"""Main training script for FRAC-MARL.

Trains a team of decentralized actor-critic agents on the MPE formation task,
under a chosen consensus/defense mode (the `mode` argument to build_agents/run):

    normal       -- Agent:            no defense; consensus_update() is a fixed-weight
                                       average (each accepted peer weighted 1/n_agents,
                                       own value keeps the remaining weight -- see Agent._average).
    naive        -- Agent:            NOT a distinct branch -- any mode other than "normal"/
                                       "resilient"/"projection"/"trimmed_mean" falls through to
                                       the same undefended Agent as "normal". The two differ only
                                       in how __main__ calls run(): "normal" is run with
                                       F_BUDGET=0 (no simulated attacks), "naive" with
                                       F_BUDGET=F_BUDGET_G -- i.e. "naive" is the undefended
                                       Agent exposed to attacks, showing what happens with no
                                       defense at all.
    resilient    -- Redundancy_Agent: tau-based message-redundancy filtering, then the same
                                       fixed-weight averaging as above over the filtered peers.
    projection   -- Projection_Agent: trimmed-mean hidden layers + gradient-recovered output step.
    trimmed_mean -- Trim_Mean_Agent:  coordinate-wise trimmed mean over all layers (peers only
                                       are sorted/trimmed; own value is always kept untrimmed).

F_BUDGET_G is the assumed number of Byzantine neighbors each agent must be resilient to;
it sets both the redundancy of the communication graph (r = 2*F_BUDGET_G + 1, see
build_neighbor_list) and, via r, the tau threshold Redundancy_Agent uses. The F_BUDGET
*argument* passed into build_agents()/run() is a separate value that sets the resilient/
projection/trimmed_mean agents' own per-call trim size (and the attack budget used to
simulate corrupted messages, see Agent.simulate_attack) -- it does not change r, so the
communication graph's redundancy stays fixed at F_BUDGET_G regardless of what F_BUDGET a
given run() call uses.

Run directly to train all five modes back-to-back for one seed:
    python train.py
Note each run() call ends with a blocking plt.show() (no non-interactive backend is set
here, unlike test_and_visualize.py) -- close each plot window for the next mode to start.
Each run() call saves a final + best actor checkpoint and (if save_rewards=True, off by
default) the per-episode reward curve. Both paths are keyed by the MODULE-LEVEL
F_BUDGET_G, not by the F_BUDGET argument that call actually used -- so all five modes'
checkpoints/rewards land in the same weights/F=<F_BUDGET_G>/ and rewards/F=<F_BUDGET_G>/
directories, consumed by test_and_visualize.py and plot_reward.py respectively.
"""

import copy
import numpy as np
import torch
import os

from environment.mpe_formation_env import MPEFormationEnv
from construct_redundant_graphs import construct_r_redundant_graph
from agents.agents import Agent
from agents.redundancy_agents import Redundancy_Agent
from agents.projection_agents import Projection_Agent
from agents.trim_mean_agent import Trim_Mean_Agent
import matplotlib.pyplot as plt

N_AGENTS = 10
MAX_EP_LEN = 35              # simple_formation_v1's own default max_cycles is 25
N_ACTIONS = 5                # [no_action, move_left, move_right, move_down, move_up]
TERMINATE_ON_SUCCESS = True   # False -> episodes always run the full MAX_EP_LEN,
                               # and `success` in the log is always 0%; set True to get
                               # a meaningful success-rate metric (formation reached to
                               # within the env's own 0.05-unit threshold)
F_BUDGET_G = 1                # GLOBAL F budget
r = 2 * F_BUDGET_G + 1
EPISODES = 10000
BATCH_EPISODES = 20
CRITIC_STEPS = 10              # per actor update
GAMMA = 0.9
CRITIC_LR = 0.01               
TEAM_REWARD_LR = 0.01                    
ACTOR_LR = 0.001
MU = 0.1
LOG_EVERY = 500
HIDDEN = 30


def mlp(in_dim, out_dim):
    """One-hidden-layer MLP (Linear -> LeakyReLU -> Linear) used for every actor,
    critic, and team-reward network in this codebase."""
    return torch.nn.Sequential(torch.nn.Linear(in_dim, HIDDEN),
                               torch.nn.LeakyReLU(0.1),
                               torch.nn.Linear(HIDDEN, out_dim))


# -------------------------------------------------
# NORMALIZER: same idea as centralized_train.fit_normalizer, kept local (rather than
# importing that one) so this file doesn't silently depend on centralized_train.py's
# own MAX_EP_LEN/N_ACTIONS constants staying in sync with the ones above.
# -------------------------------------------------
def fit_normalizer(env, max_ep_len, n_actions, n_samples=400, seed=0):
    """Collect ~n_samples states under a random policy and return a closure that
    z-score normalizes states with the resulting per-feature mean/std."""
    rng = np.random.default_rng(seed)
    S = []
    while len(S) < n_samples:
        s = env.reset(); S.append(s[0])
        for _ in range(max_ep_len):
            s, _, truncs, _ = env.step({i: int(rng.integers(0, n_actions)) for i in range(env.n)})
            S.append(s[0])
            if all(truncs):
                break
    S = np.asarray(S, np.float32)
    mean, std = S.mean(0), S.std(0) + 1e-6
    return lambda x: ((x - mean) / std).astype(np.float32)



def plotter(data, color, label, window=50, plot=False):
    """Add a moving-average curve of `data` to the current matplotlib axes; call with
    plot=True on the last series to add labels/legend and display the figure."""
    data = np.asarray(data)
    kernel = np.ones(window) / window
    mean = np.convolve(data, kernel, mode="valid")
    x = np.arange(window - 1, len(data))
    plt.plot(x, mean, label=label, color=color)


    if plot:
        plt.xlabel("Episode")
        plt.ylabel("Mean reward")
        plt.title("Mean reward over episodes")
        plt.legend()
        plt.show()


# -------------------------------------------------
# AGENTS -- identical to new_training.py: every agent starts from an IDENTICAL critic
# and team-reward net, cloned from one "proto" net.
# -------------------------------------------------
def build_agents(env, gamma=GAMMA, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, team_reward_lr=TEAM_REWARD_LR, mode="normal", F_BUDGET=F_BUDGET_G):
    """Build one agent per env agent, all starting from identical critic/team-reward
    weights (cloned from one "proto" net) so any post-training divergence between
    agents comes from training, not initialization. `mode` selects the agent class
    (see module docstring). F_BUDGET is passed to whichever class needs it -- it becomes
    tau=r-F_BUDGET for Redundancy_Agent (r is the module-level graph-redundancy constant,
    not this F_BUDGET) and the per-tail trim count F for Projection_Agent/Trim_Mean_Agent
    -- independent of the neighbor graph's own redundancy, which is fixed by F_BUDGET_G."""
    n = env.n
    state_dim = env.reset().shape[1]
    proto_critic = mlp(state_dim, 1)
    proto_reward = mlp(state_dim + n, 1)

    agents = []
    for i in range(n):
        actor = mlp(state_dim, N_ACTIONS)
        critic = mlp(state_dim, 1)
        critic.load_state_dict(proto_critic.state_dict())
        team_reward = mlp(state_dim + n, 1)
        team_reward.load_state_dict(proto_reward.state_dict())
        if mode == "normal":
            agents.append(Agent(i, n, N_ACTIONS, actor, critic, team_reward, gamma=gamma, actor_lr=actor_lr, critic_lr=critic_lr, team_reward_lr=team_reward_lr))
        elif mode == "resilient":
            agents.append(Redundancy_Agent(i, n, N_ACTIONS, actor, critic, team_reward, gamma=gamma, actor_lr=actor_lr, critic_lr=critic_lr, team_reward_lr=team_reward_lr, tau= r- F_BUDGET))
        elif mode == "projection":
            agents.append(Projection_Agent(i, n, N_ACTIONS, actor, critic, team_reward, gamma=gamma, actor_lr=actor_lr, critic_lr=critic_lr, team_reward_lr=team_reward_lr, F=F_BUDGET))
        elif mode == "trimmed_mean":
            agents.append(Trim_Mean_Agent(i, n, N_ACTIONS, actor, critic, team_reward, gamma=gamma, actor_lr=actor_lr, critic_lr=critic_lr, team_reward_lr=team_reward_lr, F=F_BUDGET))
        else:
            agents.append(Agent(i, n, N_ACTIONS, actor, critic, team_reward, gamma=gamma, actor_lr=actor_lr, critic_lr=critic_lr, team_reward_lr=team_reward_lr))

    return agents



def build_neighbor_list(n, seed):
    """Sample a fresh r-redundant communication graph (r = 2*F_BUDGET_G + 1)
    and return each agent's neighbor indices as a list of lists. Called once per batch
    in run(), so the graph topology changes every batch_episodes episodes."""
    A = construct_r_redundant_graph(n, r, seed=seed)
    neighbor_lists= [[j for j in range(n) if A[i, j] == 1] for i in range(n)]
    return neighbor_lists


# -------------------------------------------------
# ROLLOUT -- identical shape to new_training.py's; works unchanged because
# MPEFormationEnv.step() returns (state, per-agent reward, truncs, success_count) in
# the same shape enironment.py's pogema wrapper does.
# -------------------------------------------------
def rollout(env, agents, norm, mu, n, max_ep_len):
    """Run one episode with epsilon-greedy exploration (mu is the per-STEP probability
    of a fully random joint action -- see Agent.act's docstring). Returns
    (states, actions, per-agent rewards, next_states, dones, success, episode_return)
    as stacked arrays, ready to append to run()'s batch buffer."""
    s = norm(env.reset())
    S, A, Ri, S2, T = [], [], [], [], []
    ep_return, success_count = 0.0, 0
    for t in range(max_ep_len):
        explore = np.random.rand() < mu
        actions = {}
        for a in agents:
            actions.update(a.act(s, explore))

        raw, r, truncs, cov = env.step(actions)
        s2 = norm(raw)
        success_count = int(cov)
        done = success_count == n

        S.append(s[0]); A.append([actions[i] for i in range(n)])
        Ri.append(np.asarray(r, np.float32))   # per-agent local reward
        S2.append(s2[0]); T.append([float(done)])
        # r's n entries are identical (see mpe_formation_env.py's docstring) -- mean,
        # not sum, so the logged return is the actual global reward, not n times it
        ep_return += float(np.mean(r))
        s = s2
        if done or all(truncs) or t + 1 >= max_ep_len:
            break
    return (np.asarray(S, np.float32), np.asarray(A, np.int64), np.asarray(Ri, np.float32),
            np.asarray(S2, np.float32), np.asarray(T, np.float32), success_count == n, ep_return)


# -------------------------------------------------
# ONE TRAINING ROUND -- identical to new_training.py's batch_update().
# -------------------------------------------------
def batch_update(agents, buf, neighbor_list, critic_steps, F_BUDGET, mode="normal"):
    """One full training round on a batch of buffered episodes: `critic_steps` rounds of
    (local critic/reward gradient steps -> message exchange -> consensus_update), then
    one actor step per agent using the post-consensus critic/team-reward nets. Returns
    the batch's states S -- run() only checks this return value for "is not None" to
    gate its first log/checkpoint call; the states themselves aren't otherwise used
    after this call returns."""
    S, A, Ri, S2, T = (torch.as_tensor(np.concatenate([b[k] for b in buf])) for k in range(5))
    SA = torch.cat([S, A.float()], -1)
    for _ in range(critic_steps):
        for i, a in enumerate(agents):
            a.critic_step(S, S2, T, local_reward=Ri[:, i:i + 1])
            a.reward_step(SA, Ri[:, i:i + 1])

        attacked_edges, attack_budget = [], F_BUDGET
        for i, a in enumerate(agents):
            peers = [agents[j] for j in neighbor_list[i]]
            attacked_edges, attack_budget = a.receive_messages(peers, attacked_edges, attack_budget)
        if mode == "resilient":
            for i, a in enumerate(agents):
                peers = [agents[j] for j in neighbor_list[i]]
                a.receive_second_round_messages(peers)

        # Update the parameters
        if mode == "projection":
            for a in agents:
                a.consensus_update(S, SA)
        else:
            for a in agents:
                a.consensus_update()

    for a in agents:
        with torch.no_grad():
            V = a.critic(S)
            Vn = a.critic(S2)
            r_hat = a.team_reward(SA)
            adv = r_hat + a.gamma * (1 - T) * Vn - V
        a.actor_step(S, A, adv)
    return S

# -------------------------------------------------
# FULL RUN
# -------------------------------------------------
def run(episodes=EPISODES, seed=0, log_every=LOG_EVERY, batch_episodes=BATCH_EPISODES,
        critic_steps=CRITIC_STEPS, mu=MU, F_BUDGET=F_BUDGET_G, mode ="normal", save_rewards=False):
    """Train one team of agents end to end for `episodes` episodes: rollout -> buffer ->
    batch_update every `batch_episodes` episodes -> log/checkpoint every `log_every`.
    Saves a final checkpoint (always) and a best-reward checkpoint (if any logged reward
    beat the running best) to weights/F=<F_BUDGET_G>/{mode}_final{seed}.pt / _best{seed}.pt
    -- note the directory name uses the MODULE-LEVEL F_BUDGET_G constant, not this call's
    own F_BUDGET argument. Saves the full per-episode reward curve to the equivalent
    rewards/F=<F_BUDGET_G>/{mode}{seed}.npy path only when save_rewards=True (default
    False -- __main__ below does not pass it, so a plain `python train.py` run saves
    checkpoints but no reward curves for plot_reward.py to read). Also always ends with
    a blocking plt.show() of the raw (unsmoothed) return curve. Returns the list of
    per-episode returns either way, for the caller's own plotting (see plotter() above)."""
    torch.manual_seed(seed); np.random.seed(seed)
    env = MPEFormationEnv(n_agents=N_AGENTS, max_cycles=MAX_EP_LEN,terminate_on_success=TERMINATE_ON_SUCCESS, seed=seed)

    os.makedirs(f"weights/F={F_BUDGET_G}", exist_ok=True)
    os.makedirs(f"rewards/F={F_BUDGET_G}", exist_ok=True)

    final_ckpt_path = f"weights/F={F_BUDGET_G}/{mode}_final{seed}.pt"
    best_ckpt_path = f"weights/F={F_BUDGET_G}/{mode}_best{seed}.pt"
    reward_path = f"rewards/F={F_BUDGET_G}/{mode}{seed}.npy"
    
    norm = fit_normalizer(env, MAX_EP_LEN, N_ACTIONS, seed=seed)
    n = env.n
    agents = build_agents(env, mode=mode, F_BUDGET=F_BUDGET)

    def _checkpoint(episode, mean_reward):
        # deepcopy -- agents keep training after this call, so a bare reference to
        # actor.state_dict() would silently mutate underneath the saved checkpoint
        return {
            "actor_state_dicts": [copy.deepcopy(a.actor.state_dict()) for a in agents],
            "seed": seed, "n_agents": N_AGENTS, "max_ep_len": MAX_EP_LEN,
            "episode": episode, "mean_reward": mean_reward,
        }

    buf, succ, returns, expected_returns, best = [], [], [], [], -1.0
    best_reward, best_ckpt = -float("inf"), None
    last_S = None
    for ep in range(episodes):
        S, A, Ri, S2, T, success, ep_return = rollout(env, agents, norm, mu, n, MAX_EP_LEN)
        buf.append((S, A, Ri, S2, T)); succ.append(success); returns.append(ep_return); 
        SA = torch.cat([torch.as_tensor(S), torch.as_tensor(A, dtype=torch.float32)], -1)
        with torch.no_grad():
            expected_returns.append(float(np.mean([np.sum(a.team_reward(SA).detach().numpy())for a in agents])))

        if (ep + 1) % batch_episodes == 0:
            neighbor_list = build_neighbor_list(n, seed+ep)
            last_S  = batch_update(agents, buf, neighbor_list, critic_steps, F_BUDGET, mode)
            buf = []

        if (ep + 1) % log_every == 0 and last_S is not None:
            s_ = float(np.mean(succ[-log_every:])); best = max(best, s_)
            r_ = float(np.mean(returns[-log_every:]))
            exp_r_ = float(np.mean(expected_returns[-log_every:]))
            print(f"ep {ep+1:5d} | reward {r_:7.3f} | expected reward {exp_r_:.3f}", flush=True)

            if r_ > best_reward:
                best_reward = r_
                best_ckpt = _checkpoint(ep + 1, r_)


    final_r = float(np.mean(returns[-log_every:])) if returns else float("nan")
    torch.save(_checkpoint(episodes, final_r), final_ckpt_path)
    print(f"Saved final checkpoint (episode {episodes}, mean reward {final_r:.3f}) -> {final_ckpt_path}",flush=True)
    if best_ckpt is not None:
        torch.save(best_ckpt, best_ckpt_path)
        print(f"Saved best checkpoint (episode {best_ckpt['episode']}, "
              f"mean reward {best_ckpt['mean_reward']:.3f}) -> {best_ckpt_path}", flush=True)

    # save the returns for plotting later
    if save_rewards:
        np.save(reward_path, returns)

    plt.plot(np.arange(len(returns)), returns)
    plt.xlabel("Episode")
    plt.ylabel("Mean reward")
    plt.title("Mean reward over episodes")
    plt.show()
    return returns


if __name__ == "__main__":
    seed = 0

    normal_agents = run(F_BUDGET=0, mode ="normal", seed=seed)
    naive_agents = run(F_BUDGET=F_BUDGET_G, mode="naive", seed=seed)
    resilient_agents = run(F_BUDGET=F_BUDGET_G, mode="resilient", seed=seed)
    projection_agents = run(F_BUDGET=F_BUDGET_G,  mode="projection", seed=seed)
    trimmed_mean_agents = run(F_BUDGET=F_BUDGET_G, mode="trimmed_mean", seed=seed)

