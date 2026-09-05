import numpy as np
import torch
import torch.nn.functional as F


class Agent:
    """
    Base decentralized actor-critic agent: implements Algorithm 2 from Zhang, Kaiqing,
    Zhuoran Yang, Han Liu, Tong Zhang, and Tamer Basar, "Fully Decentralized Multi-Agent
    Reinforcement Learning With Networked Agents" (ICML 2018):
    https://proceedings.mlr.press/v80/zhang18n.html?ref=https://githubhelp.com

    Each agent owns its own actor, critic, and team-reward network. One training round
    looks like: a local gradient step (critic_step/reward_step) -> exchange candidate
    weights with neighbors (send_message/receive_messages, with simulate_attack()
    optionally corrupting a message on the wire -- an addition on top of the original
    algorithm, not part of it) -> consensus_update(), which here just averages every
    accepted candidate with fixed weights (see _average). Subclasses (Redundancy_Agent,
    Projection_Agent, Trim_Mean_Agent) reuse this same skeleton but override
    receive_messages/consensus_update to add actual Byzantine-resilience on top of it.
    """
    def __init__(self, index, n_agents, act_dim, actor, critic,team_reward,
                 gamma=0.95, actor_lr=0.001, critic_lr=0.01, team_reward_lr=0.01):
        """Store this agent's own actor/critic/team-reward networks and their optimizers,
        and snapshot the initial critic/team-reward weights as v_tilde/lambda_tilde (the
        candidates that would be sent to neighbors before any training has happened)."""
        self.index = index
        self.n_agents = n_agents
        self.n_actions = act_dim
        self.gamma = gamma
        self.actor = actor
        self.critic = critic
        self.team_reward = team_reward

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.team_reward_opt = torch.optim.Adam(self.team_reward.parameters(), lr=team_reward_lr)

        self.v_tilde = self._snapshot(self.critic)
        self.lambda_tilde = self._snapshot(self.team_reward)

        self._accepted_msgs = []

    @staticmethod
    def _snapshot(model):
        """Return a detached, cloned copy of a model's state_dict -- a snapshot that
        won't be mutated by the model's later training steps."""
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    # -------------------------------------------------
    # ACTION SELECTION
    # -------------------------------------------------
    def logits(self, states):
        """Return the actor's raw (pre-softmax) action logits for a batch of states."""
        return self.actor(states)

    def act(self, state, explore):
        """Pick one action: fully random if `explore` is True (a single bool decided
        once per timestep by the caller, not redrawn per agent), otherwise sampled from
        the actor's own softmax policy."""
        if explore:
            return {self.index: int(np.random.randint(self.n_actions))}
        with torch.no_grad():
            probs = F.softmax(self.logits(torch.as_tensor(state)), -1)[0].numpy()
        return {self.index: int(np.random.choice(self.n_actions, p=probs / probs.sum()))}

    # -------------------------------------------------
    # CRITIC
    # -------------------------------------------------
    def value(self, states):
        """Return the critic's value estimate for a batch of states."""
        return self.critic(states)

    def _get_lr(self, initial_lr, step, lr_decay):
        """Step-decayed learning rate helper (initial_lr / (step+1)**lr_decay); unused
        -- not called anywhere in this codebase."""
        return initial_lr / ((step + 1) ** lr_decay)

    def critic_step(self, states, next_states, done, local_reward):
        """One local TD gradient step on the critic (target = local_reward + gamma *
        next-state value), then refresh v_tilde -- the snapshot sent to neighbors -- to
        the post-step weights."""
        with torch.no_grad():
            target = local_reward + self.gamma * (1 - done) * self.critic(next_states)
        self.critic_opt.zero_grad()

        loss = F.mse_loss(self.critic(states), target)
        loss.backward()
        self.critic_opt.step()
        self.v_tilde = self._snapshot(self.critic)          # vtilde^i, ready to send
        return float(loss.item())

    # -------------------------------------------------
    # TEAM REWARD
    # -------------------------------------------------
    def predict_team_reward(self, states_actions):
        """Return the team-reward network's prediction for a batch of (state, joint-action) pairs."""
        return self.team_reward(states_actions)

    def reward_step(self, states_actions, local_reward):
        """One local regression gradient step fitting the team-reward network to
        local_reward, then refresh lambda_tilde to the post-step weights."""
        self.team_reward_opt.zero_grad()
        loss = F.mse_loss(self.team_reward(states_actions), local_reward)

        loss.backward()
        self.team_reward_opt.step()
        self.lambda_tilde = self._snapshot(self.team_reward)   # ltilde^i, ready to send
        return float(loss.item())

    # -------------------------------------------------
    # ACTOR
    # -------------------------------------------------
    def actor_step(self, states, actions, advantage):
        """One policy-gradient step on the actor using `advantage` (a (batch, 1) tensor,
        already centered by the caller); returns the batch's policy entropy for logging."""
        logits = self.logits(states)
        logp = F.log_softmax(logits, -1)

        entropy = -(logp.exp() * logp).sum(-1).mean()
        nlp = F.cross_entropy(logits, actions[:, self.index], reduction='none').unsqueeze(1)
        loss = (nlp * advantage).mean()
        self.actor_opt.zero_grad()

        loss.backward()
        self.actor_opt.step()
        return float(entropy.item())

    # -------------------------------------------------
    # COMMUNICATION   (send/receive -- compare Consensus_AC_agent.send_message / normal_receive_messages)
    # -------------------------------------------------
    def send_message(self):
        """Package this agent's current v_tilde/lambda_tilde candidates (cloned) into
        the message a peer would receive."""
        return {
            'index': self.index,
            'critic': {k: v.clone() for k, v in self.v_tilde.items()},
            'team_reward': ({k: v.clone() for k, v in self.lambda_tilde.items()}
                             if self.lambda_tilde is not None else None),
        }

    def receive_messages(self, peers, attacked_edges, attack_budget):
        """Undefended analogue of the real agent's normal_receive_messages: gather every
        peer's current weights unconditionally."""
        for peer in peers:
            msg = peer.send_message()
            msg, attacked_edges, attack_budget = self.simulate_attack(peer, msg, attacked_edges, attack_budget)
            self._accepted_msgs.append(msg)
        return attacked_edges, attack_budget


    def simulate_attack(self, peer, original_msg, attacked_edges, attack_budget):
        """Simulate a Byzantine attack on the communication channel.
        Corrupted values add a per-coordinate offset to the original vector.
        """

        msg = original_msg
        DELTA_FACTOR = 3.0   # multiples of the parameter's own typical magnitude --
                            # bias plateaus past ~2.5-3x the honest-to-honest
        EPS = 1e-6            # floor so delta doesn't collapse to ~0 for near-zero params

        def _directional_offset(state_dict):
            return {
                k: (v + DELTA_FACTOR * v.abs().mean().clamp_min(EPS)).clamp(max=1.0)
                for k, v in state_dict.items()
            }

        corrupted_msgs = {
            'index': np.random.randint(0, self.n_agents),
            'critic': _directional_offset(self.v_tilde),
            'team_reward': _directional_offset(self.lambda_tilde)
        }
        if (peer.index, self.index) in attacked_edges or (self.index, peer.index) in attacked_edges:
            msg = corrupted_msgs
        elif (attack_budget > 0):
            attacked_edges.append((peer.index, self.index))
            msg = corrupted_msgs
            attack_budget -= 1

        return msg, attacked_edges, attack_budget


    # -------------------------------------------------
    # Consensus
    # -------------------------------------------------

    def _average(self, model, own_tilde, peer_key):
        """Fixed-weight consensus step: each accepted peer's candidate is weighted
        1/n_agents and this agent's own value keeps the rest (1 - #accepted/n_agents)
        -- i.e. mixing matrix W = I - (1/n_agents)*L, not a plain mean over however
        many peers happened to be accepted."""
        k = len(self._accepted_msgs)
        w = 1.0 / self.n_agents
        keys = list(own_tilde)
        dst  = list(model.state_dict().values())                   # live refs, not copies
        torch._foreach_copy_(dst, [own_tilde[q] for q in keys])    # dst <- own_tilde
        torch._foreach_mul_(dst, 1.0 - k * w)                      # W = I - (1/n)L
        for msgs in self._accepted_msgs:
            torch._foreach_add_(dst, [msgs[peer_key][q] for q in keys], alpha=w)


    def consensus_update(self):
        """Average the critic and team-reward candidates over every accepted message
        via _average (no Byzantine defense here), refresh v_tilde/lambda_tilde to the
        post-consensus weights, and clear the message buffer for the next round."""
        self._average(self.critic, self.v_tilde, 'critic')
        self.v_tilde = self._snapshot(self.critic)

        self._average(self.team_reward, self.lambda_tilde,'team_reward')
        self.lambda_tilde = self._snapshot(self.team_reward)

        self._accepted_msgs = []
