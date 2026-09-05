"""Load a trained checkpoint (as saved by train.py) and visualize a rollout.

Runs one episode with the checkpoint's actors -- on-policy stochastic sampling from
each actor's softmax distribution (see act()), no random-exploration mixing -- and
renders the agents' trajectories toward their formation targets, either as an animated
GIF (render_rollout, the default CLI entry point) or a single SVG frame (generate_figure).

Usage:
    python test_and_visualize.py [checkpoint] [--out FILE.gif] [--episode-seed N] [--fps N]

    checkpoint     path to a *_final*.pt / *_best*.pt file saved by train.py
                   (default: weights/F=1/resilient_best0.pt)
    --episode-seed layout seed for the rendered episode -- independent of the training
                   seed baked into the checkpoint, so you can preview several rollouts
                   from the same policy.
"""

import argparse
import numpy as np
import torch
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from environment.mpe_formation_env import MPEFormationEnv
from train import N_ACTIONS, TERMINATE_ON_SUCCESS, fit_normalizer, mlp


def _formation_targets(env):
    """Return (landmark_xy, target_xy, radius): the landmark's position, each agent i's
    CURRENT assigned target position target_xy[i], and the formation radius.

    mpe2's Scenario._compute_formation only stores the resulting DISTANCES
    (`scenario._delta_dists`), not the target coordinates themselves, so this
    recomputes them the same way that method does: the ideal circle positions are
    anchored to whichever agent currently has the smallest angle from horizontal
    relative to the landmark, then agents are matched to slots via the same
    Hungarian/bipartite assignment the environment's own reward uses -- so
    target_xy[i] is genuinely agent i's own assigned slot, not just "slot i"."""
    world = env.env.unwrapped.world
    radius = env.env.unwrapped.scenario.TARGET_RADIUS
    n = len(world.agents)
    ideal_sep = (2 * np.pi) / n
    landmark_xy = np.asarray(world.landmarks[0].state.p_pos, dtype=np.float32)
    rel = [np.asarray(a.state.p_pos, dtype=np.float32) - landmark_xy for a in world.agents]

    def _angle(p):
        a = np.arctan2(p[1], p[0])
        return a + 2 * np.pi if a < 0 else a

    theta_min = min(_angle(p) for p in rel)
    expected = np.asarray([
        landmark_xy + radius * np.array([np.cos(theta_min + i * ideal_sep),
                                          np.sin(theta_min + i * ideal_sep)])
        for i in range(n)
    ], dtype=np.float32)

    agent_xy = np.asarray([a.state.p_pos for a in world.agents], dtype=np.float32)
    dists = np.linalg.norm(agent_xy[:, None, :] - expected[None, :, :], axis=-1)
    row, col = linear_sum_assignment(dists)     # row is [0..n-1], already sorted by scipy
    target_xy = expected[col]                   # target_xy[i] = agent i's own assigned slot

    return landmark_xy, target_xy, radius


def load_actors(ckpt_path):
    """Load a train.py checkpoint and rebuild each agent's actor network with its
    saved weights (eval mode). Returns (actors, ckpt) -- ckpt is kept around so callers
    can read its seed/n_agents/max_ep_len metadata."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    n_agents = ckpt["n_agents"]
    # rebuild a throwaway env just to read state_dim, exactly like build_agents() does
    # at training time -- guarantees the actor architecture matches the checkpoint
    tmp_env = MPEFormationEnv(n_agents=n_agents, max_cycles=ckpt["max_ep_len"])
    state_dim = tmp_env.reset().shape[1]

    actors = []
    for sd in ckpt["actor_state_dicts"]:
        actor = mlp(state_dim, N_ACTIONS)
        actor.load_state_dict(sd)
        actor.eval()
        actors.append(actor)
    return actors, ckpt


def _collect_rollout_history(ckpt_path, episode_seed):
    """Run one rollout and collect, per step, the raw data needed to render it later
    (agent/target/landmark positions, cumulative return, success flag) plus the
    per-step instantaneous rewards -- NOT rendered image frames; generate_figure()
    renders a chosen step's entry into an actual image via _render_frame()."""
    actors, ckpt = load_actors(ckpt_path)
    n = ckpt["n_agents"]
    max_ep_len = ckpt["max_ep_len"]

    # reproduce the SAME normalizer used at training time: fit_normalizer is a pure
    # function of (env, max_ep_len, n_actions, seed), so calling it again with the
    # checkpoint's saved training seed gives bit-identical mean/std -- no need to
    # separately serialize the normalizer.
    norm_env = MPEFormationEnv(n_agents=n, max_cycles=max_ep_len, seed=ckpt["seed"])
    norm = fit_normalizer(norm_env, max_ep_len, N_ACTIONS, seed=ckpt["seed"])

    env = MPEFormationEnv(n_agents=n, max_cycles=max_ep_len,
                           terminate_on_success=TERMINATE_ON_SUCCESS, seed=episode_seed)

    s = norm(env.reset())
    landmark_xy, target_xy, radius = _formation_targets(env)
    agent_xy0 = np.asarray([a.state.p_pos for a in env.env.unwrapped.world.agents], dtype=np.float32)
    history = [{"agents": agent_xy0.copy(), "targets": target_xy.copy(),
                "landmark": landmark_xy.copy(), "step": 0, "return": 0.0, "success": False}]

    ep_return, success, cov = 0.0, False, 0
    inst_rewards = []
    for t in range(max_ep_len):
        actions = act(actors, s)
        raw, r, truncs, cov = env.step(actions)
        inst_reward = float(np.mean(r))
        inst_rewards.append(inst_reward)
        ep_return += inst_reward
        s = norm(raw)
        landmark_xy, target_xy, radius = _formation_targets(env)
        agent_xy = np.asarray([a.state.p_pos for a in env.env.unwrapped.world.agents], dtype=np.float32)
        success = cov == n
        history.append({"agents": agent_xy.copy(), "targets": target_xy.copy(),
                         "landmark": landmark_xy.copy(), "step": t + 1,
                         "return": ep_return, "success": success})
        if success or all(truncs):
            break

    all_xy = np.concatenate(
        [entry["agents"] for entry in history] +
        [entry["targets"] for entry in history] +
        [entry["landmark"][None, :] for entry in history],
        axis=0,
    )
    margin = 0.25
    xlim = (float(all_xy[:, 0].min() - margin), float(all_xy[:, 0].max() + margin))
    ylim = (float(all_xy[:, 1].min() - margin), float(all_xy[:, 1].max() + margin))

    cmap = plt.cm.tab10 if n <= 10 else plt.cm.tab20
    colors = cmap(np.linspace(0, 1, max(n, 2)))[:n]
    trails = [[] for _ in range(n)]
    for h in history:
        for i in range(n):
            trails[i].append(h["agents"][i])

    return {
        "ckpt": ckpt,
        "history": history,
        "inst_rewards": inst_rewards,
        "trails": trails,
        "colors": colors,
        "xlim": xlim,
        "ylim": ylim,
        "radius": radius,
        "n": n,
        "success": success,
        "ep_return": ep_return,
    }


def act(actors, state):
    """Sample one action per actor from its softmax policy at `state` (no exploration
    mixing -- this is evaluation, not training)."""
    actions = {}
    state_t = torch.as_tensor(state, dtype=torch.float32)
    with torch.no_grad():
        for i, actor in enumerate(actors):
            probs = torch.softmax(actor(state_t), -1)[0].numpy()
            # sample
            actions[i] = int(np.random.choice(N_ACTIONS, p=probs / probs.sum()))
    return actions


def _render_frame(fig, ax, agent_xy, target_xy, landmark_xy, radius, trails, colors,
                   xlim, ylim, step, ep_return, success):
    """Draw one rollout step onto `ax` (agents as dots with trailing paths, targets as
    X markers, the formation circle and landmark) and return it as an RGB frame array."""
    ax.cla()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle=":", alpha=0.4)

    # the circle every target sits on, centered at the landmark -- watch the X markers
    # slide around it across frames: that's the formation's rotation re-anchoring as
    # agents move, not the circle itself moving.
    ax.add_patch(plt.Circle(landmark_xy, radius, fill=False, linestyle="--",
                             color="0.4", linewidth=1.5, zorder=1))
    ax.scatter(*landmark_xy, marker="*", s=220, color="black", zorder=5, label="landmark")

    for i, color in enumerate(colors):
        trail = np.asarray(trails[i])
        if len(trail) > 1:
            ax.plot(trail[:, 0], trail[:, 1], color=color, alpha=0.35, linewidth=1.5, zorder=2)
        ax.scatter(*agent_xy[i], color=color, s=90, zorder=4,
                   label=f"agent {i}", edgecolor="black", linewidth=0.5)
        ax.scatter(*target_xy[i], color=color, marker="x", s=110, zorder=4, linewidth=2.5)

    status = " -- SUCCESS" if success else ""
    ax.set_title(f"step {step:3d} | return {ep_return:7.2f}{status}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, borderaxespad=0)
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    return frame


def generate_figure(ckpt_path, frame_number, out_path="policy_frame.svg", episode_seed=42):
    """Generate a single SVG figure for a chosen rollout timestep.

    frame_number is the desired time step in the rollout history; if it exceeds the
    available episode length, the final available frame is used instead.
    """
    data = _collect_rollout_history(ckpt_path, episode_seed)
    history = data["history"]
    n = data["n"]
    radius = data["radius"]
    colors = data["colors"]
    xlim = data["xlim"]
    ylim = data["ylim"]

    frame_idx = int(np.clip(frame_number, 0, len(history) - 1))
    h = history[frame_idx]
    reward = data["inst_rewards"][frame_idx - 1] if frame_idx > 0 else 0.0
    trails = [[] for _ in range(n)]
    for entry in history[:frame_idx + 1]:
        for i in range(n):
            trails[i].append(entry["agents"][i])

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    _render_frame(fig, ax, h["agents"], h["targets"], h["landmark"], radius,
                  trails, colors, xlim, ylim, h["step"], h["return"], h["success"])
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved frame {frame_idx} with instantaneous reward {reward:.3f} -> {out_path}")
    return out_path


def render_rollout(ckpt_path, out_path, episode_seed, fps):
    """Run one full episode with the checkpoint's actors and save it as an animated GIF
    at `out_path`. Note this duplicates _collect_rollout_history's rollout loop rather
    than calling it -- it renders each step's frame inline as the episode plays out,
    instead of building the (agents/targets/landmark/...) history list first and
    rendering from it afterward the way generate_figure() does via _collect_rollout_history."""
    actors, ckpt = load_actors(ckpt_path)
    n = ckpt["n_agents"]
    max_ep_len = ckpt["max_ep_len"]

    # reproduce the SAME normalizer used at training time: fit_normalizer is a pure
    # function of (env, max_ep_len, n_actions, seed), so calling it again with the
    # checkpoint's saved training seed gives bit-identical mean/std -- no need to
    # separately serialize the normalizer.
    norm_env = MPEFormationEnv(n_agents=n, max_cycles=max_ep_len, seed=ckpt["seed"])
    norm = fit_normalizer(norm_env, max_ep_len, N_ACTIONS, seed=ckpt["seed"])

    env = MPEFormationEnv(n_agents=n, max_cycles=max_ep_len,
                           terminate_on_success=TERMINATE_ON_SUCCESS, seed=episode_seed)

    s = norm(env.reset())
    landmark_xy, target_xy, radius = _formation_targets(env)
    agent_xy0 = np.asarray([a.state.p_pos for a in env.env.unwrapped.world.agents], dtype=np.float32)
    history = [{"agents": agent_xy0.copy(), "targets": target_xy.copy(),
                "landmark": landmark_xy.copy(), "step": 0, "return": 0.0, "success": False}]
    ep_return, success, cov = 0.0, False, 0
    for t in range(max_ep_len):
        actions = act(actors, s)
        raw, r, truncs, cov = env.step(actions)
        ep_return += float(np.mean(r))
        s = norm(raw)
        landmark_xy, target_xy, radius = _formation_targets(env)
        agent_xy = np.asarray([a.state.p_pos for a in env.env.unwrapped.world.agents], dtype=np.float32)
        success = cov == n
        history.append({"agents": agent_xy.copy(), "targets": target_xy.copy(),
                         "landmark": landmark_xy.copy(), "step": t + 1,
                         "return": ep_return, "success": success})
        if success or all(truncs):
            break

    all_xy = np.concatenate(
        [h["agents"] for h in history] + [h["targets"] for h in history] +
        [h["landmark"][None, :] for h in history], axis=0)
    margin = 0.25
    xlim = (float(all_xy[:, 0].min() - margin), float(all_xy[:, 0].max() + margin))
    ylim = (float(all_xy[:, 1].min() - margin), float(all_xy[:, 1].max() + margin))

    cmap = plt.cm.tab10 if n <= 10 else plt.cm.tab20
    colors = cmap(np.linspace(0, 1, max(n, 2)))[:n]
    trails = [[] for _ in range(n)]
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    frames = []
    for h in history:
        for i in range(n):
            trails[i].append(h["agents"][i])
        frame = _render_frame(fig, ax, h["agents"], h["targets"], h["landmark"], radius,
                               trails, colors, xlim, ylim, h["step"], h["return"], h["success"])
        frames.append(frame)
    plt.close(fig)

    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    print(f"Saved {len(frames)}-frame rollout ({'success' if success else 'no success'}, "
          f"final return {ep_return:.2f}) from checkpoint episode {ckpt['episode']} -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", nargs="?", default="weights/F=1/resilient_best0.pt")
    p.add_argument("--out", default="policy_rollout.gif")
    p.add_argument("--episode-seed", type=int, default=0, help="layout seed for the rendered episode (independent of the training seed stored in the checkpoint)")
    p.add_argument("--fps", type=int, default=15)
    args = p.parse_args()
    render_rollout(args.checkpoint, args.out, episode_seed=args.episode_seed, fps=args.fps)


