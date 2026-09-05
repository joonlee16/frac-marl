import numpy as np
from mpe2 import simple_formation_v1


class MPEFormationEnv:
    """
    Thin wrapper around mpe2's simple_formation_v1 PettingZoo parallel environment,
    adapting its per-agent dict-based API to the single-joint-state, index-keyed API
    the rest of this codebase expects (train.py's rollout(), test_and_visualize.py).

    Every agent shares the exact same joint state: _flatten() concatenates all agents'
    own observations into ONE (1, total_obs_dim) row, which is what reset()/step()
    return and what every agent's actor consumes -- there is no per-agent partial
    observation here. reward_mode="per_agent_distance" (the default) replaces the
    underlying env's own reward with each agent's raw, clipped negative distance to its
    currently-assigned formation slot (scenario._delta_dists); any other reward_mode
    value falls back to mpe2's own per-agent reward instead. success_count mirrors an
    external `covered_`-style convention (n if the whole team reached formation, else 0)
    so train.py's `done = success_count == n` check needs no special-casing.
    """
    def __init__(self, n_agents=10, max_cycles=25, terminate_on_success=True, seed=0,
                 reward_mode="per_agent_distance", render_mode=None, target_radius=0.5):
        
        """Build the underlying mpe2 parallel env, then patch its TARGET_RADIUS onto the
        live scenario object -- mpe2 bakes its own default radius in at construction, so
        this has to be overridden afterward rather than passed in."""

        self.env = simple_formation_v1.parallel_env(
            N=n_agents,
            max_cycles=max_cycles,
            continuous_actions=False,     # Discrete(5): [no_action, left, right, down, up]
            terminate_on_success=terminate_on_success,
            render_mode=render_mode,      # "rgb_array" for visualize_mpe_formation.py
        )
        self.n = n_agents
        self.reward_mode = reward_mode
        self._agent_ids = [f"agent_{i}" for i in range(n_agents)]
        self._seed = seed
        self._seeded = False

        self.env.unwrapped.scenario.TARGET_RADIUS = target_radius

    def render(self):
        """Only meaningful if constructed with render_mode='rgb_array' (or 'human')."""
        return self.env.render()

    def reset(self):
        """Reset the episode and return the flattened joint state (see _flatten); the
        underlying RNG is only seeded on the very first call (see comment below), so
        later resets get varied layouts rather than replaying the same one."""
        # Seed once (for run-level reproducibility from new_training.py's top-level
        # torch/np seed), then let the env's own RNG free-run on later resets --
        # passing the SAME seed every episode would replay the identical layout.
        if self._seeded:
            obs, _ = self.env.reset()
        else:
            obs, _ = self.env.reset(seed=self._seed)
            self._seeded = True
        return self._flatten(obs)

    def step(self, actions):
        """actions: {agent_index(int) -> discrete action in [0, 4]}. Steps every agent
        forward one tick and returns (joint_state, per-agent reward, per-agent
        truncation flags, success_count) -- reward depends on reward_mode (see class
        docstring); success_count is n if the whole team reached formation this step,
        else 0."""
        mpe_actions = {self._agent_ids[i]: int(a) for i, a in actions.items()}
        obs, rewards, terms, truncs, _ = self.env.step(mpe_actions)

        state = self._flatten(obs)
        if self.reward_mode == "per_agent_distance":
            delta_dists = self.env.unwrapped.scenario._delta_dists
            reward = -np.clip(delta_dists, 0, 2).astype(np.float32)
        else:
            reward = np.array([rewards[aid] for aid in self._agent_ids], dtype=np.float32)
        trunc_list = [bool(truncs[aid]) for aid in self._agent_ids]
        # mirror enironment.py's `covered_` convention: n if the team "succeeded"
        # (formation reached, only possible when terminate_on_success=True) else 0,
        # so rollout()'s existing `done = covered == n` logic needs no changes.
        success = len(terms) > 0 and all(terms.values())
        success_count = self.n if success else 0
        return state, reward, trunc_list, success_count

    def _flatten(self, obs_dict):
        """Concatenate every agent's own observation, in agent-id order, into one
        (1, total_obs_dim) row -- the single joint state shared by every agent."""
        return np.concatenate([obs_dict[aid] for aid in self._agent_ids], dtype=np.float32).reshape(1, -1)
