import torch
from agents.agents import Agent


class Trim_Mean_Agent(Agent):
    """
    Minimal baseline reproducing Byrd-TD(lambda)'s aggregation rule from Wu, Shen,
    Chen & Ling, "Byzantine-Resilient Decentralized Policy Evaluation With Linear
    Function Approximation" (IEEE TSP 2021) -- Algorithm 1 with lambda=0
    https://ieeexplore.ieee.org/abstract/document/9462519

    Unlike Redundancy_Agent (which filters messages before averaging) or
    Projection_Agent (which trims hidden layers and takes a separate gradient step on
    the output layer), this class accepts every peer message unconditionally
    (receive_messages is inherited, unfiltered, from Agent) and puts its entire defense
    into the aggregation step itself: consensus_update() -> _trimmed_mean() sorts each
    parameter coordinate across the received peer values, drops the F largest and F
    smallest per tail, and averages what's left together with this agent's own value
    (which is always kept, never trimmed away).
    """

    def __init__(self, index, n_agents, act_dim, actor, critic, team_reward,
                 gamma=0.95, actor_lr=0.001, critic_lr=0.01, team_reward_lr=0.01, F=0):
        """Initialize like the base Agent, plus F -- the number of peer values trimmed
        from each tail, per coordinate, before averaging."""
        super().__init__(index, n_agents, act_dim, actor, critic, team_reward,
                          gamma, actor_lr, critic_lr, team_reward_lr=team_reward_lr)
        self.F = F

    # -------------------------------------------------
    # ELEMENTWISE TRIMMED MEAN per coordinate before averaging
    # -------------------------------------------------
    def _trimmed_mean(self, model, own_tilde, peer_key):
        """Per parameter tensor: sort the received peer values coordinate-wise, drop the
        F largest and F smallest, then average the remaining peers together with this
        agent's own (untrimmed) value -- falls back to just the own value if no peer
        messages were received. Writes the result directly into `model`."""
        received = [msg[peer_key] for msg in self._accepted_msgs]
        n = len(received)
        trimmed_state_dict = {}
        with torch.no_grad():
            for key in own_tilde.keys():
                own_val = own_tilde[key].float().unsqueeze(0)
                if n == 0:
                    kept = own_val
                else:
                    stacked = torch.stack([m[key].float() for m in received], dim=0)
                    sorted_vals, _ = torch.sort(stacked, dim=0)
                    middle = sorted_vals[self.F:n - self.F]
                    kept = torch.cat([middle, own_val], dim=0)
                trimmed_state_dict[key] = kept.mean(dim=0)
        model.load_state_dict(trimmed_state_dict, strict=False)

    def consensus_update(self):
        """Apply _trimmed_mean to both the critic and team-reward candidates, refresh
        v_tilde/lambda_tilde to the result, and clear the message buffer for the next round."""
        self._trimmed_mean(self.critic, self.v_tilde, 'critic')
        self.v_tilde = self._snapshot(self.critic)

        self._trimmed_mean(self.team_reward, self.lambda_tilde, 'team_reward')
        self.lambda_tilde = self._snapshot(self.team_reward)

        self._accepted_msgs = []
