import numpy as np
from collections import Counter
from agents.agents import Agent


class Redundancy_Agent(Agent):
    """
    Byzantine-resilient Agent variant defending via message redundancy rather than
    trimming: every value is expected to reach an agent through several independent
    paths (the neighbor graph's r-redundancy, see train.py's build_neighbor_list), and
    a claimed sender's value is only trusted once enough of those paths agree on it.

    Message exchange is two rounds: receive_messages() is round 1 (direct messages from
    this agent's own peers, filed under each message's own claimed index), and
    receive_second_round_messages() is round 2 (each peer relays what IT directly
    received in round 1, giving this agent corroborating copies of a claimed sender's
    value from every peer that also has a direct edge to that sender). consensus_update()
    then runs redundancy_filter() -- a majority vote requiring at least `tau` matching
    copies per sender -- before falling back to the base Agent's _average() over
    whatever passed the filter. tau is conventionally r - F_BUDGET (train.py), so a
    sender's true value only needs to beat out up to F_BUDGET corrupted copies.
    """
    def __init__(self, index, n_agents, act_dim, actor, critic, team_reward,
                 gamma=0.95, actor_lr=0.001, critic_lr=0.01, team_reward_lr=0.01, tau=0):
        """Initialize like the base Agent, plus tau (the minimum number of matching
        copies required to accept a claimed sender's value) and the buffers used for
        the two-round message relay (see class docstring)."""
        super().__init__(index, n_agents, act_dim, actor, critic, team_reward, gamma, actor_lr, critic_lr, team_reward_lr=team_reward_lr)

        self.tau = tau
        self.old_Kcal_it = {}      # peer_index -> [messages received first round]
        self.Kcal_it = {}          # peer_index -> [messages received second round]
        self.Mcal_it = []          # peer indices accepted by the redundancy filter


    def receive_messages(self, peers, attacked_edges, attack_budget):
        """Round 1: pull each peer's message (through simulate_attack) and file it under
        the message's OWN claimed index in old_Kcal_it -- note simulate_attack randomizes
        a corrupted message's claimed index, so it need not match the sending peer's true one."""
        for peer in peers:
            msg = peer.send_message()
            msg, attacked_edges, attack_budget = self.simulate_attack(peer, msg, attacked_edges, attack_budget)
            if msg['index'] != self.index:
                self.old_Kcal_it.setdefault(msg['index'], []).append(msg)
        return attacked_edges, attack_budget

    def receive_second_round_messages(self, peers):
        """Round 2: for each of this agent's own peers, copy whatever THAT peer directly
        received in round 1 (peer.old_Kcal_it) into this agent's own Kcal_it -- so each
        claimed sender's value gets corroborated by every peer that also reaches it directly."""
        for peer in peers:
            for i in range(self.n_agents):
                # Make sure they filter out their own message, since they don't send it to themselves
                if i==self.index:
                    continue
                msgs = peer.old_Kcal_it.get(i, [])

                # unpack the peer's message and store it in this agent's Kcal_it
                for msg in msgs:
                    self.Kcal_it.setdefault(i, []).append(msg)
    # -------------------------------------------------
    # FILTER   (compare Consensus_AC_agent.redundancy_filter)
    # -------------------------------------------------
    @staticmethod
    def _hash_msg(msg):
        """Round a message's critic/team-reward tensors to 6 decimals and hash them into
        a comparable tuple, so identical-valued copies collapse to the same key for voting."""
        def flat(sd):
            if sd is None:
                return None
            return tuple(tuple(np.round(v.detach().cpu().numpy().flatten(), 6)) for v in sd.values())
        return (flat(msg['critic']), flat(msg['team_reward']), msg['index'])

    def redundancy_filter(self):
        """For each claimed sender k, take the most common exact value among all copies
        received for k and accept it into _accepted_msgs only if that value's count is at least tau."""
        for k, msgs in self.Kcal_it.items():
            if not msgs:
                continue
            counts = Counter(self._hash_msg(m) for m in msgs)
            mode_hash, mode_count = counts.most_common(1)[0]
            if mode_count >= self.tau:
                self.Mcal_it.append(k)
                self._accepted_msgs.append(next(m for m in msgs if self._hash_msg(m) == mode_hash))

    def consensus_update(self):
        """Merge round-1 and round-2 messages, run redundancy_filter() to populate
        _accepted_msgs, do weighted average of the accepted candidates via the base Agent's _average(),
        then reset every buffer for the next round."""
        # merge old_Kcal_it and Kcal_it into a single dict, then clear Kcal_it for the next round
        for k, msgs in self.old_Kcal_it.items():
            self.Kcal_it.setdefault(k, []).extend(msgs)

        self.redundancy_filter()

        self._average(self.critic, self.v_tilde, 'critic')
        self.v_tilde = self._snapshot(self.critic)

        self._average(self.team_reward, self.lambda_tilde,'team_reward')
        self.lambda_tilde = self._snapshot(self.team_reward)

        self.old_Kcal_it = {}
        self.Kcal_it = {}
        self.Mcal_it = []
        self._accepted_msgs = []
