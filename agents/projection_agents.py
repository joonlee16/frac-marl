import copy
import numpy as np
import torch
import torch.nn.functional as F

from agents.agents import Agent



class Projection_Agent(Agent):
    """
    PyTorch port of Algorithm 2 from Ye, Figura, Lin, Pal, Das, Liu & Gupta, "Resilient Multi-Agent 
    Reinforcement Learning With Function Approximation" (IEEE Transactions on Automatic Control, 2024,
    DOI: 10.1109/TAC.2024.3409676), reference implementation:
    https://github.com/mainakpal08/Resilient-consensus-based-MARL/blob/v1/agents/resilient_CAC_agents.py

    critic_step()/reward_step() override the base Agent's versions to match the
    reference's critic_update_local()/TR_update_local(): take one local gradient step
    to produce a candidate (v_tilde/lambda_tilde) to send to neighbors, then immediately
    restore the live network to its pre-step state, so the local step's effect is only
    ever visible in the candidate. consensus_update() then projects every accepted
    candidate through this agent's own network to recover each one's implied error
    (_projected_errors), reduces that to one team error (_trimmed_mean_error), replaces
    the hidden layers with a trimmed mean of the candidates (_trimmed_mean_hidden), and
    finally takes one gradient step fitting the network to the team error
    (_team_output_step) -- in that order, since the projected errors must be computed
    against the OLD hidden layers, before the trim overwrites them.

    Two intentional departures from the reference:
      - the reference's `features_dot` and per-node error are computed on a
        single state at a time (its own docstring: "the projection step must
        be executed individually for each step"); here both are computed
        over the whole batch (`features_dot` = whole-batch Frobenius norm^2,
        each node's error = its batch-mean projected shift) and trimmed as
        one scalar per node instead of per-sample.
      - we perform trimmed mean on the hidden layers and then do a single
        gradient step on ALL hidden and output layers. The reference
        code does only gradient descent on the output layer, but we found that doing
        a single gradient step on all layers works better in practice.
    """

    def __init__(self, index, n_agents, act_dim, actor, critic, team_reward,
                 gamma=0.95, actor_lr=0.001, critic_lr=0.01, team_reward_lr=0.01,F=0):
        """Initialize like the base Agent, plus F (the number of candidates trimmed from
        each tail) and the pre-local-step feature caches _projected_errors() needs."""
        super().__init__(index, n_agents, act_dim, actor, critic, team_reward,
                          gamma, actor_lr, critic_lr, team_reward_lr=team_reward_lr)
        self.F = F                      # max # of Byzantine neighbors trimmed per tail
        self.critic_lr = critic_lr
        self.team_reward_lr = team_reward_lr

        # Stored hidden-layer features from the pre-local-step network,
        # used to compute each node's projected error in _projected_errors()
        self.critic_features = None
        self.team_reward_features = None

    @staticmethod
    def _snapshot_optimizer(opt):
        """Deep-copy an optimizer's state_dict -- state_dict()'s tensors are the LIVE
        momentum/second-moment buffers, not copies, so a plain reference would still
        get mutated by the next .step() call rather than preserving this moment."""
        sd = opt.state_dict()
        return {
            'state': {k: {kk: (vv.clone() if torch.is_tensor(vv) else vv)
                          for kk, vv in v.items()}
                      for k, v in sd['state'].items()},
            'param_groups': copy.deepcopy(sd['param_groups']),
        }

    def critic_step(self, states, next_states, done, local_reward):
        """Take one local critic step via the base Agent (producing v_tilde, the
        candidate sent to neighbors), caching the pre-step hidden features for
        _projected_errors(), then restore the live critic and optimizer to their
        pre-step state -- the local step survives only in v_tilde."""

        # Store the critic and optimizer before the update
        pre_step = self._snapshot(self.critic)
        pre_opt = self._snapshot_optimizer(self.critic_opt)

        with torch.no_grad():
            self.critic_features = self.critic[:-1](states).detach()
        loss = super().critic_step(states, next_states, done, local_reward)

        # Restore the live critic and its optimizer to their PRE-local-step state

        self.critic.load_state_dict(pre_step, strict=False)
        self.critic_opt.load_state_dict(pre_opt)
        return loss

    def reward_step(self, states_actions, local_reward):
        """Same as critic_step, but for the team-reward network: one local step to
        produce lambda_tilde, then restore the live network and optimizer state."""
        pre_step = self._snapshot(self.team_reward)
        pre_opt = self._snapshot_optimizer(self.team_reward_opt)
        with torch.no_grad():
            self.team_reward_features = self.team_reward[:-1](states_actions).detach()
        loss = super().reward_step(states_actions, local_reward)
        self.team_reward.load_state_dict(pre_step, strict=False)
        self.team_reward_opt.load_state_dict(pre_opt)
        return loss

    # -------------------------------------------------
    # PROJECTION FILTER  (compare RPBCAC_agent.resilient_consensus_critic /
    # resilient_consensus_TR and _trimmed_mean_error)
    # -------------------------------------------------
    @staticmethod
    def _trimmed_mean_error(errors, F):
        """Own-value-protected trimmed mean over per-node errors: errors[0] (this
        agent's own estimate) is always kept, and a neighbor's tail value is only kept
        if it doesn't overshoot past the agent's own estimate on that side."""
        own_val = float(errors[0])
        sorted_vals = np.sort(errors[1:])
        n = sorted_vals.shape[0]
        small = sorted_vals[:F]
        large = sorted_vals[n - F:]
        mid = sorted_vals[F:n - F]
        accepted = np.concatenate([[own_val], small[small >= own_val], mid, large[large <= own_val]])
        return float(accepted.mean())

    # -------------------------------------------------
    # HIDDEN-LAYER CONSENSUS  (compare RPBCAC_agent.resilient_consensus_hidden /
    # _trimmed_mean_hidden -- direct parameter assignment, no gradient, no
    # "own value protected" special-casing: plain elementwise order-statistics trim)
    # -------------------------------------------------
    @staticmethod
    def _split_keys(model):
        """Split an nn.Sequential's state_dict keys into (hidden_keys, output_keys),
        where output_keys is just the last submodule's own weight+bias."""
        out_prefix = f"{len(model) - 1}."
        all_keys = list(model.state_dict().keys())
        output_keys = [k for k in all_keys if k.startswith(out_prefix)]
        hidden_keys = [k for k in all_keys if not k.startswith(out_prefix)]
        return hidden_keys, output_keys

    def _trimmed_mean_hidden(self, model, own_tilde, peer_key):
        """Elementwise trimmed mean over HIDDEN-layer parameters across this agent's own
        candidate and every accepted peer's candidate, written directly into the live
        model (no gradient step); own value gets no special protection here, unlike
        _trimmed_mean_error."""
        hidden_keys, _ = self._split_keys(model)
        candidates = [own_tilde] + [msg[peer_key] for msg in self._accepted_msgs]
        n = len(candidates)
        params = dict(model.named_parameters())
        with torch.no_grad():
            for key in hidden_keys:
                stacked = torch.stack([c[key].float() for c in candidates], dim=0)
                sorted_vals, _ = torch.sort(stacked, dim=0)
                trimmed = sorted_vals[self.F:n - self.F]
                params[key].copy_(trimmed.mean(dim=0))

    def _projected_errors(self, model, own_tilde, peer_key, inputs, lr, cached_features):
        """Project this agent's own and every accepted peer's candidate weights through
        this agent's own (still pre-trim) network at its own input batch, and back out
        each candidate's implied local error via a first-order finite-difference formula."""
        if cached_features is None:
            raise RuntimeError("consensus_update() called before any critic_step()/"
                                "reward_step() has run -- no cached features yet")
        own_snapshot = self._snapshot(model)
        with torch.no_grad():
            own_out = model(inputs)
        feat_norm_sq = cached_features.pow(2).sum().item() + 1.0

        candidates = [own_tilde] + [msg[peer_key] for msg in self._accepted_msgs]
        errors = []
        with torch.no_grad():
            for cand in candidates:
                model.load_state_dict(cand, strict=False)
                shifted_out = model(inputs)
                err = ((shifted_out - own_out) / (2 * lr * feat_norm_sq)).mean().item()
                errors.append(err)
        model.load_state_dict(own_snapshot, strict=False)   # restore -- probing must not mutate state

        return np.asarray(errors, dtype=np.float64)

    # -------------------------------------------------
    # CONSENSUS STEP
    # -------------------------------------------------

    @staticmethod
    def _output_layer(model):
        """Return the model's output (last) submodule."""
        return model[-1]

    def _team_output_step(self, model, opt, inputs, team_error):
        """One gradient step fitting the network to (its own current prediction + the
        team-averaged error estimate); currently updates every layer, hidden and
        output, not just the output layer as the reference does (see class docstring)."""

        with torch.no_grad():
            own_out = model(inputs)
        target = (own_out + team_error).detach()

        opt.zero_grad()
        pred = model(inputs)
        loss = F.mse_loss(pred, target)
        loss.backward()

        # out_params = set(self._output_layer(model).parameters())
        # for p in model.parameters():             # Reference code does this to avoid updating hidden layers, but we do this since it performs better in practice
        #     if p not in out_params:
        #         p.grad = None

        opt.step()
        return float(loss.item())

    def consensus_update(self, states, states_actions):
        """One consensus round for both networks, order matters: compute each
        candidate's projected error against the pre-trim hidden layers and reduce it to
        one team error (_projected_errors -> _trimmed_mean_error), THEN trim the hidden
        layers in place (_trimmed_mean_hidden), THEN take the team-error-fitting
        gradient step (_team_output_step) -- matching the reference's
        resilient_consensus_critic/TR -> resilient_consensus_hidden ->
        critic_update_team/TR_update_team order."""

        # Compute each node's projected error (own + neighbors)
        critic_errors = self._projected_errors(self.critic, self.v_tilde, 'critic', states, self.critic_lr, self.critic_features)
        tr_errors = self._projected_errors(self.team_reward, self.lambda_tilde, 'team_reward', states_actions, self.team_reward_lr, self.team_reward_features)
        critic_team_error = self._trimmed_mean_error(critic_errors, self.F)
        tr_team_error = self._trimmed_mean_error(tr_errors, self.F)

        # Perform trimmed mean on the hidden layers
        self._trimmed_mean_hidden(self.critic, self.v_tilde, 'critic')
        self._trimmed_mean_hidden(self.team_reward, self.lambda_tilde, 'team_reward')

        # Perform a single gradient step on the ALL hidden and output layers
        self._team_output_step(self.critic, self.critic_opt, states, critic_team_error)
        self.v_tilde = self._snapshot(self.critic)
        self._team_output_step(self.team_reward, self.team_reward_opt, states_actions, tr_team_error)
        self.lambda_tilde = self._snapshot(self.team_reward)

        self._accepted_msgs = []
