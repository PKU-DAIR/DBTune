import torch
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optimizer

from .edge_ddpg import DDPG, Critic

# Indices of the hardware context dimensions in the 31-dim state vector
_CONTEXT_DIMS = (28, 29, 30)  # total_memory_mb, cpu_count, container_memory_mb


class MoEActor(nn.Module):
    """
    Mixture of Experts Actor for heterogeneous hardware tuning.

    Architecture:
      - Shared trunk (31→128→64): learns hardware-aware features
      - K expert heads (64→32→n_actions): each specializes on a tier
      - Gate (3→16→K): takes context dims and produces soft expert weights

    The gate uses only the 3 context dims (dims 28-30) so specialization is
    driven purely by hardware identity. Gate init is near-zero so softmax
    starts uniform (≈1/K per expert) — all experts receive gradient from step 1.
    """

    def __init__(self, n_states, n_actions, n_experts=4):
        super().__init__()
        n_ctx = len(_CONTEXT_DIMS)

        self.trunk = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.LayerNorm(64),
        )

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.LayerNorm(32),
                nn.Linear(32, n_actions),
            )
            for _ in range(n_experts)
        ])

        self.gate = nn.Sequential(
            nn.Linear(n_ctx, 16),
            nn.Tanh(),
            nn.Linear(16, n_experts),
        )

        self.act = nn.Sigmoid()
        self.n_experts = n_experts
        self._init_weights()

    def _init_weights(self):
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 1e-2)
                m.bias.data.uniform_(-0.1, 0.1)
        # Each expert gets a distinct seed to break symmetry from the first gradient step.
        # Without this, all experts are identical at init — the gate has no signal to
        # route to different experts, and symmetry-breaking relies on random noise alone.
        for k, expert in enumerate(self.experts):
            torch.manual_seed(2024 + k * 37)
            for m in expert:
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, 1e-2)
                    m.bias.data.uniform_(-0.1 - k * 0.02, 0.1 + k * 0.02)
        # Near-zero init → softmax ≈ uniform at start
        for m in self.gate:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 1e-2)
                m.bias.data.zero_()

    def forward(self, states, return_gate=False):
        h = self.trunk(states)                                        # [B, 64]
        ctx = states[:, list(_CONTEXT_DIMS)]                          # [B, 3]
        gate_weights = torch.softmax(self.gate(ctx), dim=-1)          # [B, K]
        expert_outs = torch.stack([e(h) for e in self.experts], dim=1)  # [B, K, A]
        mixed = (gate_weights.unsqueeze(-1) * expert_outs).sum(dim=1)   # [B, A]
        action = self.act(mixed)
        if return_gate:
            return action, gate_weights
        return action


class MoEDDPG(DDPG):
    """
    DDPG with MoEActor. Critic is identical to EdgeDDPG (no MoE needed there).

    Changes vs DDPG:
      - Actor: MoEActor (trunk + K experts + gate)
      - Actor optimizer: split LR (gate at 0.5x trunk/experts)
      - update(): adds load balancing loss + gate entropy regularization
      - choose_action(): can return gate weights for monitoring
    """

    def __init__(self, n_states, n_actions, opt, mean=None, var=None,
                 ouprocess=True, supervised=False, debug=False):
        self.n_experts = opt.get('n_experts', 4)
        self.load_balance_coef = opt.get('load_balance_coef', 0.002)
        # Must set these before super().__init__ because _build_network() is called inside it
        super().__init__(n_states, n_actions, opt, mean=mean, var=var,
                         ouprocess=ouprocess, supervised=supervised, debug=debug)
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Network construction (overrides DDPG)
    # ------------------------------------------------------------------

    def _make_actor_optimizer(self):
        return optimizer.Adam([
            {
                'params': list(self.actor.trunk.parameters()) +
                          [p for e in self.actor.experts for p in e.parameters()],
                'lr': self.alr,
                'weight_decay': 1e-5,
            },
            {
                'params': self.actor.gate.parameters(),
                'lr': self.alr * 0.5,   # gate learns slower to prevent oscillation
                'weight_decay': 1e-5,
            },
        ])

    def _build_actor(self):
        self.actor = MoEActor(self.n_states, self.n_actions, n_experts=self.n_experts)
        self.actor_criterion = nn.MSELoss()
        self.actor_optimizer = self._make_actor_optimizer()

    def _build_network(self):
        self.actor = MoEActor(self.n_states, self.n_actions, n_experts=self.n_experts)
        self.target_actor = MoEActor(self.n_states, self.n_actions, n_experts=self.n_experts)
        self.critic = Critic(self.n_states, self.n_actions)
        self.target_critic = Critic(self.n_states, self.n_actions)

        if len(self.model_name):
            self.load_model(model_name=self.model_name)
            self.logger.info(f"Loaded MoE model from {self.model_name}")

        DDPG._update_target(self.target_actor, self.actor, tau=1.0)
        DDPG._update_target(self.target_critic, self.critic, tau=1.0)

        self.loss_criterion = nn.MSELoss()
        self.actor_optimizer = self._make_actor_optimizer()
        self.critic_optimizer = optimizer.Adam(
            lr=self.clr, params=self.critic.parameters(), weight_decay=1e-5)

        self.logger.info("=== MoE-DDPG Configuration ===")
        self.logger.info(
            f"Actor: MoEActor({self.n_states}→128→64 trunk, "
            f"K={self.n_experts} experts 64→32→{self.n_actions}, "
            f"gate(3→16→{self.n_experts}))"
        )
        self.logger.info(f"Critic: ({self.n_states}+{self.n_actions})→256→64→1 (shared)")
        self.logger.info(f"LR trunk/experts={self.alr}, gate={self.alr * 0.5}")
        self.logger.info(f"Load balance coef={self.load_balance_coef}")

    # ------------------------------------------------------------------
    # Update (overrides DDPG — adds gate-specific losses)
    # ------------------------------------------------------------------

    def update(self, elite_sample=None, phase_step=0, bc_lambda=0.3):
        idxs, states, next_states, actions, rewards, terminates = self._sample_batch()
        batch_states = self.normalizer(states)
        batch_next_states = self.normalizer(next_states)
        batch_actions = DDPG.totensor(actions)
        batch_rewards = DDPG.totensor(rewards)
        mask = DDPG.totensor([0 if x else 1 for x in terminates])

        # --- Critic update (identical to DDPG) ---
        self.target_actor.eval()
        self.target_critic.eval()
        target_next_actions = self.target_actor(batch_next_states).detach()
        target_next_value = self.target_critic(
            batch_next_states, target_next_actions).detach().squeeze(1)
        current_value = self.critic(batch_states, batch_actions)
        target_next_value = torch.clamp(target_next_value, -1500.0, 20000.0)
        next_value = batch_rewards + mask * target_next_value * self.gamma

        td_errors = torch.abs(current_value.squeeze(1) - next_value).detach().numpy()
        for i in range(self.batch_size):
            self.replay_memory.update(idxs[i], float(td_errors[i]))

        critic_loss = self.loss_criterion(current_value.squeeze(1), next_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # --- Actor update (MoE) ---
        self.critic.eval()
        # Actor must be in train() mode so Dropout is active during gradient computation
        self.actor.train()
        actor_out, gate_weights = self.actor(batch_states, return_gate=True)
        policy_loss = -self.critic(batch_states, actor_out).mean()

        # Entropy regularization (same floor/decay as DDPG)
        entropy = -(actor_out * torch.log(actor_out + 1e-8) +
                    (1 - actor_out) * torch.log(1 - actor_out + 1e-8))
        entropy_coef = max(0.005, 0.02 * (0.995 ** phase_step))
        policy_loss = policy_loss - entropy_coef * entropy.mean()

        # Load balancing: penalize uneven expert utilization.
        # Uses variance of mean gate weights P — fully differentiable through softmax.
        # Minimized when P = [1/K, ..., 1/K] (uniform). No argmax needed.
        P = gate_weights.mean(dim=0)          # [K] — mean soft weight per expert
        load_balance_loss = self.n_experts * (P * P).sum()
        policy_loss = policy_loss + self.load_balance_coef * load_balance_loss

        # Gate entropy regularization: force gate diversity during first 100 steps of each phase.
        # Uses phase_step (not global_step) so it resets every tier transition in C2+.
        gate_entropy_coef = max(0.0, 0.02 * (1.0 - phase_step / 30.0))
        if gate_entropy_coef > 0:
            gate_H = -(gate_weights * torch.log(gate_weights + 1e-8)).sum(dim=-1).mean()
            policy_loss = policy_loss - gate_entropy_coef * gate_H

        # Behavior Cloning (unchanged from DDPG)
        if elite_sample is not None:
            elite_state = self.normalizer([elite_sample[0]])
            elite_action_target = torch.clamp(
                DDPG.totensor([elite_sample[1]]), 0.02, 0.98)
            actor_action_pred = self.actor(elite_state)
            bc_loss = nn.MSELoss()(actor_action_pred, elite_action_target)
            policy_loss = policy_loss + bc_lambda * bc_loss
            self.logger.info(f'[BC] bc_loss={bc_loss.item():.4f}')

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        self.critic.train()
        self.actor.train()

        DDPG._update_target(self.target_critic, self.critic, tau=self.tau)
        DDPG._update_target(self.target_actor, self.actor, tau=self.tau)

        gate_np = gate_weights.detach().numpy()
        self.logger.info(
            f'[Gate] step={phase_step} mean={gate_np.mean(axis=0).round(3)} '
            f'balance_loss={load_balance_loss.item():.4f} '
            f'critic_loss={critic_loss.item():.4f} policy_loss={policy_loss.item():.4f}'
        )

        return critic_loss.item(), policy_loss.item()

    # ------------------------------------------------------------------
    # Action selection (overrides DDPG — optionally returns gate weights)
    # ------------------------------------------------------------------

    def choose_action(self, x, coff=1, return_gate=False):
        self.actor.eval()
        act, gw = self.actor(self.normalizer([x]), return_gate=True)
        self.actor.train()
        action = act.squeeze(0).data.numpy()
        gate_np = gw.squeeze(0).data.numpy()
        if self.ouprocess:
            action += self.noise.noise() * coff
        action = action.clip(0.02, 0.98)
        if return_gate:
            return action, gate_np
        return action

    def get_gate_weights(self, x):
        """Return gate distribution for a given state (for monitoring/logging)."""
        self.actor.eval()
        _, gw = self.actor(self.normalizer([x]), return_gate=True)
        self.actor.train()
        return gw.squeeze(0).data.numpy()
