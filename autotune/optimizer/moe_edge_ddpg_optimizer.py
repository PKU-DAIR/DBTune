import logging
import numpy as np
import torch

from autotune.optimizer.edge_ddpg_optimizer import EdgeDDPG_Optimizer
from autotune.optimizer.surrogate.edge_ddpg.moe_edge_ddpg import MoEDDPG


class MoEEdgeDDPG_Optimizer(EdgeDDPG_Optimizer):
    """
    Drop-in replacement for EdgeDDPG_Optimizer that uses a Mixture of Experts Actor.

    Overrides:
      - create_model(): builds MoEDDPG instead of DDPG
      - get_suggestion(): logs gate weights every 5 steps for tier-specialization monitoring
      - _log_entropy(): extends base entropy log with gate distribution stats

    Everything else (initialize, update, anchor, BC, tier_elites, EMA norm, etc.) is inherited.
    """

    def __init__(self, *args, n_experts=4, load_balance_coef=0.002, **kwargs):
        self.n_experts = n_experts
        self.load_balance_coef = load_balance_coef
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # create_model: same as parent but builds MoEDDPG
    # ------------------------------------------------------------------

    def create_model(self):
        if not self.params:
            try:
                import glob
                search_pattern = f'model_params/{self.task_id}_*_actor.pth'
                candidates = glob.glob(search_pattern)
                if candidates:
                    best_step = -1
                    best_title = None
                    for c in candidates:
                        fn = __import__('os').path.basename(c)
                        title = fn.replace('_actor.pth', '')
                        if title.startswith(self.task_id):
                            parts = title.rsplit('_', 1)
                            if len(parts) == 2 and parts[1].isdigit():
                                step = int(parts[1])
                                if step > best_step:
                                    best_step = step
                                    best_title = title
                    if best_title:
                        self.params = f'model_params/{best_title}'
                        self.global_t = best_step
                        self.logger.info(
                            f"🔄 [MoE] Auto-Resuming from checkpoint: model_params/{best_title} "
                            f"(global_t restored to {best_step})"
                        )
                    else:
                        self.logger.info("🆕 [MoE] No existing checkpoint. Starting fresh.")
                else:
                    self.logger.info("🆕 [MoE] No existing checkpoint. Starting fresh.")
            except Exception as e:
                self.logger.warning(f"[MoE] Auto-resume check failed: {e}")

        if self.state_mean is None or self.state_var is None:
            return False

        self._fix_context_norm()

        ddpg_opt = {
            'tau': 0.001,
            'alr': 0.001,
            'clr': 0.001,
            'gamma': 0.99,
            'memory_size': 5000,
            'batch_size': self.batch_size,
            'model': self.params,
            'n_experts': self.n_experts,
            'load_balance_coef': self.load_balance_coef,
        }

        self.model = MoEDDPG(
            n_states=self.metrics_num,
            n_actions=len(self.config_space.get_hyperparameter_names()),
            opt=ddpg_opt,
            ouprocess=True,
            mean=self.running_stat.mean if self.running_stat else self.state_mean,
            var=self.running_stat.var if self.running_stat else self.state_var,
        )
        return True

    # ------------------------------------------------------------------
    # get_suggestion: same as parent + occasional gate weight logging
    # ------------------------------------------------------------------

    def get_suggestion(self, history_container=None, compact_space=None):
        if self.model is None or self.episode_init:
            return super().get_suggestion(history_container, compact_space)

        exploration_steps = 15
        noise_coeff = 1.0
        if self.phase_t > exploration_steps:
            decay_steps = self.phase_t - exploration_steps
            noise_coeff = max(0.5, 1.0 * (0.99 ** decay_steps))

        logging.getLogger().info(
            f'[MoE] Step {self.global_t} (phase={self.phase_t}): Noise={noise_coeff:.3f}')

        if np.random.random() < 0.9:
            action, gate_w = self.model.choose_action(self.state, noise_coeff, return_gate=True)
        else:
            action, gate_w = self.model.choose_action(self.state, 1.5, return_gate=True)

        # Log gate distribution every 5 steps to track expert specialization
        if self.global_t % 5 == 0:
            dominant = int(np.argmax(gate_w))
            self.logger.info(
                f'[Gate] tier={self.fog_tier} weights={gate_w.round(3)} '
                f'dominant=expert_{dominant}'
            )

        from autotune.optimizer.edge_ddpg_optimizer import action2config
        return action2config(action, self.config_space)

    # ------------------------------------------------------------------
    # _log_entropy: base entropy + gate distribution analysis
    # ------------------------------------------------------------------

    def _log_entropy(self):
        mean_entropy = super()._log_entropy()

        # Additional gate analysis: check expert utilization across random states
        self.model.actor.eval()
        n_samples = 50
        r_mean = self.running_stat.mean if self.running_stat and self.running_stat.n > 0 else np.zeros(self.metrics_num)
        r_std  = np.sqrt(self.running_stat.var + 1e-5) if self.running_stat and self.running_stat.n > 0 else np.ones(self.metrics_num)
        rng_states = [r_mean + r_std * np.random.randn(self.metrics_num) for _ in range(n_samples)]

        gate_weights_all = []
        with torch.no_grad():
            for raw_state in rng_states:
                normalized = self.model.normalizer([raw_state])
                _, gw = self.model.actor(normalized, return_gate=True)
                gate_weights_all.append(gw.cpu().numpy().flatten())

        gate_weights_all = np.array(gate_weights_all)    # [50, K]
        mean_per_expert = gate_weights_all.mean(axis=0)   # [K]
        top1_counts = np.array([
            (gate_weights_all.argmax(axis=1) == k).mean()
            for k in range(self.n_experts)
        ])
        gate_H = -(gate_weights_all * np.log(gate_weights_all + 1e-8)).sum(axis=1).mean()
        max_gate_H = np.log(self.n_experts)

        self.model.actor.train()

        self.logger.info(
            f'[Gate/Entropy] Step {self.global_t}: '
            f'mean_weights={mean_per_expert.round(3)}, '
            f'top1_frac={top1_counts.round(3)}, '
            f'gate_H={gate_H:.3f} ({gate_H/max_gate_H*100:.1f}% of max)'
        )

        # Gate collapse: one expert dominates >80% of states across all tiers
        if top1_counts.max() > 0.80:
            self.logger.warning(
                f'[Gate] COLLAPSE WARNING: expert_{top1_counts.argmax()} '
                f'dominates {top1_counts.max()*100:.1f}% of states. '
                f'Consider raising load_balance_coef (current={self.load_balance_coef}).'
            )

        return mean_entropy
