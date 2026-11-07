import logging
import numpy as np
import os
import math
import pickle
import pdb
import copy
from autotune.optimizer.surrogate.edge_ddpg.edge_ddpg import DDPG
from autotune.utils.history_container import Observation, HistoryContainer
from autotune.utils.config_space import Configuration, CategoricalHyperparameter
from autotune.utils.config_space.util import configs2space, max_min_distance
from autotune.utils.samplers import SobolSampler, LatinHypercubeSampler
from autotune.constraints.tier_profiles import get_profile
from autotune.utils.resource_monitor import ResourceMonitor

class RunningStat(object):
    """
    Exponential Moving Average (EMA) normalization stats.
    Replaces Welford's cumulative algorithm to prevent distribution shift
    when training across heterogeneous hardware tiers (v4).

    With EMA_ALPHA=0.02 the effective half-life is ~35 steps, so the
    normalizer adapts to a new tier in ~50 steps instead of being dominated
    by hundreds of past observations from a different tier.

    Interface is backwards-compatible with Welford (to_dict / load_dict).
    """
    EMA_ALPHA = 0.02  # half-life ≈ ln(2)/0.02 ≈ 35 steps

    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.ones(shape)   # EMA variance (not Welford sum-of-squares)

    def push(self, x):
        x = np.asarray(x)
        self._n += 1
        if self._n == 1:
            self._M = x.copy()
            self._S = np.ones_like(x)  # unit variance until we have data
        else:
            alpha = self.EMA_ALPHA
            delta = x - self._M
            self._M = self._M + alpha * delta
            # EMA variance update: Var_new = (1-α)*Var_old + α*(x-M_old)²
            self._S = (1.0 - alpha) * self._S + alpha * (delta ** 2)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        if self._n > 1:
            return self._S  # EMA variance is stored directly in _S
        else:
            return np.ones_like(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    def to_dict(self):
        return {'n': self._n, 'M': self._M, 'S': self._S}

    def load_dict(self, d):
        self._n = d['n']
        self._M = d['M']
        self._S = d['S']


# Context dims (28-30) represent hardware tier identity with known fixed ranges.
# EMA adaptation makes their normalization shift across tiers, causing the Actor to
# see the same physical tier as a different normalized input in C2 vs C1 → catastrophic
# forgetting. These dims are overridden with fixed normalization after every EMA update.
CONTEXT_NORM_OVERRIDES = {
    28: (5120.0, 3072.0 ** 2),  # total_memory_mb: center [2GB..8GB]
    29: (2.5,    1.5 ** 2),     # cpu_count
    30: (4096.0, 4096.0 ** 2),  # container_memory_mb
}


def create_output_folders():
    output_folders = ['ddpg', 'ddpg/save_memory', 'ddpg/model_params']
    for folder in output_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)


def config2action(config, config_space):
    action = copy.deepcopy(config.get_array())
    keys = config_space.get_hyperparameter_names()
    for key in keys:
        if type(config_space.get_hyperparameters_dict()[key]) == CategoricalHyperparameter:
            action[keys.index(key)] = action[keys.index(key)] / (config_space.get_hyperparameters_dict()[key].num_choices - 1)

    return action

def action2config(action, config_space):
    keys = config_space.get_hyperparameter_names()
    knob_dict = dict()
    for i, key in enumerate(keys):
        transform = config_space.get_hyperparameters_dict()[key]._transform
        if type(config_space.get_hyperparameters_dict()[key]) == CategoricalHyperparameter:
            action[i] = np.round( action[i] * (config_space.get_hyperparameters_dict()[key].num_choices - 1))
        knob_dict[key] = transform(action[i])
        if not type(config_space.get_hyperparameters_dict()[key]) == CategoricalHyperparameter:
            if knob_dict[key] < config_space.get_hyperparameters_dict()[key].lower:
                knob_dict[key] = config_space.get_hyperparameters_dict()[key].lower
            if knob_dict[key] > config_space.get_hyperparameters_dict()[key].upper:
                knob_dict[key] = config_space.get_hyperparameters_dict()[key].upper
    return Configuration(config_space, values=knob_dict)


class EdgeDDPG_Optimizer:
    def __init__(self, config_space,
                 history_container: HistoryContainer,
                 metrics_num,
                 task_id,
                 initial_trials=5,
                 init_strategy="random_explore_first",
                 mean_var_file='',
                 batch_size=16,
                 params='',
                 use_fog_reward=False,
                 fog_tier=None,
                 baseline_tps=None,
                 steps_per_phase=100,
                 recalibration_interval=0,
                 ablate_context_dims=False):

        self.task_id = task_id
        self.steps_per_phase = steps_per_phase
        self.recalibration_interval = recalibration_interval
        self.config_space = config_space
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics_num = metrics_num
        self.batch_size = batch_size
        self.params = params
        self.init_num = initial_trials
        self.initial_configurations = self.create_initial_design(init_strategy, excluded_configs=history_container.configurations)

        self.mean_var_file = mean_var_file
        self.internal_metrics = []
        self.running_stat = None # Replaces state_mean/var
        self.state_mean = None
        self.state_var = None

        # Fog Reward Integration
        self.use_fog_reward = use_fog_reward
        self.fog_tier = fog_tier
        self.baseline_tps = baseline_tps
        self._post_recal_skip = False  # discard first post-recalibration transition
        
        if self.use_fog_reward:
            self.logger.info('=== Fog-Aware Reward Enabled for Edge-DDPG ===')
            self.logger.info(f'Tier: {self.fog_tier}')

            # Get tier constraints and tier-specific baseline TPS
            tier_profile = get_profile(self.fog_tier)
            constraints = {
                'max_memory_mb': tier_profile['max_memory_mb'],
                'max_swap_mb': tier_profile['max_swap_mb'],
                'max_cpu_percent': tier_profile['max_cpu_percent']
            }
            # Tier-specific baseline overrides config value
            self.baseline_tps = tier_profile.get('baseline_tps', self.baseline_tps)
            self.logger.info(f'Baseline TPS (tier-specific): {self.baseline_tps}')
            self.logger.info(f'Constraints: {constraints}')

            # Reward computation happens in dbenv.py via FogAwareRewardV2.
            # EdgeDDPG_Optimizer reads reward from observation.objs[0] — no local instance needed.
            self.fog_reward = None
            self.resource_monitor = None
        else:
            self.logger.info('=== Using Vanilla Reward (TPS delta-based) ===')
            self.fog_reward = None
            self.resource_monitor = None

        # Ablation: if True, dims 28-30 (hardware context) are zeroed out before
        # feeding the state to the network and the EMA normalizer.
        self.ablate_context_dims = ablate_context_dims
        if ablate_context_dims:
            self.logger.info('[Ablation] Context dims 28-30 DISABLED — ablation mode active')

        self.model = None
        self.init_step = 0
        self.episode = 0
        self.global_t = 0
        self.phase_t = 0       # Steps in current phase (resets each optimize.py process)
        self._initializing = False
        self.t = 0
        self.score = 0
        self.episode_init = True
        self.state = None      # Set on first observation; must exist before get_suggestion checks it

        # [v6] Per-tier elite samples: persist across processes via pkl file.
        # Each tier keeps its best-ever config — BC anchors to the right tier's policy.
        self.tier_elites_file = f'{task_id}_tier_elites.pkl'
        self.tier_elites = {}
        if os.path.exists(self.tier_elites_file):
            try:
                with open(self.tier_elites_file, 'rb') as f:
                    self.tier_elites = pickle.load(f)
                self.logger.info(f'[v6/TierElites] Loaded elites for: {list(self.tier_elites.keys())}')
            except Exception as e:
                self.logger.warning(f'[v6/TierElites] Load failed: {e}')

        # [v6] Anchor buffer: top-50 transitions per tier, warm-start Critic each cycle.
        self._current_anchor = []

        # [v6] BC lambda decays by cycle (C1=0.5, C2=0.3, C3+=0.1). Set TRAINING_CYCLE env var.
        self.cycle_number = int(os.environ.get('TRAINING_CYCLE', '1'))
        self.logger.info(f'[v6] Cycle number: {self.cycle_number} — BC lambda: {max(0.1, 0.5 - 0.2 * (self.cycle_number - 1)):.1f}')

        create_output_folders()
        self.initialize(history_container)

    def initialize(self, history_container):
        self._initializing = True
        # [v8] Per-tier normalizer restore: in C2+, prefer tier-specific EMA stats
        # so the Actor sees the same normalized distribution as in C1 for the same tier.
        # Avoids EMA drift where C1-2GB calibration makes 6GB observations look like outliers.
        _norm_file_to_load = self.mean_var_file
        if self.fog_tier and self.cycle_number > 1 and self.mean_var_file:
            tier_norm_file = self.mean_var_file.replace('.pkl', f'_{self.fog_tier}.pkl')
            if os.path.exists(tier_norm_file):
                _norm_file_to_load = tier_norm_file
                self.logger.info(f'[Norm] C{self.cycle_number}: loading tier-specific normalizer from {tier_norm_file}')
        if _norm_file_to_load != '' and os.path.exists(_norm_file_to_load):
            with open(_norm_file_to_load, 'rb') as f:
                # DBTune 2.0: Load RunningStat object or dict
                try:
                    data = pickle.load(f)
                    if isinstance(data, dict) and 'n' in data:
                        # Validate shape to avoid loading stale stats from a different experiment
                        pkl_dim = data.get('M', np.array([])).shape[0] if hasattr(data.get('M', None), 'shape') else 0
                        if pkl_dim != self.metrics_num:
                            self.logger.warning(
                                f"[Norm] pkl shape mismatch: pkl has {pkl_dim} dims, "
                                f"expected {self.metrics_num}. Discarding stale stats."
                            )
                            self.running_stat = RunningStat(self.metrics_num)
                            self.state_mean = None
                        else:
                            self.running_stat = RunningStat(self.metrics_num)
                            self.running_stat.load_dict(data)
                            self.state_mean = self.running_stat.mean
                            self.state_var = self.running_stat.var
                            self.logger.info('Loaded Dynamic Normalization Stats.')
                    else:
                        # Fallback for old format (backward compatibility-ish)
                        # We assume it was (mean, var) but we can't reliably build running stat from it
                        # So we restart running stat but use loaded mean/var for init
                        self.state_mean = data
                        self.state_var = pickle.load(f)
                        self.running_stat = RunningStat(self.metrics_num)
                        # Initialize running stat with loaded mean to not start from 0
                        self.running_stat._M = self.state_mean
                        self.running_stat._n = 10 # Fake count to give some weight
                        self.logger.info('Loaded Legacy Stats (Converted to Dynamic).')
                except Exception as e:
                     self.logger.warning(f"Failed to load stats: {e}. Starting fresh.")
                     self.running_stat = RunningStat(self.metrics_num)
                     self.state_mean = None
        else:
            self.running_stat = RunningStat(self.metrics_num)
            # Pre-feed history (skip FAILED trials whose IM is all-zeros)
            for im in history_container.internal_metrics:
                self.internal_metrics.append(im)
                if len(im) == self.metrics_num and not np.allclose(im, 0):
                    self.running_stat.push(im)
            
            if self.running_stat.n > 0:
                self.state_mean = self.running_stat.mean
                self.state_var = self.running_stat.var

        # Fix context dims before model creation so DDPG normalizer is initialized correctly.
        self._fix_context_norm()
        if self.running_stat is not None and self.running_stat.n > 0:
            self.state_mean = self.running_stat.mean
            self.state_var = self.running_stat.var

        # If resuming (history exists), skip LHS exploration — Actor takes over immediately.
        # If fresh start (no history), preserve initial_trials (5) so LHS runs before Actor.
        # Bug: overwriting with 0 on fresh start caused model creation after 1 obs,
        # discarding 4 of 5 LHS exploration configs.
        if len(history_container.configurations) > 0:
            self.init_num = len(history_container.configurations)
        # else: keep self.init_num = initial_trials (set in __init__) for fresh start
        if self.create_model():
            # v5: Buffer starts fresh each phase.
            # PER priorities from a different tier are stale and misleading — the TD-error
            # of a "good" 6GB transition is meaningless in the context of the 2GB Critic.
            # Cross-tier knowledge persists through Actor/Critic weights (loaded via checkpoint).
            self.logger.info('[Buffer] Starting fresh PER buffer for this phase.')

            configurations = configs2space(history_container.configurations, self.config_space)
            for i in range(len(configurations)):
                objs = [history_container.perfs[i]]

                observation = Observation(config=configurations[i],
                                      objs=objs,
                                      constraints=history_container.constraint_perfs,
                                      trial_state=history_container.trial_states[i],
                                      elapsed_time=history_container.elapsed_times[i],
                                      iter_time=history_container.iter_times[i],
                                      EM=history_container.external_metrics[i],
                                      resource=history_container.resource[i],
                                      IM=history_container.internal_metrics[i],
                                      info=history_container.info, context=history_container.contexts[i])
                try:
                    self.update(observation)
                except:
                    pass
        self._initializing = False
        # After replay, self.state holds the IM of the FIRST history obs (set via episode_init path).
        # Update it to the LAST history obs so the first Actor query uses current-tier context.
        if len(history_container.internal_metrics) > 0:
            self.state = self._maybe_ablate(
                np.asarray(history_container.internal_metrics[-1], dtype=np.float32)
            )

        # [v6] Load per-tier elite from persistent store (survives process restarts & cycles).
        # Each tier has its own best-ever config; BC will anchor to the right tier's policy.
        tier_elite = self.tier_elites.get(self.fog_tier) if self.fog_tier else None
        n_actions = len(self.config_space.get_hyperparameter_names())
        if tier_elite is not None and len(tier_elite[1]) != n_actions:
            self.logger.warning(
                f'[v9/TierElites] Discarding stale elite for {self.fog_tier}: '
                f'action dims {len(tier_elite[1])} != current n_actions {n_actions}'
            )
            tier_elite = None
        if tier_elite is not None:
            self.best_sample = tier_elite
            self.best_reward = tier_elite[2]  # tuple: (state, action, reward, next_state, done)
            self.logger.info(f'[v6/TierElites] Warm BC start for {self.fog_tier}: reward={self.best_reward:.2f}')
        else:
            self.best_sample = None
            self.best_reward = -float('inf')
            self.logger.info(f'[v6/TierElites] No elite yet for {self.fog_tier} (first visit)')
        # [v6] Inject anchor transitions from previous cycle into fresh PER buffer.
        # Gives the Critic a warm start with known-good transitions instead of starting cold.
        self._inject_anchor()
        # [C2+ fix] Repopulate buffer with all real transitions from history so the Critic
        # has full C1 coverage, not just the top-50 anchor.
        self._repopulate_buffer_from_history(history_container)


    def create_model(self):
        # [DBTune 2.0] Auto-Resume Logic
        # If no specific model is requested, try to resume from the latest checkpoint of this Task ID.
        if not self.params:
            try:
                import glob
                # EdgeDDPG saves as: model_params/{task_id}_{step}_actor.pth
                # We search for this pattern.
                search_pattern = f'model_params/{self.task_id}_*_actor.pth'
                candidates = glob.glob(search_pattern)
                if candidates:
                    # Sort by STEP NUMBER (not mtime) to get the highest cumulative checkpoint.
                    # mtime can be misleading when phases overwrite low-numbered checkpoints.
                    best_step = -1
                    best_title = None
                    for c in candidates:
                        fn = os.path.basename(c)
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
                            f"🔄 Auto-Resuming from checkpoint: model_params/{best_title} "
                            f"(global_t restored to {best_step})"
                        )
                    else:
                        self.logger.info("🆕 No existing checkpoint found. Starting fresh.")
                else:
                    self.logger.info("🆕 No existing checkpoint found. Starting fresh.")
            except Exception as e:
                self.logger.warning(f"Auto-resume check failed: {e}")

        if (self.state_mean is None) or (self.state_var is None) :
            return False
        
        # Edge-DDPG: Reduced parameters for fog computing
        ddpg_opt = {
            'tau': 0.001,           # Reduced -> More conservative updates
            'alr': 0.001,           # Actor learning rate (same)
            'clr': 0.001,           # Critic learning rate (same)
            'gamma': 0.99,          # [v7.1] Reverted: TPCC has cross-step memory (cache, vacuum). 0.9 caused monotonic intra-phase degradation.
            'memory_size': 5000,    # Reduced from 100K to 5K (95% reduction)
            'batch_size': self.batch_size,
            'model': self.params,
        }

        self.model = DDPG(n_states=self.metrics_num,
                          n_actions=len(self.config_space.get_hyperparameter_names()),
                          opt=ddpg_opt,
                          ouprocess=True,
                          mean=self.running_stat.mean if self.running_stat else self.state_mean,
                          var=self.running_stat.var if self.running_stat else self.state_var)
        # Expose baseline_tps on the model for re-labeling at sample time
        self.model.baseline_tps = self.baseline_tps if self.baseline_tps else 100.0

        return True
 
    def gen_mean_var(self):
        # NOTE: Deprecated in DBTune 2.0 favor of Dynamic Update
        # Kept only for compatibility if called explicitly, but updates running_stat
        if self.running_stat is None:
             self.running_stat = RunningStat(self.metrics_num)
        
        # Only needed if starting from zero
        if self.running_stat.n == 0:
            for im in self.internal_metrics:
                if len(im) == self.metrics_num:
                     self.running_stat.push(im)
        
        self.state_mean = self.running_stat.mean
        self.state_var = self.running_stat.var
        
        if self.mean_var_file == '':
            self.mean_var_file = '{}_mean_var.pkl'.format(self.task_id)
        with open(self.mean_var_file, 'wb') as f:
            pickle.dump(self.running_stat.to_dict(), f)
        self.logger.info('Updated Dynamic Normalization Stats.')

    def get_suggestion(self, history_container=None, compact_space=None):
        if self.model is None:
            init_config = self.initial_configurations[self.init_step]
            self.init_step += 1
            return init_config

        if self.episode_init:
            self.t = 0
            self.score = 0
            self.episode_init = False
            # Don't return default — it wastes a 16-min benchmark slot per phase on a known-bad config.
            # Fall through to Actor. Exception: if self.state is None (first time after model creation,
            # no prior observation yet), we need one bootstrap step to get a real state.
            if self.state is None:
                return self.config_space.get_default_configuration()

        # Two-Stage Exploration Strategy (Overnight Plan)
        # Phase 1 (0-30): High Exploration (Noise=1.0)
        # Phase 2 (30-100): Exploitation/Fine-Tuning (Decay Noise)
        
        # [Cyclical Training Strategy]
        # global_t is cumulative (restored from checkpoint on resume).
        # Use high exploration early in each phase (phase_t < exploration_steps) then decay.
        # Using phase_t (not global_t) ensures exploration resets each tier transition.
        
        # Exploration budget and max noise shrink with each cycle.
        # C1: 25 steps full noise, max=1.0. C2: 20 steps, max=0.575. C3+: 15 steps, max=0.15.
        exploration_steps = max(15, 25 - 5 * (self.cycle_number - 1))
        noise_max = max(0.15, 1.0 - 0.425 * (self.cycle_number - 1))
        noise_coeff = noise_max

        if self.phase_t > exploration_steps:
            decay_steps = self.phase_t - exploration_steps
            noise_coeff = max(0.05, noise_max * (0.985 ** decay_steps))

        # Use root logger to ensure visibility in main log
        logging.getLogger().info(f'[DDPG] Step {self.global_t} (phase={self.phase_t}): Noise Coeff={noise_coeff:.3f}')

        if np.random.random() < 0.9: # Increased from 0.7 to 0.9 to trust the Actor more in later stages
            X_next = self.model.choose_action(self.state, noise_coeff)
        else:
            # Pure Exploration (Random Action) - Reduced probability (10%)
            X_next = self.model.choose_action(self.state, 1.5) # Boosted noise for random bursts

        # Post-discretize synchronous_commit: snap to 0.1 (off) or 0.9 (on).
        # Prevents gradient contamination at the 0.5 threshold — the behavior is binary
        # but the loss landscape is continuous, causing noisy gradients near the boundary.
        try:
            hp_names = self.config_space.get_hyperparameter_names()
            if 'synchronous_commit' in hp_names:
                sc_idx = hp_names.index('synchronous_commit')
                X_next[sc_idx] = 0.1 if X_next[sc_idx] < 0.5 else 0.9
        except Exception:
            pass

        config = action2config(X_next, self.config_space)
        return config

    def _maybe_ablate(self, state: np.ndarray) -> np.ndarray:
        if self.ablate_context_dims and len(state) > 30:
            state = np.array(state, dtype=float)
            state[28:31] = 0.0
        return state

    def update(self, observation: Observation):
        obs_im = self._maybe_ablate(observation.IM)

        if self.model is None:
            self.internal_metrics.append(obs_im)
            if len(self.internal_metrics) >= self.init_num:
                self.gen_mean_var()
                self.create_model()
                self.logger.info('Iteration {} create model'.format(self.init_step))

                # [Fix] Don't return! Initialize state immediately with this observation
                # to avoid wasting a step returning Default Config.
                pass
            else:
                return

        # Bootstrap: if model was created from pkl but no history replay occurred,
        # self.state was never set via the episode_init path. Set it now to avoid
        # a crash in add_sample() / normalizer when state is None.
        if self.state is None and not self._initializing:
            self.state = obs_im

        if self.episode_init:
            self.state = obs_im
            self.default_external_metrics = observation.objs[0]
            self.last_external_metrics = observation.objs[0]

            self.episode_init = False
            self.episode += 1
            self.t = 0
            self.logger.info('New Episode-%d, initialize' % (self.episode))
            return

        # Calculate reward (fog-aware or vanilla)
        if self.use_fog_reward:
            # FIX: The objective returned by dbenv IS ALREADY the negative Fog Reward.
            # Recalculating it here caused "Double Scaling".
            # Old Logic: tps=100 -> Reward = (100/1500)*100 = 6.6. (Crushed Signal)
            # New Logic: reward = -(-100) = 100. (Correct Signal)
            reward = -observation.objs[0]
            
            # DBTune 2.0: Dynamic Normalization Update
            # Guard: skip during history replay. Replay pushes shift the Welford mean toward a
            # cross-tier mixture, poisoning the normalizer before the first real step runs.
            if not self._initializing and hasattr(self, 'running_stat') and self.running_stat is not None:
                # 1. Update running stats with new observation
                if len(obs_im) == self.metrics_num:
                    self.running_stat.push(obs_im)
                # Keep context dims (28-30) fixed regardless of EMA drift
                self._fix_context_norm()

                # 2. Update the Agent's normalizer
                new_mean = self.running_stat.mean
                new_var = self.running_stat.var
                self.model.update_normalization(new_mean, new_var)

                # 3. Save stats periodically — both global and per-tier
                if self.global_t % 10 == 0:
                     with open(self.mean_var_file, 'wb') as f:
                        pickle.dump(self.running_stat.to_dict(), f)
                     # Save per-tier copy so C2+ can restore tier-specific EMA calibration
                     if self.fog_tier:
                         tier_norm_file = self.mean_var_file.replace('.pkl', f'_{self.fog_tier}.pkl')
                         with open(tier_norm_file, 'wb') as f:
                             pickle.dump(self.running_stat.to_dict(), f)

            # Extract metrics just for logging (optional, or we can trust dbenv logs)
            tps_proxy = reward # Just for display if needed
            
            # Log fog metrics
            if self.global_t % 5 == 0:
                self.logger.info(f'[FogReward] (From Obj) Reward={reward:.2f}')
                self.logger.info(f'[FogReward] Double-Check: Obj[0]={observation.objs[0]}')
            
            # Log fog metrics (already logged above)
            pass
        else:
            # Vanilla reward (delta-based)
            reward = self.get_reward(observation.objs[0])
        
        self.last_external_metrics = observation.objs[0]
        next_state = obs_im
        # Guard: don't advance episode counters during replay. Without this, self.t crosses 100
        # and sets episode_init=True mid-replay, causing the first real get_suggestion() to
        # return the default config and wasting a full benchmark slot (~16 min).
        if not self._initializing:
            self.t += 1
            self.global_t += 1
            self.phase_t += 1

        done = False
        # PostgreSQL tuning has no true terminal state — episodic resets within a phase
        # waste observations (~16 min each) and corrupt the transition with done=True.
        # Phase boundaries are handled by process restarts; disable intra-phase resets.
        if self.t >= 10000:
            done = True
        if done or self.score < -50:
            self.episode_init = True

        # Elite Reinforcement Strategy + Anchor Buffer [v6]
        if not hasattr(self, 'best_sample'):
            self.best_sample = None
            self.best_reward = -float('inf')

        # Compute action once for elite, buffer, and anchor (avoids duplicate call)
        action_for_model = None
        if not self._initializing:
            action_for_model = config2action(observation.config, self.config_space)

        # Guard: don't update elite from history replay — cross-tier contamination causes collapse
        if not self._initializing and reward > self.best_reward:
            self.best_reward = reward
            self.best_sample = (self.state, action_for_model, reward, next_state, done)
            # [v6] Persist tier elite immediately so it survives process restarts and cycles
            if self.fog_tier:
                self.tier_elites[self.fog_tier] = self.best_sample
                try:
                    with open(self.tier_elites_file, 'wb') as f:
                        pickle.dump(self.tier_elites, f)
                except Exception as e:
                    self.logger.warning(f'[v6/TierElites] Save failed: {e}')
            self.logger.info(f'[Elite] New Best for {self.fog_tier}! Reward={reward:.2f}')

        # Standard add — skip during history replay.
        # Also skip the first transition after a recalibration: that transition crosses
        # a Default→Agent discontinuity in the environment state.
        if not self._initializing:
            if self._post_recal_skip:
                self.logger.info('[Recalibration] Discarding post-recal transition (Default→Agent discontinuity)')
                self._post_recal_skip = False
            else:
                # Re-labeling: store (tps_raw, penalty_sum) instead of scalar reward when available.
                # model.update() recalculates reward = tps_raw/baseline_current*100 + penalty_sum.
                fog_rw = getattr(self.env, 'fog_reward', None)
                if self.use_fog_reward and fog_rw is not None and hasattr(fog_rw, 'last_tps_raw'):
                    reward_data = (fog_rw.last_tps_raw, fog_rw.last_penalty_sum)
                else:
                    reward_data = reward
                self.model.add_sample(self.state, action_for_model, reward_data, next_state, done)
            # [v6] Anchor: accumulate top-50 transitions for this tier (warm-start next cycle)
            self._current_anchor.append((self.state, action_for_model, reward, next_state, done))
            if len(self._current_anchor) > 100:  # trim periodically
                self._current_anchor.sort(key=lambda t: -t[2])
                self._current_anchor = self._current_anchor[:50]

        # Periodic Re-Injection (Mental Rehearsal)
        # Guard 1: skip during replay. Guard 2: only rehearse the CURRENT tier's elite (not
        # a different tier's best_sample, which would inject OOD states into the buffer).
        # Guard 3: same >70 threshold as BC — mediocre elites degrade rather than help.
        tier_elite_for_rehearsal = self.tier_elites.get(self.fog_tier) if self.fog_tier else None
        if (not self._initializing and tier_elite_for_rehearsal is not None
                and tier_elite_for_rehearsal[2] > 105.0 and self.global_t % 5 == 0):
             bs = tier_elite_for_rehearsal
             self.model.add_sample(bs[0], bs[1], bs[2], bs[3], bs[4])

        # Guard: only update self.state with real benchmark observations.
        # During replay, self.state would be overwritten with the last cross-tier historical IM,
        # making the first real Actor query operate on a stale state from a different tier.
        if not self._initializing:
            self.state = next_state

        if len(self.model.replay_memory) > self.batch_size and not self._initializing:
            losses = []

            # Determine elite sample for Behavior Cloning
            # [v7.1] Only guide toward elite if it's actually good (>70% of PGTune baseline).
            # BC with a mediocre elite (reward~45) causes monotonic policy degradation.
            elite_sample_for_update = None
            best_rwd = getattr(self, 'best_reward', -float('inf'))
            if hasattr(self, 'best_sample') and self.best_sample is not None and best_rwd > 105.0:
                elite_sample_for_update = self.best_sample

            # [v7.1] Lowered BC lambda: C1=0.2, C2=0.15, C3+=0.1
            # [v8] Added phase_t linear decay: lambda drops to 20% by end of phase
            # Prevents aggressive cloning of early mediocre elites in second half of each phase
            bc_lambda_base = max(0.1, 0.2 - 0.05 * (self.cycle_number - 1))
            phase_progress = self.phase_t / max(1, self.steps_per_phase)
            bc_lambda = bc_lambda_base * max(0.2, 1.0 - phase_progress)
            for _ in range(4):
                losses.append(self.model.update(elite_sample=elite_sample_for_update, phase_step=self.phase_t, bc_lambda=bc_lambda))
            if losses and self.global_t % 5 == 0:
                c_losses = [l[0] for l in losses if l is not None]
                p_losses = [l[1] for l in losses if l is not None]
                if c_losses:
                    self.logger.info(
                        f'[Update] step={self.global_t} critic_loss={sum(c_losses)/len(c_losses):.4f} '
                        f'policy_loss={sum(p_losses)/len(p_losses):.4f}'
                    )

        if self.global_t % 5 == 0 and not self._initializing:
            self.model.save_model('model_params', title='{}_{}'.format(self.task_id, self.global_t))
            self.logger.info('Save model_params to %s_%s' % (self.task_id, self.global_t))
            replay_path = '{}_replay_buffer.pkl'.format(self.task_id)
            self.model.replay_memory.save(replay_path)
            self.logger.info('[Buffer] Replay buffer saved (%d transitions) → %s' % (len(self.model.replay_memory), replay_path))
            # [v6] Save anchor for current tier (top-50 by reward, used for warm-start next cycle)
            if self.fog_tier and len(self._current_anchor) > 0:
                try:
                    sorted_anchor = sorted(self._current_anchor, key=lambda t: -t[2])[:50]
                    anchor_file = f'{self.task_id}_{self.fog_tier}_anchor.pkl'
                    with open(anchor_file, 'wb') as f:
                        pickle.dump(sorted_anchor, f)
                    self.logger.info(f'[v6/Anchor] Saved {len(sorted_anchor)} transitions for {self.fog_tier}')
                except Exception as e:
                    self.logger.warning(f'[v6/Anchor] Save failed: {e}')

        if self.global_t % 25 == 0 and self.global_t > 0:
            self._log_entropy()

        # Rolling baseline recalibration — corrects for DB growth during training.
        # Every `recalibration_interval` phase steps, measure Default config TPS and update baseline.
        # This keeps reward = TPS_agent / TPS_default_now (relative to current DB state).
        if (not self._initializing
                and self.recalibration_interval > 0
                and self.phase_t > 0
                and self.phase_t % self.recalibration_interval == 0):
            self._recalibrate_baseline()

    def _recalibrate_baseline(self):
        try:
            self.logger.info(f'[Recalibration] phase_t={self.phase_t}: measuring Default TPS...')
            new_baseline = self.env.measure_default_tps(n=1)
            if new_baseline and new_baseline > 0:
                old = self.baseline_tps
                self.baseline_tps = new_baseline
                # Propagate to FogAwareRewardV2 and DDPG model (for re-labeling at sample time)
                if hasattr(self.env, 'fog_reward') and self.env.fog_reward is not None:
                    self.env.fog_reward.baseline_tps = new_baseline
                if hasattr(self.model, 'baseline_tps'):
                    self.model.baseline_tps = new_baseline
                # Skip the next transition: it crosses a Default→Agent environment discontinuity
                self._post_recal_skip = True
                self.logger.info(f'[Recalibration] baseline_tps updated: {old:.1f} → {new_baseline:.1f}')
        except Exception as e:
            self.logger.warning(f'[Recalibration] Failed, keeping old baseline: {e}')

    def _log_entropy(self):
        """
        Evalúa la diversidad de la política del Actor usando entropía Bernoulli real.
        H = -p*log(p) - (1-p)*log(1-p), normalizada a % de la entropía máxima (ln2).
        """
        import torch
        self.model.actor.eval()
        n_samples = 50
        eps = 1e-8

        # Sample from the actual observed distribution, not standard Gaussian.
        # With mean=2.5/std=0.3, a standard Gaussian maps to inputs ~8 std from the training range,
        # making the Actor deterministic artificially — false positive of policy collapse.
        r_mean = self.running_stat.mean if self.running_stat and self.running_stat.n > 0 else np.zeros(self.metrics_num)
        r_std  = np.sqrt(self.running_stat.var + 1e-5) if self.running_stat and self.running_stat.n > 0 else np.ones(self.metrics_num)
        rng_states = [r_mean + r_std * np.random.randn(self.metrics_num) for _ in range(n_samples)]

        entropies = []
        with torch.no_grad():
            for raw_state in rng_states:
                normalized = self.model.normalizer([raw_state])
                out = self.model.actor(normalized).cpu().numpy().flatten()
                h = -out * np.log(out + eps) - (1 - out) * np.log(1 - out + eps)
                entropies.append(h.mean())

        mean_entropy = float(np.mean(entropies))
        max_entropy = np.log(2)
        entropy_ratio = mean_entropy / max_entropy

        self.model.actor.train()

        self.logger.info(
            f'[Entropy] Step {self.global_t}: H={mean_entropy:.4f} '
            f'({entropy_ratio*100:.1f}% of max={max_entropy:.3f})'
        )

        if mean_entropy < 0.15:
            self.logger.warning(
                f'[Entropy] LOW ENTROPY WARNING at step {self.global_t}: '
                f'H={mean_entropy:.4f} ({entropy_ratio*100:.1f}%). Monitor closely.'
            )
        if mean_entropy < 0.05:
            self.logger.warning(
                f'[Entropy] POLICY COLLAPSE DETECTED at step {self.global_t}! '
                f'H={mean_entropy:.4f} < 0.05. Consider reverting to earlier checkpoint.'
            )

        return mean_entropy

    def _fix_context_norm(self):
        """Override EMA stats for context dims (28-30) with fixed tier-independent normalization.
        Called after every normalizer update to prevent cross-tier distribution shift.
        In ablation mode: dims 28-30 are zeroed, so normalization must map 0→0 (mean=0, std=1)."""
        if self.running_stat is None:
            return
        if self.ablate_context_dims:
            # Ablation: zeroed dims must normalize to 0.0, not to -1.67
            for dim in CONTEXT_NORM_OVERRIDES:
                if dim < len(self.running_stat._M):
                    self.running_stat._M[dim] = 0.0
                    self.running_stat._S[dim] = 1.0
        else:
            for dim, (mean_val, var_val) in CONTEXT_NORM_OVERRIDES.items():
                if dim < len(self.running_stat._M):
                    self.running_stat._M[dim] = mean_val
                    self.running_stat._S[dim] = var_val

    def _repopulate_buffer_from_history(self, history_container):
        """[C2+ fix] Seed the fresh PER buffer with real (s, a, r, s') transitions from history.

        During initialize(), the _initializing guard prevents add_sample() from running, so
        the buffer starts with only the 50 anchor transitions. With 100 real steps in history,
        the Critic wastes the first 30-40 C2 updates on an anchor-dominated, biased sample.
        This method restores those transitions after _initializing is cleared.

        Priority is proportional to reward so good transitions dominate sampling."""
        if self.model is None or len(history_container.configurations) < 2:
            return
        try:
            configurations = configs2space(history_container.configurations, self.config_space)
        except Exception as e:
            self.logger.warning(f'[Buffer/Repop] configs2space failed: {e}')
            return

        n = len(configurations)
        injected = 0
        for i in range(n - 1):
            try:
                s = self._maybe_ablate(np.asarray(history_container.internal_metrics[i], dtype=np.float32))
                s_next = self._maybe_ablate(np.asarray(history_container.internal_metrics[i + 1], dtype=np.float32))
                if len(s) != self.metrics_num or len(s_next) != self.metrics_num:
                    continue
                if np.allclose(s, 0) or np.allclose(s_next, 0):
                    continue  # skip failed/empty trials
                a = config2action(configurations[i], self.config_space)
                r = float(-history_container.perfs[i])  # perfs store -reward
                priority = min(max(1.0, r), 150.0) if r > 0 else 1.0
                self.model.replay_memory.add(priority, (s, a, r, s_next, False))
                injected += 1
            except Exception:
                continue
        self.logger.info(f'[Buffer/Repop] Seeded {injected} real transitions from history into PER buffer (C{self.cycle_number})')

    def _inject_anchor(self):
        """[v6] Load top-N transitions from previous cycle and inject into fresh PER buffer.
        Gives the Critic a warm start with known-good tier-specific transitions instead of cold start."""
        if self.model is None or not self.fog_tier:
            self._current_anchor = []
            return
        anchor_file = f'{self.task_id}_{self.fog_tier}_anchor.pkl'
        if not os.path.exists(anchor_file):
            self._current_anchor = []
            self.logger.info(f'[v6/Anchor] No anchor for {self.fog_tier} (first cycle or new tier)')
            return
        try:
            with open(anchor_file, 'rb') as f:
                anchor = pickle.load(f)
            # Validate action dims before injecting — anchors from older runs may have
            # different action space size (e.g., v6 had 8 knobs, v7 has 6).
            n_actions = len(self.config_space.get_hyperparameter_names())
            valid = [(s, a, r, s2, d) for s, a, r, s2, d in anchor
                     if len(a) == n_actions and r > 105.0]
            discarded = len(anchor) - len(valid)
            if discarded > 0:
                self.logger.warning(f'[v6/Anchor] Discarded {discarded} transitions (wrong dims or reward≤105)')
            self._current_anchor = valid
            # Inject with high PER priority so Critic gets warm-started before first update.
            for s, a, r, s2, d in valid:
                self.model.replay_memory.add(100.0, (s, a, r, s2, d))
            self.logger.info(f'[v6/Anchor] Injected {len(valid)} warm transitions for {self.fog_tier} (high priority)')
        except Exception as e:
            self.logger.warning(f'[v6/Anchor] Inject failed: {e}')
            self._current_anchor = []

    def get_reward(self, external_metrics):

        def calculate_reward(delta0, deltat):
            if delta0 > 0:
                _reward = ((1 + delta0) ** 2 - 1) * math.fabs(1 + deltat)
            else:
                _reward = - ((1 - delta0) ** 2 - 1) * math.fabs(1 - deltat)

            if _reward > 0 and deltat < 0:
                _reward = 0
            return _reward

        if external_metrics == 0 or self.default_external_metrics == 0:
            # bad case, not enough time to restart mysql or bad knobs
            return 0
        # tps
        delta_0 = float((external_metrics - self.default_external_metrics)) / self.default_external_metrics
        delta_t = float((external_metrics - self.last_external_metrics)) / self.last_external_metrics
        reward = calculate_reward(delta_0, delta_t)
        self.score += reward

        return reward

    def create_initial_design(self, init_strategy='default', excluded_configs=None):
        default_config = self.config_space.get_default_configuration()
        num_random_config = self.init_num - 1
        if init_strategy == 'random':
            initial_configs = self.sample_random_configs(self.init_num, excluded_configs)
            return initial_configs
        elif init_strategy == 'default':
            initial_configs = [default_config] + self.sample_random_configs(num_random_config, excluded_configs)
            return initial_configs
        elif init_strategy == 'random_explore_first':
            candidate_configs = self.sample_random_configs(100, excluded_configs)
            return max_min_distance(default_config, candidate_configs, num_random_config)
        elif init_strategy == 'sobol':
            sobol = SobolSampler(self.config_space, num_random_config, random_state=self.rng)
            initial_configs = [default_config] + sobol.generate(return_config=True)
            return initial_configs
        elif init_strategy == 'latin_hypercube':
            lhs = LatinHypercubeSampler(self.config_space, num_random_config, criterion='maximin')
            initial_configs = [default_config] + lhs.generate(return_config=True)
            return initial_configs
        else:
            raise ValueError('Unknown initial design strategy: %s.' % init_strategy)

    def sample_random_configs(self, num_configs=1, excluded_configs=None):
        if excluded_configs is None:
            excluded_configs = set()

        configs = list()
        sample_cnt = 0
        max_sample_cnt = 1000
        while len(configs) < num_configs:
            config = self.config_space.sample_configuration()
            sample_cnt += 1
            if config not in configs and config not in excluded_configs:
                configs.append(config)
                sample_cnt = 0
                continue
            if sample_cnt >= max_sample_cnt:
                self.logger.warning('Cannot sample non duplicate configuration after %d iterations.' % max_sample_cnt)
                configs.append(config)
                sample_cnt = 0
        return configs



