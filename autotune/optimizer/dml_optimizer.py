"""
DML (Deep Metric Learning) Optimizer for one-shot DB configuration transfer.

Uses a pre-trained triplet embedding model to find the most similar historical
context based on 11 DBMS/OS metrics (TPS excluded), then transfers the best-known
configuration from that context.

Flow:
  Iteration 0: Run default config to collect baseline metrics
  Iteration 1: Use model to find nearest context, then train a GP on that
               context's (config -> TPS) data and recommend the config with the
               highest GP-predicted TPS (NOT restricted to configs that were
               actually tried in the source context).
  Iteration 2+: Random sampling (placeholder for future iterative algorithm)
"""
import os
import sys
import logging
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from ConfigSpace import Configuration

from autotune.optimizer.dml_metrics import (
    METRIC_NAMES,
    collect_metrics_from_prometheus,
    extract_metrics_from_observation,
)

logger = logging.getLogger(__name__)


class _SimpleObs:
    """Lightweight stand-in for Observation namedtuple, used by extract_metrics_from_observation."""
    def __init__(self, em, resource, im):
        self.EM = em
        self.resource = resource
        self.IM = im


# 12 knob columns as they appear in the result CSVs
CONFIG_PARAMS = [
    'innodb_buffer_pool_size', 'innodb_read_io_threads', 'innodb_write_io_threads',
    'innodb_flush_log_at_trx_commit', 'innodb_adaptive_hash_index', 'sync_binlog',
    'innodb_lru_scan_depth', 'innodb_buffer_pool_instances', 'innodb_change_buffer_max_size',
    'innodb_io_capacity', 'innodb_log_file_size', 'table_open_cache'
]


class EmbeddingNet(nn.Module):
    """Same architecture as used in training (DBMSTransferLearning/dataset/train.py)."""
    def __init__(self, input_dim=11, embedding_dim=16):
        super(EmbeddingNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )

    def forward(self, x):
        output = self.net(x)
        return nn.functional.normalize(output, p=2, dim=1)


class DML_Optimizer:
    """
    One-shot optimizer that uses deep metric learning to transfer configurations.

    Args:
        config_space: ConfigSpace.ConfigurationSpace for the target DB
        history_container: HistoryContainer instance
        model_path: Path to trained context_model.pth
        context_metrics_path: Path to context_default_metrics_all.csv
        result_data_dir: Directory containing *-result.csv files
        knob_config_file: Path to knob JSON config (for sys.maxsize scaling info)
        prometheus_url: Optional Prometheus URL for metric collection
        mysql_instance: Prometheus MySQL exporter instance label
        node_instance: Prometheus node exporter instance label
    """
    def __init__(self, config_space, history_container,
                 model_path, context_metrics_path, result_data_dir,
                 knob_config_file=None, prometheus_url=None,
                 mysql_instance=None, node_instance=None, **kwargs):
        self.config_space = config_space
        self.prometheus_url = prometheus_url
        self.mysql_instance = mysql_instance
        self.node_instance = node_instance
        self.matched_context_id = None
        self.transferred_config = None
        # GP recommender config (iter 1 over the matched context's data)
        self.gp_max_train = int(kwargs.get('dml_gp_max_train', 600))
        self.gp_n_candidates = int(kwargs.get('dml_gp_n_candidates', 30000))

        # Load knob config for sys.maxsize scaling info
        self.knob_scaling = {}
        if knob_config_file and os.path.exists(knob_config_file):
            with open(knob_config_file) as f:
                knobs = json.load(f)
            for name, info in knobs.items():
                if info.get('type') == 'integer' and info.get('max', 0) > sys.maxsize:
                    self.knob_scaling[name] = 1000

        # Load model
        logger.info("Loading DML model from %s", model_path)
        self.model = EmbeddingNet(input_dim=11, embedding_dim=16)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

        # Load scaler
        scaler_path = os.path.join(os.path.dirname(model_path), 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info("Loaded scaler from %s", scaler_path)
        else:
            # Fallback: fit scaler on context metrics
            logger.warning("scaler.pkl not found, fitting scaler on context metrics")
            from sklearn.preprocessing import MinMaxScaler
            context_df = pd.read_csv(context_metrics_path)
            self.scaler = MinMaxScaler()
            self.scaler.fit(context_df[METRIC_NAMES])

        # Load context metrics and pre-compute embeddings
        self.context_df = pd.read_csv(context_metrics_path)
        self.context_ids = self.context_df['context_id'].tolist()
        features_scaled = self.scaler.transform(self.context_df[METRIC_NAMES])
        with torch.no_grad():
            self.context_embeddings = self.model(
                torch.tensor(features_scaled, dtype=torch.float32)
            ).numpy()
        logger.info("Pre-computed embeddings for %d reference contexts", len(self.context_ids))

        # Load result CSVs: best config per context AND all (config, TPS) rows
        # per context (the latter is the GP training data for iter 1).
        self.context_rows = {}
        self.best_configs = self._load_best_configs(result_data_dir)
        logger.info("Loaded best configs for %d contexts; GP training data for %d contexts",
                    len(self.best_configs), len(self.context_rows))

    def _load_best_configs(self, result_data_dir):
        """Load all result CSVs and find the best config (by TPS) per context_id.

        Also populates self.context_rows[context_id] = {'knobs': (n,k) array,
        'cols': [knob names], 'tps': (n,) array} with every tried row, used to
        train a per-context GP at recommendation time.
        """
        best_configs = {}
        csv_files = [f for f in os.listdir(result_data_dir) if f.endswith('-result.csv')]

        for csv_file in csv_files:
            filepath = os.path.join(result_data_dir, csv_file)
            try:
                df = pd.read_csv(filepath)
            except Exception as e:
                logger.warning("Failed to read %s: %s", filepath, e)
                continue

            if 'tps' not in df.columns or 'workload_label' not in df.columns:
                continue

            # Hardware ID from filename (e.g., '8c12g' from '8c12g-result.csv')
            hw_id = csv_file.replace('-result.csv', '')
            present_params = [p for p in CONFIG_PARAMS if p in df.columns]

            # Group by workload and find best config per context
            for wl, group in df.groupby('workload_label'):
                context_id = f"{hw_id}_{wl}"
                best_row = group.loc[group['tps'].idxmax()]

                # Extract knob values
                knob_values = {}
                for param in CONFIG_PARAMS:
                    if param in best_row.index:
                        knob_values[param] = best_row[param]

                best_configs[context_id] = {
                    'knobs': knob_values,
                    'tps': best_row['tps'],
                }

                # Store all rows for GP training (vectorized — cheap)
                self.context_rows[context_id] = {
                    'knobs': group[present_params].to_numpy(dtype=float),
                    'cols': present_params,
                    'tps': group['tps'].to_numpy(dtype=float),
                }

        return best_configs

    def _recommend_via_gp(self, context_id):
        """Train a GP on the matched context's (config -> TPS) data and return
        the candidate configuration with the highest GP-predicted TPS.

        Returns (Configuration, predicted_tps) or (None, None) to signal the
        caller to fall back to the best-tried config.
        """
        data = self.context_rows.get(context_id)
        if not data or len(data['tps']) < 5:
            return None, None

        knobs_mat, cols, tps = data['knobs'], data['cols'], data['tps']
        n = len(tps)

        # Subsample for tractable GP fit: keep the top-half by TPS (so the GP
        # sees the high-performing region) plus a random sample of the rest.
        rng = np.random.RandomState(0)
        if n > self.gp_max_train:
            order = np.argsort(tps)[::-1]
            n_top = self.gp_max_train // 2
            top = order[:n_top]
            rest_pool = order[n_top:]
            n_rand = self.gp_max_train - n_top
            rand = rng.choice(rest_pool, size=min(n_rand, len(rest_pool)), replace=False)
            sel = np.concatenate([top, rand])
        else:
            sel = np.arange(n)

        # Convert each selected source row to a Configuration + normalized array,
        # dropping any non-finite rows (defensive).
        train_configs, X_list, y_list = [], [], []
        for i in sel:
            kv = {cols[j]: knobs_mat[i, j] for j in range(len(cols))}
            c = self._knob_values_to_configuration(kv)
            arr = c.get_array()
            if np.isfinite(arr).all() and np.isfinite(tps[i]):
                train_configs.append(c)
                X_list.append(arr)
                y_list.append(float(tps[i]))
        if len(y_list) < 5:
            return None, None
        X_train = np.asarray(X_list, dtype=float)
        y_train = np.asarray(y_list, dtype=float)

        try:
            kernel = (ConstantKernel(1.0, (1e-2, 1e3))
                      * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
                      + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e5)))
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                          n_restarts_optimizer=1, random_state=0)
            gp.fit(X_train, y_train)
        except Exception as e:
            logger.warning("DML GP fit failed (%s); falling back to best-tried config", e)
            return None, None

        # Candidate pool: random samples across the space (novel configs) + the
        # context's tried configs (so we never recommend worse than best-tried
        # in GP terms) + the default config.
        cands = self.config_space.sample_configuration(self.gp_n_candidates)
        if not isinstance(cands, list):
            cands = [cands]
        cands.extend(train_configs)
        cands.append(self.config_space.get_default_configuration())
        X_cand = np.asarray([c.get_array() for c in cands], dtype=float)

        mu = gp.predict(X_cand)
        best_i = int(np.argmax(mu))
        return cands[best_i], float(mu[best_i])

    def _recommend_via_ottertune(self, context_id, history_container):
        """OtterTune-style downstream for the SAME matched source context.

        Differs from _recommend_via_gp only in the downstream optimizer:
          - trains the GP on the source context's (config->TPS) rows
            CONCATENATED with the target's observed point(s) (as OtterTune's
            WorkloadMapping does), and
          - recommends by maximizing Expected Improvement (EI) — exploration,
            like OtterTune's BO acquisition — rather than argmax of the mean.

        This isolates the effect of the source-selection step: DML's embedding
        picks the source, but the recommendation mechanics match OtterTune.
        Returns (Configuration, predicted_tps) or (None, None) for fallback.
        """
        from scipy.stats import norm
        data = self.context_rows.get(context_id)
        if not data or len(data['tps']) < 5:
            return None, None
        knobs_mat, cols, tps = data['knobs'], data['cols'], data['tps']
        n = len(tps)

        rng = np.random.RandomState(0)
        if n > self.gp_max_train:
            order = np.argsort(tps)[::-1]
            n_top = self.gp_max_train // 2
            sel = np.concatenate([order[:n_top],
                                  rng.choice(order[n_top:],
                                             size=min(self.gp_max_train - n_top, n - n_top),
                                             replace=False)])
        else:
            sel = np.arange(n)

        train_configs, X_list, y_list = [], [], []
        for i in sel:
            kv = {cols[j]: knobs_mat[i, j] for j in range(len(cols))}
            c = self._knob_values_to_configuration(kv)
            arr = c.get_array()
            if np.isfinite(arr).all() and np.isfinite(tps[i]):
                train_configs.append(c); X_list.append(arr); y_list.append(float(tps[i]))
        if len(y_list) < 5:
            return None, None

        # Concatenate the target's observed point(s) — OtterTune builds its
        # surrogate on target + mapped source data.
        if history_container is not None and history_container.configurations:
            ems = history_container.external_metrics or []
            for k, cfg in enumerate(history_container.configurations):
                t = None
                if k < len(ems) and isinstance(ems[k], dict):
                    t = ems[k].get('tps')
                if t is None:
                    continue
                arr = cfg.get_array()
                if np.isfinite(arr).all() and np.isfinite(t):
                    X_list.append(arr); y_list.append(float(t))

        X = np.asarray(X_list, dtype=float)
        y = np.asarray(y_list, dtype=float)

        try:
            kernel = (ConstantKernel(1.0, (1e-2, 1e3))
                      * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
                      + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e5)))
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                          n_restarts_optimizer=1, random_state=0)
            gp.fit(X, y)
        except Exception as e:
            logger.warning("DML OtterTune-downstream GP fit failed (%s); falling back", e)
            return None, None

        # Expected Improvement over candidates (maximizing TPS).
        f_best = float(np.max(y))
        cands = self.config_space.sample_configuration(self.gp_n_candidates)
        if not isinstance(cands, list):
            cands = [cands]
        cands.extend(train_configs)
        cands.append(self.config_space.get_default_configuration())
        X_cand = np.asarray([c.get_array() for c in cands], dtype=float)

        mu, sigma = gp.predict(X_cand, return_std=True)
        sigma = np.clip(sigma, 1e-9, None)
        z = (mu - f_best) / sigma
        ei = (mu - f_best) * norm.cdf(z) + sigma * norm.pdf(z)
        best_i = int(np.argmax(ei))
        logger.info("DML OtterTune-downstream: EI-best predicted TPS=%.2f (f_best=%.2f)",
                    float(mu[best_i]), f_best)
        return cands[best_i], float(mu[best_i])

    def _knob_values_to_configuration(self, knob_values):
        """
        Convert a dict of knob values from the result CSV to a Configuration object.

        Only sets knobs that exist in both the result CSV and the target config_space.
        Values are clamped to ConfigSpace bounds. Unmatched knobs use defaults.
        """
        hp_dict = {}
        default_config = self.config_space.get_default_configuration()

        for hp in self.config_space.get_hyperparameters():
            name = hp.name
            if name in knob_values:
                raw_value = knob_values[name]

                if hasattr(hp, 'choices'):
                    # CategoricalHyperparameter
                    str_val = str(int(raw_value)) if isinstance(raw_value, float) else str(raw_value)
                    if str_val in hp.choices:
                        hp_dict[name] = str_val
                    else:
                        hp_dict[name] = default_config[name]
                elif hasattr(hp, 'lower'):
                    # Numeric hyperparameter
                    val = raw_value
                    # CSV stores buffer_pool_size in GB, log_file_size as MB/1000
                    if name == 'innodb_buffer_pool_size':
                        val = val * 1073741824  # GB to bytes
                    elif name == 'innodb_log_file_size':
                        val = val * 1000 * 1024 * 1024  # MB/1000 to bytes
                    # Handle sys.maxsize scaling: ConfigSpace uses value/1000
                    if name in self.knob_scaling:
                        val = val / self.knob_scaling[name]
                    # Clamp to bounds
                    val = max(hp.lower, min(hp.upper, val))
                    if hasattr(hp, 'default_value') and isinstance(hp.default_value, int):
                        val = int(round(val))
                    hp_dict[name] = val
                else:
                    hp_dict[name] = default_config[name]
            else:
                hp_dict[name] = default_config[name]

        try:
            return Configuration(self.config_space, values=hp_dict)
        except Exception as e:
            logger.warning("Failed to create Configuration from transferred knobs: %s", e)
            logger.warning("Falling back to default configuration")
            return default_config

    def get_suggestion(self, history_container=None, compact_space=None):
        """
        Return the next configuration to evaluate.

        Iteration 0: Default config (to collect baseline metrics)
        Iteration 1: Best config from the nearest historical context
        Iteration 2+: Random sampling (placeholder)
        """
        iteration = len(history_container.configurations) if history_container else 0

        if iteration == 0:
            logger.info("DML Iteration 0: returning default config to collect baseline metrics")
            return self.config_space.get_default_configuration()

        if iteration == 1:
            # Extract metrics from the first evaluation (default config run)
            # HistoryContainer stores metrics in separate lists, not as Observation objects
            em = history_container.external_metrics[0] if history_container.external_metrics else {}
            resource = history_container.resource[0] if history_container.resource else {}
            im = history_container.internal_metrics[0] if history_container.internal_metrics else []

            if self.prometheus_url:
                try:
                    target_metrics = collect_metrics_from_prometheus(
                        self.prometheus_url,
                        self.mysql_instance, self.node_instance
                    )
                except Exception as e:
                    logger.warning("Prometheus collection failed: %s. Falling back to collected metrics.", e)
                    obs = _SimpleObs(em, resource, im)
                    target_metrics = extract_metrics_from_observation(obs)
            else:
                obs = _SimpleObs(em, resource, im)
                target_metrics = extract_metrics_from_observation(obs)

            # Scale and embed
            scaled = self.scaler.transform(target_metrics.reshape(1, -1))
            with torch.no_grad():
                target_embedding = self.model(
                    torch.tensor(scaled, dtype=torch.float32)
                ).numpy()

            # Find nearest context
            distances = cdist(target_embedding, self.context_embeddings, metric='euclidean').flatten()
            sorted_indices = np.argsort(distances)

            for idx in sorted_indices:
                context_id = self.context_ids[idx]
                if context_id in self.best_configs:
                    self.matched_context_id = context_id
                    break

            # Manual override: FORCE_SOURCE_CONTEXT pins the source context (for
            # sensitivity analysis — does the source choice change the result?).
            forced = os.environ.get('FORCE_SOURCE_CONTEXT')
            if forced:
                import re
                def _compact(c):
                    return re.sub(r'_\d+-\d+-\d+-oltp_', '_', c.replace('history_', ''))
                cands = [c for c in self.context_rows if _compact(c) == forced] \
                        or [c for c in self.context_rows if forced in _compact(c)]
                if cands:
                    cands.sort(key=lambda c: ('1000000' not in c, c))  # prefer 1M table size
                    logger.info("DML FORCED source context: %s -> %s (was %s)",
                                forced, cands[0], self.matched_context_id)
                    self.matched_context_id = cands[0]
                else:
                    logger.warning("DML FORCE_SOURCE_CONTEXT=%s matched no context; using %s",
                                   forced, self.matched_context_id)

            if self.matched_context_id is None:
                logger.warning("No matching context found, returning random config")
                return self.config_space.sample_configuration()

            best = self.best_configs.get(self.matched_context_id, {'tps': float('nan')})
            try:
                dist = distances[self.context_ids.index(self.matched_context_id)]
            except ValueError:
                dist = float('nan')  # forced context may not be in the embedding reference set
            logger.info("DML matched context: %s (best-tried TPS=%.2f, distance=%.4f)",
                        self.matched_context_id, best['tps'], dist)

            # Downstream recommendation on the matched context. Selectable via
            # DML_DOWNSTREAM env: 'gp' (default; GP argmax-mean on source data)
            # or 'ottertune' (GP on target+source + EI, mirroring OtterTune) —
            # lets us isolate the effect of source selection vs downstream.
            downstream = os.environ.get('DML_DOWNSTREAM', 'gp').lower()
            if downstream == 'ottertune':
                rec_config, pred_tps = self._recommend_via_ottertune(
                    self.matched_context_id, history_container)
                tag = 'OtterTune-downstream'
            else:
                rec_config, pred_tps = self._recommend_via_gp(self.matched_context_id)
                tag = 'GP'
            if rec_config is not None:
                self.transferred_config = rec_config
                logger.info("DML %s recommendation: predicted TPS=%.2f (best-tried in context=%.2f)",
                            tag, pred_tps, best['tps'])
                logger.info("Transferred config (%s): %s", tag, dict(self.transferred_config))
            else:
                # Fallback: transfer the best config actually tried in the context
                self.transferred_config = self._knob_values_to_configuration(best['knobs'])
                logger.info("DML GP unavailable; transferred best-tried config: %s",
                            dict(self.transferred_config))

            return self.transferred_config

        # Iteration 2+: placeholder for future iterative algorithm
        logger.info("DML Iteration %d: returning random config (placeholder)", iteration)
        return self.config_space.sample_configuration()

    def update(self, observation):
        """No-op for now. Future iterative algorithms can override this."""
        pass
