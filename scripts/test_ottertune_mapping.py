"""Test OtterTune workload mapping with pruning for 2 iterations."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging
import numpy as np
from openbox.utils.config_space import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter
from openbox.utils.config_space import Configuration
from autotune.utils.history_container import HistoryContainer, Observation
from autotune.utils.constants import SUCCESS

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

NUM_METRICS = 20  # small number for testing
NUM_KNOBS = 4
NUM_SOURCE_OBS = 8
NUM_TARGET_OBS_ITER1 = 5
NUM_TARGET_OBS_ITER2 = 8  # grows as tuning progresses

np.random.seed(42)


def make_config_space():
    cs = ConfigurationSpace(seed=42)
    cs.add_hyperparameter(UniformIntegerHyperparameter('knob_buffer_pool', 128, 4096, default_value=256))
    cs.add_hyperparameter(UniformIntegerHyperparameter('knob_log_file_size', 16, 512, default_value=48))
    cs.add_hyperparameter(UniformFloatHyperparameter('knob_read_ahead', 0.0, 1.0, default_value=0.5))
    cs.add_hyperparameter(UniformIntegerHyperparameter('knob_io_threads', 1, 16, default_value=4))
    return cs


def make_observation(config, perf, num_metrics):
    """Create a mock Observation with random internal metrics."""
    im = np.random.randn(num_metrics) * 100 + 500  # random metrics around 500
    return Observation(
        config=config,
        trial_state=SUCCESS,
        constraints=None,
        objs=[-perf],  # negated (minimized internally)
        elapsed_time=60.0,
        iter_time=60.0,
        EM={'tps': perf, 'lat': 1.0 / max(perf, 1), 'qps': perf},
        IM=im,
        resource={'cpu': 50.0},
        info={'objs': ['tps'], 'constraints': []},
        context=None,
    )


def make_shared_configs(config_space, num_shared):
    """Generate a fixed set of configs shared between source and target."""
    configs = [config_space.get_default_configuration()]
    # Use a fixed seed so the same configs are generated for source and target
    rng = np.random.RandomState(123)
    for _ in range(num_shared - 1):
        config = config_space.sample_configuration()
        configs.append(config)
    return configs


def make_source_container(task_id, config_space, shared_configs, num_obs, num_metrics, metric_bias=0.0):
    """Create a source HistoryContainer with some observations.

    metric_bias shifts internal metrics — different biases simulate different workloads.
    Uses shared_configs first, then random configs for remaining observations.
    """
    hc = HistoryContainer(task_id, config_space=config_space)

    for i in range(num_obs):
        if i < len(shared_configs):
            config = shared_configs[i]
        else:
            config = config_space.sample_configuration()
        perf = np.random.uniform(50, 200)
        obs = make_observation(config, perf, num_metrics)
        obs = obs._replace(IM=obs.IM + metric_bias)
        hc.update_observation(obs)

    return hc


def make_target_container(task_id, config_space, shared_configs, num_obs, num_metrics, metric_bias=0.0):
    """Create a target HistoryContainer.

    Uses shared_configs first, then random configs for remaining observations.
    metric_bias controls how similar the target is to each source workload.
    """
    hc = HistoryContainer(task_id, config_space=config_space)

    for i in range(num_obs):
        if i < len(shared_configs):
            config = shared_configs[i]
        else:
            config = config_space.sample_configuration()
        perf = np.random.uniform(80, 180)
        obs = make_observation(config, perf, num_metrics)
        obs = obs._replace(IM=obs.IM + metric_bias)
        hc.update_observation(obs)

    return hc


def test_ottertune_mapping(prune_metrics):
    print(f"\n{'='*60}")
    print(f"Testing OtterTune mapping (prune_metrics={prune_metrics})")
    print(f"{'='*60}\n")

    cs = make_config_space()
    # Create 4 shared configs (default + 3 random) so exact matching has multiple pairs
    shared_configs = make_shared_configs(cs, num_shared=4)
    print(f"Shared configs: {len(shared_configs)} configs for exact matching")

    # Create 3 source workloads with different metric biases
    # Source "workload_A" has bias=0, "workload_B" has bias=200, "workload_C" has bias=500
    source_containers = [
        make_source_container('workload_A', cs, shared_configs, NUM_SOURCE_OBS, NUM_METRICS, metric_bias=0.0),
        make_source_container('workload_B', cs, shared_configs, NUM_SOURCE_OBS, NUM_METRICS, metric_bias=200.0),
        make_source_container('workload_C', cs, shared_configs, NUM_SOURCE_OBS, NUM_METRICS, metric_bias=500.0),
    ]

    # Target workload with bias=50 — should be closest to workload_A
    from autotune.transfer.tlbo.workload_map import WorkloadMapping

    wm = WorkloadMapping(
        config_space=cs,
        source_hpo_data=source_containers,
        seed=42,
        surrogate_type='gp',
        num_src_hpo_trial=-1,
        mapping_method='ottertune',
        prune_metrics=prune_metrics,
    )

    # --- Iteration 1 ---
    print(f"\n--- Iteration 1 ({NUM_TARGET_OBS_ITER1} target observations) ---")
    target_hc = make_target_container('target_task', cs, shared_configs,
                                       NUM_TARGET_OBS_ITER1, NUM_METRICS, metric_bias=50.0)
    wm.train(target_hc)
    print(f"Matched context: {wm.matched_context_id}")
    assert wm.matched_context_id == 'workload_A', \
        f"Expected workload_A (closest bias), got {wm.matched_context_id}"

    # Test prediction
    test_X = np.array([cs.sample_configuration().get_array() for _ in range(3)])
    mu, var = wm.predict(test_X)
    print(f"Prediction (3 test points): mu={mu.flatten()}, var={var.flatten()}")

    # --- Iteration 2 ---
    print(f"\n--- Iteration 2 ({NUM_TARGET_OBS_ITER2} target observations) ---")
    target_hc2 = make_target_container('target_task', cs, shared_configs,
                                        NUM_TARGET_OBS_ITER2, NUM_METRICS, metric_bias=50.0)
    wm.train(target_hc2)
    print(f"Matched context: {wm.matched_context_id}")
    assert wm.matched_context_id == 'workload_A', \
        f"Expected workload_A (closest bias), got {wm.matched_context_id}"

    mu2, var2 = wm.predict(test_X)
    print(f"Prediction (3 test points): mu={mu2.flatten()}, var={var2.flatten()}")

    print(f"\nTest passed! (prune_metrics={prune_metrics})")


if __name__ == '__main__':
    test_ottertune_mapping(prune_metrics=True)
    test_ottertune_mapping(prune_metrics=False)
    print("\n\nAll tests passed!")
