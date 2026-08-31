import tempfile
import threading
import unittest

from autotune.prepare.executor import ResultCache
from autotune.prepare.tasks import EvalTask, TuneTask, interleave
from autotune.prepare.workload import Workload
from tests.prepare.helpers import make_sim


class ExecutorTests(unittest.TestCase):
    def test_batching_and_dedup(self):
        ex = make_sim()
        w = Workload.make("sysbench", rows=500, threads=64)
        f1, f2 = ex.submit(TuneTask(w)), ex.submit(TuneTask(w))
        self.assertEqual(ex.wait([f1, f2])[0].best_tps, f2.result.best_tps)
        self.assertEqual(ex.tune_calls, 1)                    # same task submitted twice -> run once

    def test_cache_makes_rerun_free(self):
        with tempfile.TemporaryDirectory() as d:
            w = Workload.make("sysbench", rows=500, threads=64)
            first = make_sim(cache=ResultCache(d))
            r1 = first.run_all([TuneTask(w)])[0]
            second = make_sim(cache=ResultCache(d))
            r2 = second.run_all([TuneTask(w)])[0]
            self.assertEqual(second.tune_calls, 0)
            self.assertEqual(r1.best_config, r2.best_config)

    def test_eval_keeps_order_and_repeats(self):
        ex = make_sim()
        w = Workload.make("sysbench", rows=500, threads=64)
        cfg = ex.run_all([TuneTask(w)])[0].best_config
        res = ex.run_all([EvalTask(w, interleave([cfg], 3))])[0]
        self.assertEqual(len(res.tps), 3)

    def test_task_in_flight_in_another_batch_is_joined_not_rerun(self):
        """Two threads asking for the same task while one batch is running -> one run."""
        started, release = threading.Event(), threading.Event()

        class SlowSim(type(make_sim())):
            def _run(self, tasks):
                started.set(); release.wait(timeout=10)
                return super()._run(tasks)

        ex = SlowSim(make_sim().surface)
        w = Workload.make("sysbench", rows=500, threads=64)
        results = []
        t1 = threading.Thread(target=lambda: results.append(ex.run_all([TuneTask(w)])[0]))
        t1.start(); started.wait(timeout=10)
        t2 = threading.Thread(target=lambda: results.append(ex.run_all([TuneTask(w)])[0]))
        t2.start(); release.set()
        t1.join(timeout=10); t2.join(timeout=10)
        self.assertEqual(len(results), 2)
        self.assertEqual(ex.tune_calls, 1)

    def test_concurrent_waiters(self):
        ex = make_sim()
        results = {}

        def worker(name, rows):
            results[name] = ex.run_all([TuneTask(Workload.make("sysbench", rows=rows, threads=64))])[0]

        threads = [threading.Thread(target=worker, args=(i, 200 + 100 * i)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertEqual(len(results), 4)


if __name__ == "__main__":
    unittest.main()
