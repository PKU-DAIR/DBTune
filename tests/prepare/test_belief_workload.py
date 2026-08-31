import math
import random
import unittest

from autotune.prepare.belief import Empirical, Triangular, Uniform
from autotune.prepare.workload import Workload
from tests.prepare.helpers import BELIEFS, SPACE


class BeliefTests(unittest.TestCase):
    def test_quantile_roundtrip(self):
        for belief in (Uniform(0, 10), Triangular(0, 3, 10), Empirical([1, 2, 3, 5, 8, 13])):
            for q in (0.05, 0.3, 0.5, 0.95):
                self.assertAlmostEqual(belief.quantile_of(belief.quantile(q)), q, places=6)

    def test_triangular_mode_and_edges(self):
        t = Triangular(0, 3, 10)
        self.assertEqual(t.mode, 3)
        self.assertAlmostEqual(t.quantile(0), 0)
        self.assertAlmostEqual(t.quantile(1), 10)
        self.assertAlmostEqual(t.quantile_of(3), 0.3)      # mass below the mode = (m-lo)/(hi-lo)

    def test_covered_interval_and_mode_workload(self):
        lo, hi = BELIEFS.covered_interval("rows", 0.1)
        self.assertLess(lo, 500)
        self.assertGreater(hi, 500)
        self.assertEqual(BELIEFS.mode_workload(), Workload.make("sysbench", rows=500, threads=64))

    def test_sample_is_seeded_and_coerced(self):
        a = BELIEFS.sample(5, random.Random(1))
        b = BELIEFS.sample(5, random.Random(1))
        self.assertEqual(a, b)
        for w in a:
            self.assertEqual(w.get("rows"), round(w.get("rows")))
            self.assertTrue(100 <= w.get("rows") <= 1000)


class WorkloadTests(unittest.TestCase):
    def test_normalize_and_distance(self):
        a, b = Workload.make("s", rows=100, threads=8), Workload.make("s", rows=1000, threads=256)
        self.assertEqual(SPACE.normalize(a), (0.0, 0.0))
        self.assertEqual(SPACE.normalize(b), (1.0, 1.0))
        self.assertAlmostEqual(SPACE.distance(a, b), math.sqrt(2))
        self.assertTrue(SPACE.same_point(a, Workload.make("s", threads=8, rows=100)))

    def test_with_param_and_key(self):
        w = Workload.make("s", rows=100, threads=8)
        w2 = w.with_param("rows", 200)
        self.assertEqual(w2.get("rows"), 200)
        self.assertNotEqual(w.key(), w2.key())
        self.assertEqual(Workload.from_dict(w.as_dict()), w)


if __name__ == "__main__":
    unittest.main()
