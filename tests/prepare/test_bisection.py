import math
import unittest

from autotune.prepare.methods.bisection import BisectionPrepare
from autotune.prepare.methods.random_sample import RandomPrepare
from autotune.prepare.workload import Workload
from tests.prepare.helpers import BELIEFS, SPACE, make_sim


def prepare(method, ex):
    return method.prepare(BELIEFS, ex, {})


class BisectionTests(unittest.TestCase):
    def test_insensitive_side_costs_one_tune_and_infinite_radius(self):
        ex = make_sim({"rows:down": None, "rows:up": None, "threads:down": None, "threads:up": None})
        repo = prepare(BisectionPrepare(theta=0.9, delta=0.1, max_probes=3, repeats=1), ex)
        self.assertEqual(repo.ledger.tune_calls, 1 + 4)      # anchor + one edge probe per side, no fill
        self.assertEqual(len(repo), 5)                         # every Tune result is kept
        for e in repo.entries:                                 # radius = gap to the passing edge probe
            if e.role == "probe_pass":
                for param in ("rows", "threads"):
                    gap = abs(e.workload.get(param) - 500 if param == "rows" else e.workload.get(param) - 64)
                    if gap:
                        direction = "down" if (e.workload.get(param) < (500 if param == "rows" else 64)) else "up"
                        self.assertAlmostEqual(repo.radius(param, direction), gap)

    def test_bisection_finds_planted_radius(self):
        ex = make_sim({"rows:down": 120, "rows:up": None, "threads:down": None, "threads:up": None})
        repo = prepare(BisectionPrepare(theta=0.9, delta=0.1, max_probes=4, repeats=1, fill=False), ex)
        radius = repo.radius("rows", "down")
        failed_gaps = [500 - e.workload.get("rows") for e in repo.entries if e.role == "probe" and e.workload.get("rows") < 500]
        passed = [e for e in repo.entries if e.role == "probe_pass" and e.workload.get("rows") < 500]
        self.assertTrue(all(g > 120 for g in failed_gaps))      # only gaps beyond the true radius fail
        self.assertTrue(0 < radius <= 120 * 1.25)               # pass iff gap/r <= 1 + (1-theta)/slope
        self.assertAlmostEqual(radius, 500 - passed[0].workload.get("rows"))   # radius = first passing gap
        self.assertAlmostEqual(repo.radius("rows", "up"), 850 - 500)           # insensitive side: gap to the edge

    def test_k_cap_uses_half_innermost_failing_gap(self):
        ex = make_sim({"rows:down": 1, "rows:up": None, "threads:down": None, "threads:up": None})
        repo = prepare(BisectionPrepare(theta=0.9, delta=0.1, max_probes=2, repeats=1, fill=False), ex)
        probes = [e for e in repo.entries if e.role == "probe" and e.workload.get("threads") == 64]
        self.assertEqual(len(probes), 2)
        innermost_gap = min(500 - e.workload.get("rows") for e in probes)
        self.assertAlmostEqual(repo.radius("rows", "down"), innermost_gap / 2)

    def test_mode_at_boundary_skips_side(self):
        from autotune.prepare.belief import BeliefSet, Triangular, Uniform
        beliefs = BeliefSet("sysbench", SPACE, {"rows": Triangular(100, 100, 1000), "threads": Uniform(8, 256, mode=64)})
        ex = make_sim()
        repo = BisectionPrepare(repeats=1, fill=False).prepare(beliefs, ex, {})
        self.assertTrue(math.isinf(repo.radius("rows", "down")))

    def test_fill_covers_interval_at_radius_spacing(self):
        ex = make_sim({"rows:down": 100, "rows:up": None, "threads:down": None, "threads:up": None})
        repo = prepare(BisectionPrepare(theta=0.9, delta=0.1, max_probes=4, repeats=1), ex)
        r = repo.radius("rows", "down")
        fills = sorted(e.workload.get("rows") for e in repo.entries if e.role == "fill")
        lo, _ = BELIEFS.covered_interval("rows", 0.1)
        self.assertTrue(fills, "expected fill points on the sensitive side (lo=%s r=%s)" % (lo, r))
        self.assertTrue(all(lo <= v < 500 for v in fills), (fills, lo))
        multiples = [(500 - v) / r for v in fills]
        self.assertTrue(all(abs(m - round(m)) < 0.02 for m in multiples), multiples)   # at radius spacing

    def test_recommend_nearest_and_radius_rules(self):
        ex = make_sim()
        method = BisectionPrepare(repeats=1)
        repo = prepare(method, ex)
        target = Workload.make("sysbench", rows=180, threads=64)
        nearest = method.recommend(repo, target, SPACE)
        method.selection = "radius"
        scaled = method.recommend(repo, target, SPACE)
        self.assertIn(nearest, repo.entries)
        self.assertIn(scaled, repo.entries)


class RandomTests(unittest.TestCase):
    def test_matches_other_methods_size(self):
        ex = make_sim()
        base = prepare(BisectionPrepare(repeats=1), ex)
        rnd = RandomPrepare(match="bisection", seed=3).prepare(BELIEFS, ex, {"bisection": base})
        self.assertEqual(len(rnd), len(base))
        self.assertEqual(rnd.ledger.tune_calls, len(base))


if __name__ == "__main__":
    unittest.main()
