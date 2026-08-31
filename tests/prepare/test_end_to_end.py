import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXAMPLE = os.path.join(ROOT, "scripts", "lab", "prepare", "example_sim.json")


class EndToEndTests(unittest.TestCase):
    def test_program_runs_on_simulator_and_resumes(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = json.load(open(EXAMPLE))
            cfg["n_targets"], cfg["out_dir"] = 4, d
            path = os.path.join(d, "cfg.json")
            json.dump(cfg, open(path, "w"))
            cmd = [sys.executable, os.path.join(ROOT, "scripts", "lab", "prepare", "prepare_eval.py"), path]
            out1 = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT).stdout
            self.assertIn("bisection", out1)
            self.assertTrue(os.path.exists(os.path.join(d, "results.json")))
            n_cached = len(os.listdir(os.path.join(d, "task_cache")))
            out2 = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT).stdout
            self.assertEqual(out1, out2)                       # rerun is fully served from the cache
            self.assertEqual(n_cached, len(os.listdir(os.path.join(d, "task_cache"))))


if __name__ == "__main__":
    unittest.main()
