import unittest
from pathlib import Path
from frozenlake.configuration import load, merge, validate, alpha_at, epsilon_at
from frozenlake.initialization import initial_q
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_baseline_schedules_and_counts(self):
        cfg = load(ROOT / "config/baseline_8.json")
        self.assertEqual(alpha_at(cfg, 37499), .7)
        self.assertEqual(alpha_at(cfg, 37500), .1)
        self.assertAlmostEqual(epsilon_at(cfg, 25000), .05)
        self.assertAlmostEqual(epsilon_at(cfg, 100000), .05)
        self.assertEqual(sum(len(g["map_seeds"]) * len(g["training_seeds"]) for g in cfg["groups"].values()), 18)

    def test_matched_optimistic_overlay(self):
        base = load(ROOT / "config/baseline_16.json")
        candidate = load(ROOT / "config/optimistic_16.json")
        for key in base:
            if key not in ["name", "variant", "initialization"]:
                self.assertEqual(base[key], candidate[key], key)
        terminal = np.array([[False, True, False, True]])
        np.testing.assert_array_equal(initial_q(base, terminal), np.zeros((1, 4, 4)))
        q = initial_q(candidate, terminal)
        self.assertTrue((q[0, [0, 2]] == 1).all())
        self.assertTrue((q[0, [1, 3]] == 0).all())

    def test_no_optimistic_baseline_or_invalid_checkpoints(self):
        base = load(ROOT / "config/baseline_8.json")
        with self.assertRaises(ValueError):
            validate(merge(base, {"initialization": {"method": "optimistic", "value": 1}}))
        with self.assertRaises(ValueError):
            validate(merge(base, {"presentation": {"snapshots": [3]}}))


if __name__ == "__main__":
    unittest.main()
