import unittest
from pathlib import Path
from frozenlake.configuration import load, merge
from frozenlake.comparison import assert_matched

ROOT = Path(__file__).resolve().parents[1]


class ComparisonTests(unittest.TestCase):
    def test_initialization_is_the_only_allowed_protocol_change(self):
        baseline = load(ROOT / "config/baseline_16.json")
        optimistic = load(ROOT / "config/optimistic_16.json")
        assert_matched(baseline, optimistic)
        for overlay in [{"training": {"interactions": 1000000}}, {"environment": {"intended_probability": .8}},
                        {"groups": {"main": {"training_seeds": [99]}}}, {"evaluation": {"episodes": 10000}}]:
            with self.assertRaises(ValueError):
                assert_matched(baseline, merge(optimistic, overlay))


if __name__ == "__main__":
    unittest.main()
