import unittest
import numpy as np
from frozenlake.environment import build_model
from frozenlake.evaluation import exact_policy, monte_carlo


class EvaluationTests(unittest.TestCase):
    def test_hand_solved_deterministic_policy_and_cutoff(self):
        model = build_model(["SF", "HG"], 1.)
        q = np.zeros((4, 4))
        q[0, 2] = q[1, 1] = 1
        before = q.copy()
        short = exact_policy(q, model, 1, .9)
        self.assertEqual(short["success_probability"], 0)
        self.assertEqual(short["truncation_probability"], 1)
        result = exact_policy(q, model, 2, .9)
        self.assertEqual(result["success_probability"], 1)
        self.assertAlmostEqual(result["discounted_return"], .9)
        self.assertEqual(result["mean_episode_length"], 2)
        self.assertEqual(result["successful_episode_length"], 2)
        np.testing.assert_array_equal(q, before)

    def test_uniform_policy_one_step_and_independent_sampling(self):
        model = build_model(["SF", "HG"], 1.)
        q = np.zeros((4, 4))
        result = exact_policy(q, model, 1, .9)
        self.assertAlmostEqual(result["failure_probability"], .25)
        self.assertAlmostEqual(result["truncation_probability"], .75)
        samples = monte_carlo(q, model, 10, .9, 20000, 123)
        expected = exact_policy(q, model, 10, .9)
        self.assertLess(abs(samples[:, 0].mean() - expected["success_probability"]), .015)
        self.assertLess(abs(samples[:, 1].mean() - expected["discounted_return"]), .015)
        np.testing.assert_array_equal(samples, monte_carlo(q, model, 10, .9, 20000, 123))


if __name__ == "__main__":
    unittest.main()
