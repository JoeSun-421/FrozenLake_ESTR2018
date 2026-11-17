import unittest
import numpy as np
from frozenlake.environment import build_model, generate_map, shortest_path


class EnvironmentTests(unittest.TestCase):
    def test_original_map_and_fixed_density(self):
        rows, _ = generate_map(8, 2018, .1)
        self.assertEqual(rows, ["SFFFFFHF", "FHFHFFFF", "FFHFFFFF", "FFFFFHFF", "FFFFFFFF", "FFFFFFFF", "FFFFFFFF", "FHFFFFFG"])
        for n in [8, 12, 16]:
            for seed in [2018, 2019, 2020]:
                a = generate_map(n, seed, .1)[0]
                self.assertEqual(a, generate_map(n, seed, .1)[0])
                self.assertEqual(sum(r.count("H") for r in a), round(.1 * (n * n - 2)))
                self.assertIsNotNone(shortest_path(a))
        self.assertIsNone(shortest_path(["SH", "HG"]))

    def test_boundary_mass_and_terminal_reward(self):
        p, ns, r, done = build_model(["SF", "HG"], 1 / 3)
        np.testing.assert_allclose(p.sum(axis=-1), 1)
        self.assertAlmostEqual(p[0, 0][ns[0, 0] == 0].sum(), 2 / 3)
        self.assertTrue((r[ns == 2] == 0).all())
        self.assertTrue(done[ns == 2].all())
        self.assertEqual(r[1, 1].max(), 1)
        self.assertTrue((r[3] == 0).all())


if __name__ == "__main__":
    unittest.main()
