import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "benchmarks")]
import correction_eval as C  # noqa: E402


class CorrectionEvalTest(unittest.TestCase):
    def test_m0_check_finds_mismatch(self):
        self.assertEqual(C.rank_mismatch({"a": 1, "b": 2}, {"a": 1, "b": 2}), [])
        self.assertEqual(C.rank_mismatch({"a": 1, "b": 2}, {"a": 2, "b": 1}), ["a", "b"])
        self.assertEqual(C.rank_mismatch({"a": 1}, {"a": 1, "c": 3}), ["c"])

    def test_m2_neighbours_are_nearest_degree(self):
        deg = np.array([1, 2, 3, 10, 11, 50])
        nb = C.nearest(deg, 3)
        self.assertEqual(set(nb[0]), {0, 1, 2})
        self.assertEqual(set(nb[3]), {3, 4, 2})          # 10: itself, 11, then 3 (distance 7)
        self.assertEqual(set(nb[5]), {5, 4, 3})
        obs = np.array([0.0, 0.0, 0.0, 1.0, 3.0, 5.0])
        z = C.m2_z(obs, nb)
        self.assertAlmostEqual(z[5], (5 - 3) / 2)        # population 1, 3, 5: mean 3, SD (n - 1) 2

    def test_phipson_smyth_never_zero(self):
        np.testing.assert_allclose(C.phipson_smyth([0, 1000, 9], 1000), [1 / 1001, 1.0, 10 / 1001])

    def test_m1_rules(self):
        z = np.array([2.0, np.nan, 0.0, np.nan])
        raw = np.array([1.0, 4.0, 3.0, 2.0])
        # D22: defined z ranked among themselves and scaled by share 0.5; undefined keep raw percentile.
        np.testing.assert_allclose(C.m1_d22(z, raw), [0.5, 1.0, 0.25, 0.5])
        # last: undefined below every defined z.
        np.testing.assert_allclose(C.m1_last(z, raw), [1.0, 0.5, 0.75, 0.25])

    def test_uni_c(self):
        r_raw, r_adj = np.array([1, 2, 3, 4]), np.array([4, 3, 2, 1])
        deg = np.array([10, 1, 10, 1])                    # nodes 0, 2 high degree -> raw; 1, 3 -> adjusted
        # used ranks 1, 3, 3, 1; unused 4, 2, 2, 4 tie too, so node order decides.
        np.testing.assert_array_equal(C.uniform(r_raw, r_adj, "UNI-C", deg, 5), [1, 3, 4, 2])

    def test_uni_o(self):
        r_raw, r_adj = np.array([1, 4, 2, 3]), np.array([3, 1, 4, 2])
        # best of the two: 1, 1, 2, 2; unused: 3, 4, 4, 3
        np.testing.assert_array_equal(C.uniform(r_raw, r_adj, "UNI-O"), [1, 2, 4, 3])

    def test_uni_s(self):
        r_raw, r_adj = np.array([1, 2, 3]), np.array([3, 2, 1])
        np.testing.assert_array_equal(C.uniform(r_raw, r_adj, "UNI-S", seed_degree=9, net_degree=5), r_raw)
        np.testing.assert_array_equal(C.uniform(r_raw, r_adj, "UNI-S", seed_degree=3, net_degree=5), r_adj)


if __name__ == "__main__":
    unittest.main()
