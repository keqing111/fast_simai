import unittest

import numpy as np

from simai_a2a.traffic import (
    balanced_matrix,
    fast_uniform_matrix,
    fast_zipf_matrix,
    generate_matrix,
    zipf_inverse_cdf_fast,
)


class TrafficGeneratorTests(unittest.TestCase):
    def test_balanced_matrix_has_exact_pair_bytes(self):
        matrix = balanced_matrix(3, pair_bytes=4)

        self.assertEqual(matrix.logical_bytes, 24)
        self.assertEqual(np.count_nonzero(matrix.values), 6)
        np.testing.assert_array_equal(np.diag(matrix.values), np.zeros(3, dtype=np.int64))

    def test_uniform_is_seeded_and_preserves_sampled_zero_cells(self):
        first = fast_uniform_matrix(128, bytes_per_unit=3, seed=0)
        repeated = fast_uniform_matrix(128, bytes_per_unit=3, seed=0)
        different = fast_uniform_matrix(128, bytes_per_unit=3, seed=1)

        np.testing.assert_array_equal(first.values, repeated.values)
        self.assertFalse(np.array_equal(first.values, different.values))
        self.assertGreaterEqual(int(first.values.min()), 0)
        self.assertLessEqual(int(first.values.max()), 9999 * 3)
        off_diagonal = ~np.eye(128, dtype=bool)
        self.assertTrue(np.any(first.values[off_diagonal] == 0))
        np.testing.assert_array_equal(np.diag(first.values), np.zeros(128, dtype=np.int64))

    def test_zipf_inverse_cdf_matches_fast_literals(self):
        actual = [
            zipf_inverse_cdf_fast(probability, 10000, 0.9)
            for probability in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        self.assertEqual(actual, [1, 18, 236, 1830, 9999])

    def test_zipf_is_seeded_and_accepts_alpha_near_one(self):
        first = fast_zipf_matrix(8, alpha=0.999, bytes_per_unit=2, seed=7)
        repeated = fast_zipf_matrix(8, alpha=0.999, bytes_per_unit=2, seed=7)
        different = fast_zipf_matrix(8, alpha=0.999, bytes_per_unit=2, seed=8)

        np.testing.assert_array_equal(first.values, repeated.values)
        self.assertFalse(np.array_equal(first.values, different.values))
        np.testing.assert_array_equal(np.diag(first.values), np.zeros(8, dtype=np.int64))

    def test_fixed_total_is_exact_deterministic_and_keeps_zero_diagonal(self):
        cases = (
            balanced_matrix(4, total_bytes=101),
            fast_uniform_matrix(4, bytes_per_unit=1, seed=11, total_bytes=101),
            fast_zipf_matrix(4, alpha=0.9, bytes_per_unit=1, seed=11, total_bytes=101),
        )
        for matrix in cases:
            with self.subTest(generator=matrix.metadata["generator"]):
                self.assertEqual(matrix.logical_bytes, 101)
                self.assertEqual(matrix.metadata["normalization"], "fixed-total")
                np.testing.assert_array_equal(
                    np.diag(matrix.values), np.zeros(4, dtype=np.int64)
                )
        repeated = fast_uniform_matrix(
            4, bytes_per_unit=1, seed=11, total_bytes=101
        )
        np.testing.assert_array_equal(cases[1].values, repeated.values)

    def test_invalid_dimensions_scales_and_alpha_are_rejected(self):
        invalid_calls = [
            lambda: balanced_matrix(1, pair_bytes=1),
            lambda: balanced_matrix(2, pair_bytes=0),
            lambda: balanced_matrix(2),
            lambda: balanced_matrix(2, pair_bytes=1, total_bytes=2),
            lambda: fast_uniform_matrix(1, bytes_per_unit=1, seed=0),
            lambda: fast_uniform_matrix(2, bytes_per_unit=0, seed=0),
            lambda: fast_zipf_matrix(2, alpha=0.0, bytes_per_unit=1, seed=0),
            lambda: fast_zipf_matrix(2, alpha=1.0, bytes_per_unit=1, seed=0),
        ]
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()

    def test_scaling_overflow_is_rejected_before_sampling(self):
        limit = np.iinfo(np.int64).max
        with self.assertRaises(ValueError):
            fast_uniform_matrix(2, bytes_per_unit=limit // 9999 + 1, seed=0)
        with self.assertRaises(ValueError):
            fast_zipf_matrix(2, alpha=0.9, bytes_per_unit=limit // 10000 + 1, seed=0)

    def test_registry_rejects_unknown_generator(self):
        with self.assertRaisesRegex(ValueError, "balanced"):
            generate_matrix("unknown", num_devices=2)


if __name__ == "__main__":
    unittest.main()
