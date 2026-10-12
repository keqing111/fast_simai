import unittest

import numpy as np

from simai_a2a.metrics import gini, matrix_skew_metrics
from simai_a2a.model import TrafficMatrix


class GiniTests(unittest.TestCase):
    def test_gini_has_defined_zero_and_equal_semantics(self):
        self.assertEqual(gini(np.array([0, 0, 0], dtype=np.int64)), 0.0)
        self.assertEqual(gini(np.array([1, 1, 1], dtype=np.int64)), 0.0)

    def test_gini_detects_concentration(self):
        self.assertAlmostEqual(
            gini(np.array([0, 0, 3], dtype=np.int64)),
            2.0 / 3.0,
        )

    def test_gini_rejects_negative_values(self):
        with self.assertRaises(ValueError):
            gini(np.array([1, -1], dtype=np.int64))


class MatrixMetricTests(unittest.TestCase):
    def test_balanced_matrix_has_zero_gini_and_unit_bottleneck(self):
        values = np.ones((4, 4), dtype=np.int64)
        np.fill_diagonal(values, 0)

        metrics = matrix_skew_metrics(TrafficMatrix(values), gpus_per_server=2)

        for name in (
            "cell_gini",
            "source_gini",
            "destination_gini",
            "server_pair_gini",
        ):
            self.assertEqual(metrics[name], 0.0)
        self.assertEqual(metrics["source_bottleneck_ratio"], 1.0)
        self.assertEqual(metrics["destination_bottleneck_ratio"], 1.0)

    def test_all_zero_matrix_has_zero_bottleneck_ratios(self):
        metrics = matrix_skew_metrics(
            TrafficMatrix(np.zeros((4, 4), dtype=np.int64)),
            gpus_per_server=2,
        )

        self.assertEqual(metrics["source_bottleneck_ratio"], 0.0)
        self.assertEqual(metrics["destination_bottleneck_ratio"], 0.0)

    def test_server_aggregation_requires_divisible_device_count(self):
        matrix = TrafficMatrix(np.zeros((3, 3), dtype=np.int64))

        with self.assertRaises(ValueError):
            matrix_skew_metrics(matrix, gpus_per_server=2)


if __name__ == "__main__":
    unittest.main()
