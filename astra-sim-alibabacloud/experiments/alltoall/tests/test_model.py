import unittest

import numpy as np

from simai_a2a.model import (
    HardwareProfile,
    LinkModel,
    Schedule,
    ScheduledFlow,
    TrafficMatrix,
)


def hardware() -> HardwareProfile:
    return HardwareProfile(
        num_servers=2,
        gpus_per_server=2,
        intra=LinkModel(alpha_ns=100, beta_ns_per_byte=0.01),
        inter=LinkModel(alpha_ns=700, beta_ns_per_byte=0.08),
        topology_id="two-server",
        link_bandwidth_gbps=100.0,
        oversubscription=1.0,
    )


class TrafficMatrixTests(unittest.TestCase):
    def test_accepts_valid_int64_zero_diagonal_matrix(self):
        values = np.ones((4, 4), dtype=np.int64)
        np.fill_diagonal(values, 0)

        matrix = TrafficMatrix(values, {"generator": "test"})

        self.assertEqual(matrix.logical_bytes, 12)
        self.assertEqual(matrix.num_devices, 4)

    def test_rejects_invalid_matrix_shapes_and_values(self):
        cases = [
            np.zeros((2, 3), dtype=np.int64),
            np.zeros((2, 2), dtype=np.float64),
            np.array([[0, -1], [1, 0]], dtype=np.int64),
            np.array([[1, 0], [0, 0]], dtype=np.int64),
        ]
        for values in cases:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    TrafficMatrix(values)

    def test_profile_device_count_must_match_matrix(self):
        matrix = TrafficMatrix(np.zeros((3, 3), dtype=np.int64))

        with self.assertRaises(ValueError):
            hardware().validate_matrix(matrix)


class FlowAndScheduleTests(unittest.TestCase):
    def test_link_duration_rounds_to_integer_nanoseconds(self):
        link = LinkModel(alpha_ns=10.0, beta_ns_per_byte=0.25)
        self.assertEqual(link.duration_ns(7), 12)

    def test_flow_validation_rejects_invalid_fields(self):
        profile = hardware()
        invalid = [
            ScheduledFlow(0, 1, 1, 1, 0, "direct", 0),
            ScheduledFlow(0, 0, 1, 0, 0, "direct", 0),
            ScheduledFlow(0, 0, 1, 1, -1, "direct", 0),
            ScheduledFlow(0, 0, 4, 1, 0, "direct", 0),
        ]
        for flow in invalid:
            with self.subTest(flow=flow):
                with self.assertRaises(ValueError):
                    flow.validate(profile)

    def test_schedule_requires_deterministic_order_and_contiguous_ids(self):
        first = ScheduledFlow(0, 0, 1, 4, 10, "direct", 0)
        second = ScheduledFlow(1, 1, 0, 4, 0, "direct", 0)

        with self.assertRaises(ValueError):
            Schedule("direct-concurrent", [first, second], 20, {})

        valid = Schedule(
            "direct-concurrent",
            [
                ScheduledFlow(0, 1, 0, 4, 0, "direct", 0),
                ScheduledFlow(1, 0, 1, 4, 10, "direct", 0),
            ],
            20,
            {},
        )
        self.assertEqual([flow.flow_id for flow in valid.flows], [0, 1])


if __name__ == "__main__":
    unittest.main()
