import unittest
from collections import defaultdict

import numpy as np

from simai_a2a.model import HardwareProfile, LinkModel, TrafficMatrix
from simai_a2a.schedulers import get_scheduler
from simai_a2a.schedulers.direct import DirectConcurrentScheduler
from simai_a2a.schedulers.roundrobin import RoundRobinScheduler
from simai_a2a.schedulers.spreadout import SpreadOutScheduler


def profile() -> HardwareProfile:
    return HardwareProfile(
        num_servers=2,
        gpus_per_server=2,
        intra=LinkModel(alpha_ns=10, beta_ns_per_byte=1),
        inter=LinkModel(alpha_ns=100, beta_ns_per_byte=2),
        topology_id="test",
        link_bandwidth_gbps=100,
        oversubscription=1,
    )


def matrix() -> TrafficMatrix:
    return TrafficMatrix(
        np.array(
            [
                [0, 1, 2, 3],
                [4, 0, 5, 6],
                [7, 8, 0, 9],
                [10, 11, 12, 0],
            ],
            dtype=np.int64,
        )
    )


def tuples(schedule):
    return [(flow.src, flow.dst, flow.size_bytes) for flow in schedule.flows]


class DirectSchedulerTests(unittest.TestCase):
    def test_direct_emits_every_nonzero_cell_at_zero(self):
        demand = matrix()

        schedule = DirectConcurrentScheduler().schedule(demand, profile(), seed=9)

        self.assertEqual(len(schedule.flows), 12)
        self.assertEqual(set(tuples(schedule)), {
            (src, dst, int(demand.values[src, dst]))
            for src in range(4)
            for dst in range(4)
            if src != dst
        })
        self.assertTrue(all(flow.start_ns == 0 for flow in schedule.flows))
        self.assertTrue(all(flow.phase == "direct" for flow in schedule.flows))
        self.assertTrue(all(flow.stage_id == 0 for flow in schedule.flows))
        self.assertEqual(schedule.physical_bytes, demand.logical_bytes)

    def test_registry_lists_all_accepted_schedulers(self):
        self.assertIsInstance(get_scheduler("direct-concurrent"), DirectConcurrentScheduler)
        with self.assertRaisesRegex(
            ValueError,
            "direct-concurrent.*fast.*roundrobin.*spreadout",
        ):
            get_scheduler("unknown")


class SpreadOutSchedulerTests(unittest.TestCase):
    def test_spreadout_uses_native_cyclic_offsets_and_costs(self):
        demand = matrix()

        schedule = SpreadOutScheduler().schedule(demand, profile(), seed=1)

        self.assertEqual(set(tuples(schedule)), set(tuples(
            DirectConcurrentScheduler().schedule(demand, profile(), seed=0)
        )))
        starts_by_stage = defaultdict(set)
        pairs_by_stage = defaultdict(set)
        for flow in schedule.flows:
            starts_by_stage[flow.stage_id].add(flow.start_ns)
            pairs_by_stage[flow.stage_id].add((flow.src, flow.dst))
        self.assertEqual(dict(starts_by_stage), {0: {0}, 1: {120}, 2: {242}})
        self.assertEqual(
            dict(pairs_by_stage),
            {
                0: {(0, 1), (1, 2), (2, 3), (3, 0)},
                1: {(0, 2), (1, 3), (2, 0), (3, 1)},
                2: {(0, 3), (1, 0), (2, 1), (3, 2)},
            },
        )
        self.assertEqual(schedule.predicted_makespan_ns, 358)
        repeated = SpreadOutScheduler().schedule(demand, profile(), seed=999)
        self.assertEqual(schedule.flows, repeated.flows)
        self.assertEqual(schedule.predicted_makespan_ns, repeated.predicted_makespan_ns)


class RoundRobinSchedulerTests(unittest.TestCase):
    def test_roundrobin_is_one_to_one_and_byte_conserving(self):
        demand = matrix()

        schedule = RoundRobinScheduler().schedule(demand, profile(), seed=2)

        self.assertEqual(set(tuples(schedule)), set(tuples(
            DirectConcurrentScheduler().schedule(demand, profile(), seed=0)
        )))
        self.assertEqual(schedule.physical_bytes, demand.logical_bytes)
        self.assertEqual(schedule.predicted_makespan_ns, 264)
        starts = {stage: set() for stage in range(3)}
        for flow in schedule.flows:
            starts[flow.stage_id].add(flow.start_ns)
        self.assertEqual(starts, {0: {0}, 1: {22}, 2: {144}})
        for stage_id in range(3):
            stage = [flow for flow in schedule.flows if flow.stage_id == stage_id]
            self.assertEqual(len({flow.src for flow in stage}), len(stage))
            self.assertEqual(len({flow.dst for flow in stage}), len(stage))
        self.assertTrue(
            all(
                flow.src // 2 == flow.dst // 2
                for flow in schedule.flows
                if flow.stage_id == 0
            )
        )
        self.assertTrue(
            all(
                flow.src // 2 != flow.dst // 2
                for flow in schedule.flows
                if flow.stage_id > 0
            )
        )
        repeated = RoundRobinScheduler().schedule(demand, profile(), seed=999)
        self.assertEqual(schedule.flows, repeated.flows)
        self.assertEqual(schedule.predicted_makespan_ns, repeated.predicted_makespan_ns)


if __name__ == "__main__":
    unittest.main()
