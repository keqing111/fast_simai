import tempfile
import unittest
from pathlib import Path

from simai_a2a.model import HardwareProfile, LinkModel
from simai_a2a.topology import build_topology, write_topology


def profile(*, servers: int, gpus: int, oversubscription: float = 1.0):
    return HardwareProfile(
        num_servers=servers,
        gpus_per_server=gpus,
        intra=LinkModel(100, 0.002),
        inter=LinkModel(700, 0.08),
        topology_id="test",
        link_bandwidth_gbps=100,
        oversubscription=oversubscription,
    )


class TopologyTests(unittest.TestCase):
    def test_single_server_ids_counts_and_links(self):
        topology = build_topology(profile(servers=1, gpus=4))

        self.assertEqual(topology.gpu_ids, (0, 1, 2, 3))
        self.assertEqual(topology.nvswitch_ids, (4,))
        self.assertIsNone(topology.tor_id)
        self.assertEqual(topology.node_count, 5)
        self.assertEqual(len(topology.links), 4)
        self.assertTrue(all(link.bandwidth_gbps == 4000 for link in topology.links))
        for gpu_id in topology.gpu_ids:
            incident = [link for link in topology.links if gpu_id in (link.src, link.dst)]
            self.assertEqual(len(incident), 1)
            self.assertEqual(incident[0].latency_ns, 100)

    def test_multi_server_has_one_tor_and_oversubscription_changes_uplink(self):
        nonblocking = build_topology(profile(servers=2, gpus=2))
        oversubscribed = build_topology(
            profile(servers=2, gpus=2, oversubscription=2)
        )

        self.assertEqual(nonblocking.gpu_ids, (0, 1, 2, 3))
        self.assertEqual(nonblocking.nvswitch_ids, (4, 5))
        self.assertEqual(nonblocking.tor_id, 6)
        self.assertEqual(nonblocking.node_count, 7)
        self.assertEqual(len(nonblocking.links), 6)
        uplinks = [link for link in nonblocking.links if link.dst == nonblocking.tor_id]
        oversubscribed_uplinks = [
            link for link in oversubscribed.links if link.dst == oversubscribed.tor_id
        ]
        self.assertEqual([link.bandwidth_gbps for link in uplinks], [200, 200])
        self.assertEqual(
            [link.bandwidth_gbps for link in oversubscribed_uplinks], [100, 100]
        )
        self.assertTrue(all(link.latency_ns == 700 for link in uplinks))
        for gpu_id in nonblocking.gpu_ids:
            self.assertEqual(
                sum(gpu_id in (link.src, link.dst) for link in nonblocking.links), 1
            )

    def test_serialized_header_and_validation(self):
        hardware = profile(servers=2, gpus=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "topology.txt"
            manifest = write_topology(hardware, path)
            lines = path.read_text(encoding="utf-8").splitlines()
            header = lines[0].split()
            self.assertEqual(header[:5], ["7", "2", "2", "1", "6"])
            self.assertEqual(header[5], "H100")
            self.assertEqual(lines[1], "4 5 6")
            self.assertEqual(manifest["node_count"], 7)
            self.assertEqual(manifest["link_count"], 6)
        with self.assertRaises(ValueError):
            build_topology(hardware, expected_num_devices=8)
        with self.assertRaises(ValueError):
            HardwareProfile(
                1, 2, LinkModel(0, 0), LinkModel(0, 0), "bad", 0, 1
            )


if __name__ == "__main__":
    unittest.main()
