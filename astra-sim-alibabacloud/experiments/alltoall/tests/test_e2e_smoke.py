import json
import os
import tempfile
import unittest
from pathlib import Path

from simai_a2a.runner import run_scenario
from simai_a2a.scenario import prepare_scenario


ALGORITHMS = ("direct-concurrent", "spreadout", "roundrobin", "fast")
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class EndToEndSmokeTests(unittest.TestCase):
    def test_three_scenarios_by_four_schedulers(self):
        binary_value = os.environ.get("SIMAI_NS3_BINARY")
        if not binary_value:
            self.skipTest("set SIMAI_NS3_BINARY to run the ns-3 vertical smoke matrix")
        binary = Path(binary_value).resolve()
        hardware_root = PACKAGE_ROOT / "configs" / "hardware"
        cases = (
            (
                "single-balanced",
                hardware_root / "h100_1s4g_nvlink.json",
                {"generator": "balanced", "num_devices": 4, "pair_bytes": 65536},
            ),
            (
                "multi-balanced",
                hardware_root / "h100_2s2g_100g.json",
                {"generator": "balanced", "num_devices": 4, "pair_bytes": 65536},
            ),
            (
                "oversubscribed-zipf",
                hardware_root / "h100_2s2g_100g_2to1.json",
                {
                    "generator": "fast-zipf",
                    "num_devices": 4,
                    "alpha": 0.9,
                    "bytes_per_unit": 16,
                },
            ),
        )
        completed_runs = 0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, hardware, traffic in cases:
                case_root = root / name
                case_root.mkdir()
                config = case_root / "scenario.json"
                config.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "name": name,
                            "hardware": str(hardware),
                            "seed": 11,
                            "traffic": traffic,
                            "schedulers": list(ALGORITHMS),
                        }
                    ),
                    encoding="utf-8",
                )
                first = case_root / "prepared-a"
                second = case_root / "prepared-b"
                first_manifest = prepare_scenario(config, first)
                second_manifest = prepare_scenario(config, second)
                self.assertEqual(
                    first_manifest["matrix_sha256"], second_manifest["matrix_sha256"]
                )
                self.assertEqual(
                    first_manifest["schedule_sha256"],
                    second_manifest["schedule_sha256"],
                )
                self.assertEqual(
                    first_manifest["topology"]["topology_sha256"],
                    second_manifest["topology"]["topology_sha256"],
                )

                logical_bytes = None
                for algorithm in ALGORITHMS:
                    with self.subTest(scenario=name, algorithm=algorithm):
                        result = run_scenario(
                            first,
                            algorithm,
                            first / "topology.txt",
                            binary,
                            timeout_s=30,
                        )
                        self.assertTrue(result["success"], result)
                        summary = result["summary"]
                        if logical_bytes is None:
                            logical_bytes = summary["logical_bytes"]
                        self.assertEqual(summary["logical_bytes"], logical_bytes)
                        self.assertEqual(summary["flow_count"], 12)
                        if algorithm == "fast":
                            self.assertIn("fast_stage_violations", summary)
                        completed_runs += 1
        self.assertEqual(completed_runs, 12)


if __name__ == "__main__":
    unittest.main()
