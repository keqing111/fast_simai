import tempfile
import unittest
from pathlib import Path

import numpy as np

from simai_a2a.analysis import analyze_fct, analyze_network_monitors, fast_stage_violations
from simai_a2a.artifacts import write_matrix_artifact, write_schedule_artifact
from simai_a2a.model import Schedule, ScheduledFlow, TrafficMatrix


def _ip(node: int) -> str:
    return f"{0x0B000001 + ((node // 256) << 16) + ((node % 256) << 8):08x}"


def _artifacts(root: Path, flows: list[ScheduledFlow]) -> tuple[Path, Path]:
    matrix = TrafficMatrix(np.array([[0, 100], [200, 0]], dtype=np.int64))
    matrix_dir = root / "matrix"
    schedule_dir = root / "schedule"
    matrix_manifest = write_matrix_artifact(matrix, matrix_dir)
    schedule = Schedule("fast", flows, 160, {"logical_bytes": 300})
    write_schedule_artifact(
        schedule,
        schedule_dir,
        matrix_sha256=matrix_manifest["matrix_sha256"],
        hardware_sha256="hardware",
    )
    return matrix_dir, schedule_dir


class AnalysisTests(unittest.TestCase):
    def test_network_monitor_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qlen = root / "qlen.txt"
            qlen.write_text(
                "time, sw_id, port_id, q_id, q_len, port_len\n"
                "1000, 8, 1, 3, 200, 300\n"
                "2000, 9, 2, 3, 500, 700\n",
                encoding="utf-8",
            )
            bw = root / "bw.txt"
            bw.write_text(
                "time, node_id, port_id, bandwidth\n"
                "1000, 8, 1, 100\n"
                "1000, 9, 2, 200\n",
                encoding="utf-8",
            )
            result = analyze_network_monitors(
                qlen, bw, tor_id=9, tor_port_capacity_gbps=400
            )
            self.assertEqual(result["queue"]["peak_queue_bytes"], 500)
            self.assertEqual(result["queue"]["tor_peak_port_bytes"], 700)
            self.assertEqual(result["bandwidth"]["tor_peak_gbps"], 200)
            self.assertEqual(result["bandwidth"]["tor_peak_utilization"], 0.5)

    def test_exact_join_and_collective_statistics(self):
        flows = [
            ScheduledFlow(0, 0, 1, 100, 0, "balance", 0),
            ScheduledFlow(1, 1, 0, 200, 100, "inter", 1),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix_dir, schedule_dir = _artifacts(root, flows)
            fct = root / "fct.txt"
            fct.write_text(
                f"{_ip(0)} {_ip(1)} 1 100 100 1000 40 42 0\n"
                f"{_ip(1)} {_ip(0)} 2 100 200 1100 60 62 1\n",
                encoding="utf-8",
            )
            pfc = root / "pfc.txt"
            pfc.write_text("pause-a\n\npause-b\n", encoding="utf-8")

            summary = analyze_fct(matrix_dir, schedule_dir, fct, pfc)

            self.assertEqual(summary["collective_epoch_ns"], 1000)
            self.assertEqual(summary["actual_makespan_ns"], 160)
            self.assertEqual(summary["logical_bytes"], 300)
            self.assertEqual(summary["physical_bytes"], 300)
            self.assertEqual(summary["scheduled_bytes"], 300)
            self.assertEqual(summary["schedule_byte_amplification"], 1.0)
            self.assertEqual(summary["algorithm_bandwidth_gbps"], 15.0)
            self.assertEqual(summary["per_device_algorithm_bandwidth_gbps"], 7.5)
            self.assertEqual(summary["fct_ns"]["mean"], 50.0)
            self.assertEqual(summary["fct_ns"]["p50"], 50.0)
            self.assertEqual(summary["fct_ns"]["p95"], 59.0)
            self.assertAlmostEqual(summary["fct_ns"]["p99"], 59.8)
            self.assertEqual(summary["fct_ns"]["max"], 60)
            self.assertEqual(summary["pfc_count"], 2)
            self.assertEqual(summary["predicted_makespan_ns"], 160)
            self.assertEqual(summary["fidelity_ratio"], 1.0)

    def test_missing_duplicate_unknown_and_mismatched_ids_are_rejected(self):
        flows = [
            ScheduledFlow(0, 0, 1, 100, 0, "balance", 0),
            ScheduledFlow(1, 1, 0, 200, 100, "inter", 1),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix_dir, schedule_dir = _artifacts(root, flows)
            pfc = root / "pfc.txt"
            pfc.write_text("", encoding="utf-8")
            cases = {
                "missing": f"{_ip(0)} {_ip(1)} 1 100 100 1000 40 42 0\n",
                "duplicate": (
                    f"{_ip(0)} {_ip(1)} 1 100 100 1000 40 42 0\n"
                    f"{_ip(0)} {_ip(1)} 2 100 100 1001 40 42 0\n"
                ),
                "unknown": (
                    f"{_ip(0)} {_ip(1)} 1 100 100 1000 40 42 0\n"
                    f"{_ip(1)} {_ip(0)} 2 100 200 1100 60 62 9\n"
                ),
                "mismatch": (
                    f"{_ip(0)} {_ip(1)} 1 100 101 1000 40 42 0\n"
                    f"{_ip(1)} {_ip(0)} 2 100 200 1100 60 62 1\n"
                ),
            }
            for name, content in cases.items():
                with self.subTest(name=name):
                    fct = root / f"{name}.txt"
                    fct.write_text(content, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        analyze_fct(matrix_dir, schedule_dir, fct, pfc)

    def test_fast_stage_violation_boundaries(self):
        schedule = Schedule(
            "fast",
            [
                ScheduledFlow(0, 0, 1, 10, 0, "balance", 0),
                ScheduledFlow(1, 1, 0, 10, 100, "inter", 0),
                ScheduledFlow(2, 0, 1, 10, 200, "dispatch", 0),
            ],
            250,
        )
        observations = {
            0: {"actual_start_ns": 1000, "fct_ns": 120},
            1: {"actual_start_ns": 1100, "fct_ns": 80},
            2: {"actual_start_ns": 1200, "fct_ns": 30},
        }

        result = fast_stage_violations(schedule, observations, collective_epoch_ns=1000)

        self.assertEqual(result["boundary_count"], 2)
        self.assertEqual(result["violation_count"], 1)
        self.assertEqual(result["violation_fraction"], 0.5)
        self.assertEqual(result["max_violation_ns"], 20)
        self.assertEqual(result["sum_violation_ns"], 20)
        self.assertEqual([item["violation_ns"] for item in result["boundaries"]], [20, 0])


if __name__ == "__main__":
    unittest.main()
