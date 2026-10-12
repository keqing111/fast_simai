import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BINARY = REPO_ROOT / "bin" / "SimAI_simulator"
BASE_CONFIG = REPO_ROOT / "astra-sim-alibabacloud" / "inputs" / "config" / "SimAI.conf"


def _write_ns3_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    topology = root / "topology.txt"
    topology.write_text(
        "3 2 1 0 2 H100\n"
        "2\n"
        "0 2 900Gbps 0.000025ms 0\n"
        "1 2 900Gbps 0.000025ms 0\n",
        encoding="utf-8",
    )
    workload = root / "workload.txt"
    workload.write_text(
        "MICRO model_parallel_NPU_group: 1 ep: 2 all_gpus: 2\n"
        "1\n"
        "0 0 0 NONE 0 0 NONE 0 0 ALLTOALL 8192 0\n",
        encoding="utf-8",
    )
    schedule = root / "schedule.csv"
    schedule.write_text(
        "0,1,4096,0\n"
        "0,1,4096,50000\n",
        encoding="utf-8",
    )
    flow = root / "flow.txt"
    trace = root / "trace.txt"
    flow.write_text("0\n", encoding="utf-8")
    trace.write_text("0\n", encoding="utf-8")
    replacements = {
        "FLOW_FILE": flow,
        "TRACE_FILE": trace,
        "TRACE_OUTPUT_FILE": root / "trace.out",
        "FCT_OUTPUT_FILE": root / "fct.txt",
        "PFC_OUTPUT_FILE": root / "pfc.txt",
        "QLEN_MON_FILE": root / "qlen.txt",
        "BW_MON_FILE": root / "bw.txt",
        "RATE_MON_FILE": root / "rate.txt",
        "CNP_MON_FILE": root / "cnp.txt",
    }
    rendered = []
    for line in BASE_CONFIG.read_text(encoding="utf-8").splitlines():
        key = line.split(maxsplit=1)[0] if line.strip() else ""
        if key in replacements:
            rendered.append(f"{key} {replacements[key]}")
        elif key == "ENABLE_TRACE":
            rendered.append("ENABLE_TRACE 0")
        else:
            rendered.append(line)
    config = root / "ns3.conf"
    config.write_text("\n".join(rendered) + "\n", encoding="utf-8")
    return topology, workload, schedule, config


class Ns3AdapterTests(unittest.TestCase):
    def test_repeated_delayed_external_flows_finish_with_distinct_ids(self):
        binary = Path(os.environ.get("SIMAI_NS3_BINARY", DEFAULT_BINARY))
        if not binary.is_file():
            self.skipTest(f"ns-3 binary not found: {binary}")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            topology, workload, schedule, config = _write_ns3_fixture(root)
            completed = subprocess.run(
                [
                    str(binary),
                    "-t", "1",
                    "-w", str(workload),
                    "-n", str(topology),
                    "-c", str(config),
                    "-s", str(schedule),
                ],
                cwd=root,
                env={**os.environ, "AS_LOG_LEVEL": "ERROR"},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=20,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            rows = [line.split() for line in (root / "fct.txt").read_text().splitlines() if line]
            self.assertEqual(len(rows), 2, completed.stdout + completed.stderr)
            self.assertTrue(all(len(row) == 9 for row in rows), rows)
            self.assertEqual([int(row[8]) for row in rows], [0, 1])
            starts_ns = [int(row[5]) for row in rows]
            self.assertLessEqual(abs((starts_ns[1] - starts_ns[0]) - 50_000), 1)


if __name__ == "__main__":
    unittest.main()
