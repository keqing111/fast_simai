import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from simai_a2a.artifacts import write_matrix_artifact, write_schedule_artifact
from simai_a2a.cli import main as cli_main
from simai_a2a.model import Schedule, ScheduledFlow, TrafficMatrix
from simai_a2a.runner import run_scenario


FAKE_SUCCESS = r'''#!/usr/bin/env python3
import json
import sys
from pathlib import Path
args = sys.argv[1:]
values = {args[i]: args[i + 1] for i in range(0, len(args), 2)}
Path("argv.json").write_text(json.dumps(args))
config = {}
for line in Path(values["-c"]).read_text().splitlines():
    parts = line.split(maxsplit=1)
    if len(parts) == 2:
        config[parts[0]] = parts[1]
rows = []
for flow_id, line in enumerate(Path(values["-s"]).read_text().splitlines()):
    src, dst, size, start = map(int, line.split(","))
    ip = lambda node: f"{0x0B000001 + ((node // 256) << 16) + ((node % 256) << 8):08x}"
    rows.append(f"{ip(src)} {ip(dst)} {10000 + flow_id} 100 {size} {1000 + start} 10 12 {flow_id}")
Path(config["FCT_OUTPUT_FILE"]).write_text("\n".join(rows) + "\n")
Path(config["PFC_OUTPUT_FILE"]).write_text("")
'''


def _scenario(root: Path) -> tuple[Path, Path]:
    scenario = root / "scenario"
    matrix = TrafficMatrix(np.array([[0, 64], [64, 0]], dtype=np.int64))
    matrix_manifest = write_matrix_artifact(matrix, scenario / "matrix")
    schedule = Schedule(
        "direct-concurrent",
        [
            ScheduledFlow(0, 0, 1, 64, 0, "direct", 0),
            ScheduledFlow(1, 1, 0, 64, 0, "direct", 0),
        ],
        20,
    )
    write_schedule_artifact(
        schedule,
        scenario / "schedules" / "direct-concurrent",
        matrix_sha256=matrix_manifest["matrix_sha256"],
        hardware_sha256="hardware",
    )
    (scenario / "workload.csv").write_text(
        "MICRO model_parallel_NPU_group: 1 ep: 2 all_gpus: 2\n"
        "1\n0 0 0 NONE 0 0 NONE 0 0 ALLTOALL 64 0\n",
        encoding="utf-8",
    )
    topology = root / "topology.txt"
    topology.write_text("topology\n", encoding="utf-8")
    return scenario, topology


def _executable(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
    return path


class RunnerTests(unittest.TestCase):
    def test_run_and_analyze_cli_subcommands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario, topology = _scenario(root)
            binary = _executable(root / "fake.py", FAKE_SUCCESS)
            output = io.StringIO()
            with redirect_stdout(output):
                status = cli_main(
                    [
                        "run",
                        "--scenario", str(scenario),
                        "--algorithm", "direct-concurrent",
                        "--topology", str(topology),
                        "--binary", str(binary),
                        "--timeout", "2",
                    ]
                )
            self.assertEqual(status, 0)
            result = json.loads(output.getvalue())
            attempt = Path(result["attempt_dir"])
            output = io.StringIO()
            with redirect_stdout(output):
                status = cli_main(
                    [
                        "analyze",
                        "--scenario", str(scenario),
                        "--algorithm", "direct-concurrent",
                        "--attempt", str(attempt),
                    ]
                )
            self.assertEqual(status, 0)
            self.assertTrue((attempt / "summary.json").is_file())

    def test_success_uses_local_files_and_new_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario, topology = _scenario(root)
            binary = _executable(root / "fake.py", FAKE_SUCCESS)

            first = run_scenario(scenario, "direct-concurrent", topology, binary, timeout_s=2)
            second = run_scenario(scenario, "direct-concurrent", topology, binary, timeout_s=2)

            self.assertEqual(first["status"], "success")
            self.assertEqual(second["status"], "success")
            first_dir = Path(first["attempt_dir"])
            second_dir = Path(second["attempt_dir"])
            self.assertNotEqual(first_dir, second_dir)
            self.assertTrue(first_dir.name.endswith("0001"))
            self.assertTrue(second_dir.name.endswith("0002"))
            argv = json.loads((first_dir / "argv.json").read_text())
            self.assertEqual(set(argv[::2]), {"-n", "-c", "-w", "-s"})
            for flag in ("-n", "-c", "-w", "-s"):
                self.assertEqual(Path(argv[argv.index(flag) + 1]).parent, first_dir)
            self.assertTrue((first_dir / "summary.json").is_file())
            self.assertEqual(first["summary"]["flow_count"], 2)

    def test_timeout_and_incomplete_output_are_non_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario, topology = _scenario(root)
            sleeper = _executable(
                root / "sleep.py",
                "#!/usr/bin/env python3\nimport time\ntime.sleep(2)\n",
            )
            timeout = run_scenario(
                scenario, "direct-concurrent", topology, sleeper, timeout_s=0.05
            )
            self.assertEqual(timeout["status"], "timeout")
            self.assertFalse(timeout["success"])

            empty = _executable(root / "empty.py", "#!/usr/bin/env python3\n")
            incomplete = run_scenario(
                scenario, "direct-concurrent", topology, empty, timeout_s=2
            )
            self.assertEqual(incomplete["status"], "incomplete")
            self.assertFalse(incomplete["success"])


if __name__ == "__main__":
    unittest.main()
