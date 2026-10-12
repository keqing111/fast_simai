"""Process-based ns-3 experiment runner with strict completion checks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .analysis import analyze_fct, write_summary
from .artifacts import read_schedule_artifact


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_TEMPLATE = PACKAGE_ROOT / "configs" / "ns3.conf"


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _next_attempt(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        attempt = root / f"attempt-{index:04d}"
        try:
            attempt.mkdir()
            return attempt
        except FileExistsError:
            index += 1


def _render_config(template: Path, attempt: Path) -> Path:
    names = {
        "FLOW_FILE": attempt / "flow.txt",
        "TRACE_FILE": attempt / "trace.txt",
        "TRACE_OUTPUT_FILE": attempt / "trace.out",
        "FCT_OUTPUT_FILE": attempt / "fct.txt",
        "PFC_OUTPUT_FILE": attempt / "pfc.txt",
        "QLEN_MON_FILE": attempt / "qlen.txt",
        "BW_MON_FILE": attempt / "bw.txt",
        "RATE_MON_FILE": attempt / "rate.txt",
        "CNP_MON_FILE": attempt / "cnp.txt",
    }
    rendered = template.read_text(encoding="utf-8").format(
        **{key: str(value.resolve()) for key, value in names.items()}
    )
    config = attempt / "ns3.conf"
    config.write_text(rendered, encoding="utf-8")
    names["FLOW_FILE"].write_text("0\n", encoding="utf-8")
    names["TRACE_FILE"].write_text("0\n", encoding="utf-8")
    return config


def run_scenario(
    scenario_dir: Path,
    algorithm: str,
    topology_path: Path,
    binary_path: Path,
    *,
    timeout_s: float,
    output_root: Path | None = None,
    config_template: Path = DEFAULT_CONFIG_TEMPLATE,
) -> dict[str, Any]:
    """Run one prepared schedule and return its persisted run manifest."""
    configured_version = os.environ.get("SIMAI_NCCL_VERSION", "v2.30")
    if configured_version != "v2.30":
        raise ValueError("external scheduling supports only SIMAI_NCCL_VERSION=v2.30")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    scenario_dir = Path(scenario_dir).resolve()
    schedule_source = scenario_dir / "schedules" / algorithm
    schedule, _ = read_schedule_artifact(schedule_source)
    if schedule.algorithm != algorithm:
        raise ValueError("schedule algorithm does not match the requested algorithm")
    binary_path = Path(binary_path).resolve()
    topology_path = Path(topology_path).resolve()
    if not binary_path.is_file():
        raise ValueError(f"ns-3 binary does not exist: {binary_path}")
    if not os.access(binary_path, os.X_OK):
        raise ValueError(f"ns-3 binary is not executable: {binary_path}")
    if not topology_path.is_file():
        raise ValueError(f"topology does not exist: {topology_path}")

    attempts_root = (
        Path(output_root).resolve()
        if output_root is not None
        else scenario_dir / "runs" / algorithm
    )
    attempt = _next_attempt(attempts_root)
    local_topology = attempt / "topology.txt"
    local_workload = attempt / "workload.txt"
    local_schedule = attempt / "schedule.csv"
    shutil.copy2(topology_path, local_topology)
    shutil.copy2(scenario_dir / "workload.csv", local_workload)
    shutil.copy2(schedule_source / "schedule.csv", local_schedule)
    config = _render_config(Path(config_template), attempt)
    scenario_manifest_path = scenario_dir / "scenario_manifest.json"
    hardware_path = scenario_dir / "hardware.json"
    scenario_manifest = (
        json.loads(scenario_manifest_path.read_text(encoding="utf-8"))
        if scenario_manifest_path.is_file()
        else {"topology": {}}
    )
    hardware_data = (
        json.loads(hardware_path.read_text(encoding="utf-8"))
        if hardware_path.is_file()
        else None
    )
    command = [
        str(binary_path),
        "-n", str(local_topology.resolve()),
        "-c", str(config.resolve()),
        "-w", str(local_workload.resolve()),
        "-s", str(local_schedule.resolve()),
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "runtime_semantics_version": 3,
        "algorithm": algorithm,
        "attempt_dir": str(attempt.resolve()),
        "command": command,
        "timeout_s": timeout_s,
        "success": False,
    }
    started = time.monotonic_ns()
    try:
        completed = subprocess.run(
            command,
            cwd=attempt,
            env={**os.environ, "SIMAI_NCCL_VERSION": "v2.30"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
        )
        (attempt / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (attempt / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        manifest["returncode"] = completed.returncode
        if completed.returncode != 0:
            manifest["status"] = "failed"
        else:
            try:
                summary = analyze_fct(
                    scenario_dir / "matrix",
                    schedule_source,
                    attempt / "fct.txt",
                    attempt / "pfc.txt",
                    attempt / "qlen.txt",
                    attempt / "bw.txt",
                    tor_id=scenario_manifest["topology"].get("tor_id"),
                    tor_port_capacity_gbps=(
                        float(hardware_data["link_bandwidth_gbps"])
                        * int(hardware_data["gpus_per_server"])
                        / float(hardware_data["oversubscription"])
                    ) if hardware_data is not None else None,
                )
            except (OSError, ValueError) as exc:
                manifest["status"] = "incomplete"
                manifest["error"] = str(exc)
            else:
                write_summary(summary, attempt / "summary.json")
                manifest["status"] = "success"
                manifest["success"] = True
                manifest["summary"] = summary
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        (attempt / "stdout.log").write_text(stdout, encoding="utf-8")
        (attempt / "stderr.log").write_text(stderr, encoding="utf-8")
        manifest["status"] = "timeout"
        manifest["error"] = f"simulation exceeded {timeout_s} seconds"
    manifest["elapsed_wall_ns"] = time.monotonic_ns() - started
    _write_json_atomic(attempt / "run_manifest.json", manifest)
    return manifest
