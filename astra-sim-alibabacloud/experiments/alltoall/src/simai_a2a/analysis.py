"""Strict FCT joins and metrics for external all-to-all schedules."""

from __future__ import annotations

import json
import math
import csv
from pathlib import Path
from typing import Any

from .artifacts import read_matrix_artifact, read_schedule_artifact
from .model import Schedule


def _node_from_hex_ip(value: str) -> int:
    try:
        encoded = int(value, 16)
    except ValueError as exc:
        raise ValueError(f"invalid hexadecimal IP field: {value!r}") from exc
    return (encoded >> 8) & 0xFFFF


def _percentile(values: list[int], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of no observations")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def read_external_fct(path: Path) -> dict[int, dict[str, int]]:
    """Read the external nine-column ns-3 FCT format keyed by schedule row."""
    observations: dict[int, dict[str, int]] = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 9:
            raise ValueError(f"FCT line {line_number} must have nine columns")
        try:
            flow_id = int(fields[8], 10)
            record = {
                "src": _node_from_hex_ip(fields[0]),
                "dst": _node_from_hex_ip(fields[1]),
                "size_bytes": int(fields[4], 10),
                "actual_start_ns": int(fields[5], 10),
                "fct_ns": int(fields[6], 10),
                "standalone_fct_ns": int(fields[7], 10),
            }
        except ValueError as exc:
            raise ValueError(f"FCT line {line_number} contains an invalid integer") from exc
        if flow_id < 0 or any(value < 0 for value in record.values()):
            raise ValueError(f"FCT line {line_number} contains a negative value")
        if flow_id in observations:
            raise ValueError(f"duplicate FCT flow_id {flow_id}")
        observations[flow_id] = record
    return observations


def fast_stage_violations(
    schedule: Schedule,
    observations: dict[int, dict[str, int]],
    *,
    collective_epoch_ns: int,
) -> dict[str, Any]:
    """Measure spill from each scheduled start slot into its next slot."""
    starts = sorted({flow.start_ns for flow in schedule.flows})
    boundaries: list[dict[str, int]] = []
    for previous_start, next_start in zip(starts, starts[1:]):
        previous_ids = [
            flow.flow_id for flow in schedule.flows if flow.start_ns == previous_start
        ]
        previous_end = max(
            observations[flow_id]["actual_start_ns"]
            + observations[flow_id]["fct_ns"]
            - collective_epoch_ns
            for flow_id in previous_ids
        )
        violation = max(0, previous_end - next_start)
        boundaries.append(
            {
                "previous_start_ns": previous_start,
                "next_start_ns": next_start,
                "actual_previous_slot_end_ns": previous_end,
                "violation_ns": violation,
            }
        )
    violations = [item["violation_ns"] for item in boundaries]
    count = sum(value > 0 for value in violations)
    return {
        "boundary_count": len(boundaries),
        "violation_count": count,
        "violation_fraction": count / len(boundaries) if boundaries else 0.0,
        "max_violation_ns": max(violations, default=0),
        "sum_violation_ns": sum(violations),
        "boundaries": boundaries,
    }


def analyze_fct(
    matrix_dir: Path,
    schedule_dir: Path,
    fct_path: Path,
    pfc_path: Path,
    qlen_path: Path | None = None,
    bw_path: Path | None = None,
    tor_id: int | None = None,
    tor_port_capacity_gbps: float | None = None,
) -> dict[str, Any]:
    matrix, matrix_manifest = read_matrix_artifact(matrix_dir)
    schedule, schedule_manifest = read_schedule_artifact(schedule_dir)
    if schedule_manifest["matrix_sha256"] != matrix_manifest["matrix_sha256"]:
        raise ValueError("schedule and matrix hashes do not match")
    observations = read_external_fct(fct_path)
    expected_ids = set(range(len(schedule.flows)))
    actual_ids = set(observations)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unknown = sorted(actual_ids - expected_ids)
        raise ValueError(f"FCT flow IDs differ; missing={missing}, unknown={unknown}")
    for flow in schedule.flows:
        observed = observations[flow.flow_id]
        expected = (flow.src, flow.dst, flow.size_bytes)
        actual = (observed["src"], observed["dst"], observed["size_bytes"])
        if actual != expected:
            raise ValueError(
                f"FCT flow_id {flow.flow_id} tuple mismatch: expected {expected}, got {actual}"
            )

    release_candidates = [
        observations[flow.flow_id]["actual_start_ns"] - flow.start_ns
        for flow in schedule.flows
    ]
    collective_epoch_ns = min(release_candidates)
    actual_end_ns = max(
        observation["actual_start_ns"] + observation["fct_ns"]
        for observation in observations.values()
    )
    actual_makespan_ns = actual_end_ns - collective_epoch_ns
    fcts = [observation["fct_ns"] for observation in observations.values()]
    predicted = schedule.predicted_makespan_ns
    summary: dict[str, Any] = {
        "schema_version": 1,
        "algorithm": schedule.algorithm,
        "flow_count": len(schedule.flows),
        "collective_epoch_ns": collective_epoch_ns,
        "actual_makespan_ns": actual_makespan_ns,
        "predicted_makespan_ns": predicted,
        "fidelity_ratio": (
            actual_makespan_ns / predicted if predicted is not None and predicted > 0 else None
        ),
        "logical_bytes": matrix.logical_bytes,
        "physical_bytes": schedule.physical_bytes,
        "scheduled_bytes": schedule.physical_bytes,
        "schedule_byte_amplification": (
            schedule.physical_bytes / matrix.logical_bytes
            if matrix.logical_bytes > 0
            else None
        ),
        "algorithm_bandwidth_gbps": (
            matrix.logical_bytes * 8 / actual_makespan_ns
            if actual_makespan_ns > 0
            else None
        ),
        "per_device_algorithm_bandwidth_gbps": (
            matrix.logical_bytes * 8 / actual_makespan_ns / matrix.num_devices
            if actual_makespan_ns > 0
            else None
        ),
        "fct_ns": {
            "mean": sum(fcts) / len(fcts),
            "p50": _percentile(fcts, 0.50),
            "p95": _percentile(fcts, 0.95),
            "p99": _percentile(fcts, 0.99),
            "max": max(fcts),
        },
        "pfc_count": sum(
            bool(line.strip())
            for line in Path(pfc_path).read_text(encoding="utf-8").splitlines()
        ),
    }
    if schedule.algorithm == "fast":
        summary["fast_stage_violations"] = fast_stage_violations(
            schedule,
            observations,
            collective_epoch_ns=collective_epoch_ns,
        )
    if qlen_path is not None or bw_path is not None:
        summary["network"] = analyze_network_monitors(
            qlen_path,
            bw_path,
            tor_id=tor_id,
            tor_port_capacity_gbps=tor_port_capacity_gbps,
        )
    return summary


def _monitor_rows(path: Path | None) -> list[list[str]]:
    if path is None or not Path(path).is_file():
        return []
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, skipinitialspace=True)
        next(reader, None)
        return [[field.strip() for field in row] for row in reader if row]


def analyze_network_monitors(
    qlen_path: Path | None,
    bw_path: Path | None,
    *,
    tor_id: int | None,
    tor_port_capacity_gbps: float | None,
) -> dict[str, Any]:
    qrows = _monitor_rows(qlen_path)
    brows = _monitor_rows(bw_path)
    qvalues: list[tuple[int, int, int]] = []
    for row in qrows:
        if len(row) != 6:
            raise ValueError("queue monitor row must have six columns")
        qvalues.append((int(row[1]), int(row[4]), int(row[5])))
    bvalues: list[tuple[int, float]] = []
    for row in brows:
        if len(row) != 4:
            raise ValueError("bandwidth monitor row must have four columns")
        bvalues.append((int(row[1]), float(row[3])))
    tor_q = [value for node, value, _ in qvalues if node == tor_id]
    tor_port_q = [value for node, _, value in qvalues if node == tor_id]
    tor_bw = [value for node, value in bvalues if node == tor_id]
    nonzero_bw = [value for _, value in bvalues if value > 0]
    network: dict[str, Any] = {
        "queue": {
            "row_count": len(qvalues),
            "peak_queue_bytes": max((value for _, value, _ in qvalues), default=0),
            "peak_port_bytes": max((value for _, _, value in qvalues), default=0),
            "tor_peak_queue_bytes": max(tor_q, default=0),
            "tor_peak_port_bytes": max(tor_port_q, default=0),
        },
        "bandwidth": {
            "row_count": len(bvalues),
            "peak_gbps": max((value for _, value in bvalues), default=0.0),
            "mean_active_gbps": (
                sum(nonzero_bw) / len(nonzero_bw) if nonzero_bw else 0.0
            ),
            "tor_peak_gbps": max(tor_bw, default=0.0),
            "tor_mean_active_gbps": (
                sum(value for value in tor_bw if value > 0)
                / sum(value > 0 for value in tor_bw)
                if any(value > 0 for value in tor_bw)
                else 0.0
            ),
        },
    }
    if tor_port_capacity_gbps:
        network["bandwidth"]["tor_peak_utilization"] = (
            max(tor_bw, default=0.0) / tor_port_capacity_gbps
        )
        active_tor = [value for value in tor_bw if value > 0]
        network["bandwidth"]["tor_mean_active_utilization"] = (
            (sum(active_tor) / len(active_tor) if active_tor else 0.0)
            / tor_port_capacity_gbps
        )
    return network


def write_summary(summary: dict[str, Any], path: Path) -> None:
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
