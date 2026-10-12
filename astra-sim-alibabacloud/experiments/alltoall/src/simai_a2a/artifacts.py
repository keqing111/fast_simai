"""Versioned, hash-checked matrix and schedule artifact I/O."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .model import Schedule, ScheduledFlow, TrafficMatrix


SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_matrix_artifact(
    matrix: TrafficMatrix,
    directory: Path,
    *,
    manifest_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    matrix_path = directory / "matrix.npy"
    np.save(matrix_path, matrix.values, allow_pickle=False)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "num_devices": matrix.num_devices,
        "logical_bytes": matrix.logical_bytes,
        "matrix_sha256": _sha256(matrix_path),
        "metadata": matrix.metadata,
    }
    if manifest_fields:
        manifest.update(manifest_fields)
    _write_json(directory / "matrix_manifest.json", manifest)
    return manifest


def read_matrix_artifact(directory: Path) -> tuple[TrafficMatrix, dict[str, Any]]:
    directory = Path(directory)
    matrix_path = directory / "matrix.npy"
    manifest = json.loads((directory / "matrix_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported matrix schema version")
    if _sha256(matrix_path) != manifest.get("matrix_sha256"):
        raise ValueError("matrix hash mismatch")
    values = np.load(matrix_path, allow_pickle=False)
    matrix = TrafficMatrix(values, manifest.get("metadata", {}))
    if matrix.num_devices != manifest.get("num_devices"):
        raise ValueError("matrix shape does not match manifest")
    if matrix.logical_bytes != manifest.get("logical_bytes"):
        raise ValueError("matrix logical byte count does not match manifest")
    return matrix, manifest


def write_schedule_artifact(
    schedule: Schedule,
    directory: Path,
    *,
    matrix_sha256: str,
    hardware_sha256: str,
    manifest_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    schedule_path = directory / "schedule.csv"
    metadata_path = directory / "schedule_metadata.csv"
    with schedule_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        for flow in schedule.flows:
            writer.writerow((flow.src, flow.dst, flow.size_bytes, flow.start_ns))
    with metadata_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("flow_id", "phase", "stage_id"),
            lineterminator="\n",
        )
        writer.writeheader()
        for flow in schedule.flows:
            writer.writerow(
                {"flow_id": flow.flow_id, "phase": flow.phase, "stage_id": flow.stage_id}
            )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": schedule.algorithm,
        "matrix_sha256": matrix_sha256,
        "hardware_sha256": hardware_sha256,
        "schedule_sha256": _sha256(schedule_path),
        "schedule_metadata_sha256": _sha256(metadata_path),
        "flow_count": len(schedule.flows),
        "physical_bytes": schedule.physical_bytes,
        "predicted_makespan_ns": schedule.predicted_makespan_ns,
        "metadata": schedule.metadata,
    }
    if manifest_fields:
        manifest.update(manifest_fields)
    _write_json(directory / "schedule_manifest.json", manifest)
    return manifest


def read_schedule_artifact(directory: Path) -> tuple[Schedule, dict[str, Any]]:
    directory = Path(directory)
    schedule_path = directory / "schedule.csv"
    metadata_path = directory / "schedule_metadata.csv"
    manifest = json.loads((directory / "schedule_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported schedule schema version")
    if _sha256(schedule_path) != manifest.get("schedule_sha256"):
        raise ValueError("schedule hash mismatch")
    if _sha256(metadata_path) != manifest.get("schedule_metadata_sha256"):
        raise ValueError("schedule metadata hash mismatch")

    schedule_rows: list[tuple[int, int, int, int]] = []
    with schedule_path.open(encoding="utf-8", newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream), start=1):
            if len(row) != 4:
                raise ValueError(f"schedule row {line_number} must have four columns")
            try:
                schedule_rows.append(tuple(int(value) for value in row))
            except ValueError as exc:
                raise ValueError(f"schedule row {line_number} must contain integers") from exc

    metadata_rows: list[dict[str, str]] = []
    with metadata_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["flow_id", "phase", "stage_id"]:
            raise ValueError("schedule metadata header mismatch")
        metadata_rows = list(reader)
    if len(schedule_rows) != len(metadata_rows):
        raise ValueError("schedule and metadata row counts differ")

    flows: list[ScheduledFlow] = []
    for index, (row, metadata) in enumerate(zip(schedule_rows, metadata_rows)):
        if int(metadata["flow_id"]) != index:
            raise ValueError("schedule metadata flow_id does not match row index")
        flows.append(
            ScheduledFlow(
                index,
                row[0],
                row[1],
                row[2],
                row[3],
                metadata["phase"],
                int(metadata["stage_id"]),
            )
        )
    schedule = Schedule(
        manifest["algorithm"],
        flows,
        manifest.get("predicted_makespan_ns"),
        manifest.get("metadata", {}),
    )
    if len(schedule.flows) != manifest.get("flow_count"):
        raise ValueError("schedule flow count does not match manifest")
    if schedule.physical_bytes != manifest.get("physical_bytes"):
        raise ValueError("schedule physical byte count does not match manifest")
    return schedule, manifest
