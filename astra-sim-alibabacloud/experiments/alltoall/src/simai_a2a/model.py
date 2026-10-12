"""Typed contracts shared by traffic generators, schedulers, and runners."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class LinkModel:
    alpha_ns: float
    beta_ns_per_byte: float

    def __post_init__(self) -> None:
        if self.alpha_ns < 0 or self.beta_ns_per_byte < 0:
            raise ValueError("link alpha and beta must be non-negative")

    def duration_ns(self, size_bytes: int) -> int:
        if size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        return int(round(self.alpha_ns + self.beta_ns_per_byte * size_bytes))


@dataclass(frozen=True)
class HardwareProfile:
    num_servers: int
    gpus_per_server: int
    intra: LinkModel
    inter: LinkModel
    topology_id: str
    link_bandwidth_gbps: float
    oversubscription: float

    def __post_init__(self) -> None:
        if self.num_servers <= 0 or self.gpus_per_server <= 0:
            raise ValueError("server and GPU counts must be positive")
        if not self.topology_id:
            raise ValueError("topology_id must not be empty")
        if self.link_bandwidth_gbps <= 0 or self.oversubscription <= 0:
            raise ValueError("bandwidth and oversubscription must be positive")

    @property
    def num_devices(self) -> int:
        return self.num_servers * self.gpus_per_server

    def validate_matrix(self, matrix: "TrafficMatrix") -> None:
        if matrix.num_devices != self.num_devices:
            raise ValueError(
                f"matrix has {matrix.num_devices} devices, profile has {self.num_devices}"
            )


@dataclass(frozen=True)
class TrafficMatrix:
    values: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = self.values
        if not isinstance(values, np.ndarray):
            raise ValueError("matrix must be a NumPy array")
        if values.ndim != 2 or values.shape[0] != values.shape[1]:
            raise ValueError("matrix must be square")
        if values.dtype != np.dtype(np.int64):
            raise ValueError("matrix dtype must be int64")
        if np.any(values < 0):
            raise ValueError("matrix values must be non-negative")
        if np.any(np.diag(values) != 0):
            raise ValueError("matrix diagonal must be zero")
        try:
            json.dumps(self.metadata, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("matrix metadata must be JSON-compatible") from exc

    @property
    def num_devices(self) -> int:
        return int(self.values.shape[0])

    @property
    def logical_bytes(self) -> int:
        return sum(int(value) for value in self.values.flat)


@dataclass(frozen=True)
class ScheduledFlow:
    flow_id: int
    src: int
    dst: int
    size_bytes: int
    start_ns: int
    phase: str
    stage_id: int

    def validate(self, hardware: HardwareProfile) -> None:
        if self.flow_id < 0:
            raise ValueError("flow_id must be non-negative")
        if not (0 <= self.src < hardware.num_devices):
            raise ValueError("flow source is outside the hardware profile")
        if not (0 <= self.dst < hardware.num_devices):
            raise ValueError("flow destination is outside the hardware profile")
        if self.src == self.dst:
            raise ValueError("self flows are not allowed")
        if self.size_bytes <= 0:
            raise ValueError("flow size must be positive")
        if self.start_ns < 0:
            raise ValueError("flow start must be non-negative")
        if not self.phase:
            raise ValueError("flow phase must not be empty")
        if self.stage_id < 0:
            raise ValueError("stage_id must be non-negative")

    @property
    def sort_key(self) -> tuple[int, str, int, int, int, int]:
        return (
            self.start_ns,
            self.phase,
            self.stage_id,
            self.src,
            self.dst,
            self.size_bytes,
        )


@dataclass(frozen=True)
class Schedule:
    algorithm: str
    flows: list[ScheduledFlow]
    predicted_makespan_ns: int | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.algorithm:
            raise ValueError("algorithm must not be empty")
        if self.predicted_makespan_ns is not None and self.predicted_makespan_ns < 0:
            raise ValueError("predicted makespan must be non-negative")
        if self.flows != sorted(self.flows, key=lambda flow: flow.sort_key):
            raise ValueError("flows must be in deterministic schedule order")
        for index, flow in enumerate(self.flows):
            if flow.flow_id != index:
                raise ValueError("flow_id must equal the zero-based schedule row")
        try:
            json.dumps(self.metadata, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("schedule metadata must be JSON-compatible") from exc

    @property
    def physical_bytes(self) -> int:
        return sum(flow.size_bytes for flow in self.flows)


class AllToAllScheduler(Protocol):
    def schedule(
        self,
        matrix: TrafficMatrix,
        hardware: HardwareProfile,
        *,
        seed: int,
    ) -> Schedule:
        """Return a deterministic schedule for the supplied inputs."""
