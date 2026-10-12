"""Scheduler registry and shared deterministic schedule construction."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..model import HardwareProfile, Schedule, ScheduledFlow


RawFlow = tuple[int, int, int, int, str, int]


def _finalize_schedule(
    algorithm: str,
    raw_flows: Iterable[RawFlow],
    hardware: HardwareProfile,
    predicted_makespan_ns: int | None,
    metadata: dict[str, Any],
) -> Schedule:
    ordered = sorted(
        raw_flows,
        key=lambda flow: (flow[3], flow[4], flow[5], flow[0], flow[1], flow[2]),
    )
    flows = [
        ScheduledFlow(index, src, dst, size, start, phase, stage)
        for index, (src, dst, size, start, phase, stage) in enumerate(ordered)
    ]
    for flow in flows:
        flow.validate(hardware)
    return Schedule(algorithm, flows, predicted_makespan_ns, metadata)


def get_scheduler(name: str):
    if name == "direct-concurrent":
        from .direct import DirectConcurrentScheduler

        return DirectConcurrentScheduler()
    if name == "spreadout":
        from .spreadout import SpreadOutScheduler

        return SpreadOutScheduler()
    if name == "roundrobin":
        from .roundrobin import RoundRobinScheduler

        return RoundRobinScheduler()
    if name == "fast":
        try:
            from .fast import FastScheduler
        except ImportError as exc:
            raise RuntimeError("FAST scheduler is not installed") from exc
        return FastScheduler()
    accepted = "direct-concurrent, fast, roundrobin, spreadout"
    raise ValueError(f"unknown scheduler {name!r}; expected one of: {accepted}")


__all__ = ["get_scheduler"]
