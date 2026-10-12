"""Adapter from the C++-faithful FAST port to the common schedule model."""

from __future__ import annotations

from ..model import HardwareProfile, Schedule, TrafficMatrix
from . import RawFlow, _finalize_schedule
from . import fast_port


class FastScheduler:
    def __init__(self, scaling: bool = False) -> None:
        self.scaling = scaling

    def schedule(
        self,
        matrix: TrafficMatrix,
        hardware: HardwareProfile,
        *,
        seed: int,
    ) -> Schedule:
        hardware.validate_matrix(matrix)
        if hardware.num_servers == 1:
            raw: list[RawFlow] = []
            stage_starts: list[int] = []
            slot_ends: list[int] = []
            current_start = 0
            for offset in range(1, hardware.gpus_per_server):
                stage_starts.append(current_start)
                stage_duration = 0
                for src in range(hardware.gpus_per_server):
                    dst = (src + offset) % hardware.gpus_per_server
                    size = int(matrix.values[src, dst])
                    if size > 0:
                        raw.append(
                            (src, dst, size, current_start, "intrinsic", offset - 1)
                        )
                        stage_duration = max(
                            stage_duration,
                            hardware.intra.duration_ns(size),
                        )
                current_start += stage_duration
                slot_ends.append(current_start)
            metadata = {
                "seed": seed,
                "scaling": self.scaling,
                "logical_bytes": matrix.logical_bytes,
                "physical_bytes": sum(flow[2] for flow in raw),
                "bound_ns": current_start,
                "baseline_spreadout_ns": current_start,
                "speedup_vs_spreadout": 1.0,
                "balance_end_ns": 0,
                "stage_starts_ns": stage_starts,
                "slot_ends_ns": slot_ends,
                "final_dispatch_start_ns": current_start,
                "n_stages": 0,
                "degenerate_single_server": True,
            }
            return _finalize_schedule(
                "fast",
                raw,
                hardware,
                current_start,
                metadata,
            )
        inter = {
            "alpha": hardware.inter.alpha_ns / 1000.0,
            "beta": hardware.inter.beta_ns_per_byte * 1000.0,
        }
        intra = {
            "alpha": hardware.intra.alpha_ns / 1000.0,
            "beta": hardware.intra.beta_ns_per_byte * 1000.0,
        }
        result = fast_port.run_fast_full(
            matrix.values,
            hardware.num_servers,
            hardware.gpus_per_server,
            "ETHER400",
            "H100",
            scaling=self.scaling,
            MBpu=1e-6,
            pipeline="p2",
            inter_override=inter,
            intra_override=intra,
        )
        legacy_flows, timeline = fast_port.emit_schedule(result, MBpu=1e-6)
        raw: list[RawFlow] = []
        for flow, extra in zip(legacy_flows, timeline["flow_metadata"]):
            src, dst, size, start_us = flow
            raw.append(
                (
                    int(src),
                    int(dst),
                    int(size),
                    int(round(float(start_us) * 1000.0)),
                    str(extra["phase"]),
                    int(extra["stage_id"]),
                )
            )

        physical_bytes = sum(flow[2] for flow in raw)
        metadata = {
            "seed": seed,
            "scaling": self.scaling,
            "logical_bytes": matrix.logical_bytes,
            "physical_bytes": physical_bytes,
            "bound_ns": int(round(float(result["bound"]) * 1000.0)),
            "baseline_spreadout_ns": int(round(float(result["baseline"]) * 1000.0)),
            "speedup_vs_spreadout": float(result["speedup"]),
            "balance_end_ns": int(round(float(timeline["t_bal_us"]) * 1000.0)),
            "stage_starts_ns": [
                int(round(float(value) * 1000.0))
                for value in timeline["stage_starts_us"]
            ],
            "slot_ends_ns": [
                int(round(float(value) * 1000.0))
                for value in timeline["slot_ends_us"]
            ],
            "final_dispatch_start_ns": int(
                round(float(timeline["final_dispatch_start_us"]) * 1000.0)
            ),
            "n_stages": int(timeline["n_stages"]),
        }
        return _finalize_schedule(
            "fast",
            raw,
            hardware,
            int(round(float(result["t"]) * 1000.0)),
            metadata,
        )
