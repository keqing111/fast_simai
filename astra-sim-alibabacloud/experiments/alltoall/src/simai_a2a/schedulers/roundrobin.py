"""Deterministic server/GPU cyclic round-robin baseline."""

from ..model import HardwareProfile, Schedule, TrafficMatrix
from . import RawFlow, _finalize_schedule


class RoundRobinScheduler:
    def schedule(
        self,
        matrix: TrafficMatrix,
        hardware: HardwareProfile,
        *,
        seed: int,
    ) -> Schedule:
        hardware.validate_matrix(matrix)
        raw: list[RawFlow] = []
        current_start = 0
        stage_id = 0
        stage_starts: list[int] = []
        gpu_count = hardware.gpus_per_server

        for gpu_offset in range(1, gpu_count):
            stage_starts.append(current_start)
            duration = 0
            for server in range(hardware.num_servers):
                base = server * gpu_count
                for gpu in range(gpu_count):
                    src = base + gpu
                    dst = base + (gpu + gpu_offset) % gpu_count
                    size = int(matrix.values[src, dst])
                    if size == 0:
                        continue
                    duration = max(duration, hardware.intra.duration_ns(size))
                    raw.append((src, dst, size, current_start, "direct", stage_id))
            current_start += duration
            stage_id += 1

        for server_offset in range(1, hardware.num_servers):
            for gpu_offset in range(gpu_count):
                stage_starts.append(current_start)
                duration = 0
                for src_server in range(hardware.num_servers):
                    dst_server = (src_server + server_offset) % hardware.num_servers
                    for gpu in range(gpu_count):
                        src = src_server * gpu_count + gpu
                        dst = dst_server * gpu_count + (gpu + gpu_offset) % gpu_count
                        size = int(matrix.values[src, dst])
                        if size == 0:
                            continue
                        duration = max(duration, hardware.inter.duration_ns(size))
                        raw.append((src, dst, size, current_start, "direct", stage_id))
                current_start += duration
                stage_id += 1

        return _finalize_schedule(
            "roundrobin",
            raw,
            hardware,
            current_start,
            {
                "seed": seed,
                "logical_bytes": matrix.logical_bytes,
                "physical_bytes": sum(flow[2] for flow in raw),
                "stage_starts_ns": stage_starts,
            },
        )
