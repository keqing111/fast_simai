"""All nonzero point-to-point demands released concurrently."""

from ..model import HardwareProfile, Schedule, TrafficMatrix
from . import RawFlow, _finalize_schedule


class DirectConcurrentScheduler:
    def schedule(
        self,
        matrix: TrafficMatrix,
        hardware: HardwareProfile,
        *,
        seed: int,
    ) -> Schedule:
        hardware.validate_matrix(matrix)
        raw: list[RawFlow] = []
        for src in range(matrix.num_devices):
            for dst in range(matrix.num_devices):
                size = int(matrix.values[src, dst])
                if size > 0:
                    raw.append((src, dst, size, 0, "direct", 0))
        return _finalize_schedule(
            "direct-concurrent",
            raw,
            hardware,
            None,
            {
                "seed": seed,
                "logical_bytes": matrix.logical_bytes,
                "physical_bytes": matrix.logical_bytes,
                "stage_starts_ns": [0],
            },
        )
