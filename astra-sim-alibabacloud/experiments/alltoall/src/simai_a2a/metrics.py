"""Traffic-skew metrics used for experiment provenance and analysis."""

from __future__ import annotations

import numpy as np

from .model import TrafficMatrix


def gini(values: np.ndarray) -> float:
    flat = np.asarray(values).reshape(-1)
    if np.any(flat < 0):
        raise ValueError("Gini values must be non-negative")
    if flat.size == 0:
        return 0.0
    ordered = sorted(int(value) for value in flat)
    total = sum(ordered)
    if total == 0:
        return 0.0
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    count = len(ordered)
    return (2.0 * weighted) / (count * total) - (count + 1.0) / count


def _bottleneck_ratio(values: list[int]) -> float:
    total = sum(values)
    if total == 0:
        return 0.0
    return max(values) / (total / len(values))


def matrix_skew_metrics(
    matrix: TrafficMatrix,
    gpus_per_server: int,
) -> dict[str, float]:
    if gpus_per_server <= 0:
        raise ValueError("gpus_per_server must be positive")
    if matrix.num_devices % gpus_per_server != 0:
        raise ValueError("device count must be divisible by gpus_per_server")

    values = matrix.values
    off_diagonal = values[~np.eye(matrix.num_devices, dtype=bool)]
    source_sums = [sum(int(value) for value in values[row, :]) for row in range(matrix.num_devices)]
    destination_sums = [
        sum(int(value) for value in values[:, column])
        for column in range(matrix.num_devices)
    ]

    server_count = matrix.num_devices // gpus_per_server
    server_pairs: list[int] = []
    for src_server in range(server_count):
        src = slice(src_server * gpus_per_server, (src_server + 1) * gpus_per_server)
        for dst_server in range(server_count):
            if src_server == dst_server:
                continue
            dst = slice(dst_server * gpus_per_server, (dst_server + 1) * gpus_per_server)
            server_pairs.append(
                sum(int(value) for value in values[src, dst].flat)
            )

    return {
        "cell_gini": gini(off_diagonal),
        "source_gini": gini(np.asarray(source_sums, dtype=object)),
        "destination_gini": gini(np.asarray(destination_sums, dtype=object)),
        "server_pair_gini": gini(np.asarray(server_pairs, dtype=object)),
        "source_bottleneck_ratio": _bottleneck_ratio(source_sums),
        "destination_bottleneck_ratio": _bottleneck_ratio(destination_sums),
    }
