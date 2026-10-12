"""Deterministic synthetic traffic matrices matching FAST benchmark semantics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .model import TrafficMatrix


MAX_ELEMENT = 10000
_INT64_MAX = int(np.iinfo(np.int64).max)


def _validate_common(num_devices: int, bytes_per_unit: int) -> None:
    if num_devices < 2:
        raise ValueError("num_devices must be at least two")
    if bytes_per_unit <= 0:
        raise ValueError("byte scale must be positive")


def _normalize_total(values: np.ndarray, total_bytes: int) -> np.ndarray:
    """Scale nonnegative integer weights to an exact byte total.

    Largest-remainder allocation with row-major tie breaking makes the result
    deterministic and avoids floating-point or int64 multiplication overflow.
    """
    if total_bytes <= 0 or total_bytes > _INT64_MAX:
        raise ValueError("total_bytes must be within signed int64")
    flat = [int(value) for value in np.asarray(values).reshape(-1)]
    weight_sum = sum(flat)
    if weight_sum <= 0:
        raise ValueError("cannot normalize an all-zero weight matrix")
    quotients = [weight * total_bytes // weight_sum for weight in flat]
    remainders = [weight * total_bytes % weight_sum for weight in flat]
    missing = total_bytes - sum(quotients)
    order = sorted(range(len(flat)), key=lambda index: (-remainders[index], index))
    for index in order[:missing]:
        quotients[index] += 1
    result = np.asarray(quotients, dtype=np.int64).reshape(values.shape)
    if int(result.sum(dtype=object)) != total_bytes:
        raise AssertionError("fixed-total normalization did not conserve bytes")
    return result


def balanced_matrix(
    num_devices: int,
    pair_bytes: int | None = None,
    total_bytes: int | None = None,
) -> TrafficMatrix:
    if (pair_bytes is None) == (total_bytes is None):
        raise ValueError("exactly one of pair_bytes or total_bytes is required")
    if total_bytes is not None:
        _validate_common(num_devices, total_bytes)
        weights = np.ones((num_devices, num_devices), dtype=np.int64)
        np.fill_diagonal(weights, 0)
        values = _normalize_total(weights, total_bytes)
        return TrafficMatrix(
            values,
            {
                "generator": "balanced",
                "total_bytes": total_bytes,
                "normalization": "fixed-total",
                "seed": None,
            },
        )
    assert pair_bytes is not None
    _validate_common(num_devices, pair_bytes)
    if pair_bytes > _INT64_MAX:
        raise ValueError("pair_bytes exceeds signed int64")
    values = np.full((num_devices, num_devices), pair_bytes, dtype=np.int64)
    np.fill_diagonal(values, 0)
    return TrafficMatrix(
        values,
        {
            "generator": "balanced",
            "pair_bytes": pair_bytes,
            "seed": None,
        },
    )


def fast_uniform_matrix(
    num_devices: int,
    bytes_per_unit: int,
    seed: int,
    total_bytes: int | None = None,
) -> TrafficMatrix:
    _validate_common(num_devices, bytes_per_unit)
    if bytes_per_unit > _INT64_MAX // (MAX_ELEMENT - 1):
        raise ValueError("uniform scaling exceeds signed int64")
    rng = np.random.Generator(np.random.PCG64(seed))
    weights = rng.integers(
        0,
        MAX_ELEMENT,
        size=(num_devices, num_devices),
        dtype=np.int64,
    )
    np.fill_diagonal(weights, 0)
    if total_bytes is None:
        values = weights * np.int64(bytes_per_unit)
        normalization = "native"
    else:
        values = _normalize_total(weights, total_bytes)
        normalization = "fixed-total"
    return TrafficMatrix(
        values,
        {
            "generator": "fast-uniform",
            "bytes_per_unit": bytes_per_unit,
            "seed": seed,
            "total_bytes": total_bytes,
            "normalization": normalization,
            "weight_min": 0,
            "weight_max_exclusive": MAX_ELEMENT,
        },
    )


def zipf_inverse_cdf_fast(probability: float, maximum: int, alpha: float) -> int:
    if probability < 0.0 or probability > 1.0:
        raise ValueError("probability must be within [0, 1]")
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must be within (0, 1)")

    denominator = 1.0 - alpha
    target = probability * (
        12.0 * (math.pow(maximum, denominator) - 1.0) / denominator
        + 6.0
        - 6.0 * math.pow(maximum, -alpha)
        + alpha
        - math.pow(maximum, -1.0 - alpha) * alpha
    )
    x = maximum / 2.0
    for _ in range(1000):
        m = math.pow(x, -2.0 - alpha)
        mx = m * x
        mxx = mx * x
        mxxx = mxx * x
        a = (
            12.0 * (mxxx - 1.0) / denominator
            + 6.0 * (1.0 - mxx)
            + alpha
            - mx * alpha
            - target
        )
        b = 12.0 * mxx + 6.0 * alpha * mx + m * alpha * (alpha + 1.0)
        new_x = max(1.0, x - a / b)
        if abs(new_x - x) <= 0.01:
            return int(new_x)
        x = new_x
    raise RuntimeError("FAST Zipf inverse CDF did not converge")


def fast_zipf_matrix(
    num_devices: int,
    alpha: float,
    bytes_per_unit: int,
    seed: int,
    total_bytes: int | None = None,
) -> TrafficMatrix:
    _validate_common(num_devices, bytes_per_unit)
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must be within (0, 1)")
    if bytes_per_unit > _INT64_MAX // MAX_ELEMENT:
        raise ValueError("Zipf scaling exceeds signed int64")
    rng = np.random.Generator(np.random.PCG64(seed))
    probabilities = rng.random(num_devices * num_devices)
    weights = np.fromiter(
        (
            zipf_inverse_cdf_fast(float(probability), MAX_ELEMENT, alpha)
            for probability in probabilities
        ),
        dtype=np.int64,
        count=num_devices * num_devices,
    ).reshape((num_devices, num_devices))
    np.fill_diagonal(weights, 0)
    if total_bytes is None:
        values = weights * np.int64(bytes_per_unit)
        normalization = "native"
    else:
        values = _normalize_total(weights, total_bytes)
        normalization = "fixed-total"
    return TrafficMatrix(
        values,
        {
            "generator": "fast-zipf",
            "alpha": alpha,
            "bytes_per_unit": bytes_per_unit,
            "seed": seed,
            "total_bytes": total_bytes,
            "normalization": normalization,
            "weight_maximum": MAX_ELEMENT,
        },
    )


def generate_matrix(name: str, **kwargs: Any) -> TrafficMatrix:
    generators = {
        "balanced": balanced_matrix,
        "fast-uniform": fast_uniform_matrix,
        "fast-zipf": fast_zipf_matrix,
    }
    try:
        generator = generators[name]
    except KeyError as exc:
        accepted = ", ".join(sorted(generators))
        raise ValueError(f"unknown generator {name!r}; expected one of: {accepted}") from exc
    return generator(**kwargs)
