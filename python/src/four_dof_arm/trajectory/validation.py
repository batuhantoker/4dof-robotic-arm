from __future__ import annotations

import operator
from typing import SupportsIndex

import numpy as np
from numpy.typing import NDArray


def _finite_scalar(value: object, name: str) -> float:
    try:
        scalar = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc

    if scalar.ndim != 0 or not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite scalar")
    return float(scalar)


def _positive_duration(value: object, name: str = "duration") -> float:
    try:
        duration = _finite_scalar(value, name)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc

    if duration <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return duration


def _finite_vector(
    value: object,
    shape: tuple[int, ...],
    name: str,
) -> NDArray[np.float64]:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite array with shape {shape}") from exc

    if vector.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _point_count(value: SupportsIndex, name: str) -> int:
    try:
        count = operator.index(value)
    except TypeError as exc:
        raise ValueError(
            f"{name} must be an integer greater than or equal to 2"
        ) from exc

    if count < 2:
        raise ValueError(f"{name} must be greater than or equal to 2")
    return count


def _method_name(method: object) -> str:
    if not isinstance(method, str) or method not in ("cubic", "quintic"):
        raise ValueError("method must be 'cubic' or 'quintic'")
    return method


def _polynomial_inputs(
    q_start: object,
    q_end: object,
    t: object,
    duration: object,
) -> tuple[float, float, NDArray[np.float64], float]:
    start = _finite_scalar(q_start, "q_start")
    end = _finite_scalar(q_end, "q_end")
    total = _positive_duration(duration, "T")

    try:
        times = np.asarray(t, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("t must be a finite one-dimensional array") from exc

    if times.ndim != 1 or times.size == 0:
        raise ValueError("t must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(times)):
        raise ValueError("t must contain only finite values")
    return start, end, times, total
