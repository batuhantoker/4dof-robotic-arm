"""Input validation helpers for the robot model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def as_finite_vector(value: object, size: int, name: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite vector of shape ({size},)") from exc

    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of shape ({size},)")
    return array


def finite_scalar(value: object, name: str, *, positive: bool = False) -> float:
    qualifier = "positive finite" if positive else "finite"
    if isinstance(value, (bool, np.bool_)) or (
        isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be a {qualifier} scalar")

    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a {qualifier} scalar") from exc

    if array.shape != ():
        raise ValueError(f"{name} must be a {qualifier} scalar")

    result = float(array)
    if not np.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{name} must be a {qualifier} scalar")
    return result


def positive_integer(value: object, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def random_generator(rng: np.random.Generator | None) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    return rng
