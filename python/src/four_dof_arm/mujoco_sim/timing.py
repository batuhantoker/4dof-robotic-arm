from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _positive_finite_float(value: object, name: str) -> float:
    try:
        scalar = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if scalar.shape != ():
        raise ValueError(f"{name} must be a finite positive number")

    result = float(scalar)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _rounded_up_count(value: float) -> int:
    """Ceil a nonnegative count without promoting integral roundoff."""
    if value == 0.0:
        return 0

    nearest = round(value)
    tolerance = 8.0 * math.ulp(max(1.0, abs(value)))
    if nearest > 0 and abs(value - nearest) <= tolerance:
        return nearest
    return math.ceil(value)


def _step_count(duration: float, timestep: float) -> int:
    return _rounded_up_count(duration / timestep)


def _frame_count(duration: float, fps: float) -> int:
    return max(1, _rounded_up_count(duration * fps))


def _validate_trajectory(
    timestamps: ArrayLike,
    q_trajectory: ArrayLike,
    n_joints: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    times = np.asarray(timestamps, dtype=np.float64)
    targets = np.asarray(q_trajectory, dtype=np.float64)

    if times.ndim != 1 or times.size < 2:
        raise ValueError(
            "timestamps must be a one-dimensional array with at least two values"
        )
    if not np.all(np.isfinite(times)):
        raise ValueError("timestamps must be finite")
    if times[0] != 0.0:
        raise ValueError("timestamps must start at zero")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    if targets.shape != (times.size, n_joints):
        raise ValueError(
            f"q_trajectory must have shape ({times.size}, {n_joints}), got {targets.shape}"
        )
    if not np.all(np.isfinite(targets)):
        raise ValueError("q_trajectory must be finite")

    return times, targets


def _interpolate_target(
    timestamps: NDArray[np.float64],
    q_trajectory: NDArray[np.float64],
    time: float,
) -> NDArray[np.float64]:
    return np.array(
        [
            np.interp(time, timestamps, q_trajectory[:, joint])
            for joint in range(q_trajectory.shape[1])
        ],
        dtype=np.float64,
    )
