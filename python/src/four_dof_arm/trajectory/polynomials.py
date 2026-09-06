from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .validation import _polynomial_inputs


def cubic_polynomial(
    q_start: float,
    q_end: float,
    t: NDArray[np.float64],
    T: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cubic polynomial interpolation with zero velocity boundary conditions.

    q(t) = a0 + a1*t + a2*t² + a3*t³
    Boundary conditions: q(0)=q_start, q(T)=q_end, q'(0)=0, q'(T)=0

    Parameters
    ----------
    q_start, q_end : float
        Start and end joint angles.
    t : array
        Time vector.
    T : float
        Total trajectory duration.

    Returns
    -------
    q, qd : arrays
        Position and velocity profiles.

    Raises
    ------
    ValueError
        If inputs are invalid or output profiles are not finite in float64.
    """
    q_start, q_end, t, T = _polynomial_inputs(q_start, q_end, t, T)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        u = t / T
        dq = q_end - q_start
        q = q_start + dq * (u**2 * (3.0 - 2.0 * u))
        qd = dq * (6.0 * u * (1.0 - u)) / T

    if not all(np.all(np.isfinite(profile)) for profile in (q, qd)):
        raise ValueError("polynomial output must contain only finite values")
    return q, qd


def quintic_polynomial(
    q_start: float,
    q_end: float,
    t: NDArray[np.float64],
    T: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Quintic polynomial interpolation with zero velocity and acceleration BCs.

    Boundary conditions: q(0)=q_start, q(T)=q_end,
    q'(0)=0, q'(T)=0, q''(0)=0, q''(T)=0

    Returns
    -------
    q, qd, qdd : arrays
        Position, velocity, and acceleration profiles.

    Raises
    ------
    ValueError
        If inputs are invalid or output profiles are not finite in float64.
    """
    q_start, q_end, t, T = _polynomial_inputs(q_start, q_end, t, T)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        u = t / T
        dq = q_end - q_start
        q = q_start + dq * (u**3 * (10.0 + u * (-15.0 + 6.0 * u)))
        qd = dq * (30.0 * u**2 * (1.0 - u) ** 2) / T
        qdd = dq * (60.0 * u * (1.0 - u) * (1.0 - 2.0 * u)) / T / T

    if not all(np.all(np.isfinite(profile)) for profile in (q, qd, qdd)):
        raise ValueError("polynomial output must contain only finite values")
    return q, qd, qdd
