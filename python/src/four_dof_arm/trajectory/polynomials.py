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
    """
    q_start, q_end, t, T = _polynomial_inputs(q_start, q_end, t, T)

    a0 = q_start
    a1 = 0.0
    a2 = 3.0 * (q_end - q_start) / T**2
    a3 = -2.0 * (q_end - q_start) / T**3

    q = a0 + a1 * t + a2 * t**2 + a3 * t**3
    qd = a1 + 2.0 * a2 * t + 3.0 * a3 * t**2
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
    """
    q_start, q_end, t, T = _polynomial_inputs(q_start, q_end, t, T)

    dq = q_end - q_start
    a0 = q_start
    a3 = 10.0 * dq / T**3
    a4 = -15.0 * dq / T**4
    a5 = 6.0 * dq / T**5

    q = a0 + a3 * t**3 + a4 * t**4 + a5 * t**5
    qd = 3.0 * a3 * t**2 + 4.0 * a4 * t**3 + 5.0 * a5 * t**4
    qdd = 6.0 * a3 * t + 12.0 * a4 * t**2 + 20.0 * a5 * t**3
    return q, qd, qdd
