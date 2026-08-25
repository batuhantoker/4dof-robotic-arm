from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .polynomials import cubic_polynomial, quintic_polynomial
from .validation import _finite_vector, _method_name, _point_count, _positive_duration


class JointTrajectory:
    """Joint-space trajectory between two configurations.

    Parameters
    ----------
    q_start : array of shape (4,)
        Starting joint angles.
    q_end : array of shape (4,)
        Ending joint angles.
    duration : float
        Trajectory duration in seconds.
    n_points : int
        Number of trajectory points.
    method : str
        'cubic' or 'quintic' polynomial.
    """

    def __init__(
        self,
        q_start: NDArray[np.float64],
        q_end: NDArray[np.float64],
        duration: float = 2.0,
        n_points: int = 100,
        method: str = "quintic",
    ) -> None:
        self.q_start = _finite_vector(q_start, (4,), "q_start")
        self.q_end = _finite_vector(q_end, (4,), "q_end")
        self.duration = _positive_duration(duration)
        self.n_points = _point_count(n_points, "n_points")
        self.method = _method_name(method)

        self.t: NDArray[np.float64] = np.linspace(
            0, self.duration, self.n_points, dtype=np.float64
        )
        self._compute()

    def _compute(self) -> None:
        n_joints = len(self.q_start)
        self.q = np.zeros((self.n_points, n_joints))
        self.qd = np.zeros((self.n_points, n_joints))
        self.qdd = (
            np.zeros((self.n_points, n_joints)) if self.method == "quintic" else None
        )

        for j in range(n_joints):
            if self.method == "cubic":
                self.q[:, j], self.qd[:, j] = cubic_polynomial(
                    self.q_start[j], self.q_end[j], self.t, self.duration
                )
            else:
                assert self.qdd is not None
                self.q[:, j], self.qd[:, j], self.qdd[:, j] = quintic_polynomial(
                    self.q_start[j], self.q_end[j], self.t, self.duration
                )
