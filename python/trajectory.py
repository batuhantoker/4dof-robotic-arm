"""
Trajectory Planning for 4-DOF Manipulator

Supports joint-space polynomial interpolation and Cartesian-space
linear interpolation with configurable time parameterization.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from robot import Robot4DOF


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
    dq = q_end - q_start
    a0 = q_start
    a3 = 10.0 * dq / T**3
    a4 = -15.0 * dq / T**4
    a5 = 6.0 * dq / T**5

    q = a0 + a3 * t**3 + a4 * t**4 + a5 * t**5
    qd = 3.0 * a3 * t**2 + 4.0 * a4 * t**3 + 5.0 * a5 * t**4
    qdd = 6.0 * a3 * t + 12.0 * a4 * t**2 + 20.0 * a5 * t**3
    return q, qd, qdd


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
        self.q_start = np.asarray(q_start, dtype=np.float64)
        self.q_end = np.asarray(q_end, dtype=np.float64)
        self.duration = duration
        self.n_points = n_points
        self.method = method

        self.t = np.linspace(0, duration, n_points)
        self._compute()

    def _compute(self) -> None:
        n_joints = len(self.q_start)
        self.q = np.zeros((self.n_points, n_joints))
        self.qd = np.zeros((self.n_points, n_joints))
        self.qdd = np.zeros((self.n_points, n_joints)) if self.method == "quintic" else None

        for j in range(n_joints):
            if self.method == "cubic":
                self.q[:, j], self.qd[:, j] = cubic_polynomial(
                    self.q_start[j], self.q_end[j], self.t, self.duration
                )
            else:
                self.q[:, j], self.qd[:, j], self.qdd[:, j] = quintic_polynomial(
                    self.q_start[j], self.q_end[j], self.t, self.duration
                )


class CartesianTrajectory:
    """Cartesian-space linear trajectory with IK at each point.

    Interpolates linearly in Cartesian space between start and end positions,
    then solves IK at each waypoint.

    Parameters
    ----------
    robot : Robot4DOF
        Robot instance for IK computation.
    p_start : array of shape (3,)
        Start position [x, y, z] in mm.
    p_end : array of shape (3,)
        End position [x, y, z] in mm.
    duration : float
        Trajectory duration in seconds.
    n_points : int
        Number of waypoints.
    q_seed : array of shape (4,) or None
        Initial IK seed for first point.
    """

    def __init__(
        self,
        robot: Robot4DOF,
        p_start: NDArray[np.float64],
        p_end: NDArray[np.float64],
        duration: float = 2.0,
        n_points: int = 100,
        q_seed: NDArray[np.float64] | None = None,
    ) -> None:
        self.robot = robot
        self.p_start = np.asarray(p_start, dtype=np.float64)
        self.p_end = np.asarray(p_end, dtype=np.float64)
        self.duration = duration
        self.n_points = n_points

        self.t = np.linspace(0, duration, n_points)
        self._compute(q_seed)

    def _compute(self, q_seed: NDArray[np.float64] | None) -> None:
        # Smooth time parameterization (trapezoidal-ish via cosine blend)
        s = 0.5 * (1 - np.cos(np.pi * self.t / self.duration))

        self.positions = np.zeros((self.n_points, 3))
        self.q = np.zeros((self.n_points, 4))
        self.ik_success = np.ones(self.n_points, dtype=bool)

        q_current = q_seed if q_seed is not None else np.zeros(4)

        for i in range(self.n_points):
            target = self.p_start + s[i] * (self.p_end - self.p_start)
            self.positions[i] = target

            q_sol, success, _ = self.robot.inverse_kinematics(
                target, q0=q_current, tol=0.1
            )
            self.q[i] = q_sol
            self.ik_success[i] = success
            q_current = q_sol  # Warm-start next point


class MultiPointTrajectory:
    """Joint-space trajectory through multiple via-points.

    Concatenates piecewise polynomial segments with smooth transitions.

    Parameters
    ----------
    waypoints : list of arrays of shape (4,)
        Joint angle waypoints.
    durations : list of float
        Duration for each segment (len = len(waypoints) - 1).
    n_points_per_segment : int
        Points per segment.
    method : str
        'cubic' or 'quintic'.
    """

    def __init__(
        self,
        waypoints: list[NDArray[np.float64]],
        durations: list[float],
        n_points_per_segment: int = 50,
        method: str = "quintic",
    ) -> None:
        assert len(durations) == len(waypoints) - 1

        segments = []
        for i in range(len(durations)):
            seg = JointTrajectory(
                waypoints[i], waypoints[i + 1],
                duration=durations[i],
                n_points=n_points_per_segment,
                method=method,
            )
            segments.append(seg)

        # Concatenate (skip duplicate points at boundaries)
        t_offset = 0.0
        t_list, q_list, qd_list = [], [], []
        for i, seg in enumerate(segments):
            start = 1 if i > 0 else 0
            t_list.append(seg.t[start:] + t_offset)
            q_list.append(seg.q[start:])
            qd_list.append(seg.qd[start:])
            t_offset += seg.duration

        self.t = np.concatenate(t_list)
        self.q = np.concatenate(q_list, axis=0)
        self.qd = np.concatenate(qd_list, axis=0)
        self.duration = t_offset
        self.n_points = len(self.t)
