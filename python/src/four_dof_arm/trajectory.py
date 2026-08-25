"""
Trajectory Planning for 4-DOF Manipulator

Supports joint-space polynomial interpolation and Cartesian-space
linear interpolation with configurable time parameterization.
"""

from __future__ import annotations

import operator
from typing import SupportsIndex

import numpy as np
from numpy.typing import NDArray

from .robot import Robot4DOF

_PERIODIC_JOINTS = (0, 2)


def _continuous_bounded_configuration(
    previous: NDArray[np.float64],
    candidate: NDArray[np.float64],
    joint_limits: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    continuous = candidate.copy()
    for joint in _PERIODIC_JOINTS:
        delta = continuous[joint] - previous[joint]
        if abs(delta) <= np.pi:
            continue

        equivalent = continuous[joint] - np.copysign(2.0 * np.pi, delta)
        if not joint_limits[joint, 0] <= equivalent <= joint_limits[joint, 1]:
            return None
        continuous[joint] = equivalent

    return continuous


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
        self.p_start = _finite_vector(p_start, (3,), "p_start")
        self.p_end = _finite_vector(p_end, (3,), "p_end")
        self.duration = _positive_duration(duration)
        self.n_points = _point_count(n_points, "n_points")
        seed = (
            np.zeros(4, dtype=np.float64)
            if q_seed is None
            else _finite_vector(q_seed, (4,), "q_seed")
        )

        self.t = np.linspace(0, self.duration, self.n_points)
        self._compute(seed)

    def _compute(self, q_seed: NDArray[np.float64]) -> None:
        # Smooth time parameterization (trapezoidal-ish via cosine blend)
        s = 0.5 * (1 - np.cos(np.pi * self.t / self.duration))

        self.positions = np.zeros((self.n_points, 3))
        self.q = np.full((self.n_points, 4), np.nan)
        self.ik_success = np.zeros(self.n_points, dtype=bool)
        self.ik_error = np.full(self.n_points, np.inf)

        q_current = q_seed.copy()
        has_success = False

        for i in range(self.n_points):
            target = self.p_start + s[i] * (self.p_end - self.p_start)
            self.positions[i] = target

            q_sol, success, error = self.robot.inverse_kinematics(
                target, q0=q_current.copy(), tol=0.1
            )
            try:
                q_sol = np.asarray(q_sol, dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "inverse_kinematics must return q with shape (4,)"
                ) from exc
            if q_sol.shape != (4,):
                raise ValueError("inverse_kinematics must return q with shape (4,)")

            try:
                error_value = np.asarray(error, dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "inverse_kinematics must return a scalar error"
                ) from exc
            if error_value.ndim != 0:
                raise ValueError("inverse_kinematics must return a scalar error")

            if not isinstance(success, (bool, np.bool_)):
                raise ValueError(
                    "inverse_kinematics must return a boolean success flag"
                )

            self.ik_error[i] = float(error_value)
            valid_result = success and bool(
                np.all(np.isfinite(q_sol))
                and np.isfinite(error_value)
                and np.all(q_sol >= self.robot.joint_limits[:, 0])
                and np.all(q_sol <= self.robot.joint_limits[:, 1])
            )

            if valid_result and has_success:
                continuous = _continuous_bounded_configuration(
                    q_current, q_sol, self.robot.joint_limits
                )
                if continuous is None:
                    valid_result = False
                else:
                    q_sol = continuous

            self.q[i] = q_sol
            self.ik_success[i] = valid_result
            if valid_result:
                q_current = q_sol.copy()
                has_success = True

        self.success = bool(np.all(self.ik_success))


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
        try:
            n_waypoints = len(waypoints)
        except TypeError as exc:
            raise ValueError(
                "waypoints must contain at least two arrays with shape (4,)"
            ) from exc
        if n_waypoints < 2:
            raise ValueError(
                "waypoints must contain at least two arrays with shape (4,)"
            )

        waypoint_values = [
            _finite_vector(waypoint, (4,), f"waypoints[{i}]")
            for i, waypoint in enumerate(waypoints)
        ]

        try:
            duration_values = np.asarray(durations, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"durations must contain exactly {n_waypoints - 1} values"
            ) from exc
        if duration_values.shape != (n_waypoints - 1,):
            raise ValueError(f"durations must contain exactly {n_waypoints - 1} values")
        if not np.all(np.isfinite(duration_values)) or np.any(duration_values <= 0):
            raise ValueError("durations must contain only finite positive values")
        if not np.isfinite(np.sum(duration_values)):
            raise ValueError("total duration must be finite")

        n_points_per_segment = _point_count(
            n_points_per_segment, "n_points_per_segment"
        )
        method = _method_name(method)

        segments = []
        for i, segment_duration in enumerate(duration_values):
            seg = JointTrajectory(
                waypoint_values[i],
                waypoint_values[i + 1],
                duration=float(segment_duration),
                n_points=n_points_per_segment,
                method=method,
            )
            segments.append(seg)

        # Concatenate (skip duplicate points at boundaries)
        t_offset = 0.0
        t_list, q_list, qd_list, qdd_list = [], [], [], []
        for i, seg in enumerate(segments):
            start = 1 if i > 0 else 0
            t_list.append(seg.t[start:] + t_offset)
            q_list.append(seg.q[start:])
            qd_list.append(seg.qd[start:])
            if seg.qdd is not None:
                qdd_list.append(seg.qdd[start:])
            t_offset += seg.duration

        timestamps = np.concatenate(t_list)
        if np.any(np.diff(timestamps) <= 0.0):
            raise ValueError(
                "segment durations must produce strictly increasing timestamps"
            )

        self.t = timestamps
        self.q = np.concatenate(q_list, axis=0)
        self.qd = np.concatenate(qd_list, axis=0)
        self.qdd = np.concatenate(qdd_list, axis=0) if qdd_list else None
        self.duration = t_offset
        self.n_points = len(self.t)
