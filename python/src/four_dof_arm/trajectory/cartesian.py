from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..robot import Robot4DOF
from ..robot.validation import finite_scalar
from .validation import _finite_vector, _point_count, _positive_duration

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
        Initial IK guess, reused until the first accepted solution. The first
        accepted solution is not constrained to be near this seed.
    max_joint_step : float
        Positive finite maximum absolute change in radians for every joint
        relative to the last accepted solution (default 0.5). Bounded periodic
        equivalents are considered first. Larger jumps are marked unsuccessful
        and do not update the IK seed; rejected candidates remain in q with
        ik_success=False. No subdivision or retry is performed. This is a
        per-waypoint continuity bound, not a velocity or collision guarantee.
    """

    def __init__(
        self,
        robot: Robot4DOF,
        p_start: NDArray[np.float64],
        p_end: NDArray[np.float64],
        duration: float = 2.0,
        n_points: int = 100,
        q_seed: NDArray[np.float64] | None = None,
        max_joint_step: float = 0.5,
    ) -> None:
        self.robot = robot
        self.p_start = _finite_vector(p_start, (3,), "p_start")
        self.p_end = _finite_vector(p_end, (3,), "p_end")
        self.duration = _positive_duration(duration)
        self.n_points = _point_count(n_points, "n_points")
        self.max_joint_step = finite_scalar(
            max_joint_step, "max_joint_step", positive=True
        )
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
                    valid_result = bool(
                        np.all(np.abs(q_sol - q_current) <= self.max_joint_step)
                    )

            self.q[i] = q_sol
            self.ik_success[i] = valid_result
            if valid_result:
                q_current = q_sol.copy()
                has_success = True

        self.success = bool(np.all(self.ik_success))
