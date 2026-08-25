"""Inverse-kinematics and workspace-sampling implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .validation import (
    as_finite_vector,
    finite_scalar,
    positive_integer,
    random_generator,
)

if TYPE_CHECKING:
    from .model import Robot4DOF


def limit_configuration(
    robot: Robot4DOF, q: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.clip(q, robot.joint_limits[:, 0], robot.joint_limits[:, 1])


def inverse_kinematics(
    robot: Robot4DOF,
    target: NDArray[np.float64],
    q0: NDArray[np.float64] | None = None,
    tol: float = 1e-4,
    max_iter: int = 200,
    damping: float = 0.1,
) -> tuple[NDArray[np.float64], bool, float]:
    target = as_finite_vector(target, 3, "target")
    q: NDArray[np.float64]
    if q0 is None:
        q = np.zeros(robot.n_joints, dtype=np.float64)
    else:
        q = as_finite_vector(q0, robot.n_joints, "q0")
    q = robot._limit_configuration(q)
    tol = finite_scalar(tol, "tol", positive=True)
    max_iter = positive_integer(max_iter, "max_iter")
    damping = finite_scalar(damping, "damping", positive=True)
    used_stagnation_escape = False

    for _ in range(max_iter):
        position, _ = robot.forward_kinematics(q)
        error_vector = target - position
        error_norm = float(np.linalg.norm(error_vector))

        if error_norm < tol and np.all(
            (q >= robot.joint_limits[:, 0]) & (q <= robot.joint_limits[:, 1])
        ):
            return q, True, error_norm

        jacobian = robot.jacobian_analytical(q)
        damped_product = jacobian @ jacobian.T + (damping**2) * np.eye(3)
        step = jacobian.T @ np.linalg.solve(damped_product, error_vector)
        step = np.clip(step, -0.5, 0.5)

        next_q = robot._limit_configuration(q + step)
        if np.linalg.norm(next_q - q, ord=np.inf) <= 1e-12:
            if used_stagnation_escape:
                break
            q = robot._limit_configuration(q + np.array([0.0, 0.0, 0.1, 0.1]))
            used_stagnation_escape = True
        else:
            q = next_q

    position, _ = robot.forward_kinematics(q)
    error_norm = float(np.linalg.norm(target - position))
    within_limits = np.all(
        (q >= robot.joint_limits[:, 0]) & (q <= robot.joint_limits[:, 1])
    )
    return q, bool(error_norm < tol and within_limits), error_norm


def inverse_kinematics_multi(
    robot: Robot4DOF,
    target: NDArray[np.float64],
    n_restarts: int = 8,
    q0: NDArray[np.float64] | None = None,
    rng: np.random.Generator | None = None,
    tol: float = 1e-4,
    max_iter: int = 200,
    damping: float = 0.1,
) -> tuple[NDArray[np.float64], bool, float]:
    n_restarts = positive_integer(n_restarts, "n_restarts")
    rng = random_generator(rng)
    best_q: NDArray[np.float64] = np.zeros(4, dtype=np.float64)
    best_error = np.inf
    best_success = False

    for i in range(n_restarts):
        if i == 0:
            initial_q = q0
        else:
            initial_q = rng.uniform(
                robot.joint_limits[:, 0],
                robot.joint_limits[:, 1],
            )

        q, success, error = robot.inverse_kinematics(
            target,
            q0=initial_q,
            tol=tol,
            max_iter=max_iter,
            damping=damping,
        )
        if success:
            return q, True, error
        if error < best_error:
            best_q, best_success, best_error = q, False, error

    return best_q, best_success, best_error


def sample_workspace(
    robot: Robot4DOF,
    n_samples: int = 5000,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    n_samples = positive_integer(n_samples, "n_samples")
    rng = random_generator(rng)
    points = np.zeros((n_samples, 3))
    for i in range(n_samples):
        q = rng.uniform(
            robot.joint_limits[:, 0],
            robot.joint_limits[:, 1],
        )
        points[i], _ = robot.forward_kinematics(q)
    return points
