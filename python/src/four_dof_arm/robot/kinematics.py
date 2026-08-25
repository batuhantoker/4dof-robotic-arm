"""Forward-kinematics and Jacobian implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .validation import as_finite_vector, finite_scalar

if TYPE_CHECKING:
    from .model import Robot4DOF


RotationMatrices: TypeAlias = tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]


def _rot_z(theta: float) -> NDArray[np.float64]:
    """Return a rotation matrix about the Z axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _rot_x(theta: float) -> NDArray[np.float64]:
    """Return a rotation matrix about the X axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ]
    )


def rotation_matrices(robot: Robot4DOF, q: NDArray[np.float64]) -> RotationMatrices:
    q = as_finite_vector(q, robot.n_joints, "q")
    return _rot_z(q[0]), _rot_x(q[1]), _rot_z(q[2]), _rot_x(q[3])


def forward_kinematics(
    robot: Robot4DOF, q: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    q = as_finite_vector(q, robot.n_joints, "q")
    q1, q2, q3, q4 = q
    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)
    c3, s3 = np.cos(q3), np.sin(q3)
    c4, s4 = np.cos(q4), np.sin(q4)

    x = (
        robot.Lax * c1
        + robot.Lb * c1
        + robot.Lt * (s3 * s4 * c1 + s1 * (s2 * c4 + s4 * c2 * c3))
    )
    y = (
        robot.Lax * s1
        + robot.Lb * s1
        + robot.Lt * (s1 * s3 * s4 - c1 * (s2 * c4 + s4 * c2 * c3))
    )
    z = robot.Laz + robot.Lt * (c2 * c4 - s2 * s4 * c3)
    position = np.array([x, y, z])

    R_NA, R_AB, R_BC, R_CD = robot.rotation_matrices(q)
    R_ND = R_NA @ R_AB @ R_BC @ R_CD
    return position, R_ND


def joint_positions(
    robot: Robot4DOF, q: NDArray[np.float64]
) -> list[NDArray[np.float64]]:
    q = as_finite_vector(q, robot.n_joints, "q")
    R_NA = _rot_z(q[0])
    R_NB = R_NA @ _rot_x(q[1])
    R_ND = R_NB @ _rot_z(q[2]) @ _rot_x(q[3])

    a1 = R_NA[:, 0]
    a3 = R_NA[:, 2]
    b1 = R_NB[:, 0]
    d3 = R_ND[:, 2]

    origin = np.zeros(3)
    shoulder_top = origin + robot.Laz * a3
    shoulder = shoulder_top + robot.Lax * a1
    elbow = shoulder + robot.Lb * b1
    end_effector = elbow + robot.Lt * d3
    return [origin, shoulder_top, shoulder, elbow, end_effector]


def jacobian(
    robot: Robot4DOF, q: NDArray[np.float64], eps: float = 1e-6
) -> NDArray[np.float64]:
    q = as_finite_vector(q, robot.n_joints, "q")
    eps = finite_scalar(eps, "eps", positive=True)
    result = np.zeros((3, 4))
    position, _ = robot.forward_kinematics(q)
    for i in range(4):
        perturbed_q = q.copy()
        perturbed_q[i] += eps
        perturbed_position, _ = robot.forward_kinematics(perturbed_q)
        result[:, i] = (perturbed_position - position) / eps
    return result


def jacobian_analytical(
    robot: Robot4DOF, q: NDArray[np.float64]
) -> NDArray[np.float64]:
    q = as_finite_vector(q, robot.n_joints, "q")
    q1, q2, q3, q4 = q
    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)
    c3, s3 = np.cos(q3), np.sin(q3)
    c4, s4 = np.cos(q4), np.sin(q4)
    Lax, Lb, Lt = robot.Lax, robot.Lb, robot.Lt

    J11 = -(Lax * s1 + Lb * s1 + Lt * (s1 * s3 * s4 - c1 * (s2 * c4 + s4 * c2 * c3)))
    J12 = Lt * s1 * (c2 * c4 - s2 * s4 * c3)
    J13 = Lt * s4 * (c1 * c3 - s1 * s3 * c2)
    J14 = Lt * (s3 * c1 * c4 - s1 * (s2 * s4 - c2 * c3 * c4))

    J21 = Lax * c1 + Lb * c1 + Lt * (s3 * s4 * c1 + s1 * (s2 * c4 + s4 * c2 * c3))
    J22 = -Lt * c1 * (c2 * c4 - s2 * s4 * c3)
    J23 = Lt * s4 * (s1 * c3 + s3 * c1 * c2)
    J24 = Lt * (s1 * s3 * c4 + c1 * (s2 * s4 - c2 * c3 * c4))

    J31 = 0.0
    J32 = -Lt * (s2 * c4 + s4 * c2 * c3)
    J33 = Lt * s2 * s3 * s4
    J34 = -Lt * (s4 * c2 + s2 * c3 * c4)

    return np.array(
        [
            [J11, J12, J13, J14],
            [J21, J22, J23, J24],
            [J31, J32, J33, J34],
        ]
    )
