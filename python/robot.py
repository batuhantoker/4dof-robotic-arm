"""
4-DOF Serial Manipulator — Kinematics Engine

Joint configuration (from AUTOLEV Kane's method formulation):
    q1: Z-axis rotation (base yaw)
    q2: X-axis rotation (shoulder pitch)
    q3: Z-axis rotation (elbow yaw)
    q4: X-axis rotation (wrist pitch)

Rotation chain: N→A(Z,q1) → B(X,q2) → C(Z,q3) → D(X,q4)

Position vectors:
    P_O_P = Laz * A3   (base height along A's z-axis)
    P_P_R = Lax * A1   (shoulder offset along A's x-axis)
    P_R_S = Lb  * B1   (upper arm along B's x-axis)
    P_S_G = Lt  * D3   (forearm to gripper along D's z-axis)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


def _rot_z(theta: float) -> NDArray[np.float64]:
    """Rotation matrix about Z-axis (Simprot axis 3)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def _rot_x(theta: float) -> NDArray[np.float64]:
    """Rotation matrix about X-axis (Simprot axis 1)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0,  c, -s],
        [0.0,  s,  c],
    ])


class Robot4DOF:
    """4-DOF serial manipulator with Z-X-Z-X joint configuration.

    Parameters
    ----------
    Lax : float
        Shoulder offset (mm). Distance from base column top to shoulder joint
        along the A-frame x-axis.
    Laz : float
        Base height (mm). Vertical distance from origin to shoulder level
        along the A-frame z-axis.
    Lb : float
        Upper arm length (mm). Distance from shoulder to elbow along B-frame x-axis.
    Lt : float
        Forearm length (mm). Distance from wrist to gripper along D-frame z-axis.
    """

    def __init__(
        self,
        Lax: float = 50.0,
        Laz: float = 55.0,
        Lb: float = 100.0,
        Lt: float = 90.0,
    ) -> None:
        self.Lax = Lax
        self.Laz = Laz
        self.Lb = Lb
        self.Lt = Lt

        # Joint limits (radians) — conservative defaults
        self.joint_limits = np.array([
            [-np.pi, np.pi],       # q1: full base rotation
            [-np.pi / 2, np.pi / 2],  # q2: shoulder ±90°
            [-np.pi, np.pi],       # q3: full elbow rotation
            [-np.pi / 2, np.pi / 2],  # q4: wrist ±90°
        ])

    @property
    def n_joints(self) -> int:
        return 4

    def rotation_matrices(
        self, q: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Compute individual rotation matrices N_A, A_B, B_C, C_D.

        Returns
        -------
        R_NA, R_AB, R_BC, R_CD : tuple of (3,3) arrays
        """
        return _rot_z(q[0]), _rot_x(q[1]), _rot_z(q[2]), _rot_x(q[3])

    def forward_kinematics(
        self, q: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute end-effector position and orientation.

        Parameters
        ----------
        q : array of shape (4,)
            Joint angles [q1, q2, q3, q4] in radians.

        Returns
        -------
        position : array of shape (3,)
            End-effector [x, y, z] in mm.
        R_ND : array of shape (3, 3)
            Rotation matrix from base (N) to end-effector (D) frame.
        """
        q1, q2, q3, q4 = q
        c1, s1 = np.cos(q1), np.sin(q1)
        c2, s2 = np.cos(q2), np.sin(q2)
        c3, s3 = np.cos(q3), np.sin(q3)
        c4, s4 = np.cos(q4), np.sin(q4)

        # From AUTOLEV loop equation (lines 141-145 of 4DOF_kane.all)
        x = (self.Lax * c1
             + self.Lb * c1
             + self.Lt * (s3 * s4 * c1 + s1 * (s2 * c4 + s4 * c2 * c3)))
        y = (self.Lax * s1
             + self.Lb * s1
             + self.Lt * (s1 * s3 * s4 - c1 * (s2 * c4 + s4 * c2 * c3)))
        z = self.Laz + self.Lt * (c2 * c4 - s2 * s4 * c3)

        position = np.array([x, y, z])

        # Full rotation: R_ND = R_NA @ R_AB @ R_BC @ R_CD
        R_NA, R_AB, R_BC, R_CD = self.rotation_matrices(q)
        R_ND = R_NA @ R_AB @ R_BC @ R_CD

        return position, R_ND

    def joint_positions(
        self, q: NDArray[np.float64]
    ) -> list[NDArray[np.float64]]:
        """Compute positions of all joints and end-effector for visualization.

        Returns
        -------
        points : list of 5 arrays of shape (3,)
            [origin, shoulder_top, shoulder_offset, elbow, end_effector]
        """
        R_NA = _rot_z(q[0])
        R_NB = R_NA @ _rot_x(q[1])
        R_ND = R_NB @ _rot_z(q[2]) @ _rot_x(q[3])

        # A-frame unit vectors in N
        a1 = R_NA[:, 0]
        a3 = R_NA[:, 2]
        # B-frame unit vectors in N
        b1 = R_NB[:, 0]
        # D-frame unit vectors in N
        d3 = R_ND[:, 2]

        origin = np.zeros(3)
        p_shoulder_top = origin + self.Laz * a3           # P: top of base column
        p_shoulder = p_shoulder_top + self.Lax * a1       # R: shoulder offset
        p_elbow = p_shoulder + self.Lb * b1               # S: elbow
        p_ee = p_elbow + self.Lt * d3                     # G: gripper

        return [origin, p_shoulder_top, p_shoulder, p_elbow, p_ee]

    def jacobian(
        self, q: NDArray[np.float64], eps: float = 1e-6
    ) -> NDArray[np.float64]:
        """Compute the positional Jacobian via finite differences.

        Parameters
        ----------
        q : array of shape (4,)
            Joint angles in radians.
        eps : float
            Perturbation for numerical differentiation.

        Returns
        -------
        J : array of shape (3, 4)
            Positional Jacobian dp/dq.
        """
        J = np.zeros((3, 4))
        p0, _ = self.forward_kinematics(q)
        for i in range(4):
            q_pert = q.copy()
            q_pert[i] += eps
            p_pert, _ = self.forward_kinematics(q_pert)
            J[:, i] = (p_pert - p0) / eps
        return J

    def jacobian_analytical(
        self, q: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute the analytical positional Jacobian from AUTOLEV derivatives.

        Matches Zeroconfig rows 1-3 from 4DOF_kane.all (lines 154-167),
        negated because the AUTOLEV zero equations are x - f(q) = 0,
        so df/dq = -d(zero)/dq.

        Parameters
        ----------
        q : array of shape (4,)

        Returns
        -------
        J : array of shape (3, 4)
        """
        q1, q2, q3, q4 = q
        c1, s1 = np.cos(q1), np.sin(q1)
        c2, s2 = np.cos(q2), np.sin(q2)
        c3, s3 = np.cos(q3), np.sin(q3)
        c4, s4 = np.cos(q4), np.sin(q4)

        Lax, Lb, Laz, Lt = self.Lax, self.Lb, self.Laz, self.Lt

        # dx/dq (negate AUTOLEV Zeroconfig row 1)
        J11 = -(Lax * s1 + Lb * s1 + Lt * (s1 * s3 * s4 - c1 * (s2 * c4 + s4 * c2 * c3)))
        J12 = Lt * s1 * (c2 * c4 - s2 * s4 * c3)
        J13 = Lt * s4 * (c1 * c3 - s1 * s3 * c2)
        J14 = Lt * (s3 * c1 * c4 - s1 * (s2 * s4 - c2 * c3 * c4))

        # dy/dq (negate AUTOLEV Zeroconfig row 2)
        J21 = Lax * c1 + Lb * c1 + Lt * (s3 * s4 * c1 + s1 * (s2 * c4 + s4 * c2 * c3))
        J22 = -Lt * c1 * (c2 * c4 - s2 * s4 * c3)
        J23 = Lt * s4 * (s1 * c3 + s3 * c1 * c2)
        J24 = Lt * (s1 * s3 * c4 + c1 * (s2 * s4 - c2 * c3 * c4))

        # dz/dq (negate AUTOLEV Zeroconfig row 3)
        J31 = 0.0
        J32 = -Lt * (s2 * c4 + s4 * c2 * c3)
        J33 = Lt * s2 * s3 * s4
        J34 = -Lt * (s4 * c2 + s2 * c3 * c4)

        return np.array([
            [J11, J12, J13, J14],
            [J21, J22, J23, J24],
            [J31, J32, J33, J34],
        ])

    def inverse_kinematics(
        self,
        target: NDArray[np.float64],
        q0: NDArray[np.float64] | None = None,
        tol: float = 1e-4,
        max_iter: int = 200,
        damping: float = 0.1,
    ) -> tuple[NDArray[np.float64], bool, float]:
        """Solve inverse kinematics using damped least-squares (Levenberg-Marquardt).

        Parameters
        ----------
        target : array of shape (3,)
            Desired end-effector position [x, y, z] in mm.
        q0 : array of shape (4,) or None
            Initial joint angle guess. If None, uses zeros.
        tol : float
            Position error tolerance in mm.
        max_iter : int
            Maximum iterations.
        damping : float
            Damping factor for singularity robustness.

        Returns
        -------
        q : array of shape (4,)
            Joint angles solution in radians.
        success : bool
            True if converged within tolerance.
        error : float
            Final position error norm in mm.
        """
        q = q0.copy() if q0 is not None else np.zeros(4)
        target = np.asarray(target, dtype=np.float64)

        for _ in range(max_iter):
            pos, _ = self.forward_kinematics(q)
            error_vec = target - pos
            error_norm = np.linalg.norm(error_vec)

            if error_norm < tol:
                return q, True, error_norm

            J = self.jacobian_analytical(q)
            # Damped least-squares: dq = J^T (J J^T + λ²I)^{-1} e
            JJT = J @ J.T + (damping ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, error_vec)

            q = q + dq
            # Wrap to [-π, π]
            q = np.arctan2(np.sin(q), np.cos(q))

        pos, _ = self.forward_kinematics(q)
        error_norm = float(np.linalg.norm(target - pos))
        return q, error_norm < tol, error_norm

    def inverse_kinematics_multi(
        self,
        target: NDArray[np.float64],
        n_restarts: int = 8,
        **kwargs,
    ) -> tuple[NDArray[np.float64], bool, float]:
        """IK with multiple random restarts to escape local minima.

        Parameters
        ----------
        target : array of shape (3,)
        n_restarts : int
            Number of random initial guesses to try.
        **kwargs
            Passed to inverse_kinematics.

        Returns
        -------
        Best solution from all restarts.
        """
        best_q = np.zeros(4)
        best_error = np.inf
        best_success = False

        for i in range(n_restarts):
            if i == 0:
                q0 = np.zeros(4)
            else:
                q0 = np.random.uniform(
                    self.joint_limits[:, 0],
                    self.joint_limits[:, 1],
                )

            q, success, error = self.inverse_kinematics(target, q0=q0, **kwargs)

            if error < best_error:
                best_q, best_success, best_error = q, success, error

            if success:
                break

        return best_q, best_success, best_error

    def workspace_boundary(
        self, n_samples: int = 5000
    ) -> NDArray[np.float64]:
        """Sample the reachable workspace by random joint configurations.

        Parameters
        ----------
        n_samples : int
            Number of random joint configurations.

        Returns
        -------
        points : array of shape (n_samples, 3)
            Reachable end-effector positions.
        """
        points = np.zeros((n_samples, 3))
        for i in range(n_samples):
            q = np.random.uniform(
                self.joint_limits[:, 0],
                self.joint_limits[:, 1],
            )
            points[i], _ = self.forward_kinematics(q)
        return points

    def __repr__(self) -> str:
        return (
            f"Robot4DOF(Lax={self.Lax}, Laz={self.Laz}, "
            f"Lb={self.Lb}, Lt={self.Lt})"
        )
