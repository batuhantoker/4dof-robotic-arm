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


def _as_finite_vector(value: object, size: int, name: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite vector of shape ({size},)") from exc

    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of shape ({size},)")
    return array


def _finite_scalar(value: object, name: str, *, positive: bool = False) -> float:
    qualifier = "positive finite" if positive else "finite"
    if isinstance(value, (bool, np.bool_)) or (
        isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be a {qualifier} scalar")

    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a {qualifier} scalar") from exc

    if array.shape != ():
        raise ValueError(f"{name} must be a {qualifier} scalar")

    result = float(array)
    if not np.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{name} must be a {qualifier} scalar")
    return result


def _positive_integer(value: object, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _random_generator(
    rng: np.random.Generator | None,
) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    return rng


def _rot_z(theta: float) -> NDArray[np.float64]:
    """Rotation matrix about Z-axis (Simprot axis 3)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _rot_x(theta: float) -> NDArray[np.float64]:
    """Rotation matrix about X-axis (Simprot axis 1)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ]
    )


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
        self.Lax = _finite_scalar(Lax, "Lax", positive=True)
        self.Laz = _finite_scalar(Laz, "Laz", positive=True)
        self.Lb = _finite_scalar(Lb, "Lb", positive=True)
        self.Lt = _finite_scalar(Lt, "Lt", positive=True)

        # Joint limits (radians) — conservative defaults
        self.joint_limits = np.array(
            [
                [-np.pi, np.pi],  # q1: full base rotation
                [-np.pi / 2, np.pi / 2],  # q2: shoulder ±90°
                [-np.pi, np.pi],  # q3: full elbow rotation
                [-np.pi / 2, np.pi / 2],  # q4: wrist ±90°
            ]
        )

    @property
    def n_joints(self) -> int:
        return 4

    def rotation_matrices(
        self, q: NDArray[np.float64]
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Compute individual rotation matrices N_A, A_B, B_C, C_D.

        Returns
        -------
        R_NA, R_AB, R_BC, R_CD : tuple of (3,3) arrays
        """
        q = _as_finite_vector(q, self.n_joints, "q")
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
        q = _as_finite_vector(q, self.n_joints, "q")
        q1, q2, q3, q4 = q
        c1, s1 = np.cos(q1), np.sin(q1)
        c2, s2 = np.cos(q2), np.sin(q2)
        c3, s3 = np.cos(q3), np.sin(q3)
        c4, s4 = np.cos(q4), np.sin(q4)

        # From AUTOLEV loop equation (lines 141-145 of 4DOF_kane.all)
        x = (
            self.Lax * c1
            + self.Lb * c1
            + self.Lt * (s3 * s4 * c1 + s1 * (s2 * c4 + s4 * c2 * c3))
        )
        y = (
            self.Lax * s1
            + self.Lb * s1
            + self.Lt * (s1 * s3 * s4 - c1 * (s2 * c4 + s4 * c2 * c3))
        )
        z = self.Laz + self.Lt * (c2 * c4 - s2 * s4 * c3)

        position = np.array([x, y, z])

        # Full rotation: R_ND = R_NA @ R_AB @ R_BC @ R_CD
        R_NA, R_AB, R_BC, R_CD = self.rotation_matrices(q)
        R_ND = R_NA @ R_AB @ R_BC @ R_CD

        return position, R_ND

    def joint_positions(self, q: NDArray[np.float64]) -> list[NDArray[np.float64]]:
        """Compute positions of all joints and end-effector for visualization.

        Returns
        -------
        points : list of 5 arrays of shape (3,)
            [origin, shoulder_top, shoulder_offset, elbow, end_effector]
        """
        q = _as_finite_vector(q, self.n_joints, "q")
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
        p_shoulder_top = origin + self.Laz * a3  # P: top of base column
        p_shoulder = p_shoulder_top + self.Lax * a1  # R: shoulder offset
        p_elbow = p_shoulder + self.Lb * b1  # S: elbow
        p_ee = p_elbow + self.Lt * d3  # G: gripper

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
        q = _as_finite_vector(q, self.n_joints, "q")
        eps = _finite_scalar(eps, "eps", positive=True)
        J = np.zeros((3, 4))
        p0, _ = self.forward_kinematics(q)
        for i in range(4):
            q_pert = q.copy()
            q_pert[i] += eps
            p_pert, _ = self.forward_kinematics(q_pert)
            J[:, i] = (p_pert - p0) / eps
        return J

    def jacobian_analytical(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
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
        q = _as_finite_vector(q, self.n_joints, "q")
        q1, q2, q3, q4 = q
        c1, s1 = np.cos(q1), np.sin(q1)
        c2, s2 = np.cos(q2), np.sin(q2)
        c3, s3 = np.cos(q3), np.sin(q3)
        c4, s4 = np.cos(q4), np.sin(q4)

        Lax, Lb, Lt = self.Lax, self.Lb, self.Lt

        # dx/dq (negate AUTOLEV Zeroconfig row 1)
        J11 = -(
            Lax * s1 + Lb * s1 + Lt * (s1 * s3 * s4 - c1 * (s2 * c4 + s4 * c2 * c3))
        )
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

        return np.array(
            [
                [J11, J12, J13, J14],
                [J21, J22, J23, J24],
                [J31, J32, J33, J34],
            ]
        )

    def _limit_configuration(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

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
        target = _as_finite_vector(target, 3, "target")
        q: NDArray[np.float64]
        if q0 is None:
            q = np.zeros(self.n_joints, dtype=np.float64)
        else:
            q = _as_finite_vector(q0, self.n_joints, "q0")
        q = self._limit_configuration(q)
        tol = _finite_scalar(tol, "tol", positive=True)
        max_iter = _positive_integer(max_iter, "max_iter")
        damping = _finite_scalar(damping, "damping", positive=True)
        used_stagnation_escape = False

        for _ in range(max_iter):
            pos, _ = self.forward_kinematics(q)
            error_vec = target - pos
            error_norm = float(np.linalg.norm(error_vec))

            if error_norm < tol and np.all(
                (q >= self.joint_limits[:, 0]) & (q <= self.joint_limits[:, 1])
            ):
                return q, True, error_norm

            J = self.jacobian_analytical(q)
            # Damped least-squares: dq = J^T (J J^T + λ²I)^{-1} e
            JJT = J @ J.T + (damping**2) * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, error_vec)
            # Large unconstrained steps can pin the bounded solver at hard stops.
            dq = np.clip(dq, -0.5, 0.5)

            next_q = self._limit_configuration(q + dq)
            if np.linalg.norm(next_q - q, ord=np.inf) <= 1e-12:
                if used_stagnation_escape:
                    break
                # q=0 cannot move toward exact x-z plane errors at first order.
                q = self._limit_configuration(q + np.array([0.0, 0.0, 0.1, 0.1]))
                used_stagnation_escape = True
            else:
                q = next_q

        pos, _ = self.forward_kinematics(q)
        error_norm = float(np.linalg.norm(target - pos))
        within_limits = np.all(
            (q >= self.joint_limits[:, 0]) & (q <= self.joint_limits[:, 1])
        )
        return q, bool(error_norm < tol and within_limits), error_norm

    def inverse_kinematics_multi(
        self,
        target: NDArray[np.float64],
        n_restarts: int = 8,
        q0: NDArray[np.float64] | None = None,
        rng: np.random.Generator | None = None,
        tol: float = 1e-4,
        max_iter: int = 200,
        damping: float = 0.1,
    ) -> tuple[NDArray[np.float64], bool, float]:
        """IK with multiple random restarts to escape local minima.

        Parameters
        ----------
        target : array of shape (3,)
        n_restarts : int
            Number of random initial guesses to try.
        q0 : array of shape (4,) or None
            Initial guess for the first attempt.
        rng : numpy.random.Generator or None
            Random generator used for subsequent attempts.
        tol, max_iter, damping
            Solver options passed to inverse_kinematics.

        Returns
        -------
        first successful solution is returned; if all attempts fail, the lowest-error failure is returned.
        """
        n_restarts = _positive_integer(n_restarts, "n_restarts")
        rng = _random_generator(rng)
        best_q: NDArray[np.float64] = np.zeros(4, dtype=np.float64)
        best_error = np.inf
        best_success = False

        for i in range(n_restarts):
            if i == 0:
                initial_q = q0
            else:
                initial_q = rng.uniform(
                    self.joint_limits[:, 0],
                    self.joint_limits[:, 1],
                )

            q, success, error = self.inverse_kinematics(
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
        self,
        n_samples: int = 5000,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Sample the reachable workspace by random joint configurations.

        Parameters
        ----------
        n_samples : int
            Number of random joint configurations.
        rng : numpy.random.Generator or None
            Random generator used for sampling.

        Returns
        -------
        points : array of shape (n_samples, 3)
            Reachable end-effector positions.
        """
        n_samples = _positive_integer(n_samples, "n_samples")
        rng = _random_generator(rng)
        points = np.zeros((n_samples, 3))
        for i in range(n_samples):
            q = rng.uniform(
                self.joint_limits[:, 0],
                self.joint_limits[:, 1],
            )
            points[i], _ = self.forward_kinematics(q)
        return points

    def __repr__(self) -> str:
        return f"Robot4DOF(Lax={self.Lax}, Laz={self.Laz}, Lb={self.Lb}, Lt={self.Lt})"
