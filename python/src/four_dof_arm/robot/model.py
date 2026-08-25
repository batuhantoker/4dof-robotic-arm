"""Public 4-DOF robot model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import inverse, kinematics
from .kinematics import RotationMatrices
from .validation import finite_scalar


class Robot4DOF:
    """4-DOF serial manipulator with a Z-X-Z-X joint configuration.

    Link lengths are measured in millimeters.
    """

    def __init__(
        self,
        Lax: float = 50.0,
        Laz: float = 55.0,
        Lb: float = 100.0,
        Lt: float = 90.0,
    ) -> None:
        self.Lax = finite_scalar(Lax, "Lax", positive=True)
        self.Laz = finite_scalar(Laz, "Laz", positive=True)
        self.Lb = finite_scalar(Lb, "Lb", positive=True)
        self.Lt = finite_scalar(Lt, "Lt", positive=True)
        self.joint_limits = np.array(
            [
                [-np.pi, np.pi],
                [-np.pi / 2, np.pi / 2],
                [-np.pi, np.pi],
                [-np.pi / 2, np.pi / 2],
            ]
        )

    @property
    def n_joints(self) -> int:
        """Number of joints in the manipulator."""
        return 4

    def rotation_matrices(self, q: NDArray[np.float64]) -> RotationMatrices:
        """Compute the four individual joint rotation matrices."""
        return kinematics.rotation_matrices(self, q)

    def forward_kinematics(
        self, q: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute the end-effector position and orientation."""
        return kinematics.forward_kinematics(self, q)

    def joint_positions(self, q: NDArray[np.float64]) -> list[NDArray[np.float64]]:
        """Compute all joint and end-effector positions."""
        return kinematics.joint_positions(self, q)

    def jacobian(
        self, q: NDArray[np.float64], eps: float = 1e-6
    ) -> NDArray[np.float64]:
        """Compute the positional Jacobian via finite differences."""
        return kinematics.jacobian(self, q, eps)

    def jacobian_analytical(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the analytical positional Jacobian."""
        return kinematics.jacobian_analytical(self, q)

    def _limit_configuration(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        return inverse.limit_configuration(self, q)

    def inverse_kinematics(
        self,
        target: NDArray[np.float64],
        q0: NDArray[np.float64] | None = None,
        tol: float = 1e-4,
        max_iter: int = 200,
        damping: float = 0.1,
    ) -> tuple[NDArray[np.float64], bool, float]:
        """Solve inverse kinematics with damped least squares."""
        return inverse.inverse_kinematics(
            self,
            target,
            q0=q0,
            tol=tol,
            max_iter=max_iter,
            damping=damping,
        )

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
        """Solve inverse kinematics with multiple random restarts."""
        return inverse.inverse_kinematics_multi(
            self,
            target,
            n_restarts=n_restarts,
            q0=q0,
            rng=rng,
            tol=tol,
            max_iter=max_iter,
            damping=damping,
        )

    def sample_workspace(
        self,
        n_samples: int = 5000,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Sample reachable positions from random joint configurations."""
        return inverse.sample_workspace(self, n_samples=n_samples, rng=rng)

    def __repr__(self) -> str:
        return f"Robot4DOF(Lax={self.Lax}, Laz={self.Laz}, Lb={self.Lb}, Lt={self.Lt})"
