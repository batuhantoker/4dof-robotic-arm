"""Kinematics and trajectory planning for the 4-DOF robotic arm."""

from .robot import Robot4DOF
from .trajectory import (
    CartesianTrajectory,
    JointTrajectory,
    MultiPointTrajectory,
    cubic_polynomial,
    quintic_polynomial,
)

__all__ = [
    "CartesianTrajectory",
    "JointTrajectory",
    "MultiPointTrajectory",
    "Robot4DOF",
    "cubic_polynomial",
    "quintic_polynomial",
]
