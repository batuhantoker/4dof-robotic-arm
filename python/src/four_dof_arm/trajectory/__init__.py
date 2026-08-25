"""
Trajectory Planning for 4-DOF Manipulator

Supports joint-space polynomial interpolation and Cartesian-space
linear interpolation with configurable time parameterization.
"""

from .cartesian import CartesianTrajectory
from .joint import JointTrajectory
from .multi_point import MultiPointTrajectory
from .polynomials import cubic_polynomial, quintic_polynomial

__all__ = [
    "cubic_polynomial",
    "quintic_polynomial",
    "JointTrajectory",
    "CartesianTrajectory",
    "MultiPointTrajectory",
]
