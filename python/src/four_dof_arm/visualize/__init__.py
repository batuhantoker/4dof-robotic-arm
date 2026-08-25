"""Matplotlib visualizations for the 4-DOF manipulator."""

from .animation import animate_trajectory
from .arm import plot_arm
from .plots import plot_joint_angles, plot_workspace

__all__ = [
    "plot_arm",
    "animate_trajectory",
    "plot_joint_angles",
    "plot_workspace",
]
