"""MuJoCo simulation API for the 4-DOF arm."""

from .cli import demo_trajectory, main
from .core import MODEL_PATH, ArmSim
from .rendering import render_video
from .validation import validate_fk

__all__ = [
    "ArmSim",
    "MODEL_PATH",
    "validate_fk",
    "render_video",
    "demo_trajectory",
    "main",
]
