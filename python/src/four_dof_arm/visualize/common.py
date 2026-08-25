"""Shared visualization constants and helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..robot import Robot4DOF


LINK_COLOR = "#2c3e50"
JOINT_COLOR = "#e74c3c"
EE_COLOR = "#f39c12"
TRAIL_COLOR = "#95a5a6"
BG_COLOR = "#fafafa"


def finite_joint_matrix(
    value: object,
    name: str,
    *,
    non_empty: bool = False,
) -> NDArray[np.float64]:
    description = (
        "a non-empty finite array with shape (N, 4)"
        if non_empty
        else "a finite array with shape (N, 4)"
    )
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be {description}") from exc

    if array.ndim != 2 or array.shape[1] != 4 or (non_empty and array.shape[0] == 0):
        raise ValueError(f"{name} must be {description}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def finite_time_vector(value: object) -> NDArray[np.float64]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("t must be a finite one-dimensional array") from exc

    if array.ndim != 1:
        raise ValueError("t must be a finite one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError("t must contain only finite values")
    return array


def positive_integer(value: object, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def format_3d_axes(ax: Axes3D, robot: Robot4DOF) -> None:
    """Apply consistent formatting to 3D axes."""
    horizontal_reach = abs(robot.Lax) + abs(robot.Lb) + abs(robot.Lt)
    vertical_offset = abs(robot.Lt)
    z_min = min(0.0, robot.Laz - vertical_offset)
    z_max = max(0.0, robot.Laz + vertical_offset)
    xy_padding = max(horizontal_reach * 0.05, 1.0)
    z_padding = max((z_max - z_min) * 0.05, 1.0)
    xy_limit = horizontal_reach + xy_padding

    ax.set_xlim(-xy_limit, xy_limit)
    ax.set_ylim(-xy_limit, xy_limit)
    ax.set_zlim(z_min - z_padding, z_max + z_padding)
    ax.set_box_aspect((2 * xy_limit, 2 * xy_limit, z_max - z_min + 2 * z_padding))

    ax.set_xlabel("X (mm)", fontsize=11)
    ax.set_ylabel("Y (mm)", fontsize=11)
    ax.set_zlabel("Z (mm)", fontsize=11)
    ax.set_facecolor(BG_COLOR)
