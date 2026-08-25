"""Trajectory animation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation as mpl_animation
from matplotlib.artist import Artist
from mpl_toolkits.mplot3d.axes3d import Axes3D

from .common import (
    BG_COLOR,
    EE_COLOR,
    LINK_COLOR,
    TRAIL_COLOR,
    finite_joint_matrix,
    format_3d_axes,
    positive_integer,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..robot import Robot4DOF


def animate_trajectory(
    robot: Robot4DOF,
    trajectory_q: NDArray[np.float64],
    interval: int = 30,
    trail: bool = True,
    save_path: str | Path | None = None,
    show: bool = False,
) -> mpl_animation.FuncAnimation:
    """Animate the arm following a joint trajectory.

    Parameters
    ----------
    robot : Robot4DOF
    trajectory_q : array of shape (N, 4)
        Joint angles at each time step.
    interval : int
        Milliseconds between frames.
    trail : bool
        Whether to draw the end-effector trail.
    save_path : str, Path, or None
        If provided, saves animation to file (mp4/gif).
    show : bool
        Whether to call plt.show().

    Returns
    -------
    anim : FuncAnimation
    """
    trajectory_q = finite_joint_matrix(
        trajectory_q,
        "trajectory_q",
        non_empty=True,
    )
    interval = positive_integer(interval, "interval")

    fig = plt.figure(figsize=(10, 8), facecolor=BG_COLOR)
    ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))

    trail_xs: list[float] = []
    trail_ys: list[float] = []
    trail_zs: list[float] = []

    (line,) = ax.plot([], [], [], "o-", color=LINK_COLOR, linewidth=3, markersize=8)
    (trail_line,) = ax.plot([], [], [], "-", color=TRAIL_COLOR, linewidth=1, alpha=0.6)
    (ee_marker,) = ax.plot(
        [],
        [],
        [],
        color=EE_COLOR,
        linestyle="None",
        marker="D",
        markersize=11,
    )

    format_3d_axes(ax, robot)
    ax.set_title("Trajectory Animation", fontsize=14, fontweight="bold", pad=15)

    def init() -> tuple[Artist, Artist, Artist]:
        trail_xs.clear()
        trail_ys.clear()
        trail_zs.clear()
        line.set_data([], [])
        line.set_3d_properties([])
        trail_line.set_data([], [])
        trail_line.set_3d_properties([])
        ee_marker.set_data_3d([], [], [])
        return line, trail_line, ee_marker

    def update(frame: int) -> tuple[Artist, Artist, Artist]:
        q = trajectory_q[frame]
        points = robot.joint_positions(q)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]

        line.set_data(xs, ys)
        line.set_3d_properties(zs)

        ee = points[-1]
        ee_marker.set_data_3d([ee[0]], [ee[1]], [ee[2]])

        if trail:
            trail_xs.append(ee[0])
            trail_ys.append(ee[1])
            trail_zs.append(ee[2])
            trail_line.set_data(trail_xs, trail_ys)
            trail_line.set_3d_properties(trail_zs)

        return line, trail_line, ee_marker

    anim = mpl_animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(trajectory_q),
        interval=interval,
        blit=False,
        repeat=False,
    )

    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        writer = "pillow" if path.suffix.lower() == ".gif" else "ffmpeg"
        anim.save(path, writer=writer)

    if show:
        plt.show()
    return anim
