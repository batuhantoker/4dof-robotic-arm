"""Static arm visualization."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from .common import BG_COLOR, EE_COLOR, JOINT_COLOR, LINK_COLOR, format_3d_axes

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..robot import Robot4DOF


def plot_arm(
    robot: Robot4DOF,
    q: NDArray[np.float64],
    ax: Axes3D | None = None,
    title: str | None = None,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot the arm in a single static configuration.

    Parameters
    ----------
    robot : Robot4DOF
    q : array of shape (4,)
        Joint angles.
    ax : Axes3D or None
        If None, creates a new figure.
    title : str or None
    show : bool
        Whether to call plt.show().
    save_path : str, Path, or None
        If provided, saves the figure to this path.

    Returns
    -------
    fig : Figure
    """
    points = robot.joint_positions(q)

    if ax is None:
        fig = plt.figure(figsize=(10, 8), facecolor=BG_COLOR)
        ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))
    else:
        parent = ax.get_figure()
        if not isinstance(parent, Figure):
            raise ValueError("ax must belong to a matplotlib Figure")
        fig = parent

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]

    ax.plot(xs, ys, zs, "o-", color=LINK_COLOR, linewidth=3, markersize=8)

    for p in points[:-1]:
        ax.scatter(
            p[0], p[1], p[2], color=JOINT_COLOR, s=80, zorder=5, depthshade=False
        )

    ax.scatter(
        points[-1][0],
        points[-1][1],
        points[-1][2],
        color=EE_COLOR,
        s=120,
        marker="D",
        zorder=5,
        depthshade=False,
    )

    format_3d_axes(ax, robot)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig
