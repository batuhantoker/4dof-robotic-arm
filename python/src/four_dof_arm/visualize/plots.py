"""Joint trajectory and workspace plots."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from .common import BG_COLOR, finite_joint_matrix, finite_time_vector

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..robot import Robot4DOF


def plot_joint_angles(
    t: NDArray[np.float64],
    q: NDArray[np.float64],
    qd: NDArray[np.float64] | None = None,
    title: str = "Joint Trajectories",
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot joint angle (and optionally velocity) profiles over time.

    Parameters
    ----------
    t : array of shape (N,)
        Time vector.
    q : array of shape (N, 4)
        Joint angles.
    qd : array of shape (N, 4) or None
        Joint velocities.
    title : str
    save_path : str, Path, or None
    show : bool
        Whether to call plt.show().

    Returns
    -------
    fig : Figure
    """
    t = finite_time_vector(t)
    q = finite_joint_matrix(q, "q")
    if len(t) != len(q):
        raise ValueError("t and q must have matching lengths")
    if qd is not None:
        qd = finite_joint_matrix(qd, "qd")
        if qd.shape != q.shape:
            raise ValueError("qd must have the same shape as q")

    joint_names = ["q1 (base)", "q2 (shoulder)", "q3 (elbow)", "q4 (wrist)"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    n_rows = 2 if qd is not None else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows), facecolor=BG_COLOR)
    if n_rows == 1:
        axes = [axes]

    for j in range(4):
        axes[0].plot(
            t, np.degrees(q[:, j]), color=colors[j], linewidth=2, label=joint_names[j]
        )
    axes[0].set_ylabel("Angle (deg)", fontsize=12)
    axes[0].set_title(title, fontsize=14, fontweight="bold")
    axes[0].legend(loc="best", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    if qd is not None:
        for j in range(4):
            axes[1].plot(
                t,
                np.degrees(qd[:, j]),
                color=colors[j],
                linewidth=2,
                label=joint_names[j],
            )
        axes[1].set_ylabel("Velocity (deg/s)", fontsize=12)
        axes[1].set_xlabel("Time (s)", fontsize=12)
        axes[1].legend(loc="best", fontsize=10)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[0].set_xlabel("Time (s)", fontsize=12)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_workspace(
    robot: Robot4DOF,
    n_samples: int = 10000,
    save_path: str | Path | None = None,
    show: bool = False,
    rng: np.random.Generator | None = None,
) -> Figure:
    """Visualize the reachable workspace as a 3D point cloud.

    Parameters
    ----------
    robot : Robot4DOF
    n_samples : int
    save_path : str, Path, or None
    show : bool
        Whether to call plt.show().
    rng : numpy.random.Generator or None
        Random generator forwarded to workspace sampling.

    Returns
    -------
    fig : Figure
    """
    points = robot.sample_workspace(n_samples=n_samples, rng=rng)

    fig = plt.figure(figsize=(12, 5), facecolor=BG_COLOR)

    ax1 = cast(Axes3D, fig.add_subplot(121, projection="3d"))
    distances = np.linalg.norm(points, axis=1)
    sc = ax1.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=distances,
        cmap="viridis",
        s=1,
        alpha=0.3,
    )
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_zlabel("Z (mm)")
    ax1.set_title("Reachable Workspace (3D)", fontsize=13, fontweight="bold")
    plt.colorbar(sc, ax=ax1, shrink=0.6, label="Distance (mm)")

    ax2 = fig.add_subplot(122)
    ax2.scatter(
        points[:, 0], points[:, 1], c=points[:, 2], cmap="coolwarm", s=1, alpha=0.3
    )
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.set_title("Workspace (Top View, color=Z)", fontsize=13, fontweight="bold")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
