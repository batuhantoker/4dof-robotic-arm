"""
3D Visualization for 4-DOF Manipulator

Provides real-time arm rendering, trajectory animation, joint plots,
and workspace visualization using matplotlib.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.artist import Artist
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .robot import Robot4DOF


JOINT_COLOR = "#e74c3c"
EE_COLOR = "#f39c12"
TRAIL_COLOR = "#95a5a6"
BG_COLOR = "#fafafa"


def _finite_joint_matrix(
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


def _finite_time_vector(value: object) -> NDArray[np.float64]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("t must be a finite one-dimensional array") from exc

    if array.ndim != 1:
        raise ValueError("t must be a finite one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError("t must contain only finite values")
    return array


def _positive_integer(value: object, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


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

    # Links
    ax.plot(xs, ys, zs, "o-", color="#2c3e50", linewidth=3, markersize=8)

    # Highlight joints
    for p in points[:-1]:
        ax.scatter(
            p[0], p[1], p[2], color=JOINT_COLOR, s=80, zorder=5, depthshade=False
        )

    # End-effector
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

    _format_3d_axes(ax, robot)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def animate_trajectory(
    robot: Robot4DOF,
    trajectory_q: NDArray[np.float64],
    interval: int = 30,
    trail: bool = True,
    save_path: str | Path | None = None,
    show: bool = False,
) -> animation.FuncAnimation:
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
    trajectory_q = _finite_joint_matrix(
        trajectory_q,
        "trajectory_q",
        non_empty=True,
    )
    interval = _positive_integer(interval, "interval")

    fig = plt.figure(figsize=(10, 8), facecolor=BG_COLOR)
    ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))

    trail_xs: list[float] = []
    trail_ys: list[float] = []
    trail_zs: list[float] = []

    (line,) = ax.plot([], [], [], "o-", color="#2c3e50", linewidth=3, markersize=8)
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

    _format_3d_axes(ax, robot)
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

    anim = animation.FuncAnimation(
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
    t = _finite_time_vector(t)
    q = _finite_joint_matrix(q, "q")
    if len(t) != len(q):
        raise ValueError("t and q must have matching lengths")
    if qd is not None:
        qd = _finite_joint_matrix(qd, "qd")
        if qd.shape != q.shape:
            raise ValueError("qd must have the same shape as q")

    joint_names = ["q1 (base)", "q2 (shoulder)", "q3 (elbow)", "q4 (wrist)"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    n_rows = 2 if qd is not None else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows), facecolor=BG_COLOR)
    if n_rows == 1:
        axes = [axes]

    # Position plot
    for j in range(4):
        axes[0].plot(
            t, np.degrees(q[:, j]), color=colors[j], linewidth=2, label=joint_names[j]
        )
    axes[0].set_ylabel("Angle (deg)", fontsize=12)
    axes[0].set_title(title, fontsize=14, fontweight="bold")
    axes[0].legend(loc="best", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Velocity plot
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

    # 3D scatter
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

    # Top-down view (XY plane)
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


def _format_3d_axes(ax: Axes3D, robot: Robot4DOF) -> None:
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
