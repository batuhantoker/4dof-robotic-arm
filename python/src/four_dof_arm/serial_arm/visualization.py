"""Optional Matplotlib analysis plots and CPU-only Pillow animations in SI units."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from numpy.typing import NDArray

from .model import SerialArm


def render_analysis(
    arm: SerialArm,
    record: dict[str, NDArray[np.float64]],
    workspace: NDArray[np.float64],
    output_dir: Path,
    *,
    animate: bool = False,
) -> None:
    """Write overview.png and, optionally, motion.gif without showing windows.

    Times must be finite, nonempty, and strictly increasing (one pose is allowed).
    GIF poses are interpolated at 20 fps, excluding the terminal sample. Playback
    rounds up to a 50 ms frame, with a minimum of one frame and no duration cap.
    Scene bounds include up to 201 recorded poses and every rendered GIF pose.
    Only figures created here are closed, including when rendering fails.
    """
    t = record["t"]
    if (
        t.ndim != 1
        or t.size == 0
        or not np.all(np.isfinite(t))
        or np.any(np.diff(t) <= 0)
    ):
        raise ValueError("t must be finite, nonempty, and strictly increasing")
    shapes = {
        "q": (len(t), arm.n_joints),
        "qd": (len(t), arm.n_joints),
        "qdd": (len(t), arm.n_joints),
        "target_q": (len(t), arm.n_joints),
        "ee_pos": (len(t), 3),
        "target_ee_pos": (len(t), 3),
        "ee_speed": (len(t),),
    }
    for key in ("torque", "requested_torque"):
        if key in record:
            shapes[key] = (len(t), arm.n_joints)
    for key, shape in shapes.items():
        if record[key].shape != shape or not np.all(np.isfinite(record[key])):
            raise ValueError(f"{key} must be finite with shape {shape}")
    if (
        workspace.ndim != 2
        or workspace.shape[1] != 3
        or not np.all(np.isfinite(workspace))
    ):
        raise ValueError("workspace must be finite with shape (K, 3)")

    q = record["q"]
    representative = len(t) // 2
    pose = arm.joint_positions(q[representative])
    indices = np.linspace(0, len(t) - 1, min(len(t), 201), dtype=int)
    scene_points = [
        workspace,
        record["ee_pos"],
        record["target_ee_pos"],
        pose,
        np.concatenate([arm.joint_positions(q[index]) for index in indices]),
    ]
    fps = 20
    frame_times = np.empty(0, dtype=np.float64)
    frame_poses: list[NDArray[np.float64]] = []
    if animate:
        # Suppress floating-point noise at exact frame boundaries.
        count = max(1, int(np.ceil(round(float(t[-1] - t[0]) * fps, 10))))
        frame_times = t[0] + np.arange(count) / fps
        frame_q = np.column_stack(
            [np.interp(frame_times, t, q[:, joint]) for joint in range(arm.n_joints)]
        )
        frame_poses = [arm.joint_positions(values) for values in frame_q]
        scene_points.extend(frame_poses)
    points = np.concatenate(scene_points)
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = (lower + upper) / 2
    half_span = 0.55 * float(np.max(upper - lower))
    if half_span == 0:
        half_span = 0.05  # Give a fully coincident chain a visible metric scale.
    colors = plt.get_cmap("tab10")(np.arange(arm.n_joints) % 10)

    def scene(ax: Any) -> tuple[list[Any], Any]:
        ax.scatter(
            *workspace.T,
            s=3,
            color="#8296a8",
            alpha=0.18,
            depthshade=False,
            label="Workspace samples",
        )
        ax.plot(
            *record["target_ee_pos"].T,
            color="#64748b",
            linestyle="--",
            linewidth=1.5,
            label="Target tip path",
        )
        segments = []
        for joint, link in enumerate(arm.links):
            (segment,) = ax.plot(
                *pose[joint : joint + 2].T,
                color=colors[joint],
                linewidth=4,
                solid_capstyle="round",
                label=link.name,
            )
            segments.append(segment)
        (tool,) = ax.plot(*pose[-2:].T, color="#334155", linewidth=3, label="Tool")
        segments.append(tool)
        (joints,) = ax.plot(
            *pose.T, linestyle="none", marker="o", color="#334155", markersize=4
        )
        ax.set(
            xlabel="X (m)",
            ylabel="Y (m)",
            zlabel="Z (m)",
            xlim=(center[0] - half_span, center[0] + half_span),
            ylim=(center[1] - half_span, center[1] + half_span),
            zlim=(center[2] - half_span, center[2] + half_span),
        )
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=24, azim=-55)
        ax.set_autoscale_on(False)
        return segments, joints

    overview: Any = None
    motion: Any = None
    with plt.rc_context(
        {
            "figure.facecolor": "#f8f9fa",
            "axes.facecolor": "#f8f9fa",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#334155",
            "font.size": 9,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "grid.alpha": 0.25,
            "lines.linewidth": 1.6,
        }
    ):
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            overview = plt.figure(figsize=(14, 13), layout="constrained")
            overview.suptitle(f"{arm.name} | Motion analysis", fontsize=16)
            spatial: Any = overview.add_subplot(3, 2, 1, projection="3d")
            scene(spatial)
            spatial.plot(*record["ee_pos"].T, color="#dc2626", label="Actual tip path")
            spatial.set_title(f"Workspace and pose at t = {t[representative]:.3f} s")
            spatial.legend(loc="upper left", ncol=2)

            profiles = (
                ("q", "Joint angles (solid actual, dashed target)", "Angle (rad)"),
                ("qd", "Joint velocities", "Velocity (rad/s)"),
                ("qdd", "Joint accelerations", "Acceleration (rad/s^2)"),
                ("ee_speed", "Tool speed", "Speed (m/s)"),
                (
                    "torque",
                    "Joint torques (solid applied, dashed requested)",
                    "Torque (N m)",
                ),
            )
            for panel, (key, title, ylabel) in enumerate(profiles, start=2):
                ax: Any = overview.add_subplot(3, 2, panel)
                ax.set(title=title, xlabel="Time (s)", ylabel=ylabel)
                ax.grid(True)
                ax.spines[["top", "right"]].set_visible(False)
                if key == "ee_speed":
                    ax.plot(t, record[key], color="#dc2626", label="Tool")
                elif key == "torque" and not any(
                    field in record for field in ("torque", "requested_torque")
                ):
                    ax.set_axis_off()
                    ax.set_title("Joint torques")
                    ax.text(
                        0.5,
                        0.5,
                        "No torque data recorded",
                        ha="center",
                        transform=ax.transAxes,
                    )
                    continue
                else:
                    for joint, link in enumerate(arm.links):
                        if key in record:
                            ax.plot(
                                t,
                                record[key][:, joint],
                                color=colors[joint],
                                label=link.name,
                            )
                        target = {"q": "target_q", "torque": "requested_torque"}.get(
                            key
                        )
                        if target in record:
                            ax.plot(
                                t,
                                record[target][:, joint],
                                color=colors[joint],
                                linestyle="--",
                                alpha=0.75,
                                label=link.name if key not in record else "_nolegend_",
                            )
                ax.legend(loc="best", ncol=min(arm.n_joints, 3))
            overview.savefig(output_dir / "overview.png", dpi=150)

            if animate:
                motion = plt.figure(figsize=(7, 6), layout="constrained")
                animated: Any = motion.add_subplot(111, projection="3d")
                segments, joints = scene(animated)
                (trail,) = animated.plot(
                    [], [], [], color="#dc2626", label="Actual tip trail"
                )
                animated.legend(loc="upper left", ncol=2)
                tips = np.array([positions[-1] for positions in frame_poses])
                writer = PillowWriter(fps=fps)
                with writer.saving(motion, str(output_dir / "motion.gif"), dpi=90):
                    for frame, positions in enumerate(frame_poses):
                        for joint, segment in enumerate(segments):
                            segment.set_data_3d(*positions[joint : joint + 2].T)
                        joints.set_data_3d(*positions.T)
                        trail.set_data_3d(*tips[: frame + 1].T)
                        animated.set_title(
                            f"{arm.name} | t = {frame_times[frame]:.3f} s"
                        )
                        writer.grab_frame()
        finally:
            if overview is not None:
                plt.close(overview)
            if motion is not None:
                plt.close(motion)
