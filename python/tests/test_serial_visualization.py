from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image
from PIL.GifImagePlugin import GifImageFile

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.figure import Figure

from four_dof_arm.serial_arm.model import Link, SerialArm
from four_dof_arm.serial_arm.visualization import render_analysis

Case = tuple[SerialArm, dict[str, NDArray[np.float64]], NDArray[np.float64]]


@pytest.fixture(params=[2, 6], ids=["two-joints", "six-joints"])
def motion_case(request: pytest.FixtureRequest) -> Case:
    n = int(request.param)
    arm = SerialArm(
        links=tuple(
            Link(
                name=f"hinge_{joint}",
                axis=(0, 0, 1) if joint % 2 == 0 else (0, 1, 0),
                offset=(0.25 + 0.02 * joint, 0, 0.08),
                mass=0.5,
            )
            for joint in range(n)
        ),
        name="Test chain",
        base=(1.0, -0.7, 0.4),
        tool=(0.1, 0.02, 0.05),
    )
    t = np.array([2.0, 2.02, 2.1])
    q = np.outer([0.0, 0.6, 1.0], np.linspace(0.15, 0.5, n))
    qd = np.gradient(q, t, axis=0)
    ee_pos = np.array([arm.joint_positions(values)[-1] for values in q])
    target_q = q + 0.05
    record = {
        "t": t,
        "q": q,
        "qd": qd,
        "qdd": np.gradient(qd, t, axis=0),
        "target_q": target_q,
        "ee_pos": ee_pos,
        "target_ee_pos": np.array(
            [arm.joint_positions(values)[-1] for values in target_q]
        ),
        "ee_speed": np.linalg.norm(np.gradient(ee_pos, t, axis=0), axis=1),
    }
    workspace = np.vstack((arm.workspace(12, seed=5), [[-1.5, 2.0, -0.8]]))
    return arm, record, workspace


@pytest.mark.parametrize(
    "torques", [(), ("torque",), ("requested_torque",), ("torque", "requested_torque")]
)
def test_overview_png_profiles_and_bounds(
    motion_case: Case,
    torques: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm, record, workspace = motion_case
    for key in torques:
        record[key] = np.full_like(record["q"], 0.4 if key == "torque" else 0.6)
    figures: list[Any] = []
    savefig = Figure.savefig

    def capture(figure: Any, *args: Any, **kwargs: Any) -> None:
        figures.append(figure)
        savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    output_dir = tmp_path / "analysis"
    render_analysis(arm, record, workspace, output_dir)

    assert {path.name for path in output_dir.iterdir()} == {"overview.png"}
    with Image.open(output_dir / "overview.png") as image:
        assert image.format == "PNG"
        assert image.width > 0 and image.height > 0
        image.verify()
    assert plt.get_fignums() == []
    axes = figures[0].axes
    assert len(axes) == 6
    scene = axes[0]
    limits = np.array([scene.get_xlim(), scene.get_ylim(), scene.get_zlim()])
    np.testing.assert_allclose(
        np.diff(limits, axis=1), np.full((3, 1), np.diff(limits[0])[0])
    )
    points = np.concatenate(
        [workspace, record["ee_pos"], record["target_ee_pos"]]
        + [arm.joint_positions(q) for q in record["q"]]
    )
    assert np.all(points >= limits[:, 0])
    assert np.all(points <= limits[:, 1])
    np.testing.assert_allclose(
        scene.get_box_aspect(), np.full(3, scene.get_box_aspect()[0])
    )
    scene_lines = {line.get_label(): line for line in scene.lines}
    np.testing.assert_allclose(
        scene_lines["Actual tip path"].get_data_3d(), record["ee_pos"].T
    )
    np.testing.assert_allclose(
        scene_lines["Target tip path"].get_data_3d(), record["target_ee_pos"].T
    )
    for index, key in enumerate(("q", "qd", "qdd"), start=1):
        lines = {line.get_label(): line for line in axes[index].lines}
        assert set(lines) >= {link.name for link in arm.links}
        for joint, link in enumerate(arm.links):
            np.testing.assert_allclose(lines[link.name].get_xdata(), record["t"])
            np.testing.assert_allclose(
                lines[link.name].get_ydata(), record[key][:, joint]
            )
        assert axes[index].get_xlabel() == "Time (s)"
    targets = [line for line in axes[1].lines if line.get_linestyle() == "--"]
    np.testing.assert_allclose(
        [line.get_ydata() for line in targets], record["target_q"].T
    )
    assert [ax.get_ylabel() for ax in axes[1:5]] == [
        "Angle (rad)",
        "Velocity (rad/s)",
        "Acceleration (rad/s^2)",
        "Speed (m/s)",
    ]
    np.testing.assert_allclose(axes[4].lines[0].get_ydata(), record["ee_speed"])
    assert len(axes[5].lines) == len(torques) * arm.n_joints
    if torques:
        assert axes[5].get_ylabel() == "Torque (N m)"
        for key in torques:
            style = "-" if key == "torque" else "--"
            values = [
                line.get_ydata()
                for line in axes[5].lines
                if line.get_linestyle() == style
            ]
            np.testing.assert_allclose(values, record[key].T)


def test_real_gif_interpolation_timing_and_fixed_scene(
    motion_case: Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arm, record, workspace = motion_case
    frames: list[dict[str, Any]] = []
    grab_frame = PillowWriter.grab_frame

    def capture(writer: Any, **kwargs: Any) -> None:
        ax = writer.fig.axes[0]
        frames.append(
            {
                "axes": ax,
                "limits": [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()],
                "title": ax.get_title(),
                "lines": [np.array(line.get_data_3d(), copy=True) for line in ax.lines],
            }
        )
        grab_frame(writer, **kwargs)

    monkeypatch.setattr(PillowWriter, "grab_frame", capture)
    render_analysis(arm, record, workspace, tmp_path, animate=True)

    assert {path.name for path in tmp_path.iterdir()} == {"overview.png", "motion.gif"}
    assert plt.get_fignums() == []
    with Image.open(tmp_path / "motion.gif") as image:
        assert isinstance(image, GifImageFile)
        assert image.format == "GIF"
        assert image.n_frames == 2
        assert image.info["loop"] == 0
        durations = []
        for index in range(image.n_frames):
            image.seek(index)
            image.load()
            durations.append(image.info["duration"])
        assert durations == [50, 50]
    assert len(frames) == 2
    tips = []
    for index, frame in enumerate(frames):
        time = record["t"][0] + index / 20
        q = np.array([np.interp(time, record["t"], column) for column in record["q"].T])
        positions = arm.joint_positions(q)
        tips.append(positions[-1])
        assert frame["axes"] is frames[0]["axes"]
        assert len(frame["axes"].figure.axes) == 1
        assert frame["title"] == f"Test chain | t = {time:.3f} s"
        np.testing.assert_allclose(frame["limits"], frames[0]["limits"])
        for joint in range(arm.n_joints + 1):
            np.testing.assert_allclose(
                frame["lines"][joint + 1], positions[joint : joint + 2].T
            )
        np.testing.assert_allclose(frame["lines"][-2], positions.T)
        np.testing.assert_allclose(frame["lines"][-1], np.array(tips).T)


@pytest.mark.parametrize("stage", ["png", "gif"])
def test_render_failure_closes_figures(
    motion_case: Case,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("render failed")

    if stage == "png":
        monkeypatch.setattr(Figure, "savefig", fail)
    else:
        monkeypatch.setattr(PillowWriter, "finish", fail)
    arm, record, workspace = motion_case
    with pytest.raises(OSError, match="render failed"):
        render_analysis(arm, record, workspace, tmp_path, animate=True)
    assert plt.get_fignums() == []


@pytest.mark.parametrize("field", ["t", "q", "ee_pos", "torque", "workspace"])
def test_invalid_inputs_fail_before_creating_figures(
    motion_case: Case, tmp_path: Path, field: str
) -> None:
    arm, record, workspace = motion_case
    if field == "workspace":
        workspace = np.zeros((3, 2))
    elif field == "t":
        record["t"] = np.array([0.0, 0.0, 0.1])
    else:
        record[field] = np.array([np.nan])
    with pytest.raises(ValueError, match=field):
        render_analysis(arm, record, workspace, tmp_path, animate=True)
    assert plt.get_fignums() == []
    assert list(tmp_path.iterdir()) == []
