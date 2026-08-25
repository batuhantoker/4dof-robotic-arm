from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import four_dof_arm.visualize.animation as animation_module
import four_dof_arm.visualize.arm as arm_module
import four_dof_arm.visualize.plots as plots_module
from four_dof_arm import Robot4DOF, demo, visualize
from four_dof_arm.visualize import (
    animate_trajectory,
    plot_arm,
    plot_joint_angles,
    plot_workspace,
)


def test_visualize_facade_exports_owner_functions() -> None:
    assert visualize.__all__ == [
        "plot_arm",
        "animate_trajectory",
        "plot_joint_angles",
        "plot_workspace",
    ]
    assert visualize.plot_arm is plot_arm is arm_module.plot_arm
    assert (
        visualize.animate_trajectory
        is animate_trajectory
        is animation_module.animate_trajectory
    )
    assert (
        visualize.plot_joint_angles
        is plot_joint_angles
        is plots_module.plot_joint_angles
    )
    assert visualize.plot_workspace is plot_workspace is plots_module.plot_workspace


@pytest.fixture(autouse=True)
def close_figures_after_test() -> Iterator[None]:
    yield
    plt.close("all")


def test_plotting_defaults_do_not_show_or_write(monkeypatch, tmp_path: Path) -> None:
    show = Mock()
    monkeypatch.setattr(plt, "show", show)
    monkeypatch.chdir(tmp_path)

    robot = Robot4DOF()
    trajectory_q = np.array(
        [
            np.zeros(4),
            np.deg2rad([20.0, 10.0, -15.0, 5.0]),
        ],
    )
    t = np.array([0.0, 1.0])

    plot_arm(robot, trajectory_q[0])
    plot_joint_angles(t, trajectory_q)
    plot_workspace(robot, n_samples=8, rng=np.random.default_rng(7))
    trajectory_animation = animate_trajectory(robot, trajectory_q)
    plt.gcf().canvas.draw()

    assert trajectory_animation is not None
    show.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_save_paths_and_show_true_use_real_renderers(
    monkeypatch,
    tmp_path: Path,
) -> None:
    show = Mock()
    monkeypatch.setattr(plt, "show", show)

    robot = Robot4DOF()
    trajectory_q = np.array(
        [
            np.zeros(4),
            np.deg2rad([15.0, 20.0, -10.0, 5.0]),
        ],
    )
    t = np.array([0.0, 1.0])
    qd = np.array(
        [
            np.zeros(4),
            np.deg2rad([1.0, 2.0, 3.0, 4.0]),
        ],
    )
    paths = [
        tmp_path / "arm.png",
        tmp_path / "joints.png",
        tmp_path / "workspace.png",
        tmp_path / "trajectory.gif",
    ]

    plot_arm(robot, trajectory_q[0], save_path=paths[0], show=True)
    plot_joint_angles(t, trajectory_q, qd, save_path=paths[1], show=True)
    plot_workspace(
        robot,
        n_samples=8,
        save_path=paths[2],
        show=True,
        rng=np.random.default_rng(7),
    )
    trajectory_animation = animate_trajectory(
        robot,
        trajectory_q,
        interval=1,
        save_path=paths[3],
        show=True,
    )

    assert trajectory_animation is not None
    assert show.call_count == 4
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_animation_callbacks_reset_and_update_all_coordinates(monkeypatch) -> None:
    captured: dict[str, object] = {}
    animation_result = Mock()

    def capture_animation(figure, update, **kwargs):
        captured.update(figure=figure, update=update, **kwargs)
        return animation_result

    monkeypatch.setattr(
        animation_module.mpl_animation,
        "FuncAnimation",
        capture_animation,
    )

    robot = Robot4DOF()
    trajectory_q = np.array(
        [
            np.zeros(4),
            np.deg2rad([15.0, 20.0, -10.0, 5.0]),
        ],
    )
    result = animate_trajectory(robot, trajectory_q, interval=np.int64(25))

    assert result is animation_result
    assert captured["frames"] == 2
    assert captured["interval"] == 25
    assert captured["blit"] is False
    assert captured["repeat"] is False

    init_callback = captured["init_func"]
    update_callback = captured["update"]
    assert callable(init_callback)
    assert callable(update_callback)

    initial_artists = init_callback()
    first_artists = update_callback(0)
    updated_artists = update_callback(1)
    expected_points = np.asarray(robot.joint_positions(trajectory_q[1]))
    expected_end_effectors = np.asarray(
        [robot.joint_positions(q)[-1] for q in trajectory_q]
    )

    assert len(initial_artists) == 3
    assert first_artists == updated_artists
    np.testing.assert_allclose(
        np.asarray(updated_artists[0].get_data_3d(), dtype=np.float64),
        expected_points.T,
    )
    np.testing.assert_allclose(
        np.asarray(updated_artists[1].get_data_3d(), dtype=np.float64),
        expected_end_effectors.T,
    )
    np.testing.assert_allclose(
        np.asarray(updated_artists[2].get_data_3d(), dtype=np.float64),
        expected_end_effectors[-1, :, np.newaxis],
    )

    reset_artists = init_callback()
    for artist in reset_artists:
        assert all(len(coordinates) == 0 for coordinates in artist.get_data_3d())


@pytest.mark.parametrize(
    "trajectory_q",
    [
        np.empty((0, 4)),
        np.zeros(4),
        np.zeros((2, 3)),
        np.array([[0.0, 0.0, 0.0, np.nan]]),
        [["invalid", 0.0, 0.0, 0.0]],
    ],
)
def test_animation_rejects_invalid_trajectory(trajectory_q: object) -> None:
    with pytest.raises(ValueError, match="trajectory_q"):
        animate_trajectory(Robot4DOF(), trajectory_q)  # type: ignore[arg-type]

    assert plt.get_fignums() == []


@pytest.mark.parametrize("interval", [0, -1, 1.5, True, "30"])
def test_animation_rejects_invalid_interval(interval: object) -> None:
    with pytest.raises(ValueError, match="interval.*positive integer"):
        animate_trajectory(
            Robot4DOF(),
            np.zeros((1, 4)),
            interval=interval,  # type: ignore[arg-type]
        )

    assert plt.get_fignums() == []


def test_joint_plots_contain_time_angles_and_velocities() -> None:
    t = np.array([0.0, 0.5, 1.0])
    q = np.deg2rad(
        [
            [0.0, 10.0, 20.0, 30.0],
            [5.0, 15.0, 25.0, 35.0],
            [10.0, 20.0, 30.0, 40.0],
        ]
    )
    qd = np.deg2rad(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
        ]
    )

    figure = plot_joint_angles(t, q, qd)

    for joint, line in enumerate(figure.axes[0].lines):
        np.testing.assert_array_equal(line.get_xdata(), t)
        np.testing.assert_allclose(line.get_ydata(), np.degrees(q[:, joint]))
    for joint, line in enumerate(figure.axes[1].lines):
        np.testing.assert_array_equal(line.get_xdata(), t)
        np.testing.assert_allclose(line.get_ydata(), np.degrees(qd[:, joint]))


@pytest.mark.parametrize(
    "t",
    [
        np.zeros((2, 1)),
        np.array([0.0, np.nan]),
        ["invalid", 1.0],
    ],
)
def test_joint_plot_rejects_invalid_time(t: object) -> None:
    with pytest.raises(ValueError, match="t must"):
        plot_joint_angles(t, np.zeros((2, 4)))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "q",
    [
        np.zeros(4),
        np.zeros((2, 3)),
        np.array([[0.0, 0.0, 0.0, np.inf], [0.0, 0.0, 0.0, 0.0]]),
        [["invalid", 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
    ],
)
def test_joint_plot_rejects_invalid_positions(q: object) -> None:
    with pytest.raises(ValueError, match="q must"):
        plot_joint_angles(np.array([0.0, 1.0]), q)  # type: ignore[arg-type]


def test_joint_plot_rejects_mismatched_time_and_positions() -> None:
    with pytest.raises(ValueError, match="t and q must have matching lengths"):
        plot_joint_angles(np.array([0.0]), np.zeros((2, 4)))


@pytest.mark.parametrize(
    "qd",
    [
        np.zeros((2, 3)),
        np.zeros((3, 4)),
        np.array([[0.0, 0.0, 0.0, np.nan], [0.0, 0.0, 0.0, 0.0]]),
        [["invalid", 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
    ],
)
def test_joint_plot_rejects_invalid_velocities(qd: object) -> None:
    with pytest.raises(ValueError, match="qd must"):
        plot_joint_angles(
            np.array([0.0, 1.0]),
            np.zeros((2, 4)),
            qd,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "q",
    [
        np.zeros(3),
        np.array([0.0, 0.0, 0.0, np.nan]),
    ],
)
def test_plot_arm_uses_robot_joint_validation_before_creating_figure(
    q: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="q must"):
        plot_arm(Robot4DOF(), q)

    assert plt.get_fignums() == []


def test_plot_workspace_uses_robot_sample_count_validation() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        plot_workspace(Robot4DOF(), n_samples=0)

    assert plt.get_fignums() == []


def test_arm_axes_contain_the_robot_reach() -> None:
    robot = Robot4DOF(Lax=50.0, Laz=55.0, Lb=100.0, Lt=90.0)
    figure = plot_arm(robot, np.zeros(4))
    axes = figure.axes[0]

    horizontal_reach = robot.Lax + robot.Lb + robot.Lt
    z_min = min(0.0, robot.Laz - robot.Lt)
    z_max = max(0.0, robot.Laz + robot.Lt)

    assert axes.get_xlim()[0] <= -horizontal_reach
    assert axes.get_xlim()[1] >= horizontal_reach
    assert axes.get_ylim()[0] <= -horizontal_reach
    assert axes.get_ylim()[1] >= horizontal_reach
    assert axes.get_zlim()[0] <= z_min
    assert axes.get_zlim()[1] >= z_max

    valid_configurations = np.array(
        [
            [0.0, 0.0, np.pi / 2, np.pi / 2],
            [np.pi, 0.0, np.pi / 2, np.pi / 2],
            [0.0, np.pi / 2, 0.0, np.pi / 2],
            [0.0, -np.pi / 2, 0.0, -np.pi / 2],
        ],
    )
    for q in valid_configurations:
        points = np.asarray(robot.joint_positions(q))
        assert np.all(
            (axes.get_xlim()[0] <= points[:, 0]) & (points[:, 0] <= axes.get_xlim()[1])
        )
        assert np.all(
            (axes.get_ylim()[0] <= points[:, 1]) & (points[:, 1] <= axes.get_ylim()[1])
        )
        assert np.all(
            (axes.get_zlim()[0] <= points[:, 2]) & (points[:, 2] <= axes.get_zlim()[1])
        )


def test_workspace_forwards_sample_count_and_generator(monkeypatch) -> None:
    robot = Robot4DOF()
    generator = np.random.default_rng(42)
    sample_workspace = Mock(wraps=robot.sample_workspace)
    monkeypatch.setattr(robot, "sample_workspace", sample_workspace)

    plot_workspace(robot, n_samples=16, rng=generator)

    sample_workspace.assert_called_once_with(n_samples=16, rng=generator)


def test_demo_cli_arguments() -> None:
    args = demo._parse_args(["--show", "--output-dir", "renders", "--seed", "19"])

    assert args.show is True
    assert args.output_dir == Path("renders")
    assert args.seed == 19


def test_demo_real_agg_smoke_without_output(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    monkeypatch.setattr(demo, "_WORKSPACE_SAMPLES", 16)
    monkeypatch.chdir(tmp_path)

    demo.main(show=False, output_dir=None, seed=23)

    assert list(tmp_path.iterdir()) == []
    assert plt.get_fignums() == []
    assert "Done." in capsys.readouterr().out


def test_demo_real_agg_writes_expected_pngs(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    monkeypatch.setattr(demo, "_WORKSPACE_SAMPLES", 16)
    output_dir = tmp_path / "renders"

    demo.main(show=False, output_dir=output_dir, seed=23)

    expected = {
        output_dir / "arm_pose.png",
        output_dir / "joint_trajectory.png",
        output_dir / "workspace.png",
    }
    assert set(output_dir.iterdir()) == expected
    assert all(path.stat().st_size > 0 for path in expected)
    assert plt.get_fignums() == []
    assert "Done." in capsys.readouterr().out
