from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from four_dof_arm import mujoco_sim
from four_dof_arm.mujoco_sim import ArmSim
from four_dof_arm.mujoco_sim import cli as mujoco_cli
from four_dof_arm.mujoco_sim import core as mujoco_core
from four_dof_arm.mujoco_sim import rendering as mujoco_rendering
from four_dof_arm.mujoco_sim import validation as mujoco_validation
from four_dof_arm.trajectory import MultiPointTrajectory

CUSTOM_MODEL = """
<mujoco model="actuator_test">
  <option timestep="0.002" gravity="0 0 0"/>
  <default>
    <geom type="sphere" size="0.01" mass="0.1" contype="0" conaffinity="0"/>
  </default>
  <worldbody>
    <body name="link1">
      <joint name="q1" type="hinge"/>
      <geom/>
      <body name="link2" pos="0.02 0 0">
        <joint name="q2" type="hinge"/>
        <geom/>
        <body name="link3" pos="0.02 0 0">
          <joint name="q3" type="hinge"/>
          <geom/>
          <body name="link4" pos="0.02 0 0">
            <joint name="q4" type="hinge"/>
            <geom/>
            {site}
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    {actuators}
  </actuator>
</mujoco>
"""


def _write_custom_model(
    tmp_path: Path,
    actuators: str,
    *,
    site: str = '<site name="end_effector" pos="0.02 0 0"/>',
) -> Path:
    model_path = tmp_path / "custom_model.xml"
    model_path.write_text(
        CUSTOM_MODEL.format(actuators=actuators, site=site), encoding="utf-8"
    )
    return model_path


class _FakeVideoSim:
    def __init__(self, dt: float = 0.05) -> None:
        self.model = object()
        self.data = SimpleNamespace(time=0.0)
        self.n_joints = 4
        self.dt = dt
        self.step_targets: list[np.ndarray] = []

    def step(self, target: np.ndarray) -> None:
        self.step_targets.append(np.array(target, copy=True))
        self.data.time += self.dt


class _VideoHarness:
    def __init__(self) -> None:
        self.return_code = 0
        self.stderr = b""
        self.break_pipe = False
        self.render_error: Exception | None = None
        self.renderers: list[object] = []
        self.processes: list[object] = []


@pytest.fixture
def video_harness(monkeypatch: pytest.MonkeyPatch) -> _VideoHarness:
    harness = _VideoHarness()

    class FakeStdin:
        def __init__(self) -> None:
            self.closed = False
            self.writes: list[bytes] = []

        def write(self, data: bytes) -> int:
            if harness.break_pipe:
                raise BrokenPipeError
            self.writes.append(data)
            return len(data)

        def close(self) -> None:
            self.closed = True

    class FakeProcess:
        def __init__(
            self,
            command: list[str],
            *,
            stdin: int,
            stderr: object,
        ) -> None:
            assert stdin == mujoco_rendering.subprocess.PIPE
            self.stdin = FakeStdin()
            self.running = True
            self.terminated = False
            self.killed = False
            self.wait_calls: list[float | None] = []
            Path(command[-1]).write_bytes(b"fake video")
            if harness.stderr:
                stderr.write(harness.stderr)  # type: ignore[attr-defined]
            harness.processes.append(self)

        def wait(self, timeout: float | None = None) -> int:
            self.wait_calls.append(timeout)
            self.running = False
            return harness.return_code

        def poll(self) -> int | None:
            return None if self.running else harness.return_code

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

    class FakeRenderer:
        def __init__(self, _model: object, *, height: int, width: int) -> None:
            self.height = height
            self.width = width
            self.update_times: list[float] = []
            self.closed = False
            harness.renderers.append(self)

        def update_scene(self, data: SimpleNamespace, _camera: object) -> None:
            self.update_times.append(float(data.time))

        def render(self) -> np.ndarray:
            if harness.render_error is not None:
                raise harness.render_error
            return np.zeros((1, 1, 3), dtype=np.uint8)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(mujoco_rendering.shutil, "which", lambda _name: "/fake/ffmpeg")
    monkeypatch.setattr(mujoco_rendering.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(mujoco_rendering.mujoco, "Renderer", FakeRenderer)
    return harness


@pytest.fixture
def sim() -> ArmSim:
    return ArmSim()


def test_package_facade_exports_public_api() -> None:
    assert mujoco_sim.__all__ == [
        "ArmSim",
        "MODEL_PATH",
        "validate_fk",
        "render_video",
        "demo_trajectory",
        "main",
    ]
    assert mujoco_sim.ArmSim is mujoco_core.ArmSim
    assert mujoco_sim.MODEL_PATH is mujoco_core.MODEL_PATH
    assert mujoco_sim.demo_trajectory is mujoco_cli.demo_trajectory
    assert mujoco_sim.main is mujoco_cli.main
    assert mujoco_sim.render_video is mujoco_rendering.render_video
    assert mujoco_sim.validate_fk is mujoco_validation.validate_fk


def test_fk_validation_matches_analytical_model(
    sim: ArmSim, capsys: pytest.CaptureFixture[str]
) -> None:
    assert mujoco_validation.validate_fk(sim)
    output = capsys.readouterr().out
    assert "Rot err (deg)" in output
    assert "ALL PASS" in output


def test_fk_validation_rejects_orientation_mismatch(
    sim: ArmSim,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_fk = sim.robot.forward_kinematics
    quarter_turn = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    def mismatched_orientation(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        position, rotation = original_fk(q)
        return position, rotation @ quarter_turn

    monkeypatch.setattr(sim.robot, "forward_kinematics", mismatched_orientation)

    assert not mujoco_validation.validate_fk(sim)
    assert "SOME FAILED" in capsys.readouterr().out


def test_initial_derived_state_matches_analytical_fk(sim: ArmSim) -> None:
    analytical_position, analytical_rotation = sim.robot.forward_kinematics(sim.q)

    np.testing.assert_allclose(sim.ee_pos, analytical_position / 1000.0, atol=1e-12)
    np.testing.assert_allclose(sim.ee_rotation, analytical_rotation, atol=1e-12)


def test_post_step_derived_state_matches_analytical_fk(sim: ArmSim) -> None:
    q0 = np.array([0.1, -0.2, 0.15, -0.1])
    target = q0 + np.array([0.2, -0.1, 0.1, 0.15])
    sim.reset(q0)

    sim.step(target)

    analytical_position, _ = sim.robot.forward_kinematics(sim.q)
    np.testing.assert_allclose(sim.ee_pos, analytical_position / 1000.0, atol=1e-12)

    post_step_bias = sim.data.qfrc_bias[: sim.n_joints].copy()
    mujoco_core.mujoco.mj_forward(sim.model, sim.data)
    np.testing.assert_allclose(post_step_bias, sim.data.qfrc_bias[: sim.n_joints])


def test_step_converts_desired_joint_torque_using_actuator_gears(sim: ArmSim) -> None:
    q0 = np.deg2rad([10.0, -15.0, 20.0, -25.0])
    target = q0 + np.array([0.1, -0.2, 0.3, -0.4])
    sim.reset(q0)

    desired_torque = (
        sim.Kp * (target - sim.q) - sim.Kd * sim.qd + sim.data.qfrc_bias[: sim.n_joints]
    )
    actuator_scale = sim.model.actuator_gear[:, 0] * sim.model.actuator_gainprm[:, 0]

    sim.step(target)

    np.testing.assert_allclose(sim.data.ctrl, desired_torque / actuator_scale)
    np.testing.assert_allclose(
        sim.data.qfrc_actuator[: sim.n_joints], desired_torque, atol=1e-12
    )


def test_step_supports_reordered_non_unit_actuator_mappings(tmp_path: Path) -> None:
    model_path = _write_custom_model(
        tmp_path,
        """
        <general name="motor_q3" joint="q3" gear="-2" gainprm="1.5"/>
        <general name="motor_q1" joint="q1" gear="4" gainprm="0.5"/>
        <general name="motor_q4" joint="q4" gear="0.25" gainprm="8"/>
        <general name="motor_q2" joint="q2" gear="-5" gainprm="0.4"/>
        """,
    )
    custom_sim = ArmSim(model_path)
    q0 = np.array([0.1, -0.2, 0.3, -0.4])
    target = q0 + np.array([0.2, -0.1, 0.15, -0.05])
    custom_sim.reset(q0)
    desired_torque = (
        custom_sim.Kp * (target - custom_sim.q)
        - custom_sim.Kd * custom_sim.qd
        + custom_sim.data.qfrc_bias[: custom_sim.n_joints]
    )

    custom_sim.step(target)

    np.testing.assert_array_equal(custom_sim._actuator_ids, [1, 3, 0, 2])
    np.testing.assert_allclose(
        custom_sim._control_to_joint_torque, [2.0, -2.0, -3.0, 2.0]
    )
    expected_ctrl = np.zeros(4)
    expected_ctrl[[1, 3, 0, 2]] = desired_torque / np.array([2.0, -2.0, -3.0, 2.0])
    np.testing.assert_allclose(custom_sim.data.ctrl, expected_ctrl)
    np.testing.assert_allclose(
        custom_sim.data.qfrc_actuator[: custom_sim.n_joints],
        desired_torque,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "site",
    ['<site name="tool_tip" pos="0.02 0 0"/>', ""],
    ids=["wrong-name", "no-sites"],
)
def test_rejects_model_without_named_end_effector_site(
    tmp_path: Path, site: str
) -> None:
    model_path = _write_custom_model(
        tmp_path,
        """
        <motor joint="q1"/>
        <motor joint="q2"/>
        <motor joint="q3"/>
        <motor joint="q4"/>
        """,
        site=site,
    )

    with pytest.raises(
        ValueError, match="model must define a site named 'end_effector'"
    ):
        ArmSim(model_path)


@pytest.mark.parametrize(
    ("first_actuator", "error"),
    [
        (
            '<general site="end_effector" gear="1 0 0 0 0 0"/>',
            "direct joint transmission",
        ),
        (
            '<general joint="q1" gaintype="affine" gainprm="1 2"/>',
            "fixed gain",
        ),
        ('<position joint="q1" kp="2"/>', "actuator bias"),
        (
            '<general joint="q1" dyntype="filter" dynprm="0.1"/>',
            "activation dynamics",
        ),
    ],
)
def test_rejects_unsupported_actuator_semantics(
    tmp_path: Path, first_actuator: str, error: str
) -> None:
    model_path = _write_custom_model(
        tmp_path,
        f"""
        {first_actuator}
        <motor joint="q2"/>
        <motor joint="q3"/>
        <motor joint="q4"/>
        """,
    )

    with pytest.raises(ValueError, match=error):
        ArmSim(model_path)


@pytest.mark.parametrize(
    "target",
    [
        np.zeros(3),
        np.zeros((1, 4)),
        np.array([0.0, 0.0, np.nan, 0.0]),
        np.array([0.0, 0.0, 0.0, np.inf]),
    ],
)
def test_step_rejects_invalid_targets(sim: ArmSim, target: np.ndarray) -> None:
    with pytest.raises(ValueError):
        sim.step(target)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("duration", 0.0),
        ("duration", -1.0),
        ("duration", np.nan),
        ("duration", np.inf),
        ("duration", -np.inf),
        ("tol", 0.0),
        ("tol", -1.0),
        ("tol", np.nan),
        ("tol", np.inf),
        ("tol", -np.inf),
    ],
)
def test_settle_rejects_non_positive_or_non_finite_limits(
    sim: ArmSim, argument: str, value: float
) -> None:
    options = {"duration": 0.1, "tol": 1e-3, argument: value}

    with pytest.raises(ValueError, match=argument):
        sim.settle(np.zeros(sim.n_joints), **options)


def test_settle_reduces_error_and_converges(sim: ArmSim) -> None:
    target = np.array([0.2, -0.15, 0.1, -0.1])
    initial_error = float(np.linalg.norm(target - sim.q))

    final_error = sim.settle(target, duration=2.0, tol=1e-3)

    assert final_error < initial_error / 100.0
    assert final_error < 1e-3
    assert np.linalg.norm(sim.qd) < 1e-3
    assert final_error == pytest.approx(np.linalg.norm(target - sim.q))


@pytest.mark.parametrize(
    "timestamps",
    [
        np.array([0.0]),
        np.array([[0.0, 1.0]]),
        np.array([0.1, 1.0]),
        np.array([0.0, 0.5, 0.5]),
        np.array([0.0, 1.0, 0.5]),
        np.array([0.0, np.nan]),
        np.array([0.0, np.inf]),
    ],
)
def test_run_trajectory_rejects_invalid_timestamps(
    sim: ArmSim, timestamps: np.ndarray
) -> None:
    trajectory = np.zeros((timestamps.size, sim.n_joints))
    with pytest.raises(ValueError):
        sim.run_trajectory(timestamps, trajectory)


@pytest.mark.parametrize(
    "trajectory",
    [
        np.zeros((2, 3)),
        np.zeros((3, 4)),
        np.array([[0.0, 0.0, 0.0, 0.0], [0.0, np.nan, 0.0, 0.0]]),
    ],
)
def test_run_trajectory_rejects_invalid_joint_trajectories(
    sim: ArmSim, trajectory: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        sim.run_trajectory(np.array([0.0, 1.0]), trajectory)


def test_run_trajectory_controls_current_time_and_records_post_step_reference(
    sim: ArmSim, monkeypatch: pytest.MonkeyPatch
) -> None:
    duration = 3.0 * sim.dt
    endpoint = np.array([0.3, -0.6, 0.9, -1.2])
    controls: list[np.ndarray] = []
    q_snapshots = [sim.q]
    qd_snapshots = [sim.qd]
    ee_pos_snapshots = [sim.ee_pos]
    original_step = sim.step

    def record_control(target: np.ndarray) -> None:
        controls.append(np.array(target, copy=True))
        original_step(target)
        q_snapshots.append(sim.q)
        qd_snapshots.append(sim.qd)
        ee_pos_snapshots.append(sim.ee_pos)

    monkeypatch.setattr(sim, "step", record_control)
    record = sim.run_trajectory(
        np.array([0.0, duration]),
        np.vstack([np.zeros(sim.n_joints), endpoint]),
    )

    expected_controls = np.outer([0.0, 1.0 / 3.0, 2.0 / 3.0], endpoint)
    expected_recorded_targets = np.outer([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], endpoint)
    np.testing.assert_allclose(controls, expected_controls, atol=1e-14)
    np.testing.assert_allclose(
        record["target_q"], expected_recorded_targets, atol=1e-14
    )
    np.testing.assert_allclose(
        record["t"], [0.0, sim.dt, 2.0 * sim.dt, 3.0 * sim.dt], atol=1e-14
    )
    np.testing.assert_allclose(record["q"], q_snapshots, atol=1e-14)
    np.testing.assert_allclose(record["qd"], qd_snapshots, atol=1e-14)
    np.testing.assert_allclose(record["ee_pos"], ee_pos_snapshots, atol=1e-14)


def test_run_trajectory_exact_grid_roundoff_does_not_add_step(sim: ArmSim) -> None:
    grid_duration = 5.0 * sim.dt
    duration = np.nextafter(grid_duration, np.inf)

    record = sim.run_trajectory(np.array([0.0, duration]), np.zeros((2, sim.n_joints)))

    assert len(record["t"]) == 6
    assert record["t"][-1] == pytest.approx(grid_duration, abs=1e-15)


def test_run_trajectory_off_grid_duration_rounds_up_one_step(sim: ArmSim) -> None:
    duration = 5.25 * sim.dt

    record = sim.run_trajectory(np.array([0.0, duration]), np.zeros((2, sim.n_joints)))

    assert len(record["t"]) == 7
    assert record["t"][-1] == pytest.approx(6.0 * sim.dt, abs=1e-15)
    assert record["t"][-1] - duration < sim.dt


@pytest.mark.parametrize(
    ("duration", "expected_frames"),
    [
        (np.nextafter(0.3, np.inf), 3),
        (0.35, 4),
    ],
)
def test_render_video_uses_zero_based_unique_frames_and_preserves_duration(
    tmp_path: Path,
    video_harness: _VideoHarness,
    duration: float,
    expected_frames: int,
) -> None:
    fake_sim = _FakeVideoSim()
    output = tmp_path / "video.mp4"
    trajectory = np.vstack([np.zeros(4), np.ones(4)])

    result = mujoco_rendering.render_video(
        fake_sim,
        np.array([0.0, duration]),
        trajectory,
        output_path=output,
        fps=10.0,
    )

    process = video_harness.processes[0]
    renderer = video_harness.renderers[0]
    assert result == str(output)
    assert len(process.stdin.writes) == expected_frames  # type: ignore[attr-defined]
    assert len(renderer.update_times) == expected_frames  # type: ignore[attr-defined]
    assert renderer.update_times[0] == 0.0  # type: ignore[attr-defined]
    assert len(set(renderer.update_times)) == expected_frames  # type: ignore[attr-defined]
    assert renderer.update_times[-1] < duration  # type: ignore[attr-defined]
    assert abs(expected_frames / 10.0 - duration) <= 0.1
    assert fake_sim.data.time == pytest.approx(duration, abs=1e-12)
    assert fake_sim.data.time > renderer.update_times[-1]  # type: ignore[attr-defined]
    np.testing.assert_array_equal(fake_sim.step_targets[0], np.zeros(4))
    assert process.stdin.closed  # type: ignore[attr-defined]
    assert renderer.closed  # type: ignore[attr-defined]


def test_render_video_applies_piecewise_targets_through_final_simulation_step(
    tmp_path: Path, video_harness: _VideoHarness
) -> None:
    fake_sim = _FakeVideoSim()
    timestamps = np.array([0.0, 0.1, 0.35])
    trajectory = np.array(
        [
            [0.0, 10.0, -4.0, 1.0],
            [2.0, 8.0, 0.0, -1.0],
            [-3.0, 3.0, 5.0, 4.0],
        ]
    )

    mujoco_rendering.render_video(
        fake_sim,
        timestamps,
        trajectory,
        output_path=tmp_path / "video.mp4",
        fps=10.0,
    )

    expected_targets = np.array(
        [
            [0.0, 10.0, -4.0, 1.0],
            [1.0, 9.0, -2.0, 0.0],
            [2.0, 8.0, 0.0, -1.0],
            [1.0, 7.0, 1.0, 0.0],
            [0.0, 6.0, 2.0, 1.0],
            [-1.0, 5.0, 3.0, 2.0],
            [-2.0, 4.0, 4.0, 3.0],
        ]
    )
    renderer = video_harness.renderers[0]
    np.testing.assert_allclose(fake_sim.step_targets, expected_targets, atol=1e-14)
    np.testing.assert_allclose(
        renderer.update_times,
        [0.0, 0.1, 0.2, 0.3],
        atol=1e-14,  # type: ignore[attr-defined]
    )
    assert fake_sim.data.time == pytest.approx(0.35, abs=1e-14)


def test_render_video_reports_missing_ffmpeg_before_allocating_resources(
    tmp_path: Path,
    video_harness: _VideoHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mujoco_rendering.shutil, "which", lambda _name: None)

    with pytest.raises(FileNotFoundError, match="ffmpeg executable"):
        mujoco_rendering.render_video(
            _FakeVideoSim(),
            np.array([0.0, 0.1]),
            np.zeros((2, 4)),
            output_path=tmp_path / "video.mp4",
            fps=10.0,
        )

    assert not video_harness.renderers
    assert not video_harness.processes


def test_render_video_reports_ffmpeg_nonzero_and_closes_resources(
    tmp_path: Path, video_harness: _VideoHarness
) -> None:
    video_harness.return_code = 7
    video_harness.stderr = b"encoder failed"

    with pytest.raises(RuntimeError, match="status 7: encoder failed"):
        mujoco_rendering.render_video(
            _FakeVideoSim(),
            np.array([0.0, 0.1]),
            np.zeros((2, 4)),
            output_path=tmp_path / "video.mp4",
            fps=10.0,
        )

    process = video_harness.processes[0]
    renderer = video_harness.renderers[0]
    assert process.stdin.closed  # type: ignore[attr-defined]
    assert process.wait_calls == [None]  # type: ignore[attr-defined]
    assert renderer.closed  # type: ignore[attr-defined]


def test_render_video_reports_broken_pipe_and_closes_resources(
    tmp_path: Path, video_harness: _VideoHarness
) -> None:
    video_harness.break_pipe = True
    video_harness.return_code = 1
    video_harness.stderr = b"pipe closed"

    with pytest.raises(RuntimeError, match="status 1: pipe closed"):
        mujoco_rendering.render_video(
            _FakeVideoSim(),
            np.array([0.0, 0.1]),
            np.zeros((2, 4)),
            output_path=tmp_path / "video.mp4",
            fps=10.0,
        )

    process = video_harness.processes[0]
    renderer = video_harness.renderers[0]
    assert process.stdin.closed  # type: ignore[attr-defined]
    assert process.wait_calls == [None]  # type: ignore[attr-defined]
    assert renderer.closed  # type: ignore[attr-defined]


def test_render_video_terminates_process_when_rendering_fails(
    tmp_path: Path, video_harness: _VideoHarness
) -> None:
    video_harness.render_error = RuntimeError("render failed")

    with pytest.raises(RuntimeError, match="render failed"):
        mujoco_rendering.render_video(
            _FakeVideoSim(),
            np.array([0.0, 0.1]),
            np.zeros((2, 4)),
            output_path=tmp_path / "video.mp4",
            fps=10.0,
        )

    process = video_harness.processes[0]
    renderer = video_harness.renderers[0]
    assert process.stdin.closed  # type: ignore[attr-defined]
    assert process.terminated  # type: ignore[attr-defined]
    assert process.wait_calls == [5]  # type: ignore[attr-defined]
    assert renderer.closed  # type: ignore[attr-defined]


def test_main_returns_failure_status_when_fk_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mujoco_cli, "ArmSim", object)
    monkeypatch.setattr(mujoco_cli, "validate_fk", lambda _sim: False)
    monkeypatch.setattr(sys, "argv", ["mujoco_sim.py"])

    assert mujoco_cli.main() == 1


def test_main_default_runs_real_simulation_with_short_trajectory(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    trajectory = MultiPointTrajectory(
        [np.zeros(4), np.array([0.01, -0.02, 0.03, -0.04])],
        [0.004],
        n_points_per_segment=3,
        method="quintic",
    )
    monkeypatch.setattr(mujoco_cli, "demo_trajectory", lambda: trajectory)

    assert mujoco_cli.main([]) == 0

    output = capsys.readouterr().out
    assert "Result: ALL PASS" in output
    assert "TRAJECTORY EXECUTION" in output
    assert "Planned points: 3" in output
    assert "Simulation samples: 3" in output
    assert "Duration: 0.004s (requested 0.004s)" in output
    assert "Mean tracking error:" in output
    assert "Max tracking error:" in output
    assert "Final EE position:" in output


def test_main_video_forwards_demo_trajectory_and_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = mujoco_cli.demo_trajectory()
    calls: list[tuple[np.ndarray, np.ndarray, str | Path]] = []

    def capture_render_video(
        _sim: ArmSim,
        timestamps: object,
        q_trajectory: object,
        output_path: str | Path = "arm_physics_sim.mp4",
        **_kwargs: object,
    ) -> str:
        calls.append(
            (
                np.asarray(timestamps, dtype=np.float64).copy(),
                np.asarray(q_trajectory, dtype=np.float64).copy(),
                output_path,
            )
        )
        return str(output_path)

    monkeypatch.setattr(mujoco_cli, "render_video", capture_render_video)
    output_path = tmp_path / "wired-video.mp4"

    assert mujoco_cli.main(["--video", "--output", str(output_path)]) == 0

    assert len(calls) == 1
    timestamps, q_trajectory, forwarded_output_path = calls[0]
    np.testing.assert_array_equal(timestamps, expected.t)
    np.testing.assert_array_equal(q_trajectory, expected.q)
    assert forwarded_output_path == str(output_path)
