"""MuJoCo simulation for the 4-DOF arm.

Validates the analytical kinematic model in robot.py against MuJoCo's
kinematic chain. Supports PD control with gravity compensation, timestamped
trajectory execution, and video rendering for headless environments.

Usage:
    python -m four_dof_arm.mujoco_sim       # Package invocation
    four-dof-arm-mujoco --validate-only     # Installed console command
    four-dof-arm-mujoco --video             # Render demo video
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .robot import Robot4DOF
from .trajectory import MultiPointTrajectory

MODEL_PATH = Path(__file__).parent / "mujoco_model.xml"


def _positive_finite_float(value: object, name: str) -> float:
    try:
        scalar = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if scalar.shape != ():
        raise ValueError(f"{name} must be a finite positive number")

    result = float(scalar)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _rounded_up_count(value: float) -> int:
    """Ceil a nonnegative count without promoting integral roundoff."""
    if value == 0.0:
        return 0

    nearest = round(value)
    tolerance = 8.0 * math.ulp(max(1.0, abs(value)))
    if nearest > 0 and abs(value - nearest) <= tolerance:
        return nearest
    return math.ceil(value)


def _step_count(duration: float, timestep: float) -> int:
    return _rounded_up_count(duration / timestep)


def _validate_trajectory(
    timestamps: ArrayLike,
    q_trajectory: ArrayLike,
    n_joints: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    times = np.asarray(timestamps, dtype=np.float64)
    targets = np.asarray(q_trajectory, dtype=np.float64)

    if times.ndim != 1 or times.size < 2:
        raise ValueError(
            "timestamps must be a one-dimensional array with at least two values"
        )
    if not np.all(np.isfinite(times)):
        raise ValueError("timestamps must be finite")
    if times[0] != 0.0:
        raise ValueError("timestamps must start at zero")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    if targets.shape != (times.size, n_joints):
        raise ValueError(
            f"q_trajectory must have shape ({times.size}, {n_joints}), got {targets.shape}"
        )
    if not np.all(np.isfinite(targets)):
        raise ValueError("q_trajectory must be finite")

    return times, targets


def _interpolate_target(
    timestamps: NDArray[np.float64],
    q_trajectory: NDArray[np.float64],
    time: float,
) -> NDArray[np.float64]:
    return np.array(
        [
            np.interp(time, timestamps, q_trajectory[:, joint])
            for joint in range(q_trajectory.shape[1])
        ],
        dtype=np.float64,
    )


class ArmSim:
    """MuJoCo simulation wrapper for the 4-DOF arm.

    Parameters
    ----------
    model_path : str or Path
        Path to mujoco_model.xml.
    """

    def __init__(self, model_path: str | Path = MODEL_PATH) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self._end_effector_site_id = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        )
        if self._end_effector_site_id < 0:
            raise ValueError("model must define a site named 'end_effector'")
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.robot = Robot4DOF()

        self.n_joints = 4
        self.dt = float(self.model.opt.timestep)
        self.Kp = np.array([3.0, 3.0, 2.0, 1.5])
        self.Kd = np.array([0.3, 0.3, 0.2, 0.15])
        self._actuator_ids, self._control_to_joint_torque = (
            self._find_joint_actuator_transmissions()
        )

    def _find_joint_actuator_transmissions(
        self,
    ) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
        if self.model.nu != self.n_joints:
            raise ValueError(
                f"model must define exactly {self.n_joints} controlled actuators"
            )

        actuator_ids = np.full(self.n_joints, -1, dtype=int)
        torque_scales = np.zeros(self.n_joints, dtype=np.float64)

        for actuator_id in range(self.model.nu):
            if self.model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
                raise ValueError(
                    f"actuator {actuator_id} must use a direct joint transmission"
                )
            if self.model.actuator_gaintype[actuator_id] != mujoco.mjtGain.mjGAIN_FIXED:
                raise ValueError(f"actuator {actuator_id} must use a fixed gain")
            if self.model.actuator_biastype[actuator_id] != mujoco.mjtBias.mjBIAS_NONE:
                raise ValueError(f"actuator {actuator_id} must not use actuator bias")
            if (
                self.model.actuator_dyntype[actuator_id] != mujoco.mjtDyn.mjDYN_NONE
                or self.model.actuator_actadr[actuator_id] != -1
            ):
                raise ValueError(
                    f"actuator {actuator_id} must not use activation dynamics"
                )

            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"actuator {actuator_id} must drive a hinge joint")
            dof_id = int(self.model.jnt_dofadr[joint_id])
            if not 0 <= dof_id < self.n_joints:
                raise ValueError(
                    f"actuator {actuator_id} drives an uncontrolled joint DOF"
                )
            if actuator_ids[dof_id] != -1:
                raise ValueError(f"multiple actuators drive joint DOF {dof_id}")

            gear = float(self.model.actuator_gear[actuator_id, 0])
            gain = float(self.model.actuator_gainprm[actuator_id, 0])
            scale = gear * gain
            if not np.isfinite(scale) or scale == 0.0:
                raise ValueError(
                    f"actuator {actuator_id} has an invalid torque transmission"
                )

            actuator_ids[dof_id] = actuator_id
            torque_scales[dof_id] = scale

        if np.any(actuator_ids < 0):
            raise ValueError(
                "each controlled joint must have one joint-transmission actuator"
            )

        return actuator_ids, torque_scales

    def _validate_target(self, target_q: ArrayLike) -> NDArray[np.float64]:
        target = np.asarray(target_q, dtype=np.float64)
        if target.shape != (self.n_joints,):
            raise ValueError(
                f"target_q must have shape ({self.n_joints},), got {target.shape}"
            )
        if not np.all(np.isfinite(target)):
            raise ValueError("target_q must be finite")
        return target

    def reset(self, q0: ArrayLike | None = None) -> None:
        """Reset simulation to the initial configuration."""
        mujoco.mj_resetData(self.model, self.data)
        if q0 is not None:
            self.data.qpos[: self.n_joints] = self._validate_target(q0)
        mujoco.mj_forward(self.model, self.data)

    @property
    def q(self) -> NDArray[np.float64]:
        """Current joint positions."""
        return np.array(self.data.qpos[: self.n_joints], dtype=np.float64, copy=True)

    @property
    def qd(self) -> NDArray[np.float64]:
        """Current joint velocities."""
        return np.array(self.data.qvel[: self.n_joints], dtype=np.float64, copy=True)

    @property
    def ee_pos(self) -> NDArray[np.float64]:
        """End-effector position in meters."""
        return np.array(
            self.data.site_xpos[self._end_effector_site_id],
            dtype=np.float64,
            copy=True,
        )

    def step(self, target_q: ArrayLike) -> None:
        """Advance one model timestep toward a finite four-joint target."""
        target = self._validate_target(target_q)
        q_error = target - self.data.qpos[: self.n_joints]
        qd = self.data.qvel[: self.n_joints]
        desired_joint_torque = (
            self.Kp * q_error - self.Kd * qd + self.data.qfrc_bias[: self.n_joints]
        )
        self.data.ctrl[self._actuator_ids] = (
            desired_joint_torque / self._control_to_joint_torque
        )
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def settle(
        self,
        target_q: ArrayLike,
        duration: float = 2.0,
        tol: float = 1e-3,
    ) -> float:
        """Run simulation until the arm settles at the target or times out."""
        target = self._validate_target(target_q)
        duration_value = _positive_finite_float(duration, "duration")
        tolerance = _positive_finite_float(tol, "tol")
        n_steps = _step_count(duration_value, self.dt)
        for _ in range(n_steps):
            self.step(target)
            if (
                np.linalg.norm(target - self.q) < tolerance
                and np.linalg.norm(self.qd) < tolerance
            ):
                break
        return float(np.linalg.norm(target - self.q))

    def run_trajectory(
        self,
        timestamps: ArrayLike,
        q_trajectory: ArrayLike,
    ) -> dict[str, NDArray[np.float64]]:
        """Execute timestamped joint targets and record every simulation step.

        The returned time vector is relative to the start of this call. It
        contains the initial state at zero and ends within one model timestep
        of the requested trajectory duration.
        """
        times, targets = _validate_trajectory(timestamps, q_trajectory, self.n_joints)
        duration = float(times[-1])
        n_steps = _step_count(duration, self.dt)
        n_samples = n_steps + 1
        record = {
            "t": np.zeros(n_samples),
            "q": np.zeros((n_samples, self.n_joints)),
            "qd": np.zeros((n_samples, self.n_joints)),
            "ee_pos": np.zeros((n_samples, 3)),
            "target_q": np.zeros((n_samples, self.n_joints)),
        }

        start_time = float(self.data.time)
        record["q"][0] = self.q
        record["qd"][0] = self.qd
        record["ee_pos"][0] = self.ee_pos
        record["target_q"][0] = targets[0]

        for index in range(1, n_samples):
            elapsed = float(self.data.time) - start_time
            target = _interpolate_target(times, targets, elapsed)
            self.step(target)

            sample_time = float(self.data.time) - start_time
            record["t"][index] = sample_time
            record["q"][index] = self.q
            record["qd"][index] = self.qd
            record["ee_pos"][index] = self.ee_pos
            record["target_q"][index] = _interpolate_target(times, targets, sample_time)

        return record


def validate_fk(sim: ArmSim) -> bool:
    """Compare MuJoCo end-effector poses against analytical FK."""
    test_configs = {
        "Home": np.array([0.0, 0.0, 0.0, 0.0]),
        "Base 45deg": np.deg2rad([45.0, 0.0, 0.0, 0.0]),
        "Shoulder 30deg": np.deg2rad([0.0, 30.0, 0.0, 0.0]),
        "Elbow 45deg": np.deg2rad([0.0, 0.0, 45.0, 0.0]),
        "Wrist 30deg": np.deg2rad([0.0, 0.0, 0.0, 30.0]),
        "Complex": np.deg2rad([30.0, 45.0, -30.0, 20.0]),
        "All joints": np.deg2rad([20.0, -15.0, 40.0, -25.0]),
    }

    print("=" * 70)
    print("FK VALIDATION: robot.py (analytical) vs MuJoCo kinematics")
    print("=" * 70)
    print(
        f"{'Config':<20} {'Analytical (mm)':<30} {'MuJoCo (mm)':<30} "
        f"{'Pos err (mm)':>12} {'Rot err (deg)':>13}"
    )
    print("-" * 110)

    all_pass = True
    for name, q in test_configs.items():
        pos_analytical, rotation_analytical = sim.robot.forward_kinematics(q)
        sim.reset(q)
        pos_mujoco = sim.ee_pos * 1000
        rotation_mujoco = np.array(
            sim.data.site_xmat[sim._end_effector_site_id],
            dtype=np.float64,
            copy=True,
        ).reshape(3, 3)

        position_error = float(np.linalg.norm(pos_analytical - pos_mujoco))
        relative_rotation = rotation_analytical.T @ rotation_mujoco
        rotation_cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
        rotation_error = float(np.arccos(rotation_cosine))
        status = "OK" if position_error < 1.0 and rotation_error < 1e-6 else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(
            f"  {name:<18} "
            f"({pos_analytical[0]:7.2f}, {pos_analytical[1]:7.2f}, "
            f"{pos_analytical[2]:7.2f})  "
            f"({pos_mujoco[0]:7.2f}, {pos_mujoco[1]:7.2f}, "
            f"{pos_mujoco[2]:7.2f})  "
            f"{position_error:11.3f} {np.rad2deg(rotation_error):13.6f} {status}"
        )

    print()
    print(
        f"Result: {'ALL PASS' if all_pass else 'SOME FAILED'} "
        "(tolerances: 1.0 mm, 0.000057 deg)"
    )
    return all_pass


def render_video(
    sim: ArmSim,
    timestamps: ArrayLike,
    q_trajectory: ArrayLike,
    output_path: str | Path = "arm_physics_sim.mp4",
    fps: float = 30.0,
    width: int = 640,
    height: int = 480,
) -> str:
    """Simulate timestamped targets and encode timestamp-driven video frames."""
    times, targets = _validate_trajectory(timestamps, q_trajectory, sim.n_joints)
    fps_value = _positive_finite_float(fps, "fps")
    for name, value in (("width", width), ("height", height)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(output_path, (str, Path)) or not str(output_path):
        raise ValueError("output_path must be a non-empty path")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise FileNotFoundError("ffmpeg executable was not found on PATH")

    output = Path(output_path)
    duration = float(times[-1])
    frame_count = max(1, _rounded_up_count(duration * fps_value))
    frame_times = np.arange(frame_count, dtype=np.float64) / fps_value
    ffmpeg_cmd = [
        ffmpeg_path,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps_value),
        "-i",
        "-",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "fast",
        "-crf",
        "23",
        str(output),
    ]

    camera = mujoco.MjvCamera()
    camera.azimuth = 135
    camera.elevation = -20
    camera.distance = 0.45
    camera.lookat[:] = [0.08, 0.0, 0.08]

    renderer = mujoco.Renderer(sim.model, height=height, width=width)
    proc = None
    start_time = float(sim.data.time)
    print(f"Rendering {frame_count} frames to {output}...")

    try:
        with tempfile.TemporaryFile() as stderr_file:
            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stderr=stderr_file,
            )
            if proc.stdin is None:
                raise RuntimeError("failed to open FFmpeg input pipe")

            broken_pipe = False
            try:
                steps_taken = 0
                for frame_time in frame_times:
                    frame_step_count = _step_count(float(frame_time), sim.dt)
                    while steps_taken < frame_step_count:
                        elapsed = float(sim.data.time) - start_time
                        sim.step(_interpolate_target(times, targets, elapsed))
                        steps_taken += 1

                    renderer.update_scene(sim.data, camera)
                    proc.stdin.write(renderer.render().tobytes())

                final_step_count = _step_count(duration, sim.dt)
                while steps_taken < final_step_count:
                    elapsed = float(sim.data.time) - start_time
                    sim.step(_interpolate_target(times, targets, elapsed))
                    steps_taken += 1
            except BrokenPipeError:
                broken_pipe = True
            finally:
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    broken_pipe = True

            return_code = proc.wait()
            stderr_file.seek(0)
            ffmpeg_error = stderr_file.read().decode("utf-8", errors="replace").strip()
            if broken_pipe or return_code != 0:
                detail = ffmpeg_error or "no error output"
                raise RuntimeError(f"FFmpeg failed with status {return_code}: {detail}")
    finally:
        if proc is not None and proc.poll() is None:
            if proc.stdin is not None and not proc.stdin.closed:
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        renderer.close()

    size_mb = output.stat().st_size / (1024 * 1024)
    simulated_duration = float(sim.data.time) - start_time
    encoded_duration = frame_count / fps_value
    print(
        f"Done: {output} ({size_mb:.1f} MB, planned {duration:.3f}s, "
        f"simulated {simulated_duration:.3f}s, encoded {encoded_duration:.3f}s)"
    )
    return str(output)


def demo_trajectory() -> MultiPointTrajectory:
    """Create a demo multi-waypoint trajectory."""
    waypoints = [
        np.deg2rad([0.0, 0.0, 0.0, 0.0]),
        np.deg2rad([45.0, 30.0, -30.0, 15.0]),
        np.deg2rad([-30.0, 45.0, 20.0, -10.0]),
        np.deg2rad([60.0, -20.0, 45.0, 30.0]),
        np.deg2rad([0.0, 0.0, 0.0, 0.0]),
    ]
    durations = [2.0, 2.0, 2.0, 2.0]
    return MultiPointTrajectory(
        waypoints, durations, n_points_per_segment=60, method="quintic"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="4-DOF Arm MuJoCo Simulation")
    parser.add_argument("--video", action="store_true", help="Render demo video")
    parser.add_argument(
        "--validate-only", action="store_true", help="FK validation only"
    )
    parser.add_argument(
        "--output", default="arm_physics_sim.mp4", help="Video output path"
    )
    args = parser.parse_args(argv)

    sim = ArmSim()
    if not validate_fk(sim):
        return 1
    if args.validate_only:
        return 0

    print()
    print("=" * 70)
    print("TRAJECTORY EXECUTION")
    print("=" * 70)

    traj = demo_trajectory()
    sim.reset()
    record = sim.run_trajectory(traj.t, traj.q)

    tracking_error = np.linalg.norm(record["q"] - record["target_q"], axis=1)
    print(f"  Planned points: {len(traj.q)}")
    print(f"  Simulation samples: {len(record['t'])}")
    print(f"  Duration: {record['t'][-1]:.3f}s (requested {traj.duration:.3f}s)")
    print(f"  Mean tracking error: {np.mean(tracking_error):.4f} rad")
    print(f"  Max tracking error:  {np.max(tracking_error):.4f} rad")
    print(
        f"  Final EE position:   ({record['ee_pos'][-1][0] * 1000:.1f}, "
        f"{record['ee_pos'][-1][1] * 1000:.1f}, "
        f"{record['ee_pos'][-1][2] * 1000:.1f}) mm"
    )

    if args.video:
        print()
        sim.reset()
        render_video(sim, traj.t, traj.q, output_path=args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
