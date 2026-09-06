from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import ArrayLike

from .core import ArmSim
from .timing import (
    _frame_count,
    _interpolate_target,
    _positive_finite_float,
    _step_count,
    _validate_trajectory,
)


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
            or value % 2 != 0
        ):
            raise ValueError(f"{name} must be a positive even integer")
    if not isinstance(output_path, (str, Path)) or not str(output_path):
        raise ValueError("output_path must be a non-empty path")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise FileNotFoundError("ffmpeg executable was not found on PATH")

    output = Path(output_path)
    duration = float(times[-1])
    frame_count = _frame_count(duration, fps_value)
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

    sim.model.vis.global_.offwidth = max(sim.model.vis.global_.offwidth, width)
    sim.model.vis.global_.offheight = max(sim.model.vis.global_.offheight, height)
    proc = None
    renderer = mujoco.Renderer(sim.model, height=height, width=width)
    try:
        start_time = float(sim.data.time)
        print(f"Rendering {frame_count} frames to {output}...")
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
        try:
            if proc is not None:
                # Stop the reader before close can flush a blocked, buffered pipe.
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
                if proc.stdin is not None and not proc.stdin.closed:
                    try:
                        proc.stdin.close()
                    except BrokenPipeError:
                        pass
        finally:
            renderer.close()

    size_mb = output.stat().st_size / (1024 * 1024)
    simulated_duration = float(sim.data.time) - start_time
    encoded_duration = frame_count / fps_value
    print(
        f"Done: {output} ({size_mb:.1f} MB, planned {duration:.3f}s, "
        f"simulated {simulated_duration:.3f}s, encoded {encoded_duration:.3f}s)"
    )
    return str(output)
