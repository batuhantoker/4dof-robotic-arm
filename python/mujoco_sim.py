#!/usr/bin/env python3
"""
MuJoCo Physics Simulation for 4-DOF Arm

Validates the kinematic model (robot.py) against MuJoCo physics.
Supports PD control with gravity compensation, trajectory execution,
and video rendering for headless environments.

Usage:
    python mujoco_sim.py                    # Run validation + demo trajectory
    python mujoco_sim.py --video            # Render demo video
    python mujoco_sim.py --validate-only    # FK validation only
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import mujoco

from robot import Robot4DOF
from trajectory import JointTrajectory, MultiPointTrajectory


# Resolve model path relative to this file
MODEL_PATH = Path(__file__).parent / "mujoco_model.xml"


class ArmSim:
    """MuJoCo simulation wrapper for the 4-DOF arm.
    
    Parameters
    ----------
    model_path : str or Path
        Path to mujoco_model.xml.
    """

    def __init__(self, model_path: str | Path = MODEL_PATH) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.robot = Robot4DOF()  # Kinematic model for comparison
        
        self.n_joints = 4
        self.joint_names = ["q1", "q2", "q3", "q4"]
        self.dt = self.model.opt.timestep
        
        # PD gains (tuned for small arm with gravity comp)
        self.Kp = np.array([3.0, 3.0, 2.0, 1.5])
        self.Kd = np.array([0.3, 0.3, 0.2, 0.15])

    def reset(self, q0: np.ndarray | None = None) -> None:
        """Reset simulation to initial configuration."""
        mujoco.mj_resetData(self.model, self.data)
        if q0 is not None:
            self.data.qpos[:self.n_joints] = q0
        mujoco.mj_forward(self.model, self.data)

    @property
    def q(self) -> np.ndarray:
        """Current joint positions."""
        return self.data.qpos[:self.n_joints].copy()

    @property 
    def qd(self) -> np.ndarray:
        """Current joint velocities."""
        return self.data.qvel[:self.n_joints].copy()

    @property
    def ee_pos(self) -> np.ndarray:
        """End-effector position in meters."""
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        return self.data.site_xpos[ee_id].copy()

    def step(self, target_q: np.ndarray) -> None:
        """Step simulation with PD + gravity compensation.
        
        Parameters
        ----------
        target_q : array of shape (4,)
            Target joint angles in radians.
        """
        q_error = target_q - self.data.qpos[:self.n_joints]
        qd = self.data.qvel[:self.n_joints]
        gravity_comp = self.data.qfrc_bias[:self.n_joints]
        
        self.data.ctrl[:self.n_joints] = (
            self.Kp * q_error - self.Kd * qd + gravity_comp
        )
        mujoco.mj_step(self.model, self.data)

    def settle(self, target_q: np.ndarray, duration: float = 2.0, tol: float = 1e-3) -> float:
        """Run simulation until arm settles at target or timeout.
        
        Returns
        -------
        error : float
            Final joint angle error norm.
        """
        n_steps = int(duration / self.dt)
        for _ in range(n_steps):
            self.step(target_q)
            if np.linalg.norm(target_q - self.q) < tol and np.linalg.norm(self.qd) < tol:
                break
        return float(np.linalg.norm(target_q - self.q))

    def run_trajectory(
        self, q_trajectory: np.ndarray, steps_per_point: int = 10
    ) -> dict:
        """Execute a joint trajectory and record data.
        
        Parameters
        ----------
        q_trajectory : array of shape (N, 4)
            Joint angle waypoints.
        steps_per_point : int
            Simulation steps per trajectory point.
            
        Returns
        -------
        dict with keys: t, q, qd, ee_pos, target_q
        """
        n_points = len(q_trajectory)
        record = {
            "t": np.zeros(n_points),
            "q": np.zeros((n_points, 4)),
            "qd": np.zeros((n_points, 4)),
            "ee_pos": np.zeros((n_points, 3)),
            "target_q": q_trajectory.copy(),
        }
        
        for i, target in enumerate(q_trajectory):
            for _ in range(steps_per_point):
                self.step(target)
            
            record["t"][i] = self.data.time
            record["q"][i] = self.q
            record["qd"][i] = self.qd
            record["ee_pos"][i] = self.ee_pos
        
        return record


def validate_fk(sim: ArmSim) -> bool:
    """Compare MuJoCo end-effector positions against analytical FK.
    
    Tests multiple configurations and reports discrepancies.
    """
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
    print("FK VALIDATION: robot.py (analytical) vs MuJoCo (physics)")
    print("=" * 70)
    print(f"{'Config':<20} {'Analytical (mm)':<30} {'MuJoCo (mm)':<30} {'Err (mm)':>8}")
    print("-" * 90)
    
    all_pass = True
    for name, q in test_configs.items():
        # Analytical FK (returns mm)
        pos_analytical, _ = sim.robot.forward_kinematics(q)
        
        # MuJoCo FK (set joints directly, no dynamics)
        sim.reset(q)
        pos_mujoco = sim.ee_pos * 1000  # m → mm
        
        error = np.linalg.norm(pos_analytical - pos_mujoco)
        status = "OK" if error < 1.0 else "FAIL"
        if error >= 1.0:
            all_pass = False
        
        print(
            f"  {name:<18} "
            f"({pos_analytical[0]:7.2f}, {pos_analytical[1]:7.2f}, {pos_analytical[2]:7.2f})  "
            f"({pos_mujoco[0]:7.2f}, {pos_mujoco[1]:7.2f}, {pos_mujoco[2]:7.2f})  "
            f"{error:7.3f} {status}"
        )
    
    print()
    print(f"Result: {'ALL PASS' if all_pass else 'SOME FAILED'} (tolerance: 1.0 mm)")
    return all_pass


def render_video(
    sim: ArmSim,
    q_trajectory: np.ndarray,
    output_path: str = "arm_physics_sim.mp4",
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    steps_per_frame: int = 10,
) -> str:
    """Render trajectory to MP4 using offscreen MuJoCo renderer.
    
    Parameters
    ----------
    sim : ArmSim
    q_trajectory : array of shape (N, 4)
    output_path : str
    fps : int
    
    Returns
    -------
    path : str
        Output video file path.
    """
    renderer = mujoco.Renderer(sim.model, height=height, width=width)
    
    camera = mujoco.MjvCamera()
    camera.azimuth = 135
    camera.elevation = -20
    camera.distance = 0.45
    camera.lookat[:] = [0.08, 0.0, 0.08]
    
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "23",
        output_path,
    ]
    
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    
    print(f"Rendering {len(q_trajectory)} frames to {output_path}...")
    
    for i, target in enumerate(q_trajectory):
        for _ in range(steps_per_frame):
            sim.step(target)
        
        renderer.update_scene(sim.data, camera)
        pixels = renderer.render()
        proc.stdin.write(pixels.tobytes())
    
    proc.stdin.close()
    proc.wait()
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    duration = len(q_trajectory) / fps
    print(f"Done: {output_path} ({size_mb:.1f} MB, {duration:.1f}s)")
    return output_path


def demo_trajectory() -> MultiPointTrajectory:
    """Create a demo multi-waypoint trajectory."""
    waypoints = [
        np.deg2rad([0.0, 0.0, 0.0, 0.0]),        # home
        np.deg2rad([45.0, 30.0, -30.0, 15.0]),    # reach right
        np.deg2rad([-30.0, 45.0, 20.0, -10.0]),   # reach left-up
        np.deg2rad([60.0, -20.0, 45.0, 30.0]),    # sweep
        np.deg2rad([0.0, 0.0, 0.0, 0.0]),         # home
    ]
    durations = [2.0, 2.0, 2.0, 2.0]
    return MultiPointTrajectory(
        waypoints, durations, n_points_per_segment=60, method="quintic"
    )


def main():
    parser = argparse.ArgumentParser(description="4-DOF Arm MuJoCo Simulation")
    parser.add_argument("--video", action="store_true", help="Render demo video")
    parser.add_argument("--validate-only", action="store_true", help="FK validation only")
    parser.add_argument("--output", default="arm_physics_sim.mp4", help="Video output path")
    args = parser.parse_args()
    
    sim = ArmSim()
    
    # Always validate FK
    fk_ok = validate_fk(sim)
    
    if args.validate_only:
        return
    
    # Run trajectory
    print()
    print("=" * 70)
    print("TRAJECTORY EXECUTION")
    print("=" * 70)
    
    traj = demo_trajectory()
    sim.reset()
    record = sim.run_trajectory(traj.q)
    
    # Compare trajectory tracking
    tracking_error = np.linalg.norm(record["q"] - record["target_q"], axis=1)
    print(f"  Points: {len(traj.q)}")
    print(f"  Duration: {traj.duration:.1f}s")
    print(f"  Mean tracking error: {np.mean(tracking_error):.4f} rad")
    print(f"  Max tracking error:  {np.max(tracking_error):.4f} rad")
    print(f"  Final EE position:   ({record['ee_pos'][-1][0]*1000:.1f}, "
          f"{record['ee_pos'][-1][1]*1000:.1f}, {record['ee_pos'][-1][2]*1000:.1f}) mm")
    
    # Render video if requested
    if args.video:
        print()
        sim.reset()
        render_video(sim, traj.q, output_path=args.output)


if __name__ == "__main__":
    main()
