from __future__ import annotations

import argparse

import numpy as np

from ..trajectory import MultiPointTrajectory
from .core import ArmSim
from .rendering import render_video
from .validation import validate_fk


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
    parser = argparse.ArgumentParser(
        prog="four-dof-arm-mujoco",
        description="4-DOF Arm MuJoCo Simulation",
    )
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
