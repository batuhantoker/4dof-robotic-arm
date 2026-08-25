"""
Demo: 4-DOF Robotic Arm — Forward/Inverse Kinematics & Trajectory Planning

Demonstrates the full pipeline:
1. Forward kinematics for sample configurations
2. Inverse kinematics to reach a target point
3. Multi-point trajectory generation and animation
4. Workspace visualization
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .robot import Robot4DOF
from .trajectory import JointTrajectory, MultiPointTrajectory
from .visualize import animate_trajectory, plot_arm, plot_joint_angles, plot_workspace

_WORKSPACE_SAMPLES = 15_000


def main(
    *,
    show: bool = False,
    output_dir: str | Path | None = None,
    seed: int = 0,
) -> None:
    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # --- Robot Setup ---
    # Link lengths from CAD/SimScape model (mm)
    robot = Robot4DOF(Lax=50.0, Laz=55.0, Lb=100.0, Lt=90.0)
    print(f"Robot: {robot}")
    print(f"Total reach: {robot.Lax + robot.Lb + robot.Lt:.0f} mm (horizontal)")
    print()

    # --- 1. Forward Kinematics ---
    print("=" * 60)
    print("FORWARD KINEMATICS")
    print("=" * 60)

    test_configs = {
        "Home (all zeros)": np.array([0.0, 0.0, 0.0, 0.0]),
        "Base rotated 45°": np.deg2rad([45.0, 0.0, 0.0, 0.0]),
        "Shoulder up 30°": np.deg2rad([0.0, 30.0, 0.0, 0.0]),
        "Elbow bent 45°": np.deg2rad([0.0, 0.0, 45.0, 0.0]),
        "Complex pose": np.deg2rad([30.0, 45.0, -30.0, 20.0]),
    }

    for name, q in test_configs.items():
        pos, _ = robot.forward_kinematics(q)
        print(f"  {name}:")
        print(f"    q = [{', '.join(f'{np.degrees(a):6.1f}°' for a in q)}]")
        print(f"    pos = ({pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:7.2f}) mm")
        print()

    # --- 2. Jacobian Verification ---
    print("=" * 60)
    print("JACOBIAN VERIFICATION (analytical vs numerical)")
    print("=" * 60)

    q_test = np.deg2rad([30.0, 45.0, -30.0, 20.0])
    J_analytical = robot.jacobian_analytical(q_test)
    J_numerical = robot.jacobian(q_test)
    max_diff = np.max(np.abs(J_analytical - J_numerical))
    print(f"  Max difference: {max_diff:.2e} (should be < 1e-4)")
    print(f"  Status: {'PASS' if max_diff < 1e-4 else 'FAIL'}")
    print()

    # --- 3. Inverse Kinematics ---
    print("=" * 60)
    print("INVERSE KINEMATICS")
    print("=" * 60)

    # Test: reach a known point
    q_known = np.deg2rad([20.0, 30.0, -15.0, 10.0])
    target_pos, _ = robot.forward_kinematics(q_known)
    print(
        f"  Target: ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}) mm"
    )
    print(f"  (from q = [{', '.join(f'{np.degrees(a):.1f}°' for a in q_known)}])")

    q_sol, success, error = robot.inverse_kinematics_multi(target_pos, rng=rng)
    print(f"  Solution: [{', '.join(f'{np.degrees(a):6.1f}°' for a in q_sol)}]")
    print(f"  Error: {error:.4f} mm | Success: {success}")

    # Verify solution
    pos_check, _ = robot.forward_kinematics(q_sol)
    print(f"  Verify: ({pos_check[0]:.2f}, {pos_check[1]:.2f}, {pos_check[2]:.2f}) mm")
    print()

    # --- 4. Static Arm Plot ---
    print("Plotting arm in complex pose...")
    q_demo = np.deg2rad([30.0, 45.0, -30.0, 20.0])
    arm_path = output_path / "arm_pose.png" if output_path is not None else None
    fig = plot_arm(
        robot,
        q_demo,
        title="4-DOF Arm — Complex Pose",
        show=show,
        save_path=arm_path,
    )
    if arm_path is not None:
        print(f"  Saved: {arm_path}")
    plt.close(fig)

    # --- 5. Joint Trajectory ---
    print("\nGenerating quintic trajectory...")
    q_start = np.deg2rad([0.0, 0.0, 0.0, 0.0])
    q_end = np.deg2rad([45.0, 30.0, -60.0, 20.0])
    traj = JointTrajectory(q_start, q_end, duration=3.0, n_points=150, method="quintic")

    joint_path = (
        output_path / "joint_trajectory.png" if output_path is not None else None
    )
    fig = plot_joint_angles(
        traj.t,
        traj.q,
        traj.qd,
        title="Quintic Joint Trajectory",
        save_path=joint_path,
        show=show,
    )
    if joint_path is not None:
        print(f"  Saved: {joint_path}")
    plt.close(fig)

    # --- 6. Multi-point Trajectory Animation ---
    print("\nGenerating multi-point trajectory...")
    waypoints = [
        np.deg2rad([0.0, 0.0, 0.0, 0.0]),
        np.deg2rad([45.0, 30.0, -30.0, 15.0]),
        np.deg2rad([-30.0, 45.0, 20.0, -10.0]),
        np.deg2rad([60.0, -20.0, 45.0, 30.0]),
        np.deg2rad([0.0, 0.0, 0.0, 0.0]),
    ]
    durations = [1.5, 1.5, 1.5, 1.5]
    multi_traj = MultiPointTrajectory(waypoints, durations, n_points_per_segment=60)

    print(f"  Total duration: {multi_traj.duration:.1f}s, {multi_traj.n_points} points")
    if show:
        print("  Animating... (close window to continue)")
        animate_trajectory(robot, multi_traj.q, interval=30, trail=True, show=True)
        plt.close()
    else:
        print("  Animation skipped (use --show to display it)")

    # --- 7. Workspace ---
    print("\nPlotting workspace...")
    workspace_path = output_path / "workspace.png" if output_path is not None else None
    fig = plot_workspace(
        robot,
        n_samples=_WORKSPACE_SAMPLES,
        save_path=workspace_path,
        show=show,
        rng=rng,
    )
    if workspace_path is not None:
        print(f"  Saved: {workspace_path}")
    plt.close(fig)

    print("\nDone.")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--show",
        action="store_true",
        help="display plots and animation interactively",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for PNG outputs; no files are written when omitted",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for IK and workspace sampling (default: 0)",
    )
    return parser.parse_args(argv)


def cli(argv: list[str] | None = None) -> int:
    """Run the command-line demo."""
    args = _parse_args(argv)
    main(show=args.show, output_dir=args.output_dir, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
