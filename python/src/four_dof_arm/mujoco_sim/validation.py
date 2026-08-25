from __future__ import annotations

import numpy as np

from .core import ArmSim

_VALIDATION_CONFIGURATIONS = {
    "Home": np.array([0.0, 0.0, 0.0, 0.0]),
    "Base 45deg": np.deg2rad([45.0, 0.0, 0.0, 0.0]),
    "Shoulder 30deg": np.deg2rad([0.0, 30.0, 0.0, 0.0]),
    "Elbow 45deg": np.deg2rad([0.0, 0.0, 45.0, 0.0]),
    "Wrist 30deg": np.deg2rad([0.0, 0.0, 0.0, 30.0]),
    "Complex": np.deg2rad([30.0, 45.0, -30.0, 20.0]),
    "All joints": np.deg2rad([20.0, -15.0, 40.0, -25.0]),
}


def validate_fk(sim: ArmSim) -> bool:
    """Compare MuJoCo end-effector poses against analytical FK."""
    print("=" * 70)
    print("FK VALIDATION: four_dof_arm.robot vs MuJoCo kinematics")
    print("=" * 70)
    print(
        f"{'Config':<20} {'Analytical (mm)':<30} {'MuJoCo (mm)':<30} "
        f"{'Pos err (mm)':>12} {'Rot err (deg)':>13}"
    )
    print("-" * 110)

    all_pass = True
    for name, q in _VALIDATION_CONFIGURATIONS.items():
        pos_analytical, rotation_analytical = sim.robot.forward_kinematics(q)
        sim.reset(q)
        pos_mujoco = sim.ee_pos * 1000
        rotation_mujoco = sim.ee_rotation

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
