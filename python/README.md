# Python Implementation — 4-DOF Robotic Arm

Modern Python port of the kinematics engine, with trajectory planning and 3D visualization.

## Quick Start

```bash
cd python
pip install -r requirements.txt
python demo.py
```

## Modules

### `robot.py` — Kinematics Engine
- `Robot4DOF` class with configurable link lengths
- Forward kinematics (closed-form, matching AUTOLEV formulation)
- Analytical Jacobian (derived from Kane's method zero-config equations)
- Inverse kinematics via damped least-squares with multi-restart
- Workspace sampling

```python
from robot import Robot4DOF
import numpy as np

robot = Robot4DOF(Lax=50, Laz=55, Lb=100, Lt=90)

# Forward kinematics
q = np.deg2rad([30, 45, -30, 20])
pos, R = robot.forward_kinematics(q)
print(f"End-effector: {pos}")

# Inverse kinematics
q_sol, ok, err = robot.inverse_kinematics_multi(pos)
print(f"IK solution: {np.rad2deg(q_sol)}, error: {err:.4f} mm")
```

### `trajectory.py` — Trajectory Planning
- Cubic and quintic polynomial joint interpolation (zero velocity/acceleration BCs)
- Cartesian linear interpolation with IK
- Multi-point via-point trajectories

```python
from trajectory import JointTrajectory
import numpy as np

traj = JointTrajectory(
    q_start=np.zeros(4),
    q_end=np.deg2rad([45, 30, -60, 20]),
    duration=3.0,
    method="quintic",
)
# traj.t, traj.q, traj.qd, traj.qdd available
```

### `visualize.py` — 3D Visualization
- Static arm pose rendering
- Animated trajectory playback with end-effector trail
- Joint angle/velocity profile plots
- Workspace point cloud visualization

```python
from visualize import plot_arm, animate_trajectory, plot_workspace

plot_arm(robot, q)
plot_workspace(robot, n_samples=10000)
animate_trajectory(robot, traj.q)
```

## Joint Configuration

| Joint | Axis | Rotation | AUTOLEV |
|-------|------|----------|---------|
| q1 | Z | Base yaw | `Simprot(N,A,3,q1)` |
| q2 | X | Shoulder pitch | `Simprot(A,B,1,q2)` |
| q3 | Z | Elbow yaw | `Simprot(B,C,3,q3)` |
| q4 | X | Wrist pitch | `Simprot(C,D,1,q4)` |

## Link Parameters

| Parameter | Description | Default (mm) |
|-----------|-------------|-------------|
| `Laz` | Base height | 55 |
| `Lax` | Shoulder offset | 50 |
| `Lb` | Upper arm length | 100 |
| `Lt` | Forearm to gripper | 90 |
