# Python Implementation - 4-DOF Robotic Arm

The `four_dof_arm` package provides NumPy-based kinematics and trajectory
planning, with optional Matplotlib visualization and MuJoCo simulation.

FK is cross-validated; MuJoCo dynamics use simplified inertial parameters and
are not validated against CAD/Simscape dynamics.

## Install

The core package only requires NumPy. From an unpacked source distribution,
install it with:

```bash
python -m pip install .
```

Install the extra for each optional command or API:

| Extra | Contents | Package install |
|---|---|---|
| `visualization` | Matplotlib plots and animation | `python -m pip install ".[visualization]"` |
| `simulation` | MuJoCo validation, dynamics, and rendering | `python -m pip install ".[simulation]"` |

A local wheel path can be used in place of `.`. Both console commands support
`--help` with a core-only installation and report the required extra when
invoked.

## Run

The demo writes nothing unless `--output-dir` is supplied:

```bash
four-dof-arm-demo
four-dof-arm-demo --output-dir generated
```

Run MuJoCo FK validation by itself or execute the controlled trajectory:

```bash
four-dof-arm-mujoco --validate-only
four-dof-arm-mujoco
```

Video rendering additionally requires FFmpeg. `--output` selects a destination,
but rendering only occurs when `--video` is present. On a headless Linux host,
select MuJoCo's EGL backend explicitly:

```bash
MUJOCO_GL=egl four-dof-arm-mujoco --video --output arm_physics_sim.mp4
```

The committed reference render is
[`simulations/arm_physics_sim.mp4`](https://github.com/batuhantoker/4dof-robotic-arm/blob/main/simulations/arm_physics_sim.mp4).
CI validates FK and smoke-checks both commands; it does not render video.

## Core API

Importing `four_dof_arm` loads only NumPy-based functionality. It does not
import Matplotlib or MuJoCo.

```python
import numpy as np

from four_dof_arm import JointTrajectory, Robot4DOF

robot = Robot4DOF(Lax=50, Laz=55, Lb=100, Lt=90)
q = np.deg2rad([30, 45, -30, 20])
position, rotation = robot.forward_kinematics(q)

q_solution, success, error = robot.inverse_kinematics_multi(position)
trajectory = JointTrajectory(
    q_start=np.zeros(4),
    q_end=np.deg2rad([45, 30, -60, 20]),
    duration=3.0,
    method="quintic",
)
```

Visualization APIs remain in the optional module:

```python
from four_dof_arm.visualize import animate_trajectory, plot_arm, plot_workspace

plot_arm(robot, q, show=True)
plot_workspace(robot, n_samples=10_000, show=True)
animate_trajectory(robot, trajectory.q, show=True)
```

## Development

The following contributor commands require a repository checkout and must be
run from the repository root because they refer to its `python/` path:

```bash
uv sync --project python --all-extras
uv run --project python ruff check python
uv run --project python ruff format --check python
uv run --project python mypy --config-file python/pyproject.toml python/src/four_dof_arm
MPLBACKEND=Agg uv run --project python pytest --cov=four_dof_arm \
  --cov-report=term-missing python/tests
uv run --project python python -m build python
```

Repository-generated plots and videos should remain under the ignored
`python/generated/` directory unless intentionally curated.

## Joint Configuration

| Joint | Axis | Rotation | AUTOLEV |
|---|---|---|---|
| q1 | Z | Base yaw | `Simprot(N,A,3,q1)` |
| q2 | X | Shoulder pitch | `Simprot(A,B,1,q2)` |
| q3 | Z | Elbow yaw | `Simprot(B,C,3,q3)` |
| q4 | X | Wrist pitch | `Simprot(C,D,1,q4)` |

## Link Parameters

| Parameter | Description | Default (mm) |
|---|---|---|
| `Laz` | Base height | 55 |
| `Lax` | Shoulder offset | 50 |
| `Lb` | Upper arm length | 100 |
| `Lt` | Forearm to gripper | 90 |
