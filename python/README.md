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
The public MuJoCo API, focused implementation modules, and packaged MJCF model
live under [`src/four_dof_arm/mujoco_sim/`](src/four_dof_arm/mujoco_sim/).

The model excludes contact between the anchored column and its mounting surfaces.
Shortened collision capsules leave clearance around the shared wrist pivot while
preserving folded upper-arm/forearm collisions; visual geometry and inertias are
unchanged. Ground and other nonadjacent-link collisions remain enabled. This
does not make planned trajectories collision-free.
The committed reference video has not been regenerated with these exclusions.

`ArmSim.dt` reads and updates `sim.model.opt.timestep` in seconds. Change it
between simulation calls, not during trajectory execution or rendering.
`reset(q0)` validates and copies its input before clearing simulation state.
The `render_video` API accepts positive even pixel dimensions, including HD
sizes, and grows the model's offscreen framebuffer as needed.

## CAD Mesh Preview

From a repository checkout, convert all five STEP parts and render the original
CAD assembly with:

```bash
uv run --script python/scripts/preview_cad.py
```

The script installs its conversion/rendering dependencies in an isolated uv
environment; it does not add OpenCascade to the core package. It writes binary
STL meshes, `cad_preview.xml`, `cad_preview.png`, and a looping `cad_preview.gif`
to `python/generated/cad/`. GIF output does not require FFmpeg. Use
`--output-dir PATH` to choose another destination or `--no-render` to export
only the meshes and MJCF. Existing preview outputs in that directory are replaced.
On headless Linux, set `MUJOCO_GL=egl` when rendering.

This is a **separate kinematic preview**, not a CAD skin on the analytical model.
The assembly export has parallel shoulder/elbow/wrist axes and different link
dimensions from the analytical Z-X-Z-X chain. Part placements and hinge axes come
from `matlab/Montaj1.xml`; STEP coordinates are converted from millimetres to metres.
Zero joint displacement reproduces the exported assembly pose. The animation
applies illustrative joint displacements around that pose using forward kinematics,
not a physics controller. Collisions are disabled and inertias are placeholders;
the generated model must not be used to evaluate dynamics or physical clearance.
The existing `ArmSim`, analytical model, and collision geometry remain unchanged.

Run a larger set of measured CAD motions with:

```bash
uv run --script python/scripts/preview_cad.py --suite
```

This produces six scenarios under `python/generated/cad/runs/`: base yaw
(+/-60 degrees), shoulder (+/-20), elbow (+/-30), wrist (+/-45), coordinated
motion, and a multi-pose transfer-and-return. All angles are offsets from the
exported assembly pose, not absolute joint angles or validated joint limits.
Quintic joint profiles stop at each waypoint with zero velocity and acceleration.

Each scenario has a GIF with a blue tool-tip trail, a representative PNG, a CSV
with time/joint offsets/velocities/accelerations/tool position/speed, and JSON
measurements. `overview.png` compares the six cases; `summary.json` collects the
results. Tool position uses the pointed STEP vertex at `(-2.75, 0, 0)` mm in
link4; speed uses the CAD model's site Jacobian. Path length and reported extrema
are sampled at 20 Hz, and tool height is not a collision-clearance measurement.
CSV data includes the terminal sample; GIFs omit that repeated pose so their
playback duration matches the prescribed motion duration.
The durations describe prescribed motion, not physical tracking performance.

## Core API

Importing `four_dof_arm` loads only NumPy-based functionality. It does not
import Matplotlib or MuJoCo.

The `robot`, `trajectory`, `visualize`, and `mujoco_sim` directories expose
stable APIs through their `__init__.py` files while keeping implementation
details in focused modules.

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

### Planning Contracts

- Analytical positions and link lengths use millimetres; MuJoCo positions use
  metres. Joint angles use radians and time uses seconds.
- IK solves position only, not a full end-effector pose. Check its success flag
  and residual before using a returned configuration.
- `CartesianTrajectory(..., max_joint_step=0.5)` limits every joint's change
  relative to the last accepted waypoint. The first accepted configuration is
  not constrained to be near `q_seed`, which is only an initial IK guess.
- A Cartesian IK failure or excessive joint step sets `ik_success=False` for
  that waypoint and `success=False` for the trajectory. Rejected configurations
  remain in `q` for diagnostics; do not execute a failed trajectory. There is no
  automatic retry or subdivision, and the step bound is not a velocity limit.
- Joint and multi-point trajectories use zero-velocity endpoints. Multi-point
  trajectories stop at every via-point; quintic segments also have zero endpoint
  acceleration. Unrepresentable polynomial outputs raise `ValueError`.
- Planning does not enforce actuator velocity/acceleration limits or certify
  collision-free motion between waypoints. Simulation is not hardware validation.

## Development

The following contributor commands require a repository checkout and must be
run from the repository root because they refer to its `python/` path:

```bash
uv sync --project python --all-extras
uv run --project python ruff check python
uv run --project python ruff format --check python
uv run --project python mypy --config-file python/pyproject.toml python/src/four_dof_arm
MPLBACKEND=Agg uv run --project python pytest --cov=four_dof_arm \
  --cov-config=python/pyproject.toml --cov-report=term-missing python/tests
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
