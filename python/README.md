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

A local wheel path can be used in place of `.`. The two legacy console commands
support `--help` with a core-only installation and report the required extra
when invoked.

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

## Configurable Serial Arms

The separate `four_dof_arm.serial_arm` workflow analyzes fixed-base serial chains
of revolute joints in **SI units**: metres, kilograms, radians, seconds, and N m
for torque. The legacy `Robot4DOF` API below still uses millimetres.

From the repository root, install the development environment and run the
three-link example with dynamic tracking and animation:

```bash
uv sync --project python --all-extras
uv run --project python serial-arm-analyze python/examples/serial_arm/spatial_3link.json \
  --scenario python/examples/serial_arm/reach_scenario.json \
  --output-dir python/generated/serial-3link --simulate --animate
```

For an installed package, use `serial-arm-analyze` directly with your JSON paths
and install extras as described under [Install](#install).
Without `--simulate`, `--plots`, or `--animate`, the CLI requires only NumPy,
including MJCF export. `--simulate` needs the `simulation` extra; `--plots` and
`--animate` need `visualization`. `--plots` writes only the overview PNG as its
visual output; `--animate` writes both the overview and GIF. Neither FFmpeg nor
OpenGL is needed for this workflow. The generated `model.xml` can also be
inspected separately in a MuJoCo viewer.

`--output-dir` is required and must be **new or empty**; existing files are not
overwritten. Outputs are:

| File | Contents |
|---|---|
| `robot.json` | Normalized configuration with defaults and unit-length axes |
| `model.xml` | Primitive-geometry MuJoCo model (MJCF) |
| `report.json` | Scenario, sampling, diagnostics, metrics, and limitations |
| `trajectory.csv` | Time, joint/tool motion and references; torques when simulated |
| `workspace.csv` | Sampled tool positions in metres |
| `overview.png`, `motion.gif` | Optional overview and animation, respectively |

Joint CSV headers use `joint_name.field_unit`, for example `shoulder.q_rad`.

### Robot Configuration

Root keys are `units`, `name`, `base`, `tool`, `payload_mass`, and `links`.
`units` is required and must be exactly `"SI"`; `links` must be nonempty.
Optional defaults are `name="serial_arm"`, `base=[0,0,0]`, `tool=[0,0,0]`, and
`payload_mass=0`. A minimal two-link configuration is:

```json
{
  "units": "SI",
  "name": "two_link",
  "links": [
    {"name": "shoulder", "axis": [0, 0, 1], "offset": [0.3, 0, 0], "mass": 0.6},
    {"name": "elbow", "axis": [0, 0, 1], "offset": [0.25, 0, 0], "mass": 0.4}
  ]
}
```

Each link requires `name`, `axis`, `offset`, and positive `mass`. Names must be
unique identifiers; nonzero axes are normalized. Optional link defaults are
`radius=0.015`, `limits=[-pi, pi]` (numeric radians in JSON), `damping=0.05`,
`armature=0.01`, `torque_limit=10`, `kp=20`, and `kd=2`.

Frames first translate by `base`, then for each link rotate about its local
`axis` by the joint angle and translate by that link's `offset` in the rotated
frame. Each subsequent hinge is at the preceding link's tip. `tool` adds a
massless offset in the final frame; positive `payload_mass` adds a sphere at the
tool tip. Zero-length links are allowed and use spheres instead of capsules.

[Example configurations](examples/serial_arm/) include planar two-link, spatial
three- and six-link arms, and `original_4dof.json`. The original preset matches
legacy FK and the position Jacobian after millimetre-to-metre conversion, **not
the legacy physical model**.

```python
from four_dof_arm.serial_arm import SerialArm
from four_dof_arm.serial_arm.analysis import analyze

arm = SerialArm.from_json("python/examples/serial_arm/spatial_3link.json")
report, trajectory, workspace = analyze(arm)
```

### Scenarios and Limits

Scenario JSON accepts `duration` (seconds), `waypoints` (at least two joint-angle
vectors in radians, within limits), and `targets` (Cartesian positions in metres).
**Targets are independent position-only IK diagnostics, not executed goals.**
If waypoints are omitted, motion goes from home (zero clipped to joint limits)
to home +0.35 rad, home -0.35 rad, and back home, with all poses clipped to limits.
Duration defaults to 6 seconds. Quintic segments have equal durations and stop
at every waypoint with zero velocity and acceleration. `--timestep` (default
0.002 s) is a maximum: a uniform step is adjusted to hit the duration exactly,
including the terminal sample.

Reports include joint-limit workspace sampling, IK success/residuals, start/end
reference position-Jacobian ranks, and sampled path length, speeds,
accelerations, and joint-limit violations. `--simulate` adds MuJoCo PD tracking
with bias compensation, requested/applied torque, tracking errors, and torque
saturation fractions. Workspace samples are not a certified reachable volume;
sampled extrema may miss peaks, and IK failure does not prove unreachability.

All contacts are disabled: there is **no collision or clearance analysis**.
Speed and acceleration limits are not enforced. Primitive mass/inertia models,
gains, damping, and motor limits are approximate; this is neither structural
stress analysis nor hardware safety validation. Prismatic joints, branched
mechanisms, and closed-loop mechanisms are not supported yet.

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
