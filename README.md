# 4-DOF Robotic Arm - Kinematics and Simulation

![MATLAB](https://img.shields.io/badge/MATLAB-R2022a-blue?logo=mathworks)
![Simulink](https://img.shields.io/badge/Simulink-SimScape-orange?logo=mathworks)
![MuJoCo](https://img.shields.io/badge/MuJoCo-simulation-purple)
![Python](https://img.shields.io/badge/Python-3.10+-green?logo=python)
![SolidWorks](https://img.shields.io/badge/SolidWorks-CAD-red)

End-to-end study of a 4-DOF serial manipulator: symbolic kinematics (Kane's
method), Simscape multibody dynamics, a Python kinematics and trajectory
package, and a MuJoCo model with PD control and gravity compensation.

**FK cross-validation: 0.000 mm error** across the documented test
configurations: the analytical implementation matches the MuJoCo kinematic
chain.

FK is cross-validated; MuJoCo dynamics use simplified inertial parameters and
are not validated against CAD/Simscape dynamics.

![1_uHBulGzv38jcE_R2PtPH4Q](https://user-images.githubusercontent.com/55883119/210860480-249bc993-9c25-4c12-96b2-f342db412f1f.gif)

## Architecture

The manipulator is a 4-DOF serial robot arm with the following joint configuration:

| Joint | Axis | Motion | Description |
|-------|------|--------|-------------|
| q1 | Z | Rotation | Base yaw |
| q2 | X | Rotation | Shoulder pitch |
| q3 | Z | Rotation | Elbow yaw |
| q4 | X | Rotation | Wrist pitch |

The alternating Z-X-Z-X joint pattern gives the arm a good balance of workspace coverage for a 4-DOF system. Link parameters (`Laz`, `Lax`, `Lb`, `Lt`) define the base height, shoulder offset, upper arm length, and forearm-to-gripper length respectively.

## Kinematics — Kane's Method

The kinematic formulation uses **Kane's method** via AUTOLEV, a symbolic multibody dynamics tool. The approach defines successive body-fixed reference frames connected by simple rotations:

```
N → A: Simprot(Z, q1)   Base rotation
A → B: Simprot(X, q2)   Shoulder rotation
B → C: Simprot(Z, q3)   Elbow rotation
C → D: Simprot(X, q4)   Wrist rotation
```

The end-effector position is derived from the vector loop equation `P_O_G = Laz·A3 + Lax·A1 + Lb·B1 + Lt·D3`, yielding closed-form expressions for forward kinematics:

```
x = Lax·cos(q1) + Lb·cos(q1) + Lt·(sin(q3)·sin(q4)·cos(q1) + sin(q1)·(sin(q2)·cos(q4) + sin(q4)·cos(q2)·cos(q3)))
y = Lax·sin(q1) + Lb·sin(q1) + Lt·(sin(q1)·sin(q3)·sin(q4) - cos(q1)·(sin(q2)·cos(q4) + sin(q4)·cos(q2)·cos(q3)))
z = Laz + Lt·(cos(q2)·cos(q4) - sin(q2)·sin(q4)·cos(q3))
```

The full AUTOLEV source, including angular velocities, accelerations, and the analytical Jacobian, is in [`kinematics/4DOF_kane.all`](kinematics/4DOF_kane.all).

## SimScape Simulation

The matlab/Simulink SimScape Multibody model imports SolidWorks CAD geometry directly and provides:

- **Position control** — Joint trajectory tracking (`Montaj1.slx`)
- **Velocity analysis** — Joint velocity profiles (`Montaj1Hiz.slx`)
- **Torque computation** — Required actuator torques (`Montaj1tork.slx`)

CAD parts (links 0–3) are modeled in SolidWorks and exported as STEP files for SimScape import.

## MuJoCo Simulation

The packaged [MuJoCo](https://mujoco.org/) model cross-validates analytical
forward kinematics and provides a controlled dynamics demonstration:

- **MJCF model** matching exact link parameters (Laz=55mm, Lax=50mm, Lb=100mm, Lt=90mm)
- **PD joint control** with gravity compensation
- **Quintic trajectory planning** for smooth joint-space motion
- **FK cross-validation**: analytical forward kinematics vs. MuJoCo body positions - **0.000 mm error** across 7 test configurations
- **Headless Linux rendering** via EGL when `MUJOCO_GL=egl` is set

```bash
uv sync --project python --extra simulation
uv run --project python four-dof-arm-mujoco --validate-only
```

To render a new video, install FFmpeg and pass both `--video` and the desired
output path. On headless Linux, select the EGL backend as shown:

```bash
mkdir -p python/generated
MUJOCO_GL=egl uv run --project python four-dof-arm-mujoco --video \
  --output python/generated/arm_physics_sim.mp4
```

Generated output under `python/generated/` is ignored. The committed reference
render is [`simulations/arm_physics_sim.mp4`](simulations/arm_physics_sim.mp4).
CI validates the simulation and smoke-checks the installed commands, but does
not render video.
See [`mujoco_model.xml`](python/src/four_dof_arm/mujoco_model.xml) for the MJCF
model and [`mujoco_sim.py`](python/src/four_dof_arm/mujoco_sim.py) for the
simulation driver.

## Python Implementation

A packaged Python port provides forward/inverse kinematics, trajectory planning,
and optional 3D visualization without requiring MATLAB. See
[`python/README.md`](python/README.md) for the full API and development guide.

Highlights:
- Forward kinematics matching the AUTOLEV formulation exactly
- Numerical inverse kinematics via damped least-squares
- Cubic/quintic polynomial trajectory planning
- Real-time 3D matplotlib visualization and workspace plotting

All commands below are safe to run from the repository root:

```bash
uv sync --project python --extra visualization
uv run --project python four-dof-arm-demo --output-dir python/generated
```

Optional groups are `visualization` for Matplotlib, `simulation` for MuJoCo,
and `dev` for all runtime extras plus pytest/coverage, Ruff, mypy, and package
build tools. Use `uv sync --project python --all-extras` for the complete
environment.

## Repository Structure

```
├── .github/workflows/python.yml  # Python 3.10/3.12 checks
├── cad/                         # SolidWorks assembly and part files
├── figures/                     # Figures and diagrams
├── kinematics/
│   └── 4DOF_kane.all            # AUTOLEV Kane's method formulation
├── matlab/                      # Simscape Multibody models and CAD geometry
├── python/                      # Installable Python project
│   ├── pyproject.toml           # Package metadata and tool configuration
│   ├── uv.lock                  # Reproducible dependency lock
│   ├── README.md                # Python usage and development guide
│   ├── src/four_dof_arm/
│   │   ├── __init__.py          # Core public API
│   │   ├── cli.py               # Dependency-light console wrappers
│   │   ├── robot.py             # Forward/inverse kinematics and Jacobian
│   │   ├── trajectory.py        # Trajectory planning
│   │   ├── visualize.py         # Optional Matplotlib visualization
│   │   ├── demo.py              # Visualization CLI
│   │   ├── mujoco_sim.py        # Optional MuJoCo simulation CLI
│   │   └── mujoco_model.xml     # Packaged MJCF model
│   └── tests/                   # Python test suite
├── simulations/
│   └── arm_physics_sim.mp4      # Committed MuJoCo reference render
└── README.md
```

## Further Reading

Detailed derivation and results are documented in the accompanying article:

📄 [4-Eksenli Robot Kolunun Kontrolü — Medium](https://medium.com/@tokerb/d-%C3%B6rt-eksenli-robot-kolunun-kontrol%C3%BC-e33744d69f49)

## License

No software license has been selected. Choosing and adding one remains a
project-owner decision.
