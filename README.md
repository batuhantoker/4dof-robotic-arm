# 4-DOF Robotic Arm — Kinematics, Simulation & Validated Physics

![MATLAB](https://img.shields.io/badge/MATLAB-R2022a-blue?logo=mathworks)
![Simulink](https://img.shields.io/badge/Simulink-SimScape-orange?logo=mathworks)
![MuJoCo](https://img.shields.io/badge/MuJoCo-3.6-purple)
![Python](https://img.shields.io/badge/Python-3.10+-green?logo=python)
![SolidWorks](https://img.shields.io/badge/SolidWorks-CAD-red)

End-to-end study of a 4-DOF serial manipulator: symbolic kinematics (Kane's method), SimScape multibody dynamics, and a **validated MuJoCo physics simulation** with PD control and gravity compensation.

**FK validation: 0.000 mm error** across all test configurations — analytical kinematics match physics engine exactly.

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

## MuJoCo Physics Simulation

A full physics simulation in [MuJoCo 3.6](https://mujoco.org/) validates the analytical kinematics against a rigid-body physics engine:

- **MJCF model** matching exact link parameters (Laz=55mm, Lax=50mm, Lb=100mm, Lt=90mm)
- **PD joint control** with gravity compensation
- **Quintic trajectory planning** for smooth joint-space motion
- **FK cross-validation**: analytical forward kinematics vs. MuJoCo body positions — **0.000 mm error** across 7 test configurations
- **Headless rendering** via EGL for CI/server environments

```bash
cd python && pip install mujoco numpy && python mujoco_sim.py
```

Output: `arm_physics_sim.mp4` — rendered trajectory with real physics.

See [`python/mujoco_model.xml`](python/mujoco_model.xml) for the MJCF model and [`python/mujoco_sim.py`](python/mujoco_sim.py) for the simulation driver.

## Python Implementation

A standalone Python port provides forward/inverse kinematics, trajectory planning, and 3D visualization — no MATLAB license required. See [`python/README.md`](python/README.md) for quickstart.

Highlights:
- Forward kinematics matching the AUTOLEV formulation exactly
- Numerical inverse kinematics via Jacobian-based Newton-Raphson
- Cubic/quintic polynomial trajectory planning
- Real-time 3D matplotlib visualization and workspace plotting

## Repository Structure

```
├── cad/                  # SolidWorks assembly and part files
├── figures/             # Figures and diagrams
├── kinematics/            # AUTOLEV Kane's method formulation
│   └── 4DOF_kane.all     # Full symbolic kinematics source
├── matlab/               # SimScape Multibody models
│   ├── Montaj1.slx       # Position control simulation
│   ├── Montaj1Hiz.slx    # Velocity analysis
│   ├── Montaj1tork.slx   # Torque computation
│   └── *.SLDPRT/STEP     # CAD geometry for SimScape
├── python/               # Python implementation + MuJoCo simulation
│   ├── robot.py          # Forward/inverse kinematics, Jacobian
│   ├── trajectory.py     # Trajectory planning
│   ├── visualize.py      # 3D visualization & animation
│   ├── demo.py           # Demo script
│   ├── mujoco_model.xml  # MJCF physics model
│   ├── mujoco_sim.py     # MuJoCo simulation + FK validation
│   ├── arm_physics_sim.mp4  # Rendered simulation output
│   └── requirements.txt  # Dependencies
├── simulations/        # Simulation results and data
└── README.md
```

## Further Reading

Detailed derivation and results are documented in the accompanying article:

📄 [4-Eksenli Robot Kolunun Kontrolü — Medium](https://medium.com/@tokerb/d-%C3%B6rt-eksenli-robot-kolunun-kontrol%C3%BC-e33744d69f49)

## License

Academic/personal project. Feel free to reference with attribution.
