"""Headless physics tracking for generated serial-arm models only."""

from __future__ import annotations

import mujoco
import numpy as np

from .model import FloatArray, SerialArm


def run_simulation(
    arm: SerialArm, reference: dict[str, FloatArray], *, timestep: float
) -> dict[str, FloatArray]:
    """Track validated references using unit-gear torque motors and bias compensation.

    State, acceleration and torque samples are instantaneous, before each step.
    The last sample records the terminal state without an additional step.
    """
    model = mujoco.MjModel.from_xml_string(arm.to_mjcf(timestep=timestep))
    data = mujoco.MjData(model)
    joint_ids = np.array([model.joint(link.name).id for link in arm.links])
    qpos_ids = model.jnt_qposadr[joint_ids]
    dof_ids = model.jnt_dofadr[joint_ids]
    motor_ids = np.array([model.actuator(f"motor_{i}").id for i in range(arm.n_joints)])
    site_id = model.site("end_effector").id
    kp = np.array([link.kp for link in arm.links])
    kd = np.array([link.kd for link in arm.links])
    torque_limits = np.array([link.torque_limit for link in arm.links])
    result = {key: value.copy() for key, value in reference.items()}
    result["torque"] = np.zeros_like(reference["q"])
    result["requested_torque"] = np.zeros_like(reference["q"])
    data.qpos[qpos_ids] = reference["target_q"][0]
    data.qvel[dof_ids] = reference["target_qd"][0]
    jacobian = np.zeros((3, model.nv))
    mujoco.mj_forward(model, data)
    for index, time in enumerate(reference["t"]):
        if not np.isclose(data.time, time, atol=timestep * 1e-6, rtol=1e-9):
            raise ValueError("simulation became unstable and reset its clock")
        requested = (
            kp * (reference["target_q"][index] - data.qpos[qpos_ids])
            + kd * (reference["target_qd"][index] - data.qvel[dof_ids])
            + data.qfrc_bias[dof_ids]
        )
        data.ctrl[motor_ids] = np.clip(requested, -torque_limits, torque_limits)
        mujoco.mj_forward(model, data)
        result["q"][index] = data.qpos[qpos_ids]
        result["qd"][index] = data.qvel[dof_ids]
        result["qdd"][index] = data.qacc[dof_ids]
        result["ee_pos"][index] = data.site_xpos[site_id]
        result["torque"][index] = data.qfrc_actuator[dof_ids]
        result["requested_torque"][index] = requested
        mujoco.mj_jacSite(model, data, jacobian, None, site_id)
        result["ee_speed"][index] = np.linalg.norm(jacobian @ data.qvel)
        if any(warning.number for warning in data.warning):
            raise ValueError(
                "MuJoCo reported a simulation warning; check model and timestep"
            )
        if index < len(reference["t"]) - 1:
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
    return result
