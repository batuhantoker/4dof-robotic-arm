from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..robot import Robot4DOF
from .timing import (
    _interpolate_target,
    _positive_finite_float,
    _step_count,
    _validate_trajectory,
)

MODEL_PATH = Path(__file__).with_name("mujoco_model.xml")


class ArmSim:
    """MuJoCo simulation wrapper for the 4-DOF arm.

    Parameters
    ----------
    model_path : str or Path
        Path to mujoco_model.xml.
    """

    def __init__(self, model_path: str | Path = MODEL_PATH) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self._end_effector_site_id = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        )
        if self._end_effector_site_id < 0:
            raise ValueError("model must define a site named 'end_effector'")
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.robot = Robot4DOF()

        self.n_joints = 4
        self.dt = float(self.model.opt.timestep)
        self.Kp = np.array([3.0, 3.0, 2.0, 1.5])
        self.Kd = np.array([0.3, 0.3, 0.2, 0.15])
        self._actuator_ids, self._control_to_joint_torque = (
            self._find_joint_actuator_transmissions()
        )

    def _find_joint_actuator_transmissions(
        self,
    ) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
        if self.model.nu != self.n_joints:
            raise ValueError(
                f"model must define exactly {self.n_joints} controlled actuators"
            )

        actuator_ids = np.full(self.n_joints, -1, dtype=int)
        torque_scales = np.zeros(self.n_joints, dtype=np.float64)

        for actuator_id in range(self.model.nu):
            if self.model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
                raise ValueError(
                    f"actuator {actuator_id} must use a direct joint transmission"
                )
            if self.model.actuator_gaintype[actuator_id] != mujoco.mjtGain.mjGAIN_FIXED:
                raise ValueError(f"actuator {actuator_id} must use a fixed gain")
            if self.model.actuator_biastype[actuator_id] != mujoco.mjtBias.mjBIAS_NONE:
                raise ValueError(f"actuator {actuator_id} must not use actuator bias")
            if (
                self.model.actuator_dyntype[actuator_id] != mujoco.mjtDyn.mjDYN_NONE
                or self.model.actuator_actadr[actuator_id] != -1
            ):
                raise ValueError(
                    f"actuator {actuator_id} must not use activation dynamics"
                )

            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"actuator {actuator_id} must drive a hinge joint")
            dof_id = int(self.model.jnt_dofadr[joint_id])
            if not 0 <= dof_id < self.n_joints:
                raise ValueError(
                    f"actuator {actuator_id} drives an uncontrolled joint DOF"
                )
            if actuator_ids[dof_id] != -1:
                raise ValueError(f"multiple actuators drive joint DOF {dof_id}")

            gear = float(self.model.actuator_gear[actuator_id, 0])
            gain = float(self.model.actuator_gainprm[actuator_id, 0])
            scale = gear * gain
            if not np.isfinite(scale) or scale == 0.0:
                raise ValueError(
                    f"actuator {actuator_id} has an invalid torque transmission"
                )

            actuator_ids[dof_id] = actuator_id
            torque_scales[dof_id] = scale

        if np.any(actuator_ids < 0):
            raise ValueError(
                "each controlled joint must have one joint-transmission actuator"
            )

        return actuator_ids, torque_scales

    def _validate_target(self, target_q: ArrayLike) -> NDArray[np.float64]:
        target = np.asarray(target_q, dtype=np.float64)
        if target.shape != (self.n_joints,):
            raise ValueError(
                f"target_q must have shape ({self.n_joints},), got {target.shape}"
            )
        if not np.all(np.isfinite(target)):
            raise ValueError("target_q must be finite")
        return target

    def reset(self, q0: ArrayLike | None = None) -> None:
        """Reset simulation to the initial configuration."""
        initial_q = None if q0 is None else self._validate_target(q0).copy()
        mujoco.mj_resetData(self.model, self.data)
        if initial_q is not None:
            self.data.qpos[: self.n_joints] = initial_q
        mujoco.mj_forward(self.model, self.data)

    @property
    def dt(self) -> float:
        """Current model timestep in seconds; change between simulation calls."""
        return _positive_finite_float(self.model.opt.timestep, "timestep")

    @dt.setter
    def dt(self, value: float) -> None:
        self.model.opt.timestep = _positive_finite_float(value, "timestep")

    @property
    def q(self) -> NDArray[np.float64]:
        """Current joint positions."""
        return np.array(self.data.qpos[: self.n_joints], dtype=np.float64, copy=True)

    @property
    def qd(self) -> NDArray[np.float64]:
        """Current joint velocities."""
        return np.array(self.data.qvel[: self.n_joints], dtype=np.float64, copy=True)

    @property
    def ee_pos(self) -> NDArray[np.float64]:
        """End-effector position in meters."""
        return np.array(
            self.data.site_xpos[self._end_effector_site_id],
            dtype=np.float64,
            copy=True,
        )

    @property
    def ee_rotation(self) -> NDArray[np.float64]:
        """End-effector rotation matrix."""
        return np.array(
            self.data.site_xmat[self._end_effector_site_id],
            dtype=np.float64,
            copy=True,
        ).reshape(3, 3)

    def step(self, target_q: ArrayLike) -> None:
        """Advance one model timestep toward a finite four-joint target."""
        target = self._validate_target(target_q)
        q_error = target - self.data.qpos[: self.n_joints]
        qd = self.data.qvel[: self.n_joints]
        desired_joint_torque = (
            self.Kp * q_error - self.Kd * qd + self.data.qfrc_bias[: self.n_joints]
        )
        self.data.ctrl[self._actuator_ids] = (
            desired_joint_torque / self._control_to_joint_torque
        )
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def settle(
        self,
        target_q: ArrayLike,
        duration: float = 2.0,
        tol: float = 1e-3,
    ) -> float:
        """Run simulation until the arm settles at the target or times out."""
        target = self._validate_target(target_q)
        duration_value = _positive_finite_float(duration, "duration")
        tolerance = _positive_finite_float(tol, "tol")
        n_steps = _step_count(duration_value, self.dt)
        for _ in range(n_steps):
            self.step(target)
            if (
                np.linalg.norm(target - self.q) < tolerance
                and np.linalg.norm(self.qd) < tolerance
            ):
                break
        return float(np.linalg.norm(target - self.q))

    def run_trajectory(
        self,
        timestamps: ArrayLike,
        q_trajectory: ArrayLike,
    ) -> dict[str, NDArray[np.float64]]:
        """Execute timestamped joint targets and record every simulation step.

        The returned time vector is relative to the start of this call. It
        contains the initial state at zero and ends within one model timestep
        of the requested trajectory duration.
        """
        times, targets = _validate_trajectory(timestamps, q_trajectory, self.n_joints)
        duration = float(times[-1])
        n_steps = _step_count(duration, self.dt)
        n_samples = n_steps + 1
        record = {
            "t": np.zeros(n_samples),
            "q": np.zeros((n_samples, self.n_joints)),
            "qd": np.zeros((n_samples, self.n_joints)),
            "ee_pos": np.zeros((n_samples, 3)),
            "target_q": np.zeros((n_samples, self.n_joints)),
        }

        start_time = float(self.data.time)
        record["q"][0] = self.q
        record["qd"][0] = self.qd
        record["ee_pos"][0] = self.ee_pos
        record["target_q"][0] = targets[0]

        for index in range(1, n_samples):
            elapsed = float(self.data.time) - start_time
            target = _interpolate_target(times, targets, elapsed)
            self.step(target)

            sample_time = float(self.data.time) - start_time
            record["t"][index] = sample_time
            record["q"][index] = self.q
            record["qd"][index] = self.qd
            record["ee_pos"][index] = self.ee_pos
            record["target_q"][index] = _interpolate_target(times, targets, sample_time)

        return record
