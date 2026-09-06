"""Configuration-driven motion and numerical analysis, with optional dynamics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .model import FloatArray, SerialArm, _integer, _number, _vector


def analyze(
    arm: SerialArm,
    scenario: dict[str, Any] | None = None,
    *,
    timestep: float = 0.002,
    workspace_samples: int = 1000,
    seed: int = 0,
    simulate: bool = False,
) -> tuple[dict[str, Any], dict[str, FloatArray], FloatArray]:
    """Analyze a stop-at-waypoints quintic motion in SI units.

    Scenarios contain duration, joint waypoints, and optional Cartesian targets
    for independent position-only IK diagnostics. Targets do not alter the motion.
    """
    scenario = {} if scenario is None else scenario
    if not isinstance(scenario, dict):
        raise ValueError("scenario must be an object")
    unknown = set(scenario) - {"duration", "waypoints", "targets"}
    if unknown:
        raise ValueError(f"unknown scenario keys: {unknown}")
    duration = _number(scenario.get("duration", 6.0), "duration", positive=True)
    requested_dt = _number(timestep, "timestep", positive=True)
    seed = _integer(seed, "seed")
    workspace_samples = _integer(workspace_samples, "workspace_samples", minimum=1)
    lower, upper = arm.joint_limits.T
    home = np.clip(np.zeros(arm.n_joints), lower, upper)
    defaults = [
        home,
        np.clip(home + 0.35, lower, upper),
        np.clip(home - 0.35, lower, upper),
        home,
    ]
    raw_waypoints = scenario.get("waypoints", defaults)
    if not isinstance(raw_waypoints, list) or len(raw_waypoints) < 2:
        raise ValueError("waypoints must be an array of at least two joint vectors")
    waypoints = np.array(
        [arm.validate_configuration(q, enforce_limits=True) for q in raw_waypoints]
    )
    raw_targets = scenario.get("targets", [])
    if not isinstance(raw_targets, list):
        raise ValueError("targets must be an array of Cartesian positions")
    targets = [_vector(target, 3, "target") for target in raw_targets]

    # Uniform steps land exactly on the endpoint and never exceed the requested dt.
    ratio = duration / requested_dt
    if not math.isfinite(ratio) or ratio > 1_000_000:
        raise ValueError("motion exceeds 1,000,000 steps; increase timestep")
    steps = max(1, math.ceil(ratio))
    times = np.linspace(0.0, duration, steps + 1)
    dt = duration / steps
    segment_duration = duration / (len(waypoints) - 1)
    if not math.isfinite(segment_duration) or segment_duration <= 0:
        raise ValueError("segment duration is not representable")
    segment = np.minimum((times / segment_duration).astype(int), len(waypoints) - 2)
    u = np.clip((times / segment_duration - segment)[:, None], 0, 1)
    start, delta = waypoints[segment], np.diff(waypoints, axis=0)[segment]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        q = start + delta * (u**3 * (10 + u * (-15 + 6 * u)))
        qd = delta * (30 * u**2 * (1 - u) ** 2) / segment_duration
        qdd = (
            delta
            * (60 * u * (1 - u) * (1 - 2 * u))
            / segment_duration
            / segment_duration
        )
    if not all(np.all(np.isfinite(values)) for values in (q, qd, qdd)):
        raise ValueError("motion profiles must be representable as finite numbers")
    q = np.clip(q, lower, upper)
    q[0], q[-1] = waypoints[0], waypoints[-1]
    qd[[0, -1]], qdd[[0, -1]] = 0.0, 0.0
    ee_pos = np.array([arm.forward_kinematics(pose)[0] for pose in q])
    jacobians = np.array([arm.jacobian(pose) for pose in q])
    ee_speed = np.linalg.norm(np.einsum("ijk,ik->ij", jacobians, qd), axis=1)
    record = {
        "t": times,
        "q": q,
        "qd": qd,
        "qdd": qdd,
        "ee_pos": ee_pos,
        "ee_speed": ee_speed,
        "target_q": q.copy(),
        "target_qd": qd.copy(),
        "target_qdd": qdd.copy(),
        "target_ee_pos": ee_pos.copy(),
    }
    if simulate:
        from .simulation import run_simulation

        record = run_simulation(arm, record, timestep=dt)
    if not all(np.all(np.isfinite(value)) for value in record.values()):
        raise ValueError("analysis produced nonfinite values")

    workspace = arm.workspace(workspace_samples, seed=seed)
    ik_results = []
    for target in targets:
        solution, success, residual = arm.inverse_kinematics(
            target, q0=waypoints[0], seed=seed
        )
        ik_results.append(
            {
                "target_m": target.tolist(),
                "q_rad": solution.tolist(),
                "success": success,
                "residual_m": residual,
            }
        )
    actual_q = record["q"]
    limit_violation = np.maximum(np.maximum(lower - actual_q, actual_q - upper), 0)
    metrics: dict[str, Any] = {
        "path_length_m": float(
            np.linalg.norm(np.diff(record["ee_pos"], axis=0), axis=1).sum()
        ),
        "peak_tool_speed_m_s": float(record["ee_speed"].max()),
        "peak_abs_joint_velocity_rad_s": np.max(np.abs(record["qd"]), axis=0).tolist(),
        "peak_abs_joint_acceleration_rad_s2": np.max(
            np.abs(record["qdd"]), axis=0
        ).tolist(),
        "max_joint_limit_violation_rad": limit_violation.max(axis=0).tolist(),
    }
    report: dict[str, Any] = {
        "robot": arm.name,
        "n_joints": arm.n_joints,
        "joint_names": [link.name for link in arm.links],
        "units": {
            "length": "m",
            "angle": "rad",
            "time": "s",
            "mass": "kg",
            "torque": "N m",
        },
        "mode": "dynamic_tracking" if simulate else "prescribed_kinematics",
        "scenario": {
            "duration": duration,
            "waypoints": waypoints.tolist(),
            "targets": [target.tolist() for target in targets],
        },
        "sampling": {
            "samples": len(times),
            "requested_timestep_s": requested_dt,
            "timestep_s": dt,
            "seed": seed,
        },
        "workspace": {
            "samples": workspace_samples,
            "min_m": workspace.min(axis=0).tolist(),
            "max_m": workspace.max(axis=0).tolist(),
        },
        "reference_position_jacobian": {
            "start_rank": int(np.linalg.matrix_rank(jacobians[0])),
            "end_rank": int(np.linalg.matrix_rank(jacobians[-1])),
            "start_singular_values_m": np.linalg.svd(
                jacobians[0], compute_uv=False
            ).tolist(),
            "end_singular_values_m": np.linalg.svd(
                jacobians[-1], compute_uv=False
            ).tolist(),
        },
        "ik": ik_results,
        "sampled_metrics": metrics,
        "limitations": [
            "All contacts are disabled; this is not a collision or clearance analysis.",
            "Workspace is joint-limit sampling, not a certified reachable volume.",
            "IK solves position only; failure does not prove unreachability.",
            "Quintic motion stops at waypoints; velocity and acceleration limits are not enforced.",
            "Primitive inertias, gains, damping and motor limits are approximate, not hardware validation.",
            "The tool offset is massless unless payload_mass is supplied; payload is a sphere.",
            "Sampled extrema may miss peaks between samples.",
        ],
    }
    if simulate:
        error = record["q"] - record["target_q"]
        report["tracking"] = {
            "rms_joint_error_rad": np.sqrt(np.mean(error**2, axis=0)).tolist(),
            "max_abs_joint_error_rad": np.max(np.abs(error), axis=0).tolist(),
            "max_tool_error_m": float(
                np.linalg.norm(record["ee_pos"] - record["target_ee_pos"], axis=1).max()
            ),
        }
        limits = np.array([link.torque_limit for link in arm.links])
        metrics["peak_abs_actuator_torque_Nm"] = np.max(
            np.abs(record["torque"]), axis=0
        ).tolist()
        metrics["peak_abs_requested_torque_Nm"] = np.max(
            np.abs(record["requested_torque"]), axis=0
        ).tolist()
        # The terminal sample is not followed by a simulated control interval.
        metrics["saturated_step_fraction"] = np.mean(
            np.abs(record["requested_torque"][:-1]) > limits, axis=0
        ).tolist()
    return report, record, workspace
