from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from four_dof_arm.serial_arm import Link, SerialArm
from four_dof_arm.serial_arm.analysis import analyze

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "serial_arm"


def make_arm(n: int = 2) -> SerialArm:
    return SerialArm(
        links=tuple(
            Link(
                name=f"hinge_{i}",
                axis=(0, 0, 1) if i % 2 == 0 else (0, 1, 0),
                offset=(0.2, 0, 0.04),
                mass=0.3,
            )
            for i in range(n)
        ),
        base=(0.1, -0.2, 0.3),
        tool=(0.03, 0.02, 0.01),
    )


@pytest.mark.parametrize("n", [1, 2, 6])
@pytest.mark.parametrize("custom", [False, True], ids=["default", "custom"])
def test_prescribed_motion_profiles_and_report(n: int, custom: bool) -> None:
    arm = make_arm(n)
    waypoints = (
        np.array([np.linspace(-0.2, 0.1, n), np.full(n, 0.6), np.full(n, -0.3)])
        if custom
        else np.array([np.zeros(n), np.full(n, 0.35), np.full(n, -0.35), np.zeros(n)])
    )
    duration = 2.4 if custom else 6.0
    scenario = (
        {"duration": duration, "waypoints": waypoints.tolist()} if custom else None
    )
    report, record, workspace = analyze(
        arm, scenario, timestep=0.05, workspace_samples=11, seed=7
    )

    samples = len(record["t"])
    assert report["mode"] == "prescribed_kinematics"
    assert report["n_joints"] == n
    assert report["joint_names"] == [link.name for link in arm.links]
    assert report["scenario"]["waypoints"] == waypoints.tolist()
    assert report["sampling"]["samples"] == samples
    assert report["sampling"]["seed"] == 7
    assert record["t"][[0, -1]].tolist() == [0, duration]
    assert workspace.shape == (11, 3)
    assert report["workspace"]["samples"] == len(workspace)
    np.testing.assert_array_equal(report["workspace"]["min_m"], workspace.min(axis=0))
    np.testing.assert_array_equal(report["workspace"]["max_m"], workspace.max(axis=0))
    assert "tracking" not in report
    assert "torque" not in record
    for field in ("q", "qd", "qdd", "ee_pos"):
        assert record[field].shape == (samples, 3 if field == "ee_pos" else n)
        np.testing.assert_array_equal(record[field], record[f"target_{field}"])
        assert not np.shares_memory(record[field], record[f"target_{field}"])
    assert record["ee_speed"].shape == (samples,)
    assert all(np.isfinite(values).all() for values in record.values())
    json.dumps(report, allow_nan=False)

    boundaries = np.linspace(0, samples - 1, len(waypoints), dtype=int)
    np.testing.assert_allclose(record["q"][boundaries], waypoints, atol=1e-14)
    np.testing.assert_allclose(record["qd"][boundaries], 0, atol=1e-14)
    np.testing.assert_allclose(record["qdd"][boundaries], 0, atol=1e-14)
    midpoints = (boundaries[:-1] + boundaries[1:]) // 2
    segment_duration = duration / (len(waypoints) - 1)
    np.testing.assert_allclose(
        record["q"][midpoints], (waypoints[:-1] + waypoints[1:]) / 2, atol=1e-14
    )
    np.testing.assert_allclose(
        record["qd"][midpoints], 1.875 * np.diff(waypoints, axis=0) / segment_duration
    )
    np.testing.assert_allclose(record["qdd"][midpoints], 0, atol=1e-14)
    for q, qd, position, speed in zip(
        record["q"], record["qd"], record["ee_pos"], record["ee_speed"], strict=True
    ):
        np.testing.assert_allclose(position, arm.forward_kinematics(q)[0], atol=1e-14)
        assert speed == pytest.approx(np.linalg.norm(arm.jacobian(q) @ qd))
    metrics = report["sampled_metrics"]
    assert metrics["path_length_m"] == pytest.approx(
        np.linalg.norm(np.diff(record["ee_pos"], axis=0), axis=1).sum()
    )
    assert metrics["peak_tool_speed_m_s"] == pytest.approx(record["ee_speed"].max())
    for field, metric in (
        ("qd", "peak_abs_joint_velocity_rad_s"),
        ("qdd", "peak_abs_joint_acceleration_rad_s2"),
    ):
        np.testing.assert_array_equal(
            metrics[metric], np.abs(record[field]).max(axis=0)
        )
    assert metrics["max_joint_limit_violation_rad"] == [0] * n


@pytest.mark.parametrize("timestep", [0.3, 0.25, 2.0])
def test_sampling_adjusts_down_and_includes_terminal_state(timestep: float) -> None:
    report, record, _ = analyze(
        make_arm(1),
        {"duration": 1.0, "waypoints": [[-0.2], [0.4]]},
        timestep=timestep,
        workspace_samples=1,
    )
    dt = report["sampling"]["timestep_s"]
    assert 0 < dt <= timestep
    assert report["sampling"]["requested_timestep_s"] == timestep
    assert len(record["t"]) == int(np.ceil(1 / timestep)) + 1
    np.testing.assert_allclose(np.diff(record["t"]), dt)
    np.testing.assert_array_equal(record["t"][[0, -1]], [0, 1])
    np.testing.assert_array_equal(record["q"][[0, -1]], [[-0.2], [0.4]])
    np.testing.assert_array_equal(record["qd"][[0, -1]], 0)
    np.testing.assert_array_equal(record["qdd"][[0, -1]], 0)


def test_derivatives_match_finite_differences_away_from_boundaries() -> None:
    _, record, _ = analyze(
        make_arm(),
        {"duration": 2, "waypoints": [[-0.2, 0.3], [0.6, -0.4]]},
        timestep=0.002,
        workspace_samples=1,
    )
    for field, derivative in (("q", "qd"), ("qd", "qdd")):
        numerical = np.gradient(record[field], record["t"], axis=0)
        np.testing.assert_allclose(numerical[1:-1], record[derivative][1:-1], atol=2e-5)


def test_default_home_and_waypoints_respect_limits_excluding_zero() -> None:
    arm = make_arm()
    arm = replace(
        arm,
        links=(
            replace(arm.links[0], limits=(0.2, 0.3)),
            replace(arm.links[1], limits=(-0.4, -0.1)),
        ),
    )
    report, record, _ = analyze(arm, timestep=0.1, workspace_samples=1)
    assert report["scenario"]["waypoints"] == [
        [0.2, -0.1],
        [0.3, -0.1],
        [0.2, -0.4],
        [0.2, -0.1],
    ]
    assert np.all(record["q"] >= arm.joint_limits[:, 0])
    assert np.all(record["q"] <= arm.joint_limits[:, 1])
    assert report["sampled_metrics"]["max_joint_limit_violation_rad"] == [0, 0]


def test_ik_reports_are_seeded_and_targets_do_not_change_motion() -> None:
    arm = make_arm()
    targets = [arm.forward_kinematics([0.4, -0.6])[0].tolist(), [10, 10, 10]]
    scenario = {"duration": 0.2, "targets": targets}
    first = analyze(arm, scenario, timestep=0.05, workspace_samples=7, seed=19)
    second = analyze(arm, scenario, timestep=0.05, workspace_samples=7, seed=19)
    baseline = analyze(
        arm, {"duration": 0.2}, timestep=0.05, workspace_samples=7, seed=20
    )
    report, record, workspace = first
    assert report == second[0]
    np.testing.assert_array_equal(workspace, second[2])
    assert not np.array_equal(workspace, baseline[2])
    for field, values in record.items():
        np.testing.assert_array_equal(values, second[1][field])
        np.testing.assert_array_equal(values, baseline[1][field])
    assert [result["success"] for result in report["ik"]] == [True, False]
    for target, result in zip(targets, report["ik"], strict=True):
        assert result["target_m"] == target
        assert len(result["q_rad"]) == arm.n_joints
        arm.validate_configuration(result["q_rad"], enforce_limits=True)
        assert result["residual_m"] == pytest.approx(
            np.linalg.norm(arm.forward_kinematics(result["q_rad"])[0] - target)
        )
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    ("scenario", "message"),
    [
        ([], "scenario"),
        ({"typo": 1}, "unknown scenario"),
        ({"duration": 0}, "duration"),
        ({"duration": True}, "duration"),
        ({"duration": float("inf")}, "duration"),
        ({"waypoints": None}, "waypoints"),
        ({"waypoints": [[0, 0]]}, "waypoints"),
        ({"waypoints": [[0, 0], [0]]}, "shape"),
        ({"waypoints": [[0, 0], [4, 0]]}, "limits"),
        ({"waypoints": [[0, 0], [float("nan"), 0]]}, "finite"),
        ({"targets": {}}, "targets"),
        ({"targets": [[0, 0]]}, "target"),
        ({"targets": [[0, float("inf"), 0]]}, "target"),
    ],
)
def test_malformed_scenarios_are_rejected(scenario: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        analyze(make_arm(), scenario, timestep=0.1, workspace_samples=1)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"timestep": 0}, "timestep"),
        ({"timestep": float("nan")}, "timestep"),
        ({"timestep": 1e-9}, "steps"),
        ({"workspace_samples": 0}, "workspace_samples"),
        ({"workspace_samples": 2.5}, "workspace_samples"),
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
    ],
)
def test_invalid_sampling_options_are_rejected(
    options: dict[str, Any], message: str
) -> None:
    kwargs = {"timestep": 0.1, "workspace_samples": 1, **options}
    with pytest.raises(ValueError, match=message):
        analyze(make_arm(), **kwargs)


def test_unrepresentable_motion_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite|representable"):
        analyze(
            make_arm(1),
            {"duration": 1e-200, "waypoints": [[0], [0.2]]},
            timestep=1e-201,
            workspace_samples=1,
        )


def test_large_finite_duration_returns_finite_data_or_validation_error() -> None:
    try:
        report, record, workspace = analyze(
            make_arm(1),
            {"duration": 1e200, "waypoints": [[0], [0.2]]},
            timestep=1e199,
            workspace_samples=1,
        )
    except ValueError:
        return
    json.dumps(report, allow_nan=False)
    assert all(np.isfinite(values).all() for values in record.values())
    assert np.isfinite(workspace).all()


@pytest.mark.parametrize("motion", ["stationary", "moving", "saturated"])
def test_dynamic_tracking_reports_actual_states_and_applied_torques(
    motion: str,
) -> None:
    pytest.importorskip("mujoco")
    arm = SerialArm.from_json(EXAMPLES / "spatial_3link.json")
    if motion == "moving":
        scenario = json.loads((EXAMPLES / "reach_scenario.json").read_text())
        scenario.pop("targets")
    else:
        scenario = {"duration": 0.2, "waypoints": [[0, 0, 0], [0, 0, 0]]}
    if motion == "saturated":
        arm = replace(
            arm, links=tuple(replace(link, torque_limit=0.01) for link in arm.links)
        )
    report, record, _ = analyze(
        arm, scenario, timestep=0.002, workspace_samples=3, simulate=True
    )
    assert report["mode"] == "dynamic_tracking"
    assert record["t"][-1] == scenario["duration"]
    for field in ("q", "qd", "qdd", "torque", "requested_torque"):
        assert record[field].shape == (len(record["t"]), arm.n_joints)
    assert all(np.isfinite(values).all() for values in record.values())
    json.dumps(report, allow_nan=False)
    np.testing.assert_array_equal(record["q"][0], record["target_q"][0])
    for q, qd, position, speed, target_q, target_position in zip(
        record["q"],
        record["qd"],
        record["ee_pos"],
        record["ee_speed"],
        record["target_q"],
        record["target_ee_pos"],
        strict=True,
    ):
        np.testing.assert_allclose(position, arm.forward_kinematics(q)[0], atol=1e-13)
        np.testing.assert_allclose(
            target_position, arm.forward_kinematics(target_q)[0], atol=1e-13
        )
        assert speed == pytest.approx(np.linalg.norm(arm.jacobian(q) @ qd), abs=1e-13)
    error = record["q"] - record["target_q"]
    tracking = report["tracking"]
    np.testing.assert_allclose(
        tracking["rms_joint_error_rad"], np.sqrt(np.mean(error**2, axis=0))
    )
    np.testing.assert_allclose(
        tracking["max_abs_joint_error_rad"], np.abs(error).max(axis=0)
    )
    assert tracking["max_tool_error_m"] == pytest.approx(
        np.linalg.norm(record["ee_pos"] - record["target_ee_pos"], axis=1).max()
    )
    limits = np.array([link.torque_limit for link in arm.links])
    assert np.all(np.abs(record["torque"]) <= limits + 1e-12)
    np.testing.assert_allclose(
        record["torque"],
        np.clip(record["requested_torque"], -limits, limits),
        atol=1e-12,
    )
    metrics = report["sampled_metrics"]
    for field, metric in (
        ("torque", "peak_abs_actuator_torque_Nm"),
        ("requested_torque", "peak_abs_requested_torque_Nm"),
    ):
        np.testing.assert_allclose(metrics[metric], np.abs(record[field]).max(axis=0))
    np.testing.assert_allclose(
        metrics["saturated_step_fraction"],
        np.mean(np.abs(record["requested_torque"][:-1]) > limits, axis=0),
    )
    if motion == "stationary":
        np.testing.assert_allclose(record["q"], record["target_q"], atol=1e-12)
        np.testing.assert_allclose(record["qd"], 0, atol=1e-12)
        np.testing.assert_allclose(record["qdd"], 0, atol=1e-12)
        assert np.abs(record["torque"]).max() > 1
    elif motion == "moving":
        assert np.abs(record["q"]).max() > 0.5
        assert 0 < np.abs(error).max() < 0.08
        assert tracking["max_tool_error_m"] < 0.03
        assert metrics["saturated_step_fraction"] == [0, 0, 0]
    else:
        assert np.any(np.array(metrics["peak_abs_requested_torque_Nm"]) > limits)
        assert max(metrics["saturated_step_fraction"]) > 0.9
        assert np.abs(error).max() > 0.01
