from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from four_dof_arm import Robot4DOF
from four_dof_arm.serial_arm import SerialArm

PRESETS = Path(__file__).resolve().parents[1] / "examples" / "serial_arm"


@pytest.mark.parametrize(
    ("name", "n_joints", "home_position"),
    [
        ("planar_2link", 2, (0.55, 0.0, 0.1)),
        ("spatial_3link", 3, (0.54, 0.0, 0.17)),
        ("spatial_6link", 6, (0.59, 0.0, 0.17)),
        ("spatial_12link", 12, (1.14, 0.0, 0.21)),
        ("original_4dof", 4, (0.15, 0.0, 0.145)),
    ],
)
def test_presets_load_and_have_finite_kinematics(
    name: str, n_joints: int, home_position: tuple[float, float, float]
) -> None:
    arm = SerialArm.from_json(PRESETS / f"{name}.json")
    assert arm.name == name
    assert arm.n_joints == n_joints
    assert arm.joint_limits.shape == (n_joints, 2)
    home = arm.validate_configuration(np.zeros(n_joints), enforce_limits=True)
    position, rotation = arm.forward_kinematics(home)
    np.testing.assert_allclose(position, home_position, rtol=0, atol=1e-14)
    np.testing.assert_allclose(rotation, np.eye(3), rtol=0, atol=1e-14)

    rng = np.random.default_rng(42)
    for q in rng.uniform(*arm.joint_limits.T, size=(16, n_joints)):
        position, rotation = arm.forward_kinematics(q)
        points = arm.joint_positions(q)
        jacobian = arm.jacobian(q)
        assert position.shape == (3,)
        assert rotation.shape == (3, 3)
        assert points.shape == (n_joints + 2, 3)
        assert jacobian.shape == (3, n_joints)
        for values in (position, rotation, points, jacobian):
            assert np.all(np.isfinite(values))
        np.testing.assert_allclose(points[-1], position, rtol=0, atol=1e-14)
        if name == "planar_2link":
            np.testing.assert_allclose(points[:, 2], arm.base[2], rtol=0, atol=1e-14)


def test_original_4dof_matches_legacy_kinematics_in_si() -> None:
    arm = SerialArm.from_json(PRESETS / "original_4dof.json")
    legacy = Robot4DOF()
    np.testing.assert_array_equal(arm.joint_limits, legacy.joint_limits)
    assert arm.base == arm.tool == (0.0, 0.0, 0.0)
    assert arm.links[2].offset == (0.0, 0.0, 0.0)

    rng = np.random.default_rng(19)
    configurations = np.vstack(
        (
            np.zeros(4),
            legacy.joint_limits.T,
            rng.uniform(*legacy.joint_limits.T, size=(64, 4)),
        )
    )
    for q in configurations:
        position, rotation = arm.forward_kinematics(q)
        legacy_position, legacy_rotation = legacy.forward_kinematics(q)
        np.testing.assert_allclose(position, legacy_position / 1000, rtol=0, atol=1e-14)
        np.testing.assert_allclose(rotation, legacy_rotation, rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            arm.jacobian(q), legacy.jacobian_analytical(q) / 1000, rtol=0, atol=1e-14
        )


def test_reach_scenario_matches_spatial_3link() -> None:
    arm = SerialArm.from_json(PRESETS / "spatial_3link.json")
    with (PRESETS / "reach_scenario.json").open(encoding="utf-8") as stream:
        scenario = json.load(stream)
    assert set(scenario) == {"duration", "waypoints", "targets"}
    assert scenario["duration"] == 6.0
    waypoints = np.asarray(scenario["waypoints"], dtype=np.float64)
    assert waypoints.ndim == 2
    assert waypoints.shape[0] >= 2
    assert waypoints.shape[1] == arm.n_joints
    for q in waypoints:
        arm.validate_configuration(q, enforce_limits=True)
    np.testing.assert_array_equal(waypoints[0], np.zeros(arm.n_joints))
    np.testing.assert_array_equal(waypoints[-1], waypoints[0])

    targets = np.asarray(scenario["targets"], dtype=np.float64)
    assert targets.shape == (2, 3)
    assert np.all(np.isfinite(targets))
    np.testing.assert_allclose(
        targets[0], arm.forward_kinematics(waypoints[1])[0], rtol=0, atol=1e-14
    )
    # Exceeding the sum of offset lengths proves unreachability independently of IK.
    max_reach = sum(float(np.linalg.norm(link.offset)) for link in arm.links)
    max_reach += float(np.linalg.norm(arm.tool))
    assert np.linalg.norm(targets[1] - arm.base) > max_reach

    for target, expected_success in zip(targets, (True, False), strict=True):
        q, success, residual = arm.inverse_kinematics(target, q0=waypoints[0])
        assert success is expected_success
        assert (residual <= 1e-5) is expected_success
        arm.validate_configuration(q, enforce_limits=True)
        assert residual == pytest.approx(
            np.linalg.norm(arm.forward_kinematics(q)[0] - target)
        )
