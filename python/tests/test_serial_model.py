from __future__ import annotations

import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from four_dof_arm import serial_arm
from four_dof_arm.serial_arm import Link, SerialArm


def planar_config(n: int = 2) -> dict[str, Any]:
    return {
        "units": "SI",
        "links": [
            {"name": f"joint_{i}", "axis": [0, 0, 2], "offset": [1, 0, 0], "mass": 0.5}
            for i in range(n)
        ],
    }


def spatial_config(n: int = 6) -> dict[str, Any]:
    rng = np.random.default_rng(28)
    config = planar_config(n)
    config.update(
        name='Spatial & "tool"',
        base=[0.1, -0.2, 0.4],
        tool=[0.06, -0.02, 0.08],
        payload_mass=0.3,
    )
    for index, link in enumerate(config["links"]):
        link.update(
            axis=rng.normal(size=3).tolist(),
            offset=(
                [0, 0, 0] if index % 3 == 0 else rng.uniform(-0.2, 0.2, 3).tolist()
            ),
            limits=[-2.0, 2.4],
            mass=0.3 + 0.1 * index,
            radius=0.012 + index * 0.001,
            damping=0.03 * index,
            armature=0.01 * index,
            torque_limit=5 + index,
            kp=12 + index,
            kd=0.5 * index,
        )
    return config


def test_facade_defaults_immutability_and_round_trip(tmp_path: Path) -> None:
    assert serial_arm.__all__ == ["SerialArm", "Link"]
    config = planar_config(1)
    arm = SerialArm.from_dict(config)
    assert isinstance(arm.links, tuple)
    assert isinstance(arm.links[0], Link)
    assert arm.name == "serial_arm"
    assert arm.base == arm.tool == (0.0, 0.0, 0.0)
    assert arm.payload_mass == 0
    link = arm.links[0]
    assert link.axis == (0.0, 0.0, 1.0)
    assert link.offset == (1.0, 0.0, 0.0)
    assert link.limits == (-math.pi, math.pi)
    assert (link.radius, link.damping, link.armature) == (0.015, 0.05, 0.01)
    assert (link.torque_limit, link.kp, link.kd) == (10, 20, 2)
    with pytest.raises(FrozenInstanceError):
        link.mass = 9  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        arm.links = ()  # type: ignore[misc]
    config["links"][0]["offset"][0] = 100
    exported = arm.to_dict()
    assert isinstance(exported["links"][0]["axis"], list)
    assert SerialArm.from_dict(exported) == arm
    path = tmp_path / "arm.json"
    path.write_text(json.dumps(exported), encoding="utf-8")
    assert SerialArm.from_json(path) == SerialArm.from_json(str(path)) == arm
    exported["links"][0]["axis"][2] = -1
    arm.joint_limits[:] = 0
    assert link.axis == (0.0, 0.0, 1.0)
    np.testing.assert_array_equal(arm.joint_limits, [[-math.pi, math.pi]])


@pytest.mark.parametrize("scale", [1e-300, 1e300])
def test_axis_normalization_avoids_underflow_and_overflow(scale: float) -> None:
    config = planar_config(1)
    config["links"][0]["axis"] = [scale, -scale, scale]
    np.testing.assert_allclose(
        SerialArm.from_dict(config).links[0].axis, np.array([1, -1, 1]) / np.sqrt(3)
    )


@pytest.mark.parametrize("n", [1, 2, 6])
def test_planar_kinematics_and_jacobian_are_analytic(n: int) -> None:
    config = planar_config(n)
    config.update(base=[0.2, -0.4, 0.1], tool=[0.3, 0.1, 0.2])
    arm = SerialArm.from_dict(config)
    q = np.linspace(0.2, 0.7, n)
    angles = np.cumsum(q)
    offsets = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(n)))
    endpoints = arm.base + np.cumsum(offsets, axis=0)
    c, s = np.cos(angles[-1]), np.sin(angles[-1])
    expected_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    expected_tip = endpoints[-1] + expected_rotation @ arm.tool
    points = np.vstack((arm.base, endpoints, expected_tip))
    position, rotation = arm.forward_kinematics(q)
    assert arm.n_joints == n
    assert position.shape == (3,)
    assert rotation.shape == (3, 3)
    np.testing.assert_allclose(position, expected_tip, atol=1e-14)
    np.testing.assert_allclose(rotation, expected_rotation, atol=1e-14)
    np.testing.assert_allclose(arm.joint_positions(q), points, atol=1e-14)
    lever = expected_tip - points[:n]
    expected_jacobian = np.vstack((-lever[:, 1], lever[:, 0], np.zeros(n)))
    np.testing.assert_allclose(arm.jacobian(q), expected_jacobian, atol=1e-14)


def test_joint_positions_include_coincident_endpoint_and_tool() -> None:
    config = planar_config(1)
    config["links"][0]["offset"] = [0, 0, 0]
    arm = SerialArm.from_dict(config)
    np.testing.assert_array_equal(arm.joint_positions([0.8]), np.zeros((3, 3)))
    np.testing.assert_array_equal(arm.jacobian([0.8]), np.zeros((3, 1)))


@pytest.mark.parametrize("n", [1, 2, 6])
def test_spatial_jacobian_matches_central_finite_differences(n: int) -> None:
    arm = SerialArm.from_dict(spatial_config(n))
    rng = np.random.default_rng(4)
    epsilon = 1e-6
    for q in rng.uniform(-1.5, 1.5, size=(5, n)):
        finite_difference = np.column_stack(
            [
                (
                    arm.forward_kinematics(q + delta)[0]
                    - arm.forward_kinematics(q - delta)[0]
                )
                / (2 * epsilon)
                for delta in np.eye(n) * epsilon
            ]
        )
        np.testing.assert_allclose(arm.jacobian(q), finite_difference, atol=2e-10)


@pytest.mark.parametrize("config", [None, [], "robot", 42])
def test_root_must_be_an_object(config: Any) -> None:
    with pytest.raises(ValueError):
        SerialArm.from_dict(config)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("units", None),
        ("units", "mm"),
        ("units", "si"),
        ("units", ["SI"]),
        ("name", ""),
        ("name", "  "),
        ("name", 3),
        ("base", [0, 1]),
        ("base", [[0, 0, 0]]),
        ("tool", "0 0 0"),
        ("payload_mass", -1),
        ("payload_mass", True),
        ("links", []),
        ("links", {}),
        ("links", [None]),
        ("links", [3]),
        ("scenario", {}),
        ("q", [0, 0]),
        ("gravity", [0, 0, -9.81]),
    ],
)
def test_invalid_root_fields(key: str, value: object) -> None:
    config = planar_config()
    config[key] = value
    with pytest.raises(ValueError):
        SerialArm.from_dict(config)


@pytest.mark.parametrize("key", ["units", "links"])
def test_required_root_fields(key: str) -> None:
    config = planar_config()
    del config[key]
    with pytest.raises(ValueError):
        SerialArm.from_dict(config)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("name", "a b"),
        ("name", "0joint"),
        ("name", "_joint"),
        ("name", "j-1"),
        ("name", "j\n"),
        ("name", 3),
        ("name", ""),
        ("axis", [0, 0, 0]),
        ("axis", [0, 1]),
        ("axis", None),
        ("offset", [[0, 0, 1]]),
        ("offset", [0, 0, True]),
        ("limits", [0, 0]),
        ("limits", [1, -1]),
        ("limits", [0, 1, 2]),
        ("mass", 0),
        ("radius", 0),
        ("damping", -1),
        ("armature", -1),
        ("torque_limit", 0),
        ("kp", 0),
        ("kd", -1),
        ("unknown", 1),
    ],
)
def test_invalid_link_fields(key: str, value: object) -> None:
    config = planar_config()
    config["links"][0][key] = value
    with pytest.raises(ValueError):
        SerialArm.from_dict(config)


@pytest.mark.parametrize("key", ["name", "axis", "offset", "mass"])
def test_required_link_fields(key: str) -> None:
    config = planar_config()
    del config["links"][0][key]
    with pytest.raises(ValueError):
        SerialArm.from_dict(config)


def test_duplicate_joint_names_are_rejected() -> None:
    config = planar_config()
    config["links"][1]["name"] = config["links"][0]["name"]
    with pytest.raises(ValueError, match="unique"):
        SerialArm.from_dict(config)


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        "1",
        None,
        {},
        float("nan"),
        float("inf"),
        -float("inf"),
        1j,
        10**1000,
    ],
)
@pytest.mark.parametrize(
    "key",
    [
        "axis",
        "offset",
        "limits",
        "mass",
        "radius",
        "damping",
        "armature",
        "torque_limit",
        "kp",
        "kd",
        "base",
        "tool",
        "payload_mass",
    ],
)
def test_all_numeric_fields_reject_invalid_numbers(key: str, value: object) -> None:
    config = planar_config(1)
    destination = (
        config if key in ("base", "tool", "payload_mass") else config["links"][0]
    )
    if key in ("base", "tool", "axis", "offset"):
        destination[key] = [value, 0, 1]
    elif key == "limits":
        destination[key] = [value, 1]
    else:
        destination[key] = value
    with pytest.raises(ValueError):
        SerialArm.from_dict(config)


@pytest.mark.parametrize(
    "text",
    [
        '{"units":"SI","units":"SI","links":[]}',
        '{"units":"SI","links":[{"name":"a","name":"b"}]}',
        '{"units":"SI","payload_mass":NaN,"links":[]}',
        '{"units":"SI","payload_mass":Infinity,"links":[]}',
        '{"units":"SI","payload_mass":1e999,"links":[]}',
        '{"units":"SI",}',
        "[]",
        "null",
    ],
)
def test_strict_json(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        SerialArm.from_json(path)


@pytest.mark.parametrize(
    "q",
    [
        None,
        0,
        [0],
        [[0, 0]],
        [True, 0],
        ["0", 0],
        [np.nan, 0],
        [np.inf, 0],
        [1j, 0],
        [10**1000, 0],
    ],
)
def test_joint_methods_validate_inputs(q: object) -> None:
    arm = SerialArm.from_dict(planar_config())
    for method in (
        arm.validate_configuration,
        arm.forward_kinematics,
        arm.joint_positions,
        arm.jacobian,
    ):
        with pytest.raises(ValueError):
            method(q)
    with pytest.raises(ValueError):
        arm.inverse_kinematics([0, 0, 0], q0=q if q is not None else [None, 0])


def test_configuration_copies_and_optionally_enforces_limits() -> None:
    arm = SerialArm.from_dict(planar_config())
    q = np.array([0, 1], dtype=np.float32)
    validated = arm.validate_configuration(q)
    assert validated.dtype == np.float64
    validated[0] = 1
    assert q[0] == 0
    arm.validate_configuration([-math.pi, math.pi], enforce_limits=True)
    arm.validate_configuration([10, 0])
    with pytest.raises(ValueError, match="limits"):
        arm.validate_configuration([10, 0], enforce_limits=True)


@pytest.mark.parametrize("n", [1, 2, 6])
def test_workspace_matches_seeded_bounded_samples(n: int) -> None:
    arm = SerialArm.from_dict(spatial_config(n))
    expected_q = np.random.default_rng(42).uniform(*arm.joint_limits.T, size=(12, n))
    expected = np.array([arm.forward_kinematics(q)[0] for q in expected_q])
    points = arm.workspace(12, seed=42)
    assert points.shape == (12, 3)
    np.testing.assert_array_equal(points, expected)
    np.testing.assert_array_equal(points, arm.workspace(12, seed=42))
    assert not np.array_equal(points, arm.workspace(12, seed=43))


@pytest.mark.parametrize("n", [1, 2, 6])
@pytest.mark.parametrize("spatial", [False, True])
def test_ik_reachable_and_unreachable_results_are_bounded_and_reproducible(
    n: int,
    spatial: bool,
) -> None:
    arm = SerialArm.from_dict(spatial_config(n) if spatial else planar_config(n))
    target = arm.forward_kinematics(np.linspace(0.2, 0.7, n))[0]
    initial = np.zeros(n)
    for goal, expected_success in [(target, True), (np.array([100, 100, 100]), False)]:
        first = arm.inverse_kinematics(goal, q0=initial, seed=7)
        second = arm.inverse_kinematics(goal, q0=initial, seed=7)
        q, success, residual = first
        assert success is expected_success
        assert residual == pytest.approx(
            np.linalg.norm(arm.forward_kinematics(q)[0] - goal)
        )
        assert (residual <= 1e-5) is expected_success
        arm.validate_configuration(q, enforce_limits=True)
        np.testing.assert_array_equal(q, second[0])
        assert first[1:] == second[1:]
    np.testing.assert_array_equal(initial, np.zeros(n))


def test_ik_restarts_escape_straight_arm_singularity() -> None:
    arm = SerialArm.from_dict(planar_config())
    target = [1.5, 0, 0]
    assert not arm.inverse_kinematics(target, restarts=0)[1]
    q, success, residual = arm.inverse_kinematics(target, seed=0)
    assert success and residual <= 1e-5
    np.testing.assert_allclose(arm.forward_kinematics(q)[0], target, atol=1e-5)


def test_ik_clips_initial_guess_and_reports_final_iteration_residual() -> None:
    arm = SerialArm.from_dict(planar_config(1))
    q, success, residual = arm.inverse_kinematics([-1, 0, 0], q0=[10], restarts=0)
    assert success and residual <= 1e-5
    np.testing.assert_array_equal(q, [math.pi])
    q, success, residual = arm.inverse_kinematics([0, 1, 0], max_iter=1, restarts=0)
    assert not success
    assert q[0] > 0
    assert residual == pytest.approx(
        np.linalg.norm(arm.forward_kinematics(q)[0] - [0, 1, 0])
    )


@pytest.mark.parametrize(
    "target",
    [[], [[0, 0, 0]], [True, 0, 0], ["0", 0, 0], [np.nan, 0, 0], [np.inf, 0, 0]],
)
def test_ik_invalid_targets(target: object) -> None:
    with pytest.raises(ValueError, match="target"):
        SerialArm.from_dict(planar_config()).inverse_kinematics(target)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("tol", 0),
        ("tol", True),
        ("tol", "0.1"),
        ("tol", np.inf),
        ("max_iter", 0),
        ("max_iter", 2.5),
        ("max_iter", True),
        ("restarts", -1),
        ("restarts", 2.5),
        ("restarts", True),
        ("seed", -1),
        ("seed", 2.5),
        ("seed", True),
    ],
)
def test_ik_invalid_options(key: str, value: Any) -> None:
    with pytest.raises(ValueError, match=key):
        SerialArm.from_dict(planar_config()).inverse_kinematics(
            [1, 0, 0], **{key: value}
        )


@pytest.mark.parametrize("value", [0, -1, True, 2.5, np.nan, "10"])
def test_workspace_invalid_count(value: Any) -> None:
    with pytest.raises(ValueError, match="n_samples"):
        SerialArm.from_dict(planar_config()).workspace(value)


@pytest.mark.parametrize("value", [-1, True, 2.5, None])
def test_workspace_invalid_seed(value: Any) -> None:
    with pytest.raises(ValueError, match="seed"):
        SerialArm.from_dict(planar_config()).workspace(seed=value)


@pytest.mark.parametrize("value", [0, -1, True, "0.002", np.nan, np.inf])
def test_mjcf_invalid_timestep(value: Any) -> None:
    with pytest.raises(ValueError, match="timestep"):
        SerialArm.from_dict(planar_config()).to_mjcf(timestep=value)


def test_mjcf_generation_does_not_import_mujoco() -> None:
    script = """
import sys
sys.modules["mujoco"] = None
from four_dof_arm.serial_arm import SerialArm
arm = SerialArm.from_dict({"units": "SI", "links": [
    {"name": "a", "axis": [0, 0, 1], "offset": [1, 0, 0], "mass": 1}
]})
assert "<mujoco" in arm.to_mjcf()
"""
    subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )


@pytest.mark.parametrize("n", [1, 2, 6])
@pytest.mark.parametrize("payload_mass", [0, 0.3])
def test_generated_mujoco_matches_fk_jacobian_and_physical_properties(
    n: int,
    payload_mass: float,
) -> None:
    mujoco = pytest.importorskip("mujoco")
    config = spatial_config(n)
    config["payload_mass"] = payload_mass
    arm = SerialArm.from_dict(config)
    xml = arm.to_mjcf(timestep=0.003)
    root = ET.fromstring(xml)
    assert root.attrib["model"] == arm.name
    compiler = root.find("compiler")
    assert compiler is not None
    assert compiler.attrib == {
        "angle": "radian",
        "autolimits": "true",
        "inertiafromgeom": "true",
    }
    assert (root.find(".//geom[@name='payload']") is not None) == (payload_mass > 0)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    assert model.nq == model.nv == model.nu == n
    assert model.opt.timestep == 0.003
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    np.testing.assert_array_equal(model.opt.gravity, [0, 0, -9.81])
    assert np.all(model.geom_contype == 0)
    assert np.all(model.geom_conaffinity == 0)
    assert np.all(model.body_inertia[1:] > 0)
    expected_masses = np.array([link.mass for link in arm.links])
    expected_masses[-1] += payload_mass
    np.testing.assert_allclose(model.body_mass[1:], expected_masses)
    np.testing.assert_allclose(model.jnt_range, arm.joint_limits)
    for index, link in enumerate(arm.links):
        joint = model.joint(link.name)
        actuator = model.actuator(f"motor_{index}")
        assert int(actuator.trnid[0]) == joint.id
        np.testing.assert_allclose(actuator.gear, [1, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(
            actuator.ctrlrange, [-link.torque_limit, link.torque_limit]
        )
        assert model.actuator_ctrllimited[actuator.id]
        np.testing.assert_allclose(joint.axis, link.axis)
        assert model.dof_damping[index] == link.damping
        assert model.dof_armature[index] == link.armature
        if not any(link.offset):
            assert model.geom(f"geom_{index}").type == mujoco.mjtGeom.mjGEOM_SPHERE
    site_id = model.site("end_effector").id
    rng = np.random.default_rng(61)
    for q in rng.uniform(*arm.joint_limits.T, size=(8, n)):
        for index, link in enumerate(arm.links):
            data.qpos[model.joint(link.name).qposadr[0]] = q[index]
        mujoco.mj_forward(model, data)
        position, rotation = arm.forward_kinematics(q)
        np.testing.assert_allclose(data.site_xpos[site_id], position, atol=2e-14)
        np.testing.assert_allclose(
            data.site_xmat[site_id].reshape(3, 3), rotation, atol=2e-14
        )
        jacobian = np.zeros((3, n))
        mujoco.mj_jacSite(model, data, jacobian, None, site_id)
        np.testing.assert_allclose(jacobian, arm.jacobian(q), atol=2e-14)
        for index in range(n):
            np.testing.assert_allclose(
                data.body(f"body_{index}").xpos,
                arm.joint_positions(q)[index],
                atol=2e-14,
            )
        assert data.ncon == 0
