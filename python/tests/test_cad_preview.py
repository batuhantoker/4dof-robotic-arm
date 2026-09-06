import runpy
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = runpy.run_path(str(ROOT / "python/scripts/preview_cad.py"))
NS = {"s": "urn:mathworks:SimscapeMultibody:import"}


@pytest.fixture
def cad_model(tmp_path):
    path = SCRIPT["build_model"](ROOT / "matlab", tmp_path)
    xml = ET.parse(path).getroot()
    for mesh in xml.findall("asset/mesh"):
        assert mesh.attrib.pop("file") == f"{mesh.attrib['name']}.stl"
        assert mesh.attrib["scale"] == "0.001 0.001 0.001"
        # Test articulation without requiring OpenCascade or generated STL files.
        mesh.set("vertex", "0 0 0 10 0 0 0 10 0 0 0 10")
    return mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))


def test_cad_preview_preserves_exported_part_placements(cad_model):
    data = mujoco.MjData(cad_model)
    mujoco.mj_forward(cad_model, data)
    expected = [
        [0.025, 0.0, 0.0],
        [0.04876568586670035, -0.01104119024156448, 0.14],
        [0.04876568586670053, -0.01104119024156389, 0.14],
        [0.17172607910203104, -0.03888103438114886, 0.14583081828776262],
        [0.23496147793857652, -0.053198357231155, 0.13804715925566817],
    ]
    for index, position in enumerate(expected):
        np.testing.assert_allclose(data.body(f"link{index}").xpos, position, atol=1e-12)
    assert cad_model.nq == cad_model.njnt == 4
    assert cad_model.nmesh == 5


@pytest.mark.parametrize(
    "q", [np.zeros(4), [0.4, -0.2, 0.3, -0.5], [-0.5, 0.3, -0.4, 0.2]]
)
def test_cad_hinge_axes_remain_coincident_during_motion(cad_model, q):
    data = mujoco.MjData(cad_model)
    data.qpos[:] = q
    mujoco.mj_forward(cad_model, data)
    assembly = ET.parse(ROOT / "matlab/Montaj1.xml").getroot()
    instances = assembly.findall("s:RootAssembly/s:InstanceTree/s:Instance", NS)
    parts = assembly.findall("s:Parts/s:Part", NS)
    names = {part.attrib["uid"]: part.attrib["name"] for part in parts}
    bodies = {
        instance.attrib["uid"]: names[instance.attrib["entityUid"]]
        for instance in instances
    }
    for constraint in assembly.findall("s:RootAssembly/s:Constraints/s:Concentric", NS):
        axes, positions = [], []
        for geometry in constraint.findall("s:ConstraintGeometry", NS):
            body = data.body(
                bodies[geometry.findtext("s:InstancePath/s:Uid", namespaces=NS)]
            )
            rotation = body.xmat.reshape(3, 3)
            axes.append(
                rotation
                @ np.fromstring(geometry.findtext("s:Axis", namespaces=NS), sep=" ")
            )
            positions.append(
                body.xpos
                + rotation
                @ np.fromstring(geometry.findtext("s:Position", namespaces=NS), sep=" ")
            )
        assert abs(np.dot(*axes)) == pytest.approx(1.0, abs=1e-12)
        np.testing.assert_allclose(
            np.cross(positions[1] - positions[0], axes[0]), 0.0, atol=1e-12
        )


def test_cad_preview_has_no_actuation_or_collision_response(cad_model):
    assert cad_model.nu == 0
    np.testing.assert_array_equal(cad_model.opt.gravity, np.zeros(3))
    np.testing.assert_array_equal(cad_model.geom_contype, 0)
    np.testing.assert_array_equal(cad_model.geom_conaffinity, 0)


@pytest.mark.parametrize(
    "name", ["base_yaw", "shoulder", "elbow", "wrist", "coordinated", "transfer_return"]
)
def test_motion_cases_return_to_rest_with_finite_profiles(name):
    times, q, qd, qdd = SCRIPT["motion_cases"]()[name]
    assert q.shape == qd.shape == qdd.shape == (len(times), 4)
    np.testing.assert_allclose(np.diff(times), 0.05, atol=1e-14)
    for profile in (q, qd, qdd):
        assert np.all(np.isfinite(profile))
        np.testing.assert_allclose(profile[[0, -1]], 0.0, atol=1e-14)
    segment_duration = 2.0 if name == "transfer_return" else 1.5
    knots = np.arange(0, len(times), round(segment_duration * 20))
    np.testing.assert_allclose(qd[knots], 0.0, atol=1e-14)
    np.testing.assert_allclose(qdd[knots], 0.0, atol=1e-14)


@pytest.mark.parametrize(
    "joint,name,amplitude",
    [(0, "base_yaw", 60), (1, "shoulder", 20), (2, "elbow", 30), (3, "wrist", 45)],
)
def test_isolated_motion_cases_move_only_selected_joint(joint, name, amplitude):
    _, q, qd, qdd = SCRIPT["motion_cases"]()[name]
    np.testing.assert_allclose(np.delete(q, joint, axis=1), 0.0)
    np.testing.assert_allclose(np.rad2deg(q[[30, 90], joint]), [amplitude, -amplitude])
    assert np.rad2deg(q[15, joint]) == pytest.approx(amplitude / 2)
    assert np.rad2deg(qd[15, joint]) == pytest.approx(amplitude * 1.875 / 1.5)
    assert qdd[15, joint] == pytest.approx(0.0)


def test_tool_trajectory_measures_source_vertex_and_stationary_speed(cad_model):
    q = np.array([[0.0, 0.0, 0.0, 0.0], [0.2, -0.1, 0.3, -0.2]])
    positions, velocities = SCRIPT["tool_trajectory"](cad_model, q, np.zeros_like(q))
    data = mujoco.MjData(cad_model)
    for position, angles in zip(positions, q, strict=True):
        data.qpos[:] = angles
        mujoco.mj_forward(cad_model, data)
        body = data.body("link4")
        expected = body.xpos * 1000 + body.xmat.reshape(3, 3) @ [-2.75, 0.0, 0.0]
        np.testing.assert_allclose(position, expected, atol=1e-12)
    np.testing.assert_array_equal(velocities, 0.0)


def test_tool_velocity_matches_finite_difference(cad_model):
    q = np.array([[0.2, -0.1, 0.3, -0.2]])
    qd = np.array([[0.1, -0.2, 0.3, -0.4]])
    _, velocity = SCRIPT["tool_trajectory"](cad_model, q, qd)
    epsilon = 1e-6
    plus, _ = SCRIPT["tool_trajectory"](cad_model, q + epsilon * qd, qd)
    minus, _ = SCRIPT["tool_trajectory"](cad_model, q - epsilon * qd, qd)
    np.testing.assert_allclose(velocity, (plus - minus) / (2 * epsilon), atol=1e-6)
