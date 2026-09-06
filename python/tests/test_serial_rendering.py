from __future__ import annotations

import builtins
import importlib
import sys
import time
import xml.etree.ElementTree as ET
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from four_dof_arm.serial_arm import Link, SerialArm, rendering
from four_dof_arm.serial_arm.model import FloatArray

mujoco = pytest.importorskip("mujoco")


def make_arm(n: int = 2) -> SerialArm:
    return SerialArm(
        tuple(
            Link(
                f"joint_{i}",
                axis=(0, 0, 1) if i % 2 == 0 else (0, 1, 0),
                offset=(0.2, 0.01, 0.04),
                mass=0.3,
            )
            for i in range(n)
        ),
        name="Spatial arm",
        base=(0.1, -0.2, 0.3),
        tool=(0.04, 0.01, 0.02),
        payload_mass=0.05,
    )


def make_record(n: int = 2, duration: float = 0.4) -> dict[str, FloatArray]:
    return {
        "t": duration * np.array([0, 0.2, 0.5, 0.85, 1]),
        "q": np.outer([0, 0.7, 0, -0.4, 0], np.linspace(0.7, 1, n)),
        "target_q": np.full((5, n), 2.0),
        "ee_pos": np.full((5, 3), 100.0),
    }


class FakeRenderer:
    def __init__(self, model: Any, *, height: int, width: int):
        self.model = model
        self.height, self.width = height, width
        self.closed = False
        self.samples: list[tuple[float, FloatArray, FloatArray]] = []

    def update_scene(self, data: Any, *, camera: Any) -> None:
        assert camera.azimuth == 135 and camera.elevation == -22
        self.samples.append((data.time, data.qpos.copy(), data.site_xpos[0].copy()))

    def render(self) -> Any:
        return np.full(
            (self.height, self.width, 3), len(self.samples) * 13, dtype=np.uint8
        )

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def renderers(monkeypatch: pytest.MonkeyPatch) -> list[FakeRenderer]:
    instances: list[FakeRenderer] = []

    def create(model: Any, *, height: int, width: int) -> FakeRenderer:
        renderer = FakeRenderer(model, height=height, width=width)
        instances.append(renderer)
        return renderer

    monkeypatch.setattr(mujoco, "Renderer", create)

    def no_physics(*args: Any, **kwargs: Any) -> None:
        pytest.fail("replay must not integrate or recompute dynamics")

    monkeypatch.setattr(mujoco, "mj_step", no_physics)
    monkeypatch.setattr(mujoco, "mj_forward", no_physics)
    return instances


@pytest.mark.parametrize("n", [2, 12])
@pytest.mark.parametrize("dynamic", [False, True])
def test_render_actual_interpolated_motion_and_bent_snapshot(
    n: int,
    dynamic: bool,
    tmp_path: Path,
    renderers: list[FakeRenderer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Image = pytest.importorskip("PIL.Image")
    ImageDraw = pytest.importorskip("PIL.ImageDraw")
    text: list[str] = []
    original = ImageDraw.ImageDraw.text

    def capture(self: Any, xy: Any, value: str, **kwargs: Any) -> Any:
        text.append(value)
        return original(self, xy, value, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture)
    arm, record = make_arm(n), make_record(n)
    if dynamic:
        record["torque"] = np.ones_like(record["q"])
    before = {key: value.copy() for key, value in record.items()}
    output = tmp_path / "render"
    rendering.render_3d(arm, record, output)
    assert {path.name for path in output.iterdir()} == {
        "simulation_3d.png",
        "simulation_3d.gif",
        "scene.xml",
    }
    renderer = renderers[0]
    assert renderer.closed
    assert (renderer.width, renderer.height) == (960, 720)
    times = [sample[0] for sample in renderer.samples]
    assert times[0] == record["t"][1]  # Midpoint is straight, not representative.
    np.testing.assert_allclose(times[1:], np.arange(8) / 20)
    for elapsed, q, tip in renderer.samples:
        expected = np.array(
            [np.interp(elapsed, record["t"], col) for col in record["q"].T]
        )
        np.testing.assert_allclose(q, expected)
        np.testing.assert_allclose(tip, arm.forward_kinematics(expected)[0], atol=1e-13)
    with Image.open(output / "simulation_3d.png") as image:
        assert image.size == (960, 720)
        assert image.getpixel((500, 400)) == (13, 13, 13)
        assert image.getpixel((0, 0)) == (15, 23, 33)
    with Image.open(output / "simulation_3d.gif") as image:
        assert image.size == (960, 720)
        assert image.n_frames == 8
        assert image.info["loop"] == 0
        for index in range(image.n_frames):
            image.seek(index)
            assert image.info["duration"] == 50
    assert f"{arm.name} | {n} joints" in text
    mode = "Physics replay" if dynamic else "Kinematic replay"
    assert f"{mode} | t = 0.080 s" in text
    assert "Contacts disabled | Approximate rigid-body model" in text
    assert mode in ET.parse(output / "scene.xml").getroot().attrib["model"]
    for key in record:
        np.testing.assert_array_equal(record[key], before[key])


@pytest.mark.parametrize(
    "duration,count", [(0.01, 1), (0.126, 3), (0.3, 6), (0.1 + 0.2, 6)]
)
def test_gif_endpoint_and_frame_timing(
    duration: float, count: int, tmp_path: Path, renderers: list[FakeRenderer]
) -> None:
    Image = pytest.importorskip("PIL.Image")
    rendering.render_3d(make_arm(), make_record(duration=duration), tmp_path)
    samples = renderers[0].samples[1:]
    np.testing.assert_allclose([sample[0] for sample in samples], np.arange(count) / 20)
    assert samples[-1][0] < duration
    with Image.open(tmp_path / "simulation_3d.gif") as image:
        assert image.n_frames == count
        assert image.info["duration"] == 50
    assert duration <= count / 20 + 1e-15 < duration + 0.05 + 1e-15


@pytest.mark.parametrize("n", [2, 12])
def test_scene_preserves_inertias_fk_and_encloses_entire_motion(n: int) -> None:
    arm, record = make_arm(n), make_record(n)
    model, data, camera, xml = rendering._scene(arm, record)
    baseline = mujoco.MjModel.from_xml_string(arm.to_mjcf())
    root = ET.fromstring(xml)
    for field in (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "dof_armature",
    ):
        np.testing.assert_allclose(
            getattr(model, field), getattr(baseline, field), atol=1e-14
        )
    assert model.nbody == baseline.nbody and model.njnt == n
    for index in range(n):
        marker = root.find(f".//geom[@name='joint_marker_{index}']")
        assert marker is not None and marker.attrib["mass"] == "0"
        assert model.geom(f"geom_{index}").type == mujoco.mjtGeom.mjGEOM_CAPSULE
    assert np.all(model.geom_contype == 0) and np.all(model.geom_conaffinity == 0)
    floor_z = model.geom("visual_floor").pos[2]
    mount = root.find(".//geom[@name='visual_mount']")
    assert mount is not None and mount.attrib["mass"] == "0"
    mount_ends = np.fromstring(mount.attrib["fromto"], sep=" ").reshape(2, 3)
    np.testing.assert_allclose(mount_ends[0], [*arm.base[:2], floor_z])
    np.testing.assert_allclose(mount_ends[1], arm.base)
    # A sphere inside the camera's vertical cone also fits its wider horizontal cone.
    visible_radius = camera.distance * np.sin(np.deg2rad(model.vis.global_.fovy / 2))
    for elapsed in np.linspace(0, record["t"][-1], 81):
        rendering._pose(model, data, record, float(elapsed))
        points = arm.joint_positions(data.qpos)
        np.testing.assert_allclose(data.xpos[1:], points[:-2], atol=1e-13)
        np.testing.assert_allclose(data.site_xpos[0], points[-1], atol=1e-13)
        np.testing.assert_allclose(data.light_xpos, model.light_pos, atol=1e-13)
        np.testing.assert_allclose(data.light_xdir, model.light_dir, atol=1e-13)
        assert (
            floor_z < points[:, 2].min() - max(link.radius for link in arm.links) * 1.35
        )
        assert (
            np.max(np.linalg.norm(points - camera.lookat, axis=1)) + 0.021
            < visible_radius
        )
    np.testing.assert_array_equal(data.qpos, record["q"][-1])


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"t": np.array([0]), "q": np.zeros((1, 2))},
        {"t": np.array(0), "q": np.zeros((2, 2))},
        {"t": np.array([[0, 1]]), "q": np.zeros((2, 2))},
        {"t": np.array([0.1, 1]), "q": np.zeros((2, 2))},
        {"t": np.array([0, 0]), "q": np.zeros((2, 2))},
        {"t": np.array([0, -1]), "q": np.zeros((2, 2))},
        {"t": np.array([0, np.inf]), "q": np.zeros((2, 2))},
        {"t": np.array([0, np.nan]), "q": np.zeros((2, 2))},
        {"t": np.array([0, 1]), "q": np.zeros((2, 3))},
        {"t": np.array([0, 1]), "q": np.zeros((3, 2))},
        {"t": np.array([0, 1]), "q": np.zeros(2)},
        {"t": np.array([0, 1]), "q": np.full((2, 2), np.nan)},
        {"t": np.array([0, 1]), "q": np.full((2, 2), np.inf)},
        {"t": np.array([0, 1]), "q": np.array([["bad"]])},
    ],
)
def test_common_record_validation(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="record"):
        rendering._scene(make_arm(), bad)


@pytest.mark.parametrize("failure", ["render", "png", "gif"])
def test_render_cleanup_on_failure(
    failure: str,
    tmp_path: Path,
    renderers: list[FakeRenderer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Image = pytest.importorskip("PIL.Image")
    frames: list[Any] = []
    original_quantize, original_save = Image.Image.quantize, Image.Image.save

    def quantize(self: Any, *args: Any, **kwargs: Any) -> Any:
        frame = original_quantize(self, *args, **kwargs)
        frames.append(frame)
        return frame

    def save(self: Any, path: Path, **kwargs: Any) -> None:
        if path.suffix == f".{failure}":
            raise OSError("encoding failed")
        original_save(self, path, **kwargs)

    def render(self: FakeRenderer) -> Any:
        if len(self.samples) == 3:
            raise OSError("render failed")
        return np.zeros((720, 960, 3), dtype=np.uint8)

    monkeypatch.setattr(Image.Image, "quantize", quantize)
    monkeypatch.setattr(Image.Image, "save", save)
    if failure == "render":
        monkeypatch.setattr(FakeRenderer, "render", render)
    with pytest.raises(OSError, match="failed"):
        rendering.render_3d(make_arm(), make_record(), tmp_path)
    assert renderers[0].closed
    for frame in frames:
        with pytest.raises(ValueError, match="closed"):
            frame.getpixel((0, 0))


@pytest.mark.parametrize("fail_sync", [False, True])
def test_viewer_realtime_loop_mouse_camera_and_import_isolation(
    fail_sync: bool,
    monkeypatch: pytest.MonkeyPatch,
    renderers: list[FakeRenderer],
    tmp_path: Path,
) -> None:
    import mujoco.viewer

    arm, record = make_arm(12), make_record(12)
    record["torque"] = np.ones_like(record["q"])
    snapshots: list[tuple[float, FloatArray]] = []
    clock = [10.0]
    sleeps: list[float] = []

    class Viewer:
        cam = mujoco.MjvCamera()
        closed = False
        data: Any = None

        def __enter__(self) -> Viewer:
            return self

        def __exit__(self, *args: Any) -> None:
            self.closed = True

        def lock(self) -> Any:
            return nullcontext()

        def is_running(self) -> bool:
            return len(snapshots) < 28

        def sync(self) -> None:
            assert self.cam.azimuth == (135 if not snapshots else 71)
            self.cam.azimuth = 71  # Native mouse orbit must not be reset each frame.
            snapshots.append((self.data.time, self.data.qpos.copy()))
            clock[0] += 0.004
            if fail_sync:
                raise RuntimeError("sync failed")

    viewer = Viewer()

    def launch(model: Any, data: Any) -> Viewer:
        viewer.data = data
        np.testing.assert_array_equal(data.qpos, record["q"][0])
        assert b"Physics replay" in model.names.split(b"\x00")[0]
        return viewer

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    original_import = builtins.__import__

    def isolated_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "PIL" or name.startswith("PIL."):
            raise ModuleNotFoundError("Pillow unavailable", name="PIL")
        if name.startswith(("matplotlib", "four_dof_arm.mujoco_sim")):
            pytest.fail(f"unexpected replay dependency: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", sleep)
    monkeypatch.setattr(builtins, "__import__", isolated_import)
    importlib.reload(rendering)
    if fail_sync:
        with pytest.raises(RuntimeError, match="sync failed"):
            rendering.show_3d(arm, record)
    else:
        rendering.show_3d(arm, record)
        np.testing.assert_allclose(sleeps, 1 / 60 - 0.004, atol=1e-14)
        np.testing.assert_allclose(
            [sample[0] for sample in snapshots[:24]], np.arange(24) / 60, atol=1e-13
        )
        assert snapshots[-1][0] < snapshots[23][0]
    assert viewer.closed
    assert not renderers
    for elapsed, q in snapshots:
        np.testing.assert_allclose(
            q, [np.interp(elapsed, record["t"], col) for col in record["q"].T]
        )
    with pytest.raises(ModuleNotFoundError, match="Pillow unavailable"):
        rendering.render_3d(arm, record, tmp_path)


def test_module_import_does_not_require_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "mujoco",
        "mujoco.viewer",
        "PIL",
        "matplotlib",
        "four_dof_arm.mujoco_sim",
    ):
        monkeypatch.setitem(sys.modules, name, None)
    importlib.reload(rendering)
