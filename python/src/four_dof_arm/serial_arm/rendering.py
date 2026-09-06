"""Solid-link visualization of recorded motion, without advancing physics."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from .model import FloatArray, SerialArm


def _scene(arm: SerialArm, record: dict[str, FloatArray]) -> tuple[Any, Any, Any, str]:
    try:
        t, q = (np.asarray(record[key], dtype=float) for key in ("t", "q"))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("record requires numeric t and q arrays") from exc
    if (
        t.ndim != 1
        or len(t) < 2
        or q.shape != (len(t), arm.n_joints)
        or not np.isfinite(t).all()
        or not np.isfinite(q).all()
        or t[0] != 0
        or not np.all(np.diff(t) > 0)
    ):
        raise ValueError(
            "record needs finite t (N,) starting at 0, increasing, and q (N, joints); N >= 2"
        )

    import mujoco

    points = np.array([arm.joint_positions(pose) for pose in q])
    lengths = np.array([np.linalg.norm(link.offset) for link in arm.links])
    reach = np.cumsum(lengths[::-1])[::-1] + np.linalg.norm(arm.tool)
    # Nearest-endpoint arc bounds also enclose poses between recorded samples.
    padding = float(np.max(np.abs(np.diff(q, axis=0)) @ reach)) / 2
    padding += max(0.008, max(link.radius for link in arm.links) * 1.35)
    lower, upper = points.min(axis=(0, 1)) - padding, points.max(axis=(0, 1)) + padding
    center = (lower + upper) / 2
    radius = float(np.linalg.norm(upper - lower)) / 2
    floor_z = lower[2] - radius * 0.15
    mode = "Physics replay" if "torque" in record else "Kinematic replay"
    root = ET.fromstring(arm.to_mjcf())
    root.set(
        "model",
        f"{arm.name} | {arm.n_joints} joints | {mode} | Contacts disabled | Approximate rigid-body model",
    )
    ET.SubElement(
        root, "statistic", center=" ".join(map(str, center)), extent=str(2 * radius)
    )
    root.extend(
        ET.fromstring("""<style>
          <visual>
            <global offwidth="960" offheight="720" fovy="45"/>
            <quality shadowsize="2048" offsamples="4"/>
            <headlight ambient="0.3 0.3 0.3" diffuse="0.6 0.6 0.6"
                       specular="0.2 0.2 0.2"/>
          </visual>
          <asset>
            <texture name="sky" type="skybox" builtin="gradient" width="512"
                     height="3072" rgb1="0.16 0.22 0.30" rgb2="0.04 0.06 0.10"/>
            <texture name="tiles" type="2d" builtin="checker" width="256"
                     height="256" rgb1="0.30 0.35 0.41" rgb2="0.34 0.39 0.45"/>
            <material name="floor" texture="tiles" texrepeat="3 3"
                      texuniform="true" reflectance="0.08"/>
          </asset>
        </style>""")
    )
    world = root.find("worldbody")
    assert world is not None
    ET.SubElement(
        world,
        "light",
        pos="-2 -3 5",
        dir="1 1 -2",
        directional="true",
        diffuse="0.8 0.8 0.8",
    )
    ET.SubElement(
        world,
        "light",
        pos="2 1 3",
        dir="-1 -1 -2",
        directional="true",
        diffuse="0.4 0.45 0.5",
        castshadow="false",
    )
    ET.SubElement(
        world,
        "geom",
        name="visual_floor",
        type="plane",
        pos=f"0 0 {floor_z}",
        size="0 0 0.05",
        material="floor",
        mass="0",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(
        world,
        "geom",
        name="visual_mount",
        type="cylinder",
        fromto=" ".join(map(str, (*arm.base[:2], floor_z, *arm.base))),
        size=str(max(arm.links[0].radius * 1.5, radius * 0.025)),
        mass="0",
        rgba="0.15 0.20 0.26 1",
    )
    colors = ("0.08 0.55 0.70 1", "0.95 0.52 0.15 1", "0.75 0.81 0.86 1")
    for index, link in enumerate(arm.links):
        body = world.find(f".//body[@name='body_{index}']")
        geom = world.find(f".//geom[@name='geom_{index}']")
        assert body is not None and geom is not None
        geom.set("rgba", colors[index % len(colors)])
        ET.SubElement(
            body,
            "geom",
            name=f"joint_marker_{index}",
            type="sphere",
            size=str(link.radius * 1.35),
            mass="0",
            rgba="0.07 0.10 0.14 1",
        )
    if any(arm.tool):
        assert body is not None
        start = np.array(arm.links[-1].offset)
        ET.SubElement(
            body,
            "geom",
            name="visual_tool",
            type="capsule",
            fromto=" ".join(map(str, np.concatenate((start, start + arm.tool)))),
            size=str(arm.links[-1].radius * 0.5),
            mass="0",
            rgba="0.3 0.75 0.95 1",
        )
    ET.indent(root)
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = center
    camera.distance = 1.3 * radius / np.sin(np.deg2rad(model.vis.global_.fovy / 2))
    camera.azimuth, camera.elevation = 135, -22
    return model, data, camera, xml


def _pose(model: Any, data: Any, record: dict[str, FloatArray], elapsed: float) -> None:
    import mujoco

    data.qpos[:] = [
        np.interp(elapsed, record["t"], column) for column in np.asarray(record["q"]).T
    ]
    data.time = elapsed
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)


def render_3d(arm: SerialArm, record: dict[str, FloatArray], output_dir: Path) -> None:
    """Write a bent-pose PNG, a 20 fps GIF (excluding the endpoint), and MJCF."""
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    model, data, camera, xml = _scene(arm, record)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scene.xml").write_text(xml, encoding="utf-8")
    mode = "Physics replay" if "torque" in record else "Kinematic replay"
    font = ImageFont.load_default()
    frames: list[Image.Image] = []
    renderer = mujoco.Renderer(model, height=720, width=960)
    try:
        bent = int(np.argmax(np.linalg.norm(record["q"] - record["q"][0], axis=1)))
        count = max(1, int(np.ceil(np.nextafter(float(record["t"][-1]) * 20, -np.inf))))
        frame_times = np.arange(count) / 20
        for index, elapsed in enumerate([float(record["t"][bent]), *frame_times]):
            _pose(model, data, record, float(elapsed))
            view = camera
            if index == 0:
                points = arm.joint_positions(record["q"][bent])
                view = mujoco.MjvCamera()
                view.lookat[:] = (points.min(axis=0) + points.max(axis=0)) / 2
                span = float(np.linalg.norm(np.ptp(points, axis=0)))
                view.distance = (
                    max(span, 4 * max(link.radius for link in arm.links)) * 1.8
                )
                view.azimuth, view.elevation = camera.azimuth, camera.elevation
            renderer.update_scene(data, camera=view)
            with Image.fromarray(renderer.render()) as image:
                with Image.new("RGB", (480, 48), (15, 23, 33)) as label:
                    draw = ImageDraw.Draw(label)
                    title = f"{arm.name} | {arm.n_joints} joints".encode(
                        "ascii", "replace"
                    ).decode()
                    draw.text((10, 4), title, font=font, fill="white")
                    draw.text(
                        (10, 18),
                        f"{mode} | t = {elapsed:.3f} s",
                        font=font,
                        fill="#76cce8",
                    )
                    draw.text(
                        (10, 32),
                        "Contacts disabled | Approximate rigid-body model",
                        font=font,
                        fill="#c1c9d4",
                    )
                    with label.resize((960, 96)) as banner:
                        image.paste(banner, (0, 0))
                if index == 0:
                    image.save(output_dir / "simulation_3d.png")
                else:
                    frames.append(image.quantize())
        frames[0].save(
            output_dir / "simulation_3d.gif",
            save_all=True,
            append_images=frames[1:],
            duration=50,
            loop=0,
            optimize=False,
            disposal=2,
        )
    finally:
        try:
            renderer.close()
        finally:
            for frame in frames:
                frame.close()


def show_3d(arm: SerialArm, record: dict[str, FloatArray]) -> None:
    """Loop recorded motion in the native viewer; macOS callers must use mjpython."""
    import mujoco.viewer

    model, data, camera, _ = _scene(arm, record)
    _pose(model, data, record, 0.0)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = camera.lookat
            viewer.cam.distance = camera.distance
            viewer.cam.azimuth, viewer.cam.elevation = camera.azimuth, camera.elevation
        start = time.monotonic()
        while viewer.is_running():
            tick = time.monotonic()
            with viewer.lock():
                _pose(model, data, record, (tick - start) % float(record["t"][-1]))
            viewer.sync()
            time.sleep(max(0.0, 1 / 60 - (time.monotonic() - tick)))
