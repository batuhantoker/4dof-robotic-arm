# /// script
# requires-python = ">=3.10"
# dependencies = ["cadquery-ocp>=7.8", "mujoco>=3.1", "numpy>=1.24", "pillow>=10.1"]
# ///
"""Convert the repository's STEP assembly and render a kinematic CAD preview."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAMESPACE = {"s": "urn:mathworks:SimscapeMultibody:import"}


def build_model(source: Path, output: Path) -> Path:
    """Build a visual-only MJCF at the exported CAD assembly pose."""
    assembly = ET.parse(source / "Montaj1.xml").getroot()
    instances = assembly.findall("s:RootAssembly/s:InstanceTree/s:Instance", NAMESPACE)
    parts = assembly.findall("s:Parts/s:Part", NAMESPACE)
    transforms = {}
    for instance in instances:
        transform = np.eye(4)
        transform[:3, :3] = np.fromstring(
            instance.findtext("s:Transform/s:Rotation", namespaces=NAMESPACE), sep=" "
        ).reshape(3, 3)
        transform[:3, 3] = np.fromstring(
            instance.findtext("s:Transform/s:Translation", namespaces=NAMESPACE),
            sep=" ",
        )
        transforms[instance.attrib["entityUid"]] = transform

    by_name = {part.attrib["name"]: part for part in parts}
    by_uid = {
        instance.attrib["uid"]: instance.attrib["entityUid"] for instance in instances
    }
    joints = {}
    for constraint in assembly.findall(
        "s:RootAssembly/s:Constraints/s:Concentric", NAMESPACE
    ):
        parent, child = constraint.findall("s:ConstraintGeometry", NAMESPACE)
        child_uid = by_uid[child.findtext("s:InstancePath/s:Uid", namespaces=NAMESPACE)]
        joints[child_uid] = (
            by_uid[parent.findtext("s:InstancePath/s:Uid", namespaces=NAMESPACE)],
            child.findtext("s:Position", namespaces=NAMESPACE),
            child.findtext("s:Axis", namespaces=NAMESPACE),
        )

    def numbers(values: np.ndarray) -> str:
        return " ".join(f"{value:.16g}" for value in values.flat)

    # CAD uses Y-up; retain the base footprint and place its yaw axis on world Z.
    world_from_base = np.array(
        [
            [-1.0, 0.0, 0.0, 0.025],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    base_uid = by_name["link0"].attrib["uid"]
    world_from_export = world_from_base @ np.linalg.inv(transforms[base_uid])
    model = ET.Element("mujoco", model="CAD assembly - kinematic preview only")
    ET.SubElement(model, "compiler", angle="radian", meshdir="meshes")
    ET.SubElement(model, "option", gravity="0 0 0")
    visual = ET.SubElement(model, "visual")
    ET.SubElement(visual, "global", offwidth="1200", offheight="900")
    ET.SubElement(visual, "headlight", ambient="0.35 0.35 0.35", diffuse="0.7 0.7 0.7")
    asset = ET.SubElement(model, "asset")
    world = ET.SubElement(model, "worldbody")
    ET.SubElement(world, "light", pos="0 -0.3 0.8", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(
        world,
        "geom",
        type="plane",
        size="1 1 0.01",
        pos="0 0 -0.003",
        rgba="0.88 0.90 0.93 1",
        contype="0",
        conaffinity="0",
    )
    bodies = {}
    for index in range(5):
        name = f"link{index}"
        part = by_name[name]
        uid = part.attrib["uid"]
        ET.SubElement(
            asset, "mesh", name=name, file=f"{name}.stl", scale="0.001 0.001 0.001"
        )
        if index == 0:
            parent_body = world
            relative = world_from_export @ transforms[uid]
        else:
            parent_uid, anchor, axis = joints[uid]
            parent_body = bodies[parent_uid]
            relative = np.linalg.inv(transforms[parent_uid]) @ transforms[uid]
        body = ET.SubElement(
            parent_body,
            "body",
            name=name,
            pos=numbers(relative[:3, 3]),
            xyaxes=numbers(relative[:3, :2].T),
        )
        bodies[uid] = body
        if index:
            ET.SubElement(
                body,
                "joint",
                name=f"cad_joint{index}",
                type="hinge",
                pos=anchor,
                axis=axis,
                limited="false",
            )
            # Placeholder inertias permit articulation; this model is never stepped.
            ET.SubElement(body, "inertial", pos="0 0 0", mass="1", diaginertia="1 1 1")
        color = part.find("s:VisualProperties/s:Diffuse", NAMESPACE)
        rgba = " ".join(color.attrib[channel] for channel in ("r", "g", "b", "a"))
        ET.SubElement(
            body,
            "geom",
            name=f"cad_{name}",
            type="mesh",
            mesh=name,
            rgba=rgba,
            contype="0",
            conaffinity="0",
            mass="0",
        )
        if index == 4:
            # Pointed tool vertex #603 in link4_Varsayilan_sldprt.STEP, in metres.
            ET.SubElement(
                body,
                "site",
                name="tool_tip",
                pos="-0.00275 0 0",
                size="0.002",
                rgba="0.1 0.8 1 1",
            )

    ET.indent(model)
    model_path = output / "cad_preview.xml"
    ET.ElementTree(model).write(model_path, encoding="utf-8", xml_declaration=True)
    return model_path


def convert_meshes(source: Path, output: Path) -> None:
    """Tessellate every STEP solid, retaining the original millimetre part frames."""
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.StlAPI import StlAPI_Writer

    mesh_dir = output / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for index in range(5):
        reader = STEPControl_Reader()
        step = source / f"link{index}_Varsayilan_sldprt.STEP"
        if reader.ReadFile(str(step)) != IFSelect_RetDone or not reader.TransferRoots():
            raise RuntimeError(f"Could not import {step}")
        shape = reader.OneShape()
        mesher = BRepMesh_IncrementalMesh(shape, 0.15, False, 0.3, True)
        if not mesher.IsDone():
            raise RuntimeError(f"Could not tessellate {step}")
        writer = StlAPI_Writer()
        writer.ASCIIMode = False
        destination = mesh_dir / f"link{index}.stl"
        if not writer.Write(shape, str(destination)):
            raise RuntimeError(f"Could not write {destination}")
        print(f"Converted {step.name}: {destination.stat().st_size / 1024:.0f} KiB")


def motion_cases() -> dict[str, tuple[np.ndarray, ...]]:
    """Sample illustrative joint offsets at 20 Hz; these are not hardware limits."""
    waypoints = {}
    for joint, (name, amplitude) in enumerate(
        zip(("base_yaw", "shoulder", "elbow", "wrist"), (60, 20, 30, 45), strict=True)
    ):
        points = np.zeros((5, 4))
        points[:, joint] = [0, amplitude, 0, -amplitude, 0]
        waypoints[name] = (points, 1.5)
    waypoints["coordinated"] = (
        np.array(
            [
                [0, 0, 0, 0],
                [35, -15, 20, -25],
                [0, 0, 0, 0],
                [-35, 15, -20, 25],
                [0, 0, 0, 0],
            ]
        ),
        1.5,
    )
    waypoints["transfer_return"] = (
        np.array(
            [
                [0, 0, 0, 0],
                [35, -15, 20, -20],
                [35, 10, -15, 20],
                [-35, 10, -15, 20],
                [-35, -15, 20, -20],
                [0, 0, 0, 0],
            ]
        ),
        2.0,
    )
    cases = {}
    for name, (points, segment_duration) in waypoints.items():
        points = np.deg2rad(points)
        segment_samples = round(segment_duration * 20)
        sample = np.arange((len(points) - 1) * segment_samples + 1)
        times = sample / 20.0
        segment = np.minimum(sample // segment_samples, len(points) - 2)
        u = (sample - segment * segment_samples)[:, None] / segment_samples
        delta = points[segment + 1] - points[segment]
        q = points[segment] + delta * u**3 * (10 + u * (-15 + 6 * u))
        qd = delta * 30 * u**2 * (1 - u) ** 2 / segment_duration
        qdd = delta * 60 * u * (1 - u) * (1 - 2 * u) / segment_duration**2
        cases[name] = (times, q, qd, qdd)
    return cases


def tool_trajectory(
    model, q: np.ndarray, qd: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Measure the CAD tool vertex in world millimetres and millimetres/second."""
    import mujoco

    data = mujoco.MjData(model)
    site_id = model.site("tool_tip").id
    positions = np.empty((len(q), 3))
    velocities = np.empty_like(positions)
    jacobian = np.empty((3, model.nv))
    for index, (position, velocity) in enumerate(zip(q, qd, strict=True)):
        data.qpos[:] = position
        mujoco.mj_forward(model, data)
        mujoco.mj_jacSite(model, data, jacobian, None, site_id)
        positions[index] = data.site_xpos[site_id] * 1000
        velocities[index] = jacobian @ velocity * 1000
    return positions, velocities


def render_preview(
    model_path: Path,
    output: Path,
    *,
    name: str = "cad_preview",
    motion: tuple[np.ndarray, ...] | None = None,
) -> dict:
    """Render the imported pose and an illustrative, non-dynamic joint sweep."""
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    if motion is None:
        times = np.linspace(0, 4, 81)
        frequency = 2 * np.pi / 4
        phase = frequency * times[:, None]
        amplitude = np.deg2rad([25, 12, 18, 20])
        motion = (
            times,
            amplitude * np.sin(phase),
            amplitude * frequency * np.cos(phase),
            -amplitude * frequency**2 * np.sin(phase),
        )
    times, q, qd, qdd = motion
    positions, velocities = tool_trajectory(model, q, qd)
    speed = np.linalg.norm(velocities, axis=1)
    camera = mujoco.MjvCamera()
    camera.azimuth = 135
    camera.elevation = -20
    camera.distance = 0.8
    camera.lookat[:] = [0.04, 0.0, 0.14]
    font = ImageFont.load_default(size=18)
    frames = []
    snapshot = 0 if name == "cad_preview" else int(np.argmax(np.linalg.norm(q, axis=1)))
    with mujoco.Renderer(model, width=960, height=720) as renderer:
        for index, time in enumerate(times):
            data.qpos[:] = q[index]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera)
            for start, end in zip(
                positions[:index], positions[1 : index + 1], strict=True
            ):
                if np.linalg.norm(end - start) < 1e-10:
                    continue
                geom = renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    np.zeros(3),
                    np.zeros(3),
                    np.eye(3).ravel(),
                    np.array([0.1, 0.65, 0.9, 1]),
                )
                mujoco.mjv_connector(
                    geom, mujoco.mjtGeom.mjGEOM_LINE, 3, start / 1000, end / 1000
                )
                renderer.scene.ngeom += 1
            image = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 960, 88), fill=(24, 31, 43))
            draw.text(
                (20, 10),
                f"Original CAD assembly | {name.replace('_', ' ').title()}",
                font=font,
                fill="white",
            )
            draw.text(
                (20, 34),
                "Kinematic preview only - no dynamics or collision validation",
                font=font,
                fill=(185, 199, 215),
            )
            draw.text(
                (20, 60),
                f"t={time:.2f}s  |  tool speed={speed[index]:.1f} mm/s  |  joint offsets (deg): "
                + ", ".join(f"{angle:.1f}" for angle in np.rad2deg(q[index])),
                font=font,
                fill=(185, 199, 215),
            )
            if index == snapshot:
                image.save(output / f"{name}.png")
            frames.append(image)
    frames[0].save(
        output / f"{name}.gif",
        save_all=True,
        # The terminal sample closes the trajectory without adding a loop hold.
        append_images=frames[1:-1],
        duration=50,
        loop=0,
    )
    columns = ["time_s"]
    for suffix in ("offset_deg", "velocity_deg_s", "acceleration_deg_s2"):
        columns.extend(f"joint{index}_{suffix}" for index in range(1, 5))
    columns.extend(["tool_x_mm", "tool_y_mm", "tool_z_mm", "tool_speed_mm_s"])
    np.savetxt(
        output / f"{name}.csv",
        np.column_stack(
            (times, np.rad2deg(q), np.rad2deg(qd), np.rad2deg(qdd), positions, speed)
        ),
        delimiter=",",
        header=",".join(columns),
        comments="",
    )
    summary = {
        "scenario": name,
        "mode": "kinematic_only",
        "collision_checked": False,
        "hardware_limits_checked": False,
        "duration_s": float(times[-1]),
        "samples": len(times),
        "sampled_tool_path_length_mm": float(
            np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
        ),
        "sampled_peak_tool_speed_mm_s": float(speed.max()),
        "tool_height_range_mm": [
            float(positions[:, 2].min()),
            float(positions[:, 2].max()),
        ],
        "return_error_mm": float(np.linalg.norm(positions[-1] - positions[0])),
        "max_abs_joint_offset_deg": np.rad2deg(np.abs(q).max(axis=0)).tolist(),
    }
    (output / f"{name}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Rendered {output / f'{name}.gif'}: "
        f"path={summary['sampled_tool_path_length_mm']:.1f} mm, "
        f"peak speed={summary['sampled_peak_tool_speed_mm_s']:.1f} mm/s"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "python/generated/cad"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--no-render", action="store_true", help="only export meshes and MJCF"
    )
    mode.add_argument(
        "--suite", action="store_true", help="render six measured CAD motion cases"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = ROOT / "matlab"
    convert_meshes(source, args.output_dir)
    model_path = build_model(source, args.output_dir)
    print(f"Exported {model_path}")
    if args.suite:
        from PIL import Image

        runs = args.output_dir / "runs"
        runs.mkdir(exist_ok=True)
        summaries = []
        overview = Image.new("RGB", (1440, 720))
        for index, (name, motion) in enumerate(motion_cases().items()):
            summaries.append(render_preview(model_path, runs, name=name, motion=motion))
            with Image.open(runs / f"{name}.png") as image:
                overview.paste(
                    image.resize((480, 360)), ((index % 3) * 480, (index // 3) * 360)
                )
        overview.save(runs / "overview.png")
        (runs / "summary.json").write_text(
            json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
        )
    elif not args.no_render:
        render_preview(model_path, args.output_dir)


if __name__ == "__main__":
    main()
