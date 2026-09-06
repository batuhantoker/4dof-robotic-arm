"""Run with ``serial-arm-analyze`` or ``python -m four_dof_arm.serial_arm``."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .analysis import analyze
from .model import SerialArm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="serial-arm-analyze",
        description="Generate a simple-link serial arm, motion, and SI analysis reports.",
    )
    parser.add_argument("config", type=Path, help="robot JSON configuration")
    parser.add_argument(
        "--scenario", type=Path, help="JSON duration, joint waypoints and IK targets"
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="new or empty output directory"
    )
    parser.add_argument(
        "--simulate", action="store_true", help="run MuJoCo torque-controlled tracking"
    )
    parser.add_argument(
        "--plots", action="store_true", help="write a Matplotlib overview PNG"
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="write a GIF and overview PNG (no FFmpeg)",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=0.002,
        help="maximum sample/physics step in seconds (default: 0.002)",
    )
    parser.add_argument("--workspace-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists() and (
            not args.output_dir.is_dir() or any(args.output_dir.iterdir())
        ):
            raise ValueError(
                "output directory must be new or empty; existing files are not overwritten"
            )
        arm = SerialArm.from_json(args.config)
        scenario = None
        if args.scenario is not None:
            with args.scenario.open(encoding="utf-8") as stream:
                scenario = json.load(stream)
            if not isinstance(scenario, dict):
                raise ValueError("scenario must be an object")
        render = None
        if args.plots or args.animate:
            from .visualization import render_analysis

            render = render_analysis
        report, record, workspace = analyze(
            arm,
            scenario,
            timestep=args.timestep,
            workspace_samples=args.workspace_samples,
            seed=args.seed,
            simulate=args.simulate,
        )
        report_json = json.dumps(report, indent=2, allow_nan=False) + "\n"
        model_xml = arm.to_mjcf(timestep=report["sampling"]["timestep_s"])
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "report.json").write_text(report_json, encoding="utf-8")
        (args.output_dir / "robot.json").write_text(
            json.dumps(arm.to_dict(), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "model.xml").write_text(model_xml + "\n", encoding="utf-8")
        columns = [record["t"][:, None]]
        headers = ["time_s"]
        for field, unit in (
            ("q", "rad"),
            ("qd", "rad_s"),
            ("qdd", "rad_s2"),
            ("target_q", "rad"),
            ("target_qd", "rad_s"),
            ("target_qdd", "rad_s2"),
            ("torque", "Nm"),
            ("requested_torque", "Nm"),
        ):
            if field in record:
                columns.append(record[field])
                headers.extend(f"{link.name}.{field}_{unit}" for link in arm.links)
        for field in ("ee_pos", "target_ee_pos"):
            columns.append(record[field])
            headers.extend(f"{field}_{axis}_m" for axis in "xyz")
        columns.append(record["ee_speed"][:, None])
        headers.append("ee_speed_m_s")
        with (args.output_dir / "trajectory.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(np.column_stack(columns))
        np.savetxt(
            args.output_dir / "workspace.csv",
            workspace,
            delimiter=",",
            header="x_m,y_m,z_m",
            comments="",
        )
        if render is not None:
            render(arm, record, workspace, args.output_dir, animate=args.animate)
    except ModuleNotFoundError as error:
        dependency = (error.name or "").split(".")[0]
        extras = {
            "mujoco": "simulation",
            "matplotlib": "visualization",
            "PIL": "visualization",
        }
        if dependency not in extras:
            raise
        print(
            f"serial-arm-analyze requires the '{extras[dependency]}' extra. "
            f"Install it with: python -m pip install 'four-dof-arm[{extras[dependency]}]'",
            file=sys.stderr,
        )
        return 2
    except (ValueError, OSError) as error:
        print(f"serial-arm-analyze: {error}", file=sys.stderr)
        return 2
    print(f"{arm.name}: {arm.n_joints} joints, {report['mode']}")
    print(f"Analysis written to {args.output_dir}")
    print("Contacts are disabled; results are approximate and not hardware validation.")
    return 0
