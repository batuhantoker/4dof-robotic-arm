"""Dependency-light console entry points for optional package features."""

from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType


def _demo_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="four-dof-arm-demo",
        description="Run the 4-DOF arm kinematics and visualization demo.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="display plots and animation interactively",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for PNG outputs; no files are written when omitted",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for IK and workspace sampling (default: 0)",
    )
    return parser


def _mujoco_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="four-dof-arm-mujoco",
        description="Run the 4-DOF arm MuJoCo simulation.",
    )
    parser.add_argument("--video", action="store_true", help="render demo video")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="run FK validation only",
    )
    parser.add_argument(
        "--output",
        default="arm_physics_sim.mp4",
        help="video output path",
    )
    return parser


def _load_optional_module(
    module_name: str,
    *,
    dependency: str,
    extra: str,
    command: str,
) -> ModuleType | None:
    try:
        return import_module(module_name)
    except ModuleNotFoundError as error:
        missing = error.name or ""
        if missing != dependency and not missing.startswith(f"{dependency}."):
            raise
        print(
            f"{command} requires the '{extra}' extra. "
            f"Install it with: python -m pip install 'four-dof-arm[{extra}]'",
            file=sys.stderr,
        )
        return None


def demo(argv: list[str] | None = None) -> int:
    """Run the visualization demo without importing Matplotlib for help."""
    args = _demo_parser().parse_args(argv)
    module = _load_optional_module(
        "four_dof_arm.demo",
        dependency="matplotlib",
        extra="visualization",
        command="four-dof-arm-demo",
    )
    if module is None:
        return 2

    module.main(show=args.show, output_dir=args.output_dir, seed=args.seed)
    return 0


def mujoco(argv: list[str] | None = None) -> int:
    """Run the MuJoCo command without importing MuJoCo for help."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    _mujoco_parser().parse_args(arguments)
    module = _load_optional_module(
        "four_dof_arm.mujoco_sim",
        dependency="mujoco",
        extra="simulation",
        command="four-dof-arm-mujoco",
    )
    if module is None:
        return 2

    return int(module.main(arguments))
