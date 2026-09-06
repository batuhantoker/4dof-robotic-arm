from __future__ import annotations

import builtins
import csv
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from four_dof_arm.serial_arm import SerialArm, cli
from four_dof_arm.serial_arm.analysis import analyze


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "input.json"
    path.write_text(
        json.dumps(
            {
                "units": "SI",
                "links": [
                    {
                        "name": "hinge",
                        "axis": [0, 0, 2],
                        "offset": [0.2, 0, 0],
                        "mass": 0.3,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        pytest.fail(f"non-JSON number in {path.name}: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


@pytest.mark.parametrize("n", [1, 2, 6])
@pytest.mark.parametrize("simulate", [False, True], ids=["kinematic", "dynamic"])
def test_cli_artifacts_and_arbitrary_joint_csv(
    tmp_path: Path,
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
    n: int,
    simulate: bool,
) -> None:
    if simulate:
        pytest.importorskip("mujoco")
    config = strict_json(config_path)
    names = ["joint", "joint_target", "joint_requested", "axis_3", "axis_4", "axis_5"]
    config["links"] = [dict(config["links"][0], name=name) for name in names[:n]]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    scenario = {
        "duration": 0.1,
        "waypoints": [np.zeros(n).tolist(), np.full(n, 0.05).tolist()],
    }
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
    output = tmp_path / "nested" / "analysis"
    argv = [
        str(config_path),
        "--scenario",
        str(scenario_path),
        "--output-dir",
        str(output),
        "--timestep",
        "0.003",
        "--workspace-samples",
        "7",
        "--seed",
        "13",
    ]
    if simulate:
        argv.append("--simulate")
    assert cli.main(argv) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert f"{n} joints" in captured.out
    assert "Contacts are disabled" in captured.out
    assert {path.name for path in output.iterdir()} == {
        "robot.json",
        "model.xml",
        "trajectory.csv",
        "workspace.csv",
        "report.json",
    }
    arm = SerialArm.from_json(config_path)
    assert strict_json(output / "robot.json") == arm.to_dict()
    assert strict_json(output / "robot.json")["links"][0]["axis"] == [0, 0, 1]
    report, record, workspace = analyze(
        arm, scenario, timestep=0.003, workspace_samples=7, seed=13, simulate=simulate
    )
    assert strict_json(output / "report.json") == report
    option = ET.parse(output / "model.xml").find("option")
    assert option is not None
    assert float(option.attrib["timestep"]) == report["sampling"]["timestep_s"]
    fields = [
        ("q", "rad"),
        ("qd", "rad_s"),
        ("qdd", "rad_s2"),
        ("target_q", "rad"),
        ("target_qd", "rad_s"),
        ("target_qdd", "rad_s2"),
    ]
    if simulate:
        fields += [("torque", "Nm"), ("requested_torque", "Nm")]
    headers = (
        ["time_s"]
        + [
            f"{link.name}.{field}_{unit}"
            for field, unit in fields
            for link in arm.links
        ]
        + [
            f"{field}_{axis}_m"
            for field in ("ee_pos", "target_ee_pos")
            for axis in "xyz"
        ]
        + ["ee_speed_m_s"]
    )
    with (output / "trajectory.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == headers
    assert len(set(headers)) == len(headers) == 8 + n * (8 if simulate else 6)
    expected = np.column_stack(
        [
            record["t"],
            *[record[field] for field, _ in fields],
            record["ee_pos"],
            record["target_ee_pos"],
            record["ee_speed"],
        ]
    )
    actual = np.array(rows[1:], dtype=float)
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    with (output / "workspace.csv").open(newline="", encoding="utf-8") as stream:
        workspace_rows = list(csv.reader(stream))
    assert workspace_rows[0] == ["x_m", "y_m", "z_m"]
    np.testing.assert_array_equal(np.array(workspace_rows[1:], dtype=float), workspace)


@pytest.mark.parametrize("existing", ["directory", "file"])
def test_existing_output_is_not_overwritten(
    tmp_path: Path, config_path: Path, capsys: pytest.CaptureFixture[str], existing: str
) -> None:
    output = tmp_path / "analysis"
    if existing == "directory":
        output.mkdir()
        sentinel = output / "report.json"
    else:
        sentinel = output
    sentinel.write_bytes(b"keep existing bytes\x00")
    assert cli.main([str(config_path), "--output-dir", str(output)]) == 2
    assert sentinel.read_bytes() == b"keep existing bytes\x00"
    if existing == "directory":
        assert list(output.iterdir()) == [sentinel]
    assert "not overwritten" in capsys.readouterr().err


def test_existing_empty_output_directory_is_allowed(
    tmp_path: Path, config_path: Path
) -> None:
    output = tmp_path / "analysis"
    output.mkdir()
    assert (
        cli.main(
            [
                str(config_path),
                "--output-dir",
                str(output),
                "--timestep",
                "1",
                "--workspace-samples",
                "1",
            ]
        )
        == 0
    )
    assert (output / "report.json").is_file()


@pytest.mark.parametrize(
    "text",
    ["{", "[]", '{"duration":NaN}', '{"waypoints":[[0],[4]]}', '{"targets":[[0,0]]}'],
)
def test_invalid_scenario_leaves_no_output(
    tmp_path: Path, config_path: Path, capsys: pytest.CaptureFixture[str], text: str
) -> None:
    scenario = tmp_path / "bad.json"
    scenario.write_text(text, encoding="utf-8")
    output = tmp_path / "analysis"
    assert (
        cli.main(
            [str(config_path), "--scenario", str(scenario), "--output-dir", str(output)]
        )
        == 2
    )
    assert not output.exists()
    captured = capsys.readouterr()
    assert "serial-arm-analyze:" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize(
    "invalid", ["missing-config", "bad-config", "timestep", "seed", "workspace-samples"]
)
def test_invalid_input_leaves_no_output(
    tmp_path: Path, config_path: Path, capsys: pytest.CaptureFixture[str], invalid: str
) -> None:
    output = tmp_path / "analysis"
    if invalid == "missing-config":
        config_path = tmp_path / "missing.json"
    elif invalid == "bad-config":
        config_path.write_text('{"units":"mm","links":[]}', encoding="utf-8")
    argv = [str(config_path), "--output-dir", str(output)]
    if invalid in {"timestep", "seed", "workspace-samples"}:
        argv += [f"--{invalid}", "-1"]
    assert cli.main(argv) == 2
    assert not output.exists()
    captured = capsys.readouterr()
    assert "serial-arm-analyze:" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize(
    ("flag", "module", "dependency", "extra"),
    [
        ("--simulate", "simulation", "mujoco", "simulation"),
        ("--plots", "visualization", "matplotlib", "visualization"),
        ("--animate", "visualization", "PIL", "visualization"),
    ],
)
def test_missing_optional_dependencies_are_actionable(
    tmp_path: Path,
    config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flag: str,
    module: str,
    dependency: str,
    extra: str,
) -> None:
    original_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == module:
            raise ModuleNotFoundError(
                f"No module named {dependency!r}", name=dependency
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    output = tmp_path / "analysis"
    assert (
        cli.main(
            [str(config_path), "--output-dir", str(output), "--timestep", "1", flag]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert f"python -m pip install 'four-dof-arm[{extra}]'" in captured.err
    assert captured.out == ""
    assert not output.exists()


@pytest.mark.parametrize("help_only", [True, False], ids=["help", "analysis"])
def test_cli_without_optional_imports_in_fresh_process(
    tmp_path: Path, config_path: Path, help_only: bool
) -> None:
    script = """
import builtins
import sys
original_import = builtins.__import__
def core_only(name, *args, **kwargs):
    if name.split('.')[0] in {'mujoco', 'matplotlib', 'PIL', 'scipy'}:
        raise AssertionError('Unexpected optional import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = core_only
from four_dof_arm.serial_arm.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    argv = (
        ["--help"]
        if help_only
        else [
            str(config_path),
            "--output-dir",
            str(tmp_path / "analysis"),
            "--timestep",
            "1",
            "--workspace-samples",
            "2",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", script, *argv],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert ("usage:" if help_only else "prescribed_kinematics") in result.stdout
