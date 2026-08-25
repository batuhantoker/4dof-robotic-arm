from __future__ import annotations

from collections.abc import Callable
from typing import NoReturn

import pytest

from four_dof_arm import cli


@pytest.mark.parametrize(
    ("entry_point", "expected_option"),
    [
        (cli.demo, "--output-dir"),
        (cli.mujoco, "--validate-only"),
    ],
)
def test_wrapper_help_does_not_import_optional_command_module(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    entry_point: Callable[[list[str] | None], int],
    expected_option: str,
) -> None:
    def fail_import(_module_name: str) -> NoReturn:
        pytest.fail("help imported the optional command module")

    monkeypatch.setattr(cli, "import_module", fail_import)

    with pytest.raises(SystemExit) as error:
        entry_point(["--help"])

    assert error.value.code == 0
    assert expected_option in capsys.readouterr().out


@pytest.mark.parametrize(
    ("entry_point", "command", "module_name", "dependency", "extra"),
    [
        (
            cli.demo,
            "four-dof-arm-demo",
            "four_dof_arm.demo",
            "matplotlib",
            "visualization",
        ),
        (
            cli.mujoco,
            "four-dof-arm-mujoco",
            "four_dof_arm.mujoco_sim",
            "mujoco",
            "simulation",
        ),
    ],
)
def test_wrapper_reports_actionable_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    entry_point: Callable[[list[str] | None], int],
    command: str,
    module_name: str,
    dependency: str,
    extra: str,
) -> None:
    def block_import(requested_module: str) -> NoReturn:
        assert requested_module == module_name
        raise ModuleNotFoundError(
            f"No module named '{dependency}'",
            name=dependency,
        )

    monkeypatch.setattr(cli, "import_module", block_import)

    assert entry_point([]) == 2
    assert capsys.readouterr().err == (
        f"{command} requires the '{extra}' extra. "
        f"Install it with: python -m pip install 'four-dof-arm[{extra}]'\n"
    )
