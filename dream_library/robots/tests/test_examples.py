"""`dreamcli example`: an example is found where the wheel or a checkout puts it, started in place of dreamcli, listed,
and self-tested headless."""

import io
import pathlib
import sys

import pytest

from robots import cli, errors, examples

## The yam-mujoco example in this checkout, the copy a source run uses.
YAM_MUJOCO = examples.EXAMPLES["yam-mujoco"]
CHECKOUT_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / YAM_MUJOCO.directory


def test_a_checkout_runs_the_example_itself() -> None:
    assert examples.example_dir(YAM_MUJOCO) == CHECKOUT_DIR
    assert (CHECKOUT_DIR / YAM_MUJOCO.entry).is_file()
    assert (CHECKOUT_DIR / YAM_MUJOCO.self_test[0]).is_file()


def test_an_installed_wheel_runs_its_own_copy_first(tmp_path: pathlib.Path) -> None:
    module = tmp_path / "site-packages" / "robots" / "examples.py"
    module.parent.mkdir(parents=True)
    module.touch()
    installed = tmp_path / "site-packages" / examples.WHEEL_DIR / YAM_MUJOCO.directory
    installed.mkdir(parents=True)
    (installed / YAM_MUJOCO.entry).touch()
    assert examples.example_dir(YAM_MUJOCO, module) == installed


def test_a_robots_without_the_examples_says_where_they_come_from(tmp_path: pathlib.Path) -> None:
    module = tmp_path / "a" / "b" / "robots" / "examples.py"
    module.parent.mkdir(parents=True)
    with pytest.raises(errors.ConfigError, match="pip install --upgrade dreamlabs"):
        examples.example_dir(YAM_MUJOCO, module)


def test_dreamcli_example_alone_lists_the_examples() -> None:
    out = io.StringIO()
    assert cli.run(["example"], out=out) == 0
    assert "yam-mujoco" in out.getvalue() and "dreamcli example <name>" in out.getvalue()


def test_an_unknown_example_is_refused_with_the_list() -> None:
    out = io.StringIO()
    assert cli.run(["example", "nope"], out=out) == 1
    assert "yam-mujoco" in out.getvalue()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no exec; the example runs as a child there")
def test_dreamcli_example_replaces_itself_with_the_examples_script(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(examples.os, "execv", lambda path, argv: calls.append((path, argv)))
    assert cli.run(["example", "yam-mujoco", "--verbose"], out=io.StringIO()) == 0
    assert calls == [(sys.executable, [sys.executable, str(CHECKOUT_DIR / YAM_MUJOCO.entry), "--verbose"])]


def test_dreamcli_example_self_test_builds_and_steps_the_scene() -> None:
    assert cli.run(["example", "yam-mujoco", "--self-test"], out=io.StringIO()) == 0
