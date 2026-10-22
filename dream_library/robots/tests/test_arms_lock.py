"""Behaviour tests for `robots.arms_lock` and the commands that take it: one holder of the arms at a time."""

import io
import os
import pathlib
import signal
import subprocess
import sys

import pytest

from robots import arms_lock, cli, errors, setup_wizard

COMMANDS = {
    arms_lock.SETUP: ["setup", "--no-browser"],
    arms_lock.ROBOT_RUN: ["robot", "run"],
    arms_lock.MOTORS: ["motors", "check"],
}


@pytest.fixture
def home(tmp_path: pathlib.Path) -> cli.Paths:
    """A station directory under tmp with a config file in it."""
    paths = cli.paths(tmp_path / "station")
    paths.home.mkdir()
    paths.config.write_text("task: placeholder\n", encoding="utf-8")
    return paths


def stand_ins(monkeypatch: pytest.MonkeyPatch, home: cli.Paths) -> list[str | None]:
    """Replaces what each command does once it holds the lock with a record of who holds the lock then."""
    seen: list[str | None] = []

    def record(*args: object, **kwargs: object) -> int:
        seen.append(arms_lock.held_by(home.lock))
        return 0

    monkeypatch.setattr(setup_wizard, "main", record)
    monkeypatch.setattr(cli, "_run_here", record)
    monkeypatch.setattr(cli, "motors", record)
    return seen


def test_a_second_holder_is_refused_with_the_first_ones_name_and_how_to_stop_it(tmp_path: pathlib.Path) -> None:
    lock = tmp_path / "station" / arms_lock.LOCK_NAME
    with arms_lock.acquire(lock, arms_lock.SETUP):
        assert lock.read_text(encoding="utf-8") == f"{os.getpid()} dreamcli setup\n"
        with pytest.raises(errors.StationError) as refused:
            arms_lock.acquire(lock, arms_lock.ROBOT_RUN)
        assert str(refused.value) == (
            f"the arms are held by dreamcli setup (pid {os.getpid()}): press Finish on the setup page, or Ctrl-C its "
            "terminal, then run this again"
        )
        assert arms_lock.held_by(lock) == str(refused.value)
        assert lock.read_text(encoding="utf-8") == f"{os.getpid()} dreamcli setup\n"
    assert arms_lock.held_by(lock) is None
    with arms_lock.acquire(lock, arms_lock.MOTORS):
        assert lock.read_text(encoding="utf-8") == f"{os.getpid()} dreamcli motors\n"


@pytest.mark.parametrize("command", list(COMMANDS))
def test_each_command_holds_the_lock_while_it_runs(
    command: str, home: cli.Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = stand_ins(monkeypatch, home)
    assert cli.run(["--home", str(home.home), *COMMANDS[command]]) == 0
    assert seen == [
        f"the arms are held by {command} (pid {os.getpid()}): {arms_lock.HOW_TO_STOP[command]}, then run this again"
    ]
    assert arms_lock.held_by(home.lock) is None


@pytest.mark.parametrize("holder", [*COMMANDS, arms_lock.STATION])
def test_every_command_refuses_while_another_holds_the_arms(
    holder: str, home: cli.Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = stand_ins(monkeypatch, home)
    line = f"the arms are held by {holder} (pid {os.getpid()}): {arms_lock.HOW_TO_STOP[holder]}, then run this again"
    with arms_lock.acquire(home.lock, holder):
        for argv in COMMANDS.values():
            out = io.StringIO()
            assert cli.run(["--home", str(home.home), *argv], out=out) == 1
            assert out.getvalue() == line + "\n", argv
        out = io.StringIO()
        code = cli.robot_up(
            home,
            command=[sys.executable, "-c", "pass"],
            out=out,
            run_command=lambda argv: 0,
            port_free=lambda host, port: True,
        )
        assert code == 1 and out.getvalue() == line + "\n"
        assert cli.running_pid(home) is None, "no station was started"
    assert seen == [], "no command got past the lock"


def test_an_agent_an_older_dreamcli_started_still_keeps_robot_run_and_motors_off_the_arms(
    home: cli.Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = stand_ins(monkeypatch, home)
    agent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        home.pid.write_text(f"{agent.pid}\n", encoding="utf-8")
        line = (
            f"the arms are held by the station agent (pid {agent.pid}): 'dreamcli robot down' stops it, then run this "
            "again"
        )
        for argv in (COMMANDS[arms_lock.ROBOT_RUN], COMMANDS[arms_lock.MOTORS]):
            out = io.StringIO()
            assert cli.run(["--home", str(home.home), *argv], out=out) == 1
            assert out.getvalue() == line + "\n", argv
        assert arms_lock.held_by(home.lock) is None, "the pid file alone refused them"
    finally:
        agent.kill()
        agent.wait()
    assert seen == [], "no command got past the pid file"


def test_a_killed_holder_frees_the_lock(home: cli.Paths) -> None:
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import pathlib, sys, time\n"
            "from robots import arms_lock\n"
            "handle = arms_lock.acquire(pathlib.Path(sys.argv[1]), arms_lock.ROBOT_RUN)\n"
            "print('held', flush=True)\n"
            "time.sleep(60)\n",
            str(home.lock),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline() == "held\n"
        assert arms_lock.held_by(home.lock) == (
            f"the arms are held by dreamcli robot run (pid {holder.pid}): Ctrl-C the robot run, then run this again"
        )
    finally:
        holder.send_signal(signal.SIGKILL)
        holder.wait()
    assert arms_lock.held_by(home.lock) is None
    with arms_lock.acquire(home.lock, arms_lock.SETUP):
        pass


def test_an_unreadable_holder_still_refuses_with_a_line(tmp_path: pathlib.Path) -> None:
    lock = tmp_path / arms_lock.LOCK_NAME
    with arms_lock.acquire(lock, arms_lock.MOTORS) as handle:
        handle.seek(0)
        handle.truncate()
        handle.write("garbage")
        handle.flush()
        assert arms_lock.held_by(lock) == (
            "the arms are held by another dreamcli command on this computer: stop it, then run this again"
        )
