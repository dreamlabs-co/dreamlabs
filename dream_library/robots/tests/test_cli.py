"""Behaviour tests for `robots.cli`: `robot up` / `down` / `log` over a stand-in for the station agent.

The agent is replaced by a small Python script that prints a line, waits for SIGINT and exits 0 (or, in the
failure tests, exits at once), so what is pinned is the process handling: the pid file, the detached start, the
wait for the agent's API (a `ready` probe the tests script, since the stand-in serves no HTTP), the signal order
of `down`, and the log tail. `dreamcli setup` is `robots.setup_wizard`, tested in
its own module; here only its dispatch is checked.
"""

import http.server
import io
import os
import pathlib
import signal
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import conftest
import pytest
import setup_fakes

from robots import arms_lock, cli, errors, robstride, startup_failure, station

AGENT_SCRIPT = f"""
import signal, sys, time
stopped = False
def on_int(*_):
    global stopped
    stopped = True
signal.signal(signal.SIGINT, on_int)
print("agent: ready", flush=True)
while not stopped:
    time.sleep(0.05)
print("INFO robots.station: {station.SWITCHED_OFF_LINE}", flush=True)
"""

NO_SWITCH_OFF_SCRIPT = """
import signal, sys, time
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
print("agent: ready", flush=True)
while True:
    time.sleep(0.05)
"""

## A stand-in agent that ignores SIGINT (a wedged process), so `down` has to escalate.
STUBBORN_SCRIPT = """
import signal, time
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print("agent: ignoring signals", flush=True)
while True:
    time.sleep(0.05)
"""

## Longest a test waits for the stand-in to reach a state.
WAIT_S = 5.0


def up_now(host: str, port: int) -> bool:
    """A `ready` probe for the stand-in agent, which serves no HTTP: up as soon as it is asked."""
    return True


@pytest.fixture
def home(tmp_path: pathlib.Path) -> cli.Paths:
    """A station directory under tmp with a config file in it (its content is never read by `cli`)."""
    paths = cli.paths(tmp_path / "station")
    paths.home.mkdir()
    paths.config.write_text("task: placeholder\nlive: {robot_id: rbt_test}\n", encoding="utf-8")
    return paths


@pytest.fixture
def agent(tmp_path: pathlib.Path) -> Callable[[str], list[str]]:
    """Writes a stand-in agent script and returns the argv that runs it with this interpreter."""

    def make(source: str) -> list[str]:
        script = tmp_path / "agent.py"
        script.write_text(source, encoding="utf-8")
        return [sys.executable, str(script)]

    return make


@pytest.fixture
def cleanup(home: cli.Paths) -> Iterator[None]:
    """Kills whatever `robot up` left running, so a failing test does not leak a process."""
    yield
    pid = cli.running_pid(home)
    if pid is not None:
        os.kill(pid, signal.SIGKILL)
        wait_until(lambda: cli.running_pid(home) is None, "the stand-in to die")


def wait_until(predicate: Callable[[], bool], what: str) -> None:
    """Polls from the test thread until `predicate` holds."""
    deadline = time.monotonic() + WAIT_S
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.02)


def test_up_starts_the_agent_detached_records_its_pid_and_reports_the_addresses(
    home: cli.Paths, agent: Callable[[str], list[str]], cleanup: None
) -> None:
    """`up` writes the pid file, the agent's output lands in the log, and a second `up` is refused."""
    out = io.StringIO()
    code = cli.robot_up(home, command=agent(AGENT_SCRIPT), out=out, ready=up_now)
    assert code == 0, out.getvalue()
    pid = cli.running_pid(home)
    assert pid is not None
    report = out.getvalue()
    assert f"station started (pid {pid})" in report
    assert "page:" not in report  # the live view is where an operator goes next, not the station page
    assert "live view: https://platform.example.com\n" in report  # the credentials' platform (conftest)
    assert "/status" not in report  # the agent's JSON API is not a link for an operator
    assert str(home.log) in report
    # The agent is in its own session: killing this process's group would not reach it.
    assert os.getsid(pid) != os.getsid(os.getpid())
    wait_until(lambda: "agent: ready" in home.log.read_text(), "the agent's first line in the log")
    assert home.log.read_text().startswith("\n=== dreamcli robot up: ")

    again = io.StringIO()
    assert cli.robot_up(home, command=agent(AGENT_SCRIPT), out=again, ready=up_now) == 1
    assert f"already running (pid {pid})" in again.getvalue()
    assert cli.running_pid(home) == pid


def test_down_sends_sigint_waits_for_the_exit_and_removes_the_pid_file(
    home: cli.Paths, agent: Callable[[str], list[str]], cleanup: None
) -> None:
    """The agent's own SIGINT handler runs (its goodbye line is in the log) and `down` reports stopped."""
    assert cli.robot_up(home, command=agent(AGENT_SCRIPT), out=io.StringIO(), ready=up_now) == 0
    pid = cli.running_pid(home)
    assert pid is not None
    wait_until(lambda: "agent: ready" in home.log.read_text(), "the agent to be ready")
    out = io.StringIO()
    assert cli.robot_down(home, out=out) == 0
    assert f"pid {pid}: stopping (the arms switch off)" in out.getvalue()
    assert "terminating" not in out.getvalue()
    assert out.getvalue().rstrip().endswith("stopped; the arms are switched off")
    assert not home.pid.exists()
    assert station.SWITCHED_OFF_LINE in home.log.read_text()
    assert cli.running_pid(home) is None


def test_down_does_not_say_switched_off_when_the_agent_did_not_log_it(
    home: cli.Paths, agent: Callable[[str], list[str]], cleanup: None
) -> None:
    home.home.mkdir(parents=True, exist_ok=True)
    home.log.write_text(f"{cli.LOG_HEADER} earlier\nINFO robots.station: {station.SWITCHED_OFF_LINE}\n")
    assert cli.robot_up(home, command=agent(NO_SWITCH_OFF_SCRIPT), out=io.StringIO(), ready=up_now) == 0
    wait_until(lambda: "agent: ready" in home.log.read_text(), "the agent to be ready")
    out = io.StringIO()
    assert cli.robot_down(home, out=out) == 1
    assert "the arms are switched off" not in out.getvalue()
    assert "they may still hold under power; support them and switch their power off" in out.getvalue()
    assert not home.pid.exists()


def test_down_escalates_to_sigterm_and_sigkill_when_sigint_is_ignored(
    home: cli.Paths, agent: Callable[[str], list[str]], cleanup: None
) -> None:
    """A process that ignores SIGINT and SIGTERM is killed after the two grace periods, and reported so; while it
    waits after SIGINT, `down` says once that the arms are parking."""
    assert cli.robot_up(home, command=agent(STUBBORN_SCRIPT), out=io.StringIO(), ready=up_now) == 0
    pid = cli.running_pid(home)
    assert pid is not None
    # `up` returns as soon as the probe says yes, so give the stand-in time to install its handlers.
    wait_until(lambda: "agent: ignoring signals" in home.log.read_text(), "the stand-in to ignore signals")
    out = io.StringIO()
    assert cli.robot_down(home, out=out, stop_grace_s=0.3, kill_grace_s=0.3, park_notice_s=0.1) == 1
    report = out.getvalue()
    assert f"pid {pid}: stopping (the arms switch off)" in report
    assert report.count(f"pid {pid}: parking the arms before they switch off (up to 0 s)") == 1, report
    assert "did not stop; terminating" in report
    assert "did not terminate; killing" in report
    assert "they may still hold under power" in report, "a killed agent never switched the arms off"
    assert not home.pid.exists()
    assert cli.running_pid(home) is None


def test_down_with_nothing_running_switches_a_yams_motors_off_directly(
    home: cli.Paths, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """the station's PC rebooted mid-session, so no agent runs, and the arms still held under power. The
    links come back down after a reboot, so `down` brings them up, then switches each link's motors off."""
    net = tmp_path / "net"
    for name in ("can_follower_l", "can_follower_r"):
        setup_fakes.net_link(net, name, up=False)
    monkeypatch.setattr(cli, "NET_DIR", net)
    home.home.mkdir(parents=True, exist_ok=True)
    home.config.write_text(
        "follower_l: {can_channel: can_follower_l}\nfollower_r: {can_channel: can_follower_r}\n", encoding="utf-8"
    )

    def run_command(argv: Sequence[str]) -> int:
        setup_fakes.net_link(net, argv[4], up=True)
        return 0

    switched: list[str] = []
    out = io.StringIO()
    assert cli.robot_down(home, out=out, run_command=run_command, switch_off=switched.append) == 0
    assert switched == ["can_follower_l", "can_follower_r"]
    assert "switching the arms' motors off directly" in out.getvalue()
    assert out.getvalue().rstrip().endswith("the arms' motors are switched off")

    def broken(link: str) -> None:
        if link == "can_follower_l":
            raise OSError("no such device")
        switched.append(link)

    switched.clear()
    failed = io.StringIO()
    assert cli.robot_down(home, out=failed, run_command=run_command, switch_off=broken) == 1
    assert switched == ["can_follower_r"], "one link's failure does not leave the other arm on"
    assert "can_follower_l: could not switch the motors off: no such device" in failed.getvalue()
    assert "support the arms and switch their power off" in failed.getvalue()


def test_down_with_nothing_running_leaves_another_body_alone(home: cli.Paths) -> None:
    home.home.mkdir(parents=True, exist_ok=True)
    home.config.write_text("body: so101\n", encoding="utf-8")
    switched: list[str] = []
    out = io.StringIO()
    assert cli.robot_down(home, out=out, switch_off=switched.append) == 0
    assert switched == []
    assert out.getvalue().strip() == "the station is not running"


def test_down_with_nothing_running_is_a_no_op_and_a_stale_pid_file_is_removed(home: cli.Paths) -> None:
    """A pid file naming a dead process is cleaned up by the first command that looks at it. With no config there
    are no arms to switch off."""
    home.config.unlink()
    out = io.StringIO()
    assert cli.robot_down(home, out=out) == 0
    assert "not running" in out.getvalue()

    # A pid no process has: the largest pid the kernel allows plus a margin is not in use.
    home.pid.write_text("4194304\n", encoding="utf-8")
    assert cli.running_pid(home) is None
    assert not home.pid.exists()
    home.pid.write_text("not a pid\n", encoding="utf-8")
    assert cli.running_pid(home) is None


def test_up_reports_an_agent_that_dies_at_once_with_the_log_tail(
    home: cli.Paths, agent: Callable[[str], list[str]]
) -> None:
    """A config the agent refuses ends it at once; `up` says so, prints the log and leaves no pid."""
    out = io.StringIO()
    script = agent("import sys\nprint('ConfigError: config key api_key is required', flush=True)\nsys.exit(2)\n")
    assert cli.robot_up(home, command=script, out=out, ready=lambda host, port: False) == 1
    report = out.getvalue()
    assert "exited with code 2 0 s after starting" in report
    assert "ConfigError: config key api_key is required" in report
    assert "nothing is running; fix the cause above" in report
    assert "station started" not in report
    assert not home.pid.exists()


def test_up_reports_an_agent_that_dies_while_opening_the_cameras_not_a_false_start(
    home: cli.Paths, agent: Callable[[str], list[str]]
) -> None:
    """The arms open, a RealSense never delivers a frame, and the agent dies seconds
    in. `up` keeps watching until the API answers or the process is gone, so the death is reported with the
    traceback's last line, and the "station started" block is never printed."""
    out = io.StringIO()
    script = agent(
        "import sys, time\n"
        "print('station: cameras opening', flush=True)\n"
        "time.sleep(0.6)\n"
        "print('  File \"cameras.py\", line 529, in __init__', flush=True)\n"
        'print("RuntimeError: Frame didn\'t arrive within 5000", flush=True)\n'
        "sys.exit(1)\n"
    )
    assert cli.robot_up(home, command=script, out=out, ready=lambda host, port: False) == 1
    report = out.getvalue()
    assert "opening the arms and cameras" in report
    assert "exited with code 1 1 s after starting" in report
    assert "RuntimeError: Frame didn't arrive within 5000" in report
    assert "station started" not in report
    assert not home.pid.exists()


def test_up_says_why_a_start_failed_first_even_under_a_long_traceback(
    home: cli.Paths, agent: Callable[[str], list[str]]
) -> None:
    """When `robot up` exits within seconds, the cause is not lost in the log: the station's could-not-start line
    (`robots.startup_failure`) is printed first, however long the traceback after it, and only this start's."""
    home.log.write_text(f"{cli.LOG_HEADER} earlier\n{startup_failure.MARKER}an arm: old Fix: old\n", encoding="utf-8")
    out = io.StringIO()
    cause = "the top camera (RealSense 260322275394): the top camera 260322275394 did not open: No device connected"
    script = agent(
        "import sys\n"
        f"print('ERROR robots.startup_failure: {startup_failure.MARKER}{cause} Fix: Check the cable', flush=True)\n"
        "for i in range(40):\n"
        "    print(f'  File \"x.py\", line {i}', flush=True)\n"
        "sys.exit(1)\n"
    )
    assert cli.robot_up(home, command=script, out=out, ready=lambda host, port: False) == 1
    report = out.getvalue().splitlines()
    said = [line for line in report if line.startswith("the station could not start: ")]
    assert said == [f"the station could not start: {cause} Fix: Check the cable"]
    assert report.index(said[0]) < next(i for i, line in enumerate(report) if "exited with code 1" in line)
    assert "'dreamcli robot log' shows the full log" in report[-1]


def test_up_names_no_cause_when_the_station_wrote_none_this_start(
    home: cli.Paths, agent: Callable[[str], list[str]]
) -> None:
    home.log.write_text(f"{cli.LOG_HEADER} earlier\n{startup_failure.MARKER}an arm: old Fix: old\n", encoding="utf-8")
    out = io.StringIO()
    assert cli.robot_up(home, command=agent("import sys\nsys.exit(3)\n"), out=out, ready=lambda h, p: False) == 1
    assert not [line for line in out.getvalue().splitlines() if line.startswith("the station could not start")]


def test_up_reports_a_station_stopped_while_it_opened_as_stopped_not_as_a_fault(
    home: cli.Paths, agent: Callable[[str], list[str]]
) -> None:
    out = io.StringIO()
    assert (
        cli.robot_up(home, command=agent("print('agent: stopped', flush=True)\n"), out=out, ready=lambda h, p: False)
        == 1
    )
    assert "the station was stopped 0 s after starting, before it answered" in out.getvalue()
    assert "fix the cause" not in out.getvalue()
    assert not home.pid.exists()


def test_a_ctrl_c_while_up_waits_ends_the_wait_and_leaves_the_station_starting(
    home: cli.Paths, agent: Callable[[str], list[str]], cleanup: None
) -> None:

    def interrupted(host: str, port: int) -> bool:
        raise KeyboardInterrupt

    out = io.StringIO()
    assert cli.robot_up(home, command=agent(AGENT_SCRIPT), out=out, ready=interrupted) == 1
    pid = cli.running_pid(home)
    assert pid is not None, "the station goes on"
    assert f"stopped watching; the station (pid {pid}) goes on starting by itself" in out.getvalue()


def test_up_waits_for_the_api_to_answer_before_it_reports_started(
    home: cli.Paths, agent: Callable[[str], list[str]], cleanup: None
) -> None:
    """The start block is printed only once the probe says the API is up, however long that takes, and the probe
    is asked for the bind host and API port the flags name."""
    asked: list[tuple[str, int]] = []

    def ready(host: str, port: int) -> bool:
        asked.append((host, port))
        return len(asked) >= 4

    out = io.StringIO()
    code = cli.robot_up(
        home, command=agent(AGENT_SCRIPT), extra_args=["--host", "10.0.0.5", "--api-port=9091"], out=out, ready=ready
    )
    assert code == 0, out.getvalue()
    assert asked == [("10.0.0.5", 9091)] * 4
    report = out.getvalue()
    assert report.index("opening the arms and cameras") < report.index("station started")
    assert "live view: https://platform.example.com" in report


def test_up_on_a_config_without_a_live_section_names_the_api_not_a_live_view(
    home: cli.Paths, agent: Callable[[str], list[str]], cleanup: None
) -> None:
    home.config.write_text("task: placeholder\n", encoding="utf-8")
    out = io.StringIO()
    code = cli.robot_up(home, command=agent(AGENT_SCRIPT), extra_args=["--api-port", "9091"], out=out, ready=up_now)
    assert code == 0, out.getvalue()
    report = out.getvalue()
    assert "live view: none; the config has no 'live' section ('dreamcli setup' registers the robot" in report
    assert "https://platform.example.com" not in report and "Start from the live view" not in report
    assert "nothing moves until a start on its API (curl -X POST http://127.0.0.1:9091/start)" in report


def test_up_reports_an_agent_that_never_answers_and_leaves_it_running(
    home: cli.Paths, agent: Callable[[str], list[str]], cleanup: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past the deadline with the process alive and the API silent, `up` says so with the log's tail and the two
    commands that help, exits 1, and leaves the process and its pid file alone."""
    monkeypatch.setattr(cli, "START_TIMEOUT_S", 0.5)
    out = io.StringIO()
    assert cli.robot_up(home, command=agent(AGENT_SCRIPT), out=out, ready=lambda host, port: False) == 1
    report = out.getvalue()
    assert "has not answered on port 8091 after" in report
    assert "agent: ready" in report, "the log's tail is shown"
    assert "dreamcli robot log" in report and "dreamcli robot down" in report
    assert "station started" not in report
    assert cli.running_pid(home) is not None, "the process is left running for the operator to judge"


class _StatusHandler(http.server.BaseHTTPRequestHandler):
    """Answers `GET /status` the way the station's API does: 200 with the controller's state."""

    def do_GET(self) -> None:
        body = b'{"state": "idle"}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_api_answers_probes_loopback_for_a_bind_on_every_interface_and_only_the_station_counts() -> None:
    for handler, expected in ((_StatusHandler, True), (http.server.SimpleHTTPRequestHandler, False)):
        server = http.server.HTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            assert cli.api_answers("0.0.0.0", port) is expected
            assert cli.api_answers("127.0.0.1", port) is expected
        finally:
            server.shutdown()
            server.server_close()
    assert cli.api_answers("0.0.0.0", port) is False


def test_up_refuses_a_port_another_program_holds_before_anything_opens(
    home: cli.Paths, agent: Callable[[str], list[str]]
) -> None:
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        taken = holder.getsockname()[1]
        assert cli.api_port_free("127.0.0.1", taken) is False
        assert cli.api_port_free("0.0.0.0", taken) is False
        out = io.StringIO()
        code = cli.robot_up(home, command=agent(AGENT_SCRIPT), extra_args=["--api-port", str(taken)], out=out)
    assert code == 1
    assert f"port {taken} on 127.0.0.1 is taken by another program" in out.getvalue()
    assert "--api-port <port>" in out.getvalue()
    assert not home.pid.exists() and not home.log.exists()
    assert cli.api_port_free("127.0.0.1", taken) is True


def test_the_station_api_binds_this_box_only_unless_told_otherwise() -> None:
    assert station.DEFAULT_HOST == "127.0.0.1"
    assert cli._serve_flags([]) == ("127.0.0.1", station.DEFAULT_API_PORT)
    assert cli._serve_flags(["--host", "0.0.0.0"]) == ("0.0.0.0", station.DEFAULT_API_PORT)
    assert cli._serve_flags(["--host=0.0.0.0", "--api-port", "9000"]) == ("0.0.0.0", 9000)


def test_up_without_a_config_points_at_setup(tmp_path: pathlib.Path) -> None:
    """No station config means `dreamcli setup` first; an explicit `--config` that is missing says so instead."""
    paths = cli.paths(tmp_path / "empty")
    out = io.StringIO()
    assert cli.robot_up(paths, out=out) == 1
    assert "run 'dreamcli setup' first" in out.getvalue()
    out = io.StringIO()
    assert cli.robot_up(paths, config=tmp_path / "other.yaml", out=out) == 1
    assert "no such file" in out.getvalue()
    assert not paths.home.exists()


def test_log_prints_the_last_n_lines(home: cli.Paths) -> None:
    """`log` is a tail: the default 40 lines, or `-n`; no log yet is exit 1 with a hint."""
    out = io.StringIO()
    assert cli.robot_log(home, out=out) == 1
    assert "not been started" in out.getvalue()
    home.log.write_text("".join(f"line {i}\n" for i in range(100)), encoding="utf-8")
    out = io.StringIO()
    assert cli.robot_log(home, out=out) == 0
    lines = out.getvalue().splitlines()
    assert lines == [f"line {i}" for i in range(60, 100)]
    out = io.StringIO()
    assert cli.robot_log(home, lines=3, out=out) == 0
    assert out.getvalue().splitlines() == ["line 97", "line 98", "line 99"]


def test_paths_come_from_the_argument_then_the_environment_then_the_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The station directory is expanded and absolute, and the three files hang off it."""
    monkeypatch.delenv(cli.HOME_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    default = cli.paths()
    assert default.home == (tmp_path / ".dreamlabs").resolve()
    assert default.config.name == cli.CONFIG_NAME
    assert default.log.name == cli.LOG_NAME
    assert default.pid.name == cli.PID_NAME
    monkeypatch.setenv(cli.HOME_ENV, str(tmp_path / "elsewhere"))
    assert cli.paths().home == (tmp_path / "elsewhere").resolve()
    assert cli.paths(tmp_path / "explicit").home == (tmp_path / "explicit").resolve()


def test_station_command_runs_this_interpreter_on_the_station_module() -> None:
    """The agent is `python -m robots.station --config <file>`, the pass-through flags, then the arms' lock."""
    argv = cli.station_command(
        pathlib.Path("/x/station.yaml"), ["--api-port", "9091", "--task", "fold"], lock=pathlib.Path("/x/arms.lock")
    )
    assert argv == [
        sys.executable,
        "-m",
        "robots.station",
        "--config",
        "/x/station.yaml",
        "--api-port",
        "9091",
        "--task",
        "fold",
        "--arms-lock",
        "/x/arms.lock",
    ]


def test_run_dispatches_the_subcommands_and_passes_unknown_flags_through_to_up(
    home: cli.Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`robot up` forwards unknown flags to the agent (the task comes with the Start, so there is no flag
    for it any more); `log -n`, `down` and `setup` dispatch."""
    monkeypatch.setenv(cli.HOME_ENV, str(home.home))
    seen: dict[str, object] = {}

    def fake_up(paths: cli.Paths, **kwargs: object) -> int:
        seen["up"] = (paths, kwargs)
        return 0

    def fake_down(paths: cli.Paths, **kwargs: object) -> int:
        seen["down"] = paths
        return 0

    def fake_log(paths: cli.Paths, **kwargs: object) -> int:
        seen["log"] = (paths, kwargs)
        return 0

    monkeypatch.setattr(cli, "robot_up", fake_up)
    monkeypatch.setattr(cli, "robot_down", fake_down)
    monkeypatch.setattr(cli, "robot_log", fake_log)

    assert cli.run(["robot", "up", "--host", "127.0.0.1", "--api-port", "9091"]) == 0
    paths, kwargs = seen["up"]  # type: ignore[misc]
    assert paths == home
    assert kwargs["extra_args"] == ["--host", "127.0.0.1", "--api-port", "9091"]
    assert kwargs["config"] is None
    assert cli.run(["robot", "up", "--config", str(home.config)]) == 0
    assert seen["up"][1]["config"] == home.config  # type: ignore[index]
    assert cli.run(["robot", "down"]) == 0
    assert seen["down"] == home
    assert cli.run(["robot", "log", "-n", "7"]) == 0
    assert seen["log"][1]["lines"] == 7  # type: ignore[index]

    from robots import setup_wizard

    def fake_wizard(config_path: pathlib.Path, **kwargs: object) -> int:
        seen["setup"] = (config_path, kwargs)
        return 0

    monkeypatch.setattr(setup_wizard, "main", fake_wizard)
    assert cli.run(["setup", "--port", "0", "--no-browser"]) == 0
    setup = seen["setup"]
    assert isinstance(setup, tuple)
    config_path, kwargs = setup
    assert isinstance(kwargs, dict)
    agent_running = kwargs.pop("agent_running")
    assert (config_path, kwargs) == (home.config, {"host": None, "port": 0, "browser": False})
    assert callable(agent_running) and isinstance(agent_running(), bool)  # asks the pid file

    def fake_tcp(config_path: pathlib.Path, **kwargs: object) -> int:
        seen["tcp"] = (config_path, kwargs)
        return 0

    monkeypatch.setattr(setup_wizard, "tcp_main", fake_tcp)
    assert cli.run(["tcp", "--port", "0", "--no-browser"]) == 0
    assert seen["tcp"] == (home.config, {"host": None, "port": 0, "browser": False, "out": sys.stdout})

    def fake_motors(action: str, **kwargs: object) -> int:
        seen["motors"] = (action, kwargs)
        return 0

    monkeypatch.setattr(cli, "motors", fake_motors)
    assert cli.run(["motors", "check"]) == 0
    assert seen["motors"] == ("check", {"links": None, "motor_ids": None, "out": sys.stdout})
    assert cli.run(["motors", "set-mit", "--link", "can_follower_r", "--motor", "7", "--motor", "6"]) == 0
    assert seen["motors"] == ("set-mit", {"links": ["can_follower_r"], "motor_ids": [7, 6], "out": sys.stdout})
    with arms_lock.acquire(home.lock, arms_lock.STATION):
        report = io.StringIO()
        assert cli.run(["motors", "check"], out=report) == 1 and "robot down" in report.getvalue()

    with pytest.raises(SystemExit):
        cli.run(["robot", "log", "--bogus"])


def test_a_ctrl_c_no_command_handles_ends_dreamcli_with_one_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:

    def interrupted(argv: Sequence[str] | None = None) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run", interrupted)
    with pytest.raises(SystemExit) as ended:
        cli.main(["motors", "check"])
    assert ended.value.code == 130
    assert capsys.readouterr().err == "\nstopped by Ctrl-C\n"


def test_can_link_state_reads_sysfs(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`up` (brought up and not reported down, whatever operstate says beyond that), `down`, and None for an
    interface that is absent or is not a CAN link."""
    net = tmp_path / "net"
    setup_fakes.net_link(net, "can_follower_l", up=True)
    setup_fakes.net_link(net, "can1", up=False)
    setup_fakes.net_link(net, "slcan0", up=True, operstate="unknown")
    setup_fakes.net_link(net, "can2", up=True, operstate="lowerlayerdown")
    setup_fakes.net_link(net, "eth0", up=True, kind="1")
    monkeypatch.setattr(cli, "NET_DIR", net)

    assert cli.can_link_state("can_follower_l") == "up"
    assert cli.can_link_state("can1") == "down"
    assert cli.can_link_state("slcan0") == "up", "brought up, and an slcan link reports operstate unknown"
    assert cli.can_link_state("can2") == "down"
    assert cli.can_link_state("eth0") is None
    assert cli.can_link_state("nothing") is None


def test_bring_up_links_skips_what_is_up_and_brings_up_what_is_down(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A station's links are down after a reboot: `robot up` runs the one `ip link` an operator would, leaves an
    already-up link alone (so a second run changes nothing), and asks for each link once however often it is named."""
    net = tmp_path / "net"
    setup_fakes.net_link(net, "can_follower_l", up=True)
    setup_fakes.net_link(net, "can_follower_r", up=False)
    monkeypatch.setattr(cli, "NET_DIR", net)
    ran: list[list[str]] = []

    def run_command(argv: Sequence[str]) -> int:
        ran.append(list(argv))
        setup_fakes.net_link(net, argv[4], up=True)
        return 0

    out = io.StringIO()
    assert cli.bring_up_links(["can_follower_l", "can_follower_r", "can_follower_r"], out=out, run_command=run_command)
    assert ran == [["sudo", "ip", "link", "set", "can_follower_r", "up", "type", "can", "bitrate", "1000000"]]
    assert "can_follower_r is down" in out.getvalue() and "can_follower_l" not in out.getvalue()

    ran.clear()
    assert cli.bring_up_links(["can_follower_l", "can_follower_r"], out=io.StringIO(), run_command=run_command)
    assert ran == [], "both are up now: running it again touches nothing"


def test_bring_up_links_reports_a_missing_link_and_a_command_that_failed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither is a crash: the operator is told which link and what to run, and `robot up` stops before the agent."""
    net = tmp_path / "net"
    setup_fakes.net_link(net, "can_follower_r", up=False)
    monkeypatch.setattr(cli, "NET_DIR", net)

    missing = io.StringIO()
    assert cli.bring_up_links(["can_follower_l"], out=missing, run_command=lambda argv: 0) is False
    assert "no CAN link called can_follower_l" in missing.getvalue()

    refused = io.StringIO()
    assert cli.bring_up_links(["can_follower_r"], out=refused, run_command=lambda argv: 1) is False
    assert "could not bring can_follower_r up" in refused.getvalue()


def busy_link(net: pathlib.Path, name: str) -> Callable[[Sequence[str]], int]:
    """A fake `ip` for a link a bus fault left with its UP flag set and its operstate `down`.

    Like the kernel, it refuses a new bitrate on a link whose UP flag is set (`RTNETLINK answers: Device or resource
    busy`, exit 2) and accepts `up type can bitrate ...` once `ip link set <link> down` has cleared the flag.

    @param net The fake sysfs directory.
    @param name The link.
    @return The `run_command` to hand to bring_up_links().
    """
    setup_fakes.net_link(net, name, up=True, operstate="down")

    def run_command(argv: Sequence[str]) -> int:
        assert list(argv[:5]) == ["sudo", "ip", "link", "set", name], argv
        if list(argv[5:]) == ["down"]:
            setup_fakes.net_link(net, name, up=False)
            return 0
        if robstride.IFF_UP & int((net / name / "flags").read_text().strip(), 16):
            return 2
        setup_fakes.net_link(net, name, up=True)
        return 0

    return run_command


def test_bring_up_links_takes_a_busy_link_down_before_bringing_it_up(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A link stuck after a bus fault keeps its UP flag, so it reads `down` here, and
    `ip link set <link> up type can bitrate 1000000` on it answers `Device or resource busy`. `robot up` runs
    `ip link set <link> down` first for such a link, and the link comes up; a link whose UP flag is clear (the
    reboot case) still gets the one up command, as before."""
    net = tmp_path / "net"
    monkeypatch.setattr(cli, "NET_DIR", net)
    run_command = busy_link(net, "can_follower_l")
    ran: list[list[str]] = []

    def recorded(argv: Sequence[str]) -> int:
        ran.append(list(argv))
        return run_command(argv)

    out = io.StringIO()
    assert cli.bring_up_links(["can_follower_l"], out=out, run_command=recorded) is True
    assert ran == [
        ["sudo", "ip", "link", "set", "can_follower_l", "down"],
        ["sudo", "ip", "link", "set", "can_follower_l", "up", "type", "can", "bitrate", "1000000"],
    ]
    assert "taking it down first: sudo ip link set can_follower_l down" in out.getvalue()
    assert cli.can_link_state("can_follower_l") == "up"


def test_a_busy_link_that_still_will_not_come_up_names_both_commands(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure line for a busy link gives the down and the up command to run by hand, in order, never the up
    command alone, which that link refuses."""
    net = tmp_path / "net"
    setup_fakes.net_link(net, "can_follower_r", up=True, operstate="down")
    monkeypatch.setattr(cli, "NET_DIR", net)

    refused = io.StringIO()
    assert cli.bring_up_links(["can_follower_r"], out=refused, run_command=lambda argv: 1) is False
    assert (
        "could not bring can_follower_r up; run 'sudo ip link set can_follower_r down' then "
        "'sudo ip link set can_follower_r up type can bitrate 1000000' yourself"
    ) in refused.getvalue()


def test_a_canable_needs_no_link_unless_an_slcand_holds_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    net, proc, dev = tmp_path / "net", tmp_path / "proc", tmp_path / "dev"
    setup_fakes.net_link(net, "can1", up=False)
    dev.mkdir()
    for name in ("ttyACM0", "ttyACM1", "ttyACM2"):
        (dev / name).write_text("")
    setup_fakes.slcand(proc, 40, "-o", "-f", "-s8", str(dev / "ttyACM1"), "can1")
    setup_fakes.slcand(proc, 41, "-o", "-s8", str(dev / "ttyACM2"))
    monkeypatch.setattr(cli, "NET_DIR", net)
    monkeypatch.setattr(robstride, "PROC_DIR", proc)
    monkeypatch.setattr(robstride, "SERIAL_DEVICE_PREFIX", f"{dev}/")
    ran: list[list[str]] = []

    def run_command(argv: Sequence[str]) -> int:
        ran.append(list(argv))
        setup_fakes.net_link(net, argv[4], up=True)
        return 0

    assert cli.bring_up_links([str(dev / "ttyACM0")], out=io.StringIO(), run_command=run_command)
    assert ran == [], "opened directly: no link, no sudo"
    out = io.StringIO()
    assert cli.bring_up_links([str(dev / "ttyACM1")], out=out, run_command=run_command)
    assert ran == [["sudo", "ip", "link", "set", "can1", "up"]] and "can1 is down" in out.getvalue()
    unnamed = io.StringIO()
    assert cli.bring_up_links([str(dev / "ttyACM2")], out=unnamed, run_command=run_command) is False
    assert "sudo kill 41" in unnamed.getvalue()
    gone = io.StringIO()
    assert cli.bring_up_links([str(dev / "ttyACM9")], out=gone, run_command=run_command) is False
    assert "is the arm's CANable plugged in" in gone.getvalue()


def test_robot_up_brings_the_configs_links_up_before_it_starts_the_agent(
    home: cli.Paths,
    agent: Callable[[str], list[str]],
    cleanup: None,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The channels come from the config's two arms, and a link that cannot come up stops the start."""
    net = tmp_path / "net"
    for name in ("can_follower_l", "can_follower_r"):
        setup_fakes.net_link(net, name, up=False)
    monkeypatch.setattr(cli, "NET_DIR", net)
    home.config.write_text(
        "follower_l: {can_channel: can_follower_l}\nfollower_r: {can_channel: can_follower_r}\n", encoding="utf-8"
    )
    ran: list[list[str]] = []

    def run_command(argv: Sequence[str]) -> int:
        ran.append(list(argv))
        setup_fakes.net_link(net, argv[4], up=True)
        return 0

    out = io.StringIO()
    code = cli.robot_up(home, command=agent(AGENT_SCRIPT), out=out, run_command=run_command, ready=up_now)
    assert code == 0, out.getvalue()
    assert [argv[4] for argv in ran] == ["can_follower_l", "can_follower_r"]
    assert cli.running_pid(home) is not None

    cli.robot_down(home, out=io.StringIO())
    setup_fakes.net_link(net, "can_follower_l", up=False)
    refused = io.StringIO()
    code = cli.robot_up(home, command=agent(AGENT_SCRIPT), out=refused, run_command=lambda argv: 1, ready=up_now)
    assert code == 1
    assert "could not bring can_follower_l up" in refused.getvalue()
    assert cli.running_pid(home) is None, "the agent never started"


def test_up_passes_the_system_0_word_to_the_agent(home: cli.Paths, monkeypatch: pytest.MonkeyPatch) -> None:
    """`robot up` leaves System 0 to the agent's default (on for a YAM); `sys0` reaches it as `--sys0`, `nosys0` as
    `--no-sys0`, and both together are refused."""
    monkeypatch.setenv(cli.HOME_ENV, str(home.home))
    seen: dict[str, object] = {}

    def fake_up(paths: cli.Paths, **kwargs: object) -> int:
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "robot_up", fake_up)
    assert cli.run(["robot", "up", "sys0", "--api-port", "9091"]) == 0
    assert seen["sys0"] is True and seen["extra_args"] == ["--api-port", "9091"]
    assert cli.run(["robot", "up", "--host", "127.0.0.1", "nosys0"]) == 0
    assert seen["sys0"] is False and seen["extra_args"] == ["--host", "127.0.0.1"]
    assert cli.run(["robot", "up", "--host", "127.0.0.1"]) == 0
    assert seen["sys0"] is None and seen["extra_args"] == ["--host", "127.0.0.1"]
    seen.clear()
    assert cli.run(["robot", "up", "sys0", "nosys0"], out=io.StringIO()) == 1
    assert not seen, "nothing started"
    config = pathlib.Path("/x/station.yaml")
    assert cli.station_command(config, ["--api-port", "9091"], lock=home.lock, sys0=True)[-1] == "--sys0"
    assert cli.station_command(config, lock=home.lock, sys0=False)[-1] == "--no-sys0"
    assert not {"--sys0", "--no-sys0"} & set(cli.station_command(config, lock=home.lock))


def test_up_passes_the_head_word_to_the_agent(home: cli.Paths, monkeypatch: pytest.MonkeyPatch) -> None:
    """No word leaves the head to the config (a YAM's `eef`); `eef` and `joint` each force one, `noeef` is still
    `joint`, and each reaches the agent as its flag beside a System 0 word and the pass-through flags."""
    monkeypatch.setenv(cli.HOME_ENV, str(home.home))
    seen: dict[str, object] = {}

    def fake_up(paths: cli.Paths, **kwargs: object) -> int:
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "robot_up", fake_up)
    assert cli.run(["robot", "up", "--host", "127.0.0.1"]) == 0
    assert seen["eef"] is None
    assert cli.run(["robot", "up", "eef", "--host", "127.0.0.1"]) == 0
    assert seen["eef"] is True
    assert cli.run(["robot", "up", "joint", "nosys0", "--host", "127.0.0.1"]) == 0
    assert seen["eef"] is False and seen["sys0"] is False and seen["extra_args"] == ["--host", "127.0.0.1"]
    assert cli.run(["robot", "up", "noeef", "--host", "127.0.0.1"]) == 0
    assert seen["eef"] is False
    config = pathlib.Path("/x/station.yaml")
    assert cli.station_command(config, lock=home.lock, eef=False)[-1] == "--no-eef"
    assert cli.station_command(config, lock=home.lock, eef=True)[-1] == "--eef"
    assert not {"--eef", "--no-eef"} & set(cli.station_command(config, lock=home.lock))


@pytest.mark.parametrize(
    ("eef", "line"),
    [(True, "head:      eef"), (False, "head:      joint"), (None, "head:      the config's action_space")],
    ids=["eef", "joint", "from-the-config"],
)
def test_up_report_says_which_head_runs(
    home: cli.Paths, agent: Callable[[str], list[str]], cleanup: None, eef: bool | None, line: str
) -> None:
    """The report names the head each word forces, and says the config decides when no word does."""
    out = io.StringIO()
    assert cli.robot_up(home, command=agent(AGENT_SCRIPT), eef=eef, out=out, ready=up_now) == 0, out.getvalue()
    assert line in out.getvalue()


@pytest.mark.parametrize(
    ("body", "sys0", "line"),
    [
        ("yam", None, "System 0:  on"),
        ("yam", True, "System 0:  on"),
        ("yam", False, "System 0:  off"),
        ("makers", None, None),
    ],
    ids=["yam-default", "yam-on", "yam-off", "makers-default"],
)
def test_up_report_says_whether_system_0_is_on(
    home: cli.Paths,
    agent: Callable[[str], list[str]],
    cleanup: None,
    body: str,
    sys0: bool | None,
    line: str | None,
) -> None:
    """The report says System 0 drives the motors when it does (a YAM's default), and off when it was turned off."""
    home.config.write_text(home.config.read_text(encoding="utf-8") + f"body: {body}\n", encoding="utf-8")
    out = io.StringIO()
    assert cli.robot_up(home, command=agent(AGENT_SCRIPT), sys0=sys0, out=out, ready=up_now) == 0, out.getvalue()
    if line is None:
        assert "System 0" not in out.getvalue()
    else:
        assert line in out.getvalue()


def test_motors_runs_on_the_yam_links_the_config_names_and_refuses_another_body(
    home: cli.Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[object] = []
    monkeypatch.setattr(cli, "motors", lambda action, **kwargs: seen.append((action, kwargs["links"])) or 0)
    home.config.write_text(
        "body: yam\nfollower_l: {can_channel: can_left}\nfollower_r: {can_channel: can_right}\n", encoding="utf-8"
    )
    assert cli.run(["--home", str(home.home), "motors", "check"]) == 0
    assert seen == [("check", ["can_left", "can_right"])]
    for body in ("makers", "metal", "so101", "custom"):
        home.config.write_text(f"body: {body}\nfollower_l: {{can_channel: /dev/ttyACM0}}\n", encoding="utf-8")
        report = io.StringIO()
        assert cli.run(["--home", str(home.home), "motors", "set-mit"], out=report) == 1
        assert f"says body: {body}, whose motors need no switch" in report.getvalue()
    assert seen == [("check", ["can_left", "can_right"])]


def test_robot_run_without_a_key_stops_before_anything_moves(
    monkeypatch: pytest.MonkeyPatch, home: cli.Paths, credentials_path: pathlib.Path
) -> None:
    from robots import run

    example = pathlib.Path(__file__).resolve().parent.parent / "configs" / "custom.example.yaml"
    home.config.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    credentials_path.write_text("DREAM_API_URL=https://platform.example.com\n", encoding="utf-8")
    started: list[str] = []

    async def fake_main(app: Any, checkpoint: str | None, *, dry_run: bool) -> None:
        started.append("loop")

    monkeypatch.setattr(run, "main", fake_main)
    monkeypatch.setattr(cli, "bring_up_links", lambda *args, **kwargs: started.append("links") or True)
    report = io.StringIO()
    assert cli.robot_run(home, out=report) == 1
    assert started == []
    assert (
        f"account: no api key: neither the environment nor {credentials_path} sets DREAM_API_KEY" in report.getvalue()
    )
    assert "the Setup API page on https://platform.example.com" in report.getvalue()


def test_robot_run_dispatches_its_flags(monkeypatch: pytest.MonkeyPatch, home: cli.Paths) -> None:
    """`robot run` takes the config, the task, the model, the action space, the dry run, the chunk view and, unlisted
    in its help since it is not advised, the skip of an arm file's tool point check, and nothing else."""
    monkeypatch.setenv(cli.HOME_ENV, str(home.home))
    seen: dict[str, object] = {}

    def fake_run(paths: cli.Paths, **kwargs: object) -> int:
        seen.update(paths=paths, **kwargs)
        return 0

    monkeypatch.setattr(cli, "robot_run", fake_run)

    assert cli.run(["robot", "run", "--task", "wave", "--checkpoint", "alice/ft", "--dry-run"]) == 0
    assert seen == {
        "paths": home,
        "config_path": None,
        "task": "wave",
        "checkpoint": "alice/ft",
        "action_space": None,
        "dry_run": True,
        "view_port": None,
        "skip_tcp_calibration": False,
        "out": sys.stdout,
    }
    assert cli.run(["robot", "run", "--skip-tcp-calibration"]) == 0
    assert seen["skip_tcp_calibration"] is True
    assert cli.run(["robot", "run", "--config", str(home.config)]) == 0
    assert seen["config_path"] == home.config and seen["dry_run"] is False
    assert cli.run(["robot", "run", "--action-space", "joint"]) == 0
    assert seen["action_space"] == "joint"
    with pytest.raises(SystemExit):
        cli.run(["robot", "run", "--action-space", "joints"])
    assert cli.run(["robot", "run", "--view"]) == 0
    assert seen["view_port"] == 8093
    assert cli.run(["robot", "run", "--view", "--view-port", "9000"]) == 0
    assert seen["view_port"] == 9000
    with pytest.raises(SystemExit):
        cli.run(["robot", "run", "--api-port", "9091"])


def test_robot_runs_help_leaves_out_the_skip_of_the_tool_point_check(capsys: pytest.CaptureFixture[str]) -> None:
    """Skipping an arm file's tool point check is not advised, so `robot run --help` does not list it; the refusal
    names it (`robots.run.check_tool_point()`)."""
    with pytest.raises(SystemExit):
        cli.run(["robot", "run", "--help"])
    shown = capsys.readouterr().out
    assert "--dry-run" in shown and "skip-tcp" not in shown


def test_robot_run_runs_the_loop_here_unless_the_station_owns_the_arms(
    monkeypatch: pytest.MonkeyPatch, home: cli.Paths
) -> None:
    """The config is loaded with the task override and run in this process; a running station, a missing file or a
    config the loop refuses ends it with a line saying why."""
    import dream

    from robots import run

    example = pathlib.Path(__file__).resolve().parent.parent / "configs" / "custom.example.yaml"
    home.config.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    seen: dict[str, object] = {}

    async def fake_main(app: Any, checkpoint: str | None, *, dry_run: bool) -> None:
        seen.update(
            task=app.task, body=app.body, checkpoint=checkpoint, dry_run=dry_run, action_space=app.action_space
        )

    monkeypatch.setattr(run, "main", fake_main)
    report = io.StringIO()

    assert cli.robot_run(home, task="wave", checkpoint="alice/ft", dry_run=True, out=report) == 0
    assert seen == {"task": "wave", "body": "custom", "checkpoint": "alice/ft", "dry_run": True, "action_space": "eef"}
    assert "dry run on" in report.getvalue()
    assert cli.robot_run(home, checkpoint="alice/ft", action_space="joint", out=report) == 0
    assert seen["action_space"] == "joint", "the flag replaces the file's action space for this run"

    assert cli.robot_run(home, config_path=home.home / "missing.yaml", out=report) == 1
    assert "no config at" in report.getvalue()
    seen.clear()
    assert cli.robot_run(home, dry_run=True, view_port=8093, out=report) == 1
    assert "a dry run saves none" in report.getvalue() and not seen, "refused before the loop"
    assert cli.robot_run(home, task="", out=report) == 1
    assert "config: --task must be a non-empty prompt" in report.getvalue()

    def raising(exc: Exception) -> Callable[..., Any]:
        async def main(app: Any, checkpoint: str | None, *, dry_run: bool) -> None:
            raise exc

        return main

    for exc, line in (
        (dream.DreamAuthError("DA-1000 UNAUTHORIZED (unknown_key)"), "account: DA-1000 UNAUTHORIZED (unknown_key)"),
        (dream.DreamBadRequest("DA-4000 BAD_REQUEST (unknown_checkpoint)"), "refused: DA-4000 BAD_REQUEST"),
        (errors.CameraError("camera 'top' returned no frame for 2 s"), "stopped: camera 'top' returned no frame"),
        (dream.DreamProtocolError("cannot reach the Dream service at http://x"), "stopped: cannot reach the Dream"),
    ):
        monkeypatch.setattr(run, "main", raising(exc))
        assert cli.robot_run(home, out=report) == 1 and line in report.getvalue()
    with arms_lock.acquire(home.lock, arms_lock.STATION):
        assert cli.robot_run(home, out=report) == 1 and "held by the station agent" in report.getvalue()


def test_robot_up_passes_ik_solve_to_the_agent_and_the_agent_overrides_the_config_for_that_run() -> None:
    """`dreamcli robot up --ik-solve traj` is an agent flag: robot up hands it through as it is, and the agent's
    config gets that solve for this run only; without it the config's own (absent: tick, as before)."""
    from robots import run, station

    home = cli.Paths(pathlib.Path("/tmp/station-x"))
    cmd = cli.station_command(pathlib.Path("/tmp/station.yaml"), ["--ik-solve", "traj"], lock=home.lock)
    assert cmd[cmd.index("--ik-solve") + 1] == "traj"
    app = conftest.make_app_config()
    assert run.with_ik_solve(app, None) is app and app.ik_solve is None
    assert run.with_ik_solve(app, "traj").ik_solve == "traj"
    with pytest.raises(errors.ConfigError, match="IK solve must be one of"):
        run.with_ik_solve(app, "fast")
    with pytest.raises(SystemExit):
        station.main(["--config", "/tmp/station.yaml", "--ik-solve", "fast"])  # argparse refuses it before loading
