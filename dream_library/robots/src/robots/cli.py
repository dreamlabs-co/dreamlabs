"""`dreamcli`: the station's command line, installed as a console script with the package.

Eight commands, seven of them on one station directory. `dreamcli setup` opens the setup wizard, a local web page
(`robots.setup_wizard`) that asks for the arms and the cameras (the account is the credentials
file, `dream.credentials`, and the model is code), and writes the station config to #CONFIG_NAME
under the station directory. `dreamcli tcp` opens that page's tool point tool alone (`robots.setup_wizard.tcp_main`),
for a config written by hand: an arm's URDF and its tool point in, the arm file its `custom.ik` names out, beside it.
`dreamcli robot up` brings the arms' CAN links up if they are down (bring_up_links(): `ip link set <channel> up
type can bitrate 1000000` through sudo, after `ip link set <channel> down` for a link a bus fault left set up but not
running, skipped for a link that is already up, so it is safe to run again; a CANable
named by its serial device needs no link, unless an `slcand` holds it, whose link it brings up the same way), then
starts the station agent (`python -m robots.station`, the resident process
with Start / Stop / Home over HTTP) on that file as a detached process whose
output goes to #LOG_NAME and whose pid is kept in #PID_NAME, and waits until the agent's API answers (the arms
and the cameras are open by then) or the agent has died, and says which; `dreamcli robot down` sends it
SIGINT, which is the agent's clean stop (a running loop is stopped, the arms are switched off and go limp, the servers
close), and waits for it to exit, and with no agent running switches a YAM's motors off over CAN directly (a station
whose PC died left them holding); `dreamcli robot log` prints the last #LOG_LINES lines of the
log. `dreamcli robot run` runs one task in this terminal instead, on the same loop with no agent (robot_run(),
`robots.run`): it starts from home, Ctrl-C homes the arms and switches them off, `--dry-run` writes no target, and
`--view` serves a live 3D page of each chunk on 127.0.0.1 (`robots.run_view`).
`dreamcli motors check` reads the firmware control mode of every motor on a YAM's two links, the ones the station
config names (`robots.motor_mode`; the links are brought up first, as `robot up` does; a config of another body is
refused, since only a YAM's motors have that mode to switch) and `dreamcli motors set-mit` switches
any motor that is not in MIT mode, the one mode the station can drive, and stores it: new OpenYAM arms often
ship with the gripper motor in position-speed mode, which reads its position and never moves. Apart from that
switch, which a power cycle of the arm then makes real, `setup` (the wizard opens the arms, and its powered checks
and Home move them) and `robot run`, nothing here touches the arms: the agent does, on request, as `robots.station`
documents. `dreamcli example` needs no robot and no station directory: it runs an example that ships with the dream
library (`robots.examples`), today `yam-mujoco`, Dream YAM Sim, two simulated YAM arms in MuJoCo that take Dream-1
tasks.

```bash
dreamcli setup                      # the wizard, on port 8092 (open the private address it prints)
dreamcli tcp                        # the tool point tool alone, on port 8094: writes arms/<name>.json, prints it
dreamcli robot up                   # the agent on ~/.dreamlabs/station.yaml; --config names another file
dreamcli robot up nosys0            # the same without System 0, which a YAM runs by default (robots.sys0)
dreamcli robot up joint             # the model's joint head (`noeef` is the old name)
dreamcli robot up eef               # the model's EEF head (the default), tool poses solved to joints by dream's IK
dreamcli robot up --ik-solve tick   # an eef run's IK row by row (the default is traj: one solve of the whole chunk)
dreamcli robot down                 # SIGINT, wait, then SIGTERM and SIGKILL if it will not go
dreamcli robot log [-n 40]          # the tail of ~/.dreamlabs/station.log
dreamcli robot run --task "..."     # one task in this terminal, no agent; --checkpoint, --dry-run, --view, --config
dreamcli robot run --action-space joint   # the same on the model's own joints: a fine-tune of this robot only
dreamcli motors check               # every motor's control mode on both links; --link names one, --motor some ids
dreamcli motors set-mit             # switches what is not MIT and stores it; then power-cycle the arms and check
dreamcli example yam-mujoco         # Dream YAM Sim, no robot; --show-side-panel, --verbose, --self-test
```

The station directory is #DEFAULT_HOME, or the value of the #HOME_ENV environment variable;
`debug_scripts/station_run.sh` is the tmux-based alternative for a station that keeps its
config elsewhere.

@see main(), robot_up(), robot_run(), robot_down(), robot_log(), motors(), Paths, robots.examples.run()
"""

import argparse
import asyncio
import collections
import dataclasses
import errno
import importlib
import json
import logging
import os
import pathlib
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import IO

import yaml

from robots import arms_lock, config, errors, motor_arm, motor_mode, robstride, startup_failure, station
from robots.run import SKIP_TCP_FLAG

## The word after `robot up` that runs the agent with System 0 driving the motors (`robots.sys0`), a YAM's default.
SYS0_WORD: str = "sys0"

## The word after `robot up` that runs the agent without System 0: the plain command writer drives the motors.
NO_SYS0_WORD: str = "nosys0"

## The word after `robot up` that runs the model's joint head (the action space `joint`): the agent's `--no-eef`.
## It is a YAM's default, and the word is how any other body asks for it.
JOINT_WORD: str = "joint"

## The word after `robot up` that runs the model's EEF head, its tool poses solved to joints with dream's IK: the
## agent's `--eef`. Every body but the YAM runs it by default; a YAM asks for it with this word.
EEF_WORD: str = "eef"

## An older name for `joint`, still accepted so a station's own notes keep working.
NO_EEF_WORD: str = "noeef"

## Where the config, the log and the pid file live unless #HOME_ENV says otherwise.
DEFAULT_HOME = "~/.dreamlabs"

## Environment variable naming another station directory (tests, or a box with several stations).
HOME_ENV = "DREAMCLI_HOME"

## The station config the wizard writes and `robot up` reads, inside the station directory.
CONFIG_NAME = "station.yaml"

## The agent's stdout and stderr, appended across starts, inside the station directory.
LOG_NAME = "station.log"

## The running agent's pid, inside the station directory; removed by `robot down` and when found stale.
PID_NAME = "station.pid"

## Lines `robot log` prints unless `-n` says otherwise.
LOG_LINES = 40

## The bitrate every YAM CAN link runs at; what `robot up` brings a down link up with.
CAN_BITRATE = 1_000_000

## Where the kernel publishes each network interface, and the interface type a SocketCAN link reads (ARPHRD_CAN).
NET_DIR = pathlib.Path("/sys/class/net")
ARPHRD_CAN = "280"

STOP_GRACE_S = 60.0

PARK_NOTICE_S = 2.0

## Seconds `robot down` waits after SIGTERM before SIGKILL.
KILL_GRACE_S = 5.0

## Longest `robot up` waits for the agent's API to answer before it stops watching. The arms open in seconds and
## each RealSense gets `cameras.REALSENSE_START_TIMEOUT_MS` (5 s) to deliver a frame, so an agent that has not
## answered by then is stuck or dying, and either way the operator is told to look at the log.
START_TIMEOUT_S = 60.0

## Seconds between two looks at the process and its API while `robot up` waits.
START_POLL_S = 0.25

## Seconds one probe of the agent's API may take; the agent binds and answers at once once it is up.
START_PROBE_TIMEOUT_S = 1.0

## Log lines `robot up` prints when the agent died, or has not answered, after the start.
START_FAIL_LINES = 15

## How far back a failed start's log is read for the station's could-not-start line (robots.startup_failure).
START_SCAN_LINES = 2000

SHUTDOWN_LINES = 50

LOG_HEADER = "=== dreamcli robot up:"


@dataclasses.dataclass(frozen=True)
class Paths:
    """The station directory and the four files in it.

    Attributes:
    - home: The station directory, expanded and absolute.
    """

    home: pathlib.Path

    @property
    def config(self) -> pathlib.Path:
        """The station config, #CONFIG_NAME."""
        return self.home / CONFIG_NAME

    @property
    def log(self) -> pathlib.Path:
        """The agent's log, #LOG_NAME."""
        return self.home / LOG_NAME

    @property
    def pid(self) -> pathlib.Path:
        """The agent's pid file, #PID_NAME."""
        return self.home / PID_NAME

    @property
    def lock(self) -> pathlib.Path:
        """The arms' lock (`robots.arms_lock`): whoever drives the arms holds it."""
        return self.home / arms_lock.LOCK_NAME


def paths(home: str | os.PathLike[str] | None = None) -> Paths:
    """The station directory to use: `home`, else #HOME_ENV, else #DEFAULT_HOME.

    @param home An explicit directory, or None to consult the environment.
    @return The #Paths; the directory itself is created by whichever command first writes into it.
    """
    chosen = os.fspath(home) if home is not None else os.environ.get(HOME_ENV) or DEFAULT_HOME
    return Paths(pathlib.Path(os.path.expanduser(chosen)).resolve())


def station_command(
    config: pathlib.Path,
    extra_args: Sequence[str] = (),
    *,
    lock: pathlib.Path,
    sys0: bool | None = None,
    eef: bool | None = None,
) -> list[str]:
    """The command `robot up` runs: this interpreter on `robots.station`, with the config, the flags and the lock.

    @param config The station config.
    @param extra_args Flags handed to `robots.station` as they are (`--host`, `--api-port`).
    @param lock The station directory's arms' lock.
    @param sys0 True adds `--sys0` (`robot up sys0`), False `--no-sys0` (`robot up nosys0`); None leaves the
        agent's default, System 0 on a YAM and the plain loop on any other body.
    @param eef True adds `--eef` (`robot up eef`), False `--no-eef` (`robot up joint`, or the older `noeef`);
        None leaves the config's own head, `eef` unless the file says `joint`.
    @return The argv.
    """
    return [
        sys.executable,
        "-m",
        "robots.station",
        "--config",
        str(config),
        *extra_args,
        "--arms-lock",
        str(lock),
        *({None: [], True: ["--sys0"], False: ["--no-sys0"]}[sys0]),
        *([] if eef is None else ["--eef" if eef else "--no-eef"]),
    ]


def running_pid(paths: Paths) -> int | None:
    """The pid in the pid file when that process exists; a stale or unreadable pid file is removed.

    @param paths The station directory.
    @return The pid, or None when no agent started by `robot up` is running.
    """
    try:
        pid = int(paths.pid.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    if _alive(pid):
        return pid
    paths.pid.unlink(missing_ok=True)
    return None


def can_link_state(channel: str) -> str | None:
    """Whether one CAN link is up, from sysfs.

    @param channel The interface name, e.g. `can_follower_l`.
    @return `"up"`, `"down"`, or None when the interface is absent or is not a CAN link.
    """
    entry = NET_DIR / channel
    try:
        if (entry / "type").read_text().strip() != ARPHRD_CAN:
            return None
        return "up" if robstride.link_state(entry) == "up" else "down"
    except OSError:
        return None


def _flagged_up(channel: str) -> bool:
    """Whether a link's UP flag is set in sysfs, whatever its operstate says.

    @param channel The interface name, e.g. `can_follower_l`.
    @return True when the flag is set; False when it is clear or sysfs cannot say.
    """
    try:
        return bool(int((NET_DIR / channel / "flags").read_text().strip(), 16) & robstride.IFF_UP)
    except (OSError, ValueError):
        return False


def bring_up_links(
    channels: Sequence[str],
    *,
    out: IO[str] = sys.stdout,
    run_command: Callable[[Sequence[str]], int] = lambda argv: subprocess.call(argv),
) -> bool:
    """Brings every down CAN link up at #CAN_BITRATE, and leaves an already-up one alone.

    A station's links come back down after a reboot, and the agent then fails to open the arms with an error about
    the bus rather than about the link, so `robot up` does what the operator would have typed. It is safe to run
    again: a link that is already up is skipped, which is also why a second `robot up` never disturbs a running
    station's buses. `ip link set` needs root, so this asks for the sudo password when the shell has no ticket.

    A link that is not up but has its UP flag set (a bus fault left it set up with its bus not running) is
    taken down first: the kernel refuses a new bitrate on a link whose flag is set, with `RTNETLINK answers: Device
    or resource busy`, so the up command alone could never bring it back. A link whose flag is clear, the reboot case,
    gets the one up command as before.

    @param channels The CAN interfaces and CANable devices the config names, in order.
    @param out Where the report goes.
    @param run_command Runs one argv and returns its exit code; injected for tests.
    @return True when every named link is up afterwards; False when one is missing or could not be brought up.
    """
    ok = True
    for channel in dict.fromkeys(channels):
        speed = ["type", "can", "bitrate", str(CAN_BITRATE)]
        if channel.startswith(robstride.SERIAL_DEVICE_PREFIX):
            if not os.path.exists(channel):
                print(f"no adapter at {channel}: is the arm's CANable plugged in? ('ls /dev/serial/by-id')", file=out)
                ok = False
                continue
            holder = robstride.slcan_holder(channel)
            if holder is None:
                continue
            if holder.link is None:
                print(
                    f"{channel} is held by slcand (pid {holder.pid}) under a link it did not name; stop it "
                    f"(sudo kill {holder.pid}) and the station opens the adapter directly",
                    file=out,
                )
                ok = False
                continue
            channel, speed = holder.link, []
        state = can_link_state(channel)
        if state is None:
            print(
                f"no CAN link called {channel}: is the arm's adapter plugged in? ('ip -br link show type can')",
                file=out,
            )
            ok = False
            continue
        if state == "up":
            continue
        argv = ["sudo", "ip", "link", "set", channel, "up", *speed]
        down = ["sudo", "ip", "link", "set", channel, "down"]
        stuck = _flagged_up(channel)
        if stuck:
            print(f"{channel} is set up but its bus is not running; taking it down first: {' '.join(down)}", file=out)
            run_command(down)  # Its exit code is not the verdict: the up after it and the link's state are.
        print(f"{channel} is down; bringing it up: {' '.join(argv)}", file=out)
        if run_command(argv) != 0 or can_link_state(channel) != "up":
            fix = f"'{' '.join(down)}' then '{' '.join(argv)}'" if stuck else "it"
            print(f"could not bring {channel} up; run {fix} yourself, then 'dreamcli robot up' again", file=out)
            ok = False
    return ok


def robot_up(
    paths: Paths,
    *,
    config: pathlib.Path | None = None,
    extra_args: Sequence[str] = (),
    sys0: bool | None = None,
    eef: bool | None = None,
    command: Sequence[str] | None = None,
    out: IO[str] = sys.stdout,
    sleep: Callable[[float], object] = time.sleep,
    run_command: Callable[[Sequence[str]], int] = lambda argv: subprocess.call(argv),
    ready: Callable[[str, int], bool] | None = None,
    port_free: Callable[[str, int], bool] | None = None,
) -> int:
    """Starts the station agent detached, records its pid, waits until it is up, and reports its API and log.

    The arms' CAN links are brought up first (bring_up_links(), from the channels the config names), because they
    come back down after a reboot and the agent can only report that as a bus error; a link already up is left
    alone, so running this twice changes nothing.

    The report names two things: the platform the credentials file names (where the robot's live view is, the
    address an operator opens next) and the log. The station page and the agent's own HTTP API are left out: the
    live view is where an operator drives the robot, and the API's `/status` answers JSON, which reads as a broken
    page to someone who clicks it.

    The process gets its own session (it outlives this shell and its SIGINT), stdin from `/dev/null`, and both
    output streams appended to the log. Then this waits, up to #START_TIMEOUT_S, for one of two things: the agent's
    API answers, which it does only after the arms and every camera have opened (`robots.station.serve`), and its
    address is reported; or the process exits, and the exit is reported with the log's tail, since a camera
    that never delivers a frame, a config the agent refuses or a missing driver all end it that way. An agent that
    has done neither by the deadline is left running and reported as not answering, with the log's tail.

    @param paths The station directory.
    @param config The config to run; the directory's #CONFIG_NAME when None.
    @param extra_args Flags passed through to `robots.station`.
    @param sys0 Whether System 0 drives the motors: True as `robot up sys0` asks, False as `robot up nosys0` asks,
        None for the agent's default (on for a YAM, the one body it drives).
    @param eef Which head runs: True as `robot up eef` asks, False as `robot up joint` (or `noeef`) asks, None
        for the config's own, `eef` unless the file says `joint`.
    @param command The argv to run instead of station_command() (tests); the config still has to exist.
    @param out Where the report goes.
    @param sleep Blocking sleep, seconds; injected so tests do not wait.
    @param run_command Runs one argv; injected for tests. Used for the `ip link` calls only.
    @param ready Answers whether the agent's API on `(host, port)` is up; api_answers() when None, a fake in tests.
    @param port_free Answers whether `(host, port)` can be bound.
    @return The exit code.
    """
    config_path = config if config is not None else paths.config
    if not config_path.is_file():
        hint = "run 'dreamcli setup' first" if config is None else "no such file"
        print(f"no station config at {config_path}: {hint}", file=out)
        return 1
    pid = running_pid(paths)
    if pid is not None:
        print(f"the station is already running (pid {pid}); 'dreamcli robot down' stops it", file=out)
        return 1
    held = arms_lock.held_by(paths.lock)
    if held is not None:
        print(held, file=out)
        return 1
    bind_host, api_port = _serve_flags(extra_args)
    if not (port_free if port_free is not None else api_port_free)(bind_host, api_port):
        print(
            f"port {api_port} on {bind_host} is taken by another program (an old station, a web server): stop it, "
            "or start on another port with 'dreamcli robot up --api-port <port>'",
            file=out,
        )
        return 1
    if not bring_up_links(_can_channels(config_path), out=out, run_command=run_command):
        return 1
    paths.home.mkdir(parents=True, exist_ok=True)
    argv = (
        list(command)
        if command is not None
        else station_command(config_path, extra_args, lock=paths.lock, sys0=sys0, eef=eef)
    )
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with open(paths.log, "ab") as log:
        log.write(f"\n{LOG_HEADER} {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(argv)}\n".encode())
        log.flush()
        process = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
    paths.pid.write_text(f"{process.pid}\n", encoding="utf-8")
    probe = ready if ready is not None else api_answers
    print(f"starting the station (pid {process.pid}) on {config_path}; opening the arms and cameras...", file=out)
    started_s = time.monotonic()
    deadline_s = started_s + START_TIMEOUT_S
    up = False
    try:
        while process.poll() is None and time.monotonic() < deadline_s:
            up = probe(bind_host, api_port)
            if up:
                break
            sleep(START_POLL_S)
    except KeyboardInterrupt:
        print(
            f"\nstopped watching; the station (pid {process.pid}) goes on starting by itself: 'dreamcli robot log'"
            " shows it, 'dreamcli robot down' stops it",
            file=out,
        )
        return 1
    elapsed_s = time.monotonic() - started_s
    if process.poll() is not None and process.returncode == 0:
        paths.pid.unlink(missing_ok=True)
        print(
            f"the station was stopped {elapsed_s:.0f} s after starting, before it answered (a 'dreamcli robot down');"
            f" nothing is running (log: {paths.log})",
            file=out,
        )
        return 1
    if process.poll() is not None:
        paths.pid.unlink(missing_ok=True)
        cause = startup_failure.find(_since_last_start(paths.log))
        if cause is not None:
            print(f"the station could not start: {cause}", file=out)
        print(
            f"the station exited with code {process.returncode} {elapsed_s:.0f} s after starting; the log ends:",
            file=out,
        )
        for line in tail(paths.log, START_FAIL_LINES):
            print(f"  {line}", file=out)
        print(
            "nothing is running; fix the cause above, then 'dreamcli robot up' again ('dreamcli robot log' shows the"
            f" full log, {paths.log})",
            file=out,
        )
        return 1
    if not up:
        print(
            f"the station (pid {process.pid}) has not answered on port {api_port} after {elapsed_s:.0f} s; it is still"
            " running and may still be opening the arms and cameras; the log ends:",
            file=out,
        )
        for line in tail(paths.log, START_FAIL_LINES):
            print(f"  {line}", file=out)
        print("watch it with 'dreamcli robot log'; 'dreamcli robot down' stops it", file=out)
        return 1
    print(f"station started (pid {process.pid}) on {config_path}", file=out)
    if not _config_has_live(config_path):
        print(
            "  live view: none; the config has no 'live' section ('dreamcli setup' registers the robot and", file=out
        )
        print("             writes it), so the Playground cannot reach this station", file=out)
        print(f"  log:       {paths.log}   (dreamcli robot log)", file=out)
        print(
            f"nothing moves until a start on its API (curl -X POST http://{bind_host}:{api_port}/start); keep a hand "
            "on the e-stop",
            file=out,
        )
        return 0
    from dream import credentials  # read only for this line of the report

    print(f"  live view: {credentials.load().platform_url}", file=out)
    print(f"  log:       {paths.log}   (dreamcli robot log)", file=out)
    is_sys0_on = sys0 if sys0 is not None else _config_body(config_path) == "yam"
    if is_sys0_on:
        print(
            "  System 0:  on; it drives the motors after the start-up ramp ('dreamcli robot up nosys0': without it)",
            file=out,
        )
    elif sys0 is False:
        print("  System 0:  off; the plain loop drives the motors ('dreamcli robot up' turns it back on)", file=out)
    if eef is None:
        print(
            "  head:      the config's action_space; eef on every body unless the file names joint "
            "('dreamcli robot up eef' or 'joint' picks one for this run)",
            file=out,
        )
    elif eef:
        print("  head:      eef; the model's tool poses, solved to joints with dream's IK", file=out)
    else:
        print("  head:      joint; the model's own joints", file=out)
    asked = list(extra_args)
    solve = asked[asked.index("--ik-solve") + 1] if "--ik-solve" in asked[:-1] else None
    print(
        f"  IK solve:  {solve} for this run"
        if solve
        else f"  IK solve:  the station file's ik_solve, else {_DEFAULT_IK_SOLVE} "
        "('dreamcli robot up --ik-solve tick|traj' picks one)",
        file=out,
    )
    print("nothing moves until a Start from the live view; keep a hand on the e-stop", file=out)
    return 0


def robot_run(
    paths: Paths,
    *,
    config_path: pathlib.Path | None = None,
    task: str | None = None,
    checkpoint: str | None = None,
    action_space: str | None = None,
    dry_run: bool = False,
    view_port: int | None = None,
    skip_tcp_calibration: bool = False,
    out: IO[str] = sys.stdout,
    run_command: Callable[[Sequence[str]], int] = lambda argv: subprocess.call(argv),
) -> int:
    """Runs one task on the config's arms in this terminal until Ctrl-C, then homes the arms and switches them off.

    The same loop a station runs (`robots.run.main`), without the station agent and without registering anything:
    the links are brought up as `robot up` does, the arms move home, the model runs until Ctrl-C (or a fault), and
    the arms move home again and are switched off as the station does when it exits. With `dry_run` no target reaches
    an arm (`robots.run.DryRun`).

    @param paths The station directory.
    @param config_path The config to run; the directory's #CONFIG_NAME when None.
    @param task The prompt; the config's `task` when None.
    @param checkpoint The model.
    @param action_space What the run executes (`robots.config.ACTION_SPACES`); the config's `action_space` when None.
    @param dry_run Run without writing a target to an arm.
    @param view_port Serve the live chunk view on 127.0.0.1 at this port.
    @param skip_tcp_calibration Run an arm file whose tool point was never marked (`robots.run.check_tool_point()`).
    @param out Where the report goes.
    @param run_command Runs one argv; injected for tests. Used for the `ip link` calls only.
    @return The exit code.
    """
    config_path = config_path if config_path is not None else paths.config
    if dry_run and view_port is not None:
        print(
            "--view reads the chunks a run saves, and a dry run saves none: drop one of --view and --dry-run", file=out
        )
        return 1
    if not config_path.is_file():
        print(f"no config at {config_path}: run 'dreamcli setup' or pass --config", file=out)
        return 1
    if _station_running(paths, out):
        return 1
    return _holding_arms(
        paths,
        arms_lock.ROBOT_RUN,
        out,
        lambda: _run_here(
            config_path, task, checkpoint, action_space, dry_run, view_port, skip_tcp_calibration, out, run_command
        ),
    )


def _run_here(
    config_path: pathlib.Path,
    task: str | None,
    checkpoint: str | None,
    action_space: str | None,
    dry_run: bool,
    view_port: int | None,
    skip_tcp_calibration: bool,
    out: IO[str],
    run_command: Callable[[Sequence[str]], int],
) -> int:
    import dream
    from dream import credentials

    from robots import run  # the loop and the dream client, only for this command

    try:
        account = credentials.load()
    except dream.DreamConfigError as exc:
        print(f"account: {exc}", file=out)
        return 1
    if not account.api_key:
        print(
            f"account: no api key: neither the environment nor {account.path} sets DREAM_API_KEY; the Setup API page "
            f"on {account.platform_url} gives the lines that write that file",
            file=out,
        )
        return 1
    if not bring_up_links(_can_channels(config_path), out=out, run_command=run_command):
        return 1
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        app = run.with_action_space(run.with_task(config.load(config_path), task), action_space)
        run.check_tool_point(app, skip=skip_tcp_calibration)
        if view_port is not None:
            from robots import run_view

            run_view.start(app, view_port)
            port = view_port
            print(
                f"chunk view on http://127.0.0.1:{port}; from a laptop: ssh -L {port}:localhost:{port} <this box>",
                file=out,
            )
        if dry_run:
            print(f"dry run on {config_path}; nothing is written to the arms", file=out)
        else:
            print(f"running on {config_path}; the arms move home; Ctrl-C homes them and switches them off", file=out)
        asyncio.run(run.main(app, checkpoint, dry_run=dry_run))
    except errors.ConfigError as exc:
        print(f"config: {exc}", file=out)
        return 1
    except dream.DreamAuthError as exc:
        print(f"account: {exc}", file=out)
        return 1
    except dream.DreamConfigError as exc:
        print(f"refused: {exc}", file=out)
        return 1
    except (errors.RobotsError, dream.DreamError) as exc:
        print(f"stopped: {exc}", file=out)
        return 1
    return 0


def _can_channels(config_path: pathlib.Path) -> list[str]:
    """The CAN interfaces the config's two arms name, read leniently: a file that will not parse yields none, and
    the agent then reports it as the config error it is.

    @param config_path The station config.
    @return The SocketCAN link names and CANable devices, in file order.
    """
    try:
        with open(config_path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError):
        return []
    if not isinstance(raw, dict) or raw.get("body") in (config.CUSTOM_BODY, "so101"):
        return []
    found = []
    for key in ("follower_l", "follower_r"):
        section = raw.get(key)
        channel = section.get("can_channel") if isinstance(section, dict) else None
        if isinstance(channel, str) and channel:
            found.append(channel)
    return found


def _config_has_live(config_path: pathlib.Path) -> bool:
    with open(config_path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return isinstance(raw, dict) and bool(raw.get("live"))


def _config_body(config_path: pathlib.Path) -> str | None:
    try:
        with open(config_path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except FileNotFoundError:
        return None
    except (OSError, yaml.YAMLError):
        return "yam"
    return str(raw.get("body", "yam")) if isinstance(raw, dict) else "yam"


def robot_down(
    paths: Paths,
    *,
    out: IO[str] = sys.stdout,
    sleep: Callable[[float], object] = time.sleep,
    stop_grace_s: float = STOP_GRACE_S,
    kill_grace_s: float = KILL_GRACE_S,
    park_notice_s: float = PARK_NOTICE_S,
    run_command: Callable[[Sequence[str]], int] = lambda argv: subprocess.call(argv),
    switch_off: Callable[[str], None] = motor_mode.switch_off,
) -> int:
    """Stops the agent: SIGINT (its clean stop), then SIGTERM after `stop_grace_s`, then SIGKILL after `kill_grace_s`.

    With no agent running, a YAM's motors are switched off directly over CAN (_switch_off_idle_arms()).

    @param paths The station directory.
    @param out Where the report goes.
    @param sleep Blocking sleep, seconds; injected so tests do not wait.
    @param stop_grace_s Seconds to wait after SIGINT.
    @param kill_grace_s Seconds to wait after SIGTERM.
    @param park_notice_s Seconds after SIGINT before the parking line.
    @param run_command Runs one argv; injected for tests. Used for the `ip link` calls only.
    @param switch_off Switches every motor on one link off; injected for tests.
    @return The exit code.
    """
    pid = running_pid(paths)
    if pid is None:
        return _switch_off_idle_arms(paths, out=out, run_command=run_command, switch_off=switch_off)
    for signum, grace_s, what in (
        (signal.SIGINT, stop_grace_s, "stopping (the arms switch off)"),
        (signal.SIGTERM, kill_grace_s, "did not stop; terminating"),
        (signal.SIGKILL, kill_grace_s, "did not terminate; killing"),
    ):
        print(f"pid {pid}: {what}", file=out)
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            break
        if signum == signal.SIGINT:
            notice_s = min(park_notice_s, grace_s)
            if _wait_gone(pid, notice_s, sleep):
                break
            print(f"pid {pid}: parking the arms before they switch off (up to {grace_s:.0f} s)", file=out)
            grace_s -= notice_s
        if _wait_gone(pid, grace_s, sleep):
            break
    else:
        print(f"pid {pid} is still alive after SIGKILL; check it by hand", file=out)
        return 1
    paths.pid.unlink(missing_ok=True)
    if _log_says_switched_off(paths):
        print("stopped; the arms are switched off", file=out)
        return 0
    print(
        "stopped, but the station did not log that it switched the arms off: they may still hold under power; "
        f"support them and switch their power off (the log: {paths.log})",
        file=out,
    )
    return 1


def _switch_off_idle_arms(
    paths: Paths,
    *,
    out: IO[str],
    run_command: Callable[[Sequence[str]], int],
    switch_off: Callable[[str], None],
) -> int:
    """With no agent running, switches a YAM's motors off over CAN directly, so the arms go limp.

    A station whose PC died or rebooted never switched the arms off, and a DM motor keeps its last command while the
    bus is quiet, so without this `robot down` would only say that the station was not running while the arms held
    under power. The links come back down after a reboot, so they are brought up first, as `robot up` does. Another
    body, or a station directory with no config, is left as it was.

    @param paths The station directory.
    @param out Where the report goes.
    @param run_command Runs one argv; used for the `ip link` calls only.
    @param switch_off Switches every motor on one link off.
    @return The exit code: 0 when every link's motors were sent the switch-off; 1 otherwise.
    """
    if _config_body(paths.config) != "yam":
        print("the station is not running", file=out)
        return 0
    links = _can_channels(paths.config) or list(motor_mode.DEFAULT_LINKS)
    print("the station is not running; switching the arms' motors off directly (they go limp: support them)", file=out)

    def sweep() -> int:
        if not bring_up_links(links, out=out, run_command=run_command):
            print("could not reach the arms: support them and switch their power off", file=out)
            return 1
        failed = []
        for link in links:
            try:
                switch_off(link)
            except Exception as exc:  # one link's failure must not stop the other's switch-off
                print(f"{link}: could not switch the motors off: {exc}", file=out)
                failed.append(link)
        if failed:
            print("support the arms and switch their power off", file=out)
            return 1
        print("the arms' motors are switched off", file=out)
        return 0

    return _holding_arms(paths, arms_lock.ROBOT_DOWN, out, sweep)


def _log_says_switched_off(paths: Paths) -> bool:
    lines = tail(paths.log, SHUTDOWN_LINES) if paths.log.is_file() else []
    headers = [index for index, line in enumerate(lines) if line.startswith(LOG_HEADER)]
    current = lines[headers[-1] :] if headers else lines
    return any(station.SWITCHED_OFF_LINE in line for line in current)


def motors(
    action: str,
    *,
    links: Sequence[str] | None = None,
    motor_ids: Sequence[int] | None = None,
    out: IO[str] = sys.stdout,
    run_command: Callable[[Sequence[str]], int] = lambda argv: subprocess.call(argv),
) -> int:
    """Checks, or switches to MIT, the firmware control mode of the arms' motors (`robots.motor_mode`).

    The links are brought up first (bring_up_links()), since a fresh station's come up down, and a link that is
    missing ends the command the way `robot up` ends. The station agent must not be running: it owns the bus.

    @param action `check` or `set-mit`.
    @param links The CAN links; a YAM's two (`robots.motor_mode.DEFAULT_LINKS`) when None.
    @param motor_ids The motor ids on each link; every motor, 1 to 7, when None.
    @param out Where the report goes.
    @param run_command Runs one argv; injected for tests. Used for the `ip link` calls only.
    @return The exit code: 0 every motor is (or now confirms) MIT; 1 otherwise, or when a link is missing.
    """
    chosen = tuple(links) if links else motor_mode.DEFAULT_LINKS
    wanted = tuple(motor_ids) if motor_ids else motor_mode.ALL_MOTORS
    if not bring_up_links(chosen, out=out, run_command=run_command):
        return 1
    if action == "check":
        return motor_mode.check(chosen, wanted, out=out)
    return motor_mode.switch(chosen, wanted, out=out)


def robot_temp(paths: Paths, warn: float | None, stop: float | None, *, out: IO[str] = sys.stdout) -> int:
    """Prints the arms' motor temperature limits, or sets either in the station config.

    The keys are `motor_temp_warn_c` and `motor_temp_stop_c`; the arms run on them the next time they open
    (`dreamcli robot up`).

    @param paths The station directory.
    @param warn The new warning temperature, degrees C: an arm above it will not open; None keeps it.
    @param stop The new stop temperature, degrees C: a run stops and the arm parks; None keeps it.
    @param out Where the report goes.
    @return The exit code: 0, or 1 when there is no config, the body has no limits, or a limit is refused.
    """
    if not paths.config.is_file():
        print(f"no station config at {paths.config}: run 'dreamcli setup' first", file=out)
        return 1
    text = paths.config.read_text(encoding="utf-8")
    tmp = paths.config.with_name(paths.config.name + ".tmp")
    for key, value in (("motor_temp_warn_c", warn), ("motor_temp_stop_c", stop)):
        if value is not None:
            line = f"{key}: {value:g}"
            text, found = re.subn(rf"^{key}:.*$", line, text, flags=re.MULTILINE)
            if not found:
                text = text.rstrip("\n") + f"\n{line}\n"
    try:
        tmp.write_text(text, encoding="utf-8")
        cfg = config.load(tmp)
        if cfg.body not in config.MOTOR_TEMP_BODIES:
            raise errors.ConfigError(f"dream does not watch the motor temperatures of 'body: {cfg.body}'")
        spec = motor_arm.with_temp_limits(importlib.import_module(f"robots.{cfg.body}").ARM, cfg.follower_l)
    except errors.ConfigError as exc:
        tmp.unlink(missing_ok=True)
        print(exc, file=out)
        return 1
    if warn is None and stop is None:
        tmp.unlink()
    else:
        os.replace(tmp, paths.config)
    print(
        f"warn at {spec.temp_warn_c:g} C (an arm above it will not open), stop at {spec.temp_stop_c:g} C (the run "
        "stops and the arm parks)",
        file=out,
    )
    if warn is not None or stop is not None:
        print("Saved. The arms run on these from the next 'dreamcli robot up'.", file=out)
    return 0


def robot_log(paths: Paths, *, lines: int = LOG_LINES, out: IO[str] = sys.stdout) -> int:
    """Prints the last `lines` lines of the agent's log.

    @param paths The station directory.
    @param lines How many lines.
    @param out Where they go.
    @return The exit code: 0, or 1 when there is no log yet.
    """
    if not paths.log.is_file():
        print(f"no log at {paths.log}: the station has not been started with 'dreamcli robot up'", file=out)
        return 1
    for line in tail(paths.log, lines):
        print(line, file=out)
    return 0


def _since_last_start(path: pathlib.Path) -> list[str]:
    """The log's lines after its last #LOG_HEADER, among its last #START_SCAN_LINES, oldest first."""
    lines = tail(path, START_SCAN_LINES)
    starts = [i for i, line in enumerate(lines) if line.startswith(LOG_HEADER)]
    return lines[starts[-1] + 1 :] if starts else lines


def tail(path: pathlib.Path, lines: int) -> list[str]:
    """The last `lines` lines of a text file, without their newlines.

    @param path The file.
    @param lines How many lines; 0 or less gives none.
    @return The lines, oldest first; fewer when the file is shorter.
    """
    if lines <= 0:
        return []
    with open(path, encoding="utf-8", errors="replace") as handle:
        return [line.rstrip("\n") for line in collections.deque(handle, maxlen=lines)]


## The IK solve an eef run uses by default (#robots.config.DEFAULT_IK_SOLVE), for code where a `config` argument
## shadows the module.
_DEFAULT_IK_SOLVE: str = config.DEFAULT_IK_SOLVE


def _distribution_source(name: str) -> str:
    """Where an installed distribution came from, from its `direct_url.json` (PEP 610): a git URL with the branch and
    commit, a local path, or "PyPI" when it records none.

    @param name The distribution's name.
    @return One line; "not installed" when it is not.
    """
    import importlib.metadata

    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"
    raw = dist.read_text("direct_url.json")
    if not raw:
        return f"{dist.version} (PyPI)"
    found = json.loads(raw)
    vcs = found.get("vcs_info") or {}
    if vcs:
        ref = vcs.get("requested_revision") or "default branch"
        return f"{dist.version} ({found['url']} @ {ref}, commit {vcs.get('commit_id', '?')[:12]})"
    return f"{dist.version} ({found['url']})"


def version_text() -> str:
    """What `dreamcli --version` prints.

    This package and the client, with where each was installed from, then what dream's IK in this install can do and
    the IK solve an `eef` run uses by default.

    @return The lines, joined.
    """
    from robots import __version__

    lines = [
        f"dreamcli {__version__}",
        f"  dreamlabs:        {_distribution_source('dreamlabs')}",
        f"  dreamlabs-client: {_distribution_source('dreamlabs-client')}",
    ]
    try:
        from dream_ik import specs

        serve = specs.ServeSpec()
        solves = "tick, traj" if hasattr(serve, "traj_jerk_weight") else "tick"
        extra = ", region box" if hasattr(specs, "Region") else ""
        lines.append(f"  dream IK:         solves {solves}{extra}")
    except ImportError:
        lines.append("  dream IK:         not installed (Python 3.11 or newer installs it)")
    lines.append(
        f"  IK solve:         {config.DEFAULT_IK_SOLVE} on every body, "
        "by default for an eef run "
        "(station file ik_solve, or dreamcli robot up --ik-solve, picks another)"
    )
    return "\n".join(lines)


class _VersionAction(argparse.Action):
    """`--version`: prints #version_text(), computed only when asked (it imports dream's IK specs), and exits."""

    def __init__(self, option_strings: Sequence[str], dest: str = argparse.SUPPRESS, **kwargs: object) -> None:
        super().__init__(option_strings, dest=dest, nargs=0, help="print the versions, their sources and the IK solve")

    def __call__(self, parser: argparse.ArgumentParser, *_: object) -> None:
        print(version_text())
        parser.exit()


def main(argv: Sequence[str] | None = None) -> None:
    """The `dreamcli` entry point; exits with the command's code, or 130 with one line after a Ctrl-C.

    @param argv Command-line arguments; `sys.argv[1:]` when None.
    """
    try:
        code = run(argv)
    except KeyboardInterrupt:
        print("\nstopped by Ctrl-C", file=sys.stderr)
        code = 130
    sys.exit(code)


def run(argv: Sequence[str] | None = None, *, out: IO[str] = sys.stdout) -> int:
    """Parses the command line and runs one command.

    @param argv Command-line arguments; `sys.argv[1:]` when None.
    @param out Where the commands' reports go.
    @return The command's exit code.
    """
    parser = _parser()
    args, passthrough = parser.parse_known_args(argv)
    if passthrough and not (args.command == "robot" and args.robot_command == "up"):
        parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
    if args.command == "example":
        from robots import examples

        if args.name is None:
            return examples.list_examples(out)
        if args.self_test:
            return examples.self_test(args.name, out=out)
        return examples.run(
            args.name,
            [*(["--show-side-panel"] if args.show_side_panel else []), *(["--verbose"] if args.verbose else [])],
            out=out,
        )
    station_paths = paths(args.home)
    if args.command == "setup":
        from robots import setup_wizard

        return _holding_arms(
            station_paths,
            arms_lock.SETUP,
            out,
            lambda: setup_wizard.main(
                station_paths.config,
                host=args.host,
                port=args.port,
                browser=not args.no_browser,
                agent_running=lambda: running_pid(station_paths) is not None,
            ),
        )
    if args.command == "tcp":
        from robots import setup_wizard

        return setup_wizard.tcp_main(
            station_paths.config, host=args.host, port=args.port, browser=not args.no_browser, out=out
        )
    if args.command == "motors":
        body = _config_body(station_paths.config)
        if body not in (None, "yam"):
            print(
                f"'dreamcli motors' switches a YAM's motors to the mode the station drives; {station_paths.config} "
                f"says body: {body}, whose motors need no switch",
                file=out,
            )
            return 1
        links = args.link or _can_channels(station_paths.config) or None
        if _station_running(station_paths, out):
            return 1
        return _holding_arms(
            station_paths,
            arms_lock.MOTORS,
            out,
            lambda: motors(args.motors_command, links=links, motor_ids=args.motor, out=out),
        )
    if args.robot_command == "up":
        # `sys0` and `nosys0` are words, not flags, so argparse leaves them among the agent's pass-through arguments.
        if SYS0_WORD in passthrough and NO_SYS0_WORD in passthrough:
            print(f"'robot up' takes {SYS0_WORD} or {NO_SYS0_WORD}, not both", file=out)
            return 1
        extra = [arg for arg in passthrough if arg not in (SYS0_WORD, NO_SYS0_WORD, NO_EEF_WORD, JOINT_WORD, EEF_WORD)]
        sys0 = None
        if SYS0_WORD in passthrough or NO_SYS0_WORD in passthrough:
            sys0 = SYS0_WORD in passthrough
        config = pathlib.Path(args.config).expanduser().resolve() if args.config else None
        # None leaves the config's own head: `eef` unless the file says `joint`.
        eef = True if EEF_WORD in passthrough else (False if {NO_EEF_WORD, JOINT_WORD} & set(passthrough) else None)
        if args.ik_solve is not None:
            extra = [*extra, "--ik-solve", args.ik_solve]
        return robot_up(station_paths, config=config, extra_args=extra, sys0=sys0, eef=eef, out=out)
    if args.robot_command == "run":
        return robot_run(
            station_paths,
            config_path=pathlib.Path(args.config).expanduser().resolve() if args.config else None,
            task=args.task,
            checkpoint=args.checkpoint,
            action_space=args.action_space,
            dry_run=args.dry_run,
            view_port=args.view_port if args.view else None,
            skip_tcp_calibration=args.skip_tcp_calibration,
            out=out,
        )
    if args.robot_command == "down":
        return robot_down(station_paths, out=out)
    if args.robot_command == "temp":
        return robot_temp(station_paths, args.warn, args.stop, out=out)
    return robot_log(station_paths, lines=args.lines, out=out)


def _station_running(paths: Paths, out: IO[str]) -> bool:
    pid = running_pid(paths)
    if pid is not None:
        print(arms_lock.holder_message(f"{pid} {arms_lock.STATION}"), file=out)
    return pid is not None


def _holding_arms(paths: Paths, holder: str, out: IO[str], command: Callable[[], int]) -> int:
    try:
        lock = arms_lock.acquire(paths.lock, holder)
    except errors.StationError as exc:
        print(exc, file=out)
        return 1
    with lock:
        return command()


def _parser() -> argparse.ArgumentParser:
    """Builds the argument parser: `setup`, `tcp`, `robot` with `up`, `run`, `down`, `log`, `example`, and `motors`.

    @return The parser.
    """
    parser = argparse.ArgumentParser(prog="dreamcli", description="Set up and run a dream robot station.")
    parser.add_argument("--version", "-V", action=_VersionAction)
    parser.add_argument(
        "--home",
        default=None,
        help=f"the station directory (config, log, pid); default ${HOME_ENV} or {DEFAULT_HOME}",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    setup = commands.add_parser("setup", help="open the setup wizard, which writes the station config")
    setup.add_argument("--host", default=None, help="bind address of the wizard page (default 0.0.0.0)")
    setup.add_argument("--port", type=int, default=None, help="port of the wizard page (default 8092)")
    setup.add_argument("--no-browser", action="store_true", help="print the address only; do not open a browser")

    tcp = commands.add_parser(
        "tcp", help="open the setup page's tool point tool alone: your arm's URDF in, the arm file custom.ik names out"
    )
    tcp.add_argument("--host", default=None, help="bind address of the page (default 0.0.0.0)")
    tcp.add_argument("--port", type=int, default=None, help="port of the page (default 8094)")
    tcp.add_argument("--no-browser", action="store_true", help="print the address only; do not open a browser")

    robot = commands.add_parser("robot", help="start, stop and watch the station agent")
    robot_commands = robot.add_subparsers(dest="robot_command", required=True)
    up = robot_commands.add_parser(
        "up",
        help="start the station agent on the station config; a YAM runs System 0 unless 'robot up nosys0', and the "
        "model's EEF head unless 'robot up joint'",
        usage="dreamcli robot up [nosys0 | sys0] [eef | joint] [--ik-solve {tick,traj}] [--config CONFIG] [flags]",
        description=(
            "Starts python -m robots.station detached on the station config. On a YAM, System 0, the learned motor "
            f"tracker, drives the motors after the start-up ramp; the word {NO_SYS0_WORD} runs the plain loop "
            f"instead, and {SYS0_WORD} asks for System 0 on any body (only a YAM has it). The word {EEF_WORD} runs "
            f"the model's tool poses through the dream client's IK, what every body runs when neither word is given, "
            f"and {JOINT_WORD} ({NO_EEF_WORD} is its old name) runs the model's own joints; without a word the "
            "config's action_space runs. Any other flag is passed through to the agent: --api-port (default "
            f"{station.DEFAULT_API_PORT}), --host, --inference-log."
        ),
    )
    up.add_argument("--config", default=None, help="run another config file instead of the station's")
    up.add_argument(
        "--ik-solve",
        choices=config.IK_SOLVES,
        default=None,
        help="how an eef run's tool poses become joints, for this run only: traj solves the whole chunk with the "
        "arm's last 3 commands, tick row by row; default: the station file's ik_solve, else "
        f"{config.DEFAULT_IK_SOLVE}",
    )
    run_cmd = robot_commands.add_parser(
        "run",
        help="run one task in this terminal, without the station agent (Ctrl-C stops it)",
        description=(
            "Moves the config's arms home, runs the loop until Ctrl-C, then homes the arms and switches them off. "
            "Needs no station agent and registers nothing; the station must not be running."
        ),
    )
    run_cmd.add_argument("--config", default=None, help="run another config file instead of the station's")
    run_cmd.add_argument("--task", default=None, help="the prompt; the config's task when left out")
    run_cmd.add_argument(
        "--checkpoint", default=None, help="the model; the config's checkpoint when left out (a YAM's default if none)"
    )
    run_cmd.add_argument(
        "--action-space",
        choices=config.ACTION_SPACES,
        default=None,
        help="what the arms execute: eef, the model's tool poses through the dream client's IK, or joint, the model's "
        "own joints; the config's action_space when left out, which is eef on every body unless the file names joint",
    )
    run_cmd.add_argument("--dry-run", action="store_true", help="run the policy but write no target to an arm")
    run_cmd.add_argument(
        "--view", action="store_true", help="serve a live 3D page of each chunk (hand paths, arm, keep-out shapes)"
    )
    run_cmd.add_argument("--view-port", type=int, default=8093, help="the --view page's port (default 8093)")
    # Not advised, so named only where it is refused (robots.run.check_tool_point()) and in the customer docs.
    run_cmd.add_argument(SKIP_TCP_FLAG, action="store_true", help=argparse.SUPPRESS)
    robot_commands.add_parser(
        "down",
        help="stop the station agent (SIGINT; the arms switch off and go limp); with none running, switch a YAM's "
        "motors off directly",
    )
    temp = robot_commands.add_parser(
        "temp", help="SO-101, Metal and Maker arms: print the motor temperature limits, or set them with --warn/--stop"
    )
    temp.add_argument("--warn", type=float, default=None, help="degrees C: an arm with a motor above it will not open")
    temp.add_argument("--stop", type=float, default=None, help="degrees C: the run stops and the arm parks")
    log = robot_commands.add_parser("log", help=f"print the last {LOG_LINES} lines of the agent's log")
    log.add_argument("-n", "--lines", type=int, default=LOG_LINES, help="how many lines")

    example = commands.add_parser(
        "example", help="run an example that ships with the library, no robot needed (yam-mujoco: Dream YAM Sim)"
    )
    example.add_argument("name", nargs="?", help="the example; leave it out for the list")
    example.add_argument("--show-side-panel", action="store_true", help="yam-mujoco: show MuJoCo's settings panels")
    example.add_argument("--verbose", action="store_true", help="yam-mujoco: print every inference and adjustment")
    example.add_argument(
        "--self-test", action="store_true", help="run the example's headless check, then exit (no display, no key)"
    )

    motors_cmd = commands.add_parser("motors", help="YAM only: check, or switch to MIT, the arms' motor control mode")
    motors_commands = motors_cmd.add_subparsers(dest="motors_command", required=True)
    for name, text in (
        ("check", "read every motor's firmware control mode; exit 0 only when all are MIT"),
        ("set-mit", "switch every motor that is not MIT to MIT and store it; then power-cycle the arms"),
    ):
        sub = motors_commands.add_parser(name, help=text)
        sub.add_argument(
            "--link",
            action="append",
            default=None,
            help="a CAN link (default: the two the station config names, else "
            f"{' and '.join(motor_mode.DEFAULT_LINKS)})",
        )
        sub.add_argument(
            "--motor", type=int, action="append", default=None, help="a motor id, 1 to 7 (default: all of them)"
        )
    return parser


def api_answers(host: str, port: int) -> bool:
    """Whether the station agent's HTTP API answers `GET /status` on `host:port`.

    @param host The bind host the agent was started with.
    @param port The API port.
    @return True on the agent's status within #START_PROBE_TIMEOUT_S.
    """
    target = "127.0.0.1" if host in ("0.0.0.0", "") else host
    try:
        with urllib.request.urlopen(f"http://{target}:{port}/status", timeout=START_PROBE_TIMEOUT_S) as reply:
            body = json.loads(reply.read())
    except (urllib.error.URLError, OSError, ValueError):
        return False
    return isinstance(body, dict) and "state" in body


def api_port_free(host: str, port: int) -> bool:
    """Whether the agent could bind its API on `host:port` now: False only when another socket holds it.

    @param host The bind host.
    @param port The API port.
    @return False when the address is in use.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError as exc:
            return exc.errno != errno.EADDRINUSE
    return True


def _serve_flags(extra_args: Sequence[str]) -> tuple[str, int]:
    """Reads the bind host and the API port out of the pass-through flags, for the readiness probe.

    @param extra_args The flags handed to `robots.station`.
    @return `(host, api_port)`, the station's defaults where the flags say nothing.
    """
    host, api_port = station.DEFAULT_HOST, station.DEFAULT_API_PORT
    args = list(extra_args)
    for i, arg in enumerate(args):
        name, _, value = arg.partition("=")
        if not value and i + 1 < len(args):
            value = args[i + 1]
        if name == "--host" and value:
            host = value
        elif name == "--api-port" and value.isdigit():
            api_port = int(value)
    return host, api_port


def _alive(pid: int) -> bool:
    """True when a process with `pid` exists (signal 0 is delivered or refused, never lost).

    A child of this process that has exited is reaped first, so a zombie is not reported alive: `robot_up` and
    `robot_down` in one process (tests) see the same answer as two shells do.

    @param pid The pid.
    @return Whether it exists.
    """
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)
        if reaped == pid:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_gone(pid: int, grace_s: float, sleep: Callable[[float], object]) -> bool:
    """Polls until the process is gone or `grace_s` has passed.

    @param pid The pid.
    @param grace_s Seconds to wait at most.
    @param sleep Blocking sleep, seconds.
    @return True when the process is gone.
    """
    deadline = time.monotonic() + grace_s
    while _alive(pid):
        if time.monotonic() >= deadline:
            return False
        sleep(0.1)
    return True


if __name__ == "__main__":
    main()
