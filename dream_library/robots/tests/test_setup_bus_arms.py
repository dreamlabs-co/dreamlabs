"""The setup wizard on a Metal arm and an SO-101 through their own drivers, the way `dreamcli robot run` opens them.

The arms are `robots.metal` on the Damiao fake (tests/metal_fakes.py) and `robots.so101` on the Feetech fake
(tests/so101_fakes.py), each under the shared bus-arm thread at its real rate. A hand moves the fake motors while the
arms are limp (`robots.motor_arm.MotorArm.disengage`: every motor off, the pose read on), and the wizard runs every
step over its routes to Save: the robot, the arms by moving them, the cameras, home, the powered checks with the page's
drawing following the arms, and the eyes answer.
The file it writes loads, and a dry run (`robots.run.main`, `dreamcli robot run --dry-run`) starts on it and runs the
policy on both arms.
"""

import asyncio
import dataclasses
import json
import math
import pathlib
import time
from collections.abc import Callable
from typing import Any

import conftest
import metal_fakes
import numpy as np
import pytest
import setup_fakes
import so101_fakes
import test_setup_wizard
import test_so101
from test_setup_wizard import FakePlatform, request

from robots import bodies, cameras, config, feetech, robstride, run, setup_cameras, setup_model, setup_wizard, so101

pytest.importorskip("dream_ik", reason="dream's IK needs Python 3.11 or newer")

## The fake platform Save registers on (test_setup_wizard.platform).
platform = test_setup_wizard.platform

## Longest one page request may take, seconds: a first IK solve compiles, over 5 s on a CI runner.
SOLVE_WAIT_S = 30.0

## The account's embodiment an SO-101 fine-tune was trained on.
SO101_EMBODIMENT = "tester/so101-pick"

EMBODIMENTS = [setup_fakes.account_embodiment(SO101_EMBODIMENT, test_so101.SO101_EMBODIMENT)]

## The model a start names on each body: the file the wizard saves names none.
MODELS = {"so101": "dream-1-ft:tester-so101", "metal": "dream-1-ft:tester-metal"}

## Each arm's adapter: `(device id, what the driver opens)`.
ADAPTERS = {
    "metal": {
        "left": ("slcan-AAA111", "/dev/serial/by-id/usb-Protofusion_Labs_CANable_AAA111-if00"),
        "right": ("slcan-BBB222", "/dev/serial/by-id/usb-Protofusion_Labs_CANable_BBB222-if00"),
    },
    "so101": {
        "left": ("serial-left", "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A0185821-if00"),
        "right": ("serial-right", "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A0185822-if00"),
    },
}

METAL_GRIPPER_STOPS_DEG = (0.0, 114.0)

METAL_HOME_RAD = (0.0, 0.0, 0.06, -0.21, 0.0, 0.0)

## Seconds between two readings of a hand move: about the reader thread's period.
STEP_S = 0.05


class Hardware:
    """The fake motors of both arms, posed in URDF units, and the operator's hands on them.

    Attributes:
    - body: `metal` or `so101`.
    - q: Per driver channel, the true joints, `[6]` radians.
    - grip: Per driver channel, the true gripper opening, 0 closed to 1 open.
    """

    def __init__(self, body: str, monkeypatch: pytest.MonkeyPatch, model: setup_model.ArmModel) -> None:
        self.body = body
        self.session: Any = None
        self.q = {channel: model.rests[side].copy() for side, (_, channel) in ADAPTERS[body].items()}
        self.grip = dict.fromkeys(self.q, 0.5)
        if body == "metal":
            self.can = metal_fakes.FakeCanFactory()
            monkeypatch.setattr(robstride, "open_can_bus", self.can)
        else:
            self.serial = so101_fakes.FakeFeetechFactory()

            def open_bus(port: str, joints: Any, calibration: Any) -> so101_fakes.FakeFeetechBus:
                bus = self.serial(port, joints, calibration)
                bus.present.update(self._ticks(port))
                return bus

            monkeypatch.setattr(feetech, "open_bus", open_bus)
        for channel in self.q:
            self.push(channel)

    def _ticks(self, port: str) -> dict[str, int]:
        """An SO-101's servo positions for its true pose: the URDF's zero is the middle tick."""
        names = [joint.name for joint in so101.JOINTS]
        ticks = {name: feetech.deg_to_ticks(math.degrees(self.q[port][i])) for i, name in enumerate(names[:-1])}
        low, high = so101_fakes.CALIBRATION["gripper"][2:]
        ticks["gripper"] = round(low + self.grip[port] * (high - low))
        return ticks

    def push(self, channel: str) -> None:
        """Puts the true pose on the motors."""
        if self.body == "metal":
            motors = self.can.bus(channel).motors
            for i in range(6):
                motors[i + 1].position_deg = math.degrees(self.q[channel][i])
            closed, opened = METAL_GRIPPER_STOPS_DEG
            motors[7].position_deg = closed + self.grip[channel] * (opened - closed)
        elif channel in self.serial.buses:
            self.serial.bus(channel).present.update(self._ticks(channel))

    def opened(self) -> dict[str, int]:
        """How many times each adapter was opened."""
        if self.body == "metal":
            return dict(self.can.opens)
        return {port: len(buses) for port, buses in self.serial.buses.items()}

    def all_powered(self) -> bool:
        """Whether every motor of both arms is powered."""
        if self.body == "metal":
            return all(m.enabled for channel in self.q for m in self.can.bus(channel).motors.values())
        return all(self.serial.bus(port).torque_log[-1] == "on" for port in self.q)

    def _step(self, channel: str) -> None:
        self.push(channel)
        time.sleep(STEP_S)
        self.session.tick()

    def move(self, channel: str, joint: int, angle: float, steps: int = 6, back: bool = True) -> None:
        """Turns one joint by `angle` radians, then back (setup_fakes.Hands.move)."""
        start = self.q[channel][joint]
        ramp = [*range(1, steps + 1), *(range(steps - 1, -1, -1) if back else [])]
        for i in ramp:
            self.q[channel][joint] = start + angle * i / steps
            self._step(channel)

    def pose(self, channel: str, target: Any, steps: int = 6) -> None:
        """Moves the whole arm to a pose in a straight line."""
        start, end = self.q[channel].copy(), np.asarray(target, dtype=np.float64)
        for i in range(1, steps + 1):
            self.q[channel] = start + (end - start) * i / steps
            self._step(channel)

    def grip_to(self, channel: str, opening: float, steps: int = 4) -> None:
        """Opens or closes the gripper by hand, 0 at its closed stop to 1 at its open stop."""
        start = self.grip[channel]
        for i in range(1, steps + 1):
            self.grip[channel] = start + (opening - start) * i / steps
            self._step(channel)


def serve(
    body: str, target: pathlib.Path, platform: FakePlatform, hardware: Hardware, model: setup_model.ArmModel
) -> tuple[setup_wizard.SetupServer, Callable[..., Any]]:
    """The wizard over the fake arms and three fake cameras, started on an ephemeral port."""
    pictures = {
        f"/dev/video{i}": setup_fakes.sample_rgb(kind) for i, kind in enumerate(("head", "wrist_left", "wrist_right"))
    }
    server = setup_wizard.SetupServer(
        target,
        platform.account,
        host="127.0.0.1",
        port=0,
        model_factory=lambda _arm: model,
        open_camera=setup_fakes.FakeCameras(pictures),
        scan=lambda: scan(body),
        preview=lambda **_: setup_cameras.sample_jpeg("head", 0),
        ask_models=lambda: setup_fakes.MODELS,
        ask_embodiments=lambda: EMBODIMENTS,
        ask_training_data=setup_fakes.no_training_data,
    )
    hardware.session = server.session
    server.start()
    base = f"http://127.0.0.1:{server.port}"

    def call(method: str, path: str, payload: Any = None, status: int = 200) -> Any:
        code, kind, data = request(base, method, path, payload, timeout=SOLVE_WAIT_S)
        reply = json.loads(data) if kind == "application/json" else data
        assert code == status, (path, reply)
        return reply

    return server, call


def scan(body: str) -> dict[str, Any]:
    """What the box sees: the two adapters and three cameras."""
    kind = "slcan" if body == "metal" else "serial"
    arms = [
        {
            "id": ident,
            "kind": kind,
            "bus": "can" if body == "metal" else "serial",
            "open": channel,
            "serial": ident[-6:],
            "usb": "1-2:1.0",
            "label": ident,
            "detail": channel,
            "state": None,
            "via": None,
        }
        for ident, channel in ADAPTERS[body].values()
    ]
    shown = [{"source": "device", "value": f"/dev/video{i}", "label": str(i), "bound": "serial"} for i in range(3)]
    return {"arms": arms, "cameras": shown, "can": []}


@pytest.mark.parametrize("body", ["metal", "so101"])
def test_the_wizard_runs_a_bus_arm_limp_by_hand_to_save_and_a_dry_run_starts_on_the_file(
    body: str, tmp_path: pathlib.Path, platform: FakePlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every step over the routes on the real drivers: the arms go limp (motors off, read on) and follow a hand, the
    3D view draws the maker's meshes on dream's per-side URDF at home and follows the arms under power, naming each
    nudged joint, the Metal's grippers are opened and closed by hand and their range recorded, the powered checks
    pass, the eyes say yes, Save writes a file `robots.config.load` reads with the gripper ends measured and no model,
    and a dry run on it that names the model runs the policy on both arms."""
    if body == "so101" and "so101_faas" not in setup_model.spec_names():
        pytest.skip("this dream IK has no so101_faas")
    monkeypatch.setattr(bodies, "embodiment_config", lambda name, key: test_so101.SO101_EMBODIMENT)
    bodies.so101_body.cache_clear()
    model = setup_model.ArmModel(so101.ARM.ik if body == "so101" else "metal")
    hardware = Hardware(body, monkeypatch, model)
    target = tmp_path / "station.yaml"
    server, call = serve(body, target, platform, hardware, model)
    try:
        pick: dict[str, Any] = {"body": body}
        if body == "so101":
            calibration = so101_fakes.write_calibration(tmp_path / "arm.json")
            pick.update(calibration=[str(calibration)], embodiment=SO101_EMBODIMENT)
        picked = call("POST", "/api/robot", pick)
        assert picked["robot"]["note"] == setup_wizard.setup_session.LIMP_NOTE
        assert picked["joints"] == list(range(5 if body == "so101" else 6)), "the page shows only the arm's own joints"
        ids = [ident for ident, _ in ADAPTERS[body].values()]
        opened = call("POST", "/api/arms/open", {"devices": ids, "settings": picked["prefill"]["settings"]})
        assert opened["opened"] == dict.fromkeys(ids, None)
        if body == "metal":
            enabled = [m.enabled for _, c in ADAPTERS[body].values() for m in hardware.can.bus(c).motors.values()]
            assert not any(enabled), "a by-hand step holds a Metal motor energized"
        else:
            assert all(hardware.serial.bus(c).torque_log == ["off", "on", "off"] for _, c in ADAPTERS[body].values())
        for side, (ident, channel) in ADAPTERS[body].items():
            hardware.move(channel, 0, 0.3)
            assert call("GET", "/api/status")["wiggle"]["moved"] == ident
            assert call("POST", "/api/arms/assign", {"device": ident, "side": side})["crossed"] is None
        view = call("GET", "/api/view")
        links = len(view["links"])
        assert view["kind"] == "meshes" and view["parts"] and view["jaws"]
        assert all(isinstance(call("GET", part["url"]), bytes) for part in view["parts"] + view["jaws"])
        now = call("GET", "/api/status")["now"]
        assert all(len(now[side]["frame"]) == links for side in ("left", "right")), now
        roles = picked["robot"]["roles"]
        rows = [
            {
                "name": role,
                "source": "device",
                "value": f"/dev/video{i}",
                "width": 640,
                "height": 480,
                "fps": 30,
                "rotate": 0,
            }
            for i, role in enumerate(roles)
        ]
        call("POST", "/api/cameras/open", {"cameras": rows})
        model = server.session.model
        assert model is not None
        for side, (_, channel) in ADAPTERS[body].items():
            hardware.pose(channel, METAL_HOME_RAD if body == "metal" else model.rests[side])
        home = call("POST", "/api/home", {})
        assert home["out_of_range"] == [], home
        if body == "metal":
            for end, opening in (("open", 1.0), ("closed", 0.0)):
                for _, channel in ADAPTERS[body].values():
                    hardware.grip_to(channel, opening)
                gripper = call("POST", "/api/gripper", {"end": end})
            closed, opened = METAL_GRIPPER_STOPS_DEG
            kept = {"gripper_closed_deg": closed, "gripper_open_deg": opened}
            assert gripper["ends"] == {"left": kept, "right": kept}, gripper
        assert picked["reach"] is True
        call("POST", "/api/reach/paint", {"on": True})
        for _, channel in ADAPTERS[body].values():
            hardware.move(channel, 0, 0.6)
        reach = call("GET", "/api/status")["reach"]
        assert reach["painting"] and set(reach["hands"]) == {"left", "right"}, reach
        assert all(len(reach["points"][side]) > 3 for side in ("left", "right")), reach
        call("POST", "/api/reach/paint", {"on": False})
        call("POST", "/api/motion/start", {"confirm": True})
        drawn, nudged = set(), set()
        deadline_s = time.monotonic() + 120.0
        while not server.session.wait_motion(0.25) and time.monotonic() < deadline_s:
            status = call("GET", "/api/status")
            drawn |= {json.dumps(frame["frame"]) for frame in status["now"].values()}
            nudged.add(json.dumps(status["motion"]["nudging"]))
        assert server.session.wait_motion(0.0) and len(drawn) > 4, "the drawing did not follow the arms under power"
        assert {json.dumps({"side": "left", "joint": 0}), json.dumps({"side": "right", "joint": 0})} <= nudged, nudged
        motion = call("GET", "/api/status")["motion"]
        failed = [r for r in motion["results"] if r["finding"]["state"] != "pass"]
        assert motion["state"] == "done" and failed == [], (motion.get("error"), failed)
        opened_before = hardware.opened()
        call("POST", "/api/motion/start", {"confirm": True})
        assert server.session.wait_motion(120.0)
        motion = call("GET", "/api/status")["motion"]
        assert motion["state"] == "done", motion.get("error")
        assert hardware.opened() == opened_before and hardware.all_powered()
        assert call("POST", "/api/motion/eyes", {"same": True})["state"] == "done"
        saved = call("POST", "/api/save", {"cameras": rows})
        assert saved["robot"]["embodiment"] == ("emb_tester_so101_pick" if body == "so101" else "makermods_metal")
    finally:
        server.stop()

    cfg = dataclasses.replace(config.load(target), task=conftest.TASK)
    assert cfg.body == body and cfg.checkpoint is None
    for side in ("left", "right"):
        assert [list(point) for point in getattr(cfg, f"follower_{side[0]}").reach] == reach["points"][side]
    assert set(run.policy_config(cfg)["ik"]["reach"]) == {"left", "right"}
    if body == "metal":
        ends = (cfg.follower_l.gripper_closed_deg, cfg.follower_l.gripper_open_deg)
        assert ends == METAL_GRIPPER_STOPS_DEG
    else:
        assert cfg.embodiment == SO101_EMBODIMENT and cfg.follower_r.calibration == str(calibration)
    client = (test_so101.So101Client if body == "so101" else conftest.FakeClient)(horizon=cfg.action_horizon)
    readers = [
        cameras.CameraReader(c.name, conftest.FakeCaptureSource(), cfg.stale_after_s, face_blur=False)
        for c in cfg.cameras
    ]
    monkeypatch.setattr(run, "make_cameras", lambda app: readers)
    monkeypatch.setattr(run, "open_client", lambda dream_config: client)

    async def dry_run() -> None:
        stop = asyncio.Event()
        main = asyncio.create_task(run.main(cfg, MODELS[body], dry_run=True, stop=stop, home_duration_s=0.0))
        deadline_s = time.monotonic() + 30.0
        while client.calls < 2 and not main.done() and time.monotonic() < deadline_s:
            await asyncio.sleep(0.01)
        stop.set()
        await main

    if body == "metal":
        held = [m.enabled for _, c in ADAPTERS[body].values() for m in hardware.can.bus(c).motors.values()]
        assert not any(held), "the wizard left a Metal motor energized"
    asyncio.run(dry_run())
    assert client.calls >= 2


def test_the_reach_area_never_blocks_skip_or_save_even_when_the_hand_cannot_be_computed(
    tmp_path: pathlib.Path, platform: FakePlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hand's forward kinematics failing mid-paint leaves the poll answering with the reason; Skip then Save
    writes no `reach`; painting again un-skips, and Save writes what was painted, without the home hand it cannot
    compute. Bad requests are 400 and change nothing."""
    model = setup_model.ArmModel("metal")
    hardware = Hardware("metal", monkeypatch, model)
    target = tmp_path / "station.yaml"
    server, call = serve("metal", target, platform, hardware, model)
    left = ADAPTERS["metal"]["left"][1]
    try:
        rows = walk_to_the_checks(hardware, call)
        assert call("POST", "/api/reach/paint", {"on": "yes"}, status=400)["error"] == "'on' must be true or false"
        assert "'side' must be left or right" in call("POST", "/api/reach/clear", {"side": "top"}, status=400)["error"]
        call("POST", "/api/reach/paint", {"on": True})
        hardware.move(left, 0, 0.6)
        painted = call("GET", "/api/status")["reach"]["points"]

        def broken(*args: Any) -> Any:
            raise RuntimeError("no tool point")

        monkeypatch.setattr(model, "tool", broken)
        hardware.move(left, 0, 0.3)
        reach = call("GET", "/api/status")["reach"]
        assert reach["error"].endswith("hand's position failed: RuntimeError: no tool point"), reach
        assert reach["points"] == painted and reach["hands"] == {}, reach
        call("POST", "/api/reach/skip", {})
        skipped = call("GET", "/api/status")["reach"]
        call("POST", "/api/save", {"cameras": rows})
        assert skipped["skipped"] and not skipped["painting"], skipped
        assert config.load(target).follower_l.reach is None and config.load(target).follower_r.reach is None
        call("POST", "/api/reach/paint", {"on": True})
        call("POST", "/api/save", {"cameras": rows})
    finally:
        server.stop()
    loaded = config.load(target)
    assert [list(point) for point in loaded.follower_l.reach] == painted["left"]
    assert [list(point) for point in loaded.follower_r.reach] == painted["right"]


def walk_to_the_checks(hardware: Hardware, call: Callable[..., Any]) -> list[dict[str, Any]]:
    """A Metal station through steps 0 to 3 over the routes: picked, both arms found, cameras, home, gripper ends.

    @return The camera rows, as Save takes them.
    """
    body = "metal"
    picked = call("POST", "/api/robot", {"body": body})
    ids = [ident for ident, _ in ADAPTERS[body].values()]
    call("POST", "/api/arms/open", {"devices": ids, "settings": picked["prefill"]["settings"]})
    for side, (ident, channel) in ADAPTERS[body].items():
        hardware.move(channel, 0, 0.3)
        call("POST", "/api/arms/assign", {"device": ident, "side": side})
    rows = [
        {
            "name": role,
            "source": "device",
            "value": f"/dev/video{i}",
            "width": 640,
            "height": 480,
            "fps": 30,
            "rotate": 0,
        }
        for i, role in enumerate(picked["robot"]["roles"])
    ]
    call("POST", "/api/cameras/open", {"cameras": rows})
    for _, channel in ADAPTERS[body].values():
        hardware.pose(channel, METAL_HOME_RAD)
    call("POST", "/api/home", {})
    for end, opening in (("open", 1.0), ("closed", 0.0)):
        for _, channel in ADAPTERS[body].values():
            hardware.grip_to(channel, opening)
        call("POST", "/api/gripper", {"end": end})
    return rows


def start_blocked_on_the_way_home(
    hardware: Hardware, call: Callable[..., Any], monkeypatch: pytest.MonkeyPatch, on_link: bool = False
) -> None:
    """A Metal station set up to step 4."""
    walk_to_the_checks(hardware, call)
    right = ADAPTERS["metal"]["right"][1]
    if on_link:
        hardware.move(right, 1, -0.3, back=False)
    hardware.move(right, 2, 0.5, back=False)
    motors = hardware.can.bus(right).motors
    shoulder, elbow = motors[2], motors[3]
    shoulder_deg, stop_deg = shoulder.position_deg, elbow.position_deg - 2.0
    elbow.obstacle_deg = stop_deg
    if on_link:
        bus = type(hardware.can.bus(right))
        send = bus.send

        def coupled(self: Any, message: Any, timeout: float | None = None) -> None:
            if self.channel == right:
                elbow.obstacle_deg = stop_deg + abs(shoulder.position_deg - shoulder_deg)
            send(self, message, timeout)

        monkeypatch.setattr(bus, "send", coupled)
    call("POST", "/api/motion/start", {"confirm": True})


def enabled(hardware: Hardware) -> dict[str, list[bool]]:
    """Per side, whether each motor of the Metal arm is switched on."""
    return {
        side: [m.enabled for m in hardware.can.bus(c).motors.values()] for side, (_, c) in ADAPTERS["metal"].items()
    }


@pytest.mark.parametrize("on_link", [False, True], ids=["on-the-joint", "on-the-link"])
def test_a_metal_joint_blocked_on_the_way_home_parks_that_arm_and_the_other_holds(
    tmp_path: pathlib.Path, platform: FakePlatform, monkeypatch: pytest.MonkeyPatch, on_link: bool
) -> None:
    model = setup_model.ArmModel("metal")
    hardware = Hardware("metal", monkeypatch, model)
    server, call = serve("metal", tmp_path / "station.yaml", platform, hardware, model)
    try:
        start_blocked_on_the_way_home(hardware, call, monkeypatch, on_link)
        assert server.session.wait_motion(60.0)
        status = call("GET", "/api/status")
        motors = enabled(hardware)
    finally:
        server.stop()
    motion, last = status["motion"], status["motion"]["results"][-1]
    assert motion["state"] == "failed" and not motion["let_go"], motion["error"]
    assert (last["group"], last["side"], last["finding"]["cause"]) == ("watchdog", "right", "blocked"), last
    assert list(status["parked"]) == ["right"] and status["powered"] is True
    assert not any(motors["right"]) and all(motors["left"]), motors


def test_a_metal_arm_that_stops_answering_in_the_checks_still_holds_and_is_said_on_the_page_and_at_the_exit(
    tmp_path: pathlib.Path, platform: FakePlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = setup_model.ArmModel("metal")
    hardware = Hardware("metal", monkeypatch, model)
    server, call = serve("metal", tmp_path / "station.yaml", platform, hardware, model)
    words = "the right arm stopped answering; it still holds under power so it does not drop: support it, then switch "
    try:
        walk_to_the_checks(hardware, call)
        right = ADAPTERS["metal"]["right"][1]
        hardware.move(right, 2, 0.5, back=False)
        call("POST", "/api/motion/start", {"confirm": True})
        deadline_s = time.monotonic() + 60.0
        while call("GET", "/api/status")["motion"]["phase"] != "moving to home" and time.monotonic() < deadline_s:
            time.sleep(0.02)
        hardware.can.bus(right).motors[3].silent = True
        assert server.session.wait_motion(60.0)
        status = call("GET", "/api/status")
        refused = call("POST", "/api/motion/start", {"confirm": True}, status=400)
    finally:
        server.stop()
    assert status["still_holding"] == {"right": f"{words}its power off"} and status["powered"] is True, status
    assert status["not_answering"] == {} and status["holding"] == [], status
    assert refused["error"].startswith(words), refused
    assert server.session.still_holding == {"right": f"{words}its power off"}, "the exit names it (setup_wizard.main)"
    assert all(enabled(hardware)["right"][:6]), "a dead brakeless arm was switched off, so it dropped"


def test_a_metal_arm_parking_keeps_the_run_on_refuses_let_go_and_start_and_stop_lets_every_arm_go(
    tmp_path: pathlib.Path, platform: FakePlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = setup_model.ArmModel("metal")
    hardware = Hardware("metal", monkeypatch, model)
    server, call = serve("metal", tmp_path / "station.yaml", platform, hardware, model)
    wait = "wait until the right arm reaches rest and switches off"
    try:
        start_blocked_on_the_way_home(hardware, call, monkeypatch)
        deadline_s = time.monotonic() + 60.0
        while (motion := call("GET", "/api/status")["motion"])["state"] == "running" and time.monotonic() < deadline_s:
            time.sleep(0.02)
        assert motion["state"] == "parking", motion
        assert motion["phase"] == "the right arm is moving to rest under power, then switching off"
        assert wait in call("POST", "/api/motion/release", {}, status=400)["error"]
        assert wait in call("POST", "/api/motion/start", {"confirm": True}, status=400)["error"]
        assert all(enabled(hardware)["right"]), "the park was over before STOP: nothing tested"
        stopped_s = time.monotonic()
        call("POST", "/api/motion/stop", {})
        assert server.session.wait_motion(10.0)
        took_s = time.monotonic() - stopped_s
        status = call("GET", "/api/status")
        motors = enabled(hardware)
    finally:
        server.stop()
    assert not any(motors["left"]) and not any(motors["right"]), motors
    assert took_s < 2.0, f"STOP let the park run on for {took_s:.1f} s"
    assert status["motion"]["let_go"] and status["powered"] is False and status["not_answering"] == {}, status
    assert status["motion"]["stop_cut"] and list(status["parked"]) == ["right"], status["parked"]
    assert "went limp where it stood, not parked at rest" in status["parked"]["right"], status["parked"]


def test_a_metal_motor_still_warm_is_refused_at_start_and_at_the_find_and_found_once_cooled(
    tmp_path: pathlib.Path, platform: FakePlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = setup_model.ArmModel("metal")
    hardware = Hardware("metal", monkeypatch, model)
    server, call = serve("metal", tmp_path / "station.yaml", platform, hardware, model)
    (left_id, left_channel), (right_id, right_channel) = ADAPTERS["metal"]["left"], ADAPTERS["metal"]["right"]
    elbow = hardware.can.bus(right_channel).motors[3]

    def powered() -> list[bool]:
        return [
            m.enabled for channel in (left_channel, right_channel) for m in hardware.can.bus(channel).motors.values()
        ]

    try:
        picked = call("POST", "/api/robot", {"body": "metal"})
        settings = picked["prefill"]["settings"]
        assert call("POST", "/api/arms/open", {"devices": [left_id, right_id], "settings": settings})["opened"] == {
            left_id: None,
            right_id: None,
        }
        for side, (ident, channel) in ADAPTERS["metal"].items():
            hardware.move(channel, 0, 0.3)
            call("POST", "/api/arms/assign", {"device": ident, "side": side})
            hardware.pose(channel, METAL_HOME_RAD)
        call("POST", "/api/home", {})
        for end, opening in (("open", 1.0), ("closed", 0.0)):
            for channel in (left_channel, right_channel):
                hardware.grip_to(channel, opening)
            call("POST", "/api/gripper", {"end": end})

        elbow.temp_rotor_c = 60
        refused = call("POST", "/api/motion/start", {"confirm": True}, status=400)
        warm = "elbow flex motor is at 60 C at its coil: let it cool below 55 C, then try again."
        assert refused["error"] == f"The right arm's {warm}", refused
        assert "ArmError" not in refused["error"], refused
        assert not any(powered()) and not server.session.powered

        found = call("POST", "/api/arms/open", {"devices": [left_id, right_id], "settings": settings})["opened"]
        assert found[left_id] is None
        assert warm in found[right_id] and "Reset" not in found[right_id], found
        assert not any(powered())

        elbow.temp_rotor_c = 33
        found = call("POST", "/api/arms/open", {"devices": [left_id, right_id], "settings": settings})["opened"]
        assert found == {left_id: None, right_id: None}
    finally:
        server.stop()
