"""Behaviour tests for `robots.setup_wizard`: the file it writes, the platform registration, the routes over a real
socket, and `dreamcli setup` itself.

What is pinned: a saved file loads through `robots.config.load` and passes `robots.run.policy_config`, with the body,
each arm's adapter (a CAN link by the name the udev rule gives it, a serial adapter by its by-id path), its settings
and its home pose, the cameras, and the loop's default values, and nothing of the account or the model; a re-save keeps
the robot the file already named; each body registers as its own embodiment; a file the agent would refuse is reported
as the `ConfigError` and leaves the target untouched; an existing file prefills its body's settings and cameras. Over
the routes, a clean fake station goes the whole way: robot, arms found by moving them, the udev rule by serial, cameras
with the swap check, home against the reference's start, the powered checks after the safety confirm with the arms
drawn from their readings and the operator's eyes answer, and Save; with the operator's dataset, home is checked
against its episode's first state. `dreamcli setup` stops before serving when the credentials are malformed or name no
key, and says to keep its terminal open.
"""

import dataclasses
import datetime
import functools
import http.server
import io
import json
import logging
import os
import pathlib
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from typing import Any

import dream
import numpy as np
import pytest
import setup_fakes
import so101_fakes
import test_setup_cameras
import test_types_adapter
import yaml
from dream import arm_file, client_ik, credentials
from dream.datasets import Embodiment
from dream.models import Model
from makers_fakes import FakeClock
from setup_fakes import FakeArm, FakeCameras, Hands, Rig, device, sample_rgb
from shared.datasets import embodiments
from shared.tests.datasets import beta_bodies

import robots
from robots import (
    beta_robots,
    bodies,
    config,
    errors,
    feetech,
    followers,
    kinematics,
    run,
    setup_access,
    setup_cameras,
    setup_checks,
    setup_devices,
    setup_model,
    setup_session,
    setup_urdf,
    setup_view,
    setup_wizard,
)
from robots.adapter import EefWire, zero_shot_embodiment


@pytest.fixture(autouse=True)
def _embodiments(monkeypatch: pytest.MonkeyPatch) -> None:
    """The account's embodiments without the network (setup_fakes.stub_embodiments())."""
    setup_fakes.stub_embodiments(monkeypatch)


## Longest a request waits.
WAIT_S = 5.0

## The keys a file from before the credentials file carried, none of which a written file may hold any more.
MOVED_KEYS = ("api_key", "policy", "checkpoint")

## A YAM station's three cameras as the page posts them.
CAMERAS = [
    {"name": "top", "source": "serial", "value": "260322275394", "width": 1280, "height": 720, "fps": 30, "rotate": 0},
    {
        "name": "left",
        "source": "device",
        "value": "/dev/v4l/by-id/usb-cam-L-video-index0",
        "width": 640,
        "height": 360,
        "fps": "30",
        "rotate": 180,
    },
    {
        "name": "right",
        "source": "device",
        "value": "/dev/v4l/by-id/usb-cam-R-video-index0",
        "width": 640,
        "height": 360,
        "fps": 30,
        "rotate": 0,
    },
]

## What a live view says of a fake camera, whose pictures are 640x360, opened at the 1280x720 a robot of your own
## starts from (`setup_wizard.CUSTOM_CAMERA_SIZE`).
FAKE_SIZE = (
    "You typed 1280x720, but this camera gives 640x360: a camera that cannot give the size asked keeps one of its "
    "own, and the station opens it the same way."
)

## The rest pose of each arm dream has a URDF of, the home the saved files carry.
HOMES = {"left": np.array([0.0, 0.5, 0.5, -0.35, 0.0, 0.0]), "right": np.array([0.1, 0.6, 0.4, -0.3, 0.0, 0.2])}


def arms(body: str = "yam", kind: str = "can") -> dict[str, dict[str, Any]]:
    """Both arms as Save takes them from the session."""
    settings = {
        "yam": {"gripper": "linear_4310"},
        "makers": {},
        "metal": {"gripper_closed_deg": 2.0, "gripper_open_deg": 120.0},
    }
    return {
        side: {
            "can_channel": setup_wizard.saved_channel(
                {"kind": kind, "open": f"/dev/serial/by-id/usb-CANable_{side}-if00", "usb": "1-2:1.0"}, side
            ),
            "settings": dict(settings[body]),
            "home": HOMES[side],
            "reach": None,
        }
        for side in ("left", "right")
    }


def robot(body: str) -> setup_session.Robot:
    return next(r for r in setup_session.robot_choices() if r.body == body)


def cameras_for(body: str) -> list[dict[str, Any]]:
    """Camera rows named by the body's roles, checked as the page's are."""
    rows = zip(CAMERAS, robot(body).roles, strict=True)
    return [setup_wizard._camera_row(dict(row, name=role), i) for i, (row, role) in enumerate(rows)]


GOOD_KEY = "sk-dream-test"

BUILTIN_EMBODIMENTS = (
    "yam",
    "openyam",
    "makermods_metal",
    "yam-alt-gripper",
    "so101-bimanual",
    "acme-hand-arm",
    "seeed-rebot-b601-dm",
    "makermods_maker_arm",
)


class FakePlatform:
    """A stand-in for the platform's `/v1/robots` on loopback: lists and registers robots for #GOOD_KEY; and its
    `/v1/meta`, to anyone.

    Attributes:
    - robots: The registered robots, as the platform answers them.
    - meta: What `GET /v1/meta` answers (here only `commit`, the build it runs, or None while it does not say).
    - calls: Every `(method, path)` seen.
    - delay_s: How long each list takes, as over the internet.
    - origin: `http://127.0.0.1:<port>`, what the credentials name as the platform so registration comes here (a
      loopback `http` the live link accepts as `ws`).
    - account: The credentials the wizard runs with: this platform and #GOOD_KEY.
    """

    def __init__(self) -> None:
        self.robots: list[dict[str, Any]] = []
        self.meta: dict[str, Any] = {"commit": None}
        self.calls: list[tuple[str, str]] = []
        self.delay_s = 0.0
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def _authorized(self) -> bool:
                if self.headers.get("Authorization") == f"Bearer {GOOD_KEY}":
                    return True
                self._reply(401, {"error": {"code": "unauthorized", "message": "Invalid api key."}})
                return False

            def _reply(self, status: int, payload: Any) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                fake.calls.append(("GET", self.path))
                time.sleep(fake.delay_s)
                if self.path == "/v1/meta":
                    return self._reply(200, fake.meta)
                if self.path != "/v1/robots":
                    return self._reply(404, {"error": {"code": "not_found", "message": self.path}})
                if self._authorized():
                    self._reply(200, fake.robots)

            def do_POST(self) -> None:
                fake.calls.append(("POST", self.path))
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if self.path != "/v1/robots":
                    return self._reply(404, {"error": {"code": "not_found", "message": self.path}})
                if not self._authorized():
                    return None
                if not (body["embodiment"] in BUILTIN_EMBODIMENTS or body["embodiment"].startswith("emb_")):
                    message = f"Unknown embodiment {body['embodiment']!r}."
                    return self._reply(404, {"error": {"code": "not_found", "message": message}})
                robot = {
                    "id": f"rbt_{len(fake.robots):016x}",
                    "name": body["name"],
                    "embodiment": body["embodiment"],
                    "status": "online",
                    "live": False,
                }
                fake.robots.append(robot)
                return self._reply(201, robot)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        port = self._server.server_address[1]
        self.origin = f"http://127.0.0.1:{port}"
        self.account = credentials.Credentials(
            gateway_url="https://api.example.com",
            platform_url=self.origin,
            data_url="https://data.example.com",
            api_key=GOOD_KEY,
            path=pathlib.Path(),
        )

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def platform(monkeypatch: pytest.MonkeyPatch, credentials_path: pathlib.Path) -> Iterator[FakePlatform]:
    """The fake platform the station registers on, written into this test's credentials file as the platform with
    #GOOD_KEY (so `dreamcli setup` itself would read it), with the `live` extra counted as installed."""
    fake = FakePlatform()
    credentials_path.write_text(f"DREAM_API_URL={fake.origin}\nDREAM_API_KEY={GOOD_KEY}\n", encoding="utf-8")
    fake.account = dataclasses.replace(fake.account, path=credentials_path)
    monkeypatch.setattr(setup_wizard, "LIVE_EXTRA_MODULES", ())
    try:
        yield fake
    finally:
        fake.stop()


@pytest.fixture
def served(tmp_path: pathlib.Path, platform: FakePlatform) -> Iterator[tuple[setup_wizard.SetupServer, str]]:
    """The wizard on an ephemeral port, writing under tmp; yields the server and its base URL."""
    server = setup_wizard.SetupServer(
        tmp_path / "home" / "station.yaml",
        platform.account,
        host="127.0.0.1",
        port=0,
        ask_models=lambda: setup_fakes.MODELS,
        ask_embodiments=lambda: setup_fakes.EMBODIMENTS,
        ask_training_data=setup_fakes.no_training_data,
    )
    server.start()
    try:
        yield server, f"http://127.0.0.1:{server.port}"
    finally:
        server.stop()


def request(base: str, method: str, path: str, body: Any = None, timeout: float = WAIT_S) -> tuple[int, str, bytes]:
    """Sends one request and returns `(status, content type, body)`, for error replies too."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as reply:
            return reply.status, reply.headers.get("Content-Type", ""), reply.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("Content-Type", ""), error.read()


def test_save_writes_a_file_the_agent_accepts_with_the_arms_homes_and_cameras(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    """The written file loads, passes the policy check, carries the body, each arm's adapter, settings and home, the
    cameras and the loop's default values, nothing of the account or the model, and the `live` section naming the
    robot Save registered after the box."""
    target = tmp_path / "home" / "station.yaml"
    mapping = setup_wizard.build_config(robot("yam"), arms(), cameras_for("yam"))
    saved = setup_wizard.save(mapping, target, platform.account, "yam")
    loaded = config.load(target)
    run.policy_config(loaded)
    assert loaded.follower_l.can_channel == "can_follower_l" and loaded.follower_r.can_channel == "can_follower_r"
    assert loaded.follower_l.gripper == "linear_4310"
    np.testing.assert_allclose(loaded.follower_l.home, HOMES["left"], atol=1e-4)
    np.testing.assert_allclose(loaded.follower_r.home, HOMES["right"], atol=1e-4)
    assert [c.name for c in loaded.cameras] == ["top", "left", "right"]
    assert loaded.cameras[0].serial == "260322275394" and loaded.cameras[1].rotate == 180
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert raw["body"] == "yam" and raw["action_horizon"] == setup_wizard.LOOP_KEYS["action_horizon"]
    assert not any(key in raw for key in MOVED_KEYS) and "gateway" not in saved.text and "task" not in raw
    assert raw["live"] == {"robot_id": platform.robots[0]["id"]}
    assert platform.robots[0]["name"] == setup_access.robot_label() and platform.robots[0]["embodiment"] == "yam"
    assert (target.stat().st_mode & 0o777) in (0o600, 0o644, 0o664)


@pytest.mark.parametrize("body", ["makers", "metal"])
def test_a_makers_or_metal_station_saves_with_its_body_by_id_adapters_and_embodiment(
    tmp_path: pathlib.Path, platform: FakePlatform, body: str
) -> None:
    """A serial adapter is saved by its by-id path, the body's own keys load, and the robot registers as the body's
    embodiment."""
    target = tmp_path / "station.yaml"
    mapping = setup_wizard.build_config(robot(body), arms(body, kind="serial"), cameras_for(body))
    setup_wizard.save(mapping, target, platform.account, setup_session.EMBODIMENTS[body])
    loaded = config.load(target)
    assert loaded.body == body and loaded.follower_l.can_channel == "/dev/serial/by-id/usb-CANable_left-if00"
    np.testing.assert_allclose(loaded.follower_r.home, HOMES["right"], atol=1e-4)
    assert platform.robots[0]["embodiment"] == setup_session.EMBODIMENTS[body]


def test_a_makers_or_metal_station_saves_no_model_and_its_own_camera_names_and_a_start_names_the_model(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    """The file names no model, since the checkpoint is picked in the live view, and a run that names one sends the
    station's own camera names."""
    target = tmp_path / "station.yaml"
    target.write_text("body: metal\ncheckpoint: dreamlabs/dream-1-beta\n", encoding="utf-8")
    mapping = setup_wizard.build_config(robot("metal"), arms("metal", kind="serial"), cameras_for("metal"))
    assert "checkpoint" not in mapping
    setup_wizard.save(mapping, target, platform.account, setup_session.EMBODIMENTS["metal"])
    loaded = config.load(target)
    assert loaded.checkpoint is None and "checkpoint" not in run.policy_config(loaded)
    named = run.policy_config(loaded, "dreamlabs/dream-1-beta")
    assert named["checkpoint"] == "dreamlabs/dream-1-beta"
    assert list(named["embodiment"]["video"]) == list(setup_session.MAKERMODS_ROLES)


def test_every_robots_camera_names_resolve_to_a_head_and_two_wrist_cameras() -> None:
    sets = [
        robot("yam").roles,
        setup_session.MAKERMODS_ROLES,
        tuple(setup_fakes.YAM_CAMERAS),
        tuple(setup_fakes.MAKERMODS_OWN["video"]),
        tuple(setup_fakes.EMBODIMENT["video"]),
        ("head", "wrist_l", "wrist_r"),
        ("top", "left_wrist", "right_wrist"),
    ]
    for roles in sets:
        kinds = [cam.kind for cam in setup_cameras.camera_roles(roles).values()]
        assert sorted(kinds) == ["head", "wrist_left", "wrist_right"], roles
    for roles in sets[1:5]:
        assert list(bodies.METAL.wire().embodiment(list(roles), 48)["video"]) == list(roles)


def test_the_robot_step_lists_the_accounts_embodiments_and_names_the_cameras_without_a_model(
    served: tuple[setup_wizard.SetupServer, str],
) -> None:
    pytest.importorskip("dream_ik", reason="dream's IK needs Python 3.11 or newer")
    server, base = served

    def call(method: str, path: str, body: Any = None) -> tuple[int, Any]:
        status, _, data = request(base, method, path, body)
        return status, json.loads(data)

    status, listed = call("GET", "/api/embodiments")
    assert status == 200 and listed["embodiments"][0] == {
        "name": "tester/so101-pick",
        "id": "emb_tester_so101_pick",
        "cameras": ["scene", "wrist_l", "wrist_r"],
    }
    assert call("GET", "/api/models?body=metal")[0] == 404
    status, picked = call("POST", "/api/robot", {"body": "metal"})
    assert status == 200 and "checkpoint" not in picked["robot"]
    assert picked["robot"]["roles"] == list(setup_session.MAKERMODS_ROLES)
    assert [row["name"] for row in picked["prefill"]["cameras"]] == list(setup_session.MAKERMODS_ROLES)
    assert server.session.robot is not None and server.session.robot.embodiment_id == "makermods_metal"


def test_an_embodiment_with_no_settings_is_listed_with_no_cameras_not_a_crash(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    """An account with an embodiment with no settings at all does not crash the setup page when it reads every
    embodiment's camera list: the embodiment is listed with no cameras, beside the others."""
    empty = [setup_fakes.account_embodiment("ops/sourccey", {}), setup_fakes.account_embodiment("ops/none", None)]  # type: ignore[arg-type]
    server = setup_wizard.SetupServer(
        tmp_path / "station.yaml",
        platform.account,
        host="127.0.0.1",
        port=0,
        ask_models=lambda: setup_fakes.MODELS,
        ask_embodiments=lambda: [*empty, *setup_fakes.EMBODIMENTS],
        ask_training_data=setup_fakes.no_training_data,
    )
    server.start()
    try:
        state = json.loads(request(f"http://127.0.0.1:{server.port}", "GET", "/api/state")[2])
        live = json.loads(request(f"http://127.0.0.1:{server.port}", "GET", "/api/status")[2])
        status, _, data = request(f"http://127.0.0.1:{server.port}", "GET", "/api/embodiments")
    finally:
        server.stop()
    listed = json.loads(data)["embodiments"]
    assert status == 200 and [(e["name"], e["cameras"]) for e in listed[:3]] == [
        ("ops/sourccey", []),
        ("ops/none", []),
        ("tester/so101-pick", ["scene", "wrist_l", "wrist_r"]),
    ]
    shown = run_page(
        tmp_path,
        {
            "replies": {
                "GET /api/state": state,
                "GET /api/status": live,
                "GET /api/embodiments": {"embodiments": listed},
            },
            "set": {"embodiments": None},
            "call": "loadEmbodiments",
            "show": ["embodiment"],
        },
    )
    assert shown["errors"] == [], shown
    text = shown["ids"]["embodiment"]["text"]
    assert "ops/sourccey (no cameras)" in text and "tester/so101-pick (cameras: scene, wrist_l, wrist_r)" in text


def test_once_both_arms_are_found_another_port_listed_says_it_is_not_an_arm(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    """Find your arms lists a third USB port beside the two arms'; once both are found, the wizard closes every
    other port, and its row says it is not an arm rather than "Not opened." as though it might be one."""
    ports = [
        {"id": f"serial-{name}", "kind": "serial", "bus": "serial", "open": f"/dev/{name}", "serial": None,
         "usb": None, "label": f"USB serial adapter {name}", "detail": f"/dev/{name}", "state": None, "via": None}
        for name in ("ttyACM0", "ttyACM1", "ttyACM2")
    ]  # fmt: skip
    row = {
        "joints_deg": [0.0] * 6,
        "gripper": 0.5,
        "moved": False,
        "score_deg": 0.0,
        "error": None,
        "calibration": None,
    }
    assigned = {"left": "serial-ttyACM0", "right": "serial-ttyACM1"}
    wiggle = {
        "devices": {ident: {**row, "side": side} for side, ident in assigned.items()},
        "moved": None,
        "several": False,
    }
    status = {"wiggle": wiggle, "assigned": assigned}
    with Station(tmp_path / "station.yaml", platform.account) as st:
        state = st.call("GET", "/api/state")[1]
        live = st.call("GET", "/api/status")[1]
    shown = run_page(
        tmp_path,
        {
            "replies": {"GET /api/state": state, "GET /api/status": live},
            "set": {"scan": {"arms": ports}, "opened": True, "joints": list(range(6)), "status": status},
            "call": "renderWiggle",
            "args": [status],
            "show": ["devices", "wiggleAsk"],
        },
    )
    assert shown["errors"] == [], shown
    text = shown["ids"]["devices"]["text"]
    assert text.count("Not an arm") == 1 and "/dev/ttyACM2Not an arm" in text
    assert "Not opened." not in text and "Both arms found." in shown["ids"]["wiggleAsk"]["text"]


def test_find_your_arms_explains_the_limp_arms_next_to_the_left_and_right_question() -> None:
    """What Find arms does to the arms was said at the top of the step, far from where the
    page asks which arm moved; it is said just under Find arms, right above that question."""
    card = setup_wizard.page_html(False).split('data-step="1"', 1)[1].split("</section>", 1)[0]
    assert card.index('id="openArms"') < card.index("goes limp") < card.index('id="wiggleAsk"')


def test_a_can_link_that_is_not_up_is_told_to_go_down_before_it_comes_up(tmp_path: pathlib.Path) -> None:
    """A link stuck after a bus fault keeps its UP flag and refuses the up command alone with
    `RTNETLINK answers: Device or resource busy`, so the line under the adapter gives the down first; on a link that
    is plainly down the down changes nothing."""
    link = {
        "id": "can-AAA111", "kind": "can", "bus": "can", "open": "can_follower_l", "serial": "AAA111",
        "usb": "1-2:1.0", "label": "CAN link can_follower_l", "detail": "can_follower_l", "state": "down", "via": None,
    }  # fmt: skip
    shown = run_page(
        tmp_path,
        {
            "set": {"scan": {"arms": [link]}, "opened": False, "joints": list(range(6))},
            "call": "renderDevices",
            "args": [None],
            "show": ["devices"],
        },
    )
    assert shown["warnings"]["devices"] == [
        "This CAN link is down: run sudo ip link set can_follower_l down, then sudo ip link set can_follower_l up "
        "type can bitrate 1000000, then look again."
    ]


def test_a_platform_that_cannot_list_models_still_sets_up_and_says_why_it_shows_our_pictures(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    """The models are asked only for the camera step's references (the account's newest fine-tune on the robot's
    embodiment); a YAM asks for none."""
    pytest.importorskip("dream_ik", reason="dream's IK needs Python 3.11 or newer")
    asked: list[str] = []

    def older_platform() -> list[Model]:
        asked.append("models")
        raise dream.DreamProtocolError("the Dream platform answered model 'x' without its camera roles ('cameras')")

    def embodiments() -> list[Embodiment]:
        asked.append("embodiments")
        return setup_fakes.EMBODIMENTS

    server = setup_wizard.SetupServer(
        tmp_path / "station.yaml",
        platform.account,
        host="127.0.0.1",
        port=0,
        model_factory=lambda _arm: yam_model(),
        ask_models=older_platform,
        ask_embodiments=embodiments,
    )
    server.start()
    try:
        base = f"http://127.0.0.1:{server.port}"
        assert request(base, "POST", "/api/robot", {"body": "yam"})[0] == 200 and asked == []
        status, _, data = request(base, "GET", "/api/embodiments")
        assert status == 200 and json.loads(data)["embodiments"][0]["name"] == "tester/so101-pick"
        assert request(base, "POST", "/api/robot", {"body": "metal"})[0] == 200
        assert server.status()["cameras"]["reference_note"] == (
            "could not list the models your account can run: the Dream platform answered model 'x' without its "
            "camera roles ('cameras'); showing our YAM pictures."
        )
        assert asked == ["embodiments", "models"]
    finally:
        server.stop()


def test_the_so101_calibration_files_say_when_each_last_changed(
    served: tuple[setup_wizard.SetupServer, str], tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The page ticks none of the files lerobot-calibrate left on this box that no plugged-in arm holds, and says when
    each last changed, so an old calibration is not saved by mistake; a file
    the existing config names that is gone is offered as not on this box."""
    found = tmp_path / "calibration"
    found.mkdir()
    left = found / "left_arm.json"
    left.write_text("{}", encoding="utf-8")
    os.utime(left, (1_760_000_000, 1_760_000_000))
    monkeypatch.setattr(setup_session, "SO101_CALIBRATION_DIR", found)
    server, _ = served
    server.existing = {"follower_r": {"calibration": str(tmp_path / "moved.json")}}
    assert server.state()["so101_calibrations"] == [
        {"path": str(left), "modified": 1_760_000_000.0, "matches": []},
        {"path": str(tmp_path / "moved.json"), "modified": None, "matches": []},
    ]


def test_the_so101_calibration_files_say_which_plugged_in_arm_holds_each_and_those_come_ticked(
    served: tuple[setup_wizard.SetupServer, str], tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Step 0 listed every calibration file lerobot-calibrate left on this box with none ticked
    and no word of which one the plugged-in arms hold. Each SO-101 serial adapter's servos are read once, read-only
    (no torque, nothing written), and each file says the adapters whose servos hold it; a CAN adapter is not opened,
    and a file no arm holds, or one that is not an SO-101's, matches none."""
    factory = so101_fakes.FakeFeetechFactory()
    monkeypatch.setattr(feetech, "open_bus", factory)
    left, right = "/dev/serial/by-id/usb-left-if00", "/dev/serial/by-id/usb-right-if00"
    factory.eeprom[right] = {"elbow_flex": feetech.Calibration(3, 0, 40, 900, 3000)}
    found = tmp_path / "calibration"
    found.mkdir()
    so101_fakes.write_calibration(found / "a_left.json")
    so101_fakes.write_calibration(found / "b_right.json", elbow_flex=(3, 40, 900, 3000))
    so101_fakes.write_calibration(found / "c_old.json", elbow_flex=(3, 7, 900, 3000))
    (found / "d_broken.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(setup_session, "SO101_CALIBRATION_DIR", found)

    def serial(port: str, label: str) -> dict[str, Any]:
        return {**device(port, label), "id": f"serial-{label}", "kind": "serial", "bus": "serial", "label": label}

    server, _ = served
    server.scan = lambda: {
        "arms": [serial(left, "USB serial adapter L"), serial(right, "USB serial adapter R"), device("can0", "AAA")],
        "cameras": [],
        "can": [],
    }
    listed = server.state()["so101_calibrations"]

    assert [(pathlib.Path(f["path"]).name, f["matches"]) for f in listed] == [
        ("a_left.json", ["USB serial adapter L"]),
        ("b_right.json", ["USB serial adapter R"]),
        ("c_old.json", []),
        ("d_broken.json", []),
    ]
    assert sorted(factory.buses) == [left, right]
    assert all(
        bus.torque_log == [] and not bus.connected and not bus.registers
        for buses in factory.buses.values()
        for bus in buses
    )


def test_an_so101_and_a_robot_of_ones_own_save_as_the_accounts_embodiment(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    """An SO-101 file carries its embodiment and each arm's calibration file, a custom one its `custom` section and
    the gripper ends its wizard measured; both load, pass policy_config, and register as the embodiment their
    fine-tune was trained on, by its id: the platform refuses its name (JD W-2: an SO-101 station's Save failed so)."""
    pytest.importorskip("dream_ik", reason="dream's IK needs Python 3.11 or newer")
    if "so101_faas" not in setup_model.spec_names():
        pytest.skip("this dream IK has no so101_faas")
    calibration = so101_fakes.write_calibration(tmp_path / "arm.json")
    so101 = setup_session.so101_robot([str(calibration)], "tester/so101-pick")
    so101_arms = arms("yam", kind="serial")
    for side in ("left", "right"):
        so101_arms[side]["settings"] = {"calibration": str(calibration)}
        so101_arms[side]["home"] = np.append(HOMES[side][:5], 0.0)
    rows = [
        setup_wizard._camera_row(dict(r, name=n), i) for i, (r, n) in enumerate(zip(CAMERAS, so101.roles, strict=True))
    ]
    mapping = setup_wizard.build_config(so101, so101_arms, rows)
    with pytest.raises(ValueError, match="Unknown embodiment 'tester/so101-pick'"):
        setup_wizard.save(mapping, tmp_path / "so101.yaml", platform.account, "tester/so101-pick")
    setup_wizard.save(mapping, tmp_path / "so101.yaml", platform.account, "emb_tester_so101_pick")
    loaded = config.load(tmp_path / "so101.yaml")
    assert loaded.body == "so101" and loaded.embodiment == "tester/so101-pick"
    assert loaded.follower_l.calibration == str(calibration)
    custom = setup_session.custom_robot("yam", "my.driver:MyArm", "tester/my-arm")
    custom_arms = arms("yam")
    for side in ("left", "right"):
        custom_arms[side]["settings"] = {"gripper_closed_deg": 15.0, "gripper_open_deg": -326.7}
    rows = [
        setup_wizard._camera_row(dict(r, name=n), i)
        for i, (r, n) in enumerate(zip(CAMERAS, custom.roles, strict=True))
    ]
    mapping = setup_wizard.build_config(custom, custom_arms, rows)
    setup_wizard.save(mapping, tmp_path / "mine.yaml", platform.account, "emb_tester_my_arm")
    loaded = config.load(tmp_path / "mine.yaml")
    assert loaded.body == "custom" and loaded.custom == config.CustomConfig("my.driver:MyArm", "yam", "tester/my-arm")
    assert (loaded.follower_r.gripper_closed_deg, loaded.follower_r.gripper_open_deg) == (15.0, -326.7)
    assert [r["embodiment"] for r in platform.robots] == ["emb_tester_so101_pick", "emb_tester_my_arm"]


def test_a_resave_keeps_the_robot_the_file_already_names(tmp_path: pathlib.Path, platform: FakePlatform) -> None:
    target = tmp_path / "station.yaml"
    mapping = setup_wizard.build_config(robot("yam"), arms(), cameras_for("yam"))
    first = setup_wizard.save(mapping, target, platform.account, "yam")
    platform.robots[0]["name"] = "renamed on the platform"
    second = setup_wizard.save(mapping, target, platform.account, "yam")
    assert second.robot["id"] == first.robot["id"] and len(platform.robots) == 1


HAND_EDITS = [
    ("live", "record", False),
    (None, "max_ee_speed_m_s", 0.3),
    (None, "control_hz", 15.0),
    (None, "ramp_s", 3.0),
    ("live", "bitrate_kbps", 1500),
    (None, "task", "fold the towel"),
    (None, "max_stall_ticks", 300),
]


@pytest.mark.parametrize(("section", "key", "value"), HAND_EDITS, ids=[edit[1] for edit in HAND_EDITS])
def test_a_resave_keeps_a_hand_edit(
    section: str | None, key: str, value: Any, tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    target = tmp_path / "station.yaml"
    mapping = setup_wizard.build_config(robot("yam"), arms(), cameras_for("yam"))
    setup_wizard.save(mapping, target, platform.account, "yam")
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    (raw[section] if section else raw)[key] = value
    target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    setup_wizard.save(mapping, target, platform.account, "yam")
    again = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert (again[section] if section else again)[key] == value
    assert len(platform.robots) == 1


def test_a_resave_drops_the_keys_this_release_removed_and_raises_the_stall_floor(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    target = tmp_path / "station.yaml"
    target.write_text(
        "body: yam\n"
        "policy: {url: 'wss://old-gateway', checkpoint: dreamlabs/dream-1-beta}\n"
        "api_key: dl_live_old\n"
        "live: {url: 'wss://old-platform', record: false}\n"
        "control_hz: 15.0\n"
        "max_stall_ticks: 10\n",
        encoding="utf-8",
    )
    mapping = setup_wizard.build_config(robot("yam"), arms(), cameras_for("yam"))
    setup_wizard.save(mapping, target, platform.account, "yam")
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert "policy" not in raw and "api_key" not in raw and "url" not in raw["live"]
    assert raw["live"] == {"robot_id": platform.robots[0]["id"], "record": False}
    assert raw["control_hz"] == 15.0 and raw["max_stall_ticks"] == config.MAX_STALL_TICKS
    assert config.load(target).max_stall_ticks == config.MAX_STALL_TICKS


def test_a_resave_keeps_only_the_live_cameras_the_file_still_has(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    target = tmp_path / "station.yaml"
    mapping = setup_wizard.build_config(robot("yam"), arms(), cameras_for("yam"))
    names = [camera["name"] for camera in mapping["cameras"]]
    setup_wizard.save(mapping, target, platform.account, "yam")
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    raw["live"]["cameras"] = ["scene", names[1]]
    target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    setup_wizard.save(mapping, target, platform.account, "yam")
    assert yaml.safe_load(target.read_text(encoding="utf-8"))["live"]["cameras"] == [names[1]]


def test_a_save_over_another_bodys_file_starts_from_the_station_values(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    target = tmp_path / "station.yaml"
    yam = setup_wizard.build_config(robot("yam"), arms(), cameras_for("yam"))
    setup_wizard.save(yam, target, platform.account, "yam")
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    raw.update(control_hz=15.0, max_ee_speed_m_s=0.3)
    raw["live"]["record"] = False
    target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    metal = setup_wizard.build_config(robot("metal"), arms("metal"), cameras_for("metal"))
    setup_wizard.save(metal, target, platform.account, setup_session.EMBODIMENTS["metal"])
    again = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert "control_hz" not in again and again["max_ee_speed_m_s"] == setup_wizard.LOOP_KEYS["max_ee_speed_m_s"]
    assert "record" not in again["live"]


def test_kept_robot_id_is_the_files_or_none(tmp_path: pathlib.Path) -> None:
    assert setup_wizard.kept_robot_id(tmp_path / "nope.yaml") is None
    broken = tmp_path / "broken.yaml"
    broken.write_text("live: [unclosed\n", encoding="utf-8")
    assert setup_wizard.kept_robot_id(broken) is None
    linked = tmp_path / "linked.yaml"
    linked.write_text("live:\n  robot_id: rbt_abc\n", encoding="utf-8")
    assert setup_wizard.kept_robot_id(linked) == "rbt_abc"


def test_save_refuses_what_the_agent_would_refuse_and_leaves_the_target_untouched(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    """A camera outside the body's roles fails `run.policy_config`, a wrong arm key fails `config.load`; the error
    names it, nothing is written, and the next save finds the robot already registered."""
    target = tmp_path / "station.yaml"
    target.write_text("previous: file\n", encoding="utf-8")
    cameras = cameras_for("yam")
    cameras[0]["name"] = "overhead"
    with pytest.raises(errors.ConfigError, match="overhead"):
        setup_wizard.save(setup_wizard.build_config(robot("yam"), arms(), cameras), target, platform.account, "yam")
    wrong = arms()
    wrong["left"]["settings"] = {"gripper_open_deg": -90.0}
    with pytest.raises(errors.ConfigError, match="gripper_open_deg"):
        setup_wizard.save(
            setup_wizard.build_config(robot("yam"), wrong, cameras_for("yam")), target, platform.account, "yam"
        )
    assert target.read_text(encoding="utf-8") == "previous: file\n"
    assert not target.with_name("station.yaml.tmp").exists()
    assert len(platform.robots) == 1


def test_register_robot_reuses_the_named_robot_then_one_named_after_the_box_then_registers(
    platform: FakePlatform,
) -> None:
    host = socket.gethostname()
    platform.robots = [
        {"id": "rbt_other", "name": host, "embodiment": "so101-bimanual", "status": "offline", "live": False},
        {"id": "rbt_named", "name": host, "embodiment": "yam", "status": "offline", "live": False},
        {"id": "rbt_mine", "name": "garage", "embodiment": "yam", "status": "offline", "live": False},
    ]
    register = setup_wizard.register_robot
    assert register(GOOD_KEY, platform.origin, "rbt_mine", host, "yam")["id"] == "rbt_mine"
    assert register(GOOD_KEY, platform.origin, "rbt_gone", host, "yam")["id"] == "rbt_named"
    assert register(GOOD_KEY, platform.origin + "/", None, host, "yam")["id"] == "rbt_named"
    created = register(GOOD_KEY, platform.origin, None, host, "makermods_metal")
    assert created["embodiment"] == "makermods_metal" and len(platform.robots) == 4


def test_a_second_box_with_the_same_hostname_gets_its_own_robot_not_the_first_ones(
    tmp_path: pathlib.Path, platform: FakePlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = {}
    for name, text in (("one", "1" * 32), ("two", "2" * 32)):
        ids[name] = tmp_path / name
        ids[name].write_text(text + "\n")
    mapping = setup_wizard.build_config(robot("yam"), arms(), cameras_for("yam"))
    monkeypatch.setattr(setup_access, "MACHINE_ID_FILES", (ids["one"],))
    first = setup_wizard.save(mapping, tmp_path / "a.yaml", platform.account, "yam")
    again = setup_wizard.save(mapping, tmp_path / "a2.yaml", platform.account, "yam")
    monkeypatch.setattr(setup_access, "MACHINE_ID_FILES", (ids["two"],))
    second = setup_wizard.save(mapping, tmp_path / "b.yaml", platform.account, "yam")
    assert first.robot["id"] == again.robot["id"], "the same box finds its own record after its file is lost"
    assert second.robot["id"] != first.robot["id"] and len(platform.robots) == 2


def test_save_fails_before_writing_when_the_platform_refuses_the_key_or_is_unreachable(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    target = tmp_path / "station.yaml"
    mapping = setup_wizard.build_config(robot("yam"), arms(), cameras_for("yam"))
    wrong_key = dataclasses.replace(platform.account, api_key="sk-dream-wrong")
    with pytest.raises(ValueError, match=f"{platform.origin} refused the api key: Invalid api key"):
        setup_wizard.save(mapping, target, wrong_key, "yam")
    closed = socket.socket()
    closed.bind(("127.0.0.1", 0))
    port = closed.getsockname()[1]
    closed.close()
    unreachable = dataclasses.replace(platform.account, platform_url=f"http://127.0.0.1:{port}")
    with pytest.raises(ValueError, match="could not reach"):
        setup_wizard.save(mapping, target, unreachable, "yam")
    assert not target.exists() and not target.with_name("station.yaml.tmp").exists()


def test_save_without_the_live_extra_registers_but_writes_no_live_section(
    tmp_path: pathlib.Path, platform: FakePlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(setup_wizard, "LIVE_EXTRA_MODULES", ("websockets", "robots_no_such_extra"))
    target = tmp_path / "station.yaml"
    saved = setup_wizard.save(
        setup_wizard.build_config(robot("yam"), arms(), cameras_for("yam")), target, platform.account, "yam"
    )
    assert saved.live_missing == ("robots_no_such_extra",) and config.load(target).live is None


def test_an_existing_file_prefills_its_bodys_settings_and_cameras() -> None:
    """The file's arm keys (not the adapter, not home) and its cameras by role come back for the same body; another
    body starts from its own defaults and sizes."""
    raw = {
        "follower_l": {"can_channel": "can_follower_l", "gripper": "anvil_gripper", "home": [0] * 6},
        "follower_r": {"can_channel": "can_follower_r", "gripper": "linear_4310"},
        "cameras": [{"name": "left", "serial": "123", "width": 848, "height": 480, "fps": 30, "rotate": 90}],
    }
    yam = setup_wizard.prefill(raw, "yam", robot("yam").roles)
    assert yaml.safe_load(yam["settings"]["left"]) == {"gripper": "anvil_gripper"}
    by_role = {row["name"]: row for row in yam["cameras"]}
    assert by_role["left"] == {
        "name": "left",
        "source": "serial",
        "value": "123",
        "width": 848,
        "height": 480,
        "fps": 30,
        "rotate": 90,
    }
    assert by_role["top"]["value"] == "" and by_role["top"]["width"] == 1280
    metal = setup_wizard.prefill(raw, "metal", robot("metal").roles)
    assert yaml.safe_load(metal["settings"]["left"]) == {"gripper_hold_torque_nm": 0.5}
    assert [row["width"] for row in metal["cameras"]] == [1280, 1280, 1280]
    makers = {"body": "makers", "cameras": [{"name": "scene", "serial": "777", "width": 1920, "height": 1080}]}
    rows = setup_wizard.prefill(makers, "makers", tuple(setup_fakes.YAM_CAMERAS))["cameras"]
    assert [(row["name"], row["value"], row["width"]) for row in rows] == [
        ("overhead", "777", 1920),
        ("wrist_l", "", 640),
        ("wrist_r", "", 640),
    ]
    old = {
        "body": "metal",
        "follower_l": {"can_channel": "can0", "gripper_open_deg": 134.0, "gripper_closed_deg": 0.0},
    }
    assert yaml.safe_load(setup_wizard.prefill(old, "metal", robot("metal").roles)["settings"]["left"]) == {
        "gripper_hold_torque_nm": 0.5
    }


def test_a_new_metal_station_captures_the_16_9_view_the_policy_frame_has() -> None:
    rows = setup_wizard.prefill({}, "metal", robot("metal").roles)["cameras"]
    policy_w, policy_h = config.AppConfig.policy_image_size
    assert [(row["width"], row["height"]) for row in rows] == [(1280, 720)] * 3
    assert all(row["width"] * policy_h == row["height"] * policy_w for row in rows)
    example = yaml.safe_load(
        (pathlib.Path(__file__).resolve().parent.parent / "configs" / "metal.example.yaml").read_text()
    )
    assert [(camera["width"], camera["height"]) for camera in example["cameras"]] == [(1280, 720)] * 3


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("gripper: [", "not YAML"),
        ("- a\n- b\n", "keys and values"),
        ("can_channel: can0\n", "set 'can_channel'"),
        ("home: [0, 0, 0, 0, 0, 0]\n", "set 'home'"),
        ("gripper_open_deg: 134\n", "set 'gripper_open_deg'"),
        (3, "must be text"),
    ],
)
def test_arm_settings_from_the_page_are_checked(text: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        setup_wizard.parse_settings(text, "left")
    assert setup_wizard.parse_settings("", "left") == {}


HOME = {
    "left": (-0.0157, 0.0227, 0.0128, -0.0742, 0.07, -0.1257),
    "right": (0.009, 0.0032, 0.0097, -0.0174, -0.004, 0.0597),
}


@functools.cache
def yam_model() -> setup_model.ArmModel:
    return setup_model.ArmModel("yam")


class Station:
    """The wizard on an ephemeral port over a fake two-arm YAM station, three fake cameras, a fake scan."""

    def __init__(self, target: pathlib.Path, account: credentials.Credentials, agent: bool = False) -> None:
        pytest.importorskip("dream_ik", reason="dream's IK needs Python 3.11 or newer")
        self.model = yam_model()
        self.rig = Rig({"can0": FakeArm(self.model.rest), "can1": FakeArm(self.model.rest)})
        self.clock = FakeClock()
        self.cams = FakeCameras(
            {
                "260322275394": sample_rgb("head"),
                "/dev/v4l/by-id/usb-cam-L-video-index0": sample_rgb("wrist_left"),
                "/dev/v4l/by-id/usb-cam-R-video-index0": sample_rgb("wrist_right"),
            }
        )
        self.devices = [device("can0", "AAA111"), device("can1", "BBB222")]
        shown = [
            {"source": row["source"], "value": row["value"], "label": row["name"], "bound": "serial"}
            for row in CAMERAS
        ]
        self.server = setup_wizard.SetupServer(
            target,
            account,
            host="127.0.0.1",
            port=0,
            agent_running=lambda: agent,
            open_arm=self.rig.open,
            model_factory=lambda _arm: self.model,
            open_camera=self.cams,
            scan=lambda: {"arms": self.devices, "cameras": shown, "can": []},
            preview=lambda **_: setup_cameras.sample_jpeg("head", 0),
            ask_models=lambda: setup_fakes.MODELS,
            ask_embodiments=lambda: setup_fakes.EMBODIMENTS,
            ask_training_data=setup_fakes.no_training_data,
            clock=self.clock,
            sleep=self.clock.sleep,
        )
        self.hands = Hands(self.server.session, self.rig, self.clock)

    def pose_home(self) -> None:
        """The operator puts both arms in their home pose (#HOME)."""
        for side, adapter in (("left", "can0"), ("right", "can1")):
            self.hands.pose(adapter, HOME[side])

    def __enter__(self) -> "Station":
        self.server.start()
        self.base = f"http://127.0.0.1:{self.server.port}"
        return self

    def __exit__(self, *exc: Any) -> None:
        self.server.stop()

    def call(self, method: str, path: str, body: Any = None) -> tuple[int, Any]:
        status, kind, data = request(self.base, method, path, body)
        return status, json.loads(data) if kind == "application/json" else data


def test_the_whole_way_over_the_routes_on_a_clean_station(tmp_path: pathlib.Path, platform: FakePlatform) -> None:
    """Robot, arms by moving them, the rule by serial, cameras (a swap is caught, then fixed), home, the powered checks
    after the confirm with the arms drawn from their readings, the eyes answer, Save; then the file carries what the
    page settled."""
    target = tmp_path / "station.yaml"
    with Station(target, platform.account) as st:
        status, page = st.call("GET", "/")
        assert status == 200 and b"Setup wizard" in page and b'src="/setup_wizard_v2.js"' in page
        for script in setup_wizard.SCRIPTS:
            assert st.call("GET", f"/{script}")[0] == 200
        status, state = st.call("GET", "/api/state")
        assert [r["body"] for r in state["robots"]] == list(config.BODY_NAMES)
        assert state["ik_arms"] == setup_session.own_arms(None) and state["driver_dir"] == str(tmp_path.resolve())
        assert state["robots"][-1]["unavailable"] == (
            None if "so101_faas" in setup_model.spec_names() else "this install's dream IK has no so101_faas"
        )
        assert state["picked"] is None
        assert st.call("POST", "/api/arms/open", {"devices": ["can-AAA111"], "settings": {}})[0] == 400
        status, picked = st.call("POST", "/api/robot", {"body": "yam"})
        assert status == 200 and picked["robot"]["roles"] == ["top", "left", "right"]

        assert [d["id"] for d in st.call("GET", "/api/scan")[1]["arms"]] == ["can-AAA111", "can-BBB222"]
        settings = picked["prefill"]["settings"]
        status, opened = st.call(
            "POST", "/api/arms/open", {"devices": ["can-AAA111", "can-BBB222"], "settings": settings}
        )
        assert status == 200 and opened["opened"] == {"can-AAA111": None, "can-BBB222": None}
        for adapter, ident, side in (("can0", "can-AAA111", "left"), ("can1", "can-BBB222", "right")):
            st.hands.move(adapter, 0, 0.3)
            wiggle = st.call("GET", "/api/status")[1]["wiggle"]
            assert wiggle["moved"] == ident and wiggle["devices"][ident]["moved"]
            status, said = st.call("POST", "/api/arms/assign", {"device": ident, "side": side})
            assert status == 200 and said["crossed"] is None
        assert said["done"]
        status, rule = st.call("GET", "/api/udev_rule")
        assert 'ATTRS{serial}=="AAA111", NAME="can_follower_l"' in rule["rule"] and rule["in_effect"] == {
            "left": False,
            "right": False,
        }

        assert st.call("GET", "/api/preview?source=device&value=x&width=640&height=480&fps=30")[0] == 200
        swapped = [dict(row) for row in CAMERAS]
        swapped[1]["value"], swapped[2]["value"] = swapped[2]["value"], swapped[1]["value"]
        assert st.call("POST", "/api/cameras/open", {"cameras": swapped})[0] == 200
        assert [p["roles"] for p in st.call("GET", "/api/cameras/match")[1]["swapped"]] == [["left", "right"]]
        assert st.call("POST", "/api/cameras/open", {"cameras": CAMERAS})[0] == 200
        assert st.call("GET", "/api/cameras/match")[1] == {"swapped": [], "checked": ["left", "right", "top"]}
        live = {"state": "live", "error": None, "tries": 0}
        assert st.call("GET", "/api/status")[1]["cameras"]["states"] == dict.fromkeys(("top", "left", "right"), live)
        status, ref = st.call("GET", "/api/reference?role=top&episode=1")
        assert status == 200 and ref == setup_cameras.sample_jpeg("head", 1)
        status, model_view = st.call("GET", "/api/cameras/frame?role=left&view=model")
        policy_size = config.AppConfig.policy_image_size
        assert status == 200 and setup_cameras.decode_jpeg(model_view).shape == (policy_size[1], policy_size[0], 3)
        assert st.call("GET", "/api/cameras/frame?role=left&view=wide")[0] == 400

        assert st.call("GET", "/api/home/pose?episode=1")[0] == 404
        status, view = st.call("GET", "/api/view")
        links = len(view["links"])
        assert status == 200 and view["kind"] == "meshes" and len(view["joints"]) == 6
        now = st.call("GET", "/api/status")[1]["now"]
        assert all(len(now[side]["frame"]) == links for side in ("left", "right"))
        assert st.call("POST", "/api/motion/start", {"confirm": True})[0] == 400  # no home yet
        st.pose_home()
        status, home = st.call("POST", "/api/home", {})
        assert status == 200 and home["out_of_range"] == [] and "absolute" not in home
        again = st.call("GET", "/api/state")[1]["picked"]
        assert again["robot"]["body"] == "yam" and again["robot"]["roles"] == ["top", "left", "right"]
        assert [row["value"] for row in again["prefill"]["cameras"]] == [row["value"] for row in CAMERAS]
        assert all(arm.writes == 0 for arm in st.rig.arms.values())

        assert st.call("POST", "/api/motion/start", {})[0] == 400
        assert st.call("POST", "/api/motion/start", {"confirm": True})[0] == 200
        assert st.server.session.wait_motion(120.0)
        status, after = st.call("GET", "/api/status")
        motion = after["motion"]
        assert motion["state"] == "done" and all(r["finding"]["state"] == "pass" for r in motion["results"])
        held = st.server.session.motion.latest
        for side in ("left", "right"):
            assert after["now"][side]["frame"] == st.model.frame(held[side].joints)
        assert st.call("POST", "/api/motion/eyes", {"same": "yes"})[0] == 400
        status, eyes = st.call("POST", "/api/motion/eyes", {"same": True})
        assert status == 200 and eyes["state"] == "done" and eyes["results"][-1]["group"] == "eyes"

        status, saved = st.call("POST", "/api/save", {"cameras": CAMERAS})
        assert status == 200, saved
        fresh = st.call("GET", "/api/status")[1]
        assert fresh["powered"] is True and fresh["motion"]["state"] == "done"
        assert "click Turn off motors first" in st.call("POST", "/api/robot", {"body": "yam"})[1]["error"]
        assert st.call("POST", "/api/motion/release", {})[1]["powered"] is False
        assert st.call("POST", "/api/exit")[0] == 200 and st.server.done.is_set()
    loaded = config.load(target)
    np.testing.assert_allclose(loaded.follower_l.home, HOME["left"], atol=1e-4)
    assert loaded.follower_l.can_channel == "can_follower_l" and saved["robot"]["embodiment"] == "yam"


def test_swap_arms_names_both_arms_the_other_side_in_one_request(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    """Swap arms is one request: both arms change side together and stay open, and it is refused before both are
    found, so no arm is ever named twice."""
    with Station(tmp_path / "station.yaml", platform.account) as st:
        settings = st.call("POST", "/api/robot", {"body": "yam"})[1]["prefill"]["settings"]
        st.call("POST", "/api/arms/open", {"devices": ["can-AAA111", "can-BBB222"], "settings": settings})
        assert "find both arms" in st.call("POST", "/api/arms/swap", {})[1]["error"]
        for adapter, ident, side in (("can0", "can-AAA111", "left"), ("can1", "can-BBB222", "right")):
            st.hands.move(adapter, 0, 0.3)
            assert st.call("GET", "/api/status")[1]["wiggle"]["moved"] == ident
            assert st.call("POST", "/api/arms/assign", {"device": ident, "side": side})[0] == 200
        status, swapped = st.call("POST", "/api/arms/swap", {})
        assert status == 200 and swapped["assigned"] == {"left": "can-BBB222", "right": "can-AAA111"}
        assert sorted(st.server.session.bank.devices()) == ["can-AAA111", "can-BBB222"]


def test_two_saves_at_once_register_one_robot(tmp_path: pathlib.Path, platform: FakePlatform) -> None:
    target = tmp_path / "station.yaml"
    with Station(target, platform.account) as st:
        settings = st.call("POST", "/api/robot", {"body": "yam"})[1]["prefill"]["settings"]
        st.call("POST", "/api/arms/open", {"devices": ["can-AAA111", "can-BBB222"], "settings": settings})
        for adapter, ident, side in (("can0", "can-AAA111", "left"), ("can1", "can-BBB222", "right")):
            st.hands.move(adapter, 0, 0.3)
            assert st.call("GET", "/api/status")[1]["wiggle"]["moved"] == ident
            assert st.call("POST", "/api/arms/assign", {"device": ident, "side": side})[0] == 200
        for ident, side in (("can-AAA111", "left"), ("can-BBB222", "right")):
            st.server.session.devices[ident]["open"] = setup_devices.CAN_LINKS[side]
        st.pose_home()
        assert st.call("POST", "/api/home", {})[0] == 200
        platform.delay_s = 0.3
        replies: list[tuple[int, Any]] = []
        clicks = [
            threading.Thread(target=lambda: replies.append(st.call("POST", "/api/save", {"cameras": CAMERAS})))
            for _ in range(2)
        ]
        for click in clicks:
            click.start()
        for click in clicks:
            click.join()
    assert [status for status, _ in replies] == [200, 200], replies
    assert len(platform.robots) == 1 and {saved["robot"]["id"] for _, saved in replies} == {platform.robots[0]["id"]}
    assert setup_wizard.kept_robot_id(target) == platform.robots[0]["id"]


def test_a_running_agent_stops_the_page_from_opening_arms_or_cameras(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    with Station(tmp_path / "station.yaml", platform.account, agent=True) as st:
        assert st.call("GET", "/api/state")[1]["agent_running"] is True
        status, reply = st.call("POST", "/api/robot", {"body": "yam"})
        assert status == 400 and "dreamcli robot down" in reply["error"]


def test_the_scan_offers_what_fits_the_robot_and_look_again_lets_go(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    with Station(tmp_path / "station.yaml", platform.account) as st:
        st.devices.append(dict(device("/dev/ttyUSB0", "FTBENQYQ"), id="serial-ftdi", kind="serial", bus="serial"))
        everything = st.call("GET", "/api/scan")[1]
        assert len(everything["arms"]) == 3 and everything["others"] == [], "no robot yet: nothing is ruled out"
        st.call("POST", "/api/robot", {"body": "yam"})
        found = st.call("GET", "/api/scan")[1]
        assert [d["id"] for d in found["arms"]] == ["can-AAA111", "can-BBB222"]
        assert [d["id"] for d in found["others"]] == ["serial-ftdi"]
        both = {"devices": ["can-AAA111", "can-BBB222"], "settings": {}}
        assert st.call("POST", "/api/arms/open", both)[1]["opened"] == {"can-AAA111": None, "can-BBB222": None}
        st.hands.move("can0", 0, 0.3)
        assert st.call("POST", "/api/arms/assign", {"device": "can-AAA111", "side": "left"})[0] == 200
        assert st.call("POST", "/api/arms/close", {})[1] == {"ok": True}
        assert all(handle.closed for handle in st.rig.handles) and st.server.session.assigned == {}
        assert st.call("POST", "/api/arms/open", both)[1]["opened"] == {"can-AAA111": None, "can-BBB222": None}


## Runs the page's own script on node, on a stub page (its header says what it stubs and what it prints).
PAGE = pathlib.Path(__file__).with_name("setup_page.mjs")


def run_page(tmp_path: pathlib.Path, scenario: dict[str, Any]) -> dict[str, Any]:
    """Runs setup_wizard.js (or the scenario's `script`) on node with the scenario (tests/setup_page.mjs) and returns
    what the page shows."""
    file = tmp_path / "scenario.json"
    file.write_text(json.dumps(scenario), encoding="utf-8")
    done = subprocess.run(["node", str(PAGE), str(file)], capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    shown: dict[str, Any] = json.loads(done.stdout)
    return shown


def picked_page(picked: dict[str, Any]) -> dict[str, Any]:
    """What the page keeps of a pick reply (applyPick()): the robot, and the server's labels of each camera."""
    return {"robot": picked["robot"], "labels": {cam["role"]: cam for cam in picked["cameras"]}}


def test_rotate_turns_a_live_view_a_quarter_turn_and_opens_the_views_again(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    """Rotation is not only a text box among the capture settings. Each live
    view's picture has a Rotate button: it turns that camera's `rotate`, the key the saved file keeps, a quarter turn
    clockwise and opens the views again. Its chip says the current turn."""
    with Station(tmp_path / "station.yaml", platform.account) as st:
        state = st.call("GET", "/api/state")[1]
        picked = st.call("POST", "/api/robot", {"body": "yam"})[1]
        opened = st.call("POST", "/api/cameras/open", {"cameras": CAMERAS})[1]
        live = st.call("GET", "/api/status")[1]
    shown = run_page(
        tmp_path,
        {
            "replies": {"GET /api/state": state, "GET /api/status": live, "POST /api/cameras/open": opened},
            "set": {**picked_page(picked), "cams": {row["name"]: row for row in CAMERAS}},
            "call": "rotateCamera",
            "args": ["left"],
            "show": ["camErr", "compareRows"],
        },
    )
    turned = [{**row, "rotate": 270} if row["name"] == "left" else row for row in CAMERAS]
    assert shown["ids"]["camErr"]["text"] == ""
    assert shown["requests"][-1] == {"key": "POST /api/cameras/open", "body": {"cameras": turned}}
    rows = shown["ids"]["compareRows"]["text"]
    assert re.findall(r"\u21bbRotate 90\u00b0now (\d+)\u00b0", rows) == ["0", "270", "0"]


def test_rotate_on_a_camera_picture_in_the_role_step_turns_it_and_its_roles_rotate(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    """A camera mounted upside down or sideways needs turning at the role step too, not only on the live views.
    Each picture of the role step has a Rotate button:
    a quarter turn clockwise of the picture, kept as the `rotate` of the role holding that camera, the value the live
    views open with and the saved file keeps. A picture of a role whose `rotate` is already set shows it turned."""
    with Station(tmp_path / "station.yaml", platform.account) as st:
        state = st.call("GET", "/api/state")[1]
        picked = st.call("POST", "/api/robot", {"body": "yam"})[1]
        scan = st.call("GET", "/api/scan")[1]
        opened = st.call("POST", "/api/cameras/open", {"cameras": CAMERAS})[1]
        live = st.call("GET", "/api/status")[1]
    assert [cam["value"] for cam in scan["cameras"]] == [row["value"] for row in CAMERAS]
    replies = {"GET /api/state": state, "GET /api/status": live, "GET /api/scan": scan, "GET /api/preview": "picture"}
    shown = run_page(
        tmp_path,
        {
            "replies": {**replies, "POST /api/cameras/open": opened},
            "set": {**picked_page(picked), "cams": {row["name"]: row for row in CAMERAS}},
            "call": "shootAll",
            "press": [["gallery", "button.turn", 2]],
            "click": ["liveViews"],
            "show": ["camErr", "gallery"],
        },
    )
    turned = [{**row, "rotate": 90} if row["name"] == "right" else row for row in CAMERAS]
    assert shown["ids"]["camErr"]["text"] == ""
    gallery = shown["ids"]["gallery"]
    assert re.findall(r"\u21bbRotate 90\u00b0now (\d+)\u00b0", gallery["text"]) == ["0", "180", "90"]
    assert gallery["turned"] == ["rotate(180deg)", "rotate(90deg) scale(0.5625)"]
    assert shown["requests"][-1] == {"key": "POST /api/cameras/open", "body": {"cameras": turned}}


def test_show_live_views_draws_every_camera_from_the_routes_own_reply(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    """POST /api/cameras/open answers each role under "opened", and the page reads the roles from there, not off
    the reply itself. The page's script runs here on the replies the routes send, the head camera a 4:3 one, whose
    shape note shows under its row as a caption, not a warning: any shape is fine for setup."""
    with Station(tmp_path / "station.yaml", platform.account) as st:
        st.cams.pictures[CAMERAS[0]["value"]] = sample_rgb("head")[:, :480]  # (360, 480, 3): 4:3
        state = st.call("GET", "/api/state")[1]
        picked = st.call("POST", "/api/robot", {"body": "yam"})[1]
        status, opened = st.call("POST", "/api/cameras/open", {"cameras": CAMERAS})
        assert status == 200 and opened["opened"]["top"]["warning"] and not opened["opened"]["left"]["warning"]
        live = st.call("GET", "/api/status")[1]
    shown = run_page(
        tmp_path,
        {
            "replies": {"GET /api/state": state, "GET /api/status": live, "POST /api/cameras/open": opened},
            "set": {**picked_page(picked), "cams": {row["name"]: row for row in CAMERAS}},
            "call": "liveViews",
            "show": ["camErr", "compareRows", "compare"],
        },
    )
    assert shown["ids"]["camErr"]["text"] == ""
    assert (
        shown["ids"]["compareRows"]["children"] == len(picked["robot"]["roles"])
        and not shown["ids"]["compare"]["hidden"]
    )
    note = opened["opened"]["top"]["warning"]
    assert note in shown["ids"]["compareRows"]["text"] and note not in shown["warnings"]["compareRows"]
    assert shown["requests"][-1] == {"key": "POST /api/cameras/open", "body": {"cameras": CAMERAS}}


def test_the_live_views_say_the_size_each_camera_delivers_and_warn_when_it_is_not_the_one_typed(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    """A camera that cannot give the mode asked keeps one of its own, so changing its width or height in the
    setup wizard can seem to do nothing. Each live view says the size its camera delivers, from the
    open's answer, turned back to the camera's own width and height when `rotate` is a quarter turn; a size other
    than the one typed is a warning. Here the head camera gives 480x360 for the 1280x720 typed, and the right one,
    turned 90, gives its typed 640x360."""
    turned = [{**row, "rotate": 90} if row["name"] == "right" else row for row in CAMERAS]
    with Station(tmp_path / "station.yaml", platform.account) as st:
        st.cams.pictures[CAMERAS[0]["value"]] = sample_rgb("head")[:, :480]  # (360, 480, 3)
        st.cams.pictures[CAMERAS[2]["value"]] = np.ascontiguousarray(sample_rgb("wrist_right").transpose(1, 0, 2))
        state = st.call("GET", "/api/state")[1]
        picked = st.call("POST", "/api/robot", {"body": "yam"})[1]
        opened = st.call("POST", "/api/cameras/open", {"cameras": turned})[1]
        live = st.call("GET", "/api/status")[1]
    assert (opened["opened"]["right"]["width"], opened["opened"]["right"]["height"]) == (360, 640)
    shown = run_page(
        tmp_path,
        {
            "replies": {"GET /api/state": state, "GET /api/status": live, "POST /api/cameras/open": opened},
            "set": {**picked_page(picked), "cams": {row["name"]: row for row in turned}},
            "call": "liveViews",
            "show": ["camErr", "compareRows"],
        },
    )
    assert shown["ids"]["camErr"]["text"] == ""
    rows = shown["ids"]["compareRows"]["text"]
    assert re.findall(r"Your camera delivers (\d+x\d+)\.", rows) == ["480x360", "640x360", "640x360"]
    mismatch = FAKE_SIZE.replace("640x360", "480x360")
    assert shown["warnings"]["compareRows"] == [mismatch]
    assert all(opened["opened"][role]["warning"] in rows for role in ("top", "right"))


def test_a_typed_camera_size_opens_the_live_views_again_once_typing_stops(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    """A width, height or frame rate typed under Capture settings changed nothing on the screen until *Show live
    views* was pressed again. While the live views are shown, a typed size opens them again with it once
    typing stops for half a second, and only for a whole number: "1280x" opens nothing, and only the last of
    several values typed in a row is sent."""
    with Station(tmp_path / "station.yaml", platform.account) as st:
        state = st.call("GET", "/api/state")[1]
        picked = st.call("POST", "/api/robot", {"body": "yam"})[1]
        opened = st.call("POST", "/api/cameras/open", {"cameras": CAMERAS})[1]
        live = st.call("GET", "/api/status")[1]
    shown = run_page(
        tmp_path,
        {
            "replies": {"GET /api/state": state, "GET /api/status": live, "POST /api/cameras/open": opened},
            "set": {**picked_page(picked), "cams": {row["name"]: dict(row) for row in CAMERAS}},
            "call": "markRoles",
            "after": [
                "liveViews",
                {"input": ["camRows", "input", 0, "12"]},
                {"input": ["camRows", "input", 0, "1280x"]},
                {"wait": 500},
                {"input": ["camRows", "input", 0, "192"]},
                {"input": ["camRows", "input", 0, "1920"]},
                {"wait": 500},
            ],
            "show": ["camErr"],
        },
    )
    assert shown["ids"]["camErr"]["text"] == ""
    opens = [request["body"] for request in shown["requests"] if request["key"] == "POST /api/cameras/open"]
    assert opens == [{"cameras": CAMERAS}, {"cameras": [{**CAMERAS[0], "width": "1920"}, *CAMERAS[1:]]}]


def test_dreams_model_of_your_own_arm_starts_unpicked_and_a_rerun_brings_back_the_files(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    """The custom card starts with no model of the arm picked (a wrong one draws and solves another arm): neither
    way to give it is on, and Continue stays off until one is. The driver's file picker opens in the station file's
    folder. A file saved
    for a robot of one's own brings back its driver, open in the picker at its file and class, its arm model (listed
    by its name though the tiles above offer it), its embodiment, and Continue. The dry-run line is Save's, with the
    model a robot of your own's run must name."""
    embodiments = {"embodiments": [{"name": "tester/mine", "cameras": ["scene"]}]}
    show = [
        "armUrdf",
        "armOurs",
        "armList",
        "urdfBox",
        "driverDir",
        "driverList",
        "driverClasses",
        "driverPicked",
        "customDriver",
        "pickRobot",
        "embodiment",
        "customRun",
    ]
    driver_file = tmp_path / "my_robot" / "my_robot.py"
    driver_file.parent.mkdir()
    driver_file.write_text("class Arm:\n    pass\n\n\nclass ReadOnlyArm(Arm):\n    pass\n", encoding="utf-8")
    with Station(tmp_path / "station.yaml", platform.account) as st:
        fresh = st.call("GET", "/api/state")[1]
        shown = run_page(
            tmp_path,
            {
                "replies": {"GET /api/embodiments": embodiments},
                "base": st.base,
                "set": {"pick": "custom"},
                "call": "showPick",
                "show": show,
            },
        )
    ids = shown["ids"]
    assert shown["errors"] == [] and ids["pickRobot"]["blocked"]
    assert ids["armList"]["hidden"] and ids["urdfBox"]["hidden"]  # neither way to give the model is picked
    assert ids["driverDir"]["text"] == str(tmp_path.resolve()) and "my_robot/" in ids["driverList"]["text"]
    assert ids["driverClasses"]["hidden"] and ids["driverPicked"]["hidden"]
    assert [r["key"] for r in shown["requests"]] == [
        "GET /api/state",
        "GET /api/status",
        "GET /api/files",
        "GET /api/embodiments",
    ]
    assert ids["customRun"]["text"] == (
        f'dreamcli robot run --config {fresh["config_path"]} --checkpoint "<your model>" --task "..." --dry-run'
    )
    driver = f"{driver_file.resolve()}:Arm"
    saved = {"body": "custom", "custom": {"driver": driver, "ik": "yam", "embodiment": "tester/mine"}}
    (tmp_path / "station.yaml").write_text(yaml.safe_dump(saved), encoding="utf-8")
    with Station(tmp_path / "station.yaml", platform.account) as st:
        rerun = st.call("GET", "/api/state")[1]
        shown = run_page(
            tmp_path,
            {
                "replies": {"GET /api/embodiments": embodiments},
                "base": st.base,
                "set": {},
                "call": "showPick",
                "show": show,
            },
        )
    ids = shown["ids"]
    assert rerun["ik_arms"][-1] == {"name": "yam", "label": "yam (your station file's)"}
    assert (
        shown["errors"] == []
        and not ids["armList"]["hidden"]
        and "yam (your station file's)" in ids["armList"]["text"]
    )
    assert ids["driverDir"]["text"] == str(driver_file.parent.resolve())
    assert ids["driverClasses"]["text"] == "my_robot.py has 2 classes. Which one runs one arm?ArmReadOnlyArm"
    assert ids["driverPicked"]["text"] == "\u2713 Using my_robot.py: Arm" and ids["customDriver"]["value"] == ""
    assert ids["embodiment"]["value"] == "tester/mine" and not ids["pickRobot"]["blocked"]


def test_the_driver_is_picked_on_the_station_and_dreams_arm_by_its_plain_name(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    """The driver is picked from the station's own files, never typed as a path, and dream's model of the arm is an
    obvious choice. Choose your driver file opens the station's folders in a dialog; a folder, then a
    file of one class picks that class and closes the dialog, and the block says so with a check; then dream's model
    by its tile, none of them an arm the robot tiles are; Continue is on. A driver installed as a package is typed
    instead."""
    embodiments = {"embodiments": [{"name": "tester/mine", "cameras": ["scene"]}]}
    box = tmp_path / "box"
    driver_file = box / "my_robot" / "my_robot.py"
    driver_file.parent.mkdir(parents=True)
    driver_file.write_text("class Arm:\n    pass\n", encoding="utf-8")
    with Station(box / "station.yaml", platform.account) as st:
        picked = run_page(
            tmp_path,
            {
                "replies": {"GET /api/embodiments": embodiments},
                "base": st.base,
                "set": {"pick": "custom"},
                "values": {"embodiment": "tester/mine"},
                "call": "showPick",
                "after": [
                    {"click": "driverChoose"},
                    {"press": ["driverList", "button", 0]},
                    {"press": ["driverList", "button", 0]},
                    {"click": "armOurs"},
                    {"press": ["armList", "button", 0]},
                ],
                "show": ["driverPicked", "driverDialog", "driverClasses", "driverReq", "armList", "pickRobot"],
            },
        )
        typed = run_page(
            tmp_path,
            {
                "replies": {"GET /api/embodiments": embodiments},
                "base": st.base,
                "set": {"pick": "custom"},
                "values": {"customDriver": " my_pkg.arm:Arm "},
                "call": "showPick",
                "click": ["driverUse"],
                "show": ["driverPicked"],
            },
        )
    ids = picked["ids"]
    assert picked["errors"] == [] and ids["driverPicked"]["text"] == "\u2713 Using my_robot.py: Arm"
    assert not ids["driverDialog"]["open"] and ids["driverClasses"]["hidden"] and ids["driverReq"]["text"] == "Done"
    assert ids["armList"]["text"].startswith("ROBOTIS AI Workerai_worker_faas")
    assert not any(name in ids["armList"]["text"] for name in ("YAM", "Maker", "Metal", "SO-101", "b601"))
    assert not ids["pickRobot"]["blocked"]
    assert [r["key"] for r in picked["requests"] if r["key"] in ("GET /api/files", "GET /api/classes")] == [
        "GET /api/files",
        "GET /api/files",
        "GET /api/classes",
    ]
    assert typed["errors"] == [] and typed["ids"]["driverPicked"]["text"] == "\u2713 Using my_pkg.arm:Arm"


def test_the_driver_pickers_routes_and_the_template_link(tmp_path: pathlib.Path, platform: FakePlatform) -> None:
    """GET /api/files and /api/classes answer the driver's file picker, a path that is not one 400 with the reason;
    /driver_template.py is the file the page's "Driver template" link opens."""
    box = tmp_path / "box"
    (box / "my_robot").mkdir(parents=True)
    (box / "my_robot" / "my_robot.py").write_text("class Arm:\n    pass\n", encoding="utf-8")
    with Station(tmp_path / "station.yaml", platform.account) as st:
        status, listing = st.call("GET", f"/api/files?dir={urllib.parse.quote(str(box))}")
        assert status == 200 and (listing["dirs"], listing["files"]) == (["my_robot"], [])
        status, found = st.call(
            "GET", f"/api/classes?file={urllib.parse.quote(str(box / 'my_robot' / 'my_robot.py'))}"
        )
        assert status == 200 and found["classes"] == ["Arm"]
        assert st.call("GET", "/api/files?dir=relative")[1] == {"error": "relative is not a folder on this computer"}
        assert st.call("GET", "/api/classes")[1] == {"error": "'file' is required"}
        status, template = st.call("GET", "/driver_template.py")
        assert status == 200 and b"class TemplateArm" in template


def test_a_robot_of_your_owns_arms_are_dreams_but_the_robot_tiles_and_second_names() -> None:
    """A second "pick your robot" would be redundant with the robot tiles. Every arm dream's IK has
    is a tile's, a robot of your own's under its plain name, or a second name of one of those (b601), so a new arm in
    dream cannot go unlisted (an arm file another test built registers as `file:<hash>`, `dream.arm_file`). dream's
    zero-shot simulation robots (`dream_ik.specs.ZERO_SHOT_ROBOTS`) are not station hardware and are left
    out. An arm the list leaves out comes back when the station file names it."""
    from dream_ik.specs import ZERO_SHOT_ROBOTS

    tiles = {robot.ik_arm for robot in setup_session.robot_choices()}
    own = set(setup_session.OWN_ARM_LABELS)
    shipped = {name for name in setup_model.spec_names() if not name.startswith("file:")} - set(ZERO_SHOT_ROBOTS)
    assert shipped == tiles | own | set(setup_session.SAME_ARM) and not tiles & own
    arms = setup_session.own_arms(None)
    assert [arm["name"] for arm in arms] == list(setup_session.OWN_ARM_LABELS)
    assert arms[0] == {"name": "ai_worker_faas", "label": "ROBOTIS AI Worker"}
    assert setup_session.own_arms("b601") == [*arms, {"name": "b601", "label": "b601 (your station file's)"}]
    assert setup_session.own_arms("seeed_faas") == arms == setup_session.own_arms("/home/me/arms/mine.json")


def test_bad_requests_are_400_with_the_reason(tmp_path: pathlib.Path, platform: FakePlatform) -> None:
    with Station(tmp_path / "station.yaml", platform.account) as st:
        assert st.call("GET", "/api/nope")[0] == 404
        assert st.call("POST", "/api/robot", {"body": "spot"})[1]["error"] == "no robot 'spot' here"
        assert st.call("POST", "/api/robot", {"custom": {"ik_arm": "yam", "driver": ""}})[0] == 400
        assert st.call("GET", "/api/reference?role=top")[0] == 400  # no robot yet
        st.call("POST", "/api/robot", {"body": "yam"})
        assert (
            "not plugged in" in st.call("POST", "/api/arms/open", {"devices": ["can-X"], "settings": {}})[1]["error"]
        )
        assert st.call("POST", "/api/save", {"cameras": CAMERAS})[1]["error"] == "find both arms before saving"
        assert "powered checks to their end" in st.call("POST", "/api/motion/eyes", {"same": True})[1]["error"]
        assert st.call("POST", "/api/cameras/open", {"cameras": CAMERAS[:2]})[0] == 400
        assert st.call("GET", "/api/cameras/frame?role=top")[0] == 400
        assert st.call("POST", "/api/dataset", {"path": str(tmp_path)})[0] == 400
        assert st.call("POST", "/api/dataset", {"path": ""})[1] == {"path": None}
        if "so101_faas" not in setup_model.spec_names():
            body = {"body": "so101", "calibration": ["/x.json"], "embodiment": "tester/so101"}
            refused = st.call("POST", "/api/robot", body)[1]["error"]
            assert refused == "the SO-101 cannot be picked here: this install's dream IK has no so101_faas"
        custom = {"ik_arm": "yam", "driver": "mine:Open", "embodiment": "tester/mine"}
        unknown = {**custom, "embodiment": "tester/nope"}
        refused = st.call("POST", "/api/robot", {"custom": unknown})[1]["error"]
        assert refused.startswith("your account has no embodiment 'tester/nope'; it has: tester/so101-pick")
        status, picked = st.call("POST", "/api/robot", {"custom": custom})
        assert status == 200 and picked["robot"]["body"] == "custom" and picked["prefill"]["settings"]["left"] == ""
        assert picked["robot"]["embodiment_id"] == "emb_tester_mine" and "checkpoint" not in picked["robot"]


def test_an_arm_of_your_own_from_its_urdf_is_written_beside_the_file(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    """`POST /api/arm_file` writes the URDF and an arm file of it to `arms/` beside the station file and builds dream's
    model of the arm; it registers nothing, since the embodiment is the pick's. The file records whether the operator
    marked the tool point or skipped it, and when, so that skipping it is a deliberate choice on record. A link the
    URDF lacks, a malformed tool point or no record is refused by name."""
    specs = pytest.importorskip("dream_ik.specs", reason="dream's IK needs Python 3.11 or newer")
    added: list[tuple[str, dict[str, Any]]] = []
    target = tmp_path / "home" / "station.yaml"
    server = setup_wizard.SetupServer(
        target,
        platform.account,
        host="127.0.0.1",
        port=0,
        ask_models=lambda: setup_fakes.MODELS,
        ask_embodiments=lambda: setup_fakes.EMBODIMENTS,
        add_embodiment=lambda name, wire: added.append((name, wire)),
    )
    urdf = (specs.ASSETS / "metal.urdf").read_text()
    upload = {
        "filename": "B601 Arm.urdf",
        "urdf": urdf,
        "base": "base_link",
        "flange": "link6",
        "tcp": [0.135, 0, 0],
        "approach": [1, 0, 0],
        "jaw": [0, 1, 0],
        "tcp_calibration": "calibrated",
        "fixed": [],
    }
    server.start()
    try:
        base = f"http://127.0.0.1:{server.port}"

        def call(body: dict[str, Any]) -> tuple[int, Any]:
            status, _, data = request(base, "POST", "/api/arm_file", body)
            return status, json.loads(data)

        arms = target.parent / setup_wizard.ARMS_DIR
        joints = list(specs.EMBODIMENTS["metal"].side_arm("right").joints)
        assert call(upload) == (200, {"ik": str(arms / "b601-arm.json"), "joints": joints})
        assert (arms / "b601-arm.urdf").read_text() == urdf
        written = json.loads((arms / "b601-arm.json").read_text())
        assert {key: written[key] for key in ("urdf", "base", "flange", "tcp")} == {
            "urdf": "b601-arm.urdf",
            "base": "base_link",
            "flange": "link6",
            "tcp": [0.135, 0.0, 0.0],
        }
        assert len(written["rest"]) == 6
        record = written[kinematics.TCP_CALIBRATION]
        assert sorted(record) == ["at", "digest", "status"] and record["status"] == "calibrated"
        assert datetime.datetime.fromisoformat(record["at"]).utcoffset() == datetime.timedelta(0)
        # The digest is of the files as written, so dream's check at policy() finds the tool point a person set.
        assert arm_file.tool_point_problem(written, (arms / "b601-arm.urdf").read_bytes()) is None
        assert kinematics.tcp_calibration(str(arms / "b601-arm.json")) == "calibrated"
        assert call({**upload, "tcp_calibration": "skipped"})[0] == 200
        assert kinematics.tcp_calibration(str(arms / "b601-arm.json")) == "skipped"
        assert "digest" not in json.loads((arms / "b601-arm.json").read_text())[kinematics.TCP_CALIBRATION]
        assert call({**upload, "tcp_calibration": "later"}) == (
            400,
            {"error": "'tcp_calibration' must be one of ['calibrated', 'skipped'], got 'later'"},
        )
        assert call({key: v for key, v in upload.items() if key != "tcp_calibration"}) == (
            400,
            {"error": "'tcp_calibration' is required"},
        )
        assert call({key: v for key, v in upload.items() if key != "fixed"}) == (
            400,
            {"error": "'fixed' must be a list of the chain's joints your robot holds still"},
        )
        status, refused = call({**upload, "flange": "link9"})
        assert status == 400 and "flange 'link9' is not a link of the URDF" in refused["error"]
        assert call({**upload, "tcp": [0.1, 0]}) == (400, {"error": "'tcp' must be three numbers [x, y, z]"})
        assert added == []
    finally:
        server.stop()


def test_dreamcli_tcp_answers_the_tool_point_page_alone_and_writes_the_wizards_arm_file(
    tmp_path: pathlib.Path, platform: FakePlatform, caplog: pytest.LogCaptureFixture
) -> None:
    """`dreamcli tcp` (TcpServer, the tool point tool on its own) answers the page, its scripts and #TCP_ROUTES, and
    404s every other step's route by name. For the same URDF, end and tool point, its Set tool center point writes
    the arm file and the URDF the wizard's URDF step writes, byte for byte, and prints the path."""
    specs = pytest.importorskip("dream_ik.specs", reason="dream's IK needs Python 3.11 or newer")
    caplog.set_level(logging.INFO, logger=setup_wizard.__name__)
    upload = {
        "filename": "B601 Arm.urdf",
        "urdf": (specs.ASSETS / "metal.urdf").read_text(),
        "base": "base_link",
        "flange": "link6",
        "tcp": [0.135, 0.01, -0.02],
        "approach": [1, 0, 0],
        "jaw": [0, 1, 0],
        "tcp_calibration": "calibrated",
        "fixed": [],
    }
    written: dict[str, tuple[str, bytes]] = {}
    for kind in (setup_wizard.SetupServer, setup_wizard.TcpServer):
        target = tmp_path / kind.__name__ / "station.yaml"
        server = kind(
            target,
            platform.account,
            host="127.0.0.1",
            port=0,
            ask_models=lambda: setup_fakes.MODELS,
            ask_embodiments=lambda: setup_fakes.EMBODIMENTS,
        )
        server.start()
        try:
            base = f"http://127.0.0.1:{server.port}"
            tool = json.loads(request(base, "GET", "/api/state")[2])["tool"]
            status, _, data = request(base, "POST", "/api/arm_file", upload)
            assert status == 200, data
            ik = pathlib.Path(json.loads(data)["ik"])
            assert ik == target.parent / setup_wizard.ARMS_DIR / "b601-arm.json"
            # Byte for byte but the second the file was written at.
            written[tool] = (
                re.sub(r'"at": "[^"]+"', '"at": "-"', ik.read_text()),
                ik.with_suffix(".urdf").read_bytes(),
            )
            if kind is setup_wizard.TcpServer:
                for path in ("/", *(f"/{script}" for script in setup_wizard.SCRIPTS)):
                    assert request(base, "GET", path)[0] == 200
                for method, path in (("GET", "/api/status"), ("GET", "/api/scan"), ("POST", "/api/robot")):
                    status, _, data = request(base, method, path, None if method == "GET" else {"body": "yam"})
                    refused = f"dreamcli tcp serves the tool point page only, not {method} {path}"
                    assert (status, json.loads(data)) == (404, {"error": refused})
        finally:
            server.stop()
    assert sorted(written) == ["tcp", "wizard"] and written["tcp"] == written["wizard"]
    arm = tmp_path / "TcpServer" / setup_wizard.ARMS_DIR / "b601-arm.json"
    assert '"at": "-"' in written["tcp"][0]
    station = tmp_path / "TcpServer" / "station.yaml"
    assert f"dreamcli tcp: wrote {arm} (tool point calibrated); set custom.ik in {station}" in caplog.text


def test_an_arm_and_a_camera_the_scan_misses_are_typed_then_used_like_found_ones(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    """Setup never blocks on how an arm or a camera is connected: every discovery step offers a way to point at one's
    own class. The scan finds the left arm's adapter only; the right arm's address is
    typed, then listed after a rescan too, opened through the robot's driver, told apart by moving it, and saved as
    its `can_channel`. A camera address OpenCV opens and a camera class are listed after the scanned cameras; a class
    is saved as the camera's `source` and read back as one."""
    with Station(tmp_path / "station.yaml", platform.account) as st:
        st.devices = [device("can0", "AAA111")]
        settings = st.call("POST", "/api/robot", {"body": "yam"})[1]["prefill"]["settings"]
        status, scan = st.call("POST", "/api/arms/typed", {"address": " can1 "})
        assert status == 200 and [d["id"] for d in scan["arms"]] == ["can-AAA111", "typed-can1"]
        typed = scan["arms"][1]
        assert (typed["kind"], typed["label"], typed["open"]) == ("typed", "can1", "can1")
        assert st.call("POST", "/api/arms/typed", {"address": ""}) == (400, {"error": "'address' is required"})
        assert [d["id"] for d in st.call("GET", "/api/scan")[1]["arms"]] == ["can-AAA111", "typed-can1"]
        status, opened = st.call(
            "POST", "/api/arms/open", {"devices": ["can-AAA111", "typed-can1"], "settings": settings}
        )
        assert status == 200 and opened["opened"] == {"can-AAA111": None, "typed-can1": None}
        st.hands.move("can1", 0, 0.3)
        wiggle = st.call("GET", "/api/status")[1]["wiggle"]
        assert wiggle["moved"] == "typed-can1" and wiggle["devices"]["typed-can1"]["moved"]
        assert st.call("POST", "/api/arms/assign", {"device": "typed-can1", "side": "right"})[0] == 200
        assert setup_wizard.saved_channel(typed, "right") == "can1"
        # The page's own "Add it": the new address is listed and ticked, the field cleared.
        shown = run_page(
            tmp_path,
            {
                "replies": {},
                "base": st.base,
                "show": ["devices", "typedArm", "openArms"],
                "set": {"scan": {"arms": [scan["arms"][0]]}},
                "values": {"typedArm": "192.168.1.21"},
                "call": "typeArm",
            },
        )
        assert shown["errors"] == [], shown
        assert shown["ids"]["devices"]["children"] == 3 and "192.168.1.21" in shown["ids"]["devices"]["text"]
        assert shown["ids"]["devices"]["ticked"] == 3 and shown["ids"]["typedArm"]["value"] == ""

        found = len(st.call("GET", "/api/scan")[1]["cameras"])
        st.call("POST", "/api/cameras/typed", {"address": "rtsp://192.168.1.50/stream1"})
        status, scan = st.call("POST", "/api/cameras/typed", {"address": "vendor_cams:Zed"})
        assert status == 200 and scan["cameras"][found:] == [
            {
                "source": "device",
                "value": "rtsp://192.168.1.50/stream1",
                "label": "rtsp://192.168.1.50/stream1",
                "bound": "typed",
            },
            {
                "source": "class",
                "value": "vendor_cams:Zed",
                "label": "Your camera class vendor_cams:Zed",
                "bound": "typed",
            },
        ]
    row = {"name": "top", "source": "class", "value": "vendor_cams:Zed", "width": 640, "height": 360, "fps": 30}
    entry = setup_wizard._camera_entry({**row, "rotate": 0})
    assert entry == {"name": "top", "source": "vendor_cams:Zed", "width": 640, "height": 360, "fps": 30, "rotate": 0}
    assert setup_cameras.camera_config({**row, "rotate": 0}).source == "vendor_cams:Zed"
    filled = setup_wizard.prefill({"body": "yam", "cameras": [entry]}, "yam", ("top",))["cameras"]
    assert filled == [{**row, "rotate": 0}]


## What an owner types over each blank of a hand-filled file, per body: the adapters as each body's config takes them.
OWNER_VALUES = {
    "REPLACE_OPEN_DEG": "60",
    "REPLACE_CLOSED_DEG": "5",
    "REPLACE_USERNAME/REPLACE_EMBODIMENT": "tester/arm",
    "REPLACE_ARM": "so101_faas",
}


@pytest.mark.parametrize("body", ["yam", "makers", "metal", "so101", "custom"])
def test_set_it_up_myself_gives_the_file_save_writes_with_each_measured_value_left_to_fill(
    tmp_path: pathlib.Path, body: str
) -> None:
    """Step 0's "Set it up myself": the station file template with no wizard check on the way. It is
    build_config()'s file, the one Save writes, so the loop reads it the same way; each value a step measures or picks
    is a REPLACE_ blank, which the loop refuses by name until it is filled. The home pose is a blank too, never
    zeros (a Metal arm at zero stands straight up), and one blank, not a list: how many angles it lists is how many
    joints the arm has, which the file cannot know for a robot of your own."""
    text = setup_wizard.template(body)
    blank = tmp_path / "blank.yaml"
    blank.write_text(text, encoding="utf-8")
    with pytest.raises(errors.ConfigError, match="REPLACE"):
        config.load(blank)
    raw = yaml.safe_load(text)
    assert raw["follower_l"]["home"] == "REPLACE_HOME_RAD" and raw["body"] == body
    assert set(raw) - {"embodiment", "custom"} == {
        "body",
        "follower_l",
        "follower_r",
        "cameras",
        *setup_wizard.LOOP_KEYS,
    }
    for side in ("left", "right"):
        so101_fakes.write_calibration(tmp_path / f"{side}.json")
    filled = re.sub(r"REPLACE_(\w+)_CAMERA", r"/dev/v4l/by-path/\1", text)
    for left, right in (("left", "LEFT"), ("right", "RIGHT")):
        channel = f"can_follower_{left[0]}" if body == "yam" else f"/dev/serial/by-id/usb-{left}-if00"
        filled = filled.replace(f"REPLACE_{right}_ARM_ADAPTER", channel)
        filled = filled.replace(f"REPLACE_{right}_CALIBRATION_JSON", f"{left}.json")
    home = [0.1] * 5 + [0.0 if body == "so101" else 0.1]  # an SO-101 has five joints, then 0
    filled = filled.replace("REPLACE_HOME_RAD", str(home))
    for value, typed in OWNER_VALUES.items():
        filled = filled.replace(value, typed)
    assert "REPLACE" not in filled.split("\n\n", 1)[1]
    blank.write_text(filled, encoding="utf-8")
    loaded = config.load(blank)
    assert loaded.body == body and [cam.name for cam in loaded.cameras] == [cam["name"] for cam in raw["cameras"]]
    assert loaded.follower_l.home == pytest.approx([0.1] * 5 + [0.0 if body == "so101" else 0.1])
    with pytest.raises(ValueError, match="no station file for 'nope'"):
        setup_wizard.template("nope")


def test_set_it_up_myself_shows_the_file_and_the_commands_and_hides_every_step(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    """The second way at step 0 shows the picked robot's file (GET /api/template), a dry run and the run of the loop
    the wizard ends in, and none of the wizard's inputs or steps."""
    with Station(tmp_path / "station.yaml", platform.account) as st:
        state, live = st.call("GET", "/api/state")[1], st.call("GET", "/api/status")[1]
        status, reply = st.call("GET", "/api/template?body=metal")
        assert status == 200 and reply == {"yaml": setup_wizard.template("metal")}
        assert st.call("GET", "/api/template?body=nope")[0] == 400
    shown = run_page(
        tmp_path,
        {
            "replies": {"GET /api/state": state, "GET /api/status": live, "GET /api/template": reply},
            "set": {"pick": "metal"},
            "call": "pickWay",
            "args": ["self"],
            "show": ["selfBox", "selfYaml", "selfDry", "selfRun", "selfPath", "pickActions", "custom", "waySelf"],
        },
    )
    ids = shown["ids"]
    assert shown["errors"] == [], shown
    assert not ids["selfBox"]["hidden"] and ids["pickActions"]["hidden"] and ids["custom"]["hidden"]
    assert ids["selfYaml"]["text"] == reply["yaml"] and ids["selfPath"]["text"] == state["config_path"]
    run_line = f'dreamcli robot run --config {state["config_path"]} --checkpoint "<your model>" --task "<what to do>"'
    assert ids["selfRun"]["text"] == run_line and ids["selfDry"]["text"] == run_line + " --dry-run"
    assert shown["requests"][-1] == {"key": "GET /api/template", "body": None}


def test_dreamcli_tcp_opens_the_page_on_the_urdf_step_alone_and_set_it_up_myself_names_it(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    """With `tool` `tcp` (TcpServer.state()) the page is the URDF step alone: its own title, the URDF box shown, and
    no poll of the station. Set it up myself names `dreamcli tcp` for a robot of your own only, whose `custom.ik` is
    the arm file it writes."""
    server = setup_wizard.TcpServer(tmp_path / "station.yaml", platform.account, host="127.0.0.1", port=0)
    state = server.state()
    assert state["tool"] == "tcp"
    shown = run_page(tmp_path, {"replies": {"GET /api/state": state}, "show": ["title", "custom", "urdfBox"]})
    ids = shown["ids"]
    assert shown["errors"] == [], shown
    assert ids["title"]["text"] == "Your arm's tool point"
    assert not ids["custom"]["hidden"] and not ids["urdfBox"]["hidden"]
    assert [request["key"] for request in shown["requests"]] == ["GET /api/state"]
    with Station(tmp_path / "station.yaml", platform.account) as st:
        state, live = st.call("GET", "/api/state")[1], st.call("GET", "/api/status")[1]
        assert state["tool"] == "wizard"
        templates = {body: st.call("GET", f"/api/template?body={body}")[1] for body in ("metal", "custom")}
    for body, reply in templates.items():
        shown = run_page(
            tmp_path,
            {
                "replies": {"GET /api/state": state, "GET /api/status": live, "GET /api/template": reply},
                "set": {"pick": body},
                "call": "pickWay",
                "args": ["self"],
                "show": ["selfTcp"],
            },
        )
        assert shown["errors"] == [], shown
        assert shown["ids"]["selfTcp"]["hidden"] == (body != "custom")
    assert "`dreamcli tcp`" in templates["custom"]["yaml"]


## Shaped as Dexmate's Vega-1 with dexterous hands from its base to the left hand: a three-joint torso, the arm's seven
## joints on it, and the hand on a mount that turns without end (`continuous`, no limits), as its URDF writes it.
VEGA_LEFT = (
    '<robot name="vega"><link name="base"/>'
    + "".join(
        f'<link name="{child}"/>'
        f'<joint name="{joint}" type="{"continuous" if joint == "L_hand_mount_roll" else "revolute"}">'
        f'<parent link="{parent}"/><child link="{child}"/><origin xyz="0 0 0.1"/><axis xyz="0 {i % 2} {1 - i % 2}"/>'
        + ("" if joint == "L_hand_mount_roll" else '<limit lower="-1.5" upper="1.5" velocity="2"/>')
        + "</joint>"
        for i, (joint, parent, child) in enumerate(
            [("torso_j1", "base", "torso_1"), ("torso_j2", "torso_1", "torso_2"), ("torso_j3", "torso_2", "torso_3")]
            + [(f"L_arm_j{n}", "torso_3" if n == 1 else f"L_arm_{n - 1}", f"L_arm_{n}") for n in range(1, 8)]
            + [("L_hand_mount_roll", "L_arm_7", "L_hand")]
        )
    )
    + "</robot>"
)


## Each page script's words on the URDF end step for VEGA_LEFT: the chain it shows, what the map does not drive, and
## the start of the line once the arm is built. Both pages ask the same per joint.
PAGE_WORDS = {
    "setup_wizard.js": {
        "ends": "Your arm stands on base (blue) and ends at L_hand (green), 11 joints out. Not the end? Click the "
        "right link on the drawing.",
        "undriven": "tester/vega does not drive 4 of them; answer for each below.",
        "built": "Using",
    },
    "setup_wizard_v2.js": {
        "ends": "Your arm is mounted on base (blue) and ends at L_hand (green), 11 joints out. If that isn't the end, "
        "click the right link in the drawing.",
        "undriven": "Your embodiment tester/vega doesn't drive 4 of them. Answer for each below.",
        "built": "Added",
    },
}


@pytest.mark.parametrize("script", sorted(PAGE_WORDS))
def test_dreamcli_tcp_names_each_chain_joint_the_stations_embodiment_does_not_drive_and_a_person_decides_it(
    tmp_path: pathlib.Path, platform: FakePlatform, script: str
) -> None:
    """`dreamcli tcp` on a Vega-1 refused its hand mount, a continuous joint its embodiment does not drive
    (2026-10-18). The page now names each moving joint between base and end that the station file's embodiment
    (`custom.embodiment`, its `urdf.joints`) does not drive, and holds Yes until a person presses Moves or Held still
    on each: never the page (the TCP rule). The arm file records the held ones as `fixed`, and the arm is built
    without them. The wizard asks about none (its arm drives its whole chain); with no joint map, or none to read,
    `dreamcli tcp` asks about every one
    (test_dreamcli_tcp_asks_about_every_chain_joint_when_it_has_no_joint_map_and_says_why_a_read_failed)."""
    pytest.importorskip("dream_ik.specs", reason="dream's IK needs Python 3.11 or newer")
    station = tmp_path / "station.yaml"
    station.write_text("body: custom\ncustom:\n  embodiment: tester/vega\n", encoding="utf-8")
    arm = [f"L_arm_j{n}" for n in range(1, 8)]
    right = [f"R_arm_j{n}" for n in range(1, 8)]
    vega = setup_fakes.account_embodiment(
        "tester/vega", {**setup_fakes.EMBODIMENT, "urdf": {"joints": {"left_arm": arm, "right_arm": right}}}
    )
    kwargs = {"host": "127.0.0.1", "port": 0, "ask_embodiments": lambda: [vega, *setup_fakes.EMBODIMENTS]}
    keyless = dataclasses.replace(platform.account, api_key=None)
    assert setup_wizard.SetupServer(station, platform.account, **kwargs).joint_map() is None
    assert setup_wizard.TcpServer(station, keyless, **kwargs).joint_map() == {
        "embodiment": "tester/vega",
        "joints": None,
        "error": "could not read tester/vega's joints: your credentials hold no API key",
    }
    unnamed = setup_wizard.TcpServer(tmp_path / "none.yaml", platform.account, **kwargs).joint_map()
    assert unnamed == {"embodiment": None, "joints": None, "error": None}
    server = setup_wizard.TcpServer(station, platform.account, **kwargs)
    server.start()
    try:
        base = f"http://127.0.0.1:{server.port}"
        tree = json.loads(request(base, "POST", "/api/urdf", {"urdf": VEGA_LEFT})[2])
        assert tree["joint_map"] == {"embodiment": "tester/vega", "joints": sorted(arm + right), "error": None}
        show = ["urdfLinks", "urdfYes", "urdfSpare", "urdfNote"]

        def page(after: list[Any]) -> dict[str, Any]:
            scenario = {
                "script": script,
                "replies": {},
                "base": base,
                "show": show,
                "call": "loadUrdf",
                "args": ["Vega Left.urdf", VEGA_LEFT],
            }
            shown = run_page(tmp_path, {**scenario, "after": after})
            assert shown["errors"] == [], shown
            ids: dict[str, Any] = shown["ids"]
            return ids

        def pick(joint: str, held: bool) -> dict[str, list[Any]]:
            return {"press": ["urdfSpare", f'button[data-{"held" if held else "moves"}="{joint}"]', 0]}

        spare = ["torso_j1", "torso_j2", "torso_j3", "L_hand_mount_roll"]
        asked = page([])
        words = PAGE_WORDS[script]
        assert asked["urdfLinks"]["text"] == f"{words['ends']} {words['undriven']}"
        assert asked["urdfYes"]["disabled"] and not asked["urdfSpare"]["hidden"]
        assert asked["urdfSpare"]["children"] == 4
        assert asked["urdfSpare"]["text"].startswith("torso_j1: does your robot move this joint? MovesHeld still")
        assert page([pick(joint, True) for joint in spare[:3]])["urdfYes"]["disabled"], "the mount is still undecided"
        written = tmp_path / setup_wizard.ARMS_DIR / "vega-left.json"
        for mount_held, joints in ((True, arm), (False, [*arm, "L_hand_mount_roll"])):
            picks = [pick(joint, True) for joint in spare[:3]] + [pick("L_hand_mount_roll", mount_held)]
            assert not page(picks)["urdfYes"]["disabled"], "every joint is decided"
            decided = page([*picks, "confirmEnd", ["addUrdf", "calibrated"]])
            still = spare if mount_held else spare[:3]
            assert decided["urdfNote"]["text"] == (
                f"{words['built']} Vega Left.urdf: {len(joints)} joints ({', '.join(joints)}). Held still: "
                f"{', '.join(still)}."
            )
            assert json.loads(written.read_text())["fixed"] == still
            assert kinematics.arm_layout(str(written)).width == len(joints)
    finally:
        server.stop()


def _refuse_embodiments() -> list[Any]:
    raise dream.DreamError("401 Unauthorized: the key was revoked")


## A station file naming the Vega-1 embodiment.
NAMES_VEGA = "body: custom\ncustom:\n  embodiment: tester/vega\n"


@pytest.mark.parametrize(
    ("station_text", "ask", "error"),
    [
        (NAMES_VEGA, lambda: [setup_fakes.account_embodiment("tester/vega")], None),
        ("body: custom\n", lambda: setup_fakes.EMBODIMENTS, None),
        (
            NAMES_VEGA,
            _refuse_embodiments,
            "could not list your account's embodiments: 401 Unauthorized: the key was revoked",
        ),
        (
            NAMES_VEGA,
            lambda: setup_fakes.EMBODIMENTS,
            "your account has no embodiment 'tester/vega'; it has: tester/so101-pick, tester/so101, tester/my-arm, "
            "tester/mine, tester/my-metal, tester/cams",
        ),
    ],
    ids=["no_urdf_joints", "no_embodiment", "read_failed", "no_such_embodiment"],
)
@pytest.mark.parametrize("script", sorted(PAGE_WORDS))
def test_dreamcli_tcp_asks_about_every_chain_joint_when_it_has_no_joint_map_and_says_why_a_read_failed(
    tmp_path: pathlib.Path, platform: FakePlatform, station_text: str, ask: Any, error: str | None, script: str
) -> None:
    """2026-10-18: with no joint map to hold the chain to (no embodiment, or one without `urdf.joints`), the
    page asks about every moving joint between base and end; when the account could not be asked, it says why in one
    line first, never falling back silently. Nothing is pressed for the person, and held ones go in `fixed`."""
    pytest.importorskip("dream_ik.specs", reason="dream's IK needs Python 3.11 or newer")
    station = tmp_path / "station.yaml"
    station.write_text(station_text, encoding="utf-8")
    server = setup_wizard.TcpServer(station, platform.account, host="127.0.0.1", port=0, ask_embodiments=ask)
    server.start()
    try:
        base = f"http://127.0.0.1:{server.port}"
        tree = json.loads(request(base, "POST", "/api/urdf", {"urdf": VEGA_LEFT})[2])
        error = None if error is None else f"could not read tester/vega's joints: {error}"
        assert (tree["joint_map"]["joints"], tree["joint_map"]["error"]) == (None, error)
        chain = ["torso_j1", "torso_j2", "torso_j3", *[f"L_arm_j{n}" for n in range(1, 8)], "L_hand_mount_roll"]
        held = [*chain[:3], chain[10]]
        scenario = {
            "script": script,
            "replies": {},
            "base": base,
            "call": "loadUrdf",
            "args": ["Vega Left.urdf", VEGA_LEFT],
        }
        show = ["urdfLinks", "urdfYes", "urdfSpare", "urdfNote"]
        asked = run_page(tmp_path, {**scenario, "show": show, "after": []})
        assert asked["errors"] == [], asked
        spare = asked["ids"]["urdfSpare"]
        assert asked["ids"]["urdfLinks"]["text"] == f"{PAGE_WORDS[script]['ends']} Answer for each joint below."
        assert spare["children"] == len(chain) + (error is not None) and not spare["hidden"]
        assert spare["text"].startswith((error or "") + "torso_j1: does your robot move this joint? MovesHeld still")
        assert asked["ids"]["urdfYes"]["disabled"]
        picks = [
            {"press": ["urdfSpare", f'button[data-{"held" if joint in held else "moves"}="{joint}"]', 0]}
            for joint in chain
        ]
        decided = run_page(
            tmp_path, {**scenario, "show": show, "after": [*picks, "confirmEnd", ["addUrdf", "calibrated"]]}
        )
        assert decided["errors"] == [], decided
        assert not decided["ids"]["urdfYes"]["disabled"], "every joint is decided"
        assert json.loads((tmp_path / setup_wizard.ARMS_DIR / "vega-left.json").read_text())["fixed"] == held
    finally:
        server.stop()


## A gripper of two fingers on dream's Metal URDF's link6, so the end of its longest chain is a finger.
FINGERS = """  <link name="finger_l"/><link name="finger_r"/>
  <joint name="finger_l_joint" type="prismatic"><parent link="link6"/><child link="finger_l"/>
    <origin xyz="0.1 0.02 0"/><axis xyz="0 1 0"/><limit lower="0" upper="0.03" velocity="1"/></joint>
  <joint name="finger_r_joint" type="prismatic"><parent link="link6"/><child link="finger_r"/>
    <origin xyz="0.1 -0.02 0"/><axis xyz="0 -1 0"/><limit lower="0" upper="0.03" velocity="1"/></joint>
</robot>"""

## A made-up arm one joint longer than the loop drives: a chain of revolute joints from base_link to l17.
LONG_ARM = (
    '<robot name="long"><link name="base_link"/>'
    + "".join(
        f'<link name="l{i}"/><joint name="j{i}" type="revolute">'
        f'<parent link="{"base_link" if i == 1 else f"l{i - 1}"}"/><child link="l{i}"/>'
        '<origin xyz="0 0 0.05"/><axis xyz="0 0 1"/><limit lower="-2" upper="2" velocity="3"/>'
        "</joint>"
        for i in range(1, followers.MAX_JOINTS + 2)
    )
    + "</robot>"
)


def test_a_urdf_takes_its_file_alone_then_the_end_and_the_tool_from_buttons_on_the_drawing(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    """The URDF step asks for the file and nothing typed: the rest is given on a 3D view of the arm, not as numbers
    the operator has to know. The page's script runs on the real routes. The guessed end is link6, the fingers
    sliding past it being the gripper, so the operator only confirms it; the marker starts at link6's origin. An end
    past the most joints the loop drives (16) is named and Yes held until the operator clicks another link. The
    tool's six numbers, in link6's frame, each take a minus and a plus button by the step picked, or a typed value.
    The arm file's tool is the marker: dream's model of the arm puts its tool at rest on link6, moved and turned by
    those numbers."""
    specs = pytest.importorskip("dream_ik.specs", reason="dream's IK needs Python 3.11 or newer")
    urdf = (specs.ASSETS / "metal.urdf").read_text().replace("</robot>", FINGERS)
    tree = setup_urdf.read(urdf)
    link6 = tree["links"].index("link6")
    show = ["urdfAsk", "urdfLinks", "urdfYes", "urdfNote"]
    with Station(tmp_path / "station.yaml", platform.account) as st:

        def page(scenario: dict[str, Any]) -> dict[str, Any]:
            base = {
                "replies": {},
                "base": st.base,
                "show": show,
                "call": "loadUrdf",
                "args": ["Metal Grip.urdf", urdf],
            }
            shown = run_page(tmp_path, {**base, **scenario})
            assert shown["errors"] == [], shown
            ids: dict[str, Any] = shown["ids"]
            return ids

        guessed = page({})
        assert guessed["urdfAsk"]["text"] == "Is this the end of your arm?"
        assert guessed["urdfLinks"]["text"] == (
            "Your arm stands on base_link (blue) and ends at link6 (green), 6 joints out. Not the end? Click the "
            "right link on the drawing."
        )
        assert not guessed["urdfYes"]["disabled"]
        too_long = page({"args": ["Long.urdf", LONG_ARM]})
        assert too_long["urdfLinks"]["text"] == (
            "l17 is 17 joints out from base_link; an arm has at most 16. Click the link your gripper is mounted on."
        )
        assert too_long["urdfYes"]["disabled"]
        # An end with no joint between it and the base was once confirmed, and building the arm failed only after
        # the tool point was set ("the chain from ... has no moving joint").
        still = page({"after": [["pickLink", tree["base"]]]})
        assert still["urdfLinks"]["text"] == (
            'No joint moves between base_link and base_link. Click the link your gripper is mounted on, or "It '
            'stands on another link" if base_link is not what your arm stands on.'
        )
        assert still["urdfYes"]["disabled"]
        assert "!links || !joints ||" in setup_wizard._package_text("setup_wizard_v2.js")

        def tool(steps: list[dict[str, list[Any]]]) -> tuple[dict[str, Any], Any]:
            after = ["confirmEnd", *steps, ["addUrdf", "calibrated"]]
            shown = page({"after": after})
            assert shown["urdfNote"]["text"].startswith("Using Metal Grip.urdf: 6 joints")
            arm = tmp_path / setup_wizard.ARMS_DIR / "metal-grip.json"
            model = setup_model.ArmModel(str(arm))
            return json.loads(arm.read_text()), model.tool(model.rest)

        def press(selector: str) -> dict[str, list[Any]]:
            return {"press": ["urdfPad", f"button[{selector}]", 0]}

        def typed(dim: str, value: str) -> dict[str, list[Any]]:
            return {"type": ["urdfPad", f'input[data-dim="{dim}"]', 0, value]}

        w, x, y, z = np.asarray(tree["poses"][link6][:4]) / np.linalg.norm(tree["poses"][link6][:4])
        flange = np.eye(4)
        flange[:3, :3] = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
        flange[:3, 3] = tree["poses"][link6][4:]
        plain, plain_tool = tool([])
        assert plain[kinematics.TCP_CALIBRATION]["status"] == "calibrated"
        assert plain["tcp"] == [0, 0, 0] and plain_tool[:3, 3] == pytest.approx(flange[:3, 3], abs=1e-5)
        # x typed, then two 1 cm steps (the default); a 1 mm step down z; roll, pitch and yaw typed, yaw 270 shown as
        # -90; roll a 5 degree step up, 185 shown as -175.
        moved, moved_tool = tool(
            [typed("x", "-9.7"), press('data-nudge="x+"'), press('data-nudge="x+"')]
            + [press('data-move-step="1"'), press('data-nudge="z-"')]
            + [typed("roll", "180"), typed("pitch", "0"), typed("yaw", "270")]
            + [press('data-turn-step="5"'), press('data-nudge="roll+"')]
        )
        r, c, s = np.radians(-175), np.cos(np.radians(-90)), np.sin(np.radians(-90))
        turn = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.array(
            [[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]]
        )
        assert moved["tcp"] == pytest.approx([0.0103, 0, -0.001], abs=1e-9)
        assert moved["approach"] == pytest.approx(turn[:, 2], abs=1e-6)
        assert moved["jaw"] == pytest.approx(turn[:, 0], abs=1e-6)
        expected = np.eye(4)
        expected[:3, :3], expected[:3, 3] = turn, moved["tcp"]
        assert moved_tool == pytest.approx(flange @ expected, abs=1e-5)

        # The YAM beside it: its marker rides its tool link, which is dream's tool point at the pose drawn.
        yam = st.call("GET", "/api/reference_arm")[1]
        model = setup_model.ArmModel(setup_view.REFERENCE_ARM)
        pose = yam["frame"][yam["tool_link"]]
        assert pose[4:] == pytest.approx(model.tool(model.rest)[:3, 3].tolist(), abs=1e-4)

        broken = run_page(
            tmp_path, {"replies": {}, "base": st.base, "show": show, "call": "loadUrdf", "args": ["x", "<robot"]}
        )
        assert broken["errors"] == [
            "pickErr: this file is not a URDF: it does not parse as XML (unclosed token: line 1, column 0)"
        ]


def test_continue_waits_until_the_tool_point_is_marked_or_its_skip_confirmed_and_a_change_undoes_either(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    """Skipping the tool point is allowed, not advised, and must be a deliberate choice. With the driver,
    the embodiment and the URDF's end given, Continue stays off. Skip calibration (not advised) only opens a warning,
    whose Skip anyway stays off until "I understand" is ticked; then the arm file records `skipped` and Continue is on.
    Set tool center point records `calibrated`. A new end afterwards turns Continue off again, its file no longer the
    arm."""
    specs = pytest.importorskip("dream_ik.specs", reason="dream's IK needs Python 3.11 or newer")
    urdf = (specs.ASSETS / "metal.urdf").read_text()
    link5 = setup_urdf.read(urdf)["links"].index("link5")
    arm = tmp_path / setup_wizard.ARMS_DIR / "metal.json"
    show = ["pickRobot", "tcpSkip", "tcpSkipGo", "urdfNote"]
    with Station(tmp_path / "station.yaml", platform.account) as st:

        def page(after: list[Any]) -> tuple[dict[str, Any], list[str]]:
            shown = run_page(
                tmp_path,
                {
                    "replies": {
                        "GET /api/embodiments": {"embodiments": [{"name": "tester/mine", "cameras": ["scene"]}]}
                    },
                    "base": st.base,
                    "set": {"pick": "custom", "driver": "mine:Open", "armWay": "urdf"},
                    "values": {"embodiment": "tester/mine"},
                    "call": "loadUrdf",
                    "args": ["Metal.urdf", urdf],
                    "after": ["confirmEnd", *after],
                    "show": show,
                },
            )
            assert shown["errors"] == [], shown
            return shown["ids"], [r["key"] for r in shown["requests"] if r["key"] == "POST /api/arm_file"]

        ids, wrote = page([])
        assert ids["pickRobot"]["blocked"] and not ids["tcpSkip"]["open"] and wrote == []
        ids, wrote = page([{"click": "urdfSkip"}])
        assert ids["tcpSkip"]["open"] and ids["tcpSkipGo"]["disabled"] and ids["pickRobot"]["blocked"] and wrote == []
        ids, wrote = page([{"click": "urdfSkip"}, {"click": "tcpSkipBack"}])
        assert not ids["tcpSkip"]["open"] and ids["pickRobot"]["blocked"] and wrote == []
        ids, wrote = page([{"click": "urdfSkip"}, {"tick": "tcpSkipTick"}, {"click": "tcpSkipGo"}])
        assert not ids["tcpSkip"]["open"] and not ids["pickRobot"]["blocked"] and wrote == ["POST /api/arm_file"]
        assert ids["urdfNote"]["text"].endswith("Tool point skipped: every start warns.")
        assert kinematics.tcp_calibration(str(arm)) == "skipped"
        ids, wrote = page([{"click": "urdfAdd"}])
        assert not ids["pickRobot"]["blocked"] and ids["urdfNote"]["text"].startswith("Using Metal.urdf: 6 joints")
        assert "skipped" not in ids["urdfNote"]["text"] and kinematics.tcp_calibration(str(arm)) == "calibrated"
        ids, _ = page([{"click": "urdfAdd"}, ["pickLink", link5]])
        assert ids["pickRobot"]["blocked"] and ids["urdfNote"]["text"] == ""


def test_a_gray_continue_names_what_it_still_needs_and_set_tool_center_point_turns_it_on(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    """With no driver picked, Set tool center point leaves Continue blocked and the line beside it names the
    driver; before the tool point it names that; with the driver and the embodiment given, Set tool center point
    unblocks Continue and the line is empty. A click on the blocked Continue sends nothing and lines the driver's
    block in red, and picking the driver takes the line away."""
    specs = pytest.importorskip("dream_ik.specs", reason="dream's IK needs Python 3.11 or newer")
    urdf = (specs.ASSETS / "metal.urdf").read_text()
    with Station(tmp_path / "station.yaml", platform.account) as st:

        def page(driver: str, after: list[Any]) -> tuple[dict[str, Any], list[str]]:
            shown = run_page(
                tmp_path,
                {
                    "replies": {
                        "GET /api/embodiments": {"embodiments": [{"name": "tester/mine", "cameras": ["scene"]}]}
                    },
                    "base": st.base,
                    "set": {"pick": "custom", "driver": driver, "armWay": "urdf"},
                    "values": {"embodiment": "tester/mine"},
                    "call": "loadUrdf",
                    "args": ["Metal.urdf", urdf],
                    "after": ["confirmEnd", *after],
                    "show": ["pickRobot", "pickNeed", "driverBox"],
                },
            )
            assert shown["errors"] == [], shown
            return shown["ids"], [r["key"] for r in shown["requests"]]

        ids, _ = page("", [{"click": "urdfAdd"}])
        assert ids["pickRobot"]["blocked"]
        assert ids["pickNeed"]["text"] == "Still needed: your driver."
        ids, _ = page("mine:Open", [])
        assert ids["pickRobot"]["blocked"]
        assert (
            ids["pickNeed"]["text"]
            == "Still needed: your arm's tool center point (Set tool center point, or skip it)."
        )
        ids, _ = page("mine:Open", [{"click": "urdfAdd"}])
        assert not ids["pickRobot"]["blocked"] and ids["pickNeed"]["text"] == ""
        ids, sent = page("", [{"click": "urdfAdd"}, {"click": "pickRobot"}])
        assert ids["driverBox"]["missing"] and ids["pickRobot"]["blocked"] and "POST /api/robot" not in sent
        ids, _ = page("", [{"click": "urdfAdd"}, {"click": "pickRobot"}, ["pickDriver", "mine:Open"]])
        assert not ids["driverBox"]["missing"] and not ids["pickRobot"]["blocked"]


@pytest.mark.parametrize(
    ("field", "given", "call"),
    [("urdfFile", "robotis-omx-urdf.zip", "readUrdf"), ("urdfFolder", "robotis-omx-urdf", "readUrdfFolder")],
)
def test_a_zip_of_several_urdfs_offers_each_to_pick_and_draws_the_one_picked(
    tmp_path: pathlib.Path, platform: FakePlatform, field: str, given: str, call: str
) -> None:
    """A zip of two URDFs is a choice, not a refusal. Each URDF is a choice, by its path in the zip, with its links
    and moving joints counted; nothing is drawn until one is picked, then the picked URDF is, and another pick draws
    that one. A zip of one URDF draws it with no pick; one of none says so. The URDF's folder, picked, is read as its
    zip is, with the same choices and lines."""
    specs = pytest.importorskip("dream_ik.specs", reason="dream's IK needs Python 3.11 or newer")
    one = (specs.ASSETS / "metal.urdf").read_text()
    two = one.replace("</robot>", FINGERS)
    zipped = {
        "omx/urdf/omx_f/omx_f.urdf": one,
        "omx/urdf/omx_f_bi/omx_f_bi.urdf": two,
        "omx/meshes/link1.stl": "solid",
    }
    show = ["urdfPick", "urdfPickAsk", "urdfPickList", "urdfEnd", "urdfLinks"]
    with Station(tmp_path / "station.yaml", platform.account) as st:

        def page(files: dict[str, str], after: list[Any]) -> tuple[dict[str, Any], list[Any]]:
            shown = run_page(
                tmp_path,
                {
                    "replies": {},
                    "base": st.base,
                    "files": {field: given},
                    "zip": files,
                    "call": call,
                    "after": after,
                    "show": show,
                },
            )
            drawn = [r["body"]["urdf"] for r in shown["requests"] if r["key"] == "POST /api/urdf"]
            return shown, drawn

        shown, drawn = page(zipped, [])
        ids = shown["ids"]
        assert shown["errors"] == [] and drawn == [] and ids["urdfEnd"]["hidden"] and not ids["urdfPick"]["hidden"]
        assert ids["urdfPickAsk"]["text"] == f"{given} holds 2 URDFs. Pick your arm's:"
        assert ids["urdfPickList"]["text"] == (
            "omx/urdf/omx_f/omx_f.urdf7 links, 6 moving jointsomx/urdf/omx_f_bi/omx_f_bi.urdf9 links, 8 moving joints"
        )
        pick = [{"type": ["urdfPickList", "input", n, "on"]} for n in (1, 0)]
        shown, drawn = page(zipped, pick[:1])
        assert shown["errors"] == [] and drawn == [two] and not shown["ids"]["urdfEnd"]["hidden"]
        assert shown["ids"]["urdfLinks"]["text"].startswith("Your arm stands on base_link (blue) and ends at link6")
        shown, drawn = page(zipped, pick)
        assert shown["errors"] == [] and drawn == [two, one]
        assert shown["ids"]["urdfLinks"]["text"].startswith("Your arm stands on base_link (blue) and ends at link6")
        shown, drawn = page({"omx/urdf/omx_f.urdf": one}, [])
        assert shown["errors"] == [] and drawn == [one] and shown["ids"]["urdfPick"]["hidden"]
        shown, drawn = page({"omx/meshes/link1.stl": "solid"}, [])
        assert drawn == [] and shown["errors"] == [
            f"pickErr: {given} holds no .urdf file. Give your arm's URDF as a .zip with its meshes, or its folder."
        ]


def test_a_new_embodiment_is_registered_at_the_pick_from_the_arm_with_the_cameras_the_camera_step_names(
    tmp_path: pathlib.Path, platform: FakePlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no embodiment and no fine-tune on the account yet, the pick registers one from the operator's own steps,
    not a fine-tune's: the arm's joints,
    and the cameras the camera step's role buttons name, scene, wrist_l and wrist_r (the wrists by the names the base
    models read), with nothing typed. It is named after the arm and reused on a second pick; a name the account has
    for another layout is refused at the pick, by name. An SO-101 takes the same."""
    account = setup_fakes.Account(monkeypatch, [setup_fakes.account_embodiment("tester/makers")])
    with Station(tmp_path / "station.yaml", platform.account) as st:
        st.server._ask_embodiments = lambda: list(account.embodiments)
        st.server._add_embodiment = account.add

        def pick(arm: str) -> tuple[int, Any]:
            return st.call(
                "POST", "/api/robot", {"custom": {"ik_arm": arm, "driver": "mine:Open", "embodiment": None}}
            )

        status, picked = pick("metal")
        assert status == 200 and picked["robot"]["roles"] == ["scene", "wrist_l", "wrist_r"]
        assert (picked["robot"]["custom"]["embodiment"], picked["robot"]["embodiment_id"]) == (
            "tester/metal",
            "emb_tester_metal",
        )
        assert [cam["label"] for cam in picked["cameras"]] == [
            "Head camera (scene)",
            "Left wrist camera (wrist_l)",
            "Right wrist camera (wrist_r)",
        ]
        assert account.added == [("metal", zero_shot_embodiment(6))]
        assert picked["embodiment_note"] == "Registered a new embodiment, tester/metal, on your account."
        status, again = pick("metal")
        assert status == 200 and len(account.added) == 1
        # A second "new" pick lands on the first one, and the reply says so rather than leaving it to a guess.
        assert again["embodiment_note"].startswith("Your account already has tester/metal")
        assert pick("makers")[1]["error"].startswith(
            "your account already has an embodiment tester/makers of another layout "
            "(cameras scene, wrist_l, wrist_r): pick it in the list to use it"
        )
        assert len(account.added) == 1
        # An arm from its URDF is named after its file, not the key dream registers the arm file under.
        specs = pytest.importorskip("dream_ik.specs", reason="dream's IK needs Python 3.11 or newer")
        urdf = (specs.ASSETS / "metal.urdf").read_text()
        upload = {"filename": "B601 Arm.urdf", "urdf": urdf, "base": "base_link", "flange": "link6"}
        status, made = st.call(
            "POST",
            "/api/arm_file",
            {
                **upload,
                "tcp": [0.1, 0, 0],
                "approach": [1, 0, 0],
                "jaw": [0, 1, 0],
                "tcp_calibration": "calibrated",
                "fixed": [],
            },
        )
        assert status == 200
        status, picked = pick(made["ik"])
        assert status == 200 and picked["robot"]["custom"]["embodiment"] == "tester/b601-arm"
        assert account.added[-1] == ("b601-arm", zero_shot_embodiment(6))
        if "so101_faas" in setup_model.spec_names():
            calibration = so101_fakes.write_calibration(tmp_path / "arm.json")
            body = {"body": "so101", "calibration": [str(calibration)], "embodiment": None}
            status, picked = st.call("POST", "/api/robot", body)
            assert status == 200 and picked["robot"]["embodiment"] == "tester/so101-faas"
            # A beta robot's arm takes its layout (`robots.beta_robots`): the XLeRobot's, each arm as wide as the
            # station drives it (`kinematics.arm_layout`, five joints in six) with its gripper drive after it.
            xlerobot = beta_robots.embodiment_config(beta_robots.robots()["so101"])
            assert account.added[-1] == ("so101-faas", xlerobot)
            assert xlerobot["state"]["left_gripper"] == [6, 7] and xlerobot["head_camera"] == "head"


@pytest.mark.parametrize(
    "robot", [r for r in beta_robots.robots().values() if r.body == "custom"], ids=lambda robot: robot.id
)
def test_new_on_a_beta_testers_arm_registers_the_layout_the_tester_trained_on(
    tmp_path: pathlib.Path, platform: FakePlatform, monkeypatch: pytest.MonkeyPatch, robot: beta_robots.BetaRobot
) -> None:
    """A robot of your own on one of the beta robots' arms registers, at the pick, the body the
    robot's fine-tunes train on (`shared.tests.datasets.beta_bodies`), and its cameras take that body's roles."""
    if robot.ik not in setup_model.spec_names():
        pytest.skip(f"this dream IK has no {robot.ik}")
    account = setup_fakes.Account(monkeypatch, [])
    with Station(tmp_path / "station.yaml", platform.account) as st:
        st.server._ask_embodiments = lambda: list(account.embodiments)
        st.server._add_embodiment = account.add
        custom = {"ik_arm": robot.ik, "driver": "mine:Open", "embodiment": None}
        status, picked = st.call("POST", "/api/robot", {"custom": custom})
        assert status == 200, picked
        assert picked["robot"]["roles"] == list(robot.cameras)
        # Setting the robot up again reuses it (a second Unitree R1 setup registered a `-gripper` layout beside it).
        status, again = st.call("POST", "/api/robot", {"custom": custom})
        assert status == 200 and again["robot"]["custom"] == picked["robot"]["custom"], again
    assert account.added == [(robot.ik.replace("_", "-"), beta_bodies.WORKING[robot.id]["body"])]


def test_an_so101s_new_embodiment_carries_each_grippers_drive_so_dreams_ik_starts_on_it(
    tmp_path: pathlib.Path, platform: FakePlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    """New on an SO-101 registered the MakerMods parts without `<side>_gripper`, but dream's
    `so101_faas` arm has a gripper drive of its own (LeRobot's 0 to 100, `ArmSpec.jaw`) that the dream client makes
    from the grip under that part, so the client's IK refused the embodiment ("must name one gripper part") and the
    gripper never reached the model. The registered embodiment names it: a base model reads the grip as the gripper,
    and a fine-tune of the embodiment the gripper in LeRobot's units, as the arm's datasets record it."""
    if "so101_faas" not in setup_model.spec_names():
        pytest.skip("this dream IK has no so101_faas")
    # The account has the one New registered before the grippers: the layout with them goes beside it, not refused.
    old = setup_fakes.account_embodiment("tester/so101-faas", zero_shot_embodiment(6))
    account = setup_fakes.Account(monkeypatch, [old])
    with Station(tmp_path / "station.yaml", platform.account) as st:
        st.server._ask_embodiments = lambda: list(account.embodiments)
        st.server._add_embodiment = account.add
        calibration = so101_fakes.write_calibration(tmp_path / "arm.json")
        body = {"body": "so101", "calibration": [str(calibration)], "embodiment": None}
        status, picked = st.call("POST", "/api/robot", body)
        assert status == 200 and picked["robot"]["embodiment"] == "tester/so101-faas-gripper"
    registered = account.added[-1][1]
    embodiments.check_registration(registered)
    wire = EefWire(registered, None, "so101", kinematics.arm_layout("so101_faas"))
    roles = tuple(registered["video"])  # the XLeRobot tester's cameras (robots.beta_robots)
    observation = test_types_adapter.make_observation(camera_names=roles)
    created = datetime.datetime(2026, 10, 16, tzinfo=datetime.timezone.utc)
    base = Model(
        "dreamlabs/dream-1", "public", "dreamlabs/dream-1", "yam", True, created, ["overhead", "wrist_l", "wrist_r"]
    )
    fine_tune = dataclasses.replace(
        base, id="dream-1-ft:so101", embodiment="emb_tester_so101-faas", cameras=list(roles)
    )
    for model, scale in ((base, 1.0), (fine_tune, 100.0)):
        ik = client_ik.ClientIK({"arm": "so101_faas", "head": "eef"}, 30.0, registered, model)
        sent = ik.observe(test_types_adapter.sent(wire.to_dream(observation, "pick")))
        for side, arm in (("right", observation.state.right), ("left", observation.state.left)):
            np.testing.assert_allclose(sent[f"state.{side}_gripper"], [arm.gripper * scale])


def test_new_says_when_it_reuses_the_embodiment_named_after_the_arm(
    tmp_path: pathlib.Path, platform: FakePlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On an SO-101, New reuses the account's embodiment of the arm's name when there is one, and the summary shows
    it, not a new one. Reusing it is right (the name is the arm's), and the page says so, so it does not read as the
    pick having reverted: under New, that Continue reuses the one the list shows, and in the summary that it was
    reused, from the pick's `reused`. With none of that name, New still says it registers one."""
    metal = setup_fakes.account_embodiment("tester/metal", zero_shot_embodiment(6))
    account = setup_fakes.Account(monkeypatch, [metal])
    listed = {"embodiments": [{"name": "tester/metal", "cameras": ["scene", "wrist_l", "wrist_r"]}]}
    with Station(tmp_path / "station.yaml", platform.account) as st:
        st.server._ask_embodiments = lambda: list(account.embodiments)
        st.server._add_embodiment = account.add
        status, picked = st.call(
            "POST", "/api/robot", {"custom": {"ik_arm": "metal", "driver": "mine:Open", "embodiment": None}}
        )
        assert status == 200 and picked["reused"] is True and account.added == []

        def page(embodiments: dict[str, Any], arm: str) -> dict[str, Any]:
            return run_page(
                tmp_path,
                {
                    "replies": {"GET /api/embodiments": embodiments},
                    "base": st.base,
                    "set": {"pick": "custom", "armWay": "ours", "ourIk": arm},
                    "values": {"embodiment": "new"},
                    "call": "showPick",
                    "show": ["newBox"],
                },
            )

        reuses = page(listed, "metal")
        registers = page(listed, "yam")
        none = page({"embodiments": []}, "metal")
        shown = run_page(tmp_path, {"replies": {}, "base": st.base, "call": "applyPick", "args": [picked], "show": []})
    assert reuses["errors"] == [] and not reuses["ids"]["newBox"]["hidden"]
    assert reuses["ids"]["newBox"]["text"].startswith(
        "Your account already has tester/metal, named after this arm, so Continue reuses it instead of registering "
        "a new one."
    )
    for fresh in (registers, none):
        assert fresh["errors"] == [] and fresh["ids"]["newBox"]["text"].startswith(
            "Continue registers the embodiment on your account, named after your arm"
        )
    said = "Embodiment tester/metal (already on your account; reused, not registered again)."
    assert shown["errors"] == [] and said in shown["summaries"]["0"]


def find_own_arms(st: Station, motor: tuple[float, float] | None) -> dict[str, Any]:
    """Picks a robot of one's own whose driver reports its gripper motor between `motor` (closed, open, degrees), or
    no angle, and finds both arms over the routes.

    @return The robot as the pick reply gives it.
    """
    for arm in st.rig.arms.values():
        arm.gripper_motor_deg = motor
    custom = {"ik_arm": "yam", "driver": "mine:Open", "embodiment": "tester/mine"}
    picked = st.call("POST", "/api/robot", {"custom": custom})[1]
    devices = {"devices": ["can-AAA111", "can-BBB222"], "settings": picked["prefill"]["settings"]}
    assert st.call("POST", "/api/arms/open", devices)[0] == 200
    for adapter, ident, side in (("can0", "can-AAA111", "left"), ("can1", "can-BBB222", "right")):
        st.hands.move(adapter, 0, 0.3)
        assert st.call("GET", "/api/status")[1]["wiggle"]["moved"] == ident
        assert st.call("POST", "/api/arms/assign", {"device": ident, "side": side})[0] == 200
    robot: dict[str, Any] = picked["robot"]
    return robot


def test_a_robot_of_ones_own_records_its_grippers_range_over_the_route_and_the_page_shows_it(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    """The gripper step of a robot of one's own: POST /api/gripper takes the angles its driver
    reports, and the page shows them without the bus arms' line about stopping short of the open stop."""
    with Station(tmp_path / "station.yaml", platform.account) as st:
        robot = find_own_arms(st, (15.0, -326.7))
        for end, opening in (("open", 1.0), ("closed", 0.0)):
            for adapter in ("can0", "can1"):
                st.hands.grip(adapter, opening)
            status, gripper = st.call("POST", "/api/gripper", {"end": end})
        state, live = st.call("GET", "/api/state")[1], st.call("GET", "/api/status")[1]
    kept = {"gripper_closed_deg": 15.0, "gripper_open_deg": -326.7}
    assert status == 200 and gripper["ends"] == {"left": kept, "right": kept} and live["gripper"] == gripper
    show = ["gripBox", "gripHelp", "gripActions", "gripStop", "gripNote"]
    scenario = {"replies": {"GET /api/state": state, "GET /api/status": live}, "set": {"robot": robot}}
    shown = run_page(tmp_path, {**scenario, "call": "renderGripper", "args": [gripper], "show": show})["ids"]
    assert [shown[i]["hidden"] for i in show] == [False, False, False, True, False]
    assert shown["gripNote"]["text"].startswith("Left gripper: open at -327, closed at 15, runs 15 to -327 degrees.")


def test_a_robot_of_ones_own_whose_driver_reports_no_gripper_angle_is_told_why_the_step_is_skipped(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    with Station(tmp_path / "station.yaml", platform.account) as st:
        robot = find_own_arms(st, None)
        state, live = st.call("GET", "/api/state")[1], st.call("GET", "/api/status")[1]
        refused = st.call("POST", "/api/gripper", {"end": "open"})
    assert live["gripper"] == {"skipped": setup_session.GRIPPER_SKIPPED}
    assert refused == (400, {"error": setup_session.GRIPPER_SKIPPED})
    show = ["gripBox", "gripHelp", "gripActions", "gripNote"]
    scenario = {"replies": {"GET /api/state": state, "GET /api/status": live}, "set": {"robot": robot}}
    shown = run_page(tmp_path, {**scenario, "call": "renderGripper", "args": [live["gripper"]], "show": show})["ids"]
    assert [shown[i]["hidden"] for i in show] == [False, True, True, False]
    assert shown["gripNote"]["text"] == setup_session.GRIPPER_SKIPPED


def test_every_step_runs_on_the_routes_own_replies_for_cameras_the_embodiment_names(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    """An embodiment's own camera names (LeRobot's cam_high, cam_left_wrist, cam_right_wrist) are in neither the
    page's nor the server's tables of names, so the server labels every role itself
    (setup_cameras.camera_roles()): such a camera has no reference picture until the operator's dataset names its key.
    The page's script runs each step on the routes' own replies for such a robot, loaded afresh at each step as a
    reload would be, the operator's hands between; no step may log or show an error."""
    roles = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    dataset = test_setup_cameras.write_dataset(tmp_path / "ds", "libx264", episodes=1)
    errs = ["pickErr", "armsErr", "camErr", "homeErr", "reachErr", "motionErr", "saveErr"]
    with Station(tmp_path / "station.yaml", platform.account) as st:

        def page(scenario: dict[str, Any]) -> dict[str, Any]:
            shown = run_page(
                tmp_path, {"replies": {}, "base": st.base, **scenario, "show": [*errs, *scenario["show"]]}
            )
            assert shown["errors"] == [] and shown["warnings"] in ({}, {"compareRows": [FAKE_SIZE] * 3}), shown
            assert {i: shown["ids"][i]["text"] for i in errs} == dict.fromkeys(errs, "")
            ids: dict[str, Any] = shown["ids"]
            return ids

        for arm in st.rig.arms.values():
            arm.gripper_motor_deg = None
        custom = {"pick": "custom", "driver": "mine:Open", "armWay": "ours", "ourIk": "yam"}
        page({"set": custom, "values": {"embodiment": "tester/cams"}, "call": "pickRobot", "show": []})
        picked = st.call("GET", "/api/state")[1]["picked"]
        assert picked["robot"]["roles"] == roles and picked["cameras"] == [
            {"role": role, "kind": None, "label": f"Camera {role}", "short": role} for role in roles
        ]
        devices = {"devices": ["can-AAA111", "can-BBB222"], "settings": picked["prefill"]["settings"]}
        assert st.call("POST", "/api/arms/open", devices)[0] == 200
        for adapter, ident, side in (("can0", "can-AAA111", "left"), ("can1", "can-BBB222", "right")):
            st.hands.move(adapter, 0, 0.3)
            assert st.call("GET", "/api/status")[1]["wiggle"]["moved"] == ident
            assert st.call("POST", "/api/arms/assign", {"device": ident, "side": side})[0] == 200

        cams = page(
            {
                "values": {"datasetPath": str(dataset)},
                "call": "shootAll",
                "press": [["gallery", f'[data-role="{role}"]', i] for i, role in enumerate(roles)],
                "click": ["useDataset", "liveViews", "camsDone"],
                "after": ["checkSwap"],
                "show": ["camRows", "datasetKeys", "compareNote", "compareRows", "swap"],
            }
        )
        values = [row["value"] for row in CAMERAS]
        assert cams["camRows"]["text"] == "".join(
            f"Camera {r}: {v}widthheightfpsrotate" for r, v in zip(roles, values, strict=True)
        )
        assert cams["datasetKeys"]["text"].startswith("Camera cam_high(our YAM example)observation.images.top")
        assert st.server.dataset_keys == dict.fromkeys(roles)
        assert cams["compareNote"]["text"].startswith("We have no reference pictures for cameras with these names.")
        assert cams["compareRows"]["text"].count("No reference picture") == 3 and cams["swap"]["hidden"]
        status, refused = st.call("GET", "/api/reference?role=cam_high&episode=0")
        assert status == 400 and "the cam_high camera has no reference picture" in refused["error"]
        keys = dict(zip(roles, test_setup_cameras.KEYS, strict=True))
        assert st.call("POST", "/api/dataset/keys", {"keys": keys})[0] == 200
        status, picture = st.call("GET", "/api/reference?role=cam_high&episode=0")
        assert status == 200 and picture == setup_cameras.Dataset(dataset).first_frame(keys["cam_high"], 0)

        st.pose_home()
        page({"click": ["homeOk", *(["reachSkipGo"] if picked["reach"] else []), "safeGo"], "show": []})
        assert st.server.session.wait_motion(120.0)
        for ident, side in (("can-AAA111", "left"), ("can-BBB222", "right")):
            st.server.session.devices[ident]["open"] = setup_devices.CAN_LINKS[side]
        saved = page({"click": ["eyesYes", "motionDone", "saveBtn"], "show": ["savedLine"]})
        assert saved["savedLine"]["text"].startswith(f"Saved {tmp_path / 'station.yaml'}.")
    assert [cam.name for cam in config.load(tmp_path / "station.yaml").cameras] == roles


def test_a_new_account_finishes_setup_on_the_embodiment_its_own_steps_register(
    tmp_path: pathlib.Path, platform: FakePlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero-shot from nothing: an account with no embodiment and no fine-tune picks a robot of its
    own on a new embodiment, typing nothing for it, and the page's script runs every step on the routes' own replies,
    as the camera-names test does. The file names the cameras scene, wrist_l and wrist_r, as the camera step's role
    buttons do, and the embodiment the pick registered; the robot registers as it, and a Dream Labs base model reads
    those cameras."""
    roles = ["scene", "wrist_l", "wrist_r"]
    errs = ["pickErr", "armsErr", "camErr", "homeErr", "reachErr", "motionErr", "saveErr"]
    account = setup_fakes.Account(monkeypatch, [])
    base = next(m for m in setup_fakes.MODELS if m.id == "dreamlabs/dream-1-beta")
    with Station(tmp_path / "station.yaml", platform.account) as st:
        st.server._ask_models = lambda: [m for m in setup_fakes.MODELS if m.owner == "public"]
        st.server._ask_embodiments = lambda: list(account.embodiments)
        st.server._add_embodiment = account.add

        def page(scenario: dict[str, Any]) -> dict[str, Any]:
            shown = run_page(
                tmp_path, {"replies": {}, "base": st.base, **scenario, "show": [*errs, *scenario["show"]]}
            )
            assert shown["errors"] == [] and shown["warnings"] in ({}, {"compareRows": [FAKE_SIZE] * 3}), shown
            assert {i: shown["ids"][i]["text"] for i in errs} == dict.fromkeys(errs, "")
            ids: dict[str, Any] = shown["ids"]
            return ids

        for arm in st.rig.arms.values():
            arm.gripper_motor_deg = None
        custom = {"pick": "custom", "driver": "mine:Open", "armWay": "ours", "ourIk": "yam"}
        show = ["embodiment", "embodimentNote", "newBox", "pickRobot"]
        offered = page({"set": custom, "values": {"embodiment": "new"}, "call": "showPick", "show": show})
        assert offered["embodiment"]["children"] == 1 and not offered["newBox"]["hidden"]
        assert (
            offered["embodimentNote"]["text"] == "Your account has no embodiment yet, so Continue registers a new one."
        )
        assert not offered["pickRobot"]["blocked"]
        page({"set": custom, "values": {"embodiment": "new"}, "call": "pickRobot", "show": []})
        picked = st.call("GET", "/api/state")[1]["picked"]
        assert picked["robot"]["roles"] == roles and picked["robot"]["custom"]["embodiment"] == "tester/yam"
        assert account.added == [("yam", zero_shot_embodiment(6))]
        devices = {"devices": ["can-AAA111", "can-BBB222"], "settings": picked["prefill"]["settings"]}
        assert st.call("POST", "/api/arms/open", devices)[0] == 200
        for adapter, ident, side in (("can0", "can-AAA111", "left"), ("can1", "can-BBB222", "right")):
            st.hands.move(adapter, 0, 0.3)
            assert st.call("GET", "/api/status")[1]["wiggle"]["moved"] == ident
            assert st.call("POST", "/api/arms/assign", {"device": ident, "side": side})[0] == 200

        cams = page(
            {
                "call": "shootAll",
                "press": [["gallery", f'[data-role="{role}"]', i] for i, role in enumerate(roles)],
                "click": ["liveViews", "camsDone"],
                "after": ["checkSwap"],
                "show": ["camRows", "compareRows"],
            }
        )
        values = [row["value"] for row in CAMERAS]
        labels = ["Head camera (scene)", "Left wrist camera (wrist_l)", "Right wrist camera (wrist_r)"]
        assert cams["camRows"]["text"] == "".join(
            f"{label}: {v}widthheightfpsrotate" for label, v in zip(labels, values, strict=True)
        )
        assert "No reference picture" not in cams["compareRows"]["text"]

        st.pose_home()
        page({"click": ["homeOk", *(["reachSkipGo"] if picked["reach"] else []), "safeGo"], "show": []})
        assert st.server.session.wait_motion(120.0)
        for ident, side in (("can-AAA111", "left"), ("can-BBB222", "right")):
            st.server.session.devices[ident]["open"] = setup_devices.CAN_LINKS[side]
        saved = page({"click": ["eyesYes", "motionDone", "saveBtn"], "show": ["savedLine"]})
        assert saved["savedLine"]["text"].startswith(f"Saved {tmp_path / 'station.yaml'}.")
    loaded = config.load(tmp_path / "station.yaml")
    assert [cam.name for cam in loaded.cameras] == roles and loaded.custom is not None
    assert loaded.custom.embodiment == "tester/yam" and platform.robots[-1]["embodiment"] == "emb_tester_yam"
    assert len(account.added) == 1
    assert base.cameras is not None and client_ik.camera_slots(tuple(roles), tuple(base.cameras), base.id) == {
        "scene": "overhead",
        "wrist_l": "wrist_l",
        "wrist_r": "wrist_r",
    }


def test_a_failed_check_shows_its_one_cause_and_continuing_anyway_asks_first(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    """A failed check (here the Metal arm's shoulder) shows one cause: what it is, what it does to the robot, the
    fix, and whether continuing past it is safe. "Run checks again" is the main button; "Continue anyway" asks "Are
    you sure?" first, naming each failed check and whether continuing is safe (setup once let an operator continue
    without saying what failed), and only its own "Continue anyway" goes on to Save, whose warning names them
    again."""
    with Station(tmp_path / "station.yaml", platform.account) as st:
        state = st.call("GET", "/api/state")[1]
        robot = st.call("POST", "/api/robot", {"body": "yam"})[1]["robot"]
        live = st.call("GET", "/api/status")[1]
    followed = setup_checks.Finding("pass", "sent joint 1 +5.0 degrees; it moved +4.9", joint=0)
    short = setup_checks.Finding(
        "fail", "sent joint 2 +5.0 degrees; it moved under a fifth of that (+0.6)", "weight", 1
    )
    nudges = [("joint 1 +5 degrees", followed), ("joint 2 +5 degrees", short)]
    live["motion"] = {
        "state": "failed",
        "let_go": False,
        "stop_cut": False,
        "phase": "",
        "nudging": None,
        "notes": [],
        "error": f"left arm: {short.message}",
        "results": [
            {"group": "nudge", "side": "left", "label": label, "finding": finding.as_json()}
            for label, finding in nudges
        ],
    }
    show = ["causeTitle", "causeImpact", "causeFix", "causeGoOn", "motionBanner", "motionGo", "motionDone"]
    scenario = {
        "replies": {"GET /api/state": state, "GET /api/status": live},
        "set": {"robot": robot, "step": 5},
        "call": "poll",
        "show": [*show, "motionRetry", "motionCause", "results", "saveWarn", "skipFailed", "skipGoOn", "skipAsk"],
    }
    shown = run_page(tmp_path, scenario)["ids"]
    assert {key: shown[key]["text"] for key in show} == {
        "causeTitle": "This arm cannot hold its own weight.",
        "causeImpact": "It sags below where the model sends it.",
        "causeFix": "Add gravity compensation to your driver, or raise the stiffness of the lifting joints.",
        "causeGoOn": setup_checks.GO_ON[True],
        "motionBanner": "The motors are off.",
        "motionGo": "Run again",
        "motionDone": "Continue anyway",
    }
    assert shown["motionGo"]["hidden"] and not shown["motionRetry"]["hidden"] and not shown["motionCause"]["hidden"]
    assert not shown["motionDone"]["disabled"] and shown["results"]["children"] == 2
    for clicks, saved in (
        (["motionDone"], False),
        (["motionDone", "skipBack"], False),
        (["motionDone", "skipGo"], True),
    ):
        warn = run_page(tmp_path, {**scenario, "click": clicks})["ids"]["saveWarn"]["text"]
        assert warn.startswith("The powered checks did not all pass.") == saved, (clicks, warn)
    named = f"Left arm, joint 2: {short.message} (This arm cannot hold its own weight)"
    asked = run_page(tmp_path, {**scenario, "click": ["motionDone"]})["ids"]
    assert asked["skipAsk"]["open"] and named in asked["skipFailed"]["text"]
    assert asked["skipGoOn"]["text"] == setup_checks.GO_ON[True]
    warn = run_page(tmp_path, {**scenario, "click": ["motionDone", "skipGo"]})["ids"]["saveWarn"]["text"]
    assert named in warn and warn.endswith(setup_checks.GO_ON[True])
    # One failed check that is not safe to go on past makes the whole answer "fix this first".
    swapped = setup_checks.Finding(
        "fail", "sent joint 3 +5.0 degrees; joint 4 moved +5.0 degrees instead", "write_order", 2
    )
    live["motion"]["results"].append(
        {"group": "nudge", "side": "right", "label": "joint 3", "finding": swapped.as_json()}
    )
    asked = run_page(tmp_path, {**scenario, "click": ["motionDone"]})["ids"]
    assert (
        "Right arm, joint 3" in asked["skipFailed"]["text"] and asked["skipGoOn"]["text"] == setup_checks.GO_ON[False]
    )


def test_the_reach_area_step_reads_its_area_and_skip_asks_first_then_tells_the_server(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    """The Reach area step shows each arm's painted area (every grid cell within the brush of a point) or why there is
    none; Continue waits until every arm has the least area (the hand at home alone is a 4 cm disk); Skip opens "Are
    you sure?", Go back sends nothing, and Skip anyway tells the server and goes on to the powered checks (step 5)."""
    with Station(tmp_path / "station.yaml", platform.account) as st:
        state = st.call("GET", "/api/state")[1]
        robot = st.call("POST", "/api/robot", {"body": "yam"})[1]["robot"]
        live = st.call("GET", "/api/status")[1]
    live["reach"] = {
        "painting": True,
        "skipped": False,
        "radius_m": 0.04,
        "grid_m": 0.02,
        "min_area_m2": setup_session.REACH_MIN_AREA_M2,
        "points": {"left": [[0.2, 0.0]], "right": []},
        "hands": {"left": [0.2, 0.0]},
        "bases": {"left": [0.0, 0.0], "right": None},
        "error": None,
    }
    ok = {"ok": True}
    scenario = {
        "replies": {
            "GET /api/state": state,
            "GET /api/status": live,
            "POST /api/reach/skip": ok,
            "POST /api/reach/paint": ok,
        },
        "set": {"robot": robot, "step": 4, "reach": True},
        "call": "poll",
        "show": ["reachArea-left", "reachArea-right", "reachDone", "reachHint", "reachErr", "reachWarn"],
    }
    shown = run_page(tmp_path, scenario)
    ids = shown["ids"]
    assert ids["reachArea-left"]["text"] == "52 cm² painted" and ids["reachDone"]["blocked"]
    assert ids["reachHint"]["text"] == "Paint at least 400 cm² per arm to continue, or skip this step."
    assert ids["reachArea-right"]["text"] == "No reading from this arm."
    assert ids["reachErr"]["text"] == ids["reachWarn"]["text"] == ""
    square = [[0.2 + 0.02 * i, 0.02 * j] for i in range(8) for j in range(8)]  # 12 by 12 cm of points: 528 cm²
    one = {**live["reach"], "points": {"left": square, "right": square[:-8]}}
    enough = run_page(
        tmp_path, {**scenario, "replies": {**scenario["replies"], "GET /api/status": {**live, "reach": one}}}
    )
    assert enough["ids"]["reachArea-right"]["text"] == "480 cm² painted" and not enough["ids"]["reachDone"]["blocked"]
    assert enough["ids"]["reachHint"]["text"] == ""
    asked = run_page(tmp_path, {**scenario, "click": ["reachSkip", "reachBack"]})
    assert "POST /api/reach/skip" not in [r["key"] for r in asked["requests"]]
    skipped = run_page(tmp_path, {**scenario, "click": ["reachSkip", "reachSkipGo"]})
    keys = [r["key"] for r in skipped["requests"]]
    assert keys[-2:] == ["POST /api/reach/skip", "POST /api/reach/paint"], keys
    assert skipped["requests"][-1]["body"] == {"on": False} and skipped["ids"]["reachErr"]["text"] == ""


def test_setup_stops_without_a_key_naming_the_credentials_file(
    credentials_path: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Without a key nothing can be registered, so `dreamcli setup` stops before the page is served, naming the
    variable, the file, and the platform's Setup API page as where the lines come from."""
    credentials_path.write_text("DREAM_API_URL=https://platform.example.com\n", encoding="utf-8")
    out = io.StringIO()
    assert setup_wizard.main(tmp_path / "station.yaml", port=0, browser=False, out=out) == 1
    assert "DREAM_API_KEY" in out.getvalue() and str(credentials_path) in out.getvalue()
    assert "the Setup API page on https://platform.example.com gives the lines" in out.getvalue()


@pytest.mark.parametrize("name", ["DREAM_GATEWAY_URL", "DREAM_API_URL"])
@pytest.mark.parametrize("url", ["example.com", "ftp://example.com", "https://example.com/v1", "https://h:99999"])
def test_malformed_credentials_stop_setup_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, name: str, url: str
) -> None:
    """A malformed URL, from the file or the environment, fails `dreamcli setup` before the page is served, naming
    the variable, not every Save."""
    monkeypatch.setenv(name, url)
    out = io.StringIO()
    assert setup_wizard.main(tmp_path / "station.yaml", port=0, browser=False, out=out) == 1
    assert name in out.getvalue()


def test_setup_serves_with_the_credentials_and_says_what_it_writes(
    monkeypatch: pytest.MonkeyPatch, platform: FakePlatform, tmp_path: pathlib.Path
) -> None:
    """With a key in the file, `dreamcli setup` serves (here: a start that ends at once) and prints the target."""
    monkeypatch.setattr(setup_wizard.SetupServer, "start", lambda self: self.done.set())
    monkeypatch.setattr(setup_wizard.SetupServer, "stop", lambda self: None)
    out = io.StringIO()
    assert setup_wizard.main(tmp_path / "station.yaml", port=0, browser=False, out=out) == 0
    assert f"writes {tmp_path / 'station.yaml'}" in out.getvalue()
    assert "warning" not in out.getvalue()


def test_dreamcli_tcp_serves_without_a_key_and_says_where_the_arm_files_go(
    credentials_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """`dreamcli tcp` registers nothing, so a credentials file with no key serves (here: a start that ends at once)
    and prints the folder the arm files go to; a malformed one stops it, naming the variable."""
    credentials_path.write_text("DREAM_API_URL=https://platform.example.com\n", encoding="utf-8")
    monkeypatch.setattr(setup_wizard.TcpServer, "start", lambda self: self.done.set())
    monkeypatch.setattr(setup_wizard.TcpServer, "stop", lambda self: None)
    out = io.StringIO()
    assert setup_wizard.tcp_main(tmp_path / "station.yaml", host=None, port=0, browser=False, out=out) == 0
    said = out.getvalue()
    assert f"dreamcli tcp: serving the tool point page; writes arm files to {tmp_path / 'arms'}" in said
    assert "press Ctrl-C to end it; each Set tool center point writes the arm file again" in said
    monkeypatch.setenv("DREAM_API_URL", "example.com")
    out = io.StringIO()
    assert setup_wizard.tcp_main(tmp_path / "station.yaml", host=None, port=0, browser=False, out=out) == 1
    assert out.getvalue().startswith("dreamcli tcp: ") and "DREAM_API_URL" in out.getvalue()


def test_setup_warns_when_this_computers_build_is_not_the_one_the_platform_runs(
    monkeypatch: pytest.MonkeyPatch, platform: FakePlatform, tmp_path: pathlib.Path
) -> None:
    """The platform builds the wheel with its commit in the version (`4.0.0+g<first 12>`, robots.build_commit()).
    `dreamcli setup` compares it with `commit` of the platform's `GET /v1/meta` and warns when they differ; it asks
    nothing from a source checkout, and says nothing when they match or the platform does not say its commit."""
    deployed = "1f100a0fc9756520b2ba1f7277035ad206cafc4e"
    assert robots.build_commit("4.0.0+g1f100a0fc975") == deployed[:12]
    assert robots.build_commit("0.8.0") is None and robots.build_commit("0+unknown") is None
    assert setup_wizard.build_warning(platform.origin, None) is None and platform.calls == []
    assert setup_wizard.build_warning(platform.origin, "04dd7bd8aaaa") is None  # the platform does not say
    platform.meta = {"commit": deployed}
    assert setup_wizard.build_warning(platform.origin, deployed[:12]) is None
    monkeypatch.setattr(robots, "COMMIT", "04dd7bd8aaaa")
    monkeypatch.setattr(setup_wizard.SetupServer, "start", lambda self: self.done.set())
    monkeypatch.setattr(setup_wizard.SetupServer, "stop", lambda self: None)
    out = io.StringIO()
    assert setup_wizard.main(tmp_path / "station.yaml", port=0, browser=False, out=out) == 0
    said = f"setup wizard: warning: this computer's dream library was built from 04dd7bd8aaaa, but {platform.origin}"
    assert (
        f"{said} runs 1f100a0fc975: install it again (pip install --upgrade dreamlabs), then run dreamcli setup again"
        in out.getvalue()
    )


def test_the_page_footer_shows_this_installs_version(
    tmp_path: pathlib.Path, served: tuple[setup_wizard.SetupServer, str]
) -> None:
    server, base = served
    state = json.loads(request(base, "GET", "/api/state")[2])
    status = json.loads(request(base, "GET", "/api/status")[2])
    assert state["version"] == robots.__version__
    shown = run_page(tmp_path, {"replies": {"GET /api/state": state, "GET /api/status": status}, "show": ["version"]})
    assert shown["ids"]["version"]["text"] == f"Version {robots.__version__}"


SETUP_CHILD = """
import pathlib, sys, time
from robots import setup_wizard
stop = setup_wizard.SetupServer.stop
def stopping(self):
    print("stopping", flush=True)
    time.sleep(2.0)
    stop(self)
    print("stop ran", flush=True)
setup_wizard.SetupServer.stop = stopping
sys.exit(setup_wizard.main(pathlib.Path(sys.argv[1]), host="127.0.0.1", port=0, browser=False))
"""


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_a_kill_or_a_dropped_ssh_session_ends_setup_the_way_ctrl_c_does(
    signum: signal.Signals, platform: FakePlatform, tmp_path: pathlib.Path
) -> None:
    env = {key: value for key, value in os.environ.items() if key != "SSH_CONNECTION"}
    child = subprocess.Popen(
        [sys.executable, "-c", SETUP_CHILD, str(tmp_path / "station.yaml")],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    assert child.stdout is not None
    seen = []
    for line in child.stdout:
        seen.append(line)
        if "press Ctrl-C" in line:
            break
    assert "hold it up before it drops" in seen[-1], seen
    child.send_signal(signum)
    for line in child.stdout:
        seen.append(line)
        if line == "stopping\n":
            break
    child.send_signal(signum)
    rest, _ = child.communicate(timeout=30)
    assert child.returncode == 0 and "setup wizard: ended\n" in seen and "stop ran" in rest, (seen, rest)


def test_setup_says_whether_it_saved_and_what_to_do_when_its_port_is_taken(
    monkeypatch: pytest.MonkeyPatch, platform: FakePlatform, tmp_path: pathlib.Path
) -> None:
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        out = io.StringIO()
        assert setup_wizard.main(tmp_path / "station.yaml", host="127.0.0.1", port=port, browser=False, out=out) == 1
    assert "another dreamcli setup is probably still running" in out.getvalue()
    assert f"--port {port + 1}" in out.getvalue() and "Errno" not in out.getvalue()

    monkeypatch.setattr(setup_wizard.SetupServer, "start", lambda self: self.done.set())
    monkeypatch.setattr(setup_wizard.SetupServer, "stop", lambda self: self.done.set())
    out = io.StringIO()
    assert setup_wizard.main(tmp_path / "station.yaml", port=0, browser=False, out=out) == 0
    assert "nothing was saved" in out.getvalue() and "robot up" not in out.getvalue()

    def saved(server: setup_wizard.SetupServer) -> None:
        server.saved = setup_wizard.Saved("", {}, "", ())
        server.session.still_holding = {"right": setup_session.still_holding_words("right")}
        server.done.set()

    monkeypatch.setattr(setup_wizard.SetupServer, "start", saved)
    opened: list[str] = []
    monkeypatch.setattr(setup_wizard.webbrowser, "open", lambda url: opened.append(url) or True)
    monkeypatch.setattr(setup_wizard, "_has_display", lambda: False)
    out = io.StringIO()
    assert setup_wizard.main(tmp_path / "station.yaml", port=0, browser=True, out=out) == 0
    assert "setup wizard: the right arm stopped answering; it still holds under power so it does not drop" in (
        out.getvalue()
    )
    # The next step to copy on a line of its own, and Live view's link, which only a box with a display opens.
    lines = out.getvalue().splitlines()
    live = f"{platform.account.platform_url}/live"
    assert "    dreamcli robot up" in lines and f"then press Start in Live view: {live}" in lines
    assert opened == []
    monkeypatch.setattr(setup_wizard, "_has_display", lambda: True)
    assert setup_wizard.main(tmp_path / "station.yaml", port=0, browser=True, out=io.StringIO()) == 0
    assert opened[-1] == live


def test_setup_says_to_keep_the_terminal_open_and_where_the_page_is(
    monkeypatch: pytest.MonkeyPatch, platform: FakePlatform, tmp_path: pathlib.Path
) -> None:
    """The page lives only as long as the process, so `dreamcli setup` says to leave its terminal open and to use a
    second one for other commands, on a line of its own, before the addresses. The addresses are this box's own (its
    interfaces and tailnet name, never its hostname, which can name another machine), each with the run's token; a
    browser opens on the loopback one when the box has a display. Over ssh the same addresses come first and the
    tunnel after them, for when none opens."""
    monkeypatch.setattr(setup_wizard.SetupServer, "start", lambda self: self.done.set())
    monkeypatch.setattr(setup_wizard.SetupServer, "stop", lambda self: None)
    monkeypatch.setattr(setup_access, "listen_addresses", lambda: ["192.168.1.21"])
    monkeypatch.setattr(setup_access, "tailnet_name", lambda: "station-a.tail1.ts.net")
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    opened: list[str] = []
    monkeypatch.setattr(setup_wizard.webbrowser, "open", lambda url: opened.append(url) or True)

    out = io.StringIO()
    assert setup_wizard.main(tmp_path / "station.yaml", port=0, browser=False, out=out) == 0
    lines = out.getvalue().splitlines()
    assert setup_wizard.KEEP_TERMINAL_OPEN in lines
    assert "Leave this terminal open while you use the setup page." in setup_wizard.KEEP_TERMINAL_OPEN
    assert "Open a second terminal for other commands." in setup_wizard.KEEP_TERMINAL_OPEN
    (intro,) = [line for line in lines if line.startswith("Open the setup page in a browser at one of these")]
    assert lines.index(setup_wizard.KEEP_TERMINAL_OPEN) < lines.index(intro)
    shown = "\n".join(lines)
    assert "http://192.168.1.21:0/?token=" in shown and "http://station-a.tail1.ts.net:0/?token=" in shown
    assert f"http://{os.uname().nodename}:" not in shown
    assert opened == []

    monkeypatch.setattr(setup_wizard, "_has_display", lambda: True)
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.5 51234 192.168.1.21 22")
    out = io.StringIO()
    assert setup_wizard.main(tmp_path / "station.yaml", port=0, browser=True, out=out) == 0
    lines = out.getvalue().splitlines()
    assert setup_wizard.KEEP_TERMINAL_OPEN in lines
    (tunnel,) = [i for i, line in enumerate(lines) if line.startswith("  ssh -N -L 0:127.0.0.1:0 ")]
    assert lines[tunnel].endswith("@192.168.1.21")
    (tailnet,) = [i for i, line in enumerate(lines) if line.startswith("  http://station-a:0/?token=")]
    assert tailnet < tunnel, "the tailnet name first, the tunnel the fallback"
    assert len(opened) == 1 and opened[0].startswith("http://127.0.0.1:0/?token=")

    # At the robot's own screen: the page opens there, and the terminal lists no network address to puzzle over.
    monkeypatch.delenv("SSH_CONNECTION")
    out = io.StringIO()
    assert setup_wizard.main(tmp_path / "station.yaml", port=0, browser=True, out=out) == 0
    shown = out.getvalue()
    assert "The setup page is opening in your browser." in shown and setup_wizard.KEEP_TERMINAL_OPEN in shown
    assert "192.168.1.21" not in shown and "station-a" not in shown and "ssh" not in shown
    (paste,) = [line for line in shown.splitlines() if line.startswith("If it does not open, paste this")]
    assert paste.endswith(opened[-1]) and len(opened) == 2


def test_setup_flushes_its_instructions_before_it_waits(
    monkeypatch: pytest.MonkeyPatch, platform: FakePlatform, tmp_path: pathlib.Path
) -> None:
    """When stdout is not a terminal Python buffers it by the block, so the keep-open line and the address would
    stay unseen for as long as the page is served, while the log lines on stderr show. The command flushes them
    before it starts waiting for the page."""
    # What had reached the bytes underneath when the wait ended, before main() returns and anything flushes on exit.
    seen_at_wait: list[str] = []
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    raw = io.BytesIO()
    out = io.TextIOWrapper(raw, encoding="utf-8")
    monkeypatch.setattr(setup_wizard.SetupServer, "start", lambda self: self.done.set())
    monkeypatch.setattr(setup_wizard.SetupServer, "stop", lambda self: seen_at_wait.append(raw.getvalue().decode()))

    assert setup_wizard.main(tmp_path / "station.yaml", port=0, browser=False, out=out) == 0
    (written,) = seen_at_wait
    assert setup_wizard.KEEP_TERMINAL_OPEN in written
    assert "Open the setup page in a browser at one of these" in written


def test_the_gripper_switch_route_is_off_until_its_flag_is_on(
    served: tuple[setup_wizard.SetupServer, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset, `DREAM_SETUP_GRIPPER_MIT` leaves no route and tells the page not to offer the button."""
    server, base = served
    monkeypatch.setattr(setup_wizard.station, "SETUP_GRIPPER_MIT", False)
    assert request(base, "POST", "/api/motors/gripper_mit", {"device": "can0"})[0] == 404
    assert server.state()["gripper_switch"] is False

    monkeypatch.setattr(setup_wizard.station, "SETUP_GRIPPER_MIT", True)
    assert server.state()["gripper_switch"] is True


def test_the_so101_check_names_the_file_each_plugged_in_arm_holds_and_what_is_no_calibration(
    tmp_path: pathlib.Path, platform: FakePlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each listed file is read as a calibration and compared with what each SO-101 adapter's servos hold: the arm's
    own file matches it, an old arm's matches none, and a home pose file is no calibration at all; an adapter that
    cannot be read says so without hiding the others' answers."""
    found = tmp_path / "calibration"
    found.mkdir()
    mine = so101_fakes.write_calibration(found / "acm2_follower_eeprom.json")
    old = so101_fakes.write_calibration(found / "4_bimanual_left.json", gripper=(6, 100, 2000, 3400))
    (found / "xlerobot_home_pose.json").write_text('{"home": [0, 0, 0, 0, 0, 0]}', encoding="utf-8")
    monkeypatch.setattr(setup_session, "SO101_CALIBRATION_DIR", found)
    adapters = [
        {"id": "serial-a", "kind": "serial", "open": "/dev/ttyACM2", "label": "USB serial adapter A"},
        {"id": "serial-b", "kind": "serial", "open": "/dev/ttyACM3", "label": "USB serial adapter B"},
    ]
    monkeypatch.setattr(setup_devices, "scan", lambda: {"arms": adapters, "cameras": [], "can": []})

    def held(port: str) -> dict[str, Any]:
        if port == "/dev/ttyACM3":
            raise errors.ArmError("servo 4 did not answer")
        return so101_fakes.HELD

    server = setup_wizard.SetupServer(tmp_path / "station.yaml", platform.account, held_calibration=held)
    reply = server.so101_match()
    files = {pathlib.Path(f["path"]).name: f for f in reply["files"]}
    assert files["acm2_follower_eeprom.json"]["matches"] == ["serial-a"]
    assert files["4_bimanual_left.json"] == {"path": str(old), "valid": True, "why": None, "matches": []}
    assert files["xlerobot_home_pose.json"]["valid"] is False
    assert [a["error"] for a in reply["adapters"]] == [None, "servo 4 did not answer"]
    assert str(mine) in [f["path"] for f in reply["files"]]


def test_the_gripper_switch_lets_go_of_the_arms_then_switches_only_that_adapters_link(
    tmp_path: pathlib.Path, platform: FakePlatform
) -> None:
    """The switch runs on the named CAN adapter's link after setup has closed every arm, and refuses an adapter
    the scan did not find."""
    switched: list[str] = []
    server = setup_wizard.SetupServer(
        tmp_path / "station.yaml", platform.account, gripper_to_mit=lambda link: switched.append(link) or "mit"
    )
    server.devices = [{"id": "can0", "kind": "can", "open": "can_follower_l"}]
    closed: list[bool] = []
    server.session.release_arms = lambda: closed.append(True)

    assert server.switch_gripper({"device": "can0"}) == {"mode": "mit"}
    assert switched == ["can_follower_l"] and closed == [True]
    with pytest.raises(ValueError, match="not a CAN adapter"):
        server.switch_gripper({"device": "can9"})


def test_the_redesigned_page_and_its_connection_route_follow_the_flag(
    served: tuple[setup_wizard.SetupServer, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With `DREAM_SETUP_WIZARD_V2` off, `/` is the current page and there is no connection route; on, `/` is the
    redesigned page and `/api/connection` answers. Each page runs its own script."""
    _, base = served
    monkeypatch.setattr(setup_wizard.station, "SETUP_WIZARD_V2", False)
    status, _, body = request(base, "GET", "/")
    assert status == 200 and b'src="/setup_wizard.js"' in body and b"/api/connection" not in body
    assert request(base, "GET", "/api/connection")[0] == 404

    monkeypatch.setattr(setup_wizard.station, "SETUP_WIZARD_V2", True)
    status, content_type, body = request(base, "GET", "/")
    assert status == 200 and content_type.startswith("text/html")
    assert b"Setup wizard" in body and b'src="/setup_wizard_v2.js"' in body
    assert request(base, "GET", "/api/connection")[0] == 200


def test_the_redesigned_page_has_every_element_the_wizard_script_uses() -> None:
    """The redesign runs its own script, `setup_wizard_v2.js`: every id it looks up is on the redesigned page, and so
    is each of its seven steps with the chip and summary the script fills."""
    import re

    page = setup_wizard.page_html(True)
    script = setup_wizard._package_text("setup_wizard_v2.js")
    used = set(re.findall(r'\$\("([A-Za-z0-9_]+)"\)', script))
    present = set(re.findall(r'id="([A-Za-z0-9_-]+)"', page))
    assert used and not used - present, sorted(used - present)
    for step in range(7):
        card = page.split(f'data-step="{step}"', 1)[1].split("</section>", 1)[0]
        assert "data-pill" in card and "data-summary" in card, step


def test_the_redesigned_page_loads_three_js_exactly_as_the_current_page_does() -> None:
    """One import map, pinned by version and hash, for both pages: a bump on one cannot leave the other loading a
    version its hashes no longer match."""

    def import_map(page: str) -> str:
        return page.split('<script type="importmap">', 1)[1].split("</script>", 1)[0]

    assert import_map(setup_wizard.page_html(True)) == import_map(setup_wizard.page_html(False))
    assert setup_wizard.page_html(True).isascii()


def test_the_redesigned_page_shows_the_driver_links_and_moves_every_view_like_a_cad_tool() -> None:
    """The driver template, the reBot example and the docs are links under the Driver heading, not in a collapsed
    section; and every 3D view the redesign makes gets cadNavigation() (right-drag moves it, the wheel zooms toward
    the pointer, Fit, Front, Side, Top and Iso), which the current page never calls."""
    import re

    page = setup_wizard.page_html(True)
    driver = page.split('id="driverBox"', 1)[1].split('id="calibCard"', 1)[0]
    for link in ("templateLink", "exampleLink", "docsLink"):
        assert f'id="{link}"' in driver, link
    assert "<details" not in driver.split('id="docsLink"', 1)[0]
    script = setup_wizard._package_text("setup_wizard_v2.js")
    views = re.findall(r"new (?:ArmView|UrdfView)\(", script)
    assert len(views) == 3 and len(re.findall(r"\bcadNavigation\(", script)) == len(views)
    assert "export function cadNavigation(" in setup_wizard._package_text("setup_viewer.js")
    assert "cadNavigation" not in setup_wizard._package_text("setup_wizard.js")


def test_the_redesigned_page_links_its_stylesheet_and_step_list(served: tuple[setup_wizard.SetupServer, str]) -> None:
    """The redesigned page links its stylesheet and the step-list script, and has the step list and the station line
    those fill; the current page keeps its own styles (DREAM_SETUP_WIZARD_V2 off)."""
    _, base = served
    for v2 in (True,):
        page = setup_wizard.page_html(v2)
        assert '<link rel="stylesheet" href="/setup_wizard.css">' in page and 'src="/setup_page.js"' in page, v2
        assert 'id="progress"' in page and 'id="stationName"' in page and 'id="steps"' in page, v2
        assert "<style>" not in page, v2
    status, content_type, body = request(base, "GET", "/setup_wizard.css")
    assert status == 200 and content_type.startswith("text/css") and b"nav.progress" in body


def test_dreamcli_tcp_serves_the_stylesheet_the_redesigned_page_links(
    tmp_path: pathlib.Path, platform: FakePlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With `DREAM_SETUP_WIZARD_V2` on, `dreamcli tcp` serves the redesigned page and the stylesheet it links, so
    the tool point page is styled: TcpServer answers `/setup_wizard.css` as text/css, with the redesigned page on or
    off."""
    css = setup_wizard._package_text(setup_wizard.STYLESHEET).encode("utf-8")
    for v2 in (True, False):
        monkeypatch.setattr(setup_wizard.station, "SETUP_WIZARD_V2", v2)
        server = setup_wizard.TcpServer(tmp_path / "station.yaml", platform.account, host="127.0.0.1", port=0)
        server.start()
        try:
            base = f"http://127.0.0.1:{server.port}"
            page = request(base, "GET", "/")[2].decode("utf-8")
            status, content_type, body = request(base, "GET", "/setup_wizard.css")
        finally:
            server.stop()
        assert ('<link rel="stylesheet" href="/setup_wizard.css">' in page) == v2
        assert (status, content_type.startswith("text/css"), body == css) == (200, True, True), v2


def test_the_redesigned_tool_point_page_opens_on_its_urdf_step_with_no_step_count() -> None:
    """The redesigned page under `dreamcli tcp` opens on the URDF step, not on Choose setup method, and shows no
    "Step 1 of 8" over it: the step list skips the choice and the count on the tool point page alone
    (main.tcp-only)."""
    steps = setup_wizard._package_text("setup_page.js")
    assert 'if (choosing && ((rerun && later()) || main.classList.contains("tcp-only")))' in steps
    assert 'if (main.classList.contains("tcp-only")) {\n      $("stepOf").textContent = "";' in steps


def test_every_step_is_its_own_card_and_only_the_current_one_shows_its_later_ones_gated() -> None:
    """Each of the redesigned page's seven steps is a card, after the page's own Choose setup; the right column shows
    the current one alone, and the step list on the left goes back only to a done step, a later one waiting until the
    ones above it are done."""
    import re

    page = setup_wizard.page_html(True)
    assert len(re.findall(r'<section class="card[^"]*" data-step="\d"', page)) == 7
    assert '<aside class="side">' in page and 'id="progress"' in page.split("<main>", 1)[0]
    css = setup_wizard._package_text(setup_wizard.STYLESHEET)
    assert "section.card:not(.open) { display: none; }" in css
    steps = setup_wizard._package_text("setup_page.js")
    assert '!card.classList.contains("done") || !window.setupGo' in steps
    assert 'link.disabled = finished() || choosing || card.classList.contains("later")' in steps
    way = page.split('id="wayCard"', 1)[1].split("</section>", 1)[0]
    assert 'id="ways"' in way and "Choose setup" in way and 'id="wayNext"' in page  # Continue sits outside its card


def test_the_redesigned_page_draws_a_joint_card_per_arm_from_the_status() -> None:
    """The joint cards are setup_page.js's, drawn into the arms step's `armCards` from `GET /api/status` while the arms
    are let go, each joint lit past 5 degrees from the arm's first reading."""
    page = setup_wizard.page_html(True)
    script = setup_wizard._package_text("setup_page.js")
    assert 'id="armCards"' in page.split('data-step="1"', 1)[1].split("</section>", 1)[0]
    assert 'fetch("/api/status")' in script and "> 5" in script and "status.wiggle.devices" in script


def test_the_connection_check_says_whether_the_platform_takes_the_key(platform: FakePlatform) -> None:
    """`ok` for the key the platform takes, `bad_key` with the platform's own message for a 401, `unreachable` when
    nothing answers, and nothing is registered by asking."""
    ok = setup_wizard.check_connection(GOOD_KEY, platform.origin + "/")
    assert ok["state"] == "ok" and ok["platform"] == platform.origin

    refused = setup_wizard.check_connection("sk-wrong", platform.origin)
    assert refused["state"] == "bad_key" and "401" in refused["message"] and "Invalid api key." in refused["message"]

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed = probe.getsockname()[1]
    assert setup_wizard.check_connection(GOOD_KEY, f"http://127.0.0.1:{closed}")["state"] == "unreachable"

    assert platform.robots == [] and all(method == "GET" for method, _ in platform.calls)


def test_the_state_names_the_station_as_save_registers_it(tmp_path: pathlib.Path, platform: FakePlatform) -> None:
    """`station_name` is the hostname register_robot() is given, so the page header names the robot Save makes."""
    state = setup_wizard.SetupServer(tmp_path / "station.yaml", platform.account, port=0).state()
    assert state["station_name"] == socket.gethostname()


def test_a_home_joint_out_of_range_is_named_with_its_reading_the_range_and_which_way_to_move(
    tmp_path: pathlib.Path,
) -> None:
    """The home step said only "Outside dream's model of the arm: left arm joints 2"."""
    out = [
        {"side": "left", "joint": 2, "deg": 104.2, "lower_deg": -90.0, "upper_deg": 90.0},
        {"side": "right", "joint": 5, "deg": None, "lower_deg": -90.0, "upper_deg": 90.0},
    ]
    reply = {"home": {"left": [0.0] * 6, "right": [0.0] * 6}}
    shown = run_page(
        tmp_path,
        {
            "replies": {"POST /api/home": {**reply, "out_of_range": out}},
            "set": {"status": {"gripper": None}, "joints": list(range(6))},
            "call": "homeOk",
            "show": ["homeErr"],
        },
    )
    assert shown["ids"]["homeErr"]["text"] == (
        "Left joint 2 reads 104°, the model allows -90° to 90°: move it back toward 90°, or your driver reads it in "
        "other units or with another zero. Right joint 5 reads nothing usable. Fix that and press OK again, or "
        "Continue anyway"
    )
