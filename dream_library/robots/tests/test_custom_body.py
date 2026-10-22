"""A robot of your own (`body: custom`): its config, its driver, its embodiment, and a run of the shared loop on the
template driver.

The embodiment lookup is the account's (`dream.DreamClient.get_embodiments`); every test here replaces it.
"""

import asyncio
import dataclasses
import datetime
import json
import logging
import pathlib
import sys
import time
from collections.abc import Iterator
from typing import Any, ClassVar

import conftest
import dream
import numpy as np
import numpy.typing as npt
import pytest
import test_end_to_end
import test_so101
import yaml
from dream.datasets import Embodiment
from dream.models import Model
from shared.datasets import embodiments

from robots import (
    adapter,
    bodies,
    cameras,
    cli,
    config,
    control,
    driver_template,
    errors,
    followers,
    home,
    kinematics,
    run,
    station,
)

## The shipped example; loaded as-is so the file and load() cannot drift apart.
EXAMPLE_PATH = pathlib.Path(__file__).resolve().parent.parent / "configs" / "custom.example.yaml"

## The embodiment the example names, as the account answers it: two arms recorded as joints and grip, one camera.
EMBODIMENT: dict[str, Any] = {
    "state": {
        "right_arm": [0, 6],
        "right_grip": [6, 7],
        "right_eef_xyz": [7, 10],
        "right_eef_rot6": [10, 16],
        "left_arm": [16, 22],
        "left_grip": [22, 23],
        "left_eef_xyz": [23, 26],
        "left_eef_rot6": [26, 32],
    },
    "action": {
        "right_eef_xyz": [0, 3],
        "right_eef_rot6": [3, 9],
        "right_grip": [9, 10],
        "left_eef_xyz": [10, 13],
        "left_eef_rot6": [13, 19],
        "left_grip": [19, 20],
    },
    "video": {"scene": "observation.images.scene"},
}

## The account lookup, kept before the autouse fixture replaces it.
LOOKUP = bodies.embodiment_config

## The model check, kept before conftest's autouse fixture replaces it.
CHECK = bodies.check_model

CREATED = datetime.datetime(2026, 10, 13, tzinfo=datetime.UTC)


class Account(dream.DreamClient):
    """A real `dream.DreamClient` whose account has a fine-tune of its arm (the example's embodiment) and one of its
    SO-101, an embodiment with none yet, a base model and a fine-tune shared from another account.

    A subclass, not a stand-in, because the tests patch it in as `dream.DreamClient` for the model check
    (`robots.bodies`) and a run opens its own client from that same name (`robots.run.open_client`): the run's
    client then answers the account from here and runs the model as the real client does.
    """

    def get_embodiments(self) -> list[Embodiment]:
        return [
            Embodiment("alice/my-arm", EMBODIMENT, CREATED, "emb_mine"),
            Embodiment("alice/so101", EMBODIMENT, CREATED, "emb_so101"),
            Embodiment("alice/new-arm", EMBODIMENT, CREATED, "emb_new"),
        ]

    def get_models(self) -> list[Model]:
        return [
            Model("dreamlabs/dream-1-flagship", "public", "dreamlabs/dream-1-flagship", "yam", True, CREATED),
            Model("dream-1-ft:aa-my-arm", "mine", "dreamlabs/dream-1-flagship", "emb_mine", True, CREATED),
            Model("dream-1-ft:bb-so101", "mine", "dreamlabs/dream-1-flagship", "emb_so101", True, CREATED),
            Model("dream-1-ft:cc-shared", "public", "dreamlabs/dream-1-flagship", "emb_bob", True, CREATED),
        ]


## A driver file of the customer's own, next to the config: a TemplateArm under another name.
DRIVER_FILE = "from robots import driver_template\n\n\nclass MyArm(driver_template.TemplateArm):\n    pass\n"


class RecordingArm(driver_template.TemplateArm):
    """The template's arm in memory, recording which of its methods the run called."""

    ## Every call, in order, across both arms.
    calls: ClassVar[list[str]] = []
    ## Every target written, `[6 joints, gripper]`, in order, across both arms.
    targets: ClassVar[list[list[float]]] = []

    def __init__(self, cfg: config.ArmConfig, side: str) -> None:
        RecordingArm.calls.append("open")
        super().__init__(cfg, side)

    def write(self, positions_rad: npt.NDArray[np.float64], gripper: float) -> None:
        RecordingArm.calls.append("write")
        RecordingArm.targets.append([*positions_rad, gripper])
        super().write(positions_rad, gripper)

    def close(self) -> None:
        RecordingArm.calls.append("close")

    def torque_off(self) -> None:
        RecordingArm.calls.append("torque_off")


class BrakelessArm(RecordingArm):
    """A #RecordingArm without brakes: disengaged it would fall (`drops_when_limp`)."""

    drops_when_limp = True
    fault: ClassVar[str | None] = None
    sided: ClassVar[list[tuple[str, list[float]]]] = []

    def read(self) -> followers.Reading:
        if self._side == "right" and BrakelessArm.fault is not None:
            raise errors.ArmError(BrakelessArm.fault)
        return super().read()

    def write(self, positions_rad: npt.NDArray[np.float64], gripper: float) -> None:
        BrakelessArm.sided.append((self._side, [*positions_rad, gripper]))
        super().write(positions_rad, gripper)

    def disengage(self) -> bool:
        RecordingArm.calls.append("disengage")
        return True


@pytest.fixture(autouse=True)
def account(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """The account's embodiments are the example's; each body is built afresh. Yields the names looked up."""
    looked_up: list[str] = []

    def lookup(name: str, key: str) -> dict[str, Any]:
        assert key == "custom.embodiment"
        looked_up.append(name)
        return EMBODIMENT

    monkeypatch.setattr(bodies, "embodiment_config", lookup)
    bodies.custom_body.cache_clear()
    RecordingArm.calls.clear()
    RecordingArm.targets.clear()
    yield looked_up
    bodies.custom_body.cache_clear()


def example() -> dict[str, Any]:
    """The example file as a mapping."""
    loaded: dict[str, Any] = yaml.safe_load(EXAMPLE_PATH.read_text(encoding="utf-8"))
    return loaded


def load(tmp_path: pathlib.Path, document: dict[str, Any]) -> config.AppConfig:
    """Writes `document` as the config file and loads it."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return config.load(path)


# The config


def test_the_example_loads_and_runs_the_template_on_dreams_ik_and_the_named_embodiment() -> None:
    """The example names the template driver, an IK and an embodiment; each arm has a device and a home."""
    cfg = config.load(EXAMPLE_PATH)

    assert cfg.body == config.CUSTOM_BODY and cfg.control_hz == 20.0
    assert cfg.custom == config.CustomConfig(
        driver="robots.driver_template:TemplateArm", ik="makers", embodiment="alice/my-arm", grip_scale=None
    )
    assert cfg.follower_l.can_channel == "/dev/serial/by-id/usb-my-arm-left"
    assert cfg.follower_r.home == (0.0,) * 6
    assert cfg.action_space == "eef"


def test_the_action_space_is_read_from_the_file(tmp_path: pathlib.Path) -> None:
    """`action_space: joint` asks for the model's own joints; the file's default is the tool poses."""
    assert load(tmp_path, {**example(), "action_space": "joint"}).action_space == "joint"


def test_a_driver_file_is_read_from_the_config_files_directory(tmp_path: pathlib.Path) -> None:
    """`file.py:Class` is stored absolute, so the station finds it whatever directory it starts in."""
    (tmp_path / "my_arm.py").write_text(DRIVER_FILE, encoding="utf-8")

    cfg = load(tmp_path, {**example(), "custom": {**example()["custom"], "driver": "my_arm.py:MyArm"}})

    assert cfg.custom is not None and cfg.custom.driver == f"{tmp_path.resolve() / 'my_arm.py'}:MyArm"
    arm = bodies.for_config(cfg).open_arm(cfg.follower_r, "right")
    assert isinstance(arm, followers.FollowerArm) and arm.name == "right"


def test_an_arm_file_is_read_from_the_config_files_directory(tmp_path: pathlib.Path) -> None:
    """`custom.ik` may be an arm file of your URDF (`dream.arm_file`); like a driver file it is stored absolute."""
    cfg = load(tmp_path, {**example(), "custom": {**example()["custom"], "ik": "arms/my-arm.json"}})

    assert cfg.custom is not None and cfg.custom.ik == str(tmp_path.resolve() / "arms" / "my-arm.json")


def test_a_driver_file_imports_the_files_beside_it(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A driver file imports a helper module beside it, and the station loads it by path from another directory; the
    folder goes last on sys.path, so it shadows nothing."""
    (tmp_path / "beside.py").write_text("NAME = 'beside'\n", encoding="utf-8")
    (tmp_path / "my_arm.py").write_text("import beside\n" + DRIVER_FILE, encoding="utf-8")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "beside", raising=False)
    monkeypatch.chdir(tmp_path.parent)

    driver = bodies.load_driver(f"{tmp_path / 'my_arm.py'}:MyArm")

    assert issubclass(driver, followers.Driver) and sys.modules["beside"].NAME == "beside"
    assert sys.path[len(sys.path) - 1] == str(tmp_path.resolve())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"custom": None}, r"config key 'custom' is required"),
        ({"custom": {"driver": "my_arm", "ik": "makers", "embodiment": "a/b"}}, r"package.module:Class"),
        ({"custom": {"driver": "m:C", "ik": "makers"}}, r"'custom.embodiment'"),
        ({"custom": {"driver": "m:C", "ik": "makers", "embodiment": "a/b", "joints": 5}}, r"unknown keys"),
        ({"follower_l": {"can_channel": "/dev/ttyACM0"}}, r"'follower_l' needs 'home'"),
        ({"follower_l": {"can_channel": "x", "home": [0] * 6, "gripper_hold_torque_nm": 0.5}}, r"unknown keys"),
        ({"follower_l": {"can_channel": "x", "home": [0] * 6, "gripper_open_deg": -96}}, r"sets only 'gripper_open"),
        (
            {"follower_l": {"can_channel": "x", "home": [0] * 6, "gripper_open_deg": 5, "gripper_closed_deg": 5}},
            r"both 5: the gripper would not move",
        ),
        ({"body": "makers"}, r"config key 'custom' is for 'body: custom' only"),
        ({"action_space": "joints"}, r"config key 'action_space' must be one of \['eef', 'joint'\], got 'joints'"),
    ],
)
def test_a_custom_file_refuses_what_it_cannot_run(
    tmp_path: pathlib.Path, change: dict[str, Any], message: str
) -> None:
    """No section, a driver that is not `module:Class`, a missing key, an arm without a home, one gripper end or two
    equal ones, or a section for another body fails at load, by name."""
    document = {key: value for key, value in {**example(), **change}.items() if value is not None}

    with pytest.raises(errors.ConfigError, match=message):
        load(tmp_path, document)


def test_a_custom_arm_keeps_the_gripper_ends_the_wizard_measured_and_nan_without(tmp_path: pathlib.Path) -> None:
    """`gripper_closed_deg` and `gripper_open_deg`, the setup wizard's gripper step on a driver that reports its
    gripper motor's angle, reach the driver's config as written, either way round; without them both are NaN, never
    another arm's numbers."""
    document = example()
    document["follower_l"] = {**document["follower_l"], "gripper_closed_deg": 15.0, "gripper_open_deg": -326.7}

    cfg = load(tmp_path, document)

    assert (cfg.follower_l.gripper_closed_deg, cfg.follower_l.gripper_open_deg) == (15.0, -326.7)
    assert np.isnan(cfg.follower_r.gripper_closed_deg) and np.isnan(cfg.follower_r.gripper_open_deg)


def test_robot_up_brings_up_no_link_for_a_custom_robot(tmp_path: pathlib.Path) -> None:
    """Your driver opens its own device, whatever it is; `robot up` leaves it alone."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({**example(), "follower_l": {"can_channel": "can0", "home": [0] * 6}}))

    assert cli._can_channels(path) == []


# The body


def test_policy_config_is_the_named_embodiment_the_loop_rate_and_dreams_ik_of_the_arm(account: list[str]) -> None:
    """The embodiment is looked up once, when the wire is first needed, and is the dream config's block."""
    cfg = config.load(EXAMPLE_PATH)

    built = run.policy_config(cfg)
    named = run.policy_config(cfg, "alice/my-arm-ft")

    assert built == {
        "embodiment": EMBODIMENT,
        "action_hz": 20.0,
        "sync_above_ms": run.SYNC_ABOVE_MS,
        "ik": {"arm": "makers", "head": "eef", "solve": "traj"},
    }
    assert named["checkpoint"] == "alice/my-arm-ft"
    assert embodiments.parse(built["embodiment"]).state_dim == 32
    assert account == ["alice/my-arm"]
    assert run.loop_settings(cfg)["wire"] is bodies.for_config(cfg).wire()


def test_opening_and_homing_a_custom_robot_needs_no_network(account: list[str]) -> None:
    """The arms open through the driver the config names, with the config's home; `robots.home` drives them there."""
    cfg = config.load(EXAMPLE_PATH)
    home_pose = (0.1, -0.2, 0.3, 0.0, 0.0, 0.0)
    cfg = dataclasses.replace(cfg, follower_l=config.ArmConfig(can_channel="sim", home=home_pose))

    assert home.main(cfg, duration_s=0.05) == 0
    arm = bodies.for_config(cfg).open_arm(cfg.follower_l, "left")
    np.testing.assert_array_equal(arm.home_rad, home_pose)
    assert account == []


def test_a_driver_that_does_not_load_or_is_not_a_driver_fails_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """A module that is not there, a name it does not have, or a class that is not a `followers.Driver`."""
    arm_cfg = config.ArmConfig(can_channel="sim", home=(0.0,) * 6)

    def open_with(driver: str) -> None:
        bodies.custom_body(config.CustomConfig(driver=driver, ik="makers", embodiment="a/b")).open_arm(arm_cfg, "left")

    with pytest.raises(errors.ConfigError, match=r"module 'no_such_module' does not load"):
        open_with("no_such_module:Arm")
    with pytest.raises(errors.ConfigError, match=r"has no 'NoArm'"):
        open_with("robots.driver_template:NoArm")
    monkeypatch.setattr(driver_template, "NotADriver", lambda cfg, side: object(), raising=False)
    with pytest.raises(errors.ConfigError, match=r"opened object, not a robots.followers.Driver"):
        open_with("robots.driver_template:NotADriver")


def test_the_embodiment_is_looked_up_by_name_on_the_account(monkeypatch: pytest.MonkeyPatch) -> None:
    """A name the account does not have fails, listing the names it does."""

    class Embodiment:
        def __init__(self, name: str) -> None:
            self.name = name
            self.embodiment_config = {"name": name}

    class Client:
        def get_embodiments(self) -> list[Embodiment]:
            return [Embodiment("alice/my-arm"), Embodiment("alice/other")]

    monkeypatch.setattr(bodies.dream, "DreamClient", Client)

    assert LOOKUP("alice/my-arm", "custom.embodiment") == {"name": "alice/my-arm"}
    with pytest.raises(errors.ConfigError, match=r"'custom.embodiment' names 'bob/arm', which this account does not"):
        LOOKUP("bob/arm", "custom.embodiment")


def test_a_fine_tune_of_another_embodiment_is_refused_naming_both_and_what_to_pick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SO-101 fine-tune on this arm is refused, at the check and so in the dream config; the arm's own
    fine-tune, a base model, a shared fine-tune and a checkpoint the account cannot run (the client's) pass."""
    monkeypatch.setattr(bodies.dream, "DreamClient", Account)
    cfg = config.load(EXAMPLE_PATH)
    refusal = (
        r"^'dream-1-ft:bb-so101' was trained on 'alice/so101', but config key 'custom.embodiment' names "
        r"'alice/my-arm': .* pick one of your fine-tunes of 'alice/my-arm' \(dream-1-ft:aa-my-arm\) or a Dream Labs "
        r"base model$"
    )

    with pytest.raises(errors.ConfigError, match=refusal):
        CHECK(cfg, "dream-1-ft:bb-so101")
    monkeypatch.setattr(bodies, "check_model", CHECK)
    with pytest.raises(errors.ConfigError, match=refusal):
        run.policy_config(cfg, "dream-1-ft:bb-so101")
    for runs in ("dream-1-ft:aa-my-arm", "dreamlabs/dream-1-flagship", "dream-1-ft:cc-shared", "not-listed"):
        CHECK(cfg, runs)
    new_arm = dataclasses.replace(cfg, custom=dataclasses.replace(cfg.custom, embodiment="alice/new-arm"))
    with pytest.raises(errors.ConfigError, match=r"your account has no fine-tune of 'alice/new-arm' yet"):
        CHECK(new_arm, "dream-1-ft:aa-my-arm")


def test_joints_run_a_fine_tune_of_this_robot_on_the_joint_head_and_the_run_logs_its_head(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """With `action_space: joint` the arm's own fine-tune is served on the client's head `joint`, and the line the
    run logs names the head; with the default the same model is served on `eef`."""
    monkeypatch.setattr(bodies.dream, "DreamClient", Account)
    monkeypatch.setattr(bodies, "check_model", CHECK)
    cfg = config.load(EXAMPLE_PATH)
    joints = dataclasses.replace(cfg, action_space="joint")

    with caplog.at_level(logging.INFO, logger="robots.run"):
        served = run.policy_config(joints, "dream-1-ft:aa-my-arm")

    assert served["ik"] == {"arm": "makers", "head": "joint", "solve": "traj"}
    assert "dream client for dream-1-ft:aa-my-arm: ik head 'joint' (the model's own joints)" in caplog.text
    assert run.policy_config(cfg, "dream-1-ft:aa-my-arm")["ik"] == {"arm": "makers", "head": "eef", "solve": "traj"}


def test_a_reach_area_on_a_joint_run_is_kept_and_warned_about(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The client clamps only head `eef`, so a joint run with a reach area says the hands are not kept in it."""
    monkeypatch.setattr(bodies.dream, "DreamClient", Account)
    monkeypatch.setattr(bodies, "check_model", CHECK)
    cfg = config.load(EXAMPLE_PATH)
    painted = dataclasses.replace(cfg, follower_r=dataclasses.replace(cfg.follower_r, reach=((0.3, 0.0),)))

    with caplog.at_level(logging.WARNING, logger="robots.run"):
        served = run.policy_config(dataclasses.replace(painted, action_space="joint"), "dream-1-ft:aa-my-arm")
        eef = run.policy_config(painted, "dream-1-ft:aa-my-arm")

    assert served["ik"]["reach"] == eef["ik"]["reach"] == {"right": [[0.3, 0.0]]}
    assert caplog.text.count("the reach area applies only to eef runs; this joint run does not keep the hands") == 1


@pytest.mark.parametrize(
    "checkpoint",
    ["dreamlabs/dream-1-flagship", "dream-1-ft:cc-shared", "not-listed"],
    ids=["base", "shared", "unlisted"],
)
def test_joints_refuse_a_model_that_is_not_a_fine_tune_of_this_robot_naming_what_to_pick(
    monkeypatch: pytest.MonkeyPatch, checkpoint: str
) -> None:
    """A base model answers in YAM joints, and a shared or unlisted model's embodiment cannot be matched to this
    robot's: with `action_space: joint` each is refused, naming the switch back to tool poses and the arm's own
    fine-tunes; the same models pass on tool poses."""
    monkeypatch.setattr(bodies.dream, "DreamClient", Account)
    cfg = config.load(EXAMPLE_PATH)
    refusal = (
        rf"^config key 'action_space' is 'joint', but '{checkpoint}' is not one of this account's fine-tunes of "
        r"'alice/my-arm': .* set 'action_space' to 'eef' \(Tool pose in the Live view\), or pick one of your "
        r"fine-tunes of 'alice/my-arm' \(dream-1-ft:aa-my-arm\)$"
    )

    with pytest.raises(errors.ConfigError, match=refusal):
        CHECK(dataclasses.replace(cfg, action_space="joint"), checkpoint)
    CHECK(cfg, checkpoint)


def test_joints_refuse_another_embodiments_fine_tune_and_a_robot_that_names_no_embodiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Another embodiment's fine-tune is refused as on tool poses; an embodiment with no fine-tune yet says so; a
    config that names no embodiment (a Makers arm) cannot match a fine-tune to its robot and runs tool poses only,
    without asking the account."""
    monkeypatch.setattr(bodies.dream, "DreamClient", Account)
    cfg = dataclasses.replace(config.load(EXAMPLE_PATH), action_space="joint")
    assert cfg.custom is not None

    with pytest.raises(errors.ConfigError, match=r"^'dream-1-ft:bb-so101' was trained on 'alice/so101'"):
        CHECK(cfg, "dream-1-ft:bb-so101")
    new_arm = dataclasses.replace(cfg, custom=dataclasses.replace(cfg.custom, embodiment="alice/new-arm"))
    with pytest.raises(errors.ConfigError, match=r"or fine-tune one of 'alice/new-arm' first: your account has none"):
        CHECK(new_arm, "dreamlabs/dream-1-flagship")
    monkeypatch.setattr(bodies.dream, "DreamClient", lambda: pytest.fail("asked the account"))
    makers = dataclasses.replace(cfg, body="makers", custom=None)
    with pytest.raises(errors.ConfigError, match=r"a makers config names no embodiment to match 'dream-1-ft:x' to"):
        CHECK(makers, "dream-1-ft:x")


def test_a_body_that_names_no_embodiment_is_not_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    """A YAM, Makers or Metal config names no embodiment of the account's: nothing is asked."""

    def no_client() -> None:
        raise AssertionError("asked the account")

    monkeypatch.setattr(bodies.dream, "DreamClient", no_client)

    CHECK(dataclasses.replace(config.load(EXAMPLE_PATH), custom=None), "dream-1-ft:bb-so101")


def test_a_start_with_another_embodiments_fine_tune_fails_before_its_client_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model the Live view picks reaches the station's start; each run's client is built through the check, so the
    run fails with its words, no client is opened, and no arm is written."""
    monkeypatch.setattr(bodies, "check_model", CHECK)
    monkeypatch.setattr(bodies.dream, "DreamClient", Account)
    monkeypatch.setattr(driver_template, "RecordingArm", RecordingArm, raising=False)
    monkeypatch.setattr(run, "open_client", lambda config: pytest.fail(f"a client was opened with {config}"))
    cfg = config.load(EXAMPLE_PATH)
    assert cfg.custom is not None
    custom = dataclasses.replace(cfg.custom, driver="robots.driver_template:RecordingArm")
    controller = station.StationController.from_config(
        dataclasses.replace(cfg, custom=custom), inference_log_path=None
    )
    written = len(RecordingArm.targets)

    async def scenario() -> None:
        await controller.start(conftest.TASK, "dream-1-ft:bb-so101")
        deadline = time.monotonic() + 10.0
        while controller.state == station.STARTING:
            assert time.monotonic() < deadline, "the run never failed"
            await asyncio.sleep(0.01)

    asyncio.run(scenario())

    assert controller.state == station.ERROR
    assert controller.last_error is not None and controller.last_error["error_code"] == "ConfigError"
    assert "was trained on 'alice/so101'" in controller.last_error["message"]
    assert len(RecordingArm.targets) == written
    assert asyncio.run(controller.close())


## What the fake DreamAPI's SO-101 fine-tune answers on its joint head: each arm's joints in radians (the sixth is
## SO-101's held dummy joint, at 0) and the grip; it sends them in SO-101's recorded units.
JOINT_TARGETS = np.array([0.10, -0.08, 0.06, 0.05, -0.05, 0.0])
JOINT_GRIP = 0.7
## Rows per reply on the joint-head run.
JOINT_HORIZON = 48


class JointHeadAPI(test_end_to_end.FakeDreamAPI):
    """The fake gateway answering like the account's SO-101 fine-tune (`dream-1-ft:bb-so101`): `per_key` only, the
    joint head #JOINT_TARGETS in SO-101's recorded units, the grip, and tool poses the joint head leaves unread."""

    def __init__(self, units: dict[str, list[float]]) -> None:
        """@param units Per side, #JOINT_TARGETS in the checkpoint's units."""
        super().__init__(JOINT_HORIZON)
        self.per_key: dict[str, Any] = {}
        for side, row in units.items():
            self.per_key[f"action.{side}_arm"] = [row] * JOINT_HORIZON
            self.per_key[f"action.{side}_eef_xyz"] = [[0.2, 0.0, 0.1]] * JOINT_HORIZON
            self.per_key[f"action.{side}_eef_rot6"] = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]] * JOINT_HORIZON
            self.per_key[f"action.{side}_grip"] = [JOINT_GRIP] * JOINT_HORIZON
        roles = list(test_so101.SO101_EMBODIMENT["video"])
        self.models = [
            {
                "id": "dream-1-ft:bb-so101",
                "owner": "mine",
                "base_model": "dreamlabs/dream-1-flagship",
                "embodiment": "emb_so101",
                "servable": True,
                "created_at": "2026-10-13T00:00:00Z",
                "cameras": roles,
                "embodiment_config": None,
            }
        ]

    def _reply(self, body: dict[str, Any], index: int) -> tuple[int, dict[str, Any]]:
        """The base fake's sessions, with the actions replaced."""
        status, reply = super()._reply(body, index)
        reply["actions"] = {"per_key": self.per_key}
        return status, reply


def test_a_start_on_joints_drives_the_arms_with_the_fine_tunes_own_joints_through_the_real_client(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A robot of your own on dream's SO-101 IK, started as the Live view does with its own fine-tune on joints: the
    check passes it, the real dream client decodes the joint head from SO-101's recorded units (LeRobot's range, the
    held sixth joint) to radians, and the arms reach exactly those joints and the grip; the run logs its head."""
    specs = pytest.importorskip("dream_ik.specs", reason="dream's IK needs Python 3.11 or newer")
    units = {}
    for side in ("right", "left"):
        arm = specs.EMBODIMENTS["so101_faas"].side_arm(side)
        scale, offset = (np.asarray(u, dtype=np.float64) for u in arm.joint_units)
        units[side] = ((JOINT_TARGETS - offset) / scale).tolist()
    fake = JointHeadAPI(units)
    fake.start()
    monkeypatch.setenv("DREAM_GATEWAY_URL", f"http://{fake.host}:{fake.port}")
    monkeypatch.setenv("DREAM_API_URL", f"http://{fake.host}:{fake.port}")
    monkeypatch.setattr(bodies, "embodiment_config", lambda name, key: test_so101.SO101_EMBODIMENT)
    monkeypatch.setattr(bodies, "check_model", CHECK)
    monkeypatch.setattr(bodies.dream, "DreamClient", Account)
    monkeypatch.setattr(driver_template, "RecordingArm", RecordingArm, raising=False)
    monkeypatch.setattr(
        cameras.CameraReader,
        "from_config",
        lambda cfg, stale_after_s, **_: cameras.CameraReader(cfg.name, conftest.FakeCaptureSource(), stale_after_s),
    )
    height, width, _ = conftest.FRAME_SHAPE
    cfg = conftest.make_app_config(
        body=config.CUSTOM_BODY,
        custom=config.CustomConfig(
            driver="robots.driver_template:RecordingArm", ik="so101_faas", embodiment="alice/so101"
        ),
        # The station is what is tested, not the solve: tick needs no compile, which on CI's runners starved the loop.
        ik_solve="tick",
        follower_l=config.ArmConfig(can_channel="sim", home=(0.0,) * 6),
        follower_r=config.ArmConfig(can_channel="sim", home=(0.0,) * 6),
        cameras=[
            config.CameraConfig(name=name, device=i, width=width, height=height)
            for i, name in enumerate(test_so101.SO101_EMBODIMENT["video"])
        ],
        control_hz=30.0,
        action_horizon=JOINT_HORIZON,
        action_timeout_s=3.0,
        max_stall_ticks=60,
    )
    controller = station.StationController.from_config(cfg, inference_log_path=None)
    controller.open_cameras()

    def reached() -> bool:
        last = RecordingArm.targets[-2:]
        return len(last) == 2 and all(np.allclose(row, [*JOINT_TARGETS, JOINT_GRIP], atol=1e-6) for row in last)

    async def scenario() -> None:
        status = await controller.start(conftest.TASK, "dream-1-ft:bb-so101", "joint")
        assert status["action_space"] == "joint"
        deadline = time.monotonic() + 30.0
        while not reached():
            assert time.monotonic() < deadline, (controller.state, controller.last_error, RecordingArm.targets[-2:])
            await asyncio.sleep(0.02)
        assert controller.last_error is None
        await controller.close()

    with caplog.at_level(logging.INFO, logger="robots.run"):
        try:
            asyncio.run(scenario())
        finally:
            fake.close()

    assert "dream client for dream-1-ft:bb-so101: ik head 'joint' (the model's own joints)" in caplog.text
    sent = fake.requests[0].body
    assert sent["checkpoint"] == "dream-1-ft:bb-so101" and "ik" not in sent


# An arm of your own from its URDF

## The Metal arm as an arm of your own: dream's Metal URDF and its tool, as the setup wizard writes an arm file of it.
METAL_ARM: dict[str, Any] = {
    "urdf": "arm.urdf",
    "base": "base_link",
    "flange": "link6",
    "tcp": [0.135, 0.0, 0.0],
    "approach": [1.0, 0.0, 0.0],
    "jaw": [0.0, 1.0, 0.0],
    "rest": [0.0, -1.2, 1.5, 0.0, 0.0, 0.0],
    "tcp_calibration": {"status": "calibrated", "at": "2026-10-15T08:00:00+00:00"},
}

## The model a zero-shot run asks for: a Dream Labs base model, trained on YAM arms.
BASE = "dreamlabs/dream-1-flagship"


def metal_arm_file(folder: pathlib.Path) -> str:
    """Writes #METAL_ARM and dream's Metal URDF to `folder`; returns the arm file's path."""
    specs = pytest.importorskip("dream_ik.specs", reason="dream's IK needs Python 3.11 or newer")
    (folder / "arm.urdf").write_text((specs.ASSETS / "metal.urdf").read_text())
    path = folder / "arm.json"
    path.write_text(json.dumps(METAL_ARM))
    return str(path)


def test_an_arm_file_is_dreams_model_of_the_arm_for_the_layout_and_the_speed_cap(tmp_path: pathlib.Path) -> None:
    """An arm file is registered with dream's IK under its own name: its six joints all go out, the loop fills the
    gripper, and the speed cap's forward kinematics is the Metal's, whose URDF it is; dream's own names pass as is."""
    ik = metal_arm_file(tmp_path)

    assert kinematics.dream_arm(ik).startswith("file:") and kinematics.dream_arm("metal") == "metal"
    assert kinematics.arm_layout(ik) == adapter.ArmLayout(
        joints=adapter.ALL_JOINTS, width=6, client_gripper=False, slots=6
    )
    joints = np.array([0.1, -1.0, 1.3, 0.05, -0.05, 0.0])
    own = kinematics.DreamKinematics.both_arms(ik, 30.0)["right"].ee_position(joints)
    metal = kinematics.DreamKinematics.both_arms("metal", 30.0)["right"].ee_position(joints)
    np.testing.assert_allclose(own, metal, atol=1e-9)


def test_a_malformed_arm_file_fails_by_name(tmp_path: pathlib.Path) -> None:
    """dream's message on the arm file reaches the station as a ConfigError."""
    ik = metal_arm_file(tmp_path)
    pathlib.Path(ik).write_text(json.dumps({**METAL_ARM, "flange": "link9"}))

    with pytest.raises(errors.ConfigError, match=r"flange 'link9' is not a link of the URDF"):
        kinematics.arm_layout(ik)


def test_every_eef_launch_checks_the_arm_files_tool_point_was_marked_or_its_skip_chosen(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Skipping the tool point is discouraged, never blocked: the operator has to be deliberate about it.
    A marked arm file runs quietly; one skipped on the page runs with one
    warning line; one with no record, written by hand, is refused naming `dreamcli tcp` unless the hidden flag is
    passed, which warns. A joint run and dream's own arm are not checked; a malformed record fails by name."""
    ik = metal_arm_file(tmp_path)
    custom = config.CustomConfig(driver="robots.driver_template:RecordingArm", ik=ik, embodiment="alice/new-arm")
    app = conftest.make_app_config(body=config.CUSTOM_BODY, custom=custom)

    def check(record: dict[str, str] | None, *, skip: bool = False, action_space: str = "eef") -> list[str]:
        arm = {key: v for key, v in METAL_ARM.items() if key != kinematics.TCP_CALIBRATION}
        pathlib.Path(ik).write_text(json.dumps(arm if record is None else {**arm, kinematics.TCP_CALIBRATION: record}))
        caplog.clear()
        run.check_tool_point(dataclasses.replace(app, action_space=action_space), skip=skip)
        return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]

    at = "2026-10-15T08:00:00+00:00"
    assert check({"status": "calibrated", "at": at}) == []
    off = "so hand positions may land off target; 'dreamcli tcp' marks it"
    assert check({"status": "skipped", "at": at}) == [f"the tool point of {ik} was skipped, {off}"]
    with pytest.raises(errors.ConfigError) as refused:
        check(None)
    assert str(refused.value) == (
        f"the arm file {ik} (custom.ik) has no tool point: run 'dreamcli tcp', put the marker where your gripper's "
        "fingertips close and press Set tool center point; or, not advised, pass --skip-tcp-calibration"
    )
    assert check(None, skip=True) == [f"the tool point of {ik} was never marked (--skip-tcp-calibration), {off}"]
    assert check(None, action_space="joint") == []
    makers = dataclasses.replace(app, custom=dataclasses.replace(custom, ik="makers"))
    run.check_tool_point(makers, skip=False)
    with pytest.raises(errors.ConfigError, match=r"tcp_calibration must be .*got \{'status': 'later'"):
        check({"status": "later", "at": at})


def test_the_zero_shot_embodiment_registers_and_is_the_wire_of_an_arm_as_wide_as_its_joints() -> None:
    """What the setup wizard registers for an arm with no fine-tune yet: the data service's registration check passes
    it, its vectors tile, and an EefWire builds on it for a five- and a six-joint arm, sending each arm's joints and
    grip, and the scene camera before the wrists."""
    for width in (5, 6):
        wire_config = adapter.zero_shot_embodiment(width)
        embodiments.check_registration(wire_config)
        assert embodiments.parse(wire_config).state_dim == 2 * width + 20
        layout = adapter.ArmLayout(joints=tuple(range(width)), width=width, client_gripper=False, slots=width)
        wire = adapter.EefWire(wire_config, None, config.CUSTOM_BODY, layout)
        assert wire.camera_roles == ("scene", "wrist_l", "wrist_r")


class BaseModelAPI(test_end_to_end.FakeDreamAPI):
    """The fake gateway answering like a Dream Labs base model on its tool-pose head: every row one tool pose and the
    grip per arm, and YAM's six joints, which the client leaves unread on this arm."""

    def __init__(self, xyz: list[float], rot6: list[float]) -> None:
        """@param xyz The tool's position each arm is asked for, metres. @param rot6 Its rotation's first columns."""
        super().__init__(JOINT_HORIZON)
        self.per_key: dict[str, Any] = {}
        for side in ("right", "left"):
            self.per_key[f"action.{side}_eef_xyz"] = [xyz] * JOINT_HORIZON
            self.per_key[f"action.{side}_eef_rot6"] = [rot6] * JOINT_HORIZON
            self.per_key[f"action.{side}_grip"] = [JOINT_GRIP] * JOINT_HORIZON
            self.per_key[f"action.{side}_arm"] = [[0.0] * 6] * JOINT_HORIZON
        self.models = [
            {
                "id": BASE,
                "owner": "public",
                "base_model": BASE,
                "embodiment": "yam",
                "servable": True,
                "created_at": "2026-10-13T00:00:00Z",
                "cameras": ["overhead", "wrist_l", "wrist_r"],
                "embodiment_config": None,
            }
        ]

    def _reply(self, body: dict[str, Any], index: int) -> tuple[int, dict[str, Any]]:
        """The base fake's sessions, with the actions replaced."""
        status, reply = super()._reply(body, index)
        reply["actions"] = {"per_key": self.per_key}
        return status, reply


def test_a_robot_of_your_own_from_its_urdf_runs_a_base_model_zero_shot_through_the_real_client(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A robot of your own whose `custom.ik` is an arm file (of the Metal's URDF), on the embodiment the setup wizard
    registers for it, started with a Dream Labs base model on tool poses: the check passes it, the real dream client
    sends the model YAM's zero joints, the gripper and the cameras under their own roles, and solves its tool poses on
    the arm file's model, so both hands reach the pose the model asked for, with the grip it asked for. The scene
    camera keeps its name (scene) on the station and goes to the model as its scene camera (overhead)."""
    specs = pytest.importorskip("dream_ik.specs", reason="dream's IK needs Python 3.11 or newer")
    from dream_ik import decode
    from dream_ik.solver import ChunkIK

    ik = metal_arm_file(tmp_path)
    solver = ChunkIK(kinematics.dream_arm(ik), specs.SERVE.tool, 30.0)
    pose = np.asarray(solver.fk(np.array([0.1, -1.0, 1.3, 0.05, -0.05, 0.0]), "right"))  # (4, 4)
    target = pose[:3, 3]
    fake = BaseModelAPI(target.tolist(), np.asarray(decode.mat_to_rot6d(pose[:3, :3])).tolist())
    fake.start()
    monkeypatch.setenv("DREAM_GATEWAY_URL", f"http://{fake.host}:{fake.port}")
    monkeypatch.setenv("DREAM_API_URL", f"http://{fake.host}:{fake.port}")
    monkeypatch.setattr(bodies, "embodiment_config", lambda name, key: adapter.zero_shot_embodiment(6))
    monkeypatch.setattr(bodies, "check_model", CHECK)
    monkeypatch.setattr(bodies.dream, "DreamClient", Account)
    monkeypatch.setattr(driver_template, "RecordingArm", RecordingArm, raising=False)
    monkeypatch.setattr(
        cameras.CameraReader,
        "from_config",
        lambda cfg, stale_after_s, **_: cameras.CameraReader(cfg.name, conftest.FakeCaptureSource(), stale_after_s),
    )
    height, width, _ = conftest.FRAME_SHAPE
    cfg = conftest.make_app_config(
        body=config.CUSTOM_BODY,
        custom=config.CustomConfig(driver="robots.driver_template:RecordingArm", ik=ik, embodiment="alice/new-arm"),
        # The station is what is tested, not the solve: tick needs no compile, which on CI's runners starved the loop.
        ik_solve="tick",
        follower_l=config.ArmConfig(can_channel="sim", home=(0.0,) * 6),
        follower_r=config.ArmConfig(can_channel="sim", home=(0.0,) * 6),
        cameras=[
            config.CameraConfig(name=name, device=i, width=width, height=height)
            for i, name in enumerate(adapter.ZERO_SHOT_CAMERAS)
        ],
        control_hz=30.0,
        action_horizon=JOINT_HORIZON,
        action_timeout_s=3.0,
        max_stall_ticks=60,
    )
    controller = station.StationController.from_config(cfg, inference_log_path=None)
    controller.open_cameras()

    def misses() -> list[float]:
        """How far each of the last two commanded rows puts the tool from the target, metres."""
        rows = np.asarray(RecordingArm.targets[-2:])  # (2, 7)
        return [float(np.linalg.norm(np.asarray(solver.fk(row[:6], "right"))[:3, 3] - target)) for row in rows]

    def reached() -> bool:
        """Both hands within 2 mm of the target with the grip asked for (the gripper steps 0.05 a tick)."""
        rows = RecordingArm.targets[-2:]
        return len(rows) == 2 and max(misses()) < 2e-3 and all(abs(row[6] - JOINT_GRIP) < 1e-6 for row in rows)

    async def scenario() -> None:
        await controller.start(conftest.TASK, BASE, "eef")
        deadline = time.monotonic() + 30.0
        while not reached():
            assert time.monotonic() < deadline, (controller.state, controller.last_error, RecordingArm.targets[-2:])
            await asyncio.sleep(0.02)
        assert controller.last_error is None
        await controller.close()

    try:
        asyncio.run(scenario())
    finally:
        fake.close()

    sent = fake.requests[0].body
    assert sent["checkpoint"] == BASE and "ik" not in sent
    observation = sent["observation"]
    for side in ("right", "left"):
        assert observation[f"state.{side}_arm"] == [0.0] * 6 and len(observation[f"state.{side}_gripper"]) == 1
    assert sorted(key for key in observation if key.startswith("video.")) == [
        "video.overhead",
        "video.wrist_l",
        "video.wrist_r",
    ]


# The loop


def test_the_shared_loop_drives_the_template_arm_on_the_custom_wire() -> None:
    """Every request carries the embodiment's parts and cameras; the arms follow the policy's rows."""
    cfg = config.load(EXAMPLE_PATH)
    body = bodies.for_config(cfg)
    arms = {side: body.open_arm(cfg.follower_l, side) for side in ("left", "right")}
    clock = conftest.ManualClock()
    client = conftest.FakeClient(horizon=4)
    stale_after_s = conftest.LOOP_DEFAULTS["stale_after_s"]
    reader = cameras.CameraReader("scene", conftest.FakeCaptureSource(), stale_after_s, face_blur=False)
    loop = control.ControlLoop(
        followers=arms,
        cameras=[reader],
        client=client,
        task=conftest.TASK,
        kinematics=None,
        wire=body.wire(),
        camera_owner=False,
        clock=clock,
        sleep=clock.sleep,
        **conftest.LOOP_DEFAULTS,
    )

    async def scenario() -> None:
        async with loop:
            for tick in range(3):
                await loop.step(float(tick))

    # Started here, as the station does, so the manual clock cannot outrun the first frame.
    reader.start()
    try:
        deadline_s = time.monotonic() + 5.0
        while reader.frame is None and time.monotonic() < deadline_s:
            time.sleep(0.001)
        asyncio.run(scenario())
    finally:
        reader.stop()

    assert list(client.observations[0]) == [
        "prompt",
        "state.right_arm",
        "state.right_grip",
        "state.left_arm",
        "state.left_grip",
        "video.scene",
    ]
    state = arms["right"].read()
    assert state.commanded_rad is not None
    np.testing.assert_array_equal(state.positions_rad, state.commanded_rad)
    assert state.positions_rad[0] > 0.0, "the arm followed the policy's rows"
    assert adapter.VIDEO_PREFIX + "scene" in client.observations[0]


def test_the_template_arm_goes_limp_then_drops_the_first_target_after() -> None:
    """disengage() lets the template's arm go limp; the next write() re-engages it where it is and is dropped, and the
    one after is followed."""
    cfg = config.load(EXAMPLE_PATH)
    arm = driver_template.TemplateArm(cfg.follower_l, "left")
    pose = arm.read().positions_rad

    assert arm.disengage(), "the template arm has a passive mode"
    arm.write(pose + 0.1, 0.5)
    np.testing.assert_array_equal(arm.read().positions_rad, pose)
    arm.write(pose + 0.1, 0.5)
    np.testing.assert_array_equal(arm.read().positions_rad, pose + 0.1)


# A run in this terminal (`dreamcli robot run`)


@pytest.mark.parametrize("dry_run", [False, True])
def test_a_run_drives_the_arms_until_stopped_then_homes_and_switches_them_off_and_a_dry_run_never_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture, dry_run: bool
) -> None:
    """`robots.run.main` runs the shared loop until its stop is set, then moves every arm home and switches it off, and
    saves the run as an episode; a dry run runs the policy the same way, but no target and no switch-off reaches an
    arm (it is only closed, and its last line says so) and it saves no episode, whose targets no arm received."""
    cfg = load(tmp_path, {**example(), "custom": {**example()["custom"], "driver": "test_custom_body:RecordingArm"}})
    rollouts_dir = tmp_path / "rollouts"
    cfg = dataclasses.replace(cfg, rollouts=config.RolloutsConfig(enabled=True, dir=str(rollouts_dir)))
    client = conftest.FakeClient(horizon=cfg.action_horizon)
    reader = cameras.CameraReader("scene", conftest.FakeCaptureSource(), cfg.stale_after_s, face_blur=False)
    monkeypatch.setattr(run, "make_cameras", lambda app: [reader])
    monkeypatch.setattr(run, "make_kinematics", lambda app: None)
    monkeypatch.setattr(run, "open_client", lambda dream_config: client)

    async def scenario() -> None:
        stop = asyncio.Event()
        main = asyncio.create_task(run.main(cfg, "alice/my-arm-ft", dry_run=dry_run, stop=stop, home_duration_s=0.0))
        deadline_s = time.monotonic() + 10.0
        while client.calls < 2 and not main.done() and time.monotonic() < deadline_s:
            await asyncio.sleep(0.01)
        stop.set()
        await main

    with caplog.at_level(logging.INFO, logger="robots.run"):
        asyncio.run(scenario())

    assert client.calls >= 2
    messages = [record.getMessage() for record in caplog.records]
    stop_lines = [message for message in messages if message.startswith(("run: stopping", "run: dry"))]
    if dry_run:
        assert RecordingArm.calls == ["open", "open", "close", "close"]
        assert stop_lines == [
            "run: stopping; a dry run moved no arm",
            "run: dry run over: the arms are only closed, as your driver's close() leaves them",
        ]
        assert not rollouts_dir.exists()
    else:
        assert stop_lines == ["run: stopping; the arms hold, move home, then switch off"]
        assert len([path for path in rollouts_dir.iterdir() if path.is_dir()]) == 1
        assert "write" in RecordingArm.calls and RecordingArm.calls[-2:] == ["torque_off", "torque_off"]
        assert "close" not in RecordingArm.calls
        assert max(abs(target[0]) for target in RecordingArm.targets) > 0.0, "the policy's rows moved the arms"
        assert RecordingArm.targets[-2:] == [[0.0] * 6 + [home.HOME_GRIPPER]] * 2, "the run ended at home"


@pytest.mark.parametrize(
    ("fault", "homes_ok", "raised", "moves"),
    [
        (errors.ControlLoopError("no chunk"), True, r"no chunk", 2),
        (errors.ArmError("left arm stopped answering"), True, r"left arm stopped answering", 1),
        (None, False, r"an arm did not reach home before the run", 1),
    ],
)
def test_a_run_that_faults_switches_the_arms_off_then_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    fault: Exception | None,
    homes_ok: bool,
    raised: str,
    moves: int,
) -> None:
    """The loop's fault ends the run: the arms move home (not after an arm fault: the loop disengaged them, and they
    stay where they are) and are switched off, and the fault is what main() raises. An arm that cannot reach home
    before the run ends it before the loop starts."""
    cfg = load(tmp_path, {**example(), "custom": {**example()["custom"], "driver": "test_custom_body:RecordingArm"}})
    reader = cameras.CameraReader("scene", conftest.FakeCaptureSource(), cfg.stale_after_s, face_blur=False)
    monkeypatch.setattr(run, "make_cameras", lambda app: [reader])
    monkeypatch.setattr(run, "make_kinematics", lambda app: None)
    monkeypatch.setattr(run, "open_client", lambda dream_config: conftest.FakeClient(horizon=cfg.action_horizon))
    homed: list[list[str]] = []

    def return_to_home(arms: dict[str, followers.FollowerArm], *, duration_s: float) -> bool:
        homed.append(sorted(arms))
        return homes_ok

    async def loop_fault(self: control.ControlLoop) -> None:
        assert fault is not None, "the loop must not start"
        raise fault

    monkeypatch.setattr(home, "return_to_home", return_to_home)
    monkeypatch.setattr(control.ControlLoop, "run", loop_fault)

    with pytest.raises(errors.RobotsError, match=raised):
        asyncio.run(run.main(cfg, "alice/my-arm-ft", stop=asyncio.Event()))
    assert homed == [["left", "right"]] * moves
    assert RecordingArm.calls[-2:] == ["torque_off", "torque_off"]


def test_an_arm_fault_holds_both_brakeless_arms_where_they_are_then_switches_them_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    cfg = load(tmp_path, {**example(), "custom": {**example()["custom"], "driver": "test_custom_body:BrakelessArm"}})
    client = conftest.FakeClient(horizon=cfg.action_horizon)
    reader = cameras.CameraReader("scene", conftest.FakeCaptureSource(), cfg.stale_after_s, face_blur=False)
    monkeypatch.setattr(run, "make_cameras", lambda app: [reader])
    monkeypatch.setattr(run, "make_kinematics", lambda app: None)
    monkeypatch.setattr(run, "open_client", lambda dream_config: client)
    monkeypatch.setattr(BrakelessArm, "fault", None)
    monkeypatch.setattr(BrakelessArm, "sided", [])

    async def scenario() -> None:
        main = asyncio.create_task(run.main(cfg, "alice/my-arm-ft", stop=asyncio.Event(), home_duration_s=0.0))
        deadline_s = time.monotonic() + 10.0
        while client.calls < 2 and not main.done() and time.monotonic() < deadline_s:
            await asyncio.sleep(0.01)
        BrakelessArm.fault = "arm 'right' wrist_yaw reports fault 0x9, under voltage"
        await main

    with pytest.raises(errors.ArmError, match="under voltage"):
        asyncio.run(scenario())

    assert client.calls >= 2
    assert "disengage" not in RecordingArm.calls
    assert RecordingArm.calls[-2:] == ["torque_off", "torque_off"]
    for side in ("left", "right"):
        written = [target for name, target in BrakelessArm.sided if name == side]
        assert len(written) >= 2 and written[-1] == written[-2], f"the {side} arm holds its last target"


def test_a_model_that_fails_to_set_up_opens_no_arm(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """The arms open last, so a run that cannot start never moves one (switching an arm off parks it)."""
    cfg = load(tmp_path, {**example(), "custom": {**example()["custom"], "driver": "test_custom_body:RecordingArm"}})
    monkeypatch.setattr(run, "make_cameras", lambda app: [])
    monkeypatch.setattr(run, "make_kinematics", lambda app: None)

    def refuse(dream_config: dict[str, Any]) -> None:
        raise errors.ConfigError("no key")

    monkeypatch.setattr(run, "open_client", refuse)

    with pytest.raises(errors.ConfigError, match="no key"):
        asyncio.run(run.main(cfg, "alice/my-arm-ft", stop=asyncio.Event()))
    assert RecordingArm.calls == []
