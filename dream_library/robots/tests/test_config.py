"""Pins `robots.config.load`: one YAML file, every key checked, ConfigError naming the key.

Each test writes a small YAML file under `tmp_path`, starting from a full valid document and
removing or corrupting one key, so a failure points at the one rule it exercises.
"""

import copy
import logging
import math
import pathlib
from typing import Any

import pytest
import yaml

from robots import config, errors, gripper_force

## The shipped example config; loaded as-is when present so the file and load() cannot drift apart.
EXAMPLE_PATH = pathlib.Path(__file__).resolve().parent.parent / "configs" / "yam.example.yaml"

## A complete, valid document; tests copy it and change one thing.
FULL_CONFIG: dict[str, Any] = {
    "follower_l": {"can_channel": "can_left", "gripper": "linear_4310"},
    "follower_r": {"can_channel": "can_right"},
    "cameras": [
        {"name": "top", "device": 0, "width": 640, "height": 360, "fps": 30, "rotate": 0},
        {"name": "left", "device": "/dev/video1", "width": 640, "height": 360},
        {"name": "right", "device": "/dev/video2", "width": 640, "height": 360, "rotate": 180},
    ],
    "task": "pick up the cup and place it on the plate",
    "control_hz": 30,
    "action_horizon": 48,
    "action_timeout_s": 2.0,
    "stale_after_s": 0.2,
    "max_stall_ticks": 240,
    "max_ee_speed_m_s": 0.85,
    "first_chunk_timeout_s": 45.0,
    "ramp_s": 1.5,
    "ramp_max_rad_s": 0.8,
    "max_gripper_step": 0.1,
    "trace_path": "/var/tmp/robots-trace.csv",
}

## Top-level keys that have no default and therefore must be present.
REQUIRED_KEYS = (
    "follower_l",
    "follower_r",
    "cameras",
    "action_horizon",
    "stale_after_s",
    "max_stall_ticks",
    "max_ee_speed_m_s",
)


def write_config(tmp_path: pathlib.Path, document: Any) -> pathlib.Path:
    """Dumps `document` as YAML into `tmp_path` and returns the file path."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def load_variant(tmp_path: pathlib.Path, **changes: Any) -> config.AppConfig:
    """Loads FULL_CONFIG with top-level `changes` applied; a value of `None` deletes the key."""
    document = copy.deepcopy(FULL_CONFIG)
    for key, value in changes.items():
        if value is None:
            document.pop(key, None)
        else:
            document[key] = value
    return config.load(write_config(tmp_path, document))


# Full documents


def test_load_full_document_builds_app_config(tmp_path: pathlib.Path) -> None:
    """Every section lands in its dataclass, with the types the loop and dream expect."""
    cfg = load_variant(tmp_path)

    assert isinstance(cfg, config.AppConfig)
    assert cfg.follower_l == config.ArmConfig(can_channel="can_left", gripper="linear_4310")
    assert cfg.follower_r == config.ArmConfig(can_channel="can_right", gripper="linear_4310")
    assert [camera.name for camera in cfg.cameras] == ["top", "left", "right"]
    assert cfg.cameras[0] == config.CameraConfig(name="top", device=0, width=640, height=360, fps=30, rotate=0)
    assert cfg.cameras[1].device == "/dev/video1"
    assert cfg.cameras[2].rotate == 180
    assert cfg.task == "pick up the cup and place it on the plate"
    assert cfg.control_hz == 30.0
    assert type(cfg.control_hz) is float
    assert cfg.action_horizon == 48
    assert cfg.action_timeout_s == 2.0
    assert cfg.stale_after_s == 0.2
    assert cfg.max_stall_ticks == 240
    assert cfg.max_ee_speed_m_s == 0.85
    assert cfg.first_chunk_timeout_s == 45.0
    assert cfg.ramp_s == 1.5
    assert cfg.ramp_max_rad_s == 0.8
    assert cfg.max_gripper_step == 0.1
    assert cfg.trace_path == "/var/tmp/robots-trace.csv"


@pytest.mark.skipif(not EXAMPLE_PATH.exists(), reason="configs/yam.example.yaml not shipped")
def test_load_shipped_example_config() -> None:
    """The example file is what `python -m robots.run --config` loads; it must parse as written."""
    cfg = config.load(EXAMPLE_PATH)

    assert [camera.name for camera in cfg.cameras] == ["top", "left", "right"]
    assert cfg.task
    assert cfg.action_horizon == 48
    assert cfg.control_hz == 20.0
    assert cfg.action_timeout_s == 3.0


def test_load_accepts_os_pathlike_and_str(tmp_path: pathlib.Path) -> None:
    """`load(path)` takes `str | os.PathLike[str]`."""
    path = write_config(tmp_path, FULL_CONFIG)

    assert config.load(path) == config.load(str(path))


# Defaults


def test_tick_rate_and_refill_deadline_default_to_the_package_values(tmp_path: pathlib.Path) -> None:
    """`control_hz` (20) and `action_timeout_s` (3.0) are the package's numbers when the file has none."""
    cfg = load_variant(tmp_path, control_hz=None, action_timeout_s=None)

    assert cfg.control_hz == 20.0
    assert cfg.action_timeout_s == 3.0
    assert config.AppConfig.control_hz == 20.0
    assert config.AppConfig.action_timeout_s == 3.0


def test_refill_at_remaining_is_gone_and_fails_by_name(tmp_path: pathlib.Path) -> None:
    """The loop sizes its refill from its measured round trip; a file setting the key fails by name."""
    with pytest.raises(errors.ConfigError, match="'refill_at_remaining' is gone: the loop sizes its refill"):
        load_variant(tmp_path, refill_at_remaining=24)


def test_first_chunk_timeout_defaults_to_two_minutes(tmp_path: pathlib.Path) -> None:
    """`first_chunk_timeout_s` defaults to 120 s, room for a serve to boot, when absent."""
    cfg = load_variant(tmp_path, first_chunk_timeout_s=None)

    assert cfg.first_chunk_timeout_s == 120.0
    assert config.AppConfig.first_chunk_timeout_s == 120.0


def test_ramp_defaults_to_two_seconds_at_half_a_radian_per_second(tmp_path: pathlib.Path) -> None:
    """`ramp_s` and `ramp_max_rad_s` default to the YAM driver's ramp when absent."""
    cfg = load_variant(tmp_path, ramp_s=None, ramp_max_rad_s=None)

    assert cfg.ramp_s == 2.0
    assert cfg.ramp_max_rad_s == 0.5
    assert type(cfg.ramp_s) is float


def test_chunk_handoff_defaults(tmp_path: pathlib.Path) -> None:
    """The gripper clamp is 0.05 per tick, and no trace is written unless asked for; the end-effector
    cap has no default (it is in REQUIRED_KEYS)."""
    cfg = load_variant(tmp_path, max_gripper_step=None, trace_path=None)

    assert cfg.max_gripper_step == 0.05
    assert cfg.trace_path is None


def test_the_grip_force_cap_defaults_to_the_package_value_and_reads_from_the_file(
    tmp_path: pathlib.Path,
) -> None:
    """`grip_force_cap` is 0.65 (i2rt `gripper_eff` units) when the file has none, and a file may name
    another; 0 is accepted and turns the cap off."""
    assert load_variant(tmp_path, grip_force_cap=None).grip_force_cap == 0.65
    assert config.AppConfig.grip_force_cap == gripper_force.DEFAULT_CAP
    assert load_variant(tmp_path, grip_force_cap=3).grip_force_cap == 3.0
    assert load_variant(tmp_path, grip_force_cap=0).grip_force_cap == 0.0


def test_home_on_key_refused_is_on_unless_the_file_turns_it_off(tmp_path: pathlib.Path) -> None:
    """`home_on_key_refused` is True when the file has none (the arms go home after a refused key), and a
    file may turn it off; a value that is not a bool fails by name."""
    assert load_variant(tmp_path, home_on_key_refused=None).home_on_key_refused is True
    assert load_variant(tmp_path, home_on_key_refused=False).home_on_key_refused is False
    with pytest.raises(errors.ConfigError, match="home_on_key_refused"):
        load_variant(tmp_path, home_on_key_refused="yes")


def test_the_sync_threshold_is_the_clients_unless_the_file_sets_one(tmp_path: pathlib.Path) -> None:
    """`sync_above_ms` absent is None (`run.policy_config()` sizes it); a file may raise it, an int is widened."""
    assert load_variant(tmp_path, sync_above_ms=None).sync_above_ms is None
    assert load_variant(tmp_path, sync_above_ms=900).sync_above_ms == 900.0
    assert load_variant(tmp_path, sys0_sync_above_ms=None).sys0_sync_above_ms is None
    assert load_variant(tmp_path, sys0_sync_above_ms=1100).sys0_sync_above_ms == 1100.0


@pytest.mark.parametrize("bad", [0, -700, "900", True, math.nan])
def test_the_sync_threshold_rejects_a_value_that_is_not_a_positive_number(tmp_path: pathlib.Path, bad: Any) -> None:
    with pytest.raises(errors.ConfigError, match="config key 'sync_above_ms'"):
        load_variant(tmp_path, sync_above_ms=bad)
    with pytest.raises(errors.ConfigError, match="config key 'sys0_sync_above_ms'"):
        load_variant(tmp_path, sys0_sync_above_ms=bad)


# `None` is not in the list: load_variant reads it as "delete the key", which is the default path.
@pytest.mark.parametrize("bad", [-0.5, "1.5", True, math.nan, math.inf])
def test_the_grip_force_cap_rejects_a_value_of_the_wrong_type_or_range(tmp_path: pathlib.Path, bad: Any) -> None:
    """A cap that is not a finite number at or above zero fails at load naming the key."""
    with pytest.raises(errors.ConfigError, match="config key 'grip_force_cap'"):
        load_variant(tmp_path, grip_force_cap=bad)


def test_the_retired_gripper_close_snap_key_is_rejected_naming_its_replacement(tmp_path: pathlib.Path) -> None:
    """An arm section still carrying `gripper_close_snap` fails by that dotted name and names
    `grip_force_cap`, rather than as an unknown key, so an operator learns what replaced it."""
    with pytest.raises(errors.ConfigError, match=r"'follower_l.gripper_close_snap'.*'grip_force_cap'"):
        load_variant(tmp_path, follower_l={"can_channel": "can_left", "gripper_close_snap": 0.2})


def test_the_retired_joint_step_key_is_rejected_naming_its_replacement(tmp_path: pathlib.Path) -> None:
    """A 0.3.0 config still carrying `max_joint_step_rad` fails by that name and names `max_ee_speed_m_s`,
    rather than as an unknown key, so an operator learns what replaced it."""
    with pytest.raises(errors.ConfigError, match=r"'max_joint_step_rad'.*'max_ee_speed_m_s'"):
        load_variant(tmp_path, max_joint_step_rad=0.02)


@pytest.mark.parametrize("value", ["", 3, ["trace.csv"]])
def test_trace_path_must_be_a_non_empty_string_when_given(tmp_path: pathlib.Path, value: Any) -> None:
    """`trace_path` is a file name or null; an empty string or another type is rejected by name."""
    with pytest.raises(errors.ConfigError, match="'trace_path'"):
        load_variant(tmp_path, trace_path=value)


@pytest.mark.parametrize("key", ["trace_path", "inference_log_path"])
def test_a_leading_tilde_in_a_log_path_is_the_home_directory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    """A station config that writes `~/robots-trace.csv` means the home directory: nothing between the file and
    `open()` is a shell, and unexpanded the path would name a `~` directory that does not exist, so every Start
    would fail on FileNotFoundError. Absolute and relative paths pass unchanged."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    assert getattr(load_variant(tmp_path, **{key: "~/robots.log"}), key) == str(tmp_path / "home" / "robots.log")
    assert getattr(load_variant(tmp_path, **{key: "/var/log/robots.log"}), key) == "/var/log/robots.log"
    assert getattr(load_variant(tmp_path, **{key: "logs/robots.log"}), key) == "logs/robots.log"


def test_arm_gripper_defaults_to_linear_4310(tmp_path: pathlib.Path) -> None:
    """`ArmConfig.gripper` defaults to `linear_4310`, the gripper shipped with the YAM."""
    cfg = load_variant(tmp_path, follower_r={"can_channel": "can_right"})

    assert cfg.follower_r.gripper == "linear_4310"


def test_camera_fps_and_rotate_default(tmp_path: pathlib.Path) -> None:
    """`CameraConfig.fps` defaults to 30 and `rotate` to 0 when the entry omits them."""
    cfg = load_variant(tmp_path, cameras=[{"name": "top", "device": 0, "width": 640, "height": 360}])

    assert cfg.cameras[0].fps == 30
    assert cfg.cameras[0].rotate == 0


# Missing keys


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_missing_required_key_raises_naming_it(tmp_path: pathlib.Path, key: str) -> None:
    """A missing required key fails at load with ConfigError naming the key."""
    with pytest.raises(errors.ConfigError, match=f"'{key}'"):
        load_variant(tmp_path, **{key: None})


def test_missing_arm_can_channel_raises_with_dotted_path(tmp_path: pathlib.Path) -> None:
    """Nested required keys are named with their section, e.g. `follower_l.can_channel`."""
    with pytest.raises(errors.ConfigError, match="follower_l.can_channel"):
        load_variant(tmp_path, follower_l={"gripper": "linear_4310"})


@pytest.mark.parametrize("key", ["name", "device", "width", "height"])
def test_missing_camera_key_raises_with_index(tmp_path: pathlib.Path, key: str) -> None:
    """A camera entry missing a required key is named as `cameras[i].<key>`."""
    entry = dict(FULL_CONFIG["cameras"][0])
    del entry[key]

    with pytest.raises(errors.ConfigError, match=rf"cameras\[0\]\.{key}"):
        load_variant(tmp_path, cameras=[entry])


# Unknown keys


def test_unknown_top_level_key_raises_naming_it(tmp_path: pathlib.Path) -> None:
    """Unknown top-level keys fail hard; a typo must not fall back to a default."""
    with pytest.raises(errors.ConfigError, match="refill_at_remianing"):
        load_variant(tmp_path, refill_at_remianing=4)


def test_a_yam_file_runs_the_eef_head_unless_it_names_joint(tmp_path: pathlib.Path) -> None:
    """A YAM runs its model's EEF head, the tool poses solved to its joints by the client's IK, unless its file names
    `action_space: joint`, the model's own joints; a word that is neither fails by name."""
    assert load_variant(tmp_path).action_space == config.YAM_ACTION_SPACE == "eef"
    assert load_variant(tmp_path, action_space="joint").action_space == "joint"

    with pytest.raises(errors.ConfigError, match=r"config key 'action_space' must be one of"):
        load_variant(tmp_path, action_space="pose")


def test_unknown_arm_key_raises_naming_section(tmp_path: pathlib.Path) -> None:
    """Unknown keys inside `follower_*` are rejected with the section named."""
    with pytest.raises(errors.ConfigError, match="baud") as info:
        load_variant(tmp_path, follower_l={"can_channel": "can_left", "baud": 1000000})

    assert "follower_l" in str(info.value)


def test_unknown_camera_key_raises_naming_entry(tmp_path: pathlib.Path) -> None:
    """Unknown keys inside a camera entry are rejected with `cameras[i]` named."""
    entry = dict(FULL_CONFIG["cameras"][0], exposure=3)

    with pytest.raises(errors.ConfigError, match="exposure") as info:
        load_variant(tmp_path, cameras=[entry])

    assert "cameras[0]" in str(info.value)


@pytest.mark.parametrize(
    ("key", "value", "fragment"),
    [
        ("policy", {"host": "h", "checkpoint": "c"}, "the gateway is DREAM_GATEWAY_URL in the credentials file"),
        ("policy", {"checkpoint": "c"}, "the model is the top-level 'checkpoint'"),
        ("api_key", "sk-test", "it is DREAM_API_KEY in the credentials file"),
    ],
)
def test_the_keys_that_held_the_account_fail_by_name_with_where_each_went(
    tmp_path: pathlib.Path, key: str, value: Any, fragment: str
) -> None:
    """A file still carrying `policy` or `api_key`, which belong in the credentials file, is refused naming the key
    and where its value lives, never as an unknown key."""
    document = copy.deepcopy(FULL_CONFIG)
    document[key] = value

    with pytest.raises(errors.ConfigError, match=f"config key '{key}' is gone") as excinfo:
        config.load(write_config(tmp_path, document))
    assert fragment in str(excinfo.value)
    assert "unknown" not in str(excinfo.value)


def test_a_stall_budget_shorter_than_one_late_reply_reads_as_the_default_and_says_so(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="robots.config"):
        cfg = load_variant(tmp_path, max_stall_ticks=10)

    assert cfg.max_stall_ticks == config.MAX_STALL_TICKS
    assert "max_stall_ticks 10 holds the arms 0.33 s at 30 Hz, less than one late reply (action_timeout_s=2)" in (
        caplog.text
    )
    assert "config.yaml" in caplog.text
    assert load_variant(tmp_path, max_stall_ticks=60).max_stall_ticks == 60
    fast = load_variant(tmp_path, max_stall_ticks=10, control_hz=200.0)
    assert fast.max_stall_ticks == math.ceil(fast.action_timeout_s * 200.0) > config.MAX_STALL_TICKS


# Wrong types


@pytest.mark.parametrize("key", ["action_horizon", "max_stall_ticks"])
def test_bool_for_int_key_raises(tmp_path: pathlib.Path, key: str) -> None:
    """`True` is an int in Python but never a count; a bool where an int is expected is rejected."""
    with pytest.raises(errors.ConfigError, match=f"'{key}'"):
        load_variant(tmp_path, **{key: True})


@pytest.mark.parametrize(
    "key",
    [
        "control_hz",
        "action_timeout_s",
        "stale_after_s",
        "first_chunk_timeout_s",
        "ramp_s",
        "ramp_max_rad_s",
        "max_ee_speed_m_s",
        "max_gripper_step",
    ],
)
def test_bool_for_float_key_raises(tmp_path: pathlib.Path, key: str) -> None:
    """A bool where a positive number is expected is rejected by name."""
    with pytest.raises(errors.ConfigError, match=f"'{key}'"):
        load_variant(tmp_path, **{key: True})


@pytest.mark.parametrize("value", ["30", 2.5, [30]])
def test_non_int_for_int_key_raises(tmp_path: pathlib.Path, value: Any) -> None:
    """`action_horizon` must be an int; strings, floats and lists are wrong types."""
    with pytest.raises(errors.ConfigError, match="'action_horizon'"):
        load_variant(tmp_path, action_horizon=value)


@pytest.mark.parametrize("value", [0, -1])
def test_non_positive_action_horizon_raises(tmp_path: pathlib.Path, value: int) -> None:
    """A chunk has at least one row; `action_horizon` below 1 is a range error."""
    with pytest.raises(errors.ConfigError, match="'action_horizon'"):
        load_variant(tmp_path, action_horizon=value)


@pytest.mark.parametrize("value", [0, -5.0, "fast", None])
def test_non_positive_or_non_numeric_control_hz_raises(tmp_path: pathlib.Path, value: Any) -> None:
    """`control_hz` is a positive number; zero, negatives, strings and null are rejected."""
    document = copy.deepcopy(FULL_CONFIG)
    document["control_hz"] = value

    with pytest.raises(errors.ConfigError, match="'control_hz'"):
        config.load(write_config(tmp_path, document))


@pytest.mark.parametrize(
    "key", ["ramp_s", "ramp_max_rad_s", "first_chunk_timeout_s", "max_ee_speed_m_s", "max_gripper_step"]
)
@pytest.mark.parametrize("value", [0, -0.5, "slow", None])
def test_non_positive_or_non_numeric_ramp_value_raises(tmp_path: pathlib.Path, key: str, value: Any) -> None:
    """A ramp value, a clamp limit or the first-chunk deadline, when given, is a
    positive number; zero would be a step change, an arm that cannot move, or no time at all for the pool
    to answer."""
    document = copy.deepcopy(FULL_CONFIG)
    document[key] = value

    with pytest.raises(errors.ConfigError, match=f"'{key}'"):
        config.load(write_config(tmp_path, document))


def test_int_for_float_key_is_widened(tmp_path: pathlib.Path) -> None:
    """YAML `control_hz: 30` is an int on disk and a float in AppConfig."""
    cfg = load_variant(tmp_path, control_hz=30, stale_after_s=1)

    assert cfg.control_hz == 30.0
    assert type(cfg.stale_after_s) is float


def test_a_config_without_a_task_loads_with_an_empty_one(tmp_path: pathlib.Path) -> None:
    """A file `dreamcli setup` writes has no `task`: the operator sends it with the Start."""
    document = copy.deepcopy(FULL_CONFIG)
    del document["task"]
    assert config.load(write_config(tmp_path, document)).task == ""


@pytest.mark.parametrize("value", ["", 42, None])
def test_task_must_be_non_empty_str(tmp_path: pathlib.Path, value: Any) -> None:
    """`task`, when present, is the prompt sent with every observation; empty or non-string is
    rejected by name."""
    document = copy.deepcopy(FULL_CONFIG)
    document["task"] = value

    with pytest.raises(errors.ConfigError, match="'task'"):
        config.load(write_config(tmp_path, document))


def test_follower_must_be_a_mapping(tmp_path: pathlib.Path) -> None:
    """A `follower_*` section that is not a mapping is rejected by name."""
    with pytest.raises(errors.ConfigError, match="'follower_r'"):
        load_variant(tmp_path, follower_r="can_right")


def test_camera_device_must_be_int_or_str(tmp_path: pathlib.Path) -> None:
    """`CameraConfig.device` is an int index or a `/dev/...` path; anything else is rejected."""
    entry = dict(FULL_CONFIG["cameras"][0], device=1.5)

    with pytest.raises(errors.ConfigError, match=r"cameras\[0\]\.device"):
        load_variant(tmp_path, cameras=[entry])


def test_camera_serial_selects_a_realsense_camera(tmp_path: pathlib.Path) -> None:
    """`serial` instead of `device` names a RealSense camera; `device` stays None."""
    entry = {"name": "top", "serial": "260322275394", "width": 640, "height": 360}

    cfg = load_variant(tmp_path, cameras=[entry])

    assert cfg.cameras[0] == config.CameraConfig(name="top", serial="260322275394", width=640, height=360)
    assert cfg.cameras[0].device is None


def test_camera_with_both_device_and_serial_raises_naming_both(tmp_path: pathlib.Path) -> None:
    """A camera is an OpenCV device or a RealSense, never both; the error names both keys."""
    entry = dict(FULL_CONFIG["cameras"][0], serial="260322275394")

    with pytest.raises(errors.ConfigError, match=r"cameras\[0\]\.device") as info:
        load_variant(tmp_path, cameras=[entry])

    assert "cameras[0].serial" in str(info.value)


def test_camera_with_neither_device_nor_serial_raises_naming_both(tmp_path: pathlib.Path) -> None:
    """An entry with neither key is rejected with both alternatives named, so the fix is obvious."""
    entry = {"name": "top", "width": 640, "height": 360}

    with pytest.raises(errors.ConfigError, match=r"cameras\[0\]\.device") as info:
        load_variant(tmp_path, cameras=[entry])

    assert "cameras[0].serial" in str(info.value)


def test_camera_source_names_a_camera_class_of_your_own(tmp_path: pathlib.Path) -> None:
    """A camera's `source` names a class of the operator's own, for an SDK camera OpenCV cannot open: `module:Class`
    as given, `file.py:Class` made absolute from the config file's folder, as a robot of your own's driver is."""
    cameras = [
        {"name": "top", "source": "my_cams.py:Zed", "width": 640, "height": 360},
        {"name": "left", "source": "vendor.cams:Gige", "width": 640, "height": 360},
    ]

    cfg = load_variant(tmp_path, cameras=cameras)

    assert [camera.source for camera in cfg.cameras] == [
        f"{tmp_path.resolve() / 'my_cams.py'}:Zed",
        "vendor.cams:Gige",
    ]
    assert (cfg.cameras[0].device, cfg.cameras[0].serial) == (None, None)


def test_camera_with_a_class_and_a_device_raises_naming_both(tmp_path: pathlib.Path) -> None:
    """A camera is one of a device, a RealSense serial and a class of your own; the error names the keys set."""
    entry = dict(FULL_CONFIG["cameras"][0], source="my_cams:Zed")

    with pytest.raises(errors.ConfigError, match=r"cameras\[0\]\.device and cameras\[0\]\.source are set"):
        load_variant(tmp_path, cameras=[entry])


@pytest.mark.parametrize("value", ["my_cams", "my_cams:", "my_cams:2cam"])
def test_camera_source_must_name_a_class(tmp_path: pathlib.Path, value: str) -> None:
    entry = {"name": "top", "source": value, "width": 640, "height": 360}

    with pytest.raises(errors.ConfigError, match=r"cameras\[0\]\.source' must be package.module:Class"):
        load_variant(tmp_path, cameras=[entry])


@pytest.mark.parametrize("value", [260322275394, "", True])
def test_camera_serial_must_be_non_empty_str(tmp_path: pathlib.Path, value: Any) -> None:
    """Librealsense matches the serial as a string; an unquoted YAML number is rejected by name."""
    entry = {"name": "top", "serial": value, "width": 640, "height": 360}

    with pytest.raises(errors.ConfigError, match=r"cameras\[0\]\.serial"):
        load_variant(tmp_path, cameras=[entry])


def test_camera_rotate_must_be_a_right_angle(tmp_path: pathlib.Path) -> None:
    """Rotation is 0/90/180/270; any other value is a config error naming the entry."""
    entry = dict(FULL_CONFIG["cameras"][0], rotate=45)

    with pytest.raises(errors.ConfigError, match=r"cameras\[0\]\.rotate"):
        load_variant(tmp_path, cameras=[entry])


def test_camera_width_must_be_positive_int(tmp_path: pathlib.Path) -> None:
    """Frame size is declared in the embodiment, so a zero or bool width is rejected by name."""
    with pytest.raises(errors.ConfigError, match=r"cameras\[0\]\.width"):
        load_variant(tmp_path, cameras=[dict(FULL_CONFIG["cameras"][0], width=0)])
    with pytest.raises(errors.ConfigError, match=r"cameras\[0\]\.width"):
        load_variant(tmp_path, cameras=[dict(FULL_CONFIG["cameras"][0], width=True)])


# Cameras list


def test_duplicate_camera_names_raise(tmp_path: pathlib.Path) -> None:
    """Each camera name becomes an `image_window.<role>` key, so two cameras cannot share one."""
    cameras = [
        {"name": "top", "device": 0, "width": 640, "height": 360},
        {"name": "top", "device": 1, "width": 640, "height": 360},
    ]

    with pytest.raises(errors.ConfigError, match="top") as info:
        load_variant(tmp_path, cameras=cameras)

    assert "cameras" in str(info.value)


@pytest.mark.parametrize("value", [[], {"name": "top"}, "top"])
def test_cameras_must_be_a_non_empty_list(tmp_path: pathlib.Path, value: Any) -> None:
    """The policy needs at least one image window; `cameras` must be a non-empty list."""
    with pytest.raises(errors.ConfigError, match="'cameras'"):
        load_variant(tmp_path, cameras=value)


def test_camera_entry_must_be_a_mapping(tmp_path: pathlib.Path) -> None:
    """A list item that is not a mapping is named by its index."""
    with pytest.raises(errors.ConfigError, match=r"cameras\[1\]"):
        load_variant(tmp_path, cameras=[FULL_CONFIG["cameras"][0], "left"])


# File shape


@pytest.mark.parametrize("document", [None, [1, 2], "just a string"])
def test_file_must_be_a_mapping(tmp_path: pathlib.Path, document: Any) -> None:
    """An empty file, a list or a scalar at the top level is a ConfigError, not a crash elsewhere."""
    with pytest.raises(errors.ConfigError, match="mapping"):
        config.load(write_config(tmp_path, document))


def test_missing_file_raises_os_error(tmp_path: pathlib.Path) -> None:
    """rule 4: an unreadable path is an ordinary OSError from `open`; load() does not wrap it."""
    with pytest.raises(OSError):
        config.load(tmp_path / "absent.yaml")


# Policy window keys


def test_policy_window_defaults_to_the_station_values(tmp_path: pathlib.Path) -> None:
    """640x360 policy frames, nine of them, sampled at 5 fps, when the file says nothing."""
    cfg = load_variant(tmp_path)

    assert cfg.policy_image_size == (640, 360)
    assert cfg.obs_window == 9
    assert cfg.obs_fps == 5.0


def test_policy_window_keys_are_read(tmp_path: pathlib.Path) -> None:
    """`policy_image_size` is `WIDTHxHEIGHT` parsed to ints; `obs_window` int; `obs_fps` float."""
    cfg = load_variant(tmp_path, policy_image_size="1280x720", obs_window=4, obs_fps=10)

    assert cfg.policy_image_size == (1280, 720)
    assert cfg.obs_window == 4
    assert cfg.obs_fps == 10.0
    assert type(cfg.obs_fps) is float


@pytest.mark.parametrize("value", ["640", "640x0", "0x360", "640x360x3", "axb", "640 x 360", 640, [640, 360], None])
def test_policy_image_size_must_be_width_x_height(tmp_path: pathlib.Path, value: Any) -> None:
    """Anything but two positive ints joined by `x` is a ConfigError naming the key."""
    document = copy.deepcopy(FULL_CONFIG)
    document["policy_image_size"] = value

    with pytest.raises(errors.ConfigError, match="'policy_image_size'.*WIDTHxHEIGHT"):
        config.load(write_config(tmp_path, document))


@pytest.mark.parametrize("value", [0, -1, 2.5, True, "9"])
def test_obs_window_must_be_a_positive_int(tmp_path: pathlib.Path, value: Any) -> None:
    """The window holds at least one frame; a non-int or a count below 1 is a ConfigError."""
    with pytest.raises(errors.ConfigError, match="'obs_window'"):
        load_variant(tmp_path, obs_window=value)


@pytest.mark.parametrize("value", [0, -5.0, "5", True])
def test_obs_fps_must_be_a_positive_number(tmp_path: pathlib.Path, value: Any) -> None:
    """The sampling rate is a positive number; 0 would make the stride infinite."""
    with pytest.raises(errors.ConfigError, match="'obs_fps'"):
        load_variant(tmp_path, obs_fps=value)


# The live section


def test_live_section_is_optional(tmp_path: pathlib.Path) -> None:
    """No `live` key leaves `cfg.live` None: the station runs no agent."""
    assert load_variant(tmp_path).live is None


@pytest.mark.parametrize("url", ["", None, "wss://platform.example.com/live/robot"])
def test_a_live_url_fails_by_name_with_where_it_went(tmp_path: pathlib.Path, url: Any) -> None:
    """A file still carrying `live.url` (even empty), a key the credentials file replaced, is refused naming the key
    and the credentials file, never as an unknown key."""
    with pytest.raises(errors.ConfigError, match="config key 'live.url' is gone") as excinfo:
        load_variant(tmp_path, live={"url": url, "robot_id": "rbt_1"})
    assert "DREAM_API_URL in the credentials file" in str(excinfo.value)


def test_live_section_is_parsed_with_its_defaults(tmp_path: pathlib.Path) -> None:
    """`robot_id` set: every camera of the config, in the config's order."""
    cfg = load_variant(tmp_path, live={"robot_id": "rbt_1"})

    assert cfg.live == config.LiveConfig(robot_id="rbt_1", cameras=("top", "left", "right"))


def test_live_section_keys_are_read(tmp_path: pathlib.Path) -> None:
    """`cameras` keeps the file's order."""
    cfg = load_variant(tmp_path, live={"robot_id": "rbt_2", "cameras": ["right", "top"]})

    assert cfg.live == config.LiveConfig(robot_id="rbt_2", cameras=("right", "top"))


def test_an_older_files_encoding_and_record_keys_are_ignored_with_one_warning(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A station file with encoder settings or `record` in its `live` section still loads: they set nothing (the
    encoding is the platform's, every run's dream client records it), and one warning says so and names them."""
    section = {
        "robot_id": "rbt_3",
        "width_px": 320,
        "fps": 15,
        "bitrate_kbps": 700,
        "encoder": "libvpx",
        "record": False,
    }
    with caplog.at_level(logging.WARNING, logger="robots.config"):
        cfg = load_variant(tmp_path, live=section)

    assert cfg.live == config.LiveConfig(robot_id="rbt_3", cameras=("top", "left", "right"))
    warnings = [record.getMessage() for record in caplog.records if "no longer set anything" in record.getMessage()]
    assert len(warnings) == 1 and "live.record" in warnings[0] and "live.width_px" in warnings[0]


@pytest.mark.parametrize(
    ("section", "fragment"),
    [
        ({"robot_id": "r", "extra": 1}, "unknown keys ['extra'] under 'live'"),
        ({}, "'live.robot_id' is required"),
        ({"robot_id": "r", "cameras": ["nope"]}, "cameras the config does not have: ['nope']"),
        ({"robot_id": "r", "cameras": []}, "'live.cameras' must be a non-empty list"),
    ],
)
def test_bad_live_values_raise_naming_the_key(tmp_path: pathlib.Path, section: Any, fragment: str) -> None:
    """Each `live` rule fails at load with the dotted key and the rule in the message."""
    with pytest.raises(errors.ConfigError) as excinfo:
        load_variant(tmp_path, live=section)
    assert fragment in str(excinfo.value)


def test_live_must_be_a_mapping(tmp_path: pathlib.Path) -> None:
    with pytest.raises(errors.ConfigError, match="config key 'live' must be a mapping"):
        load_variant(tmp_path, live=["wss://x"])


@pytest.mark.skipif(not EXAMPLE_PATH.exists(), reason="configs/yam.example.yaml not shipped")
def test_shipped_example_config_has_the_live_agent_off() -> None:
    """The example ships its `live` section commented out, so a copied config runs no agent until the robot's id is
    filled in."""
    assert config.load(EXAMPLE_PATH).live is None


# --- exposure --------------------------------------------------------------------------------------


def load_with(tmp_path: pathlib.Path, overrides: dict[str, Any]) -> config.AppConfig:
    """Loads the full document with `overrides` laid over its top level."""
    return config.load(write_config(tmp_path, {**copy.deepcopy(FULL_CONFIG), **overrides}))


def test_exposure_is_managed_with_the_station_defaults_when_the_section_is_absent(tmp_path: Any) -> None:
    cfg = load_with(tmp_path, {})
    assert cfg.exposure == config.ExposureConfig(
        managed=True, target_median=90.0, limit_us=None, seed_us=None, smooth_idle=True
    )


def test_exposure_section_is_parsed_and_checked(tmp_path: Any) -> None:
    cfg = load_with(
        tmp_path, {"exposure": {"managed": True, "target_median": 120, "limit_us": 18000, "seed_us": 6000}}
    )
    assert cfg.exposure == config.ExposureConfig(managed=True, target_median=120.0, limit_us=18000.0, seed_us=6000.0)
    assert load_with(tmp_path, {"exposure": {"managed": False}}).exposure.managed is False
    assert load_with(tmp_path, {"exposure": {"limit_us": None}}).exposure.limit_us is None
    assert load_with(tmp_path, {"exposure": {"smooth_idle": True}}).exposure.smooth_idle is True
    assert load_with(tmp_path, {"exposure": {"smooth_idle": False}}).exposure.smooth_idle is False
    for bad, message in [
        ({"exposure": "auto"}, "must be a mapping"),
        ({"exposure": {"gain": 1}}, "unknown"),
        ({"exposure": {"smooth_idle": "yes"}}, "smooth_idle"),
        ({"exposure": {"target_median": 0}}, "target_median"),
        ({"exposure": {"target_median": 255}}, "within 1 to 254"),
        ({"exposure": {"limit_us": -1}}, "limit_us"),
        ({"exposure": {"seed_us": 20000, "limit_us": 18000}}, "must not exceed"),
        ({"exposure": {"managed": "yes"}}, "managed"),
    ]:
        with pytest.raises(errors.ConfigError, match=message):
            load_with(tmp_path, bad)
