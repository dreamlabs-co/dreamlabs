"""Pins `robots.startup_failure` and its call in `robots.station.main()`.

A station whose start fails writes one line saying which device, what error and what fix, ahead of its traceback,
for `dreamcli robot up` to show (its test is in test_cli.py); the station still exits with its own error.
"""

import logging
import pathlib
from typing import Any

import conftest
import pytest

from robots import cameras, cli, config, errors, startup_failure, station

## A D405 as a YAM station file names it, by serial.
SERIAL = "260322275394"

## The words `robots.cameras.RealSenseSource` raises for that camera on a USB2 link.
USB2_ERROR = (
    f"Couldn't resolve requests: RealSense camera {SERIAL} is on a USB 2.1 link, too slow for 1280x720 at 30 fps; "
    f"{cameras.USB3_FIX}"
)


def _config(*, live: bool = True) -> config.AppConfig:
    """The test config with a RealSense as `top` and, unless `live` is False, a `live` section naming the robot."""
    top = config.CameraConfig(name="top", serial=SERIAL, width=1280, height=720)
    section = config.LiveConfig(robot_id="rbt_0123456789abcdef", cameras=("top",)) if live else None
    return conftest.make_app_config(cameras=[top], live=section)


def _marker_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The could-not-start lines the station logged."""
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(startup_failure.MARKER)]


def test_a_camera_on_usb2_is_named_with_its_role_and_the_usb3_fix() -> None:
    said = startup_failure.describe(RuntimeError(USB2_ERROR), _config())

    assert said["device"] == f"the top camera (RealSense {SERIAL})"
    assert said["error"] == USB2_ERROR
    assert "USB3 port" in said["fix"] and "dreamcli robot up" in said["fix"]


def test_other_failures_name_the_part_of_the_robot_their_error_is_about() -> None:
    cfg = _config()

    camera = startup_failure.describe(errors.CameraError("camera 'top' could not open device 4"), cfg)
    arm = startup_failure.describe(errors.ArmError("arm 'left' did not answer on can_left"), cfg)
    other = startup_failure.describe(OSError(98, "Address already in use"), cfg)

    assert camera["device"] == f"the top camera (RealSense {SERIAL})" and "cable" in camera["fix"]
    assert arm["device"] == "an arm" and "emergency stop" in arm["fix"]
    assert other["device"] == "the station" and "dreamcli robot log" in other["fix"]
    assert other["error"] == "[Errno 98] Address already in use"


def test_log_writes_one_line_with_the_device_the_error_and_the_fix(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="robots.startup_failure")

    startup_failure.log(RuntimeError(USB2_ERROR), _config())

    said = startup_failure.describe(RuntimeError(USB2_ERROR), _config())
    line = f"{startup_failure.MARKER}{said['device']}: {said['error']} -- Fix: {said['fix']}"
    assert _marker_lines(caplog) == [line]


def test_find_answers_the_last_could_not_start_line_or_none() -> None:
    lines = [
        "12:00 INFO robots.station: opening the arms",
        f"12:00 ERROR robots.startup_failure: {startup_failure.MARKER}an arm: first Fix: one",
        f"12:01 ERROR robots.startup_failure: {startup_failure.MARKER}the station: second Fix: two",
        "Traceback (most recent call last):",
    ]
    assert startup_failure.find(lines) == "the station: second Fix: two"
    assert startup_failure.find(lines[:1]) is None


def _run_main(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, serve: Any) -> None:
    monkeypatch.setattr(station, "serve", serve)
    monkeypatch.setattr(config, "load", lambda path: _config())
    monkeypatch.setenv(cli.HOME_ENV, str(tmp_path / "home"))
    station.main(["--config", "station.yaml"])


def test_a_station_that_fails_before_it_is_ready_says_why_then_exits_with_its_own_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A YAM station: the camera's RuntimeError ended `robot up` within seconds; the log now opens the failure with
    which
    device, the error and the fix, and the error still reaches the log and the exit code as it did."""
    caplog.set_level(logging.INFO, logger="robots.startup_failure")

    async def failing(cfg: config.AppConfig, **kwargs: Any) -> None:
        raise RuntimeError(USB2_ERROR)

    with pytest.raises(RuntimeError, match="Couldn't resolve requests"):
        _run_main(monkeypatch, tmp_path, failing)
    assert len(_marker_lines(caplog)) == 1
    assert _marker_lines(caplog)[0].startswith(f"{startup_failure.MARKER}the top camera (RealSense {SERIAL}): ")

    async def refused(cfg: config.AppConfig, **kwargs: Any) -> None:
        raise errors.ArmError("arm 'left' did not answer on can_left")

    with pytest.raises(SystemExit) as exited:
        _run_main(monkeypatch, tmp_path, refused)
    assert exited.value.code == 1
    assert _marker_lines(caplog)[-1].startswith(f"{startup_failure.MARKER}an arm: ")


def test_a_camera_librealsense_cannot_find_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    """A station file naming a camera that is not plugged in: librealsense's bare "No device connected" would only
    say "the station", so the open says which camera, as setup does, same type."""
    from robots import run

    def absent(cfg: config.CameraConfig, *args: Any, **kwargs: Any) -> cameras.CameraReader:
        raise RuntimeError("No device connected")

    monkeypatch.setattr(cameras.CameraReader, "from_config", absent)
    cfg = _config()
    with pytest.raises(RuntimeError) as info:
        run.make_cameras(cfg)
    assert str(info.value) == f"the top camera {SERIAL} did not open: No device connected"
    said = startup_failure.describe(info.value, cfg)
    assert said["device"] == f"the top camera (RealSense {SERIAL})"
    assert said["fix"].startswith("Check the camera's cable and USB port")


def test_a_station_that_was_ready_says_nothing_of_a_failed_start_whatever_ends_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Only a start that failed is a startup failure: a station that came up, or was stopped while it started, is
    not called one."""
    caplog.set_level(logging.INFO, logger="robots.startup_failure")

    async def ready_then_failing(cfg: config.AppConfig, **kwargs: Any) -> None:
        kwargs["ready"].set()
        raise RuntimeError("after the start")

    async def stopped(cfg: config.AppConfig, **kwargs: Any) -> None:
        return None

    with pytest.raises(RuntimeError, match="after the start"):
        _run_main(monkeypatch, tmp_path, ready_then_failing)
    _run_main(monkeypatch, tmp_path, stopped)

    assert _marker_lines(caplog) == []
