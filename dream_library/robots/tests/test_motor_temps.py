"""The owner's motor temperature limits: the station file's `motor_temp_warn_c` and `motor_temp_stop_c`
(`robots.config`), applied when an arm opens (`robots.motor_arm.with_temp_limits`) and set by `dreamcli robot temp`
(`robots.cli.robot_temp`)."""

import io
import pathlib
import shutil

import pytest

from robots import cli, config, errors, makers, motor_arm

CONFIGS = pathlib.Path(__file__).resolve().parents[1] / "configs"


def station(tmp_path: pathlib.Path, body: str = "makers", extra: str = "") -> cli.Paths:
    """A station directory holding the shipped example of `body` as its station file, with `extra` lines added."""
    paths = cli.Paths(tmp_path)
    shutil.copy(CONFIGS / f"{body}.example.yaml", paths.config)
    if extra:
        paths.config.write_text(paths.config.read_text(encoding="utf-8") + extra, encoding="utf-8")
    return paths


def test_the_station_file_raises_or_lowers_the_bodys_limits_and_leaves_them_when_unset(tmp_path: pathlib.Path) -> None:
    """Unset keeps the body's own limits; set, either may go above or below them."""
    plain = config.load(station(tmp_path).config)
    assert motor_arm.with_temp_limits(makers.ARM, plain.follower_l) == makers.ARM

    cfg = config.load(station(tmp_path, extra="motor_temp_warn_c: 70\nmotor_temp_stop_c: 80\n").config)
    spec = motor_arm.with_temp_limits(makers.ARM, cfg.follower_r)
    assert (spec.temp_warn_c, spec.temp_stop_c) == (70.0, 80.0)

    cfg = config.load(station(tmp_path, extra="motor_temp_warn_c: 40\n").config)
    spec = motor_arm.with_temp_limits(makers.ARM, cfg.follower_l)
    assert (spec.temp_warn_c, spec.temp_stop_c) == (40.0, makers.ARM.temp_stop_c)


def test_a_warning_temperature_at_or_above_the_stop_temperature_is_refused(tmp_path: pathlib.Path) -> None:
    """Both in the file, or one against the body's own other limit."""
    with pytest.raises(
        errors.ConfigError, match=r"'motor_temp_warn_c' \(70\) must be below 'motor_temp_stop_c' \(60\)"
    ):
        config.load(station(tmp_path, extra="motor_temp_warn_c: 70\nmotor_temp_stop_c: 60\n").config)
    cfg = config.load(station(tmp_path, extra="motor_temp_warn_c: 90\n").config)
    with pytest.raises(errors.ConfigError, match=r"warning temperature \(90 C\) must be below the stop temperature"):
        motor_arm.with_temp_limits(makers.ARM, cfg.follower_l)


def test_a_yam_station_file_with_the_limits_is_refused(tmp_path: pathlib.Path) -> None:
    """dream watches no YAM motor temperature, so a limit there would do nothing."""
    with pytest.raises(errors.ConfigError, match="does not watch the motor temperatures of 'body: yam'"):
        config.load(station(tmp_path, body="yam", extra="motor_temp_stop_c: 70\n").config)


def test_robot_temp_prints_the_limits_then_saves_new_ones_in_the_station_file(tmp_path: pathlib.Path) -> None:
    """No flag only prints; a flag replaces the key's line or adds one, and the next load runs on it."""
    paths = station(tmp_path)
    before = paths.config.read_text(encoding="utf-8")
    out = io.StringIO()
    assert cli.robot_temp(paths, None, None, out=out) == 0
    assert "warn at 55 C" in out.getvalue() and "stop at 65 C" in out.getvalue()
    assert paths.config.read_text(encoding="utf-8") == before

    assert cli.robot_temp(paths, 60.0, 72.5, out=io.StringIO()) == 0
    assert cli.robot_temp(paths, 58.0, None, out=io.StringIO()) == 0
    text = paths.config.read_text(encoding="utf-8")
    assert text.startswith(before.rstrip("\n")) and text.count("motor_temp_warn_c") == 1
    cfg = config.load(paths.config)
    assert (cfg.follower_l.temp_warn_c, cfg.follower_l.temp_stop_c) == (58.0, 72.5)


def test_robot_temp_refuses_a_bad_limit_and_leaves_the_file_as_it_was(tmp_path: pathlib.Path) -> None:
    """A warning above the stop temperature, or a YAM, says why and changes nothing."""
    paths = station(tmp_path)
    before = paths.config.read_text(encoding="utf-8")
    out = io.StringIO()
    assert cli.robot_temp(paths, 80.0, None, out=out) == 1
    assert "must be below the stop temperature (65 C)" in out.getvalue()
    assert paths.config.read_text(encoding="utf-8") == before
    assert not paths.config.with_name(paths.config.name + ".tmp").exists()

    (tmp_path / "yam").mkdir()
    yam = station(tmp_path / "yam", body="yam")
    out = io.StringIO()
    assert cli.robot_temp(yam, None, None, out=out) == 1
    assert "does not watch the motor temperatures of 'body: yam'" in out.getvalue()
