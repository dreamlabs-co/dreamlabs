"""Behaviour tests for `robots.motor_mode` against a stand-in python-can: the frames that go out, the replies that
are taken, and the two command-line reports."""

import io
import struct
import sys
import types
from typing import Any

import pytest

from robots import motor_mode


class FakeBus:
    """A socketcan bus whose motors answer from a `modes` table: a parameter read answers the motor's mode, a write
    updates it and answers the new value, a disable and a store get no reply, and a motor absent from the table says
    nothing. A stray status frame precedes every real reply, so a caller that took the first frame would be wrong."""

    def __init__(self, modes: dict[int, int], log: list[Any]) -> None:
        self.modes, self.log, self.queued = modes, log, []  # type: dict[int, int], list[Any], list[Any]

    def __enter__(self) -> "FakeBus":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.log.append("closed")

    def send(self, message: Any) -> None:
        data = bytes(message.data)
        self.log.append((message.arbitration_id, data))
        if message.arbitration_id != motor_mode.DM_PARAMETER_ID:
            return  # a special command to a motor's own id: no reply
        motor, command = data[0], data[2]
        if motor not in self.modes or command == motor_mode.DM_STORE_PARAMETERS:
            return
        if command == motor_mode.DM_WRITE_PARAMETER:
            self.modes[motor] = struct.unpack("<I", data[4:])[0]
        self.queued.append(types.SimpleNamespace(arbitration_id=0x10 + motor, data=bytes([0x17, 0, 0, 0, 0, 0, 0, 0])))
        self.queued.append(
            types.SimpleNamespace(arbitration_id=0x10 + motor, data=data[:4] + struct.pack("<I", self.modes[motor]))
        )

    def recv(self, timeout: float) -> Any:
        return self.queued.pop(0) if self.queued else None


def install(monkeypatch: pytest.MonkeyPatch, modes: dict[int, int]) -> list[Any]:
    """Puts the stand-in in place of python-can for `link`, and returns its frame log."""
    log: list[Any] = []

    def bus(interface: str, channel: str) -> FakeBus:
        assert (interface, channel) == ("socketcan", "can_follower_r")
        return FakeBus(modes, log)

    monkeypatch.setitem(sys.modules, "can", types.SimpleNamespace(Message=Message, Bus=bus))
    monkeypatch.setattr(motor_mode.time, "sleep", lambda s: None)
    return log


class Message:
    def __init__(self, arbitration_id: int, data: bytes = b"", is_extended_id: bool = False) -> None:
        self.arbitration_id, self.data = arbitration_id, data


def test_read_modes_sends_one_parameter_read_per_motor_and_takes_the_reply_by_its_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = install(monkeypatch, {1: 1, 2: 1, 7: 2})
    assert motor_mode.read_modes("can_follower_r", [1, 2, 3, 7]) == [(1, 1), (2, 1), (3, None), (7, 2)]
    assert log == [(0x7FF, bytes([motor, 0, 0x33, 10, 0, 0, 0, 0])) for motor in (1, 2, 3, 7)] + ["closed"]


def test_set_mit_disables_reads_writes_only_what_is_not_mit_and_stores(monkeypatch: pytest.MonkeyPatch) -> None:
    modes = {6: 1, 7: 2}
    log = install(monkeypatch, modes)
    assert motor_mode.set_mit("can_follower_r", [6, 7, 5]) == [(6, 1, 1), (7, 2, 1), (5, None, None)]
    assert modes[7] == 1
    disable = bytes([0xFF] * 7 + [0xFD])
    store = lambda m: (0x7FF, bytes([m, 0, 0xAA, 0x01, 0, 0, 0, 0]))  # noqa: E731
    read = lambda m: (0x7FF, bytes([m, 0, 0x33, 10, 0, 0, 0, 0]))  # noqa: E731
    assert log == [
        (6, disable), read(6), store(6),
        (7, disable), read(7), (0x7FF, bytes([7, 0, 0x55, 10, 1, 0, 0, 0])), store(7),
        (5, disable), read(5),
        "closed",
    ]  # fmt: skip


def test_check_reports_every_motor_and_exits_1_on_a_wrong_mode_or_silence(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, {1: 1, 7: 2})
    report = io.StringIO()
    assert motor_mode.check(["can_follower_r"], [1, 7], out=report) == 1
    text = report.getvalue()
    assert "motor 1: mit  [ok]" in text and "motor 7: pos_speed  [NOT MIT]" in text and "set-mit" in text

    report = io.StringIO()
    assert motor_mode.check(["can_follower_r"], [1, 3], out=report) == 1
    assert "motor 3: no answer  [is the arm powered on?]" in report.getvalue()

    report = io.StringIO()
    assert motor_mode.check(["can_follower_r"], [1], out=report) == 0
    assert "every motor is in MIT mode" in report.getvalue()


def test_switch_reports_the_change_and_the_power_cycle_and_exits_1_when_a_motor_is_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(monkeypatch, {6: 1, 7: 2})
    report = io.StringIO()
    assert motor_mode.switch(["can_follower_r"], [6, 7], out=report) == 0
    text = report.getvalue()
    assert "motor 6: already mit, stored" in text and "motor 7: pos_speed -> mit, stored" in text
    assert "power-cycle the arms" in text

    report = io.StringIO()
    assert motor_mode.switch(["can_follower_r"], [7, 4], out=report) == 1
    assert "motor 4: no answer -> no answer  [FAILED]" in report.getvalue()


def test_check_reports_a_link_that_cannot_be_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    def bus(interface: str, channel: str) -> Any:
        raise OSError("No such device")

    monkeypatch.setitem(sys.modules, "can", types.SimpleNamespace(Message=Message, Bus=bus))
    report = io.StringIO()
    assert motor_mode.check(["can_follower_l"], [7], out=report) == 1
    assert "could not read the link: No such device" in report.getvalue()


def test_mode_names() -> None:
    assert [motor_mode.mode_name(m) for m in (1, 2, 3, 4, 9, None)] == [
        "mit", "pos_speed", "speed", "torque_pos", "unknown(9)", "no answer",
    ]  # fmt: skip
