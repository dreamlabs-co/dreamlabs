"""A fake of LeRobot's Feetech bus for the SO-101 tests: the calls `robots.feetech` makes, with LeRobot 0.6's
signatures (keyword-only `normalize` and `num_retry`), and servos that go to their goal at once when energized.

A test opens the fake through #FakeFeetechFactory, which `robots.feetech.FeetechChain` calls as it calls
`robots.feetech.open_bus`.
"""

import dataclasses
import json
import pathlib
import types
from typing import Any

from robots import errors, feetech, motor_arm

## A calibration as `lerobot-calibrate` writes it for an SO-101 follower: name, id, homing offset, range (ticks).
CALIBRATION = {
    "shoulder_pan": (1, -1020, 760, 3310),
    "shoulder_lift": (2, 512, 820, 3220),
    "elbow_flex": (3, -77, 860, 3090),
    "wrist_flex": (4, 1301, 810, 3260),
    "wrist_roll": (5, -43, 0, 4095),
    "gripper": (6, 905, 2000, 3400),
}

HELD = {name: feetech.Calibration(i, 0, offset, low, high) for name, (i, offset, low, high) in CALIBRATION.items()}


def write_calibration(path: pathlib.Path, **changes: tuple[int, int, int, int]) -> pathlib.Path:
    """Writes #CALIBRATION, with any entry replaced, as a LeRobot calibration json.

    @param path The file to write.
    @param changes Entries to replace, by servo name: `(id, homing_offset, range_min, range_max)`.
    @return `path`.
    """
    entries = {**CALIBRATION, **changes}
    document = {
        name: {"id": i, "drive_mode": 0, "homing_offset": offset, "range_min": low, "range_max": high}
        for name, (i, offset, low, high) in entries.items()
    }
    path.write_text(json.dumps(document, indent=4), encoding="utf-8")
    return path


class FakeFeetechBus:
    """One arm's servos: their registers, their positions in ticks, and every write a test may check.

    Attributes:
    - port: The serial device it was opened on.
    - names: The servos, base to gripper.
    - held: The calibration the servos hold (EEPROM), by name; a test changes it to plant a mismatch.
    - present: Each servo's position, ticks.
    - torque: Whether each servo is energized.
    - registers: The last value written to each `(register, servo)`.
    - goals: Every `Goal_Position` sync write, in order.
    - fail_reads: Position reads still to fail (ConnectionError, as LeRobot raises a lost status packet).
    - connected: Whether the port is open.
    - torque_log: `"on"` and `"off"` in the order the servos were switched: `"off"` once a switch-off reaches the
      gripper, the last servo, whether it went servo by servo or all at once.
    - temperature: Each servo's `Present_Temperature`, degrees C.
    - temperature_reads: Temperature sync reads so far.
    - floor: The lowest ticks a servo can reach, by name: an object in a closing gripper.
    - overload: Servos in overload protection: they answer every write with LeRobot's overload error and keep their
      torque.
    """

    def __init__(
        self, port: str, joints: tuple[motor_arm.Joint, ...], calibration: dict[str, feetech.Calibration] | None
    ) -> None:
        """Opens the fake with the servos holding the file's calibration (#HELD when opened without one), every
        servo in its middle pose."""
        self.port = port
        self.names = [joint.name for joint in joints]
        self.held = {name: dataclasses.replace(entry) for name, entry in (calibration or HELD).items()}
        self.present = dict.fromkeys(self.names, feetech.CENTER_TICK)
        self.torque = dict.fromkeys(self.names, False)
        self.registers: dict[tuple[str, str], int] = {}
        self.goals: list[dict[str, int]] = []
        self.fail_reads = 0
        self.connected = True
        self.configured = False
        self.torque_log: list[str] = []
        self.temperature = dict.fromkeys(self.names, 35)
        self.temperature_reads = 0
        self.floor: dict[str, int] = {}
        self.overload: set[str] = set()

    def read_calibration(self) -> dict[str, Any]:
        """What the servos hold, as LeRobot reads it (drive mode always 0)."""
        assert self.connected
        return {name: types.SimpleNamespace(**dataclasses.asdict(entry)) for name, entry in self.held.items()}

    def configure_motors(self) -> None:
        """LeRobot's register setup; only recorded."""
        assert self.connected and not any(self.torque.values()), "LeRobot configures with the servos off"
        self.configured = True

    def write(self, data_name: str, motor: str, value: int, *, normalize: bool = True, num_retry: int = 0) -> None:
        """One register of one servo."""
        assert self.connected and motor in self.names and isinstance(value, int), (data_name, motor, value)
        assert not self.torque[motor], f"{data_name} is written with {motor} off"
        self.registers[(data_name, motor)] = value

    def sync_read(
        self, data_name: str, motors: list[str] | None = None, *, normalize: bool = True, num_retry: int = 0
    ) -> dict[str, int]:
        """Every position or temperature, raw; a planted position failure raises as LeRobot does."""
        assert self.connected and normalize is False
        if data_name == "Present_Temperature":
            self.temperature_reads += 1
            return {name: self.temperature[name] for name in (motors or self.names)}
        assert data_name == "Present_Position"
        if self.fail_reads:
            self.fail_reads -= 1
            raise ConnectionError(f"Failed to sync read '{data_name}' after {num_retry + 1} tries. [TxRxResult] lost")
        return {name: self.present[name] for name in (motors or self.names)}

    def sync_write(
        self, data_name: str, values: dict[str, int], *, normalize: bool = True, num_retry: int = 0
    ) -> None:
        """Every goal, raw ticks; an energized servo is there at once, or at its floor."""
        assert self.connected and data_name == "Goal_Position" and normalize is False
        assert all(isinstance(value, int) and 0 <= value < feetech.TICKS_PER_TURN for value in values.values())
        self.goals.append(dict(values))
        for name, value in values.items():
            if self.torque[name]:
                self.present[name] = max(value, self.floor.get(name, value))

    def enable_torque(self, motors: list[str] | None = None, num_retry: int = 0) -> None:
        """Energizes every servo."""
        assert self.connected and motors is None
        self.torque = dict.fromkeys(self.names, True)
        self.torque_log.append("on")

    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """Switches every servo off, or the one named; one in overload raises as LeRobot does, those before it off."""
        assert self.connected
        names = self.names if motors is None else [motors] if isinstance(motors, str) else motors
        for name in names:
            if name in self.overload:
                raise RuntimeError(
                    f"Failed to write 'Torque_Enable' on id_={self.held[name].id} with '0' after {num_retry + 1} "
                    "tries. [RxPacketError] Overload error!"
                )
            self.torque[name] = False
        if names[-1] == self.names[-1]:
            self.torque_log.append("off")

    def disconnect(self, disable_torque: bool = True) -> None:
        """Closes the port; robots.feetech switches the servos off itself first."""
        assert self.connected and disable_torque is False
        self.connected = False


class FakeFeetechFactory:
    """Opens a #FakeFeetechBus per port, as `robots.feetech.open_bus` opens LeRobot's.

    Attributes:
    - buses: Every bus opened, by port, the newest last.
    - present: Ticks each new bus's servos start at, by servo name.
    - held: Calibration entries each new bus's servos hold instead of the file's, by servo name.
    - eeprom: Calibration entries one port's servos hold instead, by port, then servo name (an arm calibrated apart).
    - missing: Servo ids that do not answer on a port, by port: opening it fails as LeRobot's handshake does.
    """

    def __init__(self) -> None:
        self.buses: dict[str, list[FakeFeetechBus]] = {}
        self.present: dict[str, int] = {}
        self.held: dict[str, feetech.Calibration] = {}
        self.eeprom: dict[str, dict[str, feetech.Calibration]] = {}
        self.missing: dict[str, list[int]] = {}

    def __call__(
        self, port: str, joints: tuple[motor_arm.Joint, ...], calibration: dict[str, feetech.Calibration] | None
    ) -> FakeFeetechBus:
        if self.missing.get(port):
            raise errors.ArmError(f"servos {self.missing[port]} on {port} did not answer while the others did")
        bus = FakeFeetechBus(port, joints, calibration)
        bus.present.update(self.present)
        bus.held.update(self.held)
        bus.held.update(self.eeprom.get(port, {}))
        self.buses.setdefault(port, []).append(bus)
        return bus

    def bus(self, port: str) -> FakeFeetechBus:
        """The newest bus opened on a port."""
        return self.buses[port][-1]


class CalibratingBus:
    """One arm's servos as calibration sees them: each reads its true angle less its homing offset, as an STS3215 does
    (`Present_Position = actual - Homing_Offset`), so writing a homing offset moves every later reading.

    Attributes:
    - actual: Each servo's raw angle, ticks: where the hand has put the joint.
    - registers: The value each `(register, servo)` holds; the homing offset and range start as #HELD's.
    - torque_off: Whether disable_torque() was called.
    - connected: Whether the port is open.
    """

    def __init__(self, actual: dict[str, int]) -> None:
        self.actual = dict(actual)
        self.registers: dict[tuple[str, str], int] = {}
        for name, entry in HELD.items():
            self.registers[("Homing_Offset", name)] = entry.homing_offset
            self.registers[("Min_Position_Limit", name)] = entry.range_min
            self.registers[("Max_Position_Limit", name)] = entry.range_max
        self.torque_off = False
        self.connected = True

    def disable_torque(self, motors: Any = None, num_retry: int = 0) -> None:
        self.torque_off = True

    def write(self, data_name: str, motor: str, value: int, *, normalize: bool = True, num_retry: int = 0) -> None:
        assert self.connected and self.torque_off, f"{data_name} written with the servos on"
        assert isinstance(value, int), (data_name, motor, value)
        if data_name == "Homing_Offset":
            assert -2047 <= value <= 2047, value
        self.registers[(data_name, motor)] = value

    def sync_read(self, data_name: str, motors: Any = None, *, normalize: bool = True, num_retry: int = 0) -> dict:
        assert self.connected and data_name == "Present_Position" and normalize is False
        return {
            name: (ticks - self.registers[("Homing_Offset", name)]) % feetech.TICKS_PER_TURN
            for name, ticks in self.actual.items()
        }

    def read_calibration(self) -> dict[str, Any]:
        return {
            name: types.SimpleNamespace(
                id=entry.id,
                drive_mode=0,
                homing_offset=self.registers[("Homing_Offset", name)],
                range_min=self.registers[("Min_Position_Limit", name)],
                range_max=self.registers[("Max_Position_Limit", name)],
            )
            for name, entry in HELD.items()
        }

    def held(self) -> dict[str, feetech.Calibration]:
        """What the servos hold now, as calibration entries."""
        return {
            name: feetech.Calibration(int(e.id), 0, int(e.homing_offset), int(e.range_min), int(e.range_max))
            for name, e in self.read_calibration().items()
        }

    def disconnect(self, disable_torque: bool = True) -> None:
        assert disable_torque is False
        self.connected = False
