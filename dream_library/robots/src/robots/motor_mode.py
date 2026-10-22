"""The firmware control mode of a YAM arm's DM motors: read it, and switch a motor to MIT.

Every motor on a YAM or OpenYAM arm is a Damiao (DM) motor whose firmware runs in one of four control modes, held in
its `CTRL_MODE` register (#CTRL_MODE_REGISTER): MIT (1), position-speed (2), speed (3) or torque-position (4).
The station drives the arms through i2rt, which speaks MIT and nothing else, so a motor in any other mode answers
every status request and moves for none of them. New OpenYAM arms often ship with the gripper motor (id 7) in
position-speed mode: the wizard's gripper check then reports a gripper that did not move, and a station start
calibrates against a gripper that stands still. This module is the fix `dreamcli motors` runs.

Two operations, each on one link and a list of motor ids, each on a socket of its own opened and closed here so no
reply is left queued for anyone else to misread, the way the wizard's joint reading works:

- read_modes(): one parameter read (#DM_READ_PARAMETER) of #CTRL_MODE_REGISTER per motor, taken only when the reply
  names the motor and the register asked for. Changes nothing; works on a switched-off motor.
- set_mit(): per motor, switch it off (the DM disable command, so the write lands on an idle motor), read the mode,
  write MIT (#DM_WRITE_PARAMETER) when it is anything else, and send the store-to-flash command (#DM_STORE_PARAMETERS)
  so the mode survives a power cycle. The motor runs the new mode only after a power cycle, and answers a read with
  a timeout until then; the report says so.

The command-line reports (check() and switch()) print one line per motor and return an exit code.

@see read_modes(), set_mit(), check(), switch()
"""

import struct
import sys
import time
from collections.abc import Iterable, Sequence
from typing import IO, Any

## The DM register holding the firmware control mode (RID 10).
CTRL_MODE_REGISTER = 10

## The control modes by name, as the DM firmware numbers them. i2rt drives a motor in `mit` only.
MODES: dict[str, int] = {"mit": 1, "pos_speed": 2, "speed": 3, "torque_pos": 4}

## The one mode the station can drive.
MIT = MODES["mit"]

## A DM motor's parameter frames go to this id, carrying `[id low, id high, command, register, value...]`; the reply
## carries the same four head bytes and the value, and is taken by those bytes, so a stray status frame is skipped.
## The three commands: read a register (the wizard's joint reading uses the same one), write a register (the reply
## carries the value now held), and store every register to flash (no reply is promised).
DM_PARAMETER_ID = 0x7FF
DM_READ_PARAMETER = 0x33
DM_WRITE_PARAMETER = 0x55
DM_STORE_PARAMETERS = 0xAA

## The register byte the store command carries.
DM_STORE_REGISTER = 0x01

## The last byte of the DM special command that switches a motor off (the first seven are 0xFF), sent to the motor's
## own id; the same command the wizard sends the gripper after its check.
DM_DISABLE = 0xFD

## How long one motor gets to answer a read or a write, seconds. A motor that has just had its mode written and not
## yet been power-cycled says nothing, so a silent motor is reported, never retried forever.
REPLY_TIMEOUT_S = 0.05

## Seconds to let a motor finish writing its flash after the store command before the next frame goes out.
STORE_SETTLE_S = 0.3

## The two links of a station's arms (`dreamcli setup` writes them).
DEFAULT_LINKS: tuple[str, ...] = ("can_follower_l", "can_follower_r")

## Every motor on a YAM arm: the six joints and the gripper (id 7).
ALL_MOTORS: tuple[int, ...] = tuple(range(1, 8))


def mode_name(mode: int | None) -> str:
    """The name of a mode value, `unknown(<n>)` for one the firmware does not document, `no answer` for None.

    @param mode The register value, or None for a motor that did not answer.
    @return The name.
    """
    if mode is None:
        return "no answer"
    for name, value in MODES.items():
        if value == mode:
            return name
    return f"unknown({mode})"


def read_modes(link: str, motors: Iterable[int] = ALL_MOTORS) -> list[tuple[int, int | None]]:
    """Reads each motor's control mode on one link; switches nothing on.

    @param link The arm's CAN link.
    @param motors The motor ids to read, in order.
    @return `(motor id, mode)` per motor; the mode is None for a motor that said nothing within #REPLY_TIMEOUT_S.
    @throws ImportError python-can is not installed.
    @throws Exception Whatever python-can raises when the link cannot be opened or written.
    """
    import can

    found: list[tuple[int, int | None]] = []
    with can.Bus(interface="socketcan", channel=link) as bus:
        for motor in motors:
            found.append((motor, _parameter_exchange(bus, motor, DM_READ_PARAMETER, 0)))
    return found


def set_mit(link: str, motors: Iterable[int] = ALL_MOTORS) -> list[tuple[int, int | None, int | None]]:
    """Switches each motor on one link to MIT and stores it to flash; a motor already in MIT is stored as it is.

    Per motor, in order: the disable command to its own id (no reply is waited for), a read of the mode, a write
    of MIT when the read says anything else, then the store command and #STORE_SETTLE_S of quiet. The motor keeps
    answering the old mode's replies until it is power-cycled, so `after` is what the write's reply reported, not
    what the motor runs yet.

    @param link The arm's CAN link.
    @param motors The motor ids to switch, in order.
    @return `(motor id, mode before, mode after)` per motor; a None is a read or a write that got no answer, and
        such a motor is neither written nor stored.
    @throws ImportError python-can is not installed.
    @throws Exception Whatever python-can raises when the link cannot be opened or written.
    """
    import can

    done: list[tuple[int, int | None, int | None]] = []
    with can.Bus(interface="socketcan", channel=link) as bus:
        for motor in motors:
            bus.send(can.Message(arbitration_id=motor, data=bytes([0xFF] * 7 + [DM_DISABLE]), is_extended_id=False))
            _drain(bus)
            before = _parameter_exchange(bus, motor, DM_READ_PARAMETER, 0)
            if before is None:
                done.append((motor, None, None))
                continue
            after = before if before == MIT else _parameter_exchange(bus, motor, DM_WRITE_PARAMETER, MIT)
            if after == MIT:
                head = bytes([motor & 0xFF, motor >> 8, DM_STORE_PARAMETERS, DM_STORE_REGISTER])
                bus.send(can.Message(arbitration_id=DM_PARAMETER_ID, data=head + bytes(4), is_extended_id=False))
                time.sleep(STORE_SETTLE_S)
                _drain(bus)
            done.append((motor, before, after))
    return done


def switch_off(link: str, motors: Iterable[int] = ALL_MOTORS) -> None:
    """Sends the DM disable command to each motor on one link, so each goes limp; no reply is waited for.

    `dreamcli robot down` runs it when no station is running: a station whose PC died or rebooted never
    switched the arms off, and a DM motor keeps its last command while the bus is quiet.

    @param link The arm's CAN link.
    @param motors The motor ids to switch off, in order.
    @throws ImportError python-can is not installed.
    @throws Exception Whatever python-can raises when the link cannot be opened or written.
    """
    import can

    with can.Bus(interface="socketcan", channel=link) as bus:
        for motor in motors:
            bus.send(can.Message(arbitration_id=motor, data=bytes([0xFF] * 7 + [DM_DISABLE]), is_extended_id=False))


def check(
    links: Sequence[str] = DEFAULT_LINKS, motors: Iterable[int] = ALL_MOTORS, *, out: IO[str] = sys.stdout
) -> int:
    """Prints each motor's mode on each link and says whether every one is MIT.

    @param links The CAN links to read.
    @param motors The motor ids to read on each.
    @param out Where the report goes.
    @return 0 when every motor answered MIT; 1 when one answered another mode or did not answer.
    """
    wanted = tuple(motors)
    wrong: list[str] = []
    for link in links:
        print(f"{link}:", file=out)
        try:
            readings = read_modes(link, wanted)
        except Exception as exc:  # the report is the point; the cause is on the line
            print(f"  could not read the link: {exc}", file=out)
            wrong.append(link)
            continue
        for motor, mode in readings:
            verdict = "ok" if mode == MIT else "NOT MIT" if mode is not None else "is the arm powered on?"
            print(f"  motor {motor}: {mode_name(mode)}  [{verdict}]", file=out)
            if mode != MIT:
                wrong.append(f"{link} motor {motor}")
    if wrong:
        print(
            f"{len(wrong)} not in MIT mode: 'dreamcli motors set-mit' switches them; then power-cycle the arms",
            file=out,
        )
        return 1
    print("every motor is in MIT mode", file=out)
    return 0


def switch(
    links: Sequence[str] = DEFAULT_LINKS, motors: Iterable[int] = ALL_MOTORS, *, out: IO[str] = sys.stdout
) -> int:
    """Switches every motor on each link to MIT, stores it, and says what to do next.

    @param links The CAN links to switch.
    @param motors The motor ids to switch on each.
    @param out Where the report goes.
    @return 0 when every motor confirmed MIT; 1 when one did not answer or did not confirm.
    """
    wanted = tuple(motors)
    failed: list[str] = []
    for link in links:
        print(f"{link}:", file=out)
        try:
            results = set_mit(link, wanted)
        except Exception as exc:
            print(f"  could not write the link: {exc}", file=out)
            failed.append(link)
            continue
        for motor, before, after in results:
            if after == MIT:
                what = "already mit, stored" if before == MIT else f"{mode_name(before)} -> mit, stored"
                print(f"  motor {motor}: {what}", file=out)
            else:
                print(f"  motor {motor}: {mode_name(before)} -> {mode_name(after)}  [FAILED]", file=out)
                failed.append(f"{link} motor {motor}")
    if failed:
        print(f"{len(failed)} did not confirm MIT: is the arm powered on and the link up? Then run it again", file=out)
        return 1
    print("done: power-cycle the arms (motor power off, a few seconds, on), then 'dreamcli motors check'", file=out)
    return 0


def _parameter_exchange(bus: Any, motor: int, command: int, value: int) -> int | None:
    """Sends one parameter frame for #CTRL_MODE_REGISTER and returns the value the motor's reply carries.

    @param bus An open python-can bus.
    @param motor The motor id.
    @param command #DM_READ_PARAMETER or #DM_WRITE_PARAMETER.
    @param value The value to write; ignored by a read.
    @return The register value in the reply, or None when no matching reply came within #REPLY_TIMEOUT_S.
    """
    import can

    head = bytes([motor & 0xFF, motor >> 8, command, CTRL_MODE_REGISTER])
    frame = can.Message(arbitration_id=DM_PARAMETER_ID, data=head + struct.pack("<I", value), is_extended_id=False)
    bus.send(frame)
    deadline = time.monotonic() + REPLY_TIMEOUT_S
    while (left := deadline - time.monotonic()) > 0:
        reply = bus.recv(timeout=left)
        if reply is None:
            break
        data = bytes(reply.data)
        if len(data) == 8 and data[:4] == head:
            return int(struct.unpack("<I", data[4:])[0])
    return None


def _drain(bus: Any) -> None:
    """Reads and drops whatever the bus holds right now, so a stale frame never passes for a reply.

    @param bus An open python-can bus.
    """
    while bus.recv(timeout=0.0) is not None:
        pass
