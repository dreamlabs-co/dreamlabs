"""End to end on a Metal arm: the station, the real dream client with its IK, a fake DreamAPI that answers with tool
poses, and the Metal driver on the fake CAN bus.

The Makers Arm's end-to-end test (`test_makers_station.py`) with the Metal body: its fake gateway answers the tool
poses of fixed joint targets computed by dream's forward kinematics of the Metal arm, both arms start at dream's
Metal rest pose, and each arm's gripper runs linearly between the fixture ends (`metal_fakes.GRIPPER_ENDS`). So what is
exercised is the whole path of a Metal run: the Metal wire through dream's strict mode, the client adding the tool
poses, the chunk solved back to joints with dream's Metal IK, the loop, the driver's tick and its frames, and on close
the return to the pose each arm was opened in before torque off. The loop runs on the real clock at 30 Hz.
"""

import asyncio
import dataclasses
from collections.abc import Iterator
from typing import Any

import conftest
import dream
import metal_fakes
import numpy as np
import pytest
import test_makers_station
from test_makers_station import (
    CHANNELS,
    CHECKPOINT,
    CONTROL_HZ,
    GRIPS,
    HORIZON,
    REACH_TOL_DEG,
    ROLES,
    EEFDreamAPI,
)

from robots import cameras, config, damiao, errors, metal, motor_arm, robstride, station

## Where the grippers start, motor degrees: 20 mm open on the fixture table.
START_GRIPPER_DEG = 30.0
## The Metal gripper's CAN id.
GRIPPER_ID = 7
DISABLE = b"\xff" * 7 + bytes([damiao.CMD_DISABLE])


def stroke_deg(mm: float) -> float:
    """The fixture ends' motor angle for a stroke, degrees."""
    span = metal_fakes.GRIPPER_OPEN_DEG - metal_fakes.GRIPPER_CLOSED_DEG
    return metal_fakes.GRIPPER_CLOSED_DEG + mm / metal.GRIP_SCALE_MM * span


def stroke_mm(deg: float) -> float:
    """The fixture ends' stroke at a motor angle, mm."""
    span = metal_fakes.GRIPPER_OPEN_DEG - metal_fakes.GRIPPER_CLOSED_DEG
    return (deg - metal_fakes.GRIPPER_CLOSED_DEG) / span * metal.GRIP_SCALE_MM


@pytest.fixture(scope="module")
def solver() -> Any:
    """dream's Metal model at the loop's rate, for the targets' tool poses and the checks."""
    return test_makers_station.dream_ik_solver.ChunkIK(
        "metal", test_makers_station.dream_ik_specs.SERVE.tool, CONTROL_HZ
    )


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch, solver: Any) -> Iterator[EEFDreamAPI]:
    """A started EEF fake named as the credentials' gateway and platform; closed on teardown."""
    fake = EEFDreamAPI(test_makers_station.eef_per_key(solver))
    fake.start()
    monkeypatch.setenv("DREAM_GATEWAY_URL", f"http://{fake.host}:{fake.port}")
    monkeypatch.setenv("DREAM_API_URL", f"http://{fake.host}:{fake.port}")
    try:
        yield fake
    finally:
        fake.close()


@pytest.fixture
def bus(monkeypatch: pytest.MonkeyPatch, solver: Any) -> metal_fakes.FakeCanFactory:
    """Both arms' adapters on the fake bus, every joint at dream's Metal rest pose; the Metal driver opens them, and
    the cameras are fakes."""
    factory = metal_fakes.FakeCanFactory()
    rest_deg = np.degrees(np.asarray(solver.rest, dtype=np.float64))
    for channel in CHANNELS.values():
        motors = factory.bus(channel).motors
        for i, joint in enumerate(metal.JOINTS[:6]):
            motors[joint.motor_id].position_deg = float(rest_deg[i])
        motors[GRIPPER_ID].position_deg = START_GRIPPER_DEG
    monkeypatch.setattr(robstride, "open_can_bus", factory)

    def fake_reader(cfg: config.CameraConfig, stale_after_s: float, **_encoding: Any) -> cameras.CameraReader:
        return cameras.CameraReader(cfg.name, conftest.FakeCaptureSource(), stale_after_s)

    monkeypatch.setattr(cameras.CameraReader, "from_config", fake_reader)
    return factory


@pytest.fixture
def switch_offs(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int, float]]:
    """Every powered joint switched off on any fake adapter, in order: `(channel, CAN id, its angle then, degrees)`."""
    seen: list[tuple[str, int, float]] = []
    send = metal_fakes.FakeCanBus.send

    def recording(self: metal_fakes.FakeCanBus, message: Any, timeout: float | None = None) -> None:
        motor = self.motors.get(message.arbitration_id)
        if motor is not None and motor.can_id != GRIPPER_ID and motor.enabled and bytes(message.data) == DISABLE:
            seen.append((self.channel, motor.can_id, motor.position_deg))
        send(self, message, timeout)

    monkeypatch.setattr(metal_fakes.FakeCanBus, "send", recording)
    return seen


def metal_config() -> config.AppConfig:
    """A Metal station config over the fake cameras, at the Metal horizon and rate, both arms on the fixture table."""
    height, width, _ = conftest.FRAME_SHAPE
    return conftest.make_app_config(
        body="metal",
        follower_l=config.ArmConfig(can_channel=CHANNELS["left"], **metal_fakes.GRIPPER_ENDS),
        follower_r=config.ArmConfig(can_channel=CHANNELS["right"], **metal_fakes.GRIPPER_ENDS),
        cameras=[config.CameraConfig(name=name, device=i, width=width, height=height) for i, name in enumerate(ROLES)],
        control_hz=CONTROL_HZ,
        action_horizon=HORIZON,
        action_timeout_s=3.0,
        max_stall_ticks=60,
    )


def arm_deg(factory: metal_fakes.FakeCanFactory, side: str) -> np.ndarray:
    """One arm's six joints as its motors hold them, degrees."""
    motors = factory.buses[CHANNELS[side]].motors
    return np.array([motors[joint.motor_id].position_deg for joint in metal.JOINTS[:6]])


def enabled(factory: metal_fakes.FakeCanFactory, side: str) -> list[bool]:
    """Whether each of one arm's seven motors is powered, base to gripper."""
    return [motor.enabled for motor in factory.bus(CHANNELS[side]).motors.values()]


def assert_switched_off_at_rest(seen: list[tuple[str, int, float]], solver: Any) -> None:
    """Every joint switch-off in `seen` happened at dream's Metal rest pose, where both arms opened."""
    rest_deg = np.degrees(np.asarray(solver.rest, dtype=np.float64))
    tolerance_deg = np.degrees(motor_arm.REST_TOL_RAD)
    for channel, can_id, deg in seen:
        rest = rest_deg[can_id - 1]
        assert abs(deg - rest) <= tolerance_deg, f"{channel} motor {can_id} off at {deg:.1f} deg, rest {rest:.1f}"


async def reset_and_wait(controller: station.StationController) -> None:
    """Asks a reset and waits until it has ended, idle or in error."""
    await controller.reset()
    await test_makers_station.wait_for(lambda: controller.state in (station.IDLE, station.ERROR), "the reset to end")


def gripper_deg(factory: metal_fakes.FakeCanFactory, side: str) -> float:
    """One arm's gripper motor angle, degrees."""
    return factory.buses[CHANNELS[side]].motors[GRIPPER_ID].position_deg


def test_a_metal_station_runs_an_eef_model_through_the_clients_ik_and_returns_to_its_start_on_close(
    server: EEFDreamAPI, bus: metal_fakes.FakeCanFactory, solver: Any
) -> None:
    """Start is refused without a model; with one, the Metal wire passes strict mode, the client adds each arm's
    tool pose, solves the reply's tool poses to joints with the Metal model, and both arms end at the targets with
    their grippers at the grips' strokes; close parks both arms at the pose they opened in, grippers open, and
    switches every motor off."""
    goal = test_makers_station.targets(solver)
    # tick: this checks the Metal wire and the station, not the solve; on CI's runners the traj compile on its thread
    # (every body's default) starves the loop past the 30 s wait. The client tests and test_makers_station run traj.
    cfg = dataclasses.replace(metal_config(), ik_solve="tick")
    controller = station.StationController.from_config(cfg, inference_log_path=None)
    controller.open_cameras()

    def at_targets() -> bool:
        return all(
            np.allclose(arm_deg(bus, side), np.degrees(goal[side]), atol=REACH_TOL_DEG)
            and abs(gripper_deg(bus, side) - stroke_deg(GRIPS[side] * metal.GRIP_SCALE_MM)) < 0.1
            for side in CHANNELS
        )

    async def scenario() -> dict[str, Any]:
        with pytest.raises(errors.StationError, match=r"no checkpoint"):
            await controller.start()
        await controller.start(checkpoint=CHECKPOINT)
        await test_makers_station.wait_for(at_targets, "both arms at the targets")
        running = controller.status()
        await controller.close()
        return running

    running = asyncio.run(scenario())

    assert running["state"] == station.RUNNING and running["checkpoint"] == CHECKPOINT
    assert running["chunks"] >= 1 and running["stalls"] == 0 and running["last_error"] is None
    observation = server.requests[0].body["observation"]
    state_parts = ("arm", "gripper", "grip", "eef_xyz", "eef_rot6")
    assert set(observation) == {
        "prompt",
        *(f"state.{side}_{part}" for side in ("right", "left") for part in state_parts),
        *(f"video.{name}" for name in ROLES),
    }
    for side in ("right", "left"):
        # A few encoder counts (16 bits over +-6.28 rad, 0.19 mrad each) from the pose the motors were put in.
        np.testing.assert_allclose(np.asarray(observation[f"state.{side}_arm"]), solver.rest, atol=1e-3)
        grip = observation[f"state.{side}_grip"][0]
        assert observation[f"state.{side}_gripper"][0] == pytest.approx(grip * metal.GRIP_SCALE_MM, rel=1e-6)
        assert grip == pytest.approx(stroke_mm(START_GRIPPER_DEG) / metal.GRIP_SCALE_MM, abs=1e-3)

    rest_deg = np.degrees(np.asarray(solver.rest, dtype=np.float64))
    for side, channel in CHANNELS.items():
        motors = bus.buses[channel].motors
        assert not any(motor.enabled for motor in motors.values()), side
        assert bus.buses[channel].closed, side
        np.testing.assert_allclose(arm_deg(bus, side), rest_deg, atol=np.degrees(motor_arm.REST_TOL_RAD))
        open_goal_mm = metal.GRIP_SCALE_MM - metal.ARM.grip_margin_mm
        assert stroke_mm(gripper_deg(bus, side)) >= open_goal_mm - motor_arm.REST_TOL_MM, side
        gripper = motors[GRIPPER_ID]
        assert gripper.registers[metal_fakes.REG_TIMEOUT] == bytes(4) and gripper.mode == 1, side


def test_reset_after_a_stop_parks_each_held_arm_switches_it_off_at_rest_then_opens_both_holding(
    server: EEFDreamAPI, bus: metal_fakes.FakeCanFactory, solver: Any, switch_offs: list[tuple[str, int, float]]
) -> None:
    goal = test_makers_station.targets(solver)
    controller = station.StationController.from_config(metal_config(), inference_log_path=None)
    controller.open_cameras()

    def at_targets() -> bool:
        return all(np.allclose(arm_deg(bus, side), np.degrees(goal[side]), atol=REACH_TOL_DEG) for side in CHANNELS)

    async def scenario() -> bool:
        await controller.start(checkpoint=CHECKPOINT)
        await test_makers_station.wait_for(at_targets, "both arms at the targets")
        await controller.stop()
        assert controller.state == station.IDLE
        assert all(all(enabled(bus, side)) for side in CHANNELS), "a stop leaves both arms holding"
        await reset_and_wait(controller)
        assert controller.state == station.IDLE, controller.last_error
        assert controller.arms_alive() == {"left": True, "right": True}
        assert all(all(enabled(bus, side)) for side in CHANNELS), "both arms opened again and hold"
        switched = sorted((channel, can_id) for channel, can_id, _ in switch_offs)
        assert switched == sorted((channel, can_id) for channel in CHANNELS.values() for can_id in range(1, 7))
        return await controller.close()

    assert asyncio.run(scenario()) is True
    assert not any(any(enabled(bus, side)) for side in CHANNELS)
    assert_switched_off_at_rest(switch_offs, solver)


def test_reset_after_a_safety_stop_opens_the_parked_arm_and_the_held_one_again_dropping_neither(
    server: EEFDreamAPI, bus: metal_fakes.FakeCanFactory, solver: Any, switch_offs: list[tuple[str, int, float]]
) -> None:
    elbow = bus.bus(CHANNELS["right"]).motors[3]
    controller = station.StationController.from_config(metal_config(), inference_log_path=None)
    controller.open_cameras()

    async def scenario() -> bool:
        await controller.start(checkpoint=CHECKPOINT)
        await test_makers_station.wait_for(lambda: controller.status()["chunks"] >= 1, "the first chunk")
        elbow.temp_rotor_c = int(metal.ARM.temp_stop_c) + 1
        await test_makers_station.wait_for(lambda: controller.state == station.ERROR, "the safety stop")
        assert controller.last_error is not None and controller.last_error["type"] == "SafetyStop"
        assert controller.arms_alive() == {"left": True, "right": False}
        assert all(enabled(bus, "left")) and not any(enabled(bus, "right"))
        elbow.temp_rotor_c = int(metal.ARM.temp_warn_c)
        await reset_and_wait(controller)
        assert controller.state == station.ERROR
        assert controller.last_error is not None
        warm = "the right arm's elbow_flex is at 55 C (coil): let it cool below 55 C"
        assert warm in controller.last_error["message"]
        assert "stopped answering" not in controller.last_error["message"], "the parked arm is off, not holding"
        assert not any(enabled(bus, "left")) and not any(enabled(bus, "right"))
        elbow.temp_rotor_c = 33
        await reset_and_wait(controller)
        assert controller.state == station.IDLE, controller.last_error
        assert controller.arms_alive() == {"left": True, "right": True}
        assert all(all(enabled(bus, side)) for side in CHANNELS)
        return await controller.close()

    assert asyncio.run(scenario()) is True
    assert_switched_off_at_rest(switch_offs, solver)


def test_a_joint_pressed_on_something_while_the_station_idles_is_eased_off_then_parked(
    server: EEFDreamAPI, bus: metal_fakes.FakeCanFactory, solver: Any
) -> None:
    rest = float(np.degrees(solver.rest[2]))
    elbow = bus.bus(CHANNELS["right"]).motors[3]
    controller = station.StationController.from_config(metal_config(), inference_log_path=None)
    elbow.stop_deg = rest - 15.0

    async def scenario() -> bool:
        await test_makers_station.wait_for(lambda: bool(controller._alarms()), "the blocked elbow's alarm")
        assert elbow.commands[-1] == pytest.approx(rest - 10.0, abs=0.5)
        assert await controller.park_alarmed_arms() is True
        assert controller.state == station.ERROR and controller.last_error is not None
        assert controller.last_error["type"] == "SafetyStop"
        assert controller.last_error["message"].startswith("arm 'right': elbow_flex has been blocked for")
        assert controller.arms_alive() == {"left": True, "right": False}
        assert all(enabled(bus, "left")) and not any(enabled(bus, "right"))
        return await controller.close()

    assert asyncio.run(scenario()) is True


def test_reset_refuses_a_dead_arm_still_powered_and_robot_down_leaves_it_holding_and_says_so(
    server: EEFDreamAPI,
    bus: metal_fakes.FakeCanFactory,
    solver: Any,
    switch_offs: list[tuple[str, int, float]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    wrist = bus.bus(CHANNELS["left"]).motors[5]
    controller = station.StationController.from_config(metal_config(), inference_log_path=None)
    seen: dict[str, Any] = {}

    async def scenario() -> None:
        wrist.fault = 9
        await test_makers_station.wait_for(lambda: not controller.arms_alive()["left"], "the left arm to latch")
        await reset_and_wait(controller)
        seen["state"], seen["error"] = controller.state, controller.last_error
        seen["left"] = enabled(bus, "left")
        seen["switch_offs"] = list(switch_offs)
        seen["closed"] = await controller.close()

    asyncio.run(scenario())

    assert seen["state"] == station.ERROR
    message = seen["error"]["message"]
    assert "wrist_yaw is in fault 0x9" in message and "still powered from an earlier session" in message, message
    assert "support the arm, then switch its power off and on" in message
    assert message.endswith(
        "The left arm stopped answering and was let go where it held; it may still hold under power so it does not "
        "drop: support it, then switch its power off"
    ), message
    assert all(seen["left"][:6]), "the refused arm still holds"
    assert {channel for channel, _, _ in seen["switch_offs"]} == {CHANNELS["right"]}
    assert not any(enabled(bus, "right"))
    assert seen["closed"] is False
    assert all(enabled(bus, "left")[:6]), "robot down switched the dead arm off where it held: it drops"
    said = [record.getMessage() for record in caplog.records if "switch left arm off" in record.getMessage()]
    assert any("it still holds under power so it does not drop: support it" in line for line in said), said
    assert_switched_off_at_rest(switch_offs, solver)


def test_reset_of_an_arm_whose_motor_went_silent_names_the_arm_and_says_it_may_still_hold(
    server: EEFDreamAPI, bus: metal_fakes.FakeCanFactory, solver: Any, switch_offs: list[tuple[str, int, float]]
) -> None:
    elbow = bus.bus(CHANNELS["right"]).motors[3]
    controller = station.StationController.from_config(metal_config(), inference_log_path=None)
    seen: dict[str, Any] = {}

    async def scenario() -> None:
        elbow.silent = True
        await test_makers_station.wait_for(lambda: not controller.arms_alive()["right"], "the right arm to latch")
        await reset_and_wait(controller)
        seen["state"], seen["error"] = controller.state, controller.last_error
        seen["right"] = enabled(bus, "right")
        seen["closed"] = await controller.close()

    asyncio.run(scenario())

    assert seen["state"] == station.ERROR
    message = seen["error"]["message"]
    assert message.startswith(
        "arm 'right' did not open: can_right: elbow_flex (id 3, reply id 0x13) did not answer"
    ), message
    assert message.endswith(
        ". The right arm stopped answering and was let go where it held; it may still hold under power so it does not "
        "drop: support it, then switch its power off"
    ), message
    assert all(seen["right"][:6]), "the dead right arm still holds"
    assert seen["closed"] is False
    assert {channel for channel, _, _ in switch_offs} == {CHANNELS["left"]}


def test_a_boot_whose_right_arm_is_refused_switches_the_opened_left_arm_off_and_drops_neither(
    server: EEFDreamAPI, bus: metal_fakes.FakeCanFactory, solver: Any, switch_offs: list[tuple[str, int, float]]
) -> None:
    right = bus.bus(CHANNELS["right"]).motors
    for joint in metal.JOINTS[:6]:
        right[joint.motor_id].enabled = True

    with pytest.raises(damiao.StartupPose, match="arm 'right' refuses to start: shoulder_pan is still powered"):
        station.StationController.from_config(metal_config(), inference_log_path=None)

    assert not any(enabled(bus, "left")), "the left arm opened and was left powered"
    assert all(enabled(bus, "right")[:6]), "the refused right arm still holds"
    assert {channel for channel, _, _ in switch_offs} == {CHANNELS["left"]}
    assert_switched_off_at_rest(switch_offs, solver)


def test_reset_whose_other_arm_cannot_open_leaves_the_opened_one_switched_off_at_rest(
    server: EEFDreamAPI, bus: metal_fakes.FakeCanFactory, solver: Any, switch_offs: list[tuple[str, int, float]]
) -> None:
    controller = station.StationController.from_config(metal_config(), inference_log_path=None)
    seen: dict[str, Any] = {}

    async def scenario() -> None:
        bus.fail_opens.add(CHANNELS["right"])
        await reset_and_wait(controller)
        seen["state"], seen["error"] = controller.state, controller.last_error
        seen["powered"] = [side for side in CHANNELS if any(enabled(bus, side))]
        bus.fail_opens.clear()
        await reset_and_wait(controller)
        seen["second"] = controller.state
        seen["closed"] = await controller.close()

    asyncio.run(scenario())

    assert seen["state"] == station.ERROR and seen["error"]["message"].startswith(
        "the right arm's CAN adapter can_right failed (CanInitializationError: could not open can_right): unplug the "
        "adapter and plug it back in, or switch the arm's power off and on"
    )
    assert seen["powered"] == [], "no arm is left powered with nothing driving it"
    assert seen["second"] == station.IDLE and seen["closed"] is True
    assert_switched_off_at_rest(switch_offs, solver)


@pytest.mark.parametrize("confirms_at_exit", [True, False])
def test_a_reset_whose_rollback_switch_off_does_not_confirm_keeps_that_arm_for_robot_down(
    server: EEFDreamAPI,
    bus: metal_fakes.FakeCanFactory,
    solver: Any,
    monkeypatch: pytest.MonkeyPatch,
    confirms_at_exit: bool,
) -> None:
    controller = station.StationController.from_config(metal_config(), inference_log_path=None)
    first_left = controller._arms["left"]._driver.chain
    original = damiao.DamiaoChain._disable
    lost = {"elbow_flex"}

    def new_left_loses(self: damiao.DamiaoChain, joint: str) -> bool:
        if self.name == "left" and self is not first_left and joint in lost:
            return False
        return original(self, joint)

    monkeypatch.setattr(damiao.DamiaoChain, "_disable", new_left_loses)
    seen: dict[str, Any] = {}

    async def scenario() -> None:
        bus.fail_opens.add(CHANNELS["right"])
        await reset_and_wait(controller)
        seen["state"], seen["error"], seen["arms"] = controller.state, controller.last_error, controller.arms_alive()
        seen["left"] = enabled(bus, "left")
        if confirms_at_exit:
            lost.clear()
        seen["closed"] = await controller.close()

    asyncio.run(scenario())

    assert seen["state"] == station.ERROR
    message = seen["error"]["message"]
    assert "could not open can_right" in message, message
    assert "The left arm opened but could not be switched off after it, so it may still hold under power" in message
    assert seen["arms"]["left"] is False
    assert seen["left"][2] and not any(seen["left"][:2] + seen["left"][3:]), "only the elbow is still powered"
    assert seen["closed"] is confirms_at_exit
    assert enabled(bus, "left")[2] is not confirms_at_exit


def test_a_metal_station_compiles_the_clients_ik_before_it_opens_either_arm(
    server: EEFDreamAPI, bus: metal_fakes.FakeCanFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The IK compile (seconds on a station's CPU) happens before the grippers are powered: at a Start it starved the
    arm thread past the grippers' 500 ms CAN timeout (fault 0xD)."""
    events: list[str] = []
    warm = station.warm_client_ik
    monkeypatch.setattr(station, "warm_client_ik", lambda dream_config: (events.append("warm"), warm(dream_config)))

    def open_can_bus(*args: Any, **kwargs: Any) -> Any:
        events.append("open")
        return bus(*args, **kwargs)

    monkeypatch.setattr(robstride, "open_can_bus", open_can_bus)
    controller = station.StationController.from_config(metal_config(), inference_log_path=None)
    asyncio.run(controller.close())

    assert events[0] == "warm" and "open" in events, events


class _Model:
    id = "dreamlabs/eef-model"


def test_warm_client_ik_builds_the_clients_ik_for_the_configs_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """It builds one `ClientIK` from the config's `ik`, rate and embodiment and the checkpoint's platform record."""
    built: list[tuple[Any, ...]] = []
    monkeypatch.setattr(station.dream, "DreamClient", lambda: type("C", (), {"get_models": lambda self: [_Model()]})())
    monkeypatch.setattr(station.dream.client_ik, "ClientIK", lambda *args: built.append(args))
    ik = {"arm": "metal", "head": "eef"}

    station.warm_client_ik({"checkpoint": _Model.id, "ik": ik, "action_hz": 20.0, "embodiment": {"e": 1}})

    assert len(built) == 1
    assert built[0][0] == ik and built[0][0] is not ik
    assert built[0][1:3] == (20.0, {"e": 1}) and isinstance(built[0][3], _Model)


def test_warm_client_ik_does_nothing_without_ik(monkeypatch: pytest.MonkeyPatch) -> None:
    """A body without the client's IK (a YAM) asks the platform nothing."""
    monkeypatch.setattr(station.dream, "DreamClient", lambda: pytest.fail("no client without ik"))

    station.warm_client_ik({"checkpoint": _Model.id, "action_hz": 20.0})


@pytest.mark.parametrize("models", [[], None])
def test_warm_client_ik_only_logs_when_it_cannot_warm_up(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, models: list[Any] | None
) -> None:
    """An unknown checkpoint or an unreachable platform leaves the compile to the Start, as before, and boots on."""

    def get_models(self: Any) -> list[Any]:
        if models is None:
            raise dream.DreamError("the platform cannot be reached")
        return models

    monkeypatch.setattr(station.dream, "DreamClient", lambda: type("C", (), {"get_models": get_models})())

    station.warm_client_ik({"checkpoint": _Model.id, "ik": {"arm": "metal", "head": "eef"}, "action_hz": 20.0})

    assert "IK warm-up skipped" in caplog.text
