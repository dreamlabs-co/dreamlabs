"""Pins the loop's own types and the serve seam: key names, shapes, dtypes.

Everything here is checked against `robots.adapter`'s own output. Strict mode's verdict on the
adapter (observation and chunk against the embodiment) is `test_end_to_end.py`'s job, through
the real `dream` client; nothing private of dream is touched here. The frames here carry the
encoded frame a camera's encode thread would have given them (`Frame.jpeg`, by `robots.cameras.encode_jpeg`);
cv2 is imported only to decode it, the way the serve does.
"""

import base64
import dataclasses
from typing import Any

import conftest
import cv2
import numpy as np
import pytest

from robots import adapter, cameras, errors, types

## Rows per chunk in the tests below; small so a wrong-horizon chunk is cheap to build.
HORIZON = 4

## Per-channel slack for a solid colour after a quality-85 JPEG round trip.
JPEG_TOLERANCE = 4

## The block embodiment_spec() must produce for the three roles at horizon 48; spelled out, not derived.
EXPECTED_SPEC: dict[str, Any] = {
    "state": {
        "arm_r.joint_pos": {"shape": [6], "dtype": "float32"},
        "arm_r.gripper_pos": {"shape": [1], "dtype": "float32"},
        "arm_l.joint_pos": {"shape": [6], "dtype": "float32"},
        "arm_l.gripper_pos": {"shape": [1], "dtype": "float32"},
    },
    "image_window": {"top": {}, "left": {}, "right": {}},
    "action": {
        "q_r_chunk": {"shape": [48, 7], "dtype": "float32"},
        "q_l_chunk": {"shape": [48, 7], "dtype": "float32"},
    },
}


def make_arm_state(base: float, gripper: float = 0.5, commanded: bool = False) -> types.ArmState:
    """Builds an ArmState with `base + i` per joint so arms and rows stay distinguishable."""
    positions = base + np.arange(conftest.ARM_DOF, dtype=np.float64)
    return types.ArmState(
        positions_rad=positions,
        velocities_rad_s=np.zeros(conftest.ARM_DOF),
        gripper=gripper,
        commanded_rad=positions.copy() if commanded else None,
        commanded_gripper=gripper if commanded else None,
    )


def make_observation(
    camera_names: tuple[str, ...] = conftest.CAMERA_NAMES,
    now_s: float = 1.0,
    rgb: tuple[int, int, int] = (255, 0, 0),
) -> types.Observation:
    """Builds an Observation with distinct left/right arms and one solid RGB frame per camera, each carrying
    its own pixels encoded, as a reader that encodes every capture would publish."""
    images = {name: make_frame(name, rgb, now_s) for name in camera_names}
    state = types.RobotState(
        left=make_arm_state(base=10.0, gripper=0.25),
        right=make_arm_state(base=20.0, gripper=0.75),
        timestamp_s=now_s,
    )
    return types.Observation(state=state, images=images, timestamp_s=now_s)


def decode_frame(jpeg: str) -> np.ndarray:
    """Decodes one base64 JPEG frame the way the serve does: cv2, so the pixels come back BGR."""
    encoded = np.frombuffer(base64.b64decode(jpeg), dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert decoded is not None
    return np.asarray(decoded)


def sent(tick: tuple[dict[str, str], dict[str, Any]]) -> dict[str, Any]:
    """What the dream client sends for a tick a wire built, as the episode's first: the observation, then each frame
    key's window of that one frame.

    @param tick `(frames, observation)` from a wire's to_dream().
    @return The observation the client would send.
    """
    frames, observation = tick
    return {**observation, **{key: [frame] for key, frame in frames.items()}}


def make_frame(camera: str, rgb: tuple[int, int, int] = (255, 0, 0), now_s: float = 1.0) -> types.Frame:
    """Builds a solid RGB Frame whose encoded frame is that same frame, encoded once, as the encode thread does."""
    data = _solid_frame(rgb)
    jpeg = base64.b64encode(cameras.encode_jpeg(data)).decode("ascii")
    return types.Frame(data=data, camera=camera, timestamp_s=now_s, jpeg=jpeg, jpeg_timestamp_s=now_s)


def _solid_frame(rgb: tuple[int, int, int]) -> np.ndarray:
    frame = np.empty(conftest.FRAME_SHAPE, dtype=np.uint8)
    frame[...] = rgb
    return frame


# Types


@pytest.mark.parametrize(
    "instance, field",
    [
        (make_frame("top"), "camera"),
        (make_arm_state(base=0.0), "gripper"),
        (make_observation().state, "timestamp_s"),
        (make_observation(), "images"),
        (types.ArmCommand(positions_rad=np.zeros(conftest.ARM_DOF), gripper=0.5), "gripper"),
        (adapter.from_dream(conftest.make_chunk(HORIZON), 0.0, HORIZON).actions[0], "left"),
        (adapter.from_dream(conftest.make_chunk(HORIZON), 0.0, HORIZON), "observed_at_s"),
    ],
    ids=["Frame", "ArmState", "RobotState", "Observation", "ArmCommand", "Action", "ActionChunk"],
)
def test_types_are_frozen_dataclasses(instance: Any, field: str) -> None:
    """All seven types are frozen dataclasses; rebinding a field raises."""
    assert dataclasses.is_dataclass(instance)
    assert type(instance).__dataclass_params__.frozen
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field, None)


def test_arm_state_commanded_fields_are_none_before_first_write() -> None:
    """`commanded_rad` / `commanded_gripper` are None until the arm was written once."""
    fresh = make_arm_state(base=0.0)
    written = make_arm_state(base=0.0, commanded=True)

    assert fresh.commanded_rad is None
    assert fresh.commanded_gripper is None
    assert written.commanded_rad is not None
    assert written.commanded_gripper == 0.5


def test_action_requires_both_arms() -> None:
    """`Action` is always bimanual; `left` and `right` are required, not optional."""
    command = types.ArmCommand(positions_rad=np.zeros(conftest.ARM_DOF), gripper=0.0)

    with pytest.raises(TypeError):
        types.Action(left=command)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        types.Action(right=command)  # type: ignore[call-arg]


# to_dream


def test_to_dream_hands_one_frame_per_role_apart_from_the_prompt_and_state_keys() -> None:
    """A tick is one frame per `image_window.<role>`, apart from the observation, which is `prompt` and the
    four `state.*` keys (right arm first); `set_observation()` refuses a frame key inside the observation."""
    frames, payload = adapter.to_dream(make_observation(), conftest.TASK)

    assert adapter.STATE_KEYS == (
        "state.arm_r.joint_pos",
        "state.arm_r.gripper_pos",
        "state.arm_l.joint_pos",
        "state.arm_l.gripper_pos",
    )
    assert tuple(key for key in payload if key.startswith("state.")) == adapter.STATE_KEYS
    assert set(payload) == {"prompt", *adapter.STATE_KEYS}
    assert set(frames) == {f"image_window.{name}" for name in conftest.CAMERA_NAMES}
    assert all(isinstance(frame, str) and frame for frame in frames.values())
    assert payload["prompt"] == conftest.TASK


def test_to_dream_state_arrays_are_float32_with_dim_6_and_1() -> None:
    """Joint positions are float32 `[6]`, grippers float32 `[1]`, from the loop's float64 fields."""
    obs = make_observation()

    _, payload = adapter.to_dream(obs, conftest.TASK)

    for key in adapter.STATE_KEYS:
        assert payload[key].dtype == np.float32, key
    assert payload["state.arm_r.joint_pos"].shape == (6,)
    assert payload["state.arm_l.joint_pos"].shape == (6,)
    assert payload["state.arm_r.gripper_pos"].shape == (1,)
    assert payload["state.arm_l.gripper_pos"].shape == (1,)
    np.testing.assert_array_equal(payload["state.arm_r.joint_pos"], obs.state.right.positions_rad)
    np.testing.assert_array_equal(payload["state.arm_l.joint_pos"], obs.state.left.positions_rad)
    assert payload["state.arm_r.gripper_pos"][0] == pytest.approx(0.75)
    assert payload["state.arm_l.gripper_pos"][0] == pytest.approx(0.25)


def test_to_dream_frame_is_one_base64_jpeg_that_decodes_to_the_capture() -> None:
    """Each camera's frame goes to the client as one base64 JPEG string, BGR-encoded like the serve
    wire; the client sends the window of them.

    The serve decodes with cv2 and converts BGR to RGB, so a pure red RGB frame must decode to
    cv2's `[0, 0, 255]`; encoding the RGB array as-is would land the red in the blue channel.
    """
    obs = make_observation(camera_names=("top", "left"), rgb=(255, 0, 0))

    frames, _ = adapter.to_dream(obs, conftest.TASK)

    assert set(frames) == {"image_window.top", "image_window.left"}
    for name in ("top", "left"):
        decoded = decode_frame(frames[f"image_window.{name}"])
        assert decoded.shape == conftest.FRAME_SHAPE
        np.testing.assert_allclose(decoded[0, 0], [0, 0, 255], atol=JPEG_TOLERANCE)
        np.testing.assert_allclose(decoded[..., ::-1], obs.images[name].data, atol=JPEG_TOLERANCE)


def test_to_dream_hands_the_encoded_frame_verbatim() -> None:
    """`image_window.<role>` is the frame's `jpeg`, as the encode thread made it; the adapter never
    re-encodes, so a tick costs no cv2 call at all."""
    obs = make_observation(camera_names=("top",))
    obs.images["top"] = dataclasses.replace(obs.images["top"], jpeg="the-encoded-frame")

    frames, _ = adapter.to_dream(obs, conftest.TASK)

    assert frames == {"image_window.top": "the-encoded-frame"}


def test_to_dream_never_touches_cv2(monkeypatch: pytest.MonkeyPatch) -> None:
    """The frames arrive encoded; an adapter that encoded again would double the CPU cost per request."""

    def refusing_imencode(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("adapter.to_dream called cv2.imencode")

    obs = make_observation()
    monkeypatch.setattr(cv2, "imencode", refusing_imencode)

    frames, _ = adapter.to_dream(obs, conftest.TASK)

    assert frames["image_window.top"] == obs.images["top"].jpeg


def test_to_dream_rejects_a_frame_without_an_encoded_frame() -> None:
    """The serve needs a frame per role; a frame no reader encoded is a ControlLoopError naming the
    camera."""
    obs = make_observation(camera_names=("top",))
    frame = obs.images["top"]
    obs.images["top"] = types.Frame(data=frame.data, camera="top", timestamp_s=frame.timestamp_s)

    with pytest.raises(errors.ControlLoopError, match="'top'.*no encoded frame"):
        adapter.to_dream(obs, conftest.TASK)


def test_to_dream_rejects_a_camera_that_is_not_a_serve_role() -> None:
    """The serve accepts `top`, `left`, `right` only; another camera name is a ControlLoopError naming it."""
    with pytest.raises(errors.ControlLoopError, match="'overhead'") as info:
        adapter.to_dream(make_observation(camera_names=("top", "overhead")), conftest.TASK)

    assert "top" in str(info.value)
    assert adapter.CAMERA_ROLES == ("top", "left", "right")


def test_to_dream_does_not_mutate_the_observation() -> None:
    """rule 11: the adapter returns new dicts; the observation's arrays keep their dtype and values."""
    obs = make_observation()
    before = obs.state.right.positions_rad.copy()
    pixels_before = obs.images["top"].data.copy()

    adapter.to_dream(obs, conftest.TASK)

    assert obs.state.right.positions_rad.dtype == np.float64
    np.testing.assert_array_equal(obs.state.right.positions_rad, before)
    np.testing.assert_array_equal(obs.images["top"].data, pixels_before)


# from_dream


def test_from_dream_splits_h_by_7_rows_into_actions_with_the_gripper_last() -> None:
    """`q_r_chunk` / `q_l_chunk` rows are six joints then the gripper; one bimanual Action per row."""
    chunk = conftest.make_chunk(HORIZON, base=10.0)
    chunk["q_l_chunk"] = [[value + 100.0 for value in row[:6]] + [row[6]] for row in chunk["q_l_chunk"]]

    result = adapter.from_dream(chunk, observed_at_s=3.5, horizon=HORIZON)

    assert isinstance(result, types.ActionChunk)
    assert result.observed_at_s == 3.5
    assert len(result.actions) == HORIZON
    for i, action in enumerate(result.actions):
        np.testing.assert_array_equal(action.right.positions_rad, np.full(6, 10.0 + i))
        np.testing.assert_array_equal(action.left.positions_rad, np.full(6, 110.0 + i))
        assert action.right.gripper == pytest.approx(((10 + i) % 10) / 10)
        assert action.left.gripper == pytest.approx(((10 + i) % 10) / 10)


def test_from_dream_accepts_arrays_as_well_as_lists() -> None:
    """The wire carries lists, a fake may hand arrays; both read the same."""
    lists = conftest.make_chunk(HORIZON, base=2.0)
    arrays = {key: np.asarray(value, dtype=np.float32) for key, value in lists.items()}

    from_lists = adapter.from_dream(lists, observed_at_s=0.0, horizon=HORIZON)
    from_arrays = adapter.from_dream(arrays, observed_at_s=0.0, horizon=HORIZON)

    for a, b in zip(from_lists.actions, from_arrays.actions, strict=True):
        np.testing.assert_allclose(a.right.positions_rad, b.right.positions_rad, rtol=1e-6)
        assert a.left.gripper == pytest.approx(b.left.gripper, abs=1e-6)


def test_from_dream_ignores_the_serve_extras() -> None:
    """`q_r`, `q_l` and `per_key` ride along in the serve's reply and are not read."""
    chunk: dict[str, Any] = conftest.make_chunk(HORIZON)
    chunk["q_r"] = chunk["q_r_chunk"][0]
    chunk["q_l"] = chunk["q_l_chunk"][0]
    chunk["per_key"] = {"action.right_arm": [[0.0] * 6] * HORIZON}

    result = adapter.from_dream(chunk, observed_at_s=0.0, horizon=HORIZON)

    assert len(result.actions) == HORIZON


@pytest.mark.parametrize("key", adapter.ACTION_KEYS)
def test_from_dream_wrong_horizon_raises_naming_the_key(key: str) -> None:
    """Every key must carry exactly `horizon` rows; the error names the key and both shapes."""
    chunk = conftest.make_chunk(HORIZON)
    chunk[key] = chunk[key][: HORIZON - 1]

    with pytest.raises(errors.ControlLoopError, match=key) as info:
        adapter.from_dream(chunk, observed_at_s=0.0, horizon=HORIZON)

    assert f"({HORIZON - 1}, 7)" in str(info.value)
    assert f"({HORIZON}, 7)" in str(info.value)


def test_from_dream_row_without_the_gripper_column_raises_naming_the_key() -> None:
    """A `[H, 6]` chunk has no gripper column; it is a shape error, never a silent default."""
    chunk = conftest.make_chunk(HORIZON)
    chunk["q_l_chunk"] = [row[:6] for row in chunk["q_l_chunk"]]

    with pytest.raises(errors.ControlLoopError, match="q_l_chunk"):
        adapter.from_dream(chunk, observed_at_s=0.0, horizon=HORIZON)


@pytest.mark.parametrize("key", adapter.ACTION_KEYS)
def test_from_dream_missing_key_raises_naming_the_key(key: str) -> None:
    """A reply without one of the two chunk keys is a ControlLoopError naming it and listing what came."""
    chunk = conftest.make_chunk(HORIZON)
    del chunk[key]

    with pytest.raises(errors.ControlLoopError, match=key) as info:
        adapter.from_dream(chunk, observed_at_s=0.0, horizon=HORIZON)

    assert "got keys" in str(info.value)


def test_from_dream_clips_gripper_to_unit_interval() -> None:
    """The gripper is a float in [0, 1]; out-of-range values are clipped, never passed on."""
    chunk = conftest.make_chunk(HORIZON)
    for row, gripper in zip(chunk["q_r_chunk"], (-0.5, 0.5, 1.5, 2.0), strict=True):
        row[6] = gripper
    for row, gripper in zip(chunk["q_l_chunk"], (2.0, 1.0, 0.0, -1.0), strict=True):
        row[6] = gripper

    result = adapter.from_dream(chunk, observed_at_s=0.0, horizon=HORIZON)

    assert [a.right.gripper for a in result.actions] == pytest.approx([0.0, 0.5, 1.0, 1.0])
    assert [a.left.gripper for a in result.actions] == pytest.approx([1.0, 1.0, 0.0, 0.0])


def test_from_dream_returns_float64_positions_and_python_float_grippers() -> None:
    """The loop's ArmCommand holds float64 `[6]` positions and a float gripper, whatever the wire carried."""
    result = adapter.from_dream(conftest.make_chunk(HORIZON), observed_at_s=0.0, horizon=HORIZON)

    for action in result.actions:
        for command in (action.left, action.right):
            assert command.positions_rad.dtype == np.float64
            assert command.positions_rad.shape == (6,)
            assert type(command.gripper) is float


## An SO-101's joint slots: five joints, the sixth held at 0.
FIVE_JOINTS = (0, 1, 2, 3, 4)


def test_from_dream_on_a_five_joint_arm_fills_its_slots_and_holds_the_sixth_at_zero() -> None:
    """An SO-101's chunk rows are its five joints then the gripper; the loop's sixth slot reads 0."""
    chunk = {key: [row[:5] + row[6:] for row in rows] for key, rows in conftest.make_chunk(HORIZON, 1.0).items()}

    result = adapter.from_dream(chunk, observed_at_s=0.0, horizon=HORIZON, joints=FIVE_JOINTS)

    for i, action in enumerate(result.actions):
        np.testing.assert_array_equal(action.right.positions_rad, [1.0 + i] * 5 + [0.0])
        assert action.right.gripper == pytest.approx(((1 + i) % 10) / 10)


def test_from_dream_on_a_five_joint_arm_refuses_a_six_joint_row() -> None:
    """A chunk as wide as another arm's is a shape error naming the width the arm takes."""
    with pytest.raises(errors.ControlLoopError, match=r"expected \(4, 6\)"):
        adapter.from_dream(conftest.make_chunk(HORIZON), observed_at_s=0.0, horizon=HORIZON, joints=FIVE_JOINTS)


def test_commanded_joints_of_a_five_joint_arm_are_its_five() -> None:
    """The client's `commanded` holds the arm's own joints, as wide as the chunks it returns."""
    right = types.ArmCommand(positions_rad=np.arange(6.0), gripper=0.5)
    left = types.ArmCommand(positions_rad=np.arange(6.0) + 10.0, gripper=0.5)

    commanded = adapter.commanded_joints(types.Action(left=left, right=right), FIVE_JOINTS)

    np.testing.assert_array_equal(commanded["q_r_chunk"], np.arange(5.0))
    np.testing.assert_array_equal(commanded["q_l_chunk"], np.arange(5.0) + 10.0)


# embodiment_spec


def test_embodiment_spec_declares_state_image_window_and_action_groups() -> None:
    """The block names what to_dream() sends and from_dream() reads; JPEG windows are presence-only."""
    spec = adapter.embodiment_spec(["top", "left", "right"], horizon=48)

    assert spec == EXPECTED_SPEC
    assert list(spec["state"]) == ["arm_r.joint_pos", "arm_r.gripper_pos", "arm_l.joint_pos", "arm_l.gripper_pos"]


def test_embodiment_spec_lists_only_the_cameras_given() -> None:
    """A station with fewer cameras declares fewer windows; strict mode then rejects a surplus one."""
    spec = adapter.embodiment_spec(["top"], horizon=HORIZON)

    assert spec["image_window"] == {"top": {}}
    assert spec["action"]["q_r_chunk"] == {"shape": [HORIZON, 7], "dtype": "float32"}
