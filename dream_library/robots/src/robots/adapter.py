"""The seam between the loop's types and the serve's dicts; the only module that knows the wire's key names.

Two wires live here. The station wire, first, which is deprecated (below): `dream` forwards the observation
verbatim to a dreamzero serve and returns the serve's
action dict unchanged, so the names here are the serve wire's: a `prompt`, `state.arm_r.joint_pos` style state
keys, `image_window.<role>` JPEG windows, and `q_r_chunk` / `q_l_chunk` action chunks of
`[H, 7]` (six joints then the gripper). to_dream() turns an #robots.types.Observation into
what the loop hands `dream.DreamClient.set_observation()` on a tick: one encoded frame per window key, and
the state and the prompt; the client keeps each key's window and sends it with the newest state.
from_dream() turns the action dict back into an #robots.types.ActionChunk, and embodiment_spec() produces the
`embodiment` block that makes `dream`'s strict mode agree with both. commanded_joints() gives the client, beside
each request, the joints the loop last commanded per chunk key (never sent), so each new chunk takes over where
the arms are on the old one.

No pixels are touched here: each frame already carries its encoded JPEG (`Frame.jpeg`, filled by the
camera's encode thread in #robots.cameras), so to_dream() is a relabelling and a tick costs no encoding.
`image_window.<role>` is the only image key: `image.<role>` singles would be rejected by dream's
strict mode as undeclared, and the serve ignores them when a window is present.

Every other body (a Makers Arm, a Metal arm, an SO-101, a robot of your own) speaks the wire of a FaaS fine-tune
instead, and one class builds it from the embodiment config the model was trained on (#EefWire): per arm the parts the
body's datasets record, `state.right_arm` (the arm's joints), `state.right_grip` (`[1]`, the gripper, 1 open) and,
when the datasets record it, `state.right_gripper` (`[1]`, the gripper in the datasets' own unit, the stroke in mm on
a MakerMods arm), the same for the left arm, and a frame per camera under `video.<role>`, whose window the client
keeps. The embodiment block is that config, which is how dream's strict mode reads the wire. The model answers with
tool poses, which the dream client solves to the same `q_r_chunk` / `q_l_chunk` joint chunks (`dream.client_ik`, the
config key `ik`), so from_dream() reads every body: `[H, joints + 1]`, the arm's joints then the gripper. How each
arm's joints and gripper go out is dream's spec of the arm (#ArmLayout): an SO-101 sends its five joints and leaves
`<side>_gripper` to the client. An arm may have any number of joints dream's spec has, 1 to
`robots.followers.MAX_JOINTS`; the loop binds the wire to its arms (`Wire.for_arms()`), which refuses an arm with
another count than dream's spec, so every command is as wide as the arm it goes to. The
Makers and Metal datasets share one layout, #MAKERMODS_EMBODIMENT; an arm with no fine-tune yet gets the one
zero_shot_embodiment() builds, on which Dream Labs' base models run.

#Wire is what the loop and `robots.run` take: an #EefWire, or the deprecated #YAM_WIRE. The YAM speaks its model's
own wire, an #EefWire on #YAM_EMBODIMENT (#YAM_VIDEO_WIRE): the same values under `state.right_arm`,
`state.right_gripper` and so on, and each camera under `video.<role>` for its station name (#YAM_VIDEO_ROLES); the
serve packs the model's joint answer into the same `q_r_chunk` / `q_l_chunk` on either wire, so from_dream() and
commanded_joints() read its replies too. The station wire (to_dream(), embodiment_spec(), #YAM_WIRE) is what no
body speaks: its `state.arm_r.*` and `image_window.<role>` keys are deprecated, and it goes when the API
drops them.

@see to_dream(), from_dream(), commanded_joints(), embodiment_spec(), Wire, EefWire
"""

import dataclasses
from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from shared.datasets import embodiments

from robots import errors, types

# Names

## Camera roles the serve accepts; a camera in the config must be named after one of them.
CAMERA_ROLES = ("top", "left", "right")
## Observation state keys, right arm first, as the serve reads them.
STATE_KEYS = (
    "state.arm_r.joint_pos",
    "state.arm_r.gripper_pos",
    "state.arm_l.joint_pos",
    "state.arm_l.gripper_pos",
)
## Action chunk keys; each is `[H, 7]`, right arm first.
ACTION_KEYS = ("q_r_chunk", "q_l_chunk")
## Prefix of every camera key; the camera role follows it.
IMAGE_WINDOW_PREFIX = "image_window."
## The language prompt key; carries no group, so strict mode leaves it alone.
PROMPT_KEY = "prompt"
## Joints per arm on the YAM's wires: the dim of every `joint_pos` key and the first six columns of an action row.
ARM_DOF = 6
## Columns of an action row: the six joints then the gripper.
ACTION_DIM = ARM_DOF + 1
## Every joint slot of an arm, in order: the YAM wire's, and a Makers or Metal arm's.
ALL_JOINTS = tuple(range(ARM_DOF))

# Shapes

## What `dream` sends for state arrays; actions arrive as JSON lists and carry no dtype.
_ARRAY_DTYPE = "float32"

# A FaaS fine-tune's wire

## The MakerMods datasets' embodiment config, the Makers Arm's and the Metal arm's alike: the state parts with their
## `[start, end)` spans, the action parts (the tool poses and grips the dream client solves, then the joint head) and
## the video key each camera role stands for. The tool poses in the state (`*_eef_*`) are added by the dream client
## from the joints (`dream.client_ik`).
MAKERMODS_EMBODIMENT: dict[str, dict[str, Any]] = {
    "state": {
        "left_arm": [0, 6],
        "left_gripper": [6, 7],
        "right_arm": [7, 13],
        "right_gripper": [13, 14],
        "right_eef_xyz": [14, 17],
        "right_eef_rot6": [17, 23],
        "right_grip": [23, 24],
        "left_eef_xyz": [24, 27],
        "left_eef_rot6": [27, 33],
        "left_grip": [33, 34],
    },
    "action": {
        "right_eef_xyz": [0, 3],
        "right_eef_rot6": [3, 9],
        "right_grip": [9, 10],
        "left_eef_xyz": [10, 13],
        "left_eef_rot6": [13, 19],
        "left_grip": [19, 20],
        "left_arm": [20, 26],
        "right_arm": [26, 32],
    },
    "video": {
        "scene": "observation.images.scene",
        "overhead": "observation.images.scene",
        "head": "observation.images.scene",
        "wrist_l": "observation.images.left_wrist",
        "left_wrist": "observation.images.left_wrist",
        "wrist_r": "observation.images.right_wrist",
        "right_wrist": "observation.images.right_wrist",
    },
}
## The YAM's own modalities (`shared.datasets.embodiments`, the layout of every YAM dataset): the four state parts the
## loop fills, the four action parts the model answers (which the serve also packs into the station's joint chunks),
## and its three cameras, named by the station's camera names through #YAM_VIDEO_ROLES. The YAM speaks this wire
## (#YAM_VIDEO_WIRE, `robots.bodies.YAM`).
YAM_EMBODIMENT: dict[str, dict[str, Any]] = {
    "state": {"right_arm": [0, 6], "right_gripper": [6, 7], "left_arm": [7, 13], "left_gripper": [13, 14]},
    "action": {"right_arm": [0, 6], "right_gripper": [6, 7], "left_arm": [7, 13], "left_gripper": [13, 14]},
    "video": {
        "overhead": "observation.images.overhead",
        "wrist_l": "observation.images.wrist_l",
        "wrist_r": "observation.images.wrist_r",
    },
}
## The YAM's embodiment config on the EEF head (`robots.run.policy_config`, the default): #YAM_EMBODIMENT's parts, then
## per arm the tool pose and the `[0, 1]` grip, which the dream client adds from YAM's kinematics (`ik` arm `yam`). It
## answers each arm's joints, tool pose and grip (the EEF models' `yam_eef_joint` action head, with no
## `<side>_gripper`); the client solves the poses to the joints from_dream() reads.
YAM_POSE_EMBODIMENT: dict[str, dict[str, Any]] = {
    "state": {
        **YAM_EMBODIMENT["state"],
        "right_eef_xyz": [14, 17],
        "right_eef_rot6": [17, 23],
        "right_grip": [23, 24],
        "left_eef_xyz": [24, 27],
        "left_eef_rot6": [27, 33],
        "left_grip": [33, 34],
    },
    "action": {
        "right_arm": [0, 6],
        "left_arm": [6, 12],
        "right_eef_xyz": [12, 15],
        "right_eef_rot6": [15, 21],
        "right_grip": [21, 22],
        "left_eef_xyz": [22, 25],
        "left_eef_rot6": [25, 31],
        "left_grip": [31, 32],
    },
    "video": YAM_EMBODIMENT["video"],
}
## A YAM station's camera names (#CAMERA_ROLES, what `dreamcli setup` writes) to the model's video roles, so one
## station config speaks either wire.
YAM_VIDEO_ROLES: dict[str, str] = {"top": "overhead", "left": "wrist_l", "right": "wrist_r"}
## The cameras of an embodiment zero_shot_embodiment() builds, as the setup wizard's camera step names them by the role
## each camera gets there: the scene camera, then a wrist camera per arm, left then right. The Dream-1 base models read
## the wrists by exactly these names and the scene camera by any name (`dream.client_ik.camera_slots`).
ZERO_SHOT_CAMERAS: tuple[str, ...] = ("scene", *embodiments.WRIST_ROLES)
## Prefix of every camera key on a fine-tune's wire; the camera role follows it.
VIDEO_PREFIX = "video."
## The state parts per arm an #EefWire fills, in the order it sends them; `<side>_eef_*` parts are the dream client's.
EEF_WIRE_PARTS = ("arm", "gripper", "grip")


@dataclasses.dataclass(frozen=True)
class ArmLayout:
    """How one arm's joints and gripper go out on a fine-tune's wire: dream's spec of the arm (`dream_ik.specs`), read
    by `robots.kinematics.arm_layout`.

    Attributes:
    - joints: The joint slots the arm drives, in order: what `state.<side>_arm`, `commanded` and each chunk carry
      (the SO-101's first five; dream's IK holds its sixth at 0).
    - width: The checkpoint's `<side>_arm`, as wide as dream's spec of the arm: the dream client puts the held
      joints back and converts the joints to the checkpoint's units on the way out.
    - client_gripper: The dream client makes the gripper part the checkpoint reads (`<side>_gripper` on an SO-101)
      from the grip, so the loop leaves it out.
    - slots: The joints of dream's spec of the arm, held ones included (the SO-101's six): the count an arm the
      wire drives must have (`EefWire.for_arms()`).
    """

    joints: tuple[int, ...]
    width: int
    client_gripper: bool
    slots: int


def zero_shot_embodiment(width: int, gripper: bool = False) -> dict[str, dict[str, Any]]:
    """The embodiment config of an arm with no fine-tune yet, on which Dream Labs' base models run.

    The setup wizard registers it on the account from the operator's own steps: #MAKERMODS_EMBODIMENT's parts in its
    order, each `<side>_arm` as wide as the arm, and the cameras #ZERO_SHOT_CAMERAS. An arm whose gripper drive the
    dream client makes (#ArmLayout `client_gripper`, the SO-101's LeRobot 0 to 100) also gets `<side>_gripper`, right
    then left, last: the client sends that drive under it, and its IK refuses the arm without it. Any other arm gets
    none: the dream client sends a base model the grip as its gripper.

    @param width The arm's joint slots, as wide as the station drives it (#ArmLayout).
    @param gripper Whether the dream client makes the arm's gripper drive (#ArmLayout `client_gripper`).
    @return `{"state", "action", "video"}`, each vector's spans tiling it from 0.
    """
    eef = [("right_eef_xyz", 3), ("right_eef_rot6", 6), ("right_grip", 1)]
    eef += [("left_eef_xyz", 3), ("left_eef_rot6", 6), ("left_grip", 1)]
    drives = [("right_gripper", 1), ("left_gripper", 1)] if gripper else []
    state = [("left_arm", width), ("right_arm", width), *eef, *drives]
    action = [*eef, ("left_arm", width), ("right_arm", width), *drives]
    video = {role: f"observation.images.{role}" for role in ZERO_SHOT_CAMERAS}
    return {"state": _spans(state), "action": _spans(action), "video": video}


def _spans(parts: list[tuple[str, int]]) -> dict[str, list[int]]:
    """Each part to its `[start, end)` span, laid end to end from 0 in order."""
    ends = np.cumsum([width for _, width in parts])
    return {part: [int(end - width), int(end)] for (part, width), end in zip(parts, ends, strict=True)}


def to_dream(obs: types.Observation, task: str) -> tuple[dict[str, str], dict[str, Any]]:
    """Builds one tick of `client.set_observation()`: the frame per window key, and the rest of the observation.

    Each camera's `Frame.jpeg`, the base64 JPEG its encode thread made last, goes under `image_window.<role>`;
    the client keeps the window of each key and sends it, oldest first, with the newest state. State arrays are
    cast to float32 (`[6]` joints, `[1]` grippers).

    @param obs The observation assembled this tick; camera names must be #CAMERA_ROLES.
    @param task The language prompt for the episode, sent as `prompt`.
    @return `(frames, observation)`: frames keyed `image_window.*`, one JPEG string each, and the observation keyed
        `prompt` and `state.*`; together they match embodiment_spec().
    @throws ControlLoopError A camera is not named after a serve role, or its frame carries no encoded frame.
    """
    right, left = obs.state.right, obs.state.left
    payload: dict[str, Any] = {
        PROMPT_KEY: task,
        "state.arm_r.joint_pos": np.asarray(right.positions_rad, dtype=np.float32),
        "state.arm_r.gripper_pos": np.asarray([right.gripper], dtype=np.float32),
        "state.arm_l.joint_pos": np.asarray(left.positions_rad, dtype=np.float32),
        "state.arm_l.gripper_pos": np.asarray([left.gripper], dtype=np.float32),
    }
    frames: dict[str, str] = {}
    for name, frame in obs.images.items():
        if name not in CAMERA_ROLES:
            raise errors.ControlLoopError(
                f"camera '{name}' is not a serve camera role; expected one of {CAMERA_ROLES}"
            )
        frames[f"{IMAGE_WINDOW_PREFIX}{name}"] = _encoded(name, frame)
    return frames, payload


def from_dream(
    action: dict[str, Any],
    observed_at_s: float,
    horizon: int,
    joints: tuple[int, ...] = ALL_JOINTS,
    slots: int = ARM_DOF,
) -> types.ActionChunk:
    """Turns the action dict `client.get_action` returned into an #robots.types.ActionChunk.

    Each of `q_r_chunk` / `q_l_chunk` is read as `[horizon, len(joints) + 1]`: the joint targets in radians, then
    the gripper, clipped to `[0, 1]`. The targets fill the arm's `joints` slots of its `slots`; any other slot reads 0.
    Arrays come out float64 regardless of what the wire carried; extra keys (`q_r`, `q_l`, `per_key`) are ignored.

    @param action The serve's action dict.
    @param observed_at_s Monotonic time of the observation this chunk answers, seconds.
    @param horizon The checkpoint's action horizon; every key must carry exactly this many rows.
    @param joints The joint slots the arm drives (#ArmLayout): every slot on the YAM wire.
    @param slots Each arm's joints (#ArmLayout): each command's width. Six on the YAM wire.
    @return The chunk, `len(actions) == horizon`.
    @throws ControlLoopError A key is missing, or its shape is not `[horizon, len(joints) + 1]`.
    """
    right = _chunk_rows(action, "q_r_chunk", horizon, len(joints) + 1)
    left = _chunk_rows(action, "q_l_chunk", horizon, len(joints) + 1)
    actions = [
        types.Action(left=_arm_command(left[i], joints, slots), right=_arm_command(right[i], joints, slots))
        for i in range(horizon)
    ]
    return types.ActionChunk(actions=actions, observed_at_s=observed_at_s)


def commanded_joints(
    action: types.Action | None, joints: tuple[int, ...] = ALL_JOINTS
) -> dict[str, npt.NDArray[np.float64]]:
    """The joints the loop last commanded each arm, keyed by its chunk key, for the dream client's `commanded`.

    The client never sends them: its seam join, and on a Makers Arm its IK, take over from the row after them, where
    the loop is on the previous chunk (`dream.DreamClient.get_action`). The loop reads its observation before it
    submits the tick's row, so these are the row before it; the client adds the one row itself. They are the row the
    loop submitted, not the arms' last write (`ArmState.commanded_rad`): the command thread eases each arm toward a
    row over about one tick, so that write trails the row by a fraction of one.

    @param action The row the loop submitted last, or None before its first.
    @param joints The joint slots the arm drives (#ArmLayout): every slot on the YAM wire.
    @return `{"q_r_chunk": [n], "q_l_chunk": [n]}` radians, float64, `n = len(joints)`; empty before the first row.
    """
    if action is None:
        return {}
    return {
        key: np.asarray(command.positions_rad, dtype=np.float64)[list(joints)]
        for key, command in zip(ACTION_KEYS, (action.right, action.left), strict=True)
    }


def embodiment_spec(camera_names: list[str], horizon: int) -> dict[str, Any]:
    """Builds the `embodiment` block that makes `dream`'s strict mode agree with this adapter.

    Declares exactly the keys to_dream() hands the client and from_dream() reads: float32 `[6]` / `[1]`
    state, one `image_window.<role>` entry per camera (presence only: the window of JPEGs the client sends has no
    array shape), and `[horizon, 7]` action chunks.

    @param camera_names The camera roles in config order.
    @param horizon The checkpoint's action horizon.
    @return The nested `{group: {key: {"shape": [...], "dtype": "..."}}}` dict.
    """
    state: dict[str, Any] = {}
    for key in STATE_KEYS:
        dim = 1 if key.endswith(".gripper_pos") else ARM_DOF
        state[key.removeprefix("state.")] = {"shape": [dim], "dtype": _ARRAY_DTYPE}
    return {
        "state": state,
        "image_window": {name: {} for name in camera_names},
        "action": {key: {"shape": [horizon, ACTION_DIM], "dtype": _ARRAY_DTYPE} for key in ACTION_KEYS},
    }


class Wire(Protocol):
    """What a body speaks to its model: the observation it sends, the block dream's strict mode checks it with, and
    how it reads the joint chunks back.

    Attributes:
    - camera_roles: The camera names the wire accepts; a config's cameras must be named after these.

    @see YamWire, EefWire
    """

    camera_roles: tuple[str, ...]

    def to_dream(self, obs: types.Observation, task: str) -> tuple[dict[str, str], dict[str, Any]]:
        """Builds one tick of `client.set_observation()`.

        @param obs The observation assembled this tick.
        @param task The language prompt.
        @return `(frames, observation)`: one JPEG string per window key, and every other key.
        """

    def embodiment(self, camera_names: list[str], horizon: int) -> dict[str, Any]:
        """Builds the `embodiment` block of the dream config.

        @param camera_names The camera roles in config order.
        @param horizon The checkpoint's action horizon.
        @return The block.
        """

    def from_dream(self, action: dict[str, Any], observed_at_s: float, horizon: int) -> types.ActionChunk:
        """Reads the client's reply into a chunk (from_dream()).

        @param action The reply.
        @param observed_at_s Monotonic time of the observation it answers, seconds.
        @param horizon The checkpoint's action horizon.
        @return The chunk.
        """

    def commanded(self, action: types.Action | None) -> dict[str, npt.NDArray[np.float64]]:
        """The joints the loop last commanded each arm, for the client's `commanded` (commanded_joints()).

        @param action The row the loop submitted last, or None.
        @return Per chunk key, the joints.
        """

    def for_arms(self, dofs: Mapping[str, int]) -> "Wire":
        """The wire, once each arm it drives has the joints of the wire's model of it, as wide as every command.

        @param dofs Per arm, `"left"` and `"right"`, its joints (`robots.followers.FollowerArm.dof`).
        @return This wire.
        @throws ConfigError An arm has another count of joints than the wire's model of it.
        """


class YamWire:
    """The YAM station's wire: to_dream(), embodiment_spec(), from_dream() and commanded_joints(), looked up when
    called (tests replace them).

    Attributes:
    - camera_roles: #CAMERA_ROLES.
    """

    camera_roles: tuple[str, ...] = CAMERA_ROLES

    def to_dream(self, obs: types.Observation, task: str) -> tuple[dict[str, str], dict[str, Any]]:
        """to_dream().

        @param obs The observation assembled this tick.
        @param task The language prompt.
        @return `(frames, observation)`.
        """
        return to_dream(obs, task)

    def embodiment(self, camera_names: list[str], horizon: int) -> dict[str, Any]:
        """embodiment_spec().

        @param camera_names The camera roles in config order.
        @param horizon The checkpoint's action horizon.
        @return The block.
        """
        return embodiment_spec(camera_names, horizon)

    def from_dream(self, action: dict[str, Any], observed_at_s: float, horizon: int) -> types.ActionChunk:
        """from_dream().

        @param action The reply.
        @param observed_at_s Monotonic time of the observation it answers, seconds.
        @param horizon The checkpoint's action horizon.
        @return The chunk.
        """
        return from_dream(action, observed_at_s, horizon)

    def commanded(self, action: types.Action | None) -> dict[str, npt.NDArray[np.float64]]:
        """commanded_joints().

        @param action The row the loop submitted last, or None.
        @return Per chunk key, the joints.
        """
        return commanded_joints(action)

    def for_arms(self, dofs: Mapping[str, int]) -> "YamWire":
        """This wire, once both arms are a YAM's six joints.

        @param dofs Per arm, its joints.
        @return This wire.
        @throws ConfigError An arm does not have six joints.
        """
        _check_yam_arms(dofs)
        return self


## The station wire every YAM spoke before it spoke its model's own (#YAM_VIDEO_WIRE): deprecated, no body speaks it.
YAM_WIRE = YamWire()


class EefWire:
    """The wire of a FaaS fine-tune, built from the embodiment config its model was trained on; also the YAM's own
    wire (#YAM_VIDEO_WIRE), whose config names no tool pose.

    The config's state parts the loop fills, per arm, right first: `<side>_arm` float32, the arm's joints in radians
    (#ArmLayout), `<side>_gripper` float32 `[1]`, the gripper in the datasets' own unit (`gripper * grip_scale`, the
    stroke in mm on a MakerMods arm) unless the dream client makes it, and `<side>_grip` float32 `[1]`, the loop's
    `[0, 1]` gripper (1 open). The tool poses, `<side>_eef_*`, are the dream client's to add from `<side>_arm`
    (`dream.client_ik`). Each camera's frame goes under `video.<role>`, one of the config's `video` roles; the
    client keeps its window.

    Attributes:
    - camera_roles: The config's `video` roles, in its order.

    @see to_dream(), embodiment()
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        grip_scale: float | None,
        body: str,
        layout: ArmLayout,
        roles: Mapping[str, str] | None = None,
    ) -> None:
        """Reads the config's parts once; nothing is per tick.

        @param config The embodiment config (`state`, `action` and `video`, as `dream.DreamClient.add_embodiment`
            takes it; #MAKERMODS_EMBODIMENT for a MakerMods arm).
        @param grip_scale What `<side>_gripper` reads with the gripper fully open (the loop's 1.0), or None when the
            config has no such part or the dream client makes it.
        @param body The body, for messages.
        @param layout How the arm's joints and gripper go out, from dream's spec of the arm; both arms share it.
        @param roles The camera names the config may use, each with the `video` role it goes out as (#YAM_VIDEO_ROLES
            on a YAM); None takes the config's `video` roles as the names.
        @throws ConfigError A state part is not one the loop or the client fills (#EEF_WIRE_PARTS, or a
            `<side>_eef_*` part), is not as wide as it (`<side>_arm` as `layout.width`, the others 1), or is
            `<side>_gripper` with no `grip_scale` and no client to make it.
        """
        state: Mapping[str, list[int]] = config["state"]
        widths = {"arm": layout.width, "gripper": 1, "grip": 1}
        for part, span in state.items():
            side, _, kind = part.partition("_")
            if side not in ("left", "right") or not (kind in EEF_WIRE_PARTS or kind.startswith("eef_")):
                raise errors.ConfigError(
                    f"the {body} embodiment's state part {part!r} is not one the loop fills: it fills "
                    f"<side>_{', <side>_'.join(EEF_WIRE_PARTS)} (side left or right), and the dream client adds "
                    "<side>_eef_*"
                )
            if kind in widths and span[1] - span[0] != widths[kind]:
                raise errors.ConfigError(
                    f"the {body} embodiment's state part {part!r} spans {span}, expected {widths[kind]} wide"
                )
        if grip_scale is None and not layout.client_gripper and any(part.endswith("_gripper") for part in state):
            raise errors.ConfigError(
                f"the {body} embodiment records <side>_gripper: give its value with the gripper fully open "
                "('grip_scale')"
            )
        sent = [kind for kind in EEF_WIRE_PARTS if not (kind == "gripper" and layout.client_gripper)]
        ## `(observation key, side, kind)` per part the loop fills, in send order.
        self._parts = [
            (f"state.{side}_{kind}", side, kind)
            for side in ("right", "left")
            for kind in sent
            if f"{side}_{kind}" in state
        ]
        self._grip_scale = 0.0 if grip_scale is None else grip_scale
        self._joints = layout.joints
        self._layout = layout
        # A view of the first slots when the arm drives them in order; otherwise the slots it drives.
        in_order = layout.joints == tuple(range(len(layout.joints)))
        self._driven: slice | list[int] = slice(0, len(layout.joints)) if in_order else list(layout.joints)
        self._config = config
        self._body = body
        self._roles = dict(roles) if roles is not None else {name: name for name in config["video"]}
        self.camera_roles = tuple(self._roles)

    def to_dream(self, obs: types.Observation, task: str) -> tuple[dict[str, str], dict[str, Any]]:
        """Builds one tick of `client.set_observation()`: a frame per camera and the state parts the loop fills.

        @param obs The observation assembled this tick; camera names must be #camera_roles.
        @param task The language prompt for the episode, sent as `prompt`.
        @return `(frames, observation)`: frames keyed `video.<role>` (each camera's role, #YAM_VIDEO_ROLES on a YAM),
            one JPEG string each, and the observation keyed `prompt` and `state.*`; together they match embodiment().
        @throws ControlLoopError A camera is not one of the roles, or its frame carries no encoded frame.
        """
        payload: dict[str, Any] = {PROMPT_KEY: task}
        for key, side, kind in self._parts:
            arm = obs.state.right if side == "right" else obs.state.left
            if kind == "arm":
                payload[key] = np.asarray(arm.positions_rad[self._driven], dtype=np.float32)
            elif kind == "gripper":
                payload[key] = np.asarray([arm.gripper * self._grip_scale], dtype=np.float32)
            else:
                payload[key] = np.asarray([arm.gripper], dtype=np.float32)
        frames: dict[str, str] = {}
        for name, frame in obs.images.items():
            if name not in self.camera_roles:
                raise errors.ControlLoopError(
                    f"camera '{name}' is not a {self._body} camera role; expected one of {self.camera_roles}"
                )
            frames[f"{VIDEO_PREFIX}{self._roles[name]}"] = _encoded(name, frame)
        return frames, payload

    def embodiment(self, camera_names: list[str], horizon: int) -> dict[str, Any]:
        """Builds the `embodiment` block: the config, its `video` cut to the cameras the config has.

        Strict mode then checks every `state.<part>` is as wide as its span, every `video.<role>` is a frame window,
        and the reply's `per_key` carries every `action.<part>`. An embodiment config has no action
        shape, so the horizon is not in it.

        @param camera_names The camera roles in config order.
        @param horizon The checkpoint's action horizon; unused.
        @return `{"state", "action", "video"}`, new lists and dicts on every call.
        """
        del horizon
        keys = [self._config["video"][self._roles[name]] for name in camera_names]
        if len(set(keys)) != len(keys):
            raise errors.ConfigError(
                f"cameras {camera_names} name one {self._body} camera twice; give each camera one name"
            )
        return {
            "state": {part: list(span) for part, span in self._config["state"].items()},
            "action": {part: list(span) for part, span in self._config["action"].items()},
            "video": {self._roles[name]: self._config["video"][self._roles[name]] for name in camera_names},
        }

    def from_dream(self, action: dict[str, Any], observed_at_s: float, horizon: int) -> types.ActionChunk:
        """from_dream() on the arm's joints.

        @param action The reply.
        @param observed_at_s Monotonic time of the observation it answers, seconds.
        @param horizon The checkpoint's action horizon.
        @return The chunk.
        """
        return from_dream(action, observed_at_s, horizon, self._joints, self._layout.slots)

    def commanded(self, action: types.Action | None) -> dict[str, npt.NDArray[np.float64]]:
        """commanded_joints() on the arm's joints.

        @param action The row the loop submitted last, or None.
        @return Per chunk key, the joints.
        """
        return commanded_joints(action, self._joints)

    def for_arms(self, dofs: Mapping[str, int]) -> "EefWire":
        """This wire, once each arm has dream's spec's joints, as many as its home lists.

        An arm with more would have its extra joints commanded 0 on every row, unseen by the end-effector speed
        clamp (`robots.kinematics`); one with fewer cannot take the rows.

        @param dofs Per arm, `"left"` and `"right"`, its joints (`robots.followers.FollowerArm.dof`).
        @return This wire.
        @throws ConfigError An arm has another count of joints than dream's spec of it.
        """
        for side, dof in dofs.items():
            if dof != self._layout.slots:
                raise errors.ConfigError(
                    f"arm '{side}' has {dof} joints (its home in the config lists {dof} angles), but dream's model of "
                    f"the {self._body} arm has {self._layout.slots}: give its home one angle per joint of the model, "
                    "and have its driver read them all"
                )
        return self


def _check_yam_arms(dofs: Mapping[str, int]) -> None:
    """Raises unless both arms have a YAM's six joints, the only width the YAM's wires carry.

    @param dofs Per arm, its joints.
    @throws ConfigError An arm does not have six.
    """
    for side, dof in dofs.items():
        if dof != ARM_DOF:
            raise errors.ConfigError(
                f"arm '{side}' has {dof} joints (its home in the config lists {dof} angles); a YAM's arm has {ARM_DOF}"
            )


def _encoded(name: str, frame: types.Frame) -> str:
    """One camera's frame for the client: the JPEG its encode thread made last.

    @param name The camera, for the message.
    @param frame Its frame this tick.
    @return The base64 JPEG string.
    @throws ControlLoopError The frame carries no encoded frame; a reader encodes its first capture before it
        publishes, so only a hand-built frame gets here.
    """
    if not frame.jpeg:
        raise errors.ControlLoopError(
            f"camera '{name}' frame carries no encoded frame; the policy needs one per camera"
        )
    return frame.jpeg


## The YAM's wire (`robots.bodies.YAM`): #YAM_EMBODIMENT under the station's camera names,
## every joint driven, the gripper sent as itself (the loop's `[0, 1]`, so a grip scale of 1). The model answers in
## the YAM's own joints, which the serve packs into the joint chunks from_dream() reads, so no `ik` goes with it.
YAM_VIDEO_WIRE = EefWire(
    YAM_EMBODIMENT,
    1.0,
    "yam",
    ArmLayout(ALL_JOINTS, ARM_DOF, client_gripper=False, slots=ARM_DOF),
    roles=YAM_VIDEO_ROLES,
)
## The YAM's wire on the EEF head (#YAM_POSE_EMBODIMENT): #YAM_VIDEO_WIRE's parts and cameras, plus the loop's
## `[0, 1]` gripper as `<side>_grip`; the poses are the dream client's to add (`robots.run.policy_config`).
YAM_POSE_WIRE = EefWire(
    YAM_POSE_EMBODIMENT,
    1.0,
    "yam",
    ArmLayout(ALL_JOINTS, ARM_DOF, client_gripper=False, slots=ARM_DOF),
    roles=YAM_VIDEO_ROLES,
)


def _chunk_rows(action: dict[str, Any], key: str, horizon: int, width: int) -> npt.NDArray[np.float64]:
    """Reads one chunk key as float64 `[horizon, width]`, or raises naming the key and shapes.

    @param action The serve's action dict.
    @param key The chunk key to read, one of #ACTION_KEYS.
    @param horizon The checkpoint's action horizon; the row count the key must carry.
    @param width The arm's joints and the gripper.
    @return The `[horizon, width]` float64 array.
    @throws ControlLoopError The key is missing, or its shape is not `[horizon, width]`.
    """
    if key not in action:
        raise errors.ControlLoopError(f"action chunk is missing key '{key}'; got keys {sorted(action)}")
    rows = np.asarray(action[key], dtype=np.float64)
    if rows.shape != (horizon, width):
        raise errors.ControlLoopError(
            f"action key '{key}' has shape {rows.shape}, expected ({horizon}, {width}) for action_horizon={horizon}"
        )
    return rows


def _arm_command(row: npt.NDArray[np.float64], joints: tuple[int, ...], slots: int) -> types.ArmCommand:
    """Splits one row into joint targets and a gripper clipped to `[0, 1]`.

    @param row One action row: the arm's joint targets in radians then the gripper.
    @param joints The joint slots the targets fill; any other slot reads 0.
    @param slots The arm's joints: the targets' width.
    @return The #robots.types.ArmCommand for that row.
    """
    count = len(joints)
    if joints == tuple(range(slots)):
        positions = row[:slots]
    else:
        positions = np.zeros(slots)
        positions[list(joints)] = row[:count]
    return types.ArmCommand(positions_rad=positions, gripper=float(np.clip(row[count], 0.0, 1.0)))
