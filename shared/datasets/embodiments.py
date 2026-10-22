"""The embodiment config: the one JSON a customer writes to describe a robot body, and its one parser.

An embodiment on the platform is `username/name` plus this config, stored as `embodiments.embodiment_config`. Its
required shape is `{"state": {part: [start, end]}, "action": {part: [start, end]}, "video": {role: video_key}}`: the
named parts of the `observation.state` vector with their `[start, end)` spans, the same for `action`, and the camera
roles dream trains on mapped to the LeRobot feature key each one reads (`observation.images.overhead`). The state and
action maps are, verbatim, what the training metadata script takes as `--state-keys` and `--action-keys`.

One optional key, `head_camera`, names the role (a key of `video`) of the fixed head camera: the view the data
service sends to Gemini for dense annotation. Every recorder names that camera differently, so the config says which
of its roles it is; left out, the service takes the first head role the config has (`overhead`, `top`, `scene`,
`head`: the service's `HEAD_CAMERA_ROLES`), and the service registers no body with neither
(when it creates the body), since every dataset is labelled. Rate, prompt, the video augmentation and
the normalization statistic are the training recipe's, not the body's. `video` names at most #MAX_CAMERAS roles
(check_cameras(), at registration and on every job): a robot recorded with more cameras names the ones to train, and
its datasets keep the others, which training leaves out.

Three more optional keys say what each part is, as the New robot page asks. `parts` answers, for every part of `state`
and of `action`, its kind (#KINDS: an arm's joints, a gripper, a gripper's position or rotation, anything else, or a
part training leaves out), its arm (#SIDES) and what that kind needs (#Answer), and for an action part whether its
values are deltas (`"delta": true`: the change to the next step, which the data service turns into a target, #Answer);
`actions` says the actions are targets (#ACTIONS), and `parts` needs it; `urdf` (#Urdf) names the arm's URDF file by
its hash, the URDF joint each arm value drives, the values that turn their joint the other way and each arm's grasp
point, and needs `parts`. A body with `parts` trains on the #Plan parse() derives from them (_plan()): each part under
the name training keys on, rotations converted to 6D and grippers to 0 closed to 1 open, the gripper poses that cannot
train and the parts marked not trained left out, and the end-effector pose trained (`use_eef`) exactly when both arms'
positions, rotations and grippers are in both vectors and no URDF gives the poses from the joints. The data service
applies the plan (the payload's spans, the processed parquets, the job's `use_eef`); the job does not choose. A body
without `parts` trains its parts under their own names, and `use_eef` is the job's (below).

The parts of `state` and `action` are the robot's joints under any names the customer chooses, with one exception: a
fine-tune asked for with `use_eef` (end-effector, the pose of each gripper rather than the arm's joint angles) reads
the six parts #EEF_PARTS, by those exact names and at those exact widths, in BOTH `state` and `action`; every other
part is still joints, and training uses both. check_eef() is that rule: the data service runs it when a job is
created with `use_eef` and the dream client before sending one, and refuses a body that lacks a part or has one at
another width. For a body without
`parts`, nothing about its shape is inferred: it trains EEF only when its job says so. A body whose `action` carries
an EEF rotation part (#GRIPPER_FRAME_ROT_PARTS) trains only with `use_eef`: the recipe trains those parts in the
gripper frame under any tag, and dream's joints-only transform at the FaaS pin normalizes them, which its loader
refuses at init, so check_joints() refuses such a job at submit instead; it also refuses an action EEF position
(#GRIPPER_FRAME_XYZ_PARTS), which the recipe trains in the gripper frame too, when the state lacks the position and
rotation that frame is read from. Registration allows any `action` up to #MAX_DIMS: a body wider than the base's
action head would train at #MAX_DIMS with the values past the base's as their own weights (the base's width read from
its checkpoint), but the training service's dream version carries no such extension, so check_action_width() holds
the action to #PIN_ACTION_DIM at submit. The state has no such extension:
a body registers a state that trains up to #MAX_DIMS values, but a job trains only one the base reads
(#BASE_STATE_DIM), which check_state_width() holds at submit. Both caps count only what trains (trains()): a part
marked `not_trained` stays in the data, so either vector may be wider.

Two keys a config may still carry, `state_space` and `action_space` (#RETIRED_KEYS), are retired: training never
reads them. check_registration() refuses them on a config being registered, with a line saying so; parse() ignores
them on a stored body, so such a body still lists and takes datasets.

parse() is the one place the shape is enforced; the dream client runs it before checking a dataset against a config
(`validation`) and the data service when it reads a stored body, and both run check_registration() on a config being
registered, so a config the service accepts is a config the client accepts. It answers an #Embodiment, whose
`state_dim` and `action_dim` are derived (the largest span end), and refuses anything else with an
#EmbodimentInvalidError whose message is the one line the user sees, naming the key or part that is wrong. Standard
library only: the service image carries neither numpy nor pyarrow, and this module must import there.

The YAM body, right-first as every real YAM dataset is laid out:

```json
{
  "state": {"right_arm": [0, 6], "right_gripper": [6, 7], "left_arm": [7, 13], "left_gripper": [13, 14]},
  "action": {"right_arm": [0, 6], "right_gripper": [6, 7], "left_arm": [7, 13], "left_gripper": [13, 14]},
  "video": {
    "overhead": "observation.images.overhead",
    "wrist_l": "observation.images.wrist_l",
    "wrist_r": "observation.images.wrist_r"
  },
  "head_camera": "overhead"
}
```
"""

import dataclasses
import itertools
import math
import re
from typing import Any

## The most values of state or action a config may train (trains()); parts marked `not_trained` do not count.
MAX_DIMS: int = 128

## The widest state a job trains: the base model's state input (`max_state_dim` of the base models), which has
## no extension; check_state_width().
BASE_STATE_DIM: int = 64

## The widest action a job trains: the base's action head. The training service's dream version has no action
## extension, so a wider job fails at the shape preflight on the GPU node; check_action_width().
PIN_ACTION_DIM: int = 34

## The three keys a config must have.
CONFIG_KEYS: tuple[str, ...] = ("state", "action", "video")

## The keys a config may also have: the head camera, what each part is (#Answer), that the actions are absolute
## targets (#ACTIONS) and the arm's URDF (#Urdf).
OPTIONAL_KEYS: tuple[str, ...] = ("head_camera", "parts", "actions", "urdf")

## Keys a config may carry that nothing reads: parse() ignores them on a stored body, and check_registration()
## refuses them on a config being registered (#RETIRED_KEY_RULE).
RETIRED_KEYS: tuple[str, ...] = ("state_space", "action_space")

## How check_registration() refuses a retired key; the key is named first.
RETIRED_KEY_RULE: str = (
    "is no longer part of the embodiment config: training never read it. The parts of state and action are joint "
    "positions, and a fine-tune that trains the end-effector pose asks for it with use_eef on the job and names the "
    "six EEF parts by their fixed names; remove the key."
)

## The longest part name or camera role, and the longest video key, check_registration() takes: a fine-tuned body's
## names are listed to every account (`GET /v1/models`), so none may run to megabytes. Registration only, as
## #RETIRED_KEYS: a body already stored is read with parse() whatever its lengths.
NAME_MAX_CHARS: int = 64
VIDEO_KEY_MAX_CHARS: int = 256

## The most cameras a fine-tune trains, so the most roles `video` names (check_cameras()). A dataset recorded with more
## keeps them: `validation` takes it when `video` names this many of its streams, and training leaves the others
## out. The platform site's own `MAX_CAMERAS` is pinned to this one by its tests.
MAX_CAMERAS: int = 3

## The six parts a `use_eef` fine-tune reads, in BOTH `state` and `action`, by these names at these widths: each
## gripper's position (3), its orientation as a 6D rotation (6) and its opening (1), right then left. Two arms only;
## a single-arm EEF body is not supported. The base's EEF head was trained on one convention: xyz is the tool
## centre point (on YAM 0.1347 m along link_6's +z from the flange) in metres, rot6 is link_6's orientation as the
## first two columns of the rotation matrix, both in each arm's own base frame (the state parts reach the model as
## recorded, so the frame matters), and grip is the opening, 0 closed to 1 open (the Fine-tuning Tutorial,
## "End-effector pose (two-arm robots)"). The pipeline keys only on the names (training's default EEF transform reads
## them by name), so check_eef() cannot verify that convention: a flange pose or a row-major rot6 passes it.
EEF_PARTS: dict[str, int] = {
    "right_eef_xyz": 3,
    "right_eef_rot6": 6,
    "right_grip": 1,
    "left_eef_xyz": 3,
    "left_eef_rot6": 6,
    "left_grip": 1,
}

## The EEF parts the training recipe trains in the gripper frame whatever the job's tag, which the training
## service's joints-only transform normalizes and its loader then refuses ("gripper-frame rotation keys ... reach a
## normalizer"): in `action` they train only with `use_eef` (check_joints()). The grips train as joints without it.
GRIPPER_FRAME_ROT_PARTS: tuple[str, ...] = ("right_eef_rot6", "left_eef_rot6")

## The EEF positions the recipe also trains in the gripper frame whatever the tag, each mapped to the rotation that
## frame is read from: an action position becomes R0^T (pt - p0) from the state's same position (3 wide) and that
## rotation (6 wide) at the chunk's anchor, and the loader fails at sample time on the GPU node when the state lacks
## either, so check_joints() asks for both.
GRIPPER_FRAME_XYZ_PARTS: dict[str, str] = {"right_eef_xyz": "right_eef_rot6", "left_eef_xyz": "left_eef_rot6"}

## The wrist cameras' roles; every other role of a body is a scene camera. Training gives a body's cameras their slots
## by role: the scene cameras first, then these, left then right (the training service spells the same order again,
## because its image carries no `shared`, and its tests pin the two equal). So a model reads its one scene camera in
## the same slot whatever the body named it.
WRIST_ROLES: tuple[str, ...] = ("wrist_l", "wrist_r")

## The body the Dream-1 base models were trained on, as the platform's model record names it (`/v1/models`
## `embodiment`): a model trained on it reads YAM's joints and cameras whatever arm runs it.
YAM_BODY: str = "yam"

## The camera roles a model trained on #YAM_BODY reads, in their slots: the scene camera `overhead`, then the wrists
## (the YAM body in this module's docstring).
YAM_VIDEO_ROLES: tuple[str, ...] = ("overhead", *WRIST_ROLES)

## What a part name or a camera role may look like, as dream's `PART_RE` has it: lowercase letters, digits and
## underscores, starting with a letter (a part is a key of dream's state/action dicts; a role is half of a video key).
## Applied with fullmatch(): `$` alone would let a trailing newline through.
NAME_RE: re.Pattern[str] = re.compile(r"^[a-z][a-z0-9_]*$")

## A URDF file's SHA-256 as `urdf.sha256` gives it.
SHA256_RE: re.Pattern[str] = re.compile(r"^[0-9a-f]{64}$")

## How #NAME_RE reads in a refusal.
_NAME_RULE: str = "lowercase letters, digits and underscores, starting with a letter"

## What a part of `state` or `action` may be (#Answer `kind`): an arm's joints, a gripper's opening, a gripper's
## position, a gripper's rotation, anything else (a mobile base, a head, a hand), or a part the customer marks not
## trained: it registers and stays in the dataset, but training reads none of it and the dream client sends none of
## it and answers no target for it, so the robot holds it itself (a torso, a head, a base the policy need not move).
KINDS: tuple[str, ...] = ("arm", "gripper", "position", "rotation", "other", "not_trained")

## Which arm a part of any kind but `other` and `not_trained` belongs to; `single` on a one-arm robot, which trains
## as #SINGLE_SIDE.
SIDES: tuple[str, ...] = ("left", "right", "single")

## The side a one-arm robot's parts train as: YAM's data is right-first, and its right arm takes the head's first
## joint columns.
SINGLE_SIDE: str = "right"

## The formats a rotation part may be recorded in, each to its width; the data service converts each to the 6D of
## #EEF_PARTS (the first two columns of the rotation matrix, `[R00, R10, R20, R01, R11, R21]`). Euler angles turn
## about fixed axes in the order named (`euler_xyz`: roll about x, then pitch about y, then yaw about z, so
## R = Rz Ry Rx, URDF's `rpy`), in radians unless `_deg`; `axis_angle` is the rotation vector (angle times unit axis,
## radians); `rot6_rows` is the matrix's first two rows; `matrix_rows` its nine entries row by row, `matrix_cols`
## column by column.
ROTATION_FORMATS: dict[str, int] = {
    "quat_xyzw": 4,
    "quat_wxyz": 4,
    "euler_xyz": 3,
    "euler_zyx": 3,
    "euler_xyz_deg": 3,
    "euler_zyx_deg": 3,
    "axis_angle": 3,
    "rot6_cols": 6,
    "rot6_rows": 6,
    "matrix_rows": 9,
    "matrix_cols": 9,
}

## Where a position part's point is: the grasp point between the fingers, or the flange the gripper is bolted to.
## Only the grasp point trains (#EEF_PARTS); a flange position is recorded, not trained.
POSITION_POINTS: tuple[str, ...] = ("grasp", "flange")

## The frame a position part is in: its arm's own base, or one world frame for both arms. Only the arm's own base
## trains, as for #POSITION_POINTS.
POSITION_FRAMES: tuple[str, ...] = ("arm_base", "world")

## What `actions` may say: the actions are targets, each action part the value to reach unless its answer says
## `"delta": true` (#Answer). dream trains on absolute targets and derives the relative ones itself, so the data
## service turns a delta part into one before training, and the dream client a fine-tune's targets back into changes
## (`shared.datasets.conversion`).
ACTIONS: tuple[str, ...] = ("absolute",)

## The keys an #Answer of each kind has: the first tuple always; the second, a pose's convention, whenever given and
## required when that pose trains (_plan()).
ANSWER_KEYS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "arm": (("kind", "side"), ()),
    "gripper": (("kind", "side", "closed", "open"), ()),
    "position": (("kind", "side"), ("point", "frame")),
    "rotation": (("kind", "side"), ("format",)),
    "other": (("kind",), ()),
    "not_trained": (("kind",), ()),
}

## The width of a part of each kind that has a fixed one (a rotation's is its format's, #ROTATION_FORMATS).
KIND_WIDTHS: dict[str, int] = {"gripper": 1, "position": 3}

## The prefix an `other` part trains under, so no name the customer picks is one training keys on.
OTHER_PREFIX: str = "other_"


class EmbodimentInvalidError(ValueError):
    """A config that is not an embodiment; `str(error)` is the one line the user sees."""


@dataclasses.dataclass(frozen=True)
class Answer:
    """What one part of `state` or `action` is (its entry in `parts`), as the customer confirmed it.

    Attributes:
    - kind: One of #KINDS.
    - side: One of #SIDES; None for `other` and `not_trained`.
    - closed: A gripper's value fully closed, in its recorded units; None for every other kind.
    - open: A gripper's value fully open; the gripper trains as `(value - closed) / (open - closed)`, 0 to 1.
    - format: A rotation's #ROTATION_FORMATS key; None for other kinds and for a rotation that gives none.
    - point: A position's #POSITION_POINTS entry; None as for `format`.
    - frame: A position's #POSITION_FRAMES entry; None as for `format`.
    - delta: Whether an action part's values are the instantaneous delta, the change to the next step (`action_t =
      x(t+1) - x(t)`, where `x` is the state part training reads under the same name), which trains as the target
      `state_t + action_t` (`shared.datasets.conversion.to_targets`). `"delta": true` in the config; False for a state
      part and for an action part without it. A rotation or gripper part cannot be one.
    """

    kind: str
    side: str | None
    closed: float | None
    open: float | None
    format: str | None
    point: str | None
    frame: str | None
    delta: bool


@dataclasses.dataclass(frozen=True)
class Grasp:
    """One arm's grasp point and turn, in the frame of the URDF link it is on.

    `xyz` is metres from the origin of the link `link`, and `rpy` (URDF's roll, pitch, yaw in radians) turns that
    link's frame into the base model's tool frame: x the way the fingers close, y up, z out of the gripper.
    """

    link: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]


@dataclasses.dataclass(frozen=True)
class Urdf:
    """The arm's URDF as the config names it (`urdf`); the data service keeps the file itself beside the body.

    Attributes:
    - sha256: The file's SHA-256 in lowercase hex; the data service refuses a registration whose file has another.
    - joints: Each `state` part of kind `arm`, and any of kind `gripper`, to the URDF joint each of its values drives,
      in value order. Two arms naming the same joints share a one-arm URDF, drawn once per arm.
    - reversed: Each part in `joints` that has values turning their URDF joint the other way (the joint's value is
      minus the recorded one) to those values' places in it, counted from 0, in increasing order.
    - grasp: The side of each `state` arm part to its arm's #Grasp.
    """

    sha256: str
    joints: dict[str, tuple[str, ...]]
    reversed: dict[str, tuple[int, ...]]
    grasp: dict[str, Grasp]


@dataclasses.dataclass(frozen=True)
class Slot:
    """One part of a vector training reads: its name there, the customer's part it reads, and its width there (a
    rotation's is 6, whatever its format)."""

    name: str
    part: str
    width: int


@dataclasses.dataclass(frozen=True)
class Plan:
    """What training reads of a body with `parts` (_plan()), slot by slot in the customer's vector order.

    Attributes:
    - use_eef: Whether the gripper poses train (the job's `use_eef`, which the body decides).
    - state: The training `observation.state`. Under `use_eef` a gripper is read twice, as the joint
      `<side>_gripper` and the opening `<side>_grip`, as YAM's EEF data has both.
    - action: The training `action`.
    - recorded: `<vector>.<part>` of each part training leaves out (a pose part that cannot train, or one marked
      `not_trained`), to the line saying why.
    """

    use_eef: bool
    state: tuple[Slot, ...]
    action: tuple[Slot, ...]
    recorded: dict[str, str]

    def spans(self, vector: str) -> dict[str, tuple[int, int]]:
        """Each slot's `[start, end)` in the training vector, as the training metadata script takes it.

        @param vector `state` or `action`.
        @return Slot name to span, in slot order.
        """
        spans: dict[str, tuple[int, int]] = {}
        start = 0
        for slot in {"state": self.state, "action": self.action}[vector]:
            spans[slot.name] = (start, start + slot.width)
            start += slot.width
        return spans


@dataclasses.dataclass(frozen=True)
class Embodiment:
    """One robot body as parse() read it from its config.

    Attributes:
    - state: Part name to its `[start, end)` span of the `observation.state` vector; the spans tile `[0, state_dim)`.
    - action: Part name to its `[start, end)` span of the `action` vector; the spans tile `[0, action_dim)`.
    - video: Camera role to the LeRobot feature key of the video it reads.
    - head_camera: The role (a key of `video`) of the fixed head camera, the view dense annotation sends to
      Gemini; None when the config omits it, in which case the data service takes the first head role it has.
    - parts: `state` and `action` to each part's #Answer; None when the config has no `parts`.
    - urdf: The arm's #Urdf; None when the config names none.
    - plan: What training reads (#Plan); None without `parts`, when the parts train under their own names.
    """

    state: dict[str, tuple[int, int]]
    action: dict[str, tuple[int, int]]
    video: dict[str, str]
    head_camera: str | None
    parts: dict[str, dict[str, Answer]] | None
    urdf: Urdf | None
    plan: Plan | None

    @property
    def head_video_key(self) -> str | None:
        """The LeRobot feature key of the head camera (`video[head_camera]`), or None when none is named."""
        return self.video[self.head_camera] if self.head_camera is not None else None

    @property
    def state_dim(self) -> int:
        """The width of the `observation.state` vector: the end of its last span."""
        return max(end for _, end in self.state.values())

    @property
    def action_dim(self) -> int:
        """The width of the `action` vector: the end of its last span."""
        return max(end for _, end in self.action.values())


def parse(config: object) -> Embodiment:
    """The #Embodiment a config describes, or the one line saying why it describes none.

    A retired key (#RETIRED_KEYS) is ignored here, so a body stored with one still reads; check_registration() is
    where it is refused.

    @param config The config as JSON gave it (a dict, though anything is refused with a line rather than a crash).
    @return The #Embodiment.
    @throws EmbodimentInvalidError When the config is not an object of #CONFIG_KEYS and #OPTIONAL_KEYS (the unknown
        key is named); a part name or a role does not match #NAME_RE; a span is not two ints with `0 <= start < end`;
        the spans of `state` or `action` leave a gap or overlap; `state` or `action` trains more than #MAX_DIMS
        values (trains(), so a part marked `not_trained` does not count); `video` is empty or maps a role
        to something other than a non-empty string, or to a path (a slash, a backslash, a NUL, `.` or `..`); a
        `head_camera` that is not one of the `video` roles; `parts`, `actions` or `urdf` that _answers(), _urdf() or
        _plan() refuse, `actions` or `urdf` without `parts`, `parts` without `actions`, or `actions` other than
        #ACTIONS.
    """
    if not isinstance(config, dict):
        raise EmbodimentInvalidError("The embodiment config must be a JSON object with state, action and video.")
    for key in config:
        if key not in CONFIG_KEYS and key not in OPTIONAL_KEYS and key not in RETIRED_KEYS:
            raise EmbodimentInvalidError(
                f"Unknown key {key!r} in the embodiment config; it takes {', '.join(CONFIG_KEYS + OPTIONAL_KEYS)}."
            )
    for key in CONFIG_KEYS:
        if key not in config:
            raise EmbodimentInvalidError(f"The embodiment config has no {key!r}.")
    state = _spans("state", config["state"])
    action = _spans("action", config["action"])
    video = _video(config["video"])
    spans = {"state": state, "action": action}
    parts = _answers(config["parts"], spans) if "parts" in config else None
    for key in ("actions", "urdf"):
        if key in config and parts is None:
            raise EmbodimentInvalidError(f"{key} needs parts: what each part of state and action is.")
    if parts is not None and "actions" not in config:
        raise EmbodimentInvalidError(
            'parts needs actions: say the actions are absolute targets ("actions": "absolute").'
        )
    if parts is not None and config["actions"] not in ACTIONS:
        raise EmbodimentInvalidError(
            f'actions {config["actions"]!r}: say "absolute"; an action part whose values are deltas says '
            '"delta": true in parts.'
        )
    urdf = _urdf(config["urdf"], parts["state"], state) if parts is not None and "urdf" in config else None
    body = Embodiment(
        state=state,
        action=action,
        video=video,
        head_camera=_head_camera(config.get("head_camera"), video),
        parts=parts,
        urdf=urdf,
        plan=_plan(spans, parts, urdf) if parts is not None else None,
    )
    for what in ("state", "action"):
        if trains(body, what) > MAX_DIMS:
            raise EmbodimentInvalidError(width_line(what, trains(body, what), MAX_DIMS))
    return body


def check_registration(config: object) -> Embodiment:
    """The #Embodiment a config being registered describes, by parse() plus the refusals registration adds.

    Those refusals are a retired key, a name or video key over its length (#NAME_MAX_CHARS, #VIDEO_KEY_MAX_CHARS), more
    cameras than a fine-tune trains (check_cameras()), and a `urdf` whose joints are not each driven by one value
    (_joints_once()). The data service's `POST /v1/embodiments`
    and the client's `add_embodiment` run this; a body already stored is read with parse() alone, so a stored
    `state_space` or `action_space`, or a stored URDF mapping that names a joint twice, costs nothing.

    @param config The config as JSON gave it.
    @return The #Embodiment.
    @throws EmbodimentInvalidError Everything parse() refuses; a config carrying a key of #RETIRED_KEYS
        (#RETIRED_KEY_RULE, the key named first); a part name or role over #NAME_MAX_CHARS, or a video key over
        #VIDEO_KEY_MAX_CHARS, named by its first characters; more than #MAX_CAMERAS roles in `video`
        (check_cameras()); a `urdf` joint named twice (_joints_once()).
    """
    body = parse(config)
    assert isinstance(config, dict)  # parse() refused anything else
    for key in RETIRED_KEYS:
        if key in config:
            raise EmbodimentInvalidError(f"{key} {RETIRED_KEY_RULE}")
    for what, kind, names in (
        ("state", "part", body.state),
        ("action", "part", body.action),
        ("video", "role", body.video),
    ):
        for name in names:
            if len(name) > NAME_MAX_CHARS:
                raise EmbodimentInvalidError(
                    f"{what}: {kind} {name[:NAME_MAX_CHARS]!r}... is {len(name)} characters; at most {NAME_MAX_CHARS}."
                )
    for role, key in body.video.items():
        if len(key) > VIDEO_KEY_MAX_CHARS:
            raise EmbodimentInvalidError(
                f"video: role {role!r} names a video key of {len(key)} characters; at most {VIDEO_KEY_MAX_CHARS}."
            )
    check_cameras(body)
    if body.urdf is not None:
        _joints_once(body.urdf)
    return body


def check_cameras(body: Embodiment) -> None:
    """Refuses a body whose `video` names more cameras than a fine-tune trains (#MAX_CAMERAS).

    check_registration() runs it on a config being registered, the data service on every job (a body stored with
    more) and `validation` before a dataset is checked; parse() does not, so such a body still lists and reads.

    @param body The parsed body.
    @throws EmbodimentInvalidError Naming the roles, the cap and how to pick the cameras to train.
    """
    if len(body.video) > MAX_CAMERAS:
        raise EmbodimentInvalidError(
            f"video names {len(body.video)} cameras ({', '.join(body.video)}); fine-tuning trains at most "
            f"{MAX_CAMERAS}. Register a robot whose video names the {MAX_CAMERAS} to train: its dataset keeps the "
            "other camera streams, and training leaves them out."
        )


def _joints_once(urdf: Urdf) -> None:
    """Refuses a `urdf` whose part names a joint twice, or that names a joint another part names, unless the two are
    arms drawn on a one-arm URDF, which share all of their joints."""
    for part, names in urdf.joints.items():
        twice = next((name for name in names if names.count(name) > 1), None)
        if twice is not None:
            raise EmbodimentInvalidError(f"urdf.joints.{part} names {twice!r} twice; each value drives its own joint.")
    for part, other in itertools.combinations(urdf.joints, 2):
        shared = [name for name in urdf.joints[part] if name in urdf.joints[other]]
        if shared and urdf.joints[part] != urdf.joints[other]:
            raise EmbodimentInvalidError(
                f"urdf.joints.{part} and urdf.joints.{other} both name {shared[0]!r}; only two arms drawn on a "
                "one-arm URDF share joints, and then all of them."
            )


def trained(body: Embodiment) -> Embodiment:
    """The body as training reads it, which check_eef() and check_joints() run on.

    A body with `parts` reads as its #Plan's spans, under the names training keys on, and those spans are the
    payload's; any other body reads as it is.

    @param body A parsed body.
    @return The #Embodiment training reads, with no `parts`, `urdf` or plan of its own.
    """
    if body.plan is None:
        return body
    return Embodiment(
        state=body.plan.spans("state"),
        action=body.plan.spans("action"),
        video=body.video,
        head_camera=body.head_camera,
        parts=None,
        urdf=None,
        plan=None,
    )


def trains(body: Embodiment, what: str) -> int:
    """How many values of `state` or `action` training reads.

    They are its plan's slots, or every value of a body without `parts`; a part marked `not_trained` is not one of
    them. Both width caps count these (#MAX_DIMS at parse(), #BASE_STATE_DIM at check_state_width()).

    @param body A parsed body.
    @param what `state` or `action`.
    @return The count.
    """
    if body.plan is None:
        return {"state": body.state_dim, "action": body.action_dim}[what]
    return sum(slot.width for slot in {"state": body.plan.state, "action": body.plan.action}[what])


def width_line(what: str, trained: int, limit: int) -> str:
    """The line for a vector that trains more than `limit` values, the robot page's `widthLine()` word for word.

    @param what `state` or `action`.
    @param trained How many of its values train (trains()).
    @param limit The most that can train.
    @return The line.
    """
    return (
        f"The {what} trains {trained} values; at most {limit} can. Mark parts like the torso, head or base not "
        "trained."
    )


def check_state_width(body: Embodiment) -> None:
    """Refuses a body whose state trains more values than the base model reads (#BASE_STATE_DIM).

    Registration allows a state that trains up to #MAX_DIMS, as it does an action, but only the action trains past
    the base's width. Only what trains counts (trains()): the state may be wider with parts marked `not_trained`.
    The data service runs it on every job, and the training app once more on the payload.

    @param body The parsed body (parse()), whose widths the line names.
    @throws EmbodimentInvalidError width_line(), then where to mark the parts.
    """
    if trains(body, "state") > BASE_STATE_DIM:
        raise EmbodimentInvalidError(
            f"{width_line('state', trains(body, 'state'), BASE_STATE_DIM)} Register the robot again "
            "with them marked, and upload the dataset again."
        )


def check_action_width(body: Embodiment) -> None:
    """Refuses a body whose action trains more values than the training service can (#PIN_ACTION_DIM).

    A wider action needs dream's action extension, which the training service's dream version lacks, so such a job
    would fail at the shape preflight on the GPU node. Only what trains counts (trains()). The data service runs it
    on every job.

    @param body The parsed body (parse()), whose widths the line names.
    @throws EmbodimentInvalidError Naming the trained count, the dataset's width and the cap.
    """
    trained = trains(body, "action")
    if trained > PIN_ACTION_DIM:
        raise EmbodimentInvalidError(
            f"The action trains {trained} values (the dataset has {body.action_dim}); fine-tuning trains at most "
            f"{PIN_ACTION_DIM} for now. Mark the parts the policy need not drive (a torso, a head, a base) as not "
            "trained; they stay in the data."
        )


def check_joints(body: Embodiment) -> None:
    """Refuses a body a joints-only fine-tune (no `use_eef`) cannot train.

    That is one whose `action` carries an EEF rotation (#GRIPPER_FRAME_ROT_PARTS), or an EEF position
    (#GRIPPER_FRAME_XYZ_PARTS) that is not 3 wide or lacks, in `state`, its own position 3 wide and its rotation 6
    wide. The data service runs it when a job is created without `use_eef`, and the training app once more on the
    payload.

    @param body The parsed body of the dataset's embodiment.
    @throws EmbodimentInvalidError Naming the first such part.
    """
    for part in GRIPPER_FRAME_ROT_PARTS:
        if part in body.action:
            raise EmbodimentInvalidError(
                f"The embodiment's action part {part!r} is an end-effector rotation, which trains only with use_eef "
                "(a joints-only job fails on it); submit with use_eef=True or rename the part."
            )
    needed: list[tuple[str, str, dict[str, tuple[int, int]], str]] = []
    for xyz, rot in GRIPPER_FRAME_XYZ_PARTS.items():
        if xyz in body.action:
            needed += [
                (xyz, "action", body.action, xyz),
                (xyz, "state", body.state, xyz),
                (xyz, "state", body.state, rot),
            ]
    for xyz, side, parts, part in needed:
        start, end = parts.get(part, (0, 0))
        if end - start == EEF_PARTS[part]:
            continue
        has = f"it {end - start} wide" if part in parts else "no such part"
        raise EmbodimentInvalidError(
            f"The embodiment's action part {xyz!r} trains relative to the gripper pose, which needs the {side} part "
            f"{part!r} {EEF_PARTS[part]} wide; the embodiment config has {has}. Fix the part or rename {xyz!r}."
        )


def check_eef(body: Embodiment) -> None:
    """Refuses a body a `use_eef` fine-tune cannot train.

    The six #EEF_PARTS must be in both `state` and `action` at their exact widths.

    The data service runs it when a job is created with `use_eef`, the dream client before sending one, and the
    training app once more on the payload. It reads nothing but the config: no part name, camera or width other than
    these six says anything about how a body trains.

    @param body The parsed body of the dataset's embodiment.
    @throws EmbodimentInvalidError Naming the first missing or wrong part and its side (`state` or `action`).
    """
    for side, parts in (("state", body.state), ("action", body.action)):
        for part, width in EEF_PARTS.items():
            if part not in parts:
                raise EmbodimentInvalidError(
                    f"use_eef needs the {side} part {part!r} ({width} wide); the embodiment config has no such "
                    f"{side} part."
                )
            start, end = parts[part]
            if end - start != width:
                raise EmbodimentInvalidError(
                    f"use_eef needs the {side} part {part!r} {width} wide; the embodiment config has it "
                    f"{end - start} wide."
                )


def _spans(what: str, spans: Any) -> dict[str, tuple[int, int]]:
    """The `[start, end)` map of `state` or `action`, typed and checked to tile `[0, dim)`."""
    if not isinstance(spans, dict) or not spans:
        raise EmbodimentInvalidError(f"{what} must be a non-empty object of part name to [start, end].")
    parsed: dict[str, tuple[int, int]] = {}
    for part, span in spans.items():
        if not NAME_RE.fullmatch(part):
            raise EmbodimentInvalidError(f"{what}: part {part!r} is not a name dream takes; use {_NAME_RULE}.")
        if not (isinstance(span, list) and len(span) == 2 and all(_is_int(bound) for bound in span)):
            raise EmbodimentInvalidError(f"{what}: part {part!r} must be [start, end], two whole numbers.")
        start, end = int(span[0]), int(span[1])
        if start < 0 or end <= start:
            raise EmbodimentInvalidError(f"{what}: part {part!r} covers [{start}, {end}); it needs 0 <= start < end.")
        parsed[part] = (start, end)
    expected = 0
    for part, (start, end) in sorted(parsed.items(), key=lambda item: item[1]):
        if start > expected:
            raise EmbodimentInvalidError(
                f"{what}: nothing covers [{expected}, {start}); part {part!r} starts at {start}, expected {expected}."
            )
        if start < expected:
            raise EmbodimentInvalidError(f"{what}: part {part!r} covers [{start}, {end}), overlapping another part.")
        expected = end
    return parsed


def _video(video: Any) -> dict[str, str]:
    """The role-to-feature-key map, typed and checked."""
    if not isinstance(video, dict) or not video:
        raise EmbodimentInvalidError("video must be a non-empty object of camera role to the dataset's video key.")
    for role, key in video.items():
        if not NAME_RE.fullmatch(role):
            raise EmbodimentInvalidError(f"video: role {role!r} is not a name dream takes; use {_NAME_RULE}.")
        if not isinstance(key, str) or not key:
            raise EmbodimentInvalidError(
                f"video: role {role!r} must name the dataset's video key, a non-empty string."
            )
        # The key names a directory under each `videos/chunk-*/` (training renames it), so it must not be a path.
        if key in (".", "..") or any(char in key for char in "/\\\0"):
            raise EmbodimentInvalidError(
                f"video: role {role!r} names {key!r}; a video key is one name such as observation.images.overhead, "
                "not a path."
            )
    return dict(video)


def _head_camera(raw: Any, video: dict[str, str]) -> str | None:
    """The role of the fixed head camera, one of `video`'s roles; None when the config omits it."""
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw:
        raise EmbodimentInvalidError("head_camera must name one of the video roles, a non-empty string.")
    if raw not in video:
        raise EmbodimentInvalidError(f"head_camera: {raw!r} is not one of the video roles ({', '.join(video)}).")
    return raw


def _answers(raw: Any, spans: dict[str, dict[str, tuple[int, int]]]) -> dict[str, dict[str, Answer]]:
    """`parts` read: an #Answer for exactly every part of `state` and of `action`."""
    if not isinstance(raw, dict) or set(raw) != {"state", "action"}:
        raise EmbodimentInvalidError(
            "parts must be an object with state and action, saying what each of their parts is."
        )
    answers: dict[str, dict[str, Answer]] = {}
    for vector in ("state", "action"):
        given = raw[vector]
        if not isinstance(given, dict) or set(given) != set(spans[vector]):
            raise EmbodimentInvalidError(
                f"parts.{vector} must say what each {vector} part is, exactly these: {', '.join(spans[vector])}."
            )
        answers[vector] = {
            part: _answer(f"parts.{vector}.{part}", given[part], end - start, vector == "action")
            for part, (start, end) in spans[vector].items()
        }
    return answers


def _answer(where: str, raw: Any, width: int, acts: bool) -> Answer:
    """One part's #Answer: the keys #ANSWER_KEYS gives its kind (the convention all or none), each value one its key
    takes, and the width its kind or format has; an action part (`acts`) may add `"delta": true`, except a rotation or
    a gripper."""
    if not isinstance(raw, dict) or raw.get("kind") not in KINDS:
        raise EmbodimentInvalidError(f"{where} must be an object whose kind is one of {', '.join(KINDS)}.")
    kind = raw["kind"]
    delta = "delta" in raw
    if delta and not acts:
        raise EmbodimentInvalidError(f"{where}: delta is for action parts; a state part is the value it reads.")
    if delta and raw["delta"] is not True:
        raise EmbodimentInvalidError(f"{where}: delta must be true; leave it out for a part whose values are targets.")
    if delta and kind == "rotation":
        raise EmbodimentInvalidError(
            f"{where}: a rotation part cannot be a delta; turning a change of rotation into a target is not built, "
            "so record the absolute rotation."
        )
    if delta and kind == "not_trained":
        raise EmbodimentInvalidError(f"{where}: a part that does not train has no target, so it cannot be a delta.")
    if delta and kind == "gripper":
        raise EmbodimentInvalidError(
            f"{where}: a gripper part cannot be a delta; its closed and open values are apertures to reach, so record "
            "the absolute aperture."
        )
    always, convention = ANSWER_KEYS[kind]
    if set(raw) - {"delta"} not in (set(always), set(always + convention)):
        takes = ", ".join(always) + (f", and {' and '.join(convention)} when it trains" if convention else "")
        raise EmbodimentInvalidError(f"{where}: a {kind} part takes {takes}; it has {', '.join(sorted(raw))}.")
    for key, allowed in (("side", SIDES), ("point", POSITION_POINTS), ("frame", POSITION_FRAMES)):
        if key in raw and raw[key] not in allowed:
            raise EmbodimentInvalidError(f"{where}: {key} {raw[key]!r} is not one of {', '.join(allowed)}.")
    if "format" in raw and raw["format"] not in ROTATION_FORMATS:
        raise EmbodimentInvalidError(f"{where}: format {raw['format']!r} is not one of {', '.join(ROTATION_FORMATS)}.")
    fixed = ROTATION_FORMATS[raw["format"]] if "format" in raw else KIND_WIDTHS.get(kind)
    if fixed is not None and width != fixed:
        raise EmbodimentInvalidError(f"{where}: this {kind} is {fixed} wide; the part is {width}.")
    if kind == "gripper" and not (
        _is_number(raw["closed"]) and _is_number(raw["open"]) and raw["closed"] != raw["open"]
    ):
        raise EmbodimentInvalidError(
            f"{where}: closed and open must be two different numbers, the gripper's values fully closed and open."
        )
    return Answer(
        kind=kind,
        side=raw["side"] if "side" in raw else None,
        closed=float(raw["closed"]) if kind == "gripper" else None,
        open=float(raw["open"]) if kind == "gripper" else None,
        format=raw["format"] if "format" in raw else None,
        point=raw["point"] if "point" in raw else None,
        frame=raw["frame"] if "frame" in raw else None,
        delta=delta,
    )


def _urdf(raw: Any, answers: dict[str, Answer], state: dict[str, tuple[int, int]]) -> Urdf:
    """`urdf` read against the state's answers: the file's hash, a URDF joint per value of every arm part (and of a
    gripper part the customer mapped), the values that turn their joint the other way, and a grasp point per arm.
    That each value drives its own joint is a registration rule (_joints_once()), so a stored body parses either
    way."""
    if not isinstance(raw, dict) or set(raw) != {"sha256", "joints", "reversed", "grasp"}:
        raise EmbodimentInvalidError("urdf must be an object with sha256, joints, reversed and grasp.")
    if not (isinstance(raw["sha256"], str) and SHA256_RE.fullmatch(raw["sha256"])):
        raise EmbodimentInvalidError("urdf.sha256 must be the URDF file's SHA-256, 64 lowercase hex digits.")
    arms = [part for part, answer in answers.items() if answer.kind == "arm"]
    grippers = [part for part, answer in answers.items() if answer.kind == "gripper"]
    joints = raw["joints"]
    if not arms:
        raise EmbodimentInvalidError("urdf needs a state part of kind arm: the model moves by the arm's joint values.")
    if not (isinstance(joints, dict) and set(arms) <= set(joints) <= set(arms + grippers)):
        raise EmbodimentInvalidError(
            f"urdf.joints must map every state arm part ({', '.join(arms)}), and may map a gripper part, to the "
            "URDF joints their values drive."
        )
    for part, names in joints.items():
        width = state[part][1] - state[part][0]
        if not (isinstance(names, list) and len(names) == width and all(isinstance(n, str) and n for n in names)):
            raise EmbodimentInvalidError(f"urdf.joints.{part} must name {width} URDF joints, one per value of {part}.")
    flipped = raw["reversed"]
    if not (isinstance(flipped, dict) and set(flipped) <= set(joints)):
        raise EmbodimentInvalidError(
            "urdf.reversed must map parts urdf.joints maps to the values that turn their joint the other way."
        )
    for part, places in flipped.items():
        width = len(joints[part])
        if not (
            isinstance(places, list)
            and places
            and all(isinstance(k, int) and not isinstance(k, bool) and 0 <= k < width for k in places)
            and places == sorted(set(places))
        ):
            raise EmbodimentInvalidError(
                f"urdf.reversed.{part} must list, in increasing order and at least one, which of its {width} values "
                "(counted from 0) turn their URDF joint the other way."
            )
    sides = sorted({str(answers[part].side) for part in arms})
    grasp = raw["grasp"]
    if not (isinstance(grasp, dict) and set(grasp) == set(sides)):
        raise EmbodimentInvalidError(f"urdf.grasp must give each arm's grasp point: {', '.join(sides)}.")
    points: dict[str, Grasp] = {}
    for side, point in grasp.items():
        if not (
            isinstance(point, dict)
            and set(point) == {"link", "xyz", "rpy"}
            and isinstance(point["link"], str)
            and point["link"]
            and all(
                isinstance(point[key], list) and len(point[key]) == 3 and all(_is_number(v) for v in point[key])
                for key in ("xyz", "rpy")
            )
        ):
            raise EmbodimentInvalidError(
                f'urdf.grasp.{side} must be {{"link": a URDF link, "xyz": three numbers in metres, "rpy": three '
                "numbers in radians}."
            )
        x, y, z = (float(v) for v in point["xyz"])
        roll, pitch, yaw = (float(v) for v in point["rpy"])
        points[side] = Grasp(link=point["link"], xyz=(x, y, z), rpy=(roll, pitch, yaw))
    return Urdf(
        sha256=raw["sha256"],
        joints={part: tuple(names) for part, names in joints.items()},
        reversed={part: tuple(places) for part, places in flipped.items()},
        grasp=points,
    )


def _rest(answers: dict[str, dict[str, Answer]], urdf: Urdf | None) -> str | None:
    """None when the gripper poses train; else the line saying why they are recorded, not trained."""
    if urdf is not None and any(answer.kind == "arm" for answer in answers["action"].values()):
        return "with a URDF and joint actions the gripper poses come from the joints, which is not built yet"
    if all(answer.side in (None, "single") for vector in answers.values() for answer in vector.values()):
        return "the base model trains gripper poses on two-arm robots only"
    missing = [
        f"{vector} {side} {kind}"
        for vector in ("state", "action")
        for side in ("left", "right")
        for kind in ("position", "rotation", "gripper")
        if not any(answer.kind == kind and answer.side == side for answer in answers[vector].values())
    ]
    if missing:
        return (
            f"gripper poses train with both arms' position, rotation and gripper in state and action; no {missing[0]}"
        )
    return None


def _off_convention(answers: dict[str, dict[str, Answer]]) -> str | None:
    """None when every position gives the grasp point in its arm's own base frame, the one the base model reads."""
    for vector in ("state", "action"):
        for part, answer in answers[vector].items():
            if answer.kind == "position" and (answer.point, answer.frame) != ("grasp", "arm_base"):
                return (
                    f"the base model reads the grasp point in each arm's own base frame and {vector}.{part} is the "
                    f"{answer.point} in the {answer.frame} frame, which is not converted yet"
                )
    return None


def _names(vector: str, part: str, answer: Answer, width: int, use_eef: bool) -> list[tuple[str, int]]:
    """The names and widths one trained part takes in its training vector (#Plan)."""
    side = SINGLE_SIDE if answer.side == "single" else answer.side
    if answer.kind == "arm":
        return [(f"{side}_arm", width)]
    if answer.kind == "gripper" and not use_eef:
        return [(f"{side}_gripper", 1)]
    if answer.kind == "gripper":
        return [(f"{side}_gripper", 1), (f"{side}_grip", 1)] if vector == "state" else [(f"{side}_grip", 1)]
    if answer.kind == "position":
        return [(f"{side}_eef_xyz", 3)]
    if answer.kind == "rotation":
        return [(f"{side}_eef_rot6", 6)]
    return [(f"{OTHER_PREFIX}{part}", width)]


def _plan(
    spans: dict[str, dict[str, tuple[int, int]]], answers: dict[str, dict[str, Answer]], urdf: Urdf | None
) -> Plan:
    """What training reads of a body with `parts`, or the line saying why it cannot train.

    Each pose part gives its convention when _rest() finds no reason the gripper poses cannot train (it may give it
    always), and they train when its positions are also the grasp point in each arm's own base frame
    (_off_convention()); otherwise they are recorded, not trained, as is every part marked `not_trained`. Refused: a
    one-arm part beside a left or right one, two parts training under one name, an action arm whose state lacks that
    arm at its width (dream trains arm targets relative to it), a delta action part whose state lacks the part it is
    added to at its width, an action that drives no arm.
    """
    sides = {answer.side for vector in answers.values() for answer in vector.values() if answer.side is not None}
    if "single" in sides and sides != {"single"}:
        raise EmbodimentInvalidError(
            "parts: one part is on a single arm and another on the left or right; a one-arm robot's parts are all "
            "single, a two-arm robot's left or right."
        )
    shape = _rest(answers, urdf)
    why = shape if shape is not None else _off_convention(answers)
    slots: dict[str, list[Slot]] = {"state": [], "action": []}
    recorded: dict[str, str] = {}
    for vector, out in slots.items():
        for part, (start, end) in sorted(spans[vector].items(), key=lambda item: item[1]):
            answer = answers[vector][part]
            if answer.kind == "not_trained":
                recorded[f"{vector}.{part}"] = "marked not trained"
                continue
            if answer.kind in ("position", "rotation"):
                convention = ANSWER_KEYS[answer.kind][1]
                given = answer.format is not None or answer.point is not None
                if shape is None and not given:
                    raise EmbodimentInvalidError(
                        f"parts.{vector}.{part}: the gripper poses train, so this {answer.kind} needs its "
                        f"{' and '.join(convention)}."
                    )
                if why is not None:
                    recorded[f"{vector}.{part}"] = why
                    continue
            for name, width in _names(vector, part, answer, end - start, why is None):
                taken = next((slot.part for slot in out if slot.name == name), None)
                if taken is not None:
                    raise EmbodimentInvalidError(
                        f"parts.{vector}: {taken} and {part} both train as {name}; an arm has one part of each kind."
                    )
                out.append(Slot(name=name, part=part, width=width))
    arms = {f"{side}_arm" for side in ("left", "right")}
    state = {slot.name: slot.width for slot in slots["state"]}
    for slot in slots["action"]:
        if slot.name in arms and state.get(slot.name) != slot.width:
            raise EmbodimentInvalidError(
                f"parts.action.{slot.part}: arm targets train relative to the same arm's joints, so the state needs "
                f"that arm's joints {slot.width} wide."
            )
        if answers["action"][slot.part].delta and state.get(slot.name) != slot.width:
            raise EmbodimentInvalidError(
                f"parts.action.{slot.part}: its values are deltas, which become targets by adding the state part "
                f"that trains as {slot.name}, so the state needs that part {slot.width} wide."
            )
    if why is not None and not any(slot.name in arms for slot in slots["action"]):
        raise EmbodimentInvalidError(
            "parts.action: no part drives an arm; training needs the arms' joint targets, or both arms' gripper poses."
        )
    return Plan(use_eef=why is None, state=tuple(slots["state"]), action=tuple(slots["action"]), recorded=recorded)


def _is_number(value: Any) -> bool:
    """Whether a JSON value is a finite number (a bool is not)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_int(value: Any) -> bool:
    """Whether a JSON value is a whole number (a bool is not, though Python says it is an int)."""
    return isinstance(value, int) and not isinstance(value, bool)
