"""Client-side IK for an EEF checkpoint: forward kinematics on the way out, the chunk solve on the way back; internal.

A model fine-tuned with `use_eef` reads each arm's tool pose (`state.<side>_eef_xyz`, `state.<side>_eef_rot6`) and
answers with tool poses (`action.<side>_eef_xyz`, `action.<side>_eef_rot6`, `action.<side>_grip`) beside a joint head
(`action.<side>_arm`): on a MakerMods Metal arm, a Makers Arm, one of the FaaS beta robots (#ARMS) or an arm of your
own from its URDF (`dream.arm_file`). The model runs no IK for the caller, so with the config key `ik` the client does
both ends, with dream's own IK (`dream_ik`, vendored into the library), the same code and arm models that computed the
training poses, so every pose is in the training frame.

The loop speaks the arm's URDF units: `state.<side>_arm` is the arm's joints in radians, and each chunk the client
returns is those joints, then the grip (0 closed, 1 open). A beta robot's checkpoint reads and writes its joints and
its gripper as the robot recorded them, and dream's arm spec says how (#_Wire): the client converts both ways from the
spec alone. Metal and Makers checkpoints read the URDF's joints already, and their caller sends the gripper parts
itself.

Per call: the tool pose of each arm, from forward kinematics of its measured joints `state.<side>_arm`, is added to the
observation, and a beta robot's joints and gripper drive go out in the checkpoint's units. A model trained on YAM arms
(the Dream-1 base models) is sent zeros in its joint slots on every one of these arms but the YAM: another arm's
joints are out of its range whatever their signs, and the model does better on them with zeros;
the hand pose and the gripper sent are the arm's own. Each request also names the robot `other_embodiment`
(`dream.client`, the `robot` config key's default here), on which the serve withholds the whole state, joints, hand
pose and grip alike: the model reads none of it, and the hand pose sent is only the anchor its answer is composed
on. On a YAM (`ik` arm `yam`, for a base model that reads each hand's pose
beside the joints) the model reads the arm's own joints and gripper, the client adds the poses and `<side>_grip`, and
head `joint` runs the model's joint answer, which is this arm's. With head `eef` the reply's tool poses are
filtered over the whole chunk (dream's
Savitzky-Golay on the hand path), then solved to joints (`dream_ik.decode.solve_chunk`, under the arm's speed caps and
inside its margined limits) from the joints the arm is commanded to at the row the loop installs the chunk at, and
written as `q_r_chunk` and `q_l_chunk`: the joints, then the grip. dream's own seam blend is off, because the client's
seam join (`dream.smoothing`) joins every chunk. The `ik` key's optional `solve` picks dream's solve: `traj` (the
default) solves all its rows at once with velocity and acceleration costs, continuing the previous plan's motion into
the takeover row, and `tick` solves the chunk one row at a time (#SOLVES). With head `joint`, which a model trained on
YAM arms may not run (its joint answer is another robot's joints, not run on this arm, for safety), a beta robot's
joint head is converted the same way, and a Metal or Makers model's joint chunks are kept. See #ClientIK.
"""

import logging
import math
import numbers
import os
import threading
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
from shared.datasets import embodiments

from dream import arm_file, datasets
from dream.errors import DreamActionError, DreamConfigError, DreamObservationError
from dream.models import Model
from dream.smoothing import SeamSmoother

## The arms the client solves for, by their dream IK names: the ones whose EEF training poses dream's IK computed. The
## YAM, whose tool point is the one the models' eef_tcp bake used (`dream_ik.specs.TCP_OFFSET_M`), the Metal and
## Makers arms, then the FaaS beta robots, each checked against its checkpoint's training labels in dream.
ARMS = (
    "yam",
    "metal",
    "makers",
    "ai_worker_faas",
    "trossen_wb_faas",
    "mantis_faas",
    "robo_t_faas",
    "openarm2_faas",
    "bracketbot_faas",
    "so101_faas",
    "galaxea_faas",
    "seeed_faas",
    "unitree_r1_faas",
)

## `eef` solves the reply's tool poses to joints; `joint` keeps the model's joint answer.
HEADS = ("eef", "joint")

## Per side of the arm: the chunk key its joints are returned under, and the observation key of its measured joints.
SIDES = {"right": ("q_r_chunk", "state.right_arm"), "left": ("q_l_chunk", "state.left_arm")}

## Per chunk key, the observation key of the measured joints the seam join anchors on (`SeamSmoother`).
POSE_KEYS = dict(SIDES.values())

MODEL_JOINT_ROWS = ("q_r", "q_l")

## The state parts of a side, beside its gripper drive, that an EEF checkpoint reads (`<side>_` then each).
EEF_STATE_PARTS = ("arm", "eef_xyz", "eef_rot6", "grip")

## A test switch, not part of the stable API. Names what a model trained on YAM arms reads as its state:
## `zero` is what it reads without the switch, `raw` the arm's own joints.
PROPRIO_TEST_ENV = "DREAM_PROPRIO_TEST"
PROPRIO_TEST_MODES = ("raw", "zero", "avg", "avg_hand", "avg_all")

## For the test switch: the base model's state at normalized 0, the middle of its state range.
PROPRIO_AVG = {
    "state.right_arm": (0.0826, 1.7384, 1.5751, -0.0177, -0.0166, 0.045),
    "state.right_eef_xyz": (0.3017, 0.1342, 0.1645),
    "state.right_eef_rot6": (-0.0019, 0.0574, 0.0005, 0.0272, 0.0017, 0.0035),
    "state.left_arm": (-0.3028, 1.7185, 1.5681, -0.0202, 0.0284, -0.0042),
    "state.left_eef_xyz": (0.2793, -0.0788, 0.1768),
    "state.left_eef_rot6": (0.0036, 0.0029, -0.0021, 0.0012, -0.0015, 0.0353),
}

_log = logging.getLogger("dream.client_ik")

## Rows the solver is compiled for when the client is built: the action horizon of Dream-1 checkpoints. A reply of
## another length is compiled for on its first call.
WARMUP_ROWS = 48

## The reach area's brush, metres: with `reach`, an arm's hand targets stay within this of one of its points, seen
## from above (#clamp_to_reach). The robots setup wizard paints with it.
REACH_RADIUS_M = 0.04
## How far past the reach area's edge a solved hand may go before the arm holds instead (#reach_fence), metres: room
## for the solve's own miss, less than a camera mount's width.
REACH_SLACK_M = 0.01


## dream's chunk solves the `ik` key's `solve` names (`dream_ik.specs.ServeSpec.solve`): `traj`, the default,
## solves every row of the chunk in one problem with velocity and acceleration costs and the previous plan's two rows
## before the takeover as history, which in offline replays of recorded teleop with model-like pose error cuts joint
## jerk about 75% and direction reversals about 80% after the client's seam join, hand error unchanged, for about
## 10 ms more per chunk. `tick` solves one row at a time from the row before.
SOLVES = ("tick", "traj")

## How far (m) an arm's centreline may leave its `region` before `ClientIK.fence` holds it (dream's
## `ServeSpec.region_slack`).
SERVE_REGION_SLACK_M = 0.01


class ClientIK:
    """Adds the tool poses to each observation and, for head `eef`, solves each reply's tool poses to joint chunks."""

    def __init__(self, config: Any, action_hz: float, embodiment: Any, model: Model) -> None:
        """Checks the config, then builds dream's solver for the arm at the loop's row rate and compiles it.

        @param config The value of the `ik` config key.
        @param action_hz The loop's row rate, rows per second: the solver's tick, which its speed caps are per.
        @param embodiment The `embodiment` config key, or None: a beta robot whose gripper drive has a key of its own
            reads that key's name from it (#_drive_key), and the body's cameras are its `video` (#camera_slots),
            which a model trained on YAM arms needs.
        @param model The platform's record of the session's checkpoint.
        """
        check_config(config)
        ## Whether the arm is a YAM: a model trained on YAM arms then reads this arm's own joints, and its joint
        ## answer is this arm's joints, so head `joint` runs it.
        self.on_yam: bool = config["arm"] == "yam"
        if config["head"] == "joint" and model.embodiment == embodiments.YAM_BODY and not self.on_yam:
            raise DreamConfigError(
                f"config key 'ik' names the head 'joint', but {model.id!r} was trained on YAM arms, so its joint "
                "answer is another robot's joints and is not run on this arm, for safety: use the head 'eef', or a "
                "fine-tune of this robot"
            )
        try:
            from dream_ik import decode
            from dream_ik.solver import ChunkIK
            from dream_ik.specs import EMBODIMENTS, SERVE, Box, Capsule, Region, ServeSpec
        except ImportError as error:
            raise DreamConfigError(
                f"config key 'ik' needs dream's IK, which the library ships for Python 3.11 or newer ({error}); in a "
                f"Python 3.11 or newer environment, {datasets._install_hint()}"
            ) from None
        self.head: str = config["head"]
        self._model_id = model.id
        ## Whether this session said its model has no EEF head and runs its joint answer (#solve, on a YAM).
        self._said_joint_answer = False
        # dream's name for the arm: one it ships, or the one an arm file is registered under.
        arm = config["arm"] if config["arm"] in ARMS else arm_file.register(config["arm"])
        spec = EMBODIMENTS[arm]
        ## Per side, the joint slots of a model trained on YAM arms, sent as zeros (#observe), which also makes
        ## `dream.client` name the robot `other_embodiment`; None for any other model, and on a YAM.
        trained_on_yam = model.embodiment == embodiments.YAM_BODY
        self.yam_joints = len(EMBODIMENTS["yam"].arm.joints) if trained_on_yam and not self.on_yam else None
        self.proprio_test = _proprio_test(model)
        self.cameras: dict[str, str] = {}
        roles = _video_roles(embodiment)
        if roles is None and self.yam_joints is not None:
            raise DreamConfigError(
                "config key 'ik' with a model trained on YAM arms needs the body's embodiment config, whose video "
                "names the cameras the client sends in the model's slots: pass `embodiment`"
            )
        if roles is not None and model.cameras is not None:
            self.cameras = camera_slots(roles, tuple(model.cameras), model.id)
        ## Per side, the map between the loop's units and the checkpoint's (#_Wire).
        self.wires = {
            side: _Wire(
                spec.side_arm(side), spec.jaw_in_arm, _drive_key(side, spec, embodiment, self.yam_joints is None)
            )
            for side in spec.sides
        }
        # Imported here, with the solver, so the first get_action() does not pay for the import (scipy's, mostly).
        self._decode: Any = decode
        obstacles = _keep_out(config["keep_out"], spec, (Box, Capsule)) if "keep_out" in config else None
        ## Per side with a `region`, the box its arm stays inside (#_region): clipped targets, a cost, and #fence.
        self.region = _region(config["region"], spec.sides, Region) if "region" in config else {}
        if self.region:
            kept = dict(spec.obstacles) if obstacles is None else obstacles
            obstacles = {s: kept[s] + ((self.region[s],) if s in self.region else ()) for s in spec.sides}
        ## Per side with a `reach`, its points `[n, 2]`, metres, in that arm's base frame (#clamp_to_reach).
        self.reach = _reach(config["reach"], spec.sides) if "reach" in config else {}
        self._solver: Any = ChunkIK(arm, SERVE.tool, action_hz, obstacles)
        # Both arms in one FK call when they share one arm spec: one compiled FK.
        self._one_arm = not isinstance(spec.arm, dict)
        ## Per chunk key, the joints of the arm it drives, as the loop reads and writes them.
        self.joints = {SIDES[side][0]: wire.joints for side, wire in self.wires.items()}
        # dream's filter, run here over the whole chunk before the rows from the clock row are cut out (#solve).
        self._smooth: Any = SERVE
        # The solve's own filter off (done above), and dream's seam blend off: the client's seam join
        # (dream.smoothing) joins every chunk, whichever the head and the solve. The clip and the speed cap still hold.
        self._serve: Any = ServeSpec(
            smooth_window=0, seam_blend_ticks=0, solve=config["solve"] if "solve" in config else "traj"
        )
        ## The same with `tick`: the solve while `traj` compiles.
        self._tick: Any = self._serve.model_copy(update={"solve": "tick"})
        ## Set once the `traj` solve is compiled (#_warm_traj); until then each chunk is solved with `tick`.
        self.traj_ready = threading.Event()
        if self._serve.solve == "traj" and self.head == "eef":
            # The trajectory solve compiles in 17-20 s on one slow CPU core (tick: 1-4 s), long enough to miss a
            # robot program's start-up deadlines. So the client starts on the tick solve's
            # compile and compiles traj on a thread; the first chunks after it is ready, a few seconds in on a normal
            # machine, and every chunk after, are traj. A
            # `joint` head never solves a chunk, so it compiles neither beyond the warm-up (the thread would only take
            # CPU from the control loop). The thread is not a daemon: a program that ends while it compiles waits for
            # it, where a daemon thread killed inside XLA's teardown aborted the process ("terminate called ... FATAL:
            # exception not rethrown").
            self._solver.warmup(WARMUP_ROWS, self._tick)
            threading.Thread(target=self._warm_traj, name="dream-client-ik-traj-warmup").start()
        else:
            self._solver.warmup(WARMUP_ROWS, self._tick if self.head == "joint" else self._serve)
            self.traj_ready.set()
        self._fk(self._solver.side_rest)
        for i, side in enumerate(self._solver.sides):  # the fence's forward kinematics of one arm's chunk (#fence)
            if side in self.reach:
                self._solver.fk(np.repeat(self._solver.side_rest[i][None], WARMUP_ROWS, axis=0), side)

    def _warm_traj(self) -> None:
        """Compile the `traj` solve (on #__init__'s thread), then set #traj_ready; a failure is logged and the client
        stays on `tick`."""
        try:
            self._solver.warmup(WARMUP_ROWS, self._serve)
        except Exception:  # noqa: BLE001 - the loop keeps running on tick; say why traj never came
            _log.exception("the traj IK solve failed to compile; solving with tick")
            return
        self.traj_ready.set()

    def _solve_spec(self) -> Any:
        """The ServeSpec this chunk is solved with: the configured one, or `tick` while `traj` compiles.

        @return A `dream_ik.specs.ServeSpec`.
        """
        return self._serve if self.traj_ready.is_set() else self._tick

    def sent_checks(self, checks: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """The strict checks of the body's observation, made to read what #observe sends.

        A body camera's `video.<role>` goes out under the model's role. For a model trained on YAM arms, each side's
        `state.<side>_arm` is YAM's zeros and its gripper `state.<side>_gripper`, whatever the body's own joints and
        gripper drive (a seven-joint arm, Trossen's drive in `<side>_arm`, a Unitree R1's `<side>_hand`). On a YAM
        (#on_yam) the body's block is the YAM's own, which names no pose: each side's tool pose and grip, which
        #observe adds, are read too.

        @param checks Observation key to its entry, from the body's `embodiment` block.
        @return A new dict.
        """
        out: dict[str, dict[str, Any]] = {}
        for key, entry in checks.items():
            group, _, role = key.partition(".")
            out[f"video.{self.cameras[role]}" if group == "video" and role in self.cameras else key] = entry
        if self.on_yam:
            for side in self.wires:
                out.setdefault(f"state.{side}_eef_xyz", {"shape": [3]})
                out.setdefault(f"state.{side}_eef_rot6", {"shape": [6]})
                out.setdefault(f"state.{side}_grip", {"shape": [1]})
        if self.yam_joints is None:
            return out
        for side, wire in self.wires.items():
            if wire.drive_key is not None:
                del out[wire.drive_key]
            out[SIDES[side][1]] = {"shape": [self.yam_joints]}
            out[f"state.{side}_gripper"] = {"shape": [1]}
        return out

    def reply_widths(self, widths: dict[str, int]) -> dict[str, int]:
        """The strict widths of the reply's `per_key`, made to read what the model answers.

        A model trained on YAM arms answers `action.<side>_arm` as YAM's joints, and not the body's own gripper drive.
        On a YAM (#on_yam) a model that reads each hand's pose answers the gripper as `action.<side>_grip`, not
        `action.<side>_gripper`, so neither is required: the serve packs whichever it answers into the joint chunks.
        With head `eef` the tool poses are not required either: #solve reads them when the model answers them and
        runs the joint answer of a model with no EEF head.

        @param widths `action.<part>` to its width, from the body's `embodiment` block.
        @return A new dict.
        """
        out = dict(widths)
        if self.on_yam:
            optional = ("gripper", "eef_xyz", "eef_rot6", "grip") if self.head == "eef" else ("gripper",)
            for side in self.wires:
                for part in optional:
                    out.pop(f"action.{side}_{part}", None)
        if self.yam_joints is None:
            return out
        for side, wire in self.wires.items():
            drive = "" if wire.drive_key is None else f"action.{wire.drive_key.partition('.')[2]}"
            if drive in out:
                del out[drive]
            if f"action.{side}_arm" in out:
                out[f"action.{side}_arm"] = self.yam_joints
        return out

    def observe(self, observation: dict[str, Any]) -> dict[str, Any]:
        """The observation with each arm's tool pose added, from forward kinematics of its measured joints.

        @param observation The caller's observation; it is not changed.
        @return A shallow copy with, per side, `state.<side>_eef_xyz` (`[3]`, metres) and `state.<side>_eef_rot6`
            (`[6]`, the first two columns of the rotation), in that arm's base frame. For a beta robot also
            `state.<side>_arm` in the checkpoint's units (`_Wire.arm`) and its gripper drive under its own key, from
            `state.<side>_grip`. For a model trained on YAM arms, `state.<side>_arm` as YAM's joints, all zero
            (#yam_joints), the gripper `state.<side>_gripper` as `state.<side>_grip`, and each camera under the
            model's role (#cameras). On a YAM arm (#on_yam) the joints and the gripper are the caller's own, and
            `state.<side>_grip` is added from `state.<side>_gripper` when the caller did not send it.
        @throws DreamObservationError The observation carries a tool pose or a gripper drive itself (the client
            computes them), or lacks a side's `state.<side>_arm` or holds it as anything other than the arm's joints
            (#_measured), or, for a beta robot or a model trained on YAM arms, lacks its `state.<side>_grip` or holds
            it as anything other than one number, or, when the client sends the cameras in the model's roles
            (#cameras), its cameras are not exactly the body's.
        """
        sides = self._solver.sides
        keys = (f"state.{side}_eef_{part}" for side in sides for part in ("xyz", "rot6"))
        sent = sorted(key for key in keys if key in observation)
        if sent:
            raise DreamObservationError(
                f"observation keys {sent} are computed by the client from state.<side>_arm (config key 'ik'); "
                "leave them out"
            )
        drive_keys = [wire.drive_key for wire in self.wires.values() if wire.drive_key is not None]
        drives = sorted(key for key in drive_keys if key in observation)
        if drives:
            raise DreamObservationError(
                f"observation keys {drives} are the gripper drives the client computes from state.<side>_grip "
                "(config key 'ik'); leave them out"
            )
        joints = {side: _measured(observation, SIDES[side][1], wire.joints) for side, wire in self.wires.items()}
        poses = self._fk(np.stack([self.wires[side].spec(joints[side]) for side in sides]))  # (arms, 4, 4)
        out = dict(observation)
        for i, side in enumerate(sides):
            out[f"state.{side}_eef_xyz"] = poses[i, :3, 3]
            out[f"state.{side}_eef_rot6"] = self._decode.mat_to_rot6d(poses[i, :3, :3])
            wire = self.wires[side]
            if self.yam_joints is not None:
                out[SIDES[side][1]] = np.zeros(self.yam_joints)
                out[f"state.{side}_gripper"] = _grip(observation, f"state.{side}_grip")
                continue
            if wire.own:
                if self.on_yam and f"state.{side}_grip" not in observation:
                    # A YAM's gripper is already the grip, 0 closed to 1 open: the EEF parts read it again.
                    out[f"state.{side}_grip"] = _grip(observation, f"state.{side}_gripper")
                continue
            grip = _grip(observation, f"state.{side}_grip")
            out[SIDES[side][1]] = wire.arm(joints[side], grip)
            if wire.drive_key is not None:
                out[wire.drive_key] = wire.drive(grip)
        if self.proprio_test is not None:
            _apply_proprio_test(out, self.proprio_test, joints)
        if self.cameras:
            _rename_cameras(out, self.cameras)
        return out

    def solve(
        self,
        action: dict[str, Any],
        observation: dict[str, Any],
        smoother: SeamSmoother,
        rtt_s: float,
        commanded: Mapping[str, npt.NDArray[np.float64]] | None,
    ) -> dict[str, Any]:
        """The reply with `q_r_chunk` and `q_l_chunk` solved from its tool poses (head `eef`), or from its joint head.

        Each arm's solve starts from the joints it is commanded to when the loop installs the chunk: the previous
        chunk's row at the clock row, as the seam join finds them (`SeamSmoother.takeover`), or the measured joints
        for an episode's first chunk and an arm off its plan. The tool poses are filtered over the whole chunk, then
        the rows from the clock row on are solved; the rows before it are in the past by the time the chunk lands
        and hold the start. An arm with a `reach` has its targets moved into its reach area first (#clamp_to_reach);
        the seam join then holds it where its hand would leave the area (#fence).

        @param action The model's action dict, as validated.
        @param observation The caller's observation the reply answers (#observe's input).
        @param smoother The client's seam smoother, holding the chunk returned last.
        @param rtt_s The request's round trip, seconds: with the row rate, the clock row.
        @param commanded Per chunk key, the joints the loop last commanded that arm (`PolicyClient.get_action`), or
            None.
        @return A shallow copy of `action` whose `q_r_chunk` and `q_l_chunk` are `[H, joints + 1]` arrays: the arm's
            joints in radians, then `action.<side>_grip`, without #MODEL_JOINT_ROWS; for head `joint`, a beta
            robot's joint head converted (#joint_head), and a Metal, Makers or YAM reply itself; on a YAM with head
            `eef`, a reply with no tool poses (a model with no EEF head) itself, said once per session.
        @throws DreamActionError The reply has no `per_key` dict, lacks a tool pose or grip of a side, holds one as
            anything other than rows of its width, or its parts disagree on the number of rows.
        """
        if self.head == "joint":
            if all(wire.own for wire in self.wires.values()):
                return action
            return self.joint_head(action)
        sides = self._solver.sides
        per_key = action.get("per_key")
        if self.on_yam and not (isinstance(per_key, dict) and f"action.{sides[0]}_eef_xyz" in per_key):
            # A YAM model with no EEF head answers only the YAM's joints, which run as they come, as on head
            # `joint`; a model with one runs its EEF head on the next session.
            if not self._said_joint_answer:
                self._said_joint_answer = True
                _log.warning(
                    "%s has no EEF head (it answers no tool poses): running its joint head on this YAM instead",
                    self._model_id,
                )
            return action
        parts = _parts(action, {f"{side}_{part}": width for side in sides for part, width in _EEF_PARTS})
        count = parts[f"{sides[0]}_grip"].shape[0]
        starts, befores, skips = [], [], []
        for side in sides:
            chunk_key, pose_key = SIDES[side]
            width = self.wires[side].joints
            measured = _measured(observation, pose_key, width)
            last = None if commanded is None else commanded.get(chunk_key)
            takeover = smoother.takeover_history(chunk_key, (count, width + 1), measured, rtt_s, last)
            skip, start, *before = takeover if takeover is not None else (0, measured, measured, measured)
            skips.append(skip)
            starts.append(start)
            befores.append(before)
        # dream's filter over the whole chunk first: it is centred, so filtering after the cut below would bend the
        # last rows kept toward the padding.
        poses = self._decode.smooth_poses(
            self._decode.head_poses(
                np.stack([parts[f"{side}_eef_xyz"] for side in sides]),
                np.stack([parts[f"{side}_eef_rot6"] for side in sides]),
            ),
            self._smooth.smooth_window,
            self._smooth.smooth_polyorder,
        )  # (arms, rows, 4, 4)
        # The rows from each clock row on, padded back to the full length: one compiled shape whatever the clock row.
        # The `tick` solve is causal, so a hold at the last pose never changes the rows kept. The `traj` solve sees the
        # whole chunk, so a hold brakes the rows before it (on a YAM, the last ~10 executed rows at ~70% of the
        # plan's speed); with `traj` the pad continues the chunk's last motion instead.
        serve = self._solve_spec()
        padded = np.stack(
            [_pad(self._decode, poses[i], skip, serve.solve == "traj") for i, skip in enumerate(skips)]
        )  # (arms, rows, 4, 4)
        xyz = padded[..., :3, 3].copy()  # (arms, rows, 3)
        for i, side in enumerate(sides):
            if side in self.reach:
                xyz[i, :, :2] = clamp_to_reach(xyz[i, :, :2], self.reach[side])
        rot6 = self._decode.mat_to_rot6d(padded[..., :3, :3])
        start_joints = np.stack([self.wires[side].spec(start) for side, start in zip(sides, starts, strict=True)])
        # The motion the arm is on as the chunk takes over, the two rows before, newest first, for the `traj` solve
        # (the `tick` solve reads none): `[arms, 2, joints]`.
        before_joints = np.stack(
            [np.stack([self.wires[side].spec(b) for b in before]) for side, before in zip(sides, befores, strict=True)]
        )
        # np.inf: no cap of the client's own, so the arm spec's per-joint speed caps are the cap.
        solved, _ = self._decode.solve_chunk(
            self._solver, xyz, rot6, start_joints, np.inf, serve, q_before=before_joints
        )
        out = _arm_reply(action)
        for i, (side, skip, start) in enumerate(zip(sides, skips, starts, strict=True)):
            rows = solved[i, : count - skip][:, self.wires[side].free]
            joints = np.concatenate([np.repeat(start[None], skip, axis=0), rows])
            out[SIDES[side][0]] = np.column_stack([joints, parts[f"{side}_grip"]])
        return out

    def fence(self, key: str, rows: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """One arm's chunk as the loop runs it, held where it would leave its reach area or its region.

        The hand leaving the reach area is #reach_fence; the arm's centreline leaving its `region` by more than
        #SERVE_REGION_SLACK_M, or entering a `keep_out` obstacle, is dream's `ChunkIK.fence_keep_out` (with a
        `region` set and dream's `ServeSpec.fences` on, which it is not by default).

        The seam join (`dream.smoothing`) calls it on the joined rows, since the join carries the old motion's speed
        on into the new chunk. Moving the targets in (#clamp_to_reach) is not enough alone: the solve keeps each
        target's rotation, which a pulled-in hand may not reach, and a path across a gap in the area steps through it.

        @param key The chunk key, `q_r_chunk` or `q_l_chunk`.
        @param rows `[rows, joints + 1]` the arm's joints in radians, then the grip.
        @return `[rows, joints + 1]`: `rows` itself when the arm has no `reach` or `region` or no row leaves, else
            a copy whose joints hold from that row on; the grip is kept.
        """
        side = next(side for side, (chunk_key, _) in SIDES.items() if chunk_key == key)
        out = rows
        if side in self.reach:
            hands = np.asarray(self._solver.fk(self.wires[side].spec(rows[:, :-1]), side))[:, :2, 3]  # (rows, 2)
            row = reach_fence(hands, self.reach[side])
            if row is not None:
                out = rows.copy()
                out[row + 1 :, :-1] = rows[row, :-1]
        if side in self.region and self._serve.fences:
            # dream's keep-out fence on the joined rows, with ServeSpec.fences only (off by default: the region is
            # then clipped targets and a cost in the solve): from the first row that leaves the region (or goes
            # deeper out than the chunk's first row, for an arm already outside) or puts a point of the arm in a
            # `keep_out` obstacle such as a camera pole, hold the row before
            joints = self.wires[side].spec(out[:, :-1])
            _, held = self._solver.fence_keep_out(joints[1:], joints[0], SERVE_REGION_SLACK_M, side)
            if held[0] >= 0:
                out = out.copy()
                out[held[0] + 1 :, :-1] = out[held[0], :-1]
        return out

    def joint_head(self, action: dict[str, Any]) -> dict[str, Any]:
        """A beta robot's reply with `q_r_chunk` and `q_l_chunk` built from its joint head, in the loop's units.

        @param action The model's action dict, as validated.
        @return A shallow copy of `action` whose `q_r_chunk` and `q_l_chunk` are `[H, joints + 1]` arrays: the joint
            head `action.<side>_arm` in radians (`_Wire.joints_of`), then `action.<side>_grip`, without
            #MODEL_JOINT_ROWS.
        @throws DreamActionError The reply has no `per_key` dict, lacks a side's joint head or grip, holds one as
            anything other than rows of its width, or its parts disagree on the number of rows.
        """
        widths = {f"{side}_arm": wire.width for side, wire in self.wires.items()}
        parts = _parts(action, {**widths, **{f"{side}_grip": 1 for side in self.wires}})
        out = _arm_reply(action)
        for side, wire in self.wires.items():
            out[SIDES[side][0]] = np.column_stack([wire.joints_of(parts[f"{side}_arm"]), parts[f"{side}_grip"]])
        return out

    def _fk(self, joints: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Each arm's tool pose.

        @param joints `[arms, dof]` each arm's joints in its spec's order, radians.
        @return `[arms, 4, 4]` each tool pose in its arm's base frame.
        """
        if self._one_arm:
            return np.asarray(self._solver.fk(joints))
        return np.stack([self._solver.fk(joints[i], side) for i, side in enumerate(self._solver.sides)])


def check_config(config: Any) -> None:
    """Checks the `ik` config key's shape, before anything is looked up or imported for it.

    @param config The value of the `ik` config key.
    """
    if not isinstance(config, dict) or not {"arm", "head"} <= set(config) <= {
        "arm",
        "head",
        "keep_out",
        "reach",
        "region",
        "solve",
    }:
        raise DreamConfigError(
            "config key 'ik' must be a dict with the keys 'arm', 'head' and optionally 'keep_out', 'reach', 'region' "
            f"and 'solve', got {config!r}"
        )
    if "solve" in config and config["solve"] not in SOLVES:
        raise DreamConfigError(f"config key 'ik' names the solve {config['solve']!r}; it takes {list(SOLVES)}")
    if not isinstance(config["arm"], str) or (config["arm"] not in ARMS and not config["arm"].endswith(".json")):
        raise DreamConfigError(
            f"config key 'ik' names the arm {config['arm']!r}; it takes {list(ARMS)} or the path of an arm file "
            "(.json, dream.arm_file)"
        )
    if config["head"] not in HEADS:
        raise DreamConfigError(f"config key 'ik' names the head {config['head']!r}; it takes {list(HEADS)}")


class _Wire:
    """One arm's map between the loop's units and the checkpoint's, from dream's arm spec (`dream_ik.specs`).

    The loop drives the spec's joints less the held ones (SO-101's dummy joint 6), in radians, and the gripper as the
    grip, 0 closed to 1 open. The checkpoint reads `<side>_arm` as every spec joint in its recorded units
    (`ArmSpec.joint_units`: radians = offset + scale * unit), then the gripper drive when the drive rides in the arm
    key (`Embodiment.jaw_in_arm`), and otherwise the drive under its own key; drive = closed + grip * (open - closed)
    (`ArmSpec.jaw`). A spec with none of these (Metal, Makers) is the loop's own units (#own).
    """

    def __init__(self, spec: Any, jaw_in_arm: bool, drive_key: str | None) -> None:
        """Reads the map off one side's arm spec.

        @param spec The side's `dream_ik.specs.ArmSpec`.
        @param jaw_in_arm Whether the checkpoint's `<side>_arm` ends with the gripper drive.
        @param drive_key The observation key of the gripper drive, or None when it has none (#_drive_key).
        """
        held = set(spec.held_joints)
        ## Indices, in the spec's joint order, of the joints the loop drives.
        self.free = [i for i, name in enumerate(spec.joints) if name not in held]
        ## Joints the loop drives.
        self.joints = len(self.free)
        self._dof = len(spec.joints)
        units = spec.joint_units
        self._scale = np.ones(self._dof) if units is None else np.asarray(units[0], dtype=np.float64)
        self._offset = np.zeros(self._dof) if units is None else np.asarray(units[1], dtype=np.float64)
        self._jaw: tuple[float, float] | None = spec.jaw
        self._jaw_in_arm = jaw_in_arm
        ## Width of the checkpoint's `<side>_arm`.
        self.width = self._dof + 1 if jaw_in_arm else self._dof
        ## The observation key the gripper drive goes out under, or None.
        self.drive_key = drive_key
        ## Whether the checkpoint reads and writes the loop's own units: no gripper drive, joint units or held joints.
        self.own = spec.jaw is None and units is None and not held and not jaw_in_arm

    def spec(self, joints: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The loop's joints as the spec's: the held joints put back, at 0.

        @param joints `[..., joints]` radians.
        @return `[..., dof]` radians, in the spec's order.
        """
        out = np.zeros((*joints.shape[:-1], self._dof))
        out[..., self.free] = joints
        return out

    def drive(self, grip: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The gripper drive the checkpoint reads.

        @param grip `[1]` 0 closed, 1 open.
        @return `[1]` in the drive's own units.
        """
        assert self._jaw is not None
        closed, opened = self._jaw
        return closed + grip * (opened - closed)

    def arm(self, joints: npt.NDArray[np.float64], grip: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The checkpoint's `<side>_arm`.

        @param joints `[joints]` radians.
        @param grip `[1]` 0 closed, 1 open.
        @return `[width]`: every spec joint in the recorded units, then the drive when it rides in the arm key.
        """
        units = (self.spec(joints) - self._offset) / self._scale
        return np.concatenate([units, self.drive(grip)]) if self._jaw_in_arm else units

    def joints_of(self, rows: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The loop's joints from the checkpoint's joint head.

        @param rows `[count, width]` the checkpoint's `<side>_arm`.
        @return `[count, joints]` radians.
        """
        return (self._offset + self._scale * rows[:, : self._dof])[:, self.free]


## Per side, the reply parts of the EEF head and their widths.
_EEF_PARTS = (("eef_xyz", 3), ("eef_rot6", 6), ("grip", 1))


def _proprio_test(model: Model) -> str | None:
    """The #PROPRIO_TEST_ENV test-switch mode, or None when it is unset.

    @param model The session's checkpoint.
    @return The mode.
    @throws DreamConfigError The mode is unknown, or the model was not trained on YAM arms.
    """
    mode = os.environ.get(PROPRIO_TEST_ENV)
    if not mode:
        return None
    if mode not in PROPRIO_TEST_MODES:
        raise DreamConfigError(f"{PROPRIO_TEST_ENV}={mode!r}: use one of {list(PROPRIO_TEST_MODES)}, or unset it")
    if model.embodiment != embodiments.YAM_BODY:
        raise DreamConfigError(f"{PROPRIO_TEST_ENV} is for a model trained on YAM arms; {model.id!r} is not")
    _log.warning("PROPRIO TEST %s=%s: the model is sent test state, not the arm's", PROPRIO_TEST_ENV, mode)
    return mode


def refuse_proprio_test_without_ik(ik: Any) -> None:
    """Refuses #PROPRIO_TEST_ENV where the client builds no state (a YAM station's serve does).

    @param ik The `ik` config key, or None.
    @throws DreamConfigError The mode is set but `ik` is not.
    """
    if ik is None and os.environ.get(PROPRIO_TEST_ENV):
        raise DreamConfigError(
            f"{PROPRIO_TEST_ENV} works only on arms the client solves IK for (Metal, Makers): on a YAM station the "
            "serve builds the model's state from the joints; unset it"
        )


def _apply_proprio_test(out: dict[str, Any], mode: str, joints: dict[str, npt.NDArray[np.float64]]) -> None:
    """Replaces the joints or hand pose the model reads, per #PROPRIO_TEST_MODES (the test switch).

    @param out The observation going out, changed in place.
    @param mode The mode.
    @param joints Per side, the arm's joints as it reads them.
    """
    for side, measured in joints.items():
        arm = SIDES[side][1]
        if mode == "raw":
            out[arm] = measured
        if mode == "zero":
            out[arm] = np.zeros_like(out[arm])
        if mode in ("avg", "avg_all"):
            out[arm] = np.asarray(PROPRIO_AVG[arm])
        if mode in ("avg_hand", "avg_all"):
            for part in ("eef_xyz", "eef_rot6"):
                out[f"state.{side}_{part}"] = np.asarray(PROPRIO_AVG[f"state.{side}_{part}"])


def camera_slots(roles: tuple[str, ...], model_roles: tuple[str, ...], checkpoint: str) -> dict[str, str]:
    """Which of the model's camera roles each of the body's cameras fills, by slot.

    @param roles The body's camera roles.
    @param model_roles The camera roles the model reads.
    @param checkpoint The model, for the message.
    @return Per body role, the model's role.
    """
    wrists = embodiments.WRIST_ROLES
    scene = [role for role in roles if role not in wrists]
    model_scene = [role for role in model_roles if role not in wrists]
    if len(scene) == 1 and len(model_scene) == 1:
        slots = {scene[0]: model_scene[0], **{role: role for role in roles if role in wrists}}
    else:
        slots = {role: role for role in roles}
    missing = sorted(set(model_roles) - set(slots.values()))
    extra = sorted(role for role, model_role in slots.items() if model_role not in model_roles)
    if missing or extra:
        raise DreamConfigError(
            f"the body's cameras {list(roles)} cannot be sent the way {checkpoint!r} reads them, which is "
            f"{list(model_roles)} (one scene camera, then {' and '.join(wrists)}, matched by name): missing "
            f"{missing}, extra {extra}. Name the body's cameras after those roles"
        )
    return slots


def _video_roles(embodiment: Any) -> tuple[str, ...] | None:
    video = embodiment.get("video") if isinstance(embodiment, dict) else None
    return tuple(video) if isinstance(video, dict) else None


def _rename_cameras(observation: dict[str, Any], cameras: dict[str, str]) -> None:
    sent = {key.partition(".")[2] for key in observation if key.startswith("video.")}
    missing, extra = sorted(set(cameras) - sent), sorted(sent - set(cameras))
    if missing or extra:
        raise DreamObservationError(
            f"the observation's cameras must be the body's {sorted(cameras)}: missing {missing}, extra {extra}"
        )
    windows = {role: observation.pop(f"video.{role}") for role in cameras}
    for role, window in windows.items():
        observation[f"video.{cameras[role]}"] = window


def _keep_out(keep_out: Any, spec: Any, kinds: tuple[Any, ...]) -> dict[str, tuple[Any, ...]]:
    if not isinstance(keep_out, dict) or not keep_out or not set(keep_out) <= set(spec.sides):
        raise DreamConfigError(
            f"config key 'ik' has 'keep_out' {keep_out!r}: it must be a dict with some of the keys "
            f"{sorted(spec.sides)}, each a list of obstacles (an empty list for none)"
        )
    by_kind = {cls.model_fields["kind"].default: cls for cls in kinds}
    out = dict(spec.obstacles)
    for side in keep_out:
        if not isinstance(keep_out[side], list):
            raise DreamConfigError(f"config key 'ik' has 'keep_out.{side}' {keep_out[side]!r}: it must be a list")
        built = []
        for entry in keep_out[side]:
            kind = entry.get("kind") if isinstance(entry, dict) else None
            if kind not in by_kind:
                raise DreamConfigError(
                    f"config key 'ik' has an obstacle {entry!r} in 'keep_out.{side}': its 'kind' must be one of "
                    f"{sorted(by_kind)}"
                )
            try:
                built.append(by_kind[kind].model_validate(entry))
            except ValueError as error:
                raise DreamConfigError(f"config key 'ik' has a bad obstacle in 'keep_out.{side}': {error}") from None
        out[side] = tuple(built)
    return out


def _region(region: Any, sides: tuple[str, ...], kind: Any) -> dict[str, Any]:
    """The `region` key: per side, `{"lo": [x, y, z], "hi": [x, y, z]}`, metres in that arm's base frame, any value
    null (that face open), either key left out (all open); built as dream's `Region` (`kind`)."""
    if not isinstance(region, dict) or not region or not set(region) <= set(sides):
        raise DreamConfigError(
            f"config key 'ik' has 'region' {region!r}: it must be a dict with some of the keys {sorted(sides)}, each "
            "{'lo': [x, y, z], 'hi': [x, y, z]} in metres, null for an open face"
        )
    out = {}
    for side, box in region.items():
        faces = isinstance(box, dict) and set(box) <= {"lo", "hi"}
        if not faces or not all(
            isinstance(v, list) and len(v) == 3 and all(x is None or _finite(x) for x in v) for v in box.values()
        ):
            raise DreamConfigError(
                f"config key 'ik' has 'region.{side}' {box!r}: it must be {{'lo': [x, y, z], 'hi': [x, y, z]}}, "
                "each value a finite number in metres or null"
            )
        try:
            out[side] = kind(**{face: tuple(v) for face, v in box.items()})
        except (ValueError, AssertionError) as error:
            raise DreamConfigError(f"config key 'ik' has 'region.{side}' {box!r}: {error}") from None
    return out


def _reach(reach: Any, sides: tuple[str, ...]) -> dict[str, npt.NDArray[np.float64]]:
    if not isinstance(reach, dict) or not reach or not set(reach) <= set(sides):
        raise DreamConfigError(
            f"config key 'ik' has 'reach' {reach!r}: it must be a dict with some of the keys {sorted(sides)}, each a "
            "list of [x, y] points in metres"
        )
    for side, points in reach.items():
        if not isinstance(points, list) or not points:
            raise DreamConfigError(
                f"config key 'ik' has 'reach.{side}' {points!r}: it must be a list of [x, y] points, at least one"
            )
        for point in points:
            pair = isinstance(point, (list, tuple)) and len(point) == 2
            if not pair or not all(_finite(v) for v in point):
                raise DreamConfigError(
                    f"config key 'ik' has a bad point {point!r} in 'reach.{side}': it must be [x, y], two finite "
                    "numbers in metres"
                )
    return {side: np.asarray(points, dtype=np.float64) for side, points in reach.items()}


def _finite(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def clamp_to_reach(xy: npt.NDArray[np.float64], points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Hand targets kept inside a reach area seen from above: the disks of #REACH_RADIUS_M around its points.

    A row farther than the radius from every point moves straight toward its nearest point until it is exactly the
    radius away, the nearest place in the area; a row inside is returned as it came.

    @param xy `[rows, 2]` the targets' x and y, metres.
    @param points `[n, 2]` the area's points, metres, at least one.
    @return `[rows, 2]` a new array.
    """
    # |xy - p|^2 = |xy|^2 - 2 (xy . p - |p|^2 / 2): the nearest point has the largest xy . p - |p|^2 / 2.
    score = xy @ points.T  # (rows, n)
    score -= np.einsum("nk,nk->n", points, points) / 2.0
    nearest = np.argmax(score, axis=1)  # (rows,)
    away = xy - points[nearest]  # (rows, 2)
    distance = np.hypot(away[:, 0], away[:, 1])  # (rows,)
    out = xy.copy()
    outside = distance > REACH_RADIUS_M
    out[outside] = points[nearest[outside]] + away[outside] * (REACH_RADIUS_M / distance[outside])[:, None]
    return out


def reach_fence(hands: npt.NDArray[np.float64], points: npt.NDArray[np.float64]) -> int | None:
    """The row an arm holds at so that its hand stays in its reach area, or None when no row leaves it.

    A row leaves when its hand is farther than #REACH_RADIUS_M plus #REACH_SLACK_M from every point and more than
    1 mm farther out than the closest the rows before it came, so a hand that starts outside the area may still move
    back in, and wobble a little on its way.

    @param hands `[rows, 2]` the hand's x and y at each row, metres; the first row is where it starts.
    @param points `[n, 2]` the area's points, metres, at least one.
    @return The row before the first row that leaves: the arm holds it from there on.
    """
    distance = _nearest_distance(hands, points)  # (rows,)
    closest = np.minimum.accumulate(distance)[:-1]  # (rows - 1,) the closest before each row from the second
    leaves = distance[1:] > np.maximum(REACH_RADIUS_M + REACH_SLACK_M, closest + 1e-3)
    return int(np.argmax(leaves)) if leaves.any() else None


def _nearest_distance(xy: npt.NDArray[np.float64], points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """`[rows]` each row's distance to its nearest point."""
    away = xy[:, None, :] - points[None]  # (rows, n, 2)
    distance: npt.NDArray[np.float64] = np.sqrt(np.einsum("rnk,rnk->rn", away, away).min(axis=1))
    return distance


def _drive_key(side: str, spec: Any, embodiment: Any, read: bool = True) -> str | None:
    """The observation key a side's gripper drive goes out under: the one part of that side the body's state names
    beside #EEF_STATE_PARTS (`<side>_gripper` on most bodies, `<side>_hand` on a Unitree R1).

    A model trained on YAM arms never reads the drive (`ClientIK.observe` sends it the grip as `state.<side>_gripper`),
    so for it (`read` False) a body that names no such part is no error: an SO-101 station whose body the robots setup
    wizard registered before it named the gripper ran no base model at all.

    @param side The side.
    @param spec The arm's `dream_ik.specs.Embodiment`.
    @param embodiment The `embodiment` config key (an embodiment config or a shape spec), or None.
    @param read Whether the session's model reads the drive: False for a model trained on YAM arms.
    @return `state.<part>`, or None when the arm sends no drive (no `ArmSpec.jaw`) or sends it in `<side>_arm`, or
        the model does not read it and the body names none.
    @throws DreamConfigError The drive needs a key and `embodiment` is not a dict with a `state` dict, or names not
        exactly one such part for the side (more than one, whatever the model).
    """
    if spec.side_arm(side).jaw is None or spec.jaw_in_arm:
        return None
    state = embodiment.get("state") if isinstance(embodiment, dict) else None
    if not isinstance(state, dict):
        raise DreamConfigError(
            "config key 'ik' needs the body's embodiment config, whose state names each arm's gripper part: pass "
            "`embodiment` (DreamClient.policy() loads it for your own fine-tunes with `strict` on)"
        )
    known = {f"{side}_{part}" for part in EEF_STATE_PARTS}
    others = sorted(part for part in state if part.startswith(f"{side}_") and part not in known)
    if not others and not read:
        return None
    if len(others) != 1:
        raise DreamConfigError(
            f"config key 'embodiment' must name one gripper part of the {side} arm beside {sorted(known)}, "
            f"got {others}"
            + (
                "; a fine-tune reads the gripper, so train it on an embodiment that names one (dreamcli setup's New)"
                if not others
                else ""
            )
        )
    return f"state.{others[0]}"


def _measured(observation: dict[str, Any], key: str, joints: int) -> npt.NDArray[np.float64]:
    """One arm's measured joints from the observation.

    @param observation The observation.
    @param key The arm's joint key, `state.<side>_arm`.
    @param joints The arm's joints.
    @return `[joints]` radians.
    @throws DreamObservationError The key is absent or its value is not `joints` numbers.
    """
    if key not in observation:
        raise DreamObservationError(f"observation has no '{key}', the arm's joints the tool pose is computed from")
    try:
        values = np.asarray(observation[key], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise DreamObservationError(f"observation key '{key}' is not numbers: {error}") from None
    if values.shape != (joints,):
        raise DreamObservationError(f"observation key '{key}' has shape {values.shape}; the arm has {joints} joints")
    return values


def _grip(observation: dict[str, Any], key: str) -> npt.NDArray[np.float64]:
    """One arm's gripper from the observation.

    @param observation The observation.
    @param key The arm's grip key, `state.<side>_grip`.
    @return `[1]`, 0 closed, 1 open.
    @throws DreamObservationError The key is absent or its value is not one number.
    """
    if key not in observation:
        raise DreamObservationError(
            f"observation has no '{key}', the gripper (0 closed, 1 open) the drive is made from"
        )
    try:
        grip = np.asarray(observation[key], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise DreamObservationError(f"observation key '{key}' is not numbers: {error}") from None
    if grip.shape != (1,):
        raise DreamObservationError(f"observation key '{key}' has shape {grip.shape}; it needs (1,)")
    return grip


def _arm_reply(action: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in action.items() if key not in MODEL_JOINT_ROWS}


def _parts(action: dict[str, Any], widths: dict[str, int]) -> dict[str, npt.NDArray[np.float64]]:
    """Parts of the reply's `per_key`, each as rows of its width, all with the same number of rows.

    @param action The model's action dict.
    @param widths Per part (`<side>_<part>`), its width.
    @return Per part, `[rows, width]`.
    @throws DreamActionError The reply has no `per_key` dict, a part is absent, not rows of its width, or the parts
        disagree on the number of rows.
    """
    per_key = action.get("per_key")
    if not isinstance(per_key, dict):
        raise DreamActionError(
            "action has no 'per_key' dict, where the model returns the parts the client solves from"
        )
    parts = {part: _rows(per_key, f"action.{part}", width) for part, width in widths.items()}
    counts = {f"action.{part}": rows.shape[0] for part, rows in parts.items()}
    if len(set(counts.values())) != 1:
        raise DreamActionError(f"action 'per_key' parts disagree on the number of rows: {counts}")
    return parts


def _rows(per_key: dict[str, Any], key: str, width: int) -> npt.NDArray[np.float64]:
    """One part of the reply as `[rows, width]`; a width-1 part may come as `[rows]`.

    @param per_key The reply's `per_key`.
    @param key The part, `action.<side>_<part>`.
    @param width The part's width.
    @return `[rows, width]`.
    @throws DreamActionError The part is absent, not numbers, or not rows of `width`.
    """
    if key not in per_key:
        raise DreamActionError(f"action 'per_key' lacks '{key}', which config key 'ik' solves from")
    try:
        rows = np.asarray(per_key[key], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise DreamActionError(f"action 'per_key' entry '{key}' is not rows of numbers: {error}") from None
    if width == 1 and rows.ndim == 1:
        rows = rows[:, None]
    if rows.ndim != 2 or rows.shape[1] != width or rows.shape[0] == 0:
        raise DreamActionError(f"action 'per_key' entry '{key}' has shape {rows.shape}; it needs (H, {width})")
    return rows


def _pad(decode: Any, poses: npt.NDArray[np.float64], skip: int, continue_motion: bool) -> npt.NDArray[np.float64]:
    """The poses from `skip` on, padded back to the same length: a hold at the last pose, or the last rows' motion.

    @param decode dream's `dream_ik.decode` module (`extend_poses`).
    @param poses `[count, 4, 4]`.
    @param skip Rows to drop from the front, below `count`.
    @param continue_motion Pad with `decode.extend_poses` (the position in a line, the rotation about a fixed axis)
        instead of a hold.
    @return `[count, 4, 4]`.
    """
    kept = poses[skip:]
    if continue_motion:
        return np.asarray(decode.extend_poses(kept, skip))
    return np.concatenate([kept, np.repeat(poses[-1:], skip, axis=0)])
