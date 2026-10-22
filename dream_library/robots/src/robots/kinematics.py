"""Forward kinematics of one arm: its joint angles in, the end effector's position in metres out.

The rate clamp on the arms is a cap on the end effector's speed (`max_ee_speed_m_s`, 0.6 m/s on
the station), not on joint angular steps: a per-joint cap says nothing about how fast the hand
moves, since 0.02 rad of base yaw swings an outstretched arm a centimetre while 0.02 rad of wrist
roll moves the grasp point almost nothing, and such a cap throttles the policy on a third
of its ticks. Speed in metres needs the arm's geometry, so #ArmKinematics runs
MuJoCo's forward kinematics on the same model the IK uses: `i2rt` composes the
YAM arm's XML with the configured gripper's (`combine_arm_and_gripper_xml`) and the end effector
is that model's `grasp_site`.

Every other arm's end effector (a Makers Arm, a Metal arm, an SO-101, a robot of your own) is its tool centre point,
the frame dream computed that arm's datasets' tool poses in, so #DreamKinematics runs dream's own forward kinematics
(`dream_ik`, the model the dream client's IK solves on) rather than a second model of the arm. arm_layout() reads the
same spec for how the arm's joints and gripper go out on its wire. An arm of your own from its URDF is an arm file
(`dream.arm_file`), which dream_arm() registers with dream's IK; tcp_calibration() reads whether the operator marked
its tool point.

`mujoco`, `i2rt` and `dream_ik` are imported lazily inside the constructors, so the package, its tests and a
station without the `hardware` extra import this module; the loop takes any #Kinematics, and the
tests inject linear stand-ins.

@see ArmKinematics, DreamKinematics, Kinematics, arm_layout(), dream_arm(), tcp_calibration()
"""

import json
import pathlib
from typing import Any, Protocol

import dream
import numpy as np
import numpy.typing as npt
from dream import arm_file

from robots import adapter, config, errors, followers, yam

## The MuJoCo site at the gripper's grasp point, the end effector the IK targets too.
EE_SITE: str = "grasp_site"

## The arm file key that says whether the operator marked the tool point, written by `dreamcli setup`'s URDF step and
## `dreamcli tcp` (`robots.setup_wizard.SetupServer.add_arm`): `{"status", "at"}`, `status` one of #TCP_STATUSES and
## `at` when, UTC, ISO 8601, plus `digest` on a `calibrated` one (`dream.arm_file.tool_digest()` of the file as the
## operator confirmed it). dream warns once at `policy()` when the tool point is not one a person confirmed
## (`dream.arm_file.tool_point_problem()`).
TCP_CALIBRATION = arm_file.TCP_CALIBRATION

## `calibrated`: the operator put the marker where the fingertips close and pressed Set tool center point.
## `skipped`: they chose "Skip calibration (not advised)" and confirmed, so the tool point is wherever the marker was.
TCP_STATUSES = ("calibrated", "skipped")


class Kinematics(Protocol):
    """What the loop needs of an arm: one function from joint angles to the end effector's position.

    #ArmKinematics is the MuJoCo implementation; tests inject linear stand-ins.

    @see ee_position()
    """

    def ee_position(self, joints_rad: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The end effector's position for the arm's joint angles.

        @param joints_rad Joint angles, shape `[dof]` (`robots.followers.FollowerArm.dof`), radians.
        @return The position, shape `[3]`, metres, in the arm's base frame.
        """
        ...


class ArmKinematics:
    """Forward kinematics of one YAM arm on the i2rt MuJoCo model.

    One `MjModel` and one `MjData` per instance, allocated once: ee_position() writes the six
    joint angles into `qpos[:6]`, runs `mj_kinematics` (positions only, no dynamics; under a
    microsecond) and reads the grasp site. The model has eight coordinates, the six arm joints
    then the gripper's two coupled finger slides, which are left at zero: the grasp site sits on
    the gripper body, not on a finger, so opening the fingers does not move it (checked on the
    linear_4310 model: the site is unchanged with the slides at their limit). Not thread-safe,
    since every call shares the one `MjData`; the loop calls it from its tick alone, and the
    station reuses one instance per arm across runs.

    Attributes:
    - joint_range_rad: The six arm joints' limits in the model, `[6, 2]`, radians.

    @see from_config(), ee_position()
    """

    def __init__(self, model_xml_path: str) -> None:
        """Loads the model and finds the grasp site; mujoco is imported here, so only a station needs it.

        @param model_xml_path A MuJoCo XML of the arm with its gripper, as `combine_arm_and_gripper_xml` writes it.
        @throws ConfigError mujoco is not installed, or the model has no #EE_SITE or fewer than six coordinates.
        """
        try:
            import mujoco  # type: ignore[import-untyped]
        except ImportError as exc:
            raise errors.ConfigError(
                "the end-effector speed clamp needs mujoco, which is not installed; "
                "the dream library brings it: in your dream env (source ~/dream-env/bin/activate) run "
                "pip install --upgrade dreamlabs"
            ) from exc
        self._mujoco = mujoco
        self._model = mujoco.MjModel.from_xml_path(model_xml_path)
        self._data = mujoco.MjData(self._model)
        self._site_id = int(mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE))
        if self._site_id < 0 or self._model.nq < yam.JOINTS:
            raise errors.ConfigError(
                f"arm model '{model_xml_path}' needs a site '{EE_SITE}' and at least {yam.JOINTS} "
                f"coordinates; it has nq={self._model.nq}"
            )
        self.joint_range_rad: npt.NDArray[np.float64] = np.array(self._model.jnt_range[: yam.JOINTS], dtype=np.float64)

    @classmethod
    def from_config(cls, cfg: config.ArmConfig) -> "ArmKinematics":
        """Builds the YAM arm with the config's gripper through i2rt, the model the IK uses.

        @param cfg The arm config; only `gripper` is read (the CAN channel plays no part in the geometry).
        @return The kinematics.
        @throws ConfigError i2rt or mujoco is not installed.
        @throws ValueError `gripper` is neither an i2rt gripper type name nor one `config.GRIPPER_DRIVERS` maps to
            one (an Anvil gripper is built on the `linear_4310` model that drives it).
        """
        try:
            # i2rt is not on PyPI, so it is an optional extra and mypy cannot see it either.
            from i2rt.robots.utils import (  # type: ignore[import-not-found]
                ArmType,
                GripperType,
                combine_arm_and_gripper_xml,
            )
        except ImportError as exc:
            raise errors.ConfigError(
                "the end-effector speed clamp needs i2rt, which is not installed; "
                "the dream library brings it: in your dream env (source ~/dream-env/bin/activate) run "
                "pip install --upgrade dreamlabs"
            ) from exc
        return cls(combine_arm_and_gripper_xml(ArmType.YAM, GripperType(config.i2rt_gripper(cfg.gripper))))

    def ee_position(self, joints_rad: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The grasp site's position for six joint angles, with the finger slides at zero.

        Allocates nothing but the returned 3-vector: the model's data is reused, which is why a
        second call overwrites the site position the first one read.

        @param joints_rad Joint angles, shape `[6]`, radians.
        @return The position, shape `[3]`, metres, in the arm's base frame.
        """
        self._data.qpos[: yam.JOINTS] = joints_rad
        self._mujoco.mj_kinematics(self._model, self._data)
        return np.array(self._data.site_xpos[self._site_id], dtype=np.float64)


class DreamKinematics:
    """Forward kinematics of one arm on dream's own model of it (`dream_ik`: `maker_arm_ik.urdf`, `metal.urdf`, the
    SO-101's URDF of each side).

    The end effector is the tool centre point (`dream_ik.specs.SERVE.tool`), where the arm's datasets' tool poses
    are. both_arms() builds the model once for both arms; each instance reads its own side, since the sides may be
    different arms (the SO-101's two URDFs). A call costs about 0.13 ms on the Makers Arm (on a
    CPU); the model is built and its forward kinematics compiled once, which takes a few seconds. Not
    thread-safe; the loop calls it from its tick alone.

    @see both_arms(), ee_position()
    """

    def __init__(self, model: Any, side: str) -> None:
        """One side of a model both_arms() built.

        @param model dream's `ChunkIK` of the robot.
        @param side `"left"` or `"right"`.
        """
        self._model = model
        self._side = side
        self._dof = int(model.side_rest.shape[1])

    @classmethod
    def both_arms(cls, arm: str, hz: float) -> dict[str, "DreamKinematics"]:
        """Builds dream's model of the robot and compiles its forward kinematics on each side.

        @param arm dream's name for the arm, as the client's `ik` names it (`"makers"`, `"metal"`, `"so101_faas"`),
            or an arm file's path (dream_arm()).
        @param hz The loop's tick rate; the solver's row rate, which forward kinematics does not read.
        @return `{"left": ..., "right": ...}`.
        """
        try:
            from dream_ik.solver import ChunkIK
            from dream_ik.specs import SERVE
        except ImportError as exc:
            raise errors.ConfigError(
                f"the {arm} arm's end-effector speed clamp needs dream's IK, which the dream library ships for "
                f"Python 3.11 or newer (the platform's Setup API page installs it with Python 3.12): {exc}"
            ) from exc
        model = ChunkIK(dream_arm(arm), SERVE.tool, hz)
        arms = {side: cls(model, side) for side in ("left", "right")}
        for side, rest in zip(model.sides, model.side_rest, strict=True):
            model.fk(rest, side)
        return arms

    def ee_position(self, joints_rad: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The tool centre point's position for the arm's joints.

        @param joints_rad Joint angles, shape `[dof]`, radians, in the order and sign the arm reports them; dream's
            model reads the first of them, as many as it has (an arm may list more, held at 0).
        @return The position, shape `[3]`, metres, in the arm's base frame.
        """
        joints = np.asarray(joints_rad[: self._dof], dtype=np.float64)
        return np.asarray(self._model.fk(joints, self._side)[:3, 3], dtype=np.float64)


def arm_layout(arm: str) -> adapter.ArmLayout:
    """How the arm dream's IK names goes out on a fine-tune's wire, from dream's spec of it (`dream_ik.specs`).

    @param arm dream's name for the arm, as the client's `ik` names it, or an arm file's path (dream_arm()).
    @return The #robots.adapter.ArmLayout: the joints the loop drives (the spec's, less the ones dream holds at 0),
        the checkpoint's `<side>_arm` width, whether the dream client makes the gripper part, and the spec's joints.
    @throws ConfigError dream's IK is not installed, does not know the arm, the arm file is malformed, or the arm's
        sides differ in layout (dream's solver runs both sides as one array of joints) or have more joints than
        `robots.followers.MAX_JOINTS`.
    """
    try:
        from dream_ik.specs import EMBODIMENTS
    except ImportError as exc:
        raise errors.ConfigError(
            f"the {arm} arm needs dream's IK, which the dream library ships for Python 3.11 or newer "
            f"(the platform's Setup API page installs it with Python 3.12): {exc}"
        ) from exc
    arm = dream_arm(arm)
    if arm not in EMBODIMENTS:
        raise errors.ConfigError(f"dream's IK has no arm {arm!r}; it has {sorted(EMBODIMENTS)}")
    robot = EMBODIMENTS[arm]
    specs = [robot.side_arm(side) for side in robot.sides]
    layouts = {
        adapter.ArmLayout(
            joints=tuple(i for i, name in enumerate(spec.joints) if name not in spec.held_joints),
            width=len(spec.joints) + (1 if robot.jaw_in_arm else 0),
            client_gripper=spec.jaw is not None and not robot.jaw_in_arm,
            slots=len(spec.joints),
        )
        for spec in specs
    }
    if len(layouts) != 1 or any(len(spec.joints) > followers.MAX_JOINTS for spec in specs):
        raise errors.ConfigError(
            f"dream's {arm} arm is not one the loop drives: its sides must share one layout of at most "
            f"{followers.MAX_JOINTS} joints, got {sorted(layouts, key=repr)}"
        )
    return layouts.pop()


def dream_arm(ik: str) -> str:
    """dream's name for the arm the client's `ik` names: the arm itself, or the one its arm file is registered under.

    An arm file (a `.json`) is registered with dream's IK in this process, by `dream.arm_file.register`.

    @param ik One of the arms dream's IK ships, or the path of an arm file of your URDF.
    @return The key of `dream_ik.specs.EMBODIMENTS`.
    @throws ConfigError The arm file or its URDF is missing or malformed; dream's message says which.
    """
    if not ik.endswith(".json"):
        return ik
    try:
        return arm_file.register(ik)
    except dream.DreamConfigError as exc:
        raise errors.ConfigError(str(exc)) from exc


def tcp_calibration(ik: str) -> str | None:
    """What an arm file records of its tool point (`TCP_CALIBRATION`).

    @param ik The path of an arm file (`dream.arm_file`).
    @return One of #TCP_STATUSES, or None when the file records neither: it was written by hand.
    @throws ConfigError The file is missing or not JSON, or its record is malformed.
    """
    path = pathlib.Path(ik)
    if not path.is_file():
        raise errors.ConfigError(f"config key 'ik' names the arm file {ik!r}, which is not a file")
    try:
        arm = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise errors.ConfigError(f"arm file {ik} is not JSON: {exc}") from None
    if not isinstance(arm, dict):
        raise errors.ConfigError(f"arm file {ik} must be a JSON object, got {type(arm).__name__}")
    if TCP_CALIBRATION not in arm:
        return None
    record = arm[TCP_CALIBRATION]
    if (
        not isinstance(record, dict)
        or not {"status", "at"} <= set(record) <= {"status", "at", "digest"}
        or record["status"] not in TCP_STATUSES
    ):
        raise errors.ConfigError(
            f'arm file {ik}: {TCP_CALIBRATION} must be {{"status": one of {list(TCP_STATUSES)}, "at": a time}} and '
            f"maybe a digest, got {record!r}"
        )
    status: str = record["status"]
    return status
