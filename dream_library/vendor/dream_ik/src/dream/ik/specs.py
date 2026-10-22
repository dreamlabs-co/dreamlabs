"""Embodiment specs for the IK: every robot is data here, the solver (``dream.ik.solver``) is generic.

Adding a robot = one ``ArmSpec`` (its URDF under ``src/dream/ik/assets``, joint names, tool
frames, rest pose, speed caps, joint signs), or one per side when the sides differ, and one ``Embodiment``
entry (how many arms, what is in their way). No solver or serve code changes.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, model_validator

# The YAM tool point: this far along link_6 z, the value the eef_tcp bake used.
# Repeated here, not imported, so the package installs without the model code (a test pins the two equal).
TCP_OFFSET_M = 0.1347

ASSETS = (
    Path(__file__).resolve().parent / "assets"
)  # the URDFs, next to this module in the repo and in the dream-ik wheel


class ToolPose(BaseModel):
    """The controlled point on the flange: origin ``offset`` (m) in the flange frame and ``rotation``
    (3x3 rows) of its axes in the flange frame, so tool pose = flange pose @ [rotation | offset]."""

    model_config = {"frozen": True}

    offset: tuple[float, float, float]
    rotation: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


class Capsule(BaseModel):
    """Vertical capsule obstacle (a pole, a camera stick), in the arm's base frame."""

    model_config = {"frozen": True, "extra": "forbid"}  # a station's keep_out is hand-written: a typo must fail

    kind: Literal["capsule"] = "capsule"
    center: tuple[float, float, float]
    radius: float
    height: float


class Box(BaseModel):
    """Axis-aligned box obstacle (a mount, a table), in the arm's base frame. ``size`` is full extent."""

    model_config = {"frozen": True, "extra": "forbid"}

    kind: Literal["box"] = "box"
    center: tuple[float, float, float]
    size: tuple[float, float, float]


class Region(BaseModel):
    """Where an arm may be, in its own base frame: an axis-aligned box, each face a bound in metres or None (open).
    It bounds the arm's centreline: the centers of its collision spheres along every link, the tool point (the grasp
    point between the fingertips) among them, so a face at a table lets the hand reach the table. The solve's
    collision cost holds the centreline ``collision_margin`` in from each face, the hand targets are clipped that far
    in before the solve, and ``decode`` holds the arm at the last row before one that leaves the region by more
    than ``ServeSpec.region_slack`` (a guarantee, where the cost is not). A table is a ``lo`` z, the space behind
    the base a ``lo`` x, the other arm's half a ``hi`` or ``lo`` y (which also keeps two arms apart, hand-overs
    included)."""

    model_config = {"frozen": True, "extra": "forbid"}

    kind: Literal["region"] = "region"
    lo: tuple[float | None, float | None, float | None] = (None, None, None)
    hi: tuple[float | None, float | None, float | None] = (None, None, None)

    @model_validator(mode="after")
    def _a_box(self):
        for a, b in zip(self.lo, self.hi, strict=True):
            assert a is None or b is None or a < b, (self.lo, self.hi)
        return self


Obstacle = Capsule | Box | Region


class IKWeights(BaseModel):
    """Cost weights. pos/ori as our teleop solves. ``limit`` and ``collision``
    are stiff pushes back inside the margined joint limits and out to ``collision_margin`` (m) from every
    obstacle; ``last_pose`` is the pull toward the previous tick's joints."""

    model_config = {"frozen": True}

    pos: float = 50.0
    ori: float = 10.0
    limit: float = 1000.0
    # 1.0: 3.0 was tried (10/01 sweep on 1,072 real chunks at the station's operating point: 5% fewer ticks at the
    # cap, 5% less hand miss p95, 6% less joint acceleration) but lags fast hand paths by 2 to 4 mm, past the
    # round-trip tolerances of the IK tests; the gain is not worth the tracking.
    last_pose: float = 1.0
    collision: float = 1000.0
    collision_margin: float = 0.02


class ArmSpec(BaseModel):
    """One arm type. Joint vectors (rest pose, weights, speeds) are in ``joints`` order."""

    model_config = {"frozen": True}

    urdf: Path
    joints: tuple[str, ...]  # actuated joints, command order
    fixed_joints: tuple[str, ...] = ()  # other movable URDF joints (the jaws), held at 0 for the IK
    # joints the IK keeps at 0 (margined limits 0..0): a five-joint arm's dummy joint6 on a six-joint wire
    held_joints: tuple[str, ...] = ()
    flange: str  # link the tool frames hang off
    tools: dict[str, ToolPose]
    rest_pose: tuple[float, ...]
    rest_weight: tuple[float, ...]
    max_speed: tuple[float, ...]  # rad/s per joint, direction-preserving cap on the command
    # per joint, +1 or -1: this arm's joints into YAM's sign (its own inverse). None: no YAM counterpart, so no
    # checkpoint's joint keys can be mapped onto this arm (dream.ik.serve refuses it, but for the zero-shot sim
    # robots, whose joint slots read the nearest YAM posture; a station decodes it itself)
    joint_sign: tuple[float, ...] | None
    limit_margin: float = float(np.radians(2.0))  # rad, inside the URDF limits
    link_radius: float  # m, collision spheres laid along the kinematic chain
    weights: IKWeights = IKWeights()
    # A FaaS EEF bake's gripper: (closed, open) values of the robot's gripper drive, the checkpoint's
    # {side}_grip = (drive - closed) / (open - closed), unclipped; open may be below closed. None: YAM's layout.
    jaw: tuple[float, float] | None
    # A FaaS EEF bake that kept the robot's own joint units (SO-101: LeRobot -100..100 of each arm's calibration):
    # (scale, offset) per joint, recorded joint c = offset + scale * c rad. None: radians.
    joint_units: tuple[tuple[float, ...], tuple[float, ...]] | None

    @model_validator(mode="after")
    def _held_rest_at_zero(self):
        rest = dict(zip(self.joints, self.rest_pose))
        assert all(rest[j] == 0.0 for j in self.held_joints), f"held joints {self.held_joints} must rest at 0"
        return self

    @model_validator(mode="after")
    def _units_per_joint(self):
        n = len(self.joints)
        assert self.joint_units is None or [len(u) for u in self.joint_units] == [n, n], (self.joints, self.joint_units)
        return self


class Embodiment(BaseModel):
    """A robot = ``arm`` instanced once per side (command order), or a dict side -> ArmSpec when the sides'
    chains differ (mirrored arms, per-side URDFs), plus fixed obstacles per side in that arm's base frame (a station
    whose mounts differ passes its own to ``ChunkIK``)."""

    model_config = {"frozen": True}

    arm: ArmSpec | dict[str, ArmSpec]
    sides: tuple[str, ...]
    obstacles: dict[str, tuple[Obstacle, ...]]
    # Trossen's FaaS layout: the checkpoint's {side}_arm is the arm joints then the jaw drive (``ArmSpec.jaw``).
    jaw_in_arm: bool = False
    # Each side's base origin (m) in one shared frame, the bases sharing one orientation: what lets the trajectory
    # solve (``ServeSpec.solve`` "traj") keep the two arms apart. None = not measured: the arms are solved apart.
    side_base: dict[str, tuple[float, float, float]] | None = None

    @model_validator(mode="after")
    def _an_arm_per_side(self):
        assert not isinstance(self.arm, dict) or set(self.arm) == set(self.sides), (sorted(self.arm), self.sides)
        assert not self.jaw_in_arm or all(self.side_arm(s).jaw is not None for s in self.sides), "needs ArmSpec.jaw"
        return self

    def side_arm(self, side: str) -> ArmSpec:
        return self.arm[side] if isinstance(self.arm, dict) else self.arm


_IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

# YAM: dream's own yam.urdf and TCP (action_space.yam_fk: link_6 axes, TCP_OFFSET_M along its z).
YAM_ARM = ArmSpec(
    urdf=ASSETS / "yam.urdf",
    joints=tuple(f"joint{i}" for i in range(1, 7)),
    flange="link_6",
    tools={"tcp": ToolPose(offset=(0.0, 0.0, TCP_OFFSET_M), rotation=_IDENTITY)},
    rest_pose=(0.0, 0.5, 0.5, -0.35, 0.0, 0.0),  # our teleop's rest pose
    rest_weight=(0.15, 0.3, 0.3, 0.3, 0.3, 0.3),  # our teleop's rest weights
    max_speed=(3.0,) * 6,  # our teleop's joint speed cap
    joint_sign=(1.0,) * 6,
    link_radius=0.04,
    jaw=None,
    joint_units=None,
)

# Metal: the vendor chain (assets/metal.urdf). TCP 0.135 m along link6 x with its axes relabelled
# into YAM's link_6 convention (z along the fingers): the frame the model's eef keys live in, as in
# our Metal policy and the simulation evaluation's EEF space.
METAL_ARM = ArmSpec(
    urdf=ASSETS / "metal.urdf",
    joints=tuple(f"joint{i}" for i in range(1, 7)),
    flange="link6",
    tools={"tcp": ToolPose(offset=(0.135, 0.0, 0.0), rotation=((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))},
    rest_pose=(0.0, -1.2, 1.5, 0.0, 0.0, 0.0),  # our Metal teleop's rest pose
    rest_weight=(0.15, 0.3, 0.3, 0.3, 0.3, 0.3),  # our Metal teleop's rest weights
    max_speed=(4.5, 4.5, 4.5, 12.0, 12.0, 12.0),  # our Metal teleop's speed cap
    # The shoulder turns the other way from YAM's (Metal 0 to -3.14 rad, YAM 0 to +3.65), as in
    # our Metal policy and the simulation evaluation's EEF space.
    joint_sign=(1.0, -1.0, 1.0, 1.0, 1.0, 1.0),
    link_radius=0.04,
    jaw=None,
    joint_units=None,
)

# Seeed reBot B601-DM: assets/b601_dm.urdf takes the DM motor angles the robot reports (the vendor DM links,
# the station's soft limits, the 45 mm table drop; its header has the evidence). TCP = the station's grasp
# point, 20 mm behind the fingertips (the vendor DM end_link, 0.15539 m out along link6 z), axes turned -90 deg
# about link6 z into YAM's link_6 convention.
B601_ARM = ArmSpec(
    urdf=ASSETS / "b601_dm.urdf",
    joints=tuple(f"joint{i}" for i in range(1, 7)),
    flange="link6",
    tools={
        "tcp": ToolPose(
            offset=(0.0, 0.0, 0.15539 - 0.020), rotation=((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        )
    },
    rest_pose=tuple(np.radians([0.0, -35.0, -60.0, 40.0, 0.0, 0.0])),  # our B601 teleop's rest pose
    rest_weight=(0.15, 0.3, 0.3, 0.3, 0.3, 0.3),  # our B601 teleop's rest weights
    max_speed=(1.0, 1.0, 1.0, 1.5, 1.5, 1.5),  # our B601 teleop's speed cap
    # Shoulder, elbow and wrist pitch turn the other way from YAM's (as our B601
    # teleop signs them): through it a YAM holds the gripper as the B601 does, 0.0007 deg p99 on the take.
    joint_sign=(1.0, -1.0, -1.0, -1.0, 1.0, 1.0),
    link_radius=0.04,
    jaw=None,
    joint_units=None,
)

# Maker Arm v1 (a Maker Arm station): assets/maker_arm_ik.urdf is the vendor chain with the served base frame, the
# wrist-yaw
# sign and the follower's soft limits baked in (its header), so its joints are what the station reports. TCP: the
# URDF's grasp_center, 0.1696 m along link_007 -z, axes relabelled into YAM's (z = approach = -z_007, x = the jaw
# travel +y_007).
MAKERS_ARM = ArmSpec(
    urdf=ASSETS / "maker_arm_ik.urdf",
    joints=tuple(f"link_00{i}_joint" for i in range(2, 8)),
    flange="link_007",
    tools={
        "tcp": ToolPose(offset=(0.0, 0.0, -0.1696387), rotation=((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)))
    },
    # mean of the two arms' median recorded postures (94 episodes of a Maker Arm station); they mirror in pan and wrist
    rest_pose=(0.0, -1.2, 0.81, -0.1, 0.0, 0.0),
    rest_weight=(0.15, 0.3, 0.3, 0.3, 0.3, 0.3),
    max_speed=(float(np.radians(120.0)),) * 6,  # the station's CommandStepper: 2 deg per 60 Hz tick
    # No YAM counterpart for these joints: a YAM checkpoint's joint keys cannot be mapped onto them.
    joint_sign=None,
    link_radius=0.04,
    jaw=None,
    joint_units=None,
)


def _tool_axes(approach: tuple[float, ...], jaw: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:
    """A FaaS bake's TCP axes in the flange frame: z = approach, x = jaw made orthogonal to it, y = z cross x, as the
    rows of the (3, 3) whose columns are those axes."""
    z = np.asarray(approach) / np.linalg.norm(approach)
    x = np.asarray(jaw) - (np.asarray(jaw) @ z) * z
    x = x / np.linalg.norm(x)
    return tuple(map(tuple, np.stack([x, np.cross(z, x), z], axis=1).tolist()))


def _faas_side(
    asset: str,
    joints: tuple[str, ...],
    flange: str,
    tcp: tuple[float, float, float],
    axes: tuple[tuple[float, ...], ...],
    rest: tuple[float, ...],
    max_speed: tuple[float, ...],
    jaw: tuple[float, float],
    limit_margin: float,
) -> ArmSpec:
    """One arm of a FaaS EEF robot as its bake framed it: ``assets/{asset}.urdf`` the chain from the bake's base link
    to its flange, ``joints`` in the recorded column order and signs (the checkpoint's {side}_arm), the bake's TCP.
    Rest: the data's first-frame median. joint_sign None: these checkpoints train on the robot's own joints."""
    return ArmSpec(
        urdf=ASSETS / f"{asset}.urdf",
        joints=joints,
        flange=flange,
        tools={"tcp": ToolPose(offset=tcp, rotation=axes)},
        rest_pose=rest,
        rest_weight=(0.15,) + (0.3,) * (len(joints) - 1),
        max_speed=max_speed,
        joint_sign=None,
        limit_margin=limit_margin,
        link_radius=0.04,
        jaw=jaw,
        joint_units=None,
    )


# The FaaS EEF beta robots: FK of each spec on the recorded joints of the run's
# upload gives its eef labels to 0.0004 mm and 0.0001 deg. The sim eval's <twin>_faas specs, Trossen's TCP refit.
_RAD2 = float(np.radians(2.0))
_APPROACH_X = _tool_axes((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))  # flange x = approach, y = jaw: Metal's relabel
_R1 = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
_ROBO = ("Shoulder", "ShoulderRotate", "ArmRoll", "Elbow", "WristRoll", "WristPitch", "WristYaw")

# Trossen WidowX AI: each arm in its own base_link, TCP along link_6 x with its axes relabelled into YAM's convention.
# 0.146062 m, not the 0.1476 of the bake's notes: the labels of the Trossen EEF bake 1006b sit 1.538 mm closer to the
# flange on every frame. Rest: both arms' first-frame median, wrist roll 0. The jaw is the carriage (m).
TROSSEN_ARM = ArmSpec(
    urdf=ASSETS / "wxai.urdf",
    joints=tuple(f"joint_{i}" for i in range(6)),
    flange="link_6",
    tools={"tcp": ToolPose(offset=(0.146062, 0.0, 0.0), rotation=((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))},
    rest_pose=(-0.001, 0.001, 0.0135, -0.0085, 0.0135, 0.0),
    rest_weight=(0.15, 0.3, 0.3, 0.3, 0.3, 0.3),
    max_speed=(3.0,) * 6,
    joint_sign=None,
    link_radius=0.04,
    jaw=(0.0, 0.044),
    joint_units=None,
)

FAAS_ARMS: dict[str, dict[str, ArmSpec]] = {
    # ROBOTIS AI Worker ffw_sg2: both arms on the torso arm_base_link. Gripper drive 1.1 closed, 0 open.
    "ai_worker_faas": {
        s: _faas_side(
            f"ai_worker_faas_{s}",
            tuple(f"arm_{s[0]}_joint{i}" for i in range(1, 8)),
            f"arm_{s[0]}_link7",
            (0.0, 0.0, -0.2472),
            _tool_axes((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
            rest,
            (3.0,) * 7,
            (1.1, 0.0),
            _RAD2,
        )
        for s, rest in (
            ("left", (0.2127, 0.52, 0.0685, -1.792, 0.1741, 0.0, -1.3972)),
            ("right", (0.2127, -0.5198, -0.0685, -1.792, -0.1741, 0.0, 1.3971)),
        )
    },
    # Enactic OpenArm v2: each arm on its own openarm_{side}_base_link. The left jaw opens to +0.785, the right to
    # -0.785 (the data's grip runs -0.2..1.34: do not clip it).
    "openarm2_faas": {
        s: _faas_side(
            f"openarm2_faas_{s}",
            tuple(f"openarm_{s}_joint{i}" for i in range(1, 8)),
            f"openarm_{s}_ee_base_link",
            tcp,
            _tool_axes(*axes),
            rest,
            (4.0,) * 7,
            (0.0, open_),
            _RAD2,
        )
        for s, tcp, axes, rest, open_ in (
            (
                "left",
                (-0.0016, -0.0001, -0.159),
                ((0.0, 0.0064, -1.0), (0.003, 1.0, 0.0014)),
                (0.3683, -0.1173, -0.261, 2.0834, -0.3145, -0.3624, -0.3874),
                0.785,
            ),
            (
                "right",
                (-0.0016, 0.0001, -0.159),
                ((0.0, -0.0064, -1.0), (-0.003, 1.0, -0.0014)),
                (-0.3393, 0.0903, 0.2203, 2.0708, 0.2411, 0.3439, 0.473),
                -0.785,
            ),
        )
    },
    # Bracketbot: both arms on "base", j0 = the lift slide (m, 0 = top, 0.5 m/s cap). No limit margin: the lift rests
    # at its top limit and the data passes the joints' +-2.094 rad. Gripper 0 closed .. 1 open.
    "bracketbot_faas": {
        s: _faas_side(
            f"bracketbot_faas_{s}",
            tuple(f"{s[0]}j{i}" for i in range(7)),
            f"{s[0]}j6_link",
            tcp,
            _tool_axes(*axes),
            rest,
            (0.5,) + (4.0,) * 6,
            (0.0, 1.0),
            0.0,
        )
        for s, tcp, axes, rest in (
            (
                "left",
                (0.0238, 0.0211, -0.1433),
                ((0.0044, 0.0203, -0.9998), (-0.9993, 0.0307, 0.0232)),
                (0.0, -0.1402, 0.0874, 1.6939, 0.0683, -0.0061, -0.1204),
            ),
            (
                "right",
                (-0.0193, 0.1435, 0.0198),
                ((0.0001, 0.9998, 0.0199), (-0.9992, 0.0271, -0.0296)),
                (0.0, -0.1004, -0.1121, 1.6637, -0.0965, 0.0015, -0.0976),
            ),
        )
    },
    # Blue Worm Mantis: both arms on C_Link. The recorded columns 1 and 2 are Degree3 and Degree2 (roll and yaw
    # exchanged), so the joints list in recorded order. Gripper 1 closed, 0 open.
    "mantis_faas": {
        s: _faas_side(
            f"mantis_faas_{s}",
            tuple(f"A_{s}_Degree{i}_joint" for i in (1, 3, 2, 4, 5, 6, 7)),
            f"A_{s}_Degree7_link",
            (0.1896, 0.0, 0.0326),
            _APPROACH_X,
            (0.0,) * 7,
            (3.0,) * 7,
            (1.0, 0.0),
            _RAD2,
        )
        for s in ("left", "right")
    },
    # Galaxea R1 Lite: the zero-shot galaxea_{side}.urdf, each arm on its served_base. Gripper 0 closed .. 100.
    "galaxea_faas": {
        s: _faas_side(
            f"galaxea_faas_{s}",
            tuple(f"joint{i}" for i in range(1, 7)),
            f"{s}_gripper_link",
            (0.09996, 0.0, 0.0),
            _APPROACH_X,
            rest,
            (3.0,) * 6,
            (0.0, 100.0),
            _RAD2,
        )
        for s, rest in (
            ("left", (0.0138, 0.0541, -0.0367, 0.0057, 0.013, 0.0274)),
            ("right", (0.0053, 0.0367, -0.0367, 0.0096, 0.0249, 0.0181)),
        )
    },
    # Unitree R1 with A7 arms and Dex1 grippers: both arms on waist_yaw_link. Dex1 0 closed .. 5.4 rad open.
    "unitree_r1_faas": {
        s: _faas_side(
            f"unitree_r1_faas_{s}",
            tuple(f"{s}_{j}_joint" for j in _R1),
            f"{s}_wrist_yaw_link",
            tcp,
            _APPROACH_X,
            rest,
            (3.0,) * 7,
            (0.0, 5.4),
            _RAD2,
        )
        for s, tcp, rest in (
            ("left", (0.175, 0.0, -0.024), (0.3212, -0.0017, -0.002, -0.2636, 0.0111, 0.0279, 0.0003)),
            ("right", (0.175, 0.0, 0.0257), (0.3188, -0.0015, 0.0006, -0.2587, 0.0032, 0.0287, 0.0056)),
        )
    },
    # Seeed reBot B601-DM: B601_ARM's chain and TCP in the recorded DM motor signs. Gripper 0 closed .. 1 open.
    "seeed_faas": {
        s: _faas_side(
            "b601_dm",
            tuple(f"joint{i}" for i in range(1, 7)),
            "link6",
            (0.0, 0.0, 0.15539 - 0.020),
            _tool_axes((0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
            rest,
            (4.5,) * 6,
            (0.0, 1.0),
            _RAD2,
        )
        for s, rest in (
            ("left", (0.001, -0.0547, -0.0752, 0.0086, -0.0181, 0.0)),
            ("right", (0.046, -0.0601, -0.0692, 0.0423, 0.0071, -0.054)),
        )
    },
    # ROBO-T1 (Robo-Tv2_dreamlabs.urdf): each arm on its {S}.ShoulderFixed, the joints whose recorded sign is -1 with
    # their axes turned over and limits mirrored, so the joints are the recorded columns. The wrists come within 0.7
    # deg of their limits: margin 0.5 deg. Gripper 1 closed .. 0 open.
    "robo_t_faas": {
        s: _faas_side(
            f"robo_t_faas_{s}",
            tuple(f"{s[0].upper()}.{j}" for j in _ROBO),
            f"{s[0].upper()}.Gripper",
            tcp,
            _tool_axes((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
            rest,
            (4.0,) * 7,
            (1.0, 0.0),
            float(np.radians(0.5)),
        )
        for s, tcp, rest in (
            ("left", (0.0, 0.0, -0.1309), (0.0389, -0.2132, -0.0015, 0.9006, 0.2022, -0.1788, -0.6432)),
            ("right", (0.0, 0.0, -0.1308), (0.0422, 0.1415, -0.1747, 0.3524, -0.0065, 0.565, -0.1189)),
        )
    },
    # XLeRobot with two SO-101 arms (an SO-101 station): each arm on its served_base, TCP where the closed jaws meet.
    # Five
    # joints; joint6 is the wire's constant-zero pad (held). The arm columns are LeRobot units of that robot's own
    # calibration: rad = offset + scale * unit. Gripper 0 closed .. 100 (LeRobot units too).
    "so101_faas": {
        s: ArmSpec(
            urdf=ASSETS / f"so101_faas_{s}.urdf",
            joints=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
            held_joints=("joint6",),
            flange="finger_lf",
            tools={
                "tcp": ToolPose(
                    offset=(-0.0097, -0.0002, -0.09653), rotation=_tool_axes((0.0, 0.0, -1.0), (-1.0, 0.0, 0.0))
                )
            },
            rest_pose=rest,
            rest_weight=(0.15, 0.3, 0.3, 0.3, 0.3, 0.3),
            max_speed=(5.0,) * 6,
            joint_sign=None,
            limit_margin=_RAD2,
            link_radius=0.04,
            jaw=(0.0, 100.0),
            joint_units=(scale + (1.0,), offset + (0.0,)),
        )
        for s, rest, scale, offset in (
            (
                "left",
                (0.3409, -1.7104, 1.5285, 1.1724, -1.4119, 0.0),
                (0.0176867985, 0.0181546626, 0.0166973809, 0.0181086432, 0.0314082566),
                (0.3006602344, 0.0268446638, -0.1158155495, 0.0283786446, 0.0479125096),
            ),
            (
                "right",
                (-0.3722, -1.6763, 1.459, 1.0564, -1.4099, 0.0),
                (0.0166283517, 0.0177481577, 0.0170195168, 0.0191440802, 0.0314082566),
                (-0.2960582921, 0.0490873852, -0.2155243007, -0.0705631162, 0.0479125096),
            ),
        )
    },
}


def _zero_shot_arm(robot: str, side: str, rest: tuple[float, ...]) -> ArmSpec:
    """One side of a zero-shot sim robot: the simulation evaluation's arm split writes its arm on a root
    ``served_base`` at YAM's arm base pose for that side, joints joint1..6 (joint1..7 for a 7-joint arm with no
    joint locked), jaws jaw_drive and jaw_f1.., and a ``tcp`` link in YAM's TCP convention, so its EEF poses mean
    what YAM's do. ``rest`` (the sim's, one value per joint) is kept 0.1 rad inside the URDF limits. A five-joint
    arm's joint6 is the split's dummy (+-0.001 rad): held at 0."""
    urdf = ASSETS / f"{robot}_{side}.urdf"
    found = {j.get("name"): j for j in ET.parse(urdf).getroot().findall("joint")}
    joints = tuple(f"joint{i}" for i in range(1, 8) if f"joint{i}" in found)
    assert joints in (tuple(f"joint{i}" for i in range(1, 7)), tuple(f"joint{i}" for i in range(1, 8))), (urdf, joints)
    assert len(rest) == len(joints), f"{robot} {side}: {len(rest)} rest values for {len(joints)} joints"
    lim = np.array([[float(found[j].find("limit").get(k)) for k in ("lower", "upper")] for j in joints])  # (dof, 2) rad
    jaws = tuple(n for n in found if n.startswith("jaw_"))
    dummy = lim[:, 1] - lim[:, 0] < 0.01  # (dof,)
    return ArmSpec(
        urdf=urdf,
        joints=joints,
        fixed_joints=jaws,
        held_joints=tuple(f"joint{i + 1}" for i in np.flatnonzero(dummy)),
        flange="tcp",
        tools={"tcp": ToolPose(offset=(0.0, 0.0, 0.0), rotation=_IDENTITY)},
        rest_pose=tuple(float(x) for x in np.where(dummy, 0.0, np.clip(rest, lim[:, 0] + 0.1, lim[:, 1] - 0.1))),
        rest_weight=(0.15,) + (0.3,) * (len(joints) - 1),
        max_speed=(3.0,) * len(joints),
        joint_sign=None,  # no YAM counterpart: eef head only, joint slots from serve.nearest_yam_joints
        link_radius=0.04,
        jaw=None,
        joint_units=None,
    )


_ZERO_SHOT_REST = {
    "galaxea": {
        "right": (0.678, 2.254, -0.351, -1.093, -0.507, 0.471615),
        "left": (-0.681, 2.254, -0.353, -1.09, 0.509, -0.473957),
    },
    "unitree_r1": {  # all seven joints (2026-10-06), wrist roll fifth
        "right": (0.678, -0.176, -0.049, -0.95, 0.0, 0.548, -0.038),
        "left": (0.678, 0.176, 0.049, -0.95, 0.0, 0.548, 0.038),
    },
    "trossen_wb": {
        "right": (0.03, 0.001, 0.02, 0.008, 0.031, -0.00083),
        "left": (-0.032, 0.001, 0.007, -0.025, -0.004, -0.000348),
    },
    "ai_worker": {  # all seven joints, elbows raised and out at the same rest hand pose (2026-10-06)
        "right": (-0.6379, -1.2725, 1.3835, -1.6977, -0.6657, -0.357, 0.6593),
        "left": (-0.6379, 1.2725, -1.3835, -1.6977, 0.6657, -0.357, -0.6593),
    },
    "so101": {  # five joints, joint6 the dummy
        "right": (-0.106, -0.838, 0.74, 1.145, 0.0, 0.0),
        "left": (0.104, -0.838, 0.74, 1.145, 0.0, 0.0),
    },
    "bracketbot": {  # left lift lj0 at -0.1, right at 0; left = the right rest mirrored (2026-10-06)
        "right": (1.8014, -1.5695, -1.0869, 1.3603, 1.8058, 1.285),
        "left": (1.8014, 1.5695, 1.0869, 1.3603, -1.8058, -1.285),
    },
    "openyam": {
        "right": (0.051, 0.063, 0.076, -0.327, -0.278, 0.137),
        "left": (0.128, 0.006, 0.02, 0.0, 0.154, 0.057),
    },
    "yam7": {  # all seven joints (2026-10-06), joint3 at its old lock value
        "right": (-0.015, 1.815, -1.5708, -1.994, 0.001, -0.028, -0.943),
        "left": (-0.015, 1.815, -1.5708, -1.994, 0.001, -0.028, -0.943),
    },
    "arx_x5": {
        "right": (-0.001, 0.001, 0.004, -0.005, -0.004, 0.021),
        "left": (0.002, 0.001, 0.004, -0.004, 0.003, 0.002),
    },
    "openarm2": {  # all seven joints (2026-10-06), joint3 at its old lock value
        "right": (-0.336, 0.08, 0.226, 2.073, 0.237, 0.347, 0.486),
        "left": (0.369, -0.121, -0.262, 2.084, -0.303, -0.365, -0.389),
    },
    "mantis": {  # all seven joints (2026-10-06)
        "right": (-0.3, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0),
        "left": (-0.3, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0),
    },
    "aloha": {  # ALOHA's ACT start pose
        "right": (0.0, -0.96, 1.16, 0.0, -0.3, 0.0),
        "left": (0.0, -0.96, 1.16, 0.0, -0.3, 0.0),
    },
    "yumi": {  # vendor joint 7 (between joints 2 and 3) locked at 0
        "right": (-1.5283, 0.7592, 0.4114, 2.2813, 1.9802, -1.6159),
        "left": (1.5243, 0.7592, 0.4115, -2.2814, 1.9802, -1.5257),
    },
    "pr2": {  # all seven joints (2026-10-06)
        "right": (-0.7615, -0.1551, -1.55, -1.5336, -1.4165, -1.5971, -2.365854),
        "left": (0.7615, -0.1551, 1.55, -1.5336, 1.4165, -1.5971, 2.365834),
    },
    "baxter": {  # all seven joints (2026-10-06)
        "right": (0.1064, -0.7827, 0.8441, 1.8018, -0.7415, 0.902, -2.633559),
        "left": (-0.1064, -0.7827, -0.8441, 1.8018, 0.7415, 0.902, 2.633539),
    },
    # gripper re-mount variants (2026-10-06): the gripper turned on its flange (right +90, left -90 deg), rest re-solved
    "openarm2_rm": {
        "right": (-0.1623, 1.0314, -0.5506, 2.0715, -0.5866, 0.3346, -0.2427),
        "left": (0.2375, -1.0865, 0.4959, 2.0828, 0.4983, -0.3981, 0.2892),
    },
    "unitree_r1_rm": {
        "right": (0.6857, -0.4418, 0.1354, -0.84, -1.5415, 0.3539, 0.4158),
        "left": (0.6716, 0.5201, -0.2112, -0.8293, 1.5366, 0.4628, -0.4023),
    },
    # dual copies of single arms (2026-10-06): one at each YAM base spot, both driven; left = the right rest's TCP mirrored
    "ur5e_dual": {
        "right": (-0.1199, -2.4536, 2.4875, 0.5467, 1.8679, 0.05402),
        "left": (-3.02169, -0.68799, -2.4875, 2.59489, -1.8679, -0.05402),
    },
    "franka_dual": {  # all seven joints; the hand sits -45 deg on the flange, so the left wrist roll is not the mirror
        "right": (1.4434, -0.456, -1.2003, -2.7175, 2.4685, 3.8602, -1.534851),
        "left": (-1.61801, -0.81807, 1.27978, -2.78942, -2.12467, 4.10648, 2.72276),
    },
    "kinova_gen3_dual": {  # left = the true mirror branch (joint 1 = pi minus the right's), elbow out
        "right": (1.3209, 0.2584, -1.6126, 2.2022, -2.8278, -0.9739, 1.097284),
        "left": (1.82035, -0.25846, -1.62907, 2.20227, -0.31933, 0.94867, -1.1912),
    },
    "piper_dual": {  # the left rest is the joint-space mirror (the arm is planar: joints 1, 4 and 6 flip)
        "right": (0.0, 0.662, -0.1, -0.001, -0.216, 0.001),
        "left": (0.0, 0.662, -0.1, 0.001, -0.216, -0.001),
    },
    "widowx250_dual": {  # the hand 24.5 cm ahead of its base, 16.4 cm up, pointing down; both sides the same joints
        "right": (0.0, -0.16727, 0.48414, 0.0, 1.25392, 0.0),
        "left": (0.0, -0.16727, 0.48414, 0.0, 1.25392, 0.0),
    },
    # two more dual copies (2026-10-07): the xArm 7 rest is planar (roll joints at 0 or pi), so its left rest equals the right
    "xarm7_dual": {
        "right": (3.0632, 0.1944, -3.0736, 0.7729, -3.1257, -0.9669, 3.120917),
        "left": (3.0632, 0.1944, -3.0736, 0.7729, -3.1257, -0.9669, 3.120917),
    },
    "sawyer_dual": {  # the right upper arm swings out to the right; the left is the IK solution nearest the mirror image
        "right": (-2.4283, -1.5229, 0.0, -2.3949, 0.0, -0.7946, -0.682981),
        "left": (0.28718, -1.48828, -0.95902, 2.33933, 0.09838, 0.75722, 1.00348),
    },
    # single-arm robots: the right arm only (2026-10-06), the left EEF side served as an idle YAM arm
    "openyam_r": {"right": (0.051, 0.063, 0.076, -0.327, -0.278, 0.137)},
    "piper": {"right": (0.0, 0.662, -0.1, -0.001, -0.216, 0.001)},
    "widowx250": {"right": (-1.4485, 0.0696, 0.2102, 0.0, 1.291, -1.4485)},  # on a 5 cm block, rest down
    "franka": {"right": (1.4434, -0.456, -1.2003, -2.7175, 2.4685, 3.8602, -1.534851)},  # all seven joints (2026-10-06)
    "ur5e": {"right": (-0.1199, -2.4536, 2.4875, 0.5467, 1.8679, 0.05402)},
    "iiwa14": {"right": (0.0, 0.2059, 0.0, -1.8064, 0.0, 1.1293, 1.5708)},  # all seven joints (2026-10-06)
    "sawyer": {"right": (-2.4283, -1.5229, 0.0, -2.3949, 0.0, -0.7946, -0.682981)},  # all seven joints (2026-10-06)
    "kinova_gen3": {
        "right": (1.3209, 0.2584, -1.6126, 2.2022, -2.8278, -0.9739, 1.097284)
    },  # all seven joints (2026-10-06)
    "xarm7": {"right": (3.0632, 0.1944, -3.0736, 0.7729, -3.1257, -0.9669, 3.120917)},  # all seven joints (2026-10-06)
}
ZERO_SHOT_ROBOTS = tuple(_ZERO_SHOT_REST)  # the robots of our simulation zero-shot eval

# YAM box camera stick: 2 in radius, midway between bases 0.61 m apart, arms yawed 90 deg in the box
# frame (the YAM box's workspace guard). In each arm's own base frame it stands 0.305 m to the side.
_STICK_RADIUS_M, _HALF_SEPARATION_M = 0.0508, 0.305

EMBODIMENTS: dict[str, Embodiment] = {
    "yam": Embodiment(
        arm=YAM_ARM,
        sides=("right", "left"),  # action_space.YAM_ARMS order
        obstacles={
            "right": (Capsule(center=(0.0, _HALF_SEPARATION_M, 0.0), radius=_STICK_RADIUS_M, height=2.0),),
            "left": (Capsule(center=(0.0, -_HALF_SEPARATION_M, 0.0), radius=_STICK_RADIUS_M, height=2.0),),
        },
        # the pole stands +0.305 m in the right base frame and -0.305 m in the left's: the left base is 0.61 m out
        side_base={"right": (0.0, 0.0, 0.0), "left": (0.0, 2 * _HALF_SEPARATION_M, 0.0)},
    ),
    # the camera mount is the station's, not the arm's: a station passes its own (its station file's keep_out)
    "metal": Embodiment(arm=METAL_ARM, sides=("right", "left"), obstacles={"right": (), "left": ()}),
    "b601": Embodiment(arm=B601_ARM, sides=("right", "left"), obstacles={"right": (), "left": ()}),
    # camera mount not measured
    "makers": Embodiment(arm=MAKERS_ARM, sides=("right", "left"), obstacles={"right": (), "left": ()}),
    # Trossen's checkpoint reads the jaw drive as the last entry of {side}_arm, with no gripper key
    "trossen_wb_faas": Embodiment(
        arm=TROSSEN_ARM, sides=("right", "left"), obstacles={"right": (), "left": ()}, jaw_in_arm=True
    ),
    **{
        name: Embodiment(arm=arms, sides=("right", "left"), obstacles={"right": (), "left": ()})
        for name, arms in FAAS_ARMS.items()
    },
    # zero-shot sim robots: mirrored arms, an ArmSpec per side (or one side, a single arm)
    **{
        robot: Embodiment(
            arm={s: _zero_shot_arm(robot, s, r) for s, r in rest.items()},
            sides=tuple(s for s in ("right", "left") if s in rest),
            obstacles={s: () for s in rest},
        )
        for robot, rest in _ZERO_SHOT_REST.items()
    },
}
# yamumi (YAM's arms with GenRobot's DAS gripper) for a YAM checkpoint: the TCP sits where YAM's does against its
# fingertips, the robot files' rule (the sim's TIP_PAST_TCP_M): 7.9 mm short of the DAS fingertip midpoint at
# half its widest gap, (0, 41.5, 189.2) mm in link_6. YAM's own TCP is 134.7 mm out, its fingertips 142.6 mm.
EMBODIMENTS["yamumi"] = EMBODIMENTS["yam"].model_copy(
    update={
        "arm": YAM_ARM.model_copy(update={"tools": {"tcp": ToolPose(offset=(0.0, 0.0415, 0.1813), rotation=_IDENTITY)}})
    }
)


# Which arm's joint convention a checkpoint's joint keys (state and action ``{side}_arm``) are in, by embodiment
# tag. Serving on another arm maps its joints through both ``joint_sign``s. Our base models' tags are YAM's. None: the
# robot's own joints and gripper as recorded (the FaaS EEF contract of 2026-10-06: every part but the six EEF parts
# is the customer's own joint, so a Metal fine-tune reads Metal's shoulder sign and its gripper in mm).
MODEL_JOINTS: dict[str, str | None] = {"yam_eef_joint": "yam", "yam_gripper": "yam", "faas_new_embodiment_eef": None}


class ServeSpec(BaseModel):
    """How an EEF head is served to a joints-only consumer (``dream.ik.serve``)."""

    model_config = {"frozen": True}

    tool: str = "tcp"  # the frame every EEF checkpoint's eef keys are in (the eef_tcp bake)
    # Savitzky-Golay over the chunk's hand targets before the solve: the EEF head is 2.3x noisier
    # than the joint head, and unsmoothed its targets put 2 of 3 ticks on the speed cap (our eef-ik branch).
    smooth_window: int = 15  # ticks, odd; 0 = off
    smooth_polyorder: int = 3
    # Each new chunk eases in from the last command over this many ticks instead of jumping to its plan: the
    # chunk seam carries most of the large steps in sim (09/29, rows q1388-q1390). 0 = no ease-in (the clip to the
    # margined limits and the speed cap still hold). 12 (09/30 sim hill
    # climb, with contract v1.2): the step into a new chunk went from 5.5x an average step to 0.4x; 6 gave 0.8x.
    seam_blend_ticks: int = 12
    # "tick" solves the chunk tick by tick (warm start, a pull to the last tick) then eases it in (seam_blend_ticks).
    # "traj" solves every tick of every arm in one problem: velocity and acceleration costs along the chunk, and the
    # last two commands fixed as its history, so it continues the arm's motion with no ease-in (which braked the arm
    # to 0.39x its speed at every seam in the 2026-10-17 replays), and the two arms kept ``interarm_margin`` apart
    # where ``Embodiment.side_base`` says where they stand. Offline replays of teleop on five arms (Metal, Makers,
    # Seeed reBot, SO-101, YAM; 2026-10-17), the head's error modelled as a drifting chunk bias plus jitter: joint
    # jerk -93..-95%, direction flips -91%, joint error vs the demonstration -3..-47%, the median hand error within
    # +0.6 mm; on exact targets the median hand error is 0.1-0.8 mm higher. About 20 ms a chunk against 12. The
    # robot stacks pick "traj" themselves (the dream client's ik.solve: robot results in
    # our IK notes); dream's own default, which sim eval's decode runs, stays "tick" until a simulation A/B: on exact
    # sim targets traj's smoothing round-trips hand paths 7-16 mm off where tick holds 2 mm (an internal test).
    solve: str = "tick"
    # The trajectory solve's smoothing: velocity 20 / acceleration 100 / jerk 200, history weight 5x (three history
    # rows). On a Metal (2026-10-18, same robot, task, model and code): speed-cap clamps per chunk 1.65-2.06 against
    # 2.73 on tick, joint jerk 436-1557 deg/s^3, direction flips 0.10-0.49/s. A lighter 5 / 50 / 100 / 1x, chosen
    # offline for a faster step response (a still arm stepped ~0.1 rad arrives in 9-16 rows instead of 15-37; the
    # velocity cost in radians outweighs the pose cost in metres on small arms such as SO-101), was worse on that
    # Metal: clamps 3.09-3.79, jerk 879-2207 deg/s^3, flips 0.45-0.80/s, worse than tick on the speed cap. The weak
    # history weight lets a new chunk start off the arm's motion, which the speed cap then clips.
    # 2026-10-18, YAM: 20 / 100 braked the arm over each chunk's last rows (0.88 of the demonstrated hand speed, tick
    # 0.98) and slowed its answer to a correction (a 3 cm step reached 90% in 13.5 rows, Metal worst 28, tick 11).
    # Velocity 5 and acceleration 50, jerk and history weight unchanged: 0.95, 11 rows (Metal worst 16), the arms
    # still kept apart under a pull of the hands together; the sim playground, the same 100 YAM tasks: 49 against tick's 49
    # in the same run (20 / 100: 46 against 52).
    traj_vel_weight: float = 5.0  # joint velocity, every tick
    traj_acc_weight: float = 50.0  # joint acceleration, every tick
    # Joint jerk, every tick, and with three history rows across the seam too; a third history row without a jerk
    # cost changes nothing (no cost reads it).
    traj_jerk_weight: float = 200.0
    traj_history_weight: float = 5.0  # x the three above where the chunk meets the last three commands
    # The two arms kept apart (the trajectory solve, where side_base is known): every sphere pair of the two arms at
    # least interarm_margin apart, except two gripper spheres (flange to tool point), which may overlap by up to
    # -interarm_hand_margin: the hands may touch (a hand-over), the arms may not. In the 2026-10-17 YAM teleop
    # replays every close approach was gripper to gripper (to 23 mm into each other on hand-overs); hand to arm kept
    # 32 mm or more and arm to arm 45 mm or more, so the cost never fights teleop. 0 turns it off.
    interarm_weight: float = 2000.0
    interarm_margin: float = 0.03  # m between the two arms' collision spheres
    interarm_hand_margin: float = -0.05  # m between two gripper spheres: negative, a touch is allowed
    # A ``Region`` the arm's centreline leaves by more than this (m) holds the arm at the row before, for the
    # rest of the chunk, unless that row is no farther out than the arm already was (an arm outside may come back).
    region_slack: float = 0.01
    # Hard fences after the solve: hold the arm at the row before when a point of it enters the camera pole, a mount
    # or past a Region (fence_keep_out), or when the two arms touch (fence_interarm). Off (2026-10-18): the trajectory
    # solve runs like the tick solve, kept off the pole, the mounts, a Region and the other arm by its costs alone, and
    # never stopped mid-chunk. On: those holds as before.
    fences: bool = False


SERVE = ServeSpec()
