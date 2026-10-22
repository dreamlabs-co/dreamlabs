"""The beta robots the setup wizard knows, each with the layout its fine-tunes train on.

`beta_robots.json` holds, per robot, what a guided setup would otherwise ask: dream's IK arm, the wizard tile that
carries it, where the gripper drive sits in the state, which arm the state records first, the cameras by role and the
head camera among them, and the rate its fine-tunes run at. It is written from each robot's fine-tuning body config
and training layout; no spans are stored, since embodiment_config() derives them from dream's spec of the arm.
tests/test_beta_robots.py diffs each layout against the body's, field by field.

A new embodiment the wizard registers for one of these arms takes the robot's layout instead of
#robots.adapter.zero_shot_embodiment's, so its fine-tunes and Dream Labs' base models both run on what the setup
saves, and its data uploads into the robot's dataset shape.
"""

import dataclasses
import json
import pathlib
from typing import Any

from robots import adapter, kinematics

TABLE = pathlib.Path(__file__).with_name("beta_robots.json")

## The tool-pose parts every FaaS fine-tune reads, right arm first, in both vectors (the training service's).
EEF_PARTS: tuple[tuple[str, int], ...] = (
    ("right_eef_xyz", 3),
    ("right_eef_rot6", 6),
    ("right_grip", 1),
    ("left_eef_xyz", 3),
    ("left_eef_rot6", 6),
    ("left_grip", 1),
)


@dataclasses.dataclass(frozen=True)
class BetaRobot:
    """One beta robot.

    Attributes:
    - id: Its key in the table.
    - label: What the page calls it.
    - ik: dream's IK arm (`dream_ik.specs.EMBODIMENTS`).
    - body: The wizard's tile that carries it: one of our bodies (`metal`, `makers`, `so101`) or `custom`.
    - gripper: Where the drive sits in the state: `part` (`<side>_gripper` after each arm), `in_arm` (the last
      entry of `<side>_arm`, Trossen's) or `after_arms` (`<side>_gripper` after both arms, Unitree R1's).
    - first_arm: The side the state records first, `left` or `right` (reBot and OpenArm 2 record the right first).
    - cameras: Each video role to the dataset key its data records it under.
    - head_camera: The role the data service labels dense text from.
    - control_hz: The rate FaaS trains its fine-tunes at, a 15 fps dataset resampled up to it. A station plays them
      at its own `control_hz`, 20 by default (`robots.config`).
    """

    id: str
    label: str
    ik: str
    body: str
    gripper: str
    first_arm: str
    cameras: dict[str, str]
    head_camera: str
    control_hz: float


def robots() -> dict[str, BetaRobot]:
    """Every robot of #TABLE, by id, in its order.

    @return Each #BetaRobot by its id.
    """
    return {row["id"]: BetaRobot(**row) for row in json.loads(TABLE.read_text(encoding="utf-8"))["robots"]}


def by_ik(ik: str) -> BetaRobot | None:
    """The robot on the arm `ik` whose new embodiment takes the robot's layout.

    That is an SO-101 or a robot of your own. A MakerMods arm keeps its body's built-in layout, the same as the
    table's; picked as a robot of your own it gets the zero-shot one, whose gripper needs no `grip_scale`.

    @param ik dream's name of the arm (`dream_ik.specs.EMBODIMENTS`).
    @return The robot, or None when no SO-101 or robot of your own of the table is on that arm.
    """
    return next((r for r in robots().values() if r.ik == ik and r.body in ("so101", "custom")), None)


def embodiment_config(robot: BetaRobot) -> dict[str, Any]:
    """The embodiment config of the robot's own layout.

    Its arms are as wide as dream's spec of the arm, in the order its data records them, its gripper drive where
    its data records it, then #EEF_PARTS, its cameras and head camera.

    @param robot The robot.
    @return `{"state", "action", "video", "head_camera"}`, each vector's spans tiling it from 0.
    @throws ConfigError dream's IK has no such arm.
    """
    width = kinematics.arm_layout(robot.ik).width
    sides = ("left", "right") if robot.first_arm == "left" else ("right", "left")
    if robot.gripper == "part":
        arms = [part for side in sides for part in ((f"{side}_arm", width), (f"{side}_gripper", 1))]
    elif robot.gripper == "after_arms":
        arms = [*((f"{side}_arm", width) for side in sides), *((f"{side}_gripper", 1) for side in sides)]
    else:
        arms = [(f"{side}_arm", width) for side in sides]
    state = adapter._spans([*arms, *EEF_PARTS])
    action = adapter._spans([*EEF_PARTS, ("left_arm", width), ("right_arm", width)])
    return {"state": state, "action": action, "video": dict(robot.cameras), "head_camera": robot.head_camera}
