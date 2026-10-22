"""Return both follower arms to their home pose: each arm's configured joints (`FollowerArm.home_rad`), grippers open.

Home is the pose runs start from, so the station homes between runs and before an operator touches the scene. Each
arm's home is its config section's `home` (#robots.config.ArmConfig; zeros on a YAM). The move is
a linear interpolation from each arm's measured pose to its
home joints with the gripper at 1.0 (open) over `duration_s` in `steps` steps, about 40 Hz, and a hold at the end. It
writes `FollowerArm.write` directly at the step rate; the loop's command thread is not involved, and must not be
running: the caller (the station controller) guarantees no loop owns the arms while a home move runs, so the arms are
never written from two places at once.

Best-effort and per arm: an arm whose read or write fails is logged and left holding
whatever it last accepted, the other arm finishes its move, and the return value says which
arm did not make it, and why. Nothing here raises for a follower fault mid-move.

```bash
python -m robots.home --config station12.yaml    # exit 0 when both arms reached home, 1 otherwise
```
"""

import argparse
import logging
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from robots import bodies, config, followers, types

## Seconds the default move takes.
HOME_DURATION_S: float = 3.0

## Steps of the default move; 120 over 3 s is 40 Hz.
HOME_STEPS: int = 120

HOME_GRIPPER: float = 1.0

_log = logging.getLogger(__name__)


def return_to_home(
    arms: dict[str, followers.Arm],
    *,
    duration_s: float = HOME_DURATION_S,
    steps: int = HOME_STEPS,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Any] = time.sleep,
) -> bool:
    """move_home(), as whether every arm made it.

    @param arms The arms to home, keyed by name.
    @param duration_s Seconds the move takes.
    @param steps Steps the move is cut into.
    @param clock Monotonic clock, seconds.
    @param sleep Blocking sleep, seconds.
    @return True when every arm reached home without a fault.
    """
    return not move_home(arms, duration_s=duration_s, steps=steps, clock=clock, sleep=sleep)


def move_home(
    arms: dict[str, followers.Arm],
    *,
    duration_s: float = HOME_DURATION_S,
    steps: int = HOME_STEPS,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Any] = time.sleep,
) -> dict[str, str]:
    """Moves every arm linearly from its measured pose to its home joints with the gripper open, then holds.

    @param arms The arms to home, keyed by name.
    @param duration_s Seconds the move takes.
    @param steps Steps the move is cut into.
    @param clock Monotonic clock, seconds.
    @param sleep Blocking sleep, seconds.
    @return The arms that did not reach home, by name, with why.
    """
    steps = max(1, steps)
    failed: dict[str, str] = {}
    origins = _read_origins(arms, failed)
    moving = dict(origins)
    if not moving:
        _log.error("home: no arm could be read; nothing moved")
        return failed
    _log.info(
        "home: moving %s home with grippers open over %.1f s in %d steps; largest joint delta %.3f rad",
        sorted(moving),
        duration_s,
        steps,
        max(float(np.max(np.abs(origin.positions_rad - arms[name].home_rad))) for name, origin in moving.items()),
    )
    period_s = duration_s / steps
    started_s = clock()
    for step in range(1, steps + 1):
        alpha = step / steps
        for name in list(moving):
            command = _toward_home(moving[name], arms[name].home_rad, alpha)
            try:
                arms[name].write(command)
            except Exception as exc:
                # The one broad except: a driver fault on one arm must not stop the other arm's
                # move, and the arm keeps its last accepted PD target, which is the hold.
                _log.exception("home: arm '%s' write failed at step %d/%d; it holds where it is", name, step, steps)
                del moving[name]
                failed[name] = f"its write failed at step {step}/{steps}: {exc}"
        if not moving:
            break
        sleep(max(0.0, started_s + step * period_s - clock()))
    for name in moving:
        try:
            arms[name].hold()
        except Exception as exc:
            _log.exception("home: arm '%s' hold at the end failed", name)
            failed[name] = f"its hold at the end failed: {exc}"
    if failed:
        _log.error("home: arms %s did not reach home", sorted(failed))
        return failed
    _log.info("home: arms %s at home and holding", sorted(arms))
    return failed


def _read_origins(arms: dict[str, followers.Arm], failed: dict[str, str]) -> dict[str, types.ArmCommand]:
    """Reads every arm's measured pose as the command the move starts from; an arm that fails to read is left out.

    @param arms The arms to read.
    @param failed Where an arm that fails to read is put, with why.
    @return The origin per arm name, the measured gripper clipped into `[0, 1]` so every blend validates.
    """
    origins: dict[str, types.ArmCommand] = {}
    for name, arm in arms.items():
        try:
            state = arm.read()
        except Exception as exc:
            # Same boundary as the write: the other arm still homes.
            _log.exception("home: arm '%s' read failed; it is not moved", name)
            failed[name] = f"its read failed: {exc}"
            continue
        origins[name] = types.ArmCommand(
            positions_rad=np.asarray(state.positions_rad, dtype=np.float64),
            gripper=min(max(state.gripper, 0.0), 1.0),
        )
    return origins


def _toward_home(origin: types.ArmCommand, home_rad: npt.NDArray[np.float64], alpha: float) -> types.ArmCommand:
    """`(1 - alpha) * origin + alpha * home` for the joints and `origin + alpha * (open - origin)` for the gripper.

    @param origin The pose the move started from.
    @param home_rad The arm's home joints, radians.
    @param alpha Blend weight in `(0, 1]`; 1 is exactly the home pose.
    @return The command for this step.
    """
    return types.ArmCommand(
        positions_rad=(1.0 - alpha) * origin.positions_rad + alpha * home_rad,
        gripper=origin.gripper + alpha * (HOME_GRIPPER - origin.gripper),
    )


def main(
    cfg: config.AppConfig,
    *,
    duration_s: float = HOME_DURATION_S,
    build_arm: Callable[[config.ArmConfig, str], followers.FollowerArm] | None = None,
) -> int:
    """Builds both followers from the config and homes them.

    @param cfg The station config; `body`, `follower_l` and `follower_r` are used.
    @param duration_s Seconds the move takes.
    @param build_arm Opens one arm from its config and name; None opens the config's body's driver
        (`robots.bodies`), a test passes a fake.
    @return The process exit code: 0 when both arms reached home, 1 otherwise.
    @throws ConfigError The body's driver is not installed (i2rt for a YAM, python-can for a Makers or Metal arm).
    """
    open_arm = bodies.for_config(cfg).open_arm if build_arm is None else build_arm
    arms: dict[str, followers.Arm] = {
        "left": open_arm(cfg.follower_l, "left"),
        "right": open_arm(cfg.follower_r, "right"),
    }
    return 0 if return_to_home(arms, duration_s=duration_s) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move both follower arms to their home pose, grippers open.")
    parser.add_argument("--config", required=True, help="Path to the YAML application config.")
    parser.add_argument("--duration-s", type=float, default=HOME_DURATION_S, help="Seconds the move takes.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(main(config.load(args.config), duration_s=args.duration_s))
