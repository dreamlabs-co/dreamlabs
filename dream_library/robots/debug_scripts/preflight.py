"""Read-only preflight on a station: the config loads, the arm driver imports, the arm model's forward kinematics
place the grasp point where a zero pose should (one sanity line per arm), every camera delivers frames through the
product's own `CameraReader`, and DreamAPI answers two real inferences for those frames. The arms are never
opened: the observation carries a zero arm state, so nothing moves.

    .venv/bin/python debug_scripts/preflight.py station.yaml

Run it before `station_run.sh start` to find a dead camera, a bad key or an unreachable gateway while the arms are
still off. It prints one line per check and `preflight done` at the end; a failing check raises and stops it.
"""

import argparse
import time

import numpy as np
from i2rt.robots.get_robot import get_yam_robot  # noqa: F401  (the import proves the hardware extra is installed)
from i2rt.robots.utils import GripperType

from robots import adapter, config, errors, kinematics, run, types

## Seconds the capture threads get to deliver a first frame before the cameras are read.
CAMERA_SETTLE_S: float = 2.0

## Real inferences asked of DreamAPI; the second shows the warm latency.
INFERENCE_COUNT: int = 2


def main(config_path: str) -> None:
    """Runs the checks in order and prints one line per check.

    @param config_path The station config, the file `robots.run` takes.
    @throws ConfigError A key is missing, a camera is not a serve role, or mujoco is missing.
    @throws CameraError A camera cannot be opened.
    @throws DreamError DreamAPI refuses the key, the checkpoint or the observation.
    """
    cfg = config.load(config_path)
    print(
        "config ok:",
        repr(cfg.task),
        run.DEFAULT_CHECKPOINT,
        [camera.name for camera in cfg.cameras],
        GripperType(cfg.follower_l.gripper),
    )
    for name, arm in (("left", cfg.follower_l), ("right", cfg.follower_r)):
        fk = kinematics.ArmKinematics.from_config(arm)
        at_zero = np.round(fk.ee_position(np.zeros(6)), 4).tolist()
        print(f"fk: {name} {kinematics.EE_SITE} at zero pose = {at_zero} m (cap {cfg.max_ee_speed_m_s} m/s)")
    images = capture_frames(cfg)
    now_s = time.monotonic()
    zero = types.ArmState(
        positions_rad=np.zeros(6),
        velocities_rad_s=np.zeros(6),
        gripper=0.0,
        commanded_rad=None,
        commanded_gripper=None,
    )
    state = types.RobotState(left=zero, right=zero, timestamp_s=now_s)
    frames, observation = adapter.to_dream(types.Observation(state=state, images=images, timestamp_s=now_s), cfg.task)
    print("observation keys:", sorted([*frames, *observation]))
    client = run.open_client(run.policy_config(cfg))
    try:
        # One tick: each camera's encoded frame, timed by the newest capture among them, and the zero arm state.
        captured_s = max(frame.jpeg_timestamp_s for frame in images.values())
        client.set_observation(frames, observation, timestamp=captured_s)
        for index in range(INFERENCE_COUNT):
            started_s = time.perf_counter()
            action = client.get_action()
            chunk = adapter.from_dream(action, now_s, cfg.action_horizon)
            print(
                f"real inference {index + 1} from the station: {(time.perf_counter() - started_s) * 1000:.0f} ms,",
                f"{len(chunk.actions)} actions;",
                "first right target rad:",
                np.round(chunk.actions[0].right.positions_rad, 3).tolist(),
            )
        client.reset_episode()
    finally:
        client.close()
    print("preflight done")


def capture_frames(cfg: config.AppConfig) -> dict[str, types.Frame]:
    """Starts every camera from the config, reads one frame each after #CAMERA_SETTLE_S, stops them.

    @param cfg The loaded station config.
    @return One frame per camera name, as the adapter takes them.
    @throws CameraError A camera delivered nothing.
    """
    readers = run.make_cameras(cfg)
    for reader in readers:
        reader.start()
    time.sleep(CAMERA_SETTLE_S)
    now_s = time.monotonic()
    images: dict[str, types.Frame] = {}
    for camera, reader in zip(cfg.cameras, readers, strict=True):
        frame = reader.frame
        if frame is None:
            print(camera.name, "no frame")
        else:
            print(
                camera.name, f"{frame.data.shape} age {now_s - frame.timestamp_s:.3f} s mean {frame.data.mean():.0f}"
            )
            images[camera.name] = frame
    for reader in readers:
        reader.stop()
    missing = [camera.name for camera in cfg.cameras if camera.name not in images]
    if missing:
        raise errors.CameraError(f"cameras {missing} delivered no frame within {CAMERA_SETTLE_S} s")
    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="the station config, e.g. station.yaml")
    main(parser.parse_args().config)
