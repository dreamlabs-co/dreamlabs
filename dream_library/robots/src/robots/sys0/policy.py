"""The System 0 network: the observation it reads and the bundled weights, evaluated with numpy.

The tracker is a PPO-trained actor: an MLP of two 256-wide ELU
layers from a 98-float observation to a 7-float action in `[-1, 1]`, one network for both arms. A station runs only
its deterministic mean, so the weights ship as arrays in `tracker.npz` beside this module and the forward pass is
three matrix products: no PyTorch on the box, and about 30 microseconds a call.

`tracker.npz` holds `w0`, `b0`, `w1`, `b1`, `w2`, `b2` (each dense layer's weight `[out, in]` and bias, float32),
`obs_mean` and `obs_std` (the normaliser, `[OBS_DIM]`) and `meta`, a JSON string (`actuator`, `iteration`, the
`sha256` of the checkpoint it came from). A station with a tracker of its own trained for the same observation and
action replaces the file with one in this form; load() refuses a file whose widths do not fit.

The observation is SONIC's tracker input at arm scale, per arm: the measured joints and gripper `q`, the joint
velocities scaled by #VEL_SCALE, the previous action, and the reference now and at the next #FUTURE_FRAMES
look-ahead steps of #FUTURE_DT_S, each as its error from `q`. It is normalised with the running mean and standard
deviation frozen at training, clipped to +-#OBS_CLIP.

@see Policy, act(), observation(), load()
"""

import dataclasses
import json
import pathlib
from typing import Any

import numpy as np
import numpy.typing as npt

## Joints per arm; the gripper is the seventh entry of every per-arm vector here.
ARM_JOINTS: int = 6
## Width of one arm's slot: six joints and the normalised gripper.
ARM_DOF: int = 7
## Look-ahead reference frames in the observation, and their spacing in seconds (SONIC: 10 frames, 0.1 s apart).
FUTURE_FRAMES: int = 10
FUTURE_DT_S: float = 0.1
## Joint velocities enter the network scaled by this (rad/s to order one).
VEL_SCALE: float = 0.1
## The normalised observation is clipped to this many standard deviations.
OBS_CLIP: float = 10.0
## Width of the observation: q, q-dot, the previous action, then the reference now and the look-ahead as errors.
OBS_DIM: int = ARM_DOF * 3 + (FUTURE_FRAMES + 1) * ARM_DOF
## The weights the station runs: the YAM tracker, in the form the module docstring gives.
WEIGHTS_PATH: pathlib.Path = pathlib.Path(__file__).with_name("tracker.npz")


@dataclasses.dataclass(frozen=True)
class Policy:
    """A frozen tracker's arrays: normaliser, dense layers, and the training metadata.

    Attributes:
    - layers: `(weight, bias)` per dense layer, float32, weight `[out, in]`; ELU between them, none after the last.
    - obs_mean: Mean the observation is normalised with, `[OBS_DIM]`.
    - obs_std: Standard deviation it is normalised with, `[OBS_DIM]`.
    - meta: What the file records: `actuator`, `iteration`, `sha256` and the training settings.

    @see load(), act()
    """

    layers: tuple[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]], ...]
    obs_mean: npt.NDArray[np.float64]
    obs_std: npt.NDArray[np.float64]
    meta: dict[str, Any]


def act(net: Policy, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The network's deterministic action for one observation.

    @param net The tracker.
    @param obs The observation, `[OBS_DIM]`, from observation().
    @return The action, `[ARM_DOF]`, clipped to `[-1, 1]`.
    """
    # float32 first, as training packed it: the normaliser then sees the same values training did.
    x = np.clip((obs.astype(np.float32) - net.obs_mean) / net.obs_std, -OBS_CLIP, OBS_CLIP).astype(np.float32)
    for index, (weight, bias) in enumerate(net.layers):
        x = x @ weight.T + bias
        if index < len(net.layers) - 1:
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
    return np.clip(x.astype(np.float64), -1.0, 1.0)


def load(path: pathlib.Path = WEIGHTS_PATH) -> Policy:
    """Reads a tracker file.

    @param path A tracker in the form the module docstring gives; the bundled one by default.
    @return The policy.
    @throws ValueError The file's observation width is not #OBS_DIM, or its action not #ARM_DOF wide.
    """
    with np.load(path) as data:
        count = sum(1 for key in data.files if key.startswith("w"))
        layers = tuple((data[f"w{i}"].astype(np.float32), data[f"b{i}"].astype(np.float32)) for i in range(count))
        policy = Policy(
            layers=layers,
            obs_mean=data["obs_mean"].astype(np.float64),
            obs_std=data["obs_std"].astype(np.float64),
            meta=json.loads(str(data["meta"])),
        )
    if policy.layers[0][0].shape[1] != OBS_DIM or policy.layers[-1][0].shape[0] != ARM_DOF:
        raise ValueError(
            f"{path}: a {policy.layers[0][0].shape[1]}-input, {policy.layers[-1][0].shape[0]}-output network; "
            f"the station's tracker is {OBS_DIM} in, {ARM_DOF} out"
        )
    return policy


def observation(
    q: npt.NDArray[np.float64],
    qd: npt.NDArray[np.float64],
    a_prev: npt.NDArray[np.float64],
    rows: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Packs one arm's observation.

    @param q Measured joints and gripper, `[ARM_DOF]`, the gripper in `[0, 1]`.
    @param qd Measured joint velocities, `[ARM_DOF]` (the gripper's is 0).
    @param a_prev The previous action, `[ARM_DOF]`.
    @param rows The reference now, then every #FUTURE_DT_S ahead, `[FUTURE_FRAMES + 1, ARM_DOF]`.
    @return The observation, `[OBS_DIM]`.
    @throws ValueError `rows` is not `[FUTURE_FRAMES + 1, ARM_DOF]`.
    """
    if rows.shape != (FUTURE_FRAMES + 1, ARM_DOF):
        raise ValueError(f"reference rows must be {(FUTURE_FRAMES + 1, ARM_DOF)}, got {rows.shape}")
    return np.concatenate([q, qd * VEL_SCALE, a_prev, (rows - q[None, :]).ravel()])
