"""The bodies a station can drive, one small description each: what differs between a YAM, a Makers Arm, a Metal arm
and an SO-101.

One loop drives them all (#robots.control.ControlLoop): the config's `body` key picks a #Body here (for_config()),
and the body says what the loop and the station need that is not the same for every arm: the wire it speaks (its
camera roles, the serve observation and the embodiment block, #robots.adapter.Wire), the model it runs unless a run
names another, the dream client's IK (`dream.client_ik`), the driver that opens one arm, and the forward kinematics
the end-effector speed cap reads.

- #YAM: the i2rt YAM, the default. It speaks its model's own wire
  (#robots.adapter.YAM_VIDEO_WIRE), the `state` and `video` keys the model reads; the station wire
  (`state.arm_r.joint_pos`, `image_window.top` and the rest) is deprecated and no body speaks it.
- #MAKERS: the MakerMods Makers Arm (#robots.makers), on a FaaS fine-tune that answers with tool poses: the client
  solves them to joints with dream's IK.
- #METAL: the MakerMods Metal arm (#robots.metal), the same way: the Makers datasets' layout with the Metal grip, and
  dream's IK of the Metal arm.
- so101_body(): an SO-101 (#robots.so101), the same way again, on the embodiment its fine-tune was trained on (the
  config's `embodiment`, looked up on the account by name).

A bus arm's body is built from its table (#robots.motor_arm.ArmSpec) and its datasets' embodiment config
(#robots.adapter.EefWire); nothing in it is per robot. A robot of your own (`body: custom`, custom_body()) is built the
same way from the config: your driver (#robots.driver_template), dream's IK of your arm, and the embodiment your
fine-tune was trained on, looked up on your account by name.

The functions look their driver, adapter opener and kinematics up when called, not when this module is imported, so a
test that replaces `yam.open_arm`, `robstride.open_can_bus` or `ArmKinematics.from_config` replaces them here too.

A fine-tune runs only on the embodiment it was trained on: check_model() refuses one of the account's other
embodiments before a run is built, and runs the model's own joints (`action_space: joint`) for a fine-tune of the
robot's own embodiment only. The bodies' `ik` names the head `eef`; `robots.run.policy_config` sets the run's.

@see Body, BODIES, for_config(), so101_body(), custom_body(), check_model()
"""

import dataclasses
import functools
from collections.abc import Callable, Mapping
from typing import Any

import dream

from robots import (
    adapter,
    config,
    errors,
    feetech,
    followers,
    kinematics,
    makers,
    metal,
    motor_arm,
    robstride,
    so101,
    user_class,
    yam,
)

## The model a YAM station runs unless a run names another (`robots.station.StationController.start`): a catalog slug,
## which the gateway resolves to its checkpoint, the base model with an EEF head, the YAM's default head
## (`robots.run.policy_config`). A checkpoint id the key is not entitled to is refused (`DA-1000 not_entitled`), so
## the slug is named, never an id. The model is dream's to choose, so the config file does not carry it.
YAM_CHECKPOINT = "dreamlabs/dream-1-beta"


@dataclasses.dataclass(frozen=True)
class Body:
    """What one kind of arm pair needs from the station that the loop does not decide.

    Attributes:
    - name: The config's `body` value.
    - checkpoint: The model a run uses unless it names another; None when there is no default (a Makers or Metal
      arm runs the customer's own fine-tune, which a run must name).
    - ik: The dream client's `ik` config key (`dream.client_ik`), or None for a model that answers in joints.
    - wire: Returns the body's #robots.adapter.Wire: its camera roles, the serve observation and the `embodiment`
      block.
    - open_arm: Opens one follower arm from its config and its name (`"left"` or `"right"`).
    - kinematics: Builds the forward kinematics of both arms, keyed `"left"` and `"right"`, from the config.
    - open_arm_read_only: Opens one follower arm with no motor switched on, its pose read on, for a dry run; None when
      the driver cannot (i2rt opens a YAM powered; a robot of your own's driver decides).
    """

    name: str
    checkpoint: str | None
    ik: dict[str, str] | None
    wire: Callable[[], adapter.Wire]
    open_arm: Callable[[config.ArmConfig, str], followers.FollowerArm]
    kinematics: Callable[[config.AppConfig], dict[str, kinematics.Kinematics]]
    open_arm_read_only: Callable[[config.ArmConfig, str], followers.FollowerArm] | None = None


def _yam_open_arm(cfg: config.ArmConfig, name: str) -> followers.FollowerArm:
    """#robots.yam.open_arm, looked up per call."""
    return yam.open_arm(cfg, name)


def _yam_kinematics(app: config.AppConfig) -> dict[str, kinematics.Kinematics]:
    """One i2rt MuJoCo model per arm, from each arm's gripper type (#robots.kinematics.ArmKinematics)."""
    return {
        "left": kinematics.ArmKinematics.from_config(app.follower_l),
        "right": kinematics.ArmKinematics.from_config(app.follower_r),
    }


def _dream_kinematics(ik: str) -> Callable[[config.AppConfig], dict[str, kinematics.Kinematics]]:
    """dream's own FK of the arm named `ik`, one model for both arms, each arm on its own side of it."""

    def build(app: config.AppConfig) -> dict[str, kinematics.Kinematics]:
        return dict(kinematics.DreamKinematics.both_arms(ik, app.control_hz))

    return build


def _bus_body(
    spec: motor_arm.ArmSpec,
    embodiment: Callable[[], Mapping[str, Any]],
    opener: Callable[[], Callable[..., Any]],
) -> Body:
    """An arm on a motor bus on a FaaS EEF fine-tune: the shared thread on the arm's table, on the bus the config
    names, and the wire of the embodiment its datasets record.

    The client solves the model's tool poses with dream's IK of the arm (`spec.ik`), whose spec also says how the
    arm's joints and gripper go out (#robots.kinematics.arm_layout). Both are read when the wire is first asked for,
    so opening or homing the arms needs neither dream's IK nor the network.

    @param spec The arm's table.
    @param embodiment Returns the embodiment config.
    @param opener Returns the function that opens the arm's bus.
    """

    @functools.cache
    def wire() -> adapter.Wire:
        return adapter.EefWire(embodiment(), spec.grip_scale_mm, spec.name, kinematics.arm_layout(spec.ik))

    def open_arm(cfg: config.ArmConfig, name: str) -> followers.FollowerArm:
        return motor_arm.open_arm(spec, cfg, name, opener=opener())

    def open_arm_read_only(cfg: config.ArmConfig, name: str) -> followers.FollowerArm:
        return motor_arm.open_arm(spec, cfg, name, opener=opener(), energize=False)

    return Body(
        name=spec.name,
        checkpoint=None,
        ik={"arm": spec.ik, "head": "eef"},
        wire=wire,
        open_arm=open_arm,
        kinematics=_dream_kinematics(spec.ik),
        open_arm_read_only=open_arm_read_only,
    )


## The i2rt YAM.
YAM = Body(
    name="yam",
    checkpoint=YAM_CHECKPOINT,
    ik=None,
    wire=lambda: adapter.YAM_VIDEO_WIRE,
    open_arm=_yam_open_arm,
    kinematics=_yam_kinematics,
)

## The MakerMods Makers Arm (#robots.makers.ARM).
MAKERS = _bus_body(makers.ARM, lambda: adapter.MAKERMODS_EMBODIMENT, lambda: robstride.open_can_bus)

## The MakerMods Metal arm (#robots.metal.ARM): the Makers datasets' layout, with the Metal grip.
METAL = _bus_body(metal.ARM, lambda: adapter.MAKERMODS_EMBODIMENT, lambda: robstride.open_can_bus)

## Every body by its config name (#robots.config.BODY_NAMES) but the SO-101 (so101_body()), whose embodiment is the
## config's.
BODIES: dict[str, Body] = {body.name: body for body in (YAM, MAKERS, METAL)}


def for_config(app: config.AppConfig) -> Body:
    """The body the config names: one of #BODIES, an SO-101 (so101_body()) or a robot of your own (custom_body()).

    @param app The parsed application config.
    @return Its #Body.
    """
    if app.custom is not None:
        return custom_body(app.custom)
    if app.embodiment is not None:
        return so101_body(app.embodiment)
    return BODIES[app.body]


@functools.cache
def so101_body(embodiment: str) -> Body:
    """An SO-101 (#robots.so101.ARM) on the embodiment its fine-tune was trained on.

    @param embodiment The config's `embodiment`, `<username>/<name>`, looked up on the account when the wire is first
        asked for (embodiment_config()).
    @return The #Body, named `so101`.
    """
    return _bus_body(so101.ARM, lambda: embodiment_config(embodiment, "embodiment"), lambda: feetech.open_bus)


@functools.cache
def custom_body(custom: config.CustomConfig) -> Body:
    """A robot of your own: your driver, dream's IK of your arm, and the wire of your fine-tune's embodiment.

    Built once per config: the embodiment is looked up on the account (embodiment_config()) the first time the wire
    is asked for, so opening or homing the arms needs no network.

    @param custom The config's `custom` section.
    @return The #Body, named #robots.config.CUSTOM_BODY.
    """

    @functools.cache
    def wire() -> adapter.Wire:
        return adapter.EefWire(
            embodiment_config(custom.embodiment, "custom.embodiment"),
            custom.grip_scale,
            custom.embodiment,
            kinematics.arm_layout(custom.ik),
        )

    def open_arm(cfg: config.ArmConfig, name: str) -> followers.FollowerArm:
        driver = load_driver(custom.driver)(cfg, name)
        if not isinstance(driver, followers.Driver):
            raise errors.ConfigError(
                f"your driver {custom.driver} opened {type(driver).__name__}, not a robots.followers.Driver; "
                "start from robots.driver_template"
            )
        return followers.FollowerArm(name, driver, home_rad=cfg.home)

    return Body(
        name=config.CUSTOM_BODY,
        checkpoint=None,
        ik={"arm": custom.ik, "head": "eef"},
        wire=wire,
        open_arm=open_arm,
        kinematics=_dream_kinematics(custom.ik),
    )


@functools.cache
def load_driver(name: str) -> Callable[[config.ArmConfig, str], Any]:
    """Your driver's class, from #robots.config.CustomConfig's `driver` (#robots.user_class.load).

    Read once per process; the setup wizard clears the cache each time it opens the arms, so a `file.py` edited since
    is read again. A `package.module` stays as Python first imported it.

    @param name `package.module:Class`, or `/path/to/file.py:Class`.
    @return The class (any callable taking `(arm_config, side)`).
    @throws ConfigError The module or file does not load, or has no such name.
    """
    driver: Callable[[config.ArmConfig, str], Any] = user_class.load(name, "driver")
    return driver


def embodiment_config(name: str, key: str) -> dict[str, Any]:
    """The config of one of the account's embodiments, by name (`dream.DreamClient.get_embodiments`).

    @param name `<username>/<name>`.
    @param key The config key that names it, for the message.
    @return Its `embodiment_config`.
    @throws ConfigError The account has no embodiment of that name.
    @throws DreamError The data service could not be asked (no key, refused, unreachable).
    """
    found = {body.name: body.embodiment_config for body in dream.DreamClient().get_embodiments()}
    if name not in found:
        raise errors.ConfigError(
            f"config key '{key}' names {name!r}, which this account does not have; its embodiments are "
            f"{sorted(found)}: name the one your fine-tune was trained on"
        )
    return found[name]


def check_model(app: config.AppConfig, checkpoint: str) -> None:
    """Refuses a model this robot must not run: another embodiment's fine-tune, or for joints any but its own.

    That is a fine-tune of another of the account's embodiments on a robot whose config names its embodiment, and,
    with `action_space: joint`, anything but one of the account's fine-tunes of that embodiment.

    An SO-101 (`embodiment`) or a robot of your own (`custom.embodiment`) sends the parts and cameras of the
    embodiment its config names, and nothing else would notice a fine-tune trained on another one: it would drive
    this arm through its IK with another robot's poses. The model's record (`dream.DreamClient.get_models`) names the
    id of the embodiment it was trained on, one of `get_embodiments`. A Dream Labs base model (trained on YAM arms,
    which the dream client runs on every arm its IK models) and a fine-tune shared from another account (an
    embodiment this account does not have) pass, as the setup wizard offers them; a checkpoint the account cannot run
    is the dream client's to refuse. Every other body names no embodiment and passes without a call, and a YAM passes
    on either action space: its models answer in its own joints.

    The model's own joints drive the motors as they come, so `joint` runs only one of the account's fine-tunes of the
    embodiment the config names: a base model answers in YAM joints, and a shared fine-tune's embodiment cannot be
    matched to this robot's. A body that names no embodiment (a Makers or Metal arm) runs `eef` only.

    @param app The parsed application config.
    @param checkpoint The model the run asks.
    @throws ConfigError The checkpoint is a fine-tune of another of the account's embodiments, or `action_space` is
        `joint` and it is not one of the account's fine-tunes of the config's embodiment (or the config names none);
        the message names what to pick.
    @throws DreamError The platform or the data service could not be asked (no key, refused, unreachable).
    """
    if app.body == YAM.name and app.custom is None and app.embodiment is None:
        # Every YAM model answers in the YAM's own joints, so either head runs it (robots.run.policy_config).
        return
    named, key = (app.embodiment, "embodiment") if app.custom is None else (app.custom.embodiment, "custom.embodiment")
    joints = app.action_space == "joint"
    if named is None:
        if joints:
            raise errors.ConfigError(
                f"config key 'action_space' is 'joint', which runs a fine-tune of this robot only, and a {app.body} "
                f"config names no embodiment to match {checkpoint!r} to: set 'action_space' to 'eef'"
            )
        return
    client = dream.DreamClient()
    names = {body.id: body.name for body in client.get_embodiments()}
    models = client.get_models()
    model = next((m for m in models if m.id == checkpoint), None)
    trained = None if model is None else names.get(model.embodiment)
    if trained == named or (trained is None and not joints):
        return
    fits = [m.id for m in models if names.get(m.embodiment) == named]
    if trained is None:
        pick = (
            f"pick one of your fine-tunes of {named!r} ({', '.join(fits)})"
            if fits
            else f"fine-tune one of {named!r} first: your account has none yet"
        )
        raise errors.ConfigError(
            f"config key 'action_space' is 'joint', but {checkpoint!r} is not one of this account's fine-tunes of "
            f"{named!r}: its joints are another robot's (a base model's are a YAM's), and they would drive this robot "
            f"as they come; set 'action_space' to 'eef' (Tool pose in the Live view), or {pick}"
        )
    pick = (
        f"pick one of your fine-tunes of {named!r} ({', '.join(fits)}) or a Dream Labs base model"
        if fits
        else f"your account has no fine-tune of {named!r} yet: pick a Dream Labs base model, or fine-tune one"
    )
    raise errors.ConfigError(
        f"{checkpoint!r} was trained on {trained!r}, but config key '{key}' names {named!r}: it would drive this "
        f"robot with another robot's poses; {pick}"
    )
