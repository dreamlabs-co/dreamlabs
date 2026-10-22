# robots {#mainpage}

The public inference control loop: a fixed-rate loop that reads two follower arms (a YAM, a MakerMods Makers
Arm or Metal arm, an SO-101, or a robot of your own behind one small driver) and the cameras, drains action chunks from a `dream` policy client, and commands the arms. Nothing here
trains, records or teleoperates.

| Module | Job |
|---|---|
| `robots.control` | `ControlLoop`: tick clock, chunk buffer, prefetch, the one `get_action` in flight, the end-effector speed clamp, `state` / `video` / `plan` snapshots |
| `robots.kinematics` | `ArmKinematics`: forward kinematics of one YAM arm on the i2rt MuJoCo model (`grasp_site`), what the speed clamp and the plan's speeds are measured with; the only module that knows mujoco. `DreamKinematics`: the same on dream's own model of a bus arm, one side each; `arm_layout()`: the joints an arm drives, from dream's spec |
| `robots.followers` | `FollowerArm` over a `Driver`: the one arm interface every robot implements, in its URDF units (read, write, close, torque off); the wrapper adds the command checks, hold and disengage |
| `robots.yam` | `I2rtDriver`: the i2rt YAM behind `Driver`, `open_arm()` (the only module that knows i2rt) |
| `robots.motor_arm` | `MotorArm`: the one thread every bus arm runs, CAN or serial (rate, startup ramp, speed cap, stale target, faults, the park before torque off) over its robot's table (`ArmSpec`, `Joint`) and its bus's chain |
| `robots.bodies` | `Body`, `BODIES`, `for_config()`: what differs between the robots (wire, default model, the client's IK, driver, kinematics), picked by the config's `body`; `so101_body()` builds an SO-101 on its config's embodiment, `custom_body()` builds a robot of your own from its config |
| `robots.driver_template` | `TemplateArm`: the commented starting point for a robot of your own (`body: custom`), an arm in memory as shipped |
| `robots.makers` | The Makers Arm's table (`ARM`): its seven RobStride motors, 60 Hz, its linear gripper stroke, dream's `makers` IK |
| `robots.robstride` | `RobstrideBus`: the RobStride MIT codec and request/reply bus on one CAN adapter (the only module that knows python-can); `RobstrideChain`, a RobStride arm for `MotorArm` (full-turn wrap, stale zero, gripper hold, silence watch), a port of Dream Labs' Makers driver |
| `robots.makers_hold` | `GripperHold`: the Makers gripper's holding torque and closing cap; `HoldingController`, the holding law the Metal gripper runs too |
| `robots.metal` | The Metal arm's table (`ARM`): its seven Damiao motors, 30 Hz, the Metal grip, dream's `metal` IK |
| `robots.metal_gripper` | `MetalGripper`: the Metal gripper's setup, holding control and safety stops (watchdog, temperature, fault latch) |
| `robots.damiao` | `DamiaoBus`: the Damiao MIT codec and request/reply bus on one CAN adapter; `DamiaoChain`, a Damiao arm for `MotorArm`, a port of MakerMods' own Metal follower |
| `robots.so101` | The SO-101's table (`ARM`): its six Feetech servos (five joints and the gripper), 60 Hz, LeRobot's 0 to 100 gripper, dream's `so101_faas` IK |
| `robots.feetech` | `FeetechChain`: an arm's Feetech servos on LeRobot's own bus for `MotorArm`, refused when the servos hold another calibration than the arm's file (the only module that knows LeRobot) |
| `robots.cameras` | `CameraReader`: one capture thread per camera publishing into a latest-frame slot, over a cv2 device or a `RealSenseSource` (the only module that knows cv2 and pyrealsense2) |
| `robots.adapter` | The seam with `dream`: `Observation` to flat dicts (`YAM_WIRE`, and `EefWire`, every fine-tune's wire built from its embodiment config), action dicts to `ActionChunk`, and the joints last commanded for the client; the only module that knows dream's key names |
| `robots.types` | The seven frozen dataclasses: `Frame`, `ArmState`, `RobotState`, `Observation`, `ArmCommand`, `Action`, `ActionChunk` |
| `robots.config` | `AppConfig` and `load()`: the one YAML file that builds everything, checked key by key |
| `robots.run` | `main()`: builds cameras, the dream client and arms from the config and runs the loop once from home, then homes the arms and switches them off (`dreamcli robot run`); `DryRun` for a run that writes no target |
| `robots.station` | `StationController`: owns the arms for the process, one run at a time, Start / Stop / Home / prompt; `python -m robots.station` |
| `robots.station_api` | The HTTP control API over the controller (stdlib server, JSON) |
| `robots.home` | `return_to_home()` (`move_home()`: which arm did not make it, and why): both arms to each arm's configured home with the grippers open; `python -m robots.home` |
| `robots.inference_log` | The JSON Lines inference log the loop writes: one record per inference, error, stall and run boundary |
| `robots.errors` | `RobotsError` and its five subclasses |

**Start here**

- [Package list](namespaces.html), [Class list](annotated.html), [File list](files.html).
- `robots.control.ControlLoop.step()` is the recipe; its call graph shows every seam.
- Pages: the README, the changelog and the setup wizard's guide (`docs/SETUP_WIZARD.md`).
