"""System 0: a learned tracker between the policy's plan and the arms' motors, a YAM station's default.

Without it the station turns each 20 Hz row of the plan into a position target for the motors' PD (the command
writer, `robots.control`), and the arm reaches each row about two hundred milliseconds after it was asked to. System
0 is a motor-level tracker:
a small network that sees
the measured joints and the next second of the plan, and sets each joint's torque every 4 ms so the measured arm
follows the plan. On a YAM station the follower-side trackers take the arm's lag behind the plan from
about 200 ms on the PD path to 60 to 90 ms.

A YAM's station agent runs it unless started without it, `dreamcli robot up nosys0` (`robots.station --no-sys0`);
every other body runs without it. Nothing else about it is configurable: the weights ship in the package
(`policy`, which also says how to drop in a tracker of one's own), the shaping and the safety limits are fixed.

- `policy`: the network, evaluated with numpy from the bundled `tracker.npz`, and its observation.
- `tracker`: one arm's tick: the plan, shaped as the YAM follower shapes it, in; a supervised motor command out.
- `motors`: each arm's 250 Hz tracker tick, and the path that hands the motors between it and the command writer.
- `arm_process`: each arm in a process of its own, with its tracker, so the 4 ms tick never waits on the agent.
- `fast_loop`: i2rt's motor loops sped up, the transport the network expects.

@see robots.sys0.arm_process, robots.sys0.motors.MotorPath
"""
