# Robot examples: start your own robot from one

Each folder here is a working driver and config for a robot that runs Dream-1 on our control loop. To run a robot of your own, start from the closest one, by hand or with Claude Code: point Claude at this folder with the prompt below.

## The control loop

A robot of your own is a config with `body: custom`: the dream library's own control loop runs it (`dreamcli robot run`, `dreamcli robot up` and Live view, the setup wizard), and you write only the driver, one Python class for one arm.

- The driver is a `robots.followers.Driver`: `read()` returns the joints (radians, in the order of your arm's URDF) and the gripper (0 closed to 1 open); `write()` takes a target in the same units; `close()` lets the arm go without moving it; `torque_off()` switches its motors off. `disengage()` (limp for the setup wizard's steps by hand) and `drops_when_limp` (an arm without brakes) are optional.
- The config's `custom` section names the driver (`my_robot.py:Arm`, a file beside the config), `ik` (dream's model of your arm) and `embodiment` (the one your fine-tune was trained on, `<username>/<name>`).
- The dream client turns the model's tool poses into your arm's joints with that model (IK, on your computer), so the driver only ever sees joints.
- The loop does everything else: the model calls, the timing, the speed cap, the home move, the cameras and the rollouts.
- The driver template that comes with the library, `robots/driver_template.py`, is the commented starting point: `python -c "import robots.driver_template as t; print(t.__file__)"` prints where it is.

## The examples

| Folder | Robot | Its SDK | What it shows |
| --- | --- | --- | --- |
| `seeed_rebot/` | Seeed reBot B601-DM: two 6-joint arms on Damiao motors, one serial bridge each | Seeed's LeRobot plugin, `lerobot-robot-seeed-b601` | Motor degrees to radians and back, a jaw mapped through its measured travel, a refused reading past a joint limit, an arm without brakes, and an arm opened on the plugin's lower-level bus, because its `connect()` and `disconnect()` can move the arm |
| `trossen/` | Trossen Stationary AI: two WXAI follower arms over Ethernet | Trossen's `trossen-arm` | An SDK already in radians, the jaw's opening in metres, and a limp arm held up by gravity compensation; not yet run on an arm |

Each folder holds the driver, `<robot>.py`, and its config, `<robot>.yaml`. Each driver has two classes: `Arm`, which switches the motors on holding where the arm is, and `ReadOnlyArm`, which opens the arm, switches no motor on and refuses `write()`, for the dry run. The configs leave each arm's `home` and the `embodiment` for you to fill in, and refuse to load until you do.

## Start your own robot

1. Fetch these files with your key: the Setup robots page of the platform and the docs page "Your own robot" give the command, which saves them in `~/my_robot/examples`.
2. Copy the closest example's two files to `~/my_robot/my_robot.py` and `~/my_robot/my_robot.yaml`, and replace its SDK calls with your robot's. Keep a `ReadOnlyArm`.
3. In the config, set `custom.driver` to `my_robot.py:ReadOnlyArm`, `custom.ik`, `custom.embodiment`, each arm's `can_channel` (the device your driver opens) and your cameras. Set each `home` to one 0 per joint of the arm for now: a dry run never moves an arm, so it never goes there.
4. With the arms powered, dry run: `dreamcli robot run --config ~/my_robot/my_robot.yaml --checkpoint <your fine-tune> --task "..." --dry-run`. The model is called for real and nothing is written to the arms; the log shows what each arm reads once a second. Copy each arm's reading at rest into its `home`.
5. Then set `custom.driver` to `my_robot.py:Arm` and run without `--dry-run`, with a hand on the emergency stop. A run starts by moving both arms to `home`.

## The prompt for Claude Code

Start Claude Code in `~/my_robot`, fill in the second line, and paste:

```text
Write a dream driver and config for my robot, starting from the examples in ~/my_robot/examples.
My robot: <maker and model; how many arms and joints; its motors and how they connect; its SDK or LeRobot plugin; its cameras>

First read ~/my_robot/examples/README.md, the example closest to my robot (its .py and .yaml), the driver template that comes with the dream library (python -c "import robots.driver_template as t; print(t.__file__)"), and, if the dreamlabs-docs MCP server is connected, its page "Your own robot".
Then write ~/my_robot/my_robot.py, with an Arm and a ReadOnlyArm as the example has, and ~/my_robot/my_robot.yaml from the example's config, with custom.driver my_robot.py:ReadOnlyArm and each home one 0 per joint of the arm for now.
Then show me ReadOnlyArm and the dry-run command, and run the dry run only when I say so: dreamcli robot run --config ~/my_robot/my_robot.yaml --checkpoint <my fine-tune> --task "..." --dry-run

Rules:
- Nothing moves without me. Never run Arm, a run without --dry-run, the setup wizard, or anything else that switches a motor on or sends a target: those are mine to run, with a hand on the emergency stop.
- The dry run comes first, on ReadOnlyArm: it opens the arm, switches no motor on and refuses write().
- The tool center point is mine to measure. If my arm needs an arm file for custom.ik, write the rest with its tcp, approach and jaw left empty, then ask me to measure and confirm the tool center point with dreamcli tcp (or by hand), and continue only once I have confirmed it. Never fill in tcp, approach or jaw, or pick or estimate the point yourself: a link in my URDF and the YAM's numbers are not my tool center point.
- close() and every disconnect path never move the arm.
- Before calling a vendor plugin's connect() or disconnect(), read them in its source: some home, rezero, close the gripper or move the whole arm. If either can move anything, open the plugin's lower-level bus instead, as the reBot example does.
```

With the dreamlabs-docs MCP server connected (the Setup API page of the platform gives the command), you can also just ask Claude Code for the reBot example: the docs page it finds there says how to fetch these files and what to do with them.
