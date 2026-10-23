# robots

Inference control loop for a bimanual YAM follower setup: observe the arms and cameras at a fixed
rate, ask a Dream-1 policy for action chunks through `dream` and DreamAPI, command the arms. The
same loop drives a MakerMods Maker Arm or Metal Arm or an SO-101 on a fine-tune that answers with tool poses
(*Maker Arm*, *Metal Arm* and *SO-101* below), and a robot of your own through one small driver class (*A robot of
your own*).

`robots` is the inference stack and nothing more. It does not teleoperate, collect data, train or
serve models; it can save what each run did on the operator's disk (*Your rollouts*, off by default). It owns four things: two follower arms, one capture thread per camera, a
fixed-rate `ControlLoop` that drains action chunks, and the seam to `dream`, which talks to
DreamAPI, which forwards to the serve that owns the model. `dream`'s own README
is `../dream_client/README.md`.

## Install

A station installs the dream library, which carries this package (with `dreamcli`), the dream client, the YAM arm
driver and the RealSense wheel. The platform's Setup API page gives one block that does it all: a Python 3.12
environment at `~/dream-env`, that site's build of the library, and `~/.dreamlabs/credentials`. By hand:

```bash
pip install dreamlabs                           # from PyPI: this package, dreamcli, the client (dreamlabs-client) and the driver
pip install ./dreamlabs_client-*.whl ./dreamlabs-*.whl   # or the two files the Setup API page downloads (/downloads/latest-client, /downloads/latest)
pip install "dreamlabs[so101]"                  # then, on an SO-101 only: LeRobot's Feetech bus (`body: so101`, Python 3.12 or newer), in the same environment
```

`pip install dream` from PyPI is another project, not this library. `dream-robots`, this directory on its own, is not
on any package index: it is the development install inside this repository (`uv sync` here).

From a checkout, `pip install ./dream_library/dream_client ./dream_library` installs all of it as the `dreamlabs` distribution and the `dreamlabs-client` it depends on, i2rt and the RealSense wheel included; that one list is everything a station runs, and the `hardware`, `ui` and `live` extras are empty aliases, kept so older instructions still resolve (`dream_library/README.md`).

Import name is `robots`. i2rt is not on PyPI; the `hardware` extra pins it from git
(`i2rt-robotics/i2rt` at `4e462ecc`) together with `mujoco`, and only `robots.yam.open_arm`
and `ArmKinematics.from_config` import them, lazily, so the core and the test suite run without
them (one kinematics test needs both and is skipped otherwise). Inside this repo the package installs the client, `dreamlabs-client`, from
the sibling `../dream_client` directory through a uv path source.

Stations with Intel RealSense cameras (a YAM station: three D405s) need librealsense's Python
wheel, `pyrealsense2`. The `dreamlabs` distribution carries it where it exists (Linux, Python 3.12
or older; its wheels stop there), and this package does not, because it ships prebuilt for only
some platforms; `RealSenseSource` imports it lazily either way, with a `CameraError` naming the
package when it is missing.

## Command line

Installing the package puts `dreamcli` on the path: eight commands and one station directory, `~/.dreamlabs`
(or `$DREAMCLI_HOME`), holding the config, the agent's log and its pid file. This is how a station is set up and
run; the `python -m` entry points under *Run* are what it drives.

```bash
dreamcli setup                      # the setup wizard, a web page on port 8092, opened from the private address it prints; writes ~/.dreamlabs/station.yaml
dreamcli tcp                        # the setup page's tool point tool alone, on port 8094: your arm's URDF in, arms/<name>.json for custom.ik out
dreamcli robot up                   # brings the arms' CAN links up, then starts the agent; nothing moves until a Start; a YAM runs System 0 (*System 0*)
dreamcli robot up nosys0            # the same, without System 0: the command thread drives the motors as on every other body
dreamcli robot up noeef             # a YAM on its model's joint head instead of its EEF head, the default
dreamcli robot up --ik-solve tick   # an eef run's IK row by row, this run only (every body's default is traj: one solve of the whole chunk with the arm's last 3 commands)
dreamcli robot down                 # stops it: SIGINT (a running loop is stopped, the arms switch off and go limp), then SIGTERM, then SIGKILL; says whether the arms are off
dreamcli robot log [-n 40]          # the last 40 lines of ~/.dreamlabs/station.log
dreamcli robot run --task "..."     # one task in this terminal, no agent; Ctrl-C homes the arms and switches them off
dreamcli motors check               # every motor's firmware control mode on both arms; exit 0 only when all are MIT
dreamcli motors set-mit             # switches any motor that is not MIT, stores it; then power-cycle the arms and check
```

`dreamcli motors` (`robots.motor_mode`) is for an arm whose motors are not in MIT mode, the one firmware control mode i2rt drives: new OpenYAM arms often ship with the gripper motor (id 7) in position-speed mode, which answers every status request and moves for none, so the setup wizard reports a gripper that did not move. `check` reads the `CTRL_MODE` register of every motor on both links, the two the station config names (`can_follower_l` and `can_follower_r` before there is one; `--link` names one link, `--motor` one id, both repeatable), one parameter read per motor that switches nothing on, and exits 1 naming any motor that is not MIT or did not answer; `set-mit` switches each motor off, writes MIT where the read says otherwise, and sends the store-to-flash command, and the arm runs the new mode only after a power cycle (motor power off, a few seconds, on), which is why `check` is run again after it. Both bring the links up first, as `robot up` does, and refuse while another program holds the arms (*Run*), and on a station whose config names another body (a Maker Arm, a Metal Arm, an SO-101, your own), whose motors have no such mode to switch.

`dreamcli setup` (`robots.setup_wizard`) serves one page on the stdlib server and opens it in a browser when the
box has a display; on a headless station it prints the address to open from a laptop (`--host`, `--port` and
`--no-browser` change that). The printed address carries a token, fresh on each run
(`?token=`, kept in a cookie after the first visit); a request without it gets 403. Set `DREAM_SETUP_TOKEN` once where the command runs (in `~/.bashrc`, or the shell that runs `dreamcli setup`) and that is the token instead, so the address stays the same from run to run and can be bookmarked; `docs/SETUP_WIZARD.md`, *Opening the page*, says how to make one. Anyone with the address can power the arms, so keep it private.
Over ssh the command prints the
box's network name and interface addresses to open from your own computer, then the `ssh -N -L` line and the
`127.0.0.1` address behind it for when none of them opens. It sets up any robot dream has a URDF of (a YAM, a Maker Arm, a
Metal Arm, an SO-101 with its LeRobot calibration file, or a robot with a driver its owner wrote) and checks that the driver reads and writes joints the way dream
does: radians, the URDF's zero, direction and order, and the gripper as 0 closed to 1 open. The steps, in order: pick
the robot; find each arm by moving it by hand (every adapter that fits the robot's bus is offered, named by its serial
number, and the rest are listed apart as not arms; each ticked adapter is opened and let go; a YAM's page gives a udev
rule naming each arm's CAN link after that serial, and a CANable needs none); give each camera its role and compare it with a
reference view (the operator's own LeRobot v2.1 dataset, or our YAM pictures) and with what the model sees; set the
home pose by putting each arm where every run starts and pressing OK, with the arms drawn in 3D from their readings
; on a Maker or Metal Arm, or a robot of your own whose driver reports its gripper motor's angle, measure each
gripper's range by hand on the same step (open both fully and press Record open, close both and press Record closed); then, after a safety popup,
small moves under power (a 5 degree nudge of
each joint and the gripper, a 3 cm and 10 degree hand sweep judged by forward kinematics) with Stop on screen
throughout, while the page draws each arm in 3D from its readings, lighting up the joint being nudged, and at the end
one question: did each joint and the hand move like the drawing? Nothing moves under power before that popup. The page loads three.js from a CDN, so setup needs
internet.
`docs/SETUP_WIZARD.md` has each step, each check, the causes it names, the mistakes the tests plant, and what it does
not catch.

The task prompt is not on the page and not in the file it writes: the operator sends it with the Start, from the
Playground's live view or the API. The account is not on the page either: the key, where DreamAPI is and where the
platform is are the credentials file, `~/.dreamlabs/credentials` (*Credentials* below), which `dreamcli setup` reads
before it serves and shows on the page; without a key it stops and says where to write one. On a YAM the model is not
on the page and not in the file: the station runs `robots.run.DEFAULT_CHECKPOINT` (`dreamlabs/dream-1-beta`, the
Dream-1 base model with an EEF head, a YAM's default head) unless a run names another (`start` on the station API, or the live view's
prompt). Every other robot has no default model, so the first step asks for one: the account's fine-tunes on that
robot, then the shared ones, then the base models (on any arm: the client sends them zero joints), or one typed by
id. The file names it as `checkpoint`, and the cameras get the names that model reads (*Config file*, `cameras`).
Save
registers the station on the credentials' platform with the key (`GET /v1/robots`, then the robot the file already
names under `live.robot_id`, else one named after the box (its hostname and a short code from its machine id), else `POST /v1/robots` with the body's
embodiment: `yam`, `makermods_maker_arm` or `makermods_metal`, and on an SO-101 or a robot of your own the account
embodiment's id, `emb_...`) and writes the `live` section with
that robot's id, so the
Playground's live view links as soon as the station starts; the page says which robot the station is, and a re-save
keeps that robot. A key the platform refuses, or a platform it cannot reach, fails the save before anything is
written. When the `live` extra (PyAV and websockets) is not installed the file gets no `live` section, because the
agent would refuse one at boot, and the page says what to install. The file names each arm's adapter by what stays the
same: a USB CAN link by the name the udev rule gives it (`can_follower_l`, `can_follower_r`; Save waits until the
rule is in effect), a CANable or another USB serial adapter by its `/dev/serial/by-id/` path, or by its USB port
(`/dev/serial/by-path/`) when it has no serial of its own; and each arm's `home`,
in radians. A robot of one's own is saved as `body: custom` with its `custom` section: the driver, the `ik` arm and
the account embodiment picked on the first step. Every other key in the file is the
example's value, unless Save replaces a file of the same body: then that file's other
keys stay as they are, so a hand edit is never undone. A file of another body starts from the example's values. *Save* builds the file, loads it through `robots.config.load` and checks
it through `robots.run.policy_config`, exactly as the agent will, and only then moves it into place; a file the agent
would refuse is never written, and the page shows the `ConfigError` naming the key instead. An existing file prefills
the page (the arm settings and cameras of the same body), and a device that drove the other arm in it is called out
as a left and right swap, so re-running the wizard edits it.

`dreamcli setup` serves the redesigned page, `setup_wizard_v2.html`, beside the first, `setup_wizard.html`, which `DREAM_SETUP_WIZARD_V2=0 dreamcli setup` serves instead, over the same routes plus one: the variable is a feature flag (`shared.contracts.station.SETUP_WIZARD_V2`), read by the wizard on the station, and unset, empty or `1` is on, `0` is off. The redesigned page runs the same steps with the same script (`setup_wizard.js`, which finds its elements on either page by id) and the same pinned three.js, in the platform's look, which the current page shares (one stylesheet, `setup_wizard.css`, and one script, `setup_page.js`, for the header that names the station and the file it writes and the step list that follows the steps), plus its own wording and a connection check before anything else (`GET /api/connection`, served only with the flag on: `GET /v1/robots` with the key, the first call Save makes, answered as `ok`, `bad_key` for a 401, `unreachable` for no answer, or `error` for any other answer) that says how to fix each and holds the steps until the answer is `ok`. `GET /api/state` also carries `station_name`, the hostname Save registers the station under, which the page shows in its header.

`DREAM_SETUP_GRIPPER_MIT=1 dreamcli setup` adds a Switch gripper to MIT button to a YAM arm card whose gripper motor is in another mode, in place of the four steps that stop setup and run `dreamcli motors set-mit`: the variable is a feature flag (`shared.contracts.station.SETUP_GRIPPER_MIT`), read by the wizard on the station, and unset, empty or `0` is off. The button calls `POST /api/motors/gripper_mit` (served only with the flag on, and named in `GET /api/state` as `gripper_switch`), which lets go of every arm, then switches that adapter's gripper motor to MIT and stores it (`robots.motor_mode.set_mit()`); the operator then power-cycles the arm and clicks Start over.

`DREAM_SETUP_SO101_MATCH=1 dreamcli setup` checks each SO-101 calibration file against the plugged-in arms when an SO-101 is picked: the variable is a feature flag (`shared.contracts.station.SETUP_SO101_MATCH`), read by the wizard on the station, and unset, empty or `0` is off. The page calls `GET /api/so101/match` (served only with the flag on, and named in `GET /api/state` as `so101_match`), which reads the calibration each SO-101 adapter's servos hold (`robots.feetech.held_calibration()`, no register written) and answers per file whether it is a calibration file and which adapters it matches, so the page can tick the matching files and grey out the rest.

`dreamcli robot up` (`robots.cli`) first checks that the agent's API port is free, then runs `python -m robots.station --config ~/.dreamlabs/station.yaml` in its own
session with its output appended to the log, records the pid, then waits (up to 60 s) until the agent's API answers,
which it does only once the arms and every camera have opened, and reports the live view (the platform the credentials file names, `DREAM_API_URL`) and the log;
on a config with no `live` section it names the agent's API to start runs on instead. An
agent that dies first (a camera that never delivers a frame, a config it refuses, a missing driver) has switched
the arms off and logged one line saying why; `up` reports it
with its exit code, the seconds it lived and the log's tail, and exits 1; one that is still silent at the
deadline is left running and reported the same way, with `dreamcli robot log` and `dreamcli robot down` as the next
steps. A Ctrl-C while `up` waits ends only the wait.
`--config` runs another file, and any other flag (`--api-port`, `--host`, `--inference-log`) is passed through to `robots.station`. A second `up` while the agent runs
is refused. `dreamcli robot down` sends SIGINT, the agent's clean stop (*Station agent*), waits up to 60 s while the arms park,
saying so once after 2 s, then SIGTERM and SIGKILL with 5 s each; a stale pid file is removed on sight. It says "the arms are switched off" only
when the agent's log says so; otherwise it says the arms may still hold under power and exits 1. Apart from `setup`,
`robot run` and `motors`, nothing in `dreamcli` touches the arms: the agent does, on request. `debug_scripts/station_run.sh` is the tmux-based
alternative for a station that keeps its config elsewhere.

`dreamcli robot run` (`robots.cli.robot_run`) runs one task in the terminal instead of the agent: the same loop
(`robots.run.main`), no agent, nothing registered on the platform. It brings the links up as `up` does, refuses
while another program holds the arms (*Run*), moves the arms home (`robots.home`) and runs until Ctrl-C or a fault; then
the loop holds the arms, they move home again, and every arm is switched off the way the agent does when it exits (a
brakeless arm parks first, a YAM goes limp at home). After an arm fault the arms are switched off where they are, as
the loop left them. A hangup (a dropped ssh session, a closed terminal) stops it the same way, and a second Ctrl-C
is ignored until all that is done. `--config` runs another file, `--task` and `--checkpoint` name the prompt and the model,
`--action-space` what the arms run (`action_space` below), and `--dry-run` runs the policy without writing a target to an arm (`robots.run.DryRun`): each arm's reading and the
target it was asked for are logged once a second, nothing homes, nothing moves, and no rollout is saved. A Maker Arm,
Metal Arm or SO-101 is opened read only, with no motor switched on; a YAM's arms, which i2rt opens powered, stay
powered and stiff where they are until their power is cut. `--view` (on an EEF body: Maker Arm, Metal Arm, SO-101, your own)
serves a live 3D page of each chunk on `http://127.0.0.1:8093` (`--view-port` names another; from a laptop
`ssh -L 8093:localhost:8093 <robot box>`), per arm: the hand path the model asked for, where dream's IK sends the
hand, the arm at the chunk's first and last row, and the `keep_out` shapes the IK keeps it off (`robots.run_view`). It
reads the chunk files the run saves, so it needs `rollouts: {enabled: true}` and is refused with `--dry-run`. A missing key, or a
platform that cannot be reached, ends the run before anything moves; a refused key or checkpoint or a fault that stops
the loop ends it after the arms are switched off. Each prints one line naming it, and the command exits 1.

## Run

```bash
python -m robots.station --config configs/yam.example.yaml    # the station agent: Start / Stop / Home on request
python -m robots.run --config configs/yam.example.yaml        # one shot: run until Ctrl-C (dreamcli robot run)
```

A station runs the agent (*Station agent* below): it opens the arms once, keeps them, and starts,
stops and homes on request from its HTTP API or the Playground's live view. `robots.run` is the one-shot form:

`run.main` checks the config against what the serve accepts (camera roles), builds one `CameraReader` per entry in
`cameras`, the dream client from `robots.run.DEFAULT_CHECKPOINT` and the credentials (`dream.DreamClient()` reads the gateway and the key from `~/.dreamlabs/credentials` itself, and `client.policy()` opens the model on it, `robots.run.open_client()`), both arms from `follower_l` / `follower_r` last (so a camera or model that fails to set up never moves an arm), moves them home (`robots.home.return_to_home`, 3 s), then builds the loop and runs it inside
`async with loop:` until SIGINT, SIGTERM or a fault. Leaving that block commands both arms to a safe hold (or
disengages them after an `ArmError`) and closes the client; then the arms move home again (not after an `ArmError`),
the camera threads stop and every arm is switched off (`FollowerArm.torque_off()`), and a fault is raised after that.
An arm that does not reach home fails the run with an `ArmError`, after the switch-off. `--dry-run` writes no target to an arm (*Command
line*). Logging goes to the root logger at INFO.

Copy `configs/yam.example.yaml` per station and replace the placeholders: the camera
device paths or RealSense serials and `task`; the account is the credentials file, not this one. A Maker Arm
station starts from `configs/makers.example.yaml` instead, a Metal Arm station from `configs/metal.example.yaml`
(with each arm's gripper ends and home filled in), a robot of your own from `configs/custom.example.yaml`, and
there the model is the file's `checkpoint` or `--checkpoint`, since none of them has a
default model. Both entry
points take `--task "..."` to override the config's `task` for that process; `dreamcli` does not, because a
prompt is sent with the Start, from the live view or the API, and changed the same way.

One program drives the arms at a time. Whichever holds them, the station agent, `dreamcli setup` (while it serves the
page), `dreamcli robot run` or `dreamcli motors`, holds a lock on `arms.lock` in the station directory, and the others
refuse with one line naming the holder and how to stop it, such as `the arms are held by dreamcli setup (pid 4242):
press Finish on the setup page, or Ctrl-C its terminal, then run this again`. The lock goes with its process, so a
crash leaves nothing to clean up. An agent an older release started holds no lock: stop it with `dreamcli robot down`
after an upgrade.

## Config file

One YAML file, loaded by `robots.config.load(path) -> AppConfig`. Every key without a default
is required; a missing one raises `ConfigError` naming the key, and so does a value of the wrong
type or an unknown key at the top level, in `follower_*` or in a `cameras` entry. The file holds the hardware and the loop and nothing about the account or the model: a file from before the credentials file, still carrying `policy`, `api_key` or `live.url`, fails at load naming the key and where its value now lives (*Credentials* below), never as an unknown key. There are no silent defaults
beyond the ones listed.

### `AppConfig` (top level)

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `body` | `str` | `"yam"` | Which arms these are: `yam` (i2rt), `makers` (MakerMods Maker Arm, *Maker Arm* below), `metal` (MakerMods Metal Arm, *Metal Arm* below), `so101` (an SO-101, *SO-101* below) or `custom` (a robot of your own, *A robot of your own* below). It picks the arm driver, the arm section's keys, the camera roles, the wire and the forward kinematics (`robots.bodies.for_config`). Every body but `yam` runs EEF checkpoints only: a model that answers in joints is refused (`robots.run.refuse_joint_space`), since on another arm those are another robot's joint targets |
| `custom` | mapping | `body: custom` only | Your robot: `driver` (`package.module:Class`, or `file.py:Class` read from the config file's directory), `ik` (dream's name for your arm's URDF) and `embodiment` (`<username>/<name>`, the embodiment your fine-tune was trained on), all required, and `grip_scale` (what the embodiment's `<side>_gripper` reads fully open; required only when it records one). Refused on every other body |
| `embodiment` | `str` | `body: so101` only, required | The embodiment your SO-101 fine-tune was trained on, `<username>/<name>`, looked up on your account the first time a run needs it; its config is the wire. Refused on every other body |
| `checkpoint` | `str` | none | The model the station runs unless a run names another (`dreamcli robot run --checkpoint`, the model on a Start). Without it a YAM runs `robots.run.DEFAULT_CHECKPOINT` and every other body needs a run to name one. `dreamcli setup` writes none: the live view's Start names the model |
| `follower_l` | `ArmConfig` | required | Left follower arm (below) |
| `follower_r` | `ArmConfig` | required | Right follower arm (below) |
| `cameras` | `list[CameraConfig]` | required | One entry per camera (below); names must be the body's roles: `top`, `left`, `right` on a YAM; on a Maker or Metal Arm the names its model reads (the setup wizard takes them from the model's platform record, `cameras`): `overhead` (base models), `scene` (fine-tunes on the platform's Makers or Metal embodiment, whose record names none) or `head` (Dream Labs' shared fine-tunes), then `wrist_l` or `left_wrist` and `wrist_r` or `right_wrist`, one name per camera; the embodiment's `video` roles on an SO-101 or a robot of your own |
| `task` | `str` | none (empty) | The language prompt, sent as `prompt` with every observation. A file without it is fine for `robots.station`, which refuses Start until a prompt is given with `start`; `robots.run` needs it here or in `--task` |
| `control_hz` | `float` | `20.0` | Tick rate. Arms are read and commanded once per tick; the policy is called once per chunk. 20 by default on every body, a Maker Arm, a Metal arm and an SO-101 included (`config.MAKERS_CONTROL_HZ`, `config.METAL_CONTROL_HZ`, `config.SO101_CONTROL_HZ`; at 30 the zero-shot base model swung a Metal arm near home): the checkpoint's rows are 30 Hz actions, so 20 Hz plays every plan at two thirds speed, the safer pace on a wide-area link and around people |
| `action_horizon` | `int` | required | Rows in every chunk the checkpoint returns (the model's data config's `action_horizon`; 48 for our base model's config (not the serve's `execution_horizon`, which is a refill knob)). Needed to declare the embodiment before the first call |
| `action_timeout_s` | `float` | `3.0` | A refill started at `t0` is late once `now - t0 > action_timeout_s`; see *Failure handling*. 3 s by default: a wide-area round trip and its jitter, with room for a slow reply |
| `stale_after_s` | `float` | required | A frame older than this at the tick fails the tick |
| `max_stall_ticks` | `int` | required | Consecutive empty-buffer ticks (holding the last command) before the loop stops with `ControlLoopError`; counted only once the first chunk has landed. 240 in the example configs and in what `dreamcli setup` writes (8 s at 30 Hz). A value that holds the arms for less than `action_timeout_s` (10, say) is read as 240, with a warning naming the file |
| `max_ee_speed_m_s` | `float` | required | Cap on each arm's end-effector speed from one command to the next, metres per second, measured through the arm's forward kinematics; a faster command is scaled back to it (*Kinematics and the speed clamp*). 0.6 in the example configs. A file still carrying the older `max_joint_step_rad` fails at load naming this key |
| `first_chunk_timeout_s` | `float` | `120.0` | Seconds the first chunk may take after the first tick, or after DreamAPI's last `pool_warming` answer: how long a wait with no answer from DreamAPI may last. A pool that keeps answering `pool_warming` is waited out up to `control.POOL_WARMING_LIMIT_S` (30 minutes). See *Cold pool before motion* |
| `ramp_s` | `float` | `2.0` | Seconds over which the first commands after start, and after every new session, blend from the arms' measured pose into the policy's targets; see *Start-up ramp* |
| `ramp_max_rad_s` | `float` | `0.5` | Joint speed cap during that blend: the ramp lasts `max(ramp_s, largest joint delta / ramp_max_rad_s)` seconds |
| `max_gripper_step` | `float` | `0.05` | Most the gripper command may move per tick, in `[0, 1]` units |
| `grip_force_cap` | `float` | `0.65` | Gripper effort a grasp is held at, in i2rt's `gripper_eff` units (motor torque, not newtons): the loop backs a commanded close off until the measured effort sits there (`robots.gripper_force`). 0 turns the cap off and leaves grip force to the motor PD. A Maker or Metal Arm reports no gripper effort, so the cap passes its commands through; its gripper's hold (`gripper_hold_torque_nm`) does that job |
| `trace_path` | `str` or null | null | CSV appended with one row per commanded tick (`robots.control.trace_columns()`) for post-mortems |
| `inference_log_path` | `str` or null | null | JSON Lines file appended with one record per inference, error, stall and run boundary (*Inference log* under *Station agent*); `robots.station` writes `~/robots-inference.jsonl` when unset, `robots.run` nothing |
| `command_hz` | `float` | `250` | Rate the command thread writes the arms at |
| `command_interp_max_s` | `float` | `0.10` | Longest run from one written command to the next row, seconds |
| `action_space` | `str` | `"eef"` | What a Maker Arm, Metal Arm, SO-101 or robot of your own runs, on the dream client's IK: `eef`, the model's tool poses solved to the arm's joints, or `joint`, the model's own joints, which only one of your fine-tunes of the embodiment the file names may run (`robots.bodies.check_model` refuses any other model, a base model included, and a Maker or Metal Arm, whose file names none). A run may name the other (`dreamcli robot run --action-space`, the Actions toggle beside the Live view's model picker), and the station keeps it for the runs after. A YAM takes the key too, and its default is `eef`; `dreamcli robot up eef` runs its EEF head for one run and `dreamcli robot up joint` (or the older `noeef`) its joint head, either on any YAM model (*The YAM's EEF head* below) |
| `ik_solve` | `str` | `"traj"` | How an `eef` run's tool poses become joints: `traj`, the whole chunk in one solve that continues the arm's last three commands (dream's velocity 5 / acceleration 50), or `tick`, row by row (the dream client's `ik` key `solve`). When the key is absent every body runs `traj` (`robots.config.DEFAULT_IK_SOLVE`); the client solves its first chunks with `tick` while `traj` compiles. `dreamcli robot up --ik-solve tick|traj` overrides it for one agent run. Anything else fails at load |
| `sync_above_ms` | `float` or null | null | The dream client's sync threshold without System 0, milliseconds (`dream.pacing`): a round trip above it runs the whole chunk and holds the arms through the next request. Null is `run.SYNC_ABOVE_MS`, 900, on every body (the client's own default is 700). Set, it replaces that |
| `sys0_sync_above_ms` | `float` or null | null | The same threshold with System 0 (a YAM's default). Null is 900 plus `run.SYS0_SYNC_EXTRA_MS`, 100 ms (1000): System 0 slows sooner into a sync hold than the plain loop, so it stays async longer, the client asking early enough that a reply may land as the last rows run. Set, it replaces that |
| `policy_image_size` | `str` | `"640x360"` | `WIDTHxHEIGHT` every policy frame is downscaled to (`cv2.INTER_AREA`) before JPEG; capture stays at the camera size |
| `obs_window` | `int` | `9` | Frames per camera in every observation, oldest first: the checkpoint's `video_context_frames`. The dream client keeps the window; a value other than 9 is passed to it as its `obs_window` |
| `obs_fps` | `float` | `5.0` | Rate those frames are kept at (the training video fps): each camera encodes one capture every `1 / obs_fps` seconds and the dream client keeps them at this rate; a value other than 5 is passed to it as its `obs_fps` |
| `face_blur` | `bool` | `true` | Blur faces out of every sampled frame at capture (`shared.faceblur`) and out of every frame the live agent encodes; off sends raw frames |
| `home_before_motors_off` | `bool` | `false` | `dreamcli setup` only: after the powered checks, *Switch off the motors* drives the held arms home at the checks' own pace, then switches them off; if that move fails or takes past 15 s the motors go off anyway, where the arms stopped, and the page says so. Off, the arms go limp where they hold, as before. The wizard reads it from the file when it starts and Save keeps it; the station agent ignores it |
| `home_on_key_refused` | `bool` | `true` | The station agent only: a run DreamAPI ends by refusing the key as `disabled` (the account is out of credit or over its spending limit, or the key was revoked) is followed by the Home move, and the run ends in `error` with `error_code` `key_refused` (*Station agent*). `false`, the arms hold where the run left them, as for any other fault. Save in `dreamcli setup` keeps it |
| `exposure` | mapping | managed | Managed colour exposure for RealSense cameras, a YAM station's default: `managed` (`true`), `target_median` (`90`, the grey level the frame median is held at), `limit_us` (null, a ceiling under 85% of the frame period: an exposure of the whole period leaves the sensor no time to read the image out and it delivers half its rate), `seed_us` (null, half the ceiling), `smooth_idle` (`true`; settle at full speed once after the camera opens, then follow drift between runs only after 3 s off target and at most 5% a step, so an idle station's live view does not flicker; `false` brings back full-speed steps between runs). The firmware's auto-exposure is off and the capture thread steers exposure, then gain, between runs; the station holds both for the length of a run. Highlights come before the median: a frame with more than 1% of its pixels clipped is darkened whatever its median, and one clipping over 0.5% is never brightened, so a wrist camera metered with a dark backdrop in view keeps the table's detail. `managed: false` is the firmware's own auto-exposure. Ignored by OpenCV devices |
| `rollouts` | `RolloutsConfig` | off | Saves every run as a LeRobot v2.1 episode on this disk too (below, *Your rollouts*): `enabled` (`false`), `dir` (`~/.dreamlabs/rollouts`), `robot_name` (this machine's hostname) |
| `live` | `LiveConfig` or absent | absent | Live View (below): absent, the station shows nothing on the platform's Live View page |

### `LiveConfig` (`live`)

Live View (*Live view* under *Station agent*): the station runs a `dream.LiveViewClient`, which streams the cameras a viewer is watching to the platform over one outbound `wss` link and takes the page's buttons back. Needs the `live` extra. The section present is what turns it on; the link and its credential are the credentials' platform and key, as for every `dream` client. The encoding (VP8, 640 px wide, 30 fps, 1400 kbps per camera) is the client's own and has no setting.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `robot_id` | `str` | required | This station's robot on the platform: `dreamcli setup` registers it and writes the id; by hand, `POST /v1/robots` with the api key |
| `cameras` | `list[str]` | every camera | Camera names from `cameras` a viewer may subscribe to, in `hello` order; an unknown name fails at load |

An older file's `width_px`, `fps`, `bitrate_kbps`, `encoder` and `record` keys are ignored with one warning (`robots.config.LIVE_IGNORED_KEYS`): the encoding is the client's, and the run is recorded by its own dream client.

### `RolloutsConfig` (`rollouts`)

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `enabled` | `bool` | `false` | Save every run on this disk. `false` writes nothing here; the run's dream client still records it on a platform that takes recordings |
| `dir` | `str` | `~/.dreamlabs/rollouts` | The folder one subfolder per episode is written into |
| `robot_name` | `str` | the machine's hostname | The name written into every episode's `rollout.json` |

### `ArmConfig` (`follower_l`, `follower_r`)

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `can_channel` | `str` | required | SocketCAN interface the arm is on, e.g. `can_follower_l` |
| `gripper` | `str` | `"linear_4310"` | i2rt `GripperType` name, or `anvil_gripper`, the gripper an OpenYAM is sold with, which the station opens as i2rt's `linear_4310` (`config.GRIPPER_DRIVERS`: the same DM4310 linear drivetrain, a shorter stroke) |
| `ee_mass_kg` | `float` or null | `0.62` | End-effector mass handed to i2rt's gravity model; the measured linear_4310 gripper |
| `gravity_comp_factor` | `float` or null | `1.0` | One scalar multiplied into the gravity torque on every joint; null keeps i2rt's per-joint defaults |
| `sdk_gripper_force_limit` | `bool` | `false` | Keep i2rt's hardcoded 50 N gripper limiter; off, because it fights the loop-side `grip_force_cap`, which is the regulator here. Worth turning on only for a station running `grip_force_cap: 0` |
| `temp_export` | `bool` | `true` | i2rt includes per-motor temperatures in its observations |
| `home` | six `float` | zeros | The arm's home pose, radians in its URDF's frame: where Home and Stop move it, the gripper open. Every body takes it; zeros on a YAM and a Maker Arm, required on a Metal Arm, an SO-101 and a robot of your own, whose zero pose is no safe home |

On a Maker Arm (`body: makers`) an arm section takes these keys instead, and the YAM's keys above fail at load as
unknown, the same way these do on a YAM:

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `can_channel` | `str` | required | The arm's CAN adapter: a SocketCAN link (`ip -br link show type can`; a candleLight under the kernel's gs_usb driver, up at 1 Mbit/s) or the `/dev/serial/by-id/...` path of a CANable on slcan firmware, which is opened directly (no root, no `slcand`, and it comes back by itself after a replug; it needs `pyserial`). When an `slcand` holds that CANable, the arm goes through the link that `slcand` made, and `dreamcli robot up` brings that link up. A raw-USB `gs_usb:` channel fails at load naming the SocketCAN link to use |
| `gripper_open_deg` | `float` | `-96.0` | The gripper motor's angle fully open: the open end of the stroke and its soft limit there |
| `gripper_closed_deg` | `float` | `-2.5` | The angle with the jaws closed; must be above `gripper_open_deg` (the gripper closes toward increasing angles) |
| `gripper_hold_torque_nm` | `float` or null | `0.5` | How hard a grasped object is held, N.m at the motor, 0.1 to 2.0; null turns the hold off (plain position control at full gain) |
| `home` | six `float` | zeros | As on a YAM: zeros, the folded calibration pose |

On a Metal Arm (`body: metal`) an arm section takes these keys, and every other fails at load as unknown:

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `can_channel` | `str` | required | The arm's CAN adapter, as on a Maker Arm: a SocketCAN link at 1 Mbit/s classic CAN, or the `/dev/serial/by-id/...` path of a CANable on slcan firmware (opened directly, or through the link of an `slcand` holding it); `gs_usb:` fails at load |
| `gripper_open_deg` | `float` | required, no default | The gripper motor's angle with the jaws fully open, degrees, within 0 to 137.5: the model's grip 1 (100 mm). The grip runs linearly in the motor angle between the two ends. The setup wizard measures both by hand, the open end at the stop; every gripper goal stays the arm's `grip_margin_mm` (2 mm) inside it. A missing end fails at load saying what to supply |
| `gripper_closed_deg` | `float` | required, no default | The angle with the jaws closed: the grip 0. Must be below `gripper_open_deg` (the Metal gripper opens toward increasing angles) |
| `gripper_hold_torque_nm` | `float` | `0.5` | How hard a grasped object is held, N.m at the motor, 0.1 to 2.0; never null (the Metal gripper always holds) |
| `home` | six `float` | required, no default | The arm's home pose; its zero pose stands the arm straight up. The setup wizard captures one, or copy what `dreamcli robot run --dry-run` logs with the arm where it should start |

On an SO-101 (`body: so101`) an arm section takes these keys, and every other fails at load as unknown:

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `can_channel` | `str` | required | The arm's serial port by id, `/dev/serial/by-id/...` (`ls -l /dev/serial/by-id/`), or by USB port, `/dev/serial/by-path/...`, when two adapters report one serial. A `/dev/ttyACM*` name fails at load: it changes when a cable is replugged, so the arms could swap |
| `calibration` | `str` | required | The arm's LeRobot calibration file, what `dreamcli setup` (or `lerobot-calibrate`) wrote for it with the arm in LeRobot's middle pose (by default under `~/.cache/huggingface/lerobot/calibration/robots/so101_follower/`); a relative path is read from the config file's directory. The gripper's closed and open ends are its gripper range. An arm whose servos hold another calibration refuses to start |
| `home` | six `float` | required, no default | The arm's home pose, five joints and a 0 (the SO-101 has five joints; a sixth angle other than 0 fails at load). Its zero pose stands the arm up |

On a robot of your own (`body: custom`) an arm section takes `can_channel` (the device your driver opens: a CAN
link, a serial port) and `home` (required), and every other key fails at load; your driver reads the rest itself.

Every arm the dream client solves for (Maker Arm, Metal Arm, SO-101 and a robot of your own; not a YAM, which runs joints)
also takes `keep_out`: what dream's IK keeps that arm off, a camera mount or a pole, in the arm's base frame (metres,
x forward, z up, from the base). Each entry is `{kind: box, center: [x, y, z], size: [x, y, z]}` (axis-aligned, full
extent) or `{kind: capsule, center: [x, y, z], radius: r, height: h}` (upright). The IK keeps the arm 2 cm off
each. They replace, for that arm, the
ones dream's model of the arm has: none on the Metal Arm and the Maker Arm, since a mount belongs to the station, not the arm. So
without the key the arm is kept off nothing; if a camera mount or a pole stands where an arm can reach it, measure it
and list it here, or the IK may drive the arm into it. A malformed entry fails at load. For example, a pole beside the
right arm:

```yaml
follower_r:
  # ...
  keep_out:
    - {kind: capsule, center: [-0.05, 0.32, 0.30], radius: 0.02, height: 0.60}
```

The same arms also take `reach`: the area the hand's targets are kept inside, seen from above, as `[x, y]` points in the arm's base frame, metres. The setup wizard paints it on its Reach area step: you move each hand over the whole area it should work in, and every point the hand passes is kept (on a 2 cm grid). dream's IK then moves any hand target farther than 4 cm (`dream.client_ik.REACH_RADIUS_M`) from every point to 4 cm from the nearest one, in x and y only; height and rotation are kept. In short, dream pulls the hand's targets into the area, and if the arm still heads out past the edge, it stops there for the rest of that move. So a pole outside the area is kept out at any height, but the elbow is not guarded, and only head `eef` is clamped. Paint one solid area in front of the arm and stop short of the mount. Without the key the hand can go anywhere; a malformed value fails at load. Nothing in the stack knows where a station's camera mount is, and painting the area needs no measuring and works on every robot, where `keep_out` needs each mount measured.

The same arms also take `region`: a box the whole arm stays inside, `{"lo": [x, y, z], "hi": [x, y, z]}` in the arm's base frame, metres, any value `null` for an open face and either key left out for all open. It bounds every link and the grasp point, so a `lo` z at the table lets the hand reach the table while no link goes through it, a `lo` x keeps the arm from reaching behind its base, and a `hi` y on the right arm (or `lo` y on the left) keeps it out of the other arm's half. The dream client clips the hand targets 2 cm inside, its IK keeps the arm 2 cm in, and a row that would leave the box by more than 1 cm, or put any point of the arm inside a `keep_out` obstacle, holds the arm where it is (the dream client's `ik` key `region`). Without the key there is no box; a malformed value fails at load.

The top-level key `ik_solve` picks how the dream client turns an `eef` run's tool poses into joints: `traj` (every body's default) solves the whole chunk at once with the arm's last three commands as history, at dream's velocity 5 / acceleration 50 weights, and `tick` solves the chunk row by row (the dream client's `ik` key `solve`). The client compiles `traj` on a thread when it starts and solves its first chunks with `tick` until it is ready, so a slow machine meets the station's start-up deadlines. On a YAM `traj` did as well as `tick` in our tests, so keep the default.

```yaml
follower_r:
  # ...
  reach: [[0.30, -0.10], [0.32, -0.10], [0.32, -0.08]]
```

### `CameraConfig` (each entry of `cameras`)

Each entry names its camera in exactly one of three ways: `device` opens a V4L2 webcam or any
address OpenCV opens through `cv2.VideoCapture`; `serial` opens an Intel RealSense through
librealsense (`RealSenseSource`), which is how a D405 is read, because cv2 cannot open
their V4L2 nodes; `source` opens a camera class of your own. Setting two or none raises
`ConfigError` naming the keys.

A RealSense needs a USB3 port and cable for 1280x720 colour at 30 fps. On a USB2 link librealsense refuses the mode with "Couldn't resolve requests"; `RealSenseSource` reads the camera's link (`cameras.usb_link()`) and adds the link and that fix to the same `RuntimeError`, every open logs the link, and the setup wizard shows it under each RealSense's picture.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `name` | `str` | required | The camera role the serve reads: `top`, `left` or `right` (the key after `image_window.`) on a YAM, the name the model reads (after `video.`) on a Maker or Metal Arm (`cameras` above), a `video` role of the embodiment on an SO-101 or a robot of your own; anything else fails at startup |
| `device` | `int \| str` | one of the three | OpenCV index, a `/dev/...` path or an address (`rtsp://...`); prefer `/dev/v4l/by-path` symlinks so a reboot cannot swap cameras |
| `source` | `str` | one of the three | A camera class of your own, `package.module:Class` or `file.py:Class` (beside the config unless absolute), for a camera OpenCV cannot open (a GigE or ZED SDK, a ROS topic): `Class(camera_config)` opens it, `read()` returns `(ok, frame)` with a BGR `[height, width, 3]` uint8 frame as `cv2.VideoCapture.read()` does, and `release()` lets it go |
| `serial` | `str` | one of the three | RealSense serial number, quoted (`"260322275394"`; `rs-enumerate-devices` lists them). The colour stream is requested as `bgr8`, so the frame reaches the reader exactly as a cv2 frame would |
| `width` | `int` | required | Capture width, pixels |
| `height` | `int` | required | Capture height, pixels |
| `fps` | `int` | `30` | Requested capture rate |
| `rotate` | `int` | `0` | 0, 90, 180 or 270 degrees, applied before the frame is published. Frames travel as JPEG, which carries no declared size, so neither the resolution nor the rotation enters the embodiment |

### Credentials

Where DreamAPI is, where the platform is and the key for both are not in the config file: they are the credentials file, `~/.dreamlabs/credentials` (or the file `DREAM_CREDENTIALS` names), lines a shell could `source`, which `dream.credentials` reads (the `dream` README, *Credentials*, has the rules: the environment wins over the file name by name, a missing file is production, a malformed URL fails naming the variable and the file). The install block on the platform's Setup API page writes the file for its own environment (the fourth line, `DREAM_DATA_URL`, is the data service, which a station does not use):

```
DREAM_GATEWAY_URL=https://api.dreamlabs.ai
DREAM_API_URL=https://platform.dreamlabs.ai
DREAM_DATA_URL=https://data.dreamlabs.ai
DREAM_API_KEY=dl_live_...
```

`dream.DreamClient()` takes the gateway and the key from it, so the station passes neither; `robots.station` reads the platform and the key from it for the live link, and `dreamcli setup` reads all three and refuses to serve without a key. Moving a station to another environment is editing that one file and restarting the agent. Keep it owner-only (`chmod 600`): it holds the key.

The dream config the station builds (`robots.run.policy_config`) is the model, `robots.run.DEFAULT_CHECKPOINT`, and the `embodiment` block strict mode validates observations and actions against, filled from `robots.adapter.embodiment_spec(camera_names, action_horizon)`: one source of truth, no drift between what the adapter sends and what the client checks. A run may name another checkpoint (`start` on the station API, the live view's prompt), and so may the file (`checkpoint`). On every other body the dict is the body's embodiment config (`robots.adapter.EefWire`: `MAKERMODS_EMBODIMENT` on a Maker or Metal Arm, the embodiment `embodiment` names on an SO-101 and `custom.embodiment` on a robot of your own), `action_hz` and the client's `ik`, and the model is the run's, else the file's `checkpoint`; with neither there is none and the run is refused. On an SO-101 or a robot of your own the model is checked against the embodiment the file names (`robots.bodies.check_model`): a fine-tune of another of your account's embodiments is refused with a `ConfigError` naming both and your fine-tunes of the file's, when the setup wizard saves, when the station boots and when each run's client is built, so a model a run names (the live view's pick) is checked too; a Dream Labs base model and a fine-tune shared from another account pass. The client's IK runs the `eef` head unless the run's `action_space` is `joint` (`run.with_action_space`), which the same check allows for one of your fine-tunes of the file's embodiment only, and the run logs the head it served (`dream client for <model>: ik head 'joint' (the model's own joints)`).

### Live status timing

The station's Live View client (`dream.LiveViewClient`, handed the run's `DreamClient`) puts the control loop's `get_action` round trips in the `status` it sends the live relay (`shared.contracts.live.LiveTiming`: the last round trip, the run's p50 and p95, and the last one's network, DreamAPI and inference parts and its upload rate `up_mbps`, and `recent_ms`, every round trip since the previous such status), at most every `TIMING_STATUS_S` = 2 s, and the platform's Live view shows them as a Response chip whose hover card graphs the last 30 seconds, and as a banner saying the network or the model is slow while the typical round trip stays slow. It needs no setting; a relay older than the field closes the link `4002` on the unknown key, and every environment's relay carries it (`status.timing`).

## Module map

```
src/robots/
├── __init__.py    docstring and __version__; exports nothing else (import modules: from robots import control)
├── types.py       the frozen dataclasses: Frame, ArmState, RobotState, Observation, ArmCommand, Action, ActionChunk
├── errors.py      RobotsError and its subclasses ControlLoopError, ArmError, CameraError, ConfigError
├── config.py      ArmConfig, CameraConfig, AppConfig and load(path)
├── bodies.py      Body, BODIES, for_config(), so101_body(), custom_body(): what differs between the robots (wire, model, IK, driver, kinematics)
├── followers.py   Driver, the one arm interface in URDF units (read, write, close, torque_off); FollowerArm over it: checks, hold, disengage
├── yam.py         I2rtDriver, open_arm(): the i2rt YAM behind Driver, with System 0's MIT block and motor-loop hook; the only module that knows i2rt (with sys0/fast_loop.py)
├── motor_arm.py   MotorArm: the one thread of every bus arm, CAN or serial (ramp, cap, stale target, faults, the park) over its table (ArmSpec, Joint)
├── makers.py      the Maker Arm's table: JOINTS, ARM (60 Hz, linear gripper stroke, dream's makers IK)
├── makers_hold.py GripperHold: the Maker Arm gripper's holding torque and closing cap; HoldingController, the law the Metal gripper runs too
├── metal.py       the Metal Arm's table: JOINTS, ARM (30 Hz, the Metal grip, dream's metal IK)
├── metal_gripper.py  MetalGripper: the Metal gripper's setup, holding control and safety stops
├── damiao.py      DamiaoBus: the Damiao MIT codec and request/reply bus; DamiaoChain, a Damiao arm for MotorArm (MakerMods' follower)
├── so101.py       the SO-101's table: JOINTS, ARM (60 Hz, five joints, LeRobot's 0 to 100 gripper, dream's so101_faas IK)
├── feetech.py     FeetechChain: an arm's Feetech servos on LeRobot's bus for MotorArm, with the calibration check; the only module that knows LeRobot
├── robstride.py   RobstrideBus: the RobStride MIT codec and request/reply bus; RobstrideChain, a RobStride arm for MotorArm; open_can_bus (a CANable directly, or through an slcand's link); the only module that knows python-can
├── driver_template.py  TemplateArm: the commented starting point for a robot of your own, an arm in memory as shipped
├── cameras.py     CameraReader: one capture thread, one latest-frame slot; RealSenseSource; encode_jpeg(); the only module that knows cv2 and pyrealsense2
├── adapter.py     Observation <-> serve dict conversion, commanded_joints(), embodiment_spec(); Wire, YAM_WIRE and EefWire, every fine-tune's wire from its embodiment config; the only module that knows the wire's key names
├── control.py     ControlLoop: tick clock, chunk buffer, refill, publish, dispatch; set_task(), timing, the plan snapshot, the inference log writes
├── gripper_force.py  GripperForceCap and EffortFilter: the hybrid grip force cap the loop runs once a tick, one per arm
├── kinematics.py  ArmKinematics (one YAM arm on the i2rt MuJoCo model; the only module that knows mujoco) and DreamKinematics (dream's IK model of a bus arm or yours, one side each); arm_layout(), the joints an arm drives from dream's spec; for the speed clamp
├── inference_log.py  InferenceLog: the JSON Lines file, one record per inference / error / stall / run boundary
├── rollouts.py    Recorder: saves each run as a LeRobot v2.1 episode off the tick
├── rollout_episode.py  the episode on disk: layout, finishing it from the crash-safe raw files, recovering
├── home.py        return_to_home(): both arms to each arm's home with the grippers open (move_home(): which arm did not make it, and why); python -m robots.home
├── run.py         main(config) and the one-shot CLI (dreamcli robot run), DryRun; loop_settings(), make_cameras(), policy_config() shared with the station
├── run_view.py    dreamcli robot run --view: the live 3D page of each saved chunk and the keep-out shapes (run_view.html beside it)
├── station.py     StationController: owns the arms for the process, one run at a time; ArmsNotOpenedError; python -m robots.station
├── startup_failure.py  The one line a station that could not start writes: device, error, fix
├── station_api.py HTTP control API over the controller (stdlib http.server, JSON)
├── cli.py         dreamcli: setup, robot up / run / down / log over the station directory (~/.dreamlabs); the console script
├── arms_lock.py   one holder of the arms at a time: the flock on the station directory's arms.lock and the refusal line
├── setup_wizard.py  the setup page (setup_wizard.html, .js and the 3D arm setup_viewer.js beside it), its routes, and Save (docs/SETUP_WIZARD.md)
├── setup_view.py     how the page draws an arm in 3D: maker meshes on dream's URDF or simple shapes (setup_view/), the poses it draws
├── setup_access.py   the wizard's network edges: the run's token on every request, the address it prints (network name, interfaces, ssh tunnel), the robot label per machine id
├── setup_session.py  the wizard's arms while a hand moves them, and what the page has settled
├── setup_checks.py   the wizard's judgments: readings in, a finding with its likely cause out
├── setup_motion.py   the wizard's powered checks: the hold check, home, the nudges, the hand sweep, Stop
├── setup_model.py    dream's URDF of the arm (dream_ik) as the wizard checks against it: FK, IK, limits, each link's pose
├── setup_devices.py  what the box sees: arm adapters (one row each) and cameras by serial, the udev rule, a camera's picture, failures in plain words (explain)
├── setup_cameras.py  the camera step: the operator's LeRobot dataset, our reference pictures, the swap check, the live feeds
├── so101.py          an SO-101 on its Feetech bus for the wizard: its LeRobot calibration file, the EEPROM check, units to radians
├── so101_calibration.py  the wizard's SO-101 calibration: middle pose, ranges, zero matched to dream's so101_faas arm
└── sys0/          System 0, the learned motor tracker a YAM's `dreamcli robot up` runs (YAM only); internal, nothing in it is public API
    ├── policy.py      the network, evaluated with numpy from the bundled tracker.npz, and its observation
    ├── tracker.py     one arm's tick: the plan in, shaped; a supervised MIT-mode command out (pure logic)
    ├── motors.py      ArmMotors (one arm's 250 Hz tracker and the stale-block check), MotorPath (the writer for the ramp, the trackers after it)
    ├── arm_process.py open_arms(), RemoteArm: each arm in a process of its own with its tracker, so the 4 ms tick never waits on the agent
    └── fast_loop.py   i2rt's motor loops sped up at runtime, the transport the network expects
```

Every quantity carries its unit: `control_hz`, `action_timeout_s`, `timestamp_s`,
`positions_rad`, `velocities_rad_s`. Joint positions are radians, because i2rt reports them
that way and the checkpoints were trained on them; the gripper is a float in `[0, 1]`, 0 closed
and 1 open. `Frame.data` is RGB uint8 `[H, W, 3]`, converted from OpenCV's BGR at capture.

The loop publishes three read-only snapshots per tick, as public attributes rather than getters,
so a viewer or a recorder can read them from a concurrent task without ever blocking
control:

```python
state = loop.state  # RobotState, or None before the first tick
frames = loop.video  # dict[str, Frame], the same immutable Frames the capture threads publish
plan = loop.plan  # PlanSnapshot: the rows still to run and the last 3 s of commanded ticks (*Plan vs actual*)
```

## The seam with dream

`dream` forwards the observation verbatim to the model's serve behind DreamAPI and returns
the serve's action dict unchanged, so the key names are the ones the model reads: on a YAM, the parts and
camera roles of its own embodiment config (`robots.adapter.YAM_EMBODIMENT`, the `embodiment` block the client is
built with), as on every other body. `robots.adapter` converts in both directions and is the only place that knows
them (`robots.adapter.YAM_VIDEO_WIRE`).

The loop drives one `dream.DreamClient` per run, its policy opened by `robots.run.open_client()` on the config `robots.run.policy_config()` builds. The client keeps each camera's window of frames: the loop hands it each camera's newest encoded frame and the arms' state with `client.set_observation(frames, observation, timestamp=...)`, once every camera has encoded a new frame (timed by the newest capture among them) and on every tick a refill starts, and a refill calls `client.get_action(commanded=...)`, which sends the window the client kept and the newest state. A new episode (a new prompt, a retry after a fault) is the client's `reset_episode()`, on the refill's thread before its request; `close()` ends the run.

| Direction | Key | Shape | Type |
| --- | --- | --- | --- |
| observation | `prompt` | | `str`, the config's `task` |
| observation | `state.right_arm`, `state.left_arm` | `[6]` | float32, radians |
| observation | `state.right_gripper`, `state.left_gripper` | `[1]` | float32, in `[0, 1]` |
| observation | `video.overhead`, `video.wrist_l`, `video.wrist_r` | list of up to `obs_window`, oldest first (the window the dream client kept of the frames the loop handed it; one on an episode's first request) | base64 JPEG, quality 85, BGR-encoded; the config's `top`, `left` and `right` cameras |
| action | `q_r_chunk`, `q_l_chunk` | `[H, 7]` | six joints in radians, then the gripper |

`H` is `action_horizon`. Right arm first, as the serve lists them. The serve answers the YAM's parts under
`per_key` (which the client's strict mode checks against the embodiment config) and packs them into the two
chunks, which the adapter reads; the serve's other keys (`q_r`, `q_l`) are ignored. The gripper is the seventh
column of each row, clipped to `[0, 1]` by the adapter. A chunk whose rows disagree with `action_horizon`, or whose
rows are not seven wide, raises `ControlLoopError` naming the key and shapes. Each frame of a window goes
on the wire as one JPEG: `robots.cameras.encode_jpeg()` flips the captured RGB frame to BGR
before encoding, because the serve decodes with cv2 and converts BGR to RGB, so the colours
arrive as captured.

The older YAM keys, `state.arm_r.joint_pos`, `state.arm_r.gripper_pos`, `state.arm_l.joint_pos`, `state.arm_l.gripper_pos` and `image_window.top`, `image_window.left`, `image_window.right`, with `robots.adapter.embodiment_spec()` as the block, carry the same values; the serve still reads them, but they are deprecated and no body speaks them. `robots.adapter.to_dream()`, `embodiment_spec()` and `YAM_WIRE` keep that wire while the API accepts it. Nothing in a config file differs between the two.

Beside every request the loop passes `commanded`: per chunk key, the joints it last commanded
each arm (`robots.adapter.commanded_joints`: the row the loop submitted last, nothing before the
first). The client never sends them: its seam join, and on a Maker Arm its IK, take over from the
row after them, where the arms are on the previous chunk. The tick reads its observation before it
writes its row, so these are the previous tick's row, and the client adds the one row itself.

`get_action` blocks and allows one request in flight per client, so the loop runs it in a worker
thread (`asyncio.to_thread`) and never has more than one outstanding; `set_observation` never waits on a request,
so the ticks go on handing over frames meanwhile. Nothing in `dream` is
changed or extended; if its shapes and the loop's diverge, the fix is in `robots.adapter`.

## Start-up ramp

A Dream-1 policy's first targets sit near its home pose, and the arms rest wherever the last
run or the operator left them, so sending the first chunk as is would be a step change at the
first tick. Instead the loop takes the arms' measured joint positions and grippers on the first
commanded tick as the ramp origin and sends `origin + alpha * (target - origin)`, with `alpha`
rising linearly from one step to 1 over the ramp, after which commands pass through unchanged.
The ramp lasts `max(ramp_s, largest joint delta / ramp_max_rad_s)` seconds, sized from the
first target of the chunk, and is counted in commanded ticks, so a stall or an overrun
stretches it rather than skipping ahead. The buffer drains at the normal rate underneath; the ramp blends, it does not
pause. It runs again after a `DreamSessionExpired` reset, from the pose measured on that tick,
because a new episode's first targets are near the home pose again. One INFO line announces
each ramp with its duration and largest delta.

## Chunk alignment

A chunk answers the observation it was computed from: row `k` is the target for
`observed_at_s + k * period`. Inference takes time (about 1.2 s for our base model checkpoint), so
by the time a chunk lands the arms have moved on. The loop skips the rows they have executed
since that observation, never more than the wall clock says are due, and installs the rest; the
per-chunk INFO line reports both numbers
(`chunk 7: 48 actions, 11 executed since its observation, 36 due by wall time, after 1200 ms`).
With a continuous prefetch the two agree (the latency in rows: 36 of 48 at 30 Hz). After a hold
they differ, and only the executed count is right: the arms advance along a plan only while rows
are commanded, so a chunk that lands after the buffer drained resumes the plan where they
stopped instead of jumping ahead to where they would have been. Installing every chunk from row
0 would rewind the commanded target by the inference latency on every replacement, and skipping
by wall time after a hold jumped it forward by the length of the hold; both are step changes
that on a YAM drive a current spike through the PD loop. A chunk that lands after its whole
horizon has passed by wall time is discarded with a WARNING naming the latency and the horizon,
and the next due tick refills.

When the refill starts is not a setting of the station: the dream client decides it (its README, "Pacing"). Every tick with no request in flight the loop asks `client.refill_due(executed, remaining)`, the rows of the current chunk the arms have run since its install and the rows still queued. The client knows the loop's rate (`control_hz`, 20, passed to it as `action_hz`) and picks the mode from the median of the last five round trips: up to 900 ms on every body (`sync_above_ms`, `run.SYNC_ABOVE_MS`; 1000 ms with System 0, `sys0_sync_above_ms`; the client's own default is 700 ms) it is async, the next request timed so its reply lands as the arms finish 30 rows of the current chunk (the client's execution horizon), the rest of the chunk being runway for a late reply; past it, sync, the arms running the whole chunk (`action_horizon` rows), the next request sent only once they all have, and the arms holding through the inference. The first sync chunk after a switch was requested in async while the arms moved, so it installs past the rows they ran during the round trip, like any async chunk, and runs to its end. A chunk that took longer than its whole horizon while the arms held installs whole rather than being discarded as past. The loop's `chunk N:` log line and every `inference` record carry the client's `mode`.

## Chunk handoff

Every chunk is a fresh plan from a new inference, so even a perfectly aligned one need not start
where the last command left the arms. The loop does nothing about it: the blend from one chunk
into the next is the dream client's (`dream_library/dream_client`), which filters every chunk and
crossfades its head from the previous chunk at the row after the joints the loop last commanded
(*The seam with dream*), before the loop sees a row. So the rows pop out of the buffer exactly as `get_action` returned them, and there
is no smoothing stage in the loop and no key to turn one on. The very first chunk, and the first
after a session reset, are the ramp's alone.

The rate clamp is the last line of defence, independent of everything above; it is a cap on
each arm's end-effector speed, described next, plus a per-tick step cap on the gripper
(`max_gripper_step`, default 0.05, in `[0, 1]` units). The first command of all is never clamped,
and the ramp starts from the measured pose so its first step is small.

## Grip force cap

A policy commands a gripper position, and the motor PD turns whatever position error is left into force, so a
commanded close on a grasped object squeezes as hard as that leftover error implies rather than as hard as
anyone chose: a thin object is crushed, and the same checkpoint grips differently on two stations whose grippers
calibrate a little apart. `grip_force_cap` (default 0.65) is the number that replaces the accident. In free
motion the gripper command passes through; in contact the loop offsets it back toward open until the measured
gripper effort sits at the cap, integrating the signed error so it freezes there instead of hunting around it or
creeping back into an over-squeeze, and re-grips when the effort falls (an object compressing, or slipping). The
back-off bleeds away in free motion or on a deliberate open, and a ramp resets it, so every grasp starts from
zero. The cap runs before the rate clamp, which still has the last word on how fast the gripper may move.

In contact the command is also bounded on both sides of the measured jaw: never more than `cap / stiffness` closed
of it (a floor, so a full close on a rigid object never asks the motor PD for more than the
cap), and never open past it (an effort reading that lags the jaw must not let go of the object; without this
bound a held grasp cycled between squeeze and release about twice a second). Contact means a high effort on a
near-stalled jaw, so a fast free close is not throttled, and a tick with no effort reading keeps the back-off
instead of snapping back to the raw close.

The unit is i2rt's `gripper_eff`, the gripper motor's torque, not newtons, which is why the key is not named for
one. A driver that reports no effort has
nothing to regulate on and is commanded as the policy sent it, and `grip_force_cap: 0` turns the cap off, which
is the only case where `sdk_gripper_force_limit` is worth turning back on: i2rt's motor-level limiter and this
cap fight when both run. The implementation is `robots.gripper_force`, with one `GripperForceCap` per arm held by the loop.

## Kinematics and the speed clamp

The rule is that the end effector never moves faster than 0.6 m/s, so the clamp is
`max_ee_speed_m_s` (required; 0.6 in the example configs), a cap in metres per second, not a joint step. A
per-joint step cap (the older `max_joint_step_rad`) says nothing about how fast the hand moves (0.02 rad of base yaw
swings an outstretched arm a centimetre, 0.02 rad of wrist roll almost nothing) and, at 0.02 rad
per 20 Hz tick, cut the policy on a third of all ticks of a run (1115 clamps in 160 s); a config
still carrying it fails at load naming the replacement.

Speed in metres needs the arm's geometry. `robots.kinematics.ArmKinematics` runs MuJoCo's
forward kinematics on the model i2rt builds from the YAM arm XML and the
configured gripper's (`ArmConfig.gripper`): `ee_position(q6)` sets the six arm joints, leaves the
gripper's two finger slides at zero (the `grasp_site` sits on the gripper body, so the fingers do
not move it) and returns the site's position in the arm's base frame. One model and one data per
arm, allocated once and reused across runs; a call is under a microsecond and allocates only the
returned 3-vector. `robots.run.make_kinematics(cfg)` builds the pair; the station keeps them for
its lifetime and every loop takes them as `kinematics=`. Tests inject a linear stand-in; without
kinematics (`None`) the joints are not clamped and every end-effector speed is NaN.

Per arm per tick: `p_prev = fk(previous command)`, `p_new = fk(target)`,
`speed = |p_new - p_prev| * control_hz`. Over the cap, the joint delta `target - previous` is
scaled by `cap / speed`; because forward kinematics is nonlinear the scaled step is measured once
more and scaled again by what it came to when still over, then accepted. The first clamp of a run
is logged (`command clamped: left end effector asked 2.31 m/s, limit 0.6 m/s`), every clamped
tick counts in `ControlLoop.clamp_count`, and `ControlLoop.max_ee_speed_requested_m_s` keeps the
largest speed either arm asked for before the clamp, so the status panel shows how hard the policy
pushes against the cap. The trace records the requested speeds per tick (`ee_speed_l_m_s`,
`ee_speed_r_m_s`) and whether the tick was clamped.

`trace_path` names a CSV file the loop appends to, one row per commanded tick (hold ticks write
nothing), flushed every 30 rows and closed on exit; the header goes in when the file is new. The
columns, `robots.control.trace_columns()`, with a joint column per joint of each arm (`l0` to `l5` on a six-joint arm):

| Column | Meaning |
| --- | --- |
| `t_s` | Monotonic tick time, seconds |
| `chunk_id` | Ordinal of the chunk the row came from, 1 for the first |
| `row_index` | Index of the row within that chunk, 0-based |
| `ramp_alpha` | Blend weight of the start-up ramp on this tick; 1 once it has ended |
| `clamped` | 1 when the rate clamp bounded either arm's command this tick |
| `measured_l0..`, `measured_r0..` | Measured joint positions, radians, left arm then right, one per joint |
| `gripper_l`, `gripper_r` | Measured gripper openings |
| `commanded_l0..`, `commanded_r0..` | Joint targets written this tick, after ramp and clamp, one per joint |
| `cmd_gripper_l`, `cmd_gripper_r` | Gripper targets written this tick |
| `ee_speed_l_m_s`, `ee_speed_r_m_s` | End-effector speed each arm asked for this tick, before the clamp, metres per second; `nan` on the first command and without kinematics |

## Command thread and policy frames

The tick does not write the arms. It submits the ramped and clamped command to a command thread that
writes both arms at `command_hz`: each period it moves linearly
from where the previous run got to toward the newest command over the row cadence (clamped to
`[1 / command_hz, command_interp_max_s]`) and writes it. i2rt applies each `command_joint_pos` straight to the
motor PD, so a row re-sent as-is at the row rate would be a position staircase; the interpolation is rate conversion
for the PD, not smoothing. There is no filter on this path: the rows arrive smoothed from the dream client. A run always starts from
where the last one got to, never from the target. `ControlLoop(command_thread=False)` writes synchronously for
tests; the trace's last column, `commands_written`, counts the periods written.

The policy sees a window, not a frame, and the dream client keeps it: `obs_window` frames per camera, one every `1 / obs_fps` seconds. Each camera has an encode thread beside its capture thread that blurs faces out of one capture every `1 / obs_fps` seconds (the first capture of each such slot of the monotonic clock, the same slots on every camera), downscales it to `policy_image_size` and JPEG-encodes it (quality 85, BGR); every `Frame` carries the newest as `Frame.jpeg`, and the body's wire (`robots.adapter.YAM_VIDEO_WIRE` on a YAM) hands it to the client verbatim under `video.<role>`, so a tick encodes nothing. At the start of an episode the window is shorter, never empty. Capture at 1280x720@30 where the USB3 link allows it. The downscale does not
keep the aspect, so capture in the aspect the model's data was recorded in: 16:9 for YAM, Makers and Metal data. A
Metal camera's 640x480 mode is a centred 4:3 crop of its 16:9 view, which the downscale would stretch 1.32x wide; the setup wizard writes 1280x720 for Metal.

How the followers are configured: `ee_mass_kg` reaches i2rt's
gravity model, `gravity_comp_factor` is set on the driver, the SDK gripper limiter is off, temperatures are
exported, and the gripper target reaches the wire as the loop hands it down: how hard a commanded close ends up
squeezing is regulated by `grip_force_cap` on the measured effort, not by rounding a near-closed target to zero.
`disengage()` releases in that order: gravity-comp idle, then zero torque.

## System 0

On a YAM, `dreamcli robot up` (`python -m robots.station`) hands the arms' motors to System 0, a learned motor tracker (`robots.sys0`). Without it the command thread turns each row into a position target for the motors' own PD, and the measured arm reaches a row about 150 to 220 ms after it was asked to. With it, a small network looks at the measured joints and the next second of the plan and sets each joint's torque every 4 ms, with the firmware's stock damping left in place, so the arm follows the plan as it moves. The network ships in the package (`sys0/tracker.npz`, about 380 KB, evaluated with numpy; no GPU and no PyTorch on the box). It is the YAM's default; nothing about it is configurable: `dreamcli robot up nosys0` (`--no-sys0`) runs without it, and `dreamcli robot down` then `dreamcli robot up` or `dreamcli robot up nosys0` switches. Every other body runs without it, and asking one for it (`dreamcli robot up sys0`, `--sys0`) is refused at start. A station with a tracker of its own, trained for the same observation and action, replaces `tracker.npz` with a file of the same arrays (`robots.sys0.policy` lists them).

What changes on the station when it is on:

- The arms open with i2rt's fast motor loop (`robots.sys0.fast_loop`): all seven MIT frames go out back to back and the robot loop sleeps 0.5 ms instead of 4, so a torque reaches the arm in about 9 ms instead of 16.5. This is why it is chosen when the agent starts, for the life of the process.
- The start-up ramp is unchanged and runs on the command thread. Once it ends, the 250 Hz System 0 thread (`MotorPath`) takes the motors: each tick it samples the plan (the command the loop just submitted, then the rows still buffered, one row per tick from now) now and every 0.1 s for a second ahead, keeps it 0.03 rad inside every joint limit, under 0.8 of the safety trip speed per joint and under `max_ee_speed_m_s` at the grasp point, and writes each arm's MIT block. A new ramp (a session the gateway replaced, a prompt change) hands the arms back to the command thread for the ramp; a stall keeps them on the last command.
- Every command passes through a supervisor. A joint faster than 4.5 rad/s (7 on the wrist), a joint past its limit, a tracking error over 0.4 rad held for 0.2 s, a torque clamped for 25 ticks in a row, or a gap of more than 50 ms between two ticks puts that arm on a stiff PD hold where it is, and the run stops with `System 0 safety hold on the <arm> arm: <reason>` as its error. A check on i2rt's own motor loop does the same when the System 0 thread stops writing for 50 ms, since i2rt would otherwise keep sending its last torque.
- The gripper stays a position servo on the loop's own command, after the grip force cap, held between ticks; the
  network's gripper output is not used. The cap is what regulates a grasp, with or without System 0 (on a spoon,
  a residual on the gripper target and a reference drifting toward the raw close made a held grasp jitter).
- A stop leaves each arm on its PD at the last target System 0 wrote, as a stop without it does.

`dreamcli robot up` on a YAM reports `System 0: on`, and `dreamcli robot up nosys0` `System 0: off`; the `run_start` record of the inference log and the `loop ready` line say whether it is on; every stop logs `System 0: N ticks driven, worst gap X ms, K ticks later than 12 ms`.

### The YAM's EEF head

A YAM runs its model's EEF head by default: the dream client adds each hand's pose to every observation from the YAM's kinematics and solves the model's tool poses to the YAM's joints (`ik` `{"arm": "yam", "head": "eef"}`, with `robots.adapter.YAM_POSE_EMBODIMENT`: the YAM's parts and the poses read, each arm's joints, tool pose and grip answered). The station's default model is `dreamlabs/dream-1-beta` (`robots.bodies.YAM_CHECKPOINT`), a Dream-1 base model with an EEF head. A model with no EEF head answers the YAM's joints only: the client runs those, and the station log says so once per run (`<model> has no EEF head (it answers no tool poses): running its joint head on this YAM instead`); the next run of a model with one runs its EEF head again. `dreamcli robot up noeef` (`--no-eef`, the action space `joint`) runs every model's joint head, on the YAM's own embodiment (`robots.adapter.YAM_EMBODIMENT`). The loop and System 0 are the same on both heads: the client returns joint chunks, and System 0 tracks the rows the loop commands, so on the EEF head it drives the motors to the IK solution. `dreamcli robot up` reports `head: eef` or `head: joint`, and each run logs `dream client for <model>: ik head ...`.

## Cold pool before motion

The first `get_action` on a cold DreamAPI pool is refused with a retryable `DA-2000` and a
`retry_after_s` hint while the serve boots: about a minute for a warm model, more than ten for a cold
fine-tune. After any retryable
`DreamServerError` the loop starts no refill before `now + retry_after_s` (`RETRY_AFTER_DEFAULT_S`,
1 s, when the envelope carries none) and logs the wait at WARNING with the reason: every `pool_warming`
answer, which also says how long the policy has been starting, and any other failure once per
`RETRY_LOG_EVERY_S` (5 s) while it repeats word for word, with how many tries over how long. Until the first chunk
has landed nothing is at stake, so those empty ticks hold the arms and count no stall. Every
`pool_warming` answer restarts `first_chunk_timeout_s` (default 120 s), which is kept for a DreamAPI
that does not answer at all; a pool still warming after `POOL_WARMING_LIMIT_S` (30 minutes) ends the
run with `ControlLoopError` saying so. A first chunk that lands more than `MOVE_AFTER_WAIT_LIMIT_S` (60 s)
after the first tick is not run: the run stops with `ModelReady` before its first move, so the arms never
start moving on their own after a wait the operator may have walked away from, and the Live view asks for
Start again, which finds the model up. Once a chunk is in, `max_stall_ticks`
bounds every empty-buffer burst as before, and a retryable refusal still holds the next refill for
its hint.

## Failure handling

Ordinary calls raise. The one narrow `except` sits at the tick boundary, in `ControlLoop.step`,
and logs; nothing below it catches.

| Failure | Behavior |
| --- | --- |
| Camera frame older than `stale_after_s` | Fail the tick, hold last command, log once per burst |
| Camera thread dies or never publishes a first frame (5 s) | Stop the loop with `CameraError`, which says why in plain words and the next step ("check its cable; dreamcli setup shows the cameras"); a policy running on stale or missing video is the unsafe case. A station opens a dead camera again by itself (below) |
| `get_action` raises `DreamAuthError`, `DreamConfigError` or `DreamObservationError` | Stop the loop: a retry fails the same way; the operator fixes the key, the config or the code. Unless the station file sets `home_on_key_refused: false`, a `DreamAuthError` whose reason is `disabled` (out of credit) is then followed by a home move (*Station agent*) |
| `get_action` raises `DreamSessionExpired` | DreamAPI replaced the session mid-episode (restart, eviction, serve death) and the client adopted the new one: log `episode restarted server-side` at WARNING, drop the buffered rows (they answer the dead session), refill when next due |
| `get_action` raises a retryable `DreamServerError` (`DA-2000` cold pool or rate limit, `DA-5000`, `DA-6000`) | Log the wait once at WARNING with the reason; no refill starts before `now + retry_after_s` (1 s when absent); keep draining the current chunk |
| `get_action` raises the same `DA-5000` (serve fault) with reason `error`, word for word, `SERVE_FAULT_REPEATS` (3) times in a row (a `timeout` or `unreachable` serve is not counted: the platform replaces it) | The serve fails that request every time (a camera the checkpoint reads and the request lacks): stop the loop with `ControlLoopError` carrying DreamAPI's message, recorded as the episode's `error` in `rollout.json` |
| `get_action` raises a `DreamServerError` DreamAPI marks not retryable (`DA-2000 max_serves`) | Stop the loop at once with `ControlLoopError` carrying DreamAPI's message and its reason, recorded as the episode's `error` in `rollout.json`: asking again answers the same |
| The chunk has the wrong shape for the body (`DreamActionError` in strict mode, `ControlLoopError` from the adapter otherwise) | Asking again at once answers the same, so no refill starts before `now + RETRY_AFTER_DEFAULT_S`; keep draining the current chunk. Once the buffer empties, `max_stall_ticks` ends the run with a `ControlLoopError` whose `reason` (and `last_error["error_code"]`) is the refusal's code, so the page says what to fix |
| `get_action` raises `DreamProtocolError` (nothing answers at the gateway's address, e.g. a refused connection) | Asking again at once fails the same, so no refill starts before `now + RETRY_AFTER_DEFAULT_S` (1 s) and the same failure is logged once per `RETRY_LOG_EVERY_S` (5 s), with how many tries over how long; keep draining. The run ends on the loop's limits: `first_chunk_timeout_s` before the first chunk, `max_stall_ticks` once the buffer is dry after it, with dream's words in the `ControlLoopError` and as the episode's `error` in `rollout.json` |
| `get_action` raises any other `DreamError` (a timeout) | Log at the tick boundary (once per `RETRY_LOG_EVERY_S` while it repeats), keep draining the current chunk, retry the refill when next due |
| `get_action` exceeds `action_timeout_s` | Keep draining; a result that lands late is discarded and a fresh refill starts on the next due tick. The worker thread is never cancelled; `dream`'s `timeout_ms` bounds the call |
| Chunk lands after its whole horizon has passed | Discard it with a WARNING naming the latency and the horizon; the next due tick refills |
| A command would move an arm's end effector faster than `max_ee_speed_m_s`, or the gripper more than `max_gripper_step` in one tick | Scale the joint delta back to the cap (the gripper to its step), warn once per run, count it; see *Kinematics and the speed clamp* |
| No chunk yet (cold pool, first inference) | Hold the arms, count no stall; `ControlLoopError` once `first_chunk_timeout_s` has passed since the first tick or the last `pool_warming` answer, or `POOL_WARMING_LIMIT_S` since the first tick; a first chunk that lands more than `MOVE_AFTER_WAIT_LIMIT_S` (60 s) after the first tick stops the run with `ModelReady` before its first move |
| Buffer empties before the next chunk arrives | Wait at most until the refill deadline, then hold last command and log the stall; `ControlLoopError` after `max_stall_ticks`, naming the refill failure that emptied the buffer (its text, and its code as `reason`) when one stands |
| Arm read/write error (`ArmError`) | Stop the loop and disengage both arms; an arm that would fall limp (a brakeless Maker Arm, Metal Arm or SO-101, `FollowerArm.drops_when_limp`) holds instead, so one arm's fault never drops the other |
| An arm's own alarm (`FollowerArm.alarm()`: a motor at its body's stop temperature for 1 s, a joint blocked for 5 s, a Metal gripper that stopped; *Motor guard*) | Stop the loop with `SafetyStop` naming the arm, the motor and what to do; the arms hold. `robots.run` then homes both and parks them; a station parks the alarmed arm and switches it off, and Reset opens it again |
| Tick overrun | Log with the measured tick duration; reset the deadline so lag does not accumulate |

A chunk is applied from the row the arms have reached (*Chunk alignment* above) and commanded as
sent (*Chunk handoff*). On any unhandled exception the loop leaves its `async with` block, whose
`__aexit__` commands both arms to a safe hold (or disengages them after an `ArmError`), stops
the cameras and closes the client.

## Station agent

`python -m robots.station --config station.yaml [--task "..."] [--api-port 8091] [--host 127.0.0.1] [--inference-log PATH]`

The API binds `127.0.0.1` unless `--host` says otherwise. `--host 0.0.0.0` exposes arm control (`POST /start` moves the arms) to the whole network with no password; the website and the live view do not need it, since the station dials out (`dream.LiveViewClient`).

The agent is the resident process a station runs all day. It checks the config, opens both
followers once and keeps them for its lifetime, serves the HTTP API on `:8091`,
opens the cameras once (a camera that cannot open stops the station at boot, before
anything listens) and keeps them publishing for the life of the process, so the
live view shows the scene between runs and a prompt can be chosen with the scene in view; then it
waits: nothing moves until a Start. Each run builds a fresh dream client and a fresh `ControlLoop`
over those readers (a reader whose thread died is rebuilt at the next start, since a stopped reader
cannot be restarted, and by a watchdog every 2 s while the station is idle, so the live view never shows a
frozen scene between runs), exactly as `robots.run` would but for the cameras, and runs it in a task. A reader's
capture thread survives a source that returns no frame for up to 2 s (a USB stall, a driver timeout) and stops
only past that. SIGINT, SIGTERM or SIGHUP stops a running loop, switches the arms off (they go limp), closes the API server and
exits. `debug_scripts/station_run.sh serve station.yaml` keeps it in a tmux session with the
log at `~/robots-station.log`.

`StationController` (`robots.station`) is the one owner of the arms and enforces three rules:
exactly one run at a time (a second start is refused), and no run without a task prompt (a start with none given and none in the config is refused); stop cancels the loop task and waits for
its `__aexit__`, which stops the command thread and leaves both arms holding, before reporting
idle; home runs only when no loop exists, so the arms are never written from two places at once.
A refused request is a `StationError`: 409 on the API, a failed acknowledgement on the live link, never a crash.
Its states: `idle`, `starting` (building the client, cameras and loop on a worker thread),
`running`, `stopping`, `homing`, `resetting` (letting both arms go and reopening their drivers, then `homing`), `error`
(the last run, home or reset failed; `last_error` says how, and a start, a home or a reset is accepted again).
Every transition is one INFO line. An arm whose driver died (i2rt's CAN thread stops when a motor stops answering,
when the bus goes away or on a joint limit trip) reads `false` in `status()["arms"]`, a start or a home is refused
naming it, one that dies while the station idles puts it in `error` with `last_error` naming the arm, why, and what to
do, and `reset()` is the way back: both arms are let go, opened again from the config and homed. An arm with no
brakes that still answers (Maker Arm, Metal Arm, SO-101) parks and switches off before its reopen; a dead arm, which
cannot park, and a YAM are closed where they hold. A dead SO-101 drops when Reset opens it again: support it before
you press Reset. When
one arm does not open, the one that did is switched off (parked first). Like a start, a reset is accepted at once and
goes on in a task (`resetting` while the arms are let go and reopened, then `homing`, then `idle`, or `error` with
`last_error` saying what failed); a reopen that fails leaves the closed drivers in place, both arms read dead, and the
next reset tries again. A home move or a reset, once started, runs to
its end whatever happens to the caller: the live agent's link dropping or an API call timing out cancels the wait,
never the move, so the state never sticks in `homing` or `resetting`. On exit the agent switches both arms off after the
cameras (`FollowerArm.torque_off()`): the driver is closed (its threads are not daemons and would keep the process alive), then the DM disable command goes to every motor on the arm's link, because a driver sends nothing on close and a motor keeps the last target it was given, so a closed arm stays stiff until its power is cut. The arms go limp and settle under gravity from wherever the last run or home left them, so the workspace under them must be clear before a `dreamcli robot down`; a motor that does not answer is logged and skipped so the others are still switched off, and then the switch-off fails naming any that may still be energized. An arm with no brakes that stopped answering cannot park, so exit leaves it holding and says to support it and switch its power off. The same switch-off runs when the agent fails at boot, so a failed start never leaves the arms holding. A process that is killed rather than stopped switches nothing off and the arms hold. A stop (`POST /stop`) still holds: only exit switches off.

A run DreamAPI ends by refusing the key mid-run brings the arms home unless the station file sets `home_on_key_refused: false` (on by default). The refusal is `DA-1000 UNAUTHORIZED` with reason `disabled`, which the gateway answers once the platform's key export marks the key disabled: the account is out of credit or over its spending limit (`PLATFORM_BILLING_CREDITS`), or the key was revoked. With it set to `false`, a run that ends on it leaves the arms holding wherever the policy last put them. Otherwise, once the loop's `__aexit__` has stopped the command thread and left the arms holding (so nothing else writes them), `StationController` runs the same move as **Home** (`homing`, then both arms at their home joints with the grippers open, at the Home move's pace) and ends in `error` with `last_error["error_code"]` `key_refused` (`robots.errors.KEY_REFUSED`), `code` and `reason` still DreamAPI's; the live view words it from the code with a link to Billing. The move is a home like any other: a start, a stop, a home or a reset while it runs is refused, and a process exit waits it out. A stop asked while the loop was exiting wins: the arms hold. Home is refused while an arm's driver is dead, so then neither arm moves: the run ends holding, with `error_code` `disabled` and a message saying the arms were not homed and why. An arm that fails during the move ends in `error` with `HomeError`, as a Home would. Every other failure, including the other `DA-1000` reasons (`missing`, `unknown_key`, `not_entitled`, `region_not_supported`), still holds the arms.

### HTTP API

Six routes on the stdlib server, JSON in and out; every call runs on the controller's event loop.
A success is 200 with the status dict; a refused transition 409, a bad body or prompt 400, an
unknown route 404, a call that does not return within 60 s 504; every error body is
`{"error": "..."}`.

```bash
curl -s localhost:8091/status | jq                                    # GET: the status dict below
curl -s -X POST localhost:8091/start                                  # start with the current prompt
curl -s -X POST localhost:8091/start -d '{"prompt": "fold the towel"}'   # ... with a prompt for this run
curl -s -X POST localhost:8091/start -d '{"checkpoint": "dreamlabs/dream-1-ft:abc"}'   # ... on another checkpoint, kept for later runs
curl -s -X POST localhost:8091/prompt -d '{"prompt": "stack the blocks"}'   # running: fresh episode with it; idle: stored
curl -s -X POST localhost:8091/stop                                   # returns once the arms hold
curl -s -X POST localhost:8091/home                                   # refused (409) while a run exists
curl -s -X POST localhost:8091/reset                                  # accepted at once: let both arms go, reopen their drivers, then home; refused (409) while a run exists
```

The status dict, from `StationController.status()`:

| Key | Meaning |
| --- | --- |
| `state` | `idle`, `starting`, `running`, `stopping`, `homing`, `resetting` (letting the arms go and reopening the drivers, then `homing`) or `error` |
| `prompt` | The task the next run starts with, or the running one's |
| `checkpoint` | The checkpoint the next run's client is built with, or the running one's: the config's until a start names another |
| `arms` | Whether each arm's driver still answers, by name (`{"left": true, "right": false}`); a dead arm refuses a start until a reset |
| `run` | Runs started since the agent came up; the current run's ordinal |
| `chunks`, `stalls`, `clamps` | The loop's `chunk_count`, `stall_count` and `clamp_count` for the current (or last) run |
| `max_ee_speed_requested_m_s` | The largest end-effector speed either arm asked for in the current (or last) run, before the clamp; 0 before one |
| `max_ee_speed_m_s` | The configured cap |
| `session_id` | The dream client's DreamAPI session id (`client.session_id`), null before its first request |
| `last_error` | `{"type", "message", "code", "reason", "error_code"}` of the last failure, or null; `error_code` is DreamAPI's reason when the failure is a refusal, else the exception's class, and `key_refused` after a refused key brought the arms home (*Station agent*) |
| `timing` | `robots.control.RefillTiming` as a dict: `count`, `last_rtt_ms`, `p50_ms`, `p95_ms` over the run, and `serve_infer_ms` / `gateway_ms` / `network_ms` / `request_bytes` / `response_bytes` when the dream client has `last_timing`, else null |
| `started_at` | ISO 8601 UTC of the last start, or null |
| `uptime_s` | Seconds since the agent came up |
| `run_elapsed_s` | Seconds since the last start, or null |
| `cameras` | The age in seconds of each camera's latest frame, null before its first; the readers are open for the life of the process |
| `live` | Only with a `live` section (*Live view*): the Live View client's `stats()`, `linked`, `node`, `viewers` (null: the link carries camera subscriptions, not viewer counts), `streams` (the cameras being encoded), `pose`, `dropped_frames` per camera, `reconnects`, `errors`, `yields` and `live_paused` (`"tick"` while the tick guard has video paused) |

`rtt_ms` is measured on the loop's clock, from the tick that started a refill to the tick that
collected it, so it is quantised to a control period; the split comes from `dream` when it
reports one.

### Plan vs actual

The owner wants to see, live, the next actions the loop will execute and whether the arms follow
them. `ControlLoop.plan` is a `PlanSnapshot`, rebound on every commanded tick like `state` and
`video`: the remaining buffered rows per arm (`[rows, joints + 1]`, the arm's joints then the gripper, as the
policy sent them) with the time each is due relative to now (row `k` of a chunk is due at
`observed_at + k / control_hz`), the end-effector speed along those rows (from the kinematics,
computed once when the chunk is installed), and the last `PLAN_HISTORY_S` (3 s) of commanded
ticks at control rate: measured joints, commanded joints after ramp and clamp, the clamped
flag and the measured end-effector speed. Fixed-size ring buffers, no per-tick loop over rows.

Nothing in this package draws it: the station's own web page, which charted it, is gone, and a viewer
reads `ControlLoop.plan` from a concurrent task.

### Prompt change mid-run

`ControlLoop.set_task(prompt)` queues the change; the next tick on which no refill is in flight
(the client refuses a reset during a request) calls the client's `reset_episode()`, drops the
buffered rows (they answer the old task), makes the start-up ramp due again (a new episode's
first targets are near the home pose, as after a server-side session restart) and refills with
the new prompt. One INFO line per change; exactly one reset per change.

### Inference log

`inference_log_path` (config) or `--inference-log`; the agent defaults to
`~/robots-inference.jsonl`, `robots.run` writes none unless configured. JSON Lines, append mode,
one line per record, flushed per record, opened in the loop's `__aenter__` and closed in its
`__aexit__`, so `tail -f` follows a run and a crash loses nothing. Every record carries `kind`,
`t` (ISO 8601 UTC, milliseconds) and `run` (the agent's run number; 1 under `robots.run`).

| `kind` | Fields |
| --- | --- |
| `run_start` | `prompt`, `control_hz`, `action_horizon`, `refill` (`"client"`: the dream client's `refill_due` decides; an int: pinned, tests only), `sys0` (whether System 0 drives the motors) |
| `inference` | One per chunk that came back: `prompt` (the request's), `session_id`, `chunk` (its ordinal, null when discarded), `rtt_ms`, `serve_infer_ms`, `gateway_ms`, `network_ms`, `request_bytes`, `response_bytes` (the dream `last_timing` fields, null without it), `rows`, `mode` (the client's pacing, `"async"` or `"sync"`, from `dream.DreamClient.mode`; null for a client without it), `executed_since_obs`, `due`, `skipped` (the alignment numbers of the `chunk N:` log line; `due` / `skipped` null when not installed), `discarded` (null, `"late"` past `action_timeout_s`, or `"horizon_past"` when the arms already executed every row of it), `motor_temps_c` (`{"left": [...], "right": [...]}`, each motor's hottest sensor base to gripper, degrees C, on an arm on a motor bus; null on a YAM) |
| `error` | One per exception the loop caught or that stopped it: `error_type` (class name), `code` (`DA-xxxx` when a `DreamServerError` carries one), `reason`, `detail` (the message), `retry_after_s`, `action` (`retry`: kept draining and refills when due; `reset`: `DreamSessionExpired`, buffer dropped; `stop`: the loop ended with it) |
| `stall` | Once per empty-buffer burst: `chunk` (chunks so far; 0 before motion), `max_stall_ticks` |
| `run_stop` | `reason` (`stopped`, `cancelled`, or the exception class), `chunks`, `stalls`, `clamps`, `session_id` |

```
{"kind": "inference", "t": "2026-09-10T14:03:07.412+00:00", "run": 3, "prompt": "fold the towel", "session_id": "s-9f2", "chunk": 12, "rtt_ms": 1180.0, "serve_infer_ms": 900.0, "gateway_ms": 50.0, "network_ms": 230.0, "request_bytes": 118234, "response_bytes": 5120, "rows": 48, "mode": "async", "executed_since_obs": 35, "due": 35, "skipped": 35, "discarded": null}
{"kind": "error", "t": "2026-09-10T14:03:09.001+00:00", "run": 3, "error_type": "DreamServerError", "code": "DA-2000", "reason": "pool_warming", "detail": "DreamAPI refused the inference request: DA-2000 NO_CAPACITY (pool_warming)", "retry_after_s": 5.0, "action": "retry"}
```

### Home

Home is the pose runs start from: each arm's `home` (zeros on a YAM and a Maker Arm, the file's own on a Metal Arm
and a robot of your own), both grippers fully open (1.0, which an arm on a motor bus sends a little inside the open
end: *Motor guard*), so a held object is released rather than carried into the next scene. `robots.home.return_to_home`: each arm is read once and driven
linearly from that pose to its home over 3 s in 120 steps (40 Hz), written with `FollowerArm.write` directly at the
step rate (no command thread), then held. Best-effort and per arm: an arm whose
read or write fails is logged and left holding what it last accepted, the other arm finishes,
and the result is False; nothing raises for a follower fault
mid-move. The agent's **Home** runs it off the event loop, and a failed one's `last_error` names the arm and why, and
says to press Reset. The agent refuses it while a loop exists; `python -m robots.home --config
station.yaml` (or `station_run.sh home station.yaml`) runs it standalone, exit code 0 when
both arms made it, 1 otherwise, and must not be run while a loop owns the arms.

### Live view

With a `live` section in the config (`LiveConfig` above) and the `live` extra installed, the station builds one `dream.LiveViewClient` at boot (`robots.station._serve_opened`) and is then an ordinary caller of it, the same client and the same calls as any control loop of your own (`dream_client/README.md`, *Live View*, and the docs site's *Live View* page). Nothing about the link, the encoding, the face blur, the tick guard or the pose stream lives here: the client holds the robot link, encodes only the cameras a viewer watches, blurs faces (`face_blur` above, on by default) and guards the control rate, on its own niced thread, and the station's control tick never waits on it. A client that cannot start (the extra missing, no api key) stops the station at boot with the opened arms switched off.

The station feeds it from its own event loop, `LIVE_FEED_HZ` (60) times a second (`robots.station._feed_live`): `stream()` with the cameras' newest frames (`StationController.frames`, the readers that stay open between runs, so the scene streams while the state is `idle` as well as `running`), the state, and the keywords a station has and your loop may not: `prompt`, `run` (the run number), `model` (`(checkpoint, action_space)`), `steps`, `arms` (each arm's joints and gripper from the running loop's last read, or the drivers' cache between runs, None for an arm whose driver no longer answers) and `error` (why no actions reach the arms: in `error` the message and code of `last_error`, while running the DreamAPI error the loop is still retrying, `ControlLoop.refill_error`). While a run is on it sets the client's `dream_client` to the run's own `DreamClient` (each run opens one), so the page's Response chip shows that run's round trips and its session. The station passes `model_fixed=False`: the page's model and action pickers stay open, and a Start runs the model picked there.

The page's buttons come back through one `register_callback` that opts in to Home, Reset and the task (`robots.station._live_press`), each run on the station's event loop exactly as the HTTP API runs it: Start is `StationController.start(get_prompt(), *get_model(), get_steps())`, Stop is `stop()` followed by `home()` (the page's Stop meaning stop and go home), Home is `home()`, Reset is `reset()` and a new task is `set_prompt()`. A `StationError` or `ValueError` is the page's error line; a Stop or Home that ends in `error` answers with `last_error`'s message. The client answers the page within 10 s and shields Start, Stop, Home and Reset from that deadline and from the link dropping, so a home move or a driver reopen is never interrupted halfway; the next `status` says where it ended.

The run is recorded by the run's own dream client, which records every observation the loop hands it over the platform's record-only link (`dream._recorder`), on a platform that takes recordings (`LIVE_RECORD_ONLY` on); the Live View client records nothing, so nothing is sent twice.

Measured on a YAM station's computer (Ryzen 7 7730U, three D405 colour streams at 1280x720, 20 Hz config) with the station's cameras through a fake running controller and the Live View client, three viewers on a separate machine over the network watching three blurred cameras at 30 fps: each viewer receives 27 to 29 fps at 2.2 cores for the whole process, blur on, at 80 to 165 ms p50 capture-to-receive, no drops, no yields, one encode 2 ms. Two things keep it there. The capture thread only reads, converts and publishes, at a flat 33 ms cadence, and an encode thread per camera does the blur and the JPEG: blurring and encoding in line, with the client's own detectors sharing the thread pool, stalls the capture thread 100 to 130 ms five times a second, so the driver's queue drains in bursts and a 57 Hz poll sees 11 distinct frames a second of the 30 captured. And the client encodes each subscribed camera on a worker thread of its own, with the detector reading a copy no wider than `shared.faceblur.DETECT_MAX_WIDTH` (640), rather than every camera in turn on one poll. The process cost is the capture path, not the client: with OpenCV's default 16-thread pool the three capture threads alone burn about 7.8 cores (pool spin-wait); with the pool capped at 2 threads the same process runs at 1.7 cores idle, 1.9 with three cameras encoded and 2.9 with the blur on. `robots.cameras` caps the pool at `OPENCV_THREADS` (2) itself on every capture thread and in the agent, and defers to `OPENCV_FOR_THREADS_NUM` when an operator sets it; a bare capture loop on one D405 colour node goes from 85 to 8 percent of a core with the cap and still reads exactly 30.0 fps, with the every-sixth-frame blur from 299 to 41 percent.

What the operator sees: `status().live` (the table under *HTTP API*). Live View holds no second credential; the robot is registered by `dreamcli setup` (which writes `robot_id`), or by hand with `POST /v1/robots`.

## Your rollouts

Every run's dream client records the run on a platform that takes recordings (`LIVE_RECORD_ONLY` on): every observation the loop hands it, cameras and state, goes up the platform's record-only link into its store (`dream._recorder`), live section or not. A copy on your disk is off by default. `rollouts: {enabled: true}` in the config turns it on, for a station's Start and for `dreamcli robot run` alike (both use the same loop, so both write the same files); a `dreamcli robot run` is recorded by its dream client as a station's run is. A `--dry-run` is not saved. The path is printed when the run starts (`rollouts: saving this run to ...`; a station writes it to `station.log`, `dreamcli robot log`). Unless you changed `rollouts.dir`:

```
~/.dreamlabs/rollouts/2026-10-12T14-03-07_run3/     one folder per episode (one prompt, one run)
```

Each folder is a LeRobot v2.1 dataset of one episode, so it loads as it is (`LeRobotDataset(repo_id, root=folder)`),
and the check `dream.datasets` runs before it uploads a dataset accepts the whole `rollouts` folder as it is:

| What | Where |
| --- | --- |
| Every camera, at the size the camera delivers, one VP9 MP4 each | `videos/chunk-000/observation.images.<camera>/episode_000000.mp4` |
| Measured joints and gripper of both arms per tick (14 numbers: left joint 0 to 5, left gripper, then the right arm) | `data/chunk-000/episode_000000.parquet`, column `observation.state` |
| What the loop commanded per tick, same 14 numbers, after the start-up ramp and the speed clamp | same file, column `action` |
| Time of each tick (`timestamp` is `frame_index / fps`; `rollout.t_monotonic_s` is the loop's own clock, with the gaps of any stalled ticks), the chunk and the row of it that was commanded | same file |
| Prompt, checkpoint, robot name, body, rate, start time, how recording ended (`stop_reason`) and the message of the error that ended the run (`error`: DreamAPI's refusal, an arm's alarm) |  `rollout.json` |
| The loop settings that change what the arms do or when a run gives up: `max_ee_speed_m_s`, `max_gripper_step`, `grip_force_cap`, `max_stall_ticks`, `action_timeout_s` (the rate is `fps`) | `rollout.json` |
| What the loop did to the plan over the episode: `clamps` (ticks the speed or gripper clamp bounded a command), `max_ee_speed_requested_m_s` (the fastest either arm asked for, before the clamp), `stalls` (ticks with nothing to command, the arms holding), `left_max_motor_temp_c` and `right_max_motor_temp_c` (the arm's hottest motor sensor at any inference; null on a YAM) | `rollout.json`, written when the episode ends |
| The model's reply to every inference, as it came back: the joint chunks and, on a Maker or Metal Arm, the tool poses | `chunks/reply_000012.npz` (the 12th inference; `numpy.load`), one per `inference` record |
| The inference log of the run | `inference.jsonl` |

The videos hold one frame per control tick, the newest the camera had at that tick, so they play at the loop's rate
(20 Hz by default), not at the camera's own.

Did the speed cap slow a run? Then `clamps` is above 0 and `max_ee_speed_requested_m_s` is above `max_ee_speed_m_s`.
An arm's alarm that stopped it (a motor too hot, a joint blocked) is `stop_reason: SafetyStop`, the motor and the sensor
in `error`. An episode recovered after a crash has the settings but not the counts.

Size: about 1.7 MB per camera per minute for a moving, noisy test scene at 640x480 and 30 Hz (5.6 MB a minute with
three cameras, measured), more for a busier real scene or bigger frames. Videos are saved as recorded.

**Sending them to Dream Labs**: a linked station's runs are on the platform already. To send a copy saved here, zip the folder and send it to your Dream Labs contact.

```bash
cd ~/.dreamlabs/rollouts && zip -r rollouts.zip .    # or just the episodes you want to send
```

The recorder itself uploads nothing.

**If something goes wrong.** Recording never stops a run or slows it. If the disk fills up or a write fails, the episode
stops recording at that point, the log says so (`rollouts: ... stopped recording`), and what was saved is kept with
`status: partial` in `rollout.json`. A run needs 2 GB free to start; with less it refuses to start and says so. If the
machine loses power or the process is killed, the folder keeps an `INCOMPLETE.txt`; the next run turns what was saved into a loadable `partial` episode, losing the last fraction of a second.
Leaving out the `rollouts` section, or `rollouts: {enabled: false}`, turns all of it off. The recorder needs PyAV (its libvpx VP9 encoder) and pyarrow, which
the dream library (the platform's download) and `dream-robots` from a checkout both bring.

It stays off the control tick: the loop's calls into the recorder only put references on queues, and a journal thread and one VP9 encoder thread per camera do the work.

## Motor guard

Every arm on a motor bus (a Maker Arm, a Metal Arm, an SO-101) runs `robots.motor_guard` on each tick of its thread,
on what the tick reads and what it is about to command:

- **A blocked joint.** A joint whose command leads its angle by 12 degrees or more while it has not moved for half a
  second is pushing on something it cannot move (the camera mount, the table, the other arm). Its goal is held 5
  degrees past where it stopped, what holding itself up takes, instead of pushed at full gain; a goal that turns back
  frees it. Held 5 s, it raises the arm's alarm and the run stops (*Failure handling*). The same watch runs while the
  arm holds with no target, between runs and after a home move.
- **A gripper blocked opening.** A gripper that stops 6 mm or more short of its open end (`gripper_open_deg` set past
  the real stop) is held a millimetre past where it stopped, and the log says where to set `gripper_open_deg`.
- **A gripper blocked closing.** On an SO-101, whose gripper has no holding law, a gripper closed on an object is held
  3 units into it, a gentle squeeze, instead of at full position gain until its overload protection trips. A Metal
  Arm and a Maker Arm hold at `gripper_hold_torque_nm`.
- **A gripper never rests on its open end.** The open end is where the gripper stood pushed open by hand against its
  stop (the wizard's `gripper_open_deg`, an SO-101's `range_max`), so a goal on it presses into the stop for as long
  as it holds. Every gripper goal (the
  model's 1.0, home, the park, a hold, the arm's first tick) stays inside it by the body's margin: 2 mm on a Metal Arm
  and a Maker Arm, 3 of 100 units on an SO-101. The reading says where the gripper is (0.98 fully open on a Metal
  arm). The closed end stays reachable for a grasp on a thin object.
- **Temperatures.** Every motor's sensors: a Damiao motor's coil and driver board (Metal), a RobStride's driver
  (Maker Arm), a Feetech servo's own (SO-101, read once a second). One at the body's warning temperature is logged once;
  one at its stop temperature for 1 s raises the alarm, naming the motor and the sensor, until every sensor is back
  under the warning temperature. An arm with a motor at or above the warning temperature refuses to open (a boot, a
  Reset, a new run, the setup wizard), naming the motor and how warm it is, before anything is switched on. The Metal Arm
  and the Maker Arm warn at 55 C and stop at 65 C; an SO-101 warns at 50 C and stops at 60 C. The inference log's `inference` records carry them.
- **A motor that stops answering.** On a Maker Arm, one motor silent for 1 s while the others answer (its power or
  its CAN cable) raises the alarm naming it, since its reading is stale.
- **A pose that is not fresh.** An arm whose thread has published no pose for 1 s (a fault word that does not clear,
  a bus call that does not return) refuses the read with `ArmError` instead of handing the loop an old one, so a run
  never drives blind; the run then ends as for an arm fault.

A guard's alarm never latches the arm: it still answers, so the run that reads it stops with `SafetyStop` and the arm
parks (`robots.run` homes both arms first; a station parks the alarmed arm and switches it off, and Reset opens it
again). Between runs the station looks at the alarms every second and parks an alarmed arm the same way.

## Maker Arm

`body: makers` (`configs/makers.example.yaml`) runs the same loop and station agent on a MakerMods Maker Arm:
seven RobStride motors per arm (six joints and the gripper) on one CAN adapter each, no brakes. The model is a FaaS
fine-tune on the Makers datasets that answers with tool poses; the dream client turns them into joints. Install the
`dreamlabs` distribution on Python 3.11 or newer, which carries python-can and dream's IK; `dream-robots[makers]` adds
python-can but not the IK, which no package index carries. Everything that differs from a YAM is one small
description, `robots.bodies.MAKERS`; the loop itself has no Maker Arm branch.

The arm itself is a table, `robots.makers.ARM`: its six joints (CAN id, motor model, gains, soft limits), the
gripper's linear stroke, 60 Hz, where it parks and dream's IK name. The driver is the thread every CAN bus arm shares
(`robots.motor_arm.MotorArm`) over a RobStride chain (`robots.robstride.RobstrideChain`); together they are a
port of Dream Labs' Maker Arm driver, with what a
policy run needs and every safety check it has, and held to that driver's CAN frames byte for byte by
`tests/test_makers_parity.py`. Each arm has its own thread at 60 Hz that steps toward the loop's newest target: from the measured pose at 30 deg/s until it has caught up, then at most
120 deg/s (the gripper 40 and 120 mm/s), every target clamped to the soft limits first. A joint a whole turn off
after a power cycle is corrected for the session; one outside its limits by anything else is a stale zero, and the
arm refuses to switch on. A target older than 0.5 s is dropped and the ramp re-armed, so a stream that resumes eases
in. A motor silent for 1 s raises the arm's alarm (*Motor guard*); an arm whose every motor is silent for 0.3 s
(arm power, e-stop, cable, adapter) is latched down and reads dead, and Reset reopens it. The gripper holds a grasp at
`gripper_hold_torque_nm`. Not ported: teleoperation, calibration, the adapter reopen after an I/O failure (here that
latches the arm down), and the raw-USB adapter transport.

The wire is the Makers datasets' (`robots.adapter.EefWire` over `MAKERMODS_EMBODIMENT`): per arm
`state.<side>_arm` (`[6]` radians), `state.<side>_gripper` (`[1]`, the stroke in mm, 0 closed to 80 open) and
`state.<side>_grip` (`[1]`, the same as a fraction, 1 open), and a frame per camera (whose window the dream client keeps) under the name the model reads
: `video.overhead` for a base model, `video.scene` for a fine-tune on the platform's Makers or
Metal embodiment, `video.head` for Dream Labs' shared fine-tunes, each with `video.wrist_l` and `video.wrist_r` (or
`left_wrist`, `right_wrist`). `MAKERMODS_EMBODIMENT` takes each camera under all of those names, so the station names
its cameras for its model and the block carries those names only; one camera under two names is refused. The
embodiment is that embodiment config (34 state and 32 action dims), which is how strict mode reads the wire. The
dream config carries `ik: {arm: makers, head: eef}`: the client adds each arm's tool pose from its joints
(`state.<side>_eef_xyz`, `state.<side>_eef_rot6`) and solves the reply's tool poses (`per_key`) to the
`q_r_chunk` / `q_l_chunk` the loop reads, with dream's own IK under the arm's 120 deg/s joint caps. Chunks are joined
in joint space, after the IK, by the client's seam join, as on a YAM. `action_hz` is the loop's rate, 20 by default.

The end-effector speed clamp measures the tool centre point with dream's own forward kinematics of the arm
(`robots.kinematics.DreamKinematics`, the model the client's IK solves on; about 0.13 ms a call, a few seconds to
build once). `grip_force_cap` does nothing here: the arm reports no gripper effort.

A Maker Arm has no default model: a Start without a checkpoint runs the file's `checkpoint`, and is refused (`no
checkpoint`) when the file has none; `robots.run` likewise takes `--checkpoint` or the file's. On exit each arm first returns to rest at 30 deg/s (every joint at zero as far as its soft limits
allow, the gripper open) and only then switches its motors off, one arm after the other, since an arm without
brakes would fall; an arm latched down cannot park, so it is left holding under power (support it, then switch its power off). `dreamcli motors` is for YAM stations. The setup
wizard sets up a Maker Arm too; for its steps by hand the arm goes limp (`MotorArm.disengage()`), so hold it
(`docs/SETUP_WIZARD.md`).

## Metal Arm

`body: metal` (`configs/metal.example.yaml`) runs the same loop and station agent on a MakerMods Metal Arm: seven
Damiao motors per arm (six joints and the gripper) on one CAN adapter each, no brakes. As on a Maker Arm, the model
is a FaaS fine-tune that answers with tool poses, the dream client turns them into joints with dream's IK, and
everything that differs from a YAM is one description, `robots.bodies.METAL`. Install the `dreamlabs` distribution on
Python 3.11 or newer; `dream-robots[metal]` adds python-can but not the IK.

The arm is a table, `robots.metal.ARM`, run by the same shared thread as the Maker Arm (`robots.motor_arm`) over a
Damiao chain (`robots.damiao.DamiaoChain`, with `robots.metal_gripper`). The chain is MakerMods' own Metal driver,
from the control loop they run policies with (their LeRobot fork's `MetalFollower`, MakerMods Lab's gripper
holding, their console's `MetalRobot`), held to the CAN frames
their code writes, byte for byte, by `tests/test_metal_parity.py`. Open sets the gripper up (MIT mode, a 500 ms CAN
timeout, its temperatures and MIT scaling checked), reads every motor, refuses a joint more than 5 degrees past a
soft limit or one still energized, and preloads every joint where it is. Every tick reads all seven motors (a missed
reply is asked once more, never served from the cache) and requires each energized; every command is clamped to the
soft limits, eased in by MakerMods' startup sync (a degree a tick until within 3 degrees), and sent with their
velocity feedforward at their gains (Kp 100, 100, 100, 50, 20, 20, Kd 5, 5, 5, 5, 0.6, 0.6, the gripper 20 and 0.6).
One thing departs from that follower: each joint's frame carries the arm's gravity torque at
the tick's pose (`robots.metal.gravity_nm`, the links of MakerMods' `metal_with_gripper.urdf` with their SDK
follower's factors). Without it the shoulder sat 0.1 rad under its target and a lifted hand stayed on the table. The
SDK's stiffer gains are not taken: they go with its 120 Hz smoothed command stream, and on this 30 Hz tick they made
the arm jerky. The parity test runs with the torque off.
The gripper holds a grasp at `gripper_hold_torque_nm` and stops on a fault, 55 C on either sensor or
a command on feedback older than 0.3 s (counted from the tick's own read, so a tick the station holds up is not a
fault): it is switched off and the arm's alarm says why while the joints hold, so the run ends and the arm parks
(*Motor guard*); a stopped process leaves it energized at
most 500 ms. Around that, each arm's thread at
30 Hz does what it does on a Maker Arm: it eases onto a new target from the measured pose at 30 deg/s (the gripper
40 mm/s) until it has caught up, then steps at most 120 deg/s (120 mm/s), drops a target older than 0.5 s and re-arms
both ramps (its own and the chain's startup sync) for the next, and latches the arm down when a tick fails. The
shared thread runs the Maker Arm numbers, not MakerMods' own ramp and cap; what the chain sends for a given goal is
MakerMods', and `tests/test_metal_parity.py` holds it to their frames byte for byte.

Units: joints are the motor angles in radians (MakerMods' frame, sign +1, no offset), which is dream's Metal model;
the soft limits are its URDF's. The loop's gripper is the Metal grip, the jaw stroke in mm over 100 (0 closed, 1
fully open), what the Metal datasets and fine-tunes use. It runs linearly in the gripper motor's angle from the arm's
`gripper_closed_deg` (0) to its `gripper_open_deg` (1, 100 mm), both required (the setup wizard measures them by hand):
every bus arm's gripper is that one linear map between its two ends (a per-arm `gripper_stroke_table` is refused as
an unknown key). A command is clipped to `[0, 1]`. The motors keep their zero, set once with MakerMods' calibration at the zero pose (upright, every joint at 0,
the gripper closed); their LeRobot calibration file carries nothing the driver reads.

The wire is the Maker Arm's (`robots.adapter.EefWire` over `MAKERMODS_EMBODIMENT`), with
`state.<side>_gripper` the stroke in mm (the grip times 100); the dream config carries `ik: {arm: metal, head: eef}`,
and the speed clamp measures the tool with dream's forward kinematics of the Metal Arm. A Metal Arm has no default
model. On exit each arm returns to the pose it was opened in at 15 deg/s, as MakerMods' console does, with the gripper
open, and switches its motors off once it is there (every joint within 5 degrees, the gripper within 8 mm, as on a
Maker Arm), stalls for 2 s, or runs out of time; an arm latched down cannot park, so it is left holding under power (support it, then switch its power off). Which
of dream's arms a body solves with (`metal`, or `metal_faas`) is its table's `ik` (`robots.metal.ARM`), one line. The station's Home moves each arm to its `home`, which a Metal file must give (its zero pose is
upright). Not ported: teleoperation, calibration, the console's planner and tracking-error check, the
gripper's current-limited mode, and the raw-USB adapter transport.

A camera pole or mount an arm can reach needs a `keep_out` entry (*`ArmConfig`*): dream's model of the Metal Arm keeps
it off nothing.

## SO-101

`body: so101` (`configs/so101.example.yaml`) runs the same loop and station agent on a pair of SO-101 arms: five
joints and a gripper on six Feetech STS3215 servos per arm, one serial bus each, no brakes. The model is a FaaS
fine-tune that answers with tool poses; the dream client turns them into joints with dream's IK of the SO-101
(`so101_faas`). Install `pip install "dreamlabs[so101]"` in the dream environment on Python 3.12 or newer: the extra is LeRobot's Feetech
bus (`lerobot[feetech]`, which brings torch), and only `robots.feetech` imports it. Never `pip install lerobot` on its
own into that environment (a newer LeRobot pulls a `click` the library forbids); for LeRobot's dataset tools too,
install both in one command: `pip install "dreamlabs[so101]" "lerobot[dataset]"`.

The arm is a table, `robots.so101.ARM`, run by the same shared thread as the Maker and Metal Arms
(`robots.motor_arm`) over a chain on LeRobot's own bus (`robots.feetech.FeetechChain` on `FeetechMotorsBus`). The
chain does what Dream Labs' SO-101 follower and policy bridge do: the port
must be a `/dev/serial/by-id/` or `/dev/serial/by-path/` path; at open it reads the calibration the servos hold and refuses the arm, before
writing anything, when that differs from the arm's calibration file (the other arm's file, or a recalibration since);
it writes LeRobot's SO-101 setup with the servos off (position mode, P 16, I 0, D 32, the gripper's torque and current
caps) and switches them on at the pose they hold. Every tick reads all six positions in one sync read and writes all
six goals in one sync write, and every second all six temperatures (*Motor guard*). A failed read holds the arm for
the tick; an arm with no read for 0.3 s is latched down. The switch-off goes servo by servo, so one in
overload does not keep the others on, and the error names it.
Around that the thread runs at 60 Hz and does what it does on the other bus arms: it eases onto a new target at
30 deg/s, then steps at most 120 deg/s (the SO-101 policy bridge caps at about the same), clamps every target to the soft limits
(the calibrated range), drops a target older than 0.5 s, and latches the arm down when a tick fails.

Units: joints are radians in the URDF's frame, 0 at the calibration's middle pose (2048 ticks), which is the zero of
dream's `so101_faas` model, so a joint in degrees is `(ticks - 2048) * 360 / 4096` on any calibrated arm. The arm fills
the loop's first five joint slots; the sixth reads 0, and dream's IK holds the model's dummy sixth joint there. The
gripper is linear between the ends of its calibrated range, closed at `range_min` (0) and open at `range_max` (1).
The servos report no velocity, so velocities read 0.

The wire is the embodiment your fine-tune was trained on (the config's `embodiment`, looked up on your account), and
what the loop fills is dream's spec of the arm (`robots.kinematics.arm_layout`): per arm `state.<side>_arm` (the five
driven joints, radians) and `state.<side>_grip` (`[1]`, 1 open), and a frame per camera under `video.<role>` (whose
window the dream client keeps), named after its `video` roles. The dream config carries `ik: {arm: so101_faas, head: eef}`: the client puts the held sixth joint
back (the embodiment records six), converts the joints to
the units the checkpoint was trained in (an SO-101 station's LeRobot units) and makes `<side>_gripper` (0 to 100) from
the grip, then solves the reply's tool poses back to radians. The speed clamp measures the tool with dream's forward
kinematics of each side's own URDF. `grip_force_cap` does nothing here: the arm reports no gripper effort, and its
gripper servo's torque cap does that job.

An SO-101 has no default model: name your fine-tune of the embodiment the file names. On exit each arm
returns to the pose it was opened in at 20 deg/s with the gripper open, then switches its servos off; an arm latched
down cannot park, so it is left holding under power (support it, then switch its power off). The station's Home moves each arm to its `home`, which an SO-101 file must give.
Calibration is the setup wizard's: once both arms are found, `dreamcli setup` calibrates each SO-101 arm whose servos hold none of the operator's files (`robots.so101_calibration`, `docs/SETUP_WIZARD.md`, *Calibrating an SO-101*), and writes the same file `lerobot-calibrate` would. Not ported: teleoperation, and the which-arm and home-capture scripts (the setup wizard's job).

## A robot of your own

`body: custom` (`configs/custom.example.yaml`) runs a robot we do not ship on everything we do: the loop, the station
or `dreamcli robot run`, the safety caps, the dream client's IK and the wire to your FaaS fine-tune. What is yours is
one small driver class, a `robots.followers.Driver`:

| Method | Does |
| --- | --- |
| `__init__(arm_config, side)` | Opens one arm on `arm_config.can_channel` (whatever device it is), powered and holding where it is. `side` is `"left"` or `"right"`, or the adapter's id while `dreamcli setup` is still finding which arm is which, so per-side calibration cannot be looked up at that point |
| `read()` | Returns a `Reading`: a position and a velocity per joint (as many as the arm's `home` lists) in radians and rad/s in your arm's URDF frame, the gripper 0 closed to 1 open |
| `write(positions_rad, gripper)` | Takes a target in the same units, already checked and speed-capped by the loop; send the gripper's 1.0 a little inside its open stop, since home holds it there between runs |
| `close()` | Releases the device without moving the arm |
| `torque_off()` | Switches the motors off and releases the device; park first if the arm would fall |
| `disengage()`, optional | Lets the arm go limp where it is and keeps reading, for the setup wizard's steps by hand; the next `write()` switches the motors on holding where the arm is and drops that first target. Without it (the default returns False) the arm is held stiff instead |
| `drops_when_limp`, a class attribute | True for an arm without brakes: a reset lets it go with `torque_off()` (so that must park it first, or it drops on every reset), an arm fault in a run holds the arms rather than disengaging them, and an arm that stops answering while it holds is reported as still holding under power (`FollowerArm.holds_dead()`) |

`robots.driver_template.TemplateArm` is the commented starting point: as shipped it is an arm in memory that reads
back what it was told, which runs the whole stack with no robot attached, and its docstring says how read() and
write() wrap a LeRobot `Robot` (`get_observation`, `send_action`), an SO-101 for one. `close()` and every path that
disconnects must never command motion: a vendor's LeRobot plugin may home, rezero, power or move the arm in
`connect()` and `disconnect()` (Seeed's `lerobot_robot_seeed_b601` may rezero every motor, enable them all and close
the gripper on `connect()`, and moves the whole arm to zero on `disconnect()`), so read its source
first, or drive the motor SDK or the plugin's lower-level bus without either. The config names three
things in its `custom` section: the driver (`my_arm.py:MyArm` beside the config file, or an importable
`package.module:MyArm`), `ik`, dream's name for your arm's URDF (the client solves the model's tool poses to joints
with it, and the speed clamp measures with it), and `embodiment`, the one your fine-tune was trained on, by its
`<username>/<name>`. The embodiment is looked up on your account (`dream.DreamClient().get_embodiments()`) the first
time a run needs it, and its config is the wire (`robots.adapter.EefWire`): per arm `state.<side>_arm`,
`state.<side>_grip` and, if your datasets record one, `state.<side>_gripper` (then `grip_scale` says what it reads
fully open), and a frame per camera under `video.<role>` (whose window the dream client keeps), named after its `video` roles. Each arm section takes the device
and a `home`, required, and the gripper's ends (`gripper_closed_deg`, `gripper_open_deg`, both or neither, NaN in
`cfg` without) when `dreamcli setup` measured them: give your driver a `gripper_deg()` (the template's says how) and
the wizard's home step records each gripper closed and open by hand, for your driver to map the gripper with. Start with `dreamcli robot run --config <file> --task "..." --checkpoint <your fine-tune>
--dry-run`: it reads the arms and logs their joints beside what the model asks, writing nothing.

Limits today: two arms of 1 to 16 joints each (`robots.followers.MAX_JOINTS`, the live view's
`shared.contracts.live.TICK_MAX_ARM_DOF`) and a gripper. An arm has as many joints as its `home` lists, and its driver
reads and takes that many; a read or a command of another count stops the run naming the arm. Both arms run on dream's
one model of the arm (`custom.ik`), since dream's IK solves both sides as one array of joints; a `home` longer than the
model holds the extra joints at 0, and the wire sends only the joints dream's spec of the arm drives. The dream
client's IK takes the arms in `dream.client_ik.ARMS` (Metal Arm, Maker Arm and the arms FaaS trains on) and arm files
(`dreamcli tcp`).
`robot up` brings no link up for a custom robot: your driver opens its own device.

`examples/` holds worked robots of your own, a driver and a config each: the Seeed reBot (`examples/seeed_rebot/`) and
the Trossen Stationary AI (`examples/trossen/`). Its `README.md` says what each shows and carries a prompt that has
Claude Code write a driver from them; the platform serves the folder under `/downloads/examples/` to an account's key.

## Development

```bash
cd dream_library/robots
make test          # uv run pytest -q; no hardware needed (cv2 is used to encode and decode test frames)
make lint          # ruff check + ruff format --check
make typecheck     # mypy --strict over src/robots
make check         # lint + typecheck + test + doxygen, what CI runs
make doxygen       # docstring check + Doxygen (warnings as errors) into documentation/html
```

`src/` layout, so tests run against the installed package. The venv is managed by `uv`
(`uv sync` once; CI uses `uv sync --frozen`). Tests replace i2rt, the capture device and the
dream client with fakes in `tests/conftest.py` and drive the tick with an injected clock, so no
test waits out a control period in real time (thread hand-offs poll with millisecond sleeps);
`tests/test_end_to_end.py` runs the real `dream` client in strict mode against a threaded HTTP
fake of DreamAPI's `/v1/get_action` and asserts that what reaches the wire is the serve's
observation; `tests/test_station*.py` drive the controller over the same fakes and the HTTP API
over a real socket on an ephemeral port; `tests/test_cli.py` runs `robot up` / `down` / `log` on a stand-in for the agent
(a script that waits for SIGINT) and `robot run` over a stand-in loop, `tests/test_custom_body.py` loads the custom example and
runs the shared loop and `robots.run.main` (a real run and a dry run) on the template's arm in memory, `tests/test_so101.py` runs the SO-101 config, chain and body over a fake of LeRobot's bus (`tests/so101_fakes.py`) and checks the observation against the real dream client's IK when dream's IK is installed, `tests/test_setup_wizard.py` drives the wizard's routes over a real socket
and loads what Save wrote through `robots.config.load`, `tests/test_setup_flow.py` plants each driver mistake in fake
arms (`tests/setup_fakes.py`) and checks the wizard catches it at its step, and `tests/test_station.py` drives the station's Live View feed and buttons over a fake client (the client itself is tested in `dream_client/tests/test_live.py`, and the whole path over a real relay in the repository's integration tests). The clamp and plan tests inject a linear stand-in for the
kinematics; `tests/test_kinematics.py` runs the real YAM model once when `mujoco` (in the dev
group) and `i2rt` (the `hardware` extra) are both installed and is skipped otherwise. Every module, class and public function carries a Doxygen-form docstring per the
house standards; `ruff` runs at line length 119.

## Operator scripts

`debug_scripts/` holds station-side scripts that touch the machine rather than the code: `station_run.sh` (`serve`,
`start`, `stop`, `home`, `log`: the agent or the one-shot loop in a tmux session, and the standalone home move),
`preflight.py` (the arm model's zero-pose forward kinematics, cameras and two real inferences with the arms off) and `hwmon_can_enable.sh` (bounces the CAN links
with auto-recovery and error reporting on when they are stuck error-passive). See `debug_scripts/README.md`. A
station set up with `dreamcli setup` runs the agent with `dreamcli robot up` instead of `station_run.sh serve`
(*Command line*); `preflight.py ~/.dreamlabs/station.yaml` still checks its config with the arms off.
