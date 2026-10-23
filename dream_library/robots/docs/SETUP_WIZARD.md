# The setup wizard (`dreamcli setup`)

The setup wizard is the page a station is set up with. It works for any robot dream has a URDF of, and it answers one
question: **does your driver read and write joints the way dream does?** The policy sends joints in the units of
dream's own URDF of the arm (`dream_ik`): radians, the URDF's zero, the URDF's direction for each joint, the URDF's
joint order, and the gripper as 0 closed to 1 open. A driver that is off in any of these runs, but the arm does
something else than the model meant. The wizard draws dream's URDF at the numbers your driver reports, moves the arm
a little under power from the home you set, and says what is off and how to fix it.

Code: `src/robots/setup_*.py` (the module map in the README lists each), the page `setup_wizard.html` with
`setup_wizard.js` and the 3D arm view `setup_viewer.js` (*The 3D arm* below). Tests: `tests/test_setup_*.py`.

## Opening the page

On a box with a screen, `dreamcli setup` opens the page in its browser and prints only that it did, with the one address on this box to paste if the browser did not open (`robots.setup_access.local_lines`): a beginner at the robot's computer reads a list of network addresses as something to do. Without a screen, or over ssh, it prints the addresses to open the page at (this box, its network name, its interface addresses, or an `ssh -N -L` tunnel), each carrying the page's token after `?token=`; a request without the token gets 403 (`robots.setup_access`). The token is fresh on each run, so the address changes every time, unless the environment variable `DREAM_SETUP_TOKEN` sets it. Set it once, in `~/.bashrc` or in the shell that runs `dreamcli setup`; this line writes a fresh one there without showing it, and a new shell then has it:

```bash
echo "export DREAM_SETUP_TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(16))')" >> ~/.bashrc
```

From then on every run prints the same address (on the same port), so bookmark it. A value shorter than 22 characters, or holding anything but letters, digits, `-` and `_`, stops the command with a message that names the variable, never its value. Anyone with the address can power the arms, so keep it private: not in a shared channel, a screenshot or a ticket.

The page's footer shows this install's version (`robots.__version__`). The dream library the platform builds carries
the commit it was built from as `0.0.3+g<first 12 of the commit>`; `dreamcli setup` compares that commit with the one
the platform runs (`commit` of its `GET /v1/meta`, when the platform returns it) and prints a warning
when they differ: install the library again from the Setup API page, so the page is the platform's build. A source
checkout asks nothing.

## The flow

| Step | What the operator does | What the wizard checks | What it keeps |
| --- | --- | --- | --- |
| 0. Select robot | First, *Use the setup wizard* (the steps below) or *Set it up myself* (*Set it up myself* below: the file and the commands, no step and no check). Then YAM, Maker Arm, Metal Arm, SO-101, or "my own robot" (a driver of your own plus the dream URDF it matches). For an SO-101, picks its embodiment from the account's, and ticks LeRobot calibration files of its own only if it recorded data with them: each SO-101 serial adapter plugged in is read once, read-only with no servo switched on (`robots.setup_session.so101_matches`), and a file its servos hold comes ticked and says "Matches the arm on <adapter>"; an arm that holds none is calibrated on step 1; for one's own, the driver, picked from the station's own folders, then dream's model of the arm by one of two large choices, an arm Dream Labs has a model of (none picked until the operator picks one, or the saved file's on a re-run) or its URDF (*Your driver and the arm model* below), then for a URDF the arm's end and its tool point on a 3D drawing with buttons (*An arm from its URDF* below), and the embodiment, the account's or a new one (*A new embodiment* below). No model is asked for (*The model* below). The page reminds, on every robot but the SO-101: make sure your motors are calibrated (zeroed) the way your robot maker says before you start. A robot that cannot be set up on this install is greyed out with the reason. | The URDF builds. | `body`, and the camera names of its embodiment. |
| 1. Connect arms | Ticks the adapters that fit the robot (*Finding the arms* below), presses *Release the arms*, then moves one arm by hand, a little, any joint, and says which arm it is; then the other. An arm the scan does not list is typed by the address its driver opens (*Finding the arms* below). | Which adapter's readings changed (every ticked adapter is open and limp while you move). An existing file that drove the other arm on that adapter means left and right were swapped: the page says so. An SO-101 whose servos hold none of the ticked calibration files still opens, to be found, and once both arms are found the page calibrates it (*Calibrating an SO-101* below); the home pose and every step after it wait until both arms are calibrated. | Each arm's adapter, by its serial number, and on a YAM a udev rule naming each arm's CAN link after that serial. |
| 2. Set up cameras | Clicks the role of each camera from a picture of every feed, then compares each live view with a reference: a frame from your own LeRobot v2.1 dataset, or our shipped YAM pictures. A camera the scan does not list is typed: an address OpenCV opens, or a camera class of your own (*Finding the arms* below). On a YAM those are the views its model learned from; on any other robot they are only an example of where such a camera sits, and the page asks for your dataset to compare with. Also shown: what the model sees (the frame at the policy's size). A camera that stops sending pictures (unplugged, a USB drop) is opened again, sooner first and then every 8 s, and its feed says so until it is back. Each RealSense's picture says the USB link it is on (`USB 3.2`); one on a USB2 link says it is too slow for 1280x720 at 30 fps and to plug it into a USB3 port with a USB3 cable, and the same words end the error when it does not open. | Two wrist cameras given each other's roles (the left feed looks more like the right reference, and the other way round). | The camera rows. |
| 3. Move to home | On a Maker or Metal Arm, or a robot of your own whose driver reports its gripper motor's angle, first opens both grippers fully by hand and presses Record open, then closes them and presses Record closed (*Gripper ends* below). Puts each arm in its home pose, the pose every run starts from, and presses OK (*Home* below). The arms stay limp. The page draws dream's model of each arm at the readings now. | Joints read outside the URDF's range are named; the operator may continue anyway. A gripper range under 20 degrees is refused. | `home` per arm, one angle per joint of dream's model of the arm in radians, and the gripper ends when they are measured here. |
| 4. Set reach area | Only on a robot the dream client solves for (not a YAM, which skips to Test joints). "Move each hand over the whole area it should work in. Paint one solid area in front of the arm, and stop short of your camera mount so it stays outside the edge. dream pulls the hand's targets into the area, and if the arm still heads out past the edge, it stops there for the rest of that move." The arms stay limp; the page draws each arm from above with the area painted so far (*Reach area* below). Skip asks "Are you sure?" first. | Nothing; nothing here can block Skip or Save. | `reach` per arm, unless skipped. |
| 5. Test joints | Reads the safety popup ("Clear the workspace. Stand outside the arm's reach. Keep a hand on the emergency stop (or the power switch).") and confirms. A big Stop stays on screen. Watches the real arms and the two drawings: during the nudges one joint moves at a time, and the drawing moves the same joint at the same moment and lights it up. Then answers "Did each joint and the hand move like the drawing?" | The hold check, a +5 degree nudge of every joint (one at a time) and the gripper, then a small hand sweep: about 3 cm along x, y and z and 10 degrees about each axis, solved with dream's IK and judged by forward kinematics of where the arm got. A No from the operator fails the step. | Nothing. |
| 6. Save station | Saves. If the powered checks did not all pass, or the reach area was skipped, the page says so but still lets you save. Finish ends the wizard; with the arms still powered it first asks, since switching them off moves each arm to its rest (a Maker Arm to zero, a Metal or SO-101 arm back to where the powered checks started) before it goes limp and drops. Then the page, and the terminal, show the next step: `dreamcli robot up` to copy, and a link to Live view, which a box with a display opens in its browser. | The file loads as the agent will load it. | `station.yaml` (with each arm's gripper ends, and no model), and the robot registered on the platform as its embodiment's id. |

The page is read in a browser that need not be on the station: an operator on a laptop SSH'd into the robot computer opens it from there. So every command the page gives says to run it on the station, in its terminal or the SSH session, the file pickers and dataset path are the station's, and only an uploaded URDF comes from the laptop. `dreamcli setup` itself says to keep the SSH session open, since closing it stops setup.

Every step is its own card, and the page shows the current one alone, beside a list of the steps on the left. The page's own first step, Choose setup method (`setup_page.js`), asks *Use the setup wizard* or *Set it up myself* before Select robot, and every number on the page counts from 1 with it; a page that reopens on a later step (a rerun) passes over it. The list goes back to a done step (the script's `go`, as `window.setupGo`); a later step waits, greyed, until the ones above it are done, and within a card a blocked button names what it still needs. In Connect arms, once the arms are let go, a Left arm and a Right arm card say whether that arm is found yet and, once it is, which adapter it is on and each joint's angle (`setup_page.js`, from `GET /api/status`), and a joint lights up while it is more than 5 degrees from where the arm was first read, so you can see the arm you named move.

Nothing moves under power before Test joints (step 5, step 4 on a YAM). Until then every arm is open and disengaged (`disengage()`), so a hand can
move it, and the wizard only reads.

There is no checklist by hand. The shipped drivers fix each joint's direction, the joint order and the units in
tested code, so the home drawing plus the powered sweep covers them. What a checklist would catch that home and
power cannot is left to one question to the operator's eyes (*The eyes check* below).

## Reach area

Nothing in the stack knows where a station's camera mount is, so an arm could be driven into it. The operator paints the area each hand may work in instead: it needs no measuring and works on every robot the dream client solves for.

While the page is on this step, each read of the limp arms adds each hand's tool point, seen from above in that arm's base frame (the same point and frame the dream client solves for), to that arm's area, on a 2 cm grid (`setup_session.REACH_GRID_M`). Each arm has a Clear button. Save writes the points, with the home hand added, as the arm's `reach` (robots README, *Config file*). The dream client then moves any hand target farther than 4 cm from every point to 4 cm from the nearest one, in x and y only: dream pulls the hand's targets into the area, and if the arm still heads out past the edge, it stops there for the rest of that move. Limits: it works on the hand at any height, so a pole outside the edge is kept out, but it does not guard the elbow, and only head `eef` is clamped. Paint one solid area in front of the arm and stop short of the mount.

Continue waits until each arm has at least 400 cm^2 painted (`setup_session.REACH_MIN_AREA_M2`): the step paints from the moment it opens, and the hand at home alone is a 4 cm disk the arm could barely move in.

Skip asks "Skip the reach area? Without a reach area, the arms can move anywhere, including into your camera mount. Setting one makes runs safer and more reliable." (Go back, Skip anyway). Skipped, Save writes no `reach` and the Save step says so; coming back to the step un-skips it. Skip works from any state, a hand whose position cannot be computed is skipped for that read and named on the page, and Save never fails because of this step. Finding the arms again (or another robot) clears the painting; a new home does not.

## Home

The operator puts each arm in its home pose, the pose every run starts from and Stop goes back to, and presses OK. The
page draws dream's model of each arm at the readings now, so a driver whose zero, direction or units is off shows as a
drawing that does not look like the real arm. Joints read outside the URDF's range are named (a driver that reports
degrees or LeRobot's -100 to 100 shows here), and the operator may continue anyway. On an SO-101, joints past the
range dream's model was recorded on are named too: an arm seated on its base is past it at the shoulder and elbow, so a
run could never send it back there; the drawn home is the seated pose lifted just inside that range.

There is no expected home pose: one drawn in advance is wrong on some arms (the Maker Arm and the SO-101 among
them) and would fail drivers that are right. The cost: a joint whose zero moved by a constant (one tooth of a re-seated SO-101 servo horn
is 14.4 degrees) passes every check; only the drawing at home shows it.

## The powered checks

First each arm is sent its own position and must not move by more than 3 degrees, nor its gripper by more than 0.15;
a driver that writes differently from how it reads moves here, so this is the first powered write and the check stops
at the first tick that goes wrong. Then both arms go slowly to home, every joint is nudged 5 degrees and back (it
passes when it moves that way by 1 to 10 degrees: a fifth of the nudge to twice it, without swinging past 10 on the way), each gripper a little, and the
sweep runs from home. A sweep move the IK cannot reach from this home, or whose solve turns
any joint more than 20 degrees from home (`setup_motion.SWEEP_MAX_TURN_RAD`), is halved, twice, then skipped, and the
page says which and why.

How the run ends, and what the arms do then:

| End | The arms | Why |
| --- | --- | --- |
| every check passed | hold at home | the next step is Save |
| Stop | hold where they are | the operator stopped them on purpose |
| a check of how the arm moved failed (a nudge, the gripper nudge, a sweep move) | hold where they stopped | the write path works, so holding is safe |
| the hold check moved a joint (`not_followed`), a joint 5 degrees past the farthest test move from home (`too_far`, checked on every read), the gripper going elsewhere than sent (`gripper_write`), a YAM joint 12 degrees from what was sent for 6 ticks (`not_followed`), or an error | let go at once with `disengage()` (`setup_checks.LET_GO_CAUSES`) | a hold goes through the same write, which is what failed; a YAM let go floats |
| a joint of an arm that drops when limp (Metal, Maker Arm, SO-101) 12 degrees from what was sent for 6 ticks, still or creeping, or pushed past its `too_far` reach after it strayed (`blocked`: something under it or on the arm, the arm on the table) | that arm parks and switches off (`torque_off()`), the other holds (`setup_checks.PARK_CAUSES`) | letting both go would drop two brakeless arms. A YAM is never parked: switched off it drops, let go it floats |
| an arm's motor guard raises its alarm (`alarm`: a motor at its stop temperature for 1 s; `FollowerArm.alarm()`) | that arm parks and switches off (`torque_off()`), the other holds | as a station parks an alarmed arm between runs |

A park drives the arm to rest under power for a few seconds: the page says which arm moves, shows STOP, and refuses
*Turn off motors* and Run checks again until it ends. STOP during a park is the emergency choice: every arm goes limp at once where it
is, so hold them up. A park whose switch-off fails does not count as off: the page says the arm may still be powered,
so support it and cut its power.

An arm that stops answering under power (a cable out, its power off, a motor gone silent) is named on the page and
let go of on its own; the other is still let go. An arm that drops when limp is never switched off then, which would
drop it, and a Damiao motor that loses its bus keeps holding its last target anyway, so the page says to support it,
switch its power off and check its cable. Nothing powers the arms until they are found again in Connect arms.

While the arms hold after the checks the wizard reads each arm's alarm, as a station does between runs: a motor that
reaches its stop temperature parks that arm, and the page says which motor and how hot. The checks do not start while
any arm's alarm is up.

The farthest test move is 20 degrees from home, or where the arm started when that is farther, so a driver that
writes 1000 times too big (milliradians) gets one tick, not six at full speed. While the arms hold, the page shows
*Turn off motors* with "Support the arms first. They'll drop when the motors turn off.": it disengages both and reads them by hand again
(`Session.release()`, `POST /api/motion/release`), so home can be set again. With `home_before_motors_off: true` in the station file (off unless set) the button first drives the held arms home at the checks' own pace, with the Stop bar showing, and then lets them go; a move that trips a check, is stopped, or takes past 15 s still ends with the motors off, and the run's notes say whether the arms got home. It is skipped when an arm was parked, its switch-off failed, or it stopped answering. A new home, or a new gripper range,
drops the run from before: Test joints starts again. Every step that needs the arms by hand refuses while they hold and
says to switch off the motors first. The Stop bar while the checks run, and a *Turn off motors* bar while the arms hold, show on
every step and on any copy of the page, and a page opened again goes on from what the wizard still holds. When the
wizard's program stops while the page is open (a crash, Ctrl-C, the terminal closed), the page says so within about a
second: the arms are in whatever state it left them, so support them and run `dreamcli setup` again.

Run again drives the arms that still hold as they are; a held arm that stopped answering is refused until you let go
of the arms. Leaving the wizard switches any arm still powered off the way a run ends (`torque_off()`: a brakeless arm
parks at rest first), so no arm is left holding once the page is gone. Ctrl-C, a kill (SIGTERM) and a dropped ssh
session (SIGHUP) all leave it this way, and a second one waits until the arms are off, so hold them up first, as Finish
asks.

### One cause for a failed check

A failed check names one cause, never a list to choose from: what it is, one short line on what it does to the robot, and one fix (`setup_checks.CAUSES`, each a `Cause(title, impact, fix)`; the page shows them in a card under the banner, headed by the arm and the joint the check failed on, such as "Left arm, joint 3"). The cause comes only from what the readings support. A nudge keeps every reading from before the move until it settled (30 a second, `Motion._traces`), and the run stops at the first failed check, so the joints nudged before it are the ones known to work. "Lifts the arm" is from dream's URDF: turning the joint moves the tool up or down at least 10 cm per radian at home (`setup_model.ArmModel.lifting()`, up being the arm's base frame's +z). That is joints 2, 3 and 4 on the Metal, the YAM and the Maker Arm, never the base turn or a wrist roll.

| What the readings show | Cause | The fix the page gives |
| --- | --- | --- |
| a joint that lifts the arm fell short of its nudge once the arm showed it has power (an earlier joint followed, or this one moved a little); or a sweep move missed and the joint farthest off lifts the arm, fell behind its move, and fell short of its own nudge by a tenth of the nudge more than any joint that lifts nothing (the base yaw) did | `weight`: This arm cannot hold its own weight. | Add gravity compensation to your driver, or raise the stiffness of the lifting joints. |
| nothing moved at all on the arm's first nudged joint | `no_power`: the arm gets no power or no commands | Check its power, its emergency stop and its cable. |
| a joint fell short but was still creeping on as it settled (by a tenth of the nudge or more) | `slow` | Make write() send every command at once, and raise the joint's stiffness. |
| a joint settled right but swung past twice its nudge on the way | `overshoot` | Lower the joint's stiffness or raise its damping. |
| another joint moved, it moved the other way, or it moved over twice as far | `write_order`, `write_flipped`, `write_size`: each its own headline, saying what was seen | one line each: the order, the sign, radians |
| a joint that does not lift the arm fell short of its nudge once the arm showed it has power | `joint_stuck` | Check that nothing blocks it and that its motor answers your driver. |
| a sweep move missed and the joint farthest off does not lift the arm, went past its move, or followed its nudge about as well as the joints that lift nothing (every DM4340P joint of a reBot, the base yaw included, moves 3.6 to 3.8 of 5 degrees) | `missed` | Raise your joints' stiffness, and check that nothing presses on the arm. |

A sweep move is judged like a nudge: along the move sent (its shift, or its turn's axis) the tool must go the same way by 0.2 to 2 times it (`setup_checks.NUDGE_MIN_FRACTION`, `NUDGE_MAX_FRACTION`), and what it does besides (off that line, and the turn of a shift or the shift of a turn) stays within the body's tolerance (`setup_checks.TOLERANCES`, 1 cm and 3 degrees on a YAM). Why: on a YAM, +z (3 cm from the folded home) reaches about 1 cm, the elbow about 2 degrees short as its own nudge is (+3.1 of +5, a pass), which a bar of 1 cm from the move sent would fail on every run. A hand that goes the wrong way, under a fifth of the move, over twice it, or sideways still fails.

After the sweep, each arm gets a note, never a fail, on how its lifting joints follow (`setup_checks.tracking_advice`): from how far short of its target each lifting joint held in the moves that turned it up and those that turned it down. Friction holds a joint behind its move either way; the arm's weight holds it low, short lifting and past lowering. So shorter lifting than lowering by 1 degree or more says to add or raise gravity compensation, the reverse that it pushes up too hard, about the same both ways (1 degree or more) that the joint is soft or sticky.

The check lines and the cause card can be selected and copied: the page polls every 150 ms and rewrites a line or the card only when what it says changed, since a rewrite drops the selection in it.

The other causes keep their checks: `not_followed`, `too_far` (the same driver cause as the mapping rows), `gripper_still`, `gripper_write`, `gripper_write_size`, `blocked`, `alarm` and `eyes`. A swing past twice the nudge fails even when the joint settles right. Not named, since the readings cannot support them: ringing on its own (a joint that swings back and forth inside the pass band) and a fast buzz, which a 30 Hz read cannot see.

After a run that did not pass, *Run test again* is the main button. *Continue anyway* opens a small prompt: "Skip this step? If you skip these checks, your robot may not perform as well." with *Go back* and *Continue anyway* (`skipAsk`, a native `<dialog>`). It is friction, not a block: the save step still says the checks did not all pass. Why: a Metal arm whose driver lacks gravity torque fails the joint 2 and 3 nudges, and a message that does not say it matters lets the operator save an arm that never lifts (`tests/test_setup_flow.py`, `test_a_metal_arm_that_cannot_hold_its_weight_fails_the_shoulder_nudge_naming_its_weight`).

**No policy, no zero-shot.** The powered moves are the nudges and the hand sweep, solved by dream's IK and judged by
forward kinematics: the wizard never runs a policy, and never sends joint-space actions from a model on any robot.

### Why a sweep move may turn a joint 20 degrees at most

A move of the hand 3 cm or 10 degrees is gentle only from a well-placed home. From the Metal Arm's folded recorded
start, near a singular pose, dream's IK reaches some moves only by swinging the base or the wrist far: 66 degrees on
the base for 3 cm sideways, 67 degrees on the wrist for a 10 degree turn about z. Without a limit those ran at full
size. From dream's rest pose the Maker Arm needs 17 degrees at most for any move, so 20 keeps every move there whole,
while the Metal Arm's sideways and z-turn moves still need 24 degrees at a quarter size and are skipped. What each
robot's sweep does from its drawn home, from `Motion._reachable_path` itself:

| Move | Metal (recorded start) | YAM (example start) | Maker Arm (rest) | SO-101 (rest) |
| --- | --- | --- | --- | --- |
| +x 3 cm | 50%, reach | full | full | 50%, turn |
| -x 3 cm | skipped, reach | skipped, reach | full | 25%, reach |
| +y 3 cm | skipped, turn | 50%, turn | full | 25%, reach |
| -y 3 cm | skipped, turn | 50%, turn | full | 25%, reach |
| +z 3 cm | full | full | full | full |
| -z 3 cm | 50%, reach | 50%, reach | full | 25%, reach |
| turn about x 10 degrees | full | full | full | 25%, reach |
| turn about y 10 degrees | full | full | full | 50%, turn |
| turn about z 10 degrees | skipped, turn | 50%, turn | full | 25%, reach |

"reach" is a move the IK cannot reach at that size, "turn" one that turns a joint more than 20 degrees. The largest
joint turn of any move that runs is 19.5 degrees (the SO-101's shoulder on +x at 50%).

## The eyes check

A driver that reads and writes a joint the same wrong way (flipped, or two joints swapped) passes every powered check:
the wizard sends a number, the driver turns it into the wrong motor move, and reads the same number back. The real arm
then moves unlike dream's model of it. While the nudges and the sweep run, the page draws both arms from their live
readings (`setup_view.now()`, from the powered checks' own reads, `setup_motion.Motion.latest`). The nudges move one
joint at a time, left arm first; the drawing moves that same joint at the same moment and lights it up
(`Motion.nudging`, sent as `nudging` in the motion status), so the operator compares each joint's direction with the
drawing one by one. When the checks pass the page asks once: "Did every joint and hand move like the drawing?" Yes
passes the step. No fails it with the cause `eyes` and the
fix: check your driver's joint directions (a joint turning the other way) and order (two joints swapped)
(`Session.confirm_eyes()`, `POST /api/motion/eyes`). Continue appears only after the answer.

## Gripper ends

A Maker or Metal Arm's gripper runs linearly in its motor angle between two ends (`gripper_closed_deg`,
`gripper_open_deg`). The wizard measures them on the home step, where the arms are limp: the operator opens both
grippers fully by hand and presses Record open, then closes them fully and presses Record closed
(`Session.record_gripper()`, `POST /api/gripper`). Each gripper keeps its closed end where the jaws met and its open
end where it stopped, the real stop. The arm keeps every gripper goal (the model's fully open, home, the park, a hold)
`grip_margin_mm` inside that end (`robots.motor_arm.ArmSpec.grip_margin_mm`: 2 mm on a Metal or Maker Arm), so its
motor never pushes into the stop: one margin, in the arm, for every body and every goal. Nothing moves under power, and
Save refuses, until both are recorded; a range under 20 degrees is refused as not opened and closed fully.

Measured, never assumed, since an open end past a gripper's real stop pushes its motor into the
stop until it overheats. An SO-101's ends are its calibration file's gripper range, and a YAM's come from its gripper
type.

A robot of your own gets the same step when its driver reports its gripper motor's angle (`Driver.gripper_deg()`),
so the gripper need not be moved closed then open outside the page. The ends are the angles
it reports, either way round; Save writes them into each arm's section, and the driver maps its gripper with them
(`robots.driver_template`). Until they are measured the arms open without them (NaN), and a grip the driver maps with
them reads `?` on the page. A driver that reports none keeps its gripper's range itself: the step is skipped and the
page says why (`setup_session.GRIPPER_SKIPPED`).

## What each mistake looks like

These are the mistakes planted in fake drivers in `tests/test_setup_flow.py`, each in the left arm's driver of a YAM
homed at the start of a recorded episode; each is caught where the table says, and a clean driver passes every
check on the YAM, the Maker Arm, the Metal Arm and a robot of one's own. "Your eyes" means every check passes, and
only the operator sees it: at home, where the drawing does not look like the real arm, or during the powered checks. `tests/test_setup_bus_arms.py` runs
every step to Save on a Metal Arm and an SO-101 through their real drivers on the bus fakes, limp by hand, then starts
a dry run on the saved file.

| Mistake in the driver | Caught at | Cause on the page |
| --- | --- | --- |
| an SO-101 whose servos hold another calibration than every ticked file | finding the arms (it does not open) | each file's mismatch (the driver's check, `tests/test_so101.py`) |
| reports degrees instead of radians | home, joints outside the URDF's range | named at home |
| reports LeRobot's -100 to 100 | home, joints outside the URDF's range | named at home |
| a joint's zero shifted 14.4 degrees (a re-seated horn) | your eyes, at home | eyes |
| a joint's zero is 90 degrees off (the shoulder) | your eyes, at home | eyes |
| a joint turns the other way, home far from 0 (joint 3, 7 degrees) | your eyes, at home or during the sweep | eyes |
| two joints in the other order, home values that differ (joints 4 and 5) | your eyes, at home or during the sweep | eyes |
| a joint turns the other way, home near 0 (joint 6, 1 degree) | your eyes, during the nudges and the sweep | eyes |
| two joints in the other order, home values alike (joints 1 and 6) | your eyes, during the nudges and the sweep | eyes |
| gripper reads 1 closed (and writes the same way) | your eyes, during the gripper nudge | eyes |
| gripper not read (the reading never changes) | powered, the gripper nudge | gripper_still |
| gripper reads mm | powered, the hold check (its reading is 80 times what was sent) | gripper_write |
| left and right arms swapped against the old file | finding the arms | crossed (a message, then Save fixes it) |
| reads right, writes one joint flipped | powered, the nudge | write_flipped |
| reads right, writes the first joint flipped, away from its zero | powered, the hold check | not_followed |
| reads right, writes degrees | powered, the hold check | not_followed |
| reads right, writes two joints swapped | powered, the nudge | write_order |
| reads right, writes the gripper the other way | powered, the hold check | gripper_write |
| reads right, writes 1000 times too big | powered, the first tick after the first write; both arms let go | too_far |
| the arm does not move at all | powered, the first nudge | no_power |
| a Metal arm without gravity torque (shoulder and elbow 4.4 degrees behind what is sent) | powered, the joint 2 nudge | weight |

## The arm, as the wizard uses it

The wizard opens an arm the way `dreamcli robot run` does, through its body (`robots.bodies`: `BODIES[body]`,
`so101_body(embodiment)` or `custom_body(custom)`, picked by `setup_session.body_of`), and uses only the
`robots.followers.FollowerArm` that `open_arm(cfg, side)` returns: `read()`, `write()`, `hold()`, `disengage()` and
`close()`. A robot of one's own is a driver class its owner picks on the page (`/path/to/file.py:Class`, or a typed
`package.module:Class`, a `robots.followers.Driver`), with the account embodiment its fine-tune was trained on.
`home` is the arm config's key (`ArmConfig.home`, one angle per joint in radians; how many it lists is how many joints the arm has). The wizard opens and closes that driver at
several steps, so find out first what its SDK does at connect and disconnect: some switch every motor on at connect,
and some drive every joint to zero at disconnect (Seeed's reBot driver does, unless told not to).

### Set it up myself

Step 0's second way, for an operator who would rather write the file: the page hides every later step and, for the
picked robot, shows the station file (`GET /api/template?body=`, `setup_wizard.template`) and the commands that run
it, a dry run first. The file is build_config()'s, the one Save writes, so `dreamcli robot run` and `dreamcli robot up`
read it as they read the wizard's; every value a step measures or picks is a `REPLACE_...` blank (each arm's adapter
and home, a Metal arm's gripper ends, an SO-101's calibration files, each camera's device, the account embodiment, a
robot of your own's model of the arm), and the loop refuses the file by name until each is filled. The home is a
blank too, never zeros: a Metal arm at zero stands straight up. Nothing on this path checks the driver, the arms or
the cameras, and nothing registers the robot, so Live view needs the `live.robot_id` the wizard's Save writes; a run
from the terminal needs none. For a robot of your own the page also names `dreamcli tcp` (*`dreamcli tcp`* below),
which writes the arm file `custom.ik` takes. The customer docs page is *Set it up yourself*.

### Your driver and the arm model

The page runs in the operator's browser and the wizard on the station, so *Your driver* browses the station's own
folders: their folders and `.py` files, hidden ones left out (`GET /api/files?dir=`, `robots.setup_files.folder`),
starting in the station file's folder (`setup_files.start`), then the picked file's top-level classes, read without
running it (`GET /api/classes?file=`, `setup_files.classes`). The pick is `custom.driver` as
`/path/to/file.py:Class`; *Installed as a Python package? Enter its name* takes `package.module:Class`. Under it, two lines
link the driver template (`GET /driver_template.py`), the reBot example and the customer docs' *A robot of your own* (on the redesigned page, as buttons under the Driver heading, above the file picker);
the dry-run command is on Save, where it applies.

*the arm model* is two large choices, *My arm's URDF* (*An arm from its URDF* below) and *An arm Dream Labs
has a model of*, a tile per arm by its plain name (`setup_session.own_arms`, `OWN_ARM_LABELS`). The tiles leave out
YAM, the Maker and Metal Arms and the SO-101, whose step 0 tiles already give them their own driver and model, and
`b601`, the same URDF and tool point as `seeed_faas` (`setup_session.SAME_ARM`). A station file may still name any
dream arm: a left-out one it names is listed as "`<name>` (your station file's)".

### An arm from its URDF

An arm dream's IK does not ship is taken from its URDF, so Dream Labs' base models run on it zero-shot. Under "Your
arm's URDF" on the robot-of-your-own card, the operator gives the file and types nothing else; the rest is done
on a 3D drawing:

| Step | What the page shows | What the operator does |
| --- | --- | --- |
| the file | nothing yet; for a `.zip` of several URDFs, one choice per URDF, by its path in the zip with its links and moving joints counted, and nothing drawn until one is picked | gives the URDF as a `.zip` of it with its meshes, or its folder (Choose .zip, Choose folder, or a drop on the redesigned page; the label reads "Your arm's URDF: a .zip of it with its meshes, or its folder" on every surface; a `.urdf` alone also works, drawn plain), and then one of its URDFs if it holds several (another pick draws that one; its meshes are found relative to it, the same way from a folder as from a zip); `POST /api/urdf` reads every link where the arm's zero puts it, with its visuals (`robots.setup_urdf.read`). The page draws each link's STL, OBJ, COLLADA or glTF (`.glb`, or `.gltf` with its buffers beside it in the zip) mesh from the zip (by its path, else the longest matching end of it, any case; a Mac's `__MACOSX` copies left out), and boxes, cylinders and spheres as they are; a link with none of those is drawn plain, and the page names it. A `.urdf` alone that names meshes is drawn plain, with a note to give the zip or folder |
| the end | the links drawn as balls and capsules, the URDF's root in blue and the end of its longest chain (most moving joints, less a gripper's fingers that slide after the last turning joint) in green, with "Is this the end of your arm?" | "Yes, it ends there", or clicks the right link on the drawing; "It's mounted on another link", then a click, moves the blue one. Yes waits while the end is not below the base, or is more joints out than the loop drives (6). Under `dreamcli tcp`, when the station file's embodiment has a URDF joint map (`urdf.joints`), the page names each moving joint between base and end that the map does not drive, each with *Moves* and *Held still* and neither pressed, and Yes waits until each has one. A held one is cut as a fixed joint at its URDF zero |
| the tool | on top a YAM drawn from i2rt's meshes, gripper included, with the triad at its tool point (`GET /api/reference_arm`, `robots.setup_view.reference`), labelled "Reference: YAM (for comparison only)" in a muted, dashed frame; below it your arm with its own triad, labelled "Your arm: drag the marker here" in a solid frame. Each view is two thirds of the card's width (the whole width under 760 px) and at most the window's height less 120 px: "Place the marker where your gripper's fingertips meet, turned like the YAM's." A triad is an arrow and a ring per axis of dream's tool frame, a fifth of the arm's size, named once in a legend: red x "fingers close", green y "up", blue z "out of the gripper". The links are plain in this step, so the triads' red, green and blue are its only colours | moves your arm's triad: dragging an arrow moves the marker along that arrow in whole mm, dragging a ring turns it about that arrow in whole degrees, and a click on a handle opens a box there for a move (mm) or a turn (degrees), which Enter applies. The six numbers below are the marker's place, x, y and z in mm, and its turn, roll, pitch and yaw in degrees, a URDF's rpy (turns about the fixed x, then y, then z), in the end link's frame. Each has a minus and a plus button that step it by the step picked (1 mm, 5 mm, 1 cm or 5 cm; 1, 5, 15 or 90 degrees; 1 cm and 15 to start) and a field that shows it, live as the triad moves, and takes a typed value; angles wrap into (-180, 180]. *Flip up and down* turns the marker half a turn about its blue arrow, so up and down swap and out stays. *Reset to start* puts it back where it started on this end link. Dragging elsewhere turns the view, the wheel zooms |

The marker starts at the end link's origin, out of the gripper along the base axis nearest the last link's direction,
and starts there again when the operator picks another end. *Set tool center point* turns it into the arm file's tool (`tcp`, its z as `approach`,
its x as `jaw`, dream's tool frame) and sends it with the two links (`POST /api/arm_file`, `SetupServer.add_arm`).
Continue waits for Set tool center point, or for *Skip calibration (not advised)*: a warning that hand-position actions may
land off target, whose *Skip anyway* stays off until "I understand the hand may miss its target" is ticked, so skipping
is a deliberate choice. Skip anyway sends the marker as it is,
with `skipped`. A new URDF, end or base afterwards turns Continue off again. The wizard:

| Does | Where |
| --- | --- |
| writes the URDF and an arm file of it (`dream.arm_file`: the links, the tool, `rest`, the URDF's zero moved inside any joint limit that excludes it, and `tcp_calibration`, `{"status": "calibrated" or "skipped", "at": UTC time}`, plus `digest` on a calibrated one: `dream.arm_file.tool_digest()` of the arm file and URDF as written, which dream checks at `policy()`, warning once when the tool point is not one a person set; and `fixed`, the joints the operator said are held still, when there are any) | `arms/<name>.urdf` and `arms/<name>.json` beside the station file; `<name>` is the URDF file's name in lowercase letters, digits and dashes |
| builds dream's model of the arm from them, which every check then draws and solves on | `robots.kinematics.dream_arm` |

The pick then uses the arm file, and Save writes its path as `custom.ik`. The same name again writes the files
again. Uploading registers nothing: the embodiment is the pick's (*A new embodiment* below). The base link's frame is
what a base model reads the hand's pose in, as YAM's base: x forward, y to the arm's left, z up. The steps after this
one draw the arm as shapes, not the zip's meshes. An expert who would rather type the links and the tool writes them in the arm file
(`custom.ik`) by hand.

Every launch of an eef run on an arm file checks its record (`robots.run.check_tool_point`, from `dreamcli robot run`,
`python -m robots.run`, and `robots.station` at its start and each Start): `calibrated` runs; `skipped` runs with one
warning line; a file with no record, written by hand, is refused with a line naming `dreamcli tcp`, unless the run
passes `--skip-tcp-calibration` (`dreamcli robot up` hands it to the station), which warns the same way. The flag is
in no `--help`: only that refusal and the customer page *Set it up yourself* name it, as not advised.

### `dreamcli tcp`: the tool point tool alone

For a station file written by hand (*Set it up myself*): the tool point tool by itself, kept simple. It is this
page and this server
(`setup_wizard.tcp_main`, `TcpServer`): the page reads `tool: tcp` from `GET /api/state` and shows *An arm from its
URDF* alone, and the server answers only the page, its scripts, the meshes and `GET /api/state`,
`GET /api/reference_arm`, `POST /api/urdf` and `POST /api/arm_file` (`TCP_ROUTES`), 404ing any other route by name. So
it opens no arm or camera. When the station file names an embodiment, it asks the account for its URDF joint map,
and the end step asks a person about each moving joint between base and end the map does not drive
(`TcpServer.joint_map`); with no embodiment or no map it asks about every one, and when the account cannot be asked
(no key, network, auth) it says why in one line first (2026-10-18). Set tool center point writes `arms/<name>.json` and `.urdf` beside the station file,
byte for byte what this step writes for the same inputs, and the page and the terminal print the path for
`custom.ik`. It listens on 8094, beside the wizard's 8092 and the run view's 8093, takes `--host`, `--port` and
`--no-browser`, prints the private address as `dreamcli setup` does, and ends on Ctrl-C. The customer docs name it on
*Set it up yourself* only.

### A new embodiment, from the operator's own steps

An account with no embodiment and no fine-tune finishes setup on Dream Labs' base models. The embodiment list of an SO-101 or a robot of your own ends with "New: one for Dream
Labs' base models, from your arm and cameras", the only entry when the account has none. It asks nothing. Continue
(`POST /api/robot` with `embodiment: null`, `SetupServer._zero_shot_embodiment`) registers on the account, before the
arms step:

| Part | From |
| --- | --- |
| the name, `<username>/<arm>` | the arm, dream's name of it in lowercase letters, digits and dashes (`so101_faas` is `so101-faas`, a URDF its file's name) |
| `state` and `action` | the arm's width (`robots.adapter.zero_shot_embodiment`: the MakerMods parts without `<side>_gripper`, except on an arm whose gripper drive the dream client makes, an SO-101's LeRobot 0 to 100, which gets `right_gripper` and `left_gripper` last: the client sends the drive under it, and its IK refuses the arm without it) |
| `video` | `scene`, `wrist_l` and `wrist_r` (`robots.adapter.ZERO_SHOT_CAMERAS`), the camera step's role buttons |

The operator gives each camera its role in the camera step, and the role names it: nothing is typed (the role buttons
read "Top", "Left", "Right"). The wrist cameras keep those two names because the base models read a wrist camera per arm by
name, and one scene camera by any name (`dream.client_ik.camera_slots`, `shared.datasets.embodiments.WRIST_ROLES`), so
`scene` goes to the model as its `overhead`. A second pick reuses the embodiment. An embodiment of that name with
another layout is refused at the pick, with its cameras: pick it in the list instead. The robot then registers as the
new embodiment's id, and a fine-tune's data of it is uploaded under it (`DreamClient().upload_dataset`). This replaces
"Register one for this arm" (`POST /api/embodiment`) and the registration at a URDF's upload: the pick's "New" is the
one way the wizard registers an embodiment.

### Limp for the steps by hand

Finding the arms and home move the arms by hand, so each arm is let go with `disengage()` and read on:

| Body | What disengage() does | What the operator does |
| --- | --- | --- |
| YAM | gravity compensation (i2rt): the arm floats where it is | nothing |
| Maker Arm, Metal Arm, SO-101 | `MotorArm.disengage()`: the arm's thread switches every motor off with no park (`Chain.relax()`), then only reads | holds the arm: it has no brakes and drops |
| your own | your driver's `disengage()` (the template's shows the shape: motors off, read on, the next write re-engages holding where the arm is); a driver without one is held stiff, since the default returns False | depends on the driver; with `drops_when_limp` it drops |

The page says "This arm has no brakes. Hold it ..." on the tile and again above *Turn off motors*
(`setup_session.LIMP_NOTE`). A limp arm stays limp until closed; the powered checks close it and open it again, which
energizes it holding where it is. A bus arm is opened powered (its thread holds for a tick or two) and goes limp as
soon as disengage() returns.

### Finding the arms

What the scan lists (`setup_devices.arm_adapters`), one row per physical adapter:

| Adapter | How it is opened | What the file keeps |
| --- | --- | --- |
| USB CAN adapter under the kernel's gs_usb (a YAM's candleLight) | its SocketCAN link | `can_follower_l` / `can_follower_r`, the names the udev rule gives; Save waits until the rule is in effect |
| CANable on slcan firmware (a Maker or Metal Arm) | directly over its serial device, at 1 Mbit/s, no root and no `slcand`; if an `slcand` holds it, through that `slcand`'s link | its `/dev/serial/by-id/...` path |
| any other USB serial adapter (an SO-101's servo bus) | its serial device | its `/dev/serial/by-id/...` path |
| an arm the scan does not list (on Ethernet, behind a bridge of your own), typed under *Don't see your arm?* (`POST /api/arms/typed`, `setup_devices.typed_arm`) | by the robot's driver, given the address exactly as typed (`192.168.1.20`, `can2`, `/dev/ttyUSB0`) | that address, as the arm's `can_channel` |

- Setup never blocks on a transport the scan does not know. A typed arm fits every robot, is ticked like a found one, and goes through the same moves by hand and
  powered checks, on whatever the driver opens. A camera the scan misses is typed the same way under *Your camera is
  not listed?* (`POST /api/cameras/typed`, `setup_devices.typed_camera`): an address OpenCV opens (`rtsp://...`) is a
  `device`; `module:Class` or `file.py:Class` is a camera class of your own, the camera's `source` (`README.md`,
  `CameraConfig`). It joins the camera pictures, and the role step and live views open it as the station will.

- A serial adapter (a CANable or any other) is named by its `/dev/serial/by-id/...` path only when its serial is its own. Identical adapters can report one serial (Damiao's USB2CAN bridges do): udev then makes a single by-id link for all of them, pointing at whichever came up last, so the page would find one adapter of two and a replug could swap the arms. Such an adapter, and one with no serial, is named by its `/dev/serial/by-path/...` path, the USB port it is in, and its row says to keep it there.
- Only adapters that fit the picked robot's bus are offered (`setup_session.ARM_KINDS`: a YAM a CAN link, a Maker
  or Metal Arm a CAN link or a CANable, an SO-101 a serial adapter). Everything else is under a closed *Other devices
  on this computer, not arms*.
- The page says what ticking does: a ticked adapter is opened and let go when you press *Release the arms*, an
  unticked one is left alone. The button shows that it is working while each adapter opens.
- A CANable opened directly comes back by itself after a replug or an arm power cycle: there is no link to go down and
  no `slcand` to die. When an `slcand` holds it, the station must use its link, since two programs
  on one serial device garble each other's frames; a link that is down is named with the command that brings it up
  (`sudo ip link set can1 up`), and `dreamcli robot up` runs that command itself.
- A failure reads as what happened and what to do (`setup_devices.explain`): a link that went down, an adapter that
  is gone, no permission (`usermod -aG dialout`), another program holding the adapter, motors that did not answer,
  pyserial missing. A Metal motor in fault is named by its code in words (0xC is the coil over-temperature: let it
  cool, not re-zero it; `damiao.FAULT_TEXT`), and an SO-101 servo missing from the bus is named by id and joint with
  "reseat that servo's cable".
- After any failure, a replug or a power cycle, *Scan again* lets go of every arm found so far,
  forgets which was which and scans again, on the same page (`POST /api/arms/close`).
- A brakeless arm sags onto its rest as it goes limp, so the session waits `SETTLE_S` (1.5 s) after letting go before
  it measures the move that finds an arm. Two arms moving at once says so and asks to rest every arm where it cannot
  move, press *Start over* and move only one.

## The SO-101

An SO-101 is opened by `robots.so101` (`robots.feetech.FeetechChain` on LeRobot's bus, the `so101` extra), the same
driver `dreamcli robot run` uses. A joint is `(tick - 2048) * 360 / 4096` degrees, the tick being the servo's position
after the homing offset LeRobot's calibration writes to the motors, so the URDF's zero is the pose the arm was
calibrated in. dream's IK models the SO-101 per side (`so101_faas`, left and right URDFs that differ only in where the
base sits and the calibrated limits); the wizard draws and checks both arms on the first side's URDF and solves each
side's sweep on its own.

- The pick lists the account's embodiments (`dream.DreamClient().get_embodiments()`, by name with their cameras) for
  the one the fine-tune was trained on (its `video` roles are the camera roles), or a new one (*A new embodiment*
  above), and lists the calibration files `lerobot-calibrate` wrote on this box
  (`~/.cache/huggingface/lerobot/calibration/robots/so101_follower/*.json`), plus the existing file's, each with when
  it last changed; none is ticked at first, and the operator ticks the files of their arms.
- Each arm's servos are read once, before it opens, for the calibration they hold in their EEPROM (homing offset and
  range per servo), and the ticked file that holds the same is the arm's, saved as its `calibration`. Files that agree
  on every value are copies of one calibration, so the first is taken. An arm whose servos hold none of the files
  opens with what they hold, written to a file of the wizard's own in a temporary folder, so it can be moved and
  found; the page then calibrates it, and that file is never saved.
- The gripper's ends are the file's gripper range (the driver's).
- Save writes `body: so101`, the `embodiment`, and each arm's `calibration` and `home` (the sixth angle 0).

### Calibrating an SO-101

Once both arms are found, each arm still to calibrate has a *Calibrate the left arm* button (an arm already calibrated
has a quiet *Calibrate the left arm again*). It does what `lerobot-calibrate` does, on the page
(`robots.so101_calibration`, `Session.start_calibration()`): the arms step lets go of that arm, its servos are switched
off and put in position mode, then two stages.

| Stage | What the operator does | What the wizard does |
| --- | --- | --- |
| 1 of 2, middle pose | Holds the arm in the pose the drawing shows (dream's model at its zero: upper arm up, forearm forward, wrist in line, gripper level and closed), close is enough, and presses *Set middle pose*. | Resets each servo's homing offset and range, reads its raw angle, and writes the homing offset that makes it read LeRobot's half turn (2047), as `set_half_turn_homings()` does. |
| 2 of 2, ranges | Moves each joint slowly to its stop at each end and back, and opens and closes the gripper. The wrist roll is not needed. A bar per joint fills when the joint has turned within 10 degrees of the model arm's range (30 degrees for the gripper), and *Finish* names any joint not there yet. | Records each servo's lowest and highest reading. On *Finish*, moves each joint's zero (below), writes the homing offset and range to every servo and the same numbers to `~/.cache/huggingface/lerobot/calibration/robots/so101_follower/<side>_<adapter serial>.json`, keeping a file already there as `<name>.json.<time>.bak`, then opens the arm again, limp, with it. |

The zero is matched to the model's arm. A middle pose put by hand is a few degrees off, differently on every arm, and a
model trained on an SO-101 reads joints relative to the calibration of the arm it was recorded on: an arm zeroed
elsewhere sends every joint off by its own error, and dream's IK puts the hand off by as much, with no error anywhere.
The stops a joint is moved between are the arm's own and the same on every SO-101, so the middle of the recorded range
says where the hand's zero is against the model arm's: `so101_calibration.reference()` reads the model arm's ranges per
side from dream_ik's `so101_faas` spec (its `joint_units`: radians = offset + scale x unit, LeRobot's -100 to 100 across
the range), and each joint's homing offset is moved by the difference. The page says by how much per joint. A joint
whose range is more than 10 degrees longer than the model arm's is another build: its zero is left where the hand set
it, and the page says so. A middle pose more than 25 degrees off is refused, to be set again. The wrist roll turns all
the way round and has no stops, so its zero is the hand's and its range the whole turn, as LeRobot gives it; the
gripper's range is its own.

*Cancel* closes the arm and opens it again with what it had. Nothing past step 1 runs while an arm is uncalibrated or
being calibrated (`Session.set_home()` and every arm setting a powered step reads refuse it).

## The model

Setup asks for no model and the file names none: the model is picked at each start, in the live view or on the
command line. A YAM runs
`robots.run.DEFAULT_CHECKPOINT`; on every other robot each start names the model: the live view's Start, or
`dreamcli robot run --checkpoint`. A `checkpoint` written by hand still works, but a re-run of setup drops it.

The cameras take the embodiment's names: the body's own on a YAM, a Maker or a Metal Arm (`scene`, `wrist_l`, `wrist_r`
on the MakerMods wire, which carries every MakerMods name), the account embodiment's `video` roles on an SO-101 or a
robot of your own (the embodiment list shows them; a new one's are `scene`, `wrist_l` and `wrist_r`). A model that reads other names gets each camera in its own slot
from the dream client (`dream.client_ik.camera_slots`). The pick's reply labels each camera
(`robots.setup_cameras.camera_roles`): a name the wizard knows as a head or wrist camera ("Head camera (top)") gets our
reference picture and a guessed dataset key; any other ("Camera cam_high") has none until the operator gives a dataset
and picks its key.

The camera step's references come from the data of the account's newest fine-tune on the robot's embodiment
(`get_models()`, then the data service's training data); a YAM, and a robot the account has no fine-tune of, show our
YAM pictures. When the models cannot be listed or the data cannot be read, the page says why and shows ours.

The robot registers on the platform as its embodiment's id (`emb_...` for an SO-101 or a robot of your own). Saves
run one at a time, so a double-click or a second tab never registers two robots for one station.

## The 3D arm

Steps 3 and 4 draw dream's model of the arm in 3D (`setup_viewer.js`). The page computes no kinematics: the server
sends every link's pose from dream's URDF (`setup_view.py`: drawing() once, now() on every poll), so the drawing moves
exactly as the checks judge. Step 3 draws the arm now; step 4 draws the arm now as it moves.
Every view opens from the side: along the base's y axis, 15 degrees from above, the base on the left and forward on
the right, the same for every robot (`SIDE_VIEW` in `setup_viewer.js`). Drag to orbit. On the redesigned page (`DREAM_SETUP_WIZARD_V2`) every view also moves the way CAD tools do (`cadNavigation()` in `setup_viewer.js`): right-drag, middle-drag or a Shift, Ctrl or Cmd drag moves it, the wheel or a pinch zooms toward the pointer, and Fit, Front, Side, Top and Iso in its corner fly to a standard view (F fits); a right or modified press never picks a link or grabs the marker. Why the side view: from a
three-quarter view, the Metal Arm's folded home (every joint near 0, the upper arm lying back and the forearm folded
forward over it) reads as upside down from the elbow.

three.js 0.170.0 (three, OrbitControls, the STL, OBJ, COLLADA and glTF loaders) loads from `cdn.jsdelivr.net/npm/three@0.170.0/`, named in the
page's import map with each file's sha384 hash, so the browser refuses any other bytes. Nothing is vendored. When the
library cannot load, the page says plainly that setup needs internet, and nothing else on it works. Loading it from
the CDN means no copy of three.js and its license to ship.

Each robot is drawn one way, chosen in `setup_view.DRAWINGS`, never as a fallback:

| Robot | Drawn as | Meshes from | License | Size |
| --- | --- | --- | --- | --- |
| Metal | the maker's meshes | makermods-robotics/makermodslab @ 64defd4e | MIT | 1.7 MB |
| SO-101 | the maker's meshes | TheRobotStudio/SO-ARM100 @ a758567c | Apache-2.0 | 1.8 MB |
| Maker Arm | the maker's meshes | makermods-robotics/maker-arm-sdk @ b30d05a2 | Apache-2.0 | 1.8 MB |
| YAM | i2rt's meshes, its linear_4310 gripper's too | i2rt-robotics/i2rt @ 4e462ecc | MIT | 1.7 MB |
| My own robot on dream's `seeed_faas` arm (the Seeed reBot B601-DM) | Seeed's meshes, its two-finger gripper too (`setup_view.ARM_DRAWINGS`) | Seeed-Projects/reBot-DevArm @ c16d03ea | CERN-OHL-W-2.0 | 1.7 MB |
| My own robot on any other arm | simple shapes: capsules for links, discs at joints | | | |

`tests/data/setup_view_build.py` fetches each maker's URDF and meshes at the pinned commit, checks every maker link
rides one of dream's links rigidly, merges and simplifies the meshes (under 2 MB per robot) and writes
`setup_view/<robot>/`, plus the robot's entry in `tests/data/setup_view_golden.json`: the maker URDF's part poses at
four fixed joint angles. `tests/test_setup_view.py` checks every mesh sits within 0.1 mm of those poses on dream's
URDF, and that the drawing's tool point is dream_ik's tool point. i2rt models the YAM's gripper apart from the arm, as
an MJCF, so the builder first puts it on i2rt's URDF of the arm (`with_gripper()`), on link_6 where dream's tool point
has it: the drawn fingertips close 8 to 10 mm past the tool point, centred on it.

A robot of one's own is drawn as shapes, except on an arm of dream's whose maker's meshes ship here
(`setup_view.ARM_DRAWINGS`): the Seeed reBot, a custom robot whose `ik` is `seeed_faas`. Seeed's URDF is dream's
b601_dm chain from link1 on, but dream's joint1 sits 45 mm lower on its base_link (the station's table drop), so the
builder lines the two up on link1 and Seeed's base is drawn 45 mm below dream's base_link, where the real base is.
The gripper is Seeed's two fingers, sliding on one rack.

## What it does not catch, and what is not there

- A joint whose zero moved by a constant (a re-seated horn, a motor calibrated in another pose): only the drawing at
  home shows it, if the operator compares (*Home*).
- A joint that turns the other way, or two joints swapped, where home does not show it: only the operator's eyes
  during the powered checks see it, and only if they watch.
- A robot of one's own whose embodiment records `<side>_gripper` needs a `grip_scale`, which the page does not ask
  for yet: add it to the saved file by hand.
- An arm from its URDF whose base link is not framed as YAM's (x forward, y left, z up): nothing checks the frame, and
  a base model's hand poses come out turned. Pick another base link, or add a fixed joint to the URDF.
- The checks need `dream_ik`, which needs Python 3.11 or newer; on an older Python the page says so when you pick a
  robot.
- Not built on purpose: a checklist by hand, replaying a dataset on the arm, a step that runs the policy, a
  recheck at every session start, and showing the reply time.

## Testing it without a robot

A virtual robot of Dream Labs' own (`--setup`) serves this page over simulated YAM arms, with simulated hands that
do what the page asks (its README, *Testing the setup page*).
