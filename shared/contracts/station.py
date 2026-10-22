"""The switches the dream library's station setup page reads: `dreamcli setup`'s feature flags.

They are read on the station, by the wizard process, never by a server, so they live here rather than in
the servers' settings, which the library does not ship. Each is off when its variable is
unset or empty (`shared.contracts.env.flag_from_env`), and each is declared in every environment file like any other
flag, which a repository test checks.
"""

from shared.contracts.env import flag_from_env

## Whether `dreamcli setup` serves the station setup page `robots/setup_wizard_v2.html`: one connection
## check that every step waits for (`GET /api/connection`), a banner while the station agent runs, a progress bar,
## and each error the wizard's routes answer shown as its own card with the fix beside it. Read
## on the station, by the wizard process, not by any server. `DREAM_SETUP_WIZARD_V2` sets it (`1`/`true`/`yes`/`on`
## or `0`/`false`/`no`/`off`); unset or empty is on, the one flag here that is: it is the page people see, and the
## one the platform's guided setup quotes by its button labels. `0` serves the other page (`robots/setup_wizard.html`).
SETUP_WIZARD_V2: bool = flag_from_env("DREAM_SETUP_WIZARD_V2", True)

## Whether the station setup page offers to switch a YAM arm's gripper motor to MIT mode itself (`POST
## /api/motors/gripper_mit`, `robots.motor_mode.set_mit()` on that arm's CAN link), in place of the four steps that
## stop setup and run `dreamcli motors set-mit`. Read on the station, by the wizard process, not by any server.
## `DREAM_SETUP_GRIPPER_MIT` sets it (`1`/`true`/`yes`/`on` or `0`/`false`/`no`/`off`); unset or empty is off, the
## four steps.
SETUP_GRIPPER_MIT: bool = flag_from_env("DREAM_SETUP_GRIPPER_MIT", False)

## Whether the station setup page checks each SO-101 calibration file against the arms that are plugged in when an
## SO-101 is picked (`GET /api/so101/match`): it reads what each arm's servos hold, without writing a register, and
## says per file which arm it matches, that it matches none, or that it is no calibration file at all. Read on the
## station, by the wizard process, not by any server. `DREAM_SETUP_SO101_MATCH` sets it (`1`/`true`/`yes`/`on` or
## `0`/`false`/`no`/`off`); unset or empty is off, the plain file list.
SETUP_SO101_MATCH: bool = flag_from_env("DREAM_SETUP_SO101_MATCH", False)
