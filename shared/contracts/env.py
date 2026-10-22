"""The environment reader every constant of `shared.contracts` uses for a boolean switch.

One reader, so a flag means the same words on a server (the servers' settings) and on a station
(`shared.contracts.station`): this module and `station` ship in the dream library, the servers' settings do not.
"""

import os


def flag_from_env(name: str, default: bool) -> bool:
    """The boolean in the environment variable `name`, or `default` when it is unset or empty.

    Empty equals unset because a deployment that passes the variable through to a container hands it an empty
    string, not an absence, when the key is left blank. The off tokens are the ones
    the gateway's own flag reader reads as off, so the deployment's two boolean knobs agree on what off is; where that
    reader takes any other word as on, this one refuses it, so a typo fails the start rather than picking a path.

    @param name The variable, e.g. `DREAMAPI_SERVE_TORCH_COMPILE`.
    @param default The shipped value.
    @return The value in effect.
    @throws ValueError If the variable is set to something other than `1`/`true`/`yes`/`on` or `0`/`false`/`no`/`off`
        (any case).
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    word = raw.strip().lower()
    if word in ("1", "true", "yes", "on"):
        return True
    if word in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be one of 1, true, yes, on, 0, false, no, off; got {raw!r}")
