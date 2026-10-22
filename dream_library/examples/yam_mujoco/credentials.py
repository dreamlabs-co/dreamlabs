"""Local API-key setup. Keys never appear in terminal output or command arguments."""

import getpass
import json
import os
import tempfile
import warnings
from pathlib import Path


def config_dir():
    """The user's directory for the sim's own files (key, history, diagnostics), never the install directory.
    @return `~/.config/dream-yam-sim`, created owner-only when missing.
    """
    path = Path.home() / ".config" / "dream-yam-sim"
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def credential_path():
    """Return the owner-local credential file path.
    @return The path of the JSON file `/login` saves the key in.
    """
    return config_dir() / "credentials.json"


def saved_key():
    """Read the saved API key without printing it.
    @return The saved key, or an empty string when none is saved.
    """
    path = credential_path()
    if not path.exists():
        return ""
    return json.loads(path.read_text()).get("api_key", "")


def save_key(key):
    """Atomically save an API key with owner-only permissions.
    @param key The API key to save.
    """
    if not key or any(c.isspace() for c in key):
        raise ValueError("Enter a nonempty API key without whitespace.")
    path = credential_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".credentials-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump({"api_key": key}, stream)
        os.replace(temporary, path)  # mkstemp creates the file with mode 0600.
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def login():
    """Prompt for and save a replacement API key.
    @return True when a key was saved, False when the prompt was cancelled or the key refused.
    """
    print("  Enter your Dream API key. Input is hidden; Enter cancels.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            key = getpass.getpass("  API key: ").strip()
        if not key:
            print("  Login cancelled.")
            return False
        save_key(key)
    except (EOFError, KeyboardInterrupt):
        print("\n  Login cancelled.")
        return False
    except (ValueError, getpass.GetPassWarning) as error:
        print(f"  Could not save key: {error}")
        return False
    print("  Key saved locally (owner-only file). Used on the next task; not yet verified by Dream.")
    if os.environ.get("DREAM_API_KEY"):
        print("  Note: DREAM_API_KEY is set in this shell and takes precedence; unset it to use the saved key.")
    return True
