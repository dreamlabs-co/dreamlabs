"""Task completion and local command history for Dream YAM Sim."""

import os
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import CompleteEvent, Completer, Completion, WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

## The task a first-time user's prompt starts with: Enter runs it.
DEFAULT_TASK = "put apple into the bowl"
RECOMMENDED_TASKS = [DEFAULT_TASK, "put orange into the bowl", "put banana on the plate"]
STYLE = Style.from_dict(
    {
        "prompt": "#d7875f bold",
        "completion-menu.completion": "bg:#292d35 #d7dce2",
        "completion-menu.completion.current": "bg:#d7875f #181a20 bold",
        "completion-menu.meta.completion": "bg:#292d35 #aab2bf",
        "completion-menu.meta.completion.current": "bg:#d7875f #181a20",
        "auto-suggestion": "#777f8c",
    }
)
SUGGESTIONS = RECOMMENDED_TASKS + [
    "place the apple into the bowl",
    "place the orange into the bowl",
    "place the banana onto the plate",
    "/help",
    "/login",
    "/reset",
    "/reset_robot",
    "/collisions",
    "/status",
    "/view",
    "/quit",
    "/model beta",
    "/actions 48",
    "/actions 24",
    "/seconds 120",
]


class TaskCompleter(Completer):
    """Complete common tasks and terminal commands."""

    def get_completions(self, document, complete_event):
        """Yield task or command completions matching the current input.
        @param document Current prompt document.
        @param complete_event Completion request context.
        @return The matching completions, one at a time.
        """
        if not document.text.strip():
            for task in RECOMMENDED_TASKS:
                yield Completion(task, display_meta="recommended task")
        else:
            yield from WordCompleter(SUGGESTIONS, sentence=True, ignore_case=True).get_completions(
                document, complete_event
            )


class CommandHistory(FileHistory):
    """Persist commands while excluding recognized credential input."""

    def store_string(self, string):
        # Login secrets are entered with getpass; also exclude accidental key pastes.
        """Persist a command unless it contains recognized credential input.
        @param string Input command text.
        """
        if string.strip().startswith("/login") or any(prefix in string for prefix in ("dl_live_", "msy_")):
            return
        super().store_string(string)


def completion_bindings():
    """Build the Tab completion key bindings.
    @return The key bindings for the prompt session.
    """
    bindings = KeyBindings()

    @bindings.add("tab")
    def complete(event):
        buffer = event.current_buffer
        if buffer.complete_state:
            buffer.complete_next()
            return
        matches = list(buffer.completer.get_completions(buffer.document, CompleteEvent(completion_requested=True)))
        if len(matches) == 1:
            buffer.apply_completion(matches[0])
        elif matches:
            buffer.start_completion(select_first=True)
        elif buffer.suggestion:
            buffer.insert_text(buffer.suggestion.text)

    return bindings


def history_path():
    """The file the command history is kept in.
    @return Its path, under the user's config directory.
    """
    return Path.home() / ".config" / "dream-yam-sim" / "history"  # credentials.config_dir(), without creating it


def is_first_run():
    """Whether no command has been entered before on this computer: the history file is missing or empty.
    @return True for a first run.
    """
    path = history_path()
    return not path.exists() or path.stat().st_size == 0


def create_session():
    """Create a styled prompt with persistent command history.
    @return The prompt session.
    """
    path = history_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    os.close(descriptor)
    path.chmod(0o600)
    return PromptSession(
        key_bindings=completion_bindings(),
        history=CommandHistory(str(path)),
        completer=TaskCompleter(),
        style=STYLE,
        reserve_space_for_menu=5,
        complete_while_typing=False,
        auto_suggest=AutoSuggestFromHistory(),
    )
