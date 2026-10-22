import tempfile
import unittest
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from terminal_input import SUGGESTIONS, CommandHistory, TaskCompleter, completion_bindings


class TerminalInputTests(unittest.TestCase):
    def test_tab_completion_and_up_arrow(self):
        with tempfile.TemporaryDirectory() as directory, create_pipe_input() as pipe:
            history = CommandHistory(str(Path(directory) / "history"))
            session = PromptSession(
                key_bindings=completion_bindings(),
                input=pipe,
                output=DummyOutput(),
                history=history,
                completer=WordCompleter(SUGGESTIONS, sentence=True),
                complete_while_typing=False,
            )
            pipe.send_text("put app\t\r")
            self.assertEqual(session.prompt("❯ "), "put apple into the bowl")
            pipe.send_text("\x1b[A\r")
            self.assertEqual(session.prompt("❯ "), "put apple into the bowl")
            persisted = CommandHistory(history.filename)
            self.assertIn("put apple into the bowl", list(persisted.load_history_strings()))

    def test_key_pastes_not_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history"
            history = CommandHistory(str(path))
            history.store_string("/login secret")
            history.store_string("dl_live_fake_test_key")
            self.assertFalse(path.exists())
            history.store_string("put apple into the bowl")
            self.assertIn("put apple into the bowl", path.read_text())

    def test_empty_tab_recommends_tasks_and_cycles(self):
        import threading
        import time

        with create_pipe_input() as pipe:
            session = PromptSession(
                key_bindings=completion_bindings(),
                input=pipe,
                output=DummyOutput(),
                completer=TaskCompleter(),
                complete_while_typing=False,
            )

            def keys():
                pipe.send_text("\t")
                time.sleep(0.1)
                pipe.send_text("\t\r")

            thread = threading.Thread(target=keys)
            thread.start()
            try:
                self.assertEqual(session.prompt("❯ "), "put orange into the bowl")
            finally:
                thread.join()
