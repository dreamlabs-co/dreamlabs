"""The driver's file picker: the station's own folders and a Python file's classes, as the page browses them, so
the operator picks a file on the station rather than typing a path."""

import pathlib
import re

import pytest

from robots import setup_files


def test_the_picker_opens_in_the_station_files_folder_or_its_nearest_parent_until_it_exists(
    tmp_path: pathlib.Path,
) -> None:
    """A first run on a new box has no station file's folder yet: the picker opens on what exists of its path."""
    assert setup_files.start(tmp_path / "station.yaml") == tmp_path.resolve()
    assert setup_files.start(tmp_path / "not" / "yet" / "station.yaml") == tmp_path.resolve()


def test_a_folder_lists_its_folders_and_python_files_sorted_without_hidden_ones(tmp_path: pathlib.Path) -> None:
    for folder in ("zeta", "Alpha", ".venv", "arms"):
        (tmp_path / folder).mkdir()
    for name in ("my_robot.py", "b.py", "station.yaml", ".hidden.py", "notes.txt"):
        (tmp_path / name).write_text("", encoding="utf-8")
    listing = setup_files.folder(tmp_path)
    assert listing == {
        "dir": str(tmp_path.resolve()),
        "parent": str(tmp_path.resolve().parent),
        "dirs": ["Alpha", "arms", "zeta"],
        "files": ["b.py", "my_robot.py"],
    }
    assert setup_files.folder(pathlib.Path("/"))["parent"] is None


def test_a_files_classes_are_read_in_its_order_without_running_it(tmp_path: pathlib.Path) -> None:
    """Only the top level's: a class inside a function is not one the config can name. Nothing runs: the file raises
    when imported."""
    driver = tmp_path / "my_robot.py"
    driver.write_text(
        'raise SystemExit("ran")\n\n\nclass Arm:\n    class Inner:\n        pass\n\n\n'
        "def helper():\n    class Hidden:\n        pass\n\n\nclass ReadOnlyArm(Arm):\n    pass\n",
        encoding="utf-8",
    )
    assert setup_files.classes(driver) == {"file": str(driver), "classes": ["Arm", "ReadOnlyArm"]}


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda tmp: pathlib.Path("relative/folder"), "relative/folder is not a folder on this computer"),
        (lambda tmp: tmp / "missing", "missing is not a folder on this computer"),
    ],
)
def test_a_path_that_is_no_folder_is_refused_by_name(tmp_path: pathlib.Path, make, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        setup_files.folder(make(tmp_path))


def test_a_file_that_is_no_python_is_refused_by_what_is_wrong(tmp_path: pathlib.Path) -> None:
    (tmp_path / "notes.txt").write_text("", encoding="utf-8")
    (tmp_path / "broken.py").write_text("class Arm(:\n", encoding="utf-8")
    with pytest.raises(ValueError, match="notes.txt is not a Python file on this computer"):
        setup_files.classes(tmp_path / "notes.txt")
    with pytest.raises(ValueError, match="broken.py is not valid Python: line 1: "):
        setup_files.classes(tmp_path / "broken.py")
