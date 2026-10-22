"""`dreamcli robot run --view` (`robots.run_view`)."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any

import metal_fakes
import numpy as np
import pytest
from test_metal_body import metal_app

from robots import cli, config, errors

pytest.importorskip("dream_ik", reason="dream's IK needs Python 3.11 or newer")

from robots import run_view  # noqa: E402

ROWS = 48

MOUNT = {"kind": "box", "center": [-0.1, 0.33, 0.3], "size": [0.04, 0.04, 0.6]}


def app_with(
    tmp_path: Any, keep_out: tuple[dict[str, Any], ...] | None = None, enabled: bool = True
) -> config.AppConfig:
    """A Metal config whose rollouts go under `tmp_path`, the right arm keeping out of `keep_out`."""
    right = config.ArmConfig(can_channel="can_right", keep_out=keep_out, **metal_fakes.GRIPPER_ENDS)
    return metal_app(follower_r=right, rollouts=config.RolloutsConfig(enabled=enabled, dir=str(tmp_path)))


def write_chunk(tmp_path: Any, view: run_view.RunView, number: int) -> np.ndarray:
    """Saves a reply as the recorder does (a dict as its repr) whose tool poses are those of a joint path."""
    from dream_ik.decode import mat_to_rot6d

    q = view.model.rest + 0.2 * np.sin(np.linspace(0.0, 1.0, ROWS))[:, None]
    q[:2] = q[0]
    tool = np.stack([view.model.tool(row) for row in q])
    per_key = {}
    for side in ("left", "right"):
        per_key[f"action.{side}_eef_xyz"] = tool[:, :3, 3].tolist()
        per_key[f"action.{side}_eef_rot6"] = mat_to_rot6d(tool[:, :3, :3]).tolist()
        per_key[f"action.{side}_grip"] = np.linspace(1.0, 0.0, ROWS).tolist()
    chunk = np.column_stack([q, np.linspace(1.0, 0.0, ROWS)])
    folder = tmp_path / "2026-10-13T01-00-59_run1" / "chunks"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"reply_{number:06d}.npz"
    np.savez(path, per_key=np.array(repr(per_key)), q_l_chunk=chunk, q_r_chunk=chunk)
    old = time.time() - 1.0
    os.utime(path, (old, old))
    return q


def test_the_view_draws_the_newest_chunk_saved_since_it_started(tmp_path: Any) -> None:
    view = run_view.RunView(app_with(tmp_path), since_s=time.time() - 5.0)
    assert view.latest() == {"chunk": None}

    q = write_chunk(tmp_path, view, 3)
    latest = view.latest()

    assert (latest["episode"], latest["chunk"]) == ("2026-10-13T01-00-59_run1", 3)
    right = latest["sides"]["right"]
    assert np.allclose(right["model"], right["ik"], atol=1e-4)
    assert right["miss_mm"] < 0.2
    assert right["start"] == view.model.frame(q[0]) and right["end"] == view.model.frame(q[-1])
    assert (right["grip_start"], right["grip_end"], right["grip_asked"]) == (1.0, 0.0, [1.0, 0.0])
    assert len(right["approach"]) == len(range(0, ROWS, run_view.FRAME_EVERY))


def test_a_chunk_from_before_the_view_started_is_not_shown(tmp_path: Any) -> None:
    view = run_view.RunView(app_with(tmp_path), since_s=time.time() - 5.0)
    write_chunk(tmp_path, view, 0)
    later = run_view.RunView(app_with(tmp_path), since_s=time.time())
    assert later.latest() == {"chunk": None}


def test_the_view_draws_the_stations_keep_out_per_arm(tmp_path: Any) -> None:
    from dream_ik.specs import EMBODIMENTS

    own = {side: [o.model_dump() for o in obstacles] for side, obstacles in EMBODIMENTS["metal"].obstacles.items()}
    assert run_view.RunView(app_with(tmp_path, keep_out=(MOUNT,)), 0.0).view()["keep_out"] == {
        "left": own["left"],
        "right": [MOUNT],
    }
    assert run_view.RunView(app_with(tmp_path), 0.0).keep_out == own


def test_the_page_and_its_api_are_served_on_localhost(tmp_path: Any) -> None:
    server = run_view.start(app_with(tmp_path, keep_out=(MOUNT,)), port=0)
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        page = urllib.request.urlopen(f"{base}/").read().decode()
        assert "Policy chunks, live" in page and "keep_out" in page
        view = json.loads(urllib.request.urlopen(f"{base}/api/view").read())
        assert view["keep_out"]["right"] == [MOUNT] and view["drawing"]["kind"] == "meshes"
        assert json.loads(urllib.request.urlopen(f"{base}/api/latest").read()) == {"chunk": None}
        assert urllib.request.urlopen(f"{base}/setup_viewer.js").read().startswith(b"//")
    finally:
        server.shutdown()
    assert server.server_address[0] == "127.0.0.1"


@pytest.mark.parametrize(
    ("app", "message"),
    [
        (lambda tmp: app_with(tmp, enabled=False), "--view reads the chunks the run saves"),
        (lambda tmp: metal_app(body="yam", follower_r=config.ArmConfig(can_channel="can_right")), "answers in joints"),
    ],
    ids=["rollouts_off", "yam"],
)
def test_the_view_refuses_a_run_it_cannot_draw(tmp_path: Any, app: Any, message: str) -> None:
    with pytest.raises(errors.ConfigError, match=message):
        run_view.start(app(tmp_path), port=0)


def test_robot_run_view_is_parsed() -> None:
    args = cli._parser().parse_args(["robot", "run", "--task", "x", "--view", "--view-port", "8111"])
    assert (args.view, args.view_port) == (True, 8111)
    assert cli._parser().parse_args(["robot", "run"]).view is False
