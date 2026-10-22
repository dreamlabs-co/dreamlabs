"""Tests of `dream.fine_tuning` against an in-process fake of the data service's `/v1/fine_tuning` routes, over real
HTTP, so the paths, the bearer and the bodies are asserted on the wire."""

from __future__ import annotations

import ast
import http.server
import inspect
import json
import pathlib
import re
import threading
from collections.abc import Iterator
from typing import Any

import pytest

import dream
from dream import fine_tuning
from dream.errors import DreamAuthError, DreamConfigError, DreamProtocolError, DreamServerError

## The api key of the fake's one account.
KEY = "dl_live_test"

## The fake's one job.
JOB_ID = "ftj_0123456789abcdef"

## That job as the service answers it.
JOB: dict[str, Any] = {
    "id": JOB_ID,
    "dataset": "alice/towels",
    "embodiment": "alice/arm",
    "base_model": "dreamlabs/dream-1-beta",
    "suffix": "towels",
    "fine_tuned_model": "dream-1-ft:0a1b2c3d-towels",
    "status": "queued",
    "hyperparameters": {"steps": 500, "learning_rate": 1e-05, "batch_size": 32},
    "queue_position": 1,
    "created_at": "2026-09-25T10:00:00Z",
    "finished_at": None,
    "error": None,
}

## The two bodies the fake's account has: joints only, and the same two arms recorded with their end-effector pose
## (the six EEF parts of `shared.datasets.embodiments.EEF_PARTS` beside the joints, 34 wide).
JOINTS_LAYOUT = {"right_arm": [0, 6], "right_gripper": [6, 7], "left_arm": [7, 13], "left_gripper": [13, 14]}
EEF_LAYOUT = {
    **JOINTS_LAYOUT,
    "right_eef_xyz": [14, 17],
    "right_eef_rot6": [17, 23],
    "right_grip": [23, 24],
    "left_eef_xyz": [24, 27],
    "left_eef_rot6": [27, 33],
    "left_grip": [33, 34],
}
VIDEO = {"overhead": "observation.images.overhead", "wrist_l": "observation.images.wrist_l"}
## Two bodies that say what each part is (`parts`): both arms' joints and jaws, which train as joints, and the same
## with each arm's grasp point and xyzw quaternion, whose gripper poses train.
_JAWS = {f"{side}_jaw": {"kind": "gripper", "side": side, "closed": 0, "open": 0.044} for side in ("left", "right")}
KINDS_PARTS = {**{f"{side}_joints": {"kind": "arm", "side": side} for side in ("left", "right")}, **_JAWS}
KINDS_LAYOUT = {"left_joints": [0, 6], "left_jaw": [6, 7], "right_joints": [7, 13], "right_jaw": [13, 14]}
_GRASP = {"kind": "position", "point": "grasp", "frame": "arm_base"}
POSE_PARTS = {
    **KINDS_PARTS,
    **{f"{side}_xyz": {**_GRASP, "side": side} for side in ("left", "right")},
    **{f"{side}_quat": {"kind": "rotation", "side": side, "format": "quat_xyzw"} for side in ("left", "right")},
}
POSE_LAYOUT = {
    **KINDS_LAYOUT,
    "left_xyz": [14, 17],
    "left_quat": [17, 21],
    "right_xyz": [21, 24],
    "right_quat": [24, 28],
}
EMBODIMENTS: dict[str, dict[str, Any]] = {
    "alice/arm": {"state": JOINTS_LAYOUT, "action": JOINTS_LAYOUT, "video": VIDEO},
    "alice/eef": {"state": EEF_LAYOUT, "action": EEF_LAYOUT, "video": VIDEO},
    "alice/kinds": {
        "state": KINDS_LAYOUT,
        "action": KINDS_LAYOUT,
        "video": VIDEO,
        "parts": {"state": KINDS_PARTS, "action": KINDS_PARTS},
        "actions": "absolute",
    },
    "alice/poses": {
        "state": POSE_LAYOUT,
        "action": POSE_LAYOUT,
        "video": VIDEO,
        "parts": {"state": POSE_PARTS, "action": POSE_PARTS},
        "actions": "absolute",
    },
}

## The fake's datasets, one per body, as `GET /v1/datasets` lists them.
DATASETS: list[dict[str, Any]] = [
    {
        "id": f"ds_{index:016x}",
        "name": name,
        "embodiment": body,
        "status": "processed",
        "size_bytes": 4096,
        "episodes": 12,
        "error": None,
        "created_at": "2026-09-25T09:00:00Z",
    }
    for index, (name, body) in enumerate(
        [
            ("alice/towels", "alice/arm"),
            ("alice/eef-towels", "alice/eef"),
            ("alice/kinds-towels", "alice/kinds"),
            ("alice/pose-towels", "alice/poses"),
        ]
    )
]

## The line as the service reports it.
QUEUE: dict[str, Any] = {
    "jobs_ahead_of_new": 1,
    "running": 2,
    "slots_free": 0,
    "slots_total": 2,
    "updated_at": "2026-09-25T10:00:00Z",
}


class FakeFineTuning:
    """The data service's fine-tuning routes in one threaded HTTP server, recording every request.

    Attributes:
    - url: The base URL.
    - requests: `(method, path, authorization, body)` per request that reached it.
    - refusal: `(status, code, message)` every request is answered with instead, when set.
    - answer: What the next successful request is answered with instead of the route's own answer, when set.
    """

    def __init__(self) -> None:
        """Starts the server on a free port."""
        self.requests: list[tuple[str, str, str | None, Any]] = []
        self.refusal: tuple[int, str, str] | None = None
        self.answer: Any = None
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802  # http.server API
                fake.handle(self)

            def do_POST(self) -> None:  # noqa: N802  # http.server API
                fake.handle(self)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002  # http.server API
                pass

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        """Stops the server."""
        self._server.shutdown()
        self._server.server_close()

    def handle(self, request: http.server.BaseHTTPRequestHandler) -> None:
        """Records one request and answers it."""
        length = int(request.headers.get("Content-Length") or 0)
        raw = request.rfile.read(length)
        body = json.loads(raw) if raw else None
        self.requests.append((request.command, request.path, request.headers.get("Authorization"), body))
        if request.headers.get("Authorization") != f"Bearer {KEY}":
            status, answer = 401, _json_error("invalid_api_key", "This api key is not valid.")
        elif self.refusal is not None:
            status, answer = self.refusal[0], _json_error(*self.refusal[1:])
        elif self.answer is not None:
            status, answer = 200, self.answer
        else:
            status, answer = self._route(request.command, request.path, body)
        payload = json.dumps(answer).encode()
        request.send_response(status)
        request.send_header("Content-Length", str(len(payload)))
        request.end_headers()
        request.wfile.write(payload)

    def _route(self, method: str, path: str, body: Any) -> tuple[int, Any]:
        if (method, path) == ("POST", "/v1/fine_tuning/jobs"):
            knobs = {**JOB["hyperparameters"], **body.get("hyperparameters", {})}
            return 201, {**JOB, "dataset": body["dataset"], "suffix": body["suffix"], "hyperparameters": knobs}
        if (method, path) == ("GET", "/v1/fine_tuning/jobs"):
            return 200, [JOB, {**JOB, "id": "ftj_fedcba9876543210", "status": "succeeded"}]
        if (method, path) == ("GET", "/v1/fine_tuning/queue"):
            return 200, QUEUE
        if (method, path) == ("GET", f"/v1/fine_tuning/jobs/{JOB_ID}"):
            return 200, JOB
        if (method, path) == ("POST", f"/v1/fine_tuning/jobs/{JOB_ID}/cancel"):
            return 200, {**JOB, "status": "cancelled", "queue_position": None}
        if (method, path) == ("GET", "/v1/fine_tuning/models/dream-1-ft%3A0a1b2c3d-towels/training_data"):
            return 200, {"url": "https://bucket.example/alice/towels/dataset.zip?sig", "video": VIDEO}
        if (method, path) == ("GET", "/v1/datasets"):
            return 200, DATASETS
        if method == "GET" and path.startswith("/v1/embodiments/"):
            name = path[len("/v1/embodiments/") :]
            if name in EMBODIMENTS:
                return 200, {
                    "name": name,
                    "embodiment_config": EMBODIMENTS[name],
                    "created_at": "2026-09-24T11:00:00Z",
                }
        return 404, _json_error("not_found", f"No {path}.")


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeFineTuning]:
    """The fake, named by `DREAM_DATA_URL` so every call reaches it; stopped after the test."""
    fake = FakeFineTuning()
    monkeypatch.setenv("DREAM_DATA_URL", fake.url)
    yield fake
    fake.close()


def test_create_posts_the_dataset_the_suffix_and_the_knobs_named_and_answers_the_job(
    service: FakeFineTuning,
) -> None:
    job = fine_tuning.create("alice/towels", "towels", learning_rate=2e-5, batch_size=16, api_key=KEY)
    assert service.requests == [
        (
            "POST",
            "/v1/fine_tuning/jobs",
            f"Bearer {KEY}",
            {
                "dataset": "alice/towels",
                "suffix": "towels",
                "hyperparameters": {"learning_rate": 2e-5, "batch_size": 16},
            },
        )
    ]
    assert job == {**JOB, "hyperparameters": {"steps": 500, "learning_rate": 2e-5, "batch_size": 16}}


def test_create_without_a_knob_sends_no_hyperparameters(service: FakeFineTuning) -> None:
    job = fine_tuning.create("alice/towels", "towels", api_key=KEY)
    (request,) = service.requests
    assert request[3] == {"dataset": "alice/towels", "suffix": "towels"}
    assert job["hyperparameters"] == JOB["hyperparameters"], "the service's defaults"


def test_create_sends_only_the_knobs_named(service: FakeFineTuning) -> None:
    """A knob left out takes the service's recipe default, so only the named ones travel."""
    fine_tuning.create("alice/towels", "towels", batch_size=16, api_key=KEY)
    (request,) = service.requests
    assert request[3] == {"dataset": "alice/towels", "suffix": "towels", "hyperparameters": {"batch_size": 16}}


def test_create_sends_the_lr_schedule_named_and_none_leaves_it_to_the_service(service: FakeFineTuning) -> None:
    fine_tuning.create("alice/towels", "towels", lr_schedule="cosine", api_key=KEY)
    dream.DreamClient(api_key=KEY).finetune("alice/towels", "towels", lr_schedule="constant")
    fine_tuning.create("alice/towels", "towels", lr_schedule=None, api_key=KEY)
    assert [request[3].get("hyperparameters") for request in service.requests] == [
        {"lr_schedule": "cosine"},
        {"lr_schedule": "constant"},
        None,
    ]


def test_create_with_use_eef_checks_the_datasets_body_first_and_sends_the_switch_in_the_knobs(
    service: FakeFineTuning,
) -> None:
    """The dataset is found by name among the account's, its embodiment fetched and held to the six EEF parts, and
    only then does the create go out, with `use_eef` beside the knobs; `use_eef=False` sends nothing."""
    job = fine_tuning.create("alice/eef-towels", "eef1", use_eef=True, api_key=KEY)
    assert [(method, path) for method, path, _, _ in service.requests] == [
        ("GET", "/v1/datasets"),
        ("GET", "/v1/embodiments/alice/eef"),
        ("POST", "/v1/fine_tuning/jobs"),
    ]
    assert service.requests[-1][3] == {
        "dataset": "alice/eef-towels",
        "suffix": "eef1",
        "hyperparameters": {"use_eef": True},
    }
    assert job["hyperparameters"]["use_eef"] is True
    service.requests.clear()
    fine_tuning.create("alice/eef-towels", "eef2", use_eef=False, api_key=KEY)
    (request,) = service.requests
    assert request[3] == {"dataset": "alice/eef-towels", "suffix": "eef2"}, "off by default: no read, no switch"


def test_create_with_use_eef_refuses_a_joints_only_body_before_anything_is_queued(
    service: FakeFineTuning,
) -> None:
    with pytest.raises(DreamConfigError) as refused:
        fine_tuning.create("alice/towels", "towels", use_eef=True, api_key=KEY)
    assert str(refused.value) == (
        "alice/towels: use_eef needs the state part 'right_eef_xyz' (3 wide); the embodiment config has no such "
        "state part."
    )
    assert [(method, path) for method, path, _, _ in service.requests] == [
        ("GET", "/v1/datasets"),
        ("GET", "/v1/embodiments/alice/arm"),
    ], "nothing was posted"
    # check_eef() on its own is the same check; an EEF body passes it.
    fine_tuning.check_eef("alice/eef-towels", api_key=KEY)


def test_use_eef_on_a_body_with_parts_passes_only_when_its_parts_train_the_gripper_poses(
    service: FakeFineTuning,
) -> None:
    """A body that says what each part is is checked as training reads it: its poses under the EEF names."""
    fine_tuning.check_eef("alice/pose-towels", api_key=KEY)
    with pytest.raises(DreamConfigError) as refused:
        fine_tuning.create("alice/kinds-towels", "kinds", use_eef=True, api_key=KEY)
    assert str(refused.value) == (
        "alice/kinds-towels: its robot's parts leave the gripper poses recorded, not trained, so it cannot take "
        "use_eef; leave use_eef out and the parts decide it."
    )
    assert service.requests[-1][:2] == ("GET", "/v1/embodiments/alice/kinds"), "nothing was posted"


def test_create_with_use_eef_on_a_dataset_the_service_does_not_list_leaves_the_refusal_to_the_service(
    service: FakeFineTuning,
) -> None:
    fine_tuning.create("alice/nothing", "x", use_eef=True, api_key=KEY)
    assert [(method, path) for method, path, _, _ in service.requests] == [
        ("GET", "/v1/datasets"),
        ("POST", "/v1/fine_tuning/jobs"),
    ], "no embodiment to check locally; the service answers not_found on the create"
    assert service.requests[-1][3]["hyperparameters"] == {"use_eef": True}


def test_list_get_cancel_and_queue_hit_their_paths_with_the_bearer_and_answer_the_json(
    service: FakeFineTuning,
) -> None:
    listed = fine_tuning.list(api_key=KEY)
    assert [job["id"] for job in listed] == [JOB_ID, "ftj_fedcba9876543210"]
    assert fine_tuning.get(JOB_ID, api_key=KEY) == JOB
    cancelled = fine_tuning.cancel(JOB_ID, api_key=KEY)
    assert (cancelled["status"], cancelled["queue_position"]) == ("cancelled", None)
    assert fine_tuning.queue(api_key=KEY) == QUEUE
    assert [(method, path, bearer) for method, path, bearer, _ in service.requests] == [
        ("GET", "/v1/fine_tuning/jobs", f"Bearer {KEY}"),
        ("GET", f"/v1/fine_tuning/jobs/{JOB_ID}", f"Bearer {KEY}"),
        ("POST", f"/v1/fine_tuning/jobs/{JOB_ID}/cancel", f"Bearer {KEY}"),
        ("GET", "/v1/fine_tuning/queue", f"Bearer {KEY}"),
    ]
    assert all(body is None for _, _, _, body in service.requests), "the reads and the cancel carry no body"


def test_training_data_asks_for_the_model_quoted_into_the_path(service: FakeFineTuning) -> None:
    found = fine_tuning.training_data("dream-1-ft:0a1b2c3d-towels", api_key=KEY)
    assert found == {"url": "https://bucket.example/alice/towels/dataset.zip?sig", "video": VIDEO}
    with pytest.raises(DreamServerError) as refused:
        fine_tuning.training_data("someone-elses", api_key=KEY)
    assert refused.value.code == "not_found"


def test_a_job_id_is_quoted_into_the_path(service: FakeFineTuning) -> None:
    with pytest.raises(DreamServerError):
        fine_tuning.get("a/b?c", api_key=KEY)
    assert service.requests[0][1] == "/v1/fine_tuning/jobs/a%2Fb%3Fc"


def test_a_refusal_is_a_server_error_with_the_services_code_and_message_and_a_bad_key_an_auth_error(
    service: FakeFineTuning,
) -> None:
    service.refusal = (409, "conflict", "Suffix 'towels' was used before; dream-1-ft:0a1b2c3d-towels is taken.")
    with pytest.raises(DreamServerError) as refused:
        fine_tuning.create("alice/towels", "towels", api_key=KEY)
    assert (refused.value.code, str(refused.value), refused.value.retryable) == (
        "conflict",
        "Suffix 'towels' was used before; dream-1-ft:0a1b2c3d-towels is taken.",
        False,
    )
    service.refusal = None
    with pytest.raises(DreamServerError) as missing:
        fine_tuning.get("ftj_0000000000000000", api_key=KEY)
    assert missing.value.code == "not_found"
    with pytest.raises(DreamAuthError):
        fine_tuning.list(api_key="dl_live_other")


def test_an_answer_that_is_not_a_json_object_is_a_protocol_error(service: FakeFineTuning) -> None:
    service.answer = ["not", "a", "job"]
    with pytest.raises(DreamProtocolError, match="not a JSON object"):
        fine_tuning.get(JOB_ID, api_key=KEY)


def test_a_key_and_a_host_left_out_come_from_the_credentials(
    service: FakeFineTuning, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DREAM_CREDENTIALS", str(tmp_path / "credentials"))
    monkeypatch.delenv("DREAM_API_KEY", raising=False)
    monkeypatch.delenv("DREAM_DATA_URL", raising=False)
    with pytest.raises(DreamConfigError, match="no api key"):
        fine_tuning.queue()
    (tmp_path / "credentials").write_text(f"DREAM_API_KEY={KEY}\nDREAM_DATA_URL={service.url}\n")
    # The key and the data service's URL both come from the credentials when neither is passed.
    assert fine_tuning.queue() == QUEUE


def _json_error(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


## The service's recipe: the bounds and defaults of the three knobs, which the client docstrings state.
RECIPES = pathlib.Path(__file__).resolve().parents[3] / "data_service" / "src" / "data_service" / "faas" / "recipes.py"


def _recipe_constants() -> dict[str, Any]:
    """The literal constants of the recipe module, read from its source: the client's environment cannot import it."""
    constants: dict[str, Any] = {}
    for node in ast.parse(RECIPES.read_text(encoding="utf-8")).body:
        target = node.target if isinstance(node, ast.AnnAssign) else None
        if isinstance(target, ast.Name) and isinstance(node, ast.AnnAssign) and node.value is not None:
            try:
                constants[target.id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return constants


def _param(doc: str, name: str) -> str:
    """One `@param` tag's text, its continuation lines joined."""
    found = re.search(rf"@param {name} (.*?)(?=\n\s*@|\Z)", doc, re.S)
    assert found, f"no @param {name}"
    return " ".join(found.group(1).split())


@pytest.mark.parametrize("member", [fine_tuning.create, dream.DreamClient.finetune], ids=["create", "finetune"])
def test_the_knob_ranges_and_defaults_the_docstrings_state_are_the_services(member: Any) -> None:
    """The docstrings state the learning rate's and the schedule's range and default from the
    data service's training recipes, so a change there fails here until both docstrings say the same; they state no
    step count and no batch size, which users are not shown."""
    recipe = _recipe_constants()
    doc = member.__doc__

    assert "steps" not in inspect.signature(member).parameters, "steps are fixed; a steps argument would be refused"
    assert f"{recipe['STEPS']} steps" not in " ".join(doc.split()), doc

    rate = re.fullmatch(
        r"The learning rate, (\S+) to (\S+); None takes the service's default of (\S+)\.", _param(doc, "learning_rate")
    )
    assert rate, _param(doc, "learning_rate")
    assert (float(rate[1]), float(rate[2])) == recipe["LEARNING_RATE_RANGE"]
    assert float(rate[3]) == recipe["DEFAULT_LEARNING_RATE"]

    assert _param(doc, "batch_size") == "Leave it as None: the service sets it."

    schedule = re.match(
        r'The learning-rate schedule, ([`"\w ]+); None takes the service\'s default, `"(\w+)"`\.',
        _param(doc, "lr_schedule"),
    )
    assert schedule, _param(doc, "lr_schedule")
    assert tuple(re.findall(r'`"(\w+)"`', schedule[1])) == recipe["LR_SCHEDULES"]
    assert schedule[2] == recipe["DEFAULT_LR_SCHEDULE"]
