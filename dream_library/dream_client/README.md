# dream

The Dream Labs client: fine-tune Dream-1 on your robot's data, and call the result for actions.

The public surface is one object, `dream.DreamClient`: it registers a robot body, checks and uploads a dataset, fine-tunes Dream-1 on it, lists the models your account may run, and runs a model for actions (see *Running a model* and *Fine-tuning as a Service*). It carries your key, so no call passes it; where the Dream Labs services are is set by Dream Labs in your credentials file, never by your code. The rest of the public API is the error types, the one warning type, `__version__`, and three submodules `DreamClient` is a front over: `dream.datasets` (embodiments and datasets on the data service), `dream.fine_tuning` (fine-tuning jobs), and `dream.models` (the checkpoints an account may run). The platform site's API documentation page is the reference, and its Fine-tuning page walks the whole path step by step, with a notebook to download.

## Quickstart

```bash
pip install dreamlabs   # the library: this client (dreamlabs-client) and everything fine-tuning needs
```

The client is `dreamlabs-client` on PyPI and the whole library, which brings it, is `dreamlabs`, for Python 3.10 to 3.12. `pip install dreamlabs` carries everything fine-tuning needs, with no extra to add; the client alone is `pip install dreamlabs-client`, which adds the dataset checks and upload with `pip install "dreamlabs-client[datasets]"` and the live view's encoder with `[live]`. A platform site also serves its own build of both, which its Setup API page installs; on a machine with no browser, fetch the two files with your api key and install both: `curl -fOJ -H "Authorization: Bearer $DREAM_API_KEY" https://<the platform site>/downloads/latest-client`, the same for `/downloads/latest`, then `pip install ./dreamlabs_client-*.whl ./dreamlabs-*.whl`. `pip install dream` installs an unrelated project.

The credentials file, `~/.dreamlabs/credentials`, is four lines a shell could `source`; the platform's Fine-tuning page prints them filled in for its own site, and these are production's. Put your key in the last line and copy the three addresses as they are printed: they are set by Dream Labs, and nothing in your code sets them.

```
DREAM_GATEWAY_URL=https://api.dreamlabs.ai
DREAM_API_URL=https://platform.dreamlabs.ai
DREAM_DATA_URL=https://data.dreamlabs.ai
DREAM_API_KEY=dl_live_...
```

`dream.DreamClient` is the whole lifecycle in one object; the key comes from that file:

```python
import dream

client = dream.DreamClient()
embodiment = client.add_embodiment("makersmod-yam", config)  # describe the robot body once
dataset = client.upload_dataset("~/data/towels", "towels-v1", embodiment.name)  # keep dataset.id
job = client.finetune(dataset.name, "towels")  # once the dataset is processed
client.policy({"checkpoint": job["fine_tuned_model"]})  # once the job publishes the checkpoint
client.set_observation(frames, observation)  # every camera tick: one base64 JPEG per camera, and the state
action = client.get_action()  # the frames the client kept, with the newest state, checked before it is sent
```

*Fine-tuning as a Service* below walks every step; the platform's Fine-tuning page is the same as a tutorial, with a notebook to download.

### Running a model

Running a model is three calls on the same object: `client.policy(config)` chooses the checkpoint and returns nothing, `client.set_observation(frames, observation)` hands the client every camera tick, and `client.get_action()` sends the frames the client kept, with the newest state, and returns the next chunk of actions. The client keeps each camera's recent frames itself, in its frame window, so a tick carries one frame per camera, never a list of them. The full loop, for the YAM model `dreamlabs/dream-1-beta`:

```python
import threading

import dream

config = {
    "timeout_ms": 15000,
    "strict": True,  # validate observations and actions against "embodiment"; leave it on
    "checkpoint": "dreamlabs/dream-1-beta",  # required: the model to run, an id get_models() lists
    "ik": {"arm": "yam", "head": "joint"},  # dream-1-beta reads hand poses: the client adds them
    "embodiment": {  # the YAM body's embodiment config: what every observation and reply is checked against
        "state": {"right_arm": [0, 6], "right_gripper": [6, 7], "left_arm": [7, 13], "left_gripper": [13, 14]},
        "action": {"right_arm": [0, 6], "right_gripper": [6, 7], "left_arm": [7, 13], "left_gripper": [13, 14]},
        "video": {role: f"observation.images.{role}" for role in ("overhead", "wrist_l", "wrist_r")},
    },
}


def tick(robot):
    """One camera tick: one frame per camera, and every other key the model reads."""
    frames = {  # each one base64 JPEG str with no data: prefix, resized to 320x176 and encoded at quality 85
        "video.overhead": robot.overhead_camera.jpeg(),
        "video.wrist_l": robot.left_wrist_camera.jpeg(),
        "video.wrist_r": robot.right_wrist_camera.jpeg(),
    }
    observation = {
        "state.right_arm": robot.right_arm.joints(),  # float32 [6], radians
        "state.right_gripper": robot.right_arm.gripper(),  # float32 [1], 0 closed to 1 open
        "state.left_arm": robot.left_arm.joints(),
        "state.left_gripper": robot.left_arm.gripper(),
        "prompt": "fold the towel",
    }
    return frames, observation


with dream.DreamClient() as client:
    client.policy(config)
    client.set_observation(*tick(robot))  # the episode's first tick, before the first get_action()

    def act():
        for _ in range(100):
            robot.play(client.get_action())  # one get_action() at a time, on this thread

    worker = threading.Thread(target=act)
    worker.start()
    while worker.is_alive():  # every camera tick, at the cameras' own rate (30 Hz on a YAM station)
        robot.wait_for_frames()
        client.set_observation(*tick(robot))
```

`policy()` sends nothing to the gateway: the session opens on the first `get_action`, which is a `POST /v1/get_action` carrying `session_id: null`; DreamAPI mints a session and every later request echoes its id. Calling `policy()` again closes the previous policy, which ends its session and drops the frames it kept, and starts over. The context manager is the documented default; `client.close()` is there for callers that cannot use one.

`set_observation(frames, observation, *, timestamp=None)` takes one camera tick, at whatever rate the cameras run. `frames` maps each window key (the observation key a camera's frames are sent under) to that tick's one frame: a base64 JPEG `str` with no `data:` prefix, which you resize and encode (320x176 at JPEG quality 85 is what the models read). The window keys are `video.<role>`: `video.overhead`, `video.wrist_l` and `video.wrist_r` for `dreamlabs/dream-1-beta` on a YAM (the older `image_window.top`, `image_window.left` and `image_window.right` are still accepted but deprecated), and one per role of a fine-tuned model's embodiment `video` (*Running a fine-tuned model*); pass the same keys on every call of an episode. `observation` is every other key the model reads, the `state.<part>` arrays and `prompt`, never a frame key; each call replaces the previous call's observation whole. `timestamp` is when the frames were captured, in seconds on a monotonic clock (`time.monotonic()` at the call when it is `None`), never earlier than the previous call's within an episode.

The client keeps one tick every `1 / obs_fps` seconds of those timestamps (5 per second by default, the tick nearest each due time) and the last `obs_window` frames per window key (9 by default); every tick, kept or not, replaces the latest observation. It never waits on the network or on a `get_action` running on another thread, so the camera loop can call it on every tick.

`get_action()` takes no observation: it sends the latest observation plus, under each window key, the frames kept so far, oldest first (fewer than `obs_window` at the start of an episode, never none), and returns the action chunk. Before any `set_observation` in the episode it raises `DreamObservationError`. One `get_action` runs at a time, and a second one at once raises `DreamError`: run it on a thread of its own while `set_observation` keeps going. Its one keyword, `commanded`, anchors seam smoothing (*EEF checkpoints and client IK*) and is never sent.

The inference surface is `policy`, `set_observation`, `get_action`, `reset_episode`, `refill_due` and `close`, and the read-only properties `last_timing` (see "Timing" below), `mode`, `execution_horizon` and `action_hz` (see "Pacing" below), and `session_id`, the session the model answers the episode under once a `get_action` opened it. Called before `policy()`, each of them raises `DreamConfigError`, except `close` (which does nothing) and `last_timing` and `session_id` (which are `None`).

`client.reset_episode()` marks an episode boundary: it ends the session and empties the frame window, so the next `set_observation` is the first tick of a new episode and the next `get_action` carries `session_id: null`, for which DreamAPI mints a fresh session with a forced reset. The client releases the old session on the model's side in one short request that never raises (it waits at most two seconds; a session it could not release ends on its own once idle). Call it between trials:

```python
with dream.DreamClient() as client:
    client.policy(config)
    for trial in range(10):
        client.reset_episode()
        robot.reset()
        run_episode(client, robot)  # set_observation() every camera tick, get_action() on its own thread
```

`get_action` returns the action dict itself. There is no `info` tuple; DreamAPI does not produce
one.

### Timing

`client.last_timing` says where the milliseconds of the most recent `get_action` went. It is
read-only, `None` before the first request, and afterwards a plain dict, recorded for every
request that reached the wire, the ones that raised included, so a timeout is visible as its
`rtt_ms`:

```python
action = client.get_action()
t = client.last_timing
# {"rtt_ms": 512.3, "serve_infer_ms": 412, "gateway_ms": 460, "network_ms": 52.3,
#  "up_mbps": 38.2, "tcp_cwnd_kb": 140.8, "tcp_retrans": 0, "tcp_rtt_ms": 21.4, "tcp_app_limited": False,
#  "request_bytes": 1_204_311, "response_bytes": 2_140}
```

`rtt_ms` is the client's own round trip around the HTTP exchange, connect included.
`serve_infer_ms` and `gateway_ms` come from DreamAPI's `timing` block: the serve's own inference
time and the wall time inside the gateway from receiving the request to building the response.
`network_ms` is `rtt_ms - gateway_ms`, the part spent on the network. The three are `None` when
DreamAPI sent no block: a refusal, or a gateway older than the block. `request_bytes` and
`response_bytes` are the body sizes on the wire; `response_bytes` is `0` when no reply was read.

Five more keys come from the kernel's own record of the request's connection, and are `None` off Linux: `up_mbps`, the upload rate achieved for this request in megabits per second, the first number to check when the round trip drifts; `tcp_cwnd_kb`, the congestion window in KB (small means the connection, not the bandwidth, limited the upload); `tcp_retrans`, the retransmits on the connection so far (a rising count is packet loss on the send path); `tcp_rtt_ms`, the connection's smoothed round-trip time; and `tcp_app_limited`, `True` when the upload was held back by the program not having bytes ready rather than by the network.

### Slow-network warning

When the client's recent round trips stay slow, `get_action` says so once with a Python warning of category `dream.DreamSlowNetworkWarning` (a `UserWarning`), and writes the same line to the `dream` logger, so it is in the log file too. It fires when the median of the last five `get_action` round trips is at or above 1000 ms, and names where the time goes: the network, when it is the largest part of the round trip (with the upload rate when it is known), or the model:

```text
DreamSlowNetworkWarning: Network is slow: get_action round trips are about 1450 ms (network 1100 ms, upload 2.1 Mb/s); the robot's motion may be less smooth. See client.last_timing.
```

It is not raised: the action is returned as usual, even when warnings are turned into errors. After one warning the client stays quiet for five minutes while the round trips stay slow, unless they first recover (their median under 800 ms) and then slow down again. To turn it off, use the standard filter; there is no setting or environment variable for it:

```python
import warnings

warnings.filterwarnings("ignore", category=dream.DreamSlowNetworkWarning)
```

### Observation and action keys

The observation is the serve's own dict, which the client builds from your calls and sends as the request's `observation` field: the latest `set_observation`'s `observation`, where `state.<part>` entries are float32 arrays (for `dreamlabs/dream-1-beta` on a YAM, `state.right_arm` of shape `[6]`, `state.right_gripper` of shape `[1]`, and the same for the left arm, plus each hand's pose, which `ik` adds) and `prompt` is the task string, plus one list per window key (`video.overhead`, `video.wrist_l`, `video.wrist_r`) holding the base64 JPEG frames the client kept for that camera, oldest first. When the config's `robot` names one, or the client picks one for it (*EEF checkpoints and client IK*), the observation also carries `embodiment`, a string: `"yam"`, the state is the model's own and is read as sent, or `"other_embodiment"`, a Dream-1 base model reads none of the state. It is the client's to send: an observation of yours that carries `embodiment` is refused with `DreamObservationError`. Numpy arrays go on the wire as JSON lists.

The action is the serve's action dict: `q_r_chunk` and `q_l_chunk` are `[H, joints + 1]` chunks, one row
per step of the horizon, with the arm's joint targets (six on a YAM) followed by the gripper as the last
column; the client filters them and blends them into the previous chunk before returning them
(see "Seam smoothing" below). The serve adds fields beyond those (`q_r`, `q_l`, `per_key`,
extra-DOF slots such as `lower_body_chunk`), and they are passed through untouched.

These are the YAM station keys of `dreamlabs/dream-1-beta`. A fine-tuned model reads the keys of the embodiment config it was trained on instead, and answers under `per_key`: *Fine-tuning as a Service*, *Running a fine-tuned model*.

The `embodiment` block groups those keys for the strict-mode checks: every group except `action`
describes observation keys, flattened to `"<group>.<key>"`, and the `action` group describes the
action keys by their bare names. An empty entry (`{}`) only requires the key to be present.

The `embodiment` block may instead be the embodiment config of a fine-tuned body, the dict `add_embodiment` took; the client tells the two apart by their values (every entry of a shape spec is a dict, a config's are `[start, end]` spans and video keys), and checks the config's keys as *Running a fine-tuned model* describes.

### Seam smoothing

DreamAPI computes every chunk from one observation and knows nothing of the chunk before it, so where a control loop swaps the old chunk for the new one the target can step in position and in velocity: the hitch a policy shows at every refill. The client therefore shapes every chunk before the loop sees it, from what it already holds: the observation the chunk answers (the arms' measured joint pose, `state.<arm>.joint_pos`, which says which row of the previous chunk the arms are on), the chunk it returned last (the plan the arms follow until this one lands), the round trip it measured, and your loop's row rate (`action_hz`), which together say which row of the new chunk your loop starts at.

This is always on and has no setting: the numbers are the library's, so every chunk a control loop receives is as smooth as the library can make it. Per arm: a Savitzky-Golay filter (window 11, order 3, run at twice the row rate) removes per-row jitter from the row your loop starts at, every column included; then the 16 rows from there are a velocity blend that starts exactly where the arm's current motion goes next, at its speed, hands that velocity over to the new plan's, and lands exactly on the new plan's row at the end, so position and velocity carry over the seam and the arm is back on the model's plan after 16 rows. An arm that has run out of rows and is holding (sync pacing) starts from rest.

Pass the arms' kinematics to have the blend done on the gripper's pose rather than on the joints: the config key `kinematics` takes `q_r_chunk` and `q_l_chunk` to an object with `pose(joints)`, returning the end effector's `(position [3], rotation [3, 3])`, and `jacobian(joints)`, its `[6, joints]` Jacobian (linear rows, then angular). The blend then runs on the path the gripper actually traces and is solved back to joints by damped least squares; a seam whose solve fails falls back to the joint-space blend. The calls come from the thread that runs `get_action`, so give the client its own instances.

Only `q_r_chunk` and `q_l_chunk` are rewritten, as lists of rows of the same shape; every other key, `per_key` included, comes back as the serve sent it. Filtered but not joined, since there is no commanded motion to continue: the first chunk of an episode, the first after `reset_episode()` or a session replaced mid-episode, an arm whose measured pose is more than 0.5 rad (L2 over the arm's joints) from every row of the previous chunk (an arm that has left the plan: a reset, a stop, a hand on it), and an arm whose pose the observation does not carry.

### Pacing

A control loop plays a chunk one row per tick and asks for the next chunk either while rows are still running (async: the arms keep moving through the inference) or once they have all run (sync: the arms hold through it). The right choice depends on how long `get_action` takes, which the client measures anyway, so the client makes it. After each call, `client.mode` is `"sync"` if the median of the last five calls' round trips is longer than `sync_above_ms` (700 ms by default) and `"async"` otherwise, so one call that waited behind another robot on a shared serve does not switch it. The loop asks `client.refill_due(executed, remaining)` once per row, with no request in flight: `executed` is how many rows of the current chunk it has run since installing it, `remaining` how many are still queued.

In async the chunk comes back whole. `refill_due` turns true once `executed` plus the round trip in rows (the shortest of the last five round trips times `action_hz`) reaches `execution_horizon` (30 by default), so the next chunk lands as the arms finish 30 rows; the rows past 30 are runway for a late reply. It is also true once the rows left are worth less than that round trip plus a quarter second. The shortest is the round trip no queue at the serve added to: a reply that waited behind another robot's inference moves the next request later, away from that robot, where sizing from a slow reply would send every following request early, onto the other robot again.

In sync the chunk also comes back whole, and `refill_due` is true only once `remaining` is 0: the arms run every row of it (the action horizon, 48 for the Dream-1 base model) and hold while the next one is computed, since dropping rows of the plan would only pay for another round trip sooner. The first sync chunk after a switch was asked for in async while the arms moved, so a loop that skips the rows run during the round trip starts it a few rows in (about 15 at 750 ms) and runs the rest.

The mode is decided again on every call, so a link that slows down is paced synchronously once most of the last five calls are slow, and one that recovers is back to async once most are fast. A loop that never calls `refill_due` is unaffected: the mode changes when the loop asks, never what comes back. At 20 Hz a 700 ms round trip is 14 rows, and 14 + 30 = 44 rows is the most a 48-row chunk can cover asynchronously with room to spare, which is where the default threshold comes from.

Connection keys are validated for type and range at `policy()`, before any connection exists; a wrong type raises `DreamConfigError` naming the key. `checkpoint` is required: without it there is nothing to route to, and `policy()` raises `DreamConfigError`. **Any other key is a `DreamConfigError` at `policy()`.** DreamAPI's `/v1/get_action` request has no config field, so a key the client cannot deliver must fail loudly rather than vanish; the message names the offending keys. The API key is never a config field: it is the credentials' key, or the `api_key` argument of `DreamClient` when one is passed, and travels as `Authorization: Bearer` on every request.

## Configuration

The config is the dict `policy()` takes, and the user owns it. The client consumes two **connection keys**, three **pacing keys** and two **window keys**, takes `kinematics` and `ik` (*Seam smoothing*, *EEF checkpoints and client IK*), sends `robot` as the observation key `embodiment` (*EEF checkpoints and client IK*), reads `checkpoint` and `embodiment`, and accepts nothing else.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `timeout_ms` | `int` | `15000` | Per-request connect, send and receive timeout |
| `strict` | `bool` | `True` | Validate observations and actions against `config["embodiment"]` |
| `execution_horizon` | `int` | `30` | Rows of each chunk the loop executes before the next takes over ("Pacing"); never sent |
| `action_hz` | `float` | `20` | The loop's row rate, rows per second; turns a round trip into rows ("Pacing"); never sent |
| `sync_above_ms` | `float` | `700` | Round trip past which the client paces synchronously ("Pacing"); never sent |
| `obs_window` | `int` | `9` | Frames the client keeps per window key, at least 1 ("Running a model"); never sent |
| `obs_fps` | `float` | `5` | Frames kept per second of `set_observation`'s timestamps ("Running a model"); never sent |
| `kinematics` | `dict` | none | Per chunk key, the arm's `pose` and `jacobian`, so the seam blend runs on the gripper's path ("Seam smoothing"); never sent |
| `ik` | `dict` | none | `{"arm": ..., "head": ...}` (optionally `keep_out`, `reach`, `region`, `solve`) for a model fine-tuned with `use_eef` (*EEF checkpoints and client IK*); never sent |
| `robot` | `str` | none | `"yam"` (these joints are the model's own: a YAM, or the robot a fine-tune was trained on; the state is read as sent) or `"other_embodiment"` (an arm that is not the one the model's state is from: a Dream-1 base model reads none of the state); sent on every request as the observation key `embodiment`. Left out, `ik` on an arm that is not a YAM with a model trained on YAM arms sends `"other_embodiment"`, and anything else sends nothing (*EEF checkpoints and client IK*) |
| `checkpoint` | `str` | required | The checkpoint id DreamAPI routes to; sent on every request as the `checkpoint` field |
| `embodiment` | `dict` | none | Shape spec for the strict-mode checks, or the embodiment config of a fine-tuned body (*Running a fine-tuned model*; `DreamClient.policy()` loads it for a fine-tune of a custom body); read locally, never sent |

Where the model is is not a config key: `policy()` refuses `host`, `port` and `scheme` with `DreamConfigError`. The gateway is the one your credentials file names, read once when the client is made, and nothing in your code changes it.

## Credentials

`DreamClient()` reads your key from its `api_key` argument, else `DREAM_API_KEY` in the environment, else the file `~/.dreamlabs/credentials`. The same file names where this platform's gateway, site and data service are: the platform's Fine-tuning and Setup API pages print it filled in for their own site. Those three addresses are set by Dream Labs and are read-only: no constructor argument, config key or method sets them, so copy them as they are printed. `repr(client)` names nothing but the class, never the key.

The file is read as a shell would read it: `NAME=value` lines, `export` and single or double quotes allowed, `#` comments, blank lines and other names ignored, and a later assignment for the same name winning. A missing file is production with no key. A file that cannot be read raises `DreamConfigError` when `DreamClient()` reads it, naming the file. Keep the file owner-only (`chmod 600`): it holds the key.

With `strict` on, every observation is checked against the non-`action` groups of `embodiment`
(keys present, no undeclared grouped keys, exact shape, exact dtype of arrays) before it is sent,
and every action against the `action` group before it is returned. Observation keys without a
group (`prompt`) are metadata the serve reads and are not checked; extra action keys from the
serve are tolerated; a list value is shaped with `np.shape` and its dtype is not checked,
because JSON carries no dtype. Values are never inspected. No `embodiment` in the config means no
validation and no error. Leave `strict` on.

## Fine-tuning as a Service

Fine-tuning as a Service is the whole path from a robot's recordings to a checkpoint you can call: describe the body, upload a dataset, fine-tune Dream-1, run the result. `dream.DreamClient` is the one object for it, and the platform's Fine-tuning page is this section as a tutorial, with a notebook to download.

`DreamClient()` binds your key and the three hosts from the credentials (*Credentials*), overridable per client in the constructor. The control-plane calls are stateless HTTP and hold nothing open; `policy()` opens an inference session on the gateway, which `close()` releases (the client is also a context manager). The library file from the platform site carries what the dataset checks and the upload need.

| Method | What it does |
| --- | --- |
| `add_embodiment(name, config)` | Parses the config, then records the robot body; answers an `Embodiment` named `<username>/<name>` |
| `get_embodiments()` | Your own embodiments, newest first |
| `validate_dataset(path, config)` | Runs the dataset checks locally against a config and answers a `Report`; nothing is sent, no key needed |
| `upload_dataset(path, name, embodiment)` | Validates, zips, uploads, completes; answers the `ready_to_process` `Dataset`, whose `id` to keep |
| `get_dataset_status(dataset_id)` | The dataset's status as the service holds it now, by the `ds_...` id `upload_dataset` answered |
| `get_datasets()` | Your datasets, newest first, each with its status as the service holds it now; a lost `ds_...` id is found here by the dataset's `name` |
| `retry_dataset(dataset_id)` | Has a dataset whose processing `failed` processed again from the zip already uploaded, and answers it `ready_to_process` (as another call left it, when the retry raced one); `DreamServerError` `conflict` when it is not failed or its upload never finished (upload it again under the same name) |
| `delete_dataset(dataset_id, *, confirm=True)` | Asks `Delete dataset <id>? [y/N]`, then deletes it (refused with `conflict` while a fine-tuning job on it is queued or running); answers it `deleted` (None if you said no) |
| `finetune(dataset, suffix, *, learning_rate=None, use_eef=False, lr_schedule=None)` | Queues a fine-tuning job on a `processed` dataset; answers the `queued` job dict. `lr_schedule="cosine"` decays the learning rate to 0 by the last step (`"constant"`, the default, holds it). `use_eef=True` trains the recorded end-effector pose beside the joints and needs the six EEF parts in the body (checked before sending); the pose is each arm's tool centre point in metres, `rot6` the first two columns of the last link's rotation matrix, both in that arm's own base frame, and the grip an opening from 0 closed to 1 open (the `finetune` docstring has the full convention) |
| `get_finetuning_status(job_id)` | The job as the service answers it now |
| `get_models()` | The checkpoints your account may run: the public ones and your own fine-tunes |
| `policy(config)` | Chooses the checkpoint and starts an empty frame window; answers None, and the first `get_action` opens the session. Called again, it closes the previous policy, ending its session |
| `set_observation(frames, observation, *, timestamp=None)` | Hands the client one camera tick, one frame per camera and the latest state; call it on every tick (*Running a model*) |
| `get_action(*, commanded=None)` | Sends the latest observation with the frames the client kept and returns the action chunk |
| `reset_episode()` | Ends the session and empties the frame window; call it between episodes |

Each method forwards to a submodule function that takes the same key explicitly, so `dream.datasets`, `dream.fine_tuning` and `dream.models` are usable on their own; the tables below name each one. A key left out is the credentials' `DREAM_API_KEY`; where the service is comes from the credentials, as for `DreamClient`.

### Embodiments

An embodiment is `<username>/<name>` plus one config you write: `{"state": {part: [start, end]}, "action": {part: [start, end]}, "video": {role: video_key}}`. `state` and `action` name the parts of the `observation.state` and `action` vectors with their `[start, end)` spans, which must tile `[0, width)` with no gap or overlap (128 wide at most; a fine-tune reads a state of at most 64); `video` maps each camera role Dream trains on to the LeRobot video key the dataset records it under (`observation.images.overhead`). Part names and roles are lowercase letters, digits and underscores, starting with a letter. One key is optional. `head_camera` names the role in `video` of the robot's fixed head camera, the view over the workspace; every recorder calls it something different (`cam_high`, `cam_left_high`, `front`), so the config says which it is, and left out, a role called `overhead` is taken as the head camera. A body with neither is refused at registration, since every dataset is labelled from that camera: the service answers `add_embodiment` with a `DreamServerError`, code `invalid_request`. Nothing else goes in it. The client parses the config before sending it, and the service refuses anything the parser refuses with the same line, so `add_embodiment` raises `DreamConfigError` on a bad config before anything is sent.

```python
import dream

client = dream.DreamClient()

config = {
    "state": {"right_arm": [0, 6], "right_gripper": [6, 7], "left_arm": [7, 13], "left_gripper": [13, 14]},
    "action": {"right_arm": [0, 6], "right_gripper": [6, 7], "left_arm": [7, 13], "left_gripper": [13, 14]},
    "video": {
        "overhead": "observation.images.overhead",
        "wrist_l": "observation.images.wrist_l",
        "wrist_r": "observation.images.wrist_r",
    },
    "head_camera": "overhead",
}
embodiment = client.add_embodiment("makersmod-yam", config)  # Embodiment(name="<your-username>/makersmod-yam", ...)
```

Dream-1 trains any joint-space body from this config, and the model it makes runs like any other. Every role in `video` is a view the model trains on: a job on a body with two or three cameras takes about 4 to 5.5 hours, and a fourth camera takes longer.

`Embodiment(name, embodiment_config, created_at, id)` is the record `add_embodiment` and `get_embodiments` answer; `name` is `<username>/<name>`, what a dataset names it by.

| `dream.datasets` function | What it does |
| --- | --- |
| `add_embodiment(name, config, *, api_key=None)` | Parses the config, then records the body |
| `list_embodiments(*, api_key=None)` | Your own embodiments, newest first |

### Datasets

A dataset directory holds one LeRobot v2.1 dataset per subdirectory, or is itself one (it has `meta/info.json`); hidden files (a `.DS_Store`, a `.git/`) are neither checked nor sent. `validate_dataset` runs the dataset checks locally and with no key, so you fix problems before uploading: `meta/info.json` must have `codebase_version` (`"v2.1"`), `fps`, `chunks_size`, `total_episodes`, `total_frames`, `data_path`, `video_path` and `features`, with `action` and `observation.state` in `features` (the check names each one missing; the Fine-tuning page of the customer docs prints a complete example that passes), the parquet widths must equal the spans' width, the video directories under `videos/` must be exactly the config's video keys, every video must sit at LeRobot's standard `videos/chunk-<NNN>/<video key>/<episode>.mp4` with `meta/info.json`'s `video_path` naming that layout and one video per episode and key, every directory must record at the same fps with the same feature names, there must be at least one episode, every `task_index` in the rows must be defined in `meta/tasks.jsonl`, a `meta/modality.json` you ship must be a modality file, though its parts need not be the config's (training writes its own from the config), and `meta/info.json`'s state and action names, when given, must be one per value. The task text the model trains on comes from that `tasks.jsonl`; the service adds the per-frame task field itself, so a valid `tasks.jsonl` is all you need to provide.

```python
report = client.validate_dataset("~/data/towels", config)  # local, no key needed
assert report.ok, report.error
dataset = client.upload_dataset("~/data/towels", "towels-v1", embodiment.name)
print(dataset.id, dataset.status, dataset.episodes, dataset.size_bytes)  # keep the id; "ready_to_process"

while client.get_dataset_status(dataset.id).status not in ("processed", "failed"):
    time.sleep(30)
```

`upload_dataset` writes the directory tree as one zip in the system's temporary directory (`tempfile.gettempdir()`, so `TMPDIR` when it is set; stored uncompressed: mp4 and parquet are compressed already, so the zip is about the dataset's size), registers the dataset with the zip's size and the episode count, uploads the zip (in parts of 64 MiB, eight at once, from 64 MiB up), then completes; the service checks the upload arrived whole, at the declared size. Whatever fails after the registration (the transfer, the completion) is reported to the service, which marks the dataset `failed` with that line, and raised as `DreamUploadError`. The zip is removed before the call returns or raises, and nothing is kept between runs. Before writing the zip it checks that the temporary directory has room for it; if not, it raises `DreamConfigError` with the space needed and free and nothing is registered: start Python with `TMPDIR` set to a directory on a disk with room, or set `tempfile.tempdir` in a running one. A name is reused, not burned: running the same upload again after a failure, or after a run the process did not survive (Ctrl+C, a crash), reuses the dataset (same id, status back to `uploading`) and sends the zip again; only a name whose dataset is `ready_to_process`, `processing` or `processed` is taken for good (a `DreamServerError` with `code == "conflict"`).

`Dataset(id, name, embodiment, status, size_bytes, episodes, error, created_at)` and `Report(ok, error, sessions, episodes, frames, files, bytes, state_dim, action_dim, fps, video_keys)` are the records the calls answer; a dataset's `name` and `embodiment` are both `<username>/<name>`, and its `status` is `uploading` until the service has the whole upload, then `ready_to_process`, then `processing` while the service prepares it and `processed` once it has, or `failed` with the one line in `error` when the upload reported a failure or the dataset could not be prepared. A dataset that could not be prepared says which of three things happened: something in your files (the line names it; fix it and upload again under the same name), a dataset too large to process (split it into smaller ones), or an error on our side, not in your data (or, for a dataset that failed before the service told the two apart, only that processing failed and to retry it), after which `retry_dataset(dataset.id)` has it processed again from the zip already uploaded, with nothing sent again; nothing is retried on its own. Only a `processed` dataset can be fine-tuned on. Keep the `id` that `upload_dataset` answers: `get_dataset_status` takes it, and `get_datasets()` finds a lost one by the dataset's `name`.

| `dream.datasets` function | What it does |
| --- | --- |
| `validate_dataset(path, config)` | Runs the dataset checks locally against a config and answers a `Report`; nothing is sent |
| `upload_dataset(path, name, embodiment, *, api_key=None)` | Fetches the embodiment's config, validates, zips, uploads, completes; answers the `ready_to_process` `Dataset` |
| `get_dataset(dataset_id, *, api_key=None)` | The dataset's status as the service holds it now |
| `list_datasets(*, api_key=None)` | Your datasets, newest first, each with its status as the service holds it now |
| `retry_dataset(dataset_id, *, api_key=None)` | Has a dataset whose processing `failed` processed again from the zip already uploaded; answers it `ready_to_process`, or, when the retry raced another call on the same dataset (a second retry, an upload under its name), as that call left it |
| `delete_dataset(dataset_id, *, confirm=True, api_key=None)` | Deletes the dataset after a `[y/N]` question; answers it `deleted` |

### Fine-tuning

`finetune` queues a job on one of your `processed` datasets: Dream-1 trained on it, published as a checkpoint only your account can call, `dream-1-ft:<uid8>-<suffix>` (`uid8` the first eight hex characters of the account's id). The training knobs (`learning_rate`, `lr_schedule`) are optional; a knob left out takes the service's recipe default, and the service refuses a knob outside its bounds. Poll `get_finetuning_status` and wait for `fine_tuned_model` to be set, which happens in the same pass that marks the job `succeeded` (a `failed` or `cancelled` job never gets one), so a poll waits for that field rather than for the status alone.

```python
job = client.finetune(dataset.name, "towels")
while job["fine_tuned_model"] is None and job["status"] in ("queued", "running", "succeeded"):
    time.sleep(60)
    job = client.get_finetuning_status(job["id"])

client.policy({"checkpoint": job["fine_tuned_model"]})  # loads the body's config to check each observation
client.set_observation(frames, observation)  # every camera tick (*Running a fine-tuned model*)
action = client.get_action()  # pool_warming while the model starts; retry after retry_after_s
```

`lr_schedule` is `"constant"` (the default) or `"cosine"`. Both warm up first; cosine then decays the learning rate to 0 by the last step.

- Cosine decay works best for specialization (a fine-tune on a single task).
- You may not get high reliability on specialist tasks yet.

How long a job takes: in our tests a job on a two- or three-camera body took about 4 to 5.5 hours from submit to model; a fourth camera takes longer. While a job is `running` its record shows no estimate.

The first `get_action` to a model that is not running (a fresh fine-tune, or any model after it has been idle a while) is refused `DA-2000` `pool_warming` (a `DreamServerError` with `retryable` and `retry_after_s`) while the model starts. In our tests a start took 25 to 50 minutes, most of it waiting for hardware, and the refusal carries no progress or estimate: call again every `retry_after_s` and allow an hour. Once it is running, a call takes about 250 to 450 ms through the gateway. A model that cannot be started is refused `DA-6000` `cannot_scale` with the reason, so a broken fine-tune is an answer, not a wait. A finished fine-tune shows up in `get_models` as one of your own (`owner == "mine"`), `servable` once it can be run, which does not mean it is running now; `Model(id, owner, base_model, embodiment, servable, created_at, cameras, embodiment_config)` is the record, and `id` is what `policy` names as `checkpoint`; `cameras` are the camera roles the model reads (`video.<role>`), None when the platform does not know them; `embodiment_config` is the config of the custom body a fine-tune was trained on, whichever account holds that body (None for a builtin body), and `policy()` checks calls against it when the config names no `embodiment`. A model's `embodiment` is the `id` of the embodiment it was trained on (`emb_...`), not its name, so it matches one of `get_embodiments()` when your account holds that body; the job record names the embodiment and the dataset by name.

A job is `{id, dataset, embodiment, base_model, suffix, fine_tuned_model, status, hyperparameters, queue_position, created_at, started_at, finished_at, progress, error}`: `dataset` and `embodiment` are `<username>/<name>`, `base_model` is the public id of the checkpoint it starts from, `fine_tuned_model` is the checkpoint id once the job is `succeeded`, `hyperparameters` is the training settings as the service filled them, `queue_position` is 1-based among `queued` jobs waiting for a training slot (null otherwise), `status` is `queued` (waiting for a slot, or for a GPU once it has one), `running`, `succeeded`, `failed` or `cancelled`, `started_at` is when training started on a GPU, `progress`, where the service reports it, carries `phase`, `loss` and `updated_at` (`phase` one of `waiting_for_gpu`, `preparing`, `training`, `saving`; `loss` the action loss over your own data at `step`), and `error`, when `failed`, is one line: what to fix in your dataset when the cause was in it, else that the training did not finish.

| `dream.fine_tuning` / `dream.models` function | What it does |
| --- | --- |
| `fine_tuning.create(dataset, suffix, *, learning_rate=None, use_eef=False, lr_schedule=None, api_key=None)` | Queues a job on the `processed` dataset `<username>/<name>` under `suffix` (`[a-z0-9-]{1,40}`) |
| `fine_tuning.list(*, api_key=None)` | Your jobs, newest first |
| `fine_tuning.get(job_id, *, api_key=None)` | One job as the service answers it now |
| `fine_tuning.cancel(job_id, *, api_key=None)` | Refused with `conflict`: a job runs to its end once queued |
| `fine_tuning.training_data(model, *, api_key=None)` | Where one of your fine-tunes' training data is: `{url, video}`, a download URL of its `dataset.zip` for an hour and its camera role to video key |
| `fine_tuning.queue(*, api_key=None)` | The line as a job queued now would see it: `{jobs_ahead_of_new, running, slots_free, slots_total, updated_at}` |
| `models.list_models(*, api_key=None)` | The checkpoints your account may run (on the platform, not the data service) |

### Running a fine-tuned model

A fine-tuned model reads the keys of the embodiment config it was trained on, not the YAM station keys of *Observation and action keys*; an observation missing one of its keys is refused with an error that names the missing keys, and one that mixes the two key sets is refused too.

| Key | One per | Value |
| --- | --- | --- |
| `state.<part>` | part of the config's `state` | In `set_observation`'s `observation`: a flat list of floats as wide as the part's span, in the dataset's units |
| `video.<role>` | role of the config's `video` | In `set_observation`'s `frames`: that tick's one frame, a base64 JPEG (quality about 85) with no `data:` prefix. The client keeps the camera's last 9, one every 0.2 s, and sends them oldest first |
| `prompt` | observation | In `observation`: the task, in words |

- At the start of an episode the client has kept fewer than 9 frames and sends the ones it has: a shorter window is accepted, its frames are spread evenly over the 9 places, and a single frame is a still image (in training, the earliest places of an episode's first windows held its first frame instead).
- Any frame size is accepted: every frame is resized to 320x176 (width x height) by stretching, never by cropping or padding, the way the recording's frames were resized for training. Send the camera's whole view, framed as in the recording; frames already at 320x176 upload fastest.
- The reply's `per_key` holds one `action.<part>` per part of the config's `action`: 48 rows, one per step ahead at the dataset's frame rate, shape `[48, width]`, or `[48]` for a one-value part such as a gripper, in the dataset's units. Seam smoothing rewrites only `q_r_chunk` and `q_l_chunk`, which only a body laid out like the YAM arms also gets, so `per_key` comes back as the model sent it.
- A robot registered with `parts` trains on the slots its plan names, not its own parts; the client converts both ways, so you still send `state.<part>` for the robot's own parts and get `action.<part>` back for each action part the model trains, in the part's units: a rotation in its own format, a gripper between its `closed` and `open`, a part marked `"delta": true` as the change each row makes (the first from the `state.<part>` you sent, each later one from the row before). A pose part recorded but not trained is not in the reply. The conversion runs whenever the client has the config (loaded with `strict` on, or passed as `embodiment`); a client with `strict` off and no `embodiment` speaks the slot names, as `POST /v1/get_action` does. The platform lists a robot's `parts` only to the account that registered it, and only to a client that converts them (3.0.0 on, whose platform calls carry `User-Agent: dream-client/<version>`); anyone else's `policy()` on such a fine-tune (a shared or public one) with `strict` on is a `DreamConfigError` naming the checkpoint and its robot (`Model.parts_withheld`), so pass that robot's config as `embodiment`.
- One session per episode: call `client.reset_episode()` between episodes. It ends the session and empties the frame window, and the next `get_action` opens a fresh session. Calling `policy()` again does the same and reads the config anew.
- Calls faster than the rate limit are refused `DA-2000` `rate_limited`, retryable: each session may make 3 calls in quick succession, then one every 0.7 s, and `retry_after_s` is 0.7. `pool_warming` and `rate_limited` both carry the label `NO_CAPACITY` in the message; `reason` tells them apart.

`DreamClient.policy()` on a fine-tune of a custom body (yours, or a shared or public one whose body another account registered) loads the embodiment config the model was trained on (the model's `embodiment_config` from `get_models()`, once per checkpoint, kept for the session), and with `strict` on, the default, the client checks every observation against it before anything is sent: each part of `state` present and as wide as its span, each role of `video` among the frame keys `set_observation` was given, and no dotted key the config does not name (a YAM station key such as `image_window.top` included). A refused observation raises `DreamObservationError` naming the missing, extra or mis-shaped key, and nothing is sent. Every reply's `per_key` is checked the same way before it is returned (each part of `action` present, as rows of its width, every part with the same number of rows), or `DreamActionError`. Pass `embodiment` yourself (the dict `add_embodiment` took) to check against another config, or `strict: False` with no `embodiment` to skip both the checks and the conversion: calls go unchecked, in the slot names the model trained on. A model of a builtin body (a Dream-1 base model) has no config. A lookup that fails (the platform unreachable or refusing, as during a deploy) raises `DreamProtocolError` at `policy`, and nothing is sent: call `policy` again. A `checkpoint` `get_models()` does not name (a typo, or a model this account may not run) is a `DreamConfigError` at `policy`, and nothing is sent. A config the parser refuses is a `DreamConfigError` at `policy`, carrying the line `add_embodiment` would refuse it with.

```python
client.policy({"checkpoint": job["fine_tuned_model"]})  # loads the config registered above

frames = {  # one frame per camera per tick: a base64 JPEG str with no data: prefix
    "video.overhead": overhead_camera.jpeg(),
    "video.wrist_l": left_wrist_camera.jpeg(),
    "video.wrist_r": right_wrist_camera.jpeg(),
}
observation = {
    "state.right_arm": right_arm.joints(),  # 6 values: the [0, 6) span
    "state.right_gripper": [right_arm.gripper()],  # 1 value
    "state.left_arm": left_arm.joints(),
    "state.left_gripper": [left_arm.gripper()],
    "prompt": "fold the towel",
}
client.set_observation(frames, observation)  # every camera tick; the client keeps the last 9 frames per camera
action = client.get_action()  # on its own thread while set_observation() keeps going
rows = action["per_key"]["action.right_arm"]  # [48, 6]: one row of joint targets per step ahead
```

### EEF checkpoints and client IK

A model fine-tuned with `use_eef` reads each arm's tool pose and answers with tool poses, beside its joint answer. The model computes no poses for you and solves none back to joints, so the client does both when you set the config key `ik`:

```python
client.policy({"checkpoint": job["fine_tuned_model"], "ik": {"arm": "metal", "head": "eef"}})
```

- `arm` is the robot the model was fine-tuned on: `"yam"` (the YAM itself, below), `"metal"` (MakerMods Metal), `"makers"` (Makers Arm), or one of the FaaS beta robots, `"ai_worker_faas"` (ROBOTIS AI Worker), `"trossen_wb_faas"` (Trossen WidowX AI), `"mantis_faas"` (Blue Worm Mantis), `"robo_t_faas"` (ROBO-T1), `"openarm2_faas"` (Enactic OpenArm 2), `"bracketbot_faas"` (Bracketbot), `"so101_faas"` (SO-101), `"galaxea_faas"` (Galaxea R1 Lite), `"seeed_faas"` (Seeed reBot B601) and `"unitree_r1_faas"` (Unitree R1).
- For an arm of your own, run by a model trained on YAM arms (the Dream-1 base models), `arm` is instead the path of an arm file, a JSON object with these keys: `urdf` (your arm's URDF, relative to the arm file), `base` (the link the hand's pose is measured from, read as YAM's base: x forward, y to the arm's left, z up), `flange` (the link the gripper hangs off), `tcp` (`[x, y, z]`, metres in the flange's frame: the point between the closed fingertips; the user measures and confirms it, with `dreamcli tcp` or by hand, and a coding agent never picks or estimates it), `approach` and `jaw` (`[x, y, z]` in the flange's frame: the way the fingers point, and the way they close) and `rest` (the arm's joints in a working pose, radians, on both sides). The client cuts the URDF to the chain from `base` to `flange`, so the gripper's own joints drop out; your loop speaks that chain's moving joints in order from `base`, in radians. Each joint on the chain is fixed, or revolute or prismatic with limits, and the solve keeps each under its URDF velocity, at most 3 rad/s (YAM's). A malformed file is a `DreamConfigError` at `policy` naming what to fix (`dream.arm_file`).
- Your loop speaks each arm's joint angles in radians (a Bracketbot's lift in metres), in the order and directions of dream's arm model, as the robot recorded them for its fine-tune (an SO-101 is the one exception, below), and on a beta robot the gripper as `state.<side>_grip`, `[1]`, from 0 closed to 1 open (the client maps it linearly onto the robot's recorded closed and open values, and back). Send `state.right_arm` and `state.left_arm` (and the grips); the client builds everything else the model reads from them.
- On the way out, every `get_action` adds `state.right_eef_xyz`, `state.right_eef_rot6`, `state.left_eef_xyz` and `state.left_eef_rot6`, computed from `state.right_arm` and `state.left_arm`. A beta robot's model reads its joints and its gripper in the units the robot recorded them in, so the client also sends `state.<side>_arm` in those units and the gripper drive under the key your embodiment config names for it (`<side>_gripper`; `<side>_hand` on a Unitree R1; on a Trossen the drive is the last entry of `<side>_arm`). It finds that key in the embodiment config (`policy()` loads it for your own fine-tunes with `strict` on), so on a beta robot `ik` without one is a `DreamConfigError`. Leave the pose keys and the drive out: an observation that carries one is refused with `DreamObservationError`.
- On the way back, with `head: "eef"`, the model's tool poses (`action.<side>_eef_xyz` and `action.<side>_eef_rot6`) are solved to joints, starting from the joints the arm is commanded to when the new chunk takes over and kept under the arm's joint speed limits. The solve is dream's own IK, with its pull toward rest along joint directions the hand barely feels off on every arm, so a folded Metal arm may turn its base and wrist together for a small sideways move of the hand. They come back as `q_r_chunk` and `q_l_chunk`, `[48, joints + 1]`: the arm's joints in radians (six on Metal and Makers, five on an SO-101, seven on the seven-joint arms), then `action.<side>_grip` (0 closed, 1 open). Seam smoothing then joins them like any other chunk, anchored on `state.right_arm` and `state.left_arm`, or better on the joints your loop last commanded: pass them as `get_action(commanded={"q_r_chunk": ..., "q_l_chunk": ...})` (the same joints, or a whole written row), since a real arm trails its command by a few rows. A reply that lacks a tool pose or a grip, or holds one of the wrong shape, is `DreamActionError`.
- `keep_out`, optional, keeps each arm off what stands around it, a camera mount or a pole: `"keep_out": {"left": [...], "right": [...]}`, either side or both, each a list (empty for none) of `{"kind": "box", "center": [x, y, z], "size": [x, y, z]}` (axis-aligned, full extent) or `{"kind": "capsule", "center": [x, y, z], "radius": r, "height": h}` (upright), in metres in that arm's base frame. The solve keeps 4 cm spheres along the arm 2 cm off each, so a tool pose inside one is reached only as near as that allows. A side listed replaces the obstacles of dream's model of the arm, a side not listed keeps them; the Metal and Makers models have none, since a camera mount is the station's, not the arm's, and each station lists its own here: without `keep_out` the arm is kept off nothing. A malformed entry is a `DreamConfigError` at `policy`. A robots station reads them from each arm section's `keep_out`.
- `reach`, optional, keeps each hand's targets inside an area seen from above: `"reach": {"left": [[x, y], ...], "right": [...]}`, either side or both, each at least one point, in metres in that arm's base frame. A hand target farther than 4 cm (`dream.client_ik.REACH_RADIUS_M`) from every point moves straight toward the nearest one until it is 4 cm away, before the solve; its height and rotation are kept, and a target inside is untouched. The solve can still miss a moved target, since it keeps the target's rotation, and a path across a gap or notch in the area steps through it, so the client also takes the hand at each row the loop will run, after the seam join (which carries the old motion's speed on into the chunk), by forward kinematics: from the first row more than 1 cm past the edge (`REACH_SLACK_M`) and more than 1 mm farther out than the chunk has come, the arm holds the row before for the rest of the chunk; the gripper still moves. A hand that starts outside the area may still move back in. So the hand stays out of a pole outside the area at any height, but the elbow is not guarded, only head `eef` is kept in (head `joint` solves nothing), and a gap narrower than about 10 cm can still be crossed: give one solid area that stops short of the mount. Without `reach` nothing changes. A side the arm lacks, an empty list, or a point that is not two finite numbers is a `DreamConfigError` at `policy`. A robots station reads it from each arm section's `reach`, which `dreamcli setup` paints. It costs about 0.5 ms per arm per chunk for the forward kinematics, plus about 0.2 ms per arm for an area of 500 points (1 ms for 3,000).
- `solve`, optional, picks how the tool poses become joints: `"traj"` (the default, as the robots stations run it) solves all the chunk's rows in one problem with costs on joint velocity, acceleration and jerk, continuing the motion the arm is on (the previous chunk's two rows before the takeover row, or the measured joints held for an episode's first chunk). The seam join, the arm's speed caps and its margined limits apply either way. In offline replays of recorded teleop on the Metal, Makers, Seeed reBot, SO-101 and YAM arms, with pose errors like a model's, `traj` cut joint jerk about 75% and direction reversals about 80%, left the hand error unchanged and took about 10 ms more per chunk (about 20 ms against 11 ms on a laptop CPU). `"tick"` solves the chunk one row at a time from the row before. `traj` compiles on a thread when `policy` opens, and the first chunks are solved with `tick` until it is ready (a few seconds on a normal machine). Anything else is a `DreamConfigError` at `policy`.
- `region`, optional, keeps each arm inside a box: `"region": {"right": {"lo": [x, y, z], "hi": [x, y, z]}, "left": {...}}`, either side or both, metres in that arm's base frame, any value `null` for an open face, either key left out for all open. It bounds the arm's centreline, every link and the grasp point between the fingertips, so a `lo` z at the table lets the hand reach the table while no link goes through it; a `lo` x keeps the arm from reaching behind its base, and a `hi` y (right arm) or `lo` y (left arm) keeps it out of the other arm's half. The hand targets are moved 2 cm inside before the solve, the solve keeps the arm 2 cm in, and after the seam join the client holds the arm at the row before the first that leaves the box by more than 1 cm (`dream.client_ik.SERVE_REGION_SLACK_M`); an arm already outside may move back in, never further out. On recorded YAM teleop a box at the teleop's own bounds changed nothing; with the model's targets driven into the table, behind the base or across a midline, 12 to 36% of rows left the box without it and none with it. A side the arm lacks, an unknown face, a face that is not three numbers or nulls, or a `lo` not below its `hi` is a `DreamConfigError` at `policy`.
- With `head: "joint"`, nothing is solved: a Metal or Makers model's joint chunks are kept, and a beta robot's joint answer (`action.<side>_arm`) comes back the same way as with `eef`, in radians, then the grip. Head `joint` is refused for a model trained on YAM arms (below) on any arm but a YAM: its joint answer is another robot's joints, not run on yours, for safety.
- Drive the arm from `q_r_chunk` and `q_l_chunk` only. Whenever the client builds them (head `eef`, and a beta robot's head `joint`), the reply's `q_r` and `q_l` are dropped: they are the model's own first row of joints (YAM's, for a model trained on YAM arms), not your arm's. `per_key` stays the model's answer as it came, in the model's convention.
- What the model reads follows from the body it was trained on, the `embodiment` of its record in `get_models()`, never from its id, so `policy` asks the platform for that record once. A model trained on your robot reads it as above. A model trained on YAM arms (the Dream-1 base models, `embodiment` `yam`, and a YAM owner's fine-tune) is sent the slots it reads whatever arm runs it: YAM's six joints, as zeros on any arm but a YAM, `state.<side>_gripper` as your `state.<side>_grip` (0 closed, 1 open; send both), each hand's pose from your arm's kinematics, and your cameras as `video.overhead`, `video.wrist_l` and `video.wrist_r`: the wrists by name and your one other camera as the overhead (so it needs your `embodiment` config). Cameras that cannot fill those three are a `DreamConfigError` at `policy` that names what is missing and what is extra. On any arm but a YAM the client also sends `embodiment` `"other_embodiment"` (`robot`, below), and a Dream-1 base model then reads none of the state: it is withheld, which means the model is given zeros in its own scaled units in place of every state value (the joints, each hand's pose and the grip), an input it was trained on half of the time, and it acts on the cameras and the prompt. Each hand's pose is still the anchor of the answer: the model answers each hand's motion from where the hand is, applied to the pose the request carries, so `per_key`'s tool poses are absolute, in the arm's base frame. It answers no joints (`per_key["action.<side>_arm"]` is null), and with head `eef` the client solves `q_r_chunk` and `q_l_chunk` from the tool poses as above. A fine-tune whose camera roles the platform knows (`cameras` of its record) gets your cameras the same way: the wrists by name and your one other camera in its one scene slot, so a camera you call `scene` reaches a fine-tune trained with `head`.
- `robot`, optional, a config key of its own beside `ik` (not inside it), says whose state a request carries: `"yam"`, these joints are the model's own (a YAM, or the robot a fine-tune was trained on), so the state is read as sent; or `"other_embodiment"`, an arm that is not the one the model's state is from, so a Dream-1 base model reads none of it, as above. The client sends it on every request as the observation key `embodiment`, which is not the config key `embodiment` (the embodiment config, never sent). Left out, `ik` on an arm that is not a YAM with a model trained on YAM arms sends `"other_embodiment"`, and anything else sends nothing. A `robot` you set always wins, which is how a loop on its own IK (no `ik`) on an arm that is not a YAM asks for the same: `"robot": "other_embodiment"`, six zeros as each `state.<side>_arm` (the slots must be present, and their values are withheld), the grip, and `state.<side>_eef_xyz` and `state.<side>_eef_rot6` from its own FK, the TCP's, absolute, in the arm's base frame, since the answer is anchored on them. The reply's `action.<side>_arm` is then null and no `q_r_chunk` or `q_l_chunk` is packed from it, so that loop solves `per_key`'s tool poses itself. Any other value is a `DreamConfigError` at `policy`, and an observation that carries `embodiment` itself is a `DreamObservationError`. `robot` changes what the Dream-1 base models read; a fine-tune reads the state as it was trained to, whatever `robot` says.
- The poses are in the frame the model was trained in: the client uses the same kinematics and arm models that computed the training poses (the tool centre point in metres, `rot6` as the first two columns of the rotation, each arm in its own base frame, the arm's own joint directions), so nothing needs converting.
- An SO-101 model reads the joints in LeRobot's -100 to 100 range of the calibration of the arm its data was recorded on, and the client converts to that range with that arm's numbers. Every SO-101 is calibrated by hand, so turn your own arm's readings into radians with your own calibration first (radians 0 is the pose the arm was calibrated in, LeRobot's middle pose, at 2 pi / 4096 rad per tick; calibrate in that pose), then pass them in `state.<side>_arm`: five joints, since the arm model's sixth is a fixed placeholder. The gripper's 0 to 100 is relative to the calibrated range too, so send the grip from your own arm's range.
- On a YAM (`arm` `"yam"`), a model trained on YAM arms that also reads hand poses (`dreamlabs/dream-1-beta`) runs with `head: "joint"`: your loop sends the YAM's own keys (`state.<side>_arm`, `state.<side>_gripper`, the three `video.*` windows), the client adds `state.<side>_eef_xyz`, `state.<side>_eef_rot6` and `state.<side>_grip` (copied from `state.<side>_gripper` unless you send it) from YAM's kinematics in the training frame, and the model's joint chunks are YAM's own, so they run as they come. A model that reads no poses needs no `ik`, and runs the same with it: the serve ignores the poses, so one config runs every YAM model. With `strict`, the YAM's own embodiment block (`state` and `action` `<side>_arm` and `<side>_gripper`) is enough: the poses and grips the client adds are read too, and an answer that carries `action.<side>_grip` in place of `action.<side>_gripper` (as such a model's does) is not refused. With `head: "eef"` the model's tool poses are solved to the YAM's joints, and a model with no EEF head runs its joint chunks, with one warning per session, so one config runs every YAM model on either head.
- `ik` needs Python 3.11 or newer; the library from the platform ships the solver. `policy` then takes a few seconds, preparing the solver (3 to 5 s the first time on a machine, less after that, since the prepared solver is cached in `~/.cache/dream_ik`), and each `get_action` adds about 7 ms on Metal and Makers (about 2 ms more with an obstacle in `keep_out`) and 8 to 18 ms on the beta robots (medians on one Xeon core).

### Service errors

A refusal by the data service or the platform is a `DreamServerError` whose `code` is the service's: `not_found` for an embodiment, dataset or job that is not yours; `conflict` for an embodiment or dataset name in use, a dataset that is not `processed`, a suffix in use, a job that is already finished, or an account at its cap on jobs `queued` or `running` at once; `invalid_request` for a bad embodiment name, suffix or knob; `validation` for a dataset the service's own completion check refused. A key the service does not accept is `DreamAuthError`, a service that cannot be reached `DreamProtocolError`, and a config the parser refuses, a dataset name outside `[a-z0-9-]{1,64}` (`upload_dataset` checks it before anything is read or sent), a path that is not a directory, a missing key, or the `datasets` extra not installed, a `DreamConfigError`. `DreamDatasetError` (the local checks refused the dataset; nothing was sent) and `DreamUploadError` (an upload that did not finish) live in `dream.errors`, not the `dream` namespace: `from dream.errors import DreamDatasetError, DreamUploadError`.
## Live View

`dream.LiveViewClient` shows your robot on the platform's Live View page: its cameras as video and its status, and the task a viewer types there comes back to your loop. The station (`dreamcli robot up`) does this on its own; this is for your own control loop.

```python
import dream


def on_episode(event: dream.EventChange) -> None:
    print("viewer pressed", event.value)  # start, or end (Stop, or a new task while running)


client = dream.DreamClient()
# The robot's id (live.robot_id in station.yaml); dream_client lets the page see the model's answers (see below).
with dream.LiveViewClient("rbt_0123456789abcdef", dream_client=client) as live:
    live.register_callback(on_episode)
    while running:
        live.stream(obs, "running")  # obs: the frames and state you hand set_observation(), in one mapping
        task = live.get_prompt()  # the task a viewer last sent, or None
```

- `stream(obs, status)` returns at once, and you can call it from any thread at any rate: the client does all of its work on its own thread, which reads your newest call 60 times a second, so your loop's pace never reaches the video. Each `video.<role>` key of `obs` (a base64 JPEG, or an RGB `uint8` array) is a camera on the page: `video.overhead` the main picture, `video.wrist_l` and `video.wrist_r` the left and right side panels, any other role under its own name; the other keys are ignored. Hand it a new array or JPEG for each new camera frame and the same one while the camera has delivered nothing new: a frame is encoded once. `status` is the state the page shows (`idle`, `running`, `error`); report `running` from a Start until the episode ends, since the page shows the task and Stop only while the robot is `running`. The first call opens the link and fixes the cameras (or pass `cameras=` to the constructor to name them up front).
- `stream()` takes optional keywords for what the page shows when your loop has it: `arms={"left": (q, gripper), "right": (q, gripper)}` draws the robot (the page's 3D view, sent only while a viewer looks), `error="..."` or `(message, code)` is the page's banner for why the arms are not moving, `prompt=` the task your code runs, `run=` your run number, `model=(checkpoint, action_space)` the model you run, and `steps=` the step count your next run takes (the page then offers one, which `get_steps()` returns after a Start).
- If the client's own thread cannot keep its rate while a run is on (the computer has no CPU to spare), it pauses the video for 30 s rather than compete with your loop (`stats()['live_paused']` is `"tick"`); a slow `get_action()` or a sync hold in your loop never does.
- A camera is encoded as VP8 (PyAV's libvpx, 640 px wide, 30 fps) only while a viewer watches it, and only the newest observation is sent, so a slow uplink drops frames rather than queueing them. Faces are blurred in every frame before it leaves the robot; `face_blur=False` sends the frames as captured.
- The key and the platform resolve as `DreamClient`'s do; a missing key is `DreamConfigError`. The link is `wss://<platform>/live/robot`, redialed with a backoff after a fault; a key or robot id the platform refuses is retried every 30 s and logged on the `dream.live` logger.
- Pass your loop's `DreamClient` as `dream_client=`: while the status is `running`, the client reports each new round trip of its `get_action()` (`last_timing`) to the page, at most every 2 s, as the station does. Without it the page never sees an answer and keeps showing "Dream-1 is getting ready", however many chunks your loop runs. A loop that opens a new `DreamClient` per run sets `live.dream_client` to it.
- Your code chooses the model: the page shows the `checkpoint` your `DreamClient` runs (`client.checkpoint`, with `dream_client=`) and locks its model and action pickers, since a model picked there would never reach your loop. `model_fixed=False` opens them: a Start then carries the viewer's pick, `get_model()` returns `(checkpoint, action_space)`, and your loop runs it.
- Callbacks run on the client's thread: keep them short, or return the work as an awaitable (`asyncio.run_coroutine_threadsafe(...)` from your own event loop), which the client waits for up to 10 s before answering the page; a Start, Stop, Home or Reset is never cancelled halfway. Return the state after the press, or `{"state": ..., "error": "..."}` when it failed. `register_callback(on_event, home=True, reset=True, prompt=True)` also takes the page's Home, Reset arms and new tasks (`EventChange.home`, `EventChange.reset`, `EventChange.prompt`); without them Home and Reset are answered "not supported" and a new task is `EventChange.end`.
- The station is one more caller of this client: `dreamcli robot up` builds one, streams it 60 times a second and answers every button with its own calls, so everything it shows on the page your loop can show too (`robots.station`, *Live view* in the robots README).
- Live View only shows the robot. Recording the run is `DreamClient`'s job, from the observations you hand `set_observation()`.
- It needs PyAV and OpenCV, which `dreamlabs` carries (`dreamlabs-client[live]` for the client alone); `import dream` loads neither. OpenCV's worker pool is started by the first `import cv2` and runs at that thread's priority; the client imports it on its own thread, at nice 10, so the resize and the blur never compete with your loop, unless your loop imported `cv2` first.

## Errors

Everything the library raises derives from `dream.DreamError`. Each type also subclasses the
built-in shown, so `except ValueError` / `except TimeoutError` / `except RuntimeError` keep
working.

| Exception | Built-in base | Raised when |
| --- | --- | --- |
| `DreamError` | `Exception` | Root of the hierarchy; also raised directly for misuse (request on a closed client, two `get_action` calls at once) |
| `DreamConfigError` | `ValueError` | A connection key has the wrong type or range, `checkpoint` is missing, or the config carries a key DreamAPI has no field for (all at `policy`); an inference call before `policy`; in the control plane, see *Fine-tuning as a Service* / *Service errors* |
| `DreamBadRequest` | `DreamConfigError`, `DreamServerError` | DreamAPI refuses the request as malformed (`DA-4000 BAD_REQUEST`), with its detail verbatim; a `DreamConfigError` (never retried) that also carries the envelope, so `reason` says `unknown_checkpoint`, `too_large`, `malformed`, `unknown_op`, `invalid_session_id` or `global_op_rejected` |
| `DreamObservationError` | `ValueError` | `set_observation` was given a tick it cannot keep (not one base64 `str` frame per window key, a frame key in the observation, other frame keys than earlier in the episode, a timestamp that goes back), `get_action` came before any `set_observation` in the episode, the observation carries `embodiment`, which the client sends from the config's `robot`, or the observation fails the embodiment check; raised before anything is sent |
| `DreamActionError` | `ValueError` | The returned action fails the embodiment check; the action is not returned |
| `DreamTimeout` | `TimeoutError` | No reply within `timeout_ms`; the connection is dropped before this propagates, so the next call opens a fresh one |
| `DreamProtocolError` | `RuntimeError` | DreamAPI cannot be reached, the reply is not JSON, or the reply carries no status envelope (the wrong thing is listening) |
| `DreamServerError` | `RuntimeError` | DreamAPI refused the request; `code`, `reason`, `retryable` and `retry_after_s` carry the status envelope, `message` DreamAPI's own wording (`DA-2000`, `DA-5000`, `DA-6000` and any future code) |
| `DreamAuthError` | `DreamServerError` | `DA-1000 UNAUTHORIZED`; never retried |
| `DreamDatasetError` (`dream.errors` only) | `ValueError` | `dream.datasets`: the local checks refused the dataset before anything was sent; the message is the one line `validate_dataset` reports |
| `DreamUploadError` (`dream.errors` only) | `RuntimeError` | `dream.datasets`: the dataset zip could not be written, the transfer did not finish, or the service refused the completion (the upload did not arrive at the declared size); after the registration the dataset is `failed` with the same line and its name is free again |
| `DreamSessionExpired` | `DreamServerError` | `DA-0001 OK_NEW_SESSION` with a `reason` other than `new` on a live session: DreamAPI replaced the session mid-episode (`expired`, `unknown`, `serve_died`). The new session is already adopted; the action that came with it is not returned |

One more type is a warning, not an error, and is never raised: `dream.DreamSlowNetworkWarning` (a `UserWarning`, not a `DreamError`), issued by `get_action` when the round trips stay slow (*Slow-network warning* above).

`DreamServerError` carries the envelope as attributes: `code` (e.g. `"DA-2000"`), `reason` (e.g.
`"rate_limited"`), `retryable`, and `retry_after_s` (seconds, or `None`). The client never acts on
them; the control loop decides whether to back off, retry or stop.

Session loss is not retried. A `DA-0001` on a request that carried a session id means the
episode restarted server-side: the client adopts the new id, raises `DreamSessionExpired`, and
leaves the next move to the caller (continue on the new session, or `reset_episode()` and start
over). Every replacement is reported, so a flapping gateway is visible as one exception per flap.

The client performs exactly one retry, at the transport layer and invisible to the caller: when
the gateway closes an idle keep-alive connection before reading a request (its idle timeout, or a
restart), the request is sent once more on a fresh connection. Nothing was processed the first
time, so it cannot double-execute; a second failure is a `DreamProtocolError`.

## Logging

The client keeps a diagnostic log of its own operation: one line per session boundary (the id the gateway minted, and whether it was started or replaced), and any warning it has to raise. This is the client's own record, not robot data. Observations, actions, prompts and the api key are never written to it.

During the beta the log is on by default. The client writes `~/.dream/logs/dream.log`, a rotating file capped at 5 MB with three rotated backups (at most about 20 MB on disk), so it cannot grow without bound. `import dream` is all it takes for the file to exist, so when a run goes wrong the log is already there to send us with a bug report.

Two environment variables control it, read once when `dream` is imported:

| Variable | Default | Effect |
| --- | --- | --- |
| `DREAM_LOG` | unset | The level: `debug`, `info`, `warning`, `error`, `critical`. A disabling value (`off`, `none`, `0`, `false`, `no`) turns the file off. |
| `DREAM_LOG_DIR` | `~/.dream/logs` | The directory the file is written in. |

Underneath, the library follows the ordinary convention: everything it logs goes to the named `dream` logger and its children (`dream.client`), and nothing configures the root logger. An application that sets up Python's `logging` therefore sees the client's records alongside its own and can route, reformat or silence them; `DREAM_LOG=off` leaves the client silent by default, exactly as a client library normally is. A home directory that cannot be written never fails the import: the client falls back to silence.

The file is meant for a human to read or for you to attach to a report. Its format carries no compatibility promise and may change in any release.

## Compatibility promise

`dream` follows semantic versioning. The public surface, and the only thing the version number
promises anything about, is:

- `dream.DreamClient`: its constructor; the methods of *Fine-tuning as a Service* (`add_embodiment`, `get_embodiments`, `get_models`, `validate_dataset`, `upload_dataset`, `get_dataset_status`, `get_datasets`, `retry_dataset`, `delete_dataset`, `finetune`, `get_finetuning_status`); the inference methods `policy`, `set_observation`, `get_action`, `reset_episode`, `refill_due` and `close`; the read-only properties `execution_horizon`, `action_hz`, `mode`, `last_timing` and `session_id`; and the context manager (`__enter__` / `__exit__`)
- the error types listed above, and `dream.DreamSlowNetworkWarning`
- `dream.__version__`
- `dream.datasets`: the functions of *Fine-tuning as a Service* and the `Embodiment`, `Report` and `Dataset` records they answer
- `dream.fine_tuning`: its five functions; the job and queue dicts they answer are the data service's `/v1` shapes, which the service's own docs promise
- `dream.models`: `list_models` and the `Model` record it answers

Everything else carries **no compatibility promise**: `dream.client`, `dream.transport`, the `PolicyClient` name and constructor, every `_`-prefixed module, function or attribute, and the wire format may change in any release, including a patch.
