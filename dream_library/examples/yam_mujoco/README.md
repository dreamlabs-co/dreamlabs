# Dream YAM Sim

A local MuJoCo playground with two YAM arms, three live RGB cameras, fruit, a bowl, and a plate. The terminal sends observations to Dream through the dream library's client (`dream.DreamClient`) and executes returned joint chunks. Assets are bundled; viewing and offline tests need no API key.

## Start

The dream library's wheel carries this example (as `dream_examples/yam_mujoco/`, `robots.examples`), so with the library installed, from the platform's Setup API page:

```bash
dreamcli example yam-mujoco          # --show-side-panel, --verbose; --self-test checks the install headless
```

From a source checkout, to change the example, use Python 3.11 or later in a virtual environment and run from this directory. The example needs the dream library it ships in (its client, and the YAM kinematics the base models read), so install that from the checkout first:

```bash
python -m pip install ../..
python yam_mujoco.py
```

The model it calls is the one your credentials file (`~/.dreamlabs/credentials`, written as your platform's Setup API page shows) names, as for the dream library; with no file, production. First-run setup prompts for a key if none is configured. `/login` replaces it. Keys are stored outside the project at `~/.config/dream-yam-sim/credentials.json` with owner-only permissions. The key is looked up in the dream library's order with the saved one in between: `DREAM_API_KEY` in the environment, then the `/login` key, then `~/.dreamlabs/credentials`. Saving does not verify the key until the next task. API tasks use account credits.

The terminal launches `mjpython` automatically on macOS. Linux and Windows use Python. Rendering requires an OpenGL context. The three live camera tiles remain visible when native MuJoCo panels are hidden.

## Tasks and controls

```text
put apple into the bowl
put orange into the bowl
put banana on the plate
```

On the first run the prompt starts with `put apple into the bowl`; press Enter to run it. Entering a task while another runs replaces it: the running task stops once its pending response returns, and the new one starts. Tab on an empty prompt shows recommendations. Tab after a prefix completes it. Up and Down recall commands, including across restarts. History is saved at `~/.config/dream-yam-sim/history`; login input and recognized API-key pastes are excluded. A live status bar updates RTT after each response.

| Command | Purpose |
| --- | --- |
| `/model` | Choose Beta, or `/model <id>` for any model, such as one of your fine-tunes |
| `/actions 24` | Rows executed per response |
| `/seconds 120` | Wall-clock trial duration, 1 to 86400 seconds (a day) |
| `/reset` | Restore the scene: objects and arms; stops a running task |
| `/reset_robot` | Drive the arms home (joints zero, grippers open) over 2 s, leaving the objects; stops a running task |
| `/collisions` | Toggle contact overlay; C in the viewer works during execution |
| `/login` | Replace the local API key |
| `/status`, `/help` | Settings and help |
| `/quit` | Close the workspace |

First Ctrl-C stops a task; a second consecutive Ctrl-C exits. The same scene and viewer persist between tasks. Use `--show-side-panel` for native MuJoCo controls or `--verbose` for detailed inference and adjustment logs.

Defaults are `dreamlabs/dream-1-beta`, joint control, 48 actions per response, and 120-second trials. Requests use the model's own keys: the measured joints and grippers as `state.right_arm`, `state.right_gripper`, `state.left_arm` and `state.left_gripper`, and each camera's last nine 320x176 JPEG frames, 0.2 s apart in simulated time, as `video.overhead`, `video.wrist_l` and `video.wrist_r` (the top, left and right cameras). A model that also reads hand poses (`dreamlabs/dream-1-beta`) is sent the YAM station's keys instead: `state.arm_r.joint_pos`, `state.arm_r.gripper_pos`, `state.arm_l.joint_pos`, `state.arm_l.gripper_pos`, `image_window.top`, `image_window.left`, `image_window.right`, and `robot.arm` `yam`, `robot.exec_head` `joint`, `robot.max_joint_vel_rad_s` 0 and `robot.gripper_open` 1, so the serve computes each hand's pose from the joints in the frame the model was trained in. The model's own keys cannot carry the poses yet: the serve keeps only the state parts the checkpoint lists, which for such a model are the joints, so poses the client adds (`ik` arm `yam`) are dropped and the request is refused with `DA-5000` "EEF checkpoint: send ik.arm ... or [state.*_eef_*]". Any other model refused for missing hand poses is reopened on the station keys. Finite joint targets are clipped to URDF limits; malformed/nonfinite commands are rejected. Physics holds during inference; a pending network request can delay shutdown. Task success is not automatically detected.

## Offline validation

```bash
python yam_env.py --headless --frames frames
python dream_control.py --dry-run --headless
python -m unittest -v
python mock_pickups.py
python audit_collisions.py
```

These commands make no inference calls. Scripted trials approach, grasp, lift, hold, and release each fruit through joint servos, without attachments or object teleporting. They test contact mechanics with known positions, not Dream policy success. Compatible headless Linux GPUs can use `MUJOCO_GL=egl`.

## Geometry and limitations

See [CALIBRATION.md](CALIBRATION.md) for the pinned station cameras and frame conversion. The world is Z-up, left/right follow the center camera, and bases have a 12 mm clearance offset. Cameras produce RGB, not depth.

Robot contacts use convex decompositions of the source meshes. Table, plate, orange, and bottle core share visual/contact primitives. Apple uses a scan hull, banana uses 16 convex sections, and the bowl preserves its open cavity. Adjacent bearing parts and opposing closed fingers are excluded from self-contact; slide limits define the finger stop. Other arm/environment and non-adjacent self-contact are enabled. Collision overlays never enter policy images.

Fine recesses remain approximate. Grip friction (6/.08/.008), 22 N force limits, and banana mass (45 g) are simulation tuning, not hardware calibration. Live apple/orange-to-bowl trials succeeded once each; banana-to-plate is inconsistent. Scripted success is not a model reliability benchmark.

## Files and assets

`yam_env.py` generates the scene. `dream_control.py` handles requests and actions. `viewer_workspace.py` owns the viewer; `camera_panel.py` and `collision_view.py` draw overlays. `credentials.py` and `terminal_input.py` provide login and input. `yam_mujoco.py` is the entry point.

Robot meshes retain [assets/LICENSE](assets/LICENSE). Fruit sources, licenses, and changes are listed in [assets/fruits/ATTRIBUTION.md](assets/fruits/ATTRIBUTION.md). The convex robot parts are packed in `assets/collisions.npz` ([assets/COLLISIONS.md](assets/COLLISIONS.md)); no decomposition tool is required at runtime. `tools/pack_assets.py` rebuilds that file and the 1024 px fruit textures after a new decomposition or scan download.
