"""Run with python -m unittest -v. Fixtures are independent URDF FK outputs."""

import json
import unittest

import mujoco
import numpy as np
from yam_env import ASSETS, YamEnv


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = YamEnv()
        self.addCleanup(self.env.close)

    def test_urdf_poses(self):
        base_rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]])
        tool_roll = base_rotation  # physical gripper = URDF TCP * Rz(-pi/2)
        for sample in json.loads((ASSETS / "fk-fixtures.json").read_text()):
            eef = np.asarray(sample["eef"])
            rotation = np.column_stack([eef[3:6], eef[6:9], np.cross(eef[3:6], eef[6:9])])
            for side, x in [("left", 0.305), ("right", -0.305)]:
                for i, q in enumerate(sample["joints"], 1):
                    self.env.data.joint(f"{side}_joint{i}").qpos[0] = q
                # FK fixtures teleport through potentially intersecting poses; only evaluate kinematics.
                mujoco.mj_kinematics(self.env.model, self.env.data)
                site = self.env.data.site(f"{side}_tcp")
                np.testing.assert_allclose(site.xpos, base_rotation @ eef[:3] + [x, 0.24, 0.772], atol=1e-8)
                np.testing.assert_allclose(site.xmat.reshape(3, 3), base_rotation @ rotation @ tool_roll, atol=1e-7)

    def test_controls_and_settling(self):
        target = np.array([0.1, 0.15, 0.2, 0.05, 0, 0, 0.0])
        for _ in range(100):
            state = self.env.step(target, target)
        for side in ("left", "right"):
            np.testing.assert_allclose(state[side][:6], target[:6], atol=0.015)
            self.assertLess(state[side][6], 0.03)
        self.assertTrue(np.isfinite(self.env.data.qpos).all())
        self.assertAlmostEqual(self.env.data.body("apple").xpos[2], 0.7914, delta=0.002)
        home = self.env.reset()
        np.testing.assert_allclose(home["left"], [0, 0, 0, 0, 0, 0, 1])

    def test_fingertips_hold_then_release_apple(self):
        # Place a real dynamic object in a pre-grasp pose; no welds/attachments.
        env = self.env
        tcp = env.data.site("left_tcp").xpos.copy()
        env.data.joint("apple_free").qpos[:3] = tcp
        for sign in (-1, 1):
            env.data.joint(f"left_finger_{sign}").qpos[0] = 0.032
            env.data.actuator(f"left_finger_{sign}").ctrl[0] = 0
        mujoco.mj_forward(env.model, env.data)
        mujoco.mj_step(env.model, env.data, nstep=1000)
        # Rotate the loaded arm through a slow 0.35 rad sweep, checking slip.
        for target in np.linspace(0, 0.35, 40):
            env.data.actuator("left_joint1").ctrl[0] = target
            mujoco.mj_step(env.model, env.data, nstep=25)
            mujoco.mj_forward(env.model, env.data)
            self.assertLess(np.linalg.norm(env.data.body("apple").xpos - env.data.site("left_tcp").xpos), 0.025)
        tcp = env.data.site("left_tcp").xpos.copy()
        held = env.data.body("apple").xpos.copy()
        self.assertLess(np.linalg.norm(held - tcp), 0.02)
        for sign in (-1, 1):
            env.data.actuator(f"left_finger_{sign}").ctrl[0] = 0.0475
        mujoco.mj_step(env.model, env.data, nstep=500)
        self.assertLess(env.data.body("apple").xpos[2], held[2] - 0.08)
        self.assertTrue(np.isfinite(env.data.qpos).all())

    def test_joint_targets_clip_to_urdf_limits(self):
        left = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        right = left.copy()
        right[1] = -5.83245939589792e-11
        left[0] = 99.0
        left[2] = -0.01
        right[3] = 2.0
        self.env.step(left, right)
        for side, original in [("left", left), ("right", right)]:
            for i in range(1, 7):
                expected = np.clip(original[i - 1], *self.env.model.joint(f"{side}_joint{i}").range)
                self.assertEqual(self.env.data.actuator(f"{side}_joint{i}").ctrl[0], expected)
        self.assertEqual(right[1], -5.83245939589792e-11)

    def test_bad_action_is_atomic(self):
        previous = self.env.data.ctrl.copy()
        with self.assertRaises(ValueError):
            self.env.step([0.1, 0, 0, 0, 0, 0, 1], [np.nan, 0, 0, 0, 0, 0, 1])
        np.testing.assert_array_equal(self.env.data.ctrl, previous)


## What a station request carries: the joints, the station fields and the three camera windows.
STATION_KEYS = (
    "state.arm_r.joint_pos",
    "state.arm_r.gripper_pos",
    "state.arm_l.joint_pos",
    "state.arm_l.gripper_pos",
    "robot.arm",
    "robot.exec_head",
    "robot.max_joint_vel_rad_s",
    "robot.gripper_open",
    "image_window.top",
    "image_window.left",
    "image_window.right",
)


class DreamContractTests(unittest.TestCase):
    def test_station_contract_and_chunk_validation(self):
        from dream_control import observation, validate_chunk

        env = YamEnv()
        self.addCleanup(env.close)
        windows = {name: ["jpeg"] * 9 for name in ("top", "left", "right")}
        for model in ("dreamlabs/test-yam-joints", "dreamlabs/dream-1-beta"):
            obs = observation(env, windows, "place tomato into the bowl", model)
            self.assertEqual(obs["prompt"], "place tomato into the bowl")
            self.assertEqual(obs["state.left_gripper"], [1.0])
            self.assertEqual(len(obs["state.right_arm"]), 6)
            self.assertEqual(len(obs["video.overhead"]), 9)
            self.assertEqual(
                {k for k in obs if k.startswith("video.")}, {"video.overhead", "video.wrist_l", "video.wrist_r"}
            )
            self.assertFalse(any(k.startswith(("image_window.", "state.arm_")) for k in obs), "no deprecated key")
        actions = {k: [[0, 0, 0, 0, 0, 0, 1]] * 48 for k in ("q_l_chunk", "q_r_chunk")}
        left, right = validate_chunk(env, actions)
        env.step(left[0], right[0])
        actions["q_r_chunk"] = [[0, 0, 0, 0, 0, 0, 99]] * 48
        with self.assertRaises(ValueError):
            validate_chunk(env, actions)

    def test_small_overshoots_and_unused_tail(self):
        from dream_control import validate_chunk

        env = YamEnv()
        self.addCleanup(env.close)
        rows = np.tile([0.0, 0, 0, 0, 0, 0, 1], (48, 1))
        rows[0, 1] = -0.001
        rows[0, 6] = 1.002
        rows[30, 1] = -1.0
        actions = {"q_l_chunk": rows.tolist(), "q_r_chunk": rows.tolist()}
        left, right = validate_chunk(env, actions, 24)
        self.assertEqual(left.shape, (24, 7))
        self.assertEqual(left[0, 1], 0.0)
        self.assertEqual(left[0, 6], 1.0)
        self.assertEqual(actions["q_l_chunk"][0][1], -0.001)
        env.step(left[0], right[0])
        full, _ = validate_chunk(env, actions, 48)
        self.assertEqual(full[30, 1], 0.0)

    def _run_with_policies(self, replies, model="dreamlabs/test-yam-joints"):
        """Runs one task on fake dream client policies: each opened policy answers the next of `replies` (an action
        dict, or an exception to raise), and the run stops after the first chunk.
        @return The configs the policies were opened with, and each policy's calls."""
        import threading
        from unittest.mock import patch

        from dream_control import main

        env = YamEnv()
        self.addCleanup(env.close)
        stop = threading.Event()
        opened, answers = [], iter(replies)

        class FakePolicy:
            def __init__(self, config):
                self.config, self.calls, self.fed = config, [], []

            def set_observation(self, frames, state, timestamp):
                self.fed.append((frames, state, timestamp))

            def get_action(self):
                self.calls.append(("get_action", dict(self.fed[-1][1], **self.fed[-1][0])))
                answer = next(answers)
                if isinstance(answer, Exception):
                    raise answer
                stop.set()
                return answer

            def close(self):
                self.calls.append(("close", None))

        def open_policy(config, key):
            opened.append(FakePolicy(config))
            return opened[-1]

        def fake_capture(env, windows, keep=True):
            for window in windows.values():
                if keep:
                    window.append("jpeg")

        with (
            patch("dream_control.settings", return_value={"DREAM_API_KEY": "test"}),
            patch("dream_control.capture", side_effect=fake_capture),
            patch("dream_control.open_policy", side_effect=open_policy),
        ):
            main(["--headless"] + (["--model", model] if model else []), shared_env=env, stop_event=stop)
        return env, opened

    def test_a_server_side_episode_restart_asks_again_up_to_three_times(self):
        class DreamSessionExpired(Exception):
            pass

        chunk = {"q_l_chunk": [[0.2, 0, 0, 0, 0, 0, 1]], "q_r_chunk": [[0.2, 0, 0, 0, 0, 0, 1]]}
        env, opened = self._run_with_policies([DreamSessionExpired(), chunk])
        self.assertEqual([call for call, _ in opened[0].calls], ["get_action", "get_action", "close"])
        self.assertEqual((len(opened), len(opened[0].fed)), (1, 9), "no new session, nothing fed again")
        with self.assertRaises(DreamSessionExpired):
            self._run_with_policies([DreamSessionExpired()] * 4)

    def test_a_line_with_no_word_is_not_a_task(self):
        from dream_control import is_task

        self.assertEqual([is_task(t) for t in ("1", "a", "?", "x y z")], [False] * 4)
        self.assertEqual([is_task(t) for t in ("put apple into the bowl", "kill", "把苹果放进碗里")], [True] * 3)

    def test_the_slow_network_warning_is_silenced(self):
        import warnings
        from unittest.mock import Mock, patch

        import dream
        from dream_control import open_policy

        with warnings.catch_warnings(record=True) as seen, patch.object(dream, "DreamClient", Mock()):
            warnings.simplefilter("always")
            open_policy({}, "key")
            warnings.warn("slow", dream.DreamSlowNetworkWarning)
            warnings.warn("other", UserWarning)
        self.assertEqual([str(w.message) for w in seen], ["other"])

    def test_cancel_pending_inference_releases_session_without_playback(self):
        chunk = {"q_l_chunk": [[0.2, 0, 0, 0, 0, 0, 1]], "q_r_chunk": [[0.2, 0, 0, 0, 0, 0, 1]]}
        env, opened = self._run_with_policies([chunk])
        self.assertEqual([call for call, _ in opened[0].calls], ["get_action", "close"], "the session is ended")
        stamps = [timestamp for _, _, timestamp in opened[0].fed]
        self.assertEqual(len(stamps), 9, "the window's nine frames before the first request")
        self.assertTrue(all(abs((b - a) - 0.2) < 1e-3 for a, b in zip(stamps, stamps[1:])), "0.2 s apart")
        self.assertNotIn("ik", opened[0].config, "a joints-only model reads no hand pose")
        self.assertAlmostEqual(env.data.actuator("left_joint1").ctrl[0], 0.0)

    def test_a_model_that_reads_hand_poses_is_reopened_on_the_station_keys(self):
        refusal = RuntimeError(
            "DA-5000 SERVE_FAULT (error): inference error: EEF checkpoint: send ik.arm and its speed cap "
            "or ['state.left_eef_xyz', 'state.left_eef_rot6']"
        )
        refusal.reason = "error"
        chunk = {"q_l_chunk": [[0.0] * 6 + [1.0]], "q_r_chunk": [[0.0] * 6 + [1.0]]}
        _, opened = self._run_with_policies([refusal, chunk], model="dream-1-ft:0a1b2c3d-yam-eef")
        self.assertEqual(len(opened), 2, "opened again after the refusal")
        self.assertEqual(opened[0].calls[-1][0], "close")
        self.assertNotIn("ik", opened[1].config)
        self.assertNotIn("embodiment", opened[1].config, "the station keys are not the strict-mode embodiment's")
        self.assertEqual(set(STATION_KEYS) - set(opened[1].calls[0][1]), set())

    def test_the_beta_model_is_the_default_and_speaks_the_station_keys(self):
        from dream_control import BETA_MODEL, STATION_ROBOT, policy_config

        chunk = {"q_l_chunk": [[0.0] * 6 + [1.0]], "q_r_chunk": [[0.0] * 6 + [1.0]]}
        _, opened = self._run_with_policies([chunk], model=None)
        self.assertEqual(opened[0].config["checkpoint"], BETA_MODEL, "the default model")
        sent = opened[0].calls[0][1]
        self.assertEqual(set(STATION_KEYS) - set(sent), set())
        self.assertEqual({k: sent[k] for k in STATION_ROBOT}, STATION_ROBOT)
        self.assertEqual(len(sent["state.arm_r.joint_pos"]), 6)
        self.assertFalse(any(k.startswith("video.") or "_eef_" in k for k in sent), "no model key, no pose")
        self.assertNotIn("ik", policy_config(BETA_MODEL, "https://api.dreamlabs.ai", True))
        self.assertFalse(policy_config(BETA_MODEL, "https://api.dreamlabs.ai", True)["_record"])
        self.assertIn("embodiment", policy_config("dreamlabs/test-yam-joints", "https://api.dreamlabs.ai", False))

    def test_a_model_is_a_shortcut_or_any_model_id(self):
        from dream_control import BETA_MODEL, observation, resolve_model

        self.assertEqual(resolve_model(" Beta "), BETA_MODEL)
        self.assertEqual(resolve_model("1"), BETA_MODEL)
        self.assertEqual(resolve_model("dream-1-ft:0a1b2c3d-towels"), "dream-1-ft:0a1b2c3d-towels")
        self.assertEqual(resolve_model("run-7/checkpoint-12000"), "run-7/checkpoint-12000")
        for bad in ("", "  ", "my model"):
            with self.assertRaises(ValueError):
                resolve_model(bad)
        env = YamEnv()
        self.addCleanup(env.close)
        windows = {name: ["jpeg"] * 9 for name in ("top", "left", "right")}
        for model in ("dream-1-ft:0a1b2c3d-towels", BETA_MODEL):
            sent = observation(env, windows, "put apple into the bowl", model)
            self.assertFalse(any(key.startswith("robot.") for key in sent), "the model's own keys only")
        sent = observation(env, windows, "put apple into the bowl", BETA_MODEL, station=True)
        self.assertEqual(set(STATION_KEYS) - set(sent), set())
        self.assertEqual(sent["state.arm_l.gripper_pos"], [1.0])

    def test_frames_reach_the_model_0_2_s_apart(self):
        from unittest.mock import patch

        from dream_control import FRAME_EVERY_STEPS, FRAMES, main

        env = YamEnv()
        self.addCleanup(env.close)
        kept = []

        def fake_capture(env, windows, keep=True):
            kept.append(keep)
            for window in windows.values():
                if keep:
                    window.append("jpeg")

        with (
            patch("dream_control.settings", return_value={"DREAM_API_KEY": "test"}),
            patch("dream_control.capture", side_effect=fake_capture),
        ):
            main(["--dry-run", "--headless"], shared_env=env)
        indices = [i for i, k in enumerate(kept) if k]
        self.assertEqual(len(indices), FRAMES)
        self.assertEqual({b - a for a, b in zip(indices, indices[1:])}, {FRAME_EVERY_STEPS}, "one frame every 0.2 s")

    def test_reference_camera_calibration(self):
        from yam_env import CALIBRATION, TOP_CAMERA_LOWER_M, WORLD_OFFSET, WORLD_ROTATION

        env = YamEnv()
        self.addCleanup(env.close)
        for name, source in [("top", "top"), ("left", "left_wrist"), ("right", "right_wrist")]:
            entry = CALIBRATION["cameras"][source]
            camera = env.model.camera(name)
            k = entry["intrinsics"]
            pixel_size = env.model.cam_sensorsize[camera.id] / env.model.cam_resolution[camera.id]
            np.testing.assert_allclose(
                env.model.cam_intrinsic[camera.id, :2] / pixel_size, [k["fx"], k["fy"]], atol=1e-4
            )
            np.testing.assert_allclose(
                env.model.cam_intrinsic[camera.id, 2:] / pixel_size, [320 - k["cx"], k["cy"] - 240], atol=1e-4
            )
            # The top camera sits TOP_CAMERA_LOWER_M below its calibrated position; the wrist cameras are as calibrated.
            expected = (
                WORLD_ROTATION @ entry["position"] + WORLD_OFFSET - [0, 0, TOP_CAMERA_LOWER_M]
                if name == "top"
                else entry["position"]
            )
            np.testing.assert_allclose(camera.pos, expected, atol=1e-9)
            if name != "top":
                self.assertEqual(env.model.cam_bodyid[camera.id], env.model.body(f"{name}_link6").id)

    def test_top_camera_is_tilted_up_from_its_calibration_by_the_set_angle(self):
        import yam_env

        def pitch():
            env = YamEnv()
            self.addCleanup(env.close)
            view = -env.data.cam_xmat[env.model.camera("top").id].reshape(3, 3)[:, 2]
            return np.degrees(np.arcsin(view[2]))

        tilted = pitch()
        original = yam_env.TOP_CAMERA_LOWER_M, yam_env.TOP_CAMERA_TILT_UP_DEG
        yam_env.TOP_CAMERA_LOWER_M, yam_env.TOP_CAMERA_TILT_UP_DEG = 0.0, 0.0
        try:
            calibrated = pitch()
        finally:
            yam_env.TOP_CAMERA_LOWER_M, yam_env.TOP_CAMERA_TILT_UP_DEG = original
        self.assertAlmostEqual(tilted - calibrated, yam_env.TOP_CAMERA_TILT_UP_DEG, places=6)

    def test_home_path_eases_to_home_and_reset_robot_leaves_the_objects(self):
        from yam_env import HOME, home_path

        env = YamEnv()
        self.addCleanup(env.close)
        away = np.r_[0.4, 0.8, 0.6, 0.2, -0.3, 0.1, 0.0]
        for _ in range(40):
            env.step(away, away * [-1, 1, 1, -1, 1, -1, 1])
        apple = env.data.body("apple").xpos.copy()
        path = home_path(env.state())
        self.assertEqual(len(path), 40)
        np.testing.assert_allclose(path[-1][0], HOME)
        np.testing.assert_allclose(path[-1][1], HOME)
        steps = [np.abs(b[0] - a[0]).max() for a, b in zip(path, path[1:])]
        self.assertLess(steps[0], steps[len(steps) // 2], "eases in")
        for left, right in path:
            env.step(left, right)
        for _ in range(20):
            env.step(HOME, HOME)
        state = env.state()
        for side in ("left", "right"):
            np.testing.assert_allclose(state[side][:6], 0, atol=0.02)
            self.assertGreater(state[side][6], 0.95)
        np.testing.assert_allclose(env.data.body("apple").xpos, apple, atol=0.002)

    def test_stock_visuals_and_white_panels(self):
        env = YamEnv()
        self.addCleanup(env.close)
        for mesh in ("tip_left.stl", "tip_right.stl", "link2_white.stl", "link3_white.stl"):
            self.assertGreaterEqual(env.model.mesh(mesh).id, 0)
        # Closing/opening keeps both stock fingers moving away from the common center.
        for side in ("left", "right"):
            positions = []
            for opening in (0, 0.0475):
                for sign in (-1, 1):
                    env.data.joint(f"{side}_finger_{sign}").qpos[0] = opening
                mujoco.mj_forward(env.model, env.data)
                pads = []
                for tip in ("tip_left", "tip_right"):
                    body = env.model.body(f"{side}_{tip}")
                    pads.append(env.data.geom_xpos[body.geomadr[0]].copy())
                positions.append(np.linalg.norm(pads[0] - pads[1]))
            self.assertGreater(positions[1], positions[0])

    def test_viewer_reset_is_undone_during_a_run_and_applies_outside_one(self):
        import contextlib
        import io
        import threading
        from unittest.mock import MagicMock, patch

        from dream_control import RESET_REFUSED, main

        env = YamEnv()
        self.addCleanup(env.close)
        apple = env.model.jnt_qposadr[env.model.body("apple").jntadr[0]]
        env.data.qpos[apple] += 0.05  # off its spawn point, where MuJoCo's reset would put it back
        mujoco.mj_forward(env.model, env.data)
        stop = threading.Event()

        class FakeViewer:
            """MuJoCo 3.13's passive viewer: Backspace only flags the reset; sync() applies it (Simulate::Sync)."""

            def __init__(self):
                self.pending_reset, self.syncs = False, 0

            def is_running(self):
                return True

            def sync(self):
                self.syncs += 1
                if self.pending_reset:
                    mujoco.mj_resetData(env.model, env.data)
                    mujoco.mj_forward(env.model, env.data)
                    self.pending_reset = False

        class FakePolicy:
            def set_observation(self, frames, state, timestamp):
                pass

            def get_action(self):
                return {"q_l_chunk": [[0.2, 0, 0, 0, 0, 0, 1]] * 6, "q_r_chunk": [[0.2, 0, 0, 0, 0, 0, 1]] * 6}

            def close(self):
                pass

        viewer = FakeViewer()
        moved = env.step

        def step(left, right):
            # Backspace twice mid-run, then end the run.
            state = moved(left, right)
            viewer.pending_reset = viewer.syncs in (2, 3)
            if viewer.syncs >= 5:
                stop.set()
            return state

        def fake_capture(env, windows, keep=True):
            for window in windows.values():
                if keep:
                    window.append("jpeg")

        printed = io.StringIO()
        with (
            patch("dream_control.settings", return_value={"DREAM_API_KEY": "test"}),
            patch("dream_control.capture", side_effect=fake_capture),
            patch("dream_control.open_policy", return_value=FakePolicy()),
            patch("dream_control.aim_viewer"),
            patch("camera_panel.CameraPanel"),
            patch.object(env, "step", side_effect=step),
            contextlib.redirect_stdout(printed),
        ):
            main(
                ["--headless", "--quiet"],
                shared_env=env,
                shared_viewer=viewer,
                stop_event=stop,
                collision_view=MagicMock(),
            )
        self.assertGreater(env.data.time, 0.1, "the run's clock was not set back")
        self.assertGreater(env.data.qpos[apple] - env.model.qpos0[apple], 0.04, "the apple stayed where it was")
        self.assertGreater(env.data.actuator("left_joint1").ctrl[0], 0.1, "the arms kept the policy's target")
        self.assertEqual(printed.getvalue().count(RESET_REFUSED), 1, "told once, not once per key repeat")
        # Outside a run the workspace syncs plainly, and MuJoCo's reset does what it always did.
        viewer.pending_reset = True
        viewer.sync()
        self.assertEqual(env.data.time, 0)
        np.testing.assert_allclose(env.data.qpos, env.model.qpos0)


if __name__ == "__main__":
    unittest.main()
