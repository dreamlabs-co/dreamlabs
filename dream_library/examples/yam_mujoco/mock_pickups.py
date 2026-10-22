"""Deterministic contact-only pick/lift/release trials; no API or object attachments."""

import json
from pathlib import Path

import mujoco
import numpy as np
from yam_env import YamEnv


def solve(env, side, target, rotation, seed):
    """Solve a joint-limited grasp pose using scratch simulation data.
    @param env Simulation environment.
    @param side Arm name.
    @param target Desired target position.
    @param rotation Desired target rotation.
    @param seed Initial joint solution.
    @return The six joint angles that reach the pose.
    """
    m = env.model
    d = mujoco.MjData(m)
    mujoco.mj_copyData(d, m, env.data)
    ids = [m.joint(f"{side}_joint{i}").id for i in range(1, 7)]
    qa = m.jnt_qposadr[ids]
    va = m.jnt_dofadr[ids]
    limits = m.jnt_range[ids]
    site = m.site(f"{side}_tcp").id
    rng = np.random.default_rng(42)
    best = (1e9, None)
    for initial in [seed] + [rng.uniform(limits[:, 0], limits[:, 1]) for _ in range(14)]:
        q = np.array(initial).copy()
        for _ in range(180):
            d.qpos[qa] = q
            mujoco.mj_forward(m, d)
            current = d.site_xmat[site].reshape(3, 3)
            rot = rotation @ current.T
            angle = np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1))
            axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
            error_r = axis * angle / max(2 * np.sin(angle), 1e-8)
            error = target - d.site_xpos[site]
            score = np.linalg.norm(error) + 0.1 * angle
            if score < best[0]:
                best = (score, q.copy())
            if np.linalg.norm(error) < 0.004 and angle < 0.06:
                return q
            jp = np.zeros((3, m.nv))
            jr = jp.copy()
            mujoco.mj_jacSite(m, d, jp, jr, site)
            J = np.vstack([jp[:, va], jr[:, va] * 0.15])
            e = np.r_[error, error_r * 0.15]
            dq = J.T @ np.linalg.solve(J @ J.T + 0.0001 * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -0.12, 0.12), limits[:, 0], limits[:, 1])
    raise ValueError(f"IK failed {side} {target}: score {best[0]:.4f}")


def run_trial(fruit, yaw=0, lift=0.15):
    """Run one scripted physical pickup and release trial.
    @param fruit Scene fruit name.
    @param yaw Grasp yaw in radians.
    @param lift Lift distance in metres.
    @return The trial's result: the object, whether it passed, and the heights and positions measured.
    """
    env = YamEnv()
    try:
        home = np.r_[np.zeros(6), 1.0]
        for _ in range(30):
            env.step(home, home)
        p = env.data.body(fruit).xpos.copy()
        initial_z = p[2]
        side = "left" if p[0] > 0 else "right"
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, s, 0], [s, -c, 0], [0, 0, -1.0]])
        q = env.state()[side][:6].copy()
        limits = env.model.jnt_range[[env.model.joint(f"{side}_joint{i}").id for i in range(1, 7)]]
        q = np.clip(q, limits[:, 0], limits[:, 1])

        def move(target, opening, steps):
            nonlocal q
            end = solve(env, side, np.asarray(target), rotation, q)
            start = q.copy()
            for alpha in np.linspace(0, 1, steps):
                command = np.r_[start + (end - start) * alpha, opening]
                env.step(command, home) if side == "left" else env.step(home, command)
            q = end

        move(p + [0, 0, lift - 0.01], 1, 60)
        move(p + [0, 0, -0.009 if fruit == "orange" else 0.003], 1, 60)
        for opening in np.linspace(1, 0, 30):
            cmd = np.r_[q, opening]
            env.step(cmd, home) if side == "left" else env.step(home, cmd)
        move(p + [0, 0, lift], 0, 70)
        min_height = 10.0
        max_slip = 0.0
        for _ in range(40):
            cmd = np.r_[q, 0.0]
            env.step(cmd, home) if side == "left" else env.step(home, cmd)
            min_height = min(min_height, env.data.body(fruit).xpos[2] - initial_z)
            max_slip = max(max_slip, np.linalg.norm(env.data.body(fruit).xpos - env.data.site(f"{side}_tcp").xpos))
        held = env.data.body(fruit).xpos.copy()
        out = Path("mock_results")
        out.mkdir(exist_ok=True)
        from PIL import Image

        Image.fromarray(env.images()["top"]).save(out / f"{fruit}-held.png")
        for _ in range(35):
            cmd = np.r_[q, 1.0]
            env.step(cmd, home) if side == "left" else env.step(home, cmd)
        drop = held[2] - env.data.body(fruit).xpos[2]
        result = dict(
            object=fruit,
            yaw=float(yaw),
            lift_m=float(min_height),
            tcp_distance_m=float(max_slip),
            release_drop_m=float(drop),
            passed=bool(min_height > 0.08 and max_slip < 0.07 and drop > 0.06),
        )
        print(json.dumps(result), flush=True)
        return result
    finally:
        env.close()


if __name__ == "__main__":
    results = []
    for fruit in ("banana", "apple", "orange"):
        try:
            results.append(
                run_trial(fruit, yaw=np.pi / 2 if fruit == "banana" else 0, lift=0.10 if fruit == "orange" else 0.15)
            )
        except ValueError as error:
            print(f"{fruit}: {error}", flush=True)
            results.append(dict(object=fruit, passed=False, error=str(error)))
    Path("mock_results").mkdir(exist_ok=True)
    Path("mock_results/results.json").write_text(json.dumps(results, indent=2))
    raise SystemExit(0 if all(r["passed"] for r in results) else 1)
