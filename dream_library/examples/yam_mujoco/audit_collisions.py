"""Audit compiled geometry bounds and collision coverage; no API or viewer needed."""

import json
from pathlib import Path

import mujoco
import numpy as np
from yam_env import YamEnv


def bounds(model, data, geom):
    """Compute world-axis bounds of one compiled geom.
    @param model Selected model or compiled MuJoCo model.
    @param data MuJoCo simulation state.
    @param geom Geom index.
    @return A (2, 3) array of the minimum and maximum corners, or None for a plane.
    """
    kind = int(model.geom_type[geom])
    rotation = data.geom_xmat[geom].reshape(3, 3)
    center = data.geom_xpos[geom]
    size = model.geom_size[geom]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        points = (
            np.einsum("ij,kj->ik", np.asarray(model.mesh_vert[start : start + count], dtype=float), rotation) + center
        )
        return np.array([points.min(0), points.max(0)])
    if kind == mujoco.mjtGeom.mjGEOM_PLANE:
        return None
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        extent = np.full(3, size[0])
    elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        extent = np.sqrt((rotation**2) @ (size**2))
    elif kind in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        axis = rotation[:, 2]
        extent = np.abs(axis) * size[1]
        extent += size[0] if kind == mujoco.mjtGeom.mjGEOM_CAPSULE else size[0] * np.sqrt(np.maximum(0, 1 - axis**2))
    else:
        extent = np.abs(rotation) @ size
    return np.array([center - extent, center + extent])


def audit():
    """Inventory visual and collision geometry for each body.
    @return The report: a description and one row per body with its visual and collision geom counts and bounds.
    """
    env = YamEnv()
    try:
        model, data = env.model, env.data
        rows = []
        for body in range(model.nbody):
            ids = [
                g for g in range(model.ngeom) if model.geom_bodyid[g] == body and bounds(model, data, g) is not None
            ]
            visual = [g for g in ids if model.geom_rgba[g, 3] > 0]
            collision = [g for g in ids if model.geom_contype[g] or model.geom_conaffinity[g]]
            if not ids:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "world"
            row = dict(body=name, visual_geoms=len(visual), collision_geoms=len(collision))
            if visual and collision:

                def union(gs):
                    boxes = np.array([bounds(model, data, g) for g in gs])
                    return np.array([boxes[:, 0].min(0), boxes[:, 1].max(0)])

                v, c = union(visual), union(collision)
                excess = np.maximum(0, np.r_[v[0] - c[0], c[1] - v[1]]) * 1000
                row["bounds_excess_mm"] = round(float(excess.max()), 3)
                row["bounds_difference_mm"] = round(float(np.abs(v - c).max() * 1000), 3)
                row["status"] = "bounds_match" if row["bounds_difference_mm"] <= 1 else "approximation_review"
                if name.startswith(("left_", "right_")):
                    row["status"] = "decomposed_mesh" if collision else "visual_only"
            else:
                row["status"] = "visual_only" if visual else "collision_only"
            rows.append(row)
        return dict(
            description="World-axis bounds at reset. Bounds agreement is necessary, not proof of surface agreement. Mesh contacts use convex hulls; robot meshes are convex decompositions; adjacent joint bearings are excluded from self-contact.",
            bodies=rows,
        )
    finally:
        env.close()


if __name__ == "__main__":
    report = audit()
    destination = Path("collision_audit.json")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    for row in report["bodies"]:
        print(f"{row['body']:28} {row['status']:24} bounds delta {row.get('bounds_difference_mm', '—')} mm")
    print(f"Full report: {destination.resolve()}")
