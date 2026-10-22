"""Viewer-only collision overlays. Never change physics or policy camera images."""

import mujoco
import numpy as np


class CollisionView:
    """Display contact geometry without changing physics or camera observations."""

    def __init__(self, env):
        """Initialize the object and its owned state.
        @param env Simulation environment.
        """
        self.env = env
        self.enabled = False

    def toggle(self, key=None):
        """Toggle the collision overlay for its keyboard shortcut.
        @param key The key code pressed in the viewer, or None to toggle unconditionally.
        """
        if key is None or key in (ord("C"), ord("c")):
            self.enabled = not self.enabled
            print(f"  Collision overlay {'ON' if self.enabled else 'OFF'} (cyan).", flush=True)

    def update(self, viewer):
        """Refresh the viewer overlay.
        @param viewer Native MuJoCo viewer.
        """
        model, data = self.env.model, self.env.data
        with viewer.lock():
            scene = viewer.user_scn
            scene.ngeom = 0
            if not self.enabled:
                return
            for i in range(model.ngeom):
                if not (model.geom_contype[i] or model.geom_conaffinity[i]):
                    continue
                if int(model.geom_type[i]) == mujoco.mjtGeom.mjGEOM_PLANE:
                    continue
                if scene.ngeom >= scene.maxgeom:
                    break
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    model.geom_type[i],
                    model.geom_size[i] * 1.002,
                    data.geom_xpos[i],
                    data.geom_xmat[i],
                    np.array([0.0, 0.85, 0.9, 0.45]),
                )
                geom.dataid = (
                    2 * model.geom_dataid[i] + 1
                    if int(model.geom_type[i]) == mujoco.mjtGeom.mjGEOM_MESH
                    else model.geom_dataid[i]
                )
                geom.objtype = mujoco.mjtObj.mjOBJ_GEOM
                geom.objid = i
                geom.category = mujoco.mjtCatBit.mjCAT_DECOR
                scene.ngeom += 1
