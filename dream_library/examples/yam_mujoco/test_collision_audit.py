"""Regression coverage for the geometry audit, including primitive extents."""

import unittest

import numpy as np
from audit_collisions import audit, bounds
from yam_env import YamEnv


class CollisionAuditTests(unittest.TestCase):
    def test_scene_bounds_and_robot_contacts_are_reported(self):
        rows = {r["body"]: r for r in audit()["bodies"]}
        for name in ("apple", "banana", "orange", "bowl", "plate", "water_bottle"):
            self.assertLessEqual(rows[name]["bounds_difference_mm"], 1)
        self.assertEqual(rows["left_link2"]["status"], "decomposed_mesh")
        self.assertEqual(rows["right_tip_left"]["status"], "decomposed_mesh")

    def test_bottle_cylinder_has_correct_radius_and_height(self):
        env = YamEnv()
        try:
            model = env.model
            geom = model.body("water_bottle").geomadr[0]
            box = bounds(model, env.data, geom)
            np.testing.assert_allclose(box[1] - box[0], [0.058, 0.058, 0.150], atol=1e-8)
        finally:
            env.close()

    def test_arm_mesh_blocks_scene_object(self):
        import mujoco

        env = YamEnv()
        try:
            m, d = env.model, env.data
            link = m.body("left_link2").id
            colliders = [g for g in range(m.ngeom) if m.geom_bodyid[g] == link and m.geom_contype[g]]
            self.assertGreater(len(colliders), 1)
            d.joint("orange_free").qpos[:3] = d.geom_xpos[colliders[0]]
            mujoco.mj_forward(m, d)
            target = {link, m.body("orange").id}
            self.assertTrue(any({int(m.geom_bodyid[int(g)]) for g in c.geom} == target for c in d.contact))
        finally:
            env.close()
