"""Three live camera tiles inside the native MuJoCo viewer's right side."""

import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw


class CameraPanel:
    """Render three live camera tiles alongside the main viewer."""

    def __init__(self, env, viewer, hz=10):
        """Initialize the object and its owned state.
        @param env Simulation environment.
        @param viewer Native MuJoCo viewer.
        @param hz Tile refresh rate, frames per second.
        """
        self.env, self.viewer = env, viewer
        self.period = 1 / hz
        self.next_frame = 0.0
        self.frames = None
        self.overlay = None
        # With both native panels hidden, no UI draw initializes 2D GL state.
        # MuJoCo draws user text before user images; this initializes that state
        # (including disabling depth testing) on the viewer's render thread.
        self.viewer.set_texts(
            (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, "LIVE CAMERAS", "")
        )

    def update(self, frames=None):
        """Refresh the viewer overlay.
        @param frames The latest camera images by name, or None to keep the last ones.
        """
        now = time.monotonic()
        if frames is not None:
            self.frames = frames
        if now < self.next_frame:
            if self.overlay is not None:
                self.viewer.set_images(self.overlay)
            return
        self.next_frame = now + self.period
        viewport = self.viewer.viewport
        if viewport is None or viewport.width < 100 or viewport.height < 100:
            return
        if frames is None:
            self.frames = self.env.images()
        # Fit all three 16:9 tiles, labels and padding even when the window shrinks.
        width = max(80, min(360, int(viewport.width * 0.30), int((viewport.height - 92) * 16 / 27)))
        height = round(width * 9 / 16)
        label, gap = 22, 6
        panel = Image.new("RGB", (width + 12, 3 * (height + label) + 4 * gap), "#181b21")
        draw = ImageDraw.Draw(panel)
        for index, (name, title) in enumerate(
            (("top", "FRONT / TOP"), ("left", "LEFT WRIST"), ("right", "RIGHT WRIST"))
        ):
            y = gap + index * (height + label + gap)
            draw.text((10, y + 4), title, fill="#f0f1f2")
            rgb = Image.fromarray(self.frames[name]).resize((width, height), Image.Resampling.BILINEAR)
            panel.paste(rgb, (6, y + label))
        rect = mujoco.MjrRect(
            viewport.left + viewport.width - panel.width - 8,
            viewport.bottom + viewport.height - panel.height - 8,
            panel.width,
            panel.height,
        )
        self.overlay = (rect, np.asarray(panel).copy())
        self.viewer.set_images(self.overlay)
