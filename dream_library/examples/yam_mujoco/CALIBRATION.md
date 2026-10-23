# Standard YAM reference calibration

The cameras and arm bases follow the YAM robot configuration of the pinned simulation reference: ARISE-Initiative/yamlab at
`ec0455d2b4ce35f21fc126418ea5e74ac567133d`,
https://github.com/ARISE-Initiative/yamlab/blob/ec0455d2b4ce35f21fc126418ea5e74ac567133d/yamlab/configs/robot/yam.yaml
`assets/sim-eval-yam.json` holds the values the simulator reads.

This is STANDARD YAM. A YAM fitted with a third-party parallel gripper and fisheye cameras is a
different embodiment with its own calibration, not this linear YAM gripper.

Applied:
- Per-camera fx, fy, cx, cy, instead of an approximate 58 degree FOV.
- Center camera world pose; wrist extrinsics relative to link_6, not TCP.
- Normalized source OpenGL wxyz quaternions (same camera convention as MuJoCo).
- 640x480 rendering, then rows 60:420 for 640x360 output, matching
  the reference's camera resize for standard YAM.
- World frame conversion: reference (x,y,z) -> (y,0.4925-x,z).
  This maps both reference arm mounts exactly to the example's mounts.
- Center stand/housing follows the new camera position and orientation.
- The center camera is then deliberately moved off the calibration: `TOP_CAMERA_LOWER_M` (0.15 m) lower and `TOP_CAMERA_TILT_UP_DEG` (5 degrees) pitched up, in `yam_env.py`, for a closer view of the table. Its images do not match the calibrated station's; set both to 0 to restore it. The wrist cameras are as calibrated.

Intrinsic convention: MuJoCo principalpixel is an image-plane offset. The example uses
(320-cx, cy-240), since rendered RGB pixels have a top-left origin. After cropping,
output cy is source cy-60; focal lengths and cx do not change.

Checked the reference's USD flange FK (yam_kinematics.json) against Dream
URDF flange FK at zero and a nonzero pose: frame differences are micrometres /
about 1e-5 in rotation entries (rounded URDF transforms). No axis flip is needed
for the link_6 wrist camera mounts.

Remaining differences, deliberately NOT overwritten by the camera import:
- Table top: reference .7517m; demo .7600m.
- USD reference joint maxima: J1 3.05433, J2/J3 3.66519 radians;
  demo Dream URDF: J1 3.13, J2 3.65, J3 3.13.
- Reference TCP extension .14256m with a small tool roll; demo .1347m and
  the stock linear4310 gripper mount from the URDF.
- Gripper slide signs differ between USD and the stock STL hierarchy; normalized
  opening remains 0=closed, 1=open.
- Demo collision geometry, dynamics, home pose and task objects are not a full
  replica of the simulation reference. Camera values are the pinned reference's calibration,
  not a new measurement of the user's physical robot.

Base clearance override: both arm base origins are raised 12 mm
to z=.772 m because the visual base extends 10 mm below its origin. The fixed
center camera stays calibrated in world space; wrist mounts follow the raised arms.
