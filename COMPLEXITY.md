# Complexity register

Same rule as mujoco-sim's `[tool.complexipy]`: **no function above a cognitive
complexity of 10**. `test_no_function_exceeds_complexity_10_unless_registered`
(in `tests/test_cli_complexity.py`) runs complexipy over `src` and `tests` and
fails on any function above 10 that isn't in the table below.

Check by hand with `uv run complexipy src tests` (from `stretch-demo/`).

## Policy

1. Over 10 means rewrite first: split it into named steps, a small state
   machine, or a lookup table.
2. A function may stay above 10 only if splitting it would make it harder to
   follow, for example a strictly ordered hardware protocol that should read top to
   bottom. It then gets a row below: its score, why it must stay, and what
   was tried.
3. Rows are removed as soon as the function drops back to 10 or below.

## Functions allowed above 10

| Function (`path::name`) | Score | Why it must stay above 10 | What was tried |
| --- | --- | --- | --- |
| *(none)* | | | |

## Rewritten while porting from mujoco-sim

All of these scored above 10 in mujoco-sim. The demo's versions are all at 10
or below:

| mujoco-sim function | Was | Demo version |
| --- | --- | --- |
| `head_shoulder.HeadShoulderRetargeter.step` | 29 | `controllers/head.py`: no modes, so the lockout and lift restore are gone; `control` / `_angles` / `_level` / `_base_level` / `_gestures` |
| `head_shoulder.ShrugGestures.update` | 29 | `tracking/gestures.py` `DoubleShrug`: one method per stage (`_expire_first`, `_track_burst`, `_quick_shrug`, `_end_burst`); the long hold isn't needed |
| `sim/harness.SimHarness.run` | 25 | `sim/harness.py`: `_step_control`, `_step_task`, `_handle`, `_draw_overlay`, `_pace` |
| `joint_config.discrete` | 18 | `tracking/gestures.py` `FlickDetector` (`update`, `_candidate_sign`, `_ready`) |
| `bio_m2_v2_controller.get_ctrl` | 16 | `controllers/hands_v2.py` `control` + `_gripper_flick`; read/hold logic is shared in `controllers/base.py` |
| `imu.SettledCalibrator.update` | 13 | `tracking/imu_tracker.py` `ImuTracker` + `NeutralCalibration` (non-blocking) |
| `head_shoulder.MobilitySettings.validate` | 13 | `tracking/mobility.py`: `_validate_motions` / `_validate_fractions` / `_validate_gestures` |
| `sim/harness.place_joints_at` | 12 | `sim/harness.py`: tendon case in `_place_tendon` |
| `joint_config.ImuTracker.update` | 11 | `tracking/imu_tracker.py`: velocity integration dropped; gyro peak in `_update_gyro` |

| `end_effector/imu_ee_postwrist.ImuEePostwristController.get_ctrl` | 29 | `controllers/ee/mode2.py`: BASE mode and the lockout aren't used by Mode 2; IMU reading/holding is the shared `controllers/base.py`; the rest is `control` / `_task_twist` / `_follow` |
| `end_effector/imu_ee_prewrist_controller.ImuEePrewristController._read_rotations` | 26 | replaced by the demo's `ImuBus` + `ImuTracker` (no Redis payload parsing) |
| `end_effector/imu_ee_prewrist_controller.ImuEePrewristController.get_ctrl` | 22 | not needed: Mode 2 only builds on the post-wrist controller's arm path |
| `end_effector/imu_ee_prewrist_controller.EeSettings.__post_init__` | 20 | `controllers/ee/settings.py`: one validator per rule (`_triple`, `_check_tilts`, `_check_signs`, `_check_speeds`, `_check_reach`) |
| `end_effector/startup_pose.scene_camera_target` | 15 | not ported: each task's own start pose and camera are used |
| `end_effector/imu_ee_mode2_controller.ImuEeMode2Controller._task_twist` | 13 | `controllers/ee/mode2.py`: `PitchStage.step` + `phase_label` + `level_camera_axes` |

These weren't ported (out of scope for the demo): `viz/feedback_layout.validate_layout` (46),
`viz/feedback_gui.FeedbackCanvas.paintEvent` (30), and the `rom_calibration` functions.
