# stretch-demo

The lab's showcase demo, run from **one terminal**. It connects to the x-IMU3s,
streams their data, calibrates the visitor's neutral pose and opens the MuJoCo
viewer on a Stretch 4. The robot does a task that people with
high-level cervical SCI want help with, and each attempt is scored.

```
x-IMU3s --(ximu3 SDK)--> in-process IMU bus --> controller --> MuJoCo viewer
                                                     task scoring --> runs/*.csv
```

It replaces four processes: ximu-api, port-listener, Redis and `uv run sim`.
Everything it needs is in this folder (code ported from mujoco-sim, ximu-api and
port-listener). No Redis, no ZMQ, no second terminal.

## Run it

```bash
cd 2026-stretch-sim/stretch-demo
uv run demo --env feeding --controller hands_vel                 # real IMUs
uv run demo --env door    --controller head --mobility profiles/p00.json
uv run demo --env pickup  --controller hands_v2 --imu fake       # rehearse, no hardware
```

| option | values |
| --- | --- |
| `--env` | `feeding`, `pickup`, `door` |
| `--controller` | `hands_vel`, `hands_v2`, `head` |
| `--imu` | `ximu` (default: the real x-IMU3s), `fake` (two still synthetic IMUs) |
| `--imu-ip IP0 IP1` | default `192.168.0.161 192.168.0.102` (ximu-api's). IMU 0 first |
| `--who LABEL` | written into the CSV |
| `--mobility FILE`, `--shoulder left/right` | `head` only: per-user ranges (mujoco-sim's `--calibrate-rom` profiles load as-is) |

While it runs, the operator types into the terminal (letter, then Enter):

| key | does |
| --- | --- |
| `c` | recalibrate neutral (the visitor holds still ~2 s) |
| `r` | reset the task (bottle back on the floor, door closed, bite restarted) |
| `n` | next visitor: robot and task reset, scores cleared, recalibrate |
| `q` | quit (closing the viewer also quits) |

The viewer shows the task, what to do next, live time and scores (top left),
plus the controller's state and each IMU's sample rate (bottom left). If an IMU
drops out, the robot holds still and the overlay names the silent IMU.

## Where the IMUs go

IMU 0 is the first `--imu-ip`, IMU 1 the second.

| controller | IMU 0 | IMU 1 |
| --- | --- | --- |
| `hands_vel`, `hands_v2` | back of the left hand (wrist) | back of the right hand (lift/arm/gripper) |
| `head` | head | top of the shoulder (acromion), right by default |

## Controllers

None of the three switch modes. The same motion always does the same thing.

**`hands_vel`**: mujoco-sim's imu_hands_vel mapping. Its wrist-pitch sign is flipped, because the Stretch 4's wrist pitch is physically reversed from the Stretch 3 it was tuned on.
- Left hand roll/pitch/yaw sets wrist roll/pitch/yaw 1:1, measured from the task's start pose.
- Turning the left hand past 40° also strafes the base that way.
- Right hand pitch sets the lift speed; forearm twist sets the arm extension speed.
- Right hand yaw past ±25° latches the gripper closed/open.

**`hands_v2`**: mujoco-sim's bio_m2_v2 without its ARM/BASE modes, with the same wrist-pitch flip. Every axis is tilt-to-move.
- Hand 1 pitch/roll/yaw drive wrist speeds. Past 40° of yaw, the base strafes instead of turning the wrist.
- Hand 2 pitch drives lift; forearm twist drives the arm.
- A quick hand-2 yaw flick (the gesture that used to switch modes) toggles the gripper.

**`head`**: mujoco-sim's bio_m2_head without modes. It's for users with little hand movement.
- Nodding moves the lift: nod down = lower.
- Turning the head turns the wrist. Past ~60° (`base_yaw_start_frac` of the head-yaw range), it strafes the base instead.
- Tilting the head toward a shoulder pitches the wrist.
- Shoulder forward/back extends/retracts the arm.
- A double shrug toggles the gripper.
- Wrist roll stays where the task put it.

All ranges and signs are in `MobilitySettings` (`src/stretch_demo/tracking/mobility.py`, editable via a `--mobility` JSON). Its defaults include the signs the Stretch 4 needed in the bio_m2_head hardware sessions.

The arm reaches forward, so the base only ever strafes: that moves the gripper
sideways along the task. Nothing drives or turns the base.

## Environments and scoring

Each run appends one CSV row per finished trial to
`runs/<env>_<controller>_<time>.csv`, including the visitor number and
`--who`. A summary prints on exit. Clocks start when calibration ends, so
calibration time is never counted.

| `--env` | the task | each trial records |
| --- | --- | --- |
| `feeding` | The gripper holds a spoon (it stays closed and gripper gestures do nothing). Dip it in the plate, then bring the bite to the seated person's mouth (green box). Touching the face outside the box is a miss. | `time_s` plate→mouth; `error_cm`, the closest approach to the mouth centre during a 1 s hold (accuracy); `path_efficiency` (straight line ÷ path travelled) |
| `pickup` | A bottle lies on the floor in front of the robot. Pick it up, strafe over to the side table on the left, and set it down in the green zone. It reappears on the floor 2.5 s later. | `grasp_s`, `place_s`, `drops` |
| `door` | Grab the handle and pull the door open past 34°. That means retracting the arm while strafing right, following the handle's arc. The wrist starts pre-rolled 90° to fit the horizontal lever. | `touch_s` (first handle contact), `open_s`, `peak_deg` |

The scenes in `scenes/` are laid out for the Stretch 4: the arm reaches along
+x and the base strafes along +y. Tests drive the robot through every task to
confirm each is physically doable and gets scored.

## Setup

- Python 3.14 and [uv](https://docs.astral.sh/uv/). This folder is a member of the
  `2026-stretch-sim` uv workspace, so `uv run` here uses the shared `.venv`.
  `uv run` installs what's missing; to sync explicitly, run
  `uv sync --all-packages` from `2026-stretch-sim/`. A plain `uv sync` here would
  remove the other members' packages from the shared venv.
- `ximu3` comes from PyPI: version 1.9.1, with a wheel for 3.14. No more building
  x-IMU3-Software or setting `LD_LIBRARY_PATH`. The lab's ximu-api used a
  1.8.1 source build. The API calls used here are the same in both.
- The Stretch 4 model is vendored in `src/stretch_demo/sim/stretch4/`
  (generated by mujoco-sim's `scripts/generate_stretch4_model.py`).
- The viewer needs a display (X11 or Wayland). No global keyboard hook is used,
  so there's no X11-only requirement any more.

## Before the event

1. `uv run pytest`: 69 tests, a few seconds. They cover scenes, scoring, every controller, and the complexity rule.
2. Rehearse each combination you'll show with `--imu fake`.
3. Put the laptop on the IMUs' network and power them on. `uv run demo ...`
   prints each connection, and the IMUs blink when connected.
4. Check the overlay shows ~400 Hz per IMU.
5. Have the visitor sit relaxed while calibration runs.

**Not yet tried on hardware:** the head controller's new head-pitch→lift and
head-yaw→base directions (flip `head_lift_sign` / `base_yaw_sign` in a
`--mobility` JSON if either goes backwards), the hands controllers' flipped
wrist pitch, and the PyPI ximu3 1.9.1 with the lab's devices. If 1.9.1
misbehaves, pin `ximu3==1.8.1` in `pyproject.toml`.

## Troubleshooting

- **`IMU setup failed: could not connect to imu N at <ip>`**: the IMU is off,
  on another network, or has another IP. Fix `--imu-ip`, or use `--imu fake`.
- **Stuck on "HOLD STILL -- calibrating"**: every IMU must stay within 2° for
  2 s. Right after power-on the x-IMU3 heading can still be swinging, so wait,
  or type `c` to restart calibration.
- **"NO DATA from ..."**: an IMU dropped out (>0.5 s silent). The robot holds
  still until it's back.
- **A direction is backwards**: hands controllers: the sign constants at the top of
  `controllers/hands_*.py`; `head`: the signs in a `--mobility` JSON.

## Code complexity

Same rule as mujoco-sim: no function above a cognitive complexity of 10
(`[tool.complexipy]` in `pyproject.toml`). A test enforces it. Any exception
has to be justified in `COMPLEXITY.md`, which lists none today, along with
how each over-limit function from mujoco-sim was rewritten.
`uv run complexipy src tests` checks it by hand.
