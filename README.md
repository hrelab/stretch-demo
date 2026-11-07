# stretch-demo

The lab's showcase demo. One command connects the x-IMU3s, calibrates the
visitor, and opens a simulated Stretch 4 doing a task. Each attempt is scored.
No Redis or second terminal needed.

## Run it

```bash
cd 2026-stretch-sim/stretch-demo
uv run demo --env feeding --controller hands_vel             # real IMUs
uv run demo --env pickup  --controller ee_mode2 --imu fake   # practice without IMUs
```

| option | values |
| --- | --- |
| `--env` | `feeding`, `pickup`, `door` |
| `--controller` | `hands_vel`, `hands_v2`, `ee_mode2`, `head` |
| `--imu fake` | two still pretend IMUs, for rehearsing |
| `--imu-serial S0 S1` | which IMUs (default `6A3409FC 6A32706F`, IMU 0 first) |
| `--imu-ip IP0 IP1` | connect by IP instead of serial |
| `--who LABEL` | visitor label saved in the results file |
| `--mobility FILE` / `--shoulder left` | `head` only: per-user ranges, e.g. `profiles/p00.json` |
| `--ee-profile FILE` | `ee_mode2` only, e.g. `profiles/imu_ee_mode2_faster.json` (25% faster) |

**Operator commands** (type in the terminal, then Enter):

| key | does |
| --- | --- |
| `c` | recalibrate (visitor holds still ~2 s) |
| `r` | reset the task / play again |
| `n` | next visitor (full reset + recalibrate) |
| `g` | open/close the gripper |
| `q` | quit |

## Where the IMUs go

IMU 0 = first serial (`6A3409FC`), IMU 1 = second (`6A32706F`).

| controller | IMU 0 | IMU 1 |
| --- | --- | --- |
| `hands_vel`, `hands_v2`, `ee_mode2` | back of left hand | back of right hand |
| `head` | head | top of the shoulder (right by default) |

The IMUs are found by serial number on whatever network the laptop is on. If
that network blocks broadcasts (some phone hotspots do), use `--imu-ip`.

## Controllers

None of them have modes: the same motion always does the same thing.

**`hands_vel`** – the left hand sets the wrist angle.
- Left hand tilt/twist/turn → wrist pitch/roll/yaw (1:1).
- Left hand turned past 40° → base slides that way.
- Right hand tilt → lift up/down. Right forearm twist → arm in/out.
- Right hand turned ±25° (held level) → gripper close/open.

**`hands_v2`** – every tilt moves something at a speed.
- Left hand tilt/twist/turn → wrist pitch/roll/yaw speed. Turned past 40° → base slides.
- Right hand tilt → lift. Right forearm twist → arm.
- Quick right-hand flick → gripper.

**`ee_mode2`** – the hands move the gripper itself; the robot works out the joints.
- Left hand tilt → up/down (the wrist tips 10° first).
- Left hand twist → turn the gripper.
- Right hand tilt → forward/back. Right hand twist → sideways.
- Space or a presenter clicker → gripper (the viewer window must be focused).

**`head`** – for users with little hand movement.
- Nod → lift down/up.
- Tilt head toward a shoulder → base slides that way.
- Turn head → wrist turns.
- Shoulder forward/back → arm out/in.
- Double shrug → gripper.

## Tasks

| `--env` | goal |
| --- | --- |
| `feeding` | A 5-bite game. Scoop from the plate, bring it to the mouth (green box): **+100**. Touching the face: **−50**. Score, time and initials go on a top-5 leaderboard. |
| `pickup` | Pick the bottle up off the floor and set it on the side table's green zone. Tip: go down first, then reach around the bottle. |
| `door` | Grab the handle and pull the door open (pull back while sliding right). |

Results are saved to `runs/<env>_<controller>_<time>.csv`; feeding scores
to `runs/leaderboard.json` (delete it to clear the board).

**On screen:** task and score (top left), leaderboard (top right), controller
and IMU status (bottom left). A **STUCK** line means a joint can't go further:
move it back the other way.

## Before the event

1. `uv run pytest` – all tests should pass (~25 s).
2. Rehearse each setup with `--imu fake`.
3. Power on the IMUs, run the demo, and check each IMU shows ~400 Hz.
4. Have the visitor sit relaxed and still while it calibrates.

**Not yet tried with real IMUs:** `ee_mode2`, the `head` tilt-to-slide
direction (flip `base_roll_sign` in a `--mobility` file if backwards), and the
hands controllers' wrist-pitch direction.

## Troubleshooting

| problem | fix |
| --- | --- |
| `not found on the network` | IMU off, on another network, or wrong serial (the message lists the IMUs it saw). Try `--imu-ip`. |
| `could not connect` | close the x-IMU3 GUI or anything else using the IMU |
| stuck on "HOLD STILL" | visitor must stay still for 2 s; wait or type `c` |
| robot drifts by itself | type `c` to recalibrate |
| "NO DATA from ..." | an IMU dropped out; the robot holds until it's back |
| a direction is backwards | hands: signs at the top of `controllers/hands_*.py`; head: `--mobility` file |

## For developers

- **Setup:** Python 3.14 + [uv](https://docs.astral.sh/uv/). This is part of the
  `2026-stretch-sim` uv workspace: sync with `uv sync --all-packages` from
  `2026-stretch-sim/` (a plain `uv sync` here removes the other projects' packages).
- **Code complexity:** no function may score above 10 (complexipy). A test
  enforces it; exceptions and past rewrites are in `COMPLEXITY.md`.
- **`ee_mode2` source:** copied from mujoco-sim's `ee_controller` branch at commit
  `85f57de` (changes for the demo are listed at the top of `controllers/ee/mode2.py`).
  To see later updates, run in `mujoco-sim/`:
  `git diff 85f57de..hrelab/ee_controller -- src/mujoco_sim/controllers/end_effector/`
- If `ximu3` 1.9.1 misbehaves with the lab's IMUs, pin `ximu3==1.8.1` in `pyproject.toml`.
