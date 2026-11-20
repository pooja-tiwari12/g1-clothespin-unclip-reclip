# Clothespin task 1: unclip, open/close in the air, put down

The seated G1 (left Inspire hand) unclips a spring clothespin from the edge of a sheet of paper, grips it by the
tail tips, lifts it, opens and closes it in the air at the times measured from my video `pin_01`, puts it back on
the paper and lets go. The robot model comes from `../sim/` (shared with the cup project).

Run from the repository root (see the main README for setup):
```bash
python sim/build_scene.py                              # robot model (once)
python clothespin/sim/build_pin_scene.py               # the scene
python clothespin/scripts/unclip_press.py --full       # the task, prints each step + FULL TASK: PASS
mjpython clothespin/scripts/unclip_press.py --full --viewer --slow 4   # watch it
```
Options of `unclip_press.py`: `--open_mm 6` (opening in the air), `--hold lock|force`, `--index_only` and
`--equal_force` (the older press methods, for comparison), `--shape --across --along --yaw --thumb_yaw` (hand
placement), `--best` (best result of the opposition search).

## Files
| file | what it does |
|---|---|
| `data/clothespin.yaml` | both pins (my mini pin, measured; a standard pin, estimated) and the setup; `use:` picks the pin |
| `sim/pin_model.py` | the pin: two legs on a preloaded torsion-spring hinge, flat jaw face |
| `sim/build_pin_scene.py` | seated G1, chair, 76 cm cabinet, letter paper overhanging 2.5 cm, pin clipped on its edge; optional fingertip pads (`FINGER_PADS`) |
| `sim/view_scene.py` | open the scene in the MuJoCo viewer, robot seated (`mjpython`) |
| `scripts/measure_pin_opening.py` | from my video: when the pin is open, shut and lifted (red pixels) → `results/pin_01_events.json` |
| `scripts/spring_moment.py` | spring torque and the finger force needed at each distance from the spring |
| `scripts/pin_alone_test.py` | the pin with no robot: stays clipped, opens when squeezed, snaps back |
| `scripts/pin_pinch_test.py` | first feasibility check: can the hand pinch the pin open? |
| `scripts/unclip_test.py`, `unclip_press.py` | the robot: reach, IK, thumb then index, press by moments, slide off, grip, lift, open/close, put down |
| `scripts/oppose_search.py` | fast geometry search: hand placements where thumb and index land opposite each other |
| `scripts/unclip_search.py` | physics search: every candidate runs the real sequence |
| `scripts/make_video.py` | the multi-view video next to my recording |
| `scripts/hold_squeeze_test.py` | an early approach (pin spawned in the hand) that failed; kept for reference |
| `results/` | events from my video, the opening plot, search results |
