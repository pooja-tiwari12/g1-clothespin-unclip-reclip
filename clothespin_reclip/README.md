# Clothespin: unclip, hold, clip it back

The seated G1 (left Inspire hand) unclips the standard clothespin from the paper edge, holds it, then clips
it back onto the paper and lets go. Builds on `../clothespin/` (scene, pin model, unclip, controllers),
which this folder imports and does not change (one hook added there: `Press.stop_after` + `Press.state`).

```bash
source .venv/bin/activate                                         # from the repository root
python sim/build_scene.py                                         # robot model (once)
python clothespin/sim/build_pin_scene.py                          # the scene (same as clothespin/)
python clothespin_reclip/scripts/reclip.py                        # run it, prints each step + PASS/FAIL
mjpython clothespin_reclip/scripts/reclip.py --viewer --slow 3    # then watch it in the MuJoCo viewer
python clothespin_reclip/scripts/make_video.py                    # results/reclip.mp4 (multi-view)
```

## Steps
| step | what the robot does |
|---|---|
| part 1 | unclip with equal and opposite moments, slide it off the paper, firm grip on the tail tips (`clothespin/scripts/unclip_press.py --full`) |
| hold | keep the grip, jaws shut (finger commands locked) |
| open | squeeze to 5 mm: moment = spring torque at 5 mm / finger distance + a correction from the measured opening; equal forces |
| align, line_up | bring the open jaws in front of the paper edge; **feedback** every 20 ms on where the jaw gap really is |
| insert | slide the open jaws 15 mm onto the paper (the reverse of the pull), same feedback |
| close | lower the target opening to 0 slowly; once the jaws are on the paper, freeze the fingers: the spring clamps |
| release, drop, retreat | lift the index off; lower the whole hand so the thumb drops away; then straighten the thumb and move back |
| watch | 1.5 s: the pin stays clipped on its own |

Result: jaws clamped at 1.0 mm on the 1 mm paper, 15.6 mm onto it, flat, hand off -> `CLIPPED BACK: PASS`.

## What went wrong on the way (and the fix)
- **Jaws hit the paper edge** (stopped 4 mm on): the arm lags its target by 2-4 mm and the gap around the paper
  is only ~2 mm each side -> aim with feedback (measure the jaw gap, nudge the hand) and pause to line up.
- **Jaws re-opened to 8 mm during the close**: force control kept curling the index as its measured force
  (one finger segment) dropped, so another segment pushed the tail -> freeze the fingers once clamped.
- **Thumb flicked the pin on the release**: turning the thumb's joint swept its tip up under the tail ->
  release in order: index up, whole hand down, then the thumb.
