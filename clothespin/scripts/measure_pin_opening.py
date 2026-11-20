"""Measure how far the clothespin is open in every frame of your video, from the red pixels.

    python clothespin/scripts/measure_pin_opening.py                       # pin_01
    python clothespin/scripts/measure_pin_opening.py --video clothespin/data/raw_videos/pin_02.mp4

Why: squeezing the pin open changes your thumb-index gap by only ~7 mm, which is close to the
hand tracker's noise. The pin itself, though, is bright red and changes shape a lot:
    closed -> a solid bar            open -> a "V" with an empty triangle between the jaws
We measure that with SOLIDITY = red area / area of its convex hull (rubber band around it):
    ~0.9 when closed, clearly lower when the V opens.
Also the pin's centre (where it is) and its size in pixels (is it near / far / hidden).

Output: clothespin/results/<clip>_opening.png (plot) and <clip>_opening.npz (per-frame numbers)
        and <clip>_opening_check.jpg (frames with the detected pin outlined, to check by eye)
"""

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TASK = Path(__file__).resolve().parent.parent


def red_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # red wraps around hue 0/180. Saturation >= 180 separates the painted pin (measured 190-246) from
    # skin (155-165): with 150 the detector latched onto the wrist in about half the frames.
    m = cv2.inRange(hsv, (0, 180, 70), (9, 255, 255)) | cv2.inRange(hsv, (168, 180, 70), (180, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def measure(frame, last_xy=None):
    """(solidity, area_px, centre_xy, contour) of the pin, or None.
    Picks the red blob nearest to where the pin was in the previous frame (it can't jump far in 1/30 s)."""
    cnts, _ = cv2.findContours(red_mask(frame), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= 80]
    if not cnts:
        return None
    if last_xy is None:
        c = max(cnts, key=cv2.contourArea)
    else:
        def dist(c):
            M = cv2.moments(c)
            return np.hypot(M["m10"] / M["m00"] - last_xy[0], M["m01"] / M["m00"] - last_xy[1])
        c = min(cnts, key=dist)
        if dist(c) > 150:                               # nothing near the last position: pin hidden
            return None
    area = cv2.contourArea(c)
    if area < 80:                                       # too small: pin hidden or not there
        return None
    hull = cv2.contourArea(cv2.convexHull(c))
    M = cv2.moments(c)
    return area / hull, area, (M["m10"] / M["m00"], M["m01"] / M["m00"]), c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(TASK / "data/raw_videos/pin_01.mp4"))
    args = ap.parse_args()
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    check_at = set(np.linspace(0, n - 1, 12).astype(int))
    sol, area, cx, cy, checks = [], [], [], [], []
    i = 0
    last_xy = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        r = measure(f, last_xy)
        if r is not None:
            last_xy = r[2]
        if r is None:
            sol.append(np.nan); area.append(np.nan); cx.append(np.nan); cy.append(np.nan)
        else:
            sol.append(r[0]); area.append(r[1]); cx.append(r[2][0]); cy.append(r[2][1])
        if i in check_at:
            vis = f.copy()
            if r is not None:
                cv2.drawContours(vis, [cv2.convexHull(r[3])], -1, (255, 255, 0), 2)
                cv2.drawContours(vis, [r[3]], -1, (0, 255, 0), 2)
            label = "no pin" if r is None else f"solidity {r[0]:.2f}"
            cv2.putText(vis, f"t={i / fps:.1f}s {label}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            checks.append(cv2.resize(vis, (480, 270)))
        i += 1
    t = np.arange(i) / fps
    sol, area, cx, cy = map(np.array, (sol, area, cx, cy))
    k = max(1, int(0.2 * fps))                         # 0.2 s smoothing
    sm = np.convolve(np.nan_to_num(sol, nan=np.nanmedian(sol)), np.ones(k) / k, mode="same")

    stem = Path(args.video).stem
    out = TASK / "results"
    out.mkdir(exist_ok=True)
    np.savez(out / f"{stem}_opening.npz", t=t, solidity=sol, solidity_smooth=sm, area_px=area, cx=cx, cy=cy)
    cv2.imwrite(str(out / f"{stem}_opening_check.jpg"), np.vstack([np.hstack(checks[r:r + 4]) for r in range(0, 12, 4)]))

    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(t, sol, ".", ms=2, alpha=0.4, color="gray"); ax[0].plot(t, sm, "r")
    ax[0].set_ylabel("solidity\n(low = V open)"); ax[0].set_title(f"{stem}: clothespin opening from red pixels")
    ax[1].plot(t, area, "k"); ax[1].set_ylabel("red area (px)")
    ax[2].plot(t, cx, label="x"); ax[2].plot(t, cy, label="y"); ax[2].set_ylabel("pin centre (px)")
    ax[2].legend(); ax[2].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out / f"{stem}_opening.png", dpi=110)
    # events: when the pin is OPEN (solidity clearly below the closed level), and when it is lifted
    base = np.nanmedian(sm)
    is_open = sm < base - 0.12
    edges = np.flatnonzero(np.diff(np.r_[0, is_open.astype(int), 0])).reshape(-1, 2)
    opens = [(round(float(t[a]), 2), round(float(t[b - 1]), 2)) for a, b in edges if t[b - 1] - t[a] >= 0.15]
    rest_y = np.nanmedian(cy[: int(3 * fps)])                       # pin height on screen at the start
    lifted = np.flatnonzero(np.nan_to_num(cy, nan=rest_y) < rest_y - 60)   # >60 px higher than at rest
    events = {"clip": stem, "fps": fps, "closed_solidity": round(float(base), 3), "open": opens,
              "lifted": [round(float(t[lifted[0]]), 2), round(float(t[lifted[-1]]), 2)] if len(lifted) else None}
    import json
    (out / f"{stem}_events.json").write_text(json.dumps(events, indent=1))
    print("pin open:", ", ".join(f"{a}-{b} s" for a, b in opens), "| lifted:", events["lifted"])
    seen = np.isfinite(sol).mean() * 100
    print(f"{stem}: pin found in {seen:.0f}% of frames; solidity median {np.nanmedian(sol):.2f}, "
          f"min (most open) {np.nanmin(sm):.2f} at t={t[np.nanargmin(sm)]:.1f}s")
    print(f"saved {out.relative_to(TASK.parent)}/{stem}_opening.png, _opening_check.jpg, _opening.npz")


if __name__ == "__main__":
    main()
