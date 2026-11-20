"""Multi-view video: unclip -> hold -> clip it back onto the paper.

    python clothespin_reclip/scripts/make_video.py
    -> clothespin_reclip/results/reclip.mp4   (plays in QuickTime)

Layout (1920 x 540): big close-up of the pin on the left (960 x 540); on the right a 2x2 grid
(like your phone, front, side, from the paper's side). Step name and jaw opening are written on it.
No phone video here: there is no recording of you clipping it back (record one and it can go in).
"""

import subprocess
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

TASK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TASK / "scripts"))
import reclip as RC  # noqa: E402

OUT = TASK / "results" / "reclip.mp4"
FPS = 50                                        # the log is every 20 ms
STEP_NAMES = {
    "rest": "resting", "reach": "reaching out", "rise": "coming up under the pin", "insert": "sliding onto the paper",
    "thumb": "thumb up until it touches", "index": "index down until it touches", "settle": "waiting until still",
    "press": "press: equal and opposite moments", "pull": "sliding it off the paper", "grip": "firm grip on the tail tips",
    "hold": "holding it", "open": "squeeze open (5 mm)", "align": "line the jaws up with the paper edge",
    "line_up": "fine aim (feedback)", "close": "close slowly: the spring clamps the paper",
    "release": "lift the index off", "drop": "lower the hand: thumb drops away", "retreat": "move the hand away",
    "watch": "clipped back",
}


def label(img, text, y=24, scale=0.6):
    """White text on a dark box (a thick outline under thin text doubles the letters in OpenCV)."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (4, y - h - 6), (12 + w, y + base + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    print("running the robot (unclip, hold, clip back)...")
    u = RC.Reclip()
    log = u.run_all()
    ok = RC.report(u, log)
    m, d = u.m, u.d
    big = mujoco.Renderer(m, 540, 960)
    small = mujoco.Renderer(m, 270, 480)
    paper_cam = mujoco.MjvCamera()                    # a free camera looking back at the robot over the paper
    paper_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    paper_cam.lookat[:] = u.state["p_pin0"]
    paper_cam.distance, paper_cam.azimuth, paper_cam.elevation = 0.30, 160, -25
    OUT.parent.mkdir(exist_ok=True)
    raw = OUT.with_name(OUT.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1920, 540))
    for i, r in enumerate(log):
        d.qpos[:] = r["qpos"]
        mujoco.mj_forward(m, d)
        big.update_scene(d, "pin_close")
        left = np.ascontiguousarray(big.render()[:, :, ::-1])
        label(left, "close-up", scale=0.8)
        part = "part 2 (clothespin_reclip)" if r["t"] >= u.reclip_from else "part 1 (clothespin unclip)"
        label(left, f"t={r['t']:4.1f}s  {STEP_NAMES.get(r['phase'], r['phase'])}", y=490, scale=0.8)
        label(left, f"{part}   jaws {r['jaws']:4.1f} mm   index {r['f_index']:.1f} N  thumb {r['f_thumb']:.1f} N",
              y=528, scale=0.6)
        tiles = []
        for cam, name in (("like_your_phone", "like your phone"), ("front", "front"), ("side", "side"),
                          (paper_cam, "from the paper side")):
            small.update_scene(d, cam)
            tiles.append(label(np.ascontiguousarray(small.render()[:, :, ::-1]), name))
        grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        writer.write(np.hstack([left, grid]))
        if i % 200 == 0:
            print(f"  frame {i}/{len(log)}", flush=True)
    writer.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", str(OUT)], check=True)
    raw.unlink()
    print(f"saved {OUT.relative_to(TASK.parent)}  ({len(log) / FPS:.1f} s){'' if ok else '  (the run did NOT pass)'}")


if __name__ == "__main__":
    main()
