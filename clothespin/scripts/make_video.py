"""Multi-view video of the full clothespin task, next to YOUR video pin_01.

    python clothespin/scripts/make_video.py
    -> clothespin/results/pin_01_full_task.mp4   (plays in QuickTime)

Layout (1920 x 540): your video on the left; on the right a 2x2 grid of robot views
(like your phone, close-up of the pin, front, side), with the step name and the jaw opening.

Timing: the robot runs the full task once (same settings as `unclip_press.py --full`); its pose is
logged every 20 ms and replayed here at real speed (50 frames per second). Your video is lined up
step by step using the moments measured from it (results/pin_01_events.json): you pressed at 9.5 s,
slid it off by 11.0 s, lifted at 14.6 s, opened it in the air 20.0-22.6 s, let go at 28.9 s. Between
those moments your video plays a little faster or slower so both show the same step together; the
open/close part plays at exactly your speed (the robot follows your timing there).
"""

import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

TASK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TASK / "scripts"))
import unclip_press as UP  # noqa: E402

OUT = TASK / "results" / "pin_01_full_task.mp4"
VIDEO = TASK / "data/raw_videos/pin_01.mp4"
FPS = 50                                        # the robot log is every 20 ms
VIEWS = [("like_your_phone", "like your phone"), ("pin_close", "close-up"), ("front", "front"), ("side", "side")]
STEP_NAMES = {
    "rest": "resting", "reach": "reaching out", "rise": "coming up under the pin", "insert": "fingers around the tails",
    "thumb": "thumb up until it touches", "index": "index down until it touches", "settle": "waiting until still",
    "press": "press: equal and opposite moments", "pull": "sliding it off the paper", "grip": "firm grip on the tail tips",
    "lift": "lift", "cycle": "open / close like your video", "carry": "carry back over the paper",
    "lower": "lower onto the paper", "release": "let go", "retreat": "move the hand away", "rest_after": "done",
}


def run_robot():
    u = UP.Press(0.622, 0.35, along=0.0235, across=0.005, yaw_deg=-36.0, thumb_yaw=1.2)   # = --full
    u.slowmo = 1
    with contextlib.redirect_stdout(io.StringIO()):
        log = u.run()
    return u, log


def video_time_map(log):
    """Robot time -> your video time, piecewise linear through matching moments."""
    ev = json.loads((TASK / "results/pin_01_events.json").read_text())
    start = {}
    for r in log:
        start.setdefault(r["phase"], r["t"])
    t_end = log[-1]["t"]
    unclip_open, air_opens = ev["open"][0], [o for o in ev["open"] if 19.5 <= o[0] <= 23.0]
    anchors = [(0.0, 0.0)]
    for phase, vt in (("press", unclip_open[0]), ("pull", unclip_open[1]), ("lift", ev["lifted"][0]),
                      ("cycle", 19.5), ("carry", 23.0), ("release", ev["open"][-1][0])):
        if phase in start:
            anchors.append((start[phase], vt))
    anchors.append((t_end, ev["lifted"][1] + 1.5))
    rt, vt = np.array(anchors).T
    assert np.all(np.diff(rt) > 0) and np.all(np.diff(vt) > 0), anchors
    print("robot time -> your video time:", [(round(a, 1), round(b, 1)) for a, b in anchors])
    return lambda t: float(np.interp(t, rt, vt))


def label(img, text, y=24, scale=0.6):
    """White text on a dark box. (A thick black outline under thin white text doubled the letters: OpenCV
    spaces thicker text wider, so the two drift apart along the line.)"""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (4, y - h - 6), (12 + w, y + base + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


class Phone:
    """Your video, read forward frame by frame (seeking is slow and inaccurate)."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.t, self.img = -1.0, np.zeros((540, 960, 3), np.uint8)

    def at(self, t_s):
        while self.t + 1.0 / self.fps <= t_s:
            ok, f = self.cap.read()
            if not ok:
                break
            self.t += 1.0 / self.fps
            self.img = cv2.resize(f, (960, 540))
        return self.img.copy()


def main():
    print("running the robot (about 20 s of simulation)...")
    u, log = run_robot()
    m, d = u.m, u.d
    to_video = video_time_map(log)
    renderer = mujoco.Renderer(m, 270, 480)
    phone = Phone(VIDEO)
    OUT.parent.mkdir(exist_ok=True)
    raw = OUT.with_name(OUT.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1920, 540))
    for i, r in enumerate(log):
        d.qpos[:] = r["qpos"]
        mujoco.mj_forward(m, d)
        tiles = []
        for cam, name in VIEWS:
            renderer.update_scene(d, cam)
            tiles.append(label(np.ascontiguousarray(renderer.render()[:, :, ::-1]), name))
        label(tiles[1], f"jaws {r['jaws']:4.1f} mm   index {r['f_index']:.1f} N  thumb {r['f_thumb']:.1f} N",
              y=260, scale=0.5)
        grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        vt = to_video(r["t"])
        left = label(phone.at(vt), f"your video  t={vt:4.1f}s", scale=0.8)
        label(left, f"robot t={r['t']:4.1f}s   {STEP_NAMES.get(r['phase'], r['phase'])}", y=530, scale=0.8)
        writer.write(np.hstack([left, grid]))
        if i % 100 == 0:
            print(f"  frame {i}/{len(log)}", flush=True)
    writer.release()
    # re-encode to H.264 so it plays in QuickTime / browsers / GitHub
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", str(OUT)], check=True)
    raw.unlink()
    print(f"saved {OUT.relative_to(TASK.parent)}  ({len(log) / FPS:.1f} s)")


if __name__ == "__main__":
    main()
