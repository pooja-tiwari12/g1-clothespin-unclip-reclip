"""Unclip the pin, HOLD it, then CLIP IT BACK onto the paper (standard pin, seated G1, left hand).

    python clothespin_reclip/scripts/reclip.py                       # run, print what happened
    mjpython clothespin_reclip/scripts/reclip.py --viewer --slow 3   # then watch it in the MuJoCo viewer
    python clothespin_reclip/scripts/make_video.py                   # multi-view video

Part 1 (reused from clothespin/, unchanged): unclip with equal and opposite moments, slide it off the paper,
firm grip on the tail tips (`unclip_press.py --full`, stopped after the grip).
Part 2 (this file):
    hold     keep the firm grip for a moment (locked finger commands, jaws shut)
    open     squeeze open to OPEN_MM, controlled: moment = spring torque at the target opening / finger
             distance, plus a correction from the measured opening; equal forces (the pin is free)
    align    move the open jaws in front of the paper edge, with the paper's mid-plane between them
    line_up  hold there while the feedback lines the gap up with the paper (every step it measures where the
             jaw gap really is and nudges the hand: the arm lags its target by a few mm)
    insert   slide the open jaws onto the paper (BITE deep), the reverse of the pull - same feedback
    close    lower the target opening to 0 slowly: the spring clamps the paper (jaws stop at ~1 mm)
    release  lift the index off the top leg (thumb stays still)
    drop     lower the whole hand: the thumb drops away from under the tail without turning its joint
    retreat  straighten the thumb, move the hand back toward the robot
    watch    1.5 s: does the pin stay clipped on its own?
"""

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

TASK = Path(__file__).resolve().parent.parent              # clothespin_reclip/
ROOT = TASK.parent
sys.path.insert(0, str(ROOT / "clothespin" / "scripts"))
sys.path.insert(0, str(ROOT / "clothespin" / "sim"))
import spring_moment  # noqa: E402
import unclip_press as UP  # noqa: E402

OPEN_MM = 5.0        # jaw opening while clipping back (paper 1 mm; 6 mm was the widest without wobble)
BITE = 0.015         # how far the jaws go onto the paper (the unclip started at 22 mm; less = easier to aim)
T = dict(hold=1.5, open=1.0, align=2.0, line_up=0.8, insert=2.0, close=1.2, release=0.6, drop=0.6, retreat=1.2,
         watch=1.5)
DT = 0.02


class Reclip(UP.Press):
    def __init__(self):
        # the placement that passes the full task (= unclip_press.py --full)
        super().__init__(0.622, 0.35, along=0.0235, across=0.005, yaw_deg=-36.0, thumb_yaw=1.2)
        self.slowmo = 1
        self.stop_after = "grip"                   # part 1 ends once the pin is held on the tail tips

    # ---------- small helpers ----------
    def hand_point(self):
        W = self.d.body("left_wrist_yaw_link")
        return W.xpos + W.xmat.reshape(3, 3) @ self.PAD

    def jaw_gap_centre(self):
        """Middle of the gap between the jaw tips (world)."""
        pA, RA = self.pin_frame()
        jaw_a = pA + RA @ np.array([0.0, -self.P["h"] / 2, -self.P["Lj"]])   # leg A's inner face at the jaw tip
        return jaw_a + RA[:, 1] * (self.jaws_mm() / 1000) / 2

    def on_paper(self):
        return self.contact_force("paper", "clothespin") + self.contact_force("paper", "pin_leg_b")

    def squeeze_force(self, target_mm, d_grip):
        """Force for EACH finger to hold the jaws target_mm open (user's moment idea: F = torque / d)."""
        tch = self.touches()
        ds = [tch[k][0] / 1000 for k in ("index", "thumb") if k in tch and tch[k][2] > 0]
        d_now = float(np.clip(np.mean(ds) if ds else d_grip, 0.008, self.P["Lt"]))
        tau = spring_moment.spring_torque(target_mm / 1000) + 0.004 * (target_mm - self.jaws_mm())
        return max(0.5, tau / d_now)

    # ---------- the whole thing ----------
    def run_all(self):
        with contextlib.redirect_stdout(io.StringIO()):
            log = self.run()                       # part 1: unclip + grip (clothespin/unclip_press.py)
        m, d, st = self.m, self.d, self.state
        q, ic, tp = st["q"].copy(), st["index_curl"], st["thumb_pitch"]
        up, along, fd = st["up"], st["along_dir"], st["fd"]
        hold = st["hold_cmd"] or (ic, tp)
        d_grip = st["grip_d"] or 0.025
        paper_front, paper_top = st["paper_front"], st["paper_top"]
        step = lambda err: float(np.clip(0.02 * err, -0.006, 0.006))
        n = int(round(DT / m.opt.timestep))
        t0 = log[-1]["t"] + DT
        stage, ts, t = "hold", t0, t0
        hand0 = delta = cmd = start_gc = None
        rel_from = None
        clamped = False
        self.reclip_from = t0
        while stage is not None:
            s = min(1.0, (t - ts) / T[stage])
            sm = UP.smooth(s)
            target = None
            if stage == "hold":                    # firm grip, jaws shut (locked finger commands)
                ic += float(np.clip(hold[0] - ic, -0.006, 0.006))
                tp += float(np.clip(hold[1] - tp, -0.006, 0.006))
            elif stage in ("open", "align", "line_up", "insert", "close"):
                goal = OPEN_MM * (sm if stage == "open" else (1 - sm) if stage == "close" else 1.0)
                if stage == "close" and s >= 1.0:
                    goal = 0.0
                if stage == "close" and (clamped or (self.jaws_mm() < 1.6 and self.on_paper() > 0.5)):
                    # the jaws are on the paper: FREEZE the fingers, the spring does the clamping. (Force control
                    # kept curling the index as its measured force dropped, another part of the finger pushed the
                    # tail and opened the jaws again to 8 mm.)
                    clamped = True
                else:
                    f = self.squeeze_force(goal, d_grip)
                    ic = min(1.6, ic + step(f - self.contact_force("L_index_intermediate", "pin_leg_b")))
                    tp = min(self.thumb_max, tp + step(f - self.contact_force("L_thumb_distal", "clothespin")))
                if stage in ("align", "line_up", "insert"):
                    # AIM WITH FEEDBACK: the arm lags its target by a few mm, and with a ~5 mm gap around 1 mm paper
                    # a 3 mm height error put the lower jaw against the paper edge. So every step: measure where the
                    # jaw gap really is, and nudge the hand target toward making it follow the planned path.
                    gc = self.jaw_gap_centre()
                    if delta is None:
                        if cmd is None:
                            cmd = self.hand_point()
                        gx = paper_front - 0.010 if stage in ("align", "line_up") else paper_front + BITE
                        start_gc, goal_pt = gc.copy(), np.array([gx, gc[1], paper_top - 0.0005])
                        delta = goal_pt - start_gc
                    want = start_gc + delta * (1.0 if stage == "line_up" else sm)
                    cmd = cmd + np.clip(0.3 * (want - gc), -0.001, 0.001)
                    target = cmd
            elif stage == "release":               # fingers open away from the pin
                if rel_from is None:
                    rel_from = (ic, tp)
                ic = rel_from[0] + (self.shape - 0.25 - rel_from[0]) * sm    # 1. index straightens: up, away
                # (the thumb stays: turning its joint now swept its tip up under the tail, opened the jaws to
                # 9 mm and flicked the pin 4 mm deeper)
            elif stage == "drop":                  # 2. whole hand down: the thumb drops away from under the tail
                if hand0 is None:
                    hand0 = self.hand_point()
                target = hand0 + np.array([0.0, 0.0, -0.012]) * sm
            elif stage == "retreat":               # 3. thumb straightens, hand back toward the robot
                if hand0 is None:
                    hand0 = self.hand_point()
                tp = rel_from[1] + (-0.1 - rel_from[1]) * sm
                target = hand0 + (along * 0.06 + np.array([0, 0, -0.02])) * sm
            if target is not None:
                q, _ = self.ik(target, up, q, dict(self.SHAPE, thumb_pitch=tp, index=ic), point=self.PAD, fdir=fd)
            for j, a in enumerate(self.aid):
                d.ctrl[a] = q[j]
            UP.command(m, d, dict(self.SHAPE, thumb_pitch=tp, index=ic))
            for _ in range(n):
                mujoco.mj_step(m, d)
            pin_xyz, Rp = self.pin_frame()
            log.append(dict(t=t, phase=stage, qpos=d.qpos.copy(), jaws=self.jaws_mm(), touch=self.touches(),
                            f_index=self.contact_force("L_index_intermediate", "pin_leg_b"),
                            f_thumb=self.contact_force("L_thumb_distal", "clothespin"),
                            on_paper=self.on_paper(), faces_up=abs(Rp[:, 1] @ up), pin_x=pin_xyz[0],
                            jaw_tip_x=(pin_xyz + Rp @ np.array([0, 0, -self.P["Lj"]]))[0]))
            t += DT
            if s >= 1.0:
                nxt = {"hold": "open", "open": "align", "align": "line_up", "line_up": "insert", "insert": "close", "close": "release",
                       "release": "drop", "drop": "retreat", "retreat": "watch", "watch": None}[stage]
                stage, ts = nxt, t
                delta = hand0 = None
        return log


def report(u, log):
    by = lambda p: [r for r in log if r["phase"] == p]
    pf = u.state["paper_front"]
    pull = by("pull")
    print(f"part 1 (clothespin/): unclipped and off the paper, held on the tips "
          f"{bool(by('grip')) and by('grip')[-1]['f_index'] > 0.2 and by('grip')[-1]['f_thumb'] > 0.2}")
    held = lambda r: r["f_index"] > 0.2 and r["f_thumb"] > 0.2
    for p in ("hold", "open", "align", "line_up", "insert", "close"):
        rs = by(p)
        if rs:
            print(f"{p:7s}: jaws {rs[-1]['jaws']:4.1f} mm, held {held(rs[-1])}, "
                  f"jaw tips {1000 * (rs[-1]['jaw_tip_x'] - pf):+5.1f} mm past the paper edge, "
                  f"touching the paper {rs[-1]['on_paper'] > 0.02}")
    end = log[-1]
    clipped = (end["on_paper"] > 0.02 and 0.5 < end["jaws"] < 1.6 and end["jaw_tip_x"] > pf + 0.005
               and end["faces_up"] > 0.9 and end["f_index"] < 0.05 and end["f_thumb"] < 0.05)
    print(f"end    : jaws {end['jaws']:.1f} mm (1.0 = clamped on the 1 mm paper), jaw tips "
          f"{1000 * (end['jaw_tip_x'] - pf):+.1f} mm onto the paper, flat {end['faces_up']:.2f}, hand off the pin "
          f"{end['f_index'] < 0.05 and end['f_thumb'] < 0.05}")
    print("CLIPPED BACK:", "PASS" if clipped else "FAIL")
    return clipped


def play(u, log, slow):
    """Replay the logged motion in the MuJoCo viewer (run with mjpython)."""
    from mujoco import viewer as mj_viewer
    m, d = u.m, u.d
    with mj_viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = u.state["p_pin0"]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.35, 200, -20
        start = time.time()
        for r in log:
            if not v.is_running():
                return
            d.qpos[:] = r["qpos"]
            mujoco.mj_forward(m, d)
            v.sync()
            lag = (r["t"] - log[0]["t"]) * slow - (time.time() - start)
            if lag > 0:
                time.sleep(lag)
        print("finished - close the MuJoCo window")
        while v.is_running():
            v.sync()
            time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true", help="watch it afterwards (run with mjpython)")
    ap.add_argument("--slow", type=float, default=1.0)
    a = ap.parse_args()
    u = Reclip()
    log = u.run_all()
    report(u, log)
    if a.viewer:
        play(u, log, a.slow)


if __name__ == "__main__":
    main()
