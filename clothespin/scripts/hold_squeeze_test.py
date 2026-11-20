"""Step 3: hold the pin in the air with a LIGHT grip (stays closed) and squeeze FIRM to open it,
following the open/close timing measured from your video (clothespin/results/pin_01_events.json).

    python clothespin/scripts/hold_squeeze_test.py                  # run, print the numbers
    python clothespin/scripts/hold_squeeze_test.py --show           # watch it (close-up, real time)
    python clothespin/scripts/hold_squeeze_test.py --find_light     # search for the light grip level

Grip level g: 0 = fingers open, 1 = the firm squeeze that opened the pin in the pinch test.
    LIGHT: holds the 1 g pin by friction, but pushes less than the spring -> jaws stay shut
    FIRM:  beats the spring -> jaws open
The pin starts between the fingertips (picking it up is step 4): for the first second it is held in
place while the fingers close to LIGHT, then let go - from then on only the fingers hold it.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

TASK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TASK / "sim"))
sys.path.insert(0, str(TASK / "scripts"))
import pin_model as pm  # noqa: E402
from pin_pinch_test import PRE, SQUEEZE, command, set_fingers, mat2quat  # noqa: E402

SCENE = TASK / "sim/pin_scene.xml"
LIGHT = 0.70          # default light grip level (set with --find_light)
VIDEO_WINDOW = (17.0, 24.5)   # part of pin_01 with the pin in the air (open at 20.0-20.7 and 21.6-22.5 s)


def grip_cfg(g):
    return {k: PRE[k] + (SQUEEZE[k] - PRE[k]) * g for k in PRE}


class Sim:
    def __init__(self, pad_friction=False, pin_mass=None):
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        m0 = self.m
        if pad_friction:                                  # soft-pad-like contact: resist rolling/twisting
            for name in ("pin_leg_a", "pin_leg_b"):
                g = m0.geom(name).id
                m0.geom_condim[g] = 6
                m0.geom_friction[g] = [1.2, 0.02, 0.002]
        if pin_mass is not None:                          # heavier pin for numerical stability
            for b in ("clothespin", "pin_leg_b"):
                i = m0.body(b).id
                k = (pin_mass / 2) / m0.body_mass[i]
                m0.body_mass[i] *= k
                m0.body_inertia[i] *= k
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.fa = m.jnt_qposadr[m.joint("pin_free").id]
        self.fv = m.jnt_dofadr[m.joint("pin_free").id]
        self.ha = m.jnt_qposadr[m.joint("pin_hinge").id]
        self.P = pm.pin_params()
        self.pin_pose = self._pin_in_hand()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("home").id)
        set_fingers(self.m, self.d, PRE)
        self._place_pin()
        self._hold_pin(True)
        mujoco.mj_forward(self.m, self.d)

    def _hold_pin(self, on):
        """Keep the pin still until BOTH fingertips are on it: its velocity is zeroed after every physics
        step (its position is never moved, so nothing jumps). Like the fixed pin in the pinch test.
        Tried first, all failed: teleporting it each step and a weld constraint (blew up - the pin is
        too light), gravity off (the index arrives first and pushes the floating pin away), gravity off
        + 1000x heavier (still pushed away, just slower)."""
        self.holding = on

    def _pin_in_hand(self):
        """Pin pose (pos, quat) with its tails where the thumb and index tips physically meet."""
        m, d = self.m, self.d
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        set_fingers(m, d, PRE)
        th, ix = m.body("L_thumb_distal").id, m.body("L_index_intermediate").id
        meet = None
        for k in range(300):
            command(m, d, grip_cfg(min(1.0, (k + 1) / 200)))
            for _ in range(10):
                mujoco.mj_step(m, d)
            for i in range(d.ncon):
                c = d.contact[i]
                if {m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]} == {th, ix}:
                    n = c.frame[:3] if m.geom_bodyid[c.geom1] == th else -c.frame[:3]
                    meet = (c.pos.copy(), n.copy())
                    break
            if meet:
                break
        y = meet[1] / np.linalg.norm(meet[1])
        fdir = d.body("left_wrist_yaw_link").xmat.reshape(3, 3)[:, 0]
        z = -(fdir - (fdir @ y) * y)
        z /= np.linalg.norm(z)
        R = np.column_stack([np.cross(y, z), y, z])
        P, zt = self.P, 0.007
        a_out = -P["h"] / 2 - P["t"]
        b_out = P["h"] / 2 - zt * np.sin(P["phi0"]) + P["t"]
        pos = meet[0] - R @ np.array([0.0, (a_out + b_out) / 2, zt])
        return pos, mat2quat(R)

    def _place_pin(self):
        d = self.d
        d.qpos[self.fa:self.fa + 3] = self.pin_pose[0]
        d.qpos[self.fa + 3:self.fa + 7] = self.pin_pose[1]
        d.qpos[self.ha] = 0.0
        d.qvel[self.fv:self.fv + 6] = 0.0

    def jaws_mm(self):
        return pm.jaw_opening(self.d.qpos[self.ha], self.P) * 1000

    def pin_offset_mm(self):
        return np.linalg.norm(self.d.qpos[self.fa:self.fa + 3] - self.pin_pose[0]) * 1000

    def run(self, schedule, hold_pin_until=1.0, on_frame=None, dt=0.02):
        """schedule(t) -> grip level. Returns list of (t, grip, jaws_mm, pin_offset_mm)."""
        m, d = self.m, self.d
        steps = int(round(dt / m.opt.timestep))
        log, t = [], 0.0
        T = schedule.duration
        while t < T:
            g = schedule(t)
            command(m, d, grip_cfg(g))
            if t >= hold_pin_until:                      # fingers have it: remove the support
                self._hold_pin(False)
            for _ in range(steps):
                mujoco.mj_step(m, d)
                if self.holding:                          # truly still: position AND velocity
                    d.qpos[self.fa:self.fa + 7] = np.r_[self.pin_pose[0], self.pin_pose[1]]
                    d.qvel[self.fv:self.fv + 6] = 0.0
            t += dt
            log.append((t, g, self.jaws_mm(), self.pin_offset_mm()))
            if on_frame:
                on_frame(t, g)
        return log


class Schedule:
    """Grip level over time. First second: close to LIGHT (pin supported). Then the video window:
    FIRM while the pin was open in your video, LIGHT otherwise (0.25 s ramps)."""

    def __init__(self, light, opens, window, firm=1.0, ramp=0.25, settle=1.5):
        self.light, self.firm, self.ramp = light, firm, ramp
        self.t0 = settle - window[0]                     # sim time = video time + t0
        self.opens = [(a + self.t0, b + self.t0) for a, b in opens if window[0] <= a <= window[1]]
        self.duration = settle + (window[1] - window[0])

    def __call__(self, t):
        if t < 1.0:
            return self.light * min(1.0, t / 0.8)
        g = self.light
        for a, b in self.opens:
            if a - self.ramp <= t <= b + self.ramp:
                up = np.clip((t - (a - self.ramp)) / self.ramp, 0, 1)
                down = np.clip(((b + self.ramp) - t) / self.ramp, 0, 1)
                g = self.light + (self.firm - self.light) * min(up, down)
        return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--find_light", action="store_true")
    ap.add_argument("--light", type=float, default=LIGHT)
    ap.add_argument("--lo", type=float, default=0.40)
    ap.add_argument("--hi", type=float, default=1.00)
    ap.add_argument("--step", type=float, default=0.05)
    args = ap.parse_args()
    events = json.loads((TASK / "results/pin_01_events.json").read_text())
    sim = Sim()

    if args.find_light:
        print("light grip level -> after 1.5 s of holding with the support removed:")
        for g in np.arange(args.lo, args.hi + 1e-9, args.step):
            sim.reset()
            sched = Schedule(g, [], (0, 1.5), settle=1.0)
            sched.duration = 2.5
            log = sim.run(sched)
            t, _, jaws, off = log[-1]
            held = off < 10
            print(f"  g={g:.2f}: jaws {jaws:4.1f} mm, pin moved {off:5.1f} mm -> "
                  f"{'HELD' if held else 'DROPPED'}{', CLOSED' if jaws < 1 else ', opened'}")
        return

    on_frame = None
    if args.show:
        import cv2
        renderer = mujoco.Renderer(sim.m, 480, 640)
        cam = mujoco.MjvCamera()
        cam.distance, cam.azimuth, cam.elevation = 0.22, 200, -15
        start = time.time()

        def on_frame(t, g):
            cam.lookat[:] = sim.d.qpos[sim.fa:sim.fa + 3]
            renderer.update_scene(sim.d, cam)
            img = renderer.render()[:, :, ::-1].copy()
            label = "FIRM (open)" if g > sim_light + 0.05 else "light (closed)"
            cv2.putText(img, f"t={t:4.1f}s  {label}  jaws {sim.jaws_mm():4.1f} mm", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("hold + squeeze (q = quit)", img)
            cv2.waitKey(max(1, int(1000 * (t - (time.time() - start)))))
    sim_light = args.light

    sim.reset()
    sched = Schedule(args.light, events["open"], VIDEO_WINDOW)
    log = sim.run(sched, on_frame=on_frame)
    t = np.array([r[0] for r in log]); g = np.array([r[1] for r in log])
    jaws = np.array([r[2] for r in log]); off = np.array([r[3] for r in log])
    print(f"light grip {args.light:.2f}, firm 1.00; video window {VIDEO_WINDOW[0]}-{VIDEO_WINDOW[1]} s")
    ok = True
    for a, b in sched.opens:
        inside = (t > a + 0.1) & (t < b)
        print(f"  you opened it {a - sched.t0:.1f}-{b - sched.t0:.1f} s -> robot jaws "
              f"{jaws[inside].max():4.1f} mm open")
        ok &= jaws[inside].max() > 6
    light_times = (t > 1.2) & np.array([all(not (a - 0.4 <= x <= b + 0.6) for a, b in sched.opens) for x in t])
    print(f"  light grip: jaws at most {jaws[light_times].max():.1f} mm (closed = 0)")
    print(f"  pin kept in the hand: moved at most {off[t > 1.0].max():.1f} mm from the start")
    ok &= jaws[light_times].max() < 1.0 and off[t > 1.0].max() < 10
    print("STEP 3:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
