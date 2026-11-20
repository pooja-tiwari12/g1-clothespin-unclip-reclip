"""Step 4: unclip the pin from the paper - on the real arrangement (seated robot, pin clipped on the paper).

    python clothespin/scripts/unclip_test.py              # run, print the checks
    python clothespin/scripts/unclip_test.py --show       # watch (close-up + side, real time)
    mjpython clothespin/scripts/unclip_test.py --viewer --slow 3   # interactive MuJoCo viewer, 3x slow motion

Sequence (left hand, "pad" pinch: thumb pad under the pin, index pad on top - like your video):
    1. rest     hand on the lap
    2. reach    to just in front of the tails, fingers open
    3. insert   slide in along the pin so the tails are between thumb and index
    4. squeeze  firm pinch while the PAPER holds the pin -> jaws open
    5. pull     straight back toward the robot, still squeezing -> pin comes off the paper
    6. relax    to a light grip -> jaws close, pin still held
Checks: jaws open while squeezing, pin off the paper, pin still between the fingers at the end,
pin did not roll over (its flat faces still facing the thumb/index).
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

TASK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TASK / "sim"))
sys.path.insert(0, str(TASK / "scripts"))
import pin_model as pm  # noqa: E402
from pin_pinch_test import PINCHES, command  # noqa: E402

SCENE = TASK / "sim/pin_scene.xml"
PRE, TOUCH, SQUEEZE = (PINCHES["pad"][k] for k in ("PRE", "TOUCH", "SQUEEZE"))
AIM_Z = 0.008          # pinch the tails 8 mm from the spring (the pad pinch failed when 3 mm nearer the tail end)
APPROACH = 0.045       # start the insert 4.5 cm out along the pin
PULL = 0.06            # pull 6 cm back
ARM = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"] + [
    f"left_{j}_joint" for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                                "wrist_roll", "wrist_pitch", "wrist_yaw")]


def blend(a, b, s):
    return {k: a[k] + (b[k] - a[k]) * s for k in a}


def smooth(s):
    s = float(np.clip(s, 0, 1))
    return s * s * (3 - 2 * s)


class Unclip:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.d = mujoco.MjData(self.m)
        self.k = mujoco.MjData(self.m)                       # scratch data for IK
        m = self.m
        ids = [m.joint(j).id for j in ARM]
        self.adr = [m.jnt_qposadr[i] for i in ids]
        self.lo, self.hi = m.jnt_range[ids, 0].copy(), m.jnt_range[ids, 1].copy()
        self.lo[:3], self.hi[:3] = -0.4, 0.4                 # human-like waist range
        self.aid = [m.actuator(j.replace("_joint", "")).id for j in ARM]
        self.fa = m.jnt_qposadr[m.joint("pin_free").id]
        self.ha = m.jnt_qposadr[m.joint("pin_hinge").id]
        self.P = pm.pin_params()
        self.PINCH, self.SQ = self._pinch_in_hand()
        self.table = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in ("table", "paper")]
        self.hand = [g for g in range(m.ngeom) if m.geom_contype[g] and
                     m.body(m.geom_bodyid[g]).name.startswith(("L_", "left_wrist", "left_elbow"))]

    def _set_fingers(self, data, cfg):
        def q(n, v): data.qpos[self.m.jnt_qposadr[self.m.joint(n).id]] = v
        q("L_index_proximal_joint", cfg["index"]); q("L_index_intermediate_joint", cfg["index"])
        q("L_thumb_proximal_yaw_joint", cfg["thumb_yaw"]); q("L_thumb_proximal_pitch_joint", cfg["thumb_pitch"])
        q("L_thumb_intermediate_joint", 1.6 * cfg["thumb_pitch"]); q("L_thumb_distal_joint", 2.4 * cfg["thumb_pitch"])
        for f in ("middle", "ring", "pinky"):
            q(f"L_{f}_proximal_joint", cfg["others"]); q(f"L_{f}_intermediate_joint", cfg["others"])

    def _pinch_in_hand(self):
        """Where the thumb and index pads PHYSICALLY meet while closing PRE -> SQUEEZE (wrist frame), and the
        squeeze direction thumb -> index. Measured by simulating the closing with the hand held up in the air
        (computing it from the hand's shape instead was a few mm off - and the pin's tail is only 7 mm wide)."""
        m, k = self.m, self.k
        sys.path.insert(0, str(TASK.parent / "sim"))
        import build_desk_scene as mug_scene
        mujoco.mj_resetDataKeyframe(m, k, m.key("home").id)
        for name, v in mug_scene.HOME.items():                # left arm raised, hand in free air
            if name.startswith("left_"):
                k.qpos[m.jnt_qposadr[m.joint(name).id]] = v
                k.ctrl[m.actuator(name.replace("_joint", "")).id] = v
        self._set_fingers(k, PRE)
        th, ix = m.body("L_thumb_distal").id, {m.body("L_index_intermediate").id, m.body("L_index_proximal").id}
        for step in range(300):
            command(m, k, blend(PRE, SQUEEZE, min(1.0, (step + 1) / 200)))
            for _ in range(20):
                mujoco.mj_step(m, k)
            for i in range(k.ncon):
                c = k.contact[i]
                b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                if (b1 == th and b2 in ix) or (b2 == th and b1 in ix):
                    n = c.frame[:3] if b1 == th else -c.frame[:3]
                    W = k.body("left_wrist_yaw_link")
                    R = W.xmat.reshape(3, 3)
                    return R.T @ (c.pos - W.xpos), R.T @ n
        raise RuntimeError("thumb and index never met")

    def ik(self, target, up, q0, fingers, seeds=0, point=None, fdir=None):
        """Arm+waist angles putting the pinch point at `target` with the squeeze direction along `up`,
        hand clear of the table/paper, wrist as straight as possible."""
        m, k = self.m, self.k
        hx = np.array([1.0, 0.0, 0.0])
        self._perp = hx - (hx @ self.SQ) * self.SQ
        self._perp /= np.linalg.norm(self._perp)
        k.qpos[:] = self.d.qpos
        self._set_fingers(k, fingers)
        wr = [ARM.index(f"left_wrist_{x}_joint") for x in ("roll", "pitch", "yaw")]

        def err(x):
            k.qpos[self.adr] = x
            mujoco.mj_kinematics(m, k)
            W = k.body("left_wrist_yaw_link")
            R = W.xmat.reshape(3, 3)
            # which way the hand faces: a hand direction PERPENDICULAR to the squeeze axis (the finger axis
            # itself is only ~65 deg from it, so asking for it directly fought the squeeze alignment)
            face = (R @ self._perp - fdir) if fdir is not None else np.zeros(0)
            return np.r_[(W.xpos + R @ (self.PINCH if point is None else point) - target) * 30, R @ self.SQ - up, face,
                         0.3 * x[:3], 0.15 * x[wr], 0.03 * (x - q0)]
        best = None
        starts = [q0] + [np.random.default_rng(i).uniform(self.lo, self.hi) for i in range(seeds)]
        for x0 in starts:
            z = least_squares(err, np.clip(x0, self.lo, self.hi), bounds=(self.lo, self.hi))
            e = err(z.x)
            k.qpos[self.adr] = z.x
            mujoco.mj_forward(m, k)
            clear = min(mujoco.mj_geomDistance(m, k, a, b, 0.05, np.zeros(6)) for a in self.hand for b in self.table)
            bad_face = fdir is not None and np.linalg.norm(e[6:9]) > 0.25
            score = (np.linalg.norm(e[:3]) / 30 > 0.003 or np.linalg.norm(e[3:6]) > 0.15 or bad_face, clear < 0.003, z.cost)
            if best is None or score < best[0]:
                best = (score, z.x, clear)
        self.last_ok = not best[0][0]
        return best[1], best[2]

    def pin_frame(self):
        B = self.d.body("clothespin")
        return B.xpos.copy(), B.xmat.reshape(3, 3).copy()

    def jaws_mm(self):
        return pm.jaw_opening(self.d.qpos[self.ha], self.P) * 1000

    def pinch_point(self):
        W = self.d.body("left_wrist_yaw_link")
        return W.xpos + W.xmat.reshape(3, 3) @ self.PINCH

    def run(self, show=False, viewer=False):
        m, d = self.m, self.d
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        for _ in range(500):                                  # let the pin settle on the paper (0.5 s)
            mujoco.mj_step(m, d)
        p_pin, R_pin = self.pin_frame()
        up, along = R_pin[:, 1], R_pin[:, 2]                  # squeeze direction, toward the tails/robot
        # aim point in the pin's own frame: x = across its 7 mm width, z = along it from the spring.
        # AIM_X re-centres the fingertips across the width (they landed on the side edge at x = -3.5 mm)
        grasp = p_pin + R_pin @ np.array([self.aim_x, 0.0007, self.aim_z])
        pre = grasp + along * APPROACH
        q_lap = d.qpos[self.adr].copy()
        low = pre - up * 0.05                                 # 5 cm below pin level, in front of it
        q_low, _ = self.ik(low, up, q_lap, PRE, seeds=24)
        q_pre, clear_pre = self.ik(pre, up, q_low, PRE)
        q_grasp, clear_grasp = self.ik(grasp, up, q_pre, PRE)
        print(f"plan: hand clearance to table/paper {clear_pre * 1000:.0f} mm at the start of the insert, "
              f"{clear_grasp * 1000:.0f} mm at the pin")
        lap_fingers = {"index": 0.35, "thumb_yaw": 0.0, "thumb_pitch": 0.0, "others": 0.35}

        # timeline (s). The hand comes UP from below, in front of the pin, then slides in - swinging straight
        # from the lap to the pin brought it in from above and left it resting on the paper (18 N), stuck.
        # squeeze in two stages (your idea): the THUMB comes up first and supports the bottom leg, THEN the
        # index presses the top tail down. Closing both at once knocked the pin around before it opened.
        T = dict(rest=1.0, reach=2.5, rise=1.0, insert=1.5, thumb=0.8, press=1.2, pull=1.5, relax=0.7, hold=1.5)
        THUMB_ON = dict(PRE, thumb_pitch=self.thumb_pitch)          # thumb up under the pin, index still open
        PRESS = dict(THUMB_ON, index=self.index_press)               # index curls down and presses
        t_edges = np.cumsum([0] + list(T.values()))
        names = list(T)
        dt = 0.02
        steps = int(round(dt / m.opt.timestep))
        renderer = None
        if show:
            import cv2
            renderer = mujoco.Renderer(m, 360, 480)
            cams = [mujoco.MjvCamera(), mujoco.MjvCamera()]
            cams[0].distance, cams[0].azimuth, cams[0].elevation = 0.25, 215, -20
            cams[1].distance, cams[1].azimuth, cams[1].elevation = 1.3, 100, -10
            wall = time.time()
        v = None
        if viewer:                                            # interactive MuJoCo viewer (run with mjpython)
            from mujoco import viewer as mj_viewer
            v = mj_viewer.launch_passive(m, d)
            v.cam.lookat[:] = [0.35, 0.10, 0.72]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.9, 150, -20
            wall_v = time.time()
        q = q_lap.copy()
        log = []
        pull_from = None
        while True:
            t = len(log) * dt
            if t >= t_edges[-1]:
                break
            i = int(np.searchsorted(t_edges, t, side="right") - 1)
            phase, s = names[i], smooth((t - t_edges[i]) / T[names[i]])
            if phase == "rest":
                fingers = lap_fingers
            elif phase == "reach":
                q = q_lap + (q_low - q_lap) * s
                fingers = blend(lap_fingers, PRE, s)
            elif phase == "rise":
                q, _ = self.ik(low + (pre - low) * s, up, q, PRE)
                fingers = PRE
            elif phase == "insert":
                q, _ = self.ik(pre + (grasp - pre) * s, up, q, PRE)
                fingers = PRE
            elif phase == "thumb":
                fingers = blend(PRE, THUMB_ON, s)
            elif phase == "press":
                fingers = blend(THUMB_ON, PRESS, s)
            elif phase == "pull":
                if pull_from is None:
                    pull_from = self.pinch_point()
                q, _ = self.ik(pull_from + along * PULL * s, up, q, PRESS)
                fingers = PRESS
            else:                                             # relax / hold: light grip
                fingers = blend(PRESS, LIGHT, s) if phase == "relax" else LIGHT
            for j, a in enumerate(self.aid):
                d.ctrl[a] = q[j]
            command(m, d, fingers)
            for _ in range(steps):
                mujoco.mj_step(m, d)
            if v is not None:
                if not v.is_running():
                    break
                v.sync()
                lag = (t + dt) * self.slowmo - (time.time() - wall_v)   # real time (or slower with --slow)
                if lag > 0:
                    time.sleep(lag)
            pin_xyz, Rp = self.pin_frame()
            log.append(dict(t=t, phase=phase, jaws=self.jaws_mm(),
                            pin_to_pinch=np.linalg.norm(pin_xyz + Rp @ np.array([self.aim_x, 0.0007, self.aim_z]) - self.pinch_point()),
                            pin_x=pin_xyz[0], faces_up=abs(Rp[:, 1] @ up),
                            nan=bool(np.isnan(d.qpos).any())))
            if renderer is not None:
                import cv2
                tiles = []
                for c in cams:
                    c.lookat[:] = pin_xyz
                    renderer.update_scene(d, c)
                    tiles.append(renderer.render())
                img = np.hstack(tiles)[:, :, ::-1].copy()
                cv2.putText(img, f"t={t:4.1f}s {phase}  jaws {self.jaws_mm():4.1f} mm", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("unclip (q = quit)", img)
                if cv2.waitKey(max(1, int(1000 * (t - (time.time() - wall))))) & 0xFF == ord("q"):
                    renderer = None
            if log[-1]["nan"]:
                print("SIMULATION BLEW UP at", f"{t:.2f}s ({phase})")
                break
        if v is not None:
            print("finished - close the MuJoCo window to see the results")
            while v.is_running():
                v.sync()
                time.sleep(0.05)
        return log, p_pin


LIGHT = None   # set in main (grip level between PRE and SQUEEZE)


def main():
    global LIGHT
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--light", type=float, default=0.85, help="light grip: fraction of the way PRE -> SQUEEZE")
    ap.add_argument("--viewer", action="store_true", help="watch in the MuJoCo viewer (run with mjpython)")
    ap.add_argument("--slow", type=float, default=1.0, help="slow motion factor for --viewer, e.g. 3")
    ap.add_argument("--thumb", type=float, default=0.30, help="thumb bend when it comes up under the pin")
    ap.add_argument("--index", type=float, default=1.30, help="index curl target when pressing the top tail")
    ap.add_argument("--aim_x", type=float, default=0.0, help="aim offset across the pin's width (mm)")
    ap.add_argument("--aim_z", type=float, default=AIM_Z * 1000, help="aim distance from the spring along the pin (mm)")
    args = ap.parse_args()
    LIGHT = blend(PRE, SQUEEZE, args.light)
    u = Unclip()
    u.slowmo = args.slow
    u.thumb_pitch, u.index_press = args.thumb, args.index
    u.aim_x, u.aim_z = args.aim_x / 1000, args.aim_z / 1000
    log, p0 = u.run(show=args.show, viewer=args.viewer)
    by = lambda ph: [r for r in log if r["phase"] == ph]
    paper_front = u.d.body("paper").xpos[0] - u.m.geom_size[u.m.geom("paper").id][0]
    pl = by("pull")[-1]
    end = log[-1]
    print(f"thumb first, then index presses (paper holds the pin): jaws {max(r['jaws'] for r in by('press')):.1f} mm open")
    print(f"pull: jaw tips {(p0[0] + 0.0 - pl['pin_x']) * 100:.1f} cm back; pin's jaws now "
          f"{'OFF' if pl['pin_x'] + u.P['Lj'] < paper_front + 0.001 else 'still ON'} the paper")
    print(f"end (light grip): jaws {end['jaws']:.1f} mm, pin {end['pin_to_pinch'] * 1000:.0f} mm from the pinch point, "
          f"flat faces still toward thumb/index: {end['faces_up']:.2f} (1 = yes, 0 = rolled over)")
    ok = (max(r['jaws'] for r in by('press')) > 6 and pl['pin_x'] + u.P['Lj'] < paper_front + 0.001
          and end['pin_to_pinch'] < 0.015 and end['faces_up'] > 0.8 and end['jaws'] < 2.0)
    print("STEP 4:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
