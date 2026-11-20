"""Step 4, "press the button" version: index placed ON the middle of the top tail first, thumb up under the
pin, then the index presses straight down with a small extra curl (no sweeping, so it can't skid).

    python clothespin/scripts/unclip_press.py                       # run, print the checks
    mjpython clothespin/scripts/unclip_press.py --viewer --slow 3   # watch in the MuJoCo viewer

Why: when the index curled all the way down from open, its fingertip moved in an arc and SKIDDED along
the pin - landing past the spring (closes the jaws) or off the tail tip - never on the middle of the tail.
Sequence (your idea + this fix):
    reach / rise   in front of the pin, index already bent into its pressing shape, thumb open
    insert         slide in until the index pad is 2 mm above the MIDDLE of the top tail
    thumb          thumb comes up under the pin until it touches (contact detected, not a guessed angle)
    press          index presses down: small extra curl, thumb holds -> jaws open
    pull           pull the pin off the paper
    relax / hold   ease the index off a little: jaws close, pin still held
"""

import argparse
import io
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

TASK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TASK / "scripts"))
from unclip_test import Unclip, PRE, blend, smooth  # noqa: E402  (scene, IK, pinch helpers)
from pin_pinch_test import command  # noqa: E402


class Press(Unclip):
    def __init__(self, shape=0.75, press=0.35, hover=0.002, along=None, across=0.0, opposed=False,
                 yaw_deg=None, thumb_yaw=None):
        super().__init__()
        self.opposed = opposed
        self.balanced = True                                  # press with both fingers
        self.moment = True                                    # ...each making the SAME moment F*d (else equal F)
        self.f_max = 8.0                                      # N per finger at the end of the press ramp
        self.thumb_max = 0.6                                  # thumb pitch joint limit
        self.clear_mm = 4.0                                   # stop pressing at this jaw opening (paper: 1 mm)
        self.hold_open_mm = 6.0                               # jaw opening held while sliding off the paper
        self.air_open_mm = 6.0                                # jaw opening in the air: widest with NO wobble (sweep)
        self.hold_mode = "lock"                               # hold after the unclip: "lock" (commands) or "force"
        self.stop_after = None                                # end the run after this stage (e.g. "grip")
        self.carry_on = True                                  # after the unclip: grip, lift, open/close, place
        self.overhang = 0.020                                 # placing: tail tips this far past the paper edge
        self.yaw_deg = yaw_deg                                # which way the fingers point (None = IK decides)
        self.thumb_yaw = PRE["thumb_yaw"] if thumb_yaw is None else thumb_yaw
        self.shape, self.press, self.hover = shape, press, hover
        self.along = 0.58 * self.P["Lt"] if along is None else along   # default: just past the tail's middle
        self.across = across
        self.SHAPE = dict(PRE, index=shape, thumb_yaw=self.thumb_yaw)   # index in pressing shape, thumb open
        self.PAD = self._index_pad()

    def _index_pad(self):
        """Index pad point (wrist frame) at the pressing shape: the point of the index's last segment that
        faces the thumb (it is what touches the pin's top)."""
        m, k = self.m, self.k
        mujoco.mj_resetDataKeyframe(m, k, m.key("home").id)
        self._set_fingers(k, self.SHAPE)
        mujoco.mj_forward(m, k)
        W = k.body("left_wrist_yaw_link")
        Rw = W.xmat.reshape(3, 3)
        best = None
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != m.body("L_index_intermediate").id or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            w = (k.geom_xmat[g].reshape(3, 3) @ v.T).T + k.geom_xpos[g]
            loc = (Rw.T @ (w - W.xpos).T).T
            s = loc @ self.SQ                                 # lowest along the squeeze direction = the pad
            i = int(np.argmin(s))
            if best is None or s[i] < best[0]:
                best = (s[i], loc[i])
        return best[1]

    def finger_dir(self, up, along_dir):
        """Unit vector the fingers point along: yaw_deg about the squeeze axis, 0 = toward the paper."""
        if self.yaw_deg is None:
            return None
        b = -(along_dir - (along_dir @ up) * up)
        b /= np.linalg.norm(b)
        c = np.cross(up, b)
        a = np.radians(self.yaw_deg)
        return np.cos(a) * b + np.sin(a) * c

    def pad_target(self):
        """Middle of leg B's top surface, `along` from the spring, `across` from the centre line."""
        B = self.d.body("pin_leg_b")
        R = B.xmat.reshape(3, 3)
        return B.xpos + R @ np.array([self.across, self.P["t"], self.along])

    def contact_force(self, finger, leg):
        f = np.zeros(6)
        tot = 0.0
        a_id, b_id = self.m.body(finger).id, self.m.body(leg).id
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if {self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]} == {a_id, b_id}:
                mujoco.mj_contactForce(self.m, self.d, i, f)
                tot += f[0]
        return tot

    def thumb_on_pin(self):
        """Closest point of the thumb tip to the pin's bottom leg, in the pin frame (mm)."""
        m, d = self.m, self.d
        B = d.body("clothespin"); R = B.xmat.reshape(3, 3)
        best = (9, np.zeros(3))
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != m.body("L_thumb_distal").id or not m.geom_contype[g]:
                continue
            ft = np.zeros(6)
            dd = mujoco.mj_geomDistance(m, d, g, m.geom("pin_leg_a").id, 0.1, ft)
            if dd < best[0]:
                best = (dd, R.T @ (ft[:3] - B.xpos) * 1000)
        return best[1]

    def touches(self):
        """Real contact points + forces: index on the top leg, thumb on the bottom leg. Returns
        dict(index=(along_mm, across_mm, N), thumb=(...), torque=opening torque on the hinge from both, N*m).
        along = distance from the spring toward the tail tip; spring needs ~0.022 N*m to open."""
        m, d = self.m, self.d
        A = d.body("clothespin"); RA = A.xmat.reshape(3, 3)
        hinge = d.body("pin_leg_b").xpos
        ix, th = m.body("L_index_intermediate").id, m.body("L_thumb_distal").id
        legs = {m.geom("pin_leg_b").id: "index"}
        legs.update({m.geom("pin_leg_a").id: "thumb"})
        out, torque, f = {}, 0.0, np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            pair = {c.geom1, c.geom2}
            for leg, who in legs.items():
                finger = ix if who == "index" else th
                other = (pair - {leg})
                if leg in pair and other and m.geom_bodyid[other.pop()] == finger:
                    mujoco.mj_contactForce(m, d, i, f)
                    n = c.frame[:3] * (1 if c.geom2 == leg else -1)      # force ON the leg
                    F = n * f[0]
                    loc = RA.T @ (c.pos - hinge) * 1000
                    al, ac, F0 = out.get(who, (0, 0, 0))
                    out[who] = (al + loc[2] * f[0], ac + loc[0] * f[0], F0 + f[0])
                    # hinge axis = pin x (across); opening = pushes tails toward each other
                    tq = np.cross(c.pos - hinge, F) @ RA[:, 0]
                    torque += tq if who == "index" else -tq
        for k, (al, ac, F0) in list(out.items()):
            out[k] = (al / F0, ac / F0, F0) if F0 > 1e-6 else (0, 0, 0)
        out["torque"] = torque
        return out

    def index_pad_on_pin(self):
        """Where the index pad is, in pin_leg_b's frame (mm): across, height above top, along."""
        W = self.d.body("left_wrist_yaw_link")
        p = W.xpos + W.xmat.reshape(3, 3) @ self.PAD
        B = self.d.body("pin_leg_b")
        loc = B.xmat.reshape(3, 3).T @ (p - B.xpos)
        return np.array([loc[0], loc[1] - self.P["t"], loc[2]]) * 1000

    def run(self, show=False, viewer=False, trace=False):
        m, d = self.m, self.d
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        for _ in range(500):                                  # pin settles on the paper
            mujoco.mj_step(m, d)
        p_pin, R_pin = self.pin_frame()
        up, along_dir = R_pin[:, 1], R_pin[:, 2]
        import spring_moment
        tau0 = spring_moment.spring_torque(0.001)                         # spring torque while clamped on paper
        tau1 = 1.25 * spring_moment.spring_torque(self.clear_mm / 1000)   # 25% more than holds it clear_mm open
        bite = min(0.55 * self.P["Lj"], 0.022)                            # same rule as build_pin_scene.jaw_bite
        x_clamp = -(self.P["Lj"] - bite / 2)                              # paper clamp: middle of the bite
        if self.opposed:
            # aim the point where thumb and index PHYSICALLY meet at the middle of the tail, between the two legs:
            # then the thumb pushes up exactly under where the index pushes down (no twisting couple)
            B = d.body("pin_leg_b"); RB = B.xmat.reshape(3, 3)
            mid_b = B.xpos + RB @ np.array([self.across, 0.0, self.along])        # leg B's inner face
            A_bottom = p_pin + R_pin @ np.array([self.across, -self.P["h"] / 2, self.along])
            pad_at = (mid_b + A_bottom) / 2
            self.PAD = self.PINCH
        else:
            pad_at = self.pad_target() + up * self.hover      # index pad 2 mm above the middle of the tail
        pre = pad_at + along_dir * 0.045
        fd = self.finger_dir(up, along_dir)
        low = pre - up * 0.05
        q_lap = d.qpos[self.adr].copy()
        q_low, _ = self.ik(low, up, q_lap, self.SHAPE, seeds=24, point=self.PAD, fdir=fd)
        q_pre, c1 = self.ik(pre, up, q_low, self.SHAPE, point=self.PAD, fdir=fd)
        _, c2 = self.ik(pad_at, up, q_pre, self.SHAPE, point=self.PAD, fdir=fd)
        print(f"plan: hand clearance {c1 * 1000:.0f} mm before the insert, {c2 * 1000:.0f} mm at the pin")
        lap = {"index": 0.35, "thumb_yaw": 0.0, "thumb_pitch": 0.0, "others": 0.35}

        # Fixed-time part (approach), then a contact-driven part (your sequence):
        #   thumb  : bend up until it TOUCHES the bottom leg, then stop there
        #   index  : curl down until it TOUCHES the top leg, then stop there
        #   settle : hold until the pin is still (both fingers on it)
        #   press  : only now press with the index, gradually -> jaws open
        T = dict(rest=1.0, reach=2.5, rise=1.0, insert=1.5)
        edges = np.cumsum([0] + list(T.values()))
        names = list(T)
        T_LATER = dict(pull=1.5, relax=0.8, hold=1.5,
                       grip=1.0, lift=1.5, carry=2.0, lower=1.5, release=0.8, retreat=1.2)
        # open/close in the air: replay the part of YOUR video pin_01 where you opened it in the air
        import json
        ev = json.loads((TASK / "results/pin_01_events.json").read_text())
        VIDEO = (19.5, 23.0)                                  # video seconds replayed (opens 20.0-20.8, 21.7-22.6)
        opens = [(a, b) for a, b in ev["open"] if VIDEO[0] <= a <= VIDEO[1]]
        tau_closed = spring_moment.spring_torque(0.0)                     # spring preload, jaws shut
        tau_open = spring_moment.spring_torque(0.008)                     # to hold the jaws 8 mm open
        z = np.array([0.0, 0.0, 1.0])
        paper_g = m.geom("paper").id
        paper_front = d.body("paper").xpos[0] - m.geom_size[paper_g][0]
        paper_top = d.geom_xpos[paper_g][2] + m.geom_size[paper_g][2]
        dt = 0.02
        steps = int(round(dt / m.opt.timestep))
        v = None
        if viewer:
            from mujoco import viewer as mj_viewer
            v = mj_viewer.launch_passive(m, d)
            v.cam.lookat[:] = p_pin
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.35, 200, -20
            wall = time.time()
        q = q_lap.copy()
        thumb_pitch, index_curl = PRE["thumb_pitch"], self.shape
        stage, stage_t0, still_for = None, None, 0.0
        index_touch = None
        pull_from = None
        place_shift = touch_down = lifted_at = hand_at = retreat_from = None
        thumb_rel = index_rel = None
        grip_d = hold_idx = hold_thb = hold_cmd = None
        left_paper = None
        blend_val = lambda a, b, s: a + (b - a) * s
        log = []
        t_end = None
        while True:
            t = len(log) * dt
            if t_end is not None and t >= t_end:
                break
            if t > 30:
                print("gave up: no progress after 30 s")
                break
            fingers = dict(self.SHAPE)
            if t < edges[-1]:
                i = int(np.searchsorted(edges, t, side="right") - 1)
                ph, s = names[i], smooth((t - edges[i]) / T[names[i]])
                if ph == "rest":
                    fingers = lap
                elif ph == "reach":
                    q = q_lap + (q_low - q_lap) * s
                    fingers = blend(lap, self.SHAPE, s)
                elif ph == "rise":
                    q, _ = self.ik(low + (pre - low) * s, up, q, self.SHAPE, point=self.PAD, fdir=fd)
                else:
                    q, _ = self.ik(pre + (pad_at - pre) * s, up, q, self.SHAPE, point=self.PAD, fdir=fd)
            else:
                if stage is None:
                    stage, stage_t0 = "thumb", t
                ph = stage
                f_thumb = self.contact_force("L_thumb_distal", "clothespin")
                f_index = self.contact_force("L_index_intermediate", "pin_leg_b")
                if stage == "thumb":                          # thumb up until it touches, then stop
                    if f_thumb > 0.05:
                        stage, stage_t0 = "index", t
                    else:
                        thumb_pitch = min(0.6, thumb_pitch + 0.004)
                elif stage == "index":                        # index down until it touches, then stop
                    if f_index > 0.05:
                        index_touch = index_curl
                        stage, stage_t0, still_for = "settle", t, 0.0
                    else:
                        index_curl = min(1.6, index_curl + 0.004)
                elif stage == "settle":                       # wait until the pin is still
                    pin_speed = np.linalg.norm(d.qvel[self.m.jnt_dofadr[self.m.joint("pin_free").id]:][:6])
                    still_for = still_for + dt if (pin_speed < 0.02 and f_thumb > 0 and f_index > 0) else 0.0
                    if still_for >= 0.3 or t - stage_t0 > 1.5:
                        stage, stage_t0 = "press", t
                elif stage == "press" and self.balanced:     # BOTH fingers squeeze, with EQUAL force
                    # (user's observation: pressing with the index alone tipped the whole pin around the paper
                    # edge instead of squeezing it). Each finger is nudged so its push follows the same target
                    # force, which ramps up: equal and opposite pushes leave nothing over to rotate the pin.
                    s = min(1.0, (t - stage_t0) / 2.0)
                    step = lambda err: float(np.clip(0.02 * err, -0.006, 0.006))
                    if self.moment:
                        # user's idea: open by MOMENT. The spring resists with torque tau about the hinge; a finger
                        # at distance d makes F*d. Ramp a target moment from the clamped spring torque up past
                        # what holds the jaws clear_mm open; each finger's force = moment / ITS OWN distance,
                        # so both legs get the SAME moment and nothing is left over to turn the whole pin.
                        tau = tau0 + (tau1 - tau0) * s
                        T = self.touches()
                        d_i = max(0.005, T["index"][0] / 1000) if "index" in T else self.along
                        d_t = max(0.005, T["thumb"][0] / 1000) if "thumb" in T else self.along
                        # Two conditions (user's moment idea + statics of the clipped pin):
                        #  1. OPEN: each finger's moment about the SPRING beats the spring: F*d >= tau
                        #  2. DON'T TURN: while clipped, the whole pin can only pivot about the PAPER CLAMP
                        #     (x_clamp, on the jaw side), so the moments about the clamp must cancel:
                        #     F_i * (d_i - x_clamp) = F_t * (d_t - x_clamp)
                        # -> the force RATIO comes from 2, the SIZE from 1.
                        a_i, a_t = d_i - x_clamp, d_t - x_clamp
                        scale = tau / min(a_t * d_i, a_i * d_t)
                        fi_target, ft_target = scale * a_t, scale * a_i
                        f_hold_i, f_hold_t = fi_target, ft_target
                    else:
                        fi_target = ft_target = 0.3 + (self.f_max - 0.3) * s
                    index_curl = min(1.6, index_curl + step(fi_target - f_index))
                    thumb_pitch = min(self.thumb_max, thumb_pitch + step(ft_target - f_thumb))
                    if self.jaws_mm() >= self.clear_mm or s >= 1.0:
                        jq = lambda n: float(d.qpos[m.jnt_qposadr[m.joint(n).id]])
                        index_curl = jq("L_index_proximal_joint")      # start the pull from where they are
                        thumb_pitch = jq("L_thumb_proximal_pitch_joint")
                        if not self.moment:
                            f_hold_i = f_hold_t = max(1.0, (f_index + f_thumb) / 2)   # keep squeezing while pulling
                        stage, stage_t0 = "pull", t
                elif stage == "press":                        # old way: the index presses, the thumb holds
                    s = smooth((t - stage_t0) / 1.5)
                    index_curl = index_touch + self.press * s
                    # ...but only until the jaws clear the 1 mm paper. Squeezing further made the index tip
                    # sweep toward the tail end and the thumb toward the spring: the pin scissored out.
                    if self.jaws_mm() >= self.clear_mm or s >= 1.0:
                        # freeze the index WHERE IT IS: its motor lags the command, so stopping the command
                        # alone let it keep curling ~0.25 s and squeeze the pin out
                        index_curl = float(d.qpos[m.jnt_qposadr[m.joint("L_index_proximal_joint").id]])
                        stage, stage_t0 = "pull", t
                elif stage == "pull":
                    s = smooth((t - stage_t0) / T_LATER["pull"])
                    if self.balanced and self.moment:
                        # while the jaws are still on the paper keep them OPEN by moment, from the fingers' CURRENT
                        # distances (if they slide toward the spring, the same force gives less moment, the jaws
                        # shut on the paper and it drags the pin back). Once off the paper: lock the grip.
                        step = lambda err: float(np.clip(0.02 * err, -0.006, 0.006))
                        # still on the paper = jaw tips not yet past the paper's front edge (geometry: with the
                        # jaws open the pin doesn't TOUCH the paper, so contact force can't tell)
                        pA, RA = self.pin_frame()
                        on_paper = (pA + RA @ np.array([0, 0, -self.P["Lj"]]))[0] > paper_front - 0.002
                        if not on_paper and left_paper is None:
                            left_paper = t                    # jaw tips past the paper edge: now close gently
                        # HOLD the jaws at a set opening: 6 mm while sliding off the paper, then lowered to 0 over
                        # 0.5 s once off (a controlled close - just dropping the force lost the pin). Moment = spring
                        # torque at that opening, plus a small correction if the opening drifts. The jaws are open
                        # around the paper, so the pin is free between the fingers -> EQUAL forces. (Pushing harder
                        # opened them to 12 mm and the wide-open pin tipped out of the fingers.)
                        target_mm = self.hold_open_mm
                        if left_paper is not None:
                            target_mm *= max(0.0, 1 - (t - left_paper) / 0.5)
                        T = self.touches()
                        ds = [T[k][0] / 1000 for k in ("index", "thumb") if k in T and T[k][2] > 0]
                        d_now = float(np.clip(np.mean(ds) if ds else self.along, 0.008, self.P["Lt"]))
                        tau_c = spring_moment.spring_torque(target_mm / 1000) + 0.004 * (target_mm - self.jaws_mm())
                        if target_mm <= 0:
                            tau_c = 0.7 * tau_closed          # shut: firm grip below the spring preload
                        f_c = max(0.5, tau_c / d_now)
                        index_curl = min(1.6, index_curl + step(f_c - f_index))
                        thumb_pitch = min(self.thumb_max, thumb_pitch + step(f_c - f_thumb))
                    elif self.balanced:                       # keep BOTH fingers pressing equally while pulling
                        step = lambda err: float(np.clip(0.02 * err, -0.006, 0.006))
                        index_curl = min(1.6, index_curl + step(f_hold_i - f_index))
                        thumb_pitch = min(self.thumb_max, thumb_pitch + step(f_hold_t - f_thumb))
                    if pull_from is None:
                        W = d.body("left_wrist_yaw_link")
                        pull_from = W.xpos + W.xmat.reshape(3, 3) @ self.PAD
                    q, _ = self.ik(pull_from + along_dir * 0.06 * s, up, q, dict(fingers, thumb_pitch=thumb_pitch,
                                                                                  index=index_curl), point=self.PAD, fdir=fd)
                    if s >= 1.0:
                        stage, stage_t0 = ("grip" if self.balanced and self.carry_on else "relax"), t
                        W = d.body("left_wrist_yaw_link")
                        hand_at = W.xpos + W.xmat.reshape(3, 3) @ self.PAD
                elif stage in ("grip", "lift", "cycle", "carry", "lower", "release", "retreat"):
                    # ---- after the unclip (user's plan): firm grip at the tail tips, lift, open/close like the
                    # video, put it back on the paper with the same grip, let go. The pin is FREE now (no paper
                    # holding it), so both fingers push with EQUAL force - unequal would push it out of the hand.
                    # Forces from the spring moment (spring_moment.py): F = torque / d, d = where the fingers are.
                    step = lambda err: float(np.clip(0.02 * err, -0.006, 0.006))
                    # where the fingers grip (fixed at the grip's start; using the LIVE distance made the force
                    # grow as the fingers slid toward the spring, which pushed them further - it ran away)
                    if grip_d is None:
                        T = self.touches()
                        ds = [T[k][0] / 1000 for k in ("index", "thumb") if k in T and T[k][2] > 0]
                        grip_d = float(np.clip(np.mean(ds) if ds else self.along, 0.010, self.P["Lt"]))
                        jq = lambda n: float(d.qpos[m.jnt_qposadr[m.joint(n).id]])
                        hold_idx, hold_thb = jq("L_index_proximal_joint"), jq("L_thumb_proximal_pitch_joint")
                    d_avg = grip_d
                    f_grip = 0.7 * tau_closed / d_avg                 # firm, but below the preload: stays SHUT
                    f_open = 1.2 * tau_open / d_avg                   # beats the spring: jaws ~8 mm open
                    f_target, target = f_grip, None
                    dur = T_LATER.get(stage, VIDEO[1] - VIDEO[0])
                    s = smooth(min(1.0, (t - stage_t0) / dur))
                    if stage == "lift":
                        target = hand_at + z * 0.08 * s
                    elif stage == "cycle":                            # video time -> open (FIRM) or closed (grip)
                        vt = VIDEO[0] + (t - stage_t0)
                        w = max([min(1.0, (vt - a + 0.15) / 0.3, (b + 0.15 - vt) / 0.3) for a, b in opens] + [0.0])
                        open_mm = self.air_open_mm * max(0.0, w)       # target opening, following your video
                        target = hand_at + z * 0.08
                    elif stage in ("carry", "lower"):
                        if place_shift is None:                       # where the pin must go: jaw end on the paper,
                            pA, RA = self.pin_frame()                 # tail tips overhanging the paper edge
                            tail_end = pA + RA @ np.array([0, 0, self.P["Lt"]])
                            face = pA + RA @ np.array([0, -self.P["h"] / 2 - self.P["t"], 0])
                            place_shift = np.array([(paper_front - self.overhang) - tail_end[0], 0.0,
                                                    paper_top - face[2]])
                            lifted_at = hand_at + z * 0.08
                        above = lifted_at + place_shift + z * 0.03
                        if stage == "carry":
                            target = lifted_at + (above - lifted_at) * s
                        else:                                         # lower until the pin touches the paper
                            f_paper = self.contact_force("paper", "clothespin") + self.contact_force("paper", "pin_leg_b")
                            if f_paper > 0.1 and touch_down is None:
                                touch_down = t
                            target = above - z * 0.032 * s if touch_down is None else None
                    elif stage == "release":                          # open the fingers: pin rests on the paper
                        thumb_pitch = blend_val(thumb_rel, PRE["thumb_pitch"], s)
                        index_curl = blend_val(index_rel, self.shape - 0.2, s)
                    elif stage == "retreat":                          # hand back toward the robot and up
                        target = retreat_from + (-along_dir * 0.0 + np.array([-0.06, 0, 0.04])) * s
                    if stage == "cycle" and open_mm > 0.3:
                        # opening in the air: CONTROL the opening. The spring is preloaded, so its torque hardly
                        # grows as it opens - any steady push above the preload snapped it fully open (13.5 mm, tails
                        # touching) and the pin was lost. Moment = spring torque at the TARGET opening + a correction
                        # from the measured opening; equal forces (the pin is free).
                        tau_c = spring_moment.spring_torque(open_mm / 1000) + 0.004 * (open_mm - self.jaws_mm())
                        f_c = max(f_grip, tau_c / d_avg)
                        index_curl = min(1.6, index_curl + step(f_c - f_index))
                        thumb_pitch = min(self.thumb_max, thumb_pitch + step(f_c - f_thumb))
                    elif stage not in ("release", "retreat") and (self.hold_mode == "force" or
                                                                    (stage == "grip" and t - stage_t0 < 0.6)):
                        # squeeze to the grip force: equal FORCE on both fingers, fixed at the grip's start (from the
                        # spring preload and grip_d - NOT the live distance, which grew the force as the fingers slid
                        # toward the spring and ran away)
                        index_curl = min(1.6, index_curl + step(f_grip - f_index))
                        thumb_pitch = min(self.thumb_max, thumb_pitch + step(f_grip - f_thumb))
                    elif stage not in ("release", "retreat"):
                        # then LOCK the motor COMMANDS (not the measured angles): the motors keep pushing ~f_grip
                        # against the pin, but the fingers stop moving. Holding by force control kept re-closing the
                        # fingers as the shut pin (a wedge, thin at the jaws) slipped, and each re-close swept the
                        # fingertips further toward the jaws - they ended up squeezing the jaws, which can't open it.
                        # Locking the MEASURED angles left no push at all (~0.3 N) and the lift shook it loose.
                        if hold_cmd is None:
                            hold_cmd = (index_curl, thumb_pitch)
                        index_curl += float(np.clip(hold_cmd[0] - index_curl, -0.006, 0.006))
                        thumb_pitch += float(np.clip(hold_cmd[1] - thumb_pitch, -0.006, 0.006))
                    if target is not None:
                        q, _ = self.ik(target, up, q, dict(fingers, thumb_pitch=thumb_pitch, index=index_curl),
                                       point=self.PAD, fdir=fd)
                    if stage != "lower" or touch_down is None:
                        done = (t - stage_t0) >= dur
                    else:
                        done = t - touch_down >= 0.3                  # touched the paper: settle, then let go
                    if done and stage == self.stop_after:
                        t_end = t                             # hand over to another script (e.g. clothespin_reclip)
                    if done:
                        nxt = {"grip": "lift", "lift": "cycle", "cycle": "carry", "carry": "lower",
                               "lower": "release", "release": "retreat", "retreat": None}[stage]
                        if nxt == "release":
                            thumb_rel, index_rel = thumb_pitch, index_curl
                        if nxt == "retreat":
                            W = d.body("left_wrist_yaw_link")
                            retreat_from = W.xpos + W.xmat.reshape(3, 3) @ self.PAD
                        if nxt is None:
                            t_end = t + 1.0                           # watch the pin lie still for 1 s
                            stage = "rest_after"
                        else:
                            stage, stage_t0 = nxt, t
                elif stage == "rest_after":
                    pass
                elif stage == "relax":                        # ease the index off: jaws close, still held
                    s = smooth((t - stage_t0) / T_LATER["relax"])
                    index_curl = index_touch + self.press * (1 - 0.6 * s)
                    if s >= 1.0:
                        stage, stage_t0 = "hold", t
                        t_end = t + T_LATER["hold"]
                fingers["thumb_pitch"], fingers["index"] = thumb_pitch, index_curl
            for j, a in enumerate(self.aid):
                d.ctrl[a] = q[j]
            command(m, d, fingers)
            for _ in range(steps):
                mujoco.mj_step(m, d)
            if v is not None:
                if not v.is_running():
                    break
                v.sync()
                lag = (t + dt) * self.slowmo - (time.time() - wall)
                if lag > 0:
                    time.sleep(lag)
            pin_xyz, Rp = self.pin_frame()
            rec = dict(t=t, phase=ph, qpos=d.qpos.copy(), jaws=self.jaws_mm(), touch=self.touches(), pad=self.index_pad_on_pin(), thumb=self.thumb_on_pin(),
                       f_index=self.contact_force("L_index_intermediate", "pin_leg_b"),
                       f_thumb=self.contact_force("L_thumb_distal", "clothespin"),
                       pin_x=pin_xyz[0], faces_up=abs(Rp[:, 1] @ up),
                       held=bool(self.contact_force("L_index_intermediate", "pin_leg_b") > 0.2
                                 and self.contact_force("L_thumb_distal", "clothespin") > 0.2),
                       pin_z=pin_xyz[2],
                       legs=(Rp[:, 2].copy(), d.body("pin_leg_b").xmat.reshape(3, 3)[:, 2].copy()),
                       on_paper=self.contact_force("paper", "clothespin") + self.contact_force("paper", "pin_leg_b"))
            log.append(rec)
            if trace and len(log) % 5 == 0 and ph in ("insert", "thumb", "press", "pull"):
                p = rec["pad"]
                th = rec["thumb"]
                print(f"   thumb tip: across {th[0]:+5.1f} mm, {th[2]:5.1f} mm from spring")
                print(f"t={t:4.1f} {ph:6s} index pad: across {p[0]:+5.1f} mm, {p[1]:+5.1f} mm above top, "
                      f"{p[2]:5.1f} mm from spring | force index {rec['f_index']:.1f} N, thumb {rec['f_thumb']:.1f} N"
                      f" | jaws {rec['jaws']:.1f} mm")
        if v is not None:
            print("finished - close the MuJoCo window to see the results")
            while v.is_running():
                v.sync()
                time.sleep(0.05)
        # state for scripts that continue from here (clothespin_reclip): arm targets, finger commands, frames
        self.state = dict(q=q.copy(), index_curl=index_curl, thumb_pitch=thumb_pitch, up=up, along_dir=along_dir,
                          fd=fd, hold_cmd=hold_cmd, grip_d=grip_d, p_pin0=p_pin, R_pin0=R_pin, t=t,
                          paper_front=paper_front, paper_top=paper_top)
        return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true", help="watch in the MuJoCo viewer (run with mjpython)")
    ap.add_argument("--slow", type=float, default=1.0)
    ap.add_argument("--trace", action="store_true", help="print the index pad position and forces")
    ap.add_argument("--shape", type=float, default=0.75, help="index bend while placing it on the tail")
    ap.add_argument("--press", type=float, default=0.35, help="extra index curl when pressing")
    ap.add_argument("--along", type=float, default=None, help="where to press, mm from the spring toward the tail "
                                                                 "(default: just past the middle of the tail)")
    ap.add_argument("--across", type=float, default=0.0, help="mm from the centre line across the width")
    ap.add_argument("--opposed", action="store_true", help="aim the thumb-index meeting point at the tail middle")
    ap.add_argument("--yaw", type=float, default=None, help="which way the fingers point (deg, 0 = toward the paper)")
    ap.add_argument("--thumb_yaw", type=float, default=None, help="how far the thumb swings across the palm")
    ap.add_argument("--index_only", action="store_true", help="old press: only the index pushes (tips the pin)")
    ap.add_argument("--equal_force", action="store_true", help="both fingers push with equal force (not equal moment)")
    ap.add_argument("--open_mm", type=float, default=6.0, help="jaw opening in the air (default 6 = widest without "
                    "the fingers wobbling: 3-6 mm fingers move ~3 mm, pin tilts <10 deg; 7+ mm wobbles; 9+ drops it)")
    ap.add_argument("--hold", default="lock", choices=["force", "lock"], help="how the grip holds after the unclip")
    ap.add_argument("--best", action="store_true", help="use the best settings from the opposition search")
    ap.add_argument("--full", action="store_true", help="the placement that passes the FULL task (standard pin): "
                    "unclip, grip on the tips, lift, open/close like pin_01, place, release")
    a = ap.parse_args()
    if a.full:
        a.shape, a.across, a.along, a.yaw, a.thumb_yaw = 0.622, 5.0, 23.5, -36.0, 1.2
    if a.best:
        import json
        b = json.loads((TASK / "results/oppose_search.json").read_text())[0]["params"]
        a.shape, a.along, a.across, a.yaw, a.thumb_yaw = b["shape"], b["along"], b["across"], b["yaw"], b["thumb_yaw"]
        print("using the best settings from the opposition search:", b)
    u = Press(a.shape, a.press, along=None if a.along is None else a.along / 1000, across=a.across / 1000, opposed=a.opposed,
              yaw_deg=a.yaw, thumb_yaw=a.thumb_yaw)
    u.balanced = not a.index_only
    u.moment = not a.equal_force
    u.hold_mode = a.hold
    u.air_open_mm = a.open_mm
    u.slowmo = a.slow
    log = u.run(viewer=a.viewer, trace=a.trace)
    by = lambda ph: [r for r in log if r["phase"] == ph]
    paper_front = u.d.body("paper").xpos[0] - u.m.geom_size[u.m.geom("paper").id][0]
    press = by("press")
    if not press:
        T = log[-1]["touch"]
        print(f"never reached the press: stuck in stage '{log[-1]['phase']}' (index touching "
              f"{'index' in T and T['index'][2] > 0}, thumb touching {'thumb' in T and T['thumb'][2] > 0})")
        print("UNCLIP: FAIL")
        return
    pad = press[-1]["pad"]
    print(f"index pad at the end of the press: {pad[0]:+.1f} mm across (0 = middle), {pad[2]:.1f} mm from the spring")
    print(f"press: jaws open up to {max(r['jaws'] for r in press):.1f} mm "
          f"(index {max(r['f_index'] for r in press):.1f} N, thumb {max(r['f_thumb'] for r in press):.1f} N)")
    off = by("pull")[-1]["pin_x"] + u.P["Lj"] < paper_front + 0.001
    print(f"pull : pin {'OFF' if off else 'still ON'} the paper, held {by('pull')[-1]['held']}")
    ok = max(r["jaws"] for r in press) >= u.clear_mm - 0.5 and off and by("pull")[-1]["held"]
    print("UNCLIP:", "PASS" if ok else "FAIL")
    if not (u.balanced and u.carry_on):
        return
    g, lf, cy = by("grip"), by("lift"), by("cycle")
    if g:
        print(f"grip : held {g[-1]['held']}, jaws {g[-1]['jaws']:.1f} mm (shut = 0), index at "
              f"{g[-1]['touch'].get('index', (0,))[0]:.0f} mm / thumb at {g[-1]['touch'].get('thumb', (0,))[0]:.0f} mm from the spring")
    if lf:
        print(f"lift : pin raised {(lf[-1]['pin_z'] - lf[0]['pin_z']) * 1000:.0f} mm, held {lf[-1]['held']}")
    ok_cycle = False
    if cy:
        t0 = cy[0]["t"]
        import json
        ev = json.loads((TASK / "results/pin_01_events.json").read_text())
        opens = [(a, b) for a, b in ev["open"] if 19.5 <= a <= 23.0]
        ok_cycle = all(r["held"] for r in cy)
        for a, b in opens:
            ins = [r["jaws"] for r in cy if a + 0.15 <= 19.5 + r["t"] - t0 <= b]
            print(f"cycle: you opened it {a:.1f}-{b:.1f} s -> robot jaws up to {max(ins, default=0):.1f} mm "
                  f"(target {u.air_open_mm:.0f})")
            ok_cycle &= max(ins, default=0) > 0.6 * u.air_open_mm
        shut = [r["jaws"] for r in cy if all(not (a - 0.3 <= 19.5 + r["t"] - t0 <= b + 0.5) for a, b in opens)]
        print(f"cycle: between opens jaws at most {max(shut, default=0):.1f} mm (shut = 0); held throughout "
              f"{all(r['held'] for r in cy)}")
        ok_cycle &= max(shut, default=9) < 1.0
        # WOBBLE: how much each fingertip moves on the pin, and how much the pin tilts, over the open/close
        wob = {}
        for k in ("index", "thumb"):
            pts = np.array([r["touch"][k][:2] for r in cy if k in r["touch"] and r["touch"][k][2] > 0.2])
            wob[k] = float(np.ptp(pts, axis=0).max()) if len(pts) > 1 else 99.0
        # pin tilt = its CENTRE LINE (between the two legs) vs just before the first open: the legs themselves
        # must rotate when the jaws open, so a single leg's angle would count the opening as wobble
        mid = lambda r: (r["legs"][0] + r["legs"][1]) / np.linalg.norm(r["legs"][0] + r["legs"][1])
        m0 = mid(cy[0])
        tilt = max(float(np.degrees(np.arccos(np.clip(mid(r) @ m0, -1, 1)))) for r in cy)
        print(f"wobble: index moved {wob['index']:.1f} mm, thumb moved {wob['thumb']:.1f} mm on the pin, "
              f"pin tilted up to {tilt:.1f} deg")
    lo = by("lower")
    if lo:
        print(f"place: pin touched the paper {max(r['on_paper'] for r in lo) > 0.1}, held while lowering {lo[-1]['held']}")
    end = log[-1]
    pin_xyz, Rp = u.pin_frame()
    on = end["on_paper"] > 0.02 and pin_xyz[0] > paper_front
    print(f"end  : pin resting on the paper {on}, flat {end['faces_up']:.2f}, jaws {end['jaws']:.1f} mm, "
          f"fingers off it {not end['held']}")
    print("FULL TASK:", "PASS" if ok and ok_cycle and on and end["faces_up"] > 0.9 else "FAIL")


if __name__ == "__main__":
    main()
