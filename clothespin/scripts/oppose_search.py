"""Search for a hand placement where the THUMB and INDEX touch the pin directly OPPOSITE each other
(same spot along and across the pin) - so pressing squeezes the tails instead of twisting the pin.

    python clothespin/scripts/oppose_search.py --samples 600
    python clothespin/scripts/unclip_press.py --best              # full unclip with the best result
    mjpython clothespin/scripts/unclip_press.py --best --viewer --slow 4

Fast (kinematics only, the pin held at its settled pose): for each candidate
    1. place the hand: index pad 2 mm above the top tail at (across, along), fingers pointing `yaw`
       degrees around the vertical, thumb swung `thumb_yaw` across the palm, index bent to `shape`
    2. bend the thumb up step by step until it would touch the bottom leg -> thumb contact point
    3. curl the index until it would touch the top leg -> index contact point
Score (mm, lower is better): distance between the two contact points (along + across), plus how far
the index is from the tail's middle. Rejected: arm can't reach, hand would hit the table/paper, or the
thumb hits something before the pin.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

TASK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TASK / "scripts"))
from unclip_press import Press  # noqa: E402

OUT = TASK / "results" / "oppose_search.json"
NAMES = ["yaw", "thumb_yaw", "shape", "across", "along"]
LO = np.array([-180.0, 0.4, 0.5, -3.0, 4.0])
HI = np.array([180.0, 1.3, 1.0, 3.0, 10.0])
sys.path.insert(0, str(TASK / "sim"))
import pin_model  # noqa: E402
_P = pin_model.pin_params()
TAIL_MID = 0.58 * _P["Lt"] * 1000                # mm from the spring: just past the tail's middle
LO[4], HI[4] = 0.38 * _P["Lt"] * 1000, 0.92 * _P["Lt"] * 1000     # aim along the tail (mm from the spring)
LO[3], HI[3] = -0.4 * _P["w"] * 1000, 0.4 * _P["w"] * 1000         # aim across the tail


class Searcher:
    def __init__(self):
        self.u = Press()
        u, m, d = self.u, self.u.m, self.u.d
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        for _ in range(500):                     # pin settled on the paper (clipped 1.6 cm)
            mujoco.mj_step(m, d)
        self.base = d.qpos.copy()
        self.q_lap = d.qpos[u.adr].copy()
        self.p_pin, self.R_pin = u.pin_frame()
        self.up, self.along_dir = self.R_pin[:, 1], self.R_pin[:, 2]
        G = lambda b: [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(b).id and m.geom_contype[g]]
        self.thumb = G("L_thumb_distal")
        self.index = G("L_index_intermediate")
        self.leg_a, self.leg_b = m.geom("pin_leg_a").id, m.geom("pin_leg_b").id
        self.others = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in ("table", "paper")]

    def closest(self, fingers, leg):
        m, k = self.u.m, self.u.k
        best = (9, None)
        for g in fingers:
            ft = np.zeros(6)
            dd = mujoco.mj_geomDistance(m, k, g, leg, 0.1, ft)
            if dd < best[0]:
                best = (dd, ft[:3].copy())
        return best

    def in_pin(self, p):
        return self.R_pin.T @ (p - self.p_pin) * 1000          # mm in the pin frame (x across, z along)

    def evaluate(self, x):
        yaw, thumb_yaw, shape, across, along = x
        u, m, k = self.u, self.u.m, self.u.k
        u.yaw_deg, u.thumb_yaw, u.shape = yaw, thumb_yaw, shape
        u.across, u.along = across / 1000, along / 1000
        u.SHAPE = dict(u.SHAPE, index=shape, thumb_yaw=thumb_yaw, thumb_pitch=-0.1)
        u.PAD = u._index_pad()
        u.d.qpos[:] = self.base
        mujoco.mj_forward(m, u.d)
        target = u.pad_target() + self.up * 0.002
        fd = u.finger_dir(self.up, self.along_dir)
        q, clear = u.ik(target, self.up, self.q_lap, u.SHAPE, seeds=6, point=u.PAD, fdir=fd)
        if not u.last_ok:
            return None, "unreachable"
        if clear < 0.002:
            return None, "hand hits table/paper"
        # place it, then bend the thumb up until it touches the bottom leg
        k.qpos[:] = self.base
        k.qpos[u.adr] = q
        thumb_pt = None
        for tp in np.arange(-0.1, 0.62, 0.01):
            u._set_fingers(k, dict(u.SHAPE, thumb_pitch=tp))
            mujoco.mj_kinematics(m, k)
            mujoco.mj_collision(m, k) if False else None
            dd, pt = self.closest(self.thumb, self.leg_a)
            if dd <= 0.0005:
                thumb_pt = pt
                break
            # thumb hitting the top leg or the table before the bottom leg = wrong side / blocked
            if self.closest(self.thumb, self.leg_b)[0] <= 0.0 or min(
                    mujoco.mj_geomDistance(m, k, g, o, 0.02, np.zeros(6)) for g in self.thumb for o in self.others) <= 0:
                return None, "thumb blocked"
        if thumb_pt is None:
            return None, "thumb never reaches the pin"
        index_pt = None
        for ic in np.arange(shape, 1.6, 0.01):
            u._set_fingers(k, dict(u.SHAPE, thumb_pitch=tp, index=ic))
            mujoco.mj_kinematics(m, k)
            dd, pt = self.closest(self.index, self.leg_b)
            if dd <= 0.0005:
                index_pt = pt
                break
        if index_pt is None:
            return None, "index never reaches the pin"
        t, i = self.in_pin(thumb_pt), self.in_pin(index_pt)
        apart = np.hypot(t[0] - i[0], t[2] - i[2])           # thumb vs index, across + along
        off_mid = np.hypot(i[0], i[2] - TAIL_MID)              # index vs middle of the tail
        edge = max(0.0, abs(t[0]) - 0.35 * _P['w'] * 1000) + max(0.0, abs(i[0]) - 0.35 * _P['w'] * 1000)  # near the edges
        score = apart + 0.5 * off_mid + 2.0 * edge
        return score, dict(thumb_across=round(t[0], 1), thumb_along=round(t[2], 1),
                           index_across=round(i[0], 1), index_along=round(i[2], 1),
                           apart_mm=round(apart, 1), clearance_mm=round(clear * 1000, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=600)
    ap.add_argument("--refine", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mid", type=float, default=None, help="where the fingers should meet, mm from the spring "
                    "(default: just past the tail's middle; user: on the tail TIPS -> e.g. 27 for the 30 mm tail)")
    a = ap.parse_args()
    global TAIL_MID
    if a.mid is not None:
        TAIL_MID = a.mid
        LO[4], HI[4] = a.mid - 6, min(a.mid + 2, _P["Lt"] * 1000 - 1)
    s = Searcher()
    rng = np.random.default_rng(a.seed)
    res, why = [], {}
    t0 = time.time()

    def record(x):
        sc, info = s.evaluate(x)
        if sc is None:
            why[info] = why.get(info, 0) + 1
            return
        res.append(dict(score=round(float(sc), 2), params={n: round(float(v), 3) for n, v in zip(NAMES, x)}, **info))

    for k in range(a.samples):
        record(LO + rng.random(5) * (HI - LO))
        if (k + 1) % 100 == 0:
            b = min(res, key=lambda r: r["score"]) if res else None
            print(f"{k + 1}/{a.samples} ({time.time() - t0:.0f} s) valid {len(res)}; best "
                  f"{b['apart_mm'] if b else '-'} mm apart; rejected {why}", flush=True)
    for k in range(a.refine):
        top = sorted(res, key=lambda r: r["score"])[:5]
        base = np.array([top[k % len(top)]["params"][n] for n in NAMES])
        record(np.clip(base + rng.normal(0, 0.06, 5) * (HI - LO), LO, HI))
    res.sort(key=lambda r: r["score"])
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("best 5 (thumb vs index, mm in the pin frame: across / along from the spring):")
    for r in res[:5]:
        print(f"  apart {r['apart_mm']} mm | thumb {r['thumb_across']}/{r['thumb_along']} | index "
              f"{r['index_across']}/{r['index_along']} | {r['params']}")
    print(f"saved {OUT.relative_to(TASK.parent)}")


if __name__ == "__main__":
    main()
