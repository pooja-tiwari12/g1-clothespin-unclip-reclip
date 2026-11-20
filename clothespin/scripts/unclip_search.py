"""Physics-scored search for the unclip: every candidate runs the FULL real sequence on the clipped pin
(reach -> thumb up until touch -> index down until touch -> settle -> press -> pull -> relax -> hold),
exactly like `unclip_press.py`. Same idea as the mug's grasp_search.py.

    python clothespin/scripts/unclip_search.py --samples 500 --refine 300
    python clothespin/scripts/unclip_press.py --best                  # rerun the winner, print numbers
    mjpython clothespin/scripts/unclip_press.py --best --viewer --slow 4

Why physics and not the faster kinematic oppose_search.py: in physics the hand settles a few mm away
from its planned spot, so the index landed near the spring (low leverage) while the kinematics said
"opposite". Only a real run shows where the fingers really press.

Settings searched:
    yaw        which way the fingers point around the vertical (deg)
    thumb_yaw  how far the thumb swings across the palm
    shape      how bent the index is while it hovers
    across     index aim across the tail (mm, 0 = centre line)
    along      index aim along the tail (mm from the spring; the tail is 13 mm, middle 7.5)
    hover      index height above the tail before the thumb comes up (mm) - so the index doesn't touch first
    press      (index-only press only; the default press squeezes both fingers with equal force)
    clear      stop pressing once the jaws are this far open (mm), then freeze the index and pull
Score: success (jaws > 6 mm, pin off the paper, still held, upright) first; then off the paper, upright,
how far the jaws opened, the opening torque reached, and how far apart the thumb and index press along
the pin (apart = a twisting pair that rotates the pin instead of squeezing it).
"""

import argparse
import contextlib
import io
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TASK / "scripts"))

OUT = TASK / "results" / "unclip_search.json"
NAMES = ["yaw", "thumb_yaw", "shape", "across", "along", "hover", "press", "clear"]
LO = np.array([-60.0, 0.9, 0.5, -3.0, 5.0, 0.5, 0.2, 3.0])
HI = np.array([30.0, 1.3, 1.0, 3.0, 12.0, 5.0, 0.7, 12.0])
sys.path.insert(0, str(TASK / "sim"))
import pin_model  # noqa: E402
_P = pin_model.pin_params()
LO[4], HI[4] = 0.38 * _P["Lt"] * 1000, 0.92 * _P["Lt"] * 1000     # aim along the tail (mm from the spring)
LO[3], HI[3] = -0.4 * _P["w"] * 1000, 0.4 * _P["w"] * 1000         # aim across the tail
SPRING_TORQUE = _P["k"] * 0.6                                       # preload: torque to start opening


def evaluate(x):
    import unclip_press as UP                              # imported in the worker
    p = dict(zip(NAMES, map(float, x)))
    try:
        u = UP.Press(p["shape"], p["press"], hover=p["hover"] / 1000, along=p["along"] / 1000,
                     across=p["across"] / 1000, yaw_deg=p["yaw"], thumb_yaw=p["thumb_yaw"])
        u.slowmo = 1
        u.clear_mm = p.get("clear", 4.0)                   # stop pressing at this jaw opening (mm)
        with contextlib.redirect_stdout(io.StringIO()):
            log = u.run()
    except Exception as e:                                  # unreachable placement etc.
        return dict(params=p, score=-1.0, error=str(e)[:80])
    press = [r for r in log if r["phase"] == "press"]
    pull = [r for r in log if r["phase"] == "pull"]
    if not press:
        return dict(params=p, score=-1.0, error="never reached the press stage")
    paper_front = u.d.body("paper").xpos[0] - u.m.geom_size[u.m.geom("paper").id][0]
    off = bool(pull) and pull[-1]["pin_x"] + u.P["Lj"] < paper_front + 0.001
    end = log[-1]
    jaws = max(r["jaws"] for r in press)
    torque = max(r["touch"]["torque"] for r in press)
    first = next((r["touch"] for r in log if r["phase"] == "settle"), {})
    unstable = int(u.d.warning[3].number)
    off = bool(off)
    success = bool(jaws >= 3.5 and off and end["held"] and end["faces_up"] > 0.8 and unstable == 0)
    # thumb vs index along the pin when the press starts: apart = twisting pair (rotates the pin)
    ia, ta = first.get("index", (0, 0, 0)), first.get("thumb", (0, 0, 0))
    apart = abs(ia[0] - ta[0]) if ia[2] > 0 and ta[2] > 0 else 15.0
    score = (100 * success + 10 * off * end["held"] + 5 * off + min(jaws, 10) + 5 * min(torque / SPRING_TORQUE, 1.5)
             - 0.5 * apart + 5 * (end["faces_up"] > 0.8))
    return dict(params={k: round(v, 3) for k, v in p.items()}, score=round(float(score), 2), success=success,
                jaws_mm=round(float(jaws), 1), torque=round(float(torque), 4), off_paper=off, held=bool(end["held"]),
                faces_up=round(float(end["faces_up"]), 2), unstable=unstable, apart_mm=round(float(apart), 1),
                index_at=[round(float(v), 1) for v in first.get("index", (0, 0, 0))],
                thumb_at=[round(float(v), 1) for v in first.get("thumb", (0, 0, 0))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--refine", type=int, default=300)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    res, t0 = [], time.time()

    def report(tag):
        best = sorted(res, key=lambda r: -r["score"])[:1]
        b = best[0] if best else {}
        print(f"{tag} ({time.time() - t0:.0f} s): {len(res)} runs, {sum(r.get('success', False) for r in res)} "
              f"successes; best jaws {b.get('jaws_mm')} mm, torque {b.get('torque')} N*m", flush=True)

    with Pool(a.workers) as pool:
        X = [LO + rng.random(len(LO)) * (HI - LO) for _ in range(a.samples)]
        seeds = TASK / "results" / "oppose_search.json"       # start from placements where the fingers oppose
        if seeds.exists():
            for r in json.loads(seeds.read_text())[:20]:
                for _ in range(3):                             # each with random hover / press / opening
                    x = LO + rng.random(len(LO)) * (HI - LO)
                    for k, n in enumerate(NAMES):
                        if n in r["params"]:
                            x[k] = r["params"][n]
                    X.append(np.clip(x, LO, HI))
        for i, r in enumerate(pool.imap_unordered(evaluate, X, chunksize=2)):
            res.append(r)
            if (i + 1) % 50 == 0:
                report(f"random {i + 1}/{len(X)}")
        rounds = max(1, a.refine // 50)
        for k in range(rounds):                             # refine around the current best ten
            top = sorted(res, key=lambda r: -r["score"])[:10]
            sigma = 0.08 * (1 - 0.6 * k / rounds)
            X = []
            for j in range(50):
                base = np.array([top[j % len(top)]["params"][n] for n in NAMES])
                X.append(np.clip(base + rng.normal(0, sigma, len(LO)) * (HI - LO), LO, HI))
            res.extend(pool.map(evaluate, X, chunksize=2))
            report(f"refine {k + 1}/{rounds}")
            OUT.write_text(json.dumps(sorted(res, key=lambda r: -r["score"]), indent=1))
    res.sort(key=lambda r: -r["score"])
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("best 5:")
    for r in res[:5]:
        print(" ", {k: r.get(k) for k in ("success", "jaws_mm", "apart_mm", "off_paper", "held", "faces_up",
                                          "index_at", "thumb_at")}, r["params"])
    print(f"saved {OUT.relative_to(TASK.parent)}")


if __name__ == "__main__":
    main()
