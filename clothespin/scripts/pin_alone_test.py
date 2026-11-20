"""Pin model on its own (paper + pin, no robot): the three checks the pin must pass.

    python clothespin/scripts/pin_alone_test.py

    1. REST:    clipped on the paper and left alone -> stays clipped, does not drift
    2. SQUEEZE: equal and opposite forces on the two tails -> jaws open
    3. RELEASE: forces removed -> jaws snap shut on the paper again
Uses the same physics settings as the robot scene (timestep, integrator, the robot's joint defaults),
so whatever passes here behaves the same in the full scene.
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

TASK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TASK / "sim"))
import pin_model as pm  # noqa: E402

TH = 0.001       # paper thickness (same as the scene)
# how far the jaws are on the paper: same rule as build_pin_scene.jaw_bite


def build():
    P = pm.pin_params()
    q_clip = pm.hinge_for_jaw_gap(TH, P)
    R = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.ravel())
    z0 = 0.5
    bite = min(0.55 * P["Lj"], 0.022)
    pos = np.array([bite - P["Lj"], 0, z0 + P["h"] / 2])
    xml = f"""<mujoco><compiler angle="radian"/>
<option timestep="0.001" integrator="implicitfast"/>
<default><joint armature="0.01" damping="0.5" frictionloss="0.05"/></default>  <!-- robot scene defaults -->
<asset>{pm.PIN_MATERIAL}</asset><worldbody>
<geom name="paper" type="box" size="0.1 0.1 {TH / 2}" pos="0.1 0 {z0 + TH / 2}" friction="0.6 0.005 0.0001" solref="0.002 1" solimp="0.95 0.99 0.0001"/>
{pm.pin_xml(pos, quat, P, free=True, clamp_q=q_clip)}
</worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    fa = m.jnt_qposadr[m.joint("pin_free").id]
    d.qpos[fa:fa + 3], d.qpos[fa + 3:fa + 7] = pos, quat
    d.qpos[m.jnt_qposadr[m.joint("pin_hinge").id]] = q_clip
    mujoco.mj_forward(m, d)
    return m, d, P, pos


def main():
    m, d, P, pos0 = build()
    print(f"pin: {P['name']}")
    fa = m.jnt_qposadr[m.joint("pin_free").id]
    ha = m.jnt_qposadr[m.joint("pin_hinge").id]
    A, B = m.body("clothespin").id, m.body("pin_leg_b").id
    jaws = lambda: pm.jaw_opening(d.qpos[ha], P) * 1000
    ok = True

    for _ in range(2000):                                   # 1. rest 2 s
        mujoco.mj_step(m, d)
    drift = np.linalg.norm(d.qpos[fa:fa + 3] - pos0) * 1000
    R0 = d.body("clothespin").xmat.reshape(3, 3).copy()
    print(f"1. REST    2 s: drifted {drift:.1f} mm, jaws {jaws():.2f} mm on the 1 mm paper, NaN {np.isnan(d.qpos).any()}")
    ok &= drift < 5 and abs(jaws() - TH * 1000) < 0.3

    up = R0[:, 1]
    at = 0.58 * P["Lt"]                                     # where the fingers press: just past the tail's middle
    F_full = P["F"] / 0.58                                  # force there that opens it fully (estimate)
    for F in (0.6 * F_full, 1.0 * F_full, 1.4 * F_full):    # 2. squeeze
        for _ in range(800):
            RA, RB = d.body("clothespin").xmat.reshape(3, 3), d.body("pin_leg_b").xmat.reshape(3, 3)
            pA = d.body("clothespin").xpos + RA @ np.array([0, -P["h"] / 2 - P["t"], at])
            pB = d.body("pin_leg_b").xpos + RB @ np.array([0, P["t"], at])
            d.qfrc_applied[:] = 0
            mujoco.mj_applyFT(m, d, -RA[:, 1] * F, np.zeros(3), pB, B, d.qfrc_applied)
            mujoco.mj_applyFT(m, d, RA[:, 1] * F, np.zeros(3), pA, A, d.qfrc_applied)
            mujoco.mj_step(m, d)
        print(f"   [instability warnings so far: {d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number}]")
        print(f"2. SQUEEZE {F:.1f} N on each tail ({at * 1000:.0f} mm from the spring): jaws {jaws():.1f} mm, NaN {np.isnan(d.qpos).any()}")
    print(f"   (this pin opens ~{P['jaw_max'] * 1000:.0f} mm at most - tails touching)")
    ok &= 0.5 * P["jaw_max"] * 1000 < jaws() < 1.1 * P["jaw_max"] * 1000 and not np.isnan(d.qpos).any()

    d.qfrc_applied[:] = 0                                   # 3. release
    for _ in range(1000):
        mujoco.mj_step(m, d)
    print(f"3. RELEASE 1 s: jaws {jaws():.2f} mm (back on the paper = ~1.0), NaN {np.isnan(d.qpos).any()}")
    ok &= jaws() < 1.5 and not np.isnan(d.qpos).any()
    unstable = d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number
    print(f"instability warnings: {unstable}")
    ok &= unstable == 0 and jaws() < 13
    print("PIN MODEL:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
