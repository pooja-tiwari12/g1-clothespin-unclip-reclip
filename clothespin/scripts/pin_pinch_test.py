"""Stage-1 feasibility: can the G1's left Inspire hand pinch the mini clothespin open?

    python clothespin/scripts/pin_pinch_test.py          # run the test, print the numbers
    python clothespin/scripts/pin_pinch_test.py --show   # also watch it in a window (close-up)

The pin is held fixed (like clipped in a stand). The arm stays still; only the fingers move:
    1. pre-pinch: thumb swung across, index half-curled, tails between the two fingertips
    2. squeeze: index curls and thumb bends -> the tails are pushed together -> jaws open
    3. release: fingers open -> the spring snaps the jaws shut
The pin is placed where this hand's thumb and index meet, with its squeeze direction along
thumb -> index (the robot's own pinch decides where the pin goes; on the real desk we then
set the pin up the same way).
"""

import argparse
import re
import sys
from pathlib import Path

import mujoco
import numpy as np

TASK = Path(__file__).resolve().parent.parent           # clothespin/
ROOT = TASK.parent                                       # repository root (shared robot model in sim/)
sys.path.insert(0, str(TASK / "sim"))
import pin_model as cp  # noqa: E402

SCENE = ROOT / "sim/g1_desk_scene.xml"                  # robot + motors (imported from the mug project)
# Tip-to-tip pinch found by search: at index 0.80 / thumb pitch 0.30 only the thumb TIP touches the index
# TIP (the rest of the thumb stays 2 cm away). SQUEEZE aims a little past it so the fingers press.
# Two pinches (see README). "tip": thumb tip meets index tip, squeezing ALONG the fingers - needs the
# fingers pointing down at a flat pin (wrist at its limit). "pad" (default): thumb pad meets the index pad,
# squeezing THROUGH the palm - palm down, fingers level, thumb under / index on top, like your video.
PINCHES = {
    "tip": dict(PRE=dict(index=0.35, thumb_yaw=1.2, thumb_pitch=0.0, others=0.0),
                TOUCH=dict(index=0.80, thumb_yaw=1.2, thumb_pitch=0.30, others=0.0),
                SQUEEZE=dict(index=0.95, thumb_yaw=1.2, thumb_pitch=0.38, others=0.0)),
    "pad": dict(PRE=dict(index=0.35, thumb_yaw=1.3, thumb_pitch=-0.1, others=1.2),
                TOUCH=dict(index=1.00, thumb_yaw=1.3, thumb_pitch=0.20, others=1.2),
                SQUEEZE=dict(index=1.05, thumb_yaw=1.3, thumb_pitch=0.40, others=1.2)),   # from a grid search
}
PINCH = "pad"
PRE, TOUCH, SQUEEZE = (PINCHES[PINCH][k] for k in ("PRE", "TOUCH", "SQUEEZE"))


def set_fingers(m, d, cfg, ctrl_too=True):
    """Put the left index + thumb at a configuration (qpos for coupled joints, motor targets)."""
    def q(name, v):
        d.qpos[m.jnt_qposadr[m.joint(name).id]] = v
    q("L_index_proximal_joint", cfg["index"]); q("L_index_intermediate_joint", cfg["index"])
    q("L_thumb_proximal_yaw_joint", cfg["thumb_yaw"]); q("L_thumb_proximal_pitch_joint", cfg["thumb_pitch"])
    q("L_thumb_intermediate_joint", 1.6 * cfg["thumb_pitch"]); q("L_thumb_distal_joint", 2.4 * cfg["thumb_pitch"])
    for f in ("middle", "ring", "pinky"):
        q(f"L_{f}_proximal_joint", cfg.get("others", 0.0)); q(f"L_{f}_intermediate_joint", cfg.get("others", 0.0))
    if ctrl_too:
        command(m, d, cfg)


def command(m, d, cfg):
    d.ctrl[m.actuator("L_index_proximal").id] = cfg["index"]
    d.ctrl[m.actuator("L_thumb_proximal_yaw").id] = cfg["thumb_yaw"]
    d.ctrl[m.actuator("L_thumb_proximal_pitch").id] = cfg["thumb_pitch"]
    for f in ("middle", "ring", "pinky"):
        d.ctrl[m.actuator(f"L_{f}_proximal").id] = cfg.get("others", 0.0)


def pad_points(m, d):
    """Closest points between the thumb tip and the index tip (world)."""
    th = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("L_thumb_distal").id and m.geom_contype[g]]
    ix = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("L_index_intermediate").id and m.geom_contype[g]]
    best = None
    for a in th:
        for b in ix:
            ft = np.zeros(6)
            dist = mujoco.mj_geomDistance(m, d, a, b, 0.1, ft)
            if best is None or dist < best[0]:
                best = (dist, ft[:3].copy(), ft[3:].copy())
    return best


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    return q


def build(offset_mm=(0.0, 0.0, 0.0)):
    """Scene + pin placed in the hand's pre-pinch; returns (model, data, info)."""
    base = mujoco.MjModel.from_xml_path(str(SCENE))
    bd = mujoco.MjData(base)
    mujoco.mj_resetDataKeyframe(base, bd, base.key("home").id)
    # Where do the fingertips PHYSICALLY meet? Run the pinch once with no pin (same finger settings as the
    # test) and record the thumb-tip/index-tip contact. The fingertips are bigger than the pin's tail
    # (7 x 13 mm), so a few mm matter: estimating from the hand's shape put the pin beside the pinch.
    P = cp.pin_params()
    set_fingers(base, bd, PRE, ctrl_too=False)
    mujoco.mj_forward(base, bd)
    pre_gap = pad_points(base, bd)[0]
    for j in range(base.njnt):
        if base.joint(j).name.startswith(("L_", "R_")):
            base.dof_damping[base.jnt_dofadr[j]] = 0.02
            base.dof_frictionloss[base.jnt_dofadr[j]] = 0.002
    th_id, ix_id = base.body("L_thumb_distal").id, base.body("L_index_intermediate").id
    meet = None
    for k in range(150):
        s_ = min(1.0, (k + 1) / 100)
        command(base, bd, {kk: PRE[kk] + (SQUEEZE[kk] - PRE[kk]) * s_ for kk in PRE})
        for _ in range(10):
            mujoco.mj_step(base, bd)
        for i in range(bd.ncon):
            c = bd.contact[i]
            if {base.geom_bodyid[c.geom1], base.geom_bodyid[c.geom2]} == {th_id, ix_id}:
                normal = c.frame[:3] if base.geom_bodyid[c.geom1] == th_id else -c.frame[:3]
                meet = (c.pos.copy(), normal.copy())                # normal points thumb -> index
                break
        if meet is not None:
            break
    if meet is None:
        raise SystemExit("thumb tip and index tip never touched - pinch settings need changing")
    touch_s = s_
    hand_pose = bd.qpos.copy()
    p_thumb, p_index = meet[0] - 0.001 * meet[1], meet[0] + 0.001 * meet[1]
    mujoco.mj_resetDataKeyframe(base, bd, base.key("home").id)       # back to the open hand for placement
    set_fingers(base, bd, PRE, ctrl_too=False)
    mujoco.mj_forward(base, bd)
    y = (p_index - p_thumb) / np.linalg.norm(p_index - p_thumb)        # squeeze direction
    fdir = bd.body("left_wrist_yaw_link").xmat.reshape(3, 3)[:, 0]     # fingers point along hand x
    z = -(fdir - (fdir @ y) * y)                                       # tails toward the palm,
    z /= np.linalg.norm(z)                                             # jaws out past the fingertips
    R = np.column_stack([np.cross(y, z), y, z])
    # pinch the tails 7 mm above the spring, centred between the two legs' outer faces
    zt = 0.007
    a_out = -P["h"] / 2 - P["t"]
    b_out = P["h"] / 2 - zt * np.sin(P["phi0"]) + P["t"]
    pinch = (p_thumb + p_index) / 2
    pos = pinch - R @ np.array([0.0, (a_out + b_out) / 2, zt])
    pos = pos + R @ (np.array(offset_mm) / 1000.0)                    # robustness test: pin misplaced

    xml = SCENE.read_text()
    xml = re.sub(r"<keyframe>.*?</keyframe>", "", xml, flags=re.S)       # pin adds a joint -> rebuild key
    xml = xml.replace("</asset>", cp.PIN_MATERIAL + "  </asset>", 1)
    xml = xml.replace("</worldbody>", cp.pin_xml(pos, mat2quat(R), P) + "\n  </worldbody>", 1)
    tmp = SCENE.with_name("_pin_test.xml")
    tmp.write_text(xml)
    try:
        m = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink()
    # finger joints: the scene uses arm-sized damping (0.5) on every joint, which makes the tiny thumb
    # lag far behind its motor (commanded 0.6, reached 0.25). Use finger-sized values here.
    for j in range(m.njnt):
        if m.joint(j).name.startswith(("L_", "R_")):
            m.dof_damping[m.jnt_dofadr[j]] = 0.02
            m.dof_frictionloss[m.jnt_dofadr[j]] = 0.002
    d = mujoco.MjData(m)
    for j in range(base.njnt):                                         # copy the home pose by joint name
        name = base.joint(j).name
        n = 7 if base.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
        d.qpos[m.jnt_qposadr[m.joint(name).id]:][:n] = bd.qpos[base.jnt_qposadr[j]:][:n]
    for a in range(base.nu):
        d.ctrl[m.actuator(base.actuator(a).name).id] = base.key_ctrl[base.key("home").id][a]
    set_fingers(m, d, PRE)
    mujoco.mj_forward(m, d)
    return m, d, dict(pre_gap_cm=pre_gap * 100, tail_width_cm=(b_out - a_out) * 100, P=P, touch_s=touch_s)


def finger_pin_forces(m, d):
    f = np.zeros(6)
    out = {}
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name
        pin = {"clothespin", "pin_leg_b"}
        if (b1 in pin) != (b2 in pin):
            other = b2 if b1 in pin else b1
            mujoco.mj_contactForce(m, d, i, f)
            out[other] = out.get(other, 0.0) + f[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--pinch", default=PINCH, choices=list(PINCHES))
    args = ap.parse_args()
    global PRE, TOUCH, SQUEEZE
    PRE, TOUCH, SQUEEZE = (PINCHES[args.pinch][k] for k in ("PRE", "TOUCH", "SQUEEZE"))
    print(f"pinch: {args.pinch}")
    m, d, info = build()
    hinge = m.jnt_qposadr[m.joint("pin_hinge").id]
    P = info["P"]
    print(f"pre-pinch fingertip gap {info['pre_gap_cm']:.1f} cm, pin tails are {info['tail_width_cm']:.1f} cm thick; "
          f"fingertips meet {info['touch_s'] * 100:.0f}% of the way from open to the pinch")
    start_contacts = finger_pin_forces(m, d)
    print("touching the pin before squeezing:", sorted(start_contacts) or "nothing (good)")

    renderer = cam = None
    if args.show:
        import cv2
        renderer = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = d.body("clothespin").xpos
        cam.distance, cam.azimuth, cam.elevation = 0.18, 210, -20

    dt = 0.02
    steps = int(round(dt / m.opt.timestep))
    log = []

    def run(T, cfg_from, cfg_to, label):
        n = int(T / dt)
        for k in range(n):
            s = min(1.0, (k + 1) / n)
            s = s * s * (3 - 2 * s)
            cfg = {key: cfg_from[key] + (cfg_to[key] - cfg_from[key]) * s for key in cfg_from}
            command(m, d, cfg)
            for _ in range(steps):
                mujoco.mj_step(m, d)
            q = d.qpos[hinge]
            forces = finger_pin_forces(m, d)
            log.append((label, q, cp.jaw_opening(q, P), forces))
            if renderer is not None:
                import cv2
                renderer.update_scene(d, cam)
                img = renderer.render()[:, :, ::-1].copy()
                cv2.putText(img, f"{label}: jaws {cp.jaw_opening(q, P) * 1000:4.1f} mm", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("clothespin pinch (q = quit)", img)
                cv2.waitKey(int(dt * 1000))
        return log[-1]

    run(0.5, PRE, PRE, "pre-pinch")
    run(2.0, PRE, SQUEEZE, "squeeze")
    held = run(1.0, SQUEEZE, SQUEEZE, "hold")
    released = run(1.0, SQUEEZE, PRE, "release")

    if np.isnan(d.qpos).any():
        print("SIMULATION BLEW UP (NaN) - contact settings need work")
        return
    max_open = max(o for _, _, o, _ in log)
    hold_forces = held[3]
    print(f"jaw opening while holding: {held[2] * 1000:.1f} mm  (max {max_open * 1000:.1f} mm; "
          f"your pin opens {P['Lj'] * 0 + 12:.0f} mm fully)")
    print("finger forces on the pin while holding:",
          {k.replace("L_", ""): round(v, 1) for k, v in hold_forces.items()})
    print(f"after release: jaws {released[2] * 1000:.1f} mm (spring snapped shut: {released[2] < 0.001})")
    ok = held[2] > 0.006 and released[2] < 0.001
    print("STAGE-1 FEASIBLE:", "YES" if ok else "NO")


if __name__ == "__main__":
    main()
