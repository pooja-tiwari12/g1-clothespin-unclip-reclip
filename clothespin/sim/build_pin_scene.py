"""Build the clothespin task scene: G1 in front of your cabinet, letter paper overhanging the edge,
mini clothespin clipped flat on the paper's edge (a loose object held only by its own spring).

    python clothespin/sim/build_pin_scene.py           # writes clothespin/sim/pin_scene.xml
    python -m mujoco.viewer --mjcf=clothespin/sim/pin_scene.xml

Shared from the mug project (imported, not copied): the robot model with motors and the "ready" arm
pose (sim/g1_inspire_scene.xml, sim/build_desk_scene.py).
Measurements from clothespin/data/clothespin.yaml.

World frame (same as the mug task): robot pelvis at x=0, y=0, facing +x; y = robot's left; z = up.
"""

import os
import re
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

TASK = Path(__file__).resolve().parent.parent            # clothespin/
ROOT = TASK.parent                                        # repository root
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(TASK / "sim"))
import build_desk_scene as mug_scene  # noqa: E402  (mug project: motors, ready pose)
import pin_model  # noqa: E402

ROBOT = ROOT / "sim" / "g1_inspire_scene.xml"
OUT = TASK / "sim" / "pin_scene.xml"

# SEATED (your suggestion): sitting lowers the shoulders to about table height, so the hand reaches the pin
# level, from the front, with an almost straight wrist (searched: 5% of the wrist's range vs 100% standing).
SEATED = True
PELVIS_Z = 0.53          # pelvis height when sitting (seat ~0.45 m; G1 is 1.3 m tall)
TABLE_EDGE_X = 0.50      # cabinet front edge, 50 cm in front of the pelvis: knees and feet clear the cabinet
SIT = {"left_hip_pitch_joint": -1.57, "left_knee_joint": 1.57,      # thighs level, shins straight down,
       "right_hip_pitch_joint": -1.57, "right_knee_joint": 1.57}       # feet flat on the footrest
# both hands resting palm-down ON the thighs: solved so the palm is 2 mm from the thigh surface
# (first version aimed at the thigh's highest point near the hip and left the hands floating 2.8 cm up),
# fingers relaxed (curl 0.35) so they drape over the thigh. Left and right are mirror images.
LAP = {"left_shoulder_pitch_joint": 0.307, "left_shoulder_roll_joint": 0.216, "left_shoulder_yaw_joint": -0.565,
       "left_elbow_joint": 0.053, "left_wrist_roll_joint": 1.33, "left_wrist_pitch_joint": 0.462,
       "left_wrist_yaw_joint": 0.419,
       "right_shoulder_pitch_joint": 0.307, "right_shoulder_roll_joint": -0.216, "right_shoulder_yaw_joint": 0.565,
       "right_elbow_joint": 0.053, "right_wrist_roll_joint": -1.33, "right_wrist_pitch_joint": 0.462,
       "right_wrist_yaw_joint": -0.419,
       **{f"{h}_{f}_{j}_joint": 0.35 for h in "LR" for f in ("index", "middle", "ring", "pinky")
          for j in ("proximal", "intermediate")}}
TABLE_DEPTH, TABLE_WIDTH = 0.40, 0.80
PIN_Y = 0.15             # pin in front of the LEFT shoulder (you used your left hand)
PAPER_THICK = 0.001      # real paper is ~0.1 mm; 1 mm keeps the contact stable in simulation
def jaw_bite(P):
    """How far the jaws reach onto the paper: 55% of the jaw (12 mm on the mini pin), but the jaw tips stay
    3 mm short of the cabinet edge (the paper overhangs 25 mm; leg A is under the paper)."""
    return min(0.55 * P["Lj"], 0.025 - 0.003)


_OLD_JAW_BITE = 0.012         # how much of the jaws grip the paper. 1.6 cm (user: spring closer to the paper) wedged the
                         # rigid paper near the spring - the jaws couldn't open enough there to let go. 1.2 cm comes off.
                         # paper edge, as suggested - steadier: shorter tails sticking out, less twisting)


def chair_xml():
    """Chair under the thighs + footrest under the feet (at this seat height the G1's feet don't reach
    the floor, like on a bar stool). Sizes MEASURED from the robot's seated shape (pelvis at 0.53 m):
    thighs' lowest point 0.378 m over x -0.06..0.39; feet soles 0.102 m at x 0.31..0.35;
    the hanging right arm is at y <= -0.164, so the seat is narrower than that (half-width 0.15)."""
    seat_top, foot_top = 0.376, 0.100
    return f"""
    <body name="chair">
      <geom name="seat" type="box" size="0.17 0.15 0.02" pos="0.11 0 {seat_top - 0.02:.3f}" rgba="0.25 0.25 0.3 1"/>
      <geom name="chair_back" type="box" size="0.015 0.15 0.22" pos="-0.10 0 {seat_top + 0.22:.3f}" rgba="0.25 0.25 0.3 1"/>
      <geom name="chair_leg" type="box" size="0.03 0.03 {(seat_top - 0.04) / 2:.3f}" pos="0.11 0 {(seat_top - 0.04) / 2:.3f}" rgba="0.2 0.2 0.22 1"/>
      <geom name="footrest" type="box" size="0.04 0.16 {foot_top / 2:.3f}" pos="0.33 0 {foot_top / 2:.3f}" rgba="0.2 0.2 0.22 1"/>
    </body>"""


# Flat FINGERTIP PADS (user's idea: more contact surface between the thumb's flat and the leg tip = more stable).
# The robot's fingertips are rounded meshes: in the grip the thumb touched the leg over ~1.5 mm, practically one
# point the pin could pivot about, and with no friction against twisting (condim 3). Each pad is a thin flat plate
# placed where that fingertip touches the pin in the grip (measured in the finger's own frame), lying flat on the
# leg's face, with twisting/rolling friction (condim 6) - like a rubber fingertip pad. Set False to compare.
FINGER_PADS = False         # tested (user idea): 1/36 full passes vs 3/36 without; jaws capped at 7 mm
PAD_HALF = (0.0045, 0.0045, 0.0006)          # 9 x 9 mm, 1.2 mm thick
PADS = {   # body: (contact point in the body frame [m], direction finger -> pin in the body frame)
    "L_thumb_distal": ((0.01872, -0.01441, -0.00625), (0.485, -0.874, -0.025)),
    "L_index_intermediate": ((-0.00467, -0.04000, -0.00311), (-0.955, -0.203, 0.218)),
}


def pads_xml_insert(xml):
    import mujoco as mj
    for body, (p, n) in PADS.items():
        n = np.array(n) / np.linalg.norm(n)
        q = np.zeros(4)
        mj.mju_quatZ2Vec(q, n)                   # pad's thin axis (z) along the push direction
        c = np.array(p) + n * (PAD_HALF[2] - 0.0002)   # pad face ~1 mm proud of the fingertip at the contact
        name = body.replace("_distal", "").replace("_intermediate", "") + "_pad"
        geom = (f'<geom name="{name}" type="box" size="{PAD_HALF[0]} {PAD_HALF[1]} {PAD_HALF[2]}" '
                f'pos="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}" quat="{q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {q[3]:.5f}" '
                f'condim="6" friction="1.2 0.004 0.0005" mass="0.0005" rgba="0.15 0.15 0.15 1"/>')
        xml, k = re.subn(rf'(<body name="{body}"[^>]*>)', r"\1\n" + geom, xml, count=1)
        assert k == 1, body
    return xml


def build():
    cfg = yaml.safe_load((TASK / "data/clothespin.yaml").read_text())["setup"]
    h_table = cfg["table_height_m"]
    paper_w, paper_l = cfg["paper_size_m"]               # 21.6 cm along the edge, 27.9 cm going back
    overhang = cfg["paper_overhang_m"]
    P = pin_model.pin_params()

    paper_front = TABLE_EDGE_X - overhang                 # paper's front edge (sticks out toward the robot)
    paper_z = h_table + PAPER_THICK / 2
    # pin frame: z along the pin with the TAILS toward the robot (-x), y (squeeze) = up, leg B on top
    R = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.ravel())
    # leg A's inner face (pin y = -h/2) touches the paper's underside; jaw tips JAW_BITE onto the paper
    bite = jaw_bite(P)
    pin_pos = np.array([paper_front + bite - P["Lj"], PIN_Y, h_table + P["h"] / 2])
    q_clip = pin_model.hinge_for_jaw_gap(PAPER_THICK, P)

    world = f"""
    <!-- ===== clothespin task (measurements: clothespin/data/clothespin.yaml) ===== -->
    <body name="table" pos="{TABLE_EDGE_X + TABLE_DEPTH / 2:.4f} 0 0">
      <geom name="table_top" type="box" size="{TABLE_DEPTH / 2} {TABLE_WIDTH / 2} 0.015" pos="0 0 {h_table - 0.015:.4f}"
            material="wood" friction="0.8 0.005 0.0001"/>
      <geom type="box" size="{TABLE_DEPTH / 2 - 0.01} {TABLE_WIDTH / 2 - 0.01} {(h_table - 0.03) / 2:.4f}"
            pos="0 0 {(h_table - 0.03) / 2:.4f}" rgba="0.75 0.72 0.66 1"/>
    </body>
    <body name="paper" pos="{paper_front + paper_l / 2:.4f} {PIN_Y:.4f} {paper_z:.5f}">
      <geom name="paper" type="box" size="{paper_l / 2} {paper_w / 2} {PAPER_THICK / 2}" material="paper"
            friction="0.6 0.005 0.0001" solref="0.002 1" solimp="0.95 0.99 0.0001"/>
    </body>
{pin_model.pin_xml(pin_pos, quat, P, free=True, clamp_q=q_clip)}
    {chair_xml() if SEATED else ""}
    <camera name="pin_close" mode="targetbody" target="clothespin" pos="{pin_pos[0] - 0.18:.3f} {PIN_Y + 0.14:.3f} {h_table + 0.12:.3f}" fovy="40"/>
    <camera name="like_your_phone" mode="targetbody" target="clothespin" pos="{pin_pos[0] - 0.55:.3f} {PIN_Y + 0.45:.3f} {h_table + 0.30:.3f}"/>
    <camera name="front" mode="targetbody" target="clothespin" pos="{TABLE_EDGE_X + 0.75:.3f} 0.05 {h_table + 0.45:.3f}"/>
    <camera name="side" mode="targetbody" target="clothespin" pos="{pin_pos[0]:.3f} 0.85 {h_table + 0.25:.3f}"/>
  """
    assets = """
    <material name="wood" rgba="0.45 0.33 0.24 1"/>
    <material name="paper" rgba="0.93 0.95 0.86 1"/>
""" + pin_model.PIN_MATERIAL

    xml = ROBOT.read_text()
    if SEATED:
        xml = xml.replace('<body name="pelvis" pos="0 0 0.793">', f'<body name="pelvis" pos="0 0 {PELVIS_Z}">', 1)
    # gravity compensation on every robot body, like the real G1's controller: without it the arm sagged
    # a few degrees under its own weight, putting the fingertips centimetres below their target
    xml = re.sub(r'<body name="([^"]+)"', r'<body name="\1" gravcomp="1"', xml)
    # robot meshes live in the mug project's sim/ folder: point there (shared, not copied)
    # RELATIVE to this scene file (MuJoCo resolves meshdir from the model's folder), so it works on any computer
    meshes = os.path.relpath(ROOT / "sim/assets/g1_inspire/meshes", OUT.parent)
    xml = xml.replace('meshdir="assets/g1_inspire/meshes"', f'meshdir="{meshes}"')
    xml = xml.replace("</worldbody>", world + "</worldbody>", 1)
    xml = xml.replace("</asset>", assets + "  </asset>", 1)
    excludes = """
  <contact>
    <exclude body1="left_wrist_yaw_link" body2="L_thumb_proximal"/>
    <exclude body1="right_wrist_yaw_link" body2="R_thumb_proximal"/>
    <exclude body1="left_wrist_yaw_link" body2="L_thumb_proximal_base"/>
    <exclude body1="right_wrist_yaw_link" body2="R_thumb_proximal_base"/>
  </contact>
"""
    # temporary "helper" weld: holds the pin in place in tests (e.g. before the fingers have it).
    # Off by default; switched on/off from code with data.eq_active.
    helper = '  <equality>\n    <weld name="pin_hold" body1="clothespin" active="false"/>\n  </equality>\n'
    # motors for the ROBOT only: the mug helper adds one per joint with a range, which also caught the pin's
    # hinge - a motor silently holding the jaws at 1 mm, so no squeeze could ever open the clipped pin
    acts = "\n".join(l for l in mug_scene.actuators_xml(xml).split("\n") if 'joint="pin_' not in l)
    xml = xml.replace("</mujoco>", excludes + helper + acts + "\n</mujoco>", 1)
    # noslip_iterations: MuJoCo's friction is slightly soft, so a steady sideways push (the shut pin is a wedge
    # between the fingertips) made the pin CREEP through the fingers at ~5 mm/s even with the fingers still.
    # The no-slip solver removes that creep - real static friction doesn't creep.
    physics = ('\n  <option integrator="implicitfast" timestep="0.001" noslip_iterations="10"/>\n'
               '  <default>\n    <joint armature="0.01" damping="0.5" frictionloss="0.05"/>\n  </default>\n')
    xml = re.sub(r"(<compiler[^>]*/>)", r"\1" + physics, xml, count=1)
    # finger joints: finger-sized damping (the arm-sized 0.5 made the thumb lag far behind its motor)
    xml = re.sub(r'(<joint name="[LR]_[a-z_]+_joint")', r'\1 damping="0.02" frictionloss="0.002"', xml)
    if FINGER_PADS:
        xml = pads_xml_insert(xml)
    OUT.write_text(xml)

    # keyframe "home": arms in the ready pose (from the mug project), pin clipped on the paper
    m = mujoco.MjModel.from_xml_path(str(OUT))
    d = mujoco.MjData(m)
    for name, value in {**mug_scene.HOME, **(dict(SIT, **LAP) if SEATED else {})}.items():
        d.qpos[m.jnt_qposadr[m.joint(name).id]] = value
    fa = m.jnt_qposadr[m.joint("pin_free").id]
    d.qpos[fa:fa + 3] = pin_pos
    d.qpos[fa + 3:fa + 7] = quat
    d.qpos[m.jnt_qposadr[m.joint("pin_hinge").id]] = q_clip
    ctrl = " ".join(f"{d.qpos[m.jnt_qposadr[m.actuator_trnid[a, 0]]]:.5g}" for a in range(m.nu))
    qpos = " ".join(f"{v:.6g}" for v in d.qpos)
    xml = xml.replace("</mujoco>", f'  <keyframe>\n    <key name="home" qpos="{qpos}" ctrl="{ctrl}"/>\n'
                                   f'  </keyframe>\n</mujoco>', 1)
    OUT.write_text(xml)
    m = mujoco.MjModel.from_xml_path(str(OUT))
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  robot {'SEATED (pelvis ' + str(PELVIS_Z) + ' m)' if SEATED else 'standing'}; "
          f"cabinet top {h_table:.2f} m, front edge {TABLE_EDGE_X:.2f} m in front of the robot")
    print(f"  paper {paper_w * 100:.1f} x {paper_l * 100:.1f} cm, overhangs {overhang * 100:.1f} cm")
    print(f"  pin clipped on the paper edge at x={pin_pos[0]:.3f} y={PIN_Y:.2f}, tails pointing at the robot; "
          f"jaws held {PAPER_THICK * 1000:.0f} mm apart by the paper")
    return m


if __name__ == "__main__":
    build()
