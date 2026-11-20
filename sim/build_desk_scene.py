"""Build the task scene: the G1 standing at your desk with the mug and coaster.

    python sim/build_desk_scene.py              # mug at start mark A
    python sim/build_desk_scene.py --start C    # mug at start mark C

Everything comes from data/objects.yaml (your measurements), so if you re-measure,
just re-run this script.

Coordinates (MuJoCo world, metres): the robot's pelvis is at x=0, y=0 and it faces +x.
    x = forward (away from the robot, across the desk)
    y = to the robot's LEFT
    z = up
Your tape-mark layout uses   d = distance from the desk edge,   s = sideways (+ = left),
so a mark at (d, s) sits at  x = DESK_EDGE_X + d,  y = s,  on top of the desk.

Output: sim/g1_desk_scene.xml (open it with:  python -m mujoco.viewer --mjcf=sim/g1_desk_scene.xml)
"""

import argparse
import re
from pathlib import Path

import mujoco
import yaml

ROOT = Path(__file__).resolve().parent
ROBOT_SCENE = ROOT / "g1_inspire_scene.xml"         # made by sim/build_scene.py
OBJECTS = ROOT.parent / "data" / "objects.yaml"
OUT = ROOT / "g1_desk_scene.xml"

DESK_EDGE_X = 0.12      # desk edge 12 cm in front of the pelvis (robot chest sticks out ~7 cm)
DESK_DEPTH = 0.60       # how far the desk goes back (x)
DESK_WIDTH = 1.20       # side to side (y)
DESK_THICK = 0.03       # desk top thickness

# "Ready" pose: left hand held up (wrists ~1.15 m high, 15 cm in front, 22 cm to each side), fingers
# forward, palms facing inward. The lowest finger is at ~1.07 m, above the 12 cm mug (top at 1.00 m)
# on any start mark, so the robot never starts inside the mug. Solved with IK; without it the G1's
# default pose puts both hands inside the desk top.
HOME = {
    "left_shoulder_pitch_joint": -0.094, "left_shoulder_roll_joint": 1.538,
    "left_shoulder_yaw_joint": 0.46, "left_elbow_joint": -0.617,
    "left_wrist_roll_joint": -1.968, "left_wrist_pitch_joint": 0.606, "left_wrist_yaw_joint": 0.763,
    # right arm is not used in this task: hanging relaxed at the robot's side
    "right_shoulder_pitch_joint": 0.093, "right_shoulder_roll_joint": -0.199,
    "right_shoulder_yaw_joint": -0.079, "right_elbow_joint": 1.376,
    "right_wrist_roll_joint": -0.183, "right_wrist_pitch_joint": 0.136, "right_wrist_yaw_joint": 0.214,
}


def desk_xml(h):
    top_z = h - DESK_THICK / 2
    cx = DESK_EDGE_X + DESK_DEPTH / 2
    legs = []
    for i, (lx, ly) in enumerate([(0.04, 0.55), (0.04, -0.55), (DESK_DEPTH - 0.04, 0.55),
                                  (DESK_DEPTH - 0.04, -0.55)]):
        legs.append(f'<geom name="desk_leg{i}" type="box" size="0.02 0.02 {(h - DESK_THICK) / 2:.4f}" '
                    f'pos="{DESK_EDGE_X + lx:.4f} {ly} {(h - DESK_THICK) / 2:.4f}" material="desk"/>')
    return (f'<body name="desk">\n'
            f'      <geom name="desk_top" type="box" size="{DESK_DEPTH / 2} {DESK_WIDTH / 2} {DESK_THICK / 2}" '
            f'pos="{cx} 0 {top_z:.4f}" material="desk" friction="0.8 0.005 0.0001"/>\n      '
            + "\n      ".join(legs) + "\n    </body>")


MIMIC_FOLLOWERS = ("_intermediate_joint", "_thumb_distal_joint")   # driven by finger couplings


def gains(joint):
    """Motor stiffness (kp, N*m/rad) and damping (kv) per joint group.

    A position motor acts like a spring pulling the joint toward its target angle:
        torque = kp * (target - angle) - kv * velocity
    Legs/waist carry the whole upper body -> stiff. Arms medium. Fingers are tiny -> soft.
    """
    if joint.startswith(("L_", "R_")):
        return 5, 0.2
    if "wrist" in joint:
        return 40, 2
    if any(k in joint for k in ("shoulder", "elbow")):
        return 150, 8
    return 500, 25   # hips, knees, ankles, waist


def actuators_xml(xml):
    """One position motor per robot joint, except finger joints that are coupled to another."""
    import re
    joints = re.findall(r'<joint name="([^"]+)"[^>]*range=', xml)
    lines = []
    for j in joints:
        if j.endswith(MIMIC_FOLLOWERS) and not j.endswith("thumb_proximal_pitch_joint"):
            continue
        kp, kv = gains(j)
        lines.append(f'    <position name="{j.replace("_joint", "")}" joint="{j}" kp="{kp}" kv="{kv}" '
                     f'inheritrange="1"/>')
    return "  <actuator>\n" + "\n".join(lines) + "\n  </actuator>\n"


def build(start):
    obj = yaml.safe_load(OBJECTS.read_text())
    h = obj["table"]["height_m"]
    cup, coaster, layout = obj["cup"], obj["coaster"], obj["layout"]
    r_cup, h_cup, m_cup = cup["diameter_m"] / 2, cup["height_m"], cup["mass_kg"]
    r_co, t_co = coaster["diameter_m"] / 2, coaster["thickness_m"]

    co = layout["coaster"]
    st = layout["mug_start"][start]
    stand = layout.get("robot_stand_s", 0.0)      # robot stands this far to YOUR left of your spot
    co_x, co_y = DESK_EDGE_X + co["d"], co["s"] - stand
    mug_x, mug_y = DESK_EDGE_X + st["d"], st["s"] - stand

    objects = f"""
    <!-- ===== task objects (sizes from data/objects.yaml) ===== -->
    {desk_xml(h)}
    <body name="coaster" pos="{co_x:.4f} {co_y:.4f} {h + t_co / 2:.4f}">
      <geom name="coaster" type="cylinder" size="{r_co:.4f} {t_co / 2:.4f}" material="coaster"
            friction="1.0 0.005 0.0001"/>
      <site name="coaster_center" pos="0 0 {t_co / 2:.4f}" size="0.005" rgba="1 0 0 1"/>
    </body>
    <body name="mug" pos="{mug_x:.4f} {mug_y:.4f} {h + h_cup / 2 + 0.001:.4f}">
      <freejoint name="mug_free"/>
      <!-- body: a cylinder with the measured diameter/height/mass -->
      <geom name="mug_body" type="cylinder" size="{r_cup:.4f} {h_cup / 2:.4f}" mass="{m_cup}"
            material="mug" friction="1.0 0.005 0.0001" condim="4"/>
      <!-- no handle: the robot grips the body, so the mug is modelled as a plain cylinder -->
      <site name="mug_center" pos="0 0 0" size="0.005" rgba="0 0 1 1"/>
    </body>

    <!-- cameras: an overview, and one roughly like your phone (angled, from the robot's left-front) -->
    <camera name="overview" pos="1.6 0.9 1.7" xyaxes="-0.49 0.87 0 -0.38 -0.21 0.9"/>
    <camera name="phone_like" mode="targetbody" target="coaster" pos="0.75 0.75 1.35"/>
    <camera name="front" mode="targetbody" target="coaster" pos="1.15 0.05 1.30"/>
    <camera name="side" mode="targetbody" target="coaster" pos="0.30 0.95 1.15"/>
    <camera name="top" mode="targetbody" target="coaster" pos="0.36 0.06 1.85"/>
    <camera name="hand" mode="targetbody" target="left_wrist_yaw_link" pos="0.75 0.55 1.20" fovy="35"/>
  """

    assets = """
    <material name="desk" rgba="0.45 0.28 0.18 1"/>
    <material name="mug" rgba="0.95 0.95 0.95 1"/>
    <material name="coaster" rgba="0.85 0.75 0.55 1"/>
"""
    # the thumb-base mesh of the Inspire hand overlaps the palm mesh: ignore that self-contact
    excludes = """
  <contact>
    <exclude body1="left_wrist_yaw_link" body2="L_thumb_proximal"/>
    <exclude body1="right_wrist_yaw_link" body2="R_thumb_proximal"/>
    <exclude body1="left_wrist_yaw_link" body2="L_thumb_proximal_base"/>
    <exclude body1="right_wrist_yaw_link" body2="R_thumb_proximal_base"/>
  </contact>
"""
    xml = ROBOT_SCENE.read_text()
    xml = xml.replace("</worldbody>", objects + "</worldbody>", 1)
    xml = xml.replace("</asset>", assets + "  </asset>", 1)
    xml = xml.replace("</mujoco>", excludes + actuators_xml(xml) + "</mujoco>", 1)
    # Real joints have motor+gearbox inertia ("armature") and friction; the URDF leaves them out.
    # Without them the tiny finger/wrist links oscillate wildly under motor control.
    # implicitfast integration also keeps stiff motors stable at the 2 ms timestep.
    physics = ('\n  <option integrator="implicitfast"/>\n'
               '  <default>\n    <joint armature="0.01" damping="0.5" frictionloss="0.05"/>\n  </default>\n')
    xml = re.sub(r"(<compiler[^>]*/>)", r"\1" + physics, xml, count=1)
    OUT.write_text(xml)

    # save the ready pose as keyframe "home" (the viewer's "Load key" / mj_resetDataKeyframe)
    model = mujoco.MjModel.from_xml_path(str(OUT))
    data = mujoco.MjData(model)
    for name, value in HOME.items():
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    qpos = " ".join(f"{v:.5g}" for v in data.qpos)
    # motor targets = the same pose, so the robot HOLDS it once physics runs
    ctrl = " ".join(f"{data.qpos[model.jnt_qposadr[model.actuator_trnid[a, 0]]]:.5g}" for a in range(model.nu))
    xml = xml.replace("</mujoco>", f'  <keyframe>\n    <key name="home" qpos="{qpos}" ctrl="{ctrl}"/>\n'
                                   f'  </keyframe>\n</mujoco>', 1)
    OUT.write_text(xml)

    model = mujoco.MjModel.from_xml_path(str(OUT))   # must compile
    print(f"wrote {OUT.relative_to(ROOT.parent)}")
    print(f"  desk top at {h:.2f} m, edge {DESK_EDGE_X:.2f} m in front of the robot")
    print(f"  mug   (start {start}) at x={mug_x:.2f} y={mug_y:+.2f}   "
          f"d={r_cup * 200:.1f} cm, h={h_cup * 100:.0f} cm, {m_cup * 1000:.0f} g")
    print(f"  coaster          at x={co_x:.2f} y={co_y:+.2f}   d={r_co * 200:.0f} cm")
    print(f"  model: {model.nbody} bodies, {model.njnt} joints")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="A", choices=list("ABCDE"), help="mug start mark")
    build(ap.parse_args().start)
