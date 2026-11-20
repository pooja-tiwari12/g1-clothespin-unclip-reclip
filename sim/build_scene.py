"""Build the MuJoCo scene for the Unitree G1 + Inspire five-finger hands.

    python sim/build_scene.py

Input : sim/assets/g1_inspire/g1_inspire.urdf   (from unitreerobotics/unitree_ros, BSD-3)
Output: sim/assets/g1_inspire/g1_inspire_raw.xml  (straight URDF -> MJCF conversion)
        sim/g1_inspire_scene.xml                  (robot standing on a floor, with lights,
                                                   finger couplings and a hand-view camera)

What this adds on top of the plain conversion:
  * The URDF has no floating base, so MuJoCo welds the pelvis to the world at z=0.
    We lift it to the G1's standing pelvis height so the robot stands on the floor
    (still fixed-base: good for hand/arm work, no balancing yet).
  * URDF <mimic> tags (e.g. "index_intermediate = 1.0 * index_proximal") are ignored by
    MuJoCo. We add them back as <equality><joint .../> constraints so that, once physics
    is on, the Inspire hand behaves like the real one: 6 motors drive 12 finger joints.
  * Floor, sky, lights, and a camera looking at the right hand.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets" / "g1_inspire"
URDF = ASSETS / "g1_inspire.urdf"
RAW = ASSETS / "g1_inspire_raw.xml"
SCENE = ROOT / "g1_inspire_scene.xml"

PELVIS_HEIGHT = 0.793  # metres, G1 standing pelvis height


def urdf_to_mjcf() -> None:
    text = URDF.read_text()
    # mesh filenames in the URDF are "meshes/x.STL" and meshdir is "meshes": strip the path
    text = text.replace('<compiler meshdir="meshes" discardvisual="false"/>',
                        '<compiler meshdir="meshes" strippath="true" discardvisual="false" '
                        'balanceinertia="true"/>')
    tmp = ASSETS / "_tmp.urdf"
    tmp.write_text(text)
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink()
    mujoco.mj_saveLastXML(str(RAW), model)


def mimic_constraints() -> list[tuple[str, str, float, float]]:
    """(joint, leader, multiplier, offset) for every <mimic> in the URDF."""
    out = []
    for j in ET.parse(URDF).getroot().findall("joint"):
        m = j.find("mimic")
        if m is not None:
            out.append((j.get("name"), m.get("joint"),
                        float(m.get("multiplier", 1)), float(m.get("offset", 0))))
    return out


def build_scene() -> None:
    raw = RAW.read_text()
    # paths in the scene file are relative to sim/
    raw = raw.replace('meshdir="meshes"', 'meshdir="assets/g1_inspire/meshes"')

    # wrap everything in <worldbody> into a "pelvis" body raised to standing height
    m = re.search(r"<worldbody>(.*)</worldbody>", raw, flags=re.S)
    body = m.group(1)
    wrapped = (f'<worldbody>\n    <light pos="0 -1 3" dir="0 0.3 -1" diffuse="0.8 0.8 0.8"/>\n'
               f'    <light pos="1.5 1 2.5" dir="-0.5 -0.3 -1" diffuse="0.4 0.4 0.4"/>\n'
               f'    <geom name="floor" type="plane" size="0 0 0.05" material="groundplane"/>\n'
               f'    <body name="pelvis" pos="0 0 {PELVIS_HEIGHT}">{body}    </body>\n'
               f'  </worldbody>')
    raw = raw[:m.start()] + wrapped + raw[m.end():]

    extras = """
  <visual>
    <headlight diffuse="0.5 0.5 0.5" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="160" elevation="-20" offwidth="1280" offheight="960"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
             rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
"""
    eq = "\n".join(
        f'    <joint name="mimic_{j}" joint1="{j}" joint2="{leader}" '
        f'polycoef="{off:g} {mult:g} 0 0 0" solref="0.005 1"/>'
        for j, leader, mult, off in mimic_constraints())
    extras += f"  <equality>\n{eq}\n  </equality>\n"

    raw = raw.replace("</mujoco>", extras + "</mujoco>")
    SCENE.write_text(raw)
    model = mujoco.MjModel.from_xml_path(str(SCENE))  # sanity check: it must compile
    print(f"wrote {SCENE.relative_to(ROOT.parent)}: {model.njnt} joints, "
          f"{model.neq} finger couplings, {model.nmesh} meshes")


if __name__ == "__main__":
    urdf_to_mjcf()
    build_scene()
