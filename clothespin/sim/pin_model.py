"""MuJoCo model of the mini spring clothespin, built from clothespin/data/clothespin.yaml.

NOTE: angles (springref, range) are in RADIANS - use <compiler angle="radian"/> (the G1 scenes do).

Pin frame (metres): the spring pivot is at the origin.
    z  along the pin: tails at +z (squeezed by the fingers), jaws at -z
    y  the squeeze direction: leg A (fixed to the stand) on -y, leg B (hinged) on +y
    x  across the pin's width

How a clothespin works, and how it is modelled:
    The two legs touch at a fulcrum (where the spring coil is). At rest the spring pushes the JAWS
    together, so the legs form a narrow V that opens toward the TAILS. Squeezing the tails rocks
    leg B about the fulcrum: the tails close and the jaws open.
    -> leg A is fixed, leg B is a hinge (joint "pin_hinge") at the fulcrum, with a torsion spring.
       hinge angle 0 = jaws closed (rest), larger = jaws open.
"""

from pathlib import Path

import numpy as np
import yaml

TASK = Path(__file__).resolve().parent.parent          # clothespin/


def pin_params():
    y = yaml.safe_load((TASK / "data/clothespin.yaml").read_text())
    p = y[y.get("use", "clothespin_mini")]           # which pin: set `use:` in the yaml
    L, t, w = p["length_m"], p["leg_thickness_m"], p["width_m"]
    Lt = p["spring_from_tail_m"]                  # tail lever (pivot -> tail end)
    Lj = L - Lt                                    # jaw lever (pivot -> jaw tip)
    h = p["tail_rest_gap_m"] * Lj / (L)            # fulcrum spacing so the tails rest p["tail_rest_gap_m"] apart
    phi0 = -np.arcsin(h / Lj)                      # leg B's rest tilt: jaw tip touches leg A
    q_max = np.arcsin(h / Lt) - phi0               # rotation until the tails touch
    # torsion spring: force F at the tail end (lever Lt) opens it fully -> torque F*Lt at q_max.
    # Spring zero beyond the closed position gives a preload (jaws held shut at rest).
    k = p["squeeze_force_n"] * Lt / (q_max + 0.6)
    return dict(L=L, t=t, w=w, Lt=Lt, Lj=Lj, h=h, phi0=phi0, q_max=q_max, k=k, mass=p["mass_kg"],
                jaw_max=p["jaw_open_max_m"], F=p["squeeze_force_n"], name=y.get("use"))


def jaw_opening(q, P=None):
    """Gap between the jaw tips (m) for hinge angle q."""
    P = P or pin_params()
    return max(0.0, P["Lj"] * np.sin(P["phi0"] + q) + P["h"])


def pin_xml(pos, quat, P=None, free=False, clamp_q=None):
    """XML for the clothespin (leg A + hinged leg B), placed at pos with orientation quat (w x y z).
    free=False: leg A is held by a stand (fixed in space). free=True: the whole pin is a loose object."""
    P = P or pin_params()
    L, t, w, Lt, Lj, h = P["L"], P["t"], P["w"], P["Lt"], P["Lj"], P["h"]
    zc = (Lt - Lj) / 2                             # centre of a leg along z
    leg_m = P["mass"] / 2
    # leg B's rest tilt as a quaternion: an "euler" attribute would be read in degrees or radians depending on
    # the scene's <compiler angle=...>, and the G1 scene uses radians (that bug jammed the pin shut)
    qb = (np.cos(P["phi0"] / 2), np.sin(P["phi0"] / 2), 0.0, 0.0)
    # stiff contact for millimetre parts: solimp's last value is how deep a contact may sink before it pushes
    # back fully. MuJoCo's default (1 mm) is the whole paper thickness - the jaw sank through the paper.
    contact = 'condim="4" friction="1.2 0.01 0.001" solref="0.002 1" solimp="0.95 0.99 0.0001"'
    # Flat gripping pad on leg B's jaw (like the real pin's flat jaw face): leg B is straight and tilted, so
    # without it the top jaw touched the paper only along its tip edge and the pin could rock around that line.
    # The pad is parallel to leg A when the jaws are clamped at hinge angle clamp_q (e.g. on the paper).
    # hinge damping/armature sized to the legs (mini pin: 0.0005 / 2e-6). They scale with mass * length^2; with
    # the mini values the 5 g standard pin snapped open and hit the tails-touching stop hard enough to go unstable
    scale = (P["mass"] / 0.001) * (L / 0.035) ** 2
    damp, arm = 0.0005 * scale, 2e-6 * scale
    jaw_pad = ""
    if clamp_q is not None:
        a = -(P["phi0"] + clamp_q)                       # rotation that makes it parallel to leg A
        L_pad = 0.36 * Lj                           # flat jaw face: the front third of the jaw
        qp = (np.cos(a / 2), np.sin(a / 2), 0.0, 0.0)
        cz = -Lj + L_pad / 2 * np.cos(a)
        cy = -L_pad / 2 * np.sin(a) * 0 + 0.0004
        jaw_pad = (f'<geom name="pin_jaw_b_pad" type="box" size="{w / 2:.5f} 0.0004 {L_pad / 2:.4f}" '
                   f'pos="0 {cy:.5f} {cz:.5f}" quat="{qp[0]:.6f} {qp[1]:.6f} 0 0" mass="0.00005" '
                   f'material="pin" {contact}/>')
    return f"""
    <body name="clothespin" pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}" quat="{quat[0]:.5f} {quat[1]:.5f} {quat[2]:.5f} {quat[3]:.5f}">
      {'<joint name="pin_free" type="free" damping="0" armature="0" frictionloss="0" stiffness="0"/>' if free else ''}
      <!-- explicit zeros: a plain <freejoint/> inherited the ROBOT scene's joint defaults (friction 0.05,
           armature 0.01, damping 0.5) - 5x the pin's weight in friction, 10x its mass in armature -->
      <!-- leg A: {'loose object' if free else 'fixed (held by the stand)'} -->
      <geom name="pin_leg_a" type="box" size="{w / 2:.5f} {t / 2:.5f} {L / 2:.5f}"
            pos="0 {-h / 2 - t / 2:.5f} {zc:.5f}" mass="{leg_m}" material="pin" {contact}/>
      {"" if free else f'<geom name="pin_stand" type="box" size="0.004 0.004 0.03" pos="0 {-h / 2 - t - 0.004:.5f} {-Lj - 0.02:.5f}" rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0" mass="0.001"/>'}
      <site name="pin_spring" pos="0 0 0" size="0.0015" rgba="0.7 0.7 0.7 1"/>
      <!-- leg B: rocks on the fulcrum, spring pushes the jaws shut -->
      <body name="pin_leg_b" pos="0 {h / 2:.5f} 0" quat="{qb[0]:.6f} {qb[1]:.6f} 0 0">
        <joint name="pin_hinge" type="hinge" axis="1 0 0" range="0 {P['q_max']:.4f}"
               stiffness="{P['k']:.5f}" springref="-0.6" damping="{damp:.5f}" armature="{arm:.2e}" frictionloss="0"
               solreflimit="0.002 1" solimplimit="0.95 0.99 0.0001"/>
        <!-- stiff limits: jaws closed (0) and tails touching (q_max) are hard stops, like the real pin.
             frictionloss="0": without it the hinge inherited the ROBOT's joint friction (0.05 N*m) from the
             scene's default class - 2x the pin's own spring, i.e. a rusty hinge that 4.6 N couldn't open -->
        <geom name="pin_leg_b" type="box" size="{w / 2:.5f} {t / 2:.5f} {L / 2:.5f}"
              pos="0 {t / 2:.5f} {zc:.5f}" mass="{leg_m}" material="pin" {contact}/>
        <site name="pin_jaw_b" pos="0 0 {-Lj:.5f}" size="0.001" rgba="0 0 0 1"/>
        {jaw_pad}
      </body>
    </body>"""


PIN_MATERIAL = '    <material name="pin" rgba="0.85 0.08 0.08 1"/>\n'


def hinge_for_jaw_gap(gap, P=None):
    """Hinge angle at which the jaw tips are `gap` apart (e.g. clamped on a sheet of paper)."""
    P = P or pin_params()
    return float(np.arcsin((gap - P["h"]) / P["Lj"]) - P["phi0"])
