"""Hand calculation (user's idea): the spring resists with a torque about the hinge; a finger pushing a tail
with force F at distance d from the hinge makes a moment F*d. To open, F*d must beat the spring torque.

    python clothespin/scripts/spring_moment.py

Spring model (pin_model.py): torsion spring, torque = k * (q + 0.6)  [q = hinge angle, 0 = jaws shut].
The +0.6 is the preload: even shut, the spring pushes the jaws together.
"""

import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TASK / "sim"))
import pin_model as pm  # noqa: E402

P = pm.pin_params()


def spring_torque(jaw_gap_m):
    """Spring torque (N*m) the fingers must beat to hold the jaws this far open."""
    q = pm.hinge_for_jaw_gap(jaw_gap_m, P) if jaw_gap_m > 0 else 0.0
    return P["k"] * (q + 0.6)


def finger_force(jaw_gap_m, d_m):
    """Force (N) a finger needs at distance d from the hinge: F = torque / d."""
    return spring_torque(jaw_gap_m) / d_m


if __name__ == "__main__":
    print(f"pin: {P['name']}  tail {P['Lt'] * 1000:.0f} mm, jaw {P['Lj'] * 1000:.0f} mm, "
          f"spring k = {P['k']:.4f} N*m/rad")
    print("\nspring torque to hold the jaws open:")
    for g in (0.001, 0.004, 0.008, P["jaw_max"]):
        print(f"  {g * 1000:4.1f} mm open: {spring_torque(g):.4f} N*m")
    print("\nforce EACH finger needs (F = torque / d), jaws 4 mm open (clears the 1 mm paper):")
    for d in (0.010, 0.015, 0.020, 0.025, 0.030):
        print(f"  pressing {d * 1000:2.0f} mm from the spring: {finger_force(0.004, d):4.1f} N")
    print("\nThe robot's Inspire fingers manage roughly 10 N; pressing near the spring needs far more.")
    print("Both legs feel the SAME spring torque, so for a pure squeeze (no turning of the whole pin) each")
    print("finger should make the same MOMENT: F_index * d_index = F_thumb * d_thumb = spring torque.")
