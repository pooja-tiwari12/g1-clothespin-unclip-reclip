"""Open the clothespin scene in the MuJoCo viewer, ALREADY in the seated pose (no "Load key" needed).

    mjpython clothespin/sim/view_scene.py

(Use mjpython for this script. The plain "python -m mujoco.viewer" command starts the robot in its
default straight-leg pose until you press Load key.)
"""

import time
from pathlib import Path

import mujoco
import mujoco.viewer

SCENE = Path(__file__).resolve().parent / "pin_scene.xml"

m = mujoco.MjModel.from_xml_path(str(SCENE))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)       # seated, hands on the lap, pin on the paper
with mujoco.viewer.launch_passive(m, d) as v:
    v.cam.lookat[:] = [0.30, 0.05, 0.70]
    v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.6, 140, -20
    while v.is_running():
        t0 = time.time()
        mujoco.mj_step(m, d)
        v.sync()
        time.sleep(max(0.0, m.opt.timestep - (time.time() - t0)))
