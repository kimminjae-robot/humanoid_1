"""Verify MuJoCo free fall and ground contact, without rendering or G1 assets."""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    model = mujoco.MjModel.from_xml_string('''
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="sphere" pos="0 0 2">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>''')
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)
    expected_z = 2.0 - 0.5 * 9.81 * data.time**2
    freefall_error = abs(float(data.qpos[2]) - expected_z)
    max_contact_count = 0
    finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    for _ in range(1400):
        mujoco.mj_step(model, data)
        max_contact_count = max(max_contact_count, data.ncon)
        finite = finite and bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    settled_z = float(data.qpos[2])
    checks = {
        "freefall_error_below_0_005_m": freefall_error < 0.005,
        "ground_contact_detected": max_contact_count > 0,
        "settled_center_within_0_005_m_of_radius": abs(settled_z - 0.1) < 0.005,
        "finite_state": finite,
        "settled_speed_below_0_01_m_s": float(np.linalg.norm(data.qvel[:3])) < 0.01,
    }
    report = {
        "timestamp": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "scope": "mujoco_engine_only_not_g1_tracking_or_sim2sim",
        "mujoco_version": mujoco.__version__,
        "physics_steps": 1500,
        "simulated_time_s": float(data.time),
        "freefall_error_m": freefall_error,
        "settled_center_height_m": settled_z,
        "checks": checks,
        "passed": all(checks.values()),
    }
    path = ROOT / "reports/mujoco_smoke.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
