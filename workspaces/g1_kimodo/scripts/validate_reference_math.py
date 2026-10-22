"""Independent analytic checks for quaternion frames, contact geometry and slip."""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from reference_core import analyze, angular_velocity_world, geometry_bottoms, interpolate, wxyz


def main():
    checks = {}
    q = np.zeros((2, 36)); q[:, 3] = 1; q[1, 3] = -1; q[1, 0] = 2
    mid = interpolate(q, np.array([0., 1.]), np.array([.5]))
    checks["antipodal_slerp_shortest_arc"] = bool(abs(mid[0, 0]-1) < 1e-12 and abs(abs(mid[0, 3])-1) < 1e-12)
    t = np.arange(30)/50
    rotation = Rotation.from_euler("z", (2*t)[:, None])*Rotation.from_euler("y", .6)
    omega = angular_velocity_world(wxyz(rotation), 50)
    checks["angular_velocity_is_world_frame"] = bool(np.max(abs(omega-np.array([0, 0, 2]))) < 1e-10)
    model = mujoco.MjModel.from_xml_path(str(ROOT/"assets/g1/g1_scene.xml")); data = mujoco.MjData(model)
    contract = json.loads((ROOT/"configs/motion_contract.json").read_text())
    pose = np.zeros(36); pose[2] = .8; pose[3] = 1
    pose[7:] = [contract["controller"]["joint_default_positions"][name] for name in contract["csv"]["joint_names"]]
    data.qpos[:] = pose; mujoco.mj_forward(model, data)
    bottom, _ = geometry_bottoms(model, data, np.arange(1, model.ngeom))
    expected = np.array([mujoco.mj_geomDistance(model, data, i, 0, 3, None) for i in range(1, model.ngeom)])
    checks["capsule_bottom_matches_mujoco_signed_distance"] = bool(np.max(abs(bottom-expected)) < 1e-9)
    static = np.tile(pose, (30, 1)); static[:, 2] -= bottom[3:10].min()
    stationary, _ = analyze(static, 50)
    translated = static.copy(); translated[:, 0] += t
    sliding, _ = analyze(translated, 50)
    checks["stationary_support_has_zero_slip"] = bool(stationary["max_stance_slip_speed_m_s"] < 1e-10)
    checks["sliding_detected_without_velocity_contact_mask"] = bool(sliding["slip_frame_fraction"] == 1 and abs(sliding["max_stance_slip_speed_m_s"]-1) < 1e-9)
    # A complete in-place turn tests the material-point rotational term, not a
    # derivative of a geometrically moving 'lowest point'.
    yaw = 2*t
    rotating = static.copy(); rotating[:, 3:7] = wxyz(Rotation.from_euler("z", yaw[:, None]))
    measured = angular_velocity_world(rotating[:, 3:7], 50)
    checks["rotation_vector_differentiation"] = bool(np.max(abs(measured[:, 2]-2)) < 1e-10)
    # MuJoCo free-joint angular qvel is body-frame, unlike bundle world angular velocity.
    initial=pose.copy(); initial[3:7]=wxyz(Rotation.from_euler("y",.6))
    speed=np.zeros(35);speed[3:6]=[0,0,2];integrated=initial.copy();dt=1e-4
    mujoco.mj_integratePos(model,integrated,speed,dt)
    observed_world=(Rotation.from_quat(integrated[3:7][[1,2,3,0]])*Rotation.from_quat(initial[3:7][[1,2,3,0]]).inv()).as_rotvec()/dt
    expected_world=Rotation.from_quat(initial[3:7][[1,2,3,0]]).apply(speed[3:6])
    checks["mujoco_free_joint_angular_qvel_is_body_frame"] = bool(np.max(abs(observed_world-expected_world))<1e-9)
    result = {"scope": "independent_reference_math_geometry_and_slip_measurement_checks", "checks": checks, "passed": all(checks.values())}
    (ROOT/"reports/reference_math_validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
