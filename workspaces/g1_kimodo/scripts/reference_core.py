"""G1 kinematic reference math and geometry QA; no policy or dynamics rollout."""

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]


def rotations(wxyz):
    return Rotation.from_quat(np.asarray(wxyz)[..., [1, 2, 3, 0]])


def wxyz(rotation):
    return rotation.as_quat()[..., [3, 0, 1, 2]]


def continuous_quaternions(quat):
    q = np.array(quat, dtype=float, copy=True)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] *= -1
    return q


def interpolate(qpos, source_times, target_times):
    result = np.empty((len(target_times), 36))
    times = np.clip(target_times, source_times[0], source_times[-1])
    for column in [0, 1, 2, *range(7, 36)]:
        result[:, column] = np.interp(times, source_times, qpos[:, column])
    result[:, 3:7] = continuous_quaternions(wxyz(Slerp(source_times, rotations(continuous_quaternions(qpos[:, 3:7])))(times)))
    return result


def derivative(values, fps):
    return np.gradient(values, 1 / fps, axis=0, edge_order=2 if len(values) > 2 else 1)


def angular_velocity_world(quats, fps):
    rotation = rotations(quats)
    increments = (rotation[1:] * rotation[:-1].inv()).as_rotvec() * fps
    output = np.empty((len(quats), 3))
    output[0], output[-1] = increments[0], increments[-1]
    output[1:-1] = (increments[:-1] + increments[1:]) / 2
    return output


def model_and_config():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/g1/g1_scene.xml"))
    config = json.loads((ROOT / "configs/reference_processing.json").read_text())
    contract = json.loads((ROOT / "configs/motion_contract.json").read_text())
    return model, config, contract


def foot_geoms(model):
    return [[i for i in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(side + "_foot")]
            for side in ["left", "right"]]


def geometry_bottoms(model, data, indices):
    # All canonical G1 collision shapes are spheres or capsules, capsule axis local Z.
    center = data.geom_xpos[indices]
    radius = model.geom_size[indices, 0]
    half = np.where(model.geom_type[indices] == mujoco.mjtGeom.mjGEOM_CAPSULE, model.geom_size[indices, 1], 0)
    axis = data.geom_xmat[indices].reshape(-1, 3, 3)[:, :, 2]
    endpoints = np.stack([center - axis * half[:, None], center + axis * half[:, None]], axis=1)
    points = endpoints.copy()
    points[:, :, 2] -= radius[:, None]
    return points.min(axis=1)[:, 2], points


def static_margin(com_xy, points):
    xy = np.unique(np.round(np.asarray(points)[:, :2], 9), axis=0)
    if len(xy) < 3 or np.linalg.matrix_rank(xy - xy.mean(axis=0)) < 2:
        return -1.0
    hull = ConvexHull(xy)
    # ConvexHull normals are normalized; positive margin means inside every edge.
    return float(np.min(-(hull.equations[:, :2] @ com_xy + hull.equations[:, 2])))


def analyze(qpos, fps, model=None):
    model, config, contract = model_and_config() if model is None else (model, json.loads((ROOT / "configs/reference_processing.json").read_text()), json.loads((ROOT / "configs/motion_contract.json").read_text()))
    data = mujoco.MjData(model)
    frames = len(qpos)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]
    feet = foot_geoms(model)
    body_pos, body_quat = np.zeros((frames, model.nbody - 1, 3)), np.zeros((frames, model.nbody - 1, 4))
    clearance = np.zeros((frames, 2))
    contact_points = [[None, None] for _ in range(frames)]
    ground_depth, self_depth = np.zeros(frames), np.zeros(frames)
    com = np.zeros((frames, 3)); margins = np.full(frames, -1.0)
    self_pairs = {}
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        body_pos[t], body_quat[t] = data.xpos[1:], data.xquat[1:]
        com[t] = data.subtree_com[1]
        bottom, _ = geometry_bottoms(model, data, np.arange(1, model.ngeom))
        ground_depth[t] = max(0, -float(bottom.min()))
        support = []
        for side, indices in enumerate(feet):
            bottom, points = geometry_bottoms(model, data, np.array(indices))
            clearance[t, side] = bottom.min()
            selected = bottom <= bottom.min() + config["heel_toe_active_height_band_m"]
            active = points[selected].reshape(-1, 3)
            active = active[active[:, 2] <= bottom.min() + config["heel_toe_active_height_band_m"]]
            contact_points[t][side] = active
            if clearance[t, side] <= config["geometry_contact_height_m"]:
                support.extend(active)
        if support:
            margins[t] = static_margin(com[t, :2], support)
        for contact in data.contact[:data.ncon]:
            if contact.geom1 != 0 and contact.geom2 != 0 and contact.dist < 0:
                depth = float(-contact.dist)
                self_depth[t] = max(self_depth[t], depth)
                key = " / ".join(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(i)) for i in [contact.geom1, contact.geom2])
                self_pairs[key] = max(self_pairs.get(key, 0), depth)
    contacts = clearance <= config["geometry_contact_height_m"]
    body_vel = derivative(body_pos, fps)
    body_omega = np.stack([angular_velocity_world(body_quat[:, i], fps) for i in range(model.nbody - 1)], axis=1)
    slips = np.zeros((frames, 2))
    for side, indices in enumerate(feet):
        body = int(model.geom_bodyid[indices[0]]) - 1
        for t in range(frames):
            offset = contact_points[t][side] - body_pos[t, body]
            material_velocity = body_vel[t, body] + np.cross(body_omega[t, body], offset)
            slips[t, side] = np.linalg.norm(material_velocity[:, :2], axis=-1).max()
    limits = np.array([contract["controller"]["joint_position_limits"][name] for name in contract["csv"]["joint_names"]])
    rated = np.array([contract["controller"]["velocity_limits"][name] for name in contract["csv"]["joint_names"]])
    joint_vel = derivative(qpos[:, 7:], fps)
    ratio = abs(joint_vel) / rated
    limit_error = np.maximum(limits[:, 0] - qpos[:, 7:], qpos[:, 7:] - limits[:, 1]).clip(min=0)
    gate = json.loads((ROOT / "configs/evaluation.json").read_text())["kinematic_gate"]
    slip_fraction = float(((slips > gate["max_contact_foot_slip_speed_m_s"]) & contacts).sum() / max(contacts.sum(), 1))
    endpoint = config["endpoint_gate"]
    root_vel = derivative(qpos[:, :3], fps)
    root_omega = angular_velocity_world(qpos[:, 3:7], fps)
    root_acceleration=derivative(root_vel,fps)
    endpoint_ok = bool(min(margins[0], margins[-1]) >= endpoint["min_static_com_support_margin_m"] and
        abs(joint_vel[[0, -1]]).max() <= endpoint["max_joint_speed_rad_s"] and
        np.linalg.norm(root_vel[[0, -1]], axis=-1).max() <= endpoint["max_root_linear_speed_m_s"] and
        np.linalg.norm(root_omega[[0, -1]], axis=-1).max() <= endpoint["max_root_angular_speed_rad_s"] and contacts[[0, -1]].all())
    checks = {"finite_values": bool(np.isfinite(qpos).all()),
        "unit_quaternions": bool(abs(np.linalg.norm(qpos[:, 3:7], axis=-1) - 1).max() <= gate["quaternion_norm_tolerance"]),
        "joint_limits": bool(limit_error.max() <= gate["joint_limit_tolerance_rad"]),
        "ground_penetration": bool(ground_depth.max() <= gate["max_ground_penetration_m"]),
        "self_collision": bool(self_depth.max() <= gate["max_self_collision_depth_m"]),
        "foot_slip_fraction": bool(slip_fraction <= gate["max_contact_slip_frame_fraction"]),
        "joint_velocity": bool(ratio.max() <= gate["max_joint_velocity_limit_ratio"] and (ratio > 1).mean() <= gate["max_joint_velocity_limit_exceedance_fraction"]),
        "support_present_all_frames": bool(contacts.any(axis=1).all()), "static_endpoint_hold": endpoint_ok,
        "root_linear_speed":bool(np.linalg.norm(root_vel,axis=1).max()<=gate.get("max_root_linear_speed_m_s",np.inf)),
        "root_linear_acceleration":bool(np.linalg.norm(root_acceleration,axis=1).max()<=gate.get("max_root_linear_acceleration_m_s2",np.inf))}
    metrics = {"frames": frames, "fps": fps, "max_ground_penetration_m": float(ground_depth.max()),
        "max_self_collision_depth_m": float(self_depth.max()), "self_collision_pairs_max_depth_m": self_pairs,
        "stance_foot_frames": int(contacts.sum()), "slip_violating_stance_foot_frames": int(((slips > .15) & contacts).sum()),
        "slip_frame_fraction": slip_fraction, "max_stance_slip_speed_m_s": float(slips[contacts].max()) if contacts.any() else None,
        "max_joint_velocity_ratio": float(ratio.max()), "velocity_limit_exceedance_fraction": float((ratio > 1).mean()),
        "max_root_linear_speed_m_s":float(np.linalg.norm(root_vel,axis=1).max()),"max_root_linear_acceleration_m_s2":float(np.linalg.norm(root_acceleration,axis=1).max()),
        "max_joint_limit_violation_rad": float(limit_error.max()), "endpoint_static_com_margin_m": margins[[0, -1]].tolist(),
        "endpoint_max_joint_speed_rad_s": float(abs(joint_vel[[0, -1]]).max()),
        "unsupported_frame_fraction": float((~contacts.any(axis=1)).mean()), "checks": checks, "passed": all(checks.values())}
    arrays = {"body_names": np.array(names), "body_pos_world_m": body_pos, "body_quat_world_wxyz": body_quat,
        "body_lin_vel_world_m_s": body_vel, "body_ang_vel_world_rad_s": body_omega,
        "joint_vel_rad_s": joint_vel, "root_lin_vel_world_m_s": root_vel, "root_ang_vel_world_rad_s": root_omega,
        "foot_contacts": contacts, "foot_clearance_m": clearance, "foot_slip_speed_m_s": slips,
        "static_com_support_margin_m": margins, "com_world_m": com}
    return metrics, arrays
