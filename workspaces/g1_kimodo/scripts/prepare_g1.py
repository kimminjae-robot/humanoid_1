"""Build one primitive-collision G1 MJCF for both physics backends."""

import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    source = ROOT / "third_party/kimodo/kimodo/assets/skeletons/g1skel34/xml/g1.xml"
    urdf = ROOT / "assets/source/g1_29dof_rev_1_0.urdf"
    contract_path = ROOT / "configs/motion_contract.json"
    contract = json.loads(contract_path.read_text())
    joint_names = contract["csv"]["joint_names"]
    out = ROOT / "assets/g1"
    out.mkdir(parents=True, exist_ok=True)
    original = ET.parse(source).getroot()
    robot = ET.Element("mujoco", model="g1_kimodo_29dof")
    ET.SubElement(robot, "compiler", angle="radian", autolimits="true")
    defaults = copy.deepcopy(original.find("default"))
    for parent in list(defaults.iter()):
        for child in list(parent):
            if child.tag == "default" and child.get("class") == "visual":
                parent.remove(child)
    # Eliminate an importer-dependent Coulomb-friction convention in baseline.
    for joint in defaults.iter("joint"):
        joint.set("frictionloss", "0")
        joint.set("damping", "0")
    robot.append(defaults)
    world = ET.SubElement(robot, "worldbody")
    pelvis = copy.deepcopy(original.find("worldbody/body"))
    world.append(pelvis)
    for parent in list(pelvis.iter()):
        for child in list(parent):
            if child.tag == "geom" and (child.get("class") == "visual" or child.get("mesh")):
                parent.remove(child)
    for geom in robot.iter("geom"):
        geom.attrib.pop("material", None)
        if geom.get("class") == "collision":
            geom.set("friction", "0.8 0.005 0.0001")
    ET.indent(robot)
    robot_path = out / "g1_robot.xml"
    ET.ElementTree(robot).write(robot_path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(robot_path))
    assert model.nq == 36 and model.nv == 35 and model.njnt == 30
    actual = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt)]
    assert actual == joint_names
    limits = {j.get("name"): j.find("limit").attrib for j in ET.parse(urdf).getroot().findall("joint") if j.find("limit") is not None}
    parameters = {}
    motors = ET.SubElement(robot, "actuator")
    for index, name in enumerate(joint_names):
        j = index + 1
        cap = float(max(abs(model.jnt_actfrcrange[j])))
        assert cap > 0 and name in limits
        armature = float(model.dof_armature[index + 6])
        neutral = -0.1 if "hip_pitch" in name else 0.3 if "knee" in name else -0.2 if "ankle_pitch" in name else 0.0
        # Initial gains are design choices, not calibrated hardware values.
        if "knee" in name:
            kp, kd = 100.0, 4.0
        elif "hip" in name or "waist" in name:
            kp, kd = 80.0, 3.0
        elif "ankle" in name:
            kp, kd = 40.0, 2.0
        elif "wrist" in name:
            kp, kd = 10.0, 0.5
        else:
            kp, kd = 40.0, 2.0
        parameters[name] = {
            "neutral_rad": neutral, "kp_nm_per_rad": kp, "kd_nm_s_per_rad": kd,
            "effort_limit_nm": cap, "velocity_limit_rad_s": float(limits[name]["velocity"]),
            "position_limit_rad": model.jnt_range[j].tolist(), "armature_kg_m2": armature,
            "frictionloss_nm": 0.0, "joint_damping_nm_s_rad": 0.0, "action_scale_rad": 0.25,
            "urdf_effort_nm": float(limits[name]["effort"]),
        }
        ET.SubElement(motors, "motor", name=name + "_motor", joint=name, gear="1", ctrllimited="true",
                      ctrlrange=f"{-cap} {cap}", forcelimited="true", forcerange=f"{-cap} {cap}")
    ET.indent(robot)
    ET.ElementTree(robot).write(robot_path, encoding="unicode")
    scene = ET.Element("mujoco", model="g1_kimodo_scene")
    ET.SubElement(scene, "include", file="g1_robot.xml")
    ET.SubElement(scene, "option", timestep="0.002", gravity="0 0 -9.81", integrator="implicitfast")
    ET.SubElement(ET.SubElement(scene, "worldbody"), "geom", name="floor", type="plane", size="10 10 0.1", friction="0.8 0.005 0.0001")
    ET.indent(scene)
    scene_path = out / "g1_scene.xml"
    ET.ElementTree(scene).write(scene_path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    bodies = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]
    neutral = np.array([parameters[n]["neutral_rad"] for n in joint_names])
    rng = np.random.default_rng(731)
    fixtures = []
    for i in range(11):
        q = neutral.copy()
        if i:
            q += rng.uniform(-0.12, 0.12, size=29)
        q = np.clip(q, model.jnt_range[1:, 0] + 0.01, model.jnt_range[1:, 1] - 0.01)
        root_quat = np.array([1.0, 0.01 * i, -0.007 * i, 0.012 * i])
        root_quat /= np.linalg.norm(root_quat)
        root_pos = [0.02 * i, -0.015 * i, 2.0 + 0.01 * i]
        mujoco.mj_resetData(model, data)
        data.qpos[:] = [*root_pos, *root_quat, *q]
        mujoco.mj_forward(model, data)
        fixtures.append({"id": i, "root_pos": root_pos, "root_quat_wxyz": root_quat.tolist(), "joint_pos": q.tolist(),
                         "body_pos_world": data.xpos[1:].tolist(), "body_quat_world_wxyz": data.xquat[1:].tolist()})
    dump(ROOT / "configs/g1_fixtures.json", {"joint_names": joint_names, "body_names": bodies, "fixtures": fixtures})
    dump(ROOT / "configs/g1_actuators.json", {"status": "initial_baseline_not_motion_tracking_validated", "source": "Kimodo MJCF effort/armature; official Unitree URDF velocity; project initial PD gains", "joints": parameters})
    manifest = {
        "robot": "g1_29dof_no_fingers", "source_mjcf_sha256": sha256(source), "velocity_source_urdf_sha256": sha256(urdf),
        "generated_robot_sha256": sha256(robot_path), "generated_scene_sha256": sha256(scene_path),
        "nq": model.nq, "nv": model.nv, "actuator_count": model.nu, "body_names": bodies, "joint_names": joint_names,
        "total_mass_kg": float(model.body_mass.sum()),
        "transformations": ["remove decorative meshes/materials and keep original primitive collisions", "preserve source hierarchy/joint axes/limits/body inertials", "remove rendering/sensors and explicit source contact pairs", "set baseline joint damping/frictionloss to zero for both backends", "add torque-only motors and shared ground friction 0.8"],
        "bodies": {name: {"mass_kg": float(model.body_mass[i]), "inertia_diagonal_kg_m2": model.body_inertia[i].tolist(),
                           "com_pos_body_m": model.body_ipos[i].tolist(), "com_quat_body_wxyz": model.body_iquat[i].tolist()} for i, name in enumerate(bodies, 1)},
        "collision_geometry_count": int(model.ngeom - 1),
        "collision_geometry": {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i): {
            "body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[i]),
            "type": mujoco.mjtGeom(int(model.geom_type[i])).name, "size": model.geom_size[i].tolist(),
            "position_body_m": model.geom_pos[i].tolist(), "quaternion_body_wxyz": model.geom_quat[i].tolist()
        } for i in range(model.ngeom) if model.geom_bodyid[i] != 0},
    }
    dump(ROOT / "reports/g1_asset_manifest.json", manifest)
    controller = contract["controller"]
    for field, parameter in [("joint_default_positions", "neutral_rad"), ("kp", "kp_nm_per_rad"), ("kd", "kd_nm_s_per_rad"), ("effort_limits", "effort_limit_nm"), ("velocity_limits", "velocity_limit_rad_s"), ("action_scale", "action_scale_rad")]:
        controller[field] = {name: parameters[name][parameter] for name in joint_names}
    controller["action_target_convention"] = "reference_plus_scaled_residual"
    controller["joint_position_limits"] = {name: values["position_limit_rad"] for name, values in parameters.items()}
    controller["position_target_clipping"] = "joint_position_limits"
    controller["reference_target_update_hz"] = controller["control_hz"]
    controller["target_hold_physics_steps"] = controller["decimation"]
    controller["torque_update_hz"] = controller["physics_hz"]
    controller["action_clipping"] = [-1.0, 1.0]
    controller["velocity_limit_handling"] = "shared_outward_torque_suppression_above_rated_speed_no_state_clipping"
    controller["robot_hashes"] = {**(controller.get("robot_hashes") or {}), "mjcf": sha256(robot_path), "scene": sha256(scene_path)}
    dump(contract_path, contract)
    # This is a gravity/contact engine smoke with a G1 body, not a balancing controller.
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = [0, 0, 2]
    data.qpos[7:] = neutral
    z_initial = float(data.qpos[2])
    for _ in range(50):
        mujoco.mj_step(model, data)
    z_after_0_1 = float(data.qpos[2])
    contacts = 0
    finite = True
    for _ in range(950):
        mujoco.mj_step(model, data)
        contacts = max(contacts, data.ncon)
        finite = finite and bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    checks = {"29_motors": model.nu == 29, "floating_base": model.nq == 36, "gravity_moves_root": z_initial - z_after_0_1 > 0.02,
              "ground_contact": contacts > 0, "finite_1000_steps": finite}
    dump(ROOT / "reports/g1_mujoco_validation.json", {"scope": "asset_load_free_base_gravity_contact_not_tracking", "checks": checks, "passed": all(checks.values()), "z_before_m": z_initial, "z_after_0_1s_m": z_after_0_1, "max_contacts": contacts, "steps": 1000})
    print(json.dumps({"nq": model.nq, "nv": model.nv, "nu": model.nu, "mass_kg": manifest["total_mass_kg"], "fixtures": len(fixtures), "checks": checks}))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
