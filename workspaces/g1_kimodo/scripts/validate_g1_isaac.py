"""Import the shared G1 MJCF and compare actual PhysX FK and parameters."""

import argparse
import hashlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--force-convert", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import numpy as np
import torch
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from isaacsim.core.utils.extensions import enable_extension

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
from isaaclab.utils.math import matrix_from_quat
from isaaclab.sensors import ContactSensor, ContactSensorCfg

ROOT = Path(__file__).resolve().parents[1]


def array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def quat_angle_deg(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    b /= np.linalg.norm(b, axis=-1, keepdims=True)
    return np.rad2deg(2 * np.arccos(np.clip(np.abs((a * b).sum(-1)), 0, 1)))


def main():
    enable_extension("isaacsim.asset.importer.mjcf")
    fixtures = json.loads((ROOT / "configs/g1_fixtures.json").read_text())
    actuators = json.loads((ROOT / "configs/g1_actuators.json").read_text())["joints"]
    manifest = json.loads((ROOT / "reports/g1_asset_manifest.json").read_text())
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.002, device=args.device,
        physx=sim_utils.PhysxCfg(solver_type=1, min_position_iteration_count=16, min_velocity_iteration_count=4)))
    converted = MjcfConverter(MjcfConverterCfg(asset_path=str(ROOT / "assets/g1/g1_robot.xml"),
        usd_dir=str(ROOT / "assets/g1/usd"), usd_file_name="g1.usd", fix_base=False,
        make_instanceable=False, import_inertia_tensor=True, self_collision=True, force_usd_conversion=args.force_convert))
    print("G1_STAGE converted", converted.usd_path, flush=True)
    # Isaac Sim 5.1's MJCF importer authors a second root on worldBody and
    # instanceable collision scopes even when make_instanceable is false.
    asset_stage = Usd.Stage.Open(converted.usd_path)
    repairs = []
    for prim in list(asset_stage.Traverse()):
        if prim.GetName() == "worldBody" and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            repairs.append({"remove_duplicate_world_articulation_root": str(prim.GetPath())})
        if prim.IsInstanceable():
            prim.SetInstanceable(False)
            repairs.append({"disable_instanceability_for_collision_access": str(prim.GetPath())})
    # Restore exact source capsule orientation, including short foot capsules
    # whose importer rotation rounding otherwise introduces ~0.02 degree drift.
    xforms = UsdGeom.XformCache()
    for prim in list(asset_stage.Traverse()):
        if not prim.IsA(UsdGeom.Capsule) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        expected = manifest["collision_geometry"][prim.GetName()]
        body = prim.GetParent()
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        parent_to_body, _ = xforms.ComputeRelativeTransform(prim.GetParent(), body)
        quaternion = expected["quaternion_body_wxyz"]
        desired = Gf.Matrix4d(1.0)
        desired.SetRotate(Gf.Quatd(quaternion[0], Gf.Vec3d(*quaternion[1:])))
        desired.SetTranslateOnly(Gf.Vec3d(*expected["position_body_m"]))
        UsdGeom.Capsule(prim).GetAxisAttr().Set("Z")
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTransformOp(precision=UsdGeom.XformOp.PrecisionDouble, opSuffix="sourceAlignment").Set(desired * parent_to_body.GetInverse())
        repairs.append({"restore_source_capsule_orientation": str(prim.GetPath())})
    asset_stage.GetRootLayer().Save()
    (ROOT / "reports/g1_usd_repairs.json").write_text(json.dumps(repairs, indent=2) + "\n")
    ground = sim_utils.GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8,
        dynamic_friction=0.8, restitution=0.0, friction_combine_mode="average"))
    ground.func("/World/Ground", ground)
    print("G1_STAGE ground_spawned", flush=True)
    robot = Articulation(ArticulationCfg(
        prim_path="/World/G1",
        spawn=sim_utils.UsdFileCfg(usd_path=converted.usd_path, activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, linear_damping=0.0, angular_damping=0.0,
                max_linear_velocity=1000.0, max_angular_velocity=1000.0, enable_gyroscopic_forces=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=False, enabled_self_collisions=True,
                solver_position_iteration_count=16, solver_velocity_iteration_count=4),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0), joint_pos={n: p["neutral_rad"] for n, p in actuators.items()}, joint_vel={".*": 0.0}),
        soft_joint_pos_limit_factor=1.0,
        actuators={"torque": IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0,
            effort_limit={n: p["effort_limit_nm"] for n, p in actuators.items()},
            effort_limit_sim={n: p["effort_limit_nm"] for n, p in actuators.items()},
            velocity_limit_sim=1000000.0, armature={n: p["armature_kg_m2"] for n, p in actuators.items()},
            friction=0.0, dynamic_friction=0.0, viscous_friction=0.0)}))
    print("G1_STAGE articulation_spawned", flush=True)
    material = sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8, restitution=0.0)
    material.func("/World/G1Material", material)
    sim_utils.bind_physics_material("/World/G1", "/World/G1Material")
    contact = ContactSensor(ContactSensorCfg(prim_path="/World/G1/pelvis/(" + "|".join(fixtures["body_names"]) + ")", update_period=0.0))
    sim.reset()
    print("G1_STAGE simulation_reset", flush=True)
    robot.update(0.002)
    source_joints, source_bodies = fixtures["joint_names"], fixtures["body_names"]
    isaac_joints, isaac_bodies = robot.joint_names, robot.body_names
    assert len(isaac_joints) == 29 and set(isaac_joints) == set(source_joints), isaac_joints
    assert set(isaac_bodies) == set(source_bodies), (isaac_bodies, source_bodies)
    joint_map = [isaac_joints.index(n) for n in source_joints]
    body_map = [isaac_bodies.index(n) for n in source_bodies]

    def reset(fixture):
        robot.reset()
        pose = torch.tensor([[*fixture["root_pos"], *fixture["root_quat_wxyz"]]], device=sim.device)
        robot.write_root_pose_to_sim(pose)
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device))
        q = torch.zeros((1, 29), device=sim.device)
        q[:, joint_map] = torch.tensor([fixture["joint_pos"]], device=sim.device)
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_effort_target(torch.zeros_like(q))
        robot.write_data_to_sim()

    fk_results = []
    for fixture in fixtures["fixtures"]:
        reset(fixture)
        sim.step(render=False)
        robot.update(0.002)
        observed_pos = array(robot.data.body_link_pos_w[0, body_map]).astype(np.float64)
        observed_quat = array(robot.data.body_link_quat_w[0, body_map]).astype(np.float64)
        position_errors = np.linalg.norm(observed_pos - np.asarray(fixture["body_pos_world"]), axis=-1)
        angle_errors = quat_angle_deg(observed_quat, fixture["body_quat_world_wxyz"])
        fk_results.append({"fixture_id": fixture["id"], "max_position_error_m": float(position_errors.max()),
                           "max_orientation_error_deg": float(angle_errors.max()), "worst_position_body": source_bodies[int(position_errors.argmax())],
                           "body_pos_world": observed_pos.tolist(), "body_quat_world_wxyz": observed_quat.tolist()})
    masses = array(robot.root_physx_view.get_masses())[0, body_map]
    inertia = array(robot.root_physx_view.get_inertias())[0, body_map].reshape(-1, 3, 3)
    armature = array(robot.root_physx_view.get_dof_armatures())[0, joint_map]
    effort = array(robot.data.joint_effort_limits)[0, joint_map]
    limits = array(robot.data.joint_pos_limits)[0, joint_map]
    expected_masses = np.array([manifest["bodies"][b]["mass_kg"] for b in source_bodies])
    expected_inertia = np.sort([manifest["bodies"][b]["inertia_diagonal_kg_m2"] for b in source_bodies], axis=-1)
    observed_inertia = np.sort(np.linalg.eigvalsh(inertia.astype(np.float64)), axis=-1)
    inertia_rotations = array(matrix_from_quat(torch.tensor([manifest["bodies"][b]["com_quat_body_wxyz"] for b in source_bodies], dtype=torch.float64)))
    expected_full_inertia = np.stack([rotation @ np.diag(manifest["bodies"][b]["inertia_diagonal_kg_m2"]) @ rotation.T for b, rotation in zip(source_bodies, inertia_rotations)])
    actual_com_positions = array(robot.data.body_com_pos_b)[0, body_map]
    expected_com_positions = np.array([manifest["bodies"][b]["com_pos_body_m"] for b in source_bodies])
    expected_armature = np.array([actuators[n]["armature_kg_m2"] for n in source_joints])
    expected_effort = np.array([actuators[n]["effort_limit_nm"] for n in source_joints])
    expected_limits = np.array([actuators[n]["position_limit_rad"] for n in source_joints])
    parameter_errors = {"max_mass_error_kg": float(np.abs(masses - expected_masses).max()),
        "max_principal_inertia_error_kg_m2": float(np.abs(observed_inertia - expected_inertia).max()),
        "max_full_body_frame_inertia_error_kg_m2": float(np.abs(inertia - expected_full_inertia).max()),
        "max_com_position_error_m": float(np.linalg.norm(actual_com_positions - expected_com_positions, axis=-1).max()),
        "max_armature_error_kg_m2": float(np.abs(armature - expected_armature).max()),
        "max_effort_limit_error_nm": float(np.abs(effort - expected_effort).max()),
        "max_joint_limit_error_rad": float(np.abs(limits - expected_limits).max())}
    reset(fixtures["fixtures"][0])
    initial_z = float(robot.data.root_pos_w[0, 2])
    finite = True
    for _ in range(50):
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(0.002)
        finite = finite and bool(torch.isfinite(robot.data.root_state_w).all() and torch.isfinite(robot.data.joint_pos).all())
    final_z = float(robot.data.root_pos_w[0, 2])
    max_total_upward_contact_force = 0.0
    for _ in range(950):
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(0.002)
        contact.update(0.002)
        finite = finite and bool(torch.isfinite(robot.data.root_state_w).all() and torch.isfinite(robot.data.joint_pos).all())
        # Internal self-contact forces cancel when summed over all robot bodies.
        total_fz = float(contact.data.net_forces_w[0, :, 2].sum())
        max_total_upward_contact_force = max(max_total_upward_contact_force, total_fz)
    checks = {"29_joint_name_bijection": len(joint_map) == 29,
        "11_fk_fixtures_position_below_5mm": max(r["max_position_error_m"] for r in fk_results) < 0.005,
        "11_fk_fixtures_rotation_below_1deg": max(r["max_orientation_error_deg"] for r in fk_results) < 1.0,
        "mass_parity": parameter_errors["max_mass_error_kg"] < 1e-5,
        "principal_inertia_parity": parameter_errors["max_principal_inertia_error_kg_m2"] < 1e-5,
        "full_inertia_tensor_parity": parameter_errors["max_full_body_frame_inertia_error_kg_m2"] < 1e-5,
        "com_position_parity": parameter_errors["max_com_position_error_m"] < 1e-5,
        "armature_parity": parameter_errors["max_armature_error_kg_m2"] < 1e-6,
        "effort_limit_parity": parameter_errors["max_effort_limit_error_nm"] < 1e-4,
        "joint_limit_parity": parameter_errors["max_joint_limit_error_rad"] < 1e-5,
        "free_base_gravity": initial_z - final_z > 0.02, "finite_physics_steps": finite}
    checks["ground_contact_force_detected"] = max_total_upward_contact_force > 50.0
    collision_prims = [str(p.GetPath()) for p in sim.stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI) and str(p.GetPath()).startswith("/World/G1")]
    checks["collision_geometry_present"] = bool(collision_prims)
    checks["collision_geometry_count_matches"] = len(collision_prims) == manifest["collision_geometry_count"]
    collider_details = []
    xform_cache = UsdGeom.XformCache()
    for path in collision_prims:
        prim = sim.stage.GetPrimAtPath(path)
        parent = prim.GetParent()
        while parent and not parent.HasAPI(UsdPhysics.RigidBodyAPI):
            parent = parent.GetParent()
        transform, _ = xform_cache.ComputeRelativeTransform(prim, parent)
        details = {"path": path, "body": parent.GetName(), "type": prim.GetTypeName(),
                   "position_body_m": list(transform.ExtractTranslation()), "attributes": {}}
        details["linear_transform_body"] = np.array(transform)[:3, :3].tolist()
        for name in ["radius", "height", "size", "axis"]:
            value = prim.GetAttribute(name).Get()
            if value is not None:
                details["attributes"][name] = value
        collider_details.append(details)
    geometry_errors = {"max_center_error_m": 0.0, "max_radius_error_m": 0.0, "max_height_error_m": 0.0, "max_axis_angle_deg": 0.0}
    geometry_types_match = True
    for collider in collider_details:
        expected = manifest["collision_geometry"][Path(collider["path"]).name]
        geometry_types_match = geometry_types_match and collider["body"] == expected["body"] and collider["type"].lower() == expected["type"].removeprefix("mjGEOM_").lower()
        geometry_errors["max_center_error_m"] = max(geometry_errors["max_center_error_m"], float(np.linalg.norm(np.asarray(collider["position_body_m"]) - expected["position_body_m"])))
        geometry_errors["max_radius_error_m"] = max(geometry_errors["max_radius_error_m"], abs(collider["attributes"]["radius"] - expected["size"][0]))
        if collider["type"] == "Capsule":
            geometry_errors["max_height_error_m"] = max(geometry_errors["max_height_error_m"], abs(collider["attributes"]["height"] - 2 * expected["size"][1]))
            axis = np.asarray(collider["linear_transform_body"])["XYZ".index(collider["attributes"]["axis"])]
            rotation = array(matrix_from_quat(torch.tensor(expected["quaternion_body_wxyz"], dtype=torch.float64)))
            cosine = np.clip(abs(np.dot(axis / np.linalg.norm(axis), rotation[:, 2])), 0, 1)
            geometry_errors["max_axis_angle_deg"] = max(geometry_errors["max_axis_angle_deg"], float(np.rad2deg(np.arccos(cosine))))
    checks["collision_geometry_type_and_body_parity"] = geometry_types_match
    checks["collision_geometry_dimensions_and_centers_parity"] = max(geometry_errors[k] for k in ["max_center_error_m", "max_radius_error_m", "max_height_error_m"]) < 1e-5
    checks["capsule_axis_parity"] = geometry_errors["max_axis_angle_deg"] < 0.01
    report = {"scope": "G1_asset_FK_inertia_actuator_and_free_base_smoke_not_motion_tracking",
        "passed": all(checks.values()), "checks": checks, "parameter_errors": parameter_errors, "geometry_errors": geometry_errors,
        "isaac_joint_names": isaac_joints, "isaac_body_names": isaac_bodies,
        "source_to_isaac_joint_indices": joint_map, "source_to_isaac_body_indices": body_map,
        "fk_results": fk_results, "collision_prims": collision_prims, "collider_details": collider_details, "usd_path": converted.usd_path,
        "usd_sha256": hashlib.sha256(Path(converted.usd_path).read_bytes()).hexdigest(),
        "usd_file_sha256": {str(p.relative_to(Path(converted.usd_path).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in Path(converted.usd_path).parent.rglob("*.usd")},
        "actual_mass_kg": masses.tolist(), "actual_inertia": inertia.tolist(),
        "free_base_z_initial_m": initial_z, "free_base_z_after_0_1s_m": final_z}
    report["dynamic_smoke_steps"] = 1000
    report["max_total_upward_contact_force_n"] = max_total_upward_contact_force
    (ROOT / "reports/g1_isaac_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    contract_path = ROOT / "configs/motion_contract.json"
    contract = json.loads(contract_path.read_text())
    contract["controller"]["backend_joint_maps"] = {"csv_source": source_joints, "isaac": isaac_joints, "mujoco_motor": source_joints,
        "source_to_isaac_indices": joint_map, "source_to_mujoco_motor_indices": list(range(29))}
    contract["controller"]["robot_hashes"]["usd"] = report["usd_sha256"]
    contract["controller"]["robot_hashes"]["usd_files"] = report["usd_file_sha256"]
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    print("G1_ISAAC_VALIDATION", json.dumps({"checks": checks, "parameter_errors": parameter_errors, "passed": report["passed"]}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        import traceback
        traceback.print_exc()
        code = 1
    finally:
        # run_g1_validation.py bounds Kit teardown outside this process/GIL.
        app.close(wait_for_replicator=False)
    raise SystemExit(code)
