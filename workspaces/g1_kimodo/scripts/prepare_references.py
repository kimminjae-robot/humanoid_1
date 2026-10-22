"""Create versioned 50 Hz G1 references and evaluate unchanged kinematic gates."""

import hashlib
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from reference_core import analyze, continuous_quaternions, foot_geoms, geometry_bottoms, interpolate, model_and_config, rotations, wxyz


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def runs(mask):
    boundaries = np.r_[0, np.flatnonzero(mask[1:] != mask[:-1]) + 1, len(mask)]
    return [(int(a), int(b), bool(mask[a])) for a, b in zip(boundaries[:-1], boundaries[1:])]


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x**3 * (10 + x * (-15 + 6 * x))


def timeline(span, forward_distance, stationary, cfg):
    fps = cfg["target_fps"]; ramp = cfg["timing"]["speed_ramp_s"]
    moving_duration = span + ramp if stationary else max(span*cfg["walk"].get("minimum_temporal_stretch",1.0), forward_distance / cfg["walk"]["target_mean_forward_speed_m_s"])
    moving_duration = np.ceil(moving_duration * fps) / fps
    moving_duration = max(moving_duration, 2 * ramp)
    t = np.arange(round((cfg["timing"]["pre_hold_s"] + moving_duration + cfg["timing"]["post_hold_s"]) * fps) + 1) / fps
    u = np.clip(t - cfg["timing"]["pre_hold_s"], 0, moving_duration)
    integral = lambda x: x**6 - 3*x**5 + 2.5*x**4
    effective = u - ramp / 2
    entering = u < ramp
    exiting = u > moving_duration - ramp
    effective[entering] = ramp * integral(u[entering] / ramp)
    effective[exiting] = moving_duration - ramp - ramp * integral((moving_duration-u[exiting]) / ramp)
    return t, np.clip(effective * span / (moving_duration - ramp), 0, span)


def foot_plan(qpos, model, cfg, stationary):
    data = mujoco.MjData(model); feet = foot_geoms(model)
    foot_ids = [int(model.geom_bodyid[items[0]]) for items in feet]
    positions = np.zeros((len(qpos), 2, 3)); yaw = np.zeros((len(qpos), 2)); heights = np.zeros((len(qpos), 2))
    flat_z = []
    for items in feet:
        axis = rotations(model.geom_quat[items]).as_matrix()[:, :, 2]
        bottom = model.geom_pos[items, 2] - model.geom_size[items, 0] - abs(axis[:, 2]) * model.geom_size[items, 1]
        flat_z.append(float(-bottom.min() + cfg["ground_clearance_m"]))
    for t, row in enumerate(qpos):
        data.qpos[:] = row; mujoco.mj_forward(model, data)
        for side in range(2):
            positions[t, side] = data.xpos[foot_ids[side]]
            rotation = data.xmat[foot_ids[side]].reshape(3, 3)
            yaw[t, side] = np.arctan2(rotation[1, 0], rotation[0, 0])
            heights[t, side] = geometry_bottoms(model, data, np.array(feet[side]))[0].min()
    if not stationary:
        positions[:, :, 0] = qpos[:, 0, None] + (positions[:, :, 0]-qpos[:, 0, None])*cfg["walk"].get("forward_spatial_scale", 1.0)
        positions[:, :, 1] = qpos[:, 1, None] + (positions[:, :, 1]-qpos[:, 1, None])*cfg["walk"].get("lateral_spatial_scale", 1.0)
    contact = heights <= cfg["geometry_contact_height_m"]
    if stationary:
        contact[:] = True
    else:
        for side in range(2):
            # Remove <60ms label islands for planning only; QA uses unfiltered geometry.
            for _ in range(2):
                for start, stop, value in runs(contact[:, side]):
                    if 0 < start and stop < len(qpos) and stop - start < 3:
                        contact[start:stop, side] = not value
        contact[0] = contact[-1] = True
    target_pos = positions.copy(); target_yaw = yaw.copy()
    for side in range(2):
        segments = runs(contact[:, side]); anchors = {}
        for start, stop, value in segments:
            if value:
                xy = np.median(positions[start:stop, side, :2], axis=0)
                angle = np.arctan2(np.sin(yaw[start:stop, side]).mean(), np.cos(yaw[start:stop, side]).mean())
                anchors[(start, stop)] = (np.r_[xy, flat_z[side]], angle)
                target_pos[start:stop, side] = anchors[(start, stop)][0]
                target_yaw[start:stop, side] = angle
        for segment_index, (start, stop, value) in enumerate(segments):
            if value:
                continue
            previous = segments[segment_index-1]; following = segments[segment_index+1]
            a, ya = anchors[previous[:2]]; b, yb = anchors[following[:2]]
            phase = np.linspace(0, 1, stop-start+2)[1:-1]
            lift = cfg["walk"]["swing_lift_fraction"]
            horizontal = smoothstep((phase-lift)/(1-2*lift))
            z = cfg["walk"]["swing_clearance_m"] * np.minimum(smoothstep(phase/lift), smoothstep((1-phase)/lift))
            target_pos[start:stop, side] = a[None] + horizontal[:, None]*(b-a)[None]
            target_pos[start:stop, side, 2] += z
            target_yaw[start:stop, side] = ya + horizontal * np.angle(np.exp(1j*(yb-ya)))
    nominal_xy=qpos[:,:2].copy()
    if not stationary and cfg["walk"].get("pelvis_follows_support_center",False):
        weights=contact.astype(float)
        nominal_xy=(target_pos[:,:,:2]*weights[:,:,None]).sum(axis=1)/weights.sum(axis=1)[:,None]
        nominal_xy[:,0]+=.02
        nominal_xy=gaussian_filter1d(nominal_xy,sigma=cfg["walk"]["pelvis_support_center_filter_s"]*cfg["target_fps"],axis=0,mode="nearest")
    pitch=np.zeros((len(qpos),2))
    if not stationary and cfg["walk"].get("stance_toe_rock_max_rad",0)>0:
        ramp=cfg["timing"]["speed_ramp_s"];t=np.arange(len(qpos))/cfg["target_fps"]
        fade=np.minimum(smoothstep(np.maximum(t-cfg["timing"]["pre_hold_s"],0)/ramp),
            smoothstep(np.maximum(t[-1]-cfg["timing"]["post_hold_s"]-t,0)/ramp))
        for side,items in enumerate(feet):
            axes=rotations(model.geom_quat[items]).as_matrix()[:,:,2]
            endpoints=(model.geom_pos[items,None,:]+np.array([-1,1])[None,:,None]*axes[:,None,:]*model.geom_size[items,1,None,None]).reshape(-1,3)
            toe=endpoints[np.argmax(endpoints[:,0])]
            for start,stop,value in runs(contact[:,side]):
                if not value:continue
                for frame in range(start,stop):
                    yaw_rotation=Rotation.from_euler("z",target_yaw[frame,side])
                    forward=yaw_rotation.apply([1,0,0])
                    lead=float(np.dot(np.r_[nominal_xy[frame],qpos[frame,2]]-target_pos[frame,side],forward))
                    amount=smoothstep((lead-cfg["walk"]["toe_rock_start_pelvis_lead_m"])/(cfg["walk"]["toe_rock_full_pelvis_lead_m"]-cfg["walk"]["toe_rock_start_pelvis_lead_m"]))
                    pitch[frame,side]=cfg["walk"]["stance_toe_rock_max_rad"]*amount*fade[frame]
                    tilted=yaw_rotation*Rotation.from_euler("y",pitch[frame,side])
                    target_pos[frame,side]+=yaw_rotation.apply(toe)-tilted.apply(toe)
            segments=runs(contact[:,side])
            for segment_index,(start,stop,value) in enumerate(segments):
                if value:continue
                a=start-1;b=stop;phase=np.linspace(0,1,stop-start+2)[1:-1];lift=cfg["walk"]["swing_lift_fraction"]
                horizontal=smoothstep((phase-lift)/(1-2*lift))
                target_pos[start:stop,side,:2]=target_pos[a,side,:2]+horizontal[:,None]*(target_pos[b,side,:2]-target_pos[a,side,:2])
                pitch[start:stop,side]=pitch[a,side]+horizontal*(pitch[b,side]-pitch[a,side])
            for frame in range(len(qpos)):
                rotation=Rotation.from_euler("z",target_yaw[frame,side])*Rotation.from_euler("y",pitch[frame,side])
                relative_bottom=rotation.apply(model.geom_pos[items])[:,2]-model.geom_size[items,0]-abs(rotation.apply(axes)[:,2])*model.geom_size[items,1]
                clearance=cfg["ground_clearance_m"]
                if not contact[frame,side]:
                    start,stop,_=next((a,b,v) for a,b,v in segments if a<=frame<b)
                    phase=(frame-start+1)/(stop-start+1);lift=cfg["walk"]["swing_lift_fraction"]
                    clearance+=cfg["walk"]["swing_clearance_m"]*min(smoothstep(phase/lift),smoothstep((1-phase)/lift))
                target_pos[frame,side,2]=clearance-relative_bottom.min()
    target_rotations = [[Rotation.from_euler("z",target_yaw[t,side])*Rotation.from_euler("y",pitch[t,side]) for side in range(2)] for t in range(len(qpos))]
    return target_pos, target_rotations, contact, foot_ids,nominal_xy


def leg_ik(qpos, model, cfg, targets, orientations, foot_ids, stationary, endpoint_weights):
    data = mujoco.MjData(model); result = qpos.copy(); ik = cfg["ik"]
    xy_bound = cfg["walk"].get("root_xy_adjustment_bound_m", 0.15)
    lower = np.r_[model.jnt_range[1:13, 0] + 1e-7, -xy_bound, -xy_bound, cfg["walk"]["root_z_adjustment_bounds_m"][0]]
    upper = np.r_[model.jnt_range[1:13, 1] - 1e-7, xy_bound, xy_bound, cfg["walk"]["root_z_adjustment_bounds_m"][1]]
    if not stationary and 'knee_min_rad' in cfg['walk']:
        lower[[3,9]]=np.maximum(lower[[3,9]],cfg['walk']['knee_min_rad'])
    errors = []; evaluations = 0
    for t, source in enumerate(qpos):
        if t and (stationary or (np.max(abs(source[:19]-qpos[t-1, :19])) < 1e-12 and np.max(abs(targets[t]-targets[t-1])) < 1e-12)):
            result[t, 7:19] = result[t-1, 7:19]; result[t, :3] = result[t-1, :3]
        else:
            prev = result[t-1] if t else source
            initial = np.clip(np.r_[prev[7:19], prev[:3]-source[:3]], lower+1e-9, upper-1e-9)
            def residual(parameters):
                data.qpos[:] = source
                data.qpos[7:19] = parameters[:12]; data.qpos[:3] = source[:3]+parameters[12:]
                mujoco.mj_kinematics(model, data)
                mujoco.mj_comPos(model, data)
                parts = []
                for side, body in enumerate(foot_ids):
                    parts.append((data.xpos[body]-targets[t, side])*ik["position_weight"])
                    parts.append((rotations(data.xquat[body])*orientations[t][side].inv()).as_rotvec()*ik["orientation_weight"])
                parts.extend([(parameters[:12]-source[7:19])*ik["pose_regularization_weight"],
                    (parameters[:12]-prev[7:19])*(ik["temporal_regularization_weight"] if stationary else cfg["walk"].get("joint_temporal_weight", ik["temporal_regularization_weight"])),
                    parameters[12:]*ik["root_height_regularization_weight"],
                    (source[:3]+parameters[12:]-prev[:3])*ik["root_height_temporal_weight"],
                    (data.subtree_com[1, :2]-(targets[t].mean(axis=0)[:2]+np.array([0.04, 0])))*20*endpoint_weights[t]])
                if not stationary and t>=2 and ik.get('acceleration_regularization_weight',0)>0:
                    parts.append((parameters[:12]-2*prev[7:19]+result[t-2,7:19])*ik['acceleration_regularization_weight'])
                if not stationary:
                    parts.append(np.array([(data.qpos[2]-cfg["walk"]["pelvis_height_m"])*cfg["walk"]["pelvis_height_weight"]]))
                return np.concatenate(parts)
            fit = least_squares(residual, initial, bounds=(lower, upper), max_nfev=ik["max_function_evaluations"], ftol=1e-8, xtol=1e-8, gtol=1e-8)
            evaluations += fit.nfev
            result[t, 7:19] = fit.x[:12]; result[t, :3] = source[:3]+fit.x[12:]
        data.qpos[:] = result[t]; mujoco.mj_kinematics(model, data)
        errors.append(max(float(np.linalg.norm(data.xpos[body]-targets[t, side])) for side, body in enumerate(foot_ids)))
    return result, {"max_foot_target_error_m": max(errors), "total_optimizer_evaluations": evaluations,
                    "root_z_correction_range_m": [float((result[:, 2]-qpos[:, 2]).min()), float((result[:, 2]-qpos[:, 2]).max())]}


def correct_arm_clearance(qpos, model, cfg, contract):
    data = mujoco.MjData(model); original = qpos.copy(); output = qpos.copy(); report = {}
    joint_names = contract["csv"]["joint_names"]
    for side, sign in [("left", 1), ("right", -1)]:
        shoulder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, side+"_shoulder_pitch_link")
        hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, side+"_wrist_yaw_link")
        index = 7 + joint_names.index(side+"_shoulder_roll_joint")
        arm_geom = {i for i in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith((side+"_hand", side+"_elbow", side+"_shoulder"))}
        weights = []
        for row in original:
            data.qpos[:] = row; mujoco.mj_kinematics(model, data)
            weights.append(float(np.clip((data.xpos[shoulder, 2]-data.xpos[hand, 2])/cfg["arm_clearance"]["fade_by_hand_height_below_shoulder_m"], 0, 1)))
        for delta in np.arange(0, cfg["arm_clearance"]["max_outward_shoulder_roll_adjustment_rad"]+1e-8, cfg["arm_clearance"]["step_rad"]):
            candidate = output.copy(); candidate[:, index] = original[:, index] + sign*delta*np.array(weights)
            depth = 0
            for row in candidate:
                data.qpos[:] = row; mujoco.mj_forward(model, data)
                for contact in data.contact[:data.ncon]:
                    if contact.geom1 and contact.geom2 and (contact.geom1 in arm_geom or contact.geom2 in arm_geom):
                        depth = max(depth, -float(contact.dist))
            if depth <= cfg["arm_clearance"]["target_self_penetration_m"]:
                break
        output = candidate
        report[side] = {"outward_shoulder_roll_delta_rad": float(delta), "remaining_self_penetration_m": depth}
    return output, report


def process(qpos, motion_id, model, cfg, contract):
    _, raw_arrays = analyze(qpos, 30, model)
    source_times = np.arange(len(qpos))/30
    start, stop = 0, len(qpos)-1
    stationary = motion_id != "walk_slow"
    if motion_id == "stand":
        start = round(cfg["stationary"]["stand_trim_start_s"]*30)
    if not stationary:
        ds = raw_arrays["foot_contacts"].all(axis=1)
        window = round(cfg["walk"]["double_support_search_window_s"]*30)
        first = np.flatnonzero(ds & (np.arange(len(qpos)) <= window))
        last = np.flatnonzero(ds & (np.arange(len(qpos)) >= len(qpos)-1-window))
        if not len(first) or not len(last):
            raise ValueError("No double-support endpoint candidates in fixed search window")
        start = int(first[np.argmax(raw_arrays["static_com_support_margin_m"][first])])
        stop = int(last[np.argmax(raw_arrays["static_com_support_margin_m"][last])])
    source = qpos[start:stop+1].copy()
    if stationary:
        source[:, :3] = np.median(source[:, :3], axis=0)
        source[:, 3:7] = continuous_quaternions(np.tile(continuous_quaternions(source[:, 3:7]).mean(axis=0), (len(source), 1)))
        source[:, 7:19] = np.median(source[:, 7:19], axis=0)
    # One constant world origin/heading transform, never a per-frame root realignment.
    heading = np.arctan2(rotations(source[0, 3:7]).as_matrix()[1, 0], rotations(source[0, 3:7]).as_matrix()[0, 0])
    align = Rotation.from_euler("z", -heading)
    origin = source[0, :3].copy(); origin[2] = 0
    source[:, :3] = align.apply(source[:, :3]-origin)
    source[:, 3:7] = continuous_quaternions(wxyz(align*rotations(source[:, 3:7])))
    if not stationary:
        source[:, 0] *= cfg["walk"].get("forward_spatial_scale", 1.0)
        if cfg["walk"].get("upright_pelvis_keep_yaw",False):
            matrices=rotations(source[:,3:7]).as_matrix()
            yaw=np.arctan2(matrices[:,1,0],matrices[:,0,0])
            source[:,3:7]=continuous_quaternions(wxyz(Rotation.from_euler("z",yaw[:,None])))
    t, mapped = timeline((stop-start)/30, float(source[-1, 0]-source[0, 0]), stationary, cfg)
    target_source = interpolate(source, np.arange(len(source))/30, mapped)
    targets, orientations, contact_plan, foot_ids,nominal_xy = foot_plan(target_source, model, cfg, stationary)
    if not stationary and cfg["walk"].get("pelvis_follows_support_center",False):
        target_source[:,:2]=nominal_xy
    ramp=cfg["timing"]["speed_ramp_s"]
    endpoint_weights=np.maximum(1-smoothstep(np.maximum(t-cfg["timing"]["pre_hold_s"],0)/ramp),
        1-smoothstep(np.maximum(t[-1]-cfg["timing"]["post_hold_s"]-t,0)/ramp))
    if not stationary and cfg["walk"].get("smooth_fixed_pelvis_xy",False):
        endpoint_pose=target_source[[0,-1]].copy()
        balanced,_=leg_ik(endpoint_pose,model,cfg,targets[[0,-1]],[orientations[0],orientations[-1]],foot_ids,False,np.ones(2))
        offsets=balanced[:,:2]-endpoint_pose[:,:2]
        start_weight=1-smoothstep(np.maximum(t-cfg["timing"]["pre_hold_s"],0)/ramp)
        end_weight=1-smoothstep(np.maximum(t[-1]-cfg["timing"]["post_hold_s"]-t,0)/ramp)
        target_source[:,:2]+=start_weight[:,None]*offsets[0]+end_weight[:,None]*offsets[1]
        target_source[:,:2]=gaussian_filter1d(target_source[:,:2],sigma=cfg["walk"]["pelvis_xy_filter_s"]*cfg["target_fps"],axis=0,mode="nearest")
        restricted=json.loads(json.dumps(cfg));restricted["walk"]["root_xy_adjustment_bound_m"]=1e-6
        processed,ik=leg_ik(target_source,model,restricted,targets,orientations,foot_ids,stationary,endpoint_weights)
    else:
        processed, ik = leg_ik(target_source, model, cfg, targets, orientations, foot_ids, stationary, endpoint_weights)
    processed, arm = correct_arm_clearance(processed, model, cfg, contract)
    # Exact reference holds, rather than allowing a numerical IK optimizer to
    # continue drifting while its input source time is constant.
    pre=round(cfg["timing"]["pre_hold_s"]*cfg["target_fps"])
    post=len(processed)-1-round(cfg["timing"]["post_hold_s"]*cfg["target_fps"])
    processed[:pre+1]=processed[pre]
    processed[post:]=processed[post]
    return processed, t, mapped+source_times[start], {"crop_source_frame_indices_inclusive": [start, stop],
        "constant_origin_xyz_m": origin.tolist(), "constant_heading_removed_rad": float(heading),
        "stationary_lower_body_projection": stationary, "forward_spatial_scale": 1.0 if stationary else cfg["walk"].get("forward_spatial_scale",1.0), "ik": ik, "arm_clearance": arm,
        "planned_stance_fraction": contact_plan.mean(axis=0).tolist(), "pre_hold_s": cfg["timing"]["pre_hold_s"],
        "post_hold_s": cfg["timing"]["post_hold_s"]}


def main():
    started = time.monotonic(); model, cfg, contract = model_and_config()
    generation = json.loads((ROOT/"reports/stage_02_generation.json").read_text())
    run_id = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S_%f")+"_stage03_"+cfg["version"]
    run = ROOT/"runs"/run_id; dest = ROOT/"data/processed"/run_id
    run.mkdir(parents=True); dest.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(run/"processing.log"),logging.StreamHandler(sys.stdout)])
    provenance = run/"provenance";provenance.mkdir()
    for path in ["configs/reference_processing.json","configs/evaluation.json","configs/motion_contract.json","scripts/reference_core.py","scripts/prepare_references.py","reports/mujoco_stage03_freeze.txt"]:
        shutil.copy2(ROOT/path,provenance/path.replace('/','__'))
    write(run/"manifest.json",{"run_id":run_id,"source_generation_run":generation["run_id"],"processing_config_sha256":sha(ROOT/"configs/reference_processing.json"),"evaluation_sha256":sha(ROOT/"configs/evaluation.json"),"robot_mjcf_sha256":sha(ROOT/"assets/g1/g1_robot.xml"),"scope":"kinematic_reference_processing_not_dynamic_tracking"})
    clips=[]
    for item in generation["clips"]:
        manifest=json.loads((ROOT/item["manifest"]).read_text()); csv=ROOT/manifest["artifacts"]["qpos.csv"]["path"]
        qpos=np.loadtxt(csv,delimiter=','); source_hash=sha(csv)
        result={"clip_id":item["clip_id"],"motion_id":item["motion_id"],"split":item["split"],"seed":item["seed"],"source_sha256":source_hash,"source_manifest":item["manifest"]}
        before=time.monotonic()
        q,t,source_t,processing=process(qpos,item["motion_id"],model,cfg,contract)
        metrics,arrays=analyze(q,cfg["target_fps"],model)
        metrics["checks"]["foot_ik_tracking"] = processing["ik"]["max_foot_target_error_m"] <= cfg["ik"]["max_foot_target_error_m"]
        metrics["passed"]=all(metrics["checks"].values())
        folder=dest/item["clip_id"];folder.mkdir(); np.savetxt(folder/"qpos.csv",q,delimiter=',')
        with np.load(ROOT/manifest["artifacts"]["motion.npz"]["path"],allow_pickle=False) as raw:
            source_contacts=raw["foot_contacts"][np.clip(np.rint(source_t*30).astype(int),0,len(qpos)-1)]
        bundle={**arrays,"fps":np.array(cfg["target_fps"]),"timestamps_s":t,"source_timestamps_s":source_t,"joint_names":np.array(contract["csv"]["joint_names"]),
            "root_pos_world_m":q[:,:3],"root_quat_world_wxyz":q[:,3:7],"joint_pos_rad":q[:,7:],"source_foot_contacts":source_contacts,
            "source_sha256":np.array(source_hash),"generation_manifest":np.array(json.dumps(manifest)),"robot_manifest_sha256":np.array(sha(ROOT/"reports/g1_asset_manifest.json"))}
        np.savez_compressed(folder/"motion_bundle.npz",**bundle)
        result.update(metrics=metrics,processing=processing,elapsed_s=time.monotonic()-before,
            qpos_csv=str((folder/"qpos.csv").relative_to(ROOT)),bundle=str((folder/"motion_bundle.npz").relative_to(ROOT)),bundle_sha256=sha(folder/"motion_bundle.npz"))
        write(folder/"manifest.json",result); clips.append(result)
        write(run/"progress.json",{"completed":len(clips),"clips":clips})
        logging.info("%s pass=%s failed=%s IK=%.5f time=%.2fs",item["clip_id"],metrics["passed"],[k for k,v in metrics["checks"].items() if not v],processing["ik"]["max_foot_target_error_m"],result["elapsed_s"])
    counts={motion:{split:sum(c["motion_id"]==motion and c["split"]==split and c["metrics"]["passed"] for c in clips) for split in ["train","validation"]} for motion in ["stand","wave","walk_slow"]}
    summary={"run_id":run_id,"clips":clips,"accepted_counts":counts,"passed":all(v['train']>=2 and v['validation']>=1 for v in counts.values()),"elapsed_s":time.monotonic()-started,"visual_review":"pending","thresholds_relaxed":False}
    write(run/"summary.json",summary);write(ROOT/"reports/stage_03_processing.json",summary)
    logging.info("Finished counts=%s gate=%s",counts,summary["passed"])
    return 0 if summary["passed"] else 2


if __name__=="__main__":
    raise SystemExit(main())
