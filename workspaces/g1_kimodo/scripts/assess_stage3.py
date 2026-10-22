"""Re-evaluate packaged references and prove schema, provenance and split gates."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from reference_core import analyze,model_and_config,rotations


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    dataset=json.loads((ROOT/"reports/stage_03_dataset.json").read_text());run=ROOT/"runs"/dataset["run_id"]
    visual=json.loads((run/"visual_review.json").read_text());model,cfg,contract=model_and_config()
    baseline=json.loads((ROOT/"reports/stage_02_generation.json").read_text())
    eval_current=json.loads((ROOT/"configs/evaluation.json").read_text())
    eval_before=json.loads((run/"provenance/configs__evaluation.json").read_text())
    maps=json.loads((ROOT/dataset["backend_maps"]).read_text())
    checks={"math_and_geometry_validation":json.loads((ROOT/"reports/reference_math_validation.json").read_text())["passed"],
        "all_15_sources_accounted_for":len(dataset["clips"])==15 and {c['clip_id'] for c in dataset['clips']}=={c['clip_id'] for c in baseline['clips']},
        "raw_source_checksums_unchanged":True,"bundle_schema_finite_and_50hz":True,"bundle_and_csv_checksums":True,
        "quaternion_and_velocity_frames":True,"recomputed_kinematic_gates":True,"stored_fk_matches_recomputed_fk":True,
        "backend_name_maps_are_bijections":all(sorted(maps[key])==list(range(size)) for key,size in [('source_to_isaac_joint_indices',29),('source_to_isaac_body_indices',30)]),
        "visual_review_all_accepted":{c['clip_id'] for c in visual['clips'] if c['semantic_preserved']}=={c['clip_id'] for c in dataset['clips'] if c['accepted']},
        "thresholds_unchanged":eval_current['kinematic_gate']==eval_before['kinematic_gate'],
        "thresholds_frozen_before_training":eval_current['status']=='frozen_before_stage_4_engineering_targets',
        "final_seeds_unused":all(c['seed'] not in [303,404] for c in dataset['clips']),"minimum_per_motion_split_counts":True}
    reviews=[];counts={motion:{'train':0,'validation':0} for motion in ['stand','wave','walk_slow']}
    for clip in dataset['clips']:
        source=json.loads((ROOT/clip['source_manifest']).read_text())
        for item in source['artifacts'].values():checks['raw_source_checksums_unchanged'] &= sha(ROOT/item['path'])==item['sha256']
        if not clip['accepted']:continue
        checks['bundle_and_csv_checksums'] &= sha(ROOT/clip['bundle'])==clip['bundle_sha256'] and sha(ROOT/clip['qpos_csv'])==clip['qpos_sha256']
        with np.load(ROOT/clip['bundle'],allow_pickle=False) as data:
            missing=set(contract['motion_bundle_required_fields'])-set(data.files)
            q=np.c_[data['root_pos_world_m'],data['root_quat_world_wxyz'],data['joint_pos_rad']]
            checks['bundle_schema_finite_and_50hz'] &= not missing and int(data['fps'])==50 and bool(np.allclose(np.diff(data['timestamps_s']),.02,atol=1e-12))
            checks['bundle_schema_finite_and_50hz'] &= all(bool(np.isfinite(data[k]).all()) for k in data.files if data[k].dtype.kind in 'fiub')
            checks['bundle_schema_finite_and_50hz'] &= data['joint_names'].tolist()==maps['joint_names_source'] and data['body_names'].tolist()==maps['body_names_source']
            checks['quaternion_and_velocity_frames'] &= bool((np.sum(q[1:,3:7]*q[:-1,3:7],axis=1)>=0).all())
            expected=np.c_[data['root_lin_vel_world_m_s'],rotations(q[:,3:7]).inv().apply(data['root_ang_vel_world_rad_s']),data['joint_vel_rad_s']]
            checks['quaternion_and_velocity_frames'] &= bool(np.allclose(data['qvel_mujoco'],expected,atol=1e-12))
            qa,arrays=analyze(q,50,model)
            qa['checks']['foot_ik_tracking']=clip['processing']['ik']['max_foot_target_error_m']<=cfg['ik']['max_foot_target_error_m']
            qa['passed']=all(qa['checks'].values());checks['recomputed_kinematic_gates'] &= qa['passed']
            checks['stored_fk_matches_recomputed_fk'] &= bool(np.allclose(data['body_pos_world_m'],arrays['body_pos_world_m'],atol=1e-10)) and bool(np.allclose(data['foot_slip_speed_m_s'],arrays['foot_slip_speed_m_s'],atol=1e-10))
            reviews.append({'clip_id':clip['clip_id'],**qa});counts[clip['motion_id']][clip['split']]+=int(qa['passed'])
    checks['minimum_per_motion_split_counts']=all(v['train']>=2 and v['validation']>=1 for v in counts.values())
    for c in visual['clips']:
        checks['visual_review_all_accepted'] &= sha(ROOT/c['contact_sheet'])==c['contact_sheet_sha256']
    result={'scope':'stage_3_kinematic_reference_readiness_not_policy_or_dynamic_sim2sim','run_id':dataset['run_id'],'checks':checks,'passed':all(checks.values()),
        'source_candidates':15,'accepted':dataset['accepted'],'rejected':dataset['rejected'],'accepted_counts':counts,'clips':reviews,'dynamic_tracking_validated':False,'sim2sim_validated':False}
    (run/'stage_assessment.json').write_text(json.dumps(result,indent=2)+'\n');(ROOT/'reports/stage_03_readiness.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='clips'},indent=2))
    return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
