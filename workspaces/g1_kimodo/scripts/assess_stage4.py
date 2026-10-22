"""Stage gate requires full nominal coverage, compatible exports and exact provenance."""
import argparse,hashlib,json
import numpy as np
import mujoco
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('evaluation_run',type=Path);a=p.parse_args();run=a.evaluation_run.resolve()
    d=json.loads((run/'result.json').read_text());cfg=json.loads((run/'resolved.json').read_text());e=json.loads((ROOT/'configs/evaluation.json').read_text());contract=json.loads((run/'export/contract.json').read_text())
    eps=d['episodes'];dataset=json.loads((ROOT/contract['dataset_index']).read_text());expected={c['clip_id'] for c in dataset['clips'] if c['accepted'] and c['split']=='validation'}
    pairs=[(x['clip_id'],x['evaluation_seed']) for x in eps]
    train=Path(cfg['args']['checkpoint']).resolve().parent;training_path=train/'result.json' if (train/'result.json').exists() else train/'checkpoint_assessment.json'
    training=json.loads(training_path.read_text()) if training_path.exists() else {}
    checks={'contact_and_hold_fixture':(ROOT/'reports/tracking_physics_fixture.json').exists() and json.loads((ROOT/'reports/tracking_physics_fixture.json').read_text())['passed'],
      'nominal_physics_logs_clean':not any('[Error]' in line and ('physx' in line.lower() or 'cuda' in line.lower()) or 'GPU contact filter for collider' in line or 'please increase' in line.lower() for line in (run/'isaac.log').read_text(errors='replace').splitlines()),
      'learned_policy_evaluation_not_teacher':d.get('controller_mode')=='evaluate',
      'stage3_ready':json.loads((ROOT/'reports/stage_03_readiness.json').read_text())['passed'],
      'all_three_motion_ids':set(d['motion_ids'])=={'stand','wave','walk_slow'},
      'all_validation_clips':{x['clip_id'] for x in eps}==expected,
      'unique_explicit_reset_cases':len(pairs)==len(set(pairs))==len(expected)*e['episodes_per_clip'],
      'all_requested_seeds':all({x['evaluation_seed'] for x in eps if x['clip_id']==clip}==set(e['evaluation_seeds']) for clip in expected),
      'nominal_90_percent_each_motion':all(d['success_rates'].get(m,0)>=e['aggregate_gate']['min_nominal_success_rate_per_motion'] for m in ['stand','wave','walk_slow']),
      'same_policy_checkpoint_export':contract['checkpoint_sha256']==sha(Path(cfg['args']['checkpoint'])),
      'export_fixture_parity':d['export']['passed'] and d['export']['max_action_absolute_error']<=1e-5,
      'export_checksum':sha(ROOT/d['export']['path'])==d['export']['sha256'],
      'evaluation_thresholds_unchanged':sha(ROOT/'configs/evaluation.json')==contract['evaluation_sha256'],
      'policy_training_finite_and_updated':bool(training.get('parameters_finite')) and training.get('actor_parameter_max_change',0)>0,
      'state_only_written_in_reset':(ROOT/'tracking/env.py').read_text().count('write_root_pose_to_sim')==1 and (ROOT/'tracking/env.py').read_text().count('write_joint_state_to_sim')==1}
    trace=np.load(run/'rollout.npz',allow_pickle=False)
    independent=[]
    checks['all_episode_traces_saved']='saturated_joint_physics_steps' in trace and trace['qpos'].shape[1]==len(eps)
    checks['independent_episode_rmse_and_saturation']=checks['all_episode_traces_saved']
    model=mujoco.MjModel.from_xml_path(str(ROOT/'assets/g1/g1_scene.xml'));data=mujoco.MjData(model)
    maps=json.loads((ROOT/dataset['backend_maps']).read_text())
    body_ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n) for n in maps['body_names_source']]
    hands=[maps['body_names_source'].index(n) for n in ['left_wrist_yaw_link','right_wrist_yaw_link']]
    if checks['all_episode_traces_saved']:
        for ep in eps:
            clip=next(c for c in dataset['clips'] if c['clip_id']==ep['clip_id'])
            with np.load(ROOT/clip['bundle'],allow_pickle=False) as ref:
                q=trace['qpos'][:ep['steps'],ep['env_id']].astype(np.float64)
                frame=trace['frame_indices'][:ep['steps'],ep['env_id']].clip(0,len(ref['timestamps_s'])-1)
                joint=np.sqrt(np.mean((q[:,7:]-ref['joint_pos_rad'][frame])**2))
                root=np.sqrt(np.mean(np.sum((q[:,:3]-ref['root_pos_world_m'][frame])**2,axis=-1)))
                actual=[]
                for state in q:
                    data.qpos[:]=state;mujoco.mj_forward(model,data);actual.append(data.xpos[body_ids].copy())
                actual=np.stack(actual);body_ref=ref['body_pos_world_m'][frame]
                body=np.sqrt(np.mean(np.sum(((actual-q[:,None,:3])-(body_ref-ref['root_pos_world_m'][frame,None]))**2,axis=-1)))
                hand=np.sqrt(np.mean(np.sum((actual[:,hands]-body_ref[:,hands])**2,axis=-1)))
            saturation=float(trace['saturated_joint_physics_steps'][:ep['steps'],ep['env_id']].sum()/(ep['steps']*29*10))
            expected=np.array([ep['rmse_joint_rad'],ep['rmse_body_m'],ep['rmse_root_m'],ep['rmse_hand_m'],ep['torque_saturation_fraction']])
            measured=np.array([joint,body,root,hand,saturation]);tol=np.array([1e-5,1e-3,1e-5,1e-3,1e-7])
            passed=bool(np.isfinite(q).all() and (abs(measured-expected)<=tol).all())
            checks['independent_episode_rmse_and_saturation'] &= passed
            independent.append({'clip_id':ep['clip_id'],'evaluation_seed':ep['evaluation_seed'],'values_joint_body_root_hand_saturation':measured.tolist(),'reported_absolute_difference':abs(measured-expected).tolist(),'passed':passed})
    if cfg['agent']['policy']['class_name']=='TrackingActorCritic':
        verification=train/('base_balance_verification_'+Path(cfg['args']['checkpoint']).stem+'.json')
        frozen=json.loads(verification.read_text()) if verification.exists() else {}
        checks['base_balance_weights_preserved']=bool(frozen.get('passed')) and frozen.get('checkpoint_sha256')==contract['checkpoint_sha256'] and frozen.get('base_checkpoint_sha256')==sha(Path(cfg['agent']['policy']['base_checkpoint']))
    def nulls(x):
        if x is None:return True
        if isinstance(x,dict):return any(nulls(v) for v in x.values())
        if isinstance(x,list):return any(nulls(v) for v in x)
        return False
    checks['no_nulls_in_export_contract']=not nulls(contract)
    training_controller=json.loads((train/'provenance/configs__tracking_contract.json').read_text())['controller']
    checks['controller_matches_training_checkpoint']={k:v for k,v in training_controller.items() if k!='status'}=={k:v for k,v in contract['controller'].items() if k!='status'}
    checks['observed_joint_permutation_correct']=[cfg['joint_names_actual'][i] for i in cfg['source_to_isaac']]==contract['controller']['backend_joint_maps']['csv_source']
    result={'scope':'stage_4_gate_not_MuJoCo_sim2sim','evaluation_run':str(run.relative_to(ROOT)),'checks':checks,'passed':all(checks.values()),'success_rates':d['success_rates'],'independent_recomputation':independent,'rollout_sha256':sha(run/'rollout.npz')}
    (ROOT/'reports/stage_04_readiness.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='independent_recomputation'},indent=2));return 0 if result['passed'] else 2
if __name__=='__main__':raise SystemExit(main())
