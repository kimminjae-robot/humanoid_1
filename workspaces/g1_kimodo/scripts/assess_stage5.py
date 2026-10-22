"""Audit saved traces independently and compare paired simulator outcomes."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def audit(run,dataset,gate):
    result=load(run/'result.json');model=mujoco.MjModel.from_xml_path(str(ROOT/'assets/g1/g1_scene.xml'));data=mujoco.MjData(model)
    bundles={c['clip_id']:dict(np.load(ROOT/c['bundle'],allow_pickle=False)) for c in dataset['clips'] if c['accepted'] and c['split']=='validation'}
    names=load(ROOT/'configs/motion_contract.json')['csv']['joint_names'];qadr=[model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in names]
    bids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n) for n in next(iter(bundles.values()))['body_names'].tolist()]
    hands=[i for i,n in enumerate(next(iter(bundles.values()))['body_names'].tolist()) if n in ['left_wrist_yaw_link','right_wrist_yaw_link']]
    maxerr=0;maxsaterr=0;metric_checks=True;traces_finite=True
    with np.load(run/'rollout.npz',allow_pickle=False) as trace:
        for e in result['episodes']:
            q=trace['qpos'][:e['steps'],e['env_id']];f=trace['frame_indices'][:e['steps'],e['env_id']];b=bundles[e['clip_id']];idx=np.minimum(f,len(b['joint_pos_rad'])-1)
            assert len(q)==e['steps'] and np.array_equal(f,np.arange(1,e['steps']+1))
            assert trace['env_clip_ids'][e['env_id']]==e['clip_id']
            traces_finite &= bool(np.isfinite(q).all())
            bodies=[]
            for frame in q:
                data.qpos[:7]=frame[:7];data.qpos[qadr]=frame[7:];mujoco.mj_kinematics(model,data);bodies.append(data.xpos[bids].copy())
            bodies=np.stack(bodies);rootref=b['root_pos_world_m'][idx];br=b['body_pos_world_m'][idx]
            errors=[np.mean((q[:,7:]-b['joint_pos_rad'][idx])**2),np.mean(np.sum(((bodies-q[:,:3,None].transpose(0,2,1))-(br-rootref[:,None]))**2,axis=-1)),np.mean(np.sum((q[:,:3]-rootref)**2,axis=-1)),np.mean(np.sum((bodies[:,hands]-br[:,hands])**2,axis=-1))]
            rmse=np.sqrt(errors);stored=np.array([e[k] for k in ['rmse_joint_rad','rmse_body_m','rmse_root_m','rmse_hand_m']]);maxerr=max(maxerr,float(np.max(abs(rmse-stored))))
            sat=float(trace['saturated_joint_physics_steps'][:e['steps'],e['env_id']].sum()/(e['steps']*290));maxsaterr=max(maxsaterr,abs(sat-e['torque_saturation_fraction']))
            threshold=[gate[k] for k in ['max_joint_position_rmse_rad','max_root_relative_body_position_rmse_m','max_world_root_position_rmse_m','max_hand_position_rmse_m']]
            metric_checks &= all(e['checks'][k]==bool(v<=limit) for k,v,limit in zip(['joint','body','root','hand'],rmse,threshold))
            metric_checks &= e['checks']['saturation']==(sat<=gate['max_torque_saturation_fraction']) and e['success']==all(e['checks'].values())
    return {'episode_count':len(result['episodes']),'max_rmse_recompute_absolute_difference':maxerr,'max_saturation_recompute_difference':maxsaterr,'metric_checks_match':bool(metric_checks),'finite_saved_states':traces_finite,'passed':bool(maxerr<1e-4 and maxsaterr<1e-6 and metric_checks and traces_finite)}

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--refined-run',type=Path);a=p.parse_args();run=a.run.resolve();cfg=load(ROOT/'configs/evaluation.json');r=load(run/'result.json');resolved=load(run/'resolved.json');irun=ROOT/resolved['source_Isaac_run'];i=load(irun/'result.json');dataset=load(ROOT/resolved['dataset_index']);fixtures=load(run/'contract_validation.json')
    key=lambda e:(e['clip_id'],e['evaluation_seed'])
    ie={key(e):e for e in i['episodes']};me={key(e):e for e in r['episodes']}
    checks={'contract_fixtures':fixtures['passed'],'60_cases_no_missing_or_duplicate':len(ie)==len(me)==len(i['episodes'])==len(r['episodes'])==60,'same_case_keys':set(ie)==set(me),'explicit_initial_states_identical':set(ie)==set(me) and all(ie[k]['initial_state']==me[k]['initial_state'] for k in ie),'same_export_policy':resolved['policy_sha256']==i['export']['sha256'],'same_evaluation_contract':resolved['evaluation_sha256']==sha(ROOT/'configs/evaluation.json'),'source_Isaac_result_unchanged':resolved['initial_states_source_sha256']==sha(irun/'result.json'),'same_dataset':resolved['dataset_index']==load(irun/'export/contract.json')['dataset_index']}
    audits={'Isaac':audit(irun,dataset,cfg['dynamic_episode_gate']),'MuJoCo':audit(run,dataset,cfg['dynamic_episode_gate'])}
    checks['Isaac_metrics_independently_recomputed']=audits['Isaac']['passed']
    checks['MuJoCo_metrics_independently_recomputed']=audits['MuJoCo']['passed']
    groups={}
    for motion in r['motion_ids']:
        left=[e for e in i['episodes'] if e['motion_id']==motion];right=[e for e in r['episodes'] if e['motion_id']==motion]
        groups[motion]={'episodes':len(right),'Isaac_successes':sum(e['success'] for e in left),'MuJoCo_successes':sum(e['success'] for e in right),'Isaac_rate':i['success_rates'][motion],'MuJoCo_rate':r['success_rates'][motion],'success_drop_percentage_points':100*(i['success_rates'][motion]-r['success_rates'][motion]),'MuJoCo_failure_check_counts':dict(Counter(k for e in right for k,v in e['checks'].items() if not v)),'MuJoCo_steps_range':[min(e['steps'] for e in right),max(e['steps'] for e in right)],'max_episode_rmse':{k:max(e[k] for e in right) for k in ['rmse_joint_rad','rmse_body_m','rmse_root_m','rmse_hand_m']},'max_torque_saturation_fraction':max(e['torque_saturation_fraction'] for e in right)}
    checks['reported_success_rates_match_episode_flags']=all(g['Isaac_rate']==g['Isaac_successes']/g['episodes'] and g['MuJoCo_rate']==g['MuJoCo_successes']/g['episodes'] for g in groups.values())
    performance={'MuJoCo_nominal_90pct_all_motions':all(v['MuJoCo_rate']>=cfg['aggregate_gate']['min_nominal_success_rate_per_motion'] for v in groups.values()),'Isaac_to_MuJoCo_drop_within_10pp':all(v['success_drop_percentage_points']<=cfg['aggregate_gate']['max_isaac_to_mujoco_success_drop_percentage_points'] for v in groups.values())}
    refinement=None
    if a.refined_run:
        rr=a.refined_run.resolve();ref=load(rr/'result.json');rc=load(rr/'resolved.json');re={key(e):e for e in ref['episodes']}
        checks['refinement_same_policy_states_and_rates']=rc['physics_dt']==resolved['physics_dt']/2 and rc['control_dt']==resolved['control_dt'] and rc['torque_dt']==resolved['torque_dt'] and rc['policy_sha256']==resolved['policy_sha256'] and set(re)==set(me) and all(re[k]['initial_state']==me[k]['initial_state'] for k in me)
        audits['MuJoCo_refined']=audit(rr,dataset,cfg['dynamic_episode_gate']);checks['refinement_metrics_recomputed']=audits['MuJoCo_refined']['passed']
        refinement={'run':str(rr.relative_to(ROOT)),'success_rates':ref['success_rates'],'changed_episode_successes':sum(me[k]['success']!=re[k]['success'] for k in me),'max_paired_termination_time_delta_s':max(abs(me[k]['steps']-re[k]['steps'])*.02 for k in me),'max_paired_episode_rmse_delta':{metric:max(abs(me[k][metric]-re[k][metric]) for k in me) for metric in ['rmse_joint_rad','rmse_body_m','rmse_root_m','rmse_hand_m']},'note':'MuJoCo integration dt halved; target 50Hz and explicit PD 500Hz unchanged. Isaac refinement not yet measured.'}
    report={'scope':'framework_integration_separate_from_dynamic_performance','run':str(run.relative_to(ROOT)),'Isaac_run':str(irun.relative_to(ROOT)),'policy_sha256':resolved['policy_sha256'],'integration_checks':checks,'integration_passed':all(checks.values()),'performance_checks':performance,'performance_passed':all(performance.values()),'per_motion':groups,'independent_audits':audits,'MuJoCo_timestep_refinement':refinement,'unmeasured':['PhysX-applied torque fixture (shared Torch PD checked, actual actuator output not captured)','Isaac timestep refinement','robustness sweeps','untouched final generation seeds']}
    write(run/'comparison.json',report);write(ROOT/'reports/stage_05_summary.json',report)
    print(json.dumps(report,indent=2));return 0 if report['integration_passed'] else 1
if __name__=='__main__':raise SystemExit(main())
