"""Independent saved-trace metric audit, separate from rollout implementation."""
import json,hashlib,argparse
from pathlib import Path
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default=json.loads((ROOT/'target.json').read_text()).get('current_evaluation_run','baseline'));args=ap.parse_args();run=ROOT/'runs'/args.run
    robot=json.loads((ROOT/'reports/robot_validation.json').read_text());result=json.loads((run/'result.json').read_text());qa=json.loads((run/'kinematic_qa.json').read_text());manifest=json.loads((ROOT/'reports/robot_manifest.json').read_text());m=mujoco.MjModel.from_xml_path(str(ROOT/'assets/k1_hx5.xml'));d=mujoco.MjData(m);ads=[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in manifest['body_joint_names']];wrists=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,s+'_wrist_roll_rubber_hand') for s in ['left','right']];maximum=0;checks={'robot_fixtures':robot['passed'],'complete_unique_episodes':len(result['episodes'])==len({(e['clip_id'],e['seed']) for e in result['episodes']})==3*len({e['clip_id'] for e in result['episodes']}),'source_robot_hash':sha(ROOT/'assets/k1_hx5.xml')==manifest['model_sha256']};notes=[];walking=[]
    for e in result['episodes']:
        z=np.load(ROOT/e['rollout_npz'],allow_pickle=False);q=z['qpos'];idx=z['frame_index'];ref=z['reference'][idx];expected=np.r_[z['initial_qpos']];assert expected.tolist()==e['initial_state']['qpos'];assert np.isfinite(q).all() and len(q)==e['steps']
        joints=float(np.sqrt(np.mean((q[:,ads]-ref[:,ads])**2)));root=float(np.sqrt(np.mean(np.sum((q[:,:3]-ref[:,:3])**2,axis=-1))));squared=[]
        for a,b in zip(q,ref):
            d.qpos[:]=a;mujoco.mj_kinematics(m,d);actual=d.xpos[wrists].copy();d.qpos[:]=b;mujoco.mj_kinematics(m,d);squared.append(np.sum((actual-d.xpos[wrists])**2,axis=-1))
        wrist=float(np.sqrt(np.mean(squared)));maximum=max(maximum,abs(joints-e['joint_rmse_rad']),abs(root-e['root_rmse_m']),abs(wrist-e['wrist_rmse_m']));assert e['success']==all(e['checks'].values())
        if e['motion_id']=='walk_slow':
            footids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,side+'_ankle_roll_link') for side in ['left','right']];positions=[]
            for row in q:
                d.qpos[:]=row;mujoco.mj_kinematics(m,d);positions.append(d.xpos[footids].copy())
            positions=np.stack(positions);counts=[]
            for side in range(2):
                raised=positions[:,side,2]>np.min(positions[:,side,2])+.025;trans=np.diff(np.r_[False,raised,False].astype(int));starts=np.flatnonzero(trans==1);ends=np.flatnonzero(trans==-1);counts.append(int(sum(end-begin>=5 for begin,end in zip(starts,ends))))
            walking.append({'clip_id':e['clip_id'],'seed':e['seed'],'actual_root_displacement_m':(q[-1,:3]-expected[:3]).tolist(),'reference_root_displacement_m':(z['reference'][-1,:3]-z['reference'][0,:3]).tolist(),'actual_left_right_swing_counts':counts,'duration_s':e['duration_s']})
    checks['independent_RMSE_under_1e_minus_6']=maximum<1e-6;checks['reported_rates_match_episodes']=all(v==sum(e['success'] for e in result['episodes'] if e['motion_id']==motion)/sum(e['motion_id']==motion for e in result['episodes']) for motion,v in result['success_rates'].items())
    rates=result['success_rates'];report={'integration_checks':checks,'integration_passed':all(checks.values()),'full_performance_passed':result['all_motions_passed'] and qa['passed'],'dynamic_success_rates':rates,'episodes_per_motion':3,'independent_max_RMSE_difference':maximum,'QA':{c['clip_id']:c['passed'] for c in qa['clips']},'model_mass_kg':manifest['total_mass_kg'],'run':args.run,'walking_execution_evidence':walking,'scope':'Kimodo_G1_to_K1_HX5_MuJoCo_controller_not_Isaac_sim2sim','limitations':['one generated validation clip per motion; three reset perturbations, not generalization','simulated hand mount is not hardware validated','new arbitrary prompts are not guaranteed trackable','torque saturation is recorded by controller, not independently reconstructed from 500Hz torque traces']}
    (run/'readiness.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 0 if report['integration_passed'] else 1
if __name__=='__main__':raise SystemExit(main())
