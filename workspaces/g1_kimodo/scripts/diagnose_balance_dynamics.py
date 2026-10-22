import argparse,json,numpy as np,mujoco
from scipy.spatial.transform import Rotation
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--clip-id',default='stand_train_22');parser.add_argument('--output',type=Path);args=parser.parse_args()
w=Path(__file__).resolve().parents[1];ds=json.loads((w/json.loads((w/'state/progress.json').read_text())['artifacts_stage_3']['dataset_index']).read_text());clip=next(c for c in ds['clips'] if c['clip_id']==args.clip_id);q0=np.loadtxt(w/clip['qpos_csv'],delimiter=',')[0]
results=[]
for profile,diag in [('tracking_contract.json','stage_04_static_support_diagnostic.json'),('tracking_contract_stiff.json','stage_04_static_support_stiff_diagnostic.json')]:
 c=json.loads((w/'configs'/profile).read_text())['controller'];names=list(c['kp']);kp=np.array([c['kp'][n] for n in names]);kd=np.array([c['kd'][n] for n in names]);lim=np.array([c['joint_position_limits'][n] for n in names]);eff=np.array([c['effort_limits'][n] for n in names]);vel=np.array([c['velocity_limits'][n] for n in names]);scale=np.array([c['action_scale'][n] for n in names]);di=next(c for c in json.loads((w/'reports'/diag).read_text())['clips'] if c['clip_id']==clip['clip_id']);bias=np.array(di['required_target_offset_rad'])/scale
 for feedback in [False,True]:
  m=mujoco.MjModel.from_xml_path(str(w/'assets/g1/g1_scene.xml'));d=mujoco.MjData(m);d.qpos[:]=q0;mujoco.mj_forward(m,d);qs=[];actions=[]
  for frame in range(362):
   rot=Rotation.from_quat(d.qpos[[4,5,6,3]]);rref=Rotation.from_quat(q0[[4,5,6,3]]);delta=(rot.inv()*rref).as_quat();delta*=np.sign(delta[3]);err=2*delta[:3];poserr=rot.inv().apply(q0[:3]-d.qpos[:3]);lin=rot.inv().apply(d.qvel[:3]);ang=d.qvel[3:6];a=bias.copy()
   if feedback:
    pitch=-8*err[1]+2*ang[1]+lin[0]-2*poserr[0];roll=-8*err[0]+2*ang[0]-lin[1]+2*poserr[1]
    a[[4,10]]+=pitch;a[[5,11]]+=roll
   a=a.clip(-1,1);target=np.clip(q0[7:]+scale*a,lim[:,0],lim[:,1])
   for _ in range(10):
    raw=kp*(target-d.qpos[7:])-kd*d.qvel[6:];tau=raw.clip(-eff,eff);tau[(abs(d.qvel[6:])>vel)&(tau*d.qvel[6:]>0)]=0;d.ctrl[:]=tau;mujoco.mj_step(m,d)
   qs.append(d.qpos.copy());actions.append(a.copy())
   if d.qpos[2]<.35:break
  results.append({'profile':profile,'feedback':feedback,'steps':len(qs),'final_root':d.qpos[:3].tolist(),'final_rpy':Rotation.from_quat(d.qpos[[4,5,6,3]]).as_euler('xyz').tolist(),'max_root_error':float(np.linalg.norm(np.array(qs)[:,:3]-q0[:3],axis=-1).max())})
print(json.dumps(results,indent=2));(w/'reports/stage_04_cpu_balance_diagnostic.json').write_text(json.dumps({'scope':'MuJoCo analytical-controller diagnosis on named clip, no PPO policy transfer result','clips':[clip['clip_id']],'results':results},indent=2))
