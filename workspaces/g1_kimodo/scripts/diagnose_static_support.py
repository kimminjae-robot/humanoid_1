import argparse,json,numpy as np,mujoco
from scipy.optimize import minimize
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--contract',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
w=Path(__file__).resolve().parents[1];c=json.loads((args.contract or w/'configs/tracking_contract.json').read_text())['controller'];ds=json.loads((w/json.loads((w/'state/progress.json').read_text())['artifacts_stage_3']['dataset_index']).read_text());m=mujoco.MjModel.from_xml_path(str(w/'assets/g1/g1_scene.xml'));v=mujoco.MjData(m);names=list(c['kp']);kp=np.array([c['kp'][n] for n in names]);scale=np.array([c['action_scale'][n] for n in names]);result=[]
for clip in ds['clips']:
 if not clip['accepted'] or clip['motion_id']=='walk_slow':continue
 q=np.loadtxt(w/clip['qpos_csv'],delimiter=',')[0];v.qpos[:]=q;v.qvel[:]=0;mujoco.mj_forward(m,v)
 forces=[];points=[]
 for gi in range(m.ngeom):
  if 'foot' not in (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,gi) or ''):continue
  axis=v.geom_xmat[gi].reshape(3,3)[:,2];half=m.geom_size[gi,1];r=m.geom_size[gi,0]
  endpoints=[v.geom_xpos[gi]+sign*half*axis for sign in [-1,1]]
  for point in endpoints:
   point=point-np.array([0,0,r])
   if point[2]>.005:continue
   jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jac(m,v,jp,jr,point,int(m.geom_bodyid[gi]));forces.append(jp[2]);points.append(point.tolist())
 J=np.array(forces).T;A=J[:6];b=v.qfrc_bias[:6];U,S,Vh=np.linalg.svd(A,full_matrices=False);rank=(S>1e-8).sum();B=U[:,:rank].T@A;rhs=U[:,:rank].T@b
 def cost(f):return np.sum(((v.qfrc_bias[6:]-J[6:]@f)/kp)**2)+1e-8*np.sum(f*f)
 sol=minimize(cost,np.ones(J.shape[1])*b[2]/J.shape[1],bounds=[(0.,500.)]*J.shape[1],constraints={'type':'eq','fun':lambda f:B@f-rhs},method='SLSQP',options={'ftol':1e-12,'maxiter':300})
 tau=v.qfrc_bias[6:]-J[6:]@sol.x;offset=tau/kp;order=np.argsort(abs(offset))[::-1][:5]
 result.append({'clip_id':clip['clip_id'],'converged':bool(sol.success),'root_equilibrium_residual':float(np.linalg.norm(A@sol.x-b)),'max_required_residual_action':float(max(abs(offset/scale))),'required_target_offset_rad':offset.tolist(),'top':[(names[i],float(offset[i]),float(tau[i])) for i in order]})
print(json.dumps(result[:3],indent=2));(args.output or w/'reports/stage_04_static_support_diagnostic.json').write_text(json.dumps({'kp_used':c['kp'],'scope':'quasistatic vertical ground wrench optimization at reference frame0, no dynamic feasibility claim','clips':result},indent=2))
