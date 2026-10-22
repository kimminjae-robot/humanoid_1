"""Contact-aware conservative walking retarget from the preserved Kimodo footsteps.
Spatially compress source steps, flatten K1 soles, transfer COM before liftoff,
solve feet and COM simultaneously. No changes to physics or evaluation gates.
"""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];G1=ROOT.parent/'g1_kimodo'
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def smooth(t):return t*t*t*(10+t*(-15+6*t))
def main():
    p=argparse.ArgumentParser();p.add_argument('--label',default='walk_v3');p.add_argument('--scale',type=float,default=.30);p.add_argument('--transfer-s',type=float,default=1.2);p.add_argument('--swing-s',type=float,default=2.);p.add_argument('--height',type=float,default=.74);p.add_argument('--source-clip',default='walk_slow_validation_101');a=p.parse_args();folder=ROOT/'data'/a.label;folder.mkdir(parents=True,exist_ok=False)
    m=mujoco.MjModel.from_xml_path(str(ROOT/'assets/k1_hx5.xml'));d=mujoco.MjData(m);manifest=json.loads((ROOT/'reports/robot_manifest.json').read_text());names=manifest['body_joint_names'];jids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]);qa=m.jnt_qposadr[jids];q0=np.load(ROOT/'assets/neutral_qpos.npy');d.qpos[:]=q0;mujoco.mj_forward(m,d);feet=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,s+'_ankle_roll_link') for s in ['left','right']];neutral_feet=d.xpos[feet].copy();footrot=d.xmat[feet].reshape(-1,3,3).copy()
    for k,f in enumerate(feet):
        gs=[g for g in range(m.ngeom) if m.geom_bodyid[g]==f and m.geom_contype[g] and m.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE];neutral_feet[k,2]=.001-min((footrot[k]@m.geom_pos[g])[2]-m.geom_size[g,0] for g in gs)
    c=next(c for c in json.loads((G1/'data/processed/20261021_115621_761537_stage03_curated/final/dataset.json').read_text())['clips'] if c['clip_id']==a.source_clip);b=np.load(G1/c['bundle']);bn=b['body_names'].tolist();sf=[bn.index(s+'_ankle_roll_link') for s in ['left','right']];srcfeet=b['body_pos_world_m'][:,sf];contact=b['foot_contacts'].astype(bool)
    # Merge short false contact flicker inside one swing. Keep the five advancing steps.
    events=[]
    for side in range(2):
        off=np.flatnonzero(np.diff(np.r_[True,contact[:,side],True].astype(int))==-1);on=np.flatnonzero(np.diff(np.r_[True,contact[:,side],True].astype(int))==1)
        ranges=[]
        for start,end in zip(off,on):
            if ranges and start-ranges[-1][1]<20:ranges[-1][1]=int(end)
            else:ranges.append([int(start),int(end)])
        for start,end in ranges:
            if srcfeet[min(end,len(srcfeet)-1),side,0]-srcfeet[max(start-1,0),side,0]>.05:events.append((start,end,side))
    events.sort();poses=[];contacts=[];coms=[];sourceframes=[];current=neutral_feet.copy();com=current[:,:2].mean(0)
    def append(feet,co,xy,src):poses.append(feet.copy());contacts.append(co.copy());coms.append(xy.copy());sourceframes.append(src)
    for _ in range(40):append(current,np.ones(2,bool),com,0)
    plan=[]
    for start,end,side in events:
        stance=1-side;startcom=com.copy();endcom=current[stance,:2].copy()
        for u in np.linspace(0,1,round(a.transfer_s*50)+1)[1:]:append(current,np.ones(2,bool),startcom+(endcom-startcom)*smooth(u),max(0,start-1))
        com=endcom;origin=current[side].copy();dest=neutral_feet[side].copy();dest[0]+=a.scale*(srcfeet[min(end,len(srcfeet)-1),side,0]-srcfeet[0,side,0])
        # Keep a fixed stance width; source lateral drift is not carried into K1.
        for u in np.linspace(0,1,round(a.swing_s*50)+1)[1:]:
            feetpos=current.copy();feetpos[side]=origin+(dest-origin)*smooth(u);feetpos[side,2]+=.055*16*u*u*(1-u)*(1-u);co=np.ones(2,bool);co[side]=u>=1;append(feetpos,co,com,round(start+(end-start)*u))
        current[side]=dest;plan.append({'source_start':start,'source_end':end,'swing_side':['left','right'][side],'destination_ankle_m':dest.tolist()})
    final=current[:,:2].mean(0);old=com.copy()
    for u in np.linspace(0,1,round(a.transfer_s*50)+1)[1:]:append(current,np.ones(2,bool),old+(final-old)*smooth(u),len(srcfeet)-1)
    for _ in range(50):append(current,np.ones(2,bool),final,len(srcfeet)-1)
    poses=np.stack(poses);contacts=np.stack(contacts);coms=np.stack(coms);out=[];diags=[];prev=np.r_[q0[:3],q0[qa]];limits=m.jnt_range[jids];legix=np.array([i for i,n in enumerate(names) if any(k in n for k in ['hip','knee','ankle'])]);variables=np.r_[np.arange(3),3+legix];base=np.r_[q0[:3],q0[qa]];base[2]=a.height
    def setq(x):
        xx=base.copy();xx[variables]=x;d.qpos[:]=q0;d.qpos[:3]=xx[:3];d.qpos[3:7]=[1,0,0,0];d.qpos[qa]=xx[3:];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
    starttime=time.monotonic()
    for frame in range(len(poses)):
        targets=poses[frame];desiredcom=coms[frame]
        def fun(x):
            setq(x);err=[(d.xpos[feet]-targets).ravel()*100]
            err.extend([Rotation.from_matrix(footrot[k].T@d.xmat[f].reshape(3,3)).as_rotvec()*20 for k,f in enumerate(feet)])
            err.append((d.subtree_com[1,:2]-desiredcom)*100);err.append(np.array([(d.qpos[2]-a.height)*10]));err.append((d.qpos[qa[legix]]-q0[qa[legix]])*.04);return np.concatenate(err)
        lo=np.r_[targets[:,:2].mean(0)-[.3,.3],a.height-.1,limits[:,0]];hi=np.r_[targets[:,:2].mean(0)+[.3,.3],a.height+.1,limits[:,1]]
        sol=least_squares(fun,np.clip(prev[variables],lo[variables]+1e-7,hi[variables]-1e-7),bounds=(lo[variables],hi[variables]),max_nfev=35,ftol=1e-7,gtol=1e-7,xtol=1e-7)
        setq(sol.x);q=d.qpos.copy();out.append(q);prev=np.r_[q[:3],q[qa]];diags.append(float(np.max(np.linalg.norm(d.xpos[feet]-targets,axis=-1))))
    out=np.stack(out);np.savez_compressed(folder/'reference.npz',qpos=out,fps=np.array(50),joint_names=np.array(names),source_foot_contacts=contacts,planned_foot_positions=poses,planned_COM_xy=coms,source_frames=np.array(sourceframes));np.savetxt(folder/'qpos.csv',out,delimiter=',')
    record={**{k:c[k] for k in ['clip_id','motion_id','split','seed','source_sha256','source_manifest']},'accepted_for_tracking':False,'clip_id':c['clip_id'],'frames':len(out),'fps':50,'reference_npz':str((folder/'reference.npz').relative_to(ROOT)),'qpos_csv':str((folder/'qpos.csv').relative_to(ROOT)),'method':'contact_aware_conservative_Kimodo_footsteps_K1_IK','variant':a.label,'max_sampled_foot_position_error_m':max(diags),'source_G1_csv':str((G1/c['qpos_csv']).relative_to(ROOT.parent)),'source_bundle_sha256':c['bundle_sha256'],'model_sha256':manifest['model_sha256'],'source_contact_events':plan,'settings':vars(a),'upper_body':'neutralized_to_isolate_gait; fingers_hold_original_targets','root_orientation':'upright_heading_zero','duration_s':(len(out)-1)/50,'forward_foot_displacement_m':(current-neutral_feet).tolist(),'elapsed_s':time.monotonic()-starttime}
    write(folder/'manifest.json',record);ds=json.loads((ROOT/'data/dataset.json').read_text());ds['clips']=[record if r['motion_id']=='walk_slow' else r for r in ds['clips']];write(ROOT/'reports'/('dataset_'+a.label+'.json'),ds);print(json.dumps({k:v for k,v in record.items() if k not in ['metrics','processing']},indent=2))
if __name__=='__main__':main()
