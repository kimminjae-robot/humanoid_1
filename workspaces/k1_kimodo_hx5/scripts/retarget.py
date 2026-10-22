"""Task-space IK from provenance-preserved Kimodo G1 references to K1/HX5.
No finger motion synthesis. Kinematic acceptance is separate from dynamics.
"""
import json,hashlib,time,argparse,shutil,re
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation,Slerp
ROOT=Path(__file__).resolve().parents[1];G1=ROOT.parent/'g1_kimodo'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def bid(m,n):return mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n)
def rot(q):return Rotation.from_quat(np.r_[q[1:],q[0]]).as_matrix()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-csv',type=Path);parser.add_argument('--source-manifest',type=Path);parser.add_argument('--clip-id',default='custom_motion');parser.add_argument('--fps',type=float,default=30);parser.add_argument('--motion-id',default='custom');parser.add_argument('--profile',choices=['v1','v2']);args=parser.parse_args();args.profile=args.profile or ('v1' if args.source_csv else 'v2')
    if args.fps<=0 or not re.fullmatch(r'[A-Za-z0-9_-]+',args.clip_id):raise ValueError('Invalid fps or clip id')
    started=time.monotonic();manifest=json.loads((ROOT/'reports/robot_manifest.json').read_text());m=mujoco.MjModel.from_xml_path(str(ROOT/'assets/k1_hx5.xml'));d=mujoco.MjData(m);q0=np.load(ROOT/'assets/neutral_qpos.npy');d.qpos[:]=q0;mujoco.mj_forward(m,d)
    names=manifest['body_joint_names'];jids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]);qa=m.jnt_qposadr[jids];limits=m.jnt_range[jids];basepos=d.xpos.copy();baserot=d.xmat.reshape(-1,3,3).copy()
    sm=mujoco.MjModel.from_xml_path(str(G1/'assets/g1/g1_scene.xml'));sd=mujoco.MjData(sm)
    source_joint_names=json.loads((G1/'configs/motion_contract.json').read_text())['csv']['joint_names']
    dataset=json.loads((G1/'data/processed/20261021_115621_761537_stage03_curated/final/dataset.json').read_text());records=[]
    selected=[next(c for c in dataset['clips'] if c['clip_id']==motion+'_validation_101') for motion in ['stand','wave','walk_slow']]
    if args.source_csv:selected=[{'clip_id':args.clip_id,'motion_id':args.motion_id,'custom':True,'source_manifest':None,'source_sha256':sha(args.source_csv),'bundle_sha256':None}]
    for c in selected:
        motion=c['motion_id']
        if not c.get('custom'):
            source=G1/c['qpos_csv'];qsrc=np.loadtxt(source,delimiter=',');b=np.load(G1/c['bundle'],allow_pickle=False);source_names=b['body_names'].tolist();srcpos=b['body_pos_world_m'];srcquat=b['body_quat_world_wxyz']
        else:
            folder=ROOT/'data/user_input'/c['clip_id'];folder.mkdir(parents=True,exist_ok=False);source=folder/'source_G1.csv';shutil.copy2(args.source_csv,source)
            if args.source_manifest:
                provenance=json.loads(args.source_manifest.read_text())
                if provenance['artifacts']['qpos.csv']['sha256']!=sha(source):raise ValueError('Generation CSV hash mismatch')
                if provenance['fps']!=args.fps:raise ValueError('Generation fps mismatch')
                snapshot=folder/'generation_manifest.json';shutil.copy2(args.source_manifest,snapshot);c['source_manifest']=str(snapshot.relative_to(ROOT));c['source_sha256']=sha(snapshot)
            raw=np.loadtxt(source,delimiter=',');assert raw.ndim==2 and raw.shape[1]==36 and len(raw)>=2 and np.isfinite(raw).all()
            t=np.arange(len(raw))/args.fps;ti=np.arange(int(np.floor(t[-1]*50))+1)/50;qsrc=np.stack([np.interp(ti,t,raw[:,j]) for j in range(36)],-1);qq=Slerp(t,Rotation.from_quat(raw[:,[4,5,6,3]]))(ti).as_quat();qsrc[:,3:7]=qq[:,[3,0,1,2]]
            source_names=[mujoco.mj_id2name(sm,mujoco.mjtObj.mjOBJ_BODY,j) for j in range(1,sm.nbody)];poses=[];quats=[]
            for q in qsrc:
                sd.qpos[:]=q;mujoco.mj_kinematics(sm,sd);poses.append(sd.xpos[1:].copy());quats.append(sd.xquat[1:].copy())
            srcpos=np.stack(poses);srcquat=np.stack(quats);footids=[source_names.index(side+'_ankle_roll_link') for side in ['left','right']];foot=srcpos[:,footids];vel=np.linalg.norm(np.gradient(foot[:,:,:2],.02,axis=0),axis=-1);contact=(foot[:,:,2]<np.percentile(foot[:,:,2],5,axis=0)+.025)&(vel<.15);b={'foot_contacts':contact}
            c['contact_schedule']='heuristic_ankle_height_and_speed_requires_QA'
        feet=[bid(m,s+'_ankle_roll_link') for s in ['left','right']];sf=[source_names.index(s+'_ankle_roll_link') for s in ['left','right']]
        rootheight=q0[2];scale=(q0[2]-basepos[feet,2].mean())/(qsrc[0,2]-srcpos[0,sf,2].mean())
        frame_ids=list(range(0,len(qsrc),2))
        if frame_ids[-1]!=len(qsrc)-1:frame_ids.append(len(qsrc)-1)
        outputs=[];diagnostics=[];prev=q0[qa].copy();initial_roots=qsrc[0,:3].copy();previous_root=q0[:3].copy()
        for frame in frame_ids:
            sr=srcpos[frame];root=q0[:3]+scale*(qsrc[frame,:3]-initial_roots)
            orientation=qsrc[frame,3:7];R=rot(orientation);qfixed=q0.copy();qfixed[:3]=root;qfixed[3:7]=orientation
            # Initializer only: position IK, not copied joint trajectories, determines the output.
            for k,name in enumerate(names):
                if name in source_joint_names:
                    idx=source_joint_names.index(name)
                    qfixed[qa[k]]=np.clip(qsrc[frame,7+idx],*limits[k])
            d.qpos[:]=qfixed;mujoco.mj_kinematics(m,d)
            targets=[];ids=[]
            for side in ['left','right']:
                shoulder=bid(m,side+'_shoulder_yaw_link');elbow=bid(m,side+'_elbow_link');wrist=bid(m,side+'_wrist_roll_rubber_hand')
                gs=[source_names.index(side+'_'+n) for n in ['shoulder_yaw_link','elbow_link','wrist_roll_link']]
                shoulder_anchor=d.xpos[shoulder].copy()
                upper=sr[gs[1]]-sr[gs[0]];lower=sr[gs[2]]-sr[gs[1]]
                e=shoulder_anchor+upper/np.linalg.norm(upper)*np.linalg.norm(basepos[elbow]-basepos[shoulder])
                w=e+lower/np.linalg.norm(lower)*np.linalg.norm(basepos[wrist]-basepos[elbow])
                ids.extend([elbow,wrist]);targets.extend([e,w])
            footpos=basepos[feet]+scale*(sr[sf]-srcpos[0,sf])
            footrot=[rot(srcquat[frame,k])@rot(srcquat[0,k]).T@baserot[dest] for k,dest in zip(sf,feet)]
            # Account for K1 foot collision spheres under the requested orientation.
            if args.profile=='v2':
                for kk,foot in enumerate(feet):
                    gs=[g for g in range(m.ngeom) if m.geom_bodyid[g]==foot and m.geom_contype[g] and m.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE]
                    bottom=min((footrot[kk]@m.geom_pos[g])[2]-m.geom_size[g,0] for g in gs)
                    clearance=.001 if b['foot_contacts'][frame,kk] else .015
                    footpos[kk,2]=max(footpos[kk,2],clearance-bottom)
            fingerids=[bid(m,f'finger_{side}_link{k}') for side in ['l','r'] for k in [4,8,12,16,20]]
            # Support-aware COM term uses the reference contact schedule, no ground teleporting.
            contacts=b['foot_contacts'][frame].astype(bool);support=footpos[contacts,:2].mean(0) if contacts.any() else root[:2]
            def setstate(x):
                d.qpos[:]=qfixed;d.qpos[:3]=x[:3];d.qpos[qa]=x[3:];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            def residual(x):
                setstate(x)
                re=[(d.xpos[feet]-footpos).ravel()*80]
                re.extend([Rotation.from_matrix(footrot[k].T@d.xmat[f].reshape(3,3)).as_rotvec()*12 for k,f in enumerate(feet)])
                re.append((d.xpos[ids]-targets).ravel()*5)
                re.append((x[:3]-root)*3)
                re.append((d.subtree_com[bid(m,'pelvis'),:2]-support)*12)
                re.append((x[3:]-qfixed[qa])*.1)
                re.append((x[3:]-prev)*(.2 if args.profile=='v2' else .08))
                # Keep passive fingers away from the pelvis/hip envelope.
                if args.profile=='v2':
                    pts=d.xpos[fingerids];local=pts-d.xpos[bid(m,'pelvis')]
                    near=(np.abs(local[:,2])<.24)&(np.abs(local[:,0])<.16)
                    side=np.r_[np.ones(5),-np.ones(5)]
                    re.append(np.where(near,np.maximum(.215-side*local[:,1],0),0)*30)
                return np.concatenate(re)
            lo=np.r_[root-np.array([.18,.14,.08]),limits[:,0]+1e-5];hi=np.r_[root+np.array([.18,.14,.08]),limits[:,1]-1e-5]
            if outputs and args.profile=='v2':
                maxdelta=np.array([manifest['body_actuator_parameters'][name]['speed'] for name in names])*.04*.65
                lo[3:]=np.maximum(lo[3:],prev-maxdelta);hi[3:]=np.minimum(hi[3:],prev+maxdelta)
            x=np.clip(np.r_[previous_root,prev],lo+1e-8,hi-1e-8)
            sol=least_squares(residual,x,bounds=(lo,hi),max_nfev=40,ftol=2e-5,xtol=2e-5,gtol=2e-5)
            setstate(sol.x);outputs.append(d.qpos.copy());prev=sol.x[3:].copy();previous_root=sol.x[:3].copy()
            diagnostics.append({'frame':frame,'foot_position_error_m':float(np.max(np.linalg.norm(d.xpos[feet]-footpos,axis=-1))),'foot_orientation_error_rad':float(max(np.linalg.norm(Rotation.from_matrix(footrot[k].T@d.xmat[f].reshape(3,3)).as_rotvec()) for k,f in enumerate(feet))),'arm_position_error_m':float(np.max(np.linalg.norm(d.xpos[ids]-targets,axis=-1))),'iterations':sol.nfev,'solver_success':bool(sol.success)})
        keys=np.stack(outputs);n=len(qsrc);times=np.arange(n);out=np.stack([np.interp(times,frame_ids,keys[:,i]) for i in range(m.nq)],-1)
        rr=Rotation.from_quat(keys[:,[4,5,6,3]]);qq=Slerp(frame_ids,rr)(times).as_quat();out[:,3:7]=qq[:,[3,0,1,2]]
        folder=ROOT/'data'/('custom' if args.source_csv else args.profile)/c['clip_id'];folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'reference.npz',qpos=out,fps=np.array(50),joint_names=np.array(names),source_qpos=qsrc,source_foot_contacts=b['foot_contacts'])
        np.savetxt(folder/'qpos.csv',out,delimiter=',')
        stats={'clip_id':c['clip_id'],'motion_id':motion,'frames':n,'fps':50,'source_G1_csv':str(source.relative_to(ROOT.parent)),'source_csv_sha256':sha(source),'original_Kimodo_manifest':c['source_manifest'],'original_source_sha256':c['source_sha256'],'source_bundle_sha256':c['bundle_sha256'],'method':'task_space_IK_'+args.profile,'profile':args.profile,'leg_scale':float(scale),'root_orientation':'preserved_Kimodo_reference','source_contact_schedule':c.get('contact_schedule','canonical_G1_bundle'),'hand_targets':'constant_zero_clipped_to_official_limits','model_sha256':manifest['model_sha256'],'max_sampled_foot_position_error_m':max(x['foot_position_error_m'] for x in diagnostics),'max_sampled_arm_position_error_m':max(x['arm_position_error_m'] for x in diagnostics),'ik_diagnostics':diagnostics,'qpos_csv':str((folder/'qpos.csv').relative_to(ROOT)),'reference_npz':str((folder/'reference.npz').relative_to(ROOT))}
        write(folder/'manifest.json',stats);records.append({k:v for k,v in stats.items() if k!='ik_diagnostics'});print(json.dumps(records[-1]),flush=True)
    write(ROOT/'reports'/('retarget_'+args.clip_id+'.json' if args.source_csv else 'retarget_summary_v2.json'),{'scope':'kinematic_Kimodo_G1_to_K1_HX5_not_dynamic_success','clips':records,'elapsed_s':time.monotonic()-started})
if __name__=='__main__':main()
