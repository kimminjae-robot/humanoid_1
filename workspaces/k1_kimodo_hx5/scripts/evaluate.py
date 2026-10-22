"""Kinematic QA and free-base MuJoCo PD/support-feedforward baseline."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import nnls
ROOT=Path(__file__).resolve().parents[1]
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--controller',choices=['pd','support_ff','balance','motion_balance','wbc','centroidal'],default='wbc');ap.add_argument('--label');ap.add_argument('--physics-dt',type=float,choices=[.002,.001],default=.002);ap.add_argument('--balance-config',type=Path);ap.add_argument('--references',type=Path,default=ROOT/'data/dataset.json');a=ap.parse_args()
    balance_cfg=json.loads(a.balance_config.read_text()) if a.balance_config else {'orientation':2.,'angular_rate':.5,'position':2.,'linear_rate':1.}
    cfg=json.loads((ROOT/'configs/evaluation.json').read_text());manifest=json.loads((ROOT/'reports/robot_manifest.json').read_text());summary=json.loads(a.references.read_text());m=mujoco.MjModel.from_xml_path(str(ROOT/'assets/k1_hx5.xml'));assert sha(ROOT/'assets/k1_hx5.xml')==manifest['model_sha256']
    m.opt.timestep=a.physics_dt;pd_substeps=round(.002/a.physics_dt);decimation=round(.02/a.physics_dt)
    d=mujoco.MjData(m);rd=mujoco.MjData(m);n=manifest['body_joint_names'];jid=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,x) for x in n];qa=m.jnt_qposadr[jid];va=m.jnt_dofadr[jid];motor=np.array([next(i for i in range(m.nu) if m.actuator_trnid[i,0]==j) for j in jid]);pars=manifest['body_actuator_parameters'];kp=np.array([pars[x]['kp'] for x in n]);kd=np.array([pars[x]['kd'] for x in n]);effort=np.array([pars[x]['effort'] for x in n]);speed=np.array([pars[x]['speed'] for x in n]);floor=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'floor');torso=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'torso_link');wrists=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,s+'_wrist_roll_rubber_hand') for s in ['left','right']];feet=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,s+'_ankle_roll_link') for s in ['left','right']];footgeoms=[[g for g in range(m.ngeom) if m.geom_bodyid[g]==b and m.geom_contype[g] and m.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE] for b in feet]
    run=ROOT/'runs'/(a.label or a.controller);run.mkdir(parents=True,exist_ok=False);reports=[];episodes=[];started=time.monotonic()
    write(run/'resolved.json',{'controller':a.controller,'physics_dt':a.physics_dt,'PD_dt':.002,'config':cfg,'robot_sha256':manifest['model_sha256'],'mujoco_version':mujoco.__version__,'numpy_version':np.__version__,'support_ff':'static vertical ground forces solve floating-base gravity balance with nonnegative least squares; actuator gravity minus J_transpose_force; no root actuation','root_state_writes':'reset_only','policy_training':False,'balance_parameters':balance_cfg,'references_path':str(a.references),'evaluation_sha256':sha(ROOT/'configs/evaluation.json')})
    for clip in summary['clips']:
        b=np.load(ROOT/clip['reference_npz'],allow_pickle=False);ref=b['qpos'];schedule=b['source_foot_contacts'].astype(bool);count=len(ref);refvel=np.zeros((count,m.nv))
        for ii in range(count-1):mujoco.mj_differentiatePos(m,refvel[ii],.02,ref[ii],ref[ii+1])
        rates=[];wrist_ref=[];foot_xyz=[];penetration=0;selfdepth=0;limiterr=0;finite=bool(np.isfinite(ref).all());massCOM=[]
        for frame,q in enumerate(ref):
            rd.qpos[:]=q;rd.qvel[:]=0;mujoco.mj_forward(m,rd);wrist_ref.append(rd.xpos[wrists].copy());foot_xyz.append(rd.xpos[feet].copy());massCOM.append(rd.subtree_com[1].copy())
            limiterr=max(limiterr,float(np.maximum(m.jnt_range[1:,0]-q[m.jnt_qposadr[1:]],q[m.jnt_qposadr[1:]]-m.jnt_range[1:,1]).max()))
            for c in rd.contact:
                if c.dist>=0:continue
                if floor in (c.geom1,c.geom2):penetration=max(penetration,float(-c.dist))
                else:selfdepth=max(selfdepth,float(-c.dist))
            if frame:
                v=np.zeros(m.nv);mujoco.mj_differentiatePos(m,v,.02,ref[frame-1],q);rates.append(v[va])
        foot_xyz=np.stack(foot_xyz);wrist_ref=np.stack(wrist_ref);ratios=abs(np.stack(rates))/speed
        slip=np.linalg.norm(np.diff(foot_xyz[:,:,:2],axis=0),axis=-1)/.02;contact=schedule[1:]&schedule[:-1];fraction=float(((slip>.15)&contact).sum()/max(1,contact.sum()))
        k=cfg['kinematic'];checks={'finite':finite,'joint_limits':limiterr<=k['joint_limit_tolerance_rad'],'ground_penetration':penetration<=k['max_ground_penetration_m'],'self_collision':selfdepth<=k['max_self_collision_depth_m'],'joint_velocity_fraction':float((ratios>1).mean())<=k['max_velocity_exceedance_fraction'],'joint_velocity_peak':float(ratios.max())<=k['max_velocity_ratio'],'stance_foot_slip':fraction<=k['max_stance_slip_fraction'],'IK_foot_accuracy':clip['max_sampled_foot_position_error_m']<=k['max_IK_foot_position_error_m']}
        qa_report={'clip_id':clip['clip_id'],'checks':checks,'passed':all(checks.values()),'ground_penetration_m':penetration,'self_collision_depth_m':selfdepth,'joint_limit_violation_rad':max(0,limiterr),'max_joint_speed_ratio':float(ratios.max()),'velocity_exceedance_fraction':float((ratios>1).mean()),'stance_slip_fraction':fraction,'max_stance_slip_speed_m_s':float(slip[contact].max()) if contact.any() else 0,'hand_reference_joint_motion_rad':float(np.max(np.ptp(ref[:,[int(m.jnt_qposadr[j]) for j in range(1,m.njnt) if j not in jid]],axis=0)))}
        reports.append(qa_report);print('QA '+json.dumps(qa_report),flush=True)
        # Precompute support force feedforward at every reference pose. No reference acceleration assumed.
        feedforward=[];residuals=[]
        for frame,q in enumerate(ref):
            rd.qpos[:]=q;rd.qvel[:]=0;mujoco.mj_forward(m,rd);J=[]
            for side,gs in enumerate(footgeoms):
                if not schedule[frame,side]:continue
                for g in gs:
                    point=rd.geom_xpos[g].copy();point[2]-=m.geom_size[g,0];jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,rd,jp,jr,point,int(m.geom_bodyid[g]));J.append(jp[2])
            if J:
                J=np.stack(J);forces,error=nnls(J[:,:6].T,rd.qfrc_bias[:6]);ff=rd.qfrc_bias[va]-J[:,va].T@forces
            else:ff=rd.qfrc_bias[va].copy();error=float(np.linalg.norm(rd.qfrc_bias[:6]))
            feedforward.append(ff);residuals.append(float(error))
        for seed in cfg['reset_seeds']:
            if a.controller=='centroidal' or (a.controller=='wbc' and clip['motion_id']=='walk_slow'):
                import sys
                sys.path.insert(0,str(ROOT/'scripts'))
                from walk_wbc import WalkingWBC
                wbc=WalkingWBC(m,ref,schedule,qa,va,effort)
            else:wbc=None
            mujoco.mj_resetData(m,d);d.qpos[:]=ref[0];d.qpos[qa]+=np.random.default_rng(seed).normal(0,.002,23);d.qvel[:]=0;mujoco.mj_forward(m,d)
            initial={'qpos':d.qpos.tolist(),'qvel':d.qvel.tolist()};record=[];sumerr=np.zeros(3);sat=0;bad=0;forbidden=False;isfinite=True;gate=cfg['dynamic'];torque_ticks=0;self_penetration=0.
            for step in range(count-1+50):
                idx=min(step,count-1);target=ref[idx,qa];ff=feedforward[idx] if a.controller in ['support_ff','balance','motion_balance','wbc'] else 0
                for tick in range(decimation):
                    held=target.copy()
                    motion_mode=a.controller=='motion_balance' and clip['motion_id']=='walk_slow'
                    if a.controller in ['balance','motion_balance','wbc']:
                        actualR=d.xmat[1].reshape(3,3);qw=ref[min(step,count-1),3:7];w,x,y,z=qw
                        refR=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
                        from scipy.spatial.transform import Rotation
                        er=Rotation.from_matrix(refR.T@actualR).as_rotvec();ep=refR.T@(d.qpos[:3]-ref[min(step,count-1),:3]);ev=refR.T@(d.qvel[:3]-(refvel[min(step,count-1),:3] if motion_mode else 0))
                        pitch=np.clip(2*er[1]+.4*d.qvel[4]+.5*ep[0]+.25*ev[0],-.35,.35)
                        roll=np.clip(2*er[0]+.4*d.qvel[3]-.5*ep[1]-.25*ev[1],-.35,.35)
                        if motion_mode:
                            par=balance_cfg;pitch=np.clip(par['orientation']*er[1]+par['angular_rate']*(d.qvel[4]-refvel[min(step,count-1),4])+par['position']*ep[0]+par['linear_rate']*ev[0],-.35,.35)
                            roll=np.clip(par['orientation']*er[0]+par['angular_rate']*(d.qvel[3]-refvel[min(step,count-1),3])-par['position']*ep[1]-par['linear_rate']*ev[1],-.35,.35)
                        for side_index,side in enumerate(['left','right']):
                            if motion_mode and not schedule[min(step,count-1),side_index]:continue
                            held[n.index(side+'_ankle_pitch_joint')]+=pitch
                            held[n.index(side+'_ankle_roll_joint')]+=roll
                        held=np.clip(held,m.jnt_range[jid,0],m.jnt_range[jid,1])
                    raw=kp*(held-d.qpos[qa])-kd*(d.qvel[va]-(refvel[min(step,count-1),va] if motion_mode else 0))+ff
                    if wbc is not None:
                        if tick%pd_substeps==0:wbc_torque=wbc.compute(d,idx)
                        raw=wbc_torque+kp*(target-d.qpos[qa])-kd*(d.qvel[va]-refvel[min(step,count-1),va])
                    if tick%pd_substeps:raw=held_raw
                    else:held_raw=raw.copy()
                    tau=np.clip(raw,-effort,effort);tau=np.where((abs(d.qvel[va])>speed)&(tau*d.qvel[va]>0),0,tau);
                    if tick%pd_substeps:tau=held_tau
                    else:held_tau=tau.copy()
                    d.ctrl[motor]=tau
                    # Official HX position actuators hold constant zero targets; no finger task.
                    sat+=int((abs(raw)>=effort).sum());torque_ticks+=23
                    gz=d.xmat[1].reshape(3,3)[2,2];bad=bad+1 if d.qpos[2]<gate['min_root_height_m'] or gz<np.cos(np.deg2rad(gate['max_tilt_deg'])) else 0
                    for ci,c in enumerate(d.contact):
                        if floor not in (c.geom1,c.geom2):self_penetration=max(self_penetration,float(-c.dist))
                        if floor in (c.geom1,c.geom2):
                            other=c.geom2 if c.geom1==floor else c.geom1
                            if m.geom_bodyid[other]==torso:
                                force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force);forbidden |= np.linalg.norm(force[:3])>1
                    mujoco.mj_step(m,d)
                mujoco.mj_forward(m,d);idx=min(step+1,count-1);record.append(d.qpos.copy());sumerr += [np.mean((d.qpos[qa]-ref[idx,qa])**2),np.sum((d.qpos[:3]-ref[idx,:3])**2),np.mean(np.sum((d.xpos[wrists]-wrist_ref[idx])**2,axis=-1))];isfinite &= bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and np.isfinite(tau).all())
                fall=bad>=round(gate['fall_persistence_s']/a.physics_dt)
                if fall or forbidden or not isfinite:break
            steps=len(record);rmse=np.sqrt(sumerr/steps);fraction=sat/max(1,torque_ticks);checks={'full_clip_plus_hold':steps>=count-1+50,'finite':isfinite,'no_fall':not fall,'no_torso_ground':not forbidden,'joint':bool(rmse[0]<=gate['max_body_joint_rmse_rad']),'root':bool(rmse[1]<=gate['max_root_position_rmse_m']),'wrist':bool(rmse[2]<=gate['max_wrist_position_rmse_m']),'saturation':fraction<=gate['max_body_torque_saturation_fraction']}
            dest=run/(clip['clip_id']+'_'+str(seed)+'.npz');np.savez_compressed(dest,qpos=np.stack(record),initial_qpos=np.array(initial['qpos']),reference=ref,frame_index=np.minimum(np.arange(1,steps+1),count-1))
            e={'clip_id':clip['clip_id'],'motion_id':clip['motion_id'],'seed':seed,'steps':steps,'duration_s':steps*.02,'joint_rmse_rad':float(rmse[0]),'root_rmse_m':float(rmse[1]),'wrist_rmse_m':float(rmse[2]),'torque_saturation_fraction':fraction,'max_dynamic_self_penetration_m':self_penetration,'checks':checks,'success':all(checks.values()),'initial_state':initial,'rollout_npz':str(dest.relative_to(ROOT)),'reference_max_static_base_balance_residual':max(residuals),'WBC_diagnostics':wbc.diagnostics() if wbc else None};episodes.append(e);print('DYNAMIC '+json.dumps({k:v for k,v in e.items() if k!='initial_state'}),flush=True)
    write(run/'kinematic_qa.json',{'scope':'kinematic_reference_only','clips':reports,'passed':all(c['passed'] for c in reports)})
    rates={motion:sum(e['success'] for e in episodes if e['motion_id']==motion)/sum(e['motion_id']==motion for e in episodes) for motion in sorted({c['motion_id'] for c in summary['clips']})};write(run/'result.json',{'scope':'MuJoCo_free_base_dynamic_PD_baseline_not_Isaac_policy_sim2sim','controller':a.controller,'episodes':episodes,'success_rates':rates,'all_motions_passed':all(v>=.9 for v in rates.values()),'kinematic_all_passed':all(c['passed'] for c in reports),'elapsed_s':time.monotonic()-started});print('RESULT '+json.dumps(rates),flush=True)
if __name__=='__main__':main()
