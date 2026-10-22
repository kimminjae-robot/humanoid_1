"""Same-policy MuJoCo transfer. State writes occur only in reset/contract fixtures.

Run with the workspace Kimodo Python (Torch), borrowing only the workspace's
pinned MuJoCo site-packages after Torch/NumPy have loaded. No shared-env install.
"""
import argparse, hashlib, json, sys, time, shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    import mujoco
except ImportError:
    sys.path.append(str(ROOT / '.venvs/mujoco/lib/python3.11/site-packages'))
    import mujoco
from tracking.core import observe, torque_pd, pd


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def tensor(x): return torch.as_tensor(np.asarray(x), dtype=torch.float32)
def rotate(q,v):
    t=2*np.cross(q[...,1:],v)
    return v+q[...,:1]*t+np.cross(q[...,1:],t)
def inverse(q): return q*np.array([1,-1,-1,-1], dtype=q.dtype)
def multiply(a,b):
    w=a[...,0]*b[...,0]-(a[...,1:]*b[...,1:]).sum(-1)
    xyz=a[...,:1]*b[...,1:]+b[...,:1]*a[...,1:]+np.cross(a[...,1:],b[...,1:])
    return np.concatenate((w[...,None],xyz),-1)
def observe_numpy(pos,q,lin,ang,joint,rate,previous,rpos,rq,rj,rlin,phase):
    qi=inverse(q);delta=multiply(qi,rq);delta*=np.where(delta[:,:1]<0,-1,1)
    gravity=np.zeros_like(pos);gravity[:,2]=-1
    return np.concatenate((rotate(qi,lin),.25*rotate(qi,ang),rotate(qi,gravity),joint-rj[:,0],.05*rate,previous,rotate(qi,rpos-pos),2*delta[:,1:],rj.reshape(-1,87),rotate(qi,rlin),np.sin(phase)[:,None],np.cos(phase)[:,None]),-1).clip(-20,20)
def pd_numpy(q,qd,target,kp,kd,effort,velocity):
    raw=kp*(target-q)-kd*qd
    tau=np.clip(raw,-effort,effort)
    return raw,np.where((np.abs(qd)>velocity)&(tau*qd>0),0,tau)


class Bridge:
    def __init__(self, export, isaac, dt):
        self.contract=json.loads((export/'contract.json').read_text());self.c=self.contract['controller']
        self.names=self.c['backend_joint_maps']['csv_source'];self.dt=dt
        assert dt in (.002,.001), 'Refinement keeps policy 50Hz and torque 500Hz'
        assert self.c['physics_hz']==500 and self.c['control_hz']==50 and self.c['decimation']==10
        for name,digest in json.loads((export/'checksums.json').read_text()).items():
            assert sha(export/name)==digest, 'Modified export: '+name
        assert sha(export/'policy.pt')==self.contract['policy']['sha256']
        assert sha(ROOT/'configs/evaluation.json')==self.contract['evaluation_sha256']
        for key,path in [('mjcf','g1_robot.xml'),('scene','g1_scene.xml')]:
            assert sha(ROOT/'assets/g1'/path)==self.c['robot_hashes'][key]
        self.policy=torch.jit.load(str(export/'policy.pt'),map_location='cpu').eval()
        self.source=json.loads((isaac/'result.json').read_text())
        self.cases=sorted(self.source['episodes'],key=lambda e:e['env_id'])
        assert len(self.cases)==self.source['episodes_expected']==60
        assert [e['env_id'] for e in self.cases]==list(range(60))
        self.dataset=json.loads((ROOT/self.contract['dataset_index']).read_text())
        self.clips={c['clip_id']:c for c in self.dataset['clips'] if c['accepted'] and c['split']=='validation'}
        self.refs={}
        for cid,c in self.clips.items():
            assert sha(ROOT/c['bundle'])==c['bundle_sha256']
            with np.load(ROOT/c['bundle'],allow_pickle=False) as b:self.refs[cid]={k:b[k].copy() for k in b.files}
        scene=ET.parse(ROOT/'assets/g1/g1_scene.xml').getroot()
        scene.find('include').set('file',str(ROOT/'assets/g1/g1_robot.xml'))
        ground=scene.find('worldbody/geom')
        ground.set('type','box');ground.set('size','100 100 0.1');ground.set('pos','0 0 -0.1')
        scene.find('option').set('timestep',str(dt))
        self.effective_xml=ET.tostring(scene,encoding='unicode')
        self.model=mujoco.MjModel.from_xml_string(self.effective_xml)
        m=self.model;m.opt.timestep=dt
        # Frozen profile: match flat ground height/size and sliding friction.
        # Solver/contact compliance remain MuJoCo defaults, recorded, not tuned to scores.
        floor=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'floor');self.floor=floor
        m.geom_friction[:,0]=.8
        jids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in self.names]
        assert min(jids)>=0 and len(set(jids))==29
        self.qadr=m.jnt_qposadr[jids];self.vadr=m.jnt_dofadr[jids]
        self.motor=np.array([next(i for i in range(m.nu) if m.actuator_trnid[i,0]==j) for j in jids])
        assert len(set(self.motor))==29
        assert np.allclose(m.actuator_gear[self.motor,0],1)
        self.bnames=next(iter(self.refs.values()))['body_names'].tolist()
        self.bids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for n in self.bnames]
        assert min(self.bids)>0
        self.hands=[self.bnames.index(n) for n in ['left_wrist_yaw_link','right_wrist_yaw_link']]
        self.torso=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'torso_link')
        for key in ['kp','kd','action_scale','effort_limits','velocity_limits','joint_position_limits']:
            setattr(self,key,np.asarray([self.c[key][n] for n in self.names],dtype=np.float32))
        assert np.allclose(m.jnt_range[jids],self.joint_position_limits)
        assert np.allclose(m.actuator_ctrlrange[self.motor],np.stack((-self.effort_limits,self.effort_limits),-1))
        self.gates=json.loads((ROOT/'configs/evaluation.json').read_text())

    def reset(self,e):
        d=mujoco.MjData(self.model);s=e['initial_state']
        d.qpos[:7]=s['root_pose_local'];d.qpos[self.qadr]=s['joint_pos_source']
        twist=np.asarray(s['root_twist_world']);d.qvel[:3]=twist[:3]
        d.qvel[3:6]=rotate(inverse(d.qpos[3:7]),twist[3:]);d.qvel[self.vadr]=s['joint_vel_source']
        mujoco.mj_forward(self.model,d)
        return d

    def state(self,d):
        return [d.qpos[:3],d.qpos[3:7],d.qvel[:3],rotate(d.qpos[3:7],d.qvel[3:6]),d.qpos[self.qadr],d.qvel[self.vadr]]

    def inputs(self,datas,cases,frames,prev):
        state=[np.stack([self.state(d)[k] for d in datas]).astype(np.float32) for k in range(6)]
        rp=[];rq=[];rj=[];rv=[];phase=[]
        for e,f in zip(cases,frames):
            b=self.refs[e['clip_id']];n=len(b['joint_pos_rad']);idx=min(f,n-1)
            rp.append(b['root_pos_world_m'][idx]);rq.append(b['root_quat_world_wxyz'][idx])
            rj.append(b['joint_pos_rad'][np.minimum(f+np.array([0,5,15]),n-1)])
            rv.append(b['root_lin_vel_world_m_s'][idx] if f<n else np.zeros(3));phase.append(2*np.pi*min(f/(n-1),1))
        return [*map(tensor,state),tensor(prev),tensor(rp),tensor(rq),tensor(rj),tensor(rv),tensor(phase)]

    def fixtures(self,export):
        fixture=np.load(export/'observation_action_fixtures.npz');checks={};errors={}
        expected=fixture['action'];actual=self.policy(tensor(fixture['observation'])).detach().numpy()
        errors['export_action_vs_Isaac_GPU']=float(np.max(np.abs(actual-expected)))
        checks['export_action']=errors['export_action_vs_Isaac_GPU']<=1e-5
        cases=self.cases[:6];datas=[self.reset(e) for e in cases]
        args=self.inputs(datas,cases,np.zeros(6,dtype=int),np.zeros((6,29),np.float32))
        observed=observe(*args).numpy()
        errors['six_explicit_reset_observations_vs_recorded_Isaac']=float(np.max(np.abs(observed-fixture['observation'][:6])))
        checks['actual_Isaac_initial_observations']=errors['six_explicit_reset_observations_vs_recorded_Isaac']<=1e-5
        # Additional asymmetric quaternion, nonzero world twist/history and motor-limit fixtures.
        rng=np.random.default_rng(7921);count=256
        args=[tensor(rng.normal(0,.3,(count,n))) for n in [3,4,3,3,29,29,29,3,4]]
        for i in [1,8]:args[i]/=torch.linalg.vector_norm(args[i],dim=-1,keepdim=True)
        args += [tensor(rng.normal(0,.3,(count,3,29))),tensor(rng.normal(0,.3,(count,3))),tensor(rng.uniform(0,2*np.pi,count))]
        errors['numpy_vs_Torch_observation_asymmetric_states']=float(np.max(np.abs(observe(*args).numpy()-observe_numpy(*[x.numpy() for x in args]))))
        checks['independent_observation_math']=errors['numpy_vs_Torch_observation_asymmetric_states']<=1e-5
        q=tensor(rng.normal(0,.3,(count,29)));rate=tensor(rng.uniform(-1.2,1.2,(count,29))*self.velocity_limits)
        ref=tensor(rng.normal(0,.3,(count,29)));action=tensor(rng.normal(0,2,(count,29)))
        target,raw,tau=pd(q,rate,ref,action,*[tensor(getattr(self,k)) for k in ['kp','kd','action_scale','joint_position_limits','effort_limits','velocity_limits']])
        nt=np.clip(ref.numpy()+self.action_scale*action.numpy().clip(-1,1),self.joint_position_limits[:,0],self.joint_position_limits[:,1])
        nr,nu=pd_numpy(q.numpy(),rate.numpy(),nt,self.kp,self.kd,self.effort_limits,self.velocity_limits)
        errors['numpy_vs_shared_Isaac_Torch_PD_Nm']=float(np.max(np.abs(nu-tau.numpy())))
        errors['target_rad']=float(np.max(np.abs(nt-target.numpy())))
        checks['PD_torque']=errors['numpy_vs_shared_Isaac_Torch_PD_Nm']<=1e-4
        checks['PD_target']=errors['target_rad']<=1e-5
        checks['speed_guard_suppresses_outward_and_keeps_braking']=bool(np.any((np.abs(rate.numpy())>self.velocity_limits)&(nu==0)) and np.any((np.abs(rate.numpy())>self.velocity_limits)&(nu*rate.numpy()<0)))
        checks['all_saved_initial_states_roundtrip']=all(np.max(np.abs(self.reset(e).qpos[self.qadr]-e['initial_state']['joint_pos_source']))<1e-8 and np.max(np.abs(self.state(self.reset(e))[3]-np.array(e['initial_state']['root_twist_world'])[3:]))<1e-8 for e in self.cases)
        return {'scope':'recorded_Isaac_reset_observations_and_GPU_actor; independent_NumPy_vs_shared_Isaac_Torch_PD_not_measured_PhysX_applied_efforts','checks':checks,'max_absolute_errors':errors,'passed':all(checks.values()),'random_fixture_seed':7921,'asymmetric_fixture_count':256,'recorded_actor_fixtures':len(expected)}

    def forbidden(self,d):
        net=np.zeros(3)
        for contact_id,c in enumerate(d.contact):
            if self.floor not in (c.geom1,c.geom2):continue
            other=c.geom2 if c.geom1==self.floor else c.geom1
            if self.model.geom_bodyid[other]!=self.torso:continue
            force=np.zeros(6);mujoco.mj_contactForce(self.model,d,contact_id,force)
            net+=(1 if c.geom1==self.floor else -1)*c.frame.reshape(3,3).T@force[:3]
        return np.linalg.norm(net)>1

    def run(self,run):
        cases=self.cases;count=len(cases);datas=[self.reset(e) for e in cases]
        frames=np.zeros(count,dtype=int);previous=np.zeros((count,29),np.float32)
        finished=np.zeros(count,bool);bad=np.zeros(count,dtype=int);forbidden=np.zeros(count,bool);finite=np.ones(count,bool)
        sums=np.zeros((count,4));sat=np.zeros(count,dtype=int);samples=np.zeros(count,dtype=int);episodes=[]
        traces=[];ftrace=[];strace=[];g=self.gates['dynamic_episode_gate'];substeps=round(.002/self.dt)
        max_steps=max(len(b['joint_pos_rad'])-1+50 for b in self.refs.values());started=time.monotonic()
        for step in range(max_steps):
            args=self.inputs(datas,cases,frames,previous)
            obs=observe(*args);actions=self.policy(obs).detach().numpy()
            targets=np.clip(args[9][:,0].numpy()+self.action_scale*actions.clip(-1,1),self.joint_position_limits[:,0],self.joint_position_limits[:,1])
            step_sat=np.zeros(count,dtype=int)
            for i,(d,e) in enumerate(zip(datas,cases)):
                if finished[i]:continue
                previous[i]=actions[i].clip(-1,1);finite[i]&=np.isfinite(actions[i]).all()
                for tick in range(10):
                    q=d.qpos[self.qadr].astype(np.float32);qd=d.qvel[self.vadr].astype(np.float32)
                    raw,tau=pd_numpy(q,qd,targets[i],self.kp,self.kd,self.effort_limits,self.velocity_limits)
                    saturated=int((np.abs(raw)>=self.effort_limits).sum());sat[i]+=saturated;step_sat[i]+=saturated
                    finite[i]&=np.isfinite(tau).all()
                    d.ctrl[self.motor]=tau
                    for _ in range(substeps):
                        gz=rotate(inverse(d.qpos[3:7]),np.array([0.,0.,-1.]))[2]
                        invalid=d.qpos[2]<g['min_base_height_m'] or gz>-np.cos(np.deg2rad(g['max_base_tilt_deg']))
                        bad[i]=bad[i]+1 if invalid else 0
                        forbidden[i]|=self.forbidden(d)
                        finite[i]&=np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
                        if not finite[i]:break
                        mujoco.mj_step(self.model,d)
                    if not finite[i]:break
                frames[i]+=1;samples[i]+=1
                mujoco.mj_forward(self.model,d) # refresh link FK at the post-step qpos
                b=self.refs[e['clip_id']];idx=min(frames[i],len(b['joint_pos_rad'])-1)
                actual=d.xpos[self.bids];pos=d.qpos[:3];bodyref=b['body_pos_world_m'][idx];rootref=b['root_pos_world_m'][idx]
                sums[i]+=np.array([np.mean((d.qpos[self.qadr]-b['joint_pos_rad'][idx])**2),np.mean(np.sum(((actual-pos)-(bodyref-rootref))**2,axis=-1)),np.sum((pos-rootref)**2),np.mean(np.sum((actual[self.hands]-bodyref[self.hands])**2,axis=-1))])
                fall=bad[i]>=round(g['fall_violation_persistence_s']/self.dt)
                full=frames[i]>=len(b['joint_pos_rad'])-1+50
                if full or fall or forbidden[i] or not finite[i]:
                    rmse=np.sqrt(sums[i]/samples[i]);frac=float(sat[i]/(samples[i]*290))
                    checks={'full_clip_plus_hold':bool(full),'finite':bool(finite[i]),'no_fall':not bool(fall),'no_forbidden_contact':not bool(forbidden[i]),'joint':bool(rmse[0]<=g['max_joint_position_rmse_rad']),'body':bool(rmse[1]<=g['max_root_relative_body_position_rmse_m']),'root':bool(rmse[2]<=g['max_world_root_position_rmse_m']),'hand':bool(rmse[3]<=g['max_hand_position_rmse_m']),'saturation':frac<=g['max_torque_saturation_fraction']}
                    episodes.append({**{k:e[k] for k in ['env_id','clip_id','motion_id','evaluation_seed','initial_state']},'steps':int(samples[i]),**dict(zip(['rmse_joint_rad','rmse_body_m','rmse_root_m','rmse_hand_m'],rmse.tolist())),'torque_saturation_fraction':frac,'checks':checks,'success':all(checks.values())})
                    finished[i]=True
            traces.append(np.stack([np.r_[d.qpos[:7],d.qpos[self.qadr]] for d in datas]));ftrace.append(frames.copy());strace.append(step_sat)
            if step%100==0:print(json.dumps({'control_step':step,'finished':int(finished.sum()),'elapsed_s':round(time.monotonic()-started,1)}),flush=True)
            if finished.all():break
        rates={m:sum(e['success'] for e in episodes if e['motion_id']==m)/sum(e['motion_id']==m for e in episodes) for m in self.source['motion_ids']}
        clipids=[e['clip_id'] for e in cases[:6]]
        np.savez_compressed(run/'rollout.npz',qpos=np.stack(traces),frame_indices=np.stack(ftrace),saturated_joint_physics_steps=np.stack(strace),env_clip_ids=np.array([e['clip_id'] for e in cases]),clip_ids=np.array(clipids))
        result={'scope':'MuJoCo_free_base_dynamic_same_policy_no_retraining','backend':'MuJoCo','episodes_expected':count,'episodes_measured':len(episodes),'motion_ids':self.source['motion_ids'],'success_rates':rates,'passed':all(v>=.9 for v in rates.values()),'episodes':sorted(episodes,key=lambda e:e['env_id']),'elapsed_s':time.monotonic()-started}
        write(run/'result.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--export',type=Path,default=ROOT/'exports/stage04_candidate_v11');p.add_argument('--isaac-run',type=Path,default=ROOT/'runs/20261021_130836_931130_stage04_multimotion_eval_v11');p.add_argument('--physics-dt',type=float,choices=[.002,.001],default=.002);p.add_argument('--run-dir',type=Path);p.add_argument('--fixtures-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    run=(a.run_dir or ROOT/'runs'/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_stage05_mujoco')).resolve();run.mkdir(parents=True,exist_ok=False)
    print('RUN '+str(run),flush=True)
    write(run/'execution.json',{'command':[sys.executable,'-I',*sys.argv],'running':True})
    provenance=run/'provenance';provenance.mkdir()
    for src in [Path(__file__),ROOT/'tracking/core.py',ROOT/'configs/evaluation.json',a.export/'contract.json',a.export/'checksums.json']:
        shutil.copy2(src,provenance/(src.parent.name+'__'+src.name))
    bridge=Bridge(a.export,a.isaac_run,a.physics_dt)
    (provenance/'effective_scene.xml').write_text(bridge.effective_xml)
    fixture=bridge.fixtures(a.export);write(run/'contract_validation.json',fixture)
    print(json.dumps(fixture),flush=True)
    if not fixture['passed']:raise RuntimeError('Contract fixtures failed; no dynamic evaluation')
    m=bridge.model
    write(run/'resolved.json',{'backend':'MuJoCo','source_Isaac_run':str(a.isaac_run.resolve().relative_to(ROOT)),'policy_sha256':sha(a.export/'policy.pt'),'source_checkpoint_sha256':bridge.contract['checkpoint_sha256'],'dataset_index':bridge.contract['dataset_index'],'evaluation_sha256':sha(ROOT/'configs/evaluation.json'),'initial_states_source_sha256':sha(a.isaac_run/'result.json'),'same_policy_no_retraining':True,'physics_dt':a.physics_dt,'torque_dt':.002,'control_dt':.02,'precision':{'physics':'float64','policy_and_PD':'float32'},'versions':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'mujoco':mujoco.__version__},'runtime_dependencies':{'torch':torch.__file__,'mujoco':mujoco.__file__},'joint_names_source':bridge.names,'source_qpos_addresses':bridge.qadr.tolist(),'source_qvel_addresses':bridge.vadr.tolist(),'source_to_motor':bridge.motor.tolist(),'ground':'static_box_200x200x0.2_center_z_minus_0.1','sliding_friction':.8,'solver':{'integrator':int(m.opt.integrator),'type':int(m.opt.solver),'iterations':int(m.opt.iterations),'tolerance':m.opt.tolerance,'geom_solref':m.geom_solref.tolist(),'geom_solimp':m.geom_solimp.tolist()},'known_engine_differences':['MuJoCo soft contact vs PhysX TGS, no restitution parameter equivalence claimed','MuJoCo collision exclusions/contact generation differ from PhysX','MuJoCo torsional/rolling friction and joint-limit compliance retained from canonical model'],'state_write_scope':'reset_only; rollout uses ctrl and mj_step'} )
    if a.fixtures_only:
        write(run/'execution.json',{'command':[sys.executable,'-I',*sys.argv],'running':False,'status':'contract_fixtures_passed'})
        return
    with torch.inference_mode(): result=bridge.run(run)
    write(run/'execution.json',{'command':[sys.executable,'-I',*sys.argv],'running':False,'status':'dynamic_evaluation_complete','integration_executed':True,'performance_passed':result['passed']})
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'}),flush=True)
if __name__=='__main__':main()
