"""Workspace-only free-base G1 reference tracking; reset is the only state write."""
import json
from pathlib import Path
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import Articulation,ArticulationCfg
from isaaclab.envs import DirectRLEnv,DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor,ContactSensorCfg
from isaaclab.utils import configclass
from .core import observe,torque_pd,rotate,inv

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=json.loads((ROOT/'configs/motion_contract.json').read_text())
CTRL=CONTRACT['controller'];NAMES=CONTRACT['csv']['joint_names']
ACT=json.loads((ROOT/'configs/g1_actuators.json').read_text())['joints']
EVAL=json.loads((ROOT/'configs/evaluation.json').read_text())


def robot_cfg():
    return ArticulationCfg(prim_path='/World/envs/env_.*/G1',
      spawn=sim_utils.UsdFileCfg(usd_path=str(ROOT/'assets/g1/usd/g1.usd'),activate_contact_sensors=True,
       rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,linear_damping=0.,angular_damping=0.,max_linear_velocity=1000.,max_angular_velocity=1000.,enable_gyroscopic_forces=True),
       articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=False,enabled_self_collisions=True,solver_position_iteration_count=16,solver_velocity_iteration_count=4),
       collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.002,rest_offset=0.)),
      init_state=ArticulationCfg.InitialStateCfg(pos=(0.,0.,.8),joint_pos=CTRL['joint_default_positions'],joint_vel={'.*':0.}),
      soft_joint_pos_limit_factor=1.,actuators={'torque':IdealPDActuatorCfg(joint_names_expr=['.*'],stiffness=0.,damping=0.,
       effort_limit=CTRL['effort_limits'],effort_limit_sim=CTRL['effort_limits'],velocity_limit_sim=1e6,
       armature={n:p['armature_kg_m2'] for n,p in ACT.items()},friction=0.,dynamic_friction=0.,viscous_friction=0.)})


@configclass
class TrackingCfg(DirectRLEnvCfg):
    decimation=10
    episode_length_s=20.
    action_space=29
    observation_space=194
    state_space=0
    is_finite_horizon=True
    sim=sim_utils.SimulationCfg(dt=.002,render_interval=10,physx=sim_utils.PhysxCfg(solver_type=1,min_position_iteration_count=16,min_velocity_iteration_count=4,gpu_max_rigid_contact_count=2**18,gpu_max_rigid_patch_count=2**15,gpu_found_lost_pairs_capacity=2**18,gpu_found_lost_aggregate_pairs_capacity=2**20,gpu_total_aggregate_pairs_capacity=2**18))
    scene=InteractiveSceneCfg(num_envs=32,env_spacing=5.,replicate_physics=True,clone_in_fabric=False)
    robot=robot_cfg()
    motions=('stand',)
    evaluation=False
    random_start=True
    controller=CTRL
    walking_sample_weight=1.


class TrackingEnv(DirectRLEnv):
    def __init__(self,cfg,**kwargs):
        self.ctrl=cfg.controller
        super().__init__(cfg,**kwargs)
        self.dataset_joint_names=NAMES
        self.jmap=torch.tensor([self.robot.joint_names.index(n) for n in NAMES],device=self.device)
        self.bnames=json.loads((ROOT/'configs/g1_fixtures.json').read_text())['body_names']
        self.bmap=torch.tensor([self.robot.body_names.index(n) for n in self.bnames],device=self.device)
        self.hands=[self.bnames.index(n) for n in ['left_wrist_yaw_link','right_wrist_yaw_link']]
        dataset_path=ROOT/json.loads((ROOT/'state/progress.json').read_text())['artifacts_stage_3']['dataset_index']
        self.dataset=json.loads(dataset_path.read_text())
        split='validation' if cfg.evaluation else 'train'
        self.clips=[c for c in self.dataset['clips'] if c['accepted'] and c['motion_id'] in cfg.motions and c['split']==split]
        assert self.clips
        counts={m:sum(c['motion_id']==m for c in self.clips) for m in cfg.motions}
        self.clip_probabilities=torch.tensor([(cfg.walking_sample_weight if c['motion_id']=='walk_slow' else 1.)/counts[c['motion_id']] for c in self.clips],device=self.device)
        self.motion_masks={m:torch.tensor([c['motion_id']==m for c in self.clips],device=self.device) for m in cfg.motions}
        fields=['root_pos_world_m','root_quat_world_wxyz','root_lin_vel_world_m_s','root_ang_vel_world_rad_s','joint_pos_rad','joint_vel_rad_s','body_pos_world_m']
        raw=[]
        for c in self.clips:
            with np.load(ROOT/c['bundle'],allow_pickle=False) as b:raw.append({k:b[k].copy() for k in fields})
        self.lengths=torch.tensor([len(b['joint_pos_rad']) for b in raw],device=self.device)
        size=int(self.lengths.max())+51
        self.refs={k:torch.tensor(np.stack([np.concatenate([b[k],np.repeat(b[k][-1:],size-len(b[k]),axis=0)]) for b in raw]),device=self.device,dtype=torch.float32) for k in fields}
        for k in ['root_lin_vel_world_m_s','root_ang_vel_world_rad_s','joint_vel_rad_s']:
            for i,n in enumerate(self.lengths.tolist()):self.refs[k][i,n:]=0
        self.cid=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self.frame=torch.zeros_like(self.cid);self.previous=torch.zeros((self.num_envs,29),device=self.device)
        self.actions=torch.zeros_like(self.previous);self.target=torch.zeros_like(self.previous)
        self.kp=self.vec('kp');self.kd=self.vec('kd');self.scale=self.vec('action_scale');self.effort=self.vec('effort_limits');self.velocity=self.vec('velocity_limits')
        self.limits=torch.tensor([self.ctrl['joint_position_limits'][n] for n in NAMES],device=self.device)
        self.sums=torch.zeros((self.num_envs,4),device=self.device);self.samples=torch.zeros_like(self.cid)
        self.sat=torch.zeros_like(self.cid);self.torque_steps=torch.zeros_like(self.cid)
        self.bad_duration=torch.zeros_like(self.cid);self.forbidden=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.nonfinite=self.forbidden.clone();self.finished=[];self.initial_states={};self.eval_counts=torch.zeros_like(self.cid)
        self.last_metrics={};self.last_qpos=None;self.last_frame=None;self.step_sat=torch.zeros_like(self.cid);self.last_step_sat=None

    def vec(self,key):return torch.tensor([self.ctrl[key][n] for n in NAMES],device=self.device)

    def _setup_scene(self):
        self.robot=Articulation(self.cfg.robot)
        ground=sim_utils.CuboidCfg(size=(200.,200.,.2),collision_props=sim_utils.CollisionPropertiesCfg(),physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.8,restitution=0.,friction_combine_mode='average'))
        ground.func('/World/ground',ground,translation=(0.,0.,-.1))
        mat=sim_utils.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.8,restitution=0.)
        mat.func('/World/G1Material',mat);sim_utils.bind_physics_material('/World/envs/env_0/G1','/World/G1Material')
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=['/World/ground'])
        self.scene.articulations['robot']=self.robot
        # Head sphere and torso capsule share torso_link. Filter ground contacts only.
        self.torso=ContactSensor(ContactSensorCfg(prim_path='/World/envs/env_.*/G1/pelvis/torso_link',update_period=0.,filter_prim_paths_expr=['/World/ground/geometry/mesh']))
        self.scene.sensors['torso']=self.torso
        light=sim_utils.DomeLightCfg(intensity=1800.,color=(.8,.8,.8));light.func('/World/Light',light)

    def ref(self,key,offset=0):
        t=(self.frame+offset).minimum(self.lengths[self.cid]-1)
        out=self.refs[key][self.cid,t]
        # qdot/twist reference becomes zero after clip, even for unusual source endpoints.
        if 'vel' in key:out=torch.where((self.frame>=self.lengths[self.cid]).view(-1,*([1]*(out.ndim-1))),0.,out)
        return out

    def state(self):
        d=self.robot.data
        return d.root_link_pos_w-self.scene.env_origins,d.root_link_quat_w,d.root_link_lin_vel_w,d.root_link_ang_vel_w,d.joint_pos[:,self.jmap],d.joint_vel[:,self.jmap]

    def _get_observations(self):
        pos,quat,lin,ang,joint,rate=self.state()
        look=torch.stack([self.ref('joint_pos_rad',i) for i in [0,5,15]],1)
        phase=2*torch.pi*(self.frame/(self.lengths[self.cid]-1)).clamp(0,1)
        obs=observe(pos,quat,lin,ang,joint,rate,self.previous,self.ref('root_pos_world_m'),self.ref('root_quat_world_wxyz'),look,self.ref('root_lin_vel_world_m_s'),phase)
        return {'policy':obs}

    def _pre_physics_step(self,actions):
        self.step_sat.zero_()
        self.actions=actions.clone().clamp(-1,1)
        self.old_actions=self.previous.clone();self.previous=self.actions.clone()
        self.target=(self.ref('joint_pos_rad')+self.scale*self.actions).maximum(self.limits[:,0]).minimum(self.limits[:,1])

    def _apply_action(self):
        d=self.robot.data;rate=d.joint_vel[:,self.jmap]
        raw,torque=torque_pd(d.joint_pos[:,self.jmap],rate,self.target,self.kp,self.kd,self.effort,self.velocity)
        saturated=(raw.abs()>=self.effort).sum(-1)
        self.sat+=saturated;self.step_sat+=saturated;self.torque_steps+=29
        self.robot.set_joint_effort_target(torque,joint_ids=self.jmap)
        pos,quat,*_=self.state();gravity=torch.zeros_like(pos);gravity[:,2]=-1
        gz=rotate(inv(quat),gravity)[:,2]
        gate=EVAL['dynamic_episode_gate']
        bad=(pos[:,2]<gate['min_base_height_m'])|(gz>-np.cos(np.deg2rad(gate['max_base_tilt_deg'])))
        self.bad_duration=torch.where(bad,self.bad_duration+1,0)
        self.forbidden|=torch.linalg.vector_norm(self.torso.data.force_matrix_w.flatten(1),dim=-1)>1.
        self.nonfinite|=~torch.isfinite(self.robot.data.root_link_state_w).all(-1)|~torch.isfinite(torque).all(-1)|~torch.isfinite(self.actions).all(-1)

    def _get_dones(self):
        self.frame+=1
        self.last_frame=self.frame.clone();self.last_step_sat=self.step_sat.clone()
        pos,quat,lin,ang,joint,rate=self.state()
        actual=self.robot.data.body_link_pos_w[:,self.bmap]-self.scene.env_origins[:,None]
        bodyref=self.ref('body_pos_world_m');rootref=self.ref('root_pos_world_m')
        qe=(joint-self.ref('joint_pos_rad')).square().mean(-1)
        be=((actual-pos[:,None])-(bodyref-rootref[:,None])).square().sum(-1).mean(-1)
        re=(pos-rootref).square().sum(-1)
        he=(actual[:,self.hands]-bodyref[:,self.hands]).square().sum(-1).mean(-1)
        self.sums+=torch.stack((qe,be,re,he),-1);self.samples+=1
        self.last_metrics={'joint':qe,'body':be,'root':re,'hand':he}
        self.last_qpos=torch.cat((pos,quat,joint),-1).detach().clone()
        failure=(self.bad_duration>=round(EVAL['dynamic_episode_gate']['fall_violation_persistence_s']/.002))|self.forbidden|self.nonfinite
        # Training-only early recovery resets; nominal evaluation never uses these tracking cutoffs.
        if not self.cfg.evaluation:failure|=(re>.5**2)|(be>.35**2)
        complete=self.frame>=self.lengths[self.cid]-1+50
        return failure,complete

    def _get_rewards(self):
        m=self.last_metrics
        tracking=2*torch.exp(-m['joint']/.15**2)+2*torch.exp(-m['body']/.08**2)+torch.exp(-m['root']/.15**2)+torch.exp(-m['hand']/.10**2)
        penalty=.03*(self.actions-self.old_actions).square().mean(-1)+.005*self.actions.square().mean(-1)
        reward=(tracking-penalty-2*self.reset_terminated.float())*.02
        self.extras['log']={'tracking/joint_rmse':m['joint'].mean().sqrt(),'tracking/body_rmse':m['body'].mean().sqrt(),'tracking/root_rmse':m['root'].mean().sqrt(),'failure/rate':self.reset_terminated.float().mean()}
        for name,cohort in self.motion_masks.items():
            mask=cohort[self.cid]
            self.extras['log']['tracking/'+name+'_root_rmse']=m['root'][mask].mean().sqrt() if mask.any() else torch.tensor(0.,device=self.device)
            self.extras['log']['failure/'+name]=self.reset_terminated[mask].float().mean() if mask.any() else torch.tensor(0.,device=self.device)
        return reward

    def _reset_idx(self,env_ids):
        if env_ids is None:env_ids=self.robot._ALL_INDICES
        # Save episode results before the framework resets the state.
        if hasattr(self,'samples') and self.cfg.evaluation:
            for i in env_ids[self.samples[env_ids]>0].tolist():
                rmse=(self.sums[i]/self.samples[i]).sqrt().tolist();frac=(self.sat[i]/self.torque_steps[i].clamp_min(1)).item()
                g=EVAL['dynamic_episode_gate'];full=self.frame[i]>=self.lengths[self.cid[i]]-1+50
                checks={'full_clip_plus_hold':bool(full),'finite':not bool(self.nonfinite[i]),'no_fall':bool(self.bad_duration[i]<50),'no_forbidden_contact':not bool(self.forbidden[i]),'joint':rmse[0]<=g['max_joint_position_rmse_rad'],'body':rmse[1]<=g['max_root_relative_body_position_rmse_m'],'root':rmse[2]<=g['max_world_root_position_rmse_m'],'hand':rmse[3]<=g['max_hand_position_rmse_m'],'saturation':frac<=g['max_torque_saturation_fraction']}
                if self.cfg.evaluation and self.eval_counts[i]==0:
                    self.finished.append({'env_id':i,'clip_id':self.clips[self.cid[i]]['clip_id'],'motion_id':self.clips[self.cid[i]]['motion_id'],'evaluation_seed':EVAL['evaluation_seeds'][i//len(self.clips)],'steps':int(self.samples[i]),'rmse_joint_rad':rmse[0],'rmse_body_m':rmse[1],'rmse_root_m':rmse[2],'rmse_hand_m':rmse[3],'torque_saturation_fraction':frac,'checks':checks,'success':all(checks.values()),'initial_state':self.initial_states[i]})
                    self.eval_counts[i]+=1
        super()._reset_idx(env_ids)
        if not hasattr(self,'cid'):return
        n=len(env_ids)
        if self.cfg.evaluation:self.cid[env_ids]=env_ids%len(self.clips)
        else:self.cid[env_ids]=torch.multinomial(self.clip_probabilities,n,replacement=True)
        self.frame[env_ids]=0
        if self.cfg.random_start and not self.cfg.evaluation:
            self.frame[env_ids]=(torch.rand(n,device=self.device)*(self.lengths[self.cid[env_ids]]-1)).long()
        root=torch.cat((self.ref('root_pos_world_m')[env_ids]+self.scene.env_origins[env_ids],self.ref('root_quat_world_wxyz')[env_ids]),-1)
        twist=torch.cat((self.ref('root_lin_vel_world_m_s')[env_ids],self.ref('root_ang_vel_world_rad_s')[env_ids]),-1)
        joint=self.ref('joint_pos_rad')[env_ids].clone();rate=self.ref('joint_vel_rad_s')[env_ids].clone()
        if not self.cfg.evaluation:
            joint+=.01*torch.randn_like(joint)
        else:
            # Each seed's small perturbation is saved explicitly for backend replay.
            for k,i in enumerate(env_ids.tolist()):
                gen=torch.Generator(device=self.device).manual_seed(EVAL['evaluation_seeds'][i//len(self.clips)])
                joint[k]+=.005*torch.randn((29,),device=self.device,generator=gen)
        self.robot.write_root_pose_to_sim(root,env_ids);self.robot.write_root_link_velocity_to_sim(twist,env_ids)
        jall=self.robot.data.default_joint_pos[env_ids].clone();vall=torch.zeros_like(jall)
        jall[:,self.jmap]=joint;vall[:,self.jmap]=rate
        self.robot.write_joint_state_to_sim(jall,vall,env_ids=env_ids)
        self.robot.reset(env_ids)
        self.previous[env_ids]=0;self.actions[env_ids]=0;self.sums[env_ids]=0;self.samples[env_ids]=0;self.sat[env_ids]=0;self.torque_steps[env_ids]=0
        self.bad_duration[env_ids]=0;self.forbidden[env_ids]=False;self.nonfinite[env_ids]=False
        if self.cfg.evaluation:
            for k,i in enumerate(env_ids.tolist()):
                if self.eval_counts[i]==0:self.initial_states[i]={'root_pose_local':torch.cat((root[k,:3]-self.scene.env_origins[i],root[k,3:])).tolist(),'root_twist_world':twist[k].tolist(),'joint_pos_source':joint[k].tolist(),'joint_vel_source':rate[k].tolist(),'previous_action':[0.]*29}
