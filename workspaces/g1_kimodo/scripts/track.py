"""Train/evaluate G1 in Isaac Lab; emit complete artifacts before bounded teardown."""
import argparse,json,sys,hashlib,shutil,time,traceback,copy
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['train','evaluate','zero','fixture','teacher'],default='train')
p.add_argument('--motions',nargs='+',default=['stand'],choices=['stand','wave','walk_slow'])
p.add_argument('--num-envs',type=int,default=32)
p.add_argument('--walking-weight',type=float,default=1.)
p.add_argument('--iterations',type=int,default=5)
p.add_argument('--checkpoint',type=Path)
p.add_argument('--base-policy',type=Path)
p.add_argument('--contract',type=Path)
p.add_argument('--support-bias-init',action='store_true')
p.add_argument('--balance-init',action='store_true')
p.add_argument('--freeze-balance-inputs',action='store_true')
p.add_argument('--noise-std',type=float,default=.3)
p.add_argument('--run-dir',type=Path,required=True)
p.add_argument('--random-start',action=argparse.BooleanOptionalAction,default=True)
p.add_argument('--learning-rate',type=float,default=3e-4)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args();app=AppLauncher(a).app
import numpy as np
import torch
from tracking.env import TrackingCfg,TrackingEnv,EVAL
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper,RslRlOnPolicyRunnerCfg,RslRlPpoActorCriticCfg,RslRlPpoAlgorithmCfg,export_policy_as_jit
from rsl_rl.runners import OnPolicyRunner


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started=time.monotonic();run=a.run_dir.resolve();run.mkdir(parents=True,exist_ok=True)
    provenance=run/'provenance';provenance.mkdir(exist_ok=True)
    for name in ['tracking/core.py','tracking/env.py','scripts/track.py','configs/motion_contract.json','configs/tracking_contract.json','configs/evaluation.json']:
        shutil.copy2(ROOT/name,provenance/name.replace('/','__'))
    source_checkpoint=a.checkpoint or a.base_policy
    contract_path=a.contract or ((source_checkpoint.resolve().parent/'provenance/configs__tracking_contract.json') if source_checkpoint else ROOT/'configs/tracking_contract.json')
    selected_contract=json.loads(contract_path.read_text())
    selected_contract['runtime'].update(ground={'shape':'static_box','size_m':[200.,200.,.2],'center_z_m':-.1,'surface_z_m':0.,'static_friction':.8,'dynamic_friction':.8,'restitution':0.,'combine_mode':'average'},contact={'offset_m':.002,'rest_offset_m':0.,'forbidden_torso_ground_force_threshold_n':1.},solver={'type':'TGS','position_iterations':16,'velocity_iterations':4})
    (provenance/'configs__tracking_contract.json').write_text(json.dumps(selected_contract,indent=2)+'\n')
    cfg=TrackingCfg();cfg.walking_sample_weight=a.walking_weight;cfg.controller=selected_contract['controller'];cfg.sim.device=a.device;cfg.scene.num_envs=a.num_envs;cfg.seed=42;cfg.motions=tuple(a.motions);cfg.random_start=a.random_start;cfg.evaluation=a.mode in ['zero','evaluate','teacher']
    if cfg.evaluation:
        data=json.loads((ROOT/json.loads((ROOT/'state/progress.json').read_text())['artifacts_stage_3']['dataset_index']).read_text())
        count=sum(c['accepted'] and c['motion_id'] in a.motions and c['split']=='validation' for c in data['clips'])
        cfg.scene.num_envs=count*EVAL['episodes_per_clip']
    raw=TrackingEnv(cfg);env=RslRlVecEnvWrapper(raw,clip_actions=1.)
    if a.mode=='fixture':
        # Independent contact fixture; deliberately horizontal free base, no policy episode.
        ids=torch.arange(raw.num_envs,device=raw.device)
        pose=raw.robot.data.root_link_pose_w.clone()
        pose[:,:3]=raw.scene.env_origins+torch.tensor([0.,0.,.15],device=raw.device)
        pose[:,3:]=torch.tensor([2**-.5,0.,2**-.5,0.],device=raw.device)
        raw.robot.write_root_pose_to_sim(pose,ids)
        raw.robot.write_root_link_velocity_to_sim(torch.zeros((raw.num_envs,6),device=raw.device),ids)
        raw.target=raw.ref('joint_pos_rad').clone();held=raw.target.clone();force=0.;finite=True
        for _ in range(250):
            raw._apply_action();raw.scene.write_data_to_sim();raw.sim.step(render=False);raw.scene.update(.002)
            force=max(force,float(torch.linalg.vector_norm(raw.torso.data.force_matrix_w.flatten(1),dim=-1).max()))
            finite &= bool(torch.isfinite(raw.robot.data.root_link_state_w).all())
        checks={'ground_filtered_torso_contact_detected':force>1.,'target_held_over_physics_steps':bool(torch.equal(held,raw.target)),'finite_physics':finite}
        result={'scope':'independent_horizontal_drop_contact_and_target_hold_fixture_not_policy_validation','checks':checks,'passed':all(checks.values()),'max_ground_filtered_torso_force_n':force,'physics_steps':250}
        (run/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('TRACK_FIXTURE',json.dumps(result),flush=True);env.close();return
    torch.set_num_threads(4)
    agent=RslRlOnPolicyRunnerCfg(num_steps_per_env=32,max_iterations=a.iterations,save_interval=50,experiment_name='g1_kimodo_tracking',obs_groups={'policy':['policy'],'critic':['policy']},
      policy=RslRlPpoActorCriticCfg(init_noise_std=a.noise_std,noise_std_type='log',actor_hidden_dims=[256,128,128],critic_hidden_dims=[256,128,128],activation='elu',actor_obs_normalization=False,critic_obs_normalization=False),
      algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.,use_clipped_value_loss=True,clip_param=.2,entropy_coef=.002,learning_rate=a.learning_rate,num_learning_epochs=5,num_mini_batches=4,schedule='adaptive',gamma=.99,lam=.95,desired_kl=.01,max_grad_norm=1.))
    agent_config=agent.to_dict()
    base_policy=a.base_policy
    if a.checkpoint:
        previous_config=json.loads((a.checkpoint.resolve().parent/'resolved.json').read_text())
        if previous_config['agent']['policy']['class_name']=='TrackingActorCritic':
            base_policy=Path(previous_config['agent']['policy']['base_checkpoint'])
            if not base_policy.exists():
                parts=base_policy.parts; marker=parts.index('g1_kimodo'); base_policy=ROOT.joinpath(*parts[marker+1:])
    if base_policy:
        from tracking.policy import TrackingActorCritic
        import rsl_rl.runners.on_policy_runner as runner_module
        runner_module.TrackingActorCritic=TrackingActorCritic
        agent_config['policy'].update(class_name='TrackingActorCritic',base_checkpoint=str(base_policy.resolve()))
        shutil.copy2(ROOT/'tracking/policy.py',provenance/'tracking__policy.py')
    (run/'resolved.json').write_text(json.dumps({'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'agent':copy.deepcopy(agent_config),'clips':[c['clip_id'] for c in raw.clips],'joint_names_actual':raw.robot.joint_names,'source_to_isaac':raw.jmap.tolist(),'dataset_run_id':raw.dataset['run_id'],'source_checkpoint_sha256':sha(source_checkpoint) if source_checkpoint else 'none','timing':{'physics_dt':.002,'control_dt':.02},'gravity':True,'self_collision':True},indent=2))
    runner=OnPolicyRunner(env,agent_config,log_dir=str(run),device=a.device)
    runner.git_status_repos=[] # workspace provenance replaces unrelated checkout diffs
    if a.checkpoint:
        runner.load(str(a.checkpoint));runner.current_learning_iteration+=1
    if a.support_bias_init:
        if a.checkpoint or a.mode!='train' or a.motions!=['stand']:
            raise ValueError('Support bias initialization requires a new stand-only training run')
        diagnostic=ROOT/selected_contract.get('quasistatic_initializer_diagnostic','reports/stage_04_static_support_diagnostic.json')
        entries=json.loads(diagnostic.read_text())['clips']
        training_ids={c['clip_id'] for c in raw.clips}
        offsets=[x['required_target_offset_rad'] for x in entries if x['clip_id'] in training_ids and x['converged']]
        if len(offsets)!=len(training_ids):raise ValueError('Missing converged train-only support solution')
        bias=torch.tensor(np.mean(offsets,axis=0),device=a.device)/raw.scale
        with torch.no_grad():
            runner.alg.policy.actor[-1].weight.mul_(.01)
            runner.alg.policy.actor[-1].bias.copy_(bias.clamp(-.95,.95))
        shutil.copy2(diagnostic,provenance/diagnostic.name)
        (run/'support_bias_initialization.json').write_text(json.dumps({'source':'quasistatic vertical support optimization; train clips only','training_ids':sorted(training_ids),'action_bias':bias.tolist(),'diagnostic_sha256':sha(diagnostic)},indent=2))
    if a.balance_init:
        if a.support_bias_init or a.checkpoint or a.mode!='train' or a.motions!=['stand']:
            raise ValueError('Balance teacher initialization requires a new stand-only training run')
        from tracking.initialize import initialize_balance
        diagnostic=ROOT/selected_contract.get('quasistatic_initializer_diagnostic','reports/stage_04_static_support_diagnostic.json')
        initialize_balance(runner.alg.policy.actor,raw,diagnostic,run)
        shutil.copy2(ROOT/'tracking/initialize.py',provenance/'tracking__initialize.py')
        shutil.copy2(diagnostic,provenance/diagnostic.name)
    if a.freeze_balance_inputs:
        if a.mode!='train' or a.motions!=['stand']:
            raise ValueError('Restricted balance inputs are for stand-only training')
        mask=torch.zeros(194,device=raw.device);mask[:6]=1;mask[96:102]=1
        with torch.no_grad():runner.alg.policy.actor[0].weight.mul_(mask[None])
        runner.alg.policy.actor[0].weight.register_hook(lambda grad:grad*mask[None])
        (run/'frozen_input_columns.json').write_text(json.dumps({'active_columns':mask.nonzero().flatten().tolist(),'scope':'actor first-layer input columns constrained during stand PPO only; critic remains full observation'},indent=2))
    initial=torch.cat([v.detach().flatten().cpu() for v in runner.alg.policy.actor.parameters()])
    export=None
    if a.mode=='train':
        runner.logger_type='tensorboard'
        runner.save(str(run/'model_initialized.pt'))
        runner.learn(a.iterations,init_at_random_ep_len=False)
        final=torch.cat([v.detach().flatten().cpu() for v in runner.alg.policy.actor.parameters()])
        result={'scope':'PPO_training_not_validation_success','iterations_run':a.iterations,'last_iteration':runner.current_learning_iteration,'parameters_finite':bool(torch.isfinite(final).all()),'actor_parameter_max_change':float((final-initial).abs().max()),'checkpoints':[str(x.relative_to(ROOT)) for x in run.glob('model_*.pt')],'observation_dim':env.get_observations()['policy'].shape[-1],'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'elapsed_s':time.monotonic()-started}
    else:
        policy=runner.get_inference_policy(device=a.device)
        if a.mode=='teacher':
            diagnostics=json.loads((ROOT/selected_contract.get('quasistatic_initializer_diagnostic','reports/stage_04_static_support_diagnostic.json')).read_text())['clips']
            train_ids={c['clip_id'] for c in raw.dataset['clips'] if c['accepted'] and c['split']=='train' and c['motion_id'] in a.motions}
            teacher_bias=torch.tensor(np.mean([c['required_target_offset_rad'] for c in diagnostics if c['clip_id'] in train_ids],axis=0),device=raw.device,dtype=torch.float32)/raw.scale
        qs=[];ids=[];saturations=[];observations=[]
        with torch.inference_mode():
            for step in range(int(raw.lengths.max())+55):
                obs=env.get_observations()
                action=torch.zeros((raw.num_envs,29),device=raw.device) if a.mode=='zero' else policy(obs)
                if a.mode=='teacher':
                    x=obs['policy'];action=teacher_bias[None].repeat(raw.num_envs,1)
                    action[:,[4,10]]+=(-8*x[:,100]+8*x[:,4]+x[:,0]-2*x[:,96])[:,None]
                    action[:,[5,11]]+=(-8*x[:,99]+8*x[:,3]-x[:,1]+2*x[:,97])[:,None]
                    action=action.clamp(-1,1)
                observations.append(obs['policy'][:len(raw.clips)].detach().cpu());env.step(action)
                qs.append(raw.last_qpos.cpu().numpy());saturations.append(raw.last_step_sat.cpu().numpy())
                ids.append(raw.last_frame.cpu().numpy())
                if len(raw.finished)==raw.num_envs:break
        np.savez_compressed(run/'rollout.npz',qpos=np.stack(qs),frame_indices=np.stack(ids),saturated_joint_physics_steps=np.stack(saturations),env_clip_ids=np.array([raw.clips[i%len(raw.clips)]['clip_id'] for i in range(raw.num_envs)]),clip_ids=np.array([c['clip_id'] for c in raw.clips]))
        groups={m:[x for x in raw.finished if x['motion_id']==m] for m in a.motions}
        rates={m:sum(e['success'] for e in episodes)/max(1,len(episodes)) for m,episodes in groups.items()}
        export=None
        if a.mode=='evaluate':
            out=run/'export';export_policy_as_jit(runner.alg.policy,None,str(out))
            exported=torch.jit.load(str(out/'policy.pt'),map_location=a.device)
            fixtures=torch.cat(observations).to(a.device)
            expected=runner.alg.policy.actor(fixtures);actual=exported(fixtures)
            gpu_err=float((expected-actual).abs().max())
            cpu_exported=torch.jit.load(str(out/'policy.pt'),map_location='cpu')
            cpu_err=float((cpu_exported(fixtures.cpu())-expected.detach().cpu()).abs().max())
            err=max(gpu_err,cpu_err);export={'path':str((out/'policy.pt').relative_to(ROOT)),'sha256':sha(out/'policy.pt'),'fixture_count':len(fixtures),'gpu_action_error':gpu_err,'cpu_vs_training_gpu_action_error':cpu_err,'max_action_absolute_error':err,'passed':err<=1e-5}
            np.savez_compressed(out/'observation_action_fixtures.npz',observation=fixtures.cpu().numpy(),action=expected.detach().cpu().numpy())
            contract=copy.deepcopy(selected_contract);contract.update(motion_ids=a.motions,scope='Isaac validation for listed motion_ids; MuJoCo transfer pending',policy=export,checkpoint_sha256=sha(a.checkpoint),validation_success_rates=rates,deployable=export['passed'] and all(v>=.9 for v in rates.values()))
            contract['status']='Isaac_nominal_pass_for_listed_motions' if contract['deployable'] else 'Isaac_nominal_failed_for_listed_motions'
            contract['controller']['status']='stage4_Isaac_validated_MuJoCo_pending' if contract['deployable'] else 'stage4_dynamic_validation_failed'
            (out/'contract.json').write_text(json.dumps(contract,indent=2))
            # Every new evaluation export can be consumed directly by sim2sim.
            (out/'checksums.json').write_text(json.dumps({name:sha(out/name) for name in ['policy.pt','contract.json','observation_action_fixtures.npz']},indent=2)+'\n')
        result={'scope':'Isaac_free_base_dynamic_validation_full_clip_plus_1s_hold','controller_mode':a.mode,'motion_ids':a.motions,'episodes_expected':raw.num_envs,'episodes_measured':len(raw.finished),'success_rates':rates,'passed':len(raw.finished)==raw.num_envs and all(v>=.9 for v in rates.values()),'episodes':raw.finished,'export':export,'elapsed_s':time.monotonic()-started}
    (run/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('TRACK_RESULT',json.dumps({k:v for k,v in result.items() if k!='episodes'}),flush=True)
    env.close()


if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc();raise
    finally:app.close()
