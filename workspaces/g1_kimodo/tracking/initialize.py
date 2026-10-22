"""Train-only synthetic balance prior for the actor; inference remains a plain MLP."""
import json
import torch
from .core import observe,mul


def initialize_balance(actor,env,diagnostic,run):
    by_id={x['clip_id']:x for x in json.loads(diagnostic.read_text())['clips']}
    device=env.device;generator=torch.Generator(device=device).manual_seed(7341)
    n=12288
    cid=torch.arange(n,device=device)%len(env.clips)
    roots=env.refs['root_pos_world_m'][cid,0];quats=env.refs['root_quat_world_wxyz'][cid,0];qref=env.refs['joint_pos_rad'][cid,0]
    offsets=torch.tensor([by_id[c['clip_id']]['required_target_offset_rad'] for c in env.clips],device=device)
    mean_bias=(offsets/env.scale).mean(0)
    bias=mean_bias[None].repeat(n,1)
    def noise(shape,scale):return scale*torch.randn(shape,device=device,generator=generator)
    # Local quaternion perturbations cover the small-angle balance region.
    xyz=noise((n,3),.04);dq=torch.cat((torch.ones((n,1),device=device),xyz/2),-1);dq=dq/dq.norm(dim=-1,keepdim=True)
    quat=mul(quats,dq);pos=roots+noise((n,3),.025);lin=noise((n,3),.12);ang=noise((n,3),.2)
    joint=qref+noise((n,29),.025);rate=noise((n,29),.3);previous=2*torch.rand((n,29),device=device,generator=generator)-1
    previous[:n//8]=0
    look=qref[:,None].repeat(1,3,1);phase=torch.rand(n,device=device,generator=generator)*2*torch.pi
    obs=observe(pos,quat,lin,ang,joint,rate,previous,roots,quats,look,torch.zeros_like(lin),phase)
    target=bias.clone()
    pitch=-8*obs[:,100]+8*obs[:,4]+obs[:,0]-2*obs[:,96]
    roll=-8*obs[:,99]+8*obs[:,3]-obs[:,1]+2*obs[:,97]
    for name in ['left_ankle_pitch_joint','right_ankle_pitch_joint']:target[:,env.dataset_joint_names.index(name)]+=pitch
    for name in ['left_ankle_roll_joint','right_ankle_roll_joint']:target[:,env.dataset_joint_names.index(name)]+=roll
    target=target.clamp(-1,1)
    optimizer=torch.optim.Adam(actor.parameters(),lr=1e-3)
    # Only true teacher inputs influence the initial MLP; PPO can later learn all fields.
    active=torch.zeros(194,device=device,dtype=torch.bool)
    active[:6]=True;active[96:102]=True
    with torch.no_grad():actor[0].weight[:,~active]=0
    actor.train()
    for _ in range(500):
        ids=torch.randint(n-1024,(512,),device=device,generator=generator)
        loss=(actor(obs[ids])-target[ids]).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        with torch.no_grad():actor[0].weight[:,~active]=0
    actor.eval()
    with torch.no_grad():
        rmse=float((actor(obs[-1024:])-target[-1024:]).square().mean().sqrt())
    with torch.no_grad():
        rc=env.refs['root_pos_world_m'][:,0];qc=env.refs['root_quat_world_wxyz'][:,0];jc=env.refs['joint_pos_rad'][:,0]
        zeros=torch.zeros_like(rc)
        reset_obs=observe(rc,qc,zeros,zeros,jc,torch.zeros_like(jc),torch.zeros_like(jc),rc,qc,jc[:,None].repeat(1,3,1),zeros,torch.zeros(len(jc),device=device))
        reset_error=float((actor(reset_obs)-mean_bias).abs().max())
    result={'initial_input_mask':'root velocity, angular velocity, root position/orientation error only; first-layer other columns zeroed during initialization, trainable again in PPO','bias_policy':'mean of train clips; no clip identification shortcut','reset_zero_history_max_action_error':reset_error,'previous_action_sampling':'uniform independent of target, with explicit zeros', 'method':'synthetic local balance teacher initialization followed by PPO','training_clip_ids':[c['clip_id'] for c in env.clips],'synthetic_samples':n,'heldout_synthetic_rmse_action':rmse,'teacher':{'ankle_pitch':'bias -8*orientation_error_y +2*omega_y +vx -2*root_error_x','ankle_roll':'bias -8*orientation_error_x +2*omega_x -vy +2*root_error_y','bias':'per-train-clip vertical support wrench equilibrium torque / kp / residual scale','all_errors':'actual root body frame','action_clip':[-1,1]},'validation_used_for_teacher':False,'inference_architecture':'unchanged MLP; teacher is not called during rollout'}
    (run/'balance_initialization.json').write_text(json.dumps(result,indent=2))
    torch.save({'observations':obs[-256:].cpu(),'teacher_actions':target[-256:].cpu()},run/'synthetic_teacher_fixtures.pt')
    return result
