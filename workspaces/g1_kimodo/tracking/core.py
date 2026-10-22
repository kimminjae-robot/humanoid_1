"""Simulator-independent torch observation/PD contract shared with deployment."""
import torch


def mul(a,b):
    aw,ax,ay,az=a.unbind(-1);bw,bx,by,bz=b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz,aw*bx+ax*bw+ay*bz-az*by,aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw),-1)


def inv(q):
    return q*torch.tensor([1.,-1.,-1.,-1.],device=q.device)


def rotate(q,v):
    t=2*torch.cross(q[...,1:],v,dim=-1)
    return v+q[...,:1]*t+torch.cross(q[...,1:],t,dim=-1)


def observe(pos,quat,lin,ang,joint,rate,previous,ref_pos,ref_quat,ref_joints,ref_lin,phase):
    qi=inv(quat);delta=mul(qi,ref_quat);delta=delta*torch.where(delta[:,:1]<0,-1.,1.)
    gravity=torch.zeros_like(pos);gravity[:,2]=-1
    return torch.cat((rotate(qi,lin),.25*rotate(qi,ang),rotate(qi,gravity),joint-ref_joints[:,0],.05*rate,
        previous,rotate(qi,ref_pos-pos),2*delta[:,1:],ref_joints.flatten(1),rotate(qi,ref_lin),torch.sin(phase)[:,None],torch.cos(phase)[:,None]),-1).clamp(-20,20)


def pd(joint,rate,reference,action,kp,kd,scale,limits,effort,velocity):
    target=(reference+scale*action.clamp(-1,1)).maximum(limits[:,0]).minimum(limits[:,1])
    raw,torque=torque_pd(joint,rate,target,kp,kd,effort,velocity)
    return target,raw,torque


def torque_pd(joint,rate,target,kp,kd,effort,velocity):
    raw=kp*(target-joint)-kd*rate
    torque=raw.maximum(-effort).minimum(effort)
    torque=torch.where((rate.abs()>velocity)&(torque*rate>0),0.,torque)
    return raw,torque
