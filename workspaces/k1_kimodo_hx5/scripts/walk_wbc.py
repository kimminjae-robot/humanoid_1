"""Centroidal contact-wrench QP with independent torque-bounded body PD; only real joint motors actuated.
63-DoF model mass includes hands. Fingers retain official position actuators;
the QP assumes small finger accelerations and controls root+23 body coordinates.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.deps'))
import osqp
import numpy as np,mujoco
from scipy import sparse
from scipy.spatial.transform import Rotation

class WalkingWBC:
    def __init__(self,m,ref,schedule,qa,va,effort):
        self.m=m;self.ref=ref;self.schedule=schedule;self.qa=qa;self.va=va;self.sel=np.r_[np.arange(6),va];self.effort=effort;self.feet=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,s+'_ankle_roll_link') for s in ['left','right']]
        self.vel=np.zeros((len(ref),m.nv));self.acc=np.zeros_like(self.vel)
        for i in range(len(ref)-1):mujoco.mj_differentiatePos(m,self.vel[i],.02,ref[i],ref[i+1])
        self.vel[1:-1]=(self.vel[:-2]+self.vel[1:-1])/2
        self.acc[1:-1]=(self.vel[2:]-self.vel[:-2])/.04
        rd=mujoco.MjData(m);positions=[];rotations=[]
        for q in ref:
            rd.qpos[:]=q;mujoco.mj_kinematics(m,rd);positions.append(rd.xpos[self.feet].copy());rotations.append(rd.xmat[self.feet].reshape(2,3,3).copy())
        self.fp=np.stack(positions);self.fr=np.stack(rotations);self.fv=np.gradient(self.fp,.02,axis=0);self.fa=np.gradient(self.fv,.02,axis=0)
        self.calls=0;self.max_primal_residual=0;self.max_base_residual=0;self.max_iterations=0;self.failures=0;self.last=None
    def compute(self,d,index):
        m=self.m;i=min(index,len(self.ref)-1);active=self.schedule[i];sel=self.sel;n=len(sel);nv=n+12
        mujoco.mj_forward(m,d);full=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,full,d.qM);M=full[np.ix_(sel,sel)];bias=d.qfrc_bias[sel]-d.qfrc_passive[sel]
        state_error=np.zeros(m.nv);mujoco.mj_differentiatePos(m,state_error,1.,d.qpos,self.ref[i]);v=d.qvel[sel];rv=self.vel[i,sel];ra=self.acc[i,sel]
        desired=ra.copy();desired[:3]+=np.array([45.,45.,100.])*state_error[:3]+np.array([13.,13.,20.])*(rv[:3]-v[:3]);desired[3:6]+=100*state_error[3:6]+20*(rv[3:6]-v[3:6])
        Jall=[]
        for foot in self.feet:
            jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jac(m,d,jp,jr,d.xpos[foot],foot);Jall.append(np.r_[jp[:,sel],jr[:,sel]])
        JT=np.concatenate([J.T for J in Jall],axis=1);desired_wrench=M[:6]@desired+bias[:6]
        # Centroidal wrench feedback and motor PD: the ground-force QP never
        # imposes a fictitious fixed-foot acceleration on the soft contact solver.
        mg=float(m.body_mass.sum()*9.81);scales=np.tile([mg,mg,mg,mg*.1,mg*.1,mg*.1],2);S=np.diag(scales)
        weight=np.diag([1.,1.,1.,10.,10.,10.]);B=weight@JT[:6]@S/mg;goal=weight@desired_wrench/mg
        P=B.T@B+np.eye(12)*1e-3;q=-B.T@goal
        A=[];lo=[];hi=[]
        def constraint(mat,l,u):A.append(np.atleast_2d(mat));lo.extend(np.atleast_1d(l));hi.extend(np.atleast_1d(u))
        for side in range(2):
            unit=np.zeros((6,12));unit[:,6*side:6*side+6]=np.eye(6)
            if not active[side]:constraint(unit,np.zeros(6),np.zeros(6));continue
            constraint(unit[2],0,1000)
            for axis in [0,1]:
                for sign in [-1,1]:constraint(sign*unit[axis]-.6*unit[2],-np.inf,0)
            for sign in [-1,1]:
                constraint(sign*(unit[3]-.066*unit[1])-.04*unit[2],-np.inf,0)
                constraint(sign*(unit[4]+.066*unit[0])-.085*unit[2],-np.inf,0)
                constraint(sign*unit[5]-.015*unit[2],-np.inf,0)
        solver=osqp.OSQP();solver.setup(P=sparse.csc_matrix(np.triu(P)),q=q,A=sparse.csc_matrix(np.vstack(A)@S/mg),l=np.asarray(lo)/mg,u=np.asarray(hi)/mg,verbose=False,eps_abs=1e-5,eps_rel=1e-5,max_iter=10000,polishing=True)
        if self.last is not None:solver.warm_start(x=self.last)
        result=solver.solve(raise_error=False);self.calls+=1
        if result.info.status_val not in [1,2]:self.failures+=1;raise RuntimeError(f'Centroidal QP {result.info.status}; frame={i}')
        forces=S@result.x;self.last=result.x.copy();tau=M[6:]@ra+bias[6:]-JT[6:]@forces
        self.max_base_residual=max(self.max_base_residual,float(np.max(np.abs(JT[:6]@forces-desired_wrench))));self.max_primal_residual=max(self.max_primal_residual,float(result.info.prim_res));self.max_iterations=max(self.max_iterations,int(result.info.iter))
        return tau
    def diagnostics(self):return {'QP_calls':self.calls,'failures':self.failures,'max_desired_base_wrench_residual':self.max_base_residual,'max_primal_residual':self.max_primal_residual,'max_iterations':self.max_iterations,'assumption':'finger_accelerations_neglected_in_reduced_dynamics','root_actuation':False}
