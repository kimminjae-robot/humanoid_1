"""Independent frame and actuator edge-case checks before policy deployment."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tracking.core import rotate,inv,mul,observe,pd

def main():
    rng=np.random.default_rng(42);rot=Rotation.random(20,random_state=rng);q=torch.tensor(rot.as_quat()[:,[3,0,1,2]],dtype=torch.float64);v=torch.tensor(rng.normal(size=(20,3)),dtype=torch.float64)
    checks={'rotation_matches_scipy':bool(np.allclose(rotate(q,v),rot.apply(v.numpy()),atol=1e-12)), 'inverse_rotation_matches_scipy':bool(np.allclose(rotate(inv(q),v),rot.inv().apply(v.numpy()),atol=1e-12)), 'quaternion_product_order':bool(np.allclose(mul(inv(q),q),np.tile([1,0,0,0],(20,1)),atol=1e-12))}
    quat=torch.tensor([[1.,0.,0.,0.]]);pos=torch.zeros((1,3));joint=torch.zeros((1,29));look=torch.zeros((1,3,29));phase=torch.zeros(1)
    obs=observe(pos,quat,pos,pos,joint,joint,joint,pos,quat,look,pos,phase)
    checks['observation_layout_and_reset_previous_action']=obs.shape==(1,194) and bool((obs[:,67:96]==0).all()) and obs[0,8]==-1
    checks['quaternion_sign_invariance']=bool(torch.allclose(obs,observe(pos,-quat,pos,pos,joint,joint,joint,pos,quat,look,pos,phase)))
    args=[torch.tensor([[0.,0.]]),torch.tensor([[0.,4.]]),torch.tensor([[10.,10.]]),torch.tensor([[2.,2.]]),torch.tensor([10.,10.]),torch.tensor([1.,1.]),torch.tensor([.25,.25]),torch.tensor([[-1.,1.],[-1.,1.]]),torch.tensor([3.,3.]),torch.tensor([2.,2.])]
    target,raw,torque=pd(*args)
    checks['target_and_effort_clipping']=bool(torch.allclose(target,torch.ones_like(target))) and raw[0,0]==10 and torque[0,0]==3
    checks['outward_torque_suppressed_above_rated_speed']=torque[0,1]==0
    args[2]=torch.tensor([[-10.,-10.]])
    checks['braking_torque_preserved_above_rated_speed']=pd(*args)[2][0,1]==-3
    checks={k:bool(v) for k,v in checks.items()};result={'checks':checks,'passed':all(checks.values())}
    (ROOT/'reports/tracking_math_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
