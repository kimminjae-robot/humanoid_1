"""Required ground wrench/COP of reference accelerations; diagnostic, not a gate change."""
import json,sys
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from reference_core import derivative,geometry_bottoms,foot_geoms,static_margin,rotations

def main():
    dataset=json.loads((ROOT/json.loads((ROOT/'state/progress.json').read_text())['artifacts_stage_3']['dataset_index']).read_text())
    m=mujoco.MjModel.from_xml_path(str(ROOT/'assets/g1/g1_scene.xml'));data=mujoco.MjData(m);feet=foot_geoms(m);clips=[]
    cfg=json.loads((ROOT/'configs/reference_processing.json').read_text())
    for clip in dataset['clips']:
        if not clip['accepted'] or clip['motion_id']!='walk_slow':continue
        with np.load(ROOT/clip['bundle'],allow_pickle=False) as b:
            q=np.c_[b['root_pos_world_m'],b['root_quat_world_wxyz'],b['joint_pos_rad']];v=b['qvel_mujoco'];acc=derivative(v,50)
        margins=[];cop=[];friction=[];fz=[];max_acc=np.abs(acc[:,6:]).max()
        for t,state in enumerate(q):
            data.qpos[:]=state;data.qvel[:]=v[t];data.qacc[:]=acc[t]
            # RNE required applied force before subtracting passive/constraint terms.
            mujoco.mj_forward(m,data);data.qacc[:]=acc[t]
            force=np.zeros(m.nv);mujoco.mj_rne(m,data,1,force)
            f=force[:3];moment=rotations(state[None,3:7]).apply(force[None,3:6])[0]
            xy=np.array([state[0]-(moment[1]+state[2]*f[0])/f[2],state[1]+(moment[0]-state[2]*f[1])/f[2]])
            points=[]
            for indices in feet:
                bottom,p=geometry_bottoms(m,data,np.array(indices));selected=bottom<=bottom.min()+cfg['heel_toe_active_height_band_m'];active=p[selected].reshape(-1,3);active=active[active[:,2]<=bottom.min()+cfg['heel_toe_active_height_band_m']]
                if bottom.min()<=cfg['geometry_contact_height_m']:points.extend(active)
            margins.append(static_margin(xy,points) if points and f[2]>0 else -1.);cop.append(xy.tolist());friction.append(float(np.linalg.norm(f[:2])/max(f[2],1e-6)));fz.append(float(f[2]))
        arrays={'required_cop_xy_world_m':cop,'support_margin_m':margins,'required_friction_ratio':friction,'vertical_force_n':fz}
        clips.append({'clip_id':clip['clip_id'],'split':clip['split'],'frames':len(q),'required_cop_outside_support_fraction':float((np.array(margins)<-.005).mean()),'min_required_cop_margin_m':float(min(margins)),'max_required_friction_ratio':float(max(friction)),'negative_vertical_force_frames':sum(x<=0 for x in fz),'max_joint_acceleration_rad_s2':float(max_acc),'arrays':arrays})
    summary={'scope':'exact_reference_root_wrench_and_COP_diagnostic_from_50Hz_finite_difference_accelerations; not proof of infeasibility within tracking-error allowances; no success threshold changed','clips':clips}
    (ROOT/'reports/stage_04_reference_wrench_diagnostic.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps([{k:v for k,v in c.items() if k!='arrays'} for c in clips],indent=2))
if __name__=='__main__':main()
