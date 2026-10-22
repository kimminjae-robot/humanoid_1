"""Independent 50Hz FK audit of posture changes and actual self contacts."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--original-reference', type=Path, required=True)
    a = p.parse_args()
    run = ROOT/'runs'/a.run
    result = json.loads((run/'result.json').read_text())
    m = mujoco.MjModel.from_xml_path(str(ROOT/'assets/k1_hx5.xml'))
    d = mujoco.MjData(m)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'torso_link')
    wrist = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'right_wrist_roll_rubber_hand')
    names = json.loads((ROOT/'reports/robot_manifest.json').read_text())['body_joint_names']
    adr = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in names]
    original = np.load(a.original_reference)['qpos']
    reference = np.load(ROOT/result['episodes'][0]['rollout_npz'])['reference']
    if original.shape != reference.shape:
        raise ValueError('This audit requires unchanged duration and fps')
    def pose(q):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        return d.xmat[torso].reshape(3,3).T @ (d.xpos[wrist]-d.xpos[torso])
    p0 = np.stack([pose(q) for q in original])
    p1 = np.stack([pose(q) for q in reference])
    episodes = []
    for e in result['episodes']:
        z = np.load(ROOT/e['rollout_npz'])
        error = []
        worst = 0.
        pair = None
        for q, i in zip(z['qpos'], z['frame_index']):
            error.append(float(np.linalg.norm(pose(q)-p1[i])))
            for c in d.contact:
                if floor not in [c.geom1, c.geom2] and c.dist < -worst:
                    worst = -float(c.dist)
                    pair = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
                            for g in [c.geom1, c.geom2]]
        episodes.append({'seed': e['seed'], 'actual_sampled_max_self_penetration_m': worst,
                         'deepest_contact_bodies': pair,
                         'right_wrist_torso_local_RMSE_m': float(np.sqrt(np.mean(np.square(error))))})
    report = {'scope': 'independent_FK_at_saved_50Hz; not_peak_contact_depth_at_500Hz',
              'max_root_translation_change_m': float(np.max(np.linalg.norm(reference[:,:3]-original[:,:3],axis=1))),
              'root_orientation_max_component_change': float(np.max(abs(reference[:,3:7]-original[:,3:7]))),
              'arm_joint_max_changes_rad': {n:float(abs(reference[:,i]-original[:,i]).max())
                                          for n,i in zip(names,adr) if any(k in n for k in ['shoulder','elbow','wrist'])},
              'right_wrist_torso_local_max_change_m': float(np.linalg.norm(p1-p0,axis=1).max()),
              'right_wrist_torso_local_max_change_hold_1_4_to_4_4_s_m': float(np.linalg.norm(p1[70:220]-p0[70:220],axis=1).max()),
              'episodes': episodes}
    (run/'fidelity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
