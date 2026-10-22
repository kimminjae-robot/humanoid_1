"""Project a standing reference onto fixed, flat feet and centered COM.

Keeps source root orientation and upper-body joints, apart from an explicit
optional left shoulder abduction offset. No dynamics or motor limits changed.
Standing is a requested conditioning mode, not an inferred universal solution.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-reference', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--height', type=float, default=.77)
    p.add_argument('--left-arm-offset', type=float, default=0.)
    p.add_argument('--time-scale', type=float, default=1.)
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.label) or a.time_scale < 1:
        raise ValueError('Invalid label or time scale')
    folder = ROOT / 'data' / 'conditioned' / a.label
    folder.mkdir(parents=True, exist_ok=False)
    source = np.load(a.source_reference, allow_pickle=False)
    if float(source['fps']) != 50:
        raise ValueError('Expected 50Hz K1 reference')
    original = source['qpos']
    m = mujoco.MjModel.from_xml_path(str(ROOT / 'assets/k1_hx5.xml'))
    d = mujoco.MjData(m)
    robot = json.loads((ROOT / 'reports/robot_manifest.json').read_text())
    names = robot['body_joint_names']
    jids = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names])
    qa = m.jnt_qposadr[jids]
    leg = np.array([i for i, n in enumerate(names) if any(k in n for k in ['hip', 'knee', 'ankle'])])
    feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, s + '_ankle_roll_link') for s in ['left', 'right']]
    q0 = np.load(ROOT / 'assets/neutral_qpos.npy')
    d.qpos[:] = q0
    mujoco.mj_forward(m, d)
    targets = d.xpos[feet].copy()
    rotations = d.xmat[feet].reshape(2, 3, 3).copy()
    for k, f in enumerate(feet):
        gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == f and m.geom_contype[g]
              and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
        targets[k, 2] = .001 - min((rotations[k] @ m.geom_pos[g])[2] - m.geom_size[g, 0] for g in gs)
    com = targets[:, :2].mean(0)
    low = np.r_[[-.2, -.2, a.height-.10], m.jnt_range[jids[leg], 0] + 1e-5]
    high = np.r_[[.2, .2, a.height+.10], m.jnt_range[jids[leg], 1] - 1e-5]
    previous = np.r_[q0[:3], q0[qa[leg]]]
    output = []
    diagnostics = []
    sample_frames = np.linspace(0, len(original)-1, round((len(original)-1)*a.time_scale)+1)
    # Quaternion interpolation is required even though standing heading varies little.
    from scipy.spatial.transform import Slerp
    quat = Slerp(np.arange(len(original)), Rotation.from_quat(original[:, [4, 5, 6, 3]]))(sample_frames).as_quat()
    resampled = np.stack([np.interp(sample_frames, np.arange(len(original)), original[:, j])
                          for j in range(m.nq)], axis=1)
    resampled[:, 3:7] = quat[:, [3, 0, 1, 2]]
    left_roll = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'left_shoulder_roll_joint')])
    for frame, src in enumerate(resampled):
        base = src.copy()
        base[left_roll] += a.left_arm_offset
        def setq(x):
            d.qpos[:] = base
            d.qpos[:3] = x[:3]
            d.qpos[qa[leg]] = x[3:]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
        def fun(x):
            setq(x)
            return np.r_[(d.xpos[feet] - targets).ravel()*100,
                         *[Rotation.from_matrix(rotations[k].T @ d.xmat[f].reshape(3, 3)).as_rotvec()*20
                           for k, f in enumerate(feet)],
                         (d.subtree_com[1, :2] - com)*100,
                         (d.qpos[2] - a.height)*10,
                         (x[3:] - q0[qa[leg]])*.04]
        sol = least_squares(fun, np.clip(previous, low+1e-7, high-1e-7), bounds=(low, high),
                            max_nfev=50, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        setq(sol.x)
        output.append(d.qpos.copy())
        previous = sol.x.copy()
        diagnostics.append({'frame': frame, 'solver_success': bool(sol.success),
                            'foot_position_error_m': float(np.max(np.linalg.norm(d.xpos[feet]-targets, axis=1))),
                            'COM_xy_error_m': float(np.linalg.norm(d.subtree_com[1, :2]-com))})
    ref = np.stack(output)
    np.savez_compressed(folder / 'reference.npz', qpos=ref, fps=np.array(50), joint_names=np.array(names),
                        source_foot_contacts=np.ones((len(ref), 2), bool), planned_foot_positions=targets,
                        planned_COM_xy=com, source_frames=sample_frames)
    np.savetxt(folder / 'qpos.csv', ref, delimiter=',')
    upper = np.array([i for i in range(len(names)) if i not in leg])
    record = {'clip_id': a.label, 'motion_id': 'custom', 'reference_npz': str((folder/'reference.npz').relative_to(ROOT)),
              'frames': len(ref), 'fps': 50, 'accepted_for_tracking': False,
              'source_reference': str(a.source_reference.resolve()), 'source_reference_sha256': sha(a.source_reference),
              'model_sha256': sha(ROOT/'assets/k1_hx5.xml'), 'method': 'fixed_flat_feet_COM_IK_preserve_upper_body',
              'settings': vars(a) | {'source_reference': str(a.source_reference)},
              'max_sampled_foot_position_error_m': max(x['foot_position_error_m'] for x in diagnostics),
              'max_COM_xy_error_m': max(x['COM_xy_error_m'] for x in diagnostics),
              'max_upper_joint_change_rad': float(np.max(abs(ref[:, qa[upper]]-resampled[:, qa[upper]]))),
              'max_root_translation_change_m': float(np.max(np.linalg.norm(ref[:, :3]-resampled[:, :3], axis=1))),
              'diagnostics': diagnostics}
    (folder/'manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    (ROOT/'reports'/('dataset_'+a.label+'.json')).write_text(json.dumps({'clips': [{k:v for k,v in record.items() if k!='diagnostics'}]}, indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='diagnostics'}, indent=2))


if __name__ == '__main__':
    main()
