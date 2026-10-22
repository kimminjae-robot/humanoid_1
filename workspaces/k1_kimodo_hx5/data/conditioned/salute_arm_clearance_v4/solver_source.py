"""Smallest-cost arm IK corrections against actual MuJoCo collision geometry.

Works on an existing K1 reference, uses no motion name or hand teleportation.
Finger joints remain constant. Every candidate must still pass full QA/dynamics.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-reference', required=True, type=Path)
    ap.add_argument('--label', required=True)
    ap.add_argument('--margin', type=float, default=.003)
    ap.add_argument('--max-correction', type=float)
    ap.add_argument('--speed-ratio', type=float)
    ap.add_argument('--initial-reference', type=Path)
    a = ap.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.label):
        raise ValueError('Invalid label')
    folder = ROOT/'data/conditioned'/a.label
    folder.mkdir(parents=True, exist_ok=False)
    source = np.load(a.source_reference, allow_pickle=False)
    ref = source['qpos'].copy()
    original = ref.copy()
    if a.initial_reference:
        ref = np.load(a.initial_reference, allow_pickle=False)['qpos'].copy()
        if ref.shape != original.shape:
            raise ValueError('Initial reference must match original shape')
    m = mujoco.MjModel.from_xml_path(str(ROOT/'assets/k1_hx5.xml'))
    d = mujoco.MjData(m)
    names = json.loads((ROOT/'reports/robot_manifest.json').read_text())['body_joint_names']
    armnames = [n for n in names if any(k in n for k in ['shoulder', 'elbow', 'wrist'])]
    jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in armnames]
    adr = m.jnt_qposadr[jids]
    low, high = m.jnt_range[jids].T
    parameters = json.loads((ROOT/'reports/robot_manifest.json').read_text())['body_actuator_parameters']
    step = np.array([parameters[n]['speed'] for n in armnames])*.02
    bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, s+'_'+n)
              for s in ['left', 'right'] for n in ['elbow_link', 'wrist_roll_rubber_hand']]
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    pairs = set()
    def scan():
        depth = 0.
        for q in ref:
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            for c in d.contact:
                if floor in [c.geom1, c.geom2]:
                    continue
                if c.dist < a.margin:
                    pairs.add(tuple(sorted([int(c.geom1), int(c.geom2)])))
                depth = max(depth, -float(c.dist))
        return depth
    initial = scan()
    history = []
    for iteration in range(4):
        active = sorted(pairs)
        previous = ref[0, adr].copy()
        for frame, src in enumerate(original):
            d.qpos[:] = src
            mujoco.mj_kinematics(m, d)
            targets = d.xpos[bodies].copy()
            base = ref[frame].copy()
            def fun(x):
                d.qpos[:] = base
                d.qpos[adr] = x
                mujoco.mj_kinematics(m, d)
                distances = np.array([mujoco.mj_geomDistance(m, d, g1, g2, .03, None) for g1, g2 in active])
                return np.r_[(x-src[adr])*.15,
                             (d.xpos[bodies]-targets).ravel()*3,
                             (x-previous)*.08,
                             np.maximum(a.margin-distances, 0)*200]
            lo, hi = low.copy()+1e-7, high.copy()-1e-7
            if a.max_correction is not None:
                lo = np.maximum(lo, src[adr]-a.max_correction)
                hi = np.minimum(hi, src[adr]+a.max_correction)
            if a.speed_ratio is not None and frame:
                lo = np.maximum(lo, previous-step*a.speed_ratio)
                hi = np.minimum(hi, previous+step*a.speed_ratio)
            if np.any(lo >= hi):
                raise RuntimeError(f'Incompatible pose/speed bounds at frame {frame}')
            sol = least_squares(fun, np.clip(ref[frame, adr], lo+1e-8, hi-1e-8),
                                bounds=(lo, hi), max_nfev=60,
                                ftol=1e-7, xtol=1e-7, gtol=1e-7)
            ref[frame, adr] = sol.x
            previous = sol.x.copy()
        depth = scan()
        history.append({'iteration': iteration, 'pairs': len(pairs), 'max_self_penetration_m': depth})
        print(history[-1], flush=True)
        if depth < .001:
            break
    payload = {k:source[k] for k in source.files}
    payload['qpos'] = ref
    np.savez_compressed(folder/'reference.npz', **payload)
    np.savetxt(folder/'qpos.csv', ref, delimiter=',')
    record = {'source_reference': str(a.source_reference.resolve()),
              'source_reference_sha256': hashlib.sha256(a.source_reference.read_bytes()).hexdigest(),
              'settings': vars(a) | {'source_reference': str(a.source_reference),
                                     'initial_reference': str(a.initial_reference) if a.initial_reference else None},
              'initial_max_self_penetration_m': initial, 'iterations': history,
              'max_arm_joint_correction_rad': float(np.max(abs(ref[:,adr]-original[:,adr]))),
              'collision_pairs': [list(pair) for pair in sorted(pairs)],
              'method': 'soft_collision_distance_arm_IK; root and legs unchanged; COM re-projection required'}
    (folder/'manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
