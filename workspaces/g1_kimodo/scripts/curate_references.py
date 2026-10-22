"""Select each source clip's latest reference passing the current unchanged gates."""

import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from reference_core import analyze


def main():
    baseline=json.loads((ROOT/"reports/stage_02_generation.json").read_text())
    required={"finite_values","unit_quaternions","joint_limits","ground_penetration","self_collision","foot_slip_fraction","joint_velocity","support_present_all_frames","static_endpoint_hold","root_linear_speed","root_linear_acceleration","foot_ik_tracking"}
    attempts=[]
    for path in sorted((ROOT/"runs").glob('*_stage03_reference_v*/summary.json'),reverse=True):
        manifest=json.loads((path.parent/'manifest.json').read_text())
        if manifest['source_generation_run']==baseline['run_id']:
            attempts.append(json.loads(path.read_text()))
    selected=[]
    for source in baseline['clips']:
        latest=None;chosen=None
        for attempt in attempts:
            clip=next(c for c in attempt['clips'] if c['clip_id']==source['clip_id'])
            if latest is None:latest=(clip,attempt['run_id'])
            if not required.issubset(clip['metrics']['checks']) or not all(clip['metrics']['checks'][key] for key in required):continue
            q=np.loadtxt(ROOT/clip['qpos_csv'],delimiter=',');qa,_=analyze(q,50)
            qa['checks']['foot_ik_tracking']=clip['processing']['ik']['max_foot_target_error_m']<=.005
            qa['passed']=all(qa['checks'].values())
            if qa['passed']:
                chosen={**clip,'metrics':qa,'processing_run_id':attempt['run_id']};break
        if chosen is None:
            clip,run_id=latest;chosen={**clip,'processing_run_id':run_id}
        profile=ROOT/'runs'/chosen['processing_run_id']/'provenance/configs__reference_processing.json'
        chosen['processing_profile_snapshot']=str(profile.relative_to(ROOT))
        chosen['processing_profile_sha256']=hashlib.sha256(profile.read_bytes()).hexdigest()
        selected.append(chosen)
    run_id=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f')+'_stage03_curated'
    run=ROOT/'runs'/run_id;run.mkdir();provenance=run/'provenance';provenance.mkdir()
    for name in ['configs/evaluation.json','configs/reference_processing.json','configs/motion_contract.json','scripts/reference_core.py','scripts/curate_references.py','reports/mujoco_stage03_freeze.txt']:
        shutil.copy2(ROOT/name,provenance/name.replace('/','__'))
    counts={motion:{split:sum(c['motion_id']==motion and c['split']==split and c['metrics']['passed'] for c in selected) for split in ['train','validation']} for motion in ['stand','wave','walk_slow']}
    result={'run_id':run_id,'source_generation_run':baseline['run_id'],'selection_rule':'latest per-source version passing all current gates, recomputed before selection; candidates without root speed/acceleration checks are ineligible','clips':selected,'accepted_counts':counts,'passed':all(c['train']>=2 and c['validation']>=1 for c in counts.values()),'source_candidate_count':15,'accepted':sum(c['metrics']['passed'] for c in selected),'rejected':sum(not c['metrics']['passed'] for c in selected),'attempt_run_ids':[a['run_id'] for a in attempts]}
    (run/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');(ROOT/'reports/stage_03_curated.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='clips'},indent=2))
    return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
