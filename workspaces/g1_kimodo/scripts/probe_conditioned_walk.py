"""Recondition walking IK in a new run; canonical stage-3 data is never overwritten."""
import argparse,hashlib,json,shutil,sys,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from reference_core import model_and_config,analyze,derivative,rotations
from prepare_references import process


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--height',type=float,default=.70);p.add_argument('--knee-min',type=float,default=.15);p.add_argument('--acceleration-weight',type=float,default=2.);p.add_argument('--lateral-scale',type=float,default=1.);p.add_argument('--label',default='v12');a=p.parse_args()
    model,cfg,contract=model_and_config();cfg['version']='stage4_walk_'+a.label
    cfg['walk'].update(pelvis_height_m=a.height,knee_min_rad=a.knee_min,joint_temporal_weight=.4,lateral_spatial_scale=a.lateral_scale)
    cfg['ik']['acceleration_regularization_weight']=a.acceleration_weight
    rid=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f')+'_stage04_walk_reference_'+a.label
    run=ROOT/'runs'/rid;out=ROOT/'data/processed'/rid;run.mkdir();out.mkdir();prov=run/'provenance';prov.mkdir()
    (prov/'configs__reference_processing.json').write_text(json.dumps(cfg,indent=2))
    for n in ['scripts/probe_conditioned_walk.py','scripts/prepare_references.py','scripts/reference_core.py','configs/evaluation.json','configs/motion_contract.json']:
        shutil.copy2(ROOT/n,prov/n.replace('/','__'))
    generation=json.loads((ROOT/'reports/stage_02_generation.json').read_text());clips=[]
    for item in generation['clips']:
        if item['motion_id']!='walk_slow':continue
        start=time.monotonic();manifest=json.loads((ROOT/item['manifest']).read_text());source=ROOT/manifest['artifacts']['qpos.csv']['path'];q0=np.loadtxt(source,delimiter=',')
        q,t,source_t,processing=process(q0,'walk_slow',model,cfg,contract)
        qa,arrays=analyze(q,50,model);qa['checks']['foot_ik_tracking']=processing['ik']['max_foot_target_error_m']<=.005;qa['passed']=all(qa['checks'].values())
        folder=out/item['clip_id'];folder.mkdir();np.savetxt(folder/'qpos.csv',q,delimiter=',')
        with np.load(ROOT/manifest['artifacts']['motion.npz']['path'],allow_pickle=False) as raw:contacts=raw['foot_contacts'][np.clip(np.rint(source_t*30).astype(int),0,len(q0)-1)]
        bundle={**arrays,'fps':np.array(50),'timestamps_s':t,'source_timestamps_s':source_t,'joint_names':np.array(contract['csv']['joint_names']),'root_pos_world_m':q[:,:3],'root_quat_world_wxyz':q[:,3:7],'joint_pos_rad':q[:,7:],'source_foot_contacts':contacts,'source_sha256':np.array(sha(source)),'generation_manifest':np.array(json.dumps(manifest)),'robot_manifest_sha256':np.array(sha(ROOT/'reports/g1_asset_manifest.json'))}
        bundle['qvel_mujoco']=np.c_[bundle['root_lin_vel_world_m_s'],rotations(q[:,3:7]).inv().apply(bundle['root_ang_vel_world_rad_s']),bundle['joint_vel_rad_s']]
        np.savez_compressed(folder/'motion_bundle.npz',**bundle)
        leg_acc=float(np.abs(derivative(bundle['joint_vel_rad_s'][:,:12],50)).max())
        clip={'clip_id':item['clip_id'],'motion_id':'walk_slow','split':item['split'],'seed':item['seed'],'source_sha256':sha(source),'source_manifest':item['manifest'],'accepted':qa['passed'],'rejection_reasons':[k for k,v in qa['checks'].items() if not v],'metrics':qa,'processing':processing,'processing_run_id':rid,'processing_profile_snapshot':str((prov/'configs__reference_processing.json').relative_to(ROOT)),'processing_profile_sha256':sha(prov/'configs__reference_processing.json'),'bundle':str((folder/'motion_bundle.npz').relative_to(ROOT)),'bundle_sha256':sha(folder/'motion_bundle.npz'),'qpos_csv':str((folder/'qpos.csv').relative_to(ROOT)),'qpos_sha256':sha(folder/'qpos.csv'),'max_leg_acceleration_rad_s2':leg_acc,'elapsed_s':time.monotonic()-start}
        clips.append(clip);(folder/'manifest.json').write_text(json.dumps(clip,indent=2));(run/'progress.json').write_text(json.dumps(clips,indent=2))
        print(item['clip_id'],qa['passed'],clip['rejection_reasons'],'IK',processing['ik']['max_foot_target_error_m'],'leg_acc',leg_acc,flush=True)
    result={'run_id':rid,'scope':'stage4_conditioning_candidate_not_adopted_or_dynamically_validated','source_candidates':5,'clips':clips,'counts':{s:sum(c['accepted'] and c['split']==s for c in clips) for s in ['train','validation']}}
    (run/'summary.json').write_text(json.dumps(result,indent=2));(ROOT/'reports/stage_04_conditioning_latest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='clips'},indent=2))
if __name__=='__main__':main()
