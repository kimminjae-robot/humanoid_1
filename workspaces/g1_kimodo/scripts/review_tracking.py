"""Render recorded Isaac policy states using canonical geometry; no MuJoCo dynamics."""
import argparse,json,subprocess,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();run=a.run.resolve()
    result=json.loads((run/'result.json').read_text());dataset=json.loads((ROOT/json.loads((ROOT/'state/progress.json').read_text())['artifacts_stage_3']['dataset_index']).read_text())
    output=run/'videos';output.mkdir(exist_ok=True);records=[]
    with np.load(run/'rollout.npz',allow_pickle=False) as b:
        for slot,clip_id in enumerate(b['clip_ids'].tolist()):
            ep=next(x for x in result['episodes'] if x['env_id']==slot)
            actual=b['qpos'][:ep['steps'],slot]
            initial=ep['initial_state'];q0=np.r_[initial['root_pose_local'],initial['joint_pos_source']]
            actual=np.concatenate([q0[None],actual]);clip=next(c for c in dataset['clips'] if c['clip_id']==clip_id)
            reference=np.loadtxt(ROOT/clip['qpos_csv'],delimiter=',');reference=reference[np.arange(len(actual)).clip(0,len(reference)-1)]
            folder=output/clip_id;folder.mkdir(exist_ok=True)
            for tag,q in [('actual_isaac',actual),('reference',reference)]:
                csv=folder/(tag+'.csv');np.savetxt(csv,q,delimiter=',')
                subprocess.run([sys.executable,'-I',str(ROOT/'scripts/preview_motion.py'),str(csv),'--fps','50'],check=True,stdout=subprocess.DEVNULL)
            path=folder/'reference_vs_isaac.mp4'
            filters="[0:v]drawtext=text='Reference':x=20:y=20:fontsize=28:fontcolor=white[a];[1:v]drawtext=text='Isaac physics - success "+str(ep['success'])+"':x=20:y=20:fontsize=28:fontcolor=white[b];[a][b]hstack=inputs=2[v]"
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(folder/'reference.mp4'),'-i',str(folder/'actual_isaac.mp4'),'-filter_complex',filters,'-map','[v]','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(path)],check=True)
            sheet=folder/'frames.png'
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(path),'-vf',f'fps=10/{len(actual)/50},scale=480:180,tile=5x2','-frames:v','1',str(sheet)],check=True)
            records.append({'clip_id':clip_id,'episode':ep,'video':str(path.relative_to(ROOT)),'contact_sheet':str(sheet.relative_to(ROOT)),'video_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    summary={'scope':'recorded_Isaac_dynamic_rollout_rendered_by_MuJoCo_FK_not_MuJoCo_sim2sim','run':str(run.relative_to(ROOT)),'cases':records}
    (run/'video_manifest.json').write_text(json.dumps(summary,indent=2));print(json.dumps({'run':summary['run'],'videos':[r['video'] for r in records]},indent=2))
if __name__=='__main__':main()
