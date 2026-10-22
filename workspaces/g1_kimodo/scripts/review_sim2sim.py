"""Render recorded dynamic traces; preview's FK is visualization only."""
import argparse,json,subprocess,sys,hashlib,html
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();run=a.run.resolve();resolved=json.loads((run/'resolved.json').read_text());irun=ROOT/resolved['source_Isaac_run']
    mr=json.loads((run/'result.json').read_text());ir=json.loads((irun/'result.json').read_text());dataset=json.loads((ROOT/resolved['dataset_index']).read_text());clips={c['clip_id']:c for c in dataset['clips']}
    mz=np.load(run/'rollout.npz',allow_pickle=False);iz=np.load(irun/'rollout.npz',allow_pickle=False);records=[]
    for slot in [0,2,4]:
        me=next(e for e in mr['episodes'] if e['env_id']==slot);ie=next(e for e in ir['episodes'] if e['env_id']==slot);cid=me['clip_id'];folder=run/'videos'/cid;folder.mkdir(parents=True,exist_ok=True)
        steps=max(me['steps'],ie['steps']);initial=np.r_[me['initial_state']['root_pose_local'],me['initial_state']['joint_pos_source']]
        traces={}
        for label,z,e in [('Isaac',iz,ie),('MuJoCo',mz,me)]:
            q=np.concatenate([initial[None],z['qpos'][:e['steps'],slot]])
            traces[label]=np.concatenate([q,np.repeat(q[-1:],steps+1-len(q),axis=0)])
        ref=np.loadtxt(ROOT/clips[cid]['qpos_csv'],delimiter=',');traces['Reference']=ref[np.minimum(np.arange(steps+1),len(ref)-1)]
        for tag,q in traces.items():
            csv=folder/(tag+'.csv');np.savetxt(csv,q,delimiter=',')
            subprocess.run([sys.executable,'-I',str(ROOT/'scripts/preview_motion.py'),str(csv),'--fps','50'],check=True,stdout=subprocess.DEVNULL)
        video=folder/'reference_isaac_mujoco.mp4';filters=[]
        for n,label in enumerate(['Reference','Isaac','MuJoCo']):
            suffix='' if n==0 else ' success '+str((ie if n==1 else me)['success'])
            f=f'[{n}:v]scale=640:480,drawtext=text=\'{label}{suffix}\':x=14:y=16:fontsize=22:fontcolor=white'
            if n:
                ep=ie if n==1 else me
                if ep['steps']<steps:f+=f",drawtext=text='Terminated - held final frame':x=14:y=52:fontsize=20:fontcolor=yellow:enable='gte(t,{(ep['steps']+1)/50})'"
            filters.append(f+'[v'+str(n)+']')
        filters.append('[v0][v1][v2]hstack=inputs=3[v]')
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y',*sum([['-i',str(folder/(tag+'.mp4'))] for tag in ['Reference','Isaac','MuJoCo']],[]),'-filter_complex',';'.join(filters),'-map','[v]','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(video)],check=True)
        sheet=folder/'frames.png';subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(video),'-vf',f'fps=10/{(steps+1)/50},scale=600:150,tile=5x2','-frames:v','1',str(sheet)],check=True)
        records.append({'clip_id':cid,'evaluation_seed':me['evaluation_seed'],'Isaac_success':ie['success'],'MuJoCo_success':me['success'],'Isaac_steps':ie['steps'],'MuJoCo_steps':me['steps'],'video':str(video.relative_to(ROOT)),'contact_sheet':str(sheet.relative_to(ROOT)),'video_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),'visual_review':'pending'})
        print('Rendered '+cid,flush=True)
    manifest={'scope':'reference_and_recorded_Isaac_and_actual_MuJoCo_dynamic_policy_states_rendered_with_canonical_FK','padding':'terminated episodes hold their final recorded state with visible label; never counted as additional rollout','cases':records}
    (run/'video_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    summary=json.loads((run/'comparison.json').read_text()) if (run/'comparison.json').exists() else None
    parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>G1 Kimodo sim-to-sim</title><style>body{background:#161b22;color:#e6edf3;font:16px system-ui;max-width:1400px;margin:30px auto;padding:20px}video,img{width:100%}section{margin:40px 0}td,th{padding:10px 24px;border-bottom:1px solid #444}a{color:#79c0ff}</style><h1>G1 Kimodo → Isaac → MuJoCo</h1><p>동일 v11 정책 · 명시적 동일 초기 상태 · 실제 자유 베이스 물리 평가. 왼쪽 reference / 가운데 Isaac / 오른쪽 MuJoCo.</p><p>60개 평가 중 seed1000의 대표 동작 3개. 종료된 동작은 마지막 프레임을 유지하며 노란색으로 표시합니다. 영상 렌더링은 저장된 물리 상태를 사용합니다.</p>']
    if summary:
        parts.append('<table><tr><th>동작</th><th>Isaac</th><th>MuJoCo</th></tr>')
        for m,g in summary['per_motion'].items():parts.append(f"<tr><td>{m}</td><td>{g['Isaac_successes']}/{g['episodes']}</td><td>{g['MuJoCo_successes']}/{g['episodes']}</td></tr>")
        parts.append('</table>')
    for r in records:
        parts.append(f"<section><h2>{html.escape(r['clip_id'])}</h2><video controls preload='metadata' src='../{r['video']}'></video><a href='../{r['video']}'>MP4</a> · <a href='../{r['contact_sheet']}'>프레임 모음</a><p>Isaac {r['Isaac_steps']/50:.2f}s / MuJoCo {r['MuJoCo_steps']/50:.2f}s · 성공: {r['Isaac_success']} / {r['MuJoCo_success']}</p></section>")
    parts.append('<p><a href="stage_05.md">평가 보고서</a> · <a href="stage_05_summary.json">전체 정량 결과</a></p></html>')
    (ROOT/'reports/stage_05_gallery.html').write_text('\n'.join(parts))
if __name__=='__main__':main()
