"""Compare FK of retargeted reference with recorded, freely simulated states."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json,subprocess,hashlib
from pathlib import Path
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='walking_improved_v4');ap.add_argument('--gallery-name',default='gallery.html');a=ap.parse_args();run=ROOT/'runs'/a.run;result=json.loads((run/'result.json').read_text());qa=json.loads((run/'kinematic_qa.json').read_text());model=mujoco.MjModel.from_xml_path(str(ROOT/'assets/k1_hx5.xml'));d=mujoco.MjData(model);width,height=640,480;model.vis.global_.offwidth=width;model.vis.global_.offheight=height;camera=mujoco.MjvCamera();camera.distance=2.6;camera.azimuth=130;camera.elevation=-14;model.vis.headlight.ambient[:]=[.5,.5,.5];option=mujoco.MjvOption();option.geomgroup[:]=0;option.geomgroup[0]=1;option.geomgroup[1]=1;option.geomgroup[2]=1
    videos=ROOT/'reports/videos'/a.run;videos.mkdir(parents=True,exist_ok=True);records=[]
    with mujoco.Renderer(model,height=height,width=width) as renderer:
        for motion in result['success_rates']:
            ep=next(e for e in result['episodes'] if e['motion_id']==motion and e['seed']==1000);b=np.load(ROOT/ep['rollout_npz'],allow_pickle=False);ref=b['reference'];actual=np.concatenate([b['initial_qpos'][None],b['qpos']]);steps=max(len(ref)+50,len(actual));folder=videos/motion;folder.mkdir(exist_ok=True);output=folder/'reference_vs_physics.mp4'
            filters="drawtext=text='K1 + HX5 - retargeted reference':x=14:y=14:fontsize=20:fontcolor=white,drawtext=text='MuJoCo physics - success "+str(ep['success'])+"':x=654:y=14:fontsize=20:fontcolor=white"
            if len(actual)<steps:filters+=f",drawtext=text='Terminated - held final state':x=654:y=46:fontsize=18:fontcolor=yellow:enable='gte(t,{len(actual)*.02})'"
            proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x480','-r','25','-i','pipe:0','-vf',filters,'-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(output)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
            try:
                for i in range(0,steps,2):
                    camera.lookat[:]=[*ref[min(i,len(ref)-1),:2],.72];images=[]
                    for q in [ref[min(i,len(ref)-1)],actual[min(i,len(actual)-1)]]:
                        d.qpos[:]=q;mujoco.mj_kinematics(model,d);mujoco.mj_comPos(model,d);renderer.update_scene(d,camera=camera,scene_option=option);images.append(renderer.render().copy())
                    proc.stdin.write(np.concatenate(images,axis=1).tobytes())
                proc.stdin.close();stderr=proc.stderr.read().decode();code=proc.wait()
                if code:raise RuntimeError(stderr)
            except BaseException:proc.kill();proc.wait();raise
            sheet=folder/'frames.png';subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(output),'-vf',f'fps=10/{steps*.02},scale=640:240,tile=5x2','-frames:v','1',str(sheet)],check=True)
            records.append({'motion_id':motion,'clip_id':ep['clip_id'],'seed':1000,'video':str(output.relative_to(ROOT)),'contact_sheet':str(sheet.relative_to(ROOT)),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'success':ep['success'],'QA':next(q['passed'] for q in qa['clips'] if q['clip_id']==ep['clip_id']),'visual_review':'pending'})
            print('Rendered '+motion,flush=True)
    (run/'video_manifest.json').write_text(json.dumps({'scope':'reference_FK_vs_recorded_free_base_dynamic_rollout; early_termination_is_labeled_final_frame_hold','cases':records},indent=2)+'\n')
    parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>K1 + HX5 Kimodo MuJoCo</title><style>body{background:#161b22;color:#e6edf3;font:16px system-ui;max-width:1300px;margin:30px auto;padding:20px}video,img{width:100%}section{margin:36px 0}a{color:#79c0ff}td,th{padding:10px 24px;border-bottom:1px solid #444}</style><h1>AI Sapiens K1 + HX5-D20</h1><p>Kimodo G1 모션 → K1 리타게팅 → MuJoCo 검증. 손가락은 일정 목표 자세로 유지합니다.</p><p>왼쪽: 변환 모션 재생 / 오른쪽: 실제 자유 베이스·중력·접촉 물리 실행. 걷기는 지면반력 QP와 몸체 PD, 서기·wave는 기존 균형 제어기를 사용합니다. Isaac 학습 정책의 sim-to-sim 결과가 아닙니다.</p><p>손목 장착은 시뮬레이션 가정입니다. 현재 run의 QA와 물리 성공·실패를 함께 표시합니다. 영상은 seed1000 대표 사례이며, 정량 평가는 동작별3회입니다.</p><table><tr><th>동작</th><th>reference QA</th><th>물리 성공</th></tr>']
    for r in records:
        eps=[e for e in result['episodes'] if e['motion_id']==r['motion_id']];parts.append(f"<tr><td>{r['motion_id']}</td><td>{r['QA']}</td><td>{sum(e['success'] for e in eps)}/{len(eps)}</td></tr>")
    parts.append('</table>')
    for r in records:parts.append(f"<section><h2>{r['motion_id']}</h2><video controls preload='metadata' src='../{r['video']}'></video><a href='../{r['video']}'>MP4</a> · <a href='../{r['contact_sheet']}'>10-frame sheet</a></section>")
    parts.append('<p><a href="baseline.md">평가 보고서</a> · <a href="../STATUS.md">현재 상태</a></p></html>');(ROOT/'reports'/a.gallery_name).write_text('\n'.join(parts))
if __name__=='__main__':main()
