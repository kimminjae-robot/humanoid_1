"""Track every Isaac job and bound only Kit teardown after a fresh result exists."""
import argparse,json,os,signal,subprocess,sys,time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--label',default='tracking');p.add_argument('--timeout-s',type=int,default=1800)
    a,rest=p.parse_known_args()
    run=ROOT/'runs'/(datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f')+'_stage04_'+a.label);run.mkdir()
    cmd=[os.environ.get('G1_ISAACLAB_PYTHON', sys.executable),'-u',str(ROOT/'scripts/track.py'),'--headless','--run-dir',str(run),*rest]
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    # Keep Kit/torch generated caches inside the workspace where possible.
    env['TORCH_EXTENSIONS_DIR']=str(ROOT/'cache/torch_extensions')
    started=time.monotonic();seen=None;error_seen=None;reason='normal'
    with (run/'isaac.log').open('w') as log:
        job=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
        live={'run_dir':str(run.relative_to(ROOT)),'pid':job.pid,'command':cmd,'log':str((run/'isaac.log').relative_to(ROOT))}
        (ROOT/'state/active_tracking.json').write_text(json.dumps(live,indent=2));print(json.dumps(live,indent=2),flush=True)
        while job.poll() is None:
            now=time.monotonic()
            if (run/'result.json').exists() and seen is None:seen=now
            if error_seen is None and 'Traceback (most recent call last)' in (run/'isaac.log').read_text(errors='replace'):error_seen=now
            if now-started>a.timeout_s or (seen and now-seen>15) or (error_seen and now-error_seen>15):
                reason='bounded_kit_teardown_after_result' if seen else ('bounded_kit_teardown_after_error' if error_seen else 'timeout_before_result')
                os.killpg(job.pid,signal.SIGTERM)
                try:job.wait(5)
                except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);job.wait(5)
                break
            try:job.wait(1)
            except subprocess.TimeoutExpired:pass
    result=json.loads((run/'result.json').read_text()) if (run/'result.json').exists() else None
    physics_errors=[line for line in (run/'isaac.log').read_text(errors='replace').splitlines() if ('[Error]' in line and ('physx' in line.lower() or 'cuda' in line.lower())) or 'GPU contact filter for collider' in line or 'please increase' in line.lower()]
    execution={**live,'physics_errors':physics_errors,'returncode':job.returncode,'termination':reason,'elapsed_s':time.monotonic()-started,'fresh_result':result is not None}
    (run/'execution.json').write_text(json.dumps(execution,indent=2));(ROOT/'reports/stage_04_last_execution.json').write_text(json.dumps(execution,indent=2))
    active=json.loads((ROOT/'state/active_tracking.json').read_text())
    if active.get('run_dir')==execution['run_dir']:
        (ROOT/'state/active_tracking.json').write_text(json.dumps({**execution,'running':False},indent=2))
    print(json.dumps(execution,indent=2));return 0 if result is not None and not physics_errors else 1
if __name__=='__main__':raise SystemExit(main())
