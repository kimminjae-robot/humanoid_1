"""One input command for generic CSV retargeting, QA, dynamics and video.

No motion-name-based walking conversion is applied. Performance failures remain
visible in the report; diagnostic videos are rendered even when QA fails.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-csv', type=Path, required=True)
    p.add_argument('--source-manifest', type=Path)
    p.add_argument('--fps', type=float, default=30)
    p.add_argument('--label', required=True)
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.label):
        raise ValueError('Invalid label')
    work = ROOT / 'runs' / (a.label + '_pipeline')
    work.mkdir(parents=True, exist_ok=False)
    state = {'label': a.label, 'source_csv': str(a.source_csv.resolve()),
             'source_csv_sha256': hashlib.sha256(a.source_csv.read_bytes()).hexdigest(),
             'stages': [], 'performance_passed': False,
             'scope': 'generic v1 IK and legacy balance; not universal dynamic tracking'}
    def save():
        (work / 'pipeline.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    save()
    snapshot = work / 'provenance'
    snapshot.mkdir()
    hashes = {}
    for relative in ['scripts/process_motion.py', 'scripts/retarget.py', 'scripts/evaluate.py',
                     'scripts/review.py', 'scripts/assess.py', 'configs/evaluation.json',
                     'reports/robot_manifest.json', 'assets/k1_hx5.xml']:
        dst = snapshot / relative.replace('/', '__')
        shutil.copy2(ROOT / relative, dst)
        hashes[relative] = hashlib.sha256(dst.read_bytes()).hexdigest()
    state['provenance_sha256'] = hashes
    retarget = ['--source-csv', str(a.source_csv.resolve()), '--fps', str(a.fps),
                '--clip-id', a.label, '--motion-id', 'custom', '--profile', 'v1']
    if a.source_manifest:
        retarget += ['--source-manifest', str(a.source_manifest.resolve())]
    stages = [('retarget', retarget),
              ('evaluate', ['--references', str(ROOT / 'reports' / ('retarget_' + a.label + '.json')),
                            '--controller', 'balance', '--label', a.label]),
              ('assess', ['--run', a.label]),
              ('review', ['--run', a.label, '--gallery-name', 'gallery_' + a.label + '.html'])]
    try:
        for script, argv in stages:
            print('Starting ' + script, flush=True)
            with (work / (script + '.log')).open('w') as log:
                proc = subprocess.run([sys.executable, '-I', str(ROOT / 'scripts' / (script + '.py')),
                                       *argv], stdout=log, stderr=subprocess.STDOUT)
            state['stages'].append({'stage': script, 'exit_code': proc.returncode})
            save()
            if proc.returncode:
                raise RuntimeError(script + ' failed; see ' + str(work / (script + '.log')))
        result = json.loads((ROOT / 'runs' / a.label / 'result.json').read_text())
        qa = json.loads((ROOT / 'runs' / a.label / 'kinematic_qa.json').read_text())
        state.update(execution_completed=True, kinematic_passed=qa['passed'],
                     dynamic_success_rates=result['success_rates'],
                     performance_passed=qa['passed'] and result['all_motions_passed'],
                     gallery='reports/gallery_' + a.label + '.html')
        save()
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0 if state['performance_passed'] else 2
    except Exception as exc:
        state.update(execution_completed=False, error=repr(exc))
        save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
