"""Fetch a clean upstream checkout at the recorded revision, without overwrites."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
spec = json.loads((ROOT / 'sources.lock.json').read_text())['kimodo']
dest = ROOT / 'workspaces/g1_kimodo/third_party/kimodo'
if dest.exists():
    actual = subprocess.check_output(['git', '-C', str(dest), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != spec['revision']:
        raise SystemExit('Existing Kimodo checkout has a different revision; choose a fresh package directory.')
else:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'init', str(dest)], check=True)
    subprocess.run(['git', '-C', str(dest), 'remote', 'add', 'origin', spec['repository']], check=True)
    subprocess.run(['git', '-C', str(dest), 'fetch', '--depth', '1', 'origin', spec['revision']], check=True)
    subprocess.run(['git', '-C', str(dest), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
print('Kimodo source verified:', spec['revision'])
