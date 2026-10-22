"""Download the exact generator snapshot; never resolve a floating revision."""
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
G1 = ROOT / 'workspaces/g1_kimodo'
os.environ.setdefault('HF_HOME', str(G1 / 'cache/huggingface'))
from huggingface_hub import snapshot_download

spec = json.loads((ROOT / 'sources.lock.json').read_text())['generator']
snapshot = Path(snapshot_download(repo_id=spec['repo_id'], revision=spec['revision']))
dest = G1 / 'checkpoints/Kimodo-G1-RP-v1'
if dest.is_symlink():
    if dest.resolve() != snapshot.resolve():
        raise SystemExit('Existing generator link points to another snapshot.')
elif dest.exists():
    raise SystemExit('Generator checkpoint directory already exists; inspect it before replacing.')
else:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.symlink_to(snapshot, target_is_directory=True)
digest = hashlib.sha256((snapshot / 'model.safetensors').read_bytes()).hexdigest()
(G1 / 'reports/kimodo_checkpoint.json').write_text(json.dumps({**spec, 'snapshot': str(snapshot), 'weights_sha256': digest}, indent=2)+'\n')
print('Pinned generator ready:', spec['revision'])
