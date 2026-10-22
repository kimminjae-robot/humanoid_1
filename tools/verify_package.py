"""Verify release hashes, internal model paths and compiled robot dimensions."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
failures = []
manifest = json.loads((ROOT / 'PACKAGE_MANIFEST.json').read_text())
for relative, expected in manifest['files'].items():
    path = ROOT / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
        failures.append('checksum: ' + relative)
for relative in ('workspaces/k1_kimodo_hx5/assets/k1_hx5.xml', 'workspaces/g1_kimodo/assets/g1/g1_scene.xml'):
    path = ROOT / relative
    for node in ET.parse(path).iter():
        name = node.get('file')
        if name:
            target = (path.parent / name).resolve()
            if Path(name).is_absolute() or not target.is_relative_to(ROOT) or not target.is_file():
                failures.append('asset path: ' + str(target))
import mujoco
for relative, expected in [('workspaces/k1_kimodo_hx5/assets/k1_hx5.xml', (70,69,63)), ('workspaces/g1_kimodo/assets/g1/g1_scene.xml', (36,35,29))]:
    model = mujoco.MjModel.from_xml_path(str(ROOT / relative))
    if (model.nq, model.nv, model.nu) != expected:
        failures.append('dimensions: ' + relative)
result = {'passed': not failures, 'files_verified': len(manifest['files']), 'mujoco': mujoco.__version__, 'failures': failures}
print(json.dumps(result, indent=2))
raise SystemExit(0 if result['passed'] else 1)
