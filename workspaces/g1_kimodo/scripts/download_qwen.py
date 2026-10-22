"""Download a pinned public alternative without changing official Kimodo."""

import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path

# Set before importing huggingface_hub to select the resumable HTTP transport.
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / "configs/alternative_encoder.json").read_text())
    source_dir = ROOT / "third_party/qwen_encoder"
    source_dir.mkdir(parents=True, exist_ok=True)
    source = source_dir / "custom_text_encoder.py"
    source_url = f"https://raw.githubusercontent.com/{config['implementation_repo']}/{config['implementation_revision']}/{config['source_path']}"
    if not source.exists():
        source.write_bytes(urllib.request.urlopen(source_url, timeout=30).read())
    if hashlib.sha256(source.read_bytes()).hexdigest() != config["implementation_sha256"]:
        raise ValueError("Experimental encoder implementation checksum mismatch")
    shutil.copy2(ROOT / "third_party/kimodo/LICENSE", source_dir / "LICENSE")
    snapshot = snapshot_download(config["encoder_repo"], revision=config["encoder_revision"], token=False, max_workers=2)
    projection = hf_hub_download(config["projection_repo"], config["projection_filename"], revision=config["projection_revision"], token=False)
    manifest = {**config, "encoder_snapshot": snapshot, "projection_file": projection,
                "implementation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "experimental": True}
    (ROOT / "reports/qwen_checkpoint.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
