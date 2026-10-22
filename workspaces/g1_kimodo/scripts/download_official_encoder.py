"""Download pinned Llama and LLM2Vec files, excluding duplicate original weights."""

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapters-only", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/official_encoder.json").read_text())
    inventory = {repo["repo_id"]: repo for repo in json.loads((ROOT / "reports/hf_inventory.json").read_text())["repositories"]}
    roles = ["mntp", "supervised"] if args.adapters_only else ["base", "mntp", "supervised"]
    manifest = {"timestamp": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(), "repositories": {}}
    for role in roles:
        spec = config["repositories"][role]
        recorded = inventory[spec["repo_id"]]
        if recorded["sha"] != spec["revision"]:
            raise ValueError("Pinned revision disagrees with recorded inventory")
        names = [item["name"] for item in recorded["files"] if "/" not in item["name"] and
                 (item["name"].endswith((".json", ".safetensors")) or item["name"] in {"LICENSE", "USE_POLICY.md", "README.md"})]
        size = sum(item["size"] for item in recorded["files"] if item["name"] in names)
        if shutil.disk_usage(ROOT).free < 20 * 2**30 + size:
            raise RuntimeError("Insufficient free space for selected repository and 20 GiB working reserve")
        print(f"Downloading {role}: {spec['repo_id']}@{spec['revision']} ({size / 2**30:.3f} GiB selected)", flush=True)
        snapshot = Path(snapshot_download(repo_id=spec["repo_id"], revision=spec["revision"],
                                          allow_patterns=names, max_workers=4))
        files = {}
        for item in recorded["files"]:
            if item["name"] not in names:
                continue
            path = snapshot / item["name"]
            if path.stat().st_size != item["size"]:
                raise ValueError(f"Downloaded size mismatch: {role}/{item['name']}")
            files[item["name"]] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        manifest["repositories"][role] = {**spec, "snapshot": str(snapshot), "files": files}
        print(f"Verified {role}: {len(files)} files", flush=True)
    manifest["complete"] = not args.adapters_only
    output = ROOT / "reports" / ("official_adapters_checkpoint.json" if args.adapters_only else "official_encoder_checkpoint.json")
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest: {output}", flush=True)


if __name__ == "__main__":
    main()
