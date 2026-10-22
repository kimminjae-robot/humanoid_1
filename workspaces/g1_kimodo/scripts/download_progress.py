"""Report cached encoder weight bytes without printing URLs or credentials."""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--encoder", choices=["official", "qwen"], default="official")
args = parser.parse_args()
if args.encoder == "official":
    inventory = json.loads((ROOT / "reports/hf_inventory.json").read_text())["repositories"]
    repo = "meta-llama/Meta-Llama-3-8B-Instruct"
    item = next(x for x in inventory if x["repo_id"] == repo)
    manifest = "official_encoder_checkpoint.json"
else:
    inventory = json.loads((ROOT / "reports/alternative_encoder_inventory.json").read_text())
    repo = "Qwen/Qwen3-8B"
    item = next(x for x in inventory if x["repo"] == repo)
    manifest = "qwen_checkpoint.json"
total = sum(f["size"] for f in item["files"] if f["name"].endswith(".safetensors"))
blobs = Path(os.environ.get("HF_HOME", ROOT / "cache/huggingface")) / "hub" / ("models--" + repo.replace("/", "--")) / "blobs"
sizes = {}
complete = set()
for path in blobs.iterdir() if blobs.exists() else []:
    if path.is_file() and re.fullmatch(r"[0-9a-f]{64}", path.name.split(".")[0]):
        key = path.name.split(".")[0]
        sizes[key] = max(sizes.get(key, 0), path.stat().st_size)
        if path.name == key:
            complete.add(key)
downloaded = sum(sizes.values())
report = {"timestamp": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(), "encoder": args.encoder, "repo_id": repo,
          "weight_gib_downloaded": round(downloaded / 2**30, 3), "weight_gib_total": round(total / 2**30, 3),
          "fraction": min(downloaded / total, 1), "complete_shards": len(complete),
          "total_shards": sum(f["name"].endswith(".safetensors") for f in item["files"]),
          "checkpoint_ready": (ROOT / "reports" / manifest).exists()}
(ROOT / "reports" / f"{args.encoder}_download_status.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
