"""Verify gated-model access without exposing tokens or response bodies."""

import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
REPO = "meta-llama/Meta-Llama-3-8B-Instruct"


def main():
    report = {"timestamp": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(), "repo_id": REPO, "access_verified": False}
    token_path = Path(os.environ.get("HF_TOKEN_PATH", str(Path(os.environ["HF_HOME"]) / "token")))
    report["token_available_in_checked_locations"] = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")) or token_path.is_file()
    try:
        spec = json.loads((ROOT / "configs/official_encoder.json").read_text())["repositories"]["base"]
        hf_hub_download(REPO, "config.json", revision=spec["revision"])
        report.update(access_verified=True, revision=spec["revision"])
    except Exception as error:
        report["error_type"] = type(error).__name__
        response = getattr(error, "response", None)
        report["http_status"] = getattr(response, "status_code", None)
    (ROOT / "reports/hf_access.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["access_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
