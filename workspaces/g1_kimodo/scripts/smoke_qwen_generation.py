"""Verify real Qwen conditioning reaches the unchanged G1 diffusion model."""

import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch
from kimodo import load_model
from kimodo.exports.mujoco import MujocoQposConverter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from qwen_cached_encoder import CachedQwenEncoder


def main():
    started = time.monotonic()
    torch.manual_seed(73101)
    torch.cuda.reset_peak_memory_stats()
    config = json.loads((ROOT / "configs/prompts.json").read_text())
    prompt = next(p for p in config["prompts"] if p["id"] == "walk_slow")["text"]
    model = load_model("Kimodo-G1-RP-v1", device="cuda", text_encoder=CachedQwenEncoder())
    with torch.inference_mode():
        motion = model(prompts=prompt, num_frames=60, num_denoising_steps=100, num_samples=1,
                       post_processing=True, return_numpy=False)
        qpos = MujocoQposConverter(model.skeleton).dict_to_qpos(motion, "cuda")
    if isinstance(qpos, torch.Tensor):
        qpos = qpos.detach().cpu().numpy()
    qpos = qpos.reshape(-1, 36)
    output = ROOT / "runs" / (datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S") + "_stage01_qwen_connectivity")
    output.mkdir(parents=True, exist_ok=True)
    csv = output / "walk_slow_connectivity.csv"
    np.savetxt(csv, qpos, delimiter=",")
    checks = {"60_frames_36_columns": qpos.shape == (60, 36), "finite_qpos": bool(np.isfinite(qpos).all()),
              "quaternion_norms": bool(np.max(np.abs(np.linalg.norm(qpos[:, 3:7], axis=-1) - 1)) < 1e-3)}
    report = {"scope": "experimental_text_to_G1_CSV_connectivity_not_motion_quality_or_dynamic_tracking",
        "prompt": prompt, "seed": 73101, "duration_s": 2.0, "fps": 30, "diffusion_steps": 100,
        "encoder": "Qwen3-8B_plus_community_projection", "experimental": True, "checks": checks,
        "passed": all(checks.values()), "csv": str(csv), "csv_sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "elapsed_s": time.monotonic() - started, "quality_evaluation": "pending_stage_2_and_3",
        "official_Llama_equivalence": "not_established"}
    (ROOT / "reports/qwen_generation_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
