"""Verify genuine official CPU embeddings condition the real CUDA G1 model."""

import hashlib
import json
import resource
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from kimodo import load_model
from kimodo.exports.motion_io import save_kimodo_npz
from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.tools import seed_everything

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from official_cached_encoder import CachedOfficialEncoder


def main():
    started = time.monotonic()
    seed_everything(73101)
    torch.cuda.reset_peak_memory_stats()
    prompts = json.loads((ROOT / "configs/prompts.json").read_text())["prompts"]
    prompt = next(item for item in prompts if item["id"] == "walk_slow")["text"]
    model = load_model("Kimodo-G1-RP-v1", device="cuda", text_encoder=CachedOfficialEncoder())
    with torch.inference_mode():
        motion = model(prompts=prompt, num_frames=60, num_denoising_steps=100, num_samples=1,
                       post_processing=True, return_numpy=False, cfg_type="separated", cfg_weight=[2.0, 2.0])
        qpos = MujocoQposConverter(model.skeleton).dict_to_qpos(motion, "cuda")
    if isinstance(qpos, torch.Tensor):
        qpos = qpos.detach().cpu().numpy()
    qpos = qpos.reshape(-1, 36)
    output = ROOT / "runs" / (datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S_%f") + "_stage01_official_connectivity")
    output.mkdir(parents=True, exist_ok=False)
    csv, npz = output / "walk_slow_connectivity.csv", output / "motion.npz"
    np.savetxt(csv, qpos, delimiter=",")
    save_kimodo_npz(str(npz), motion)
    checks = {
        "60_frames_36_columns": qpos.shape == (60, 36),
        "finite_qpos": bool(np.isfinite(qpos).all()),
        "quaternion_norms": bool(np.max(np.abs(np.linalg.norm(qpos[:, 3:7], axis=-1) - 1)) < 1e-3),
    }
    provenance = {}
    for name in ["official_encoder_checkpoint.json", "text_encoder_smoke.json", "kimodo_checkpoint.json"]:
        path = ROOT / "reports" / name
        provenance[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "manifest": json.loads(path.read_text())}
    report = {
        "scope": "official_text_to_G1_connectivity_not_motion_quality_or_dynamic_tracking",
        "timestamp": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "prompt": prompt, "seed": 73101, "seed_split": "stage_1_connectivity_only",
        "duration_s": 2.0, "fps": 30, "diffusion_steps": 100, "post_processing": True,
        "encoder": "official_llm2vec_llama3", "experimental": False,
        "cfg_type": "separated", "cfg_weight": [2.0, 2.0],
        "source_revisions": json.loads((ROOT / "configs/project.json").read_text())["source_revisions"],
        "provenance": provenance, "checks": checks, "passed": all(checks.values()),
        "artifacts": {path.name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in [csv, npz]},
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "elapsed_s": time.monotonic() - started,
        "quality_evaluation": "pending_stage_2_and_3", "physical_tracking_evaluation": "pending_stage_4_and_5",
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    (ROOT / "reports/official_generation_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["passed", "checks", "artifacts", "peak_cuda_allocated_mib", "elapsed_s"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
