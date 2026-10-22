"""Generate the fixed stage-2 baseline, preserving every attempted sample."""

import hashlib
import json
import logging
import resource
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from kimodo import load_model
from kimodo.exports.motion_io import save_kimodo_npz
from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.sanitize import sanitize_text
from kimodo.tools import seed_everything

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from official_cached_encoder import CachedOfficialEncoder


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b""):
            result.update(chunk)
    return result.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    started = time.monotonic()
    config = json.loads((ROOT / "configs/prompts.json").read_text())
    project = json.loads((ROOT / "configs/project.json").read_text())
    gate = json.loads((ROOT / "reports/stage_01_readiness.json").read_text())
    if not gate["official_route_passed"]:
        raise RuntimeError("Stage 1 official route has not passed")
    if shutil.disk_usage(ROOT).free < 2 * 2**30:
        raise RuntimeError("Need at least 2 GiB free working space")
    run_id = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S_%f") + "_stage02_official_baseline"
    run = ROOT / "runs" / run_id
    raw = ROOT / "data/raw" / run_id
    run.mkdir(parents=True, exist_ok=False)
    raw.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler(run / "generation.log"), logging.StreamHandler(sys.stdout)])
    baseline = project["scope"]["baseline_motion_ids"]
    prompts = [p for p in config["prompts"] if p["id"] in baseline]
    plan = [{"clip_id": f"{p['id']}_{split}_{seed}", "motion_id": p["id"], "split": split,
             "seed": seed, "prompt": p["text"], "sanitized_prompt": sanitize_text(p["text"])}
            for p in prompts for split in ["train", "validation"] for seed in config["seed_splits"][split]]
    if len(plan) != 15 or any(item["seed"] in config["seed_splits"]["final"] for item in plan):
        raise ValueError("Expected 15 samples with no final seeds")
    provenance = run / "provenance"
    provenance.mkdir()
    source_files = ["configs/prompts.json", "configs/project.json", "configs/motion_contract.json",
                    "configs/official_encoder.json", "reports/official_encoder_checkpoint.json",
                    "reports/text_encoder_smoke.json", "reports/kimodo_checkpoint.json",
                    "reports/g1_asset_manifest.json", "reports/kimodo_freeze.txt",
                    "scripts/generate_baseline.py", "scripts/official_cached_encoder.py"]
    snapshots = {}
    for relative in source_files:
        source = ROOT / relative
        target = provenance / relative.replace("/", "__")
        shutil.copy2(source, target)
        snapshots[relative] = {"snapshot": str(target.relative_to(ROOT)), "sha256": digest(target)}
    weights = ROOT / "checkpoints/Kimodo-G1-RP-v1/model.safetensors"
    run_manifest = {"run_id": run_id, "stage": 2, "created_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "encoder": "official_llm2vec_llama3_cached_real_cpu_embeddings", "experimental": False,
        "source_revisions": project["source_revisions"], "snapshots": snapshots,
        "generator_weights_sha256": digest(weights), "plan": plan,
        "seed_application": "immediately_before_each_sample_after_model_loading",
        "raw_output_definition": "direct_model_output_with_configured_postprocessing_no_reference_cleanup",
        "settings": {"model": config["generator_model"], "fps": config["fps"], "duration_s": config["duration_s"],
                     "frames": round(config["fps"] * config["duration_s"]), "diffusion_steps": config["diffusion_steps"],
                     "post_processing": config["postprocess"], "cfg_type": "separated", "cfg_weight": [2.0, 2.0],
                     "num_samples_per_call": 1, "constraints": [], "device": "cuda", "torch_threads": 8}}
    write(run / "manifest.json", run_manifest)
    run_hash = digest(run / "manifest.json")
    torch.set_num_threads(8)
    logging.info("Loading G1 once; run=%s; samples=%d", run_id, len(plan))
    model = load_model(config["generator_model"], device="cuda", text_encoder=CachedOfficialEncoder())
    converter = MujocoQposConverter(model.skeleton)
    results = []
    for index, item in enumerate(plan):
        destination = raw / item["clip_id"]
        destination.mkdir(exist_ok=False)
        record = {**item, "run_id": run_id, "run_manifest": str((run / "manifest.json").relative_to(ROOT)),
                  "run_manifest_sha256": run_hash, "settings": run_manifest["settings"], "status": "running"}
        before = time.monotonic()
        try:
            seed_everything(item["seed"])
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                motion = model(prompts=item["prompt"], num_frames=run_manifest["settings"]["frames"],
                    num_denoising_steps=config["diffusion_steps"], num_samples=1, post_processing=config["postprocess"],
                    return_numpy=False, cfg_type="separated", cfg_weight=[2.0, 2.0])
                qpos = converter.dict_to_qpos(motion, "cuda")
            if isinstance(qpos, torch.Tensor):
                qpos = qpos.detach().cpu().numpy()
            qpos = qpos.reshape(-1, 36)
            single = {key: value[0].detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)[0]
                      for key, value in motion.items()}
            npz, csv = destination / "motion.npz", destination / "qpos.csv"
            save_kimodo_npz(str(npz), single)
            np.savetxt(csv, qpos, delimiter=",")
            checks = {"150_frames_36_columns": qpos.shape == (150, 36), "finite_qpos": bool(np.isfinite(qpos).all()),
                "quaternion_norms": bool(np.max(np.abs(np.linalg.norm(qpos[:, 3:7], axis=-1) - 1)) < 1e-3),
                "finite_raw_npz": all(bool(np.isfinite(value).all()) for value in single.values()),
                "unbatched_npz_pose_shape": single["posed_joints"].shape == (150, 34, 3)}
            record.update(status="generated" if all(checks.values()) else "format_failed", checks=checks,
                artifacts={path.name: {"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in [npz, csv]},
                npz_shapes={key: list(value.shape) for key, value in single.items()},
                peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
            del motion, single, qpos
        except Exception as error:
            record.update(status="generation_failed", error_type=type(error).__name__, error=str(error))
            (destination / "failure.txt").write_text(traceback.format_exc())
            logging.exception("Failed sample %s", item["clip_id"])
            torch.cuda.empty_cache()
        record["elapsed_s"] = time.monotonic() - before
        write(destination / "manifest.json", record)
        results.append({**item, "status": record["status"], "manifest": str((destination / "manifest.json").relative_to(ROOT)),
                        "elapsed_s": record["elapsed_s"], "peak_cuda_allocated_mib": record.get("peak_cuda_allocated_mib")})
        write(run / "progress.json", {"completed_attempts": len(results), "expected_attempts": len(plan), "clips": results})
        logging.info("%d/%d %s: %s %.2fs", index + 1, len(plan), item["clip_id"], record["status"], record["elapsed_s"])
    summary = {"scope": "stage_2_generation_and_file_integrity_only", "run_id": run_id,
        "run_manifest": str((run / "manifest.json").relative_to(ROOT)), "clips": results,
        "expected_clips": 15, "generated_clips": sum(item["status"] == "generated" for item in results),
        "all_format_checks_passed": all(item["status"] == "generated" for item in results),
        "final_seeds_used": False, "semantic_review": "pending", "elapsed_s": time.monotonic() - started,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    write(run / "generation_summary.json", summary)
    write(ROOT / "reports/stage_02_generation.json", summary)
    write(ROOT / "data/splits" / (run_id + ".json"), {"run_id": run_id,
        "splits": {split: [item for item in results if item["split"] == split] for split in ["train", "validation"]},
        "final": "reserved_not_generated"})
    logging.info("Finished: %d/15 generated; total %.2fs", summary["generated_clips"], summary["elapsed_s"])
    return 0 if summary["all_format_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
