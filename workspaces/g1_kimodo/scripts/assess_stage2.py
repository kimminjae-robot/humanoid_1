"""Assess the stage-2 artifact/review gate, independently of dynamic tracking."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    generation = json.loads((ROOT / "reports/stage_02_generation.json").read_text())
    run = ROOT / "runs" / generation["run_id"]
    artifact = json.loads((run / "artifact_review.json").read_text())
    semantic = json.loads((run / "semantic_review.json").read_text())
    manifest_path = ROOT / generation["run_manifest"]
    manifest = json.loads(manifest_path.read_text())
    prompt_snapshot = ROOT / manifest["snapshots"]["configs/prompts.json"]["snapshot"]
    prompts = json.loads(prompt_snapshot.read_text())
    planned = {(item["motion_id"], item["split"], item["seed"]) for item in manifest["plan"]}
    actual = {(item["motion_id"], item["split"], item["seed"]) for item in generation["clips"]}
    checks = {"15_planned_samples_preserved": len(generation["clips"]) == len(actual) == len(planned) == 15 and actual == planned,
        "all_generation_completed": generation["generated_clips"] == 15 and generation["all_format_checks_passed"],
        "final_seeds_unused": not any(item["seed"] in prompts["seed_splits"]["final"] for item in generation["clips"]),
        "run_provenance_snapshots_intact": all(digest(ROOT / entry["snapshot"]) == entry["sha256"] for entry in manifest["snapshots"].values()),
        "all_formats_independently_verified": artifact["all_formats_passed"],
        "semantic_review_recorded_for_all": {c["clip_id"] for c in semantic["clips"]} == {c["clip_id"] for c in generation["clips"]}
            and all(c.get("observations") and c.get("coarse_motion_present") is not None for c in semantic["clips"]),
        "raw_checksums_and_frames": True, "per_clip_provenance_matches_run": True,
        "all_previews_valid_150_frames_30fps": True, "visual_evidence_unchanged": True}
    for clip in generation["clips"]:
        entry = json.loads((ROOT / clip["manifest"]).read_text())
        checks["per_clip_provenance_matches_run"] &= entry["run_manifest_sha256"] == digest(manifest_path)
        for name, value in entry["artifacts"].items():
            path = ROOT / value["path"]
            checks["raw_checksums_and_frames"] &= digest(path) == value["sha256"]
            if name == "qpos.csv":
                q = np.loadtxt(path, delimiter=",")
                checks["raw_checksums_and_frames"] &= q.shape == (150, 36) and bool(np.isfinite(q).all()) and bool(np.max(np.abs(np.linalg.norm(q[:, 3:7], axis=1) - 1)) < 1e-3)
            else:
                with np.load(path, allow_pickle=False) as arrays:
                    checks["raw_checksums_and_frames"] &= arrays["posed_joints"].shape == (150, 34, 3) and all(bool(np.isfinite(arrays[k]).all()) for k in arrays.files)
    for clip in artifact["clips"]:
        video = ROOT / clip["preview"]
        result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                 "stream=nb_frames,r_frame_rate", "-of", "json", str(video)], capture_output=True, text=True, check=True)
        stream = json.loads(result.stdout)["streams"][0]
        checks["all_previews_valid_150_frames_30fps"] &= stream["nb_frames"] == "150" and stream["r_frame_rate"] == "30/1" and digest(video) == clip["preview_sha256"]
    for clip in semantic["clips"]:
        checks["visual_evidence_unchanged"] &= digest(ROOT / clip["contact_sheet"]) == clip["contact_sheet_sha256"]
    result = {"scope": "stage_2_generation_and_documented_semantic_review_only", "run_id": generation["run_id"],
        "checks": checks, "passed": all(checks.values()),
        "coarse_motion_present_count": sum(c["coarse_motion_present"] for c in semantic["clips"]),
        "all_prompt_details_guaranteed": False, "physics_tracking_validated": False,
        "quality_followups": semantic["quality_followups"]}
    (run / "stage_assessment.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "reports/stage_02_readiness.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
