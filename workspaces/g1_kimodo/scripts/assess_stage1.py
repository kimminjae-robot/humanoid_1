"""Evaluate stage-1 evidence without claiming motion quality or policy success."""

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    path = ROOT / "reports" / name
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    asset = read("g1_asset_manifest.json")
    isaac = read("g1_isaac_validation.json")
    checks = {
        "storage_budget": shutil.disk_usage(ROOT).free >= 40 * 2**30,
        "kimodo_environment_freeze": (ROOT / "reports/kimodo_freeze.txt").exists(),
        "public_checkpoint_cuda_forward": bool(read("kimodo_smoke.json").get("passed")),
        "cpp_postprocessing_execution": bool(read("motion_correction_smoke.json").get("passed")),
        "mujoco_g1_physics": bool(read("g1_mujoco_validation.json").get("passed")),
        "isaac_g1_fk_parameters_collisions_physics": bool(isaac.get("passed")),
    }
    for key, path in [("generated_robot_sha256", ROOT / "assets/g1/g1_robot.xml"), ("generated_scene_sha256", ROOT / "assets/g1/g1_scene.xml")]:
        checks[key] = path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == asset.get(key)
    usd_files = isaac.get("usd_file_sha256", {})
    checks["usd_layer_hashes"] = bool(usd_files) and all((ROOT / "assets/g1/usd" / relative).exists() and
        hashlib.sha256((ROOT / "assets/g1/usd" / relative).read_bytes()).hexdigest() == digest for relative, digest in usd_files.items())
    target = read("action_target_contract_review.json")
    checks["tracking_target_range_contract"] = bool(target.get("all_actual_reference_targets_representable_with_zero_residual")) and \
        target.get("new_contract_sha256") == hashlib.sha256((ROOT / "configs/motion_contract.json").read_bytes()).hexdigest()
    qwen = {
        "real_projected_cpu_text_embeddings": bool(read("qwen_text_encoder_smoke.json").get("passed")) and read("qwen_text_encoder_smoke.json").get("device") == "cpu",
        "real_text_to_g1_csv_connectivity": bool(read("qwen_generation_smoke.json").get("passed")),
    }
    encoder = read("text_encoder_smoke.json")
    generation = read("official_generation_smoke.json")
    official = {"hf_access": bool(read("hf_access.json").get("access_verified")),
                "official_cpu_encoder": bool(encoder.get("passed")) and encoder.get("device") == "cpu" and encoder.get("encoder") == "official_llm2vec_llama3",
                "official_text_to_g1_csv_connectivity": bool(generation.get("passed")) and generation.get("encoder") == "official_llm2vec_llama3"}
    report = {
        "scope": "stage_1_setup_readiness_only",
        "common_checks": checks, "experimental_qwen_checks": qwen, "official_llama_checks": official,
        "experimental_route_passed": all(checks.values()) and all(qwen.values()),
        "official_route_passed": all(checks.values()) and all(official.values()),
        "motion_quality_validated": False, "physical_tracking_validated": False,
        "llama_qwen_quality_equivalence_validated": False,
    }
    (ROOT / "reports/stage_01_readiness.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["experimental_route_passed"] or report["official_route_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
