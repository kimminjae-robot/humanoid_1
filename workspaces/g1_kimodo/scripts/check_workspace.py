"""Check staged-project configuration without simulator dependencies."""

import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    project = json.loads((ROOT / "configs/project.json").read_text())
    motion = json.loads((ROOT / "configs/motion_contract.json").read_text())
    prompts = json.loads((ROOT / "configs/prompts.json").read_text())
    evaluation = json.loads((ROOT / "configs/evaluation.json").read_text())
    progress = json.loads((ROOT / "state/progress.json").read_text())
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    joints = motion["csv"]["joint_names"]
    check(len(joints) == len(set(joints)) == 29, "Expected 29 unique source joint names")
    check(motion["csv"]["qpos_size"] == 7 + len(joints), "qpos must contain free root + joints")
    check(motion["csv"]["qvel_size"] == 6 + len(joints), "qvel must contain free root twist + joints")
    inventory = json.loads((ROOT / "reports/source_inventory.json").read_text())
    check(joints == inventory["g1_csv_contract"]["observed_joint_names"], "Joint order differs from inspected official XML")
    check(motion["csv"]["quaternion_order"] == "wxyz", "MuJoCo CSV must use wxyz")
    check(not motion["csv"]["header"] and not motion["csv"]["frame_index_column"], "Unexpected CSV header/index")
    check(prompts["fps"] == motion["csv"]["fps"] == 30, "Source frame rate mismatch")
    controller = motion["controller"]
    check(controller["physics_hz"] == controller["control_hz"] * controller["decimation"], "Invalid control decimation")
    check(controller["decimation"] > 0, "Decimation must be positive")
    if (ROOT / "configs/g1_actuators.json").exists():
        for field in ["joint_default_positions", "kp", "kd", "action_scale", "effort_limits", "velocity_limits", "joint_position_limits"]:
            check(set(controller[field]) == set(joints), f"Incomplete 29-joint controller field: {field}")
        check(controller["action_target_convention"] == "reference_plus_scaled_residual", "Unexpected tracking target convention")
        check(controller["position_target_clipping"] == "joint_position_limits", "Position targets must honor joint limits")
        check(controller["reference_target_update_hz"] == controller["control_hz"], "Reference target update rate mismatch")
        check(controller["target_hold_physics_steps"] == controller["decimation"], "Target hold duration mismatch")
        check(controller["torque_update_hz"] == controller["physics_hz"], "PD torque must update each physics step")
    maps = controller.get("backend_joint_maps")
    if maps:
        check(sorted(maps["source_to_isaac_indices"]) == list(range(29)), "Invalid source-to-Isaac permutation")
        check([maps["isaac"][i] for i in maps["source_to_isaac_indices"]] == joints, "Isaac name permutation mismatch")
    for label, path in [("mjcf", ROOT / "assets/g1/g1_robot.xml"), ("scene", ROOT / "assets/g1/g1_scene.xml")]:
        if path.exists():
            check(hashlib.sha256(path.read_bytes()).hexdigest() == controller["robot_hashes"][label], f"Asset hash mismatch: {label}")
    for relative, digest in (controller.get("robot_hashes") or {}).get("usd_files", {}).items():
        path = ROOT / "assets/g1/usd" / relative
        check(path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"USD layer hash mismatch: {relative}")
    seen = set()
    for split, seeds in prompts["seed_splits"].items():
        check(bool(seeds), f"Empty seed split: {split}")
        check(len(seeds) == len(set(seeds)), f"Duplicate seeds in {split}")
        check(not seen.intersection(seeds), f"Generation seed overlap: {split}")
        seen.update(seeds)
    prompt_ids = [p["id"] for p in prompts["prompts"]]
    check(len(prompt_ids) == len(set(prompt_ids)), "Duplicate motion IDs")
    check(set(project["scope"]["baseline_motion_ids"]).issubset(prompt_ids), "Missing baseline prompts")
    check(evaluation["episodes_per_clip"] == len(evaluation["evaluation_seeds"]), "Episode/seed count mismatch")
    check(len(set(evaluation["evaluation_seeds"])) == len(evaluation["evaluation_seeds"]), "Duplicate evaluation seeds")
    check(evaluation["full_clip_required"], "Dynamic evaluation must cover the entire clip")
    stages = progress["stages"]
    check([s["id"] for s in stages] == list(range(7)), "Expected stages 0 through 6")
    check(all(s["status"] in {"pending", "in_progress", "complete", "blocked"} for s in stages), "Invalid stage status")
    completed = [s["id"] for s in stages if s["status"] == "complete"]
    check(completed == list(range(len(completed))), "Completed stages must form a prefix")
    for stage in stages:
        if stage["status"] == "complete":
            check(bool(stage["report"]) and (ROOT / stage["report"]).is_file(), f"Stage {stage['id']} has no report")
    for path in ["README.md", "STATUS.md", "docs/PLAN.md", "docs/ANALYSIS.md", "docs/CONTRACTS.md", "docs/STAGE_REPORT_TEMPLATE.md"]:
        check((ROOT / path).is_file(), f"Missing {path}")
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("PASS: workspace config, 29-joint CSV contract, timing, seed splits, and staged progress")
    print("Scope: configuration consistency only; controller fields and G1 physics are not validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
