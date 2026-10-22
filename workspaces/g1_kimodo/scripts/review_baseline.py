"""Render every generated baseline and measure descriptive semantic evidence."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    generation = json.loads((ROOT / "reports/stage_02_generation.json").read_text())
    run = ROOT / "runs" / generation["run_id"]
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/g1/g1_scene.xml"))
    data = mujoco.MjData(model)
    bodies = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in [
        "pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link", "left_shoulder_pitch_link",
        "right_shoulder_pitch_link", "left_ankle_roll_link", "right_ankle_roll_link"]}
    if min(bodies.values()) < 0:
        raise ValueError("Missing expected G1 semantic body names")
    results = []
    for index, clip in enumerate(generation["clips"]):
        if clip["status"] != "generated":
            results.append({"clip_id": clip["clip_id"], "format_passed": False, "semantic_review": "generation_failed"})
            continue
        manifest = json.loads((ROOT / clip["manifest"]).read_text())
        csv = ROOT / manifest["artifacts"]["qpos.csv"]["path"]
        npz = ROOT / manifest["artifacts"]["motion.npz"]["path"]
        if any(digest(ROOT / item["path"]) != item["sha256"] for item in manifest["artifacts"].values()):
            raise ValueError(f"Source checksum mismatch: {clip['clip_id']}")
        qpos = np.loadtxt(csv, delimiter=",")
        local_hands = {side: [] for side in ["left", "right"]}
        above_shoulders = {side: [] for side in ["left", "right"]}
        feet = {side: [] for side in ["left", "right"]}
        for frame in qpos:
            data.qpos[:] = frame
            mujoco.mj_forward(model, data)
            pelvis = data.xpos[bodies["pelvis"]]
            inverse_rotation = data.xmat[bodies["pelvis"]].reshape(3, 3).T
            for side in ["left", "right"]:
                hand = data.xpos[bodies[f"{side}_wrist_yaw_link"]]
                shoulder = data.xpos[bodies[f"{side}_shoulder_pitch_link"]]
                local_hands[side].append(inverse_rotation @ (hand - pelvis))
                above_shoulders[side].append(float(hand[2] - shoulder[2]))
                feet[side].append(data.xpos[bodies[f"{side}_ankle_roll_link"]].copy())
        metrics = {
            "root_displacement_xyz_m": (qpos[-1, :3] - qpos[0, :3]).tolist(),
            "max_horizontal_root_distance_from_start_m": float(np.linalg.norm(qpos[:, :2] - qpos[0, :2], axis=1).max()),
            "root_height_range_m": [float(qpos[:, 2].min()), float(qpos[:, 2].max())],
            "max_quaternion_norm_error": float(np.max(np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1))),
        }
        for side in ["left", "right"]:
            hands = np.array(local_hands[side]);foot = np.array(feet[side])
            metrics[f"{side}_hand_root_frame_range_xyz_m"] = np.ptp(hands, axis=0).tolist()
            metrics[f"{side}_hand_root_frame_path_length_m"] = float(np.linalg.norm(np.diff(hands, axis=0), axis=1).sum())
            metrics[f"{side}_hand_above_shoulder_fraction"] = float((np.array(above_shoulders[side]) > 0).mean())
            metrics[f"{side}_hand_max_height_above_shoulder_m"] = float(max(above_shoulders[side]))
            metrics[f"{side}_foot_horizontal_range_xy_m"] = np.ptp(foot[:, :2], axis=0).tolist()
        with np.load(npz, allow_pickle=False) as arrays:
            format_checks = {"150_by_36": qpos.shape == (150, 36), "finite_csv": bool(np.isfinite(qpos).all()),
                "unit_quaternions": metrics["max_quaternion_norm_error"] < 1e-3,
                "unbatched_npz_pose": arrays["posed_joints"].shape == (150, 34, 3),
                "finite_npz": all(bool(np.isfinite(arrays[key]).all()) for key in arrays.files)}
        preview = run / "previews" / clip["clip_id"]
        subprocess.run([sys.executable, "-I", str(ROOT / "scripts/preview_motion.py"), str(csv),
                        "--fps", "30", "--output-dir", str(preview)], check=True, stdout=subprocess.DEVNULL)
        indices = np.linspace(0, 149, 10).round().astype(int).tolist()
        select = "+".join(f"eq(n\\,{i})" for i in indices)
        sheet = preview / "frames.png"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(preview / "qpos.mp4"),
                        "-vf", f"select={select},scale=320:240,tile=5x2", "-frames:v", "1", str(sheet)], check=True)
        result = {**clip, "format_checks": format_checks, "format_passed": all(format_checks.values()),
            "semantic_metrics": metrics, "semantic_review": "pending_visual_review",
            "preview": str((preview / "qpos.mp4").relative_to(ROOT)), "preview_sha256": digest(preview / "qpos.mp4"),
            "contact_sheet": str(sheet.relative_to(ROOT)), "contact_sheet_frame_indices": indices}
        results.append(result)
        print(f"Reviewed artifacts {index+1}/15: {clip['clip_id']}", flush=True)
    report = {"scope": "stage_2_format_and_kinematic_semantic_evidence_not_dynamic_tracking_or_stage_3_QA",
        "run_id": generation["run_id"], "clips": results, "all_formats_passed": len(results) == 15 and all(c["format_passed"] for c in results),
        "semantic_review": "pending_visual_review", "final_seeds_used": False}
    (run / "artifact_review.json").write_text(json.dumps(report, indent=2) + "\n")
    (ROOT / "reports/stage_02_artifacts.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["all_formats_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
