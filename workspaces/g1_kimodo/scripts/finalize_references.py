"""Package QA-passing references without overwriting source or processing outputs."""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from reference_core import rotations


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary=json.loads((ROOT/"reports/stage_03_curated.json").read_text())
    run=ROOT/"runs"/summary["run_id"]
    dest=ROOT/"data/processed"/summary["run_id"]/"final"
    dest.mkdir(parents=True,exist_ok=False)
    contract=json.loads((ROOT/"configs/motion_contract.json").read_text())
    isaac=json.loads((ROOT/"reports/g1_isaac_validation.json").read_text())
    with np.load(ROOT/summary["clips"][0]["bundle"],allow_pickle=False) as first:
        body_names=first["body_names"].tolist()
    maps={"joint_names_source":contract["csv"]["joint_names"],"joint_names_isaac":isaac["isaac_joint_names"],
        "body_names_source":body_names,"body_names_isaac":isaac["isaac_body_names"],
        "source_to_isaac_joint_indices":[isaac["isaac_joint_names"].index(n) for n in contract["csv"]["joint_names"]],
        "source_to_isaac_body_indices":[isaac["isaac_body_names"].index(n) for n in body_names],
        "source_to_mujoco_joint_indices":list(range(29)),"source_to_mujoco_body_indices":list(range(30)),
        "robot_hashes":contract["controller"]["robot_hashes"],"source":"stage_1_actual_backend_name_lookup"}
    (dest/"backend_maps.json").write_text(json.dumps(maps,indent=2)+"\n")
    clips=[]
    for clip in summary["clips"]:
        entry={k:clip[k] for k in ["clip_id","motion_id","split","seed","source_sha256","source_manifest"]}
        entry.update(accepted=clip["metrics"]["passed"],rejection_reasons=[k for k,v in clip["metrics"]["checks"].items() if not v],metrics=clip["metrics"],processing=clip["processing"])
        entry.update({key:clip[key] for key in ['processing_run_id','processing_profile_snapshot','processing_profile_sha256']})
        if entry["accepted"]:
            folder=dest/clip["clip_id"];folder.mkdir()
            with np.load(ROOT/clip["bundle"],allow_pickle=False) as data:
                bundle={key:data[key] for key in data.files}
            angular_local=rotations(bundle["root_quat_world_wxyz"]).inv().apply(bundle["root_ang_vel_world_rad_s"])
            bundle["qvel_mujoco"]=np.c_[bundle["root_lin_vel_world_m_s"],angular_local,bundle["joint_vel_rad_s"]]
            bundle["qvel_mujoco_convention"]=np.array("root_linear_world_xyz; root_angular_body_xyz; 29_joint_rates")
            bundle["geometry_foot_contact_names"]=np.array(["left","right"])
            bundle["source_foot_contact_names"]=np.array(["left_ankle_roll_skel","left_toe_base","right_ankle_roll_skel","right_toe_base"])
            bundle["backend_maps_sha256"]=np.array(sha(dest/"backend_maps.json"))
            bundle["processing_manifest"]=np.array(json.dumps(clip["processing"]))
            np.savez_compressed(folder/"motion_bundle.npz",**bundle)
            shutil.copy2(ROOT/clip["qpos_csv"],folder/"qpos.csv")
            entry.update(bundle=str((folder/"motion_bundle.npz").relative_to(ROOT)),bundle_sha256=sha(folder/"motion_bundle.npz"),qpos_csv=str((folder/"qpos.csv").relative_to(ROOT)),qpos_sha256=sha(folder/"qpos.csv"))
            (folder/"manifest.json").write_text(json.dumps(entry,ensure_ascii=False,indent=2)+"\n")
        clips.append(entry)
    dataset={"scope":"kinematically_screened_reference_candidates_dynamic_tracking_not_tested","run_id":summary["run_id"],
        "fps":50,"clips":clips,"accepted_counts":summary["accepted_counts"],"source_candidates":len(clips),
        "accepted":sum(c["accepted"] for c in clips),"rejected":sum(not c["accepted"] for c in clips),
        "backend_maps":str((dest/"backend_maps.json").relative_to(ROOT)),"backend_maps_sha256":sha(dest/"backend_maps.json"),
        "split_rule":"retain original generation seed split; no final generation seeds used", "visual_review":"pending"}
    (dest/"dataset.json").write_text(json.dumps(dataset,ensure_ascii=False,indent=2)+"\n")
    (ROOT/"reports/stage_03_dataset.json").write_text(json.dumps(dataset,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"run_id":summary["run_id"],"accepted":dataset["accepted"],"rejected":dataset["rejected"],"dataset":str(dest/"dataset.json")},indent=2))


if __name__=="__main__":
    main()
