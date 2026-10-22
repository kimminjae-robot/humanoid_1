"""Render all processed references and fixed-seed raw/processed comparisons."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from reference_core import interpolate,rotations,wxyz


def main():
    dataset=json.loads((ROOT/"reports/stage_03_dataset.json").read_text());run=ROOT/"runs"/dataset["run_id"];results=[]
    for c in dataset["clips"]:
        if not c["accepted"]:continue
        csv=ROOT/c["qpos_csv"];preview=run/"previews"/c["clip_id"]
        subprocess.run([sys.executable,"-I",str(ROOT/"scripts/preview_motion.py"),str(csv),"--fps","50","--output-dir",str(preview)],check=True,stdout=subprocess.DEVNULL)
        with np.load(ROOT/c["bundle"],allow_pickle=False) as b:
            frames=len(b["timestamps_s"]);mapped=b["source_timestamps_s"]
        indices=np.linspace(0,frames-1,10).round().astype(int).tolist();select="+".join(f"eq(n\\,{i})" for i in indices)
        subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(preview/"qpos.mp4"),"-vf",f"select={select},scale=320:240,tile=5x2","-frames:v","1",str(preview/"frames.png")],check=True)
        result={"clip_id":c["clip_id"],"preview":str((preview/"qpos.mp4").relative_to(ROOT)),"frames":frames,
            "preview_sha256":hashlib.sha256((preview/"qpos.mp4").read_bytes()).hexdigest(),"contact_sheet":str((preview/"frames.png").relative_to(ROOT)),"contact_sheet_sha256":hashlib.sha256((preview/"frames.png").read_bytes()).hexdigest(),"frame_indices":indices}
        if c["seed"]==101:
            rawm=json.loads((ROOT/c["source_manifest"]).read_text());raw=np.loadtxt(ROOT/rawm["artifacts"]["qpos.csv"]["path"],delimiter=',')
            aligned=interpolate(raw,np.arange(len(raw))/30,mapped);transform=Rotation.from_euler('z',-c["processing"]["constant_heading_removed_rad"])
            aligned[:,:3]=transform.apply(aligned[:,:3]-np.array(c["processing"]["constant_origin_xyz_m"]));aligned[:,3:7]=wxyz(transform*rotations(aligned[:,3:7]))
            raw_dir=preview/"source_time_aligned";raw_dir.mkdir(exist_ok=True);np.savetxt(raw_dir/"source.csv",aligned,delimiter=',')
            subprocess.run([sys.executable,"-I",str(ROOT/"scripts/preview_motion.py"),str(raw_dir/"source.csv"),"--fps","50"],check=True,stdout=subprocess.DEVNULL)
            output=run/(c["motion_id"]+"_raw_vs_reference_seed101.mp4")
            filters="[0:v]scale=480:360,drawtext=text=raw at source time:x=8:y=12:fontcolor=white:fontsize=17[a];[1:v]scale=480:360,drawtext=text=processed reference:x=8:y=12:fontcolor=white:fontsize=17[b];[a][b]hstack=inputs=2[v]"
            subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(raw_dir/"source.mp4"),"-i",str(preview/"qpos.mp4"),"-filter_complex",filters,"-map","[v]","-c:v","libx264","-crf","20","-pix_fmt","yuv420p",str(output)],check=True)
            result["comparison"]=str(output.relative_to(ROOT))
        results.append(result);print(f"Rendered {len(results)}/{dataset['accepted']}: {c['clip_id']}",flush=True)
    report={"run_id":dataset["run_id"],"scope":"kinematic_source_qpos_previews_no_policy_or_physics_rollout","clips":results,"visual_review":"pending"}
    (run/"preview_manifest.json").write_text(json.dumps(report,indent=2)+"\n");(ROOT/"reports/stage_03_previews.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
