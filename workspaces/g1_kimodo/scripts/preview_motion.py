"""Render source qpos through MuJoCo FK; this preview uses no control policy."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    csv = args.csv.resolve()
    prefix = (args.output_dir.resolve() / csv.stem) if args.output_dir else csv.with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    qpos = np.loadtxt(csv, delimiter=",")
    if qpos.ndim != 2 or qpos.shape[1] != 36 or not np.isfinite(qpos).all():
        raise ValueError("Preview requires finite [frames,36] G1 source qpos")
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/g1/g1_scene.xml"))
    names = json.loads((ROOT / "configs/motion_contract.json").read_text())["csv"]["joint_names"]
    actual = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt)]
    if actual != names:
        raise ValueError("MuJoCo qpos joint order differs from source CSV")
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.5, 125, -15
    width, height = 960, 720
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    model.vis.headlight.ambient[:] = [0.4, 0.4, 0.4]
    model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
    model.geom_rgba[1:, :] = [0.4, 0.65, 0.85, 1]
    model.geom_rgba[0, :] = [0.2, 0.23, 0.26, 1]
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[:] = 1  # Canonical model retains collision primitives only.
    output = prefix.with_suffix(".mp4")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(args.fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-crf", "20", "-pix_fmt", "yuv420p", str(output)]
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for index, frame in enumerate(qpos):
                data.qpos[:] = frame
                mujoco.mj_forward(model, data)
                camera.lookat[:] = [*frame[:2], 0.75]
                renderer.update_scene(data, camera=camera, scene_option=scene_option)
                pixels = renderer.render()
                if index == 0:
                    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                                    "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-i", "pipe:0",
                                    "-frames:v", "1", str(prefix.with_suffix(".png"))],
                                   input=pixels.tobytes(), check=True)
                process.stdin.write(pixels.tobytes())
            process.stdin.close()
            stderr = process.stderr.read().decode()
            if process.wait() != 0:
                raise RuntimeError(f"ffmpeg failed: {stderr}")
        except BaseException:
            process.kill()
            process.wait()
            raise
    report = {
        "scope": "mujoco_kinematic_qpos_preview_no_policy_or_physics_rollout",
        "csv_sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
        "video": str(output), "video_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "frames": len(qpos), "fps": args.fps, "resolution": [width, height],
    }
    prefix.with_suffix(".preview.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
