"""Read-only environment inspection; write only the selected workspace report."""

import argparse
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ["torch", "numpy", "mujoco", "isaacsim", "isaaclab", "rsl-rl-lib", "transformers", "huggingface-hub"]


def run(command, timeout=20):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()[-1500:]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": None, "stdout": "", "stderr": str(error)}


def python_metadata(executable):
    if not executable.is_file():
        return {"exists": False}
    code = """import importlib.metadata as m,json,sys
v={}
for name in %r:
    try: v[name]=m.version(name)
    except m.PackageNotFoundError: v[name]=None
print(json.dumps({'python':sys.version.split()[0],'packages':v}))
""" % PACKAGES
    result = run([str(executable), "-I", "-c", code])
    if result["returncode"] == 0:
        return {"exists": True, **json.loads(result["stdout"])}
    return {"exists": True, "error": result["stderr"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reports/environment.json")
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    project = json.loads((ROOT / "configs/project.json").read_text())
    env = project["environment"]
    disk = shutil.disk_usage(ROOT)
    hf_dir = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
    token_path = Path(os.environ.get("HF_TOKEN_PATH", str(hf_dir / "token")))
    token_available = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")) or token_path.is_file()
    python_paths = {
        "tools": Path(os.environ.get("G1_TOOLS_PYTHON", env["tools_python"])),
        "isaaclab": Path(os.environ.get("G1_ISAACLAB_PYTHON", env["isaaclab_python"])),
        "kimodo": ROOT / env["generation_venv"] / "bin/python",
        "mujoco": ROOT / env["mujoco_venv"] / "bin/python",
    }
    interpreters = {name: python_metadata(path) for name, path in python_paths.items()}
    gpu = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"])
    prerequisites = []
    def evidence(name):
        path = ROOT / "reports" / name
        return json.loads(path.read_text()) if path.exists() else {}
    if disk.free / 2**30 < env["minimum_free_disk_gib_for_full_setup"]:
        prerequisites.append("disk_below_provisional_full_setup_budget")
    if gpu["returncode"] != 0:
        prerequisites.append("gpu_not_visible_in_this_execution_context")
    if not token_available:
        prerequisites.append("hf_token_not_found_in_checked_locations")
    if not interpreters["kimodo"]["exists"]:
        prerequisites.append("kimodo_environment_missing")
    if not interpreters["mujoco"].get("packages", {}).get("mujoco"):
        prerequisites.append("mujoco_environment_missing_or_unusable")
    if not interpreters["isaaclab"].get("packages", {}).get("isaacsim"):
        prerequisites.append("isaacsim_package_missing_or_unusable")
    if not evidence("hf_access.json").get("access_verified"):
        prerequisites.append("hf_gated_repository_authorization_unverified")
    if not evidence("kimodo_smoke.json").get("passed"):
        prerequisites.append("kimodo_checkpoint_cuda_execution_not_verified")
    if not evidence("text_encoder_smoke.json").get("passed"):
        prerequisites.append("cpu_text_encoder_loading_and_embedding_not_verified")
    if not evidence("official_generation_smoke.json").get("passed"):
        prerequisites.append("official_text_to_g1_generation_not_verified")
    if not evidence("g1_mujoco_validation.json").get("passed") or not evidence("g1_isaac_validation.json").get("passed"):
        prerequisites.append("g1_asset_and_backend_contract_not_verified")
    memory = {}
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        values = dict(line.split(":", 1) for line in meminfo.read_text().splitlines())
        memory = {key: round(int(values[key].split()[0]) / 2**20, 2) for key in ["MemTotal", "MemAvailable"]}
    report = {
        "timestamp": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "scope": "environment_metadata_only_no_model_or_simulator_initialization",
        "platform": platform.platform(),
        "memory_gib": memory,
        "disk": {"free_gib": round(disk.free / 2**30, 2), "total_gib": round(disk.total / 2**30, 2), "provisional_full_setup_free_gib": env["minimum_free_disk_gib_for_full_setup"]},
        "gpu": gpu,
        "interpreters": interpreters,
        "repository": {"version": project["environment"].get("isaaclab_version", "2.3.2"), "git_head": run(["git", "-C", str(ROOT.parents[1]), "rev-parse", "HEAD"])["stdout"]},
        "hf_token_available": token_available,
        "hf_repository_access": "verified_in_hf_access_report" if evidence("hf_access.json").get("access_verified") else "unverified",
        "build_tools": {name: shutil.which(name) for name in ["cmake", "g++", "ninja", "git"]},
        "full_pipeline_ready": not prerequisites,
        "pending_prerequisites": prerequisites,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Environment report: {args.output}")
    print(f"Free disk: {report['disk']['free_gib']} GiB; GPU visible: {gpu['returncode'] == 0}; HF token available: {token_available}")
    print(f"Full pipeline prerequisite readiness: {not prerequisites}")
    print("Pending: " + ", ".join(prerequisites))
    return 1 if args.require_ready and prerequisites else 0


if __name__ == "__main__":
    raise SystemExit(main())
