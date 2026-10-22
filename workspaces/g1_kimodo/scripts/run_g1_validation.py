"""Bound Kit teardown and require a fresh, passing physics report."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def main():
    run_dir = ROOT / "runs" / (datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S") + "_stage01_g1_asset")
    run_dir.mkdir(parents=True)
    report_path = ROOT / "reports/g1_isaac_validation.json"
    if report_path.exists():
        shutil.copy2(report_path, run_dir / "previous_report.json")
        report_path.unlink()
    log_path = ROOT / "reports/g1_isaac_validation.log"
    command = [os.environ.get('G1_ISAACLAB_PYTHON', sys.executable), "-u", str(ROOT / "scripts/validate_g1_isaac.py"), "--headless", *sys.argv[1:]]
    start, first_report_seen = time.monotonic(), None
    teardown = "normal_exit"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            now = time.monotonic()
            if report_path.exists() and first_report_seen is None:
                first_report_seen = now
            if now - start > 150 or (first_report_seen is not None and now - first_report_seen > 15):
                teardown = "terminated_stalled_kit_cleanup_after_report" if first_report_seen else "timed_out_before_report"
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                break
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
    shutil.copy2(log_path, run_dir / "isaac.log")
    report = json.loads(report_path.read_text()) if report_path.exists() else {"passed": False, "error": "No fresh validation report"}
    if report_path.exists():
        shutil.copy2(report_path, run_dir / "g1_isaac_validation.json")
    result = {"command": command, "returncode": process.returncode, "teardown": teardown,
              "fresh_report_passed": bool(report.get("passed")), "run_dir": str(run_dir), "elapsed_s": time.monotonic() - start}
    (run_dir / "execution.json").write_text(json.dumps(result, indent=2) + "\n")
    (ROOT / "reports/g1_isaac_execution.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(json.dumps(report.get("checks", {}), indent=2))
    return 0 if result["fresh_report_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
