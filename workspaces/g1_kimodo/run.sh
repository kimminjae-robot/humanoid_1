#!/usr/bin/env bash
set -euo pipefail
workspace_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tools_python="${G1_TOOLS_PYTHON:-python3.11}"
portable_root="$(cd -- "$workspace_dir/../.." && pwd)"
mujoco_python="${G1_MUJOCO_PYTHON:-$portable_root/.venv/bin/python}"
generation_python="${KIMODO_GENERATION_PYTHON:-$workspace_dir/.venvs/kimodo/bin/python}"
export HF_HOME="${G1_HF_HOME:-$workspace_dir/cache/huggingface}"
export CHECKPOINT_DIR="$workspace_dir/checkpoints"
export TEXT_ENCODER_DEVICE=cpu
export TEXT_ENCODER_MODE=local
command_name="${1:-status}"
if [[ $# -gt 0 ]]; then shift; fi
case "$command_name" in
  status) cat "$workspace_dir/STATUS.md" ;;
  plan) cat "$workspace_dir/docs/PLAN.md" ;;
  check) exec "$tools_python" -I "$workspace_dir/scripts/check_workspace.py" "$@" ;;
  doctor) exec "$tools_python" -I "$workspace_dir/scripts/doctor.py" "$@" ;;
  bootstrap-mujoco) exec bash "$portable_root/setup.sh" mujoco ;;
  smoke-mujoco) exec "$mujoco_python" -I "$workspace_dir/scripts/smoke_mujoco.py" "$@" ;;
  prepare-g1) exec "$mujoco_python" -I "$workspace_dir/scripts/prepare_g1.py" "$@" ;;
  validate-g1-isaac) exec "$tools_python" -I "$workspace_dir/scripts/run_g1_validation.py" "$@" ;;
  smoke-kimodo) exec "$generation_python" -I "$workspace_dir/scripts/smoke_kimodo.py" "$@" ;;
  probe-hf) exec "$generation_python" -I "$workspace_dir/scripts/probe_hf.py" "$@" ;;
  download-official) exec "$generation_python" -I "$workspace_dir/scripts/download_official_encoder.py" "$@" ;;
  cache-official) exec "$generation_python" -I "$workspace_dir/scripts/cache_official_embeddings.py" "$@" ;;
  smoke-official) exec "$generation_python" -I "$workspace_dir/scripts/smoke_official_generation.py" "$@" ;;
  generate-baseline) exec "$generation_python" -I "$workspace_dir/scripts/generate_baseline.py" "$@" ;;
  review-baseline) exec "$mujoco_python" -I "$workspace_dir/scripts/review_baseline.py" "$@" ;;
  assess-stage2) exec "$mujoco_python" -I "$workspace_dir/scripts/assess_stage2.py" "$@" ;;
  prepare-references) exec "$mujoco_python" -I "$workspace_dir/scripts/prepare_references.py" "$@" ;;
  curate-references) exec "$mujoco_python" -I "$workspace_dir/scripts/curate_references.py" "$@" ;;
  finalize-references) exec "$mujoco_python" -I "$workspace_dir/scripts/finalize_references.py" "$@" ;;
  review-references) exec "$mujoco_python" -I "$workspace_dir/scripts/review_references.py" "$@" ;;
  validate-reference-math) exec "$mujoco_python" -I "$workspace_dir/scripts/validate_reference_math.py" "$@" ;;
  assess-stage3) exec "$mujoco_python" -I "$workspace_dir/scripts/assess_stage3.py" "$@" ;;
  validate-tracking-math) exec "$generation_python" -I "$workspace_dir/scripts/validate_tracking_math.py" "$@" ;;
  assess-stage4) exec "$mujoco_python" -I "$workspace_dir/scripts/assess_stage4.py" "$@" ;;
  verify-frozen-balance) exec "$generation_python" -I "$workspace_dir/scripts/verify_frozen_balance.py" "$@" ;;
  track) exec "$tools_python" -I "$workspace_dir/scripts/run_tracking.py" "$@" ;;
  sim2sim) exec "$mujoco_python" -I "$workspace_dir/scripts/sim2sim.py" "$@" ;;
  assess-stage5) exec "$mujoco_python" -I "$workspace_dir/scripts/assess_stage5.py" "$@" ;;
  review-sim2sim) exec "$mujoco_python" -I "$workspace_dir/scripts/review_sim2sim.py" "$@" ;;
  preview) exec "$mujoco_python" -I "$workspace_dir/scripts/preview_motion.py" "$@" ;;
  cache-qwen) exec "$generation_python" -I "$workspace_dir/scripts/cache_qwen_embeddings.py" "$@" ;;
  download-qwen) exec "$generation_python" -I "$workspace_dir/scripts/download_qwen.py" "$@" ;;
  smoke-qwen) exec "$generation_python" -I "$workspace_dir/scripts/smoke_qwen_generation.py" "$@" ;;
  assess-stage1) exec "$tools_python" -I "$workspace_dir/scripts/assess_stage1.py" "$@" ;;
  download-progress) exec "$tools_python" -I "$workspace_dir/scripts/download_progress.py" "$@" ;;
  *) echo "Usage: $0 {status|plan|check|doctor|bootstrap-mujoco|smoke-mujoco|prepare-g1|validate-g1-isaac|smoke-kimodo|probe-hf|download-official|cache-official|smoke-official|generate-baseline|review-baseline|assess-stage2|prepare-references|curate-references|finalize-references|review-references|validate-reference-math|assess-stage3|validate-tracking-math|verify-frozen-balance|track|assess-stage4|preview|download-qwen|cache-qwen|smoke-qwen|assess-stage1|download-progress}" >&2; exit 2 ;;
esac
