#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
portable_root="$(cd -- "$root/../.." && pwd)"
python_bin="${K1_KIMODO_PYTHON:-$portable_root/.venv/bin/python}"
generation_python="${KIMODO_GENERATION_PYTHON:-$root/../g1_kimodo/.venvs/kimodo/bin/python}"
command_name="${1:-status}"
if [[ $# -gt 0 ]]; then shift; fi
case "$command_name" in
 generate) exec "$generation_python" -I "$root/scripts/generate_motion.py" "$@";;
 process) exec "$python_bin" -I "$root/scripts/process_motion.py" "$@";;
 condition-standing) exec "$python_bin" -I "$root/scripts/condition_standing.py" "$@";;
 clear-arm-collisions) exec "$python_bin" -I "$root/scripts/clear_arm_collisions.py" "$@";;
 status) cat "$root/STATUS.md";;
 prepare) exec "$python_bin" -I "$root/scripts/prepare_robot.py" "$@";;
 validate-robot) exec "$python_bin" -I "$root/scripts/validate_robot.py" "$@";;
 improve-walk) exec "$python_bin" -I "$root/scripts/improve_walk.py" "$@";;
 retarget) exec "$python_bin" -I "$root/scripts/retarget.py" "$@";;
 evaluate) exec "$python_bin" -I "$root/scripts/evaluate.py" "$@";;
 review) exec "$python_bin" -I "$root/scripts/review.py" "$@";;
 assess) exec "$python_bin" -I "$root/scripts/assess.py" "$@";;
 *) echo 'Usage: run.sh {status|generate|process|condition-standing|clear-arm-collisions|prepare|validate-robot|improve-walk|retarget|evaluate|review|assess}' >&2;exit 2;;
esac
