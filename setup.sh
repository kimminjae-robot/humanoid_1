#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${KIMODO_TOOLS_PYTHON:-python3.11}"
mode="${1:-mujoco}"
case "$mode" in
  mujoco|policy)
    if [[ ! -x "$root/.venv/bin/python" ]]; then "$python_bin" -m venv "$root/.venv"; fi
    "$root/.venv/bin/python" -I -m pip install --disable-pip-version-check -r "$root/requirements/mujoco.txt"
    if [[ "$mode" == policy ]]; then
      "$root/.venv/bin/python" -I -m pip install 'torch==2.7.0' --index-url https://download.pytorch.org/whl/cpu
    fi
    "$root/.venv/bin/python" -I "$root/tools/verify_package.py"
    ;;
  generator)
    "$python_bin" -I "$root/tools/fetch_kimodo.py"
    generation_dir="$root/workspaces/g1_kimodo/.venvs/kimodo"
    if [[ ! -x "$generation_dir/bin/python" ]]; then "$python_bin" -m venv "$generation_dir"; fi
    "$generation_dir/bin/python" -I -m pip install 'torch==2.7.0' --index-url https://download.pytorch.org/whl/cu128
    "$generation_dir/bin/python" -I -m pip install -r "$root/requirements/generation.txt"
    "$generation_dir/bin/python" -I -m pip install --no-deps -e "$root/workspaces/g1_kimodo/third_party/kimodo"
    printf '%s\n' 'Next: follow docs/REPRODUCE.md for HF login and pinned model downloads.'
    ;;
  *) printf '%s\n' 'Usage: bash setup.sh {mujoco|policy|generator}' >&2; exit 2 ;;
esac
