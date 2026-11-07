#!/bin/sh
set -eu
project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export UV_CACHE_DIR="$project_root/.tools/uv-cache"
export UV_PYTHON_INSTALL_DIR="$project_root/.tools/python"
export HF_HOME="$project_root/.tools/hf-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
if [ -x "$project_root/.tools/bootstrap/bin/uv" ]; then
    uv_command="$project_root/.tools/bootstrap/bin/uv"
else
    uv_command=uv
fi
exec "$uv_command" --project "$project_root/environments/lora" "$@"
