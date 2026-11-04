#!/bin/sh
# Isolated PyTorch profile; never synchronizes the classical environment.
set -eu
project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export UV_CACHE_DIR="$project_root/.tools/uv-cache"
export UV_PYTHON_INSTALL_DIR="$project_root/.tools/python"
if [ -x "$project_root/.tools/bootstrap/bin/uv" ]; then
    uv_command="$project_root/.tools/bootstrap/bin/uv"
else
    uv_command=uv
fi
exec "$uv_command" --project "$project_root/environments/bigru" "$@"
