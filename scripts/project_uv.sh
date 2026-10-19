#!/bin/sh
# Run the project package manager with all its local files under the repository.
set -eu
project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export UV_CACHE_DIR="$project_root/.tools/uv-cache"
export UV_PYTHON_INSTALL_DIR="$project_root/.tools/python"
if [ -x "$project_root/.tools/bootstrap/bin/uv" ]; then
    uv_command="$project_root/.tools/bootstrap/bin/uv"
elif command -v uv >/dev/null 2>&1; then
    uv_command=$(command -v uv)
else
    printf '%s\n' 'uv is missing; follow docs/20_environnement_python.md' >&2
    exit 1
fi
exec "$uv_command" --project "$project_root" "$@"
