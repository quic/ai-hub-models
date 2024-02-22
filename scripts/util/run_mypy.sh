#!/usr/bin/env bash
# shellcheck source=/dev/null

REPO_ROOT=$(git rev-parse --show-toplevel)

. "${REPO_ROOT}/scripts/util/common.sh"

set_strict_mode


cd "$(dirname "$0")/../.."

venv="${VENV_PATH:-qaihm-dev}"
echo "Activating venv in ${venv}"
source "${venv}/bin/activate"

paths=(qai_hub_models)
for path in "${paths[@]}"; do
    pathToCheck="${path}"
    echo "Running mypy on ${pathToCheck}"
    mypy --warn-unused-configs --config-file="${REPO_ROOT}/mypy.ini" "${pathToCheck}"
done
