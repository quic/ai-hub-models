#!/usr/bin/env bash

REPO_ROOT=$(git rev-parse --show-toplevel)

# Load helpers
. "${REPO_ROOT}/scripts/util/common.sh"
. "${REPO_ROOT}/scripts/util/github.sh"

set_strict_mode


print_help() {
  echo "pytest_with_coverage.sh --name=[...] --omit=[...] PYTEST_ARGS"
  echo ""
  echo "--name=[...]  Test report name."
  echo "--omit=[...]  Comma-seprated directories."
}

NAME="unnamed"
OMIT=""

for i in "$@"; do
  case $i in
    --name=*)
      NAME="${i#*=}"
      shift
      ;;
    --omit=*)
      OMIT="${i#*=}"
      shift
      ;;
    -h|--help)
      print_help
      shift
      exit 0
      ;;
    *)
      ;;
  esac
done


COV_CONFIG="$(mktemp).coveragerc"
COVERAGE_DIR="${REPO_ROOT}/build/test-coverage"
RESULTS_DIR="${REPO_ROOT}/build/test-results"

mkdir -p "$COVERAGE_DIR" "$RESULTS_DIR"

DATA_FILE="${COVERAGE_DIR}/.coverage.${NAME}"
JUNIT_REPORT="${RESULTS_DIR}/${NAME}.xml"

python "${REPO_ROOT}/scripts/util/make_coverage_config.py" \
    --base "${REPO_ROOT}/.coveragerc" \
    --data_file "${DATA_FILE}" \
    --omit "${OMIT}" \
    --output "${COV_CONFIG}"

# Coverage can be turned off by passing `--no-cov` as part of $@
pytest \
    -rxXs \
    -p no:warnings \
    --junitxml="${JUNIT_REPORT}" \
    --durations=20 \
    --durations-min=0.5 \
    --cov \
    --cov-report= \
    --cov-config="${COV_CONFIG}" \
    "$@"
