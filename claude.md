# Qualcomm AI Hub Models - Development Guide

This is the internal repository for Qualcomm AI Hub Models - a collection of ML models optimized for Qualcomm chipsets.

## Project Structure

```
qai_hub_models/
├── models/           # ~188 model implementations (each has model.py, app.py, demo.py, test.py, info.yaml)
├── configs/          # Configuration utilities
├── datasets/         # Dataset loaders for training/evaluation
├── evaluators/       # Accuracy evaluation classes
├── scorecard/        # Performance benchmarking
├── scripts/          # Internal tooling (codegen, autofill, etc.)
├── test/             # Shared test utilities
├── utils/            # Common utilities (base_model.py, image_processing.py, asset_loaders.py)
└── extern/           # External dependencies wrapped for safe import
scripts/              # Build, CI, and release tooling
```

## Code Quality

### Linting & Formatting
- **Ruff** is the primary linter/formatter (replaces flake8, isort, pyupgrade)
- Run `ruff check --fix` and `ruff format` before committing
- Key rules: E/F (pycodestyle/flakes), I (isort), D (docstrings - numpy style), PL (pylint), PT (pytest)

### Type Checking
- **mypy** with `ignore_missing_imports = true`
- All code in `qai_hub_models/` must pass mypy (except `models/_internal/`, LLM modules)
- Use type hints for function signatures

### Pre-commit Hooks
Always run `pre-commit install` after cloning. Hooks include:
- License header insertion (BSD-3)
- YAML validation, trailing whitespace, large file detection
- Ruff check + format
- mypy type checking
- pydoclint for configs/datasets docstrings

## Build & Test Workflow

### Setup
```bash
python scripts/build_and_test.py install_deps
source qaihm-dev/bin/activate
pre-commit install
```

### Common Commands
```bash
# Run codegen for a model
python qai_hub_models/scripts/run_codegen.py -m <model_id>

# Auto-fill info.yaml fields
python qai_hub_models/scripts/autofill_info_yaml.py -m <model_id>

# Run unit tests (use build_and_test.py to handle model dependencies automatically)
QAIHM_TEST_MODELS=<model_id> python scripts/build_and_test.py test_changed_models

# Install model dependencies first for remaining commands (see qai_hub_models/models/<model_id>/README.md)

# Export model to device
python -m qai_hub_models.models.<model_id>.export --target-runtime <tflite|onnx|qnn> --chipset <chipset>

# Run demo
python -m qai_hub_models.models.<model_id>.demo

# Run evaluation (only available for models with eval_datasets() defined)
python -m qai_hub_models.models.<model_id>.evaluate --help
```

### Verifying Correctness
Before submitting a PR, always run:
```bash
# Re-run codegen for affected models
python qai_hub_models/scripts/run_codegen.py -m <model_id>

# Required: Run all pre-commit hooks on all files
pre-commit run --all-files

# Run package unit tests
python scripts/build_and_test.py test_qaihm

# Run unit tests for affected models
QAIHM_TEST_MODELS=<model_id> python scripts/build_and_test.py test_changed_models
```

**If a model's architecture changed substantially**, also verify export.py and evaluate.py before merging:
```bash
# Install model dependencies first (see qai_hub_models/models/<model_id>/README.md)
python -m qai_hub_models.models.<model_id>.export --target-runtime tflite --chipset qualcomm-snapdragon-8gen3
python -m qai_hub_models.models.<model_id>.evaluate  # if available
```
- `export.py` should result in a model that **profiles successfully on device**
- `evaluate.py` results should match the expected values in `numerics.yaml` for the model

### Branch Naming
- Development branches: `dev/<username>/<branch_name>`

## Running Tests

### Quick Reference
```bash
# Run package unit tests (doesn't touch models, safe to run anytime)
python scripts/build_and_test.py test_qaihm

# Run unit tests for a specific model
QAIHM_TEST_MODELS=<model_id> python scripts/build_and_test.py test_changed_models

# Run all pre-checkin tests (excludes long-running export tests)
python scripts/build_and_test.py precheckin

# Run all tests including export/compile (long)
python scripts/build_and_test.py all_tests_long
```

**Important:** Always use `build_and_test.py` to run tests—it handles model dependencies and environment setup automatically.

**Warning:** Export/compile/profile tests can take a very long time. Never run all tests—always limit to a small set of specific models, runtimes, and precisions using environment variables (e.g., `QAIHM_TEST_MODELS`, `QAIHM_TEST_PATHS`, `QAIHM_TEST_PRECISIONS`).

### Test Commands via build_and_test.py

| Command | Description |
|---------|-------------|
| `precheckin` | Quick tests: `test_qaihm` + `test_changed_models` (no export tests, shared env) |
| `precheckin_long` | Full tests for changed models: includes export tests, fresh env per model |
| `all_tests` | All models: `test_qaihm` + `test_all_models` (no export tests) |
| `all_tests_long` | All models with full test suite including exports |
| `test_qaihm` | Run tests for core qai_hub_models package (excludes models/) |
| `test_compile_all_models` | Submit compile jobs for all models |
| `test_profile_all_models` | Submit profile jobs for all models |
| `test_inference_all_models` | Submit inference jobs for all models |

### Testing Workflow

The CI (`.github/workflows/test.yml`) runs tests with these key settings:
- `QAIHM_TEST_HUB_ASYNC=1` - Jobs submitted without waiting
- `QAIHM_TEST_DEVICES=canary` - Uses canary device set
- `QAIHM_TEST_ASYNC_HUB_FAILURES_AS_TEST_FAILURES=1` - Upstream failures cause test failures

## Environment Variables

Tests can be configured via environment variables. All variables are prefixed with `QAIHM_TEST_`.

### Model Selection
| Variable | Description | Default |
|----------|-------------|---------|
| `QAIHM_TEST_MODELS` | Comma-separated list of model IDs to test. Special values: `all`, `pytorch`, `static`, `bench` | `all` |

### Test Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `QAIHM_TEST_PRECISIONS` | Comma-separated precisions (see below), or special: `default`, `default_minus_float`, `default_quantized`, `bench` | `default` |
| `QAIHM_TEST_PATHS` | Comma-separated runtimes (see below), or special: `default`, `all`, or prefix like `qnn` | `default` |
| `QAIHM_TEST_DEVICES` | Comma-separated devices (e.g., `cs_8_elite`, `cs_8_gen_3`), or special: `all`, `canary` | `all` |
| `QAIHM_TEST_QAIRT_VERSION` | QAIRT version for compile/profile jobs | `qaihm_default` |

**Available Precisions** (defined in `qai_hub_models/models/common.py`):
- Standard: `float`, `w8a8`, `w8a16`, `w16a16`, `w4a16`, `w4`
- Mixed precision: `w8a8_mixed_int16`, `w8a16_mixed_int16`, `w8a8_mixed_fp16`, `w8a16_mixed_fp16`

**Available Paths/Runtimes** (defined in `qai_hub_models/scorecard/path_profile.py`):
- TFLite: `tflite`
- QNN: `qnn_dlc`, `qnn_dlc_via_qnn_ep`, `qnn_context_binary`, `qnn_dlc_gpu`
- ONNX: `onnx`, `precompiled_qnn_onnx`, `onnx_dml_gpu`
- LLM: `genie`, `onnxruntime_genai`

### Test Behavior
| Variable | Description | Default |
|----------|-------------|---------|
| `QAIHM_TEST_IGNORE_KNOWN_FAILURES` | If `true`, run tests even for known-failing model+runtime+precision combos | `false` |
| `QAIHM_TEST_HUB_ASYNC` | If `true`, tests submit jobs without waiting; requires running tests in sequence (compile → profile → inference) | `false` |
| `QAIHM_TEST_IGNORE_DEVICE_JOB_CACHE` | If `true`, always submit new profile jobs instead of reusing cached results | `false` |
| `QAIHM_TEST_ASYNC_HUB_FAILURES_AS_TEST_FAILURES` | If `true` (and async enabled), upstream job failures cause downstream test failures instead of skips | `false` |

### Output & Artifacts
| Variable | Description | Default |
|----------|-------------|---------|
| `QAIHM_TEST_ARTIFACTS_DIR` | Directory for test artifacts and results | `./qaihm_test_artifacts` |
| `QAIHM_TEST_DEPLOYMENT` | AI Hub Workbench deployment to target | `prod` |
| `QAIHM_TEST_S3_ARTIFACTS_DIR` | S3 path for uploading exported model zips | (empty) |

### Example Usage
```bash
# Compile then profile a single model on one device with w8a8 precision (async mode recommended)
export QAIHM_TEST_HUB_ASYNC=1
export QAIHM_TEST_MODELS=yolov7
export QAIHM_TEST_DEVICES=cs_8_elite
export QAIHM_TEST_PRECISIONS=w8a8

python scripts/build_and_test.py test_compile_all_models  # Step 1: compile
python scripts/build_and_test.py test_profile_all_models  # Step 2: profile (uses compiled artifacts)
```

## Important Notes

- **Don't import directly**: `numba`, `xtcocotools`, `git` - use `qai_hub_models.extern.*` wrappers
- **S3 assets**: Upload to `qaihub-public-assets` bucket, use versioned folders (v1, v2, ...)
  - Run `python scripts/build_and_test.py validate_aws_credentials` first (prompts for password)
  - Use AWS profile `qaihm` (e.g., `aws s3 cp --profile qaihm ...`)
- **Requirements pinning**: All model-specific deps must be pinned to exact versions
- **Global requirements**: Check `global_requirements.txt` before adding new deps
