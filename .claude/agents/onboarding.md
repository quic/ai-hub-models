# Model Onboarding Agent

Use this agent when adding a new model to qai_hub_models.

## Design Goals

1. **Modify PyTorch code to work around compilation failures.** The goal is to get models compiling and running on device, even if it requires changes to the original model architecture.

2. **PyTorch code cannot depend on GPU.** All models must run on CPU. Look at existing models for tricks to remove GPU dependencies if you get stuck.

3. **Prefer monkeypatching over SourceAsRoot.** When modifying external model code, use monkeypatching techniques first. `SourceAsRoot` (copying source into the repo) should always be the last resort.

4. **Prefer pip install from GitHub over SourceAsRoot.** If a package isn't on PyPI, install directly from GitHub (e.g., `git+https://github.com/...`). Use `SourceAsRoot` only if direct GitHub install is not possible.

5. **Merge pre/postprocessing into the model.** Include as much preprocessing and postprocessing as possible in the model itself to simplify on-device implementation.

6. **Follow existing I/O conventions.** Models with existing examples (e.g., object detectors, segmentation) should follow the input/output format of similar models in the repo for consistency.

7. **Use shared code.** Check `qai_hub_models/utils/` for existing utilities before implementing your own pre/postprocessing. Look at similar models' `app.py` files and consolidate shared code when possible.

## Terminology

- **model_id** - The folder name (e.g., `yolov7`, `ddrnet23_slim`)
- **model_name** - The published display name in info.yaml (e.g., "YOLOv7", "DDRNet23-Slim")

## Model Directory Structure

Each model lives in `qai_hub_models/models/<model_id>/` and requires:

### Required Files

1. **model.py** - PyTorch model inheriting from `BaseModel`

   **Required methods:**
   - `from_pretrained(cls)` - classmethod to load pretrained weights; all args must have defaults
   - `get_input_spec()` - staticmethod returning `InputSpec` dict of `{input_name: (shape, dtype)}`
   - `get_output_names()` - staticmethod returning list of output tensor names

   **Optional overrides (have default implementations):**
   - `_sample_inputs_impl()` - provide real sample inputs instead of random data
   - `_get_input_spec_for_instance()` - instance-specific input spec (when shapes depend on instance vars)
   - `get_channel_last_inputs()` / `get_channel_last_outputs()` - inputs/outputs to transpose for on-device performance
   - `get_hub_compile_options()` / `get_hub_profile_options()` - custom AI Hub flags
   - `get_unsupported_reason()` - marks specific device attributes that can't be supported by this model (eg hexagon version)
   - `eval_datasets()` - list of dataset names for evaluation
   - `get_evaluator()` - return evaluator class for accuracy measurement
   - `calibration_dataset_name()` - dataset for quantization calibration
   - `get_hub_litemp_percentage(precision)` - returns percentage (0-100) of sensitive layers to keep in higher precision for mixed precision quantization (e.g., `w8a8_mixed_int16`)

2. **app.py** - End-to-end application with pre/post-processing
   - `App` class taking a `Callable` (works with PyTorch or on-device inference)
   - `predict()` method for inference

3. **demo.py** - CLI demo running the app on sample data
   - Parse args, init model, run app, display/save results

4. **test.py** - Unit tests
   - `test_task`: PyTorch model accuracy
   - `test_trace`: TorchScript accuracy (mark with `@pytest.mark.trace`)
   - `test_demo`: Demo runs without error

5. **info.yaml** - Model metadata for public website
   - Auto-fill size/params: `python qai_hub_models/scripts/autofill_info_yaml.py -m <model_id>`

### Optional Files

- **code-gen.yaml** - Custom options for export.py generation
- **requirements.txt** - Model-specific dependencies (pinned versions required)

### Auto-generated Files

- `README.md`, `export.py`, `test_generated.py` via `python qai_hub_models/scripts/run_codegen.py -m <model_id>`
- `evaluate.py` - Only generated if model defines `eval_datasets()` and `get_evaluator()`
- `perf.yaml` - Generated weekly by CI

## Architecture Patterns

### Base Classes (`qai_hub_models/utils/base_model.py`)

- **BaseModel** - Standard single-model, inherits `torch.nn.Module`
- **CollectionModel** - Multi-component models (e.g., encoder-decoder where components are compiled separately)
- **BasePrecompiledModel** - Pre-compiled assets only (no PyTorch source available)

### Shared Components

- `models/_shared/` - Reusable model components (LLM tokenizers, pose estimation, etc.)
- `extern/` - Safe imports for optional deps (numba, xtcocotools, git)

## Onboarding Workflow

1. Create folder `qai_hub_models/models/<model_id>/`
2. Implement required files: `model.py`, `app.py`, `demo.py`, `test.py`, `info.yaml`
3. Add `requirements.txt` if model needs additional dependencies
4. Run codegen: `python qai_hub_models/scripts/run_codegen.py -m <model_id>`
5. Auto-fill info.yaml: `python qai_hub_models/scripts/autofill_info_yaml.py -m <model_id>`
6. Verify:
   ```bash
   # Run pre-commit on all files
   pre-commit run --all-files

   # Install model dependencies (see qai_hub_models/models/<model_id>/README.md)

   # Test export
   python -m qai_hub_models.models.<model_id>.export --target-runtime <tflite|onnx|qnn> --chipset <chipset>

   # Test demo
   python -m qai_hub_models.models.<model_id>.demo

   # Test evaluation (if model has eval_datasets() defined)
   python -m qai_hub_models.models.<model_id>.evaluate --help
   ```

## Adding Quantization Support

To enable quantized precision options (e.g., w8a8, w8a16) for a model:

### 1. Add a Dataset
Create or reuse a dataset in `qai_hub_models/datasets/`:
- Inherit from appropriate base class (e.g., `BaseDataset`)
- Implement data loading and preprocessing
- Register the dataset name

### 2. Add an Evaluator
Create or reuse an evaluator in `qai_hub_models/evaluators/`:
- Inherit from base evaluator class
- Implement accuracy metrics for your task (e.g., mAP for detection, IoU for segmentation)

### 3. Update the Model
In `model.py`, implement these methods:
```python
@staticmethod
def eval_datasets() -> list[str]:
    return ["<dataset_name>"]

@staticmethod
def calibration_dataset_name() -> str:
    return "<dataset_name>"

@classmethod
def get_evaluator(cls) -> type[BaseEvaluator]:
    return <YourEvaluator>
```

### 4. Update code-gen.yaml
Add supported precisions:
```yaml
supported_precisions:
  - float
  - w8a8
  - w8a16
```

### 5. Re-run Codegen
```bash
python qai_hub_models/scripts/run_codegen.py -m <model_id>
```

This will generate/update `evaluate.py` and add quantization options to `export.py`.

### 6. Test Quantized Accuracy
Run evaluate.py to verify quantization accuracy:
```bash
python -m qai_hub_models.models.<model_id>.evaluate --precision w8a8
```

Ensure accuracy drop from float is reasonable (10 points or less). If accuracy drop is too large, consider using mixed precision (e.g., `w8a8_mixed_int16`).

## S3 Assets

For model checkpoints or test data not available via public URLs:
- Run `python scripts/build_and_test.py validate_aws_credentials` first (prompts for password)
- Use AWS profile `qaihm` (e.g., `aws s3 cp --profile qaihm ...`)
- Upload to `qaihub-public-assets` S3 bucket under `qai-hub-models/models/<model_id>/v1/`
- Use versioned folders (v1, v2, ...) - assets cannot be deleted
- Set `MODEL_ASSET_VERSION = 1` in model.py
- Grant public-read access when uploading

## Requirements

- All packages in `requirements.txt` must be pinned to exact versions (e.g., `torch==2.0.1`)
- Check `global_requirements.txt` before adding new deps
- If a different version than global is required, set `global_requirements_incompatible: true` in `code-gen.yaml`
