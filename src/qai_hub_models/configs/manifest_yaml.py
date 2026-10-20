# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_serializer, model_validator
from qai_hub_models_cli.proto import info_pb2, numerics_pb2

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs._info_yaml_enums import (
    MODEL_DOMAIN,
    MODEL_LICENSE,
    MODEL_STATUS,
    MODEL_TAG,
    MODEL_USE_CASE,
    VOICE_AI_SDK,
)
from qai_hub_models.configs._info_yaml_llm_details import LLM_CALL_TO_ACTION, LLMDetails
from qai_hub_models.configs.model_disable_reasons import ModelDisableReasonsMapping
from qai_hub_models.configs.proto_helpers import (
    call_to_action_to_proto,
    domain_to_proto,
    form_factor_to_proto,
    license_to_proto,
    runtime_to_proto,
    status_to_proto,
    tag_to_proto,
    use_case_to_proto,
    voice_ai_sdk_to_proto,
)

# configs normally does not import from models, but lm_schema is a self-contained,
# dependency-light (pydantic-only) island -- not model code -- so the manifest can
# validate its lm_quantization_details block against it without a real layer inversion.
from qai_hub_models.models.templates.lm_schema import PrecisionSchema, Recipe
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.device import DEFAULT_EXPORT_DEVICE, FormFactor
from qai_hub_models.utils.path_helpers import (
    MODEL_IDS,
    MODELS_PACKAGE_NAME,
    QAIHM_MODELS_ROOT,
    QAIHM_PACKAGE_NAME,
    QAIHM_PACKAGE_ROOT,
    _get_qaihm_models_root,
)

__all__ = [
    "MODEL_DOMAIN",
    "MODEL_LICENSE",
    "MODEL_STATUS",
    "MODEL_TAG",
    "MODEL_USE_CASE",
    "ExternalRepoConfig",
    "LMQuantizationDetails",
    "NumericsAccuracyBenchmark",
    "PipCommand",
    "PipCommandMachine",
    "QAIHMModelManifest",
]


PipCommandMachine = Literal["any", "gpu", "cpu"]

# Substrings that would let a pip command smuggle in extra shell operations
# (chaining, redirects). These are inert inside any quoted region, so we
# only scan for them outside of single/double quotes.
_UNQUOTED_METACHAR_PATTERNS: tuple[str, ...] = (
    "&&",
    "||",
    ";",
    "|",
    "&",
    ">",
    "<",
    "\n",
    "\r",
    "\\",
)

# Substrings that trigger command substitution or variable expansion. In
# POSIX shells these still fire inside double quotes (only single quotes
# suppress them), so we reject them everywhere except inside single quotes.
# No legitimate pip command needs ``$`` or a backtick in its arguments.
_ALWAYS_REJECTED_PATTERNS: tuple[str, ...] = (
    "`",
    "$",
)


def _find_unquoted_shell_meta(cmd: str) -> str | None:
    """Return the offending sequence if any shell metacharacter appears outside single quotes.

    Also returns ``"<unbalanced quote>"`` if the command has an unclosed quote.
    Returns ``None`` if the command is clean.
    """
    i = 0
    in_single = False
    in_double = False
    while i < len(cmd):
        c = cmd[i]
        if in_single:
            if c == "'":
                in_single = False
            i += 1
            continue
        for pat in _ALWAYS_REJECTED_PATTERNS:
            if cmd.startswith(pat, i):
                return pat
        if in_double:
            if c == '"':
                in_double = False
            i += 1
            continue
        if c == "'":
            in_single = True
            i += 1
            continue
        if c == '"':
            in_double = True
            i += 1
            continue
        for pat in _UNQUOTED_METACHAR_PATTERNS:
            if cmd.startswith(pat, i):
                return pat
        i += 1
    if in_single or in_double:
        return "<unbalanced quote>"
    return None


class ExternalRepoConfig(BaseQAIHMConfig):
    """Configuration for a single external repository dependency."""

    repo_url: str
    commit_sha: str
    patches_filename: str | None = None

    @model_validator(mode="after")
    def check_fields(self) -> ExternalRepoConfig:
        if not self.repo_url:
            raise ValueError("repo_url must not be empty.")
        if not self.commit_sha:
            raise ValueError("commit_sha must not be empty.")
        return self


class PipCommand(BaseQAIHMConfig):
    """A single pip command to run before or after installing requirements.txt.

    In YAML, an entry may be written either as a bare string (when no
    ``machine`` gate is needed) or as a mapping::

        pre_pip_install_commands:
        - pip install foo             # bare string, machine="any"
        - command: pip install bar    # mapping form, needed to set machine
          machine: gpu

    Every command must start with ``"pip "``. Shell metacharacters are
    rejected so a config entry cannot smuggle in arbitrary commands.

    The ``machine`` field gates when the command runs:
      - ``"any"`` (default) runs on every host.
      - ``"gpu"`` runs only when a CUDA GPU is detected on the host.
      - ``"cpu"`` runs only when no CUDA GPU is detected.
    """

    command: str
    machine: PipCommandMachine = "any"

    @model_validator(mode="before")
    @classmethod
    def _lift_bare_string(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"command": data}
        return data

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any) -> Any:
        if self.machine == "any":
            return self.command
        return handler(self)

    @model_validator(mode="after")
    def check_fields(self) -> PipCommand:
        cmd = self.command.strip()
        if not cmd:
            raise ValueError("pip command must not be empty.")
        # The manifest is trusted config, but we still enforce a narrow shape
        # so the install list can be audited for security-relevant changes.
        # The command is passed through the shell downstream, so any
        # unquoted metacharacter would let a config entry smuggle in
        # arbitrary commands.
        try:
            tokens = shlex.split(cmd)
        except ValueError as e:
            raise ValueError(
                f"pip command is not a valid shell token sequence: {self.command!r} ({e})"
            ) from e
        if not tokens or tokens[0] != "pip":
            raise ValueError(
                f"pip command must start with 'pip '. Got: {self.command!r}"
            )
        if len(tokens) < 2:
            raise ValueError(
                f"pip command must have at least one argument after 'pip'. "
                f"Got: {self.command!r}"
            )
        offender = _find_unquoted_shell_meta(cmd)
        if offender is not None:
            raise ValueError(
                f"pip command must not contain shell metacharacter {offender!r} "
                f"outside of a quoted string (found in {self.command!r})."
            )
        return self


class NumericsAccuracyBenchmark(BaseQAIHMConfig):
    """Expected accuracy benchmark for a model on a specific dataset/metric."""

    dataset_name: str
    metric_name: str
    value: float
    unit: str
    # Where this benchmark value came from (e.g., a URL to the paper or
    # model card, or "AI Hub Models Reference Eval" for scorecard-derived values).
    source: str

    def to_proto(self) -> numerics_pb2.NumericsAccuracyBenchmark:
        return numerics_pb2.NumericsAccuracyBenchmark(
            dataset_name=self.dataset_name,
            metric_name=self.metric_name,
            value=self.value,
            unit=self.unit,
            source=self.source,
        )


TechnicalDetails = dict[str, str | int | float]


class LMQuantizationDetails(BaseQAIHMConfig):
    """The ``precision:`` + ``recipe:`` pair that produced one precision's checkpoint.

    Both halves are validated by the shared GenAI Lab schema (``lm_schema``):
    ``precision`` is the bit config (defaults reproduce AIHM's W4A16 contract), and
    ``recipe`` is the ordered technique chain (SeqMSE / AdaScale / Calibration / ...).
    Copied verbatim from the GenAI Lab config that produced the checkpoint; see the
    ``lm_quantization_details`` field on :class:`QAIHMModelManifest`.
    """

    # No default: a field that merely equals its default is stripped by
    # exclude_defaults, and an omitted `precision:` means the W4A16 contract, not
    # "nothing". _fill_precision supplies it so existing manifests still validate.
    precision: PrecisionSchema
    recipe: Recipe

    @model_validator(mode="before")
    @classmethod
    def _fill_precision(cls, data: Any) -> Any:
        if isinstance(data, dict) and "precision" not in data:
            return {**data, "precision": {}}
        return data


class QAIHMModelManifest(BaseQAIHMConfig):
    """Unified schema for manifest.yaml.

    A single manifest.yaml lives at three altitudes:
      * Full model folder — carries the website-facing metadata plus
        build/export config. The website-facing block is required here.
      * ``templates/<name>/`` template — carries dependencies (``templates:``)
        and optional external repo config only.
      * ``datasets/<name>/`` — carries dependencies (``templates:``) only.

    The website-facing fields are optional at the schema level and enforced
    by ``check_fields`` only when this manifest describes a real model
    (``id`` set and present in ``MODEL_IDS``).
    """

    # =============================================================================
    # Template/dataset dependencies
    # =============================================================================

    # Names of other templates/ folders this manifest depends on.
    # Used by template and dataset manifests; empty for model manifests.
    templates: list[str] = Field(default_factory=list)

    # Names of datasets/ folders this manifest depends on.
    # Used by template and dataset manifests; empty for model manifests.
    datasets: list[str] = Field(default_factory=list)

    # =============================================================================
    # Website-facing metadata fields (originally from info.yaml)
    # =============================================================================

    # Name of the model as it will appear on the website.
    # Should have dashes instead of underscores and all
    # words capitalized. For example, `Whisper-Base-En`.
    name: str | None = None

    # Name of the model's folder within the repo.
    id: str | None = None

    # Publish status. UNSET is only for external recipes — the CLI treats
    # it as neither published nor unpublished and runs without prompting.
    status: MODEL_STATUS = MODEL_STATUS.UNSET

    # A brief catchy headline explaining what the model does and why it may be interesting
    headline: str | None = None

    # The domain the model is used in such as computer vision, audio, etc.
    domain: MODEL_DOMAIN | None = None

    # A 2-3 sentence description of how the model can be used.
    description: str | None = None

    # What task the model is used to solve, such as object detection, classification, etc.
    use_case: MODEL_USE_CASE | None = None

    # A list of applicable tags to add to the model
    tags: list[MODEL_TAG] = Field(default_factory=list)

    # A list of real-world applicaitons for which this model could be used.
    # This is free-from and almost anything reasonable here is fine.
    applicable_scenarios: list[str] = Field(default_factory=list)

    # A list of other similar models in the repo.
    # Typically, any model that performs the same task is fine.
    # If nothing fits, this can be left blank. Limit to 3 models.
    related_models: list[str] = Field(default_factory=list)

    # A list of device types for which this model could be useful.
    # If unsure what to put here, default to `Phone` and `Tablet`.
    form_factors: list[FormFactor] = Field(default_factory=list)

    # Whether the model has a static image uploaded in S3. All published models must have this.
    has_static_banner: bool = False

    # Whether the model has an animated asset uploaded in S3. This is optional.
    has_animated_banner: bool = False

    # A list of datasets for which the model has pre-trained checkpoints
    # available as options in `model.py`. Typically only has one entry.
    dataset: list[str] = Field(default_factory=list)

    # A list of a few technical details about the model.
    #   Model checkpoint: The name of the downloaded model checkpoint file.
    #   Input resolution: The size of the model's input. For example, `2048x1024`.
    #   Number of parameters: The number of parameters in the model.
    #   Model size: The file size of the downloaded model asset.
    #   Number of output classes: The number of classes the model can classify or annotate.
    technical_details: TechnicalDetails = Field(default_factory=dict)

    # Technical details specific to certain runtimes. Keyed by runtime; empty
    # for models with no runtime-specific details.
    runtime_technical_details: dict[TargetRuntime, TechnicalDetails] = {}

    # The license type of the original model repo.
    license_type: MODEL_LICENSE | None = None

    # Device form factors for which we don't publish performance data.
    private_perf_form_factors: list[FormFactor] | None = None

    # Some models are made by company
    model_maker_id: str | None = None

    # Link to the research paper where the model was first published. Usually an arxiv link.
    research_paper: str | None = None

    # The title of the research paper.
    research_paper_title: str | None = None

    # A link to the original github repo with the model's code.
    source_repo: str | None = None

    # A link to the model's license. Most commonly found in the github repo it was cloned from.
    license: str | None = None

    # Whether the model is compatible with the IMSDK Plugin for IOT devices
    imsdk_supported: bool = False

    # If set, model assets shouldn't distributed.
    restrict_model_sharing: bool = False

    # Expected accuracy benchmark. If set, scorecard will flag results that
    # deviate from this value by more than the metric's metric_enablement_threshold.
    numerics_benchmark: NumericsAccuracyBenchmark | None = None

    # If status is private, this must have a reference to an issue with an explanation.
    status_reason: str | None = None

    # It is a large language model (LLM) or not.
    model_type_llm: bool = False

    # Add per device, download, app and if the model is available for purchase.
    llm_details: LLMDetails | None = None

    # Which Qualcomm Voice AI SDK variant the model is compatible with.
    # Unset means the model has no Voice AI SDK integration; the "Deploying
    # on-device" section that links to the SDK download page is only emitted
    # when this is set.
    voice_ai_sdk: VOICE_AI_SDK | None = None

    # =============================================================================
    # Build/export config fields
    # =============================================================================

    # External repository dependencies for this model.
    # Keys are repo names used as import paths (e.g., "gkt" -> external_repos.gkt.module).
    external_repos: dict[str, ExternalRepoConfig] | None = None

    # Direct dependencies on other model folders under qai_hub_models/models/.
    # Only the model's own direct imports; transitive deps are captured in
    # the depended-on model's own manifest.yaml.
    models: list[str] = Field(default_factory=list)

    # Whether the model is quantized with aimet.
    is_aimet: bool = False

    # The list of precisions that:
    # - Are enabled via the CLI
    # - Scorecard runs by default each week for accuracy & performance tests
    supported_precisions: list[Precision] = Field(
        default_factory=lambda: [Precision.float]
    )

    # Portable GenAI Lab quant recipes, keyed by Hub precision name (a subset of
    # supported_precisions). Each value is the precision: + recipe: block copied
    # verbatim from the GenAI Lab config that produced that precision's checkpoint,
    # validated by the shared lm_schema. Empty for models with no recorded recipe;
    # LLM/VLM folders only (lm_ prefix mirrors lm_schema / lm_driver). See the RFC
    # for the full design.
    lm_quantization_details: dict[Precision, LMQuantizationDetails] = Field(
        default_factory=dict
    )

    # aimet model can additionally specify num calibration samples to speed up
    # compilation
    num_calibration_samples: int | None = None

    # Whether the model's demo supports running on device with the `--eval-mode on-device` option.
    has_on_target_demo: bool = False

    # The reason why various paths are disabled
    disabled_paths: ModelDisableReasonsMapping = Field(
        default_factory=lambda: ModelDisableReasonsMapping()
    )

    # If set, changes the default device when running export.py for the model.
    default_device: str = DEFAULT_EXPORT_DEVICE

    # Some model outputs have low PSNR when in practice the numerical accuracy is fine.
    # This can happen when the model outputs many low confidence values that get
    # filtered out in post-processing.
    # Omit printing PSNR in `export.py` for these to avoid confusion.
    # dict<output_idx, reason_for_skip>
    outputs_to_skip_validation: dict[int, str] | None = None

    # True for Collection model comprises of components, such as Whisper model's
    # encoder and decoder.
    is_collection_model: bool = False

    # Whether the model uses the pre-compiled pattern instead of the
    # standard pre-trained pattern.
    is_precompiled: bool = False

    # If set, all paths that compile "Just In Time" to QNN on device are disabled.
    # These disabled paths are sometimes referred to as doing "on device prepare".
    #
    # In other words, if set, only paths that compile to context binary ahead of time
    # ("AOT prepare") are enabled, both in CI and in Scorecard.
    requires_aot_prepare: bool = False

    # "Orchestrator runtimes" are runtimes that require extra orchestration steps beyond just running the model in order to work.
    orchestrator_runtimes: list[TargetRuntime] = Field(default_factory=list)

    # If set, only the runtimes in orchestrator runtimes will be supported.
    only_allow_orchestrator_runtimes: bool = False

    # pip commands to run before ``pip install -r requirements.txt``.
    #
    # Each entry is a :class:`PipCommand`. Every command must start with
    # ``"pip "`` and contain no shell metacharacters outside of quoted
    # strings; the ``machine`` field gates execution on CPU-only / GPU hosts.
    #
    # Typical uses:
    #   - Pre-install a package that ``requirements.txt`` needs at build time
    #     (e.g. ``pip install "torch>=2.1"`` before a wheel that builds against
    #     torch but doesn't declare it as a build dep).
    #   - Install from a custom index on a specific host type
    #     (e.g. a CPU-only torch build for machines without a CUDA GPU).
    pre_pip_install_commands: list[PipCommand] = Field(default_factory=list)

    # pip commands to run after ``pip install -r requirements.txt``.
    #
    # Typical uses:
    #   - Install a package straight from a git URL (which cannot go in
    #     ``requirements.txt`` when other pip flags are required).
    #   - Overwrite a transitively-installed package with a GPU-specific build
    #     (e.g. ``onnxruntime-gpu`` and an AIMET wheel on GPU hosts).
    post_pip_install_commands: list[PipCommand] = Field(default_factory=list)

    # A comma separated list of metrics to print in the inference summary of `export.py`.
    inference_metrics: str = "psnr"

    # Additional details that can be set on the model's readme.
    # Use LiteralScalarString so the YAML dump writes this on multiple lines instead of dumping '\n' directly
    additional_readme_section: str = ""

    # If set, omits the "Example Usage" section from the HuggingFace readme.
    skip_example_usage: bool = False

    # If set, the repo README groups Setup / Verify with CLI Demo / Export the
    # model artifact under a single "## Export" parent (instead of the default
    # top-level "## Setup", "## Run CLI Demo", "## Export for on-device
    # deployment" sections). Used by genie LLM models whose local-device
    # deployment section links to the "#export" anchor.
    local_device_deployment: bool = False

    # The model supports python versions that are at least this version. None == Any version
    python_version_greater_than_or_equal_to: str | None = None
    python_version_greater_than_or_equal_to_reason: str | None = None

    # The model supports python versions that are less than this version. None == Any version
    python_version_less_than: str | None = None
    python_version_less_than_reason: str | None = None

    # If set, the model returns multiple (input_spec, graph_name) pairs.
    # The generated export.py will loop over compile specs, submit multiple
    # compile jobs, and link them into a single context binary.  Models
    # that do NOT set this flag get the simple single-compile-job path.
    has_multi_graph: bool = False

    # If set, the model has a separate quantize.py script. The --precision
    # option is omitted from export.py and precision is determined from
    # the checkpoint (via args.json or DEFAULT_* sentinel).
    separate_quantize_script: bool = False

    # Instructions for installing system-level dependencies before pip install.
    readme_install_system_deps: str | None = None

    # =============================================================================
    # Field name sets for website info.yaml derivation
    # =============================================================================

    # Fields that belong in the website's info.yaml (subset of manifest fields)
    _INFO_YAML_FIELDS: frozenset[str] = frozenset(
        {
            "name",
            "id",
            "status",
            "headline",
            "domain",
            "description",
            "use_case",
            "tags",
            "applicable_scenarios",
            "related_models",
            "form_factors",
            "has_static_banner",
            "has_animated_banner",
            "dataset",
            "technical_details",
            "runtime_technical_details",
            "license_type",
            "private_perf_form_factors",
            "model_maker_id",
            "research_paper",
            "research_paper_title",
            "source_repo",
            "license",
            "imsdk_supported",
            "restrict_model_sharing",
            "numerics_benchmark",
            "status_reason",
            "model_type_llm",
            "llm_details",
            "voice_ai_sdk",
        }
    )

    # =============================================================================
    # Methods from QAIHMModelInfo
    # =============================================================================

    @staticmethod
    def _technical_details_to_proto(
        details: TechnicalDetails,
    ) -> list[info_pb2.ModelInfo.TechnicalDetail]:
        protos = []
        for key, val in details.items():
            td = info_pb2.ModelInfo.TechnicalDetail(key=key)
            if isinstance(val, int):
                td.int_value = val
            elif isinstance(val, float):
                td.float_value = val
            else:
                td.string_value = str(val)
            protos.append(td)
        return protos

    def to_proto(self, aihm_version: str) -> info_pb2.ModelInfo:
        assert self.is_full_model_manifest, "to_proto only valid for model manifests"
        assert self.status is not None
        assert self.domain is not None
        assert self.use_case is not None
        assert self.license_type is not None
        technical_details = self._technical_details_to_proto(self.technical_details)

        runtime_technical_details = [
            info_pb2.ModelInfo.RuntimeTechnicalDetails(
                runtime=runtime_to_proto(runtime),
                technical_details=self._technical_details_to_proto(details),
            )
            for runtime, details in self.runtime_technical_details.items()
        ]

        llm_details = None
        if self.llm_details is not None:
            llm_details = info_pb2.ModelInfo.LLMDetails(
                call_to_action=call_to_action_to_proto(self.llm_details.call_to_action),
                genie_compatible=self.llm_details.genie_compatible,
                geniex_llamacpp_compatible=self.llm_details.geniex_llamacpp_compatible,
            )

        numerics_benchmark = None
        if self.numerics_benchmark is not None:
            numerics_benchmark = self.numerics_benchmark.to_proto()

        return info_pb2.ModelInfo(
            aihm_version=aihm_version,
            id=self.id,
            name=self.name,
            status=status_to_proto(self.status),
            status_reason=self.status_reason,
            headline=self.headline,
            domain=domain_to_proto(self.domain),
            description=self.description,
            use_case=use_case_to_proto(self.use_case),
            tags=[tag_to_proto(t) for t in self.tags],
            applicable_scenarios=self.applicable_scenarios,
            related_models=self.related_models,
            form_factors=[form_factor_to_proto(ff) for ff in self.form_factors],
            technical_details=technical_details,
            runtime_technical_details=runtime_technical_details,
            license_type=license_to_proto(self.license_type),
            model_maker_id=self.model_maker_id,
            dataset=self.dataset,
            research_paper=self.research_paper,
            research_paper_title=self.research_paper_title,
            source_repo=self.source_repo,
            license_url=self.license,
            has_static_banner=self.has_static_banner,
            has_animated_banner=self.has_animated_banner,
            imsdk_supported=self.imsdk_supported,
            restrict_model_sharing=self.restrict_model_sharing,
            numerics_benchmark=numerics_benchmark,
            model_type_llm=self.model_type_llm,
            llm_details=llm_details,
            private_perf_form_factors=[
                form_factor_to_proto(ff)
                for ff in (self.private_perf_form_factors or [])
            ],
            voice_ai_sdk=voice_ai_sdk_to_proto(self.voice_ai_sdk),
        )

    def can_promote_to_published(self) -> tuple[bool, str]:
        """
        Check whether this model meets all prerequisites for promotion to PUBLISHED.

        Returns (True, "") if promotion is safe, or (False, reason) if not.
        """
        assert self.license_type is not None
        assert self.id is not None
        if self.license_type.is_non_commerical:
            return (
                False,
                f"Models with license {self.license_type!s} cannot be published",
            )

        if not self.has_static_banner:
            return False, "model has no static banner asset"

        if not self.supports_at_least_1_runtime:
            return False, "model does not support at least one export path"

        must_have_assets = not self.restrict_model_sharing
        if self.llm_details:
            must_have_assets = must_have_assets and self.llm_details.call_to_action in {
                LLM_CALL_TO_ACTION.DOWNLOAD,
                LLM_CALL_TO_ACTION.DOWNLOAD_AND_VIEW_README,
            }

        if must_have_assets and not os.path.exists(
            QAIHM_MODELS_ROOT / self.id / "release-assets.yaml"
        ):
            return False, "no release assets available"

        return True, ""

    def get_package_name(self) -> str:
        assert self.id is not None
        return f"{QAIHM_PACKAGE_NAME}.{MODELS_PACKAGE_NAME}.{self.id}"

    def get_package_path(self, root: Path = QAIHM_PACKAGE_ROOT) -> Path:
        assert self.id is not None
        return _get_qaihm_models_root(root) / self.id

    def get_model_definition_path(self) -> str:
        assert self.id is not None
        return os.path.join(
            ASSET_CONFIG.get_qaihm_repo(self.id, relative=False), "model.py"
        )

    def get_demo_path(self) -> str:
        assert self.id is not None
        return os.path.join(
            ASSET_CONFIG.get_qaihm_repo(self.id, relative=False), "demo.py"
        )

    def get_manifest_yaml_path(self, root: Path = QAIHM_PACKAGE_ROOT) -> Path:
        return self.get_package_path(root) / "manifest.yaml"

    def get_hf_pipeline_tag(self) -> str:
        assert self.use_case is not None
        return self.use_case.map_to_hf_pipeline_tag()

    def get_hugging_face_metadata(
        self, root: Path = QAIHM_PACKAGE_ROOT
    ) -> dict[str, str | list[str]]:
        # Get the metadata for huggingface model cards.
        hf_metadata: dict[str, str | list[str]] = {}
        hf_metadata["library_name"] = "pytorch"
        # We only tag Hugging Face models with the specific license name if the source is copyleft.
        # Most models are tagged with the "other" license on HF because they use the AI Hub Models license.
        hf_metadata["license"] = (
            # 'Unlicensed' will appear only if this model is not public.
            # All models are validated to have a license if they are public.
            self.license_type or MODEL_LICENSE.UNLICENSED
        ).huggingface_name
        hf_metadata["tags"] = [tag.name.lower() for tag in self.tags] + ["android"]
        hf_metadata["pipeline_tag"] = self.get_hf_pipeline_tag()
        return hf_metadata

    def get_model_details(self) -> str:
        # Model details.
        details = (
            "- **Model Type:** "
            + self.use_case.__str__().lower().capitalize()
            + "\n- **Model Stats:**"
        )
        for name, val in self.technical_details.items():
            details += f"\n  - {name}: {val}"
        return details

    def check_geniex_runtime_technical_details(self) -> None:
        geniex_runtimes = (TargetRuntime.GENIEX_QAIRT, TargetRuntime.GENIEX_LLAMACPP)
        errors: list[str] = []

        missing = [
            rt.value
            for rt in geniex_runtimes
            if rt in self.orchestrator_runtimes
            and rt not in self.runtime_technical_details
        ]
        if missing:
            errors.append(
                "manifest.yaml must define runtime_technical_details for GenieX "
                f"runtimes listed in orchestrator_runtimes: {', '.join(missing)}."
            )

        # model_infer_id points the website at a HuggingFace repo to download, so it
        # is required exactly when we're allowed to redistribute the model's weights.
        for rt in geniex_runtimes:
            details = self.runtime_technical_details.get(rt)
            if details is None:
                continue
            has_infer_id = "model_infer_id" in details
            if self.restrict_model_sharing and has_infer_id:
                errors.append(
                    f"runtime_technical_details.{rt.value} must not define "
                    "model_infer_id when restrict_model_sharing is true; these "
                    "weights cannot be redistributed."
                )
            elif not self.restrict_model_sharing and not has_infer_id:
                errors.append(
                    f"runtime_technical_details.{rt.value} must define "
                    "model_infer_id (the HuggingFace repo the website links for "
                    "deployment), or set restrict_model_sharing: true."
                )

        if errors:
            raise ValueError(
                f"{self.id}:\n" + "\n".join(f"  - {err}" for err in errors)
            )

    def get_perf_yaml_path(self, root: Path = QAIHM_PACKAGE_ROOT) -> Path:
        return self.get_package_path(root) / "perf.yaml"

    def get_release_assets_yaml_path(self, root: Path = QAIHM_PACKAGE_ROOT) -> Path:
        return self.get_package_path(root) / "release-assets.yaml"

    def get_readme_path(self, root: Path = QAIHM_PACKAGE_ROOT) -> Path:
        return self.get_package_path(root) / "README.md"

    def get_hf_model_card_path(self, root: Path = QAIHM_PACKAGE_ROOT) -> Path:
        return self.get_package_path(root) / "HF_MODEL_CARD.md"

    def get_requirements_path(self, root: Path = QAIHM_PACKAGE_ROOT) -> Path:
        return self.get_package_path(root) / "requirements.txt"

    def has_model_requirements(self, root: Path = QAIHM_PACKAGE_ROOT) -> bool:
        return os.path.exists(self.get_requirements_path(root))

    def get_web_url(self, website_url: str = ASSET_CONFIG.models_website_url) -> str:
        return f"{website_url}/models/{self.id}"

    @property
    def is_gen_ai_model(self) -> bool:
        return MODEL_TAG.LLM in self.tags or MODEL_TAG.GENERATIVE_AI in self.tags

    def to_website_info_yaml(self, path: Path) -> Path:
        """Write an info.yaml subset for the website (excludes code-gen fields).

        This derives the website-facing info.yaml from the manifest so the website
        sees no schema change after the info.yaml → manifest.yaml migration.
        """
        non_info_fields = set(self.model_fields.keys()) - self._INFO_YAML_FIELDS
        self.to_yaml(
            path=path,
            exclude=non_info_fields,
            exclude_defaults=False,  # Website needs all fields even if default
        )
        return path

    # =============================================================================
    # Build/export methods
    # =============================================================================

    def is_supported(
        self,
        precision: Precision,
        runtime: TargetRuntime,
        consider_scorecard_failures: bool = True,
        consider_user_defined_failures: bool = True,
        consider_timeouts: bool = True,
        device: str | None = None,
    ) -> bool:
        """
        Return true if this precision + runtime combo is supported by this model.
        Return false if this model has a failure reason set for this runtime.

        If consider_scorecard_failures is False, then scorecard failures in manifest
        are ignored for the purposes of determining if a path is supported.
        If a reason lists disable_devices, it only counts when device is one of them.
        """
        return not bool(
            self.failure_reason(
                precision,
                runtime,
                consider_scorecard_failures,
                consider_user_defined_failures,
                consider_timeouts,
                device,
            )
        )

    def failure_reason(
        self,
        precision: Precision,
        runtime: TargetRuntime,
        include_scorecard_failures: bool = True,
        include_user_defined_failures: bool = True,
        include_timeouts: bool = True,
        device: str | None = None,
    ) -> str | None:
        """
        Return the reason a model failed or None if the model did not fail.

        If a reason lists disable_devices, its issue and causes_timeout only apply
        when device is one of them.
        """
        if (
            not runtime.is_orchestrator_runtime
            and self.only_allow_orchestrator_runtimes
        ):
            return f"{runtime} is not an orchestrator runtime, but only orchestrator runtimes are supported for this model."
        if (
            runtime.is_orchestrator_runtime
            and runtime not in self.orchestrator_runtimes
        ):
            return f"{runtime} is not a supported runtime for this model."

        if self.is_precompiled and not runtime.is_aot_compiled:
            return "Precompiled models are only supported via runtimes that are compiled ahead of time (is_aot_compiled)."

        if precision and not runtime.supports_precision(precision):
            return f"{runtime} does not support precision {precision!s}."

        if (
            self.requires_aot_prepare
            and not runtime.is_aot_compiled
            and runtime != TargetRuntime.GENIEX_LLAMACPP
        ):
            return "Only runtimes that are compiled to context binary ahead of time are supported."

        if (
            self.has_multi_graph
            and not runtime.uses_hub_link
            and runtime != TargetRuntime.GENIEX_LLAMACPP
        ):
            return "Multi-graph models require runtimes that support linking (uses_hub_link)."

        if (
            not self.requires_aot_prepare
            and runtime.is_aot_compiled
            and not runtime.is_orchestrator_runtime
        ):
            # Only the JIT path is tested if this model does not require AOT prepare.
            # All AOT paths will fail if QNN fails.
            runtime = TargetRuntime.QNN_DLC

        if runtime == TargetRuntime.GENIE:
            # Don't surface Genie as a supported path when GenieX (QAIRT) fails.
            runtime = TargetRuntime.GENIEX_QAIRT

        if (
            reason := self.disabled_paths.peek_disable_reasons(precision, runtime)
        ) and reason.has_failure:
            if include_scorecard_failures and (
                scorecard_failure := reason.scorecard_failure
                or reason.scorecard_accuracy_failure
            ):
                return scorecard_failure
            if reason.issue_applies_to_device(device):
                if include_user_defined_failures and reason.issue is not None:
                    return reason.issue
                if include_timeouts and reason.causes_timeout:
                    return reason.issue or "Timeout"
        return None

    @property
    def supports_at_least_1_runtime(self) -> bool:
        supports_at_least_1_runtime = False
        for precision in self.supported_precisions:
            if supports_at_least_1_runtime:
                break
            for runtime in TargetRuntime:
                if supports_at_least_1_runtime:
                    break
                supports_at_least_1_runtime = self.is_supported(precision, runtime)
        return supports_at_least_1_runtime

    @property
    def default_quantized_precision(self) -> Precision | None:
        for precision in self.supported_precisions:
            assert isinstance(precision, Precision)
            if precision.has_quantized_activations:
                return precision
        return None

    @property
    def can_use_quantize_job(self) -> bool:
        """
        Whether the model can be quantized via quantize job.
        This may return true even if the model does list support for non-float precisions.
        """
        return not self.is_precompiled and not self.is_aimet

    @property
    def supports_quantization(self) -> bool:
        return any(x != Precision.float for x in self.supported_precisions)

    @property
    def default_precision(self) -> Precision:
        return self.supported_precisions[0]

    @property
    def default_runtime(self) -> TargetRuntime:
        """Default runtime for export scripts and README sample commands."""
        passing_paths = self.get_supported_paths_for_testing(only_include_passing=True)
        if not passing_paths:
            # qnn_context_binary is not a valid export path for orchestrator-only models.
            if self.only_allow_orchestrator_runtimes and self.orchestrator_runtimes:
                return next(
                    (
                        rt
                        for rt in self.orchestrator_runtimes
                        if rt is not TargetRuntime.GENIE
                    ),
                    self.orchestrator_runtimes[0],
                )
            return TargetRuntime.QNN_CONTEXT_BINARY

        runtimes = next(iter(passing_paths.values()))
        return next(
            (rt for rt in runtimes if rt is not TargetRuntime.GENIE),
            runtimes[0],
        )

    def get_supported_paths_for_testing(
        self, only_include_passing: bool = False
    ) -> dict[Precision, list[TargetRuntime]]:
        """
        Returns a set of {precision, runtime} pairs that are enabled for testing this model in scorecard.

        Parameters
        ----------
        only_include_passing
            If True, only includes runtimes that have no known failure reasons in manifest.
            If False, includes all runtimes that are enabled for testing, even if they have known failures.

        Returns
        -------
        dict[Precision, list[TargetRuntime]]
            A dictionary mapping precision to a list of runtimes that are supported for testing for that precision.

        Notes
        -----
        Certain supported pairs may be excluded from this list if they are not enabled for testing.
        For example, models that allow JIT (on-device) compile will not test AOT runtimes; we assume that if it works on JIT it will work on AOT.
        """
        out: dict[Precision, list[TargetRuntime]] = {}
        for precision in self.supported_precisions:
            if runtimes := [
                r
                for r in TargetRuntime
                if (
                    self.is_supported(
                        precision,
                        r,
                        consider_scorecard_failures=only_include_passing,
                        consider_user_defined_failures=only_include_passing,
                        consider_timeouts=True,
                    )
                    and (
                        (self.requires_aot_prepare and r.is_aot_compiled)
                        or (not self.requires_aot_prepare and not r.is_aot_compiled)
                    )
                )
            ]:
                out[precision] = runtimes
        return out

    def get_supported_paths_for_export(
        self,
    ) -> dict[Precision, list[TargetRuntime]]:
        """Return {precision: [runtime]} pairs available via ``qai-hub-models export``.

        Unlike :func:`get_supported_paths_for_testing`, does not filter out AOT
        runtimes on JIT models (still reachable via JIT-compile + link) or paths
        with known scorecard failures.
        """
        out: dict[Precision, list[TargetRuntime]] = {}
        for precision in self.supported_precisions:
            runtimes = [
                r
                for r in TargetRuntime
                if self.is_supported(
                    precision,
                    r,
                    consider_scorecard_failures=False,
                    consider_user_defined_failures=False,
                    consider_timeouts=False,
                )
            ]
            if runtimes:
                out[precision] = runtimes
        return out

    # =============================================================================
    # Combined validators from both schemas
    # =============================================================================

    @property
    def is_full_model_manifest(self) -> bool:
        """True if this manifest describes a model in ``models/<id>/``.

        False for ``templates/<name>/`` templates and ``datasets/<name>/``
        entries, which only declare dependencies (``templates:``) and
        optional external repo config. Determined by ``id`` matching a
        known model in the repo.
        """
        return self.id is not None and self.id in MODEL_IDS

    @model_validator(mode="after")
    def check_fields(self) -> QAIHMModelManifest:
        """Validate universal manifest correctness.

        This method enforces only the checks that must hold for *any*
        recipe — in-tree or external. Website / scorecard / publication
        lint (headline style, banner reachability, canary-device
        membership, ``status: published`` gating, arxiv URL style, etc.)
        lives in ``test_manifest_yamls.py::_check_website_facing`` and
        runs per-model in CI.

        Template and dataset manifests only need their ``templates:`` /
        ``external_repos:`` fields to be well-formed and skip the
        ``is_full_model_manifest`` block entirely.
        """
        if self.is_full_model_manifest:
            assert self.id is not None
            if " " in self.id or "-" in self.id:
                raise ValueError("Model IDs cannot contain spaces or dashes.")
            if self.id.lower() != self.id:
                raise ValueError("Model IDs must be lowercase.")

            missing_required = [
                field
                for field in (
                    "name",
                    "status",
                    "headline",
                    "domain",
                    "description",
                    "use_case",
                    "technical_details",
                    "license_type",
                )
                if getattr(self, field) is None
            ]
            if missing_required:
                raise ValueError(
                    f"Model {self.id}: manifest.yaml is missing required fields: "
                    f"{missing_required}"
                )

            assert self.name is not None
            assert self.license_type is not None

            if not self.license and self.license_type != MODEL_LICENSE.COMMERCIAL:
                raise ValueError("license cannot be empty")
            if (
                self.license_type.url is not None
                and self.license != self.license_type.url
            ):
                raise ValueError(
                    f"License {self.license_type!s} must have URL {self.license_type.url}"
                )
            if (
                self.license_type.restricts_redistribution
                and not self.restrict_model_sharing
            ):
                raise ValueError(
                    f"License {self.license_type!s} does not permit redistributing "
                    f"model weights; set restrict_model_sharing: true in manifest.yaml."
                )

            if self.model_type_llm and not self.llm_details:
                raise ValueError("llm_details must be set if model type is LLM")
            if self.llm_details and not self.model_type_llm:
                raise ValueError("Model type must be LLM if llm_details is set")

        if (
            self.python_version_greater_than_or_equal_to is None
            and self.python_version_greater_than_or_equal_to_reason is not None
        ):
            raise ValueError(
                "python_version_greater_than_or_equal_to_reason is set, but python_version_greater_than_or_equal_to is not."
            )
        if (
            self.python_version_greater_than_or_equal_to is not None
            and self.python_version_greater_than_or_equal_to_reason is None
        ):
            raise ValueError(
                "python_version_greater_than_or_equal_to must have a reason (python_version_greater_than_or_equal_to_reason) set."
            )
        if (
            self.python_version_less_than_reason is None
            and self.python_version_less_than is not None
        ):
            raise ValueError(
                "python_version_less_than must have a reason (python_version_less_than_reason) set."
            )
        if (
            self.python_version_less_than_reason is not None
            and self.python_version_less_than is None
        ):
            raise ValueError(
                "python_version_less_than_reason is set, but python_version_less_than is not."
            )
        for x in self.orchestrator_runtimes:
            if not x.is_orchestrator_runtime:
                raise ValueError(
                    f"{x.value} is not an orchestrator runtime, and should not be listed in orchestrator_runtimes."
                )
        if not self.is_collection_model and any(
            p in [Precision.mixed, Precision.mixed_with_float]
            for p in self.supported_precisions
        ):
            raise ValueError("Only collection models can have mixed precisions")

        # lm_quantization_details holds portable GenAI Lab recipes, which only exist
        # for LLM/VLM folders. Both LLMs and VLMs set model_type_llm=true (verified
        # on the VLM manifests), so gate on it: a non-LLM/VLM manifest can't carry a
        # recipe section (mirrors the llm_details gating above).
        if self.lm_quantization_details and not self.model_type_llm:
            raise ValueError(
                "lm_quantization_details can only be set on LLM/VLM models "
                "(model_type_llm must be true)."
            )

        # Every lm_quantization_details key must be a precision the model supports;
        # a recipe for a precision that isn't in supported_precisions is unreachable.
        unknown_precisions = set(self.lm_quantization_details) - set(
            self.supported_precisions
        )
        if unknown_precisions:
            raise ValueError(
                f"lm_quantization_details has recipes for precision(s) "
                f"{sorted(str(p) for p in unknown_precisions)} that are not in "
                f"supported_precisions ({sorted(str(p) for p in self.supported_precisions)})."
            )

        return self

    # =============================================================================
    # Loader / writer
    # =============================================================================

    @classmethod
    def from_model(cls: type[QAIHMModelManifest], model_id: str) -> QAIHMModelManifest:
        """Load manifest.yaml for the given model."""
        model_folder = QAIHM_MODELS_ROOT / model_id
        if not os.path.exists(model_folder):
            raise ValueError(f"{model_id} does not exist")

        manifest_path = model_folder / "manifest.yaml"
        return cls.from_yaml(manifest_path)

    def to_model_yaml(self) -> Path:
        """Write manifest.yaml."""
        assert self.id is not None
        manifest_path = QAIHM_MODELS_ROOT / self.id / "manifest.yaml"
        self.to_yaml(path=manifest_path)
        return manifest_path
