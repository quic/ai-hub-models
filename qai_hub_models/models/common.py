# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Optional

import numpy as np
import qai_hub as hub
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema
from qai_hub.client import QuantizeDtype
from qai_hub.hub import _global_client
from qai_hub.public_api_pb2 import Framework
from qai_hub.public_rest_api import get_framework_list
from typing_extensions import assert_never


class QAIRTVersion:
    # Map of <Hub URL -> Valid QAIRT Versions>
    _FRAMEWORKS: dict[str, list[Framework]] = {}
    HUB_FLAG = "--qairt_version"
    DEFAULT_AI_HUB_MODELS_API_VERSION = "2.34"  # Default version used by AI Hub Models. May be different than AI Hub's default.
    DEFAULT_QAIHM_TAG = "qaihm_default"
    DEFAULT_AIHUB_TAG = "default"
    LATEST_AIHUB_TAG = "latest"

    def __init__(
        self, version_or_tag: str, return_default_if_does_not_exist: bool = False
    ):
        if re.match(r"(\d+)\.(\d+)(\.\d+)*", version_or_tag):
            api_version = version_or_tag
            api_tag = None
        else:
            api_tag = version_or_tag
            api_version = None

        self._api_url, frameworks = QAIRTVersion._load_frameworks()
        framework = None
        qaihm_default_framework = None
        default_framework = None
        for fw in frameworks:
            if (api_version and fw.full_version.startswith(api_version)) or (
                api_tag and api_tag in [x for x in fw.api_tags]
            ):
                framework = fw
                break
            if QAIRTVersion.DEFAULT_AIHUB_TAG in fw.api_tags:
                default_framework = fw
            if fw.api_version == QAIRTVersion.DEFAULT_AI_HUB_MODELS_API_VERSION:
                qaihm_default_framework = fw
                if version_or_tag == QAIRTVersion.DEFAULT_QAIHM_TAG:
                    framework = fw
                    break

        if not framework:
            if not self._api_url:
                self.api_version = api_version or "UNKNOWN"
                self.full_version = api_version or "UNKNOWN"
                self.tags = [api_tag] if api_tag else []
            elif return_default_if_does_not_exist:
                fallback_framework = qaihm_default_framework or default_framework
                assert fallback_framework is not None
                print(
                    f"Warning: QAIRT version {version_or_tag} does not exist. Using the default version ({fallback_framework.api_version}) instead."
                )
                self.api_version = fallback_framework.api_version
                self.full_version = fallback_framework.full_version
                self.tags = [x for x in fallback_framework.api_tags]
            else:
                all_versions_str = "\n    ".join([str(x) for x in QAIRTVersion.all()])
                raise ValueError(
                    f"QAIRT version {version_or_tag} is not supported by AI Hub. Available versions are:\n    {all_versions_str}"
                )
        else:
            self.api_version = framework.api_version
            self.full_version = framework.full_version
            self.tags = [x for x in framework.api_tags]
            if self.api_version == QAIRTVersion.DEFAULT_AI_HUB_MODELS_API_VERSION:
                self.tags.append(QAIRTVersion.DEFAULT_QAIHM_TAG)

    @property
    def hub_option(self) -> str:
        """String flag to pass to hub to use this Hub version."""
        if self.is_default:
            return ""
        return (
            f"{QAIRTVersion.HUB_FLAG} {self.tags[0]}"
            if self.tags and self.tags[0] != QAIRTVersion.DEFAULT_QAIHM_TAG
            else f"{QAIRTVersion.HUB_FLAG} {self.api_version}"
        )

    @property
    def is_qaihm_default(self) -> bool:
        """
        Whether this is the default QAIRT version used by AI Hub Models.

        AI HUB MODELS' DEFAULT QAIRT VERSION MIGHT BE DIFFERENT THAN AI HUB's DEFAULT VERSION.
        """
        return self.api_version == QAIRTVersion.DEFAULT_AI_HUB_MODELS_API_VERSION

    @property
    def is_default(self) -> bool:
        """Whether this is the default AI Hub QAIRT version."""
        return bool(self.tags) and QAIRTVersion.DEFAULT_AIHUB_TAG in self.tags

    def __eq__(self, other):
        if isinstance(other, str):
            return self.full_version.startswith(other) or other in self.tags
        if isinstance(other, QAIRTVersion):
            return self.full_version == other.full_version
        return False

    def __str__(self):
        return (
            f"QAIRT v{self.api_version}"
            + (
                " | UNVERIFIED - NO AI HUB ACCESS"
                if not self._api_url
                else f" | {self.full_version}"
            )
            + (f" | {', '.join(self.tags)}" if self.tags else "")
        )

    def __repr__(self):
        return str(self)

    @staticmethod
    def qaihm_default() -> QAIRTVersion:
        """
        Default QAIRT version used by AI Hub Models.

        THIS MIGHT BE DIFFERENT THAN AI HUB's DEFAULT VERSION.
        """
        try:
            return QAIRTVersion(QAIRTVersion.DEFAULT_AI_HUB_MODELS_API_VERSION)
        except ValueError as e:
            msg = e.args[0]
            if "is not supported by AI Hub" in msg:
                raise ValueError(
                    msg
                    + f"\n\nAI Hub no longer supports the standard QAIRT version ({QAIRTVersion.DEFAULT_AI_HUB_MODELS_API_VERSION}) used by this version of AI Hub Models.\n"
                    "Pass --compile-options='--qairt_version=default' and/or --profile-options='qairt_version=default' to use the current default available on AI Hub.\n"
                    "DO THIS AT YOUR OWN RISK -- Older versions of AI Hub Models are not guarunteed to work with newer versions of QAIRT."
                )
            else:
                raise e

    @staticmethod
    def default() -> QAIRTVersion:
        """Default QAIRT version on AI Hub."""
        return QAIRTVersion(QAIRTVersion.DEFAULT_AIHUB_TAG)

    @staticmethod
    def latest() -> QAIRTVersion:
        """Latest QAIRT version on AI Hub."""
        return QAIRTVersion(QAIRTVersion.LATEST_AIHUB_TAG)

    @staticmethod
    def all() -> list[QAIRTVersion]:
        _, frameworks = QAIRTVersion._load_frameworks()
        return [QAIRTVersion(f.api_version) for f in frameworks]

    @staticmethod
    def _load_frameworks() -> tuple[str, list[Framework]]:
        try:
            api_url = _global_client.config.api_url
            if api_url not in QAIRTVersion._FRAMEWORKS:
                QAIRTVersion._FRAMEWORKS[api_url] = [
                    f
                    for f in get_framework_list(_global_client.config).frameworks
                    if f.name == "QAIRT"
                ]
        except Exception:
            # No hub API Access
            api_url = ""
            QAIRTVersion._FRAMEWORKS[api_url] = []

        return (api_url, QAIRTVersion._FRAMEWORKS[api_url])


@unique
class InferenceEngine(Enum):
    """
    The inference engine that executes a TargetRuntime asset.
    """

    TFLITE = "tflite"
    QNN = "qnn"
    ONNX = "onnx"


@unique
class TargetRuntime(Enum):
    """
    Compilation target.
    """

    TFLITE = "tflite"
    QNN_DLC = "qnn_dlc"
    QNN_CONTEXT_BINARY = "qnn_context_binary"
    ONNX = "onnx"
    PRECOMPILED_QNN_ONNX = "precompiled_qnn_onnx"

    @staticmethod
    def from_hub_model_type(type: hub.SourceModelType):
        for rt in TargetRuntime:
            if rt.hub_model_type == type:
                return rt
        raise ValueError(f"Unsupported Hub model type: {type}")

    @property
    def inference_engine(self) -> InferenceEngine:
        if self == TargetRuntime.TFLITE:
            return InferenceEngine.TFLITE
        if self == TargetRuntime.QNN_CONTEXT_BINARY or self == TargetRuntime.QNN_DLC:
            return InferenceEngine.QNN
        if self == TargetRuntime.ONNX or self == TargetRuntime.PRECOMPILED_QNN_ONNX:
            return InferenceEngine.ONNX
        assert_never(self)

    @property
    def file_extension(self) -> str:
        """
        The file extension (without the .) assets for this runtime use.
        """
        if self == TargetRuntime.QNN_CONTEXT_BINARY:
            return "bin"
        if self == TargetRuntime.QNN_DLC:
            return "dlc"
        if self == TargetRuntime.ONNX:
            return "onnx"
        if self == TargetRuntime.TFLITE:
            return "tflite"
        if self == TargetRuntime.PRECOMPILED_QNN_ONNX:
            return "onnx"

        assert_never(self)

    @property
    def hub_model_type(self) -> hub.SourceModelType:
        """
        The associated hub SourceModelType for assets for this TargetRuntime.
        """
        if self == TargetRuntime.QNN_CONTEXT_BINARY:
            return hub.SourceModelType.QNN_CONTEXT_BINARY
        if self == TargetRuntime.QNN_DLC:
            return hub.SourceModelType.QNN_DLC
        if self == TargetRuntime.PRECOMPILED_QNN_ONNX or self == TargetRuntime.ONNX:
            return hub.SourceModelType.ONNX
        if self == TargetRuntime.TFLITE:
            return hub.SourceModelType.TFLITE
        assert_never(self)

    @property
    def channel_last_native_execution(self) -> bool:
        """
        If true, this runtime natively executes ops in NHWC (channel-last) format.
        If false, the runtime executes ops in NCHW (channel-first) format.
        """
        return self.is_aot_compiled or self.inference_engine in [
            InferenceEngine.QNN,
            InferenceEngine.TFLITE,
        ]

    @property
    def qairt_version_changes_compilation(self) -> bool:
        """
        Returns true if different versions of Qualcomm AI Runtime will affect how this runtime is compiled.
        """
        return self.is_aot_compiled or self.inference_engine == InferenceEngine.QNN

    def supports_precision(self, precision: Precision) -> bool:
        # Float is always supported.
        if precision == Precision.float:
            return True

        # Check quantized types
        if self == TargetRuntime.TFLITE:
            return precision == Precision.w8a8
        if self == TargetRuntime.ONNX:
            return precision in [
                Precision.w8a8,
                Precision.w8a16,
                # The following three are enabled tentatively
                # (not experimentally verified)
                Precision.w16a16,
                Precision.w4a16,
                Precision.w4,
            ]
        if (
            self == TargetRuntime.QNN_DLC
            or self == TargetRuntime.QNN_CONTEXT_BINARY
            or self == TargetRuntime.PRECOMPILED_QNN_ONNX
        ):
            return precision in [
                Precision.w8a8,
                Precision.w8a16,
                Precision.w4a16,
                Precision.w4,
                Precision.w16a16,
            ]

        assert_never(self)

    @property
    def compilation_uses_qnn_converters(self) -> bool:
        """
        If true, this runtime uses the QNN converters when compiling.
        """
        return self in [
            TargetRuntime.PRECOMPILED_QNN_ONNX,
            TargetRuntime.QNN_CONTEXT_BINARY,
        ]

    def get_target_runtime_flag(
        self,
        device: Optional[hub.Device] = None,
    ) -> str:
        """
        AI Hub job flag for compiling to this runtime.
        """
        return f"--target_runtime {self.value}"

    @property
    def default_qairt_version(self) -> QAIRTVersion:
        """
        The default QAIRT version for this runtime.
        """
        if self == TargetRuntime.PRECOMPILED_QNN_ONNX:
            # Return default in a possible future where 2.31 is deprecated on Hub.
            return QAIRTVersion("2.33", return_default_if_does_not_exist=True)
        return QAIRTVersion.qaihm_default()

    @property
    def is_aot_compiled(self) -> bool:
        """
        Returns true if this asset is fully compiled ahead of time (before running on target).
        This means the compiled asset contains a QNN context binary.
        """
        return self.aot_equivalent == self

    @property
    def aot_equivalent(self) -> TargetRuntime | None:
        """
        Returns the equivalent runtime that is compiled ahead of time.
        Returns None if there is no equivalent runtime that is compiled ahead of time.
        """
        inference_engine = self.inference_engine
        if inference_engine == InferenceEngine.ONNX:
            return TargetRuntime.PRECOMPILED_QNN_ONNX
        if inference_engine == InferenceEngine.QNN:
            return TargetRuntime.QNN_CONTEXT_BINARY
        if inference_engine == InferenceEngine.TFLITE:
            return None
        assert_never(inference_engine)

    @property
    def jit_equivalent(self) -> TargetRuntime:
        """
        Returns the equivalent runtime that is compiled "just in time" on target.
        """
        inference_engine = self.inference_engine
        if inference_engine == InferenceEngine.ONNX:
            return TargetRuntime.ONNX
        if inference_engine == InferenceEngine.QNN:
            return TargetRuntime.QNN_DLC
        if inference_engine == InferenceEngine.TFLITE:
            return TargetRuntime.TFLITE
        assert_never(inference_engine)


class Precision:
    """
    Stores a precision configuration for a model's graph.

    Precision can represent any precision that AI Hub's QuantizeJob supports.
    It also can represent floating point (no quantization).

    Mixed precision is not currently representable by this class.
    """

    # Common Precision Instances
    float: Precision
    w8a8: Precision
    w8a16: Precision
    w16a16: Precision
    w4a16: Precision
    w4: Precision

    def __init__(
        self, weights_type: QuantizeDtype | None, activations_type: QuantizeDtype | None
    ):
        self.weights_type: QuantizeDtype | None = weights_type
        self.activations_type: QuantizeDtype | None = activations_type

    @staticmethod
    def parse(obj: Any) -> Precision:
        if isinstance(obj, Precision):
            return obj
        if not isinstance(obj, str):
            raise ValueError(f"Unknown type {obj} for parsing to Precision")
        string = obj
        if string == "float":
            return Precision.float

        atype = None
        wtype = None
        if match := re.match(r"(w\d+)?(a\d+)?", string):
            wtype = match.group(1)
            atype = match.group(2)
        if match := re.match(r"(a\d+)?(w\d+)?", string):
            atype = atype or match.group(1)
            wtype = wtype or match.group(2)

        if not atype and not wtype:
            raise ValueError(
                "Unsupported quantization type string. Example valid strings: float, w8a16, a8w8, w8, a16, ..."
            )

        atype_enum_name = f"INT{atype[1:]}" if atype else None
        wtype_enum_name = f"INT{wtype[1:]}" if wtype else None
        for enum_name, bit_width, name in [
            (atype_enum_name, atype, "activations"),
            (wtype_enum_name, wtype, "weights"),
        ]:
            if not bit_width:
                continue
            if enum_name not in QuantizeDtype._member_names_:
                raise ValueError(
                    f"Unsupported bit width {bit_width} for quantization {name}. Supported: {','. join([str(x) for x in QuantizeDtype])}"
                )

        return Precision(
            QuantizeDtype[wtype_enum_name] if wtype_enum_name else None,
            QuantizeDtype[atype_enum_name] if atype_enum_name else None,
        )

    @property
    def has_float_activations(self) -> bool:
        # Returns true if model activations are not quantized.
        return self.activations_type is None

    @property
    def has_float_weights(self) -> bool:
        # Returns true if model weights are not quantized.
        return self.weights_type is None

    def __str__(self) -> str:
        if self == Precision.float:
            return "float"

        return (f"w{self.weights_type.name[3:]}" if self.weights_type else "") + (
            f"a{self.activations_type.name[3:]}" if self.activations_type else ""
        )

    def __repr__(self):
        return str(self)

    def __eq__(self, value):
        if not isinstance(value, Precision):
            return False
        return (
            value.activations_type == self.activations_type
            and value.weights_type == self.weights_type
        )

    def __hash__(self) -> int:
        return hash(str(self))

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.with_info_after_validator_function(
            lambda obj, _: cls.parse(obj),
            handler(Any),
            field_name=handler.field_name,
            serialization=core_schema.plain_serializer_function_ser_schema(
                Precision.__str__, when_used="json"
            ),
        )


Precision.float = Precision(None, None)
Precision.w8a8 = Precision(QuantizeDtype.INT8, QuantizeDtype.INT8)
Precision.w8a16 = Precision(QuantizeDtype.INT8, QuantizeDtype.INT16)
Precision.w16a16 = Precision(QuantizeDtype.INT16, QuantizeDtype.INT16)
Precision.w4a16 = Precision(QuantizeDtype.INT4, QuantizeDtype.INT16)
Precision.w4 = Precision(QuantizeDtype.INT4, None)


@unique
class SourceModelFormat(Enum):
    ONNX = 0
    TORCHSCRIPT = 1


SampleInputsType = dict[str, list[np.ndarray]]


@dataclass
class ExportResult:
    compile_job: Optional[hub.CompileJob] = None
    quantize_job: Optional[hub.QuantizeJob] = None
    profile_job: Optional[hub.ProfileJob] = None
    inference_job: Optional[hub.InferenceJob] = None
    link_job: Optional[hub.LinkJob] = None
