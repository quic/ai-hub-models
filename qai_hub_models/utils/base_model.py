# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import inspect
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Union

import torch
from qai_hub.client import Device

from qai_hub_models.models.common import (
    Precision,
    QAIRTVersion,
    SampleInputsType,
    SourceModelFormat,
    TargetRuntime,
)
from qai_hub_models.models.protocols import (
    ExecutableModelProtocol,
    FromPrecompiledProtocol,
    FromPretrainedProtocol,
    HubModelProtocol,
    PretrainedHubModelProtocol,
)
from qai_hub_models.utils.checkpoint import CheckpointSpec
from qai_hub_models.utils.input_spec import (
    InputSpec,
    broadcast_data_to_multi_batch,
    get_batch_size,
    make_torch_inputs,
)
from qai_hub_models.utils.transpose_channel import transpose_channel_first_to_last


class CollectionModel:
    """
    Model that glues together several BaseModels or BasePrecompiledModel.

    See test_base_model.py for usage examples.
    """

    component_classes: list[type[BaseModel]] = []
    component_class_names: list[str] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Make copies of the list for each subclass so they don't share state
        cls.component_classes = list(cls.component_classes)
        cls.component_class_names = list(cls.component_class_names)

    def __init__(self, *args):
        component_names = type(self).component_class_names
        components: dict[
            str, BaseModel | BasePrecompiledModel
        ] = {}  # class name -> instantiated object

        # Process positional arguments.
        if len(args) != len(component_names):
            raise ValueError(
                f"CollectionModel has {len(component_names)} ordered arguments, "
                "each of which should correspond with a single component."
            )
        for i, (name, arg) in enumerate(zip(component_names, args)):
            expected_class = type(self).component_classes[i]
            if not isinstance(arg, (expected_class, ExecutableModelProtocol)):
                raise ValueError(
                    f"Expected component '{name}' to be an instance "
                    f"of {name} or ExecutableModelProtocol, got {type(arg).__name__}"
                )
            assert isinstance(arg, (BaseModel, BasePrecompiledModel))
            components[name] = arg

        # Check that all required components are provided.
        missing = [name for name in component_names if name not in components]
        if missing:
            raise ValueError(f"Missing components for: {missing}")

        self.components = components

    @staticmethod
    def add_component(component_class: type, component_name: str | None = None):
        """
        Decorator to add a component (a subclass of BaseModel or
        BasePrecompiledModel) to a CollectionModel.  The component is
        inserted at the beginning of the class-level dictionary `component_classes`,
        so that the outer decorator appears first.

        See test_base_model.py for usage examples.
        """

        def decorator(subclass):
            name = component_name or component_class.__name__
            if name in subclass.component_class_names:
                raise ValueError(f"Component with name {name} already registered")
            # prepend list
            subclass.component_classes.insert(0, component_class)
            subclass.component_class_names.insert(0, name)

            # Component class from_pretrained would look for this folder
            # under checkpoint dir
            setattr(component_class, "default_subfolder", name)
            return subclass

        return decorator

    @staticmethod
    def reset_components():
        """
        Decorator to erase all components set on a CollectionModel.
        Useful when subclassing a CollectionModel and overriding component classes.

        See test_base_model.py for usage examples.
        """

        def decorator(subclass):
            subclass.component_classes = []
            subclass.component_class_names = []
            return subclass

        return decorator

    @staticmethod
    def eval_datasets() -> list[str]:
        """
        Returns list of strings with names of all datasets on which
        this model can be evaluated.

        All names must be registered in qai_hub_models/datasets/__init__.py
        """
        return []


class PretrainedCollectionModel(CollectionModel, FromPretrainedProtocol):
    @classmethod
    def from_pretrained(
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        host_device: Union[torch.device, str] = torch.device("cpu"),
        **kwargs,  # any extra you might want to forward
    ) -> PretrainedCollectionModel:
        """
        Instantiate the collection by delegating to each component_cls.from_pretrained,
        but only passing it the arguments it actually declares.
        """
        # common kwargs we’d like to pass
        base_kwargs = {
            "checkpoint": checkpoint,
            "host_device": host_device,
            **kwargs,
        }

        # helper to call any fn with only the supported subset of base_kwargs
        def _call_with_supported(fn, all_kwargs):
            sig = inspect.signature(fn)
            params = sig.parameters.values()

            # If fn accepts **kwargs, just pass everything through
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
                return fn(**all_kwargs)

            # Otherwise, filter to only the names fn explicitly declares
            supported = {
                name: value
                for name, value in all_kwargs.items()
                if name in sig.parameters
            }
            return fn(**supported)

        components = []
        for name, component_cls in zip(
            cls.component_class_names, cls.component_classes
        ):
            # call component_cls.from_pretrained but only with the args it accepts
            try:
                comp = _call_with_supported(component_cls.from_pretrained, base_kwargs)
            except Exception as e:
                raise AttributeError(
                    f"Component '{component_cls.__name__}' does not have "
                    f"a callable from_pretrained method. {e}"
                )
            components.append(comp)

        # now build and return your collection model however CollectionModel expects
        return cls(*components)


class PrecompiledCollectionModel(CollectionModel, FromPrecompiledProtocol):
    @classmethod
    def from_precompiled(cls):
        """
        Instantiate the CollectionModel by instantiating all registered components
        using their own from_precompiled() methods.
        """
        components = []
        for component_cls in cls.component_classes:
            if not (
                hasattr(component_cls, "from_precompiled")
                and callable(component_cls.from_precompiled)
            ):
                raise AttributeError(
                    f"Component '{component_cls.__name__}' does not have "
                    "a callable from_precompiled method"
                )
            components.append(component_cls.from_precompiled())
        return cls(*components)


class HubModel(HubModelProtocol):
    """
    Base interface for AI Hub models.
    """

    def __init__(self):
        # If a child class implements _get_input_spec_for_instance(),
        # then calling `get_input_spec` on the instance will redirect to it.
        if self._get_input_spec_for_instance.__module__ != __name__:
            self.get_input_spec = self._get_input_spec_for_instance  # type: ignore[method-assign]
        if self._get_output_names_for_instance.__module__ != __name__:
            self.get_output_names = self._get_output_names_for_instance  # type: ignore[method-assign]
        if self._get_channel_last_inputs_for_instance.__module__ != __name__:
            self.get_channel_last_inputs = self._get_channel_last_inputs_for_instance  # type: ignore[method-assign]
        if self._get_channel_last_outputs_for_instance.__module__ != __name__:
            self.get_channel_last_outputs = self._get_channel_last_outputs_for_instance  # type: ignore[method-assign]

    def _get_input_spec_for_instance(self, *args, **kwargs) -> InputSpec:
        """
        Get the input specifications for an instance of this model.

        Typically this will pre-fill inputs of get_input_spec
        with values determined by instance members of the model class.

        If this function is implemented by a child class, the initializer for BaseModel
        will automatically override get_input_spec with this function
        when the class is instantiated.
        """
        raise NotImplementedError

    def _get_output_names_for_instance(self, *args, **kwargs) -> list[str]:
        """
        Get the output names for an instance of this model.

        If this function is implemented by a child class, the initializer for BaseModel
        will automatically override get_output_names with this function
        when the class is instantiated.
        """
        raise NotImplementedError

    def _get_channel_last_inputs_for_instance(self, *args, **kwargs) -> list[str]:
        """
        Get the channel last input names for an instance of this model.

        If this function is implemented by a child class, the initializer for BaseModel
        will automatically override get_channel_last_inputs with this function
        when the class is instantiated.
        """
        raise NotImplementedError

    def _get_channel_last_outputs_for_instance(self, *args, **kwargs) -> list[str]:
        """
        Get the channel last output names for an instance of this model.

        If this function is implemented by a child class, the initializer for BaseModel
        will automatically override get_channel_last_outputs with this function
        when the class is instantiated.
        """
        raise NotImplementedError

    def sample_inputs(
        self,
        input_spec: InputSpec | None = None,
        use_channel_last_format: bool = True,
        **kwargs,
    ) -> SampleInputsType:
        """
        Returns a set of sample inputs for the model.

        For each input name in the model, a list of numpy arrays is provided.
        If the returned set is batch N, all input names must contain exactly N numpy arrays.

        Subclasses should NOT override this. They should instead override _sample_inputs_impl.

        This function will invoke _sample_inputs_impl and then apply any required channel
            format transposes.
        """
        sample_inputs = self._sample_inputs_impl(input_spec, **kwargs)
        if input_spec is not None:
            batch_size = get_batch_size(input_spec)
            if batch_size is not None and batch_size > 1:
                sample_inputs = broadcast_data_to_multi_batch(input_spec, sample_inputs)
        if use_channel_last_format and self.get_channel_last_inputs():
            return transpose_channel_first_to_last(
                self.get_channel_last_inputs(), sample_inputs
            )
        return sample_inputs

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        """
        This is a default implementation that returns a single random data array
        for each input name based on the shapes and dtypes in `get_input_spec`.

        A subclass may choose to override this and fetch a batch of real input data
        from a data source.

        See the `sample_inputs` doc for the expected format.
        """
        if not input_spec:
            input_spec = self.get_input_spec()
        inputs_dict = {}
        inputs_list = make_torch_inputs(input_spec)
        for i, input_name in enumerate(input_spec.keys()):
            inputs_dict[input_name] = [inputs_list[i].numpy()]
        return inputs_dict

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
    ) -> str:
        """
        AI Hub profile options recommended for the model.
        """
        if QAIRTVersion.HUB_FLAG not in other_profile_options:
            other_profile_options = (
                other_profile_options
                + f" {target_runtime.default_qairt_version.hub_option}"
            )

        return other_profile_options

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        """
        A list of input names that should be transposed to channel-last format
            for the on-device model in order to improve performance.
        """
        return []

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        """
        A list of output names that should be transposed to channel-last format
            for the on-device model in order to improve performance.
        """
        return []


class BaseModel(
    torch.nn.Module,
    HubModel,
    PretrainedHubModelProtocol,
    ExecutableModelProtocol,
):
    """
    A pre-trained PyTorch model with helpers for submission to AI Hub.
    """

    def __init__(self, model: torch.nn.Module | None = None):
        torch.nn.Module.__init__(self)  # Initialize Torch Module
        HubModel.__init__(self)  # Initialize Hub Model
        self.eval()
        if model is not None:
            self.model = model

    def __setattr__(self, name: str, value: Any) -> None:
        """
        When a new torch.nn.Module attribute is added, we want to set it to eval mode.
            If this model is being trained, calling `model.train()`
            will reverse all of these.
        """
        if isinstance(value, torch.nn.Module) and not self.training:
            value.eval()
        torch.nn.Module.__setattr__(self, name, value)

    def __call__(self, *args, **kwargs):
        """
        If a model is in eval mode (which equates to self.training == False),
            we don't want to compute gradients when doing the forward pass.
        """
        context_fn = nullcontext if self.training else torch.no_grad
        with context_fn():
            return torch.nn.Module.__call__(self, *args, **kwargs)

    def convert_to_torchscript(
        self, input_spec: InputSpec | None = None, check_trace: bool = True
    ) -> Any:
        """
        Converts the torch module to a torchscript trace, which
        is the format expected by qai hub.

        This is a default implementation that may be overriden by a subclass.
        """
        if not input_spec:
            input_spec = self.get_input_spec()

        # Torchscript should never be trained, so disable gradients for all parameters.
        # Need to do this on a model copy, in case the original model is being trained.
        model_copy = deepcopy(self)
        for param in model_copy.parameters():
            param.requires_grad = False

        return torch.jit.trace(
            model_copy, make_torch_inputs(input_spec), check_trace=check_trace
        )

    def convert_to_hub_source_model(
        self,
        target_runtime: TargetRuntime,
        output_path: str | Path,
        input_spec: InputSpec | None = None,
        check_trace: bool = True,
        external_onnx_weights: bool = False,
        output_names: Optional[list[str]] = None,
    ) -> Optional[str]:
        """
        Convert to a AI Hub source model appropriate for the export method.
        """
        # Local import to prevent circular dependency
        from qai_hub_models.utils.inference import prepare_compile_zoo_model_to_hub

        source_model = prepare_compile_zoo_model_to_hub(
            self,
            source_model_format=self.preferred_hub_source_model_format(target_runtime),
            target_runtime=target_runtime,
            output_path=output_path,
            input_spec=input_spec,
            check_trace=check_trace,
            external_onnx_weights=external_onnx_weights,
            output_names=output_names or self.get_output_names(),
        )
        return source_model

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Optional[Device] = None,
    ) -> str:
        """
        AI Hub compile options recommended for the model.
        """
        compile_options = ""
        if "--target_runtime" not in other_compile_options:
            compile_options = target_runtime.get_target_runtime_flag(device)
        if (
            QAIRTVersion.HUB_FLAG not in other_compile_options
            and target_runtime.qairt_version_changes_compilation
        ):
            compile_options = (
                compile_options + f" {target_runtime.default_qairt_version.hub_option}"
            )

        compile_options += f" --output_names {','.join(self.get_output_names())}"

        if target_runtime != TargetRuntime.ONNX:
            if self.get_channel_last_inputs():
                channel_last_inputs = ",".join(self.get_channel_last_inputs())
                compile_options += f" --force_channel_last_input {channel_last_inputs}"
            if self.get_channel_last_outputs():
                channel_last_outputs = ",".join(self.get_channel_last_outputs())
                compile_options += (
                    f" --force_channel_last_output {channel_last_outputs}"
                )

        if precision.activations_type is not None:
            compile_options += " --quantize_io"
            if target_runtime == TargetRuntime.TFLITE:
                # uint8 is the easiest I/O type for integration purposes,
                # especially for image applications. Images are always
                # uint8 RGB when coming from disk or a camera.
                #
                # Uint8 has not been thoroughly tested with other paths,
                # so it is enabled only for TF Lite today.
                compile_options += " --quantize_io_type uint8"

        if other_compile_options != "":
            return compile_options + " " + other_compile_options

        return compile_options

    def preferred_hub_source_model_format(
        self, target_runtime: TargetRuntime
    ) -> SourceModelFormat:
        """
        Source model format preferred for conversion on AI Hub.
        """
        return SourceModelFormat.TORCHSCRIPT

    def get_unsupported_reason(
        self, target_runtime: TargetRuntime, device: Device
    ) -> None | str:
        """
        Report the reason if any combination of runtime and device isn't
        supported.
        """
        return None

    @staticmethod
    def eval_datasets() -> list[str]:
        """
        Returns list of strings with names of all datasets on which
        this model can be evaluated.

        All names must be registered in qai_hub_models/datasets/__init__.py
        """
        return []

    @staticmethod
    def calibration_dataset_name() -> str | None:
        """
        Name of the dataset to use for calibration when quantizing the model.

        Must be registered in qai_hub_models/datasets/__init__.py
        """
        return None

    def get_hub_quantize_options(self, precision: Precision) -> str:
        """
        AI Hub quantize options recommended for the model.
        """
        return "--range_scheme min_max" if precision == Precision.w8a16 else ""


class BasePrecompiledModel(HubModel, FromPrecompiledProtocol):
    """
    A pre-compiled hub model.
    Model PyTorch source is not available, but compiled assets are available.
    """

    def __init__(self, target_model_path: str):
        self.target_model_path = target_model_path

    def get_target_model_path(self) -> str:
        """Get the path to the compiled asset for this model on disk."""
        return self.target_model_path

    def get_unsupported_reason(
        self, target_runtime: TargetRuntime, device: Device
    ) -> None | str:
        """
        Report the reason if any combination of runtime and device isn't
        supported.
        """
        return None
