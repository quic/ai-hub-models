# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import re
from enum import Enum, unique

import qai_hub as hub
from pydantic import Field
from typing_extensions import assert_never

from qai_hub_models.models.common import InferenceEngine
from qai_hub_models.scorecard.device import ScorecardDevice
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.path_helpers import QAIHM_PACKAGE_ROOT

SCORECARD_DEVICE_YAML_PATH = QAIHM_PACKAGE_ROOT / "devices_and_chipsets.yaml"


@unique
class WebsiteWorld(Enum):
    Mobile = "Mobile"
    Compute = "Compute"
    Automotive = "Automotive"
    IoT = "IoT"
    XR = "XR"

    @staticmethod
    def from_form_factor(form_factor: ScorecardDevice.FormFactor) -> WebsiteWorld:
        if (
            form_factor == ScorecardDevice.FormFactor.PHONE
            or form_factor == ScorecardDevice.FormFactor.TABLET
        ):
            return WebsiteWorld.Mobile
        if form_factor == ScorecardDevice.FormFactor.XR:
            return WebsiteWorld.XR
        if form_factor == ScorecardDevice.FormFactor.COMPUTE:
            return WebsiteWorld.Compute
        if form_factor == ScorecardDevice.FormFactor.IOT:
            return WebsiteWorld.IoT
        if form_factor == ScorecardDevice.FormFactor.AUTO:
            return WebsiteWorld.Automotive
        assert_never(form_factor)


@unique
class WebsiteIcon(Enum):
    Car = "Car"
    IoT_Chip = "IoT_Chip"
    IoT_Drone = "IoT_Drone"
    Laptop_Generic = "Laptop_Generic"
    Laptop_X_Elite = "Laptop_X_Elite"
    Phone_S21 = "Phone_S21"
    Phone_S22 = "Phone_S22"
    Phone_S23 = "Phone_S23"
    Phone_S23_Ultra = "Phone_S23_Ultra"
    Phone_S24 = "Phone_S24"
    Phone_S24_Ultra = "Phone_S24_Ultra"
    Tablet_Android = "Tablet_Android"
    XR_Headset = "XR_Headset"

    @staticmethod
    def from_device(device: ScorecardDevice) -> WebsiteIcon:
        if device.form_factor == ScorecardDevice.FormFactor.PHONE:
            if device.chipset == "qualcomm-snapdragon-888":
                return WebsiteIcon.Phone_S21
            if device.chipset == "qualcomm-snapdragon-8gen1":
                return WebsiteIcon.Phone_S22
            if device.chipset == "qualcomm-snapdragon-8gen2":
                if "Ultra" in device.reference_device_name:
                    return WebsiteIcon.Phone_S23_Ultra
                return WebsiteIcon.Phone_S23
            if device.chipset == "qualcomm-snapdragon-8gen3":
                if "Ultra" in device.reference_device_name:
                    return WebsiteIcon.Phone_S24_Ultra
                return WebsiteIcon.Phone_S24
            return WebsiteIcon.Phone_S21
        if device.form_factor == ScorecardDevice.FormFactor.COMPUTE:
            if device.chipset in [
                "qualcomm-snapdragon-8cxgen3",
                "qualcomm-snapdragon-x-plus-8-core",
                "qualcomm-snapdragon-x-elite",
            ]:
                return WebsiteIcon.Laptop_X_Elite
            return WebsiteIcon.Laptop_Generic
        if device.form_factor == ScorecardDevice.FormFactor.TABLET:
            return WebsiteIcon.Tablet_Android
        if device.form_factor == ScorecardDevice.FormFactor.XR:
            return WebsiteIcon.XR_Headset
        if device.form_factor == ScorecardDevice.FormFactor.IOT:
            if device.chipset in [
                "qualcomm-qcs6490-proxy",
                "qualcomm-qcs8250-proxy",
                "qualcomm-qcs8275-proxy",
                "qualcomm-qcs9075-proxy",
            ] and device.reference_device_name not in [
                "RB3 Gen 2 (Proxy)",
                "RB5 (Proxy)",
            ]:
                return WebsiteIcon.IoT_Chip
            return WebsiteIcon.IoT_Drone
        if device.form_factor == ScorecardDevice.FormFactor.AUTO:
            return WebsiteIcon.Car
        assert_never(device.form_factor)


class FormFactorYaml(BaseQAIHMConfig):
    display_name: str
    world: WebsiteWorld

    @staticmethod
    def from_form_factor(form_factor: ScorecardDevice.FormFactor):
        return FormFactorYaml(
            display_name=FormFactorYaml._form_factor_to_display_name(form_factor),
            world=WebsiteWorld.from_form_factor(form_factor),
        )

    @staticmethod
    def _form_factor_to_display_name(ff: ScorecardDevice.FormFactor) -> str:
        if ff == ScorecardDevice.FormFactor.AUTO:
            return "Automotive"
        return ff.value


class DeviceDetailsYaml(BaseQAIHMConfig):
    chipset: str
    os: ScorecardDevice.OperatingSystem
    form_factor: ScorecardDevice.FormFactor
    vendor: str
    icon: WebsiteIcon
    npu_count: int = 1

    @staticmethod
    def from_device(device: ScorecardDevice):
        return DeviceDetailsYaml(
            chipset=device.chipset,
            os=device.os,
            form_factor=device.form_factor,
            vendor=device.vendor,
            icon=WebsiteIcon.from_device(device),
            npu_count=device.npu_count,
        )


class ChipsetYaml(BaseQAIHMConfig):
    aliases: list[str]
    marketing_name: str
    world: WebsiteWorld

    @staticmethod
    def from_device(device: ScorecardDevice):
        world = WebsiteWorld.from_form_factor(device.form_factor)
        return ChipsetYaml(
            aliases=device.chipset_aliases,
            marketing_name=ChipsetYaml.chipset_marketing_name(device.chipset, world),
            world=world,
        )

    @staticmethod
    def chipset_marketing_name(chipset, world: WebsiteWorld | None = None) -> str:
        """Sanitize chip name to match marketing."""
        chip = " ".join([word.capitalize() for word in chipset.split("-")])
        chip = chip.replace(
            "Qualcomm Snapdragon", "Snapdragon®"
        )  # Marketing name for Qualcomm Snapdragon is Snapdragon®
        chip = chip.replace(
            "Qualcomm", "Qualcomm®"
        )  # All other Qualcomm brand names should include a registered trademark

        chip = chip.replace("Proxy", "(Proxy)")

        # 8cxgen2 -> 8cx Gen 2
        # 8gen2 -> 8 Gen 2
        chip = re.sub(r"(\w+)gen(\d+)", r"\g<1> Gen \g<2>", chip)

        # 8 Core -> 8-Core
        chip = re.sub(r"(\d+) Core", r"\g<1>-Core", chip)

        # qcs6490 -> QCS6490
        # sa8775p -> SA8775P
        chip = re.sub(
            r"(Qcs|Sa)\s*(\w+)",
            lambda m: f"{m.group(1).upper()}{m.group(2).upper()}",
            chip,
        )

        return chip + (f" {world.value}" if world == WebsiteWorld.Mobile else "")


class DevicesAndChipsetsYaml(BaseQAIHMConfig):
    """
    This class stores definitions / attributes of valid:
        * devices
        * chipsets
        * form factors
        * scorecard paths

    That the website reads from AI Hub Models perf.yaml files
    to create model card webpages.
    """

    scorecard_path_to_website_runtime: dict[
        ScorecardProfilePath, InferenceEngine
    ] = Field(default_factory=dict)
    form_factors: dict[ScorecardDevice.FormFactor, FormFactorYaml] = Field(
        default_factory=dict
    )
    devices: dict[str, DeviceDetailsYaml] = Field(default_factory=dict)
    chipsets: dict[str, ChipsetYaml] = Field(default_factory=dict)

    @staticmethod
    def from_all_runtimes_and_devices() -> DevicesAndChipsetsYaml:
        """
        Re-generate a DevicesAndChipsetsYaml configuration from the current
        set of devices / runtimes that are valid in AI Hub Models perf.yaml files.
        """
        out = DevicesAndChipsetsYaml()
        out.form_factors = {
            ff: FormFactorYaml.from_form_factor(ff) for ff in ScorecardDevice.FormFactor
        }

        for profile_path in ScorecardProfilePath.all_paths():
            if profile_path.include_in_perf_yaml:
                out.scorecard_path_to_website_runtime[
                    profile_path
                ] = profile_path.runtime.inference_engine

        # For each hub device...
        for hub_device in hub.get_devices():
            if "(Family)" in hub_device.name:
                # Exclude "Family" devices
                continue
            if hub_device.name in out.devices:
                # Exclude multiple devices with the same name
                # (eg different OS)
                continue

            device = ScorecardDevice.get(hub_device.name, return_unregistered=True)

            if "qualcomm" not in device.chipset:
                # Exclude non-qualcomm devices
                continue

            out.devices[device.reference_device_name] = DeviceDetailsYaml.from_device(
                device
            )
            if device.chipset not in out.chipsets:
                out.chipsets[device.chipset] = ChipsetYaml.from_device(device)

        return out

    @staticmethod
    def load():
        """Load this configuration from its standard YAML location in the AI Hub Models python package."""
        return DevicesAndChipsetsYaml.from_yaml(SCORECARD_DEVICE_YAML_PATH)

    def save(self):
        """Save this configuration to its standard YAML location in the AI Hub Models python package."""
        return self.to_yaml(SCORECARD_DEVICE_YAML_PATH)
