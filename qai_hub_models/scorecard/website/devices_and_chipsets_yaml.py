# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from dataclasses import dataclass, field
from enum import unique

import qai_hub as hub
from typing_extensions import assert_never

from qai_hub_models.scorecard.device import ScorecardDevice
from qai_hub_models.scorecard.results.chipset_helpers import chipset_marketing_name
from qai_hub_models.utils.base_config import BaseQAIHMConfig, ParseableQAIHMEnum
from qai_hub_models.utils.path_helpers import QAIHM_PACKAGE_ROOT

SCORECARD_DEVICE_YAML_PATH = (
    QAIHM_PACKAGE_ROOT / "scorecard" / "website" / "devices_and_chipsets.yaml"
)


@unique
class WebsiteWorld(ParseableQAIHMEnum):
    Mobile = 0
    Compute = 1
    Automotive = 2
    IoT = 3
    XR = 4

    def __str__(self) -> str:
        if self == WebsiteWorld.IoT:
            return "IoT"
        if self == WebsiteWorld.XR:
            return "XR"
        return self.name.title()

    @staticmethod
    def from_string(string: str) -> "WebsiteWorld":
        if string == "IoT":
            return WebsiteWorld.IoT
        if string == "XR":
            return WebsiteWorld.XR
        return WebsiteWorld[string.title()]

    @staticmethod
    def from_form_factor(form_factor: ScorecardDevice.FormFactor) -> "WebsiteWorld":
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
class WebsiteIcon(ParseableQAIHMEnum):
    Car = 0
    IoT_Chip = 1
    IoT_Drone = 2
    Laptop_Generic = 3
    Laptop_X_Elite = 4
    Phone_S21 = 5
    Phone_S22 = 6
    Phone_S23 = 7
    Phone_S23_Ultra = 8
    Phone_S24 = 9
    Phone_S24_Ultra = 10
    Tablet_Android = 11
    XR_Headset = 12

    @staticmethod
    def from_string(string: str) -> "WebsiteIcon":
        return WebsiteIcon[string]

    def __str__(self):
        return self.name

    @staticmethod
    def from_device(device: ScorecardDevice) -> "WebsiteIcon":
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
            return WebsiteIcon.IoT_Drone
        if device.form_factor == ScorecardDevice.FormFactor.AUTO:
            return WebsiteIcon.Car
        assert_never(device.form_factor)


@dataclass
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
        return str(ff)


@dataclass
class DeviceYaml(BaseQAIHMConfig):
    chipset: str
    form_factor: ScorecardDevice.FormFactor
    icon: WebsiteIcon

    @staticmethod
    def from_device(device: ScorecardDevice):
        return DeviceYaml(
            chipset=device.chipset,
            form_factor=device.form_factor,
            icon=WebsiteIcon.from_device(device),
        )


@dataclass
class ChipsetYaml(BaseQAIHMConfig):
    aliases: list[str]
    marketing_name: str
    world: WebsiteWorld

    @staticmethod
    def from_device(device: ScorecardDevice):
        world = WebsiteWorld.from_form_factor(device.form_factor)
        return ChipsetYaml(
            aliases=device.chipset_aliases,
            marketing_name=chipset_marketing_name(
                device.chipset, world.name if world == WebsiteWorld.Mobile else None
            ),
            world=world,
        )


@dataclass
class WebsiteDevicesAndChipsetsYaml(BaseQAIHMConfig):
    form_factors: dict[ScorecardDevice.FormFactor, FormFactorYaml] = field(
        default_factory=dict
    )
    devices: dict[str, DeviceYaml] = field(default_factory=dict)
    chipsets: dict[str, ChipsetYaml] = field(default_factory=dict)

    @staticmethod
    def from_all_hub_devices() -> "WebsiteDevicesAndChipsetsYaml":
        out = WebsiteDevicesAndChipsetsYaml()
        out.form_factors = {
            ff: FormFactorYaml.from_form_factor(ff) for ff in ScorecardDevice.FormFactor
        }

        # For each hub device...
        for hub_device in hub.get_devices():
            if "(Family)" in hub_device.name:
                # Exclude "Family" devices
                continue
            if hub_device.name in out.devices:
                # Exclude multiple devices with the same name
                # (eg different OS)
                continue

            device = ScorecardDevice(hub_device.name, hub_device.name, register=False)

            if "qualcomm" not in device.chipset:
                # Exclude non-qualcomm devices
                continue

            out.devices[device.name] = DeviceYaml.from_device(device)
            if device.chipset not in out.chipsets:
                out.chipsets[device.chipset] = ChipsetYaml.from_device(device)

        return out
