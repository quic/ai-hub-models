# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import qai_hub as hub

SCORECARD_DEVICE_NAME_TO_CHIPSET_NAME = {
    "s23": "qualcomm-snapdragon-8gen2",
    "s24": "qualcomm-snapdragon-8gen3",
    "6490": "qualcomm-qcs6490",
    "8250": "qualcomm-qcs8250",
    "8550": "qualcomm-qcs8550",
}


SCORECARD_DEVICE_NAME_TO_CHIPSET = {
    device: f"chipset:{chipset}"
    for device, chipset in SCORECARD_DEVICE_NAME_TO_CHIPSET_NAME.items()
}


def __get_device(device_name) -> hub.Device:
    # Gets a device with attributes & OS. This only comes from hub.get_devices()
    for device in hub.get_devices():
        if device.name == device_name:
            return device
    raise ValueError(f"No device named {device_name}")


REFERENCE_DEVICE_PER_SUPPORTED_CHIPSETS = {
    "qualcomm-snapdragon-8gen2": __get_device("Samsung Galaxy S23"),
    "qualcomm-snapdragon-8gen3": __get_device("Samsung Galaxy S24"),
    "qualcomm-qcs6490": __get_device("RB3 Gen 2 (Proxy)"),
    "qualcomm-qcs8250": __get_device("RB5 (Proxy)"),
    "qualcomm-qcs8550": __get_device("QCS8550 (Proxy)"),
}
