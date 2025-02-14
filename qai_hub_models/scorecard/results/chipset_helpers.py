# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import re

import qai_hub as hub


def supported_chipsets(chips: list[str]) -> list[str]:
    """
    Return all the supported chipsets given the chipset it works on.

    The order of chips in the website list mirror the order here. Order
    chips from newest to oldest to highlight newest chips most prominently.
    """
    chipset_set = set(chips)
    chipset_list = []

    if "qualcomm-snapdragon-8-elite" in chipset_set:
        chipset_list.extend(
            [
                "qualcomm-snapdragon-8-elite",
                "qualcomm-snapdragon-8gen3",
                "qualcomm-snapdragon-8gen2",
                "qualcomm-snapdragon-8gen1",
                "qualcomm-snapdragon-888",
            ]
        )
    elif "qualcomm-snapdragon-8gen3" in chipset_set:
        chipset_list.extend(
            [
                "qualcomm-snapdragon-8gen3",
                "qualcomm-snapdragon-8gen2",
                "qualcomm-snapdragon-8gen1",
                "qualcomm-snapdragon-888",
            ]
        )
    elif "qualcomm-snapdragon-8gen2" in chipset_set:
        chipset_list.extend(
            [
                "qualcomm-snapdragon-8gen2",
                "qualcomm-snapdragon-8gen1",
                "qualcomm-snapdragon-888",
            ]
        )

    if "qualcomm-snapdragon-x-elite" in chipset_set:
        chipset_list.extend(["qualcomm-snapdragon-x-elite"])
        chipset_list.extend(["qualcomm-snapdragon-x-plus-8-core"])

    chipset_order = [
        "qualcomm-qcs6490",
        "qualcomm-qcs8250",
        "qualcomm-qcs8550",
        "qualcomm-sa8775p",
        "qualcomm-sa8650p",
        "qualcomm-sa8255p",
        "qualcomm-qcs8450",
    ]
    for chipset in chipset_order:
        if chipset in chipset_set:
            chipset_list.append(chipset)

    # Add any remaining chipsets not covered
    for chipset in sorted(chipset_set):
        if chipset not in chipset_list:
            chipset_list.append(chipset)
    return chipset_list


def chipset_marketing_name(chipset) -> str:
    """Sanitize chip name to match marketing."""
    chip = " ".join([word.capitalize() for word in chipset.split("-")])
    chip = chip.replace("Qualcomm ", "")
    chip = chip.replace(
        "Snapdragon", "Snapdragon®"
    )  # Marketing name for Qualcomm Snapdragon is Snapdragon®

    # 8cxgen2 -> 8cx Gen 2
    # 8gen2 -> 8 Gen 2
    chip = re.sub(r"(\w+)gen(\d+)", r"\g<1> Gen \g<2>", chip)

    # 8 Core -> 8-Core
    chip = re.sub(r"(\d+) Core", r"\g<1>-Core", chip)

    # qcs6490 -> QCS6490
    # sa8775p -> SA8775P
    chip = re.sub(
        r"(Qcs|Sa)\s*(\w+)", lambda m: f"{m.group(1).upper()}{m.group(2).upper()}", chip
    )

    return chip


def supported_chipsets_santized(chips) -> list[str]:
    """Santize the chip name passed via hub."""
    chips = [chip for chip in chips if chip != ""]
    return [chipset_marketing_name(chip) for chip in supported_chipsets(chips)]


# Caching this information is helpful because it requires pulling data from hub.
# Pulling data from hub is slow.
__CHIP_SUPPORTED_DEVICES_CACHE: dict[str, list[str]] = {}


def get_supported_devices(chips) -> list[str]:
    """Return all the supported devices given the chipset being used."""
    supported_devices = []

    for chip in supported_chipsets(chips):
        supported_devices_for_chip = __CHIP_SUPPORTED_DEVICES_CACHE.get(chip, list())
        if not supported_devices_for_chip:
            supported_devices_for_chip = [
                device.name
                for device in hub.get_devices(attributes=f"chipset:{chip}")
                if "(Family)" not in device.name
                and "Snapdragon 8 Gen 3 QRD"
                != device.name  # this is not available to all users
            ]
            supported_devices_for_chip = sorted(set(supported_devices_for_chip))
            __CHIP_SUPPORTED_DEVICES_CACHE[chip] = supported_devices_for_chip
        supported_devices.extend(supported_devices_for_chip)
    return supported_devices
