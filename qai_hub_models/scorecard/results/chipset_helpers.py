# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import qai_hub as hub

from qai_hub_models.scorecard.device import ScorecardDevice

WEBSITE_CHIPSET_ORDER = [
    "qualcomm-snapdragon-8-elite",
    "qualcomm-snapdragon-x-elite",
    "qualcomm-snapdragon-8gen3",
    "qualcomm-snapdragon-8gen2",
    "qualcomm-snapdragon-8gen1",
    "qualcomm-snapdragon-888",
    "qualcomm-snapdragon-x-plus-8-core",
    "qualcomm-qcs6490",
    "qualcomm-qcs6490-proxy",
    "qualcomm-qcs8250",
    "qualcomm-qcs8250-proxy",
    "qualcomm-qcs8275"
    "qualcomm-qcs8275-proxy"
    "qualcomm-qcs8265p"
    "qualcomm-qcs8265p-proxy"
    "qualcomm-qcs8550",
    "qualcomm-qcs8550-proxy",
    "qualcomm-sa8775p",
    "qualcomm-sa8775p-proxy",
    "qualcomm-sa8650p",
    "qualcomm-sa8650p-proxy",
    "qualcomm-sa8255p",
    "qualcomm-sa8255p-proxy",
    "qualcomm-qcs8450",
]


def sorted_chipsets(chips: set[str]) -> list[str]:
    """
    Sort the set of chipsets in order they should show up on the website.
    """
    chips = set(chips)

    out = []
    for chipset in WEBSITE_CHIPSET_ORDER:
        if len(chips) == 0:
            break
        if chipset in chips:
            out.append(chipset)
            chips.remove(chipset)
    out.extend(sorted(chips))
    return out


def sorted_devices(devices: set[ScorecardDevice]) -> list[ScorecardDevice]:
    """
    Sort the set of devices in order they should show up on the website.
    Devices will be ordered by their chipset (see WEBSITE_CHIPSET_ORDER)
    """
    device_chipset_map: dict[str, set[ScorecardDevice]] = {}
    for device in devices:
        if device.chipset not in device_chipset_map:
            device_chipset_map[device.chipset] = set()
        device_chipset_map[device.chipset].add(device)

    out: list[ScorecardDevice] = []
    for chipset in WEBSITE_CHIPSET_ORDER:
        if len(device_chipset_map) == 0:
            break
        if chipset in device_chipset_map:
            out.extend(
                sorted(
                    device_chipset_map[chipset], key=lambda d: d.reference_device_name
                )
            )
            device_chipset_map.pop(chipset)

    for chipset in sorted(device_chipset_map.keys()):
        out.extend(
            sorted(device_chipset_map[chipset], key=lambda d: d.reference_device_name)
        )

    return out


# Caching this information is helpful because it requires pulling data from hub.
# Pulling data from hub is slow.
__CHIP_SUPPORTED_DEVICES_CACHE: dict[str, set[ScorecardDevice]] = {}


def get_supported_devices(chips: set[str]) -> list[ScorecardDevice]:
    """Return all the supported devices given the chipset being used."""
    supported_devices: set[ScorecardDevice] = set()

    for chip in chips:
        if chip not in __CHIP_SUPPORTED_DEVICES_CACHE:
            __CHIP_SUPPORTED_DEVICES_CACHE[chip] = {
                ScorecardDevice.get(device.name, return_unregistered=True)
                for device in hub.get_devices(attributes=f"chipset:{chip}")
                if "(Family)" not in device.name
                and "Snapdragon 8 Gen 3 QRD"
                != device.name  # this is not available to all users
            }
        supported_devices.update(__CHIP_SUPPORTED_DEVICES_CACHE[chip])
    return sorted_devices(supported_devices)
