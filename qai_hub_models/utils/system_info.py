# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


import platform

from qai_hub_models.utils.printing import print_with_box


def get_available_memory_in_gb():
    # On Windows, the swap (paging file) is defined as a range (initial to
    # max). Psutil only queries the initial (minimum), which is not relevant.
    if platform.system() == "Windows":
        import wmi

        c = wmi.WMI()
        total_ram_gb = (
            sum([int(pagefile.Capacity) for pagefile in c.Win32_PhysicalMemory()])
            / 1024**3
        )
        total_swap_gb = (
            sum([pagefile.MaximumSize for pagefile in c.Win32_PageFileSetting()]) / 1024
        )
        return int(total_ram_gb), int(total_swap_gb)
    import psutil

    total_ram = psutil.virtual_memory().total
    total_swap = psutil.swap_memory().total

    # Get total memory in GB
    total_ram_in_gb = total_ram / 1024**3
    total_swap_in_gb = total_swap / 1024**3

    return int(total_ram_in_gb), int(total_swap_in_gb)


def has_recommended_memory(required_memory_in_gb: float) -> None:
    """Prints out warning if system has less memory(RAM+swap-space) than recommended."""
    total_ram_in_gb, total_swap_in_gb = get_available_memory_in_gb()
    total_memory_in_gb = total_ram_in_gb + total_swap_in_gb

    # If the user is within this buffer, the memory is most likely sufficient
    # and we do not print this warning. This is because if we recommand 120 GB,
    # there are many reasons why the user may still end up slightly short of
    # suppressing this warning. For instance, on Windows swap is specified in
    # MB, so if the user puts in 120000, this will actually correspond to 117
    # GB and the user will still be 3 GB short.
    buffer_gb = 5

    if required_memory_in_gb - buffer_gb > total_memory_in_gb:
        recommended_swap = int(required_memory_in_gb - total_ram_in_gb) + 1
        warning_msgs = [
            "⚠️ Warning: Insufficient memory",
            "",
            f"Recommended memory (RAM + swap): {required_memory_in_gb} GB (currently {total_memory_in_gb} GB)",
            "",
            f"Recommended swap space: {recommended_swap} GB (currently {total_swap_in_gb} GB)",
            "",
            "The process could get killed with out-of-memory error during export/demo.",
            "",
            "This can be avoided by increasing your swap space. Please follow these instructions:",
            "",
            "  https://github.com/quic/ai-hub-apps/blob/main/tutorials/llm_on_genie/increase_swap.md",
            "",
        ]
        print_with_box(warning_msgs)
