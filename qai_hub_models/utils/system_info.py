# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import psutil

from qai_hub_models.utils.printing import print_with_box


def has_recommended_memory(required_memory_in_gb: float) -> None:
    """
    Prints out warning if system has less memory(RAM+swap-space) than recommended.
    """
    total_ram = psutil.virtual_memory().total
    total_swap = psutil.swap_memory().total

    # Get total memory in GB
    total_ram_in_gb = total_ram / 1024**3
    total_swap_in_gb = total_swap / 1024**3

    total_memory_in_gb = int(total_ram_in_gb + total_swap_in_gb)

    if required_memory_in_gb > total_memory_in_gb:
        recommended_swap = int(required_memory_in_gb - total_ram_in_gb) + 1
        warning_msgs = [
            f"Recommended minimum memory of {required_memory_in_gb} GB memory (RAM + swap-space), found {total_memory_in_gb} GB.",
            "You might see process killed error due to OOM during export/demo.",
            "",
            "Please increase your swap-space temporarily as a work-around. It might slow down export but allow you to export successfully.",
            "You can refer to https://askubuntu.com/questions/178712/how-to-increase-swap-space for instructions",
            "or run following commands: ",
            "",
            "sudo swapoff -a",
            "# bs=<amount of data that can be read/write>",
            "# count=number of <bs> to allocate for swapfile",
            "# Total size = <bs> * count",
            "#            = 1 MB * 40k = ~40GB",
            f"sudo dd if=/dev/zero of=/local/mnt/swapfile bs=1M count={recommended_swap}k",
            "" "# Set the correct permissions",
            "sudo chmod 0600 /local/mnt/swapfile",
            "",
            "sudo mkswap /local/mnt/swapfile  # Set up a Linux swap area",
            "sudo swapon /local/mnt/swapfile  # Turn the swap on",
            "",
            "You can update `count` to increase swap space that works for machine.",
            "NOTE: the above commands will not persist through a reboot.",
        ]
        print_with_box(warning_msgs)
