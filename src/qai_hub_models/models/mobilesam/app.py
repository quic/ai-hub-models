# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from mobile_sam.utils.transforms import ResizeLongestSide

from qai_hub_models.models.templates.sam.app import SAMApp


class MobileSAMApp(SAMApp):
    resize_longest_side = ResizeLongestSide
