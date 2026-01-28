# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.opus_mt.demo import opus_mt_demo
from qai_hub_models.models.opus_mt_zh_en.model import OpusMTZhEn


def main(is_test: bool = False):
    opus_mt_demo(OpusMTZhEn, "Chinese", "English", is_test)


if __name__ == "__main__":
    main()
