# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from .app import SAMApp as App  # noqa: F401

# Use SamLarge as the default. This really needs to be split into multiple
# directory like llama
from .model import MODEL_ID  # noqa: F401
from .model import SAM as Model  # noqa: F401
