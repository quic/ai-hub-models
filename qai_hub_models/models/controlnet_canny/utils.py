# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from io import BytesIO
from typing import Optional

import requests
import torch
from PIL import Image, UnidentifiedImageError

from qai_hub_models.models._shared.stable_diffusion.utils import make_canny


def download_img(url: str) -> Optional[Image.Image]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/114.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.google.com",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content))
        return image

    except UnidentifiedImageError as e:
        print(f"Cannot identify image from {url}: {e}")
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
    return None


def fetch_and_prepare(
    url: str,
    height: int = 512,
    width: int = 512,
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> Optional[torch.Tensor]:
    """
    Fetch image from URL and returns canny map (None if fetch failed).

    Args:
        height (int): Target height of the image.
        width (int): Target width of the image.
        low_threshold (int): Low threshold for Canny edge detection.
        high_threshold (int): High threshold for Canny edge detection.

    Returns:
        Optional[torch.Tensor]: The processed image tensor (NCHW) or None
        if loading fails.
    """
    image = download_img(url)
    if image is None:
        return None
    return make_canny(image, height, width, low_threshold, high_threshold)
