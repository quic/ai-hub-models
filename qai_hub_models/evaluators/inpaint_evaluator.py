# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.autograd import Variable
from torch.nn.functional import adaptive_avg_pool2d
from torchvision import models

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator


class InpaintEvaluator(BaseEvaluator):
    """Evaluator for image comparison metrics (MAE, PSNR, SSIM, FID)."""

    def __init__(self, metrics=("mae", "psnr", "ssim", "fid")):
        self.metrics = metrics
        self.reset()

    def reset(self):
        self.results: dict = {m: [] for m in self.metrics}
        self.real_images = []  # Store images for FID
        self.fake_images = []

    def postprocess(self, image):
        image = (torch.clamp(image, -1.0, 1.0) + 1) / 2.0 * 255.0
        return image.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    def add_batch(self, fake_images: torch.Tensor, real_images: torch.Tensor):
        """
        Compute accuracy for the real and fake images

        Args:
            fake_images(torch.Tensor): model output with shape (B, 3, 512, 512)
            real_images(torch.Tensor): ground truth with shape (B, 3, 512, 512)
        """
        fake_np = self.postprocess(fake_images)
        real_np = self.postprocess(real_images)

        for real, fake in zip(real_np, fake_np):
            self._update_metrics(real, fake)

        if "fid" in self.metrics:
            self.fake_images.append(fake_np)
            self.real_images.append(real_np)

    def _update_metrics(self, real, fake):
        """Calculate and store all metrics for one image pair."""

        if "mae" in self.metrics:
            real_fp, fake_fp = real.astype(np.float32), fake.astype(np.float32)
            mae = np.mean(np.abs(real_fp - fake_fp)) / 255.0
            self.results["mae"].append(mae)

        if "psnr" in self.metrics:
            self.results["psnr"].append(
                peak_signal_noise_ratio(real, fake, data_range=255)
            )

        if "ssim" in self.metrics:
            self.results["ssim"].append(
                structural_similarity(real, fake, multichannel=True, data_range=255)
            )

    def mae(self) -> float:
        """Return mean absolute error (normalized 0-1)."""
        if not self.results["mae"]:
            return 0.0
        return float(np.mean(self.results["mae"]))

    def psnr(self) -> float:
        """Return peak signal-to-noise ratio."""
        if not self.results["psnr"]:
            return 0.0
        return float(np.mean(self.results["psnr"]))

    def ssim(self) -> float:
        """Return structural similarity index."""
        if not self.results["ssim"]:
            return 0.0
        return float(np.mean(self.results["ssim"]))

    def fid(self) -> float:
        """Return Fréchet Inception Distance."""
        if not ("fid" in self.metrics and self.real_images and self.fake_images):
            return 0.0
        real_images = np.concatenate(self.real_images, axis=0)
        fake_images = np.concatenate(self.fake_images, axis=0)
        return float(self._compute_fid(real_images, fake_images))

    def get_accuracy_score(self) -> float:
        return self.ssim()

    def formatted_accuracy(self) -> str:
        """Return formatted string with all available metrics."""
        parts = []
        if "mae" in self.metrics:
            parts.append(f"mae: {self.mae():.4f}")
        if "psnr" in self.metrics:
            parts.append(f"psnr: {self.psnr():.2f}")
        if "ssim" in self.metrics:
            parts.append(f"ssim: {self.ssim():.4f}")
        if "fid" in self.metrics:
            parts.append(f"fid: {self.fid():.2f}")
        return ", ".join(parts)

    def _compute_fid(self, images1, images2, batch_size=64, dims=2048):
        """Calculate FID between two image sets"""
        model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[dims]]).cpu()
        model.eval()

        # Get activations
        act1 = self.get_activations(images1 / 255.0, model, batch_size, dims)
        act2 = self.get_activations(images2 / 255.0, model, batch_size, dims)

        # Calculate mean and covariance
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

        # Calculate FID
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                raise ValueError(f"Imaginary component {np.max(np.abs(covmean.imag))}")
            covmean = covmean.real
        tr_covmean = np.trace(covmean)
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def get_activations(self, images, model, batch_size=64, dims=2048):
        """Get Inception V3 activations."""
        images = images.transpose((0, 3, 1, 2))  # (B, 3, H, W)
        d0 = images.shape[0]
        n_batches = (d0 + batch_size - 1) // batch_size
        pred_arr = np.empty((d0, dims))
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, d0)
            batch = torch.from_numpy(images[start:end]).type(torch.FloatTensor)
            batch = Variable(batch).cpu()
            with torch.no_grad():
                pred = model(batch)[0]
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred_arr[start:end] = pred.cpu().data.numpy().reshape(end - start, -1)
        return pred_arr


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps."""

    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {64: 0, 192: 1, 768: 2, 2048: 3}

    def __init__(
        self,
        output_blocks=[DEFAULT_BLOCK_INDEX],
        resize_input=True,
        normalize_input=True,
        requires_grad=False,
    ):
        super().__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        assert self.last_needed_block <= 3, "Last possible output block index is 3"
        self.blocks = nn.ModuleList()
        inception = models.inception_v3(weights="DEFAULT")
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
        ]
        self.blocks.append(nn.Sequential(*block0))
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2),
            ]
            self.blocks.append(nn.Sequential(*block1))
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            ]
            self.blocks.append(nn.Sequential(*block3))
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        outp = []
        x = inp
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=True)
        if self.normalize_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)
            if idx == self.last_needed_block:
                break
        return outp
