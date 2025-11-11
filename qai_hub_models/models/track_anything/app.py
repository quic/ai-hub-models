# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from PIL import Image

from qai_hub_models.models.track_anything.model import (
    MemoryManager,
    TrackAnythingEncodeValue,
)
from qai_hub_models.utils.draw import create_color_map
from qai_hub_models.utils.image_processing import resize_pad, undo_resize_pad


class TrackAnythingApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with Track-Anything Model.

    The app uses 4 models:
        * TrackAnythingEncodeKeyWithShrinkage
            Given image, gives the encoded key with shrinkage, selection, f16 as output
        * TrackAnythingEncodeValue
            Given image, mask, f16 feature, hidden state, gives the probabilities, encoded value, hidden as output
        * TrackAnythingEncodeKeyWithoutShrinkage
            Given image, gives the encoded key, selection, multiscale features as output
        * TrackAnythingSegment
            Given multiscale features, memory_readout, hidden_state, gives the probabilities, hidden as output
    """

    def __init__(
        self,
        EncodeKeyWithShrinkage: Callable[
            [torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        EncodeValue: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        EncodeKeyWithoutShrinkage: Callable[
            [torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        Segment: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ],
        config: dict,
    ):
        """
        Parameters
        ----------
            EncodeKeyWithShrinkage:
                TrackAnything Key encoder with shrinkage. Must match input and output of qai_hub_models.models.track_anything.model.TrackAnythingEncodeKeyWithShrinkage
            EncodeValue:
                TrackAnything Value encoder. Must match input and output of qai_hub_models.models.track_anything.model.TrackAnythingEncodeValue
            EncodeKeyWithoutShrinkage:
                TrackAnything Key encoder without shrinkage. Must match input and output of qai_hub_models.models.track_anything.model.TrackAnythingEncodeKeyWithoutShrinkage
            Segment:
                TrackAnything Segment. Must match input and output of qai_hub_models.models.track_anything.model.TrackAnythingSegment
            config: dict
                TrackAnything model's config
        """
        self.EncodeKeyWithShrinkage = EncodeKeyWithShrinkage
        self.EncodeValue = EncodeValue
        self.EncodeKeyWithoutShrinkage = EncodeKeyWithoutShrinkage
        self.Segment = Segment
        config["top_k"] = None
        self.memory = MemoryManager(config=config)

    def predict(self, *args, **kwargs):
        return self.track(*args, **kwargs)

    def track(
        self,
        frames: list[np.ndarray],
        mask: np.ndarray,
        raw_output: bool = False,
    ) -> list[np.ndarray]:
        """
        Track the masked object in all frames.

        Parameters
        ----------
            frames: list[numpy array] (N H W C x uint8)
                channel layout RGB

            mask: numpy array of shape [1, H, W]
                mask for object to track

        Returns
        -------
            if raw_output is True,
                out_mask: list[numpy array] (N H W C x uint8)
                    masks of object for all frames

            otherwise,
                painted_image: list[numpy array] (N H W C x uint8)
                    masks applied on all frames
        """
        painted_frames = []
        out_masks = []
        frame_list = []

        # preprocess
        h, w, _ = frames[0].shape[-3:]
        model_h, model_w = TrackAnythingEncodeValue.get_input_spec()["image"][0][-2:]
        for frame_numpy in frames:
            frame = torch.from_numpy(frame_numpy).permute(2, 0, 1).float() / 255.0
            frame, scale, pad = resize_pad(frame.unsqueeze(0), (model_h, model_w))
            frame_list.append(frame)
        frame_tensor = torch.concat(frame_list, dim=0)

        mask_tensor = torch.Tensor(mask).unsqueeze(0).unsqueeze(0)
        mask_tensor = resize_pad(mask_tensor, (model_h, model_w))[0].squeeze(0)

        # Keep first frame as annotation and track the object in remaining frames
        # keep NUM_FRAMES as batch_size
        # skip the last batch of frames if it less than NUM_FRAMES
        NUM_FRAMES = 5
        i = 0
        while i <= frame_tensor.shape[0] - NUM_FRAMES:
            # track the object
            batch_probs_list = []
            for j, tensor in enumerate(frame_tensor[i : i + NUM_FRAMES]):
                if j % 5 == 0:
                    # First frame in NUM_FRAMES
                    key, shrinkage, selection, f16 = self.EncodeKeyWithShrinkage(
                        tensor[None]
                    )

                    # create new hidden state
                    self.memory.create_hidden_state(len([1]), key)

                    hidden_state = self.memory.get_hidden()
                    batch_probs, value, hidden = self.EncodeValue(
                        tensor[None], mask_tensor, f16, hidden_state
                    )

                    # save as memory
                    self.memory.add_memory(
                        key,
                        shrinkage,
                        value,
                        [1],
                        selection=selection,
                    )
                    self.memory.set_hidden(hidden)
                else:
                    # Remaining frame in NUM_FRAMES
                    key, selection, f16, f8, f4 = self.EncodeKeyWithoutShrinkage(
                        tensor[None]
                    )

                    # match mempory with key and selection
                    memory_readout = self.memory.match_memory(key, selection).unsqueeze(
                        0
                    )
                    hidden_state = self.memory.get_hidden()
                    batch_probs, hidden = self.Segment(
                        f16, f8, f4, memory_readout, hidden_state
                    )
                    self.memory.set_hidden(hidden)

                batch_probs_list.append(batch_probs.unsqueeze(0))
            batch_probs = torch.concat(batch_probs_list, dim=0)

            # convert prob to mask
            batch_out_mask = torch.argmax(batch_probs, dim=1).to(torch.float32)
            mask_tensor = batch_out_mask[-1].unsqueeze(0)
            batch_out_mask = undo_resize_pad(
                batch_out_mask.unsqueeze(1), (w, h), scale, pad
            ).squeeze(1)

            for ti, frame_np in enumerate(frames[i : i + NUM_FRAMES]):
                out_mask = batch_out_mask[ti].detach().cpu().numpy().astype(np.uint8)
                out_masks.append(out_mask)

                color_map = create_color_map(out_mask.max() + 1)
                painted_image = Image.blend(
                    Image.fromarray(frame_np), Image.fromarray(color_map[out_mask]), 0.5
                )
                painted_image_numpy = np.asarray(painted_image)
                painted_frames.append(painted_image_numpy)
            i += NUM_FRAMES - 1

        if raw_output:
            return out_masks

        return painted_frames
