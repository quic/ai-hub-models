# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from typing import Any

import torch

try:
    from transformers.cache_utils import DynamicCache
except ImportError:

    class DynamicCache:  # type: ignore[no-redef]
        pass


# This grossly violates underlying type assumptions in DynamicCache, so we
# turn mypy off for this whole file.
class SHADynamicCacheNewValueOnly(DynamicCache):
    """
    Version of DynamicCache that stores the cache as lists for the separate
    heads (so as to avoid concats/splits for SHA) and returning only the
    new values without accumulation.
    """

    def update(
        self,
        key_states: list[torch.Tensor],
        value_states: list[torch.Tensor],
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0 and hasattr(self, "_seen_tokens"):
            # self._seen_tokens += key_states.shape[-2]
            # This line is updated
            self._seen_tokens += key_states[0].shape[-2]

        # Update the cache
        if hasattr(self, "key_cache"):
            assert hasattr(self, "key_cache")
            assert hasattr(self, "value_cache")
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                # Do not concatenate the cache, we only need the latest entry
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states

            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        if len(self.layers) <= layer_idx:
            # We are violating the types of the original DynamicCache by using
            # lists
            self.layers.append([key_states, value_states])
        else:
            # Do not concatenate the cache, we only need the latest entry
            self.layers[layer_idx][0] = key_states
            self.layers[layer_idx][1] = value_states

        # return self.key_cache[layer_idx], self.value_cache[layer_idx]
        return self.layers[layer_idx][0], self.layers[layer_idx][1]

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if layer_idx is None:
            layer_idx = 0
        if hasattr(self, "key_cache"):
            if len(self.key_cache) <= layer_idx:
                return 0
            # [0] added to get shape since the outermost is list
            return self.key_cache[layer_idx][0].shape[-2]
        if len(self.layers) <= layer_idx:
            return 0
        # [0] added to get shape since the outermost is list
        return self.layers[layer_idx][0][0].shape[-2]
