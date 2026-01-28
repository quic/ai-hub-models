# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import melo
import torch
import torch.nn.functional as F
from torch import Tensor


class OptimizedFlow:
    def __init__(self, original_flow: "melo.models.TransformerCouplingBlock") -> None:
        self.channels = original_flow.channels
        self.hidden_channels = original_flow.hidden_channels
        self.n_layers = original_flow.n_layers
        self.n_flows = original_flow.n_flows
        self.gin_channels = original_flow.gin_channels
        self.half_channels = self.channels // 2
        first_coupling = original_flow.flows[0]
        self.n_heads = first_coupling.enc.n_heads
        self.k_channels = self.hidden_channels // self.n_heads
        self.attention_scale = 1.0 / self.k_channels**0.5
        self.flows = []
        for i in range(self.n_flows):
            coupling_layer = original_flow.flows[i * 2]
            flow_params = {
                "pre": coupling_layer.pre,
                "post": coupling_layer.post,
                "encoder_layers": [],
            }
            enc = coupling_layer.enc
            for j in range(enc.n_layers):
                layer = {
                    "attn": {
                        "conv_q": enc.attn_layers[j].conv_q,
                        "conv_k": enc.attn_layers[j].conv_k,
                        "conv_v": enc.attn_layers[j].conv_v,
                        "conv_o": enc.attn_layers[j].conv_o,
                    },
                    "norm1": enc.norm_layers_1[j],
                    "ffn": enc.ffn_layers[j],
                    "norm2": enc.norm_layers_2[j],
                }
                if hasattr(enc, "cond_layer_idx") and j == enc.cond_layer_idx:
                    layer["is_cond_layer"] = True
                    if self.gin_channels != 0:
                        layer["spk_emb_linear"] = enc.spk_emb_linear
                else:
                    layer["is_cond_layer"] = False
                flow_params["encoder_layers"].append(layer)
            self.flows.append(flow_params)

    def optimized_attention(self, x: Tensor, mask: Tensor, attn_params: dict) -> Tensor:
        # return the attention tensor
        batch_size = x.size(0)
        q = attn_params["conv_q"](x)
        k = attn_params["conv_k"](x)
        v = attn_params["conv_v"](x)
        q = q.view(batch_size, self.n_heads, self.k_channels, -1)
        k = k.view(batch_size, self.n_heads, self.k_channels, -1)
        v = v.view(batch_size, self.n_heads, self.k_channels, -1)

        scores = torch.matmul(q.transpose(-2, -1), k) * self.attention_scale

        if mask is not None:
            scores.masked_fill_(mask == 0, -1e4)

        attn_output = torch.matmul(F.softmax(scores, dim=-1), v.transpose(-2, -1))
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, self.hidden_channels, -1)

        return attn_params["conv_o"](attn_output)

    def optimized_encoder_layer(
        self, x: Tensor, x_mask: Tensor, layer: dict, g: Tensor | None = None
    ) -> Tensor:
        if layer["is_cond_layer"] and g is not None and self.gin_channels != 0:
            g_trans = layer["spk_emb_linear"](g.transpose(1, 2)).transpose(1, 2)
            x = (x + g_trans) * x_mask

        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)

        h = self.optimized_attention(x, attn_mask, layer["attn"])

        x = layer["norm1"]((x + h) * x_mask)
        h = layer["ffn"](x, x_mask)
        return layer["norm2"]((x + h) * x_mask)

    def optimized_coupling_transform(
        self,
        x: Tensor,
        x_mask: Tensor,
        flow_params: dict,
        g: Tensor | None = None,
        reverse: bool = False,
    ) -> Tensor:
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = flow_params["pre"](x0) * x_mask

        for layer in flow_params["encoder_layers"]:
            h = self.optimized_encoder_layer(h, x_mask, layer, g)

        stats = flow_params["post"](h) * x_mask

        x1 = x1 + stats if not reverse else x1 - stats

        return torch.cat([x0, x1], dim=1)

    def forward(
        self, x: Tensor, x_mask: Tensor, g: Tensor | None = None, reverse: bool = False
    ) -> Tensor:
        """
        Parameters
        ----------
        x
            output of encoder, shape of (1, ENCODER_HIDDEN_DIM, MAX_SEQ_LEN)
        x_mask
            mask of x, shape of (1, 1, UPSAMPLED_MAX_SEQ_LEN)
        g
            embedding of speaker ID, shape of (1, SPEAKER_EMBED_DIM, 1)
        reverse
            whether to reverse the flow

        Returns
        -------
        x
            shape of (1, ENCODER_HIDDEN_DIM, DECODER_Z_TIME_DIM)
        """
        if not reverse:
            for flow in self.flows:
                x = self.optimized_coupling_transform(x, x_mask, flow, g, reverse)
                x = x.flip(1)
        else:
            for flow in reversed(self.flows):
                x = x.flip(1)
                x = self.optimized_coupling_transform(x, x_mask, flow, g, reverse)
        return x
