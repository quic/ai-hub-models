# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import math
from typing import Callable, Optional

import diffusers.models.attention_processor as attention_processor
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU
from diffusers.models.attention import BasicTransformerBlock, FeedForward
from diffusers.models.transformers.transformer_2d import Transformer2DModel


class SHAAttention(nn.Module):
    """
    Split-Head Attention with per-head Conv2D projections and a single output
    Conv2D projection.  This implementation splits the attention heads into
    separate Conv2D projection layers and applies a single output projection
    after concatenating all heads.  Adjusted to handle spatial dimensions (H,
    W) instead of sequence length.
    """

    def __init__(self, orig_attn: attention_processor.Attention):
        """
        Initialize SHAAttention by copying weights from an existing Attention module.

        Args:
            orig_attn (attention_processor.Attention): The original Attention module to be replaced.
        """
        super().__init__()

        for f in ["group_norm", "spatial_norm", "norm_q", "norm_k", "norm_cross"]:
            if getattr(orig_attn, f) is not None:
                raise NotImplementedError(f"{f} is not supported")

        # Copy configuration from the original Attention module
        self.heads = orig_attn.heads
        self.kv_heads = int(orig_attn.inner_kv_dim / orig_attn.inner_dim * self.heads)

        # Infer dim_head from to_q.out_features and heads
        if orig_attn.to_q.out_features % self.heads != 0:
            raise ValueError(
                "to_q.out_features is not divisible by heads. Cannot infer dim_head."
            )
        self.dim_head = orig_attn.to_q.out_features // self.heads
        self.scale = 1 / math.sqrt(self.dim_head)
        self.rescale_output_factor_inv = 1 / orig_attn.rescale_output_factor

        self.residual_connection = orig_attn.residual_connection

        # Verify to_k and to_v dimensions
        expected_kv_out = self.kv_heads * self.dim_head
        if orig_attn.to_k.out_features != expected_kv_out:
            raise ValueError(
                f"to_k.out_features ({orig_attn.to_k.out_features}) does not match expected {expected_kv_out}."
            )
        if orig_attn.to_v.out_features != expected_kv_out:
            raise ValueError(
                f"to_v.out_features ({orig_attn.to_v.out_features}) does not match expected {expected_kv_out}."
            )

        # Initialize separate Conv2D projection layers for each head
        self.q_proj_sha = nn.ModuleList(
            [
                nn.Conv2d(
                    orig_attn.to_q.in_features,
                    self.dim_head,
                    kernel_size=1,
                    bias=(orig_attn.to_q.bias is not None),
                )
                for _ in range(self.heads)
            ]
        )
        self.k_proj_sha = nn.ModuleList(
            [
                nn.Conv2d(
                    orig_attn.to_k.in_features,
                    self.dim_head,
                    kernel_size=1,
                    bias=(orig_attn.to_k.bias is not None),
                )
                for _ in range(self.kv_heads)
            ]
        )
        self.v_proj_sha = nn.ModuleList(
            [
                nn.Conv2d(
                    orig_attn.to_v.in_features,
                    self.dim_head,
                    kernel_size=1,
                    bias=(orig_attn.to_v.bias is not None),
                )
                for _ in range(self.kv_heads)
            ]
        )

        self.to_out = orig_attn.to_out

        # Copy weights from the original shared Linear projections to the separate Conv2D projections
        for i in range(self.heads):
            # Query Projection
            q_weight = orig_attn.to_q.weight.data[
                i * self.dim_head : (i + 1) * self.dim_head, :
            ].clone()
            q_weight = q_weight.unsqueeze(-1).unsqueeze(
                -1
            )  # Shape: (dim_head, in_features, 1, 1)
            self.q_proj_sha[i].weight.data.copy_(q_weight)
            if orig_attn.to_q.bias is not None:
                self.q_proj_sha[i].bias.data.copy_(
                    orig_attn.to_q.bias.data[
                        i * self.dim_head : (i + 1) * self.dim_head
                    ].clone()
                )

        for i in range(self.kv_heads):
            # Key Projection
            k_weight = orig_attn.to_k.weight.data[
                i * self.dim_head : (i + 1) * self.dim_head, :
            ].clone()
            k_weight = k_weight.unsqueeze(-1).unsqueeze(
                -1
            )  # Shape: (dim_head, in_features, 1, 1)
            self.k_proj_sha[i].weight.data.copy_(k_weight)
            if orig_attn.to_k.bias is not None:
                self.k_proj_sha[i].bias.data.copy_(
                    orig_attn.to_k.bias.data[
                        i * self.dim_head : (i + 1) * self.dim_head
                    ].clone()
                )

            # Value Projection
            v_weight = orig_attn.to_v.weight.data[
                i * self.dim_head : (i + 1) * self.dim_head, :
            ].clone()
            v_weight = v_weight.unsqueeze(-1).unsqueeze(
                -1
            )  # Shape: (dim_head, in_features, 1, 1)
            self.v_proj_sha[i].weight.data.copy_(v_weight)
            if orig_attn.to_v.bias is not None:
                self.v_proj_sha[i].bias.data.copy_(
                    orig_attn.to_v.bias.data[
                        i * self.dim_head : (i + 1) * self.dim_head
                    ].clone()
                )

        del orig_attn.to_q
        del orig_attn.to_k
        del orig_attn.to_v

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        """
        Forward pass for Split-Head Cross Attention.
        Processes each head separately using head-specific Conv2D projection layers.

        Args:
            hidden_states (torch.Tensor): The hidden states (batch_size, hidden_size, H, W).
            attention_mask (Optional[torch.Tensor]): The attention mask.
            encoder_hidden_states (Optional[torch.Tensor]): The encoder hidden states for cross-attention.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple containing the attention output, attention weights, and past key-value.
        """
        bsz, hidden_size, H, W = hidden_states.size()
        residual = hidden_states

        if encoder_hidden_states is not None:
            # (N, seq_len, inner_dim) to (N, inner_dim, 1, seq_len)
            encoder_hidden_states = encoder_hidden_states.permute(0, 2, 1).unsqueeze(2)
            # encoder_hidden_states: (N, inner_dim, 1, seq_len)
        else:
            encoder_hidden_states = hidden_states

        query_states = [q_proj(hidden_states) for q_proj in self.q_proj_sha]
        key_states = [k_proj(encoder_hidden_states) for k_proj in self.k_proj_sha]
        value_states = [v_proj(encoder_hidden_states) for v_proj in self.v_proj_sha]
        # query_states, key_states, value_states: List of (bsz, dim_head, H, W)

        # Handle past_key_value for caching
        past_key_value = kwargs.get("past_key_value", None)
        if past_key_value is not None:
            raise NotImplementedError("SHAAttention does not support kv cache yet")

        # Prepare for attention computation
        attn_outputs = []
        for head_idx, (q, k, v) in enumerate(
            zip(query_states, key_states, value_states)
        ):
            q_flat = q.permute(0, 2, 3, 1)  # (bsz, H, W, dim_head)
            k_flat = k.view(
                bsz, 1, self.dim_head, -1
            )  # (bsz, 1, dim_head, H_enc*W_enc)
            v_flat = v.view(
                bsz, 1, self.dim_head, -1
            )  # (bsz, 1, dim_head, H_enc*W_enc)

            attn_scores = torch.matmul(q_flat, k_flat) * self.scale
            # attn_scores: (bsz, H, W, H_enc*W_enc)

            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask

            attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
            # attn_probs: (bsz, H, W, H_enc*W_enc)

            # Compute attention output
            v_perm = v_flat.permute(0, 1, 3, 2)  # (bsz, 1, H_enc*W_enc, dim_head)
            attn_output = torch.matmul(attn_probs, v_perm)
            # attn_output: (bsz, H, W, dim_head)

            attn_outputs.append(attn_output)

        # Concatenate all heads' outputs along the channel dimension
        attn_output = torch.cat(attn_outputs, dim=-1)  # (bsz, H, W, heads * dim_head)

        attn_output = self.to_out[0](attn_output)  # (bsz, H, W, out_features)
        attn_output = self.to_out[1](attn_output)  # (bsz, H, W, out_features)
        attn_output = attn_output.permute(0, 3, 1, 2)  # (bsz,out_features, H, W)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        if kwargs.get("output_attentions", False):
            raise NotImplementedError("output_attentions=True is not supported")

        if self.rescale_output_factor_inv != 1:
            hidden_states *= self.rescale_output_factor_inv

        return attn_output


class PermuteLayerNorm(nn.Module):
    def __init__(self, original_norm):
        super().__init__()
        self.original_norm = original_norm

    def forward(self, *args, **kwargs):
        # Assuming the first argument is the tensor to be normalized
        # Permute the tensor dimensions from (N, C, H, W) to (N, H, W, C)
        permuted_args = list(args)
        if len(permuted_args) > 0 and isinstance(permuted_args[0], torch.Tensor):
            permuted_args[0] = permuted_args[0].permute(0, 2, 3, 1)

        # Apply the original normalization
        norm_output = self.original_norm(*permuted_args, **kwargs)

        # If the output is a tuple (as in some custom norms), permute relevant tensors
        if isinstance(norm_output, tuple):
            # Permute the first tensor in the output tuple
            norm_output = (norm_output[0].permute(0, 3, 1, 2),) + norm_output[1:]
        elif isinstance(norm_output, torch.Tensor):
            norm_output = norm_output.permute(0, 3, 1, 2)

        return norm_output


def traverse_and_replace(
    model: nn.Module,
    target_type: type[torch.nn.Module],
    replacement_fn: Callable[[torch.nn.Module], torch.nn.Module],
):
    """
    Recursively traverses the model to find and replace modules of a specified type.

    Args:
        model (nn.Module): The model to traverse.
        target_type (type): The type of modules to replace (e.g., Attention, GELU).
        replacement_fn (callable): A function that takes a module instance and returns the replacement module.
    """
    for name, module in model.named_children():
        if isinstance(module, target_type):
            setattr(model, name, replacement_fn(module))

        elif isinstance(module, nn.ModuleList):
            for idx in range(len(module)):
                child = module[idx]
                if isinstance(child, target_type):
                    module[idx] = replacement_fn(child)
                else:
                    # Recursively apply to child modules
                    traverse_and_replace(child, target_type, replacement_fn)
        else:
            traverse_and_replace(module, target_type, replacement_fn)


def replace_attention_modules(model: nn.Module):
    """
    Recursively traverses the model to find and replace all instances of Attention with SHAAttention,
    including those nested within ModuleList containers.

    Args:
        model (nn.Module): The model in which to replace Attention modules.
    """
    traverse_and_replace(
        model, attention_processor.Attention, lambda orig_attn: SHAAttention(orig_attn)
    )


def replace_gelu_and_approx_gelu_with_conv2d(activation_module: nn.Module) -> nn.Module:
    """
    Replaces the projection layer in GELU and ApproximateGELU activation modules from Linear to Conv2D.

    Args:
        activation_module (nn.Module): The activation module to replace.

    Returns:
        nn.Module: The activation module with Conv2D projection.
    """
    assert isinstance(activation_module, GELU) or isinstance(
        activation_module, ApproximateGELU
    )
    dim_in = activation_module.proj.in_features
    dim_out = activation_module.proj.out_features
    bias = activation_module.proj.bias is not None

    # Define Conv2d projection
    conv = nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=1, bias=bias)

    # Copy weights from Linear to Conv2d
    with torch.no_grad():
        conv.weight.copy_(activation_module.proj.weight.view(dim_out, dim_in, 1, 1))
        if bias:
            conv.bias.copy_(activation_module.proj.bias)

    # Replace the Linear layer with Conv2d
    activation_module.proj = conv
    return activation_module


class QcGEGLU(nn.Module):
    r"""
    A reimplemented version of the GEGLU activation function using two Conv2D layers.
    This class replaces the original GEGLU's Linear projections with Conv2D projections
    and eliminates the need for the chunk operation by directly computing the gate.

    Parameters:
        original_geglu (GEGLU): The original GEGLU module to be replaced.
    """

    def __init__(self, original_geglu: GEGLU):
        super().__init__()
        # Extract dimensions from the original GEGLU
        dim_in = original_geglu.proj.in_features
        dim_out = (
            original_geglu.proj.out_features // 2
        )  # GEGLU splits output into two parts
        bias = original_geglu.proj.bias is not None

        # Define separate Conv2D layers for hidden projection and gate projection
        self.hidden_proj = nn.Conv2d(
            in_channels=dim_in, out_channels=dim_out, kernel_size=1, bias=bias
        )
        self.gate_proj = nn.Conv2d(
            in_channels=dim_in, out_channels=dim_out, kernel_size=1, bias=bias
        )

        # Initialize weights and biases from the original GEGLU's Linear layer
        with torch.no_grad():
            # Original Linear weights shape: [dim_out*2, dim_in]
            linear_weight = (
                original_geglu.proj.weight.data
            )  # Shape: [dim_out*2, dim_in]
            linear_bias = (
                original_geglu.proj.bias.data if bias else None
            )  # Shape: [dim_out*2]

            # Assign weights to hidden_proj and gate_proj Conv2D layers
            self.hidden_proj.weight.copy_(
                linear_weight[:dim_out, :].view(dim_out, dim_in, 1, 1)
            )
            if bias:
                self.hidden_proj.bias.copy_(linear_bias[:dim_out])  # type: ignore

            self.gate_proj.weight.copy_(
                linear_weight[dim_out:, :].view(dim_out, dim_in, 1, 1)
            )
            if bias:
                self.gate_proj.bias.copy_(linear_bias[dim_out:])  # type: ignore

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        return F.gelu(gate)

    def forward(self, hidden_states, *args, **kwargs):
        # Project hidden states and compute gate
        hidden_proj = self.hidden_proj(hidden_states)  # (N, dim_out, H, W)
        gate = self.gate_proj(hidden_states)  # (N, dim_out, H, W)

        # Apply GELU activation to the gate
        gate = self.gelu(gate)

        # Apply gating mechanism
        return hidden_proj * gate


def replace_geglu_with_conv2d(activation_module: nn.Module) -> nn.Module:
    """
    Replaces the original GEGLU activation module with the QcGEGLU module,
    which uses two Conv2D layers and eliminates the chunk operation.

    Args:
        activation_module (nn.Module): The GEGLU activation module to replace.

    Returns:
        nn.Module: The QcGEGLU activation module with two Conv2D projections.
    """
    if isinstance(activation_module, GEGLU):
        # Instantiate QcGEGLU with the original GEGLU module
        qc_geglu = QcGEGLU(activation_module)
        return qc_geglu
    else:
        raise TypeError(
            f"Unsupported activation module type for GEGLU replacement: {type(activation_module)}"
        )


def replace_activations_with_conv2d(model: nn.Module):
    """
    Recursively traverses the model to find and replace GELU, GEGLU, and ApproximateGELU activation projections
    from Linear layers to Conv2D layers, ensuring compatibility with NCHW input shapes.
    Also handles activations nested within ModuleList containers.

    Args:
        model (nn.Module): The model in which to perform the replacement.
    """
    # Replace GELU and ApproximateGELU
    traverse_and_replace(model, GELU, replace_gelu_and_approx_gelu_with_conv2d)
    traverse_and_replace(
        model, ApproximateGELU, replace_gelu_and_approx_gelu_with_conv2d
    )

    # Replace GEGLU
    traverse_and_replace(model, GEGLU, replace_geglu_with_conv2d)


def replace_feedforward_with_conv2d(feedforward_module: nn.Module) -> nn.Module:
    """
    Replaces the nn.Linear layer in the FeedForward module with a Conv2D layer
    to handle hidden_states of shape (N, C, H, W).

    Args:
        feedforward_module (nn.Module): The FeedForward module to replace.

    Returns:
        nn.Module: The FeedForward module with Conv2D layers instead of Linear layers.
    """
    if isinstance(feedforward_module, FeedForward):
        # Create a new ModuleList to hold the modified layers
        new_net = nn.ModuleList()
        for module in feedforward_module.net:
            if isinstance(module, nn.Linear):
                # Define Conv2d projection
                conv = nn.Conv2d(
                    in_channels=module.in_features,
                    out_channels=module.out_features,
                    kernel_size=1,
                    bias=module.bias is not None,
                )
                # Copy weights from Linear to Conv2d
                with torch.no_grad():
                    conv.weight.copy_(
                        module.weight.data.view(
                            module.out_features, module.in_features, 1, 1
                        )
                    )
                    if module.bias is not None:
                        conv.bias.copy_(module.bias.data)
                # Append the Conv2d layer instead of Linear
                new_net.append(conv)
            else:
                # Append other modules (e.g., activation functions, Dropout) unchanged
                new_net.append(module)
        # Replace the original ModuleList with the new one containing Conv2d layers
        feedforward_module.net = new_net
        return feedforward_module
    else:
        raise TypeError(
            f"Unsupported module type for FeedForward replacement: {type(feedforward_module)}"
        )


def replace_feedforward_modules(model: nn.Module):
    # Replace FeedForward modules' Linear layers with Conv2D
    traverse_and_replace(model, FeedForward, replace_feedforward_with_conv2d)


def replace_layer_norm_modules(model: nn.Module):
    """
    Recursively traverses the model to find and replace all instances of
    LayerNorm within BasicTransformerBlock with PermuteLayerNorm to be
    compatible with optimized_operate_on_continuous_inputs.

    Args:
        model (nn.Module): The model in which to replace LayerNorm modules.
    """

    def replace_layer_norm(block: BasicTransformerBlock) -> BasicTransformerBlock:
        """
        Replaces norm1, norm2, and norm3 within a BasicTransformerBlock.

        Args:
            block (BasicTransformerBlock): The transformer block to modify.

        Returns:
            BasicTransformerBlock: The modified transformer block.
        """
        block.norm1 = PermuteLayerNorm(block.norm1)
        block.norm2 = PermuteLayerNorm(block.norm2)
        if hasattr(block, "norm3"):
            block.norm3 = PermuteLayerNorm(block.norm3)
        return block

    traverse_and_replace(model, BasicTransformerBlock, replace_layer_norm)


def optimized_operate_on_continuous_inputs(self, hidden_states):
    """
    By using 4D NCHW hidden states, we can skip permutation and reshape
    required in the HF implementation.
    """
    if self.use_linear_projection:
        raise NotImplementedError(
            "This optimized method only supports use_linear_projection=False."
        )

    hidden_states = self.norm(hidden_states)
    inner_dim = hidden_states.shape[1]
    hidden_states = self.proj_in(hidden_states)
    return hidden_states, inner_dim


def optimized_get_output_for_continuous_inputs(
    self, hidden_states, residual, batch_size, height, width, inner_dim
):
    """Similar to optimized_operate_on_continuous_inputs"""
    if self.use_linear_projection:
        raise NotImplementedError(
            "This optimized method only supports use_linear_projection=False."
        )

    hidden_states = self.proj_out(hidden_states)
    return hidden_states + residual


_patch_applied = False


def _monkeypatch_hf_unet():
    # Only need to run once
    global _patch_applied
    if not _patch_applied:
        Transformer2DModel._operate_on_continuous_inputs = (
            optimized_operate_on_continuous_inputs
        )
        Transformer2DModel._get_output_for_continuous_inputs = (
            optimized_get_output_for_continuous_inputs
        )
        _patch_applied = True


def monkey_patch_model(model: UNet2DConditionModel):
    """
    1. Apply monkey patches
    2. Apply module replacements for targeted modules whenever monkeypatch
    code is too long

    Note:

    - This monkeypatch is verified against diffusers==0.31.0 on stable
    diffusion 1.5
    """
    print("Monkeypatching Unet (replacing MHA with SHA attention etc)")
    _monkeypatch_hf_unet()
    replace_attention_modules(model)
    replace_activations_with_conv2d(model)
    replace_layer_norm_modules(model)
    replace_feedforward_modules(model)
