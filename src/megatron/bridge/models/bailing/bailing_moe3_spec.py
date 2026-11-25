# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Megatron-Core module spec used by the Ling 3.0 Bridge."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from functools import partial

import torch
import torch.nn.functional as F
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.jit import jit_fuser
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.transformer.transformer_layer import MoETransformerLayer, TransformerLayerSubmodules


def validate_swiglu_limits(limits: Sequence[float] | None, *, num_layers: int, field: str) -> list[float] | None:
    """Validate HF logical-layer limits; zero denotes an unclipped layer."""
    if limits is None:
        return None
    if len(limits) != num_layers:
        raise ValueError(f"{field} must have {num_layers} logical-layer entries, got {len(limits)}.")
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in limits):
        raise ValueError(f"{field} must contain finite, nonnegative clipping limits.")
    return list(limits)


@jit_fuser
def _clamped_swiglu(inputs: torch.Tensor, limit: float) -> torch.Tensor:
    """Fuse Ling post-SiLU clipping while retaining its activation order."""
    gate, up = inputs.chunk(2, dim=-1)
    # Tensor bounds retain the runtime value when the compiler generalizes limits across layers.
    bound = torch.scalar_tensor(limit, dtype=inputs.dtype, device=inputs.device)
    return F.silu(gate).clamp(max=bound) * up.clamp(min=-bound, max=bound)


class BailingMoe3ClampedSwiGLU(torch.nn.Module):
    """Apply ``clamp(silu(gate), max=L) * clamp(up, -L, L)``.

    This follows inclusionAI's SGLang BailingMLP implementation at revision
    822e3e2dd1bb236f5223d205aa4dcae91c8b3d40. MCore's generic activation
    clamp instead clips the gate *before* SiLU and is not equivalent.
    """

    def __init__(self, *, config: object, limit: float) -> None:
        super().__init__()
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("Ling SwiGLU requires a finite positive clipping limit.")
        self.limit = limit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return _clamped_swiglu(inputs, self.limit)


def _build_clamped_mlp(*, builder, limit: float, config, **kwargs):
    """Change only the activation slot in an existing native MCore builder."""
    if config.use_transformer_engine_op_fuser:
        raise ValueError("Ling post-SiLU clipping requires disabling TE operation fusion.")
    if config.activation_func_clamp_value is not None:
        raise ValueError("Ling post-SiLU clipping cannot be combined with MCore's pre-SiLU clamp.")
    config = copy.copy(config)
    # Keep native SwiGLU fusion on unclipped siblings; clipped branches use the custom JIT activation.
    config.bias_activation_fusion = False
    config.use_te_activation_func = True
    submodules = copy.deepcopy(builder.keywords["submodules"])
    submodules.activation_func = partial(BailingMoe3ClampedSwiGLU, limit=limit)
    return builder(config=config, submodules=submodules, **kwargs)


def _build_clamped_experts(num_local_experts, config, *, builder, limit, **kwargs):
    """Preserve the native ExpertsBuilder's two positional arguments."""
    return _build_clamped_mlp(
        builder=builder, limit=limit, config=config, num_local_experts=num_local_experts, **kwargs
    )


def build_bailing_moe3_moe(
    *,
    builder: partial,
    expert_limits: Sequence[float] | None,
    shared_limits: Sequence[float] | None,
    layer_number: int,
    is_mtp_layer: bool = False,
    **kwargs: object,
) -> torch.nn.Module:
    """Attach independent routed/shared limits using global Hybrid positions.

    Each logical block occupies two positions. MTP positions restart at one
    and have no entries in the published clipping lists, so remain unclipped.
    """
    if is_mtp_layer:
        return builder(layer_number=layer_number, is_mtp_layer=True, **kwargs)
    logical_layer = (layer_number - 1) // 2
    submodules = copy.deepcopy(builder.keywords["submodules"])
    for field, limits in (("experts", expert_limits), ("shared_experts", shared_limits)):
        limit = limits[logical_layer] if limits is not None else 0
        if limit:
            wrapper = _build_clamped_experts if field == "experts" else _build_clamped_mlp
            setattr(submodules, field, partial(wrapper, builder=getattr(submodules, field), limit=limit))
    return builder(submodules=submodules, layer_number=layer_number, is_mtp_layer=False, **kwargs)


class BailingMoe3MLASelfAttention(MLASelfAttention):
    """Native MLA with Flash's KV-only latent normalization.

    MCore's QK-norm resolver also adds a fused input norm to direct Q.
    Ling already normalizes the block input and has no such Q norm. Retain
    the resolver's validation and KV selection, replacing only that builder.
    All attention arithmetic and checkpoint behavior remain native.
    """

    def _resolve_qk_norm_config(self, submodules):
        resolved = super()._resolve_qk_norm_config(submodules)
        if self.config.q_lora_rank is None:
            resolved["linear_q_proj"] = submodules.linear_q_proj
        return resolved


class BailingMoe3MoETransformerLayer(MoETransformerLayer):
    """Bind global Hybrid layer indices before native MoE construction.

    TransformerLayer passes its layer number to MoELayer only *after* building
    experts. Bind it here so the activation builders receive the correct limits.
    Forward, backward, and sharded state handling remain inherited from MCore.
    """

    def __init__(
        self,
        *,
        config: MLATransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        expert_limits: list[float] | None,
        shared_limits: list[float] | None,
        **kwargs: object,
    ) -> None:
        submodules = copy.deepcopy(submodules)
        submodules.mlp = partial(
            build_bailing_moe3_moe,
            builder=submodules.mlp,
            layer_number=layer_number,
            expert_limits=expert_limits,
            shared_limits=shared_limits,
        )
        super().__init__(config=config, submodules=submodules, layer_number=layer_number, **kwargs)


def _missing_mcore_features() -> tuple[str, ...]:
    """Inspect native Ling capabilities without requiring a particular MCore SHA."""
    missing = []
    stack = hybrid_stack_spec.submodules
    fields = MLATransformerConfig.__dataclass_fields__
    if (
        getattr(Symbols, "KDA", None) != "K"
        or getattr(stack, "kda_layer", None) is None
        or not {"kda_safe_gate", "kda_lower_bound"} <= fields.keys()
    ):
        missing.append("native HybridModel KDA")
    mla = stack.mla_layer.submodules.self_attention.submodules
    if (
        getattr(mla, "linear_gate", None) is None
        or not {"attention_output_gate", "gated_attention_proj_granularity"} <= fields.keys()
    ):
        missing.append("MLA output gating")
    return tuple(missing)


def _validate_mcore_support() -> None:
    """Fail before model construction when MCore cannot represent Ling weights."""
    missing = _missing_mcore_features()
    if missing:
        raise RuntimeError(
            f"Ling 3.0 requires Megatron-Core support for {', '.join(missing)}. "
            "Install a compatible Bridge/MCore combination with both native HybridModel KDA "
            "and head-wise MLA output gating. Selecting `bash scripts/switch_mcore.sh dev` "
            "alone does not guarantee compatibility. See docs/models/bailing/ling-3.md "
            "and the Ling 3.0 Tiny Base verification card for limitations and verified revisions."
        )


def bailing_moe3_hybrid_stack_spec(config: object) -> ModuleSpec:
    """Return the Transformer Engine Hybrid spec required by Ling 3.0.

    Both public variants store the MLA KV layernorm vector. Tiny additionally
    stores the low-rank Q layernorm, while Flash uses direct Q without a Q
    norm. Flash's resolver retains the declared native plain Q projection.

    Args:
        config: Provider whose ``q_lora_rank`` selects the low-rank Q spec used
            by Tiny versus the direct-Q spec used by Flash.

    Returns:
        A fresh, serializable module spec for the Ling 3.0 HybridModel.
    """
    _validate_mcore_support()
    spec = copy.deepcopy(hybrid_stack_spec)
    mla_attention = spec.submodules.mla_layer.submodules.self_attention.submodules
    if getattr(config, "q_lora_rank", None) is not None:
        mla_attention.q_layernorm = TENorm
    else:
        spec.submodules.mla_layer.submodules.self_attention.module = BailingMoe3MLASelfAttention
    mla_attention.kv_layernorm = TENorm

    expert_limits = getattr(config, "expert_swiglu_limit_list", None)
    shared_limits = getattr(config, "share_expert_swiglu_limit_list", None)
    if expert_limits is not None or shared_limits is not None:
        moe = spec.submodules.moe_layer
        moe.module = BailingMoe3MoETransformerLayer
        moe.params.update(expert_limits=expert_limits, shared_limits=shared_limits)

    return spec
