# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import logging
import os
from collections.abc import Callable
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.transformer.identity_op import IdentityOp
from torch import nn

from megatron.bridge.models.bailing import bailing_moe3_spec as ling_spec
from megatron.bridge.models.bailing.bailing_moe3_bridge import (
    _LING3_MTP_PATTERN,
    BailingMoeV3Bridge,
    _flash_hybrid_layer_pattern,
    _tiny_hybrid_layer_pattern,
)
from megatron.bridge.models.bailing.bailing_moe3_mappings import (
    _BailingMoe3KDAConv1dMapping,
    _BailingMoe3KDAInProjMapping,
)
from megatron.bridge.models.bailing.bailing_moe3_provider import BailingMoe3HybridProvider
from megatron.bridge.models.bailing.bailing_moe3_spec import (
    BailingMoe3ClampedSwiGLU,
    _build_clamped_mlp,
    _missing_mcore_features,
    bailing_moe3_hybrid_stack_spec,
    build_bailing_moe3_moe,
    validate_swiglu_limits,
)
from megatron.bridge.models.conversion.model_bridge import ModelConfigNotSupportedError, get_model_bridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.models.transformer_config import MLATransformerConfig


logger = logging.getLogger(__name__)
pytestmark = pytest.mark.unit
requires_ling_mcore = pytest.mark.skipif(
    bool(_missing_mcore_features()),
    reason="Requires native MCore HybridModel KDA and MLA output gating (switch_mcore.sh dev)",
)


@pytest.mark.parametrize(
    "missing",
    [
        "symbol",
        "kda_layer",
        "kda_safe_gate",
        "kda_lower_bound",
        "linear_gate",
        "attention_output_gate",
        "gated_attention_proj_granularity",
        None,
    ],
)
def test_mcore_capability_detection(monkeypatch, missing) -> None:
    fields = dict.fromkeys(
        ("kda_safe_gate", "kda_lower_bound", "attention_output_gate", "gated_attention_proj_granularity")
    )
    if missing in fields:
        del fields[missing]
    mla = SimpleNamespace(linear_gate=None if missing == "linear_gate" else object())
    stack = SimpleNamespace(
        kda_layer=None if missing == "kda_layer" else object(),
        mla_layer=SimpleNamespace(submodules=SimpleNamespace(self_attention=SimpleNamespace(submodules=mla))),
    )
    monkeypatch.setattr(ling_spec, "Symbols", SimpleNamespace() if missing == "symbol" else SimpleNamespace(KDA="K"))
    monkeypatch.setattr(ling_spec, "hybrid_stack_spec", SimpleNamespace(submodules=stack))
    monkeypatch.setattr(ling_spec, "MLATransformerConfig", SimpleNamespace(__dataclass_fields__=fields))
    if missing is None:
        assert _missing_mcore_features() == ()
    elif missing in ("symbol", "kda_layer", "kda_safe_gate", "kda_lower_bound"):
        assert _missing_mcore_features() == ("native HybridModel KDA",)
    else:
        assert _missing_mcore_features() == ("MLA output gating",)


@pytest.mark.parametrize("missing", [("native HybridModel KDA",), ("MLA output gating",)])
def test_provider_rejects_missing_mcore_capability_before_config_mapping(monkeypatch, missing) -> None:
    monkeypatch.setattr(ling_spec, "_missing_mcore_features", lambda: missing)
    with pytest.raises(RuntimeError, match="switch_mcore.sh dev") as error:
        BailingMoeV3Bridge().provider_bridge(SimpleNamespace(config=_tiny_config()))
    assert missing[0] in str(error.value)
    with pytest.raises(RuntimeError, match="switch_mcore.sh dev"):
        bailing_moe3_hybrid_stack_spec(_tiny_config())


_TINY_GOLDEN_PATTERN = "K-KEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+E"
_FLASH_GOLDEN_PATTERN = "K-K-KEKEKE+EKEKEKEKEKE+EKEKEKEKEKE+EKEKEKEKEKE+EKEKEKEKEKE+EKEKEKEKEKE+EKEKEKEKEKE+E"


@requires_ling_mcore
def test_flash_clipping_schedules_survive_config_roundtrip():
    routed = [0] * 35 + [4] * 7
    shared = [0] * 34 + [5] * 6 + [7] * 2
    hf_config = _flash_config(expert_swiglu_limit_list=routed, share_expert_swiglu_limit_list=shared)
    bridge = BailingMoeV3Bridge()
    provider = bridge.provider_bridge(SimpleNamespace(config=hf_config))
    exported = bridge.megatron_to_hf_config(provider)
    assert exported["expert_swiglu_limit_list"] == routed
    assert exported["share_expert_swiglu_limit_list"] == shared
    assert provider.bias_activation_fusion is False
    spec = bailing_moe3_hybrid_stack_spec(provider)
    assert spec.submodules.moe_layer.module is ling_spec.BailingMoe3MoETransformerLayer


def _tiny_config(**overrides):
    values = dict(
        architectures=["BailingMoeV3ForCausalLM"],
        model_type="bailing_hybrid",
        num_hidden_layers=24,
        layer_group_size=4,
        first_k_dense_replace=1,
        hidden_size=1536,
        vocab_size=157184,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=128,
        num_kv_heads_for_linear_attn=0,
        num_experts=128,
        num_experts_per_tok=8,
        moe_intermediate_size=512,
        moe_shared_expert_intermediate_size=512,
        num_shared_experts=1,
        q_lora_rank=256,
        kv_lora_rank=512,
        qk_head_dim=192,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        short_conv_kernel_size=4,
        num_nextn_predict_layers=0,
        mtp_use_kda=False,
        mtp_loss_scaling_factor=0.0,
        gated_attention_proj_granularity_type="head_wise",
        use_kda_lora=False,
        no_kda_lora=True,
        use_qkv_bias=False,
        attention_bias=False,
        use_bias=False,
        mlp_bias=False,
        scoring_func="sigmoid",
        max_position_embeddings=131072,
        rope_theta=6000000.0,
        partial_rotary_factor=0.5,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        intermediate_size=4608,
        num_hidden_layers_mtp=0,
        hidden_act="silu",
        torch_dtype="bfloat16",
        attention_dropout=0.0,
        hidden_dropout=0.0,
        embedding_dropout=0.0,
        output_dropout=0.0,
        tie_word_embeddings=False,
        use_qk_norm=True,
        linear_silu=True,
        group_norm_size=1,
        moe_router_enable_expert_bias=True,
        router_dtype="fp32",
        norm_topk_prob=True,
        scale_router_input=False,
        topk_method="noaux_tc",
        score_function="sigmoid",
        rotary_dim=64,
        rope_interleave=True,
        rope_scaling={
            "rope_theta": 6_000_000.0,
            "partial_rotary_factor": 0.5,
            "rope_type": "default",
        },
        routed_scaling_factor=2.5,
        n_group=8,
        topk_group=4,
        kda_safe_gate=True,
        kda_lower_bound=-5,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _flash_config(**overrides):
    values = dict(
        architectures=["BailingMoeV3ForCausalLM"],
        model_type="bailing_hybrid",
        num_hidden_layers=42,
        layer_group_size=6,
        first_k_dense_replace=2,
        hidden_size=2560,
        vocab_size=157184,
        num_attention_heads=32,
        num_key_value_heads=32,
        num_experts=512,
        num_experts_per_tok=8,
        moe_intermediate_size=768,
        moe_shared_expert_intermediate_size=768,
        num_shared_experts=1,
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_head_dim=192,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        head_dim=128,
        short_conv_kernel_size=4,
        num_nextn_predict_layers=1,
        mtp_use_kda=False,
        mtp_loss_scaling_factor=0.0,
        gated_attention_proj_granularity_type="head_wise",
        use_kda_lora=False,
        no_kda_lora=True,
        use_qkv_bias=False,
        attention_bias=False,
        use_bias=False,
        mlp_bias=False,
        scoring_func="sigmoid",
        max_position_embeddings=262144,
        rope_theta=6000000.0,
        partial_rotary_factor=0.5,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        intermediate_size=6144,
        hidden_act="silu",
        torch_dtype="bfloat16",
        attention_dropout=0.0,
        hidden_dropout=0.0,
        embedding_dropout=0.0,
        output_dropout=0.0,
        tie_word_embeddings=False,
        use_qk_norm=True,
        linear_silu=True,
        group_norm_size=1,
        moe_router_enable_expert_bias=True,
        router_dtype="fp32",
        norm_topk_prob=True,
        scale_router_input=False,
        topk_method="noaux_tc",
        score_function="sigmoid",
        rotary_dim=64,
        rope_interleave=True,
        rope_scaling={
            "rope_theta": 6_000_000.0,
            "partial_rotary_factor": 0.5,
            "rope_type": "default",
        },
        routed_scaling_factor=2.5,
        n_group=8,
        topk_group=4,
        kda_safe_gate=True,
        kda_lower_bound=-5.0,
        num_kv_heads_for_linear_attn=0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_tiny_hybrid_pattern_is_fixed() -> None:
    assert _tiny_hybrid_layer_pattern(_tiny_config()) == _TINY_GOLDEN_PATTERN


def test_flash_hybrid_pattern_has_dense_prefix_and_grouped_mla() -> None:
    pattern = _flash_hybrid_layer_pattern(_flash_config())

    assert pattern == _FLASH_GOLDEN_PATTERN
    assert len(pattern) == 84
    assert pattern[:4] == "K-K-"
    assert all(pattern[2 * layer] == "+" for layer in (5, 11, 17, 23, 29, 35, 41))
    assert all(pattern[2 * layer] == "K" for layer in (0, 1, 2, 3, 4, 6, 7, 8, 9, 10))
    assert pattern.count("-") == 2
    assert pattern.count("E") == 40


def test_bailing_moe3_registration_uses_architecture_and_model_type() -> None:
    config = _tiny_config()
    bridge = get_model_bridge("BailingMoeV3ForCausalLM", hf_config=config)

    assert isinstance(bridge, BailingMoeV3Bridge)
    assert bridge.SOURCE_NAME == "BailingMoeV3ForCausalLM"
    assert bridge.MODEL_TYPE == "bailing_hybrid"
    assert bridge.hf_config is config


def test_bailing_moe3_disables_incompatible_default_gpt_builder_config() -> None:
    bridge = BailingMoeV3Bridge()

    with pytest.raises(ModelConfigNotSupportedError, match="MODEL_CONFIG_CLASS is None"):
        bridge.hf_config_to_model_config(_tiny_config())


@requires_ling_mcore
def test_tiny_provider_maps_supported_config_variations() -> None:
    config = _tiny_config(
        hidden_act="gelu",
        attention_dropout=0.1,
        hidden_dropout=0.2,
        intermediate_size=4096,
        vocab_size=157200,
        q_lora_rank=128,
        kv_lora_rank=256,
        num_experts=64,
        num_experts_per_tok=4,
        routed_scaling_factor=1.5,
        n_group=4,
        topk_group=2,
        kda_safe_gate=False,
        kda_lower_bound=None,
        moe_shared_expert_intermediate_size=384,
        mtp_loss_scaling_factor=0.25,
    )

    provider = BailingMoeV3Bridge().provider_bridge(SimpleNamespace(config=config))

    assert provider.activation_func is torch.nn.functional.gelu
    assert provider.attention_dropout == 0.1
    assert provider.hidden_dropout == 0.2
    assert provider.ffn_hidden_size == 4096
    assert provider.vocab_size == 157200
    assert provider.q_lora_rank == 128
    assert provider.kv_lora_rank == 256
    assert provider.num_moe_experts == 64
    assert provider.moe_router_topk == 4
    assert provider.moe_router_topk_scaling_factor == 1.5
    assert provider.moe_router_num_groups == 4
    assert provider.moe_router_group_topk == 2
    assert provider.moe_router_dtype == "fp32"
    assert provider.moe_router_score_function == "sigmoid"
    assert provider.moe_router_pre_softmax is False
    assert provider.kda_safe_gate is False
    assert provider.kda_lower_bound is None
    assert provider.moe_shared_expert_intermediate_size == 384
    assert provider.mtp_loss_scaling_factor == 0.25


@requires_ling_mcore
def test_tiny_provider_defaults_linear_heads_when_hf_field_is_absent() -> None:
    config = _tiny_config()
    del config.num_kv_heads_for_linear_attn

    provider = BailingMoeV3Bridge().provider_bridge(SimpleNamespace(config=config))

    assert provider.linear_num_key_heads == config.num_attention_heads
    assert provider.linear_num_value_heads == config.num_attention_heads


@pytest.mark.parametrize(
    "override",
    [
        {"num_nextn_predict_layers": 2},
        {"num_nextn_predict_layers": 1, "mtp_use_kda": True},
        {"q_lora_rank": None},
        {"gated_attention_proj_granularity_type": "token_wise"},
        {"no_kda_lora": False},
        {"use_qkv_bias": True},
        {"tie_word_embeddings": True},
        {"use_qk_norm": False},
        {"moe_router_enable_expert_bias": False},
        {"router_dtype": "fp64"},
        {"scoring_func": "softmax"},
        {"score_function": "softmax"},
        {"norm_topk_prob": False},
        {"topk_method": "greedy"},
        {"rope_scaling": {"rope_type": "linear", "factor": 2.0}},
        {"layer_group_size": 3},
    ],
)
def test_tiny_validation_fails_closed(override) -> None:
    with pytest.raises(ValueError):
        BailingMoeV3Bridge._validate_config(_tiny_config(**override))


@pytest.mark.parametrize(
    "override",
    [
        {"q_lora_rank": 256},
        {"num_nextn_predict_layers": 0},
        {"mtp_use_kda": True},
    ],
)
def test_flash_validation_fails_closed(override) -> None:
    with pytest.raises(ValueError):
        BailingMoeV3Bridge._validate_config(_flash_config(**override))


@pytest.mark.parametrize("config_factory", [_tiny_config, _flash_config])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("factor", [0.5, 0.25, None])
def test_rotary_factor_validation_supports_hf_config_formats(config_factory, nested, factor) -> None:
    config = config_factory(partial_rotary_factor=factor)
    if nested:
        del config.partial_rotary_factor
        config.rope_parameters = {"rope_type": "default", "partial_rotary_factor": factor}

    if factor == 0.5:
        BailingMoeV3Bridge._validate_config(config)
    else:
        with pytest.raises(ValueError, match="partial_rotary_factor"):
            BailingMoeV3Bridge._validate_config(config)


@requires_ling_mcore
def test_tiny_provider_contains_serializable_mla_fields() -> None:
    bridge = BailingMoeV3Bridge()
    provider = bridge.provider_bridge(SimpleNamespace(config=_tiny_config()))

    assert isinstance(provider, BailingMoe3HybridProvider)
    assert provider.hybrid_layer_pattern == _TINY_GOLDEN_PATTERN
    assert provider.num_layers == 48
    assert provider.q_lora_rank == 256
    assert provider.kv_lora_rank == 512
    assert provider.qk_head_dim == 128
    assert provider.qk_pos_emb_head_dim == 64
    assert provider.v_head_dim == 128
    assert provider.position_embedding_type == "rope"
    assert provider.rotary_percent == 1.0
    assert provider.rotary_scaling_factor == 1.0
    assert provider.mscale_all_dim == 1.0
    assert provider.gated_attention_proj_granularity == "headwise"
    assert provider.moe_router_pre_softmax is False
    assert provider.moe_router_enable_expert_bias is True
    assert set(vars(provider)) <= set(provider.__dataclass_fields__)


def test_bailing_provider_combines_hybrid_construction_and_mla_config() -> None:
    provider = BailingMoe3HybridProvider(
        hidden_size=256,
        num_attention_heads=4,
        hybrid_layer_pattern="K+E-",
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_head_dim=32,
        qk_pos_emb_head_dim=16,
        v_head_dim=32,
    )

    assert isinstance(provider, HybridModelProvider)
    assert isinstance(provider, MLATransformerConfig)
    assert provider.multi_latent_attention is True
    assert provider.provide.__func__ is HybridModelProvider.provide
    assert {"q_lora_rank", "hybrid_layer_pattern", "linear_conv_kernel_dim"} <= set(provider.__dataclass_fields__)


@requires_ling_mcore
def test_bailing_provider_finalize_runs_hybrid_and_mla_validation() -> None:
    provider = BailingMoe3HybridProvider(
        hidden_size=256,
        num_attention_heads=4,
        hybrid_layer_pattern="+",
        attention_output_gate=True,
        mla_down_proj_fusion=True,
    )

    with pytest.raises(ValueError, match="MLA output gating does not support fused down projections"):
        provider.finalize()

    assert provider.num_layers == 1


@requires_ling_mcore
def test_flash_provider_uses_direct_q_and_one_mtp_layer() -> None:
    bridge = BailingMoeV3Bridge()
    provider = bridge.provider_bridge(SimpleNamespace(config=_flash_config()))

    assert provider.hybrid_layer_pattern == _FLASH_GOLDEN_PATTERN
    assert provider.num_layers == 84
    assert provider.q_lora_rank is None
    assert provider.qk_layernorm is True
    assert provider.linear_num_key_heads == 32
    assert provider.num_moe_experts == 512
    assert provider.ffn_hidden_size == 6144
    assert provider.mtp_num_layers == 1
    assert provider.mtp_hybrid_override_pattern == _LING3_MTP_PATTERN
    assert provider.mtp_loss_scaling_factor == 0.0
    assert provider.position_embedding_type == "rope"
    assert provider.rotary_percent == 1.0
    assert set(vars(provider)) <= set(provider.__dataclass_fields__)


@requires_ling_mcore
def test_flash_spec_uses_mcore_native_direct_q_and_standalone_kv_norm() -> None:
    spec = bailing_moe3_hybrid_stack_spec(_flash_config())
    mla = spec.submodules.mla_layer.submodules.self_attention

    assert mla.module is ling_spec.BailingMoe3MLASelfAttention
    assert mla.submodules.q_layernorm is IdentityOp
    assert mla.submodules.kv_layernorm is TENorm


@requires_ling_mcore
def test_tiny_provider_uses_low_rank_q_and_one_mtp_layer() -> None:
    bridge = BailingMoeV3Bridge()
    provider = bridge.provider_bridge(
        SimpleNamespace(config=_tiny_config(num_nextn_predict_layers=1, mtp_use_kda=False))
    )

    assert provider.mtp_num_layers == 1
    assert provider.mtp_hybrid_override_pattern == _LING3_MTP_PATTERN
    assert provider.mtp_loss_scaling_factor == 0.0


def test_tiny_mapping_registry_covers_low_rank_q_mtp() -> None:
    bridge = BailingMoeV3Bridge()
    bridge.hf_config = _tiny_config(num_nextn_predict_layers=1, mtp_use_kda=False)
    registry = bridge.mapping_registry()

    mtp_prefix = "mtp.layers.0.mtp_model_layer.layers.0.self_attention"
    assert (
        registry.hf_to_megatron_lookup("model.layers.24.attention.q_a_proj.weight").megatron_param
        == f"{mtp_prefix}.linear_q_down_proj.weight"
    )
    assert (
        registry.hf_to_megatron_lookup("model.layers.24.attention.q_a_layernorm.weight").megatron_param
        == f"{mtp_prefix}.q_layernorm.weight"
    )
    assert (
        registry.hf_to_megatron_lookup("model.layers.24.attention.q_b_proj.weight").megatron_param
        == f"{mtp_prefix}.linear_q_up_proj.weight"
    )
    assert (
        registry.hf_to_megatron_lookup("model.layers.24.attention.kv_b_proj.weight").megatron_param
        == f"{mtp_prefix}.linear_kv_up_proj.weight"
    )


@requires_ling_mcore
def test_spec_reuses_mcore_projection_builders_and_overrides_only_ling_norms() -> None:
    spec = bailing_moe3_hybrid_stack_spec(_tiny_config())

    default_kda = hybrid_stack_spec.submodules.kda_layer.submodules.self_attention.submodules
    default_mla = hybrid_stack_spec.submodules.mla_layer.submodules.self_attention.submodules
    kda = spec.submodules.kda_layer.submodules.self_attention.submodules
    mla = spec.submodules.mla_layer.submodules.self_attention.submodules

    assert spec is not hybrid_stack_spec
    assert kda.beta_proj is default_kda.beta_proj
    assert mla.linear_q_down_proj is default_mla.linear_q_down_proj
    assert mla.linear_kv_down_proj is default_mla.linear_kv_down_proj
    assert mla.linear_gate is default_mla.linear_gate
    assert mla.q_layernorm is TENorm
    assert mla.kv_layernorm is TENorm
    assert default_mla.q_layernorm is IdentityOp
    assert default_mla.kv_layernorm is IdentityOp


@pytest.mark.parametrize(
    ("config_factory", "expected_layers", "expected_group_size", "expected_dense_layers", "expected_mtp_layers"),
    [
        (_tiny_config, 24, 4, 1, 0),
        (_flash_config, 42, 6, 2, 1),
    ],
)
@requires_ling_mcore
def test_megatron_to_hf_config_restores_logical_architecture(
    config_factory,
    expected_layers: int,
    expected_group_size: int,
    expected_dense_layers: int,
    expected_mtp_layers: int,
) -> None:
    bridge = BailingMoeV3Bridge()
    provider = bridge.provider_bridge(SimpleNamespace(config=config_factory()))
    provider.finalize()

    exported = bridge.megatron_to_hf_config(provider)

    assert exported["num_hidden_layers"] == expected_layers
    assert exported["layer_group_size"] == expected_group_size
    assert exported["first_k_dense_replace"] == expected_dense_layers
    assert exported["gated_attention_proj_granularity_type"] == "head_wise"
    assert exported["no_kda_lora"] is True
    assert exported["num_kv_heads_for_linear_attn"] == 0
    assert exported["scoring_func"] == "sigmoid"
    assert exported["score_function"] == "sigmoid"
    assert exported["norm_topk_prob"] is True
    assert exported["topk_method"] == "noaux_tc"
    assert exported["router_dtype"] == "fp32"
    assert exported["moe_router_enable_expert_bias"] is True
    assert exported["moe_shared_expert_intermediate_size"] in (512, 768)
    assert exported["mtp_loss_scaling_factor"] == 0.0
    assert exported["partial_rotary_factor"] == 0.5
    assert exported["rotary_dim"] == 64
    assert exported["rope_interleave"] is True
    assert exported["num_nextn_predict_layers"] == expected_mtp_layers


def _export_provider(config_factory: Callable[..., SimpleNamespace]) -> SimpleNamespace:
    """Describe the export contract without constructing a native MCore model."""
    flash = config_factory is _flash_config
    return SimpleNamespace(
        hybrid_layer_pattern=_FLASH_GOLDEN_PATTERN if flash else _TINY_GOLDEN_PATTERN,
        normalization="RMSNorm",
        gated_linear_unit=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        qk_layernorm=True,
        attention_output_gate=True,
        gated_attention_proj_granularity="headwise",
        rotary_percent=1.0,
        rotary_interleaved=False,
        rope_type="rope",
        moe_grouped_gemm=True,
        moe_router_score_function="sigmoid",
        moe_router_dtype="fp32",
        moe_router_load_balancing_type="none",
        moe_router_enable_expert_bias=True,
        mtp_use_repeated_layer=False,
        mtp_num_layers=1 if flash else 0,
        mtp_hybrid_override_pattern="+E" if flash else None,
        q_lora_rank=None if flash else 256,
        kv_channels=128,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=32 if flash else 16,
        linear_num_value_heads=32 if flash else 16,
        num_attention_heads=32 if flash else 16,
        moe_shared_expert_intermediate_size=768 if flash else 512,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
    )


@pytest.mark.parametrize(
    ("config_factory", "layers", "group", "dense", "mtp"),
    [(_tiny_config, 24, 4, 1, 0), (_flash_config, 42, 6, 2, 1)],
)
def test_export_logical_architecture_without_native_mcore(
    config_factory: Callable[..., SimpleNamespace], layers: int, group: int, dense: int, mtp: int
) -> None:
    exported = BailingMoeV3Bridge.megatron_to_hf_config(_export_provider(config_factory))
    assert exported["num_hidden_layers"] == layers
    assert exported["layer_group_size"] == group
    assert exported["first_k_dense_replace"] == dense
    assert exported["num_nextn_predict_layers"] == mtp
    assert exported["qk_head_dim"] == 192
    assert exported["partial_rotary_factor"] == 0.5
    assert exported["norm_topk_prob"] is True


@pytest.mark.parametrize("pattern", [None, "K-", f"{_TINY_GOLDEN_PATTERN}/+E"])
def test_export_rejects_invalid_pattern_without_native_mcore(pattern: str | None) -> None:
    provider = _export_provider(_tiny_config)
    provider.hybrid_layer_pattern = pattern
    with pytest.raises(ValueError, match="hybrid_layer_pattern|Hybrid pattern|MTP pattern suffixes"):
        BailingMoeV3Bridge.megatron_to_hf_config(provider)


@pytest.mark.parametrize("config_factory", [_tiny_config, _flash_config])
def test_export_mtp_suffix_without_native_mcore(config_factory: Callable[..., SimpleNamespace]) -> None:
    provider = _export_provider(config_factory)
    provider.mtp_num_layers = 1
    provider.mtp_hybrid_override_pattern = "+E"
    provider.hybrid_layer_pattern += "/+E"
    exported = BailingMoeV3Bridge.megatron_to_hf_config(provider)
    assert exported["num_nextn_predict_layers"] == 1
    assert exported["num_hidden_layers"] == (24 if config_factory is _tiny_config else 42)


@pytest.mark.parametrize(
    ("config_factory", "invalid_mtp_layers"),
    [
        (_tiny_config, 2),
        (_flash_config, 0),
    ],
)
def test_megatron_to_hf_config_rejects_non_public_mtp_layout(config_factory, invalid_mtp_layers: int) -> None:
    bridge = BailingMoeV3Bridge()
    provider = _export_provider(config_factory)
    provider.mtp_num_layers = invalid_mtp_layers

    with pytest.raises(ValueError, match="export requires mtp_num_layers"):
        bridge.megatron_to_hf_config(provider)


def test_megatron_to_hf_config_does_not_infer_norm_topk_prob_from_pre_softmax() -> None:
    bridge = BailingMoeV3Bridge()
    provider = _export_provider(_tiny_config)
    provider.moe_router_pre_softmax = True

    exported = bridge.megatron_to_hf_config(provider)

    assert exported["norm_topk_prob"] is True


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("moe_router_score_function", "softmax"),
        ("moe_router_dtype", "fp64"),
        ("moe_router_load_balancing_type", "aux_loss"),
        ("moe_router_enable_expert_bias", False),
    ],
)
def test_megatron_to_hf_config_rejects_incompatible_router_semantics(name: str, value: object) -> None:
    bridge = BailingMoeV3Bridge()
    provider = _export_provider(_tiny_config)
    if not hasattr(provider, name):
        raise ValueError(f"Unknown provider override: {name}")
    setattr(provider, name, value)

    with pytest.raises(ValueError, match=f"export requires {name}"):
        bridge.megatron_to_hf_config(provider)


@pytest.mark.parametrize(
    ("config_factory", "name", "value", "match"),
    [
        (_tiny_config, "attention_output_gate", False, "attention_output_gate"),
        (_tiny_config, "gated_attention_proj_granularity", "elementwise", "gated_attention_proj_granularity"),
        (_tiny_config, "position_embedding_type", "none", "position_embedding_type"),
        (_tiny_config, "qk_layernorm", False, "qk_layernorm"),
        (_tiny_config, "add_bias_linear", True, "add_bias_linear"),
        (_tiny_config, "moe_grouped_gemm", False, "moe_grouped_gemm"),
        (_tiny_config, "rotary_percent", 0.5, "rotary_percent"),
        (_tiny_config, "q_lora_rank", None, "low-rank-Q MLA"),
        (_flash_config, "q_lora_rank", 256, "direct-Q MLA"),
        (_tiny_config, "linear_value_head_dim", 64, "head dimensions"),
        (_tiny_config, "linear_num_value_heads", 8, "head counts"),
        (_tiny_config, "moe_shared_expert_intermediate_size", 0, "shared expert"),
    ],
)
def test_megatron_to_hf_config_rejects_incompatible_structure(
    config_factory,
    name: str,
    value: object,
    match: str,
) -> None:
    bridge = BailingMoeV3Bridge()
    provider = _export_provider(config_factory)
    if not hasattr(provider, name):
        raise ValueError(f"Unknown provider override: {name}")
    setattr(provider, name, value)

    with pytest.raises(ValueError, match=match):
        bridge.megatron_to_hf_config(provider)


def test_megatron_to_hf_config_rejects_incompatible_mtp_pattern() -> None:
    bridge = BailingMoeV3Bridge()
    provider = _export_provider(_flash_config)
    provider.hybrid_layer_pattern = f"{_FLASH_GOLDEN_PATTERN}/KE"

    with pytest.raises(ValueError, match="MTP pattern suffixes"):
        bridge.megatron_to_hf_config(provider)


def test_tiny_mapping_registry_covers_logical_to_physical_layout() -> None:
    bridge = BailingMoeV3Bridge()
    bridge.hf_config = _tiny_config()
    registry = bridge.mapping_registry()

    assert registry.megatron_to_hf_lookup("decoder.layers.0.self_attention.in_proj.weight") is not None
    assert registry.megatron_to_hf_lookup("decoder.layers.6.self_attention.linear_gate.weight") is not None
    assert registry.megatron_to_hf_lookup("decoder.layers.3.mlp.router.weight") is not None
    assert registry.megatron_to_hf_lookup("decoder.layers.3.mlp.shared_experts.linear_fc2.weight") is not None

    mla_mapping = registry.hf_to_megatron_lookup("model.layers.3.attention.q_b_proj.weight")
    assert mla_mapping is not None
    assert mla_mapping.megatron_param == "decoder.layers.6.self_attention.linear_q_up_proj.weight"

    q_down = registry.hf_to_megatron_lookup("model.layers.3.attention.q_a_proj.weight")
    kv_down = registry.hf_to_megatron_lookup("model.layers.3.attention.kv_a_proj_with_mqa.weight")
    assert isinstance(q_down, AutoMapping)
    assert isinstance(kv_down, AutoMapping)

    kda_mapping = registry.megatron_to_hf_lookup("decoder.layers.0.self_attention.in_proj.weight")
    assert kda_mapping is not None
    assert list(kda_mapping.hf_param) == ["q", "k", "v", "f", "g"]


def test_flash_mapping_registry_covers_direct_q_and_mtp() -> None:
    bridge = BailingMoeV3Bridge()
    bridge.hf_config = _flash_config()
    registry = bridge.mapping_registry()

    direct_q = registry.hf_to_megatron_lookup("model.layers.5.attention.q_proj.weight")
    assert direct_q is not None
    assert direct_q.megatron_param == "decoder.layers.10.self_attention.linear_q_proj.weight"

    kda = registry.hf_to_megatron_lookup("model.layers.4.attention.q_proj.weight")
    assert kda is not None
    assert kda.megatron_param == "decoder.layers.8.self_attention.in_proj.weight"

    mtp_q = registry.hf_to_megatron_lookup("model.layers.42.attention.q_proj.weight")
    assert mtp_q is not None
    assert mtp_q.megatron_param == "mtp.layers.0.mtp_model_layer.layers.0.self_attention.linear_q_proj.weight"

    mtp_expert = registry.hf_to_megatron_lookup("model.layers.42.mlp.experts.7.down_proj.weight")
    assert mtp_expert is not None
    assert mtp_expert.megatron_param == "mtp.layers.0.mtp_model_layer.layers.1.mlp.experts.linear_fc2.weight7"

    mtp_norm = registry.hf_to_megatron_lookup("model.layers.42.enorm.weight")
    assert mtp_norm is not None
    assert mtp_norm.megatron_param == "mtp.layers.0.enorm.weight"

    kv_down = registry.hf_to_megatron_lookup("model.layers.5.attention.kv_a_proj_with_mqa.weight")
    mtp_kv_down = registry.hf_to_megatron_lookup("model.layers.42.attention.kv_a_proj_with_mqa.weight")
    assert isinstance(kv_down, AutoMapping)
    assert isinstance(mtp_kv_down, AutoMapping)


class _FakeTpInProjMapping(_BailingMoe3KDAInProjMapping):
    """Exercise section-wise TP logic without initializing distributed state."""

    _shared_splits = None

    def __init__(self, *args, rank: int, gathered: list[torch.Tensor] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._rank = rank
        self._gathered = gathered

    @property
    def tp_size(self) -> int:
        return 2

    @property
    def tp_rank(self) -> int:
        return self._rank

    def scatter_to_tp_ranks(self, splits, output_shape, dtype, device, src_rank=0):
        if splits is not None:
            type(self)._shared_splits = splits
        return type(self)._shared_splits[self._rank]

    def gather_from_tp_ranks(self, tensor):
        return self._gathered

    def broadcast_from_pp_rank(self, tensor, cache_key=None):
        return tensor


class _FakeTpConvMapping(_BailingMoe3KDAConv1dMapping):
    """Exercise KDA convolution section-wise TP logic without distributed state."""

    _shared_splits = None

    def __init__(self, *args, rank: int, gathered: list[torch.Tensor] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._rank = rank
        self._gathered = gathered

    @property
    def tp_size(self) -> int:
        return 2

    @property
    def tp_rank(self) -> int:
        return self._rank

    def scatter_to_tp_ranks(self, splits, output_shape, dtype, device, src_rank=0):
        if splits is not None:
            type(self)._shared_splits = splits
        return type(self)._shared_splits[self._rank]

    def gather_from_tp_ranks(self, tensor):
        return self._gathered

    def broadcast_from_pp_rank(self, tensor, cache_key=None):
        return tensor


def _kda_config() -> SimpleNamespace:
    return SimpleNamespace(
        linear_key_head_dim=4,
        linear_num_key_heads=1,
        linear_value_head_dim=4,
        linear_num_value_heads=1,
    )


def test_kda_projection_mapping_preserves_semantic_sections_under_tp() -> None:
    config = _kda_config()
    target = nn.Module()
    target.config = config
    target.in_proj = nn.Linear(4, 10, bias=False)
    source = {
        name: torch.arange(4, dtype=torch.float32).reshape(4, 1) + value
        for name, value in zip(("q", "k", "v", "f", "g"), (0, 10, 20, 30, 40), strict=True)
    }

    rank0 = _FakeTpInProjMapping("in_proj.weight", "q", "k", "v", "f", "g", rank=0)
    rank1 = _FakeTpInProjMapping("in_proj.weight", "q", "k", "v", "f", "g", rank=1)
    local0 = rank0.hf_to_megatron(source, target)
    local1 = rank1.hf_to_megatron(source, target)

    assert torch.equal(local0, torch.cat([source[name][:2] for name in ("q", "k", "v", "f", "g")]))
    assert torch.equal(local1, torch.cat([source[name][2:] for name in ("q", "k", "v", "f", "g")]))

    exported = _FakeTpInProjMapping(
        "in_proj.weight",
        "q",
        "k",
        "v",
        "f",
        "g",
        rank=0,
        gathered=[local0, local1],
    ).megatron_to_hf(local0, target)
    for name in source:
        assert torch.equal(exported[name], source[name])


def test_kda_conv_mapping_preserves_semantic_sections_under_tp() -> None:
    config = _kda_config()
    target = nn.Module()
    target.config = config
    target.conv1d = nn.Conv1d(1, 6, kernel_size=2, bias=False)
    source = {
        name: torch.arange(8, dtype=torch.float32).reshape(4, 1, 2) + value
        for name, value in zip(("q", "k", "v"), (0, 10, 20), strict=True)
    }

    rank0 = _FakeTpConvMapping("conv1d.weight", "q", "k", "v", rank=0)
    rank1 = _FakeTpConvMapping("conv1d.weight", "q", "k", "v", rank=1)
    local0 = rank0.hf_to_megatron(source, target)
    local1 = rank1.hf_to_megatron(source, target)

    assert torch.equal(local0, torch.cat([source[name][:2] for name in ("q", "k", "v")]))
    assert torch.equal(local1, torch.cat([source[name][2:] for name in ("q", "k", "v")]))

    exported = _FakeTpConvMapping(
        "conv1d.weight",
        "q",
        "k",
        "v",
        rank=0,
        gathered=[local0, local1],
    ).megatron_to_hf(local0, target)
    for name in source:
        assert torch.equal(exported[name], source[name])


@pytest.mark.parametrize("limit", [4.0, 5.0, 7.0, pytest.param(4.0, id="4.0-reuse")])
def test_post_silu_clamp_forward_and_analytic_backward(limit):
    # Include a gate above L whose SiLU is below L: pre-SiLU clipping fails here.
    gate = torch.tensor([-12.0, -1.0, 0.0, limit + 0.01, limit + 1.0, 12.0], dtype=torch.float64)
    up = torch.tensor([-12.0, 0.5, 2.0, -1.0, 3.0, 12.0], dtype=torch.float64)
    inputs = torch.cat([gate, up]).requires_grad_()
    actual = BailingMoe3ClampedSwiGLU(config=None, limit=limit)(inputs)
    sigmoid = 1 / (1 + torch.exp(-gate))
    silu = gate * sigmoid
    clipped_gate = torch.minimum(silu, torch.full_like(silu, limit))
    clipped_up = torch.maximum(torch.minimum(up, torch.full_like(up, limit)), torch.full_like(up, -limit))
    torch.testing.assert_close(actual, clipped_gate * clipped_up, rtol=1e-12, atol=1e-12)
    actual.sum().backward()
    gate_grad = clipped_up * sigmoid * (1 + gate * (1 - sigmoid)) * (silu <= limit)
    up_grad = clipped_gate * (up.abs() <= limit)
    torch.testing.assert_close(inputs.grad, torch.cat([gate_grad, up_grad]), rtol=1e-12, atol=1e-12)


def _capture(**kwargs):
    return kwargs


@pytest.mark.parametrize(
    "physical_layer,mtp,expert_limit,shared_limit",
    [
        (68, False, None, None),
        (70, False, None, 5),
        (72, False, 4, 5),
        (82, False, 4, 7),
        (84, False, 4, 7),
        (2, True, None, None),
    ],
)
@pytest.mark.parametrize("bias_activation_fusion", [False, True])
def test_clipping_uses_logical_layer_and_excludes_mtp(
    physical_layer, mtp, expert_limit, shared_limit, bias_activation_fusion
):
    native_experts = partial(_capture, submodules=SimpleNamespace(activation_func=None))
    native_shared = partial(_capture, submodules=SimpleNamespace(activation_func=None))
    native = partial(_capture, submodules=SimpleNamespace(experts=native_experts, shared_experts=native_shared))
    config = SimpleNamespace(
        bias_activation_fusion=bias_activation_fusion,
        use_transformer_engine_op_fuser=False,
        activation_func_clamp_value=None,
        use_te_activation_func=False,
    )
    result = build_bailing_moe3_moe(
        builder=native,
        expert_limits=[0] * 35 + [4] * 7,
        shared_limits=[0] * 34 + [5] * 6 + [7] * 2,
        layer_number=physical_layer,
        is_mtp_layer=mtp,
        config=config,
    )
    assert result["layer_number"] == physical_layer
    for name, expected in (("experts", expert_limit), ("shared_experts", shared_limit)):
        branch = getattr(result["submodules"], name)
        assert branch.keywords.get("limit") == expected
        extra = {"num_local_experts": 2} if name == "experts" else {}
        built = branch(config=config, **extra)
        if expected is None:
            assert built["config"] is config
            assert built["config"].bias_activation_fusion is bias_activation_fusion
            assert built["submodules"].activation_func is None
        else:
            assert built["config"] is not config
            assert built["config"].bias_activation_fusion is False
            assert built["config"].use_te_activation_func is True
            assert built["submodules"].activation_func.keywords["limit"] == expected
        assert config.bias_activation_fusion is bias_activation_fusion
        assert config.use_te_activation_func is False
    # A later layer must never mutate the reusable native spec.
    assert native.keywords["submodules"].experts is native_experts
    assert native_experts.keywords["submodules"].activation_func is None
    assert native_shared.keywords["submodules"].activation_func is None


@pytest.mark.parametrize(
    "op_fusion,native_clamp,message",
    [(True, None, "disabling TE operation fusion"), (False, 4.0, "pre-SiLU clamp")],
)
def test_clipped_activation_rejects_incompatible_fusion(op_fusion, native_clamp, message):
    config = SimpleNamespace(
        bias_activation_fusion=True,
        use_transformer_engine_op_fuser=op_fusion,
        activation_func_clamp_value=native_clamp,
        use_te_activation_func=False,
    )
    builder = partial(_capture, submodules=SimpleNamespace(activation_func=None))
    with pytest.raises(ValueError, match=message):
        _build_clamped_mlp(builder=builder, limit=4.0, config=config)
    assert config.bias_activation_fusion is True
    assert config.use_te_activation_func is False


@pytest.mark.parametrize("limits", [[0] * 41, [0] * 41 + [-1], [0] * 41 + [float("nan")]])
def test_reject_invalid_clipping_schedules(limits):
    with pytest.raises(ValueError, match="expert_swiglu_limit_list"):
        validate_swiglu_limits(limits, num_layers=42, field="expert_swiglu_limit_list")


@pytest.fixture(scope="module")
def native_groups(tmp_path_factory):
    """Initialize native state only for GPU cases; CPU cases never request it."""
    if not torch.cuda.is_available():
        pytest.skip("Native Transformer Engine tests require a CUDA device.")
    from megatron.bridge.models.bailing.bailing_moe3_spec import _missing_mcore_features

    missing = _missing_mcore_features()
    if missing:
        pytest.skip("Requires native Ling-capable MCore: " + ", ".join(missing))
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    # TE calls cuBLAS directly; torch.backends.matmul.allow_tf32 alone does not
    # control its FP32 grouped weight-gradient GEMM.
    previous_tf32_override = os.environ.get("NVIDIA_TF32_OVERRIDE")
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    rendezvous = tmp_path_factory.mktemp("ling_dist") / "store"
    torch.cuda.set_device(0)
    torch.distributed.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(123)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield ProcessGroupCollection.use_mpu_process_groups()
    torch.backends.cuda.matmul.allow_tf32 = old_tf32
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()
    if previous_tf32_override is None:
        os.environ.pop("NVIDIA_TF32_OVERRIDE", None)
    else:
        os.environ["NVIDIA_TF32_OVERRIDE"] = previous_tf32_override


def _compare(actual, reference, name):
    actual, reference = actual.detach().double().flatten(), reference.detach().double().flatten()
    if actual.norm() == 0 and reference.norm() == 0:
        return
    cosine = F.cosine_similarity(actual, reference, dim=0).item()
    similarity = (1 - (actual - reference).norm() / (actual.norm() + reference.norm()).clamp_min(1e-30)).item()
    logger.info("%s cosine=%.9f symmetric_similarity=%.9f", name, cosine, similarity)
    assert cosine >= 0.999, (name, cosine)
    assert similarity >= 0.999, (name, similarity)


def _compare_bf16_activation(actual, reference, name):
    assert torch.isfinite(actual).all(), name
    assert torch.isfinite(reference).all(), name
    actual, reference = actual.detach().double().flatten(), reference.detach().double().flatten()
    denominator = (actual.square() + reference.square()).sum()
    if denominator == 0:
        return
    cosine = F.cosine_similarity(actual, reference, dim=0).item()
    similarity = (2 * (actual * reference).sum() / denominator).item()
    logger.info("%s cosine=%.9f tensor_similarity=%.9f", name, cosine, similarity)
    assert cosine >= 0.999, (name, cosine)
    assert similarity >= 0.999, (name, similarity)


@pytest.mark.run_only_on("GPU")
class TestBailingMoe3NativeModules:
    """Check native construction and TE outputs/gradients with Flash expert dimensions."""

    @pytest.mark.parametrize("is_mtp", [False, True])
    def test_flash_native_mla_has_kv_norm_without_extra_direct_q_norm(self, native_groups, is_mtp):
        from megatron.core.transformer.spec_utils import build_module

        from megatron.bridge.models.bailing.bailing_moe3_bridge import BailingMoeV3Bridge
        from megatron.bridge.models.bailing.bailing_moe3_spec import bailing_moe3_hybrid_stack_spec

        config = BailingMoeV3Bridge().provider_bridge(SimpleNamespace(config=_flash_config()))
        config.init_model_with_meta_device = True
        config.finalize()
        spec = bailing_moe3_hybrid_stack_spec(config).submodules.mla_layer
        with torch.device("meta"):
            layer = build_module(
                spec,
                config=config,
                layer_number=1 if is_mtp else 11,
                is_mtp_layer=is_mtp,
                add_layer_offset=False,
                pg_collection=native_groups,
            )
        assert set(dict(layer.named_parameters())) == {
            "input_layernorm.weight",
            "self_attention.linear_q_proj.weight",
            "self_attention.linear_kv_down_proj.weight",
            "self_attention.kv_layernorm.weight",
            "self_attention.linear_kv_up_proj.weight",
            "self_attention.linear_gate.weight",
            "self_attention.linear_proj.weight",
        }

    @pytest.mark.parametrize(
        "layer_number,is_mtp,routed,shared",
        [(70, False, None, 5), (72, False, 4, 5), (84, False, 4, 7), (2, True, None, None)],
    )
    @pytest.mark.parametrize("bias_activation_fusion", [False, True])
    def test_native_layer_binds_limits_before_building_experts(
        self, native_groups, layer_number, is_mtp, routed, shared, bias_activation_fusion
    ):
        from megatron.core.transformer.spec_utils import build_module

        from megatron.bridge.models.bailing.bailing_moe3_provider import BailingMoe3HybridProvider
        from megatron.bridge.models.bailing.bailing_moe3_spec import bailing_moe3_hybrid_stack_spec

        config = BailingMoe3HybridProvider(
            num_layers=84,
            hidden_size=2560,
            num_attention_heads=32,
            ffn_hidden_size=6144,
            moe_ffn_hidden_size=768,
            moe_shared_expert_intermediate_size=768,
            num_moe_experts=512,
            moe_grouped_gemm=True,
            gated_linear_unit=True,
            activation_func=F.silu,
            add_bias_linear=False,
            bias_activation_fusion=bias_activation_fusion,
            gradient_accumulation_fusion=False,
            init_model_with_meta_device=True,
            expert_swiglu_limit_list=[0] * 35 + [4] * 7,
            share_expert_swiglu_limit_list=[0] * 34 + [5] * 6 + [7] * 2,
        )
        config.finalize()
        spec = bailing_moe3_hybrid_stack_spec(config).submodules.moe_layer
        with torch.device("meta"):
            layer = build_module(
                spec,
                config=config,
                layer_number=layer_number,
                is_mtp_layer=is_mtp,
                add_layer_offset=False,
                pg_collection=native_groups,
            )
        assert all(parameter.is_meta for parameter in layer.parameters())
        assert config.bias_activation_fusion is bias_activation_fusion
        for module, expected in ((layer.mlp.experts, routed), (layer.mlp.shared_experts, shared)):
            if expected is None:
                assert not isinstance(module.activation_func, BailingMoe3ClampedSwiGLU)
                assert module.config.bias_activation_fusion is bias_activation_fusion
            else:
                assert isinstance(module.activation_func, BailingMoe3ClampedSwiGLU)
                assert module.activation_func.limit == expected
                assert module.config.bias_activation_fusion is False
                assert module.config.use_te_activation_func is True

    @pytest.mark.parametrize("kind,limit", [("experts", 4.0), ("shared_experts", 5.0), ("shared_experts", 7.0)])
    @pytest.mark.parametrize("bias_activation_fusion", [False, True])
    def test_native_te_clipped_expert_output_and_gradients(self, native_groups, kind, limit, bias_activation_fusion):
        from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
        from megatron.core.transformer.transformer_config import TransformerConfig

        config = TransformerConfig(
            num_layers=2,
            hidden_size=2560,
            num_attention_heads=32,
            ffn_hidden_size=6144,
            moe_ffn_hidden_size=768,
            moe_shared_expert_intermediate_size=768,
            num_moe_experts=512,
            moe_grouped_gemm=True,
            gated_linear_unit=True,
            activation_func=F.silu,
            add_bias_linear=False,
            bias_activation_fusion=bias_activation_fusion,
            gradient_accumulation_fusion=False,
            params_dtype=torch.float32,
            use_cpu_initialization=False,
        )
        submodules = get_moe_module_spec(use_te=True, num_experts=512, moe_grouped_gemm=True).keywords["submodules"]
        extra = {"num_local_experts": 2} if kind == "experts" else {"gate": False}
        module = _build_clamped_mlp(
            builder=getattr(submodules, kind), limit=limit, config=config, pg_collection=native_groups, **extra
        ).cuda()
        torch.manual_seed(456)
        with torch.no_grad():
            for param in module.parameters():
                param.normal_(std=param.shape[-1] ** -0.5)
        x = (4 * torch.randn(16, 2560, device="cuda")).requires_grad_()
        rx = x.detach().double().requires_grad_()
        weights = {name: p.detach().double().requires_grad_() for name, p in module.named_parameters()}
        probs = torch.linspace(0.1, 0.9, 16, device="cuda").requires_grad_()
        rprobs = probs.detach().double().requires_grad_()

        def reference(inputs, w1, w2, scales=None):
            gate, up = (inputs @ w1.T).chunk(2, dim=-1)
            activated = gate / (1 + torch.exp(-gate))
            activated = torch.minimum(activated, torch.full_like(activated, limit))
            linear = torch.maximum(torch.minimum(up, torch.full_like(up, limit)), torch.full_like(up, -limit))
            hidden = activated * linear
            if scales is not None:
                hidden = hidden * scales.unsqueeze(-1)
            return hidden @ w2.T

        if kind == "experts":
            output, bias = module(x, torch.tensor([9, 7], device="cpu"), probs)
            expected = torch.cat(
                [
                    reference(rx[:9], weights["linear_fc1.weight0"], weights["linear_fc2.weight0"], rprobs[:9]),
                    reference(rx[9:], weights["linear_fc1.weight1"], weights["linear_fc2.weight1"], rprobs[9:]),
                ]
            )
        else:
            output, bias = module(x), None
            expected = reference(rx, weights["linear_fc1.weight"], weights["linear_fc2.weight"])
        assert bias is None
        _compare(output, expected, "output")
        upstream = torch.randn_like(output)
        (output * upstream).sum().backward()
        (expected * upstream.double()).sum().backward()
        _compare(x.grad, rx.grad, "input gradient")
        if kind == "experts":
            _compare(probs.grad, rprobs.grad, "routing probability gradient")
        for name, param in module.named_parameters():
            assert param.grad is not None, name
            _compare(param.grad, weights[name].grad, name)

    @pytest.mark.parametrize("limit", [4.0, 5.0, 7.0])
    @pytest.mark.parametrize("weighted", [False, True])
    def test_jit_clipped_swiglu_bf16_output_and_gradients(self, native_groups, limit, weighted):
        # Independent reference: inclusionAI/SGLang BailingMLP at
        # 822e3e2dd1bb236f5223d205aa4dcae91c8b3d40, post-SiLU gate clamp.
        # Keep Flash's expert width; cover saturation and both sides of the
        # BF16-representable clipping boundaries, including exact up boundaries.
        torch.manual_seed(1234)
        values = (6 * torch.randn(256, 2 * 768, device="cuda")).bfloat16()
        edges = torch.tensor(
            [
                -20,
                -limit - 0.125,
                -limit,
                -limit + 0.125,
                -1,
                0,
                1,
                limit - 0.0625,
                limit,
                limit + 0.03125,
                limit + 0.0625,
                limit + 0.125,
                limit + 1,
                20,
            ],
            device="cuda",
            dtype=torch.bfloat16,
        )
        values[0, : edges.numel()] = edges
        values[0, 768 : 768 + edges.numel()] = edges.flip(0)
        inputs = values.detach().requires_grad_()
        reference_inputs = values.float().requires_grad_()
        eager_inputs = values.detach().clone().requires_grad_()
        probabilities = torch.linspace(0.01, 0.99, values.shape[0], device="cuda").requires_grad_()
        reference_probabilities = probabilities.detach().clone().requires_grad_()
        eager_probabilities = probabilities.detach().clone().requires_grad_()

        activation = BailingMoe3ClampedSwiGLU(config=None, limit=limit)
        with torch.no_grad():
            checkpoint_output = activation(values)
        output = activation(inputs)
        _compare_bf16_activation(checkpoint_output, output, "checkpoint/recompute activation")
        gate, up = reference_inputs.chunk(2, dim=-1)
        activated = gate * torch.sigmoid(gate)
        activated = torch.where(activated > limit, limit, activated)
        clipped_up = torch.where(up > limit, limit, torch.where(up < -limit, -limit, up))
        expected = activated * clipped_up
        eager_gate, eager_up = eager_inputs.chunk(2, dim=-1)
        eager = F.silu(eager_gate).clamp(max=limit) * eager_up.clamp(-limit, limit)
        if weighted:
            # TEGroupedMLP applies FP32 router probabilities after activation,
            # then casts back to the activation dtype before the second GEMM.
            output = (output * probabilities.unsqueeze(-1)).to(inputs.dtype)
            eager = (eager * eager_probabilities.unsqueeze(-1)).to(inputs.dtype)
            expected = expected * reference_probabilities.unsqueeze(-1)
        upstream = torch.randn_like(output)
        (output * upstream).sum().backward()
        (eager * upstream).sum().backward()
        (expected * upstream.float()).sum().backward()
        for name, expected_output, expected_inputs in (
            ("FP32 reference", expected, reference_inputs),
            ("BF16 eager", eager, eager_inputs),
        ):
            _compare_bf16_activation(output, expected_output, name + " output")
            _compare_bf16_activation(inputs.grad, expected_inputs.grad, name + " input gradient")
        if weighted:
            _compare_bf16_activation(
                probabilities.grad, reference_probabilities.grad, "FP32 reference probability gradient"
            )
            _compare_bf16_activation(probabilities.grad, eager_probabilities.grad, "BF16 eager probability gradient")
