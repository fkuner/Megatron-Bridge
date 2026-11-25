import importlib
import os
from types import SimpleNamespace

import pytest
import torch
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge.models.bailing.bailing_moe3_bridge import BailingMoeV3Bridge
from megatron.bridge.training.config import apply_environment_variables
from tests.unit_tests.models.bailing.test_bailing_moe3_bridge import _flash_config, requires_ling_mcore


pytestmark = [pytest.mark.unit, requires_ling_mcore]


@pytest.mark.parametrize("local", [False, True])
def test_flash_sft_recipe_preserves_architecture_and_fsdp_contract(monkeypatch, local):
    module = importlib.import_module("megatron.bridge.recipes.bailing.h100.ling_v3_flash")
    source = _flash_config(
        expert_swiglu_limit_list=[0] * 35 + [4] * 7,
        share_expert_swiglu_limit_list=[0] * 34 + [5] * 6 + [7] * 2,
    )
    provider = BailingMoeV3Bridge().provider_bridge(SimpleNamespace(config=source))
    calls = []
    load_weights_calls = []

    class FakeBridge:
        @staticmethod
        def from_hf_pretrained(path, **kwargs):
            calls.append((path, kwargs))

            def to_provider(*, load_weights):
                load_weights_calls.append(load_weights)
                return provider

            return SimpleNamespace(to_megatron_provider=to_provider)

    monkeypatch.setattr(module, "AutoBridge", FakeBridge)
    if local:
        monkeypatch.setenv("LING_V3_FLASH_HF_PATH", "/offline/ling-flash")
    else:
        monkeypatch.delenv("LING_V3_FLASH_HF_PATH", raising=False)
    cfg = module.ling_v3_flash_sft_64gpu_h100_bf16_config()
    expected_path = "/offline/ling-flash" if local else module.LING_V3_FLASH_HF_MODEL
    revision = {} if local else {"revision": module.LING_V3_FLASH_HF_REVISION}
    assert calls == [(expected_path, {"trust_remote_code": True, **revision})]
    assert load_weights_calls == [False]
    assert cfg.checkpoint.hf_trust_remote_code
    assert cfg.tokenizer.tokenizer_model == expected_path
    assert cfg.tokenizer.hf_tokenizer_kwargs == revision
    for config in (cfg.model, cfg.ddp, cfg.scheduler, cfg.train, cfg.checkpoint):
        assert set(vars(config)) <= set(config.__dataclass_fields__)
    assert cfg.model.expert_swiglu_limit_list == source.expert_swiglu_limit_list
    assert cfg.model.share_expert_swiglu_limit_list == source.share_expert_swiglu_limit_list
    assert cfg.model.pipeline_model_parallel_size == cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.context_parallel_size == cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.attention_backend == AttnBackend.auto
    assert cfg.model.expert_model_parallel_size == 16
    assert cfg.ddp.use_megatron_fsdp and cfg.ddp.megatron_fsdp_version == 1
    assert (
        cfg.ddp.data_parallel_sharding_strategy
        == cfg.ddp.expert_data_parallel_sharding_strategy
        == "optim_grads_params"
    )
    assert cfg.ddp.megatron_fsdp_main_params_dtype == cfg.ddp.megatron_fsdp_main_grads_dtype == torch.float32
    assert cfg.checkpoint.ckpt_format == "fsdp_dtensor"
    assert cfg.checkpoint.strict_fsdp_dtensor_load is True
    assert cfg.checkpoint.load_rng is True
    assert cfg.checkpoint.save_rng is True
    assert cfg.checkpoint.save_optim is True
    assert cfg.checkpoint.load_optim is True
    assert cfg.checkpoint.save_interval == 50
    assert cfg.train.global_batch_size == 128 and cfg.train.micro_batch_size == 2
    assert cfg.train.train_iters == 100
    assert cfg.model.seq_length == cfg.dataset.seq_length == 4096
    assert cfg.dataset.enable_offline_packing is False
    assert cfg.dataset.dataset_kwargs["pad_to_max_length"] is True
    assert cfg.model.mtp_loss_scaling_factor == 0.1
    assert cfg.model.bias_activation_fusion is True
    assert cfg.model.gradient_accumulation_fusion is False
    assert cfg.ddp.suggested_communication_unit_size is None
    assert cfg.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] != "1"
    assert cfg.env_vars["NVTE_USE_CUTLASS_GROUPED_GEMM"] == "1"
    assert cfg.env_vars["NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK"] == "0"

    for name in cfg.env_vars:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    apply_environment_variables(cfg)
    assert os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"] == "1"
    assert os.environ["NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK"] == "0"
    monkeypatch.setenv("NVTE_USE_CUTLASS_GROUPED_GEMM", "0")
    apply_environment_variables(cfg)
    assert os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"] == "0"

    # Finalization must retain the complete HF architecture, including MTP.
    cfg.model.finalize()
    main_pattern, mtp_pattern = cfg.model.hybrid_layer_pattern.split("/")
    assert len(main_pattern) == cfg.model.num_layers == 2 * source.num_hidden_layers
    assert "|" not in main_pattern and mtp_pattern == "+E"
    assert cfg.model.q_lora_rank is None
    assert cfg.model.init_model_with_meta_device
    assert (cfg.model.moe_token_dispatcher_type, cfg.model.moe_flex_dispatcher_backend) == ("flex", "hybridep")
    assert cfg.model.moe_flex_dispatcher_num_sms == 8
    assert cfg.model.moe_hybridep_num_sms is None
    assert cfg.model.moe_deepep_num_sms is None
    assert cfg.model.moe_permute_fusion is True
    assert cfg.model.moe_router_fusion is True
    assert cfg.model.moe_permute_fusion_into_hybridep is False
    assert cfg.model.moe_expert_capacity_factor is None
    assert cfg.model.moe_expert_rank_capacity_factor is None
    assert (cfg.model.recompute_granularity, cfg.model.recompute_method, cfg.model.recompute_num_layers) == (
        "full",
        "uniform",
        1,
    )
    exported = BailingMoeV3Bridge.megatron_to_hf_config(cfg.model)
    for field in (
        "num_hidden_layers",
        "num_nextn_predict_layers",
        "expert_swiglu_limit_list",
        "share_expert_swiglu_limit_list",
    ):
        assert exported[field] == getattr(source, field)
