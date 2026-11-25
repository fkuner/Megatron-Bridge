# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Ling 3.0 Flash full-parameter fine-tuning recipe for 64 H100 GPUs."""

import os

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _sft_common
from megatron.bridge.recipes.utils.dataset_utils import default_squad_config
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.config import ConfigContainer


LING_V3_FLASH_HF_MODEL = "inclusionAI/Ling-3.0-flash"
LING_V3_FLASH_HF_REVISION = "fcc60a52b34c69c6271d0ac8a8cb0815020b6a43"  # pragma: allowlist secret


def ling_v3_flash_sft_64gpu_h100_bf16_config() -> ConfigContainer:
    """Full-parameter SFT with EP16, HybridEP, m-FSDP v1, and TP=PP=CP=ETP=1.

    Dense parameters shard across 64 ranks and expert parameters across four
    expert-DP ranks. Model construction reads the HF configuration only. Set
    LING_V3_FLASH_HF_PATH for a local snapshot and pass --pretrained_checkpoint
    to run_recipe.py to initialize weights from HF. Weight-only fsdp_dtensor
    initialization requires a separate RNG-load fix. The full 42-block
    architecture and MTP layer are preserved.
    Requires HybridEP and Transformer Engine with CUTLASS grouped GEMM support.
    """
    cfg = _sft_common()
    hf_path = os.environ.get("LING_V3_FLASH_HF_PATH", LING_V3_FLASH_HF_MODEL)
    revision = {"revision": LING_V3_FLASH_HF_REVISION} if hf_path == LING_V3_FLASH_HF_MODEL else {}
    cfg.model = AutoBridge.from_hf_pretrained(hf_path, trust_remote_code=True, **revision).to_megatron_provider(
        load_weights=False
    )
    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.tokenizer.tokenizer_model = hf_path
    cfg.tokenizer.hf_tokenizer_kwargs = revision
    cfg.checkpoint.hf_trust_remote_code = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 16
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.init_model_with_meta_device = True
    cfg.model.seq_length = 4096
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 8
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_router_fusion = True
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.mtp_loss_scaling_factor = 0.1
    cfg.model.bias_activation_fusion = True
    cfg.model.gradient_accumulation_fusion = False
    cfg.model.use_transformer_engine_op_fuser = False
    cfg.model.fp8 = None
    cfg.model.cuda_graph_impl = "none"
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1

    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.megatron_fsdp_version = 1
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.expert_data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.average_in_collective = False
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.megatron_fsdp_main_params_dtype = torch.float32
    cfg.ddp.megatron_fsdp_main_grads_dtype = torch.float32
    cfg.ddp.megatron_fsdp_grad_comm_dtype = torch.float32
    cfg.optimizer.use_distributed_optimizer = True

    cfg.dataset = default_squad_config(seq_length=cfg.model.seq_length, enable_offline_packing=False)
    cfg.dataset.dataset_kwargs["pad_to_max_length"] = True
    cfg.train.micro_batch_size = 2
    cfg.train.global_batch_size = 128
    cfg.train.train_iters = 100
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100
    cfg.checkpoint.ckpt_format = "fsdp_dtensor"
    cfg.checkpoint.strict_fsdp_dtensor_load = True
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.fully_parallel_save = False
    cfg.validation.eval_interval = 50
    cfg.validation.eval_iters = 4
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": "32",
        "NVIDIA_TF32_OVERRIDE": "0",
        "NVTE_USE_CUTLASS_GROUPED_GEMM": "1",
        "NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK": "0",
    }
    return cfg


__all__ = [
    "LING_V3_FLASH_HF_MODEL",
    "LING_V3_FLASH_HF_REVISION",
    "ling_v3_flash_sft_64gpu_h100_bf16_config",
]
