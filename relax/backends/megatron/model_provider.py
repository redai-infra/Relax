# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Adapt from https://github.com/NVIDIA/Megatron-LM/blob/b1efb3c7126ef7615e8c333432d76e08038e17ff/pretrain_gpt.py
import argparse
import inspect
import os
import pickle
import re
from contextlib import nullcontext
from typing import Literal

import torch
import torch.distributed as dist
from megatron.core import tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import core_transformer_config_from_args

from relax.utils.logging_utils import get_logger
from relax.utils.misc import load_function


logger = get_logger(__name__)


# Adapt from https://github.com/volcengine/verl/blob/c3b20575d2bc815fcccd84bddb4c0401fc4b632b/verl/models/llama/megatron/layers/parallel_linear.py#L82
class LinearForLastLayer(torch.nn.Linear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features=input_size, out_features=output_size, bias=bias)
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel:
            self.weight.sequence_parallel = True
            if bias:
                self.bias.sequence_parallel = True

        self.weight.data.normal_(mean=0.0, std=0.02)
        if bias:
            self.bias.data.zero_()

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        logits = super().forward(input_)
        logits = logits.float()
        if self.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
        return logits, None


# CP-PROBE: one-shot forward-pre-hook on the first attention module to verify that
# context parallelism actually splits the sequence dimension at the attention input.
# Compare seq_len across CP=1 vs CP=2 runs — it must halve.  Remove after verifying.
_CP_PROBE_INSTALLED = False


def _maybe_mark_unsplit_forward(args: argparse.Namespace, model: torch.nn.Module) -> None:
    """Mark `args.uses_unsplit_forward` when the bridge produces a model whose
    forward expects UNSPLIT input + global cu_seqlens + attention_mask and does
    CP+SP splitting internally (Qwen3VLModel family — used for Qwen3-VL and
    text-only Qwen3.5 / Qwen3.6 sharing the same architecture).

    Read by data.py / loss.py to build unsplit tokens + tp*cp*2-aligned
    cu_seqlens instead of the pre-split + cp-multiplied form, and by model.py
    to route those through the forward.
    """
    try:
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
    except ImportError:
        return
    if isinstance(model, Qwen3VLModel):
        args.uses_unsplit_forward = True


def _install_cp_probe(model: torch.nn.Module) -> None:
    global _CP_PROBE_INSTALLED
    if _CP_PROBE_INSTALLED:
        return

    from megatron.core import mpu

    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    state = {"n": 0}

    target_classes = (
        "DotProductAttention",
        "TEDotProductAttention",
        "FusedAttention",
        "FlashAttention",
    )

    def hook(module, args, kwargs):
        if state["n"] >= 2 or tp_rank != 0:
            return
        shapes: dict[str, object] = {}
        for name in (
            "query",
            "key",
            "value",
            "q",
            "k",
            "v",
            "hidden_states",
            "query_layer",
            "key_layer",
            "value_layer",
        ):
            t = kwargs.get(name)
            if torch.is_tensor(t):
                shapes[name] = tuple(t.shape)
        for i, t in enumerate(args):
            if torch.is_tensor(t):
                shapes[f"arg{i}"] = tuple(t.shape)
        for name in ("cu_seqlens_q", "cu_seqlens_kv"):
            t = kwargs.get(name)
            if torch.is_tensor(t):
                shapes[name] = t.tolist()  # one-shot sync, OK for probe
        logger.debug(f"[CP-PROBE] cp_rank={cp_rank}/{cp_size} module={type(module).__name__} shapes={shapes}")
        state["n"] += 1

    skip_prefixes = ("vision_model", "visual", "vit", "image_encoder", "projector", "audio")

    def is_llm_backbone(n: str) -> bool:
        return not any(p in n for p in skip_prefixes)

    matches = [(n, m) for n, m in model.named_modules() if type(m).__name__ in target_classes]
    llm_matches = [(n, m) for n, m in matches if is_llm_backbone(n)]
    chosen = llm_matches or matches  # fallback to vision if no LLM backbone in this stage

    if chosen:
        name, m = chosen[0]
        m.register_forward_pre_hook(hook, with_kwargs=True)
        logger.debug(
            f"[CP-PROBE] hook installed on '{name}' ({type(m).__name__}) "
            f"cp_size={cp_size} cp_rank={cp_rank} "
            f"(total_attn_modules={len(matches)}, llm_backbone={len(llm_matches)})"
        )
        _CP_PROBE_INSTALLED = True
        return

    candidates = [(n, type(m).__name__) for n, m in model.named_modules() if "attention" in n.lower()][:8]
    logger.warning(
        f"[CP-PROBE] no attention module matched on this stage (cp_rank={cp_rank}); "
        f"attention-like candidates: {candidates}"
    )


def get_model_provider_func(
    args: argparse.Namespace,
    role: Literal["actor", "critic"] = "actor",
):
    # Support custom model provider path (similar to --custom-rm-path for reward models)
    if getattr(args, "custom_model_provider_path", None):

        def wrapped_model_provider(
            pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None
        ) -> GPTModel:
            custom_model_provider = load_function(args.custom_model_provider_path)
            # Check if the custom provider supports vp_stage parameter
            has_vp_stage = "vp_stage" in inspect.signature(custom_model_provider).parameters
            if has_vp_stage:
                model = custom_model_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
            else:
                model = custom_model_provider(pre_process=pre_process, post_process=post_process)
            # Apply critic output layer if needed
            if post_process and role == "critic":
                model.output_layer = LinearForLastLayer(
                    input_size=model.config.hidden_size, output_size=1, config=model.config
                )
            _maybe_mark_unsplit_forward(args, model)
            _install_cp_probe(model)
            return model

        return wrapped_model_provider

    if args.megatron_to_hf_mode == "bridge":
        from megatron.bridge import AutoBridge

        bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
        provider = bridge.to_megatron_provider(load_weights=False)
        # Override provider attributes with matching args values
        bridge_keys = [
            "attention_backend",
            "tensor_model_parallel_size",
            "sequence_parallel",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "expert_model_parallel_size",
            "expert_tensor_parallel_size",
            "variable_seq_lengths",
            "dsa_indexer_loss_coeff",
            "dsa_indexer_use_sparse_loss",
            "attention_softmax_in_fp32",
            "bias_dropout_fusion",
            "apply_rope_fusion",
            "recompute_granularity",
            "recompute_method",
            "recompute_num_layers",
            "distribute_saved_activations",
            "moe_router_load_balancing_type",
            "moe_router_dtype",
            "moe_aux_loss_coeff",
            "moe_token_dispatcher_type",
            "moe_shared_expert_overlap",
            "moe_enable_deepep",
            "moe_flex_dispatcher_backend",
            "use_audio_in_video",
            "freeze_language_model",
            "freeze_vision_model",
            "freeze_vision_projection",
            # https://github.com/redai-infra/Megatron-Bridge/commit/960bb5f18800d3e1fb9815e95daa185ab06c09ea
            "vision_dp_when_tp",
            "calculate_per_token_loss",
            # Allow CLI to override layer count / MoE frequency for layer-reduced training
            "num_layers",
            "moe_layer_freq",
        ]

        args_dict = vars(args)
        for attr in vars(provider):
            if attr in args_dict and attr in bridge_keys:
                old_val = getattr(provider, attr)
                new_val = args_dict[attr]
                if old_val != new_val:
                    logger.info(f"Override provider.{attr}: {old_val!r} -> {new_val!r}")
                setattr(provider, attr, new_val)

        # Handle name-mismatched attributes that require explicit mapping
        if getattr(args, "decoder_first_pipeline_num_layers", None) is not None:
            provider.num_layers_in_first_pipeline_stage = args.decoder_first_pipeline_num_layers
        if getattr(args, "decoder_last_pipeline_num_layers", None) is not None:
            provider.num_layers_in_last_pipeline_stage = args.decoder_last_pipeline_num_layers

        if args.fp16:
            provider.fp16 = True
            provider.bf16 = False
            provider.params_dtype = torch.float16
        elif args.bf16:
            provider.fp16 = False
            provider.bf16 = True
            provider.params_dtype = torch.bfloat16

        provider.finalize()

        # Pickle provider for offline inspection / reproducibility (only on rank 0)
        if not dist.is_initialized() or dist.get_rank() == 0:
            save_path = getattr(args, "save", None) or "/tmp/relax"
            os.makedirs(save_path, exist_ok=True)
            pkl_path = os.path.join(save_path, "transformer_config.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(provider, f)
            logger.info(f"Provider config saved to {pkl_path}")

        original_provide = provider.provide

        def provide_with_cp_probe(*p_args, **p_kwargs):
            model = original_provide(*p_args, **p_kwargs)
            _maybe_mark_unsplit_forward(args, model)
            _install_cp_probe(model)
            return model

        return provide_with_cp_probe

    def model_provider(pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None) -> GPTModel:
        """Builds the model.

        If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

        Args:
            pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
            post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


        Returns:
            Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
        """
        use_te = args.transformer_impl == "transformer_engine"

        # Experimental loading arguments from yaml
        config: TransformerConfig = core_transformer_config_from_args(args)

        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
            # Allow the spec to be a function so that user can use customized Megatron easier.
            if callable(transformer_layer_spec):
                transformer_layer_spec = transformer_layer_spec(args, config, vp_stage)
        else:
            if args.num_experts:
                # Define the decoder block spec
                kwargs = {
                    "use_transformer_engine": use_te,
                }
                if vp_stage is not None:
                    kwargs["vp_stage"] = vp_stage
                transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        num_experts=args.num_experts,
                        moe_grouped_gemm=args.moe_grouped_gemm,
                        qk_layernorm=args.qk_layernorm,
                        multi_latent_attention=args.multi_latent_attention,
                        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
                    )
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        num_experts=args.num_experts,
                        moe_grouped_gemm=args.moe_grouped_gemm,
                        qk_layernorm=args.qk_layernorm,
                        multi_latent_attention=args.multi_latent_attention,
                        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
                    )

        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init

                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True

                # Check if fp8_model_init supports preserve_high_precision_init_val
                if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except Exception as e:
                raise RuntimeError(
                    "--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found."
                ) from e

        kwargs = {
            "config": config,
            "transformer_layer_spec": transformer_layer_spec,
            "vocab_size": args.padded_vocab_size,
            "max_sequence_length": args.max_position_embeddings,
            "pre_process": pre_process,
            "post_process": post_process,
            "fp16_lm_cross_entropy": args.fp16_lm_cross_entropy,
            "parallel_output": True,
            "share_embeddings_and_output_weights": not args.untie_embeddings_and_output_weights,
            "position_embedding_type": args.position_embedding_type,
            "rotary_percent": args.rotary_percent,
            "rotary_base": args.rotary_base,
            "rope_scaling": args.use_rope_scaling,
        }

        if vp_stage is not None:
            kwargs["vp_stage"] = vp_stage

        if args.mtp_num_layers:
            from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

            mtp_kwargs = {
                "use_transformer_engine": use_te,
            }
            if vp_stage is not None:
                mtp_kwargs["vp_stage"] = vp_stage

            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, **mtp_kwargs)
            kwargs["mtp_block_spec"] = mtp_block_spec

        with build_model_context(**build_model_context_args):
            model = GPTModel(**kwargs)

        if post_process and role == "critic":
            model.output_layer = LinearForLastLayer(input_size=config.hidden_size, output_size=1, config=config)

        _maybe_mark_unsplit_forward(args, model)
        _install_cp_probe(model)
        return model

    return model_provider


def wrap_model_provider_with_freeze(original_provider, args):
    def wrapped_provider(pre_process=True, post_process=True, vp_stage=None, **kwargs):
        sig = inspect.signature(original_provider)
        if "vp_stage" in sig.parameters:
            model = original_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        else:
            model = original_provider(pre_process=pre_process, post_process=post_process)

        freeze_model_params(model, args)

        return model

    return wrapped_provider


def freeze_model_params(model: GPTModel, args: argparse.Namespace):
    if args.only_train_params_name_list:
        for name, param in model.named_parameters():
            param.requires_grad = False
            for pattern in args.only_train_params_name_list:
                if re.search(pattern, name):
                    param.requires_grad = True
                    break

    if args.freeze_params_name_list:
        for name, param in model.named_parameters():
            for pattern in args.freeze_params_name_list:
                if re.search(pattern, name):
                    param.requires_grad = False
                    break
