# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import torch
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import split_deepstack_embs
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync
from megatron.core import InferenceParams, mpu, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerConfig as Qwen3OmniMoeThinkerConfigHF,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder as Qwen3OmniMoeAudioEncoderHF,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeVisionEncoder as Qwen3OmniMoeVisionEncoderHF,
)

from relax.models.qwen_omni.modeling_qwen3_omni.text_model import Qwen3OmniGPTModel
from relax.models.qwen_omni.modeling_qwen3_omni.transformer_config import Qwen3OmniTransformerConfig
from relax.models.qwen_omni.modeling_qwen3_omni.utils import get_rope_index


class Qwen3OmniMoeModel(MegatronModule):
    """Qwen3 Omni MoE Thinker Model for multimodal understanding.

    This model supports audio, image, and video inputs in addition to text.
    It processes multimodal inputs through separate encoders and combines them
    for the language model.

    This is a standalone implementation that does not inherit from other models
    to maintain independence from version-specific implementations.
    """

    def __init__(
        self,
        language_transformer_config: Qwen3OmniTransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        audio_transformer_config: Qwen3OmniMoeThinkerConfigHF,
        vision_transformer_config: Qwen3OmniMoeThinkerConfigHF,
        parallel_output: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        use_audio_in_video: bool = False,
        pg_collection=None,
    ):
        super().__init__(config=language_transformer_config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.pg_collection = pg_collection
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        self.encoder_hidden_state = None
        self.vision_model = None
        self.language_model = None
        self.image_token_id = language_transformer_config.image_token_id
        self.video_token_id = language_transformer_config.video_token_id
        self.vision_start_token_id = language_transformer_config.vision_start_token_id

        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        self.share_embeddings_and_output_weights = False

        self.position_id_per_seconds = language_transformer_config.position_id_per_seconds
        self.audio_token_id = language_transformer_config.audio_token_id
        self.audio_start_token_id = language_transformer_config.audio_start_token_id
        self.use_audio_in_video = use_audio_in_video
        self.audio_model = None

        if self.pre_process:
            # Initialize audio and vision models with random weights from config
            self.audio_model = Qwen3OmniMoeAudioEncoderHF._from_config(audio_transformer_config)
            self.vision_model = Qwen3OmniMoeVisionEncoderHF._from_config(vision_transformer_config)
            # Ensure HF encoder params are marked for TP grad sync and future assignments are hooked.
            hook_hf_module_setattr_for_tp_grad_sync(self.audio_model)
            hook_hf_module_setattr_for_tp_grad_sync(self.vision_model)
            # Move to device if available
            if torch.cuda.is_available():
                self.audio_model = self.audio_model.to("cuda")
                self.vision_model = self.vision_model.to("cuda")

        self.language_model = Qwen3OmniGPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_transformer_config.vocab_size,
            max_sequence_length=language_transformer_config.language_max_sequence_length,
            parallel_output=parallel_output,
            position_embedding_type="mrope",
            rotary_percent=language_transformer_config.rotary_percent,
            pre_process=self.pre_process,
            post_process=self.post_process,
            rotary_base=language_transformer_config.rotary_base,
            fp16_lm_cross_entropy=language_transformer_config.fp16_lm_cross_entropy,
            share_embeddings_and_output_weights=language_transformer_config.share_embeddings_and_output_weights,
            scatter_embedding_sequence_parallel=False,
            pg_collection=pg_collection,
        )
        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights

    def set_input_tensor(self, input_tensor) -> None:
        """Set input tensor to be used instead of forward()'s input.

        When the pipeline parallel size > 1, the input tensor is received from
        the previous pipeline stage and must be provided to the model via this method.

        Args:
            input_tensor (list or torch.Tensor): Input tensor(s) from the previous pipeline stage.
        """
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for Qwen3OmniMoeModel"

        if self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
        freeze_audio_model: bool = False,
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_vision_projection (bool): Freeze the vision projection modules.
            freeze_audio_model (bool): Freeze the audio model module.
        """
        if freeze_language_model and self.language_model is not None:
            for param in self.language_model.parameters():
                param.requires_grad = False

        if freeze_vision_model and self.vision_model is not None:
            for param in self.vision_model.parameters():
                param.requires_grad = False

        if freeze_audio_model and self.audio_model is not None:
            self.audio_model._freeze_parameters()

    def forward(
        self,
        input_ids: torch.Tensor,
        input_features: torch.Tensor = None,
        position_ids: torch.Tensor = None,  # can set at dataset
        attention_mask: torch.Tensor = None,
        feature_attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        loss_mask: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        pixel_values: torch.Tensor = None,
        pixel_values_videos: torch.Tensor = None,
        image_grid_thw: torch.Tensor = None,
        video_grid_thw: torch.Tensor = None,
        image_input_mask: torch.Tensor = None,
        video_second_per_grid=None,
    ) -> torch.Tensor:
        """Forward function of the Qwen3 Omni model.

        Args:
            input_ids (torch.Tensor): input text ids [batch, text_seq_len].
            input_features (torch.Tensor): audio features.
            position_ids (torch.Tensor): input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): attention mask for the language model.
            feature_attention_mask (torch.Tensor): attention mask for audio features.
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            loss_mask (torch.Tensor): Loss mask.
            inference_params (InferenceParams): Inference-time parameters including KV cache.
            packed_seq_params (PackedSeqParams): Packed sequence parameters.
            extra_block_kwargs (dict): Extra block kwargs.
            pixel_values (torch.Tensor): Image pixel values.
            pixel_values_videos (torch.Tensor): Video pixel values.
            image_grid_thw (torch.Tensor): Image grid dimensions.
            video_grid_thw (torch.Tensor): Video grid dimensions.
            image_input_mask (torch.Tensor): Image input mask.
            video_second_per_grid (torch.Tensor): Seconds per video grid.

        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits.
        """
        assert inference_params is None, "not support inference"

        video_start_index = 0
        vision_grid_thw = None
        vision_data = None
        image_mask = None
        video_mask = None
        deepstack_feature_lists = None
        # position ids is computed within the model
        position_ids = None
        audio_feature_lengths = None

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)

        if self.pre_process:
            # =========================
            # image / Video
            # =========================
            if image_grid_thw is not None or video_grid_thw is not None:
                if image_grid_thw is not None:
                    image_mask = image_input_mask
                    if image_mask is None:
                        image_mask = (input_ids == self.image_token_id).contiguous()
                    vision_grid_thw = image_grid_thw
                    vision_data = pixel_values
                    video_start_index = image_mask.sum().item()
                else:
                    video_start_index = 0

                # Handle videos - concatenate if both present
                if video_grid_thw is not None:
                    video_mask = (input_ids == self.video_token_id).contiguous()
                    if vision_grid_thw is not None:
                        # Both images and videos present - concatenate
                        vision_grid_thw = torch.cat([vision_grid_thw, video_grid_thw], dim=0)
                        vision_data = torch.cat([vision_data, pixel_values_videos], dim=0)
                    else:
                        # Only videos present
                        vision_grid_thw = video_grid_thw
                        vision_data = pixel_values_videos

            vision_embeds = None
            if vision_grid_thw is not None and vision_grid_thw.shape[0] > 0:
                vision_outputs = self.vision_model(
                    hidden_states=vision_data,
                    grid_thw=vision_grid_thw,
                )

                import transformers
                from packaging import version

                if version.parse(transformers.__version__) >= version.parse("5.0.0"):
                    vision_embeds = vision_outputs.pooler_output
                    deepstack_feature_lists = vision_outputs.deepstack_features
                else:
                    vision_embeds, deepstack_feature_lists = vision_outputs

            combined_embeddings = self.language_model.embedding(
                input_ids=input_ids,
                position_ids=None,  # NOTE: disable
            ).clone()  # [text_seq_len, b, h_language]

            if vision_embeds is not None:
                if video_start_index == 0:
                    image_embeds = None
                    video_embeds = vision_embeds
                elif video_start_index == vision_embeds.shape[0]:
                    image_embeds = vision_embeds
                    video_embeds = None
                elif 0 < video_start_index < vision_embeds.shape[0]:
                    image_embeds = vision_embeds[:video_start_index]
                    video_embeds = vision_embeds[video_start_index:]
                else:
                    raise ValueError(
                        f"Expect video token start index in range [0, {vision_embeds.shape[0]}], but got "
                        f"{video_start_index}"
                    )

                if image_embeds is not None:
                    combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
                    combined_embeddings[image_mask] = image_embeds
                    combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()

                if video_embeds is not None:
                    combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
                    combined_embeddings[video_mask] = video_embeds
                    combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()

                # Create visual_pos_masks for deepstack processing
                if image_embeds is not None and video_embeds is not None:
                    visual_pos_masks = image_mask | video_mask
                elif image_embeds is not None:
                    visual_pos_masks = image_mask
                elif video_embeds is not None:
                    visual_pos_masks = video_mask
                else:
                    visual_pos_masks = None
            else:
                visual_pos_masks = None

            # =========================
            # Audio
            # =========================
            if input_features is not None:
                audio_mask = (input_ids == self.audio_token_id).contiguous()
                if feature_attention_mask is not None:
                    input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)

                feature_lens = (
                    audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
                )

                # dtype from fp32 to bf16
                audio_outputs = self.audio_model(
                    input_features.to(next(self.audio_model.parameters()).dtype),
                    feature_lens=feature_lens,
                )
                audio_embeds = audio_outputs.last_hidden_state  # [num_audio_tokens, hidden]
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()

                combined_embeddings[audio_mask] = audio_embeds
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()

            if self.config.sequence_parallel:
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)
                combined_embeddings = combined_embeddings.contiguous()
        else:
            combined_embeddings = None
            visual_pos_masks = None

        cu_seqlens_padded = None
        if packed_seq_params is not None:
            if packed_seq_params.cu_seqlens_q_padded is not None:
                cu_seqlens_padded = packed_seq_params.cu_seqlens_q_padded
            else:
                cu_seqlens_padded = packed_seq_params.cu_seqlens_q

        hf_attention_mask = None
        if position_ids is None:
            input_ids_for_rope_index = input_ids
            if cu_seqlens_padded is not None:

                def thd_to_bshd(packed_values: torch.Tensor, cu_seqlens: torch.Tensor):
                    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
                    max_seq_len = seqlens.max()
                    bs = len(cu_seqlens) - 1
                    results = packed_values.new_zeros(size=(bs, max_seq_len, *packed_values.shape[2:]))
                    for i, seqlen in enumerate(seqlens):
                        results[i, :seqlen] = packed_values[0, cu_seqlens[i] : cu_seqlens[i] + seqlen]
                    return results

                def bshd_to_thd(unpacked_values: torch.Tensor, cu_seqlens: torch.Tensor):
                    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
                    total_len = cu_seqlens[-1]
                    results = unpacked_values.new_zeros(size=(1, total_len, *unpacked_values.shape[2:]))
                    for i, seqlen in enumerate(seqlens):
                        results[0, cu_seqlens[i] : cu_seqlens[i] + seqlen] = unpacked_values[i, :seqlen]
                    return results

                input_ids_for_rope_index = thd_to_bshd(input_ids, cu_seqlens_padded)

            # =========================
            # RoPE index (audio-aware)
            # =========================
            position_ids, _ = get_rope_index(
                spatial_merge_size=self.config.spatial_merge_size,
                image_token_id=self.image_token_id,
                video_token_id=self.video_token_id,
                audio_token_id=self.audio_token_id,
                vision_start_token_id=self.vision_start_token_id,
                audio_start_token_id=self.audio_start_token_id,
                input_ids=input_ids_for_rope_index,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                audio_seqlens=audio_feature_lengths,
                attention_mask=hf_attention_mask,
                use_audio_in_video=self.use_audio_in_video,
                second_per_grids=video_second_per_grid,
                position_id_per_seconds=self.position_id_per_seconds,
            )
            if cu_seqlens_padded is not None:
                position_ids = bshd_to_thd(position_ids.permute(1, 2, 0), cu_seqlens_padded).permute(2, 0, 1)

        deepstack_visual_embeds = deepstack_feature_lists

        # Split visual_pos_masks and deepstack_visual_embeds for sequence parallel / CP
        if self.config.sequence_parallel and visual_pos_masks is not None and deepstack_visual_embeds is not None:
            if self.pg_collection is not None:
                tp_size = self.pg_collection.tp.size()
                tp_rank = self.pg_collection.tp.rank()
            else:
                tp_size = mpu.get_tensor_model_parallel_world_size()
                tp_rank = mpu.get_tensor_model_parallel_rank()
            visual_pos_masks, deepstack_visual_embeds = split_deepstack_embs(
                visual_pos_masks,
                deepstack_visual_embeds,
                tp_size=tp_size,
                tp_rank=tp_rank,
                cp_size=1,
                cp_rank=0,
                sequence_parallel=True,
            )

        output = self.language_model(
            input_ids=None,
            position_ids=position_ids,  # None in encoder
            attention_mask=attention_mask,  # None in encoder
            decoder_input=combined_embeddings,  # only not None in the first decoder PP stage
            labels=labels,  # only not None in the last decoder PP stage
            loss_mask=loss_mask,
            inference_params=inference_params,  # currently always None
            packed_seq_params=packed_seq_params,  # currently always None
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **(extra_block_kwargs or {}),
        )

        return output
