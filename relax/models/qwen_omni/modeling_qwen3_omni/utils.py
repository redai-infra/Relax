# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from typing import Optional

import torch
from megatron.core.packed_seq_params import PackedSeqParams


def _get_feat_extract_output_lengths(input_lengths):
    """Computes the output length of the convolutional layers and the output
    length of the audio encoder."""

    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


def get_llm_pos_ids_for_vision(
    self,
    start_idx: int,
    vision_idx: int,
    spatial_merge_size: int,
    t_index: list[torch.Tensor],
    grid_hs: list[torch.Tensor],
    grid_ws: list[torch.Tensor],
):
    """Generate LLM position IDs for vision tokens.

    Computes position embeddings for vision tokens (images/videos) by creating
    3D position indices (temporal, height, width) based on spatial merge size.

    Args:
        self: Instance reference.
        start_idx: Starting position index offset.
        vision_idx: Index of the vision sample.
        spatial_merge_size: Size of spatial merge for grid downsampling.
        t_index: List of temporal indices.
        grid_hs: List of grid heights.
        grid_ws: List of grid widths.

    Returns:
        torch.Tensor: Position IDs of shape [3, num_tokens] with temporal, height, width indices.
    """
    llm_pos_ids_list = []
    llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
    llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten().float()
    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten().float()
    t_index = torch.Tensor(t_index).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().float()
    _llm_pos_ids = torch.stack([t_index, h_index, w_index])
    llm_pos_ids_list.append(_llm_pos_ids + start_idx)
    llm_pos_ids = torch.cat(llm_pos_ids_list, dim=1)
    return llm_pos_ids


def get_rope_index(
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    audio_token_id: int,
    vision_start_token_id: int,
    audio_start_token_id: int,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    audio_seqlens: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    use_audio_in_video: bool = False,
    second_per_grids: Optional[torch.Tensor] = None,
    position_id_per_seconds: int = 1,
    packed_seq_params: Optional[PackedSeqParams] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate RoPE position indices for multimodal inputs.

    Computes rotary position embeddings (RoPE) indices for a sequence containing
    mixed modalities (text, images, videos, audio). Handles temporal, spatial,
    and audio-specific position encoding.

    Args:
        spatial_merge_size: Size of spatial merge for grid downsampling.
        image_token_id: Token ID for image markers.
        video_token_id: Token ID for video markers.
        audio_token_id: Token ID for audio markers.
        vision_start_token_id: Token ID marking start of vision content.
        audio_start_token_id: Token ID marking start of audio content.
        input_ids: Input token IDs of shape [batch_size, seq_len].
        image_grid_thw: Image grid dimensions [num_images, 3] with (T, H, W).
        video_grid_thw: Video grid dimensions [num_videos, 3] with (T, H, W).
        audio_seqlens: Audio sequence lengths [num_audios].
        attention_mask: Attention mask indicating valid tokens.
        use_audio_in_video: Whether audio is embedded within video tokens.
        second_per_grids: Seconds per video grid frame.
        position_id_per_seconds: Position ID increment per second.
        packed_seq_params: Packed sequence parameters for variable-length sequences.

    Returns:
        tuple: (position_ids, mrope_position_deltas) where:
            - position_ids: Shape [3, batch_size, seq_len] with temporal, height, width indices.
            - mrope_position_deltas: Shape [batch_size, 1] with position delta adjustments.
    """
    # VL timestamp split logic (unchanged)
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    if packed_seq_params is not None and attention_mask is None and input_ids is not None:
        # Build an attention mask from packed sequence metadata when one is not provided.
        # cu_seqlens_q entries are cumulative lengths; their diffs give per-sample lengths.
        cu_seqlens = packed_seq_params.cu_seqlens_q
        if cu_seqlens is not None and cu_seqlens.numel() >= 2:
            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            attention_mask = torch.zeros_like(input_ids, dtype=input_ids.dtype)
            max_len = attention_mask.shape[1]
            for i, seq_len in enumerate(seq_lens.tolist()):
                valid = min(int(seq_len), max_len)
                attention_mask[i, :valid] = 1
        else:
            # Fallback to a dense mask if packed metadata is missing.
            attention_mask = torch.ones_like(input_ids)

    mrope_position_deltas = []
    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None or audio_seqlens is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=torch.float,
            device=input_ids.device,
        )
        image_index, video_index, audio_index = 0, 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]

            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            audio_nums = torch.sum(input_ids == audio_start_token_id)
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (
                (vision_tokens == audio_start_token_id).sum()
                if use_audio_in_video
                else (vision_tokens == video_token_id).sum()
            )
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums
            multimodal_nums = image_nums + audio_nums if use_audio_in_video else image_nums + video_nums + audio_nums

            for _ in range(multimodal_nums):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                if (image_token_id in input_tokens or video_token_id in input_tokens) and (
                    remain_videos > 0 or remain_images > 0
                ):
                    ed_vision_start = input_tokens.index(vision_start_token_id, st)
                else:
                    ed_vision_start = len(input_tokens) + 1
                if audio_token_id in input_tokens and remain_audios > 0:
                    ed_audio_start = input_tokens.index(audio_start_token_id, st)
                else:
                    ed_audio_start = len(input_tokens) + 1
                min_ed = min(ed_vision_start, ed_audio_start)

                # ---------- text ----------
                text_len = min_ed - st
                if text_len > 0:
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                    st_idx += text_len

                # ---------- BOS ----------
                # Audio in Video
                if min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                    bos_len, eos_len = 2, 2
                else:
                    bos_len, eos_len = 1, 1

                llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)
                st_idx += bos_len

                # Audio Only
                if min_ed == ed_audio_start:
                    audio_len = _get_feat_extract_output_lengths(audio_seqlens[audio_index])
                    llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                    llm_pos_ids_list.append(llm_pos_ids)

                    st += text_len + bos_len + audio_len + eos_len
                    audio_index += 1
                    remain_audios -= 1

                # Image Only
                elif min_ed == ed_vision_start and input_ids[ed_vision_start + 1] == image_token_id:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )

                    t_index = (torch.arange(t) * 1 * position_id_per_seconds).float()

                    llm_pos_ids_list_temp = []
                    llm_grid_h = h // spatial_merge_size
                    llm_grid_w = w // spatial_merge_size
                    h_index = (
                        torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten().float()
                    )
                    w_index = (
                        torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten().float()
                    )
                    t_index = torch.Tensor(t_index).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().float()
                    _llm_pos_ids = torch.stack([t_index, h_index, w_index])
                    llm_pos_ids_list_temp.append(_llm_pos_ids + st_idx)
                    llm_pos_ids = torch.cat(llm_pos_ids_list_temp, dim=1)

                    llm_pos_ids_list.append(llm_pos_ids)

                    image_len = image_grid_thw[image_index].prod() // (spatial_merge_size**2)
                    st += int(text_len + bos_len + image_len + eos_len)
                    image_index += 1
                    remain_images -= 1

                # Video Only
                elif min_ed == ed_vision_start and input_ids[ed_vision_start + 1] == video_token_id:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    t_index = (
                        torch.arange(t) * second_per_grids[video_index].cpu().float() * position_id_per_seconds
                    ).float()

                    llm_pos_ids_list_temp = []
                    llm_grid_h = h // spatial_merge_size
                    llm_grid_w = w // spatial_merge_size
                    h_index = (
                        torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten().float()
                    )
                    w_index = (
                        torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten().float()
                    )
                    t_index = torch.Tensor(t_index).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().float()
                    _llm_pos_ids = torch.stack([t_index, h_index, w_index])
                    llm_pos_ids_list_temp.append(_llm_pos_ids + st_idx)
                    llm_pos_ids = torch.cat(llm_pos_ids_list_temp, dim=1)

                    llm_pos_ids_list.append(llm_pos_ids)

                    video_len = video_grid_thw[video_index].prod() // (spatial_merge_size**2)
                    st += int(text_len + bos_len + video_len + eos_len)
                    video_index += 1
                    remain_videos -= 1

                # Audio in Video
                elif min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                    audio_len = _get_feat_extract_output_lengths(audio_seqlens[audio_index])
                    audio_llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx

                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )

                    t_index = (
                        torch.arange(t) * second_per_grids[video_index].cpu().float() * position_id_per_seconds
                    ).float()

                    llm_pos_ids_list_temp = []
                    llm_grid_h = h // spatial_merge_size
                    llm_grid_w = w // spatial_merge_size
                    h_index = (
                        torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten().float()
                    )
                    w_index = (
                        torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten().float()
                    )
                    t_index = torch.Tensor(t_index).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().float()
                    _llm_pos_ids = torch.stack([t_index, h_index, w_index])
                    llm_pos_ids_list_temp.append(_llm_pos_ids + st_idx)
                    llm_pos_ids = torch.cat(llm_pos_ids_list_temp, dim=1)

                    video_llm_pos_ids = llm_pos_ids

                    video_data_index, audio_data_index = 0, 0
                    while (
                        video_data_index < video_llm_pos_ids.shape[-1]
                        and audio_data_index < audio_llm_pos_ids.shape[-1]
                    ):
                        if video_llm_pos_ids[0][video_data_index] <= audio_llm_pos_ids[0][audio_data_index]:
                            llm_pos_ids_list.append(video_llm_pos_ids[:, video_data_index : video_data_index + 1])
                            video_data_index += 1
                        else:
                            llm_pos_ids_list.append(audio_llm_pos_ids[:, audio_data_index : audio_data_index + 1])
                            audio_data_index += 1
                    if video_data_index < video_llm_pos_ids.shape[-1]:
                        llm_pos_ids_list.append(video_llm_pos_ids[:, video_data_index : video_llm_pos_ids.shape[-1]])
                    if audio_data_index < audio_llm_pos_ids.shape[-1]:
                        llm_pos_ids_list.append(audio_llm_pos_ids[:, audio_data_index : audio_llm_pos_ids.shape[-1]])
                    video_len = video_grid_thw[video_index].prod() // (spatial_merge_size**2)

                    st += int(text_len + bos_len + audio_len + video_len + eos_len)
                    audio_index += 1
                    video_index += 1
                    remain_videos -= 1
                    remain_audios -= 1
                else:
                    raise (RuntimeError("unexpected error"))

                # ---------- EOS ----------
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

            # tail text
            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat([item.float() for item in llm_pos_ids_list], dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        # fallback (pure text)
        # position_ids = attention_mask.float().cumsum(-1) - 1
        # position_ids.masked_fill_(attention_mask == 0, 1)
        # position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        # max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        # mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)

        if attention_mask is not None:
            position_ids = attention_mask.float().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas
