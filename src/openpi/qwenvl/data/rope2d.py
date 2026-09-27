from __future__ import annotations

from typing import Optional

import torch


def get_rope_index_25(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate Qwen2.5-VL RoPE positions for image/video tokens."""
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    mrope_position_deltas = []

    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index = 0
        video_index = 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for batch_idx, input_row in enumerate(total_input_ids):
            input_row = input_row[attention_mask[batch_idx] == 1]
            vision_start_indices = torch.argwhere(input_row == vision_start_token_id).squeeze(1)
            vision_tokens = input_row[vision_start_indices + 1]
            image_nums = int((vision_tokens == image_token_id).sum().item())
            video_nums = int((vision_tokens == video_token_id).sum().item())
            input_tokens = input_row.tolist()
            llm_pos_ids_list: list[torch.Tensor] = []
            start = 0
            remain_images = image_nums
            remain_videos = video_nums

            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    image_pos = input_tokens.index(image_token_id, start)
                else:
                    image_pos = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    video_pos = input_tokens.index(video_token_id, start)
                else:
                    video_pos = len(input_tokens) + 1

                if image_pos < video_pos:
                    grid_t, grid_h, grid_w = image_grid_thw[image_index]
                    second_per_grid_t = 0.0
                    image_index += 1
                    remain_images -= 1
                    end = image_pos
                else:
                    grid_t, grid_h, grid_w = video_grid_thw[video_index]
                    if second_per_grid_ts is not None:
                        second_per_grid_t = float(second_per_grid_ts[video_index].item())
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    end = video_pos

                llm_grid_t = int(grid_t.item())
                llm_grid_h = int(grid_h.item() // spatial_merge_size)
                llm_grid_w = int(grid_w.item() // spatial_merge_size)
                text_len = end - start
                base_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + base_idx)

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                time_tensor = expanded_range * second_per_grid_t * 2
                t_index = time_tensor.long().flatten()
                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + base_idx)
                start = end + llm_grid_t * llm_grid_h * llm_grid_w

            if start < len(input_tokens):
                base_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                text_len = len(input_tokens) - start
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + base_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., batch_idx, attention_mask[batch_idx] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[batch_idx]))

        deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, deltas

    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        deltas = max_position_ids + 1 - attention_mask.shape[-1]
        return position_ids, deltas

    position_ids = (
        torch.arange(input_ids.shape[1], device=input_ids.device)
        .view(1, 1, -1)
        .expand(3, input_ids.shape[0], -1)
    )
    deltas = torch.zeros([input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype)
    return position_ids, deltas


def get_rope_index_2(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Best-effort fallback for Qwen2."""
    return get_rope_index_25(*args, **kwargs)


def get_rope_index_3(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Best-effort fallback for Qwen3.

    Qwen3 is not the primary supported path in this merge, but keeping this alias
    avoids hard failures when loading older checkpoints.
    """
    return get_rope_index_25(*args, **kwargs)
