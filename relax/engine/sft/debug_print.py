# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pretty-print the first processed SFT sample of a step using rich.

Used by `relax.components.sft.SFT._produce_one_step` to make the chat-template
/ loss-mask pipeline visually inspectable in logs.
"""

from typing import Any

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


_BLOCK_HEAD_TOKENS = 8
_BLOCK_TAIL_TOKENS = 8


def _decode_one(tokenizer, token_id: int) -> str:
    s = tokenizer.decode([token_id], skip_special_tokens=False)
    return s.replace("\n", "\\n").replace("\t", "\\t")


def _iter_label_blocks(loss_mask: torch.Tensor) -> list[tuple[int, int, int]]:
    """Yield (start, end_exclusive, label) for each maximal run of equal
    labels."""
    n = int(loss_mask.shape[0])
    blocks: list[tuple[int, int, int]] = []
    if n == 0:
        return blocks
    start = 0
    cur = int(loss_mask[0].item())
    for i in range(1, n):
        v = int(loss_mask[i].item())
        if v != cur:
            blocks.append((start, i, cur))
            start = i
            cur = v
    blocks.append((start, n, cur))
    return blocks


def _token_table(tokenizer, input_ids: torch.Tensor, loss_mask: torch.Tensor) -> Table:
    n = int(input_ids.shape[0])
    table = Table(title=f"Tokens (total={n})", show_lines=False, expand=False)
    table.add_column("idx", justify="right", style="dim")
    table.add_column("id", justify="right")
    table.add_column("token", overflow="fold")
    table.add_column("learn", justify="center")

    def add_row(i: int) -> None:
        tok_id = int(input_ids[i].item())
        learn = int(loss_mask[i].item())
        learn_cell = Text("1", style="bold green") if learn else Text("0", style="dim")
        tok_cell = Text(_decode_one(tokenizer, tok_id), style="green" if learn else "white")
        table.add_row(str(i), str(tok_id), tok_cell, learn_cell)

    for start, end, _ in _iter_label_blocks(loss_mask):
        block_len = end - start
        if block_len <= _BLOCK_HEAD_TOKENS + _BLOCK_TAIL_TOKENS:
            for i in range(start, end):
                add_row(i)
        else:
            for i in range(start, start + _BLOCK_HEAD_TOKENS):
                add_row(i)
            table.add_row(
                "…",
                "…",
                Text(f"(truncated {block_len - _BLOCK_HEAD_TOKENS - _BLOCK_TAIL_TOKENS} tokens)", style="dim italic"),
                "…",
            )
            for i in range(end - _BLOCK_TAIL_TOKENS, end):
                add_row(i)
    return table


def _mm_summary(mm_inputs: dict[str, Any] | None) -> str:
    if not mm_inputs:
        return "[dim]none[/dim]"
    parts = []
    for k, v in mm_inputs.items():
        if isinstance(v, torch.Tensor):
            parts.append(f"{k}: tensor{tuple(v.shape)} {v.dtype}")
        elif isinstance(v, list):
            parts.append(f"{k}: list(len={len(v)})")
        else:
            parts.append(f"{k}: {type(v).__name__}")
    return "\n".join(parts)


def print_first_sample(
    *,
    step: int,
    sample_idx: int,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    multimodal_train_inputs: dict[str, Any] | None,
    tokenizer,
) -> None:
    """Render the first sample of an SFT step to stderr via rich.

    Output: a panel containing the decoded full text, a head/tail token table
    annotated with `learn` flags, and a multimodal summary.
    """
    console = Console(stderr=True, force_terminal=False, soft_wrap=True)

    n = int(input_ids.shape[0])
    n_learn = int(loss_mask.sum().item())
    full_text = tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)

    header = (
        f"[bold]SFT step={step}[/bold]  sample_idx={sample_idx}  "
        f"len={n}  learn_tokens={n_learn} ({(n_learn / max(n, 1)) * 100:.1f}%)"
    )
    mm_panel = Panel.fit(_mm_summary(multimodal_train_inputs), title="Multimodal", border_style="magenta")

    console.rule(header, style="bold blue")
    console.rule("Rendered text", style="cyan", align="left")
    console.print(Text(full_text))
    console.print(_token_table(tokenizer, input_ids, loss_mask))
    console.print(mm_panel)
    console.rule(style="dim")
