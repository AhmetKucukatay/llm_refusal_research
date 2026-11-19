"""Tests for Arditi-specific Qwen intervention wiring."""

import torch
from torch import nn

from reproductions.arditi_2024.interventions import (
    get_qwen_ablation_hooks,
    get_qwen_addition_hooks,
)


class ToyBlock(nn.Module):
    """Minimal transformer-like block with attention and MLP modules."""

    def __init__(self) -> None:
        """Initialize the toy transformer block."""
        super().__init__()
        self.attn = nn.Identity()
        self.mlp = nn.Identity()


class ToyTransformer(nn.Module):
    """Minimal Qwen-like transformer container."""

    def __init__(self) -> None:
        """Minimal Qwen-like transformer container."""
        super().__init__()
        self.h = nn.ModuleList(
            [
                ToyBlock(),
                ToyBlock(),
            ]
        )


class ToyQwen(nn.Module):
    """Minimal model exposing the Qwen transformer.h structure."""

    def __init__(self) -> None:
        """Initialize the toy Qwen model."""
        super().__init__()
        self.transformer = ToyTransformer()


def test_qwen_ablation_hooks_cover_all_blocks_attn_and_mlp() -> None:
    """Ablation wiring covers block inputs plus attention and MLP outputs."""
    model = ToyQwen()
    direction = torch.tensor([1.0, 0.0])

    pre_hooks, output_hooks = get_qwen_ablation_hooks(
        model,
        direction,
    )

    assert len(pre_hooks) == 2
    assert len(output_hooks) == 4

    assert pre_hooks[0][0] is model.transformer.h[0]
    assert pre_hooks[1][0] is model.transformer.h[1]

    assert output_hooks[0][0] is model.transformer.h[0].attn
    assert output_hooks[1][0] is model.transformer.h[1].attn
    assert output_hooks[2][0] is model.transformer.h[0].mlp
    assert output_hooks[3][0] is model.transformer.h[1].mlp


def test_qwen_addition_hook_targets_only_source_layer() -> None:
    """Activation addition targets only the selected source layer."""
    model = ToyQwen()
    direction = torch.tensor([1.0, 0.0])

    hooks = get_qwen_addition_hooks(
        model,
        direction,
        source_layer=1,
    )

    assert len(hooks) == 1
    assert hooks[0][0] is model.transformer.h[1]