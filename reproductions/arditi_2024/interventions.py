"""Arditi-style intervention hook wiring for Qwen."""

from collections.abc import Callable

import torch
from torch import nn

from refusal_toolkit.interventions import (
    make_activation_addition_pre_hook,
    make_direction_ablation_output_hook,
    make_direction_ablation_pre_hook,
)

Hook = Callable[..., object]
HookSpec = tuple[nn.Module, Hook]


def get_qwen_ablation_hooks(
    model: nn.Module,
    direction: torch.Tensor,
) -> tuple[list[HookSpec], list[HookSpec]]:
    """Build Qwen hooks for Arditi directional ablation."""
    blocks = model.transformer.h

    pre_hooks = [
        (block, make_direction_ablation_pre_hook(direction))
        for block in blocks
    ]

    output_hooks = [
        (block.attn, make_direction_ablation_output_hook(direction))
        for block in blocks
    ]

    output_hooks += [
        (block.mlp, make_direction_ablation_output_hook(direction))
        for block in blocks
    ]

    return pre_hooks, output_hooks


def get_qwen_addition_hooks(
    model: nn.Module,
    direction: torch.Tensor,
    source_layer: int,
    coefficient: float = 1.0,
) -> list[HookSpec]:
    """Build the Qwen hook for activation addition at one source layer."""
    block = model.transformer.h[source_layer]

    return [
        (
            block,
            make_activation_addition_pre_hook(
                direction,
                coefficient=coefficient,
            ),
        )
    ]