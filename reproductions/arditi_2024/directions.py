"""Candidate direction construction for the Arditi et al. reproduction."""

from collections.abc import Callable, Sequence

import torch
from torch import nn

from reproductions.arditi_2024.extraction import (
    compute_mean_resid_pre_activations,
)


def compute_candidate_directions(
    model: nn.Module,
    block_modules: Sequence[nn.Module],
    harmful_instructions: Sequence[str],
    harmless_instructions: Sequence[str],
    tokenize_instructions_fn: Callable,
    positions: Sequence[int],
    batch_size: int = 32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compute harmful-minus-harmless candidate refusal directions."""
    harmful_mean = compute_mean_resid_pre_activations(
        model=model,
        block_modules=block_modules,
        instructions=harmful_instructions,
        tokenize_instructions_fn=tokenize_instructions_fn,
        positions=positions,
        batch_size=batch_size,
        device=device
    )

    harmless_mean = compute_mean_resid_pre_activations(
        model=model,
        block_modules=block_modules,
        instructions=harmless_instructions,
        tokenize_instructions_fn=tokenize_instructions_fn,
        positions=positions,
        batch_size=batch_size,
        device=device
    )

    return harmful_mean - harmless_mean