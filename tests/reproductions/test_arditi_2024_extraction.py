"""Tests for Arditi activation extraction."""

import torch
from torch import nn

from reproductions.arditi_2024.extraction import (
    extract_resid_pre_activations,
)


class AddOneBlock(nn.Module):
    """Transformer-like block that adds one to its input."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Add one to every hidden-state value."""
        return hidden_states + 1


class ToyModel(nn.Module):
    """Minimal model with sequential transformer-like blocks."""

    def __init__(self) -> None:
        """Create two deterministic blocks."""
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                AddOneBlock(),
                AddOneBlock(),
            ]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run deterministic hidden states through both blocks."""
        del attention_mask

        hidden_states = input_ids.unsqueeze(-1).float()

        for block in self.blocks:
            hidden_states = block(hidden_states)

        return hidden_states


def test_extract_resid_pre_activations() -> None:
    """Capture block inputs rather than block outputs."""
    model = ToyModel()

    input_ids = torch.tensor(
        [
            [10, 20, 30],  #input 1
            [40, 50, 60],  #input 2
        ]
    )
    attention_mask = torch.ones_like(input_ids)

    activations = extract_resid_pre_activations(
        model=model,
        block_modules=model.blocks,
        input_ids=input_ids,
        attention_mask=attention_mask,
        positions=[-2, -1],
    )
    # Note: output order is [batch, position, layer, hidden_dim],
    # so values from different layers are grouped under each token position.
    expected = torch.tensor( # Output order is [batch, position, layer, hidden_dim].
        [
            [
                [[20.0], [21.0]],   # position -2 across layers 0 and 1
                [[30.0], [31.0]],   #  # position -1 across layers 0 and 1
            ],
            [
                [[50.0], [51.0]],
                [[60.0], [61.0]],
            ],
        ]
    )

    assert activations.shape == (2, 2, 2, 1)
    assert torch.equal(activations, expected)