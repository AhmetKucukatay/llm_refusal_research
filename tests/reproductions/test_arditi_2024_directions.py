"""Tests for Arditi candidate direction construction."""

import torch
from torch import nn

from reproductions.arditi_2024.directions import compute_candidate_directions


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
        self.blocks = nn.ModuleList([AddOneBlock(), AddOneBlock()])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run hidden states through both blocks."""
        del attention_mask

        hidden_states = input_ids.unsqueeze(-1).float()

        for block in self.blocks:
            hidden_states = block(hidden_states)

        return hidden_states


def test_compute_candidate_directions() -> None:
    """Compute harmful-minus-harmless directions."""
    model = ToyModel()

    harmful_instructions = ["50", "70"]  #50 + 70 /2 = 60 
    harmless_instructions = ["10", "30"] #mean 20

    def tokenize_instructions_fn(batch_instructions: list[str]):
        input_ids = torch.tensor(
            [
                [0, int(instruction)]
                for instruction in batch_instructions
            ]
        )

        return type(
            "TokenizedBatch",
            (),
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            },
        )()

    directions = compute_candidate_directions(
        model=model,
        block_modules=model.blocks,
        harmful_instructions=harmful_instructions,
        harmless_instructions=harmless_instructions,
        tokenize_instructions_fn=tokenize_instructions_fn,
        positions=[-1],
        batch_size=1,
    )

    expected = torch.tensor(
        [
            [
                [40.0],   #for layer 0: (60 - 20) = 40
                [40.0],   #for layer 1: (61 - 21) = 40
            ]
        ],
        dtype=torch.float64,
    )

    assert directions.shape == (1, 2, 1)
    assert torch.equal(directions, expected)