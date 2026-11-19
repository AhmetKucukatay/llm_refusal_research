"""Tests for Arditi refusal scoring and filtering."""

import math

import torch
from torch import nn

from reproductions.arditi_2024.refusal import (
    compute_refusal_log_odds,
    compute_refusal_scores,
    filter_instructions_by_refusal,
)


def test_compute_refusal_log_odds() -> None:
    """Compute refusal log-odds from the last-token distribution."""
    probabilities = torch.tensor(
        [
            [0.1, 0.6, 0.3],
            [0.7, 0.1, 0.2],
        ],
        dtype=torch.float64,
    )

    logits = torch.log(probabilities).unsqueeze(1)

    scores = compute_refusal_log_odds(
        logits=logits,
        refusal_token_ids=[1],
    )

    expected = torch.tensor(
        [
            math.log(0.6) - math.log(0.4),
            math.log(0.1) - math.log(0.9),
        ],
        dtype=torch.float64,
    )

    assert torch.allclose(scores, expected)

def test_filter_instructions_by_refusal() -> None:
    """Keep examples according to the sign of their refusal score."""
    instructions = ["a", "b", "c"]
    scores = torch.tensor([1.5, -2.0, 0.7])

    refusals = filter_instructions_by_refusal(
        instructions,
        scores,
        keep_refusals=True,
    )

    non_refusals = filter_instructions_by_refusal(
        instructions,
        scores,
        keep_refusals=False,
    )

    assert refusals == ["a", "c"]
    assert non_refusals == ["b"]

class ToyRefusalModel(nn.Module):
    """Minimal model that returns deterministic next-token logits."""

    @property
    def device(self) -> torch.device:
        """Return the toy model device."""
        return torch.device("cpu")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """Return logits derived from the final input token."""
        del attention_mask

        batch_size, sequence_length = input_ids.shape
        vocab_size = 3

        logits = torch.zeros(
            batch_size,
            sequence_length,
            vocab_size,
            dtype=torch.float64,
        )

        final_values = input_ids[:, -1]

        for index, value in enumerate(final_values):
            if value.item() > 0:
                logits[index, -1] = torch.tensor([0.0, 3.0, 0.0])
            else:
                logits[index, -1] = torch.tensor([3.0, 0.0, 0.0])

        return type("ModelOutput", (), {"logits": logits})()

def test_compute_refusal_scores() -> None:
    """Compute one refusal score per instruction across batches."""
    model = ToyRefusalModel()

    instructions = ["refuse", "comply", "refuse"]

    def tokenize_instructions_fn(batch_instructions: list[str]):
        values = [
            1 if instruction == "refuse" else -1
            for instruction in batch_instructions
        ]

        input_ids = torch.tensor(
            [
                [0, value]
                for value in values
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

    scores = compute_refusal_scores(
        model=model,
        instructions=instructions,
        tokenize_instructions_fn=tokenize_instructions_fn,
        refusal_token_ids=[1],
        batch_size=2,
    )

    assert scores.shape == (3,)
    assert scores[0] > 0
    assert scores[1] < 0
    assert scores[2] > 0