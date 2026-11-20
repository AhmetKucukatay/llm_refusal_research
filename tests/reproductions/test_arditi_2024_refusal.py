"""Tests for Arditi refusal scoring and filtering."""

import math
from collections.abc import Sequence
from types import SimpleNamespace

import torch
from torch import nn

from refusal_toolkit.interventions import make_direction_ablation_pre_hook
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

class ToyInterventionRefusalModel(nn.Module):
    """Toy model whose refusal logits can be changed by a hook."""

    def __init__(self) -> None:
        """Initialize the intervention point."""
        super().__init__()
        self.block = nn.Identity()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """Produce logits from a hookable hidden representation."""
        del attention_mask

        values = input_ids.to(torch.float32)

        hidden = torch.stack(
            [
                values,
                torch.zeros_like(values),
            ],
            dim=-1,
        )

        hidden = self.block(hidden)

        zeros = torch.zeros_like(hidden[..., :1])

        logits = torch.cat(
            [
                hidden,
                zeros,
            ],
            dim=-1,
        )

        return SimpleNamespace(logits=logits)

def test_refusal_score_changes_under_ablation_hook() -> None:
    """Ablating the refusal dimension lowers the refusal score."""
    model = ToyInterventionRefusalModel()

    def tokenize(instructions: Sequence[str]):
        values = torch.tensor(
            [[int(instruction)] for instruction in instructions]
        )

        return SimpleNamespace(
            input_ids=values,
            attention_mask=torch.ones_like(values),
        )

    baseline_scores = compute_refusal_scores(
        model=model,
        instructions=["3"],
        tokenize_instructions_fn=tokenize,
        refusal_token_ids=[0],
    )

    direction = torch.tensor([1.0, 0.0])
    hook = make_direction_ablation_pre_hook(direction)

    ablated_scores = compute_refusal_scores(
        model=model,
        instructions=["3"],
        tokenize_instructions_fn=tokenize,
        refusal_token_ids=[0],
        pre_hooks=[(model.block, hook)],
    )

    assert baseline_scores.item() > 0.0
    assert ablated_scores.item() < 0.0
    assert ablated_scores.item() < baseline_scores.item()