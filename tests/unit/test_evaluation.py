"""Unit tests for generic evaluation utilities."""

from collections.abc import Sequence
from types import SimpleNamespace

import torch
from torch import nn

from refusal_toolkit.evaluation import (
    collect_last_token_logits,
    compute_kl_divergence_from_logits,
    compute_masked_next_token_nll,
)
from refusal_toolkit.interventions import make_direction_ablation_pre_hook


def test_kl_divergence_is_zero_for_identical_logits() -> None:
    """Identical distributions have zero KL divergence."""
    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 0.0],
        ]
    )

    result = compute_kl_divergence_from_logits(
        logits,
        logits,
    )

    expected = torch.zeros(2, dtype=torch.float64)

    assert torch.allclose(
        result,
        expected,
        atol=1e-10,
    )


def test_kl_divergence_is_positive_for_different_logits() -> None:
    """Different distributions produce positive KL divergence."""
    baseline_logits = torch.tensor([[3.0, 1.0]])
    intervention_logits = torch.tensor([[1.0, 3.0]])

    result = compute_kl_divergence_from_logits(
        baseline_logits,
        intervention_logits,
    )

    assert result.shape == (1,)
    assert result.item() > 0.0

class ToyCausalLM(nn.Module):
    """Minimal causal model exposing logits and an intervention point."""

    def __init__(self) -> None:
        """Initialize the toy intervention block."""
        super().__init__()
        self.block = nn.Identity()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """Produce simple logits from the toy hidden states."""
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


def test_collect_last_token_logits_applies_temporary_hook() -> None:
    """Collected logits reflect an intervention applied during the forward."""
    model = ToyCausalLM()

    def tokenize(instructions: Sequence[str]):
        values = torch.tensor(
            [[int(instruction)] for instruction in instructions]
        )
        mask = torch.ones_like(values)

        return SimpleNamespace(
            input_ids=values,
            attention_mask=mask,
        )

    direction = torch.tensor([1.0, 0.0])
    hook = make_direction_ablation_pre_hook(direction)

    logits = collect_last_token_logits(
        model=model,
        instructions=["3", "5"],
        tokenize_instructions_fn=tokenize,
        pre_hooks=[(model.block, hook)],
    )

    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    assert torch.allclose(logits, expected)

def test_compute_masked_next_token_nll_ignores_masked_positions():
    """Masked positions should not contribute to next-token NLL."""
    input_ids = torch.tensor([[0, 1, 2]])

    logits = torch.zeros(1, 3, 3)
    logits[0, 0, 1] = 10.0
    logits[0, 1, 2] = 10.0

    loss_mask = torch.tensor([[0, 1, 0]])

    loss, n_tokens = compute_masked_next_token_nll(
        logits,
        input_ids,
        loss_mask,
    )

    assert n_tokens.item() == 1
    assert loss.item() < 0.001