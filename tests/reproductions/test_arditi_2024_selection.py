"""Tests for Arditi candidate-direction evaluation."""

from collections.abc import Sequence
from types import SimpleNamespace

import torch
from torch import nn

from refusal_toolkit.evaluation import collect_last_token_logits
from reproductions.arditi_2024.selection import (
    CandidateEvaluation,
    evaluate_candidate,
    should_discard_candidate,
)


class ToyBlock(nn.Module):
    """Minimal Qwen-like transformer block."""

    def __init__(self) -> None:
        """Initialize attention and MLP intervention points."""
        super().__init__()
        self.attn = nn.Identity()
        self.mlp = nn.Identity()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Pass hidden states through the toy attention and MLP."""
        hidden = self.attn(hidden)
        return self.mlp(hidden)


class ToyTransformer(nn.Module):
    """Minimal container exposing Qwen's transformer.h structure."""

    def __init__(self) -> None:
        """Initialize two toy transformer blocks."""
        super().__init__()
        self.h = nn.ModuleList([ToyBlock(), ToyBlock()])


class ToyQwen(nn.Module):
    """Minimal causal model compatible with Qwen intervention wiring."""

    def __init__(self) -> None:
        """Initialize the toy transformer."""
        super().__init__()
        self.transformer = ToyTransformer()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """Produce simple vocabulary logits."""
        del attention_mask

        values = input_ids.to(torch.float32)

        hidden = torch.stack(
            [
                values,
                torch.zeros_like(values),
            ],
            dim=-1,
        )

        for block in self.transformer.h:
            hidden = block(hidden)

        zeros = torch.zeros_like(hidden[..., :1])
        logits = torch.cat([hidden, zeros], dim=-1)

        return SimpleNamespace(logits=logits)


def test_evaluate_candidate_computes_expected_metric_behavior() -> None:
    """A useful toy direction suppresses and induces refusal as expected."""
    model = ToyQwen()

    def tokenize(instructions: Sequence[str]):
        values = torch.tensor(
            [[int(instruction)] for instruction in instructions]
        )

        return SimpleNamespace(
            input_ids=values,
            attention_mask=torch.ones_like(values),
        )

    harmless = ["0"]

    baseline_harmless_logits = collect_last_token_logits(
        model=model,
        instructions=harmless,
        tokenize_instructions_fn=tokenize,
    )

    evaluation = evaluate_candidate(
        model=model,
        harmful_instructions=["3"],
        harmless_instructions=harmless,
        tokenize_instructions_fn=tokenize,
        refusal_token_ids=[0],
        baseline_harmless_logits=baseline_harmless_logits,
        direction=torch.tensor([1.0, 0.0]),
        source_layer=0,
    )

    assert evaluation.ablation_refusal_score < 0.0
    assert evaluation.steering_refusal_score > 0.0
    assert abs(evaluation.ablation_kl_score) < 1e-10

def test_candidate_filter_matches_arditi_thresholds() -> None:
    """Selection filters reject candidates that violate Arditi thresholds."""
    good = CandidateEvaluation(
        ablation_refusal_score=-2.0,
        steering_refusal_score=1.0,
        ablation_kl_score=0.05,
    )

    high_kl = CandidateEvaluation(
        ablation_refusal_score=-2.0,
        steering_refusal_score=1.0,
        ablation_kl_score=0.2,
    )

    weak_steering = CandidateEvaluation(
        ablation_refusal_score=-2.0,
        steering_refusal_score=-0.5,
        ablation_kl_score=0.05,
    )

    assert not should_discard_candidate(
        good,
        layer=10,
        n_layers=24,
    )

    assert should_discard_candidate(
        high_kl,
        layer=10,
        n_layers=24,
    )

    assert should_discard_candidate(
        weak_steering,
        layer=10,
        n_layers=24,
    )

    assert should_discard_candidate(
        good,
        layer=19,
        n_layers=24,
    )