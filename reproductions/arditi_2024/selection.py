"""Candidate-direction evaluation for the Arditi reproduction."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from refusal_toolkit.evaluation import (
    collect_last_token_logits,
    compute_kl_divergence_from_logits,
)
from reproductions.arditi_2024.interventions import (
    get_qwen_ablation_hooks,
    get_qwen_addition_hooks,
)
from reproductions.arditi_2024.refusal import compute_refusal_scores


@dataclass(frozen=True)
class CandidateEvaluation:
    """Metrics used to evaluate one candidate refusal direction."""

    ablation_refusal_score: float
    steering_refusal_score: float
    ablation_kl_score: float

@dataclass(frozen=True)
class CandidateRecord:
    """Evaluation record for one position-layer candidate."""

    position_index: int
    position: int
    layer: int
    evaluation: CandidateEvaluation
    discarded: bool


@dataclass(frozen=True)
class SelectionResult:
    """Best eligible candidate and all candidate evaluations."""

    position: int
    layer: int
    direction: torch.Tensor
    evaluation: CandidateEvaluation
    records: tuple[CandidateRecord, ...]


def evaluate_candidate(
    model: nn.Module,
    harmful_instructions: Sequence[str],
    harmless_instructions: Sequence[str],
    tokenize_instructions_fn: Callable,
    refusal_token_ids: Sequence[int],
    baseline_harmless_logits: torch.Tensor,
    direction: torch.Tensor,
    source_layer: int,
    *,
    batch_size: int = 32,
    device: torch.device | str | None = None,
) -> CandidateEvaluation:
    """Evaluate one candidate direction using Arditi's three metrics."""
    ablation_pre_hooks, ablation_output_hooks = get_qwen_ablation_hooks(
        model,
        direction,
    )

    harmful_ablation_scores = compute_refusal_scores(
        model=model,
        instructions=harmful_instructions,
        tokenize_instructions_fn=tokenize_instructions_fn,
        refusal_token_ids=refusal_token_ids,
        pre_hooks=ablation_pre_hooks,
        output_hooks=ablation_output_hooks,
        batch_size=batch_size,
        device=device,
    )

    ablated_harmless_logits = collect_last_token_logits(
        model=model,
        instructions=harmless_instructions,
        tokenize_instructions_fn=tokenize_instructions_fn,
        pre_hooks=ablation_pre_hooks,
        output_hooks=ablation_output_hooks,
        batch_size=batch_size,
        device=device,
    )

    ablation_kl_scores = compute_kl_divergence_from_logits(
        baseline_harmless_logits,
        ablated_harmless_logits,
    )

    addition_pre_hooks = get_qwen_addition_hooks(
        model=model,
        direction=direction,
        source_layer=source_layer,
        coefficient=1.0,
    )

    harmless_steering_scores = compute_refusal_scores(
        model=model,
        instructions=harmless_instructions,
        tokenize_instructions_fn=tokenize_instructions_fn,
        refusal_token_ids=refusal_token_ids,
        pre_hooks=addition_pre_hooks,
        batch_size=batch_size,
        device=device,
    )

    return CandidateEvaluation(
        ablation_refusal_score=harmful_ablation_scores.mean().item(),
        steering_refusal_score=harmless_steering_scores.mean().item(),
        ablation_kl_score=ablation_kl_scores.mean().item(),
    )


def should_discard_candidate(
    evaluation: CandidateEvaluation,
    layer: int,
    n_layers: int,
    *,
    kl_threshold: float = 0.1,
    steering_threshold: float = 0.0,
    prune_layer_percentage: float = 0.2,
) -> bool:
    """Return whether a candidate fails Arditi's selection filters."""
    scores = (
        evaluation.ablation_refusal_score,
        evaluation.steering_refusal_score,
        evaluation.ablation_kl_score,
    )

    if any(math.isnan(score) for score in scores):
        return True

    first_pruned_layer = int(
        n_layers * (1.0 - prune_layer_percentage)
    )

    if layer >= first_pruned_layer:  #if it is in last 20% of layers -> discard
        return True

    if evaluation.ablation_kl_score > kl_threshold:
        return True

    if evaluation.steering_refusal_score < steering_threshold:
        return True

    return False


def select_best_candidate(
    model: nn.Module,
    harmful_instructions: Sequence[str],
    harmless_instructions: Sequence[str],
    tokenize_instructions_fn: Callable,
    refusal_token_ids: Sequence[int],
    candidate_directions: torch.Tensor,
    positions: Sequence[int],
    *,
    batch_size: int = 32,
    device: torch.device | str | None = None,
    kl_threshold: float = 0.1,
    steering_threshold: float = 0.0,
    prune_layer_percentage: float = 0.2,
) -> SelectionResult:
    """Evaluate all candidates and select the best eligible direction."""
    n_positions, n_layers, _ = candidate_directions.shape

    if len(positions) != n_positions:
        raise ValueError(
            "positions must match the first dimension of candidate_directions"
        )

    baseline_harmless_logits = collect_last_token_logits(
        model=model,
        instructions=harmless_instructions,
        tokenize_instructions_fn=tokenize_instructions_fn,
        batch_size=batch_size,
        device=device,
    )

    records = []

    for position_index, position in enumerate(positions):
        for layer in range(n_layers):
            direction = candidate_directions[
                position_index,
                layer,
            ]

            evaluation = evaluate_candidate(
                model=model,
                harmful_instructions=harmful_instructions,
                harmless_instructions=harmless_instructions,
                tokenize_instructions_fn=tokenize_instructions_fn,
                refusal_token_ids=refusal_token_ids,
                baseline_harmless_logits=baseline_harmless_logits,
                direction=direction,
                source_layer=layer,
                batch_size=batch_size,
                device=device,
            )

            discarded = should_discard_candidate(
                evaluation=evaluation,
                layer=layer,
                n_layers=n_layers,
                kl_threshold=kl_threshold,
                steering_threshold=steering_threshold,
                prune_layer_percentage=prune_layer_percentage,
            )

            records.append(
                CandidateRecord(
                    position_index=position_index,
                    position=position,
                    layer=layer,
                    evaluation=evaluation,
                    discarded=discarded,
                )
            )

    eligible_records = [
        record
        for record in records
        if not record.discarded
    ]

    if not eligible_records:
        raise RuntimeError("All candidate directions were discarded.")

    best_record = min(
        eligible_records,
        key=lambda record: record.evaluation.ablation_refusal_score,
    )

    best_direction = candidate_directions[
        best_record.position_index,
        best_record.layer,
    ]

    return SelectionResult(
        position=best_record.position,
        layer=best_record.layer,
        direction=best_direction,
        evaluation=best_record.evaluation,
        records=tuple(records),
    )