"""Generic evaluation utilities for representation interventions."""

from collections.abc import Callable, Sequence

import torch
from torch import nn

from refusal_toolkit.interventions import temporary_hooks


def compute_kl_divergence_from_logits(
    baseline_logits: torch.Tensor,
    intervention_logits: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute per-example KL divergence from baseline to intervention."""
    baseline_logits = baseline_logits.to(torch.float64)
    intervention_logits = intervention_logits.to(torch.float64)

    baseline_probs = torch.softmax(baseline_logits, dim=-1)
    intervention_probs = torch.softmax(intervention_logits, dim=-1)

    return torch.sum(
        baseline_probs
        * (
            torch.log(baseline_probs + epsilon)
            - torch.log(intervention_probs + epsilon)
        ),
        dim=-1,
    )

def collect_last_token_logits(
    model: nn.Module,
    instructions: Sequence[str],
    tokenize_instructions_fn: Callable,
    *,
    pre_hooks: Sequence[tuple[nn.Module, Callable]] = (),
    output_hooks: Sequence[tuple[nn.Module, Callable]] = (),
    batch_size: int = 32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Collect next-token logits with optional temporary interventions."""
    collected_logits = []

    for start in range(0, len(instructions), batch_size):
        batch_instructions = instructions[start : start + batch_size]
        inputs = tokenize_instructions_fn(batch_instructions)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        if device is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

        with torch.inference_mode():
            with temporary_hooks(
                pre_hooks=list(pre_hooks),
                output_hooks=list(output_hooks),
            ):
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits

        collected_logits.append(
            logits[:, -1, :].detach()
        )

    return torch.cat(collected_logits, dim=0)

def compute_masked_next_token_nll(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return summed next-token NLL and the number of evaluated tokens."""
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    next_token_log_probs = log_probs[:, :-1].gather(
        dim=-1,
        index=input_ids[:, 1:].unsqueeze(-1),
    ).squeeze(-1)

    # Position i stores the loss for predicting token i + 1.
    # The final position therefore has no next-token target.
    next_token_log_probs = torch.cat(
        [
            next_token_log_probs,
            torch.zeros(
                next_token_log_probs.shape[0],
                1,
                dtype=next_token_log_probs.dtype,
                device=next_token_log_probs.device,
            ),
        ],
        dim=-1,
    )

    mask = loss_mask.to(
        device=next_token_log_probs.device,
        dtype=next_token_log_probs.dtype,
    )

    negative_log_likelihood = -(next_token_log_probs * mask).sum()
    n_tokens = loss_mask.sum()

    return negative_log_likelihood, n_tokens