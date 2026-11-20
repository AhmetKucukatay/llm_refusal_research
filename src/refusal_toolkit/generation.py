"""Generic text-generation utilities with optional activation interventions."""

from collections.abc import Callable, Sequence

import torch
from torch import nn
from transformers import GenerationConfig

from refusal_toolkit.interventions import temporary_hooks


def generate_completions(
    model: nn.Module,
    tokenizer,
    records: Sequence[dict],
    tokenize_instructions_fn: Callable,
    *,
    pre_hooks: Sequence[tuple[nn.Module, Callable]] = (),
    output_hooks: Sequence[tuple[nn.Module, Callable]] = (),
    batch_size: int = 8,
    max_new_tokens: int = 512,
    device: torch.device | str | None = None,
) -> list[dict]:
    """Generate deterministic completions for instruction records."""
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    completions = []

    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        instructions = [
            record["instruction"]
            for record in batch_records
        ]

        inputs = tokenize_instructions_fn(instructions)

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
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                )

        prompt_length = input_ids.shape[-1]
        new_token_ids = generated_ids[:, prompt_length:]

        for record, token_ids in zip(
            batch_records,
            new_token_ids,
            strict=True,
        ):
            completions.append(
                {
                    "category": record["category"],
                    "prompt": record["instruction"],
                    "response": tokenizer.decode(
                        token_ids,
                        skip_special_tokens=True,
                    ).strip(),
                }
            )

    return completions