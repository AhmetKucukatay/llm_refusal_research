"""Loss-evaluation utilities for the Arditi et al. reproduction."""

import itertools
import json
from collections.abc import Iterator
from pathlib import Path

import torch
from datasets import load_dataset

from refusal_toolkit.evaluation import compute_masked_next_token_nll
from refusal_toolkit.interventions import temporary_hooks
from reproductions.arditi_2024.qwen import (
    QWEN_POST_INSTRUCTION_SUFFIX,
    format_qwen_prompt,
)


def tokenize_qwen_with_outputs(
    tokenizer,
    instructions: list[str],
    outputs: list[str],
) -> dict[str, torch.Tensor]:
    """Tokenize Qwen chat prompts together with target assistant outputs."""
    prompts = [
        format_qwen_prompt(instruction) + output
        for instruction, output in zip(
            instructions,
            outputs,
            strict=True,
        )
    ]

    return tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )


def build_completion_loss_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    eoi_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Mask loss so that only assistant-completion tokens are evaluated."""
    loss_mask = attention_mask.clone()

    # The final position cannot predict another token.
    loss_mask[:, -1] = 0

    for batch_index in range(input_ids.shape[0]):
        for position in range(input_ids.shape[1]):
            candidate = input_ids[
                batch_index,
                position : position + eoi_token_ids.shape[0],
            ]

            if candidate.shape[0] != eoi_token_ids.shape[0]:
                continue

            if torch.all(candidate == eoi_token_ids):
                # Keep the final suffix position because its logits predict
                # the first token of the assistant completion.
                loss_mask[
                    batch_index,
                    : position + eoi_token_ids.shape[0] - 1,
                ] = 0
                break

    return loss_mask


def iter_chat_completion_batches(
    *,
    tokenizer,
    instructions: list[str],
    outputs: list[str],
    batch_size: int,
    eoi_token_ids: torch.Tensor,
) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
    """Yield Qwen chat batches and corresponding completion-loss masks."""
    instruction_iterator = iter(instructions)
    output_iterator = iter(outputs)

    while True:
        instruction_batch = list(
            itertools.islice(instruction_iterator, batch_size)
        )
        output_batch = list(
            itertools.islice(output_iterator, batch_size)
        )

        if not instruction_batch or not output_batch:
            break

        inputs = tokenize_qwen_with_outputs(
            tokenizer,
            instruction_batch,
            output_batch,
        )

        loss_mask = build_completion_loss_mask(
            inputs["input_ids"],
            inputs["attention_mask"],
            eoi_token_ids,
        )

        yield inputs, loss_mask


def iter_custom_completion_batches(
    *,
    path: Path,
    tokenizer,
    batch_size: int,
    eoi_token_ids: torch.Tensor,
) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
    """Yield batches from previously generated completion artifacts."""
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    instructions = [record["prompt"] for record in records]
    outputs = [record["response"] for record in records]

    yield from iter_chat_completion_batches(
        tokenizer=tokenizer,
        instructions=instructions,
        outputs=outputs,
        batch_size=batch_size,
        eoi_token_ids=eoi_token_ids,
    )


def iter_alpaca_batches(
    *,
    tokenizer,
    batch_size: int,
    eoi_token_ids: torch.Tensor,
) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
    """Yield shuffled Alpaca instruction-output batches."""
    dataset = load_dataset(
        "tatsu-lab/alpaca",
        split="train",
    )
    dataset = dataset.shuffle(seed=42)

    instructions = []
    outputs = []

    for record in dataset:
        if record["input"].strip() == "":
            instructions.append(record["instruction"])
            outputs.append(record["output"])

    yield from iter_chat_completion_batches(
        tokenizer=tokenizer,
        instructions=instructions,
        outputs=outputs,
        batch_size=batch_size,
        eoi_token_ids=eoi_token_ids,
    )


def iter_pile_batches(
    *,
    tokenizer,
    batch_size: int,
    max_length: int = 256,
) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
    """Yield streaming Pile batches."""
    dataset = load_dataset(
        "monology/pile-uncopyrighted",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    dataset_iterator = iter(dataset)

    while True:
        batch = list(
            itertools.islice(dataset_iterator, batch_size)
        )

        if not batch:
            break

        inputs = tokenizer(
            [record["text"] for record in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        loss_mask = inputs["attention_mask"].clone()
        loss_mask[:, -1] = 0

        yield inputs, loss_mask


@torch.no_grad()
def compute_loss_over_batches(
    *,
    model,
    batches,
    n_batches: int,
    pre_hooks=(),
    output_hooks=(),
) -> dict[str, float | int]:
    """Compute token-weighted CE loss and perplexity over batches."""
    total_nll = torch.tensor(
        0.0,
        dtype=torch.float64,
        device=model.device,
    )
    total_tokens = torch.tensor(
        0,
        dtype=torch.int64,
        device=model.device,
    )

    for batch_index, (inputs, loss_mask) in enumerate(batches):
        if n_batches != -1 and batch_index >= n_batches:
            break

        inputs = {
            key: value.to(model.device)
            for key, value in inputs.items()
        }
        loss_mask = loss_mask.to(model.device)

        with temporary_hooks(
            pre_hooks=pre_hooks,
            output_hooks=output_hooks,
        ):
            logits = model(**inputs).logits

        batch_nll, batch_tokens = compute_masked_next_token_nll(
            logits,
            inputs["input_ids"],
            loss_mask,
        )

        total_nll += batch_nll.to(torch.float64)
        total_tokens += batch_tokens

    ce_loss = total_nll / total_tokens
    perplexity = torch.exp(ce_loss)

    return {
        "ce_loss": ce_loss.item(),
        "perplexity": perplexity.item(),
        "n_tokens": total_tokens.item(),
    }


def get_qwen_eoi_token_ids(tokenizer) -> torch.Tensor:
    """Return token IDs marking the end of the Qwen instruction region."""
    return torch.tensor(
        tokenizer.encode(QWEN_POST_INSTRUCTION_SUFFIX),
        dtype=torch.long,
    )