"""Inspect baseline refusal filtering on the real Qwen model."""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reproductions.arditi_2024.data import sample_direction_datasets
from reproductions.arditi_2024.qwen import (
    QWEN_REFUSAL_TOKEN_IDS,
    configure_qwen_tokenizer,
    tokenize_qwen_instructions,
)
from reproductions.arditi_2024.refusal import (
    compute_refusal_scores,
    filter_instructions_by_refusal,
)

MODEL_PATH = Path("models/qwen-1_8b-chat")


def main() -> None:
    """Run the released Arditi filtering rule on sampled datasets."""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=False,
    )
    configure_qwen_tokenizer(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        use_flash_attn=True,
        bf16=False,
        fp16=True,
        fp32=False,
    ).eval()
    model.requires_grad_(False)

    harmful_train, harmless_train, harmful_val, harmless_val = (
        sample_direction_datasets()
    )

    def tokenize_batch(instructions: list[str]):
        return tokenize_qwen_instructions(
            tokenizer,
            instructions,
        )

    harmful_train_scores = compute_refusal_scores(
        model=model,
        instructions=harmful_train,
        tokenize_instructions_fn=tokenize_batch,
        refusal_token_ids=QWEN_REFUSAL_TOKEN_IDS,
    )

    harmless_train_scores = compute_refusal_scores(
        model=model,
        instructions=harmless_train,
        tokenize_instructions_fn=tokenize_batch,
        refusal_token_ids=QWEN_REFUSAL_TOKEN_IDS,
    )

    harmful_val_scores = compute_refusal_scores(
        model=model,
        instructions=harmful_val,
        tokenize_instructions_fn=tokenize_batch,
        refusal_token_ids=QWEN_REFUSAL_TOKEN_IDS,
    )

    harmless_val_scores = compute_refusal_scores(
        model=model,
        instructions=harmless_val,
        tokenize_instructions_fn=tokenize_batch,
        refusal_token_ids=QWEN_REFUSAL_TOKEN_IDS,
    )

    harmful_train_filtered = filter_instructions_by_refusal(
        harmful_train,
        harmful_train_scores,
        keep_refusals=True,
    )
    harmless_train_filtered = filter_instructions_by_refusal(
        harmless_train,
        harmless_train_scores,
        keep_refusals=False,
    )
    harmful_val_filtered = filter_instructions_by_refusal(
        harmful_val,
        harmful_val_scores,
        keep_refusals=True,
    )
    harmless_val_filtered = filter_instructions_by_refusal(
        harmless_val,
        harmless_val_scores,
        keep_refusals=False,
    )

    print(
        f"Harmful train: {len(harmful_train)} -> "
        f"{len(harmful_train_filtered)}"
    )
    print(
        f"Harmless train: {len(harmless_train)} -> "
        f"{len(harmless_train_filtered)}"
    )
    print(
        f"Harmful val: {len(harmful_val)} -> "
        f"{len(harmful_val_filtered)}"
    )
    print(
        f"Harmless val: {len(harmless_val)} -> "
        f"{len(harmless_val_filtered)}"
    )


if __name__ == "__main__":
    main()