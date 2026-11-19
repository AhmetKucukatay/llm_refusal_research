"""Generate Arditi candidate refusal directions on Qwen."""

#we'll generate [5, 24, 2048] candidate directions -> 5 * 24 = 120 candidate with 2048 hidden size each

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reproductions.arditi_2024.data import sample_direction_datasets
from reproductions.arditi_2024.directions import compute_candidate_directions
from reproductions.arditi_2024.qwen import (
    QWEN_REFUSAL_TOKEN_IDS,
    configure_qwen_tokenizer,
    get_qwen_post_instruction_token_ids,
    tokenize_qwen_instructions,
)
from reproductions.arditi_2024.refusal import (
    compute_refusal_scores,
    filter_instructions_by_refusal,
)

MODEL_PATH = Path("models/qwen-1_8b-chat")
OUTPUT_PATH = Path("artifacts/arditi_2024/qwen-1_8b-chat/candidate_directions.pt")


def main() -> None:
    """Generate and save the Qwen candidate direction bank."""
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

    harmful_train, harmless_train, _, _ = sample_direction_datasets()

    def tokenize_batch(instructions: list[str]):
        return tokenize_qwen_instructions(
            tokenizer,
            instructions,
        )

    harmful_scores = compute_refusal_scores(
        model=model,
        instructions=harmful_train,
        tokenize_instructions_fn=tokenize_batch,
        refusal_token_ids=QWEN_REFUSAL_TOKEN_IDS,
    )

    harmless_scores = compute_refusal_scores(
        model=model,
        instructions=harmless_train,
        tokenize_instructions_fn=tokenize_batch,
        refusal_token_ids=QWEN_REFUSAL_TOKEN_IDS,
    )

    harmful_train = filter_instructions_by_refusal(
        harmful_train,
        harmful_scores,
        keep_refusals=True,
    )

    harmless_train = filter_instructions_by_refusal(
        harmless_train,
        harmless_scores,
        keep_refusals=False,
    )

    print(f"Filtered harmful train: {len(harmful_train)}")
    print(f"Filtered harmless train: {len(harmless_train)}")

    suffix_ids = get_qwen_post_instruction_token_ids(tokenizer)
    positions = list(range(-len(suffix_ids), 0))

    candidate_directions = compute_candidate_directions(
        model=model,
        block_modules=model.transformer.h,
        harmful_instructions=harmful_train,
        harmless_instructions=harmless_train,
        tokenize_instructions_fn=tokenize_batch,
        positions=positions,
        batch_size=32,
        device=model.device,
    )

    expected_shape = (
        len(positions),
        model.config.num_hidden_layers,
        model.config.hidden_size,
    )

    print(f"Candidate shape: {tuple(candidate_directions.shape)}")
    print(f"Expected shape: {expected_shape}")

    assert tuple(candidate_directions.shape) == expected_shape
    assert not torch.isnan(candidate_directions).any()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(candidate_directions.cpu(), OUTPUT_PATH)

    print(f"Saved candidates to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()