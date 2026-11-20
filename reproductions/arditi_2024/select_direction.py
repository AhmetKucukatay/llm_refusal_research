"""Select the best Arditi refusal direction on Qwen validation data."""

import json
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reproductions.arditi_2024.data import sample_direction_datasets
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
from reproductions.arditi_2024.selection import select_best_candidate

MODEL_PATH = Path("models/qwen-1_8b-chat")

CANDIDATE_PATH = Path(
    "artifacts/arditi_2024/qwen-1_8b-chat/candidate_directions.pt"
)

OUTPUT_DIR = Path(
    "artifacts/arditi_2024/qwen-1_8b-chat/selection"
)


def main() -> None:
    """Run Arditi candidate selection on the filtered validation sets."""
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

    _, _, harmful_val, harmless_val = sample_direction_datasets()

    def tokenize_batch(instructions: list[str]):
        return tokenize_qwen_instructions(
            tokenizer,
            instructions,
        )

    harmful_scores = compute_refusal_scores(
        model=model,
        instructions=harmful_val,
        tokenize_instructions_fn=tokenize_batch,
        refusal_token_ids=QWEN_REFUSAL_TOKEN_IDS,
        batch_size=32,
        device=model.device,
    )

    harmless_scores = compute_refusal_scores(
        model=model,
        instructions=harmless_val,
        tokenize_instructions_fn=tokenize_batch,
        refusal_token_ids=QWEN_REFUSAL_TOKEN_IDS,
        batch_size=32,
        device=model.device,
    )

    harmful_val = filter_instructions_by_refusal(
        harmful_val,
        harmful_scores,
        keep_refusals=True,
    )

    harmless_val = filter_instructions_by_refusal(
        harmless_val,
        harmless_scores,
        keep_refusals=False,
    )

    print(f"Filtered harmful val: {len(harmful_val)}")
    print(f"Filtered harmless val: {len(harmless_val)}")

    candidate_directions = torch.load(
        CANDIDATE_PATH,
        map_location="cpu",
    )

    suffix_ids = get_qwen_post_instruction_token_ids(tokenizer)
    positions = list(range(-len(suffix_ids), 0))

    print(f"Candidate shape: {tuple(candidate_directions.shape)}")
    print(f"Positions: {positions}")

    result = select_best_candidate(
        model=model,
        harmful_instructions=harmful_val,
        harmless_instructions=harmless_val,
        tokenize_instructions_fn=tokenize_batch,
        refusal_token_ids=QWEN_REFUSAL_TOKEN_IDS,
        candidate_directions=candidate_directions,
        positions=positions,
        batch_size=32,
        device=model.device,
        kl_threshold=0.1,
        steering_threshold=0.0,
        prune_layer_percentage=0.2,
    )

    eligible_count = sum(
        not record.discarded
        for record in result.records
    )

    print()
    print("Selected direction")
    print(f"Position: {result.position}")
    print(f"Layer: {result.layer}")
    print(
        "Ablation refusal score: "
        f"{result.evaluation.ablation_refusal_score:.6f}"
    )
    print(
        "Steering refusal score: "
        f"{result.evaluation.steering_refusal_score:.6f}"
    )
    print(
        "Ablation KL score: "
        f"{result.evaluation.ablation_kl_score:.6f}"
    )
    print(
        f"Eligible candidates: {eligible_count}/{len(result.records)}"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        result.direction.cpu(),
        OUTPUT_DIR / "selected_direction.pt",
    )

    metadata = {
        "position": result.position,
        "layer": result.layer,
        "evaluation": asdict(result.evaluation),
        "num_candidates": len(result.records),
        "num_eligible_candidates": eligible_count,
        "records": [asdict(record) for record in result.records],
    }

    with open(
        OUTPUT_DIR / "selection_result.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(f"Saved selection artifacts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()