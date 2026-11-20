"""Run loss preservation evaluation for the Arditi reproduction."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reproductions.arditi_2024.interventions import (
    get_qwen_ablation_hooks,
    get_qwen_addition_hooks,
)
from reproductions.arditi_2024.loss_evaluation import (
    compute_loss_over_batches,
    get_qwen_eoi_token_ids,
    iter_alpaca_batches,
    iter_custom_completion_batches,
    iter_pile_batches,
)
from reproductions.arditi_2024.qwen import configure_qwen_tokenizer

MODEL_PATH = Path("models/qwen-1_8b-chat")

SELECTION_DIR = Path(
    "artifacts/arditi_2024/qwen-1_8b-chat/selection"
)

COMPLETIONS_DIR = Path(
    "artifacts/arditi_2024/qwen-1_8b-chat/completions"
)

OUTPUT_DIR = Path(
    "artifacts/arditi_2024/qwen-1_8b-chat/loss_evals"
)

BATCH_SIZE = 2
N_BATCHES = 2048
MAX_SEQ_LENGTH = 256


def save_result(label: str, result: dict) -> None:
    """Save one intervention's loss-evaluation results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path = OUTPUT_DIR / f"{label}_loss_eval.json"

    with path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(f"Saved {path}")


def evaluate_condition(
    *,
    label: str,
    model,
    tokenizer,
    eoi_token_ids: torch.Tensor,
    pre_hooks=(),
    output_hooks=(),
) -> dict:
    """Evaluate one intervention condition on all released loss datasets."""
    result = {}

    print(f"\n=== {label.upper()} ===")

    print("Evaluating Pile...")
    result["pile"] = compute_loss_over_batches(
        model=model,
        batches=iter_pile_batches(
            tokenizer=tokenizer,
            batch_size=BATCH_SIZE,
            max_length=MAX_SEQ_LENGTH,
        ),
        n_batches=N_BATCHES,
        pre_hooks=pre_hooks,
        output_hooks=output_hooks,
    )
    print(result["pile"])

    print("Evaluating Alpaca...")
    result["alpaca"] = compute_loss_over_batches(
        model=model,
        batches=iter_alpaca_batches(
            tokenizer=tokenizer,
            batch_size=BATCH_SIZE,
            eoi_token_ids=eoi_token_ids,
        ),
        n_batches=N_BATCHES,
        pre_hooks=pre_hooks,
        output_hooks=output_hooks,
    )
    print(result["alpaca"])

    print("Evaluating harmless baseline completions...")
    result["alpaca_custom_completions"] = compute_loss_over_batches(
        model=model,
        batches=iter_custom_completion_batches(
            path=COMPLETIONS_DIR / "harmless_baseline.json",
            tokenizer=tokenizer,
            batch_size=BATCH_SIZE,
            eoi_token_ids=eoi_token_ids,
        ),
        n_batches=-1,
        pre_hooks=pre_hooks,
        output_hooks=output_hooks,
    )
    print(result["alpaca_custom_completions"])

    save_result(label, result)

    return result


def main() -> None:
    """Run baseline, ablation, and negative-addition loss evaluation."""
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

    direction = torch.load(
        SELECTION_DIR / "selected_direction.pt",
        map_location="cpu",
    )

    with (
        SELECTION_DIR / "selection_result.json"
    ).open("r", encoding="utf-8") as file:
        selection = json.load(file)

    selected_layer = selection["layer"]
    eoi_token_ids = get_qwen_eoi_token_ids(tokenizer)

    print(f"Selected layer: {selected_layer}")
    print(f"Loss batch size: {BATCH_SIZE}")
    print(f"Loss batches: {N_BATCHES}")

    # Released pipeline: no intervention.
    evaluate_condition(
        label="baseline",
        model=model,
        tokenizer=tokenizer,
        eoi_token_ids=eoi_token_ids,
    )

    # Released pipeline: remove the refusal direction throughout the model.
    ablation_pre_hooks, ablation_output_hooks = get_qwen_ablation_hooks(
        model=model,
        direction=direction,
    )

    evaluate_condition(
        label="ablation",
        model=model,
        tokenizer=tokenizer,
        eoi_token_ids=eoi_token_ids,
        pre_hooks=ablation_pre_hooks,
        output_hooks=ablation_output_hooks,
    )

    # Released pipeline: subtract the refusal direction at its source layer.
    addition_pre_hooks = get_qwen_addition_hooks(
        model=model,
        direction=direction,
        source_layer=selected_layer,
        coefficient=-1.0,
    )

    evaluate_condition(
        label="actadd",
        model=model,
        tokenizer=tokenizer,
        eoi_token_ids=eoi_token_ids,
        pre_hooks=addition_pre_hooks,
    )

    print("\nLoss evaluation complete.")


if __name__ == "__main__":
    main()