"""Generate harmless-test completions for the Arditi reproduction."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from refusal_toolkit.generation import generate_completions
from reproductions.arditi_2024.data import sample_harmless_test_dataset
from reproductions.arditi_2024.interventions import get_qwen_addition_hooks
from reproductions.arditi_2024.qwen import (
    configure_qwen_tokenizer,
    tokenize_qwen_instructions,
)

MODEL_PATH = Path("models/qwen-1_8b-chat")

SELECTION_DIR = Path(
    "artifacts/arditi_2024/qwen-1_8b-chat/selection"
)

OUTPUT_DIR = Path(
    "artifacts/arditi_2024/qwen-1_8b-chat/completions"
)


def save_completions(
    completions: list[dict],
    filename: str,
) -> None:
    """Save completions as JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path = OUTPUT_DIR / filename

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            completions,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved {path}")


def main() -> None:
    """Generate baseline and positive-addition harmless completions."""
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

    dataset = sample_harmless_test_dataset()

    direction = torch.load(
        SELECTION_DIR / "selected_direction.pt",
        map_location="cpu",
    )

    with (
        SELECTION_DIR / "selection_result.json"
    ).open("r", encoding="utf-8") as file:
        selection_metadata = json.load(file)

    selected_layer = selection_metadata["layer"]

    def tokenize_batch(instructions: list[str]):
        return tokenize_qwen_instructions(
            tokenizer,
            instructions,
        )

    print(f"Harmless test size: {len(dataset)}")
    print(f"Selected layer: {selected_layer}")

    print("Generating harmless baseline completions...")

    baseline = generate_completions(
        model=model,
        tokenizer=tokenizer,
        records=dataset,
        tokenize_instructions_fn=tokenize_batch,
        batch_size=8,
        max_new_tokens=512,
        device=model.device,
    )

    save_completions(
        baseline,
        "harmless_baseline.json",
    )

    print("Generating harmless positive-addition completions...")

    addition_hooks = get_qwen_addition_hooks(
        model=model,
        direction=direction,
        source_layer=selected_layer,
        coefficient=1.0,
    )

    actadd = generate_completions(
        model=model,
        tokenizer=tokenizer,
        records=dataset,
        tokenize_instructions_fn=tokenize_batch,
        pre_hooks=addition_hooks,
        batch_size=8,
        max_new_tokens=512,
        device=model.device,
    )

    save_completions(
        actadd,
        "harmless_positive_addition.json",
    )

    print("Harmless generation complete.")


if __name__ == "__main__":
    main()