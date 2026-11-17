"""Check whether the original Arditi Qwen checkpoint loads in this environment."""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = Path("models/qwen-1_8b-chat")


def main() -> None:
    """Load the original Qwen checkpoint and report basic compatibility."""
    print(f"Loading tokenizer from: {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=False,
    )

    print("Tokenizer loaded successfully.")

    print(f"Loading model from: {MODEL_PATH}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        bf16=False,
        fp16=True,
        fp32=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    print("Model loaded successfully.")
    print(f"Device: {device}")
    print(f"Model class: {type(model).__name__}")
    print(f"Tokenizer class: {type(tokenizer).__name__}")
    print(f"Model dtype: {next(model.parameters()).dtype}")


if __name__ == "__main__":
    main()