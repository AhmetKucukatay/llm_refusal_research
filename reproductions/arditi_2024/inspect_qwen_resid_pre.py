"""Sanity-check resid_pre extraction on the real Qwen model."""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reproductions.arditi_2024.extraction import (
    extract_resid_pre_activations,
)
from reproductions.arditi_2024.qwen import (
    configure_qwen_tokenizer,
    format_qwen_prompt,
    get_qwen_post_instruction_token_ids,
)

MODEL_PATH = Path("models/qwen-1_8b-chat")


def main() -> None:
    """Extract post-instruction resid_pre activations from one Qwen prompt."""
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

    prompt = format_qwen_prompt("Explain why the sky is blue.")

    inputs = tokenizer(
        [prompt],
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    suffix_ids = get_qwen_post_instruction_token_ids(tokenizer)
    positions = list(range(-len(suffix_ids), 0))

    # The released Arditi Qwen wrapper uses model.transformer.h
    # as the ordered list of transformer blocks.
    block_modules = model.transformer.h

    device = model.device

    activations = extract_resid_pre_activations(
        model=model,
        block_modules=block_modules,
        input_ids=inputs.input_ids.to(device),
        attention_mask=inputs.attention_mask.to(device),
        positions=positions,
    )

    expected_shape = ( #convention [batch, position, layer, hidden_dim]
        1,
        len(positions),
        model.config.num_hidden_layers,
        model.config.hidden_size,
    )

    print(f"Positions: {positions}")
    print(f"Number of layers: {model.config.num_hidden_layers}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Activation shape: {tuple(activations.shape)}")
    print(f"Expected shape: {expected_shape}")

    assert tuple(activations.shape) == expected_shape
    assert not torch.isnan(activations).any()

    print("Real-model resid_pre extraction passed.")


if __name__ == "__main__":
    main()