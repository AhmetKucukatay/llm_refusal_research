"""Inspect the post-instruction tokens used for Qwen candidate extraction."""

from pathlib import Path

from transformers import AutoTokenizer

from reproductions.arditi_2024.qwen import (
    QWEN_POST_INSTRUCTION_SUFFIX,
    configure_qwen_tokenizer,
    format_qwen_prompt,
    get_qwen_post_instruction_token_ids,
)

MODEL_PATH = Path("models/qwen-1_8b-chat")


def main() -> None:
    """Inspect Qwen post-instruction tokenization."""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=False,
    )
    configure_qwen_tokenizer(tokenizer)

    suffix_ids = get_qwen_post_instruction_token_ids(tokenizer)

    prompt = format_qwen_prompt("Test instruction")
    prompt_ids = tokenizer.encode(prompt)

    print("Post-instruction suffix:")
    print(repr(QWEN_POST_INSTRUCTION_SUFFIX))
    print()

    print(f"Number of post-instruction tokens: {len(suffix_ids)}")
    print(f"Token IDs: {suffix_ids}")
    print()

    print("Tokens:")

    #assuming we have 5 suffik token, first suffix token -> -5.. last suffix token -> -1
    for position, token_id in zip(
        range(-len(suffix_ids), 0),   
        suffix_ids,
        strict=True,
    ):
        token_text = tokenizer.decode([token_id])
        print(f"{position}: {token_id} -> {token_text!r}")

    # Verify that the suffix matches the final tokens of the full prompt
    assert prompt_ids[-len(suffix_ids) :] == suffix_ids

    print()
    print("Suffix matches the final tokens of the full prompt.")


if __name__ == "__main__":
    main()