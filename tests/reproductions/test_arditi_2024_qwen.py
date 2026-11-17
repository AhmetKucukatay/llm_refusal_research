"""Tests for the Arditi Qwen reproduction conventions."""

from reproductions.arditi_2024.qwen import (
    QWEN_POST_INSTRUCTION_SUFFIX,
    configure_qwen_tokenizer,
    format_qwen_prompt,
    format_qwen_prompts,
    get_qwen_post_instruction_token_ids,
)


class FakeQwenTokenizer:
    """Minimal tokenizer stand-in for testing Qwen-specific configuration."""

    def __init__(self) -> None:
        """Minimal tokenizer stand-in for testing Qwen-specific configuration."""
        self.padding_side = "right"
        self.pad_token = None
        self.pad_token_id = None
        self.eod_id = 151643
        self.encoded_text = None

    def encode(self, text: str) -> list[int]:
        """Record the encoded text and return deterministic fake token IDs."""
        self.encoded_text = text
        return [10, 20, 30]


def test_format_qwen_prompt() -> None:
    """Format one instruction with the exact paper template."""
    prompt = format_qwen_prompt("Test instruction")

    assert prompt == (
        "<|im_start|>user\n"
        "Test instruction<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_format_qwen_prompts() -> None:
    """Format multiple instructions independently."""
    prompts = format_qwen_prompts(["First", "Second"])

    assert prompts == [
        "<|im_start|>user\nFirst<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nSecond<|im_end|>\n<|im_start|>assistant\n",
    ]


def test_configure_qwen_tokenizer() -> None:
    """Use the padding configuration from the released implementation."""
    tokenizer = FakeQwenTokenizer()

    configure_qwen_tokenizer(tokenizer)  #settings were wrong in FakeQwenTokenizer, now it should be matched with papers settings

    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token == "<|extra_0|>"
    assert tokenizer.pad_token_id == tokenizer.eod_id


def test_get_qwen_post_instruction_token_ids() -> None:
    """Encode exactly the fixed suffix after the user instruction."""
    tokenizer = FakeQwenTokenizer()

    token_ids = get_qwen_post_instruction_token_ids(tokenizer)

    assert tokenizer.encoded_text == QWEN_POST_INSTRUCTION_SUFFIX
    assert token_ids == [10, 20, 30]