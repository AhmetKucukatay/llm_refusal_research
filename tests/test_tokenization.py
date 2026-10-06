"""Tests for text tokenization utilities."""

import pytest
import torch
from transformers import BatchEncoding

from refusal_toolkit.tokenization import tokenize_texts


class _FakeTokenizer:
    """Minimal tokenizer double used to test wrapper behavior."""

    def __init__(self) -> None:
        """Initialize storage for the most recent tokenizer call."""
        self.last_texts: list[str] | None = None
        self.last_kwargs: dict[str, object] | None = None

    def __call__(
        self,
        texts: list[str],
        **kwargs: object,
    ) -> BatchEncoding:
        """Record the call and return a deterministic encoded batch."""
        self.last_texts = texts
        self.last_kwargs = kwargs

        return BatchEncoding(
            {
                "input_ids": torch.tensor(
                    [
                        [1, 2, 0],
                        [3, 4, 5],
                    ]
                ),
                "attention_mask": torch.tensor(
                    [
                        [1, 1, 0],
                        [1, 1, 1],
                    ]
                ),
            }
        )


def test_tokenize_texts_forwards_expected_arguments() -> None:
    """Tokenizer options should be forwarded exactly as requested."""
    tokenizer = _FakeTokenizer()

    encoded = tokenize_texts(
        tokenizer,
        ("first prompt", "second prompt"),
        padding=True,
        truncation=True,
        max_length=128,
    )

    assert tokenizer.last_texts == [
        "first prompt",
        "second prompt",
    ]
    assert tokenizer.last_kwargs == {
        "padding": True,
        "truncation": True,
        "max_length": 128,
        "return_tensors": "pt",
    }

    assert isinstance(encoded, BatchEncoding)
    assert encoded["input_ids"].shape == (2, 3)
    assert encoded["attention_mask"].shape == (2, 3)


def test_tokenize_texts_uses_safe_defaults() -> None:
    """Default tokenization should pad without truncating."""
    tokenizer = _FakeTokenizer()

    tokenize_texts(
        tokenizer,
        ["first prompt", "second prompt"],
    )

    assert tokenizer.last_kwargs == {
        "padding": True,
        "truncation": False,
        "max_length": None,
        "return_tensors": "pt",
    }


def test_tokenize_texts_rejects_empty_input() -> None:
    """At least one input text must be provided."""
    tokenizer = _FakeTokenizer()

    with pytest.raises(ValueError, match="At least one input text"):
        tokenize_texts(tokenizer, [])


def test_tokenize_texts_rejects_non_string_input() -> None:
    """Every input item must be a string."""
    tokenizer = _FakeTokenizer()

    with pytest.raises(TypeError, match="must be strings"):
        tokenize_texts(
            tokenizer,
            ["valid text", 42],
        )


@pytest.mark.parametrize(
    "max_length",
    [0, -1, -100],
)
def test_tokenize_texts_rejects_non_positive_max_length(
    max_length: int,
) -> None:
    """Maximum sequence length must be positive when specified."""
    tokenizer = _FakeTokenizer()

    with pytest.raises(ValueError, match="positive integer"):
        tokenize_texts(
            tokenizer,
            ["example"],
            max_length=max_length,
        )