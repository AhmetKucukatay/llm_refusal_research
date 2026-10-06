"""Utilities for converting text into model-ready token batches."""

from collections.abc import Sequence

from transformers import BatchEncoding, PreTrainedTokenizerBase


def tokenize_texts(
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],  #gem ["a", "b"] hem ("a","b") kabul etmesi için
    *,
    padding: bool | str = True,
    truncation: bool = False,
    max_length: int | None = None,
) -> BatchEncoding:
    """Tokenize a batch of texts into PyTorch tensors.

    Args:
        tokenizer: Hugging Face tokenizer used to encode the input texts.
        texts: Sequence of input strings to tokenize.
        padding: Padding strategy passed to the tokenizer.
        truncation: Whether sequences longer than the allowed length should
            be truncated.
        max_length: Maximum sequence length used when truncation or fixed
            padding requires an explicit limit.

    Returns:
        A Hugging Face ``BatchEncoding`` containing PyTorch tensors such as
        ``input_ids`` and ``attention_mask``.

    Raises:
        ValueError: If no input texts are provided.
        TypeError: If any input item is not a string.
        ValueError: If ``max_length`` is provided with a non-positive value.
    """
    if len(texts) == 0:
        raise ValueError("At least one input text must be provided.")

    if any(not isinstance(text, str) for text in texts):
        raise TypeError("All input texts must be strings.")

    if max_length is not None and max_length <= 0:
        raise ValueError("max_length must be a positive integer.")

    return tokenizer(
        list(texts),
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        return_tensors="pt",
    )