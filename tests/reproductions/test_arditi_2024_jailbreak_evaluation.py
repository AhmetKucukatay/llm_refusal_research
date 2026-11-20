"""Tests for Arditi jailbreak evaluation."""

from reproductions.arditi_2024.jailbreak_evaluation import (
    evaluate_substring_matching,
)


def test_substring_evaluation_distinguishes_refusal() -> None:
    """Refusal text is not counted as a jailbreak success."""
    completions = [
        {
            "category": "test",
            "prompt": "prompt 1",
            "response": "I'm sorry, but I cannot help with that.",
        },
        {
            "category": "test",
            "prompt": "prompt 2",
            "response": "Sure, here is the requested answer.",
        },
    ]

    result = evaluate_substring_matching(completions)

    assert result["substring_matching_success_rate"] == 0.5
    assert result["substring_matching_per_category"]["test"] == 0.5

    flags = [
        completion["is_jailbreak_substring_matching"]
        for completion in result["completions"]
    ]

    assert flags == [0, 1]