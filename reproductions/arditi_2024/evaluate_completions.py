"""Evaluate Arditi JailbreakBench completions."""

import json
from pathlib import Path

from reproductions.arditi_2024.jailbreak_evaluation import (
    evaluate_substring_matching,
)

COMPLETION_DIR = Path(
    "artifacts/arditi_2024/qwen-1_8b-chat/completions"
)

CONDITIONS = (
    "baseline",
    "ablation",
    "negative_addition",
)


def load_completions(condition: str) -> list[dict]:
    """Load saved JailbreakBench completions for one condition."""
    path = COMPLETION_DIR / f"jailbreakbench_{condition}.json"

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    """Evaluate all intervention conditions."""
    results = {}

    for condition in CONDITIONS:
        completions = load_completions(condition)

        evaluation = evaluate_substring_matching(
            completions
        )

        results[condition] = evaluation

        success_rate = evaluation[
            "substring_matching_success_rate"
        ]

        print(
            f"{condition:>18}: "
            f"{success_rate:.3f}"
        )

    output_path = (
        COMPLETION_DIR
        / "jailbreakbench_substring_evaluation.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(f"Saved evaluation to {output_path}")


if __name__ == "__main__":
    main()