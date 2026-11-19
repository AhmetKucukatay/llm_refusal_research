"""Tests for Arditi dataset loading and sampling."""

import json
from pathlib import Path

from reproductions.arditi_2024.data import (
    load_instruction_split,
    sample_direction_datasets,
)


def write_split(
    split_dir: Path,
    filename: str,
    instructions: list[str],
) -> None:
    """Write a minimal Arditi-style split file."""
    records = [
        {
            "instruction": instruction,
            "category": "test",
        }
        for instruction in instructions
    ]

    path = split_dir / filename
    path.write_text(json.dumps(records), encoding="utf-8")


def test_load_instruction_split(tmp_path: Path) -> None:
    """Load only instruction strings from a split file."""
    write_split(
        tmp_path,
        "harmful_train.json",
        ["first", "second"],
    )

    instructions = load_instruction_split(
        tmp_path,
        harmtype="harmful",
        split="train",
    )

    assert instructions == ["first", "second"]


def test_sample_direction_datasets_is_deterministic(tmp_path: Path) -> None:
    """Produce the same samples when using the same random seed."""
    for harmtype in ("harmful", "harmless"):
        write_split(
            tmp_path,
            f"{harmtype}_train.json",
            [f"{harmtype}-train-{i}" for i in range(10)],
        )
        write_split(
            tmp_path,
            f"{harmtype}_val.json",
            [f"{harmtype}-val-{i}" for i in range(6)],
        )

    first = sample_direction_datasets(
        split_dir=tmp_path,
        n_train=4,
        n_val=2,
        seed=42,
    )

    second = sample_direction_datasets(
        split_dir=tmp_path,
        n_train=4,
        n_val=2,
        seed=42,
    )

    assert first == second

    harmful_train, harmless_train, harmful_val, harmless_val = first

    assert len(harmful_train) == 4
    assert len(harmless_train) == 4
    assert len(harmful_val) == 2
    assert len(harmless_val) == 2