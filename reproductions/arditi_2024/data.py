"""Dataset utilities for the Arditi et al. reproduction."""

import json
import random
from pathlib import Path

VALID_HARMTYPES = {"harmful", "harmless"}
VALID_SPLITS = {"train", "val", "test"}

DEFAULT_SPLIT_DIR = Path("data/arditi_2024/splits")


def load_instruction_split(
    split_dir: Path,
    harmtype: str,
    split: str,
) -> list[str]:
    """Load instruction strings from one Arditi dataset split."""
    if harmtype not in VALID_HARMTYPES:
        raise ValueError(f"Unknown harm type: {harmtype}")

    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split: {split}")

    path = split_dir / f"{harmtype}_{split}.json"

    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    return [record["instruction"] for record in records]

def sample_direction_datasets(
    split_dir: Path = DEFAULT_SPLIT_DIR,
    n_train: int = 128,
    n_val: int = 32,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Load and deterministically sample train/validation instruction sets."""
    harmful_train_all = load_instruction_split(
        split_dir,
        harmtype="harmful",
        split="train",
    )
    harmless_train_all = load_instruction_split(
        split_dir,
        harmtype="harmless",
        split="train",
    )
    harmful_val_all = load_instruction_split(
        split_dir,
        harmtype="harmful",
        split="val",
    )
    harmless_val_all = load_instruction_split(
        split_dir,
        harmtype="harmless",
        split="val",
    )

    # Match the released pipeline's seed and sampling order without
    # modifying Python's global random state.
    rng = random.Random(seed)

    harmful_train = rng.sample(harmful_train_all, n_train)
    harmless_train = rng.sample(harmless_train_all, n_train)
    harmful_val = rng.sample(harmful_val_all, n_val)
    harmless_val = rng.sample(harmless_val_all, n_val)

    return harmful_train, harmless_train, harmful_val, harmless_val