"""Dataset utilities for the Arditi et al. reproduction."""

import json
import random
from pathlib import Path

VALID_HARMTYPES = {"harmful", "harmless"}
VALID_SPLITS = {"train", "val", "test"}

DEFAULT_SPLIT_DIR = Path("data/arditi_2024/splits")
DEFAULT_PROCESSED_DIR = Path("data/arditi_2024/processed") #for tes/evaluation


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

def load_processed_dataset(
    name: str,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> list[dict]:
    """Load an Arditi processed evaluation dataset."""
    path = processed_dir / f"{name}.json"

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

def load_split_records(
    harmtype: str,
    split: str,
    split_dir: Path = DEFAULT_SPLIT_DIR,
) -> list[dict]:
    """Load full records from an Arditi train/val/test split."""
    if harmtype not in VALID_HARMTYPES:
        raise ValueError(f"Unknown harmtype: {harmtype}")

    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split: {split}")

    path = split_dir / f"{harmtype}_{split}.json"

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

def sample_harmless_test_dataset(
    *,
    n_train: int = 128,
    n_val: int = 32,
    n_test: int = 100,
    seed: int = 42,
    split_dir: Path = DEFAULT_SPLIT_DIR,
) -> list[dict]:
    """Sample harmless test records using Arditi's RNG call order."""
    rng = random.Random(seed)

    harmful_train = load_split_records(
        "harmful",
        "train",
        split_dir,
    )
    harmless_train = load_split_records(
        "harmless",
        "train",
        split_dir,
    )
    harmful_val = load_split_records(
        "harmful",
        "val",
        split_dir,
    )
    harmless_val = load_split_records(
        "harmless",
        "val",
        split_dir,
    )
    harmless_test = load_split_records(
        "harmless",
        "test",
        split_dir,
    )

    # Consume the same four sampling calls as the released pipeline.
    rng.sample(harmful_train, n_train)
    rng.sample(harmless_train, n_train)
    rng.sample(harmful_val, n_val)
    rng.sample(harmless_val, n_val)

    return rng.sample(harmless_test, n_test)