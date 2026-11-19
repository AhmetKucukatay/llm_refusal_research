"""Inspect the sampled Arditi direction datasets."""

from reproductions.arditi_2024.data import sample_direction_datasets


def main() -> None:
    """Print sampled dataset sizes."""
    harmful_train, harmless_train, harmful_val, harmless_val = (
        sample_direction_datasets()
    )

    print(f"Harmful train: {len(harmful_train)}")
    print(f"Harmless train: {len(harmless_train)}")
    print(f"Harmful val: {len(harmful_val)}")
    print(f"Harmless val: {len(harmless_val)}")


if __name__ == "__main__":
    main()
    