"""Download the official dataset splits for the Arditi et al. reproduction."""

from pathlib import Path
from urllib.request import urlopen

ARDITI_COMMIT = "9d852fae1a9121c78b29142de733cb1340770cc3"

SPLIT_FILENAMES = (
    "harmful_train.json",
    "harmful_val.json",
    "harmful_test.json",
    "harmless_train.json",
    "harmless_val.json",
    "harmless_test.json",
)

DEFAULT_OUTPUT_DIR = Path("data/arditi_2024/splits")

PROCESSED_FILES = (  #test/evaluation
    "jailbreakbench.json",
)


def fetch_arditi_splits(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Download the official Arditi dataset splits from a pinned commit."""
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = (
        "https://raw.githubusercontent.com/"
        "andyrdt/refusal_direction/"
        f"{ARDITI_COMMIT}/dataset/splits"
    )

    for filename in SPLIT_FILENAMES:
        url = f"{base_url}/{filename}"
        destination = output_dir / filename

        print(f"Downloading {filename}...")

        with urlopen(url) as response:
            destination.write_bytes(response.read())

    print(f"Saved dataset splits to {output_dir}")

    processed_dir = Path("data/arditi_2024/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    for filename in PROCESSED_FILES:
        url = (
            "https://raw.githubusercontent.com/"
            f"andyrdt/refusal_direction/{ARDITI_COMMIT}/"
            f"dataset/processed/{filename}"
        )

        destination = processed_dir / filename

        print(f"Downloading {filename}...")
        with urlopen(url) as response:
            destination.write_bytes(response.read())


if __name__ == "__main__":
    fetch_arditi_splits()