"""Config-driven entry point for the local experiment."""

from pathlib import Path
from typing import Any

import yaml

# TODO: Load and validate the YAML configuration.
# TODO: Set random seeds and initialize output directories/run metadata.
# TODO: Run the no-training audit before permitting a training mode.
# TODO: Dispatch the selected loop and evaluate each saved checkpoint.


def main() -> None:
    """TODO: Wire configuration, model/data setup, loop execution, and evaluation."""
    raise NotImplementedError(
        "Complete the experiment runner after implementing its components."
    )


if __name__ == "__main__":
    main()
