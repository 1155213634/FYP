"""Orchestrate one iteration using an independent-commit judge reward."""

from typing import Any

# TODO: For each question, commit the judge's independent answer before using a candidate.
# TODO: Keep the judge's solve call candidate-free; then compare against candidate final answers.
# TODO: Build within-question preferences from these judge labels, never from gold references.
# TODO: Match the anchored arm's policy initialization, sampling, training budget, and audit set.
