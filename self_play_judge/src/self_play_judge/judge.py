"""Define candidate-conditioned and independent-commit reference-free judges."""

from dataclasses import dataclass
from typing import Any

# TODO: Define an explicit judgment result including accept/reject, parsed answer, and parse status.
# TODO: Anchored mode: provide the question and candidate together, with no reference answer.
# TODO: De-anchored mode: solve from the question alone and persist the committed answer first.
# TODO: Compare the committed answer with the candidate using the shared normalization rules.
# TODO: Log prompts and outputs for audits without ever including hidden gold labels.
