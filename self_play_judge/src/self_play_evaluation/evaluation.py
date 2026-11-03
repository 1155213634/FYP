"""Evaluate fixed checkpoints against held-out GSM8K references."""

import csv
import json
from pathlib import Path
from typing import Any

# TODO: Generate candidates on a fixed audit set, independent of training questions.
# TODO: Keep references inside this evaluator and compare only parsed final answers.
# TODO: Evaluate judge pass rate and true correctness on the same candidate records.
# TODO: Save per-example outcomes and aggregate metrics with denominators and run metadata.
# TODO: Track malformed/truncated outputs rather than silently dropping them.
