"""Compute judge and hidden-anchor metrics with explicit denominators."""

from collections.abc import Sequence
from typing import Any

# TODO: Report pass rate, exact-match accuracy, TPR, FPR, and TPR - FPR discrimination.
# TODO: Report judge-truth gap as pass rate minus accuracy; do not call it FPR.
# TODO: Handle empty positive/negative groups explicitly and include each denominator.
# TODO: Add Wilson intervals or paired bootstrap intervals for the final comparison.
