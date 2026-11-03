"""Orchestrate one iteration using candidate-conditioned judge labels."""

from typing import Any

# TODO: Freeze the policy checkpoint for rollout generation and judging within an iteration.
# TODO: Generate multiple candidates per question and label each with the anchored judge.
# TODO: Build within-question preferences, train the policy, then advance to the next checkpoint.
# TODO: Keep data selection, budgets, and evaluation settings matched with the de-anchored arm.
