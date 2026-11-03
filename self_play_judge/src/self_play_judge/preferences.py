"""Build DPO preferences from judge labels, never from hidden correctness labels."""

from collections.abc import Iterable
from typing import Any

# TODO: Group solver rollouts by question ID.
# TODO: Pair one judge-accepted rollout as chosen with one judge-rejected rollout as rejected.
# TODO: Skip questions without both labels; never fall back to reference-answer correctness.
# TODO: Return the schema expected by the pinned TRL DPOTrainer version.
