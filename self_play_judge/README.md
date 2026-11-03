# Self-Play Judge Study

Starter scaffold for a local GSM8K study of anchored and de-anchored reference-free judging.
The Python files intentionally contain imports and implementation notes, not experiment logic.

## First implementation order

1. Fill in `configs/experiment.yaml` after choosing a model and checking VRAM use.
2. Implement data loading and final-answer parsing. Keep reference answers out of all prompts and training records.
3. Implement the two judge channels and verify their information flow on fixed candidate examples.
4. Run the no-training capability audit before implementing DPO.
5. Implement same-question accepted/rejected rollout pairs and the two training loops.
6. Evaluate every checkpoint on a fixed, disjoint audit set and save the raw results.

## Module map

- `data.py`: GSM8K loading, stable IDs, and the boundary between prompt data and hidden labels.
- `models.py`: tokenizer/model loading, generation settings, and optional LoRA setup.
- `parsing.py`: strict output parsing and GSM8K final-answer normalization.
- `judge.py`: anchored and independent-commit judge behavior.
- `preferences.py`: within-question accepted/rejected rollout pairs for DPO.
- `training.py`: TRL/PEFT training configuration and checkpoint handling.
- `loops/`: one orchestration path per judge reward channel.
- `evaluation.py` and `metrics.py`: hidden-anchor evaluation and reported statistics.
- `run.py`: config-driven entry point; keep notebook experiments out of the source of truth.
- `tests/`: focused checks for parsing, information flow, preferences, and metrics.

Start with a small pilot. The proposed 1,000 training questions and 200 evaluation questions are scale-up targets, not assumptions about what fits or finishes locally.
