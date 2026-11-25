import json
import re
from pathlib import Path
from typing import Any

# Matches filenames like: evaluation_results_0_humanevalplus_stats.json
FILENAME_PATTERN = re.compile(r"^evaluation_results_(\d+)_(.+)_stats\.json$")


def load_success_array(path: Path, key: str = "success_array") -> list[int]:
    """Load and validate a binary success array from a stats JSON file."""
    with path.open("r", encoding="utf-8") as f:
        payload: dict[str, Any] = json.load(f)

    if key not in payload or not isinstance(payload[key], list):
        raise KeyError(f"Missing list '{key}' in {path.name}")

    arr = payload[key]
    if not all(x in (0, 1) and not isinstance(x, bool) for x in arr):
        raise ValueError(f"'{key}' in {path.name} must contain only 0 and 1 integers.")

    return arr


def compare_arrays(arr_a: list[int], arr_b: list[int]) -> dict[str, Any]:
    """Compute 0->1 and 1->0 transitions and concordant counts between two arrays."""
    if len(arr_a) != len(arr_b):
        raise ValueError(f"Length mismatch: {len(arr_a)} vs {len(arr_b)}")

    zero_to_one_indices: list[int] = []
    one_to_zero_indices: list[int] = []
    one_to_one = 0
    zero_to_zero = 0

    for idx, (a, b) in enumerate(zip(arr_a, arr_b)):
        if a == 0 and b == 1:
            zero_to_one_indices.append(idx)
        elif a == 1 and b == 0:
            one_to_zero_indices.append(idx)
        elif a == 1 and b == 1:
            one_to_one += 1
        elif a == 0 and b == 0:
            zero_to_zero += 1

    c_0_to_1 = len(zero_to_one_indices)
    c_1_to_0 = len(one_to_zero_indices)

    return {
        "total_pairs": len(arr_a),
        "0->1_count": c_0_to_1,
        "1->0_count": c_1_to_0,
        "1->1_count": one_to_one,
        "0->0_count": zero_to_zero,
        "net_gain": c_0_to_1 - c_1_to_0,
        "0->1_indices": zero_to_one_indices,
        "1->0_indices": one_to_zero_indices,
    }


def merge_pair_analyses(
    input_dir: str | Path = ".",
    output_file: str | Path = "pair_analysis_results.json",
    key: str = "success_array",
) -> dict[str, Any]:
    """Run consecutive-step and first-to-max_iter pair analyses per dataset."""
    folder = Path(input_dir)
    if not folder.is_dir():
        raise NotADirectoryError(f"Directory not found: {folder.resolve()}")

    by_dataset: dict[str, list[tuple[int, Path]]] = {}
    for path in folder.iterdir():
        if not path.is_file():
            continue
        match = FILENAME_PATTERN.match(path.name)
        if match:
            step = int(match.group(1))
            dataset = match.group(2)
            by_dataset.setdefault(dataset, []).append((step, path))

    pair_results: list[dict[str, Any]] = []

    for dataset in sorted(by_dataset.keys()):
        step_files = sorted(by_dataset[dataset], key=lambda item: item[0])
        loaded = [(step, path, load_success_array(path, key=key)) for step, path in step_files]

        # 1. Consecutive step comparisons: step_k -> step_{k+1}
        for (step_a, path_a, arr_a), (step_b, path_b, arr_b) in zip(loaded, loaded[1:]):
            metrics = compare_arrays(arr_a, arr_b)
            pair_results.append(
                {
                    "dataset": dataset,
                    "comparison_type": "consecutive",
                    "from_step": step_a,
                    "to_step": step_b,
                    "pair": f"{step_a}->{step_b}",
                    "from_file": path_a.name,
                    "to_file": path_b.name,
                    **metrics,
                }
            )

        # 2. Extra comparison: initial step (e.g., 0) -> max_iter (highest step available)
        if len(loaded) >= 2:
            init_step, init_path, init_arr = loaded[0]
            max_step, max_path, max_arr = loaded[-1]
            metrics = compare_arrays(init_arr, max_arr)
            pair_results.append(
                {
                    "dataset": dataset,
                    "comparison_type": "0_to_max_iter",
                    "from_step": init_step,
                    "to_step": max_step,
                    "pair": f"{init_step}->{max_step}",
                    "from_file": init_path.name,
                    "to_file": max_path.name,
                    **metrics,
                }
            )

    output_data = {
        "total_pairs_analyzed": len(pair_results),
        "pair_analysis": pair_results,
    }

    out_path = Path(output_file)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("Saved to:", out_path.resolve())
    return output_data


if __name__ == "__main__":
    merge_pair_analyses(
        input_dir="results/Qwen/Qwen2.5-Coder-3B-Instruct-Judge/calculated",
        output_file="pair_analysis_results.json",
        key="success_array",
    )