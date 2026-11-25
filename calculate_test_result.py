import argparse
import json
from pathlib import Path


def get_calculated_dir(target: Path) -> Path:
    base_dir = target if target.is_dir() else target.parent
    calc_dir = base_dir / "calculated"
    calc_dir.mkdir(parents=True, exist_ok=True)
    return calc_dir


def is_humaneval_item(item: dict, dataset_name: str | None) -> bool:
    dataset_str = (dataset_name or "").lower()
    task_id_str = str(item.get("task_id", "")).lower()
    return (
        "humaneval" in dataset_str
        or "humaneval" in task_id_str
        or ("max_score" in item and item["max_score"] is not None)
    )


def extract_normalized_score(item: dict, dataset_name: str | None, cull_bonus: bool = False) -> float:
    raw_score = float(item.get("score", 0.0) or 0.0)

    if is_humaneval_item(item, dataset_name):
        # Never apply --cull-bonus to HumanEval items
        max_score = item.get("max_score")
        if max_score is not None and float(max_score) > 0:
            score = raw_score / float(max_score)
        else:
            score = raw_score
        return round(score, 6)

    # Standard datasets (e.g., MBPPPlus, LeetCodeDataset)
    if cull_bonus and raw_score > 0:
        score = raw_score - 0.05
    else:
        score = raw_score
    return round(score, 6)


def process_file(path: Path, calc_dir: Path, cull_bonus: bool = False) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    dataset_name = data.get("dataset")
    results = data.get("results", [])
    adjusted_scores = [
        extract_normalized_score(item, dataset_name, cull_bonus=cull_bonus)
        for item in results
    ]

    total_tests = len(adjusted_scores)
    success_array = [1 if s >= 1.0 - 1e-6 else 0 for s in adjusted_scores]
    passed_tests = sum(success_array)

    test_accuracy = (passed_tests / total_tests) if total_tests > 0 else 0.0
    secondary_accuracy = (sum(adjusted_scores) / total_tests) if total_tests > 0 else 0.0

    stats_payload = {
        "source_file": path.name,
        "dataset": dataset_name,
        "cull_bonus": cull_bonus,
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "test_accuracy": test_accuracy,
        "secondary_test_accuracy": secondary_accuracy,
        "success_array": success_array,
    }

    output_path = calc_dir / f"{path.stem}_stats{path.suffix}"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(stats_payload, f, indent=2)

    return {
        "file": str(path),
        "output_file": str(output_path),
        **stats_payload,
    }


def collect_json_files(target: Path, calc_dir: Path, recursive: bool = False) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        pattern = "**/*.json" if recursive else "*.json"
        return sorted(
            p for p in target.glob(pattern)
            if p.is_file()
            and not p.stem.endswith("_stats")
            and calc_dir not in p.parents
        )
    raise FileNotFoundError(f"Path not found: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse a JSON file or folder of JSONs and save stats + success array into a 'calculated' subfolder."
    )
    parser.add_argument("target", help="Path to a JSON file or a folder containing JSON files")
    parser.add_argument(
        "--cull-bonus",
        action="store_true",
        help="Subtract 0.05 from scores greater than 0 for non-HumanEval datasets (off by default)",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Also search subfolders when target is a directory",
    )
    args = parser.parse_args()

    target_path = Path(args.target)
    if not target_path.exists():
        print(f"Path not found: {target_path}")
        raise SystemExit(1)

    calc_dir = get_calculated_dir(target_path)
    files = collect_json_files(target_path, calc_dir, recursive=args.recursive)
    if not files:
        print(f"No JSON files found in: {target_path}")
        raise SystemExit(1)

    for file_path in files:
        summary = process_file(file_path, calc_dir, cull_bonus=args.cull_bonus)
        print(
            f"{summary['file']} | "
            f"tests: {summary['total_tests']} | "
            f"test_acc: {summary['test_accuracy']:.4f} | "
            f"sec_acc: {summary['secondary_test_accuracy']:.4f} | "
            f"saved -> {summary['output_file']}"
        )