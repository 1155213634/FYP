from datasets import load_dataset

from docker_code import DockerSandbox
from utils import human_eval_solution_parser

dataset = load_dataset("openai/openai_humaneval", split="test")
problem = dataset[0]

reference_source = problem["prompt"] + problem["canonical_solution"]
solution = human_eval_solution_parser(reference_source, problem["entry_point"])

sandbox = DockerSandbox(image_name="python:3.10-slim", timeout_seconds=10)
passed, total = sandbox.evaluate_humaneval(
    solution,
    problem["test"],
    candidate_name_hint=problem["entry_point"],
)

print(f"{problem['task_id']}: {passed}/{total} assertions passed")
if total == 0 or passed != total:
    raise SystemExit("Canonical solution did not pass; inspect Docker/test-harness output.")