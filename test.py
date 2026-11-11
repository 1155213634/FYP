import os
import torch
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from docker_code import DockerSandbox

# ==========================================
# Configuration Constants
# ==========================================
PARENT_DIR = "trained/Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_PATH = f"{PARENT_DIR}/10_answerer_model"
TOKENIZER_PATH = f"{PARENT_DIR}/10_tokenizer"

DATASET_NAME = "openai/openai_humaneval"
DATASET_SPLIT = "test"

OUTPUT_DIR = "./results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "evaluation_results.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Main Evaluation Execution Pipeline
# ==========================================

print("Loading model and tokenizer from local folders...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    device_map="auto"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the dataset
print(f"Loading dataset '{DATASET_NAME}' ({DATASET_SPLIT} split)...")
dataset = load_dataset(DATASET_NAME)[DATASET_SPLIT]

# Initialize your Docker sandbox from docker_box.py
sandbox = DockerSandbox(image_name="python:3.10-slim", timeout_seconds=3)

passed_problems = 0
total_problems = len(dataset)
evaluation_logs = []

print(f"Running generation and sandbox evaluation on {len(dataset)} problems...")

# Loop through problems
for i, problem in enumerate(dataset):
    task_id = problem.get("task_id", f"problem_{i}")
    prompt_text = problem["prompt"]
    test_code = problem["test"]
    
    # Tokenize and generate completion
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.2,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    
    generated_solution = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Evaluate solution using DockerSandbox
    score = sandbox.evaluate(solution_code=generated_solution, test_code=test_code)
    
    # Calculate how many assertions exist in this problem
    valid_assertions = [line for line in test_code.split("\n") if line.strip()]
    max_possible_assertions = len(valid_assertions)
    
    # Must satisfy ALL assertions to be considered a success
    if max_possible_assertions > 0 and score == max_possible_assertions:
        passed_problems += 1
        status = "PASSED"
    else:
        status = "FAILED"

    print(f"[{i+1}/{total_problems}] ID: {task_id} -> {status} (Assertions: {int(score)}/{max_possible_assertions})")

    # Store detailed logs for this problem
    evaluation_logs.append({
        "task_id": task_id,
        "status": status,
        "score": score,
        "max_score": max_possible_assertions,
        "prompt": prompt_text,
        "generated_code": generated_solution,
        "test_code": test_code
    })
    
    break

# Calculate final pass rate percentage
pass_rate = (passed_problems / total_problems) * 100 if total_problems > 0 else 0

# Compile summary report
final_report = {
    "dataset": DATASET_NAME,
    "total_problems": total_problems,
    "passed_problems": passed_problems,
    "pass_rate_percentage": round(pass_rate, 2),
    "results": evaluation_logs
}

# Write results to JSON file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_report, f, indent=4, ensure_ascii=False)

print("\n--- Final Evaluation Summary ---")
print(f"Dataset: {DATASET_NAME}")
print(f"Total Problems Tested: {total_problems}")
print(f"Successfully Passed All Assertions: {passed_problems}")
print(f"Pass@1 Rate: {pass_rate:.2f}%")