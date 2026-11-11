import os
import torch
import json
import re
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from docker_code import DockerSandbox
from utils import assert_code_parser, solution_code_parser
from prompts import get_answerer_prompt

# ==========================================
# Configuration Constants
# ==========================================
PARENT_DIR = "ready/Qwen/Qwen2.5-Coder-3B-Instruct"
MODEL_PATH = f"{PARENT_DIR}/10_answerer_model"
TOKENIZER_PATH = f"{PARENT_DIR}/10_tokenizer"
# MODEL_PATH = "Qwen/Qwen2.5-Coder-3B-Instruct"
# TOKENIZER_PATH = "Qwen/Qwen2.5-Coder-3B-Instruct"

DATASET_NAME = "openai/openai_humaneval"
DATASET_SPLIT = "test"

OUTPUT_DIR = "./results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "evaluation_results_baseline.json")
STORE_LOG = True

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

def parse_human_eval_asserts(str):
    list_of_asserts = str.split("assert")[1:]
    return "\n".join("assert " + assert_statement.strip() for assert_statement in list_of_asserts)


# Loop through problems
for i, problem in enumerate(dataset):
    task_id = problem.get("task_id", f"problem_{i}")
    prompt_text = problem["prompt"]
    human_eval_test_code = problem["test"]
    
    # Tokenize and generate completion
    text_prompt = tokenizer.apply_chat_template(get_answerer_prompt(prompt_text), tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text_prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.2,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    
    generated_solution = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    test_code = assert_code_parser(parse_human_eval_asserts(human_eval_test_code))
    solution = solution_code_parser(generated_solution)
    # Evaluate solution using DockerSandbox
    score = sandbox.evaluate(solution_code=solution, test_code=test_code, normalized=False)
    
    # Calculate how many assertions exist in this problem
    valid_assertions = [line for line in test_code.split("\n") if line.strip()]
    max_possible_assertions = len(valid_assertions)
    
    # Must satisfy ALL assertions to be considered a success
    if max_possible_assertions > 0 and score >= max_possible_assertions:
        passed_problems += 1
        status = "PASSED"
    else:
        status = "FAILED"

    print(f"[{i+1}/{total_problems}] ID: {task_id} -> {status} (Assertions: {int(score)}/{max_possible_assertions})")

    # Store detailed logs for this problem
    if STORE_LOG:
        evaluation_logs.append({
            "task_id": task_id,
            "status": status,
            "score": score,
            "max_score": max_possible_assertions,
            "prompt": prompt_text,
            "generated_code": generated_solution,
            "test_code": test_code
        })
    else:
        evaluation_logs = None
    

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