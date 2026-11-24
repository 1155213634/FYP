import os
import torch
import json
import re
import gc
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM

from docker_code import DockerSandbox
from prompts import get_ame_answerer_prompt

def extract_aime_answer(response_text: str) -> int | None:
    # 1. Try to find the content inside \boxed{...}
    # We look for the last occurrence in case of multiple boxes
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(boxed_pattern, response_text, re.DOTALL)
    
    candidate = None
    if matches:
        candidate = matches[-1].strip()
    else:
        # 2. Fallback: Search for the last standalone integer (0-999) if no box is found
        all_numbers = re.findall(r'\b\d{1,3}\b', response_text)
        if all_numbers:
            candidate = all_numbers[-1]

    if candidate is not None:
        # Clean potential formatting like commas or currency symbols (e.g., "1,200" or "$500")
        candidate_cleaned = re.sub(r'[^\d]', '', candidate)
        if candidate_cleaned.isdigit():
            # AIME answers are strictly 0 to 999 (sometimes formatted with leading zeros)
            val = int(candidate_cleaned)
            return val % 1000  # or keep as integer depending on formatting
            
    return None

def test_model_performance_ame(iter, model_path, QLoRA, Judged, adapter_path=None, store_log=False):
    # ==========================================
    # Configuration Constants
    # ==========================================
    PARENT_DIR = model_path

    if iter == 0:
        MODEL_PATH = model_path
        TOKENIZER_PATH = model_path
    else:
        if QLoRA:
            if Judged:
                MODEL_PATH = f"ready/QLoRA/Judged/{PARENT_DIR}/{iter}_answerer_model"
                TOKENIZER_PATH = f"ready/QLoRA/Judged/{PARENT_DIR}/{iter}_tokenizer"
            else:
                MODEL_PATH = f"ready/QLoRA/NonJudged/{PARENT_DIR}/{iter}_answerer_model"
                TOKENIZER_PATH = f"ready/QLoRA/NonJudged/{PARENT_DIR}/{iter}_tokenizer"
        else:
            MODEL_PATH = f"ready/{PARENT_DIR}/{iter}_answerer_model"
            TOKENIZER_PATH = f"ready/{PARENT_DIR}/{iter}_tokenizer"

    print("Loading model and tokenizer from local folders...")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if QLoRA and iter > 0:
        # Use this when MODEL_PATH contains adapter_config.json.
        model = AutoPeftModelForCausalLM.from_pretrained(
            MODEL_PATH,
            is_trainable=False,
            dtype=torch.float16,
            device_map="auto",
        )
    else:
        # Loads full saved/incremental models and the original base model.
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    # ==========================================
    # Dataset Loading from Hugging Face (`Qwen/LeetCodeDataset`)
    # ==========================================
    DATASET_NAME = "project-numina/aimo-validation-aime"
    print(f"Loading dataset '{DATASET_NAME}' from Hugging Face...")
    
    # Loads LeetCodeDataset dataset (usually defaults to train/default split)
    dataset = load_dataset(DATASET_NAME, split="train")

    if QLoRA:
        if Judged:
            OUTPUT_DIR = f"./results/QLoRA/Judged/{model_path}"
        else:
            OUTPUT_DIR = f"./results/QLoRA/NonJudged/{model_path}"
    else:
        OUTPUT_DIR = f"./results/{model_path}"

    OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"evaluation_results_{iter}_MBPPPlus.json")
    STORE_LOG = True

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==========================================
    # Main Evaluation Execution Pipeline
    # ==========================================
    print(MODEL_PATH)
    model.eval()

    passed_problems = 0
    total_problems = len(dataset)
    evaluation_logs = []

    print(f"Running generation and sandbox evaluation on {total_problems} AME24 problems...")

    # Loop through problems
    for i, problem in enumerate(dataset):
        if (i+1) % 10 == 0 :
            print(f"Now Performing Test No.{i}\nAccuracy = {passed_problems/total_problems*100}%")
        # Adjust field lookups according to LeetCodeDataset's data columns schema
        task_id = problem.get("id", f"problem_{i}")
        problem_text = problem.get("problem", "")
        ground_truth_raw = problem.get("answer", 0)
        ground_truth = int(ground_truth_raw) if str(ground_truth_raw).isdigit() else 0

        # Construct prompt using the imported prompt function
        prompt_text = get_ame_answerer_prompt(problem_text)
        prompt_text = tokenizer.apply_chat_template(
            prompt_text,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize prompt and send to model's device
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        
        # Generate completion/reasoning
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
            )
        
        # Decode the generated output text
        solution = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the integer answer using the provided extraction helper
        answer = extract_aime_answer(solution)

        # Score is 1 if the extracted answer correctly matches the ground truth
        score = 1 if (answer is not None and answer == ground_truth) else 0

        # Must satisfy ALL assertions to be considered a success
        if score == 1:
            passed_problems += 1
            status = "PASSED"
        else:
            status = "FAILED"

        # print(f"[{i+1}/{total_problems}] ID: {task_id} -> {status} (Assertions: {int(score)}/{max_possible_assertions})")

        # Store detailed logs for this problem
        if STORE_LOG:
            evaluation_logs.append({
                "task_id": task_id,
                "status": status,
                "score": score,
                "prompt": prompt_text,
                "reasoning": solution,
                "answer": answer,
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
    print(f"Model: {model_path} at {iter} Iterations")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Total Problems Tested: {total_problems}")
    print(f"Successfully Passed All Assertions: {passed_problems}")
    print(f"Pass@1 Rate: {pass_rate:.2f}%")

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()