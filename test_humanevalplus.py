import os
import torch
import json
import gc
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from docker_code import DockerSandbox
from utils import solution_code_parser, human_eval_solution_parser
from prompts import get_human_eval_answerer_prompt

def test_model_performance_humanevalplus(iter, model_path, QLoRA, Judged, adapter_path=None, testset="humaneval", store_log=False):
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
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        # Loads full saved/incremental models and the original base model.
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    DATASET_NAME = "evalplus/humanevalplus"
    DATASET_SPLIT = "test"

    if QLoRA:
        if Judged:
            OUTPUT_DIR = f"./results/QLoRA/Judged/{model_path}"
        else:
            OUTPUT_DIR = f"./results/QLoRA/NonJudged/{model_path}"
    else:
        OUTPUT_DIR = f"./results/{model_path}"

    OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"evaluation_results_{iter}_humanevalplus.json")
    STORE_LOG = True

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==========================================
    # Main Evaluation Execution Pipeline
    # ==========================================


    print(MODEL_PATH)
    model.eval()

    # Load the dataset
    print(f"Loading dataset '{DATASET_NAME}' ({DATASET_SPLIT} split)...")
    dataset = load_dataset(DATASET_NAME)[DATASET_SPLIT]

    # Initialize your Docker sandbox from docker_box.py
    sandbox = DockerSandbox(image_name="fyp-sandbox", timeout_seconds=15)

    passed_problems = 0
    total_problems = len(dataset)
    evaluation_logs = []

    print(f"Running generation and sandbox evaluation on {len(dataset)} problems...")

    # Loop through problems
    for i, problem in enumerate(dataset):
        task_id = problem.get("task_id", f"problem_{i}")
        prompt_text = problem["prompt"]
        human_eval_test_code = problem["test"]

        entry_point = problem["entry_point"]
        messages = get_human_eval_answerer_prompt(prompt_text, entry_point)

        text_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text_prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        generated_solution = tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )

        solution = human_eval_solution_parser(generated_solution, entry_point)
        passed, total = sandbox.evaluate_humaneval(
            solution,
            human_eval_test_code,
            candidate_name_hint=entry_point,
            problem_prompt=prompt_text,
        )
        
        # Must satisfy ALL assertions to be considered a success
        if total > 0 and passed == total:
            passed_problems += 1
            status = "PASSED"
        else:
            status = "FAILED"

        # print(f"[{i+1}/{total_problems}] ID: {task_id} -> {status} (Assertions: {int(passed)}/{total})")

        # Store detailed logs for this problem
        if STORE_LOG:
            evaluation_logs.append({
                "task_id": task_id,
                "status": status,
                "score": passed,
                "max_score": total,
                "prompt": prompt_text,
                "generated_code": generated_solution,
                "test_code": human_eval_test_code
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