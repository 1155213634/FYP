import os
import torch
import json
import gc
import textwrap
import ast
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from docker_code import DockerSandbox
from utils import solution_code_parser, ensure_solution_function, clean_code_output
from prompts import get_answerer_prompt

class KeywordToPositional(ast.NodeTransformer):

  def __init__(self, target_func):
    self.target_func = target_func

  def visit_Call(self, node):
    self.generic_visit(node)

    if isinstance(node.func, ast.Name) and node.func.id == "candidate":
      node.func.id = "solution"

    for kw in node.keywords:
      if kw.arg is not None:
        node.args.append(kw.value)
    node.keywords = [kw for kw in node.keywords if kw.arg is None]

    return node


def test_model_performance_LeetCodeDataset(iter, model_path, QLoRA, Judged, adapter_path=None, store_log=False):
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
    DATASET_NAME = "newfacade/LeetCodeDataset"
    print(f"Loading dataset '{DATASET_NAME}' from Hugging Face...")
    
    # Loads LeetCodeDataset dataset (usually defaults to train/default split)
    dataset = load_dataset(DATASET_NAME, split="test")

    if QLoRA:
        if Judged:
            OUTPUT_DIR = f"./results/QLoRA/Judged/{model_path}"
        else:
            OUTPUT_DIR = f"./results/QLoRA/NonJudged/{model_path}"
    else:
        OUTPUT_DIR = f"./results/{model_path}"

    OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"evaluation_results_{iter}_LeetCodeDataset.json")
    STORE_LOG = True

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==========================================
    # Main Evaluation Execution Pipeline
    # ==========================================
    print(MODEL_PATH)
    model.eval()

    # Initialize your Docker sandbox from docker_code.py
    sandbox = DockerSandbox(image_name="python:3.10-slim", timeout_seconds=10)

    passed_problems = 0
    total_problems = len(dataset)
    evaluation_logs = []

    print(f"Running generation and sandbox evaluation on {total_problems} LeetCodeDataset problems...")

    # Loop through problems
    for i, problem in enumerate(dataset):
        if (i+1) % 10 == 0 :
            print(f"Now Performing Test No.{i}\nAccuracy = {passed_problems/total_problems*100}%")
        # Adjust field lookups according to LeetCodeDataset's data columns schema
        task_id = problem.get("task_id")
        prompt_text = problem.get("problem_description")
        raw_LeetCodeDataset_tests = textwrap.dedent(problem.get("test").split('\n', 1)[-1])

        tree = ast.parse(raw_LeetCodeDataset_tests)
        modified_tree = KeywordToPositional(target_func="solution").visit(tree)
        ast.fix_missing_locations(modified_tree)
        
        messages = get_answerer_prompt(prompt_text)

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

        solution = clean_code_output(generated_solution)
        try:
            solution = ensure_solution_function(generated_solution)
        except Exception as exc:
            pass

        # Evaluate using sandbox
        LeetCodeDataset_test_code = ast.unparse(modified_tree)
        score = sandbox.evaluate(solution, LeetCodeDataset_test_code)
        
        # Must satisfy ALL assertions to be considered a success
        if score == 1:
            passed_problems += 1
            status = "PASSED"
        else:
            status = "FAILED"

        # Store detailed logs for this problem
        if STORE_LOG:
            evaluation_logs.append({
                "task_id": task_id,
                "status": status,
                "score": score,
                "prompt": prompt_text,
                "generated_code": solution,
                "test_code": LeetCodeDataset_test_code
            })
        else:
            evaluation_logs = None

        # if i == 9:
        #     break

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
    print(f"Total Problems Tested: {total_problems}")
    print(f"Successfully Passed All Assertions: {passed_problems}")
    print(f"Pass Rate: {pass_rate:.2f}%")

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()