import json
import re
import torch
import ast
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts import get_proposer_prompt, get_answerer_prompt
from docker_code import DockerSandbox

LOG_FILE = "./training_rollouts.jsonl"

# ==========================================
# 0. CONSTANTS
# ==========================================
MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
# MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
# MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"

LEARN_RATE = 5e-6
MAX_ITER = 50                      # No. of total loops
GROUP_SIZE = 4                     # GRPO group size

ANS_ATTEMPTS_PROP_TRAIN = 4        # No. of answerer attempts to calculate proposer difficulty score
ANS_TOKENS_PROP_TRAIN = 256        # No. of max tokens for answerer for training proposer
ANS_TEMP_PROP_TRAIN = 0.7          # No. of max tokens for answerer for training proposer

PROP_TOKENS_ANS_TRAIN = 256        # No. of max tokens for proposer for training answerer
PROP_TEMP_ANS_TRAIN = 0.7          # No. of max tokens for proposer for training answerer

# ==========================================
# 2. UTILITIES
# ==========================================

def clean_code_output(text: str) -> str:
    """
    Extracts pure Python code if enclosed in markdown code blocks,
    otherwise returns the stripped raw text.
    """
    pattern = r"```(?:python)?\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n".join(match.strip() for match in matches)
    return text.strip()

def parse_proposer_output(proposer_text: str):
    """
    Parses proposer text into problem description and test assertions.
    Returns (None, None) if required formatting tags are missing.
    """
    if "[PROBLEM]" not in proposer_text or "[TESTS]" not in proposer_text:
        return None, None
        
    parts = proposer_text.split("[TESTS]")
    problem_desc = parts[0].replace("[PROBLEM]", "").strip()
    test_code = parts[1].strip()
    test_code = clean_code_output(test_code)
    return problem_desc, test_code

def validate_assertions(test_code_str):
    """Safely checks if test code is syntactically valid and contains assertions."""
    if not test_code_str or not isinstance(test_code_str, str):
        return 0.0
    
    score = 0.0

    # Assertion Code Needs to Run Properly 0.3
    syntax_valid = False
    tree = None 
    try:
        tree = ast.parse(test_code_str)
        score += 0.3
        syntax_valid = True
    except SyntaxError:
        return 0.0

    # Needs to Assert 0.4
    if "assert" in test_code_str:
        score += 0.4

    # Asser Less Than Four 0.3
    assertion_count = 0
    if syntax_valid:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                assertion_count += 1
    
    if assertion_count < 4:
        score += 0.3
        
    return score

def assert_code_parser(code: str) -> str:
    assert_pattern = r"(?m)^[ \t]*assert[ \t]+([A-Za-z_]\w*)[ \t]*\("
    asserts = code.split("\n")
    flags = re.DOTALL
    temp_list = []
    for assert_statement in asserts:
        temp_list.append(re.sub(assert_pattern, "assert f(", assert_statement, flags=flags))
    return "\n".join(assert_pattern.strip() for assert_pattern in temp_list)

def solution_code_parser(text: str) -> str | None:
    match = re.search(
        r"```(?:python[ \t]*)?\s*(.*?)```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    code = match.group(1) if match else text

    function_match = re.search(
        r"(?m)^[ \t]*def[ \t]+([A-Za-z_]\w*)[ \t]*\(",
        code,
    )
    if not function_match:
        return text

    original_name = function_match.group(1)
    if original_name != "f":
        code = re.sub(
            rf"(?m)^([ \t]*def[ \t]+){re.escape(original_name)}(?=[ \t]*\()",
            r"\1f",
            code,
            count=1,
        )

    return code

def log_rollout(proposer_completion, answerer_completion, reward, metadata=None, seed_prompt=None):
    """Appends a complete self-play rollout record (Proposer + Answerer) to JSONL."""
    record = {
        # "seed_prompt": seed_prompt,
        "reward": reward,
        "metadata": metadata or {},
        "proposer_completion": proposer_completion,
        "answerer_completion": answerer_completion
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ==========================================
# 3. SELF-PLAY TRAINING ORCHESTRATION
# ==========================================
def main():
    # Cleanup log file
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        pass

    sandbox = DockerSandbox(timeout_seconds=3)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # Load shared or separate models for Proposer and Answerer
    print("Loading models...")
    proposer_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")
    answerer_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")

    # Common TRL configuration
    proposer_args = GRPOConfig(
        output_dir="grpo_proposer_output",
        learning_rate=LEARN_RATE,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=GROUP_SIZE,          # Group size (G) for GRPO
        max_completion_length=256,
        temperature=0.8,            # Temperature for completions
        repetition_penalty=1.15,    # Additional penalty for repeating
        logging_steps=5,
        bf16=True,
        report_to="none"
    )

    answerer_args = GRPOConfig(
        output_dir="grpo_answerer_output",
        learning_rate=LEARN_RATE,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=GROUP_SIZE,          # Group size (G) for GRPO
        max_completion_length=256,
        logging_steps=5,
        bf16=True,
        report_to="none"
    )

    # --- ALTERNATING SELF-PLAY TRAINING LOOP ---
    for iteration in range(MAX_ITER):
        print(f"\n==============================")
        print(f"=== SELF-PLAY ITERATION {iteration + 1}/{MAX_ITER} ===")
        print(f"==============================")

        # ----------------------------------------------------
        # PHASE 1: Train Proposer (Freeze Answerer)
        # ----------------------------------------------------
        print("\n--- Phase 1: Training Proposer ---")
        for param in answerer_model.parameters():
            param.requires_grad = False
        for param in proposer_model.parameters():
            param.requires_grad = True


        def proposer_reward_func(prompts, completions, **kwargs):
            # Max reward = formatting 0.2 + assertion 0.2 + answerer correctness 0.6 = 1.0
            rewards = []
            answerer_model.eval()
            for prompt, completion in zip(prompts, completions):
                comp_text = completion[0]["content"] if isinstance(completion, list) else completion

                # Check for formatting
                has_problem_tag = comp_text.count("[PROBLEM]")
                has_tests_tag = comp_text.count("[TESTS]")

                format_score = 0.0
                if has_problem_tag == 1:
                    format_score += 0.5
                if has_tests_tag == 1:
                    format_score += 0.5
                
                if has_problem_tag > 1 or has_tests_tag > 1:
                    format_score = 0
                    
                prob_desc, test_code = parse_proposer_output(comp_text)
                
                # Reject malformed outputs by returning 0.0 reward
                if prob_desc is None or test_code is None:
                    rewards.append(0.0)

                    log_rollout(
                    seed_prompt=None,
                    proposer_completion=None,
                    answerer_completion=None,
                    reward=0.0,
                    metadata={"phase": "proposer", "failed": True}
                )
                    continue

                test_code = assert_code_parser(test_code)

                # Reward for correct assertion code
                assert_score = 0.0
                assert_score += validate_assertions(test_code)
                
                # Format requirements built into the answerer prompt
                text_prompt = tokenizer.apply_chat_template(get_answerer_prompt(prob_desc), tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text_prompt, return_tensors="pt").to(proposer_model.device)
                input_len = inputs["input_ids"].shape[1]
            
                scores = []

                with torch.no_grad():
                        outs = answerer_model.generate(**inputs, max_new_tokens=ANS_TOKENS_PROP_TRAIN,
                                                      do_sample=True, temperature=ANS_TEMP_PROP_TRAIN,
                                                      num_return_sequences=4)
                for out in outs:
                    generated_tokens = out[input_len:]
                
                    raw_solution = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    solution = clean_code_output(raw_solution)
                    
                    # Reward heuristic: Target problems that are challenging yet solvable (~0.5 pass score)
                    score = sandbox.evaluate(solution, test_code)
                    scores.append(score)

                avg_score = sum(scores) / len(scores)
                diff_reward = 1.0 - abs(avg_score - 0.5) * 2

                total_reward = format_score * 0.2 + diff_reward * 0.6 + assert_score * 0.2

                # Log results to file
                log_rollout(
                    seed_prompt=prompt,
                    proposer_completion=comp_text,
                    answerer_completion=solution,
                    reward=round(total_reward, 2),
                    metadata={"phase": "proposer", "format": round(format_score,2), 
                              "assert": round(assert_score, 2), "ans_avg":round(avg_score,2)}
                )

                rewards.append(total_reward)
            return rewards


        proposer_prompt = tokenizer.apply_chat_template(
            get_proposer_prompt(iteration), tokenize=False, add_generation_prompt=True
        )

        proposer_dataset = Dataset.from_dict({"prompt": [proposer_prompt]})
        proposer_trainer = GRPOTrainer(
            model=proposer_model,
            reward_funcs=proposer_reward_func,
            args=proposer_args,
            train_dataset=proposer_dataset,
        )
        proposer_trainer.train()

        # ----------------------------------------------------
        # PHASE 2: Train Answerer (Freeze Proposer)
        # ----------------------------------------------------
        print("\n--- Phase 2: Training Answerer ---")
        for param in proposer_model.parameters():
            param.requires_grad = False
        for param in answerer_model.parameters():
            param.requires_grad = True

        proposer_model.eval()

        prob_desc, test_code = None, None
        max_retries = 3

        for attempt in range(max_retries):
            text_prompt = tokenizer.apply_chat_template(get_proposer_prompt(iteration), tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text_prompt, return_tensors="pt").to(proposer_model.device)

            input_len = inputs["input_ids"].shape[1]
            
            with torch.no_grad():
                out = proposer_model.generate(**inputs, max_new_tokens=PROP_TOKENS_ANS_TRAIN,
                                              do_sample=True, temperature=PROP_TEMP_ANS_TRAIN)

            generated_tokens = out[0][input_len:]
            
            generated_q = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            prob_desc, test_code = parse_proposer_output(generated_q)
            test_code = assert_code_parser(test_code)
            if validate_assertions(test_code) <= 0:
                prob_desc, test_code = None, None
            if prob_desc is not None and test_code is not None:
                break
            print(f"Attempt {attempt + 1}: Proposer generated malformed output. Retrying generation...")
            


        if prob_desc is None or test_code is None:
            print("Error: Proposer failed to follow format after multiple retries. Skipping this iteration's Answerer training.")
            continue

        def answerer_reward_func(prompts, completions, **kwargs):
            assert test_code is not None

            rewards = []
            for prompt, completion in zip(prompts, completions):
                raw_sol_text = completion[0]["content"] if isinstance(completion, list) else completion
                solution = solution_code_parser(raw_sol_text)
                
                score = sandbox.evaluate(solution, test_code)

                # Log results
                log_rollout(
                    proposer_completion=generated_q,
                    answerer_completion=solution,
                    reward=score,
                    metadata={"iteration" : iteration, "phase": "answerer", "problem_description": prob_desc, "test_code": test_code, "raw_sol_text": raw_sol_text}
                )

                rewards.append(score)
            return rewards
        

        answerer_prompt = tokenizer.apply_chat_template(
            get_answerer_prompt(prob_desc), tokenize=False, add_generation_prompt=True
        )

        answerer_dataset = Dataset.from_dict({"prompt": [answerer_prompt]})
        answerer_trainer = GRPOTrainer(
            model=answerer_model,
            reward_funcs=answerer_reward_func,
            args=answerer_args,
            train_dataset=answerer_dataset,
        )
        answerer_trainer.train()

        # Saving
        if (iteration+1) % 10 == 0:
            print("Saving final models to disk...")
            proposer_model.save_pretrained(f"./trained/{MODEL_ID}/{iteration+1}_proposer_model")
            answerer_model.save_pretrained(f"./trained/{MODEL_ID}/{iteration+1}_answerer_model")
            tokenizer.save_pretrained(f"./trained/{MODEL_ID}/{iteration+1}_tokenizer")
            print("Models saved successfully!")


    print("\nSelf-Play Reinforcement Training Completed Successfully!")

if __name__ == "__main__":
    main()