import os
import json
import re
import tempfile
import docker
import torch
import ast
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts import get_proposer_prompt, get_answerer_prompt

LOG_FILE = "./training_rollouts.jsonl"

# ==========================================
# 1. SECURE DOCKER SANDBOX
# ==========================================
class DockerSandbox:
    def __init__(self, image_name: str = "python:3.10-slim", timeout_seconds: int = 3):
        self.client = docker.from_env()
        self.image_name = image_name
        self.timeout = timeout_seconds

    def evaluate(self, solution_code: str, test_code: str) -> float:
        full_script = f"""\
{solution_code}

try:
{self._indent_assert(test_code)}
    print("ALL_TESTS_PASSED")
    exit(0)
except AssertionError as ae:
    print(f"Assertion Failed: {{ae}}")
    exit(1)
except Exception as e:
    print(f"Test Error: {{e}}")
    exit(1)
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding="utf-8") as f:
            f.write(full_script)
            temp_file_path = f.name

        # Get the actual random filename inside the mounted /app directory
        script_name = os.path.basename(temp_file_path)

        container = None
        try:
            container = self.client.containers.run(
                image=self.image_name,
                command=f"python /app/{script_name}",
                volumes={os.path.dirname(temp_file_path): {'bind': '/app', 'mode': 'ro'}},
                working_dir='/app',
                network_mode="none",
                mem_limit="256m",
                cpu_period=100000,
                cpu_quota=50000,
                pids_limit=64,
                detach=True,
                remove=False
            )
            result = container.wait(timeout=self.timeout)
            logs = container.logs(stdout=True, stderr=True).decode('utf-8')
            
            if result.get('StatusCode', -1) == 0 and "ALL_TESTS_PASSED" in logs:
                return 1.0
            return 0.0
        except Exception:
            return 0.0
        finally:
            if container:
                try: container.remove(force=True)
                except Exception: pass
            if os.path.exists(temp_file_path):
                try: os.unlink(temp_file_path)
                except Exception: pass

    # def 
    # code_pattern = r"```(?:python\s*)?\n?(.*?)\n?```"
    # code_snippet = code_match.group(1).strip()

    def _indent_assert(self, code: str, spaces: int = 4) -> str:
        return "\n".join([" " * spaces + line.lstrip() for line in code.splitlines()])


# ==========================================
# 2. UTILITIES
# ==========================================
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

def log_rollout(seed_prompt, proposer_completion, answerer_completion, reward, metadata=None):
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
        return -1.5

    # Needs to Assert 0.3
    if "assert" in test_code_str:
        score += 0.15

    if "assert solution(" in test_code_str:
        score += 0.15

    # Asserted Function Is Solution() 0.4
    has_solution_call = False
    if syntax_valid:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'solution':
                    has_solution_call = True
                    break
    
    if has_solution_call:
        score += 0.4
        
    return score


# ==========================================
# 3. SELF-PLAY TRAINING ORCHESTRATION
# ==========================================
def main():
    # Cleanup log file
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        pass

    # MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"    
    sandbox = DockerSandbox(timeout_seconds=3)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # Load shared or separate models for Proposer and Answerer
    print("Loading models...")
    proposer_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")
    answerer_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")

    # Initial Seed Prompts for Proposer
    seed_prompts = [
        "Create an algorithmic problem involving arrays and target sums.",
        "Create an algorithmic problem involving string reversal or parsing."
    ]

    # Common TRL configuration
    proposer_args = GRPOConfig(
        output_dir="grpo_proposer_output",
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,          # Group size (G) for GRPO
        max_completion_length=256,
        logging_steps=5,
        bf16=True,
        report_to="none"
    )

    answerer_args = GRPOConfig(
        output_dir="grpo_answerer_output",
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,          # Group size (G) for GRPO
        max_completion_length=256,
        logging_steps=5,
        bf16=True,
        report_to="none"
    )

    # --- ALTERNATING SELF-PLAY TRAINING LOOP ---
    num_iterations = 2
    for iteration in range(num_iterations):
        print(f"\n==============================")
        print(f"=== SELF-PLAY ITERATION {iteration + 1}/{num_iterations} ===")
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
                has_problem_tag = "[PROBLEM]" in comp_text
                has_tests_tag = "[TESTS]" in comp_text

                format_score = 0.0
                if has_problem_tag:
                    format_score += 0.5
                if has_tests_tag:
                    format_score += 0.5
                    
                prob_desc, test_code = parse_proposer_output(comp_text)
                
                # Reject malformed outputs by returning 0.0 reward
                if prob_desc is None or test_code is None:
                    rewards.append(0.0)
                    continue

                # Reward for correct assertion code
                assert_score = 0.0
                assert_score += validate_assertions(test_code)
                
                # Format requirements built into the answerer prompt
                text_prompt = tokenizer.apply_chat_template(get_answerer_prompt(prob_desc), tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text_prompt, return_tensors="pt").to(proposer_model.device)
                input_len = inputs["input_ids"].shape[1]
            
                with torch.no_grad():
                    out = answerer_model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)

                generated_tokens = out[0][input_len:]
            
                raw_solution = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                solution = clean_code_output(raw_solution)
                
                # Reward heuristic: Target problems that are challenging yet solvable (~0.5 pass score)
                score = sandbox.evaluate(solution, test_code)
                # edited below diff_reward as sandbox.evaluate implies higher reward better, not difficulty yet
                diff_reward = score

                total_reward = format_score * 0.2 + diff_reward * 0.6 + assert_score * 0.2

                # Log results to file
                log_rollout(
                    seed_prompt=prompt,
                    proposer_completion=comp_text,
                    answerer_completion=solution,
                    reward=total_reward,
                    metadata={"phase": "proposer", "format": format_score, "assert": assert_score, "eval_reward": score}
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
                out = proposer_model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)

            generated_tokens = out[0][input_len:]
            
            generated_q = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            prob_desc, test_code = parse_proposer_output(generated_q)
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
                solution = clean_code_output(raw_sol_text)
                
                score = sandbox.evaluate(solution, test_code)

                # Log results
                log_rollout(
                    seed_prompt=prompt,
                    proposer_completion=prob_desc,
                    answerer_completion=solution,
                    reward=score,
                    metadata={"phase": "answerer", "test_code": test_code}
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
            proposer_model.save_pretrained(f"./trained/{iteration+1}_proposer_model")
            answerer_model.save_pretrained(f"./trained/{iteration+1}_answerer_model")
            tokenizer.save_pretrained(f"./trained/{iteration+1}_tokenizer")
            print("Models saved successfully!")


    print("\nSelf-Play Reinforcement Training Completed Successfully!")

if __name__ == "__main__":
    main()