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
from prompts import get_answerer_prompt
from test_question_and_assert import list_of_questions
import random

LOG_FILE = "./solver_test_training_rollouts.jsonl"

# ==========================================
# 1. SECURE DOCKER SANDBOX
# ==========================================
class DockerSandbox:
    def __init__(self, image_name: str = "python:3.10-slim", timeout_seconds: int = 3):
        self.client = docker.from_env()
        self.image_name = image_name
        self.timeout = timeout_seconds

    def evaluate(self, solution_code: str, test_code: str) -> float:
        list_of_assert = test_code.split("\n")
        assert_stuff = ""
        for assert_str in list_of_assert:
            assert_stuff += self._assert_code_str_generation(assert_str)
        full_script = f"""\
{solution_code}

score = 0
{assert_stuff}

print(score)
exit(0)
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
            score = logs.strip()
            
            if result.get('StatusCode', -1) == 0:
                return int(score)
            return 0.0
        # except Exception:
        #     return 0.0
        finally:
            if container:
                try: container.remove(force=True)
                except Exception: pass
            if os.path.exists(temp_file_path):
                try: os.unlink(temp_file_path)
                except Exception: pass

    def _assert_code_str_generation(self, assert_str):
        return f"""
try:
    {assert_str}
    score += 1
except AssertionError:
    pass
"""


# ==========================================
# 2. UTILITIES
# ==========================================
def assert_code_parser(text: str) -> str:
    """
    Extracts pure Python code if enclosed in markdown code blocks,
    otherwise returns the stripped raw text.
    """
    pattern = r"```(?:python)?\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n".join(match.strip() for match in matches)
    return text.strip()


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


def log_rollout(answerer_completion, reward, metadata=None):
    """Appends a complete self-play rollout record (Answerer) to JSONL."""
    record = {
        "reward": reward,
        "answerer_completion": answerer_completion,
        "metadata": metadata or {},
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, indent=4) + "\n")

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

    # Load shared or separate models for Answerer
    print("Loading models...")
    answerer_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")

    # Common TRL configuration
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
    num_iterations = 3
    for iteration in range(num_iterations):
        print(f"\n==============================")
        print(f"=== SELF-PLAY ITERATION {iteration + 1}/{num_iterations} ===")
        print(f"==============================")

        for param in answerer_model.parameters():
            param.requires_grad = True

        prob_desc, test_code = random.choice(list_of_questions)

        def answerer_reward_func(prompts, completions, **kwargs):
            assert test_code is not None

            rewards = []
            for prompt, completion in zip(prompts, completions):
                raw_sol_text = completion[0]["content"] if isinstance(completion, list) else completion
                solution = solution_code_parser(raw_sol_text)
                
                score = sandbox.evaluate(solution, test_code)

                # Log results
                log_rollout(
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
            answerer_model.save_pretrained(f"./trained/{iteration+1}_answerer_model")
            tokenizer.save_pretrained(f"./trained/{iteration+1}_tokenizer")
            print("Models saved successfully!")


    print("\nSelf-Play Reinforcement Training Completed Successfully!")

if __name__ == "__main__":
    main()