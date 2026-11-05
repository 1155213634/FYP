import os
import json
import re
import tempfile
import docker
import torch
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

LOG_FILE = "training_rollouts.jsonl"


# ==========================================
# 1. SECURE DOCKER SANDBOX
# ==========================================
class DockerSandbox:
    def __init__(self, image_name: str = "python:3.10-slim", timeout_seconds: int = 3):
        self.client = None
        self.image_name = image_name
        self.timeout = timeout_seconds
        try:
            self.client = docker.from_env()
        except Exception as exc:
            print(
                f"Warning: Docker is unavailable; sandbox evaluation is disabled. Details: {exc}"
            )

    def evaluate(self, solution_code: str, test_code: str) -> float:
        if self.client is None:
            return 0.0

        full_script = f"""
try:
{self._indent_code(self._sanitize_code(solution_code))}
except Exception as e:
    print(f"Runtime Error: {{e}}")
    exit(1)

try:
{self._indent_code(self._sanitize_code(test_code))}
    print("ALL_TESTS_PASSED")
    exit(0)
except AssertionError as ae:
    print(f"Assertion Failed: {{ae}}")
    exit(1)
except Exception as e:
    print(f"Test Error: {{e}}")
    exit(1)
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(self._sanitize_code(full_script))
            temp_file_path = f.name

        # Get the actual random filename inside the mounted /app directory
        script_name = os.path.basename(temp_file_path)

        container = None
        try:
            container = self.client.containers.run(
                image=self.image_name,
                command=f"python /app/{script_name}",
                volumes={
                    os.path.dirname(temp_file_path): {"bind": "/app", "mode": "ro"}
                },
                working_dir="/app",
                network_mode="none",
                mem_limit="256m",
                cpu_period=100000,
                cpu_quota=50000,
                pids_limit=64,
                detach=True,
                remove=False,
            )
            result = container.wait(timeout=self.timeout)
            logs = container.logs(stdout=True, stderr=True).decode(
                "utf-8", errors="replace"
            )

            if result.get("StatusCode", -1) == 0 and "ALL_TESTS_PASSED" in logs:
                return 1.0
            return 0.0
        except Exception:
            return 0.0
        finally:
            if container:
                try:
                    container.remove(force=True)
                except Exception:
                    pass
            if os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass

    def _sanitize_code(self, code: str) -> str:
        if not isinstance(code, str):
            return ""
        return (
            code.replace("\x00", "").encode("utf-8", errors="replace").decode("utf-8")
        )

    def _indent_code(self, code: str, spaces: int = 4) -> str:
        return "\n".join(
            [
                " " * spaces + line if line.strip() else line
                for line in code.splitlines()
            ]
        )


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
    return problem_desc, test_code


def clean_code_output(text: str) -> str:
    """
    Extracts pure Python code if enclosed in markdown code blocks,
    otherwise returns the stripped raw text.
    """
    pattern = r"```(?:python)?\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text.strip()


def log_rollout(
    seed_prompt, proposer_completion, answerer_completion, reward, metadata=None
):
    """Appends a complete self-play rollout record (Proposer + Answerer) to JSONL."""
    record = {
        # "seed_prompt": seed_prompt,
        "proposer_completion": proposer_completion,
        "answerer_completion": answerer_completion,
        "reward": reward,
        "metadata": metadata or {},
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

    # MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"  # Small test
    sandbox = DockerSandbox(timeout_seconds=3)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # Load shared or separate models for Proposer and Answerer
    print("Loading models...")
    proposer_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto"
    )
    answerer_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto"
    )

    # Initial Seed Prompts for Proposer
    seed_prompts = [
        "Create an algorithmic problem involving arrays and target sums.",
        "Create an algorithmic problem involving string reversal or parsing.",
    ]

    # Common TRL configuration
    training_args = GRPOConfig(
        output_dir="grpo_self_play_output",
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,  # Group size (G) for GRPO
        max_completion_length=256,
        logging_steps=1,
        bf16=True,
        report_to="none",
    )

    # --- ALTERNATING SELF-PLAY TRAINING LOOP ---
    num_iterations = 3
    for iteration in range(num_iterations):
        print(f"\n==============================")
        print(f"=== SELF-PLAY ITERATION {iteration + 1}/{num_iterations} ===")
        print(f"==============================")

        current_seed = seed_prompts[iteration % len(seed_prompts)]

        # ----------------------------------------------------
        # PHASE 1: Train Proposer (Freeze Answerer)
        # ----------------------------------------------------
        print("\n--- Phase 1: Training Proposer ---")
        for param in answerer_model.parameters():
            param.requires_grad = False
        for param in proposer_model.parameters():
            param.requires_grad = True

        def proposer_reward_func(prompts, completions, **kwargs):
            rewards = []
            answerer_model.eval()
            for prompt, completion in zip(prompts, completions):
                comp_text = (
                    completion[0]["content"]
                    if isinstance(completion, list)
                    else completion
                )
                prob_desc, test_code = parse_proposer_output(comp_text)

                # Reject malformed outputs by returning 0.0 reward
                if prob_desc is None or test_code is None:
                    rewards.append(0.0)
                    continue

                # Format requirements built into the answerer prompt
                ans_prompt = (
                    f"Write a Python function to solve this problem:\n{prob_desc}\n\n"
                    "FORMAT REQUIREMENT: Output ONLY valid Python code. Do not include any conversational filler."
                )
                inputs = tokenizer(ans_prompt, return_tensors="pt").to(
                    answerer_model.device
                )
                input_len = inputs["input_ids"].shape[1]

                with torch.no_grad():
                    generated_output = answerer_model.generate(
                        **inputs, max_new_tokens=256, do_sample=True, temperature=0.7
                    )

                generated_tokens = generated_output[0][input_len:]

                raw_solution = tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )
                solution = clean_code_output(raw_solution)
                print(f"Answerer Solution:\n{solution}")

                # Reward heuristic: Target problems that are challenging yet solvable (~0.5 pass score)
                score = sandbox.evaluate(solution, test_code)
                diff_reward = 1.0 - abs(score - 0.5) * 2

                # Log results to file
                log_rollout(
                    seed_prompt=prompt,
                    proposer_completion=comp_text,
                    answerer_completion=solution,
                    reward=diff_reward,
                    metadata={"phase": "proposer", "execution_score": score},
                )

                rewards.append(diff_reward)
            return rewards

        proposer_dataset = Dataset.from_dict({"prompt": [current_seed]})
        proposer_trainer = GRPOTrainer(
            model=proposer_model,
            reward_funcs=proposer_reward_func,
            args=training_args,
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

        # Explicit format requirements in Proposer system prompt
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert programming problem creator. "
                    "FORMAT REQUIREMENT: You must always output problems using the exact tags: "
                    "[PROBLEM] followed by the description, and [TESTS] followed by executable Python assertions."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Create an algorithmic coding problem based on this topic:\nTopic: {current_seed}\n\n"
                    "Strictly adhere to this layout:\n"
                    "[PROBLEM] <Detailed problem description>\n"
                    "[TESTS] <assert test_function(args) == expected>"
                ),
            },
        ]

        prob_desc, test_code = None, None
        max_retries = 3

        for attempt in range(max_retries):
            text_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text_prompt, return_tensors="pt").to(
                proposer_model.device
            )

            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                out = proposer_model.generate(
                    **inputs, max_new_tokens=256, do_sample=True, temperature=0.7
                )

            generated_tokens = out[0][input_len:]

            generated_q = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(f"Proposer Question:\n{generated_q}")  # TEMP

            prob_desc, test_code = parse_proposer_output(generated_q)
            if prob_desc is not None and test_code is not None:
                break
            print(
                f"Attempt {attempt + 1}: Proposer generated malformed output. Retrying generation..."
            )

        if prob_desc is None or test_code is None:
            print(
                "Error: Proposer failed to follow format after multiple retries. Skipping this iteration's Answerer training."
            )
            continue

        # Explicit format requirement added to the Answerer prompt
        answerer_prompt = (
            f"Write a Python function to solve this problem:\n{prob_desc}\n\n"
            "FORMAT REQUIREMENT: Output ONLY valid Python code. Do not include any conversational filler."
        )

        def answerer_reward_func(prompts, completions, **kwargs):
            assert test_code is not None

            rewards = []
            for prompt, completion in zip(prompts, completions):
                raw_sol_text = (
                    completion[0]["content"]
                    if isinstance(completion, list)
                    else completion
                )
                solution = clean_code_output(raw_sol_text)

                score = sandbox.evaluate(solution, test_code)

                # Log results
                log_rollout(
                    seed_prompt=prompt,
                    proposer_completion=prob_desc,
                    answerer_completion=solution,
                    reward=score,
                    metadata={"phase": "answerer", "test_code": test_code},
                )

                rewards.append(score)
            return rewards

        answerer_dataset = Dataset.from_dict({"prompt": [answerer_prompt]})
        answerer_trainer = GRPOTrainer(
            model=answerer_model,
            reward_funcs=answerer_reward_func,
            args=training_args,
            train_dataset=answerer_dataset,
        )
        answerer_trainer.train()

    print("\nSelf-Play Reinforcement Training Completed Successfully!")


if __name__ == "__main__":
    main()
