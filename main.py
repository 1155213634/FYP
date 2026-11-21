import json
import re
import torch
import ast
from pathlib import Path
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from prompts import get_proposer_prompt, get_answerer_prompt, get_judge_prompt
from docker_code import DockerSandbox
from utils import clean_code_output, parse_proposer_output, validate_assertions, assert_code_parser, solution_code_parser
from utils import check_gpu_mem, judge_response_parser, assertions_to_solution, ensure_solution_function

# ==========================================
# 0. CONSTANTS
# ==========================================
# MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
# MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
# MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
# MODEL_ID = "stabilityai/stable-code-instruct-3b"
# MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

USE_JUDGE = False
JUDGE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

DO_LOGGING = True
LOG_PERIOD = 10
LOG_FILE = f"./results/{MODEL_ID}/training_rollouts{"_judge" if USE_JUDGE else ''}.jsonl"

LEARN_RATE = 5e-6
MAX_ITER = 50                      # No. of total loops
GROUP_SIZE = 4                     # GRPO group size

ANS_ATTEMPTS_PROP_TRAIN = 4        # No. of answerer attempts to calculate proposer difficulty score
ANS_TOKENS_PROP_TRAIN = 256        # No. of max tokens for answerer for training proposer
ANS_TEMP_PROP_TRAIN = 0.7          # No. of max tokens for answerer for training proposer

PROP_TOKENS_ANS_TRAIN = 256        # No. of max tokens for proposer for training answerer
PROP_TEMP_ANS_TRAIN = 0.7          # No. of max tokens for proposer for training answerer

JUDGE_TOKENS = 1024

QUANTIZATION = False
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# ==========================================
# 2. UTILITIES
# ==========================================

def log_rollout(proposer_completion, answerer_completion, reward, metadata=None, seed_prompt=None):
    """Appends a complete self-play rollout record (Proposer + Answerer) to JSONL."""
    if not DO_LOGGING:
        return
    record = {
        # "seed_prompt": seed_prompt,
        "reward": reward,
        "metadata": metadata or {},
        "proposer_completion": proposer_completion,
        "answerer_completion": answerer_completion
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_trainable_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()},
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.print_trainable_parameters()
    return model

# ==========================================
# 3. SELF-PLAY TRAINING ORCHESTRATION
# ==========================================
def main():
    # Cleanup log file
    file_path = Path(LOG_FILE)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        pass

    sandbox = DockerSandbox(timeout_seconds=3)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # Load shared or separate models for Proposer and Answerer
    print("Loading models...")
    if QUANTIZATION:
        proposer_model = load_trainable_model()
        answerer_model = load_trainable_model()
    else:
        proposer_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        answerer_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    check_gpu_mem()
    if USE_JUDGE:
        judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_ID)
        judge_quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16, # or torch.float16 depending on your GPU
            bnb_4bit_quant_type="nf4",             # Normalized Float 4 (best for performance)
            bnb_4bit_use_double_quant=True,        # Nested quantization to save even more memory
        )
        judge_model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL_ID, quantization_config=judge_quantization_config, dtype=torch.bfloat16, device_map="auto")
        judge_model.eval()
    else:
        judge_model = None
        judge_tokenizer = None

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
        report_to="none",
        top_p=0.95,
    )

    proposer_prompt = tokenizer.apply_chat_template(
        get_proposer_prompt(0), tokenize=False, add_generation_prompt=True
    )
    proposer_dataset = Dataset.from_dict({"prompt": [proposer_prompt]})

    answerer_args = GRPOConfig(
        output_dir="grpo_answerer_output",
        learning_rate=LEARN_RATE,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=GROUP_SIZE,          # Group size (G) for GRPO
        max_completion_length=256,
        temperature=1.0,
        logging_steps=5,
        bf16=True,
        report_to="none"
    )

    answerer_prompt = tokenizer.apply_chat_template(
        get_answerer_prompt(""), tokenize=False, add_generation_prompt=True
    )

    answerer_dataset = Dataset.from_dict({"prompt": [answerer_prompt]})

    check_gpu_mem()

    # utility functions
    def logical_assertion_reward_func(prob_desc, test_code):
        text_prompt = judge_tokenizer.apply_chat_template(get_judge_prompt(prob_desc, test_code),
                                                            tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text_prompt, return_tensors="pt").to(judge_model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = judge_model.generate(
                **inputs,
                max_new_tokens=JUDGE_TOKENS,
                do_sample=False,
            )
        
        output_ids = outputs[0][input_len:]
        raw_output = tokenizer.decode(output_ids, skip_special_tokens=True)
        thinking, verdict = judge_response_parser(raw_output)

        score = 0.0
        if verdict is not None:
            if "invalid" in verdict:
                score = -1.0
            elif "valid" in verdict:
                score = 1.0

        return score, thinking, verdict, raw_output

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
                
                if i % LOG_PERIOD == 0: 
                    log_rollout(
                    seed_prompt=None,
                    proposer_completion=None,
                    answerer_completion=None,
                    reward=0.0,
                    metadata={"phase": "proposer", "failed": True}
                    )
                continue

            # Reward for correct assertion code
            assert_score = 0.0
            try:
                test_code = assertions_to_solution(test_code)
            except Exception as exc:
                assert_score = -1.0

            assert_score += validate_assertions(test_code)

            # Assert judge
            judge_score = 0.0
            thinking = ""
            verdict = ""
            judge_raw_output = ""
            if assert_score > 0.9 and USE_JUDGE:
                judge_score, thinking, verdict, judge_raw_output = logical_assertion_reward_func(prob_desc, test_code)
            
            diff_reward = 0.0
            solution = None
            avg_score = -1
            if assert_score > 0.9 and (judge_score > 0.9 or not USE_JUDGE):
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

                    can_compile = True
                    try:
                        solution = ensure_solution_function(solution)
                    except Exception as exc:
                        can_compile = False
                    
                    # Reward heuristic: Target problems that are challenging yet solvable (~0.5 pass score)
                    score = 0.0
                    if can_compile:
                        score = sandbox.evaluate(solution, test_code)
                    scores.append(score)

                avg_score = sum(scores) / len(scores)
                diff_reward = 1.0 - abs(avg_score - 0.5) * 2

            if USE_JUDGE:
                total_reward = format_score * 0.05 + assert_score * 0.05 + judge_score * 0.3 + diff_reward * 0.6  
            else:
                total_reward = format_score * 0.2 + diff_reward * 0.6 + assert_score * 0.2

            # Log results to file
            if i % LOG_PERIOD == 0: 
                log_rollout(
                    seed_prompt=prompt,
                    proposer_completion=comp_text,
                    answerer_completion=solution,
                    reward=round(total_reward, 2),
                    metadata={"phase": "proposer", "format": round(format_score,2), 
                                "assert": round(assert_score, 2), "ans_avg":round(avg_score,2), "judge":round(judge_score,2), "judge_output": (thinking, verdict, judge_raw_output)}
                )

            rewards.append(total_reward)
        return rewards


    def answerer_reward_func(prompts, completions, **kwargs):
        assert test_code is not None

        rewards = []
        for prompt, completion in zip(prompts, completions):
            raw_sol_text = completion[0]["content"] if isinstance(completion, list) else completion

            solution = ""
            can_compile = True
            try:
                solution = ensure_solution_function(raw_sol_text)
            except Exception as exc:
                can_compile = False
            
            if can_compile:
                score = sandbox.evaluate(solution, test_code)
            else:
                score = 0.0

            # Log results
            if i % LOG_PERIOD == 0: 
                log_rollout(
                    proposer_completion=generated_q,
                    answerer_completion=solution,
                    reward=score,
                    metadata={"iteration" : iteration, "phase": "answerer", "problem_description": prob_desc, "test_code": test_code, "raw_sol_text": raw_sol_text}
                )

            rewards.append(score)
        return rewards

    # initialize trainers
    proposer_trainer = GRPOTrainer(
        model=proposer_model,
        reward_funcs=proposer_reward_func,
        args=proposer_args,
        train_dataset=proposer_dataset,
    )

    answerer_trainer = GRPOTrainer(
        model=answerer_model,
        reward_funcs=answerer_reward_func,
        args=answerer_args,
        train_dataset=answerer_dataset,
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
        if not QUANTIZATION:
            for param in answerer_model.parameters():
                param.requires_grad = False
            for param in proposer_model.parameters():
                if param.is_floating_point() or param.is_complex():
                    param.requires_grad = True
            
        proposer_prompt = tokenizer.apply_chat_template(
            get_proposer_prompt(iteration), tokenize=False, add_generation_prompt=True
        )

        proposer_trainer.train_dataset = Dataset.from_dict({"prompt": [proposer_prompt]})
        proposer_trainer.train()

        # ----------------------------------------------------
        # PHASE 2: Train Answerer (Freeze Proposer)
        # ----------------------------------------------------
        print("\n--- Phase 2: Training Answerer ---")
        if not QUANTIZATION:
            for param in proposer_model.parameters():
                param.requires_grad = False
            for param in answerer_model.parameters():
                if param.is_floating_point() or param.is_complex():
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
            if prob_desc is None or test_code is None:
                continue
            try:
                test_code = assertions_to_solution(test_code)
            except Exception as exc:
                pass

            if validate_assertions(test_code) <= 0:
                prob_desc, test_code = None, None
            if prob_desc is not None and test_code is not None:
                break
            print(f"Attempt {attempt + 1}: Proposer generated malformed output. Retrying generation...")


        if prob_desc is None or test_code is None:
            print("Error: Proposer failed to follow format after multiple retries. Skipping this iteration's Answerer training.")
            continue

        answerer_prompt = tokenizer.apply_chat_template(
            get_answerer_prompt(prob_desc), tokenize=False, add_generation_prompt=True
        )

        answerer_trainer.train_dataset = Dataset.from_dict({"prompt": [answerer_prompt]})
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