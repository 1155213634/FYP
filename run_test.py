from test import test_model_performance
from test_codeelo import test_model_performance_codeelo
from test_leetcode import test_model_performance_LeetCodeDataset
import traceback

i = 0
max_steps = 50
step_size = 10
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
QLoRA = True
Judged = False

while i <= max_steps:
    try:
        test_model_performance_LeetCodeDataset(i, MODEL_ID, QLoRA, Judged)
    except Exception as exc:
        print(f"Iteration {i} failed: {exc}")
        traceback.print_exc()
    finally:
        i += step_size