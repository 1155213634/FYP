from test_humanevalplus import test_model_performance_humanevalplus
from test_codeelo import test_model_performance_codeelo
from test_leetcode import test_model_performance_LeetCodeDataset
import traceback

i = 50
max_steps = 50
step_size = 10
MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
QLoRA = True
Judged = True

while i <= max_steps:
    try:
        test_model_performance_humanevalplus(i, MODEL_ID, QLoRA, Judged)
    except Exception as exc:
        print(f"Iteration {i} failed: {exc}")
        traceback.print_exc()
    finally:
        i += step_size