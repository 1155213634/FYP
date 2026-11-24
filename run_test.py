from test_humanevalplus import test_model_performance_humanevalplus
# from test_codeelo import test_model_performance_codeelo
from test_leetcode import test_model_performance_LeetCodeDataset
from test_mbppplus import test_model_performance_mbppplus
from test_ame import test_model_performance_ame
import traceback

starting = 10
max_steps = 60
step_size = 10
MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
QLoRA = False
Judged = False

for func in [test_model_performance_humanevalplus, test_model_performance_LeetCodeDataset, test_model_performance_mbppplus, test_model_performance_ame]:
    i = starting
    while i <= max_steps:
        try:
            func(i, MODEL_ID, QLoRA, Judged)
        except Exception as exc:
            print(f"Iteration {i} failed: {exc}")
            traceback.print_exc()
        finally:
            i += step_size