from test import test_model_performance

i = 0
max_steps = 50
step_size = 10
MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
QLoRA = True
Judged = True

while i <= max_steps:
    test_model_performance(i, MODEL_ID, QLoRA, Judged)
    i += step_size