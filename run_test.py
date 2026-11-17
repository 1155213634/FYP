from test import test_model_performance

i = 0
max_steps = 70
step_size = 10
MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
testset = "mbppplus"

while i <= max_steps:
    test_model_performance(i, MODEL_ID, testset)
    i += step_size