from test import test_model_performance

i = 0
max_steps = 50
step_size = 10
MODEL_ID = "stabilityai/stable-code-instruct-3b"

while i <= max_steps:
    test_model_performance(i, MODEL_ID)
    i += step_size