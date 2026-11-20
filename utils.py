import re
import ast
import torch

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
        return 0.0

    # Needs to Assert 0.4
    if "assert" in test_code_str:
        score += 0.4

    # Asser Less Than Four 0.3
    assertion_count = 0
    if syntax_valid:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                assertion_count += 1
    
    if assertion_count < 4:
        score += 0.3
        
    return score

def assert_code_parser(code: str) -> str:
    # for training (know would not allow nested function in asserts)
    assert_not_pattern = r"(?m)^[ \t]*assert not[ \t]+([A-Za-z_]\w*)[ \t]*\("
    assert_pattern = r"(?m)^[ \t]*assert[ \t]+([A-Za-z_]\w*)[ \t]*\("
    asserts = code.split("\n")
    flags = re.DOTALL
    temp_list = []
    for assert_statement in asserts:
        if re.match(assert_not_pattern, assert_statement, flags=flags):
            temp_list.append(re.sub(assert_not_pattern, "assert not f(", assert_statement, flags=flags))
        else:
            temp_list.append(re.sub(assert_pattern, "assert f(", assert_statement, flags=flags))
    return "\n".join(assert_statement.strip() for assert_statement in temp_list)

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

def judge_response_parser(text: str) -> str | None:
    if "[THINK]" not in text or "[JUDGE]" not in text:
        return None, None

    parts = text.split("[JUDGE]")
    thinking = parts[0].replace("[THINK]", "").strip()
    validate = parts[1].strip()
    return thinking, validate

def human_eval_solution_parser(text: str, entry_point: str) -> str:
    match = re.search(
        r"```(?:python|py)?[ \t]*\r?\n(.*?)```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    code = match.group(1).strip() if match else text.strip()

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Generated solution is invalid Python: {exc}") from exc

    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if entry_point not in functions:
        raise ValueError(
            f"Generated solution does not define required entry point {entry_point!r}"
        )

    return code


def check_gpu_mem():
    # Ensure CUDA is available
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        
        # Get total VRAM on the GPU
        total_vram = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
        
        # Get VRAM currently allocated by PyTorch tensors (models, activations)
        allocated_vram = torch.cuda.memory_allocated(device_id) / (1024**3)
        
        # Get VRAM managed by PyTorch's caching allocator (reserved but maybe unused)
        reserved_vram = torch.cuda.memory_reserved(device_id) / (1024**3)
        
        # Calculate true free space
        free_vram = total_vram - reserved_vram

        print(f"--- VRAM Status Report ---")
        print(f"Total GPU VRAM:       {total_vram:.2f} GB")
        print(f"Actively Allocated:   {allocated_vram:.2f} GB (Proposer + Answerer weights/tensors)")
        print(f"Reserved by PyTorch:  {reserved_vram:.2f} GB")
        print(f"Truly Free VRAM:      {free_vram:.2f} GB")
    else:
        print("CUDA is not available. Running on CPU.")