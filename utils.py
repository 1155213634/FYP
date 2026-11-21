import re
import ast
import torch
import json
import builtins

def clean_code_output(text: str) -> str:
    """
    Extracts pure Python code if enclosed in markdown code blocks,
    and strips stray leading/trailing fences when prompts pre-open a block.
    """
    text = text.strip()
    
    # 1. Complete fenced block(s): ```python ... ```
    pattern = r"```(?:python)?\s*\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return "\n".join(match.strip() for match in matches)
    
    # 2. Stray opening fence (e.g., prompt didn't open it, model didn't close it)
    text = re.sub(r"^```(?:python)?\s*\n?", "", text, flags=re.IGNORECASE)
    
    # 3. Stray closing fence (e.g., prompt ended with ```python, model ended with ```)
    text = re.sub(r"\n?```\s*$", "", text)
    
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
    print("assert_code_parser has been depreciated. Please switch to assertions_to_solutions.")
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
    print("solution_code_parser is depreciated. Please use ensure_solution_function")
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


def get_assert_lines_codeelo(test_cases):
    def split_lines(text):
        return text.splitlines()

    inputs = test_cases[0][0]
    input_text = split_lines(inputs)
    print(input_text)
    test_length = int(input_text.pop(0))

    outputs = test_cases[0][1]
    output_text = split_lines(outputs)

    result = []
    for i in range(test_length):
        line = f"assert f('{input_text[i]}') == '{output_text[i]}'"
        result.append(line)
    
    return '\n'.join(result)


def get_assert_lines_LeetCodeDataset(test_cases):
    def split_lines(text):
        return text.splitlines()

    inputs = test_cases[0][0]
    input_text = split_lines(inputs)
    print(input_text)
    test_length = int(input_text.pop(0))

    outputs = test_cases[0][1]
    output_text = split_lines(outputs)

    result = []
    for i in range(test_length):
        line = f"assert f('{input_text[i]}') == '{output_text[i]}'"
        result.append(line)
    
    return '\n'.join(result)


def _replace_span(
    lines: list[str],
    start_line: int,
    start_col: int,
    end_line: int,
    end_col: int,
    replacement: str,
) -> None:
  """Replace a UTF-8 byte span across `lines` (1-based line numbers, 0-based byte cols)."""
  start_idx = start_line - 1
  end_idx = end_line - 1

  start_bytes = lines[start_idx].encode("utf-8")
  end_bytes = lines[end_idx].encode("utf-8")

  prefix = start_bytes[:start_col]
  suffix = end_bytes[end_col:]
  rep_bytes = replacement.encode("utf-8")

  if start_idx == end_idx:
    lines[start_idx] = (prefix + rep_bytes + suffix).decode("utf-8")
  else:
    lines[start_idx] = (prefix + rep_bytes).decode("utf-8")
    for idx in range(start_idx + 1, end_idx):
      lines[idx] = ""
    lines[end_idx] = suffix.decode("utf-8")


def _collect_top_level_calls(
    node: ast.AST | None,
    target_name: str,
    builtin_names: set[str],
    spans: list[tuple[int, int, int, int, str]],
) -> None:
  """Collect highest-level non-builtin `ast.Name` calls without descending past them."""
  if node is None:
    return

  if isinstance(node, ast.Call):
    if isinstance(node.func, ast.Name):
      name = node.func.id
      if name != target_name and name not in builtin_names:
        start_line = node.func.lineno
        start_col = node.func.col_offset
        end_line = node.func.end_lineno or start_line
        end_col = (
            node.func.end_col_offset
            if node.func.end_col_offset is not None
            else start_col + len(name.encode("utf-8"))
        )
        spans.append((start_line, start_col, end_line, end_col, target_name))
        # Stop here so nested calls inside a replaced non-builtin call aren't altered.
        return

    # If the call is a builtin (or not a direct non-builtin Name), descend into
    # its arguments/keywords/subexpressions to find the highest-level non-builtin calls.
    for child in ast.iter_child_nodes(node):
      _collect_top_level_calls(child, target_name, builtin_names, spans)
    return

  for child in ast.iter_child_nodes(node):
    _collect_top_level_calls(child, target_name, builtin_names, spans)


def assertions_to_solution(
    assert_code: str,
    target_name: str = "solution",
    *,
    skip_builtins: bool = True,
) -> str:
  """Rewrite top-level function calls inside `assert` statements to `target_name`."""
  tree = ast.parse(assert_code)
  builtin_names = set(dir(builtins)) if skip_builtins else set()
  replacements: list[tuple[int, int, int, int, str]] = []

  for node in ast.walk(tree):
    if isinstance(node, ast.Assert):
      _collect_top_level_calls(node.test, target_name, builtin_names, replacements)
      _collect_top_level_calls(node.msg, target_name, builtin_names, replacements)

  replacements.sort(key=lambda r: (r[0], r[1]), reverse=True)

  lines = assert_code.splitlines(keepends=True)
  for span in replacements:
    _replace_span(lines, *span)

  return "".join(lines)


def ensure_solution_function(
    code: str, func_name: str = "solution", *, include_async: bool = True
) -> str:
  """Ensure a function named `func_name` exists; if not, rename the first function (Python 3.8 compatible)."""
  tree = ast.parse(code)
  target_types = (
      (ast.FunctionDef, ast.AsyncFunctionDef)
      if include_async
      else (ast.FunctionDef,)
  )

  first_func = None
  for node in ast.walk(tree):
    if isinstance(node, target_types):
      if node.name == func_name:
        return code
      if first_func is None:
        first_func = node

  if first_func is None:
    return code

  # AST line numbers are 1-based; col_offset is a UTF-8 byte offset on that line.
  lines = code.splitlines(keepends=True)
  line_idx = first_func.lineno - 1
  line_bytes = lines[line_idx].encode("utf-8")

  # Slice starting at the `def` (or `async def`) keyword so decorators/indentation stay intact.
  prefix_bytes = line_bytes[: first_func.col_offset]
  rest_bytes = line_bytes[first_func.col_offset :]

  old_name_bytes = first_func.name.encode("utf-8")
  new_name_bytes = func_name.encode("utf-8")

  # Replace the first occurrence of the function name after `def` / `async def` on that line.
  updated_rest_bytes = rest_bytes.replace(old_name_bytes, new_name_bytes, 1)
  lines[line_idx] = (prefix_bytes + updated_rest_bytes).decode("utf-8")

  return "".join(lines)