def get_proposer_prompt(iter):
    # Initial Seed Prompts for Proposer
    seed_prompts = [
        "Create an algorithmic problem involving arrays and target sums.",
        "Create an algorithmic problem involving string reversal or parsing.",
        "Create an algorithmic problem involving number validations."
    ]

    current_seed = seed_prompts[iter % len(seed_prompts)]

    proposer_messages = [
        {
            "role": "system",
            "content": (
                "You are an expert programming problem creator. "
                "You must always structure your response using exact tags: "
                "[PROBLEM] followed by the problem description, and "
                "[TESTS] followed by a Python code block containing up to 3 assertions "
                "that call `solution(...)`."
                """
### Evaluation Criteria:
- Executability, your assert statements should be executable
- Correctness, your assert statements should be correct for the problem proposed"
"""
            )
        },
        # Few-shot example as an actual conversation turn
        {
            "role": "user",
            "content": "Create an algorithmic coding problem based on this topic:\nTopic: string reversal"
        },
        {
            "role": "assistant",
            "content": (
                "[PROBLEM]\n"
                "Write a function `solution(s)` that takes a string `s` and returns the reversed string.\n\n"
                "[TESTS]\n"
                "```python\n"
                "assert solution(\"hello\") == \"olleh\"\n"
                "assert solution(\"world\") == \"dlrow\"\n"
                "```"
            )
        },
        {
            "role": "user",
            "content": "Create an algorithmic coding problem based on this topic:\nTopic: array sum"
        },
        {
            "role": "assistant",
            "content": (
                "[PROBLEM]\n"
                "Write a function `solution(s)` that takes an array `l` and returns the sum of all elements in the array.\n\n"
                "[TESTS]\n"
                "```python\n"
                "assert solution([1, 2, 3]) == 6\n"
                "assert solution([3, 3, 3]) == 9\n"
                "assert solution([0]) == 0\n"
                "```"
            )
        },
        # Actual user prompt for the current iteration
        {
            "role": "user",
            "content": (
                f"Create an algorithmic coding problem based on this topic:\nTopic: {current_seed}\n\n"
                "Strictly follow the [PROBLEM] and [TESTS] tag format. Only up to 3 assertions allowed."
            )
        }
    ]
    
    return proposer_messages


def get_answerer_prompt(problem_desc):
    answerer_messages = [
            {
                "role": "system", 
                "content": ( """
### Task: Create a Python Code Snippet to solve provided problem

### Code Requirements:
- Name the entry function `f` (e.g., `def f(...): ...`)
- Ensure the function returns a value
- Include at least one input parameter
- Make the function deterministic
- AVOID THE FOLLOWING:
  * Random functions or variables
  * Date/time operations
  * I/O operations (reading files, network requests)
  * Printing or logging
  * Any external state
  * Doc strings
- Ensure execution completes within 10 seconds on a modern CPU
- All imports and class definitions should be at the very top of the code snippet
- The snippet should end with a return statement from the main function `f(...)`, DO NOT ADD ANYTHING AFTER FINAL RETURN STATEMENT"
- Add comments using # to support logic flow of code, AVOID chat response style output
- Do not return prose or example print calls.

### Formatting:
- Format your code with:
  def f(...):
      # your code here
      return ...
  
### Example Format:
def f(name: str, info: dict):
    # code logic here
    return result

### Evaluation Criteria:
- Executability, your code should be executable when given arguments into function f(...)
- Correctness, your code should be able to solve the given problem
"""
                )
            },
            {
            "role": "user",
            "content": "Write a Python function to solve this problem: string reversal\n"
            },
                    {
            "role": "assistant",
            "content": (
                "def f(string):\n"
                "    s = string[::-1]\n"
                "    return s"
            )
            },
            {
            "role": "user",
            "content": "Write a Python function to solve this problem: array sum\n"
            },
            {
            "role": "assistant",
            "content": (
                "def f(array):\n"
                "    sum = 0\n"
                "    for value in array:\n"
                "        sum += value\n"
                "    return sum"
            )
            },
            {
                "role": "user", 
                "content": (
                    f"Write a Python function to solve this problem:\n{problem_desc}\n\n"
                )
            }
        ]

    return answerer_messages


def get_judge_prompt(problem, assertion):
    judge_messages = [
        {
            "role": "system",
            "content": (
                "You are a strict Python test case validator. Determine if the assert statement logically aligns with the described problem." 
                "You must always structure your response using exact tags: "
                "[THINK] followed by the reasoning on validity of assert statements, and "
                "[JUDGE] followed by a EXACTLY one word: valid or invalid"
            )
        },
        {
            "role": "user",
            "content": (
                "Problem: Write a function that returns True if even, else False."
                "Assert: assert f(4) == True"
            )
        },
        {
            "role": "assistant",
            "content": (
                "[THINK] Each assert statement checks if the function returns True if the input number is even.\n For assert 1, it claims that input number 4 returns True, that means input number 4 is even.\n The statement is valid"
                "[JUDGE] valid"
            )
        },
        {
            "role": "user",
            "content": (
                "Problem: Write a function that returns True if prime, else False."
                "Assert: assert f(4) == True"
            )
        },
        {
            "role": "assistant",
            "content": (
                "[THINK] Each assert statement checks if the function returns True if the input number is prime.\n For assert 1, it claims that input number 4 returns True, that means input number 4 is prime.\n The statement is invalid"
                "[JUDGE] valid"
            )
        },
        {
            "role": "user",
            "content": (
                f"Problem: {problem}\n"
                f"Assert: {assertion}"
            )
        }
    ]

    return judge_messages


def get_human_eval_answerer_prompt(problem_prompt: str, entry_point: str):
    return [
        {
            "role": "system",
            "content": (
                "You solve HumanEval programming problems. "
                "Return only complete, valid Python source code—no Markdown fences "
                "or explanatory prose. Include a standalone top-level function "
                "definition using the required function name. Preserve the provided "
                "signature when one is present. Helper functions and necessary "
                "imports are allowed."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Implement the HumanEval problem below.\n"
                f"The required entry-point function name is `{entry_point}`.\n"
                "The output will be evaluated as standalone source code, so include "
                "the complete function definition.\n\n"
                f"{problem_prompt}"
            ),
        },
    ]


def get_codeelo_answerer_prompt(problem_desc):
    answerer_messages = [
            {
                "role": "system", 
                "content": ( """
### Task: Create a Python Code Snippet to solve provided problem

### Code Requirements:
- Name the entry function `f` (e.g., `def f(...): ...`)
- Ensure the function returns only one string
- Include only one input parameter
- Make the function deterministic
- AVOID THE FOLLOWING:
  * Random functions or variables
  * Date/time operations
  * I/O operations (reading files, network requests)
  * Printing or logging
  * Any external state
  * Doc strings
- Ensure execution completes within 10 seconds on a modern CPU
- All imports and class definitions should be at the very top of the code snippet
- The snippet should end with a return statement from the main function `f(...)`, DO NOT ADD ANYTHING AFTER FINAL RETURN STATEMENT"
- Add comments using # to support logic flow of code, AVOID chat response style output
- Do not return prose or example print calls.
- Function arguments are in the form of strings such as "1 9". Return values should also be strings such as "10 10"

### Formatting:
- Format your code with:
  def f(...):
      # your code here
      return ...
  
### Example Format:
def f(input:str):
    # code logic here
    return result

### Evaluation Criteria:
- Executability, your code should be executable when given arguments into function f(...)
- Correctness, your code should be able to solve the given problem
"""
                )
            },
            {
            "role": "user",
            "content": "Write a Python function to solve this problem: string reversal\n"
            },
                    {
            "role": "assistant",
            "content": (
                "def f(string):\n"
                "    s = string[::-1]\n"
                "    return s"
            )
            },
            {
            "role": "user",
            "content": "Write a Python function to solve this problem: array sum\n"
            },
            {
            "role": "assistant",
            "content": (
                "def f(input_str):\n"
                "    array = input_str.split(' ')"
                "    sum = 0\n"
                "    for value in array:\n"
                "        sum += value\n"
                "    return str(sum)"
            )
            },
            {
                "role": "user", 
                "content": (
                    f"Write a Python function to solve this problem:\n{problem_desc}\n\n"
                )
            }
        ]

    return answerer_messages