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
                "Reply with EXACTLY one word: valid or invalid."
            )
        },
        {
            "role": "user",
            "content": (
                "Problem: Write a function that returns True if even, else False."
                "Assert: solution(4) == True"
            )
        },
        {
            "role": "assistant",
            "content": "valid"
        },
        {
            "role": "user",
            "content": (
                "Problem: Write a function that returns True if prime, else False."
                "Assert: solution(4) == True"
            )
        },
        {
            "role": "assistant",
            "content": "invalid"
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