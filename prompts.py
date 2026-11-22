def get_proposer_prompt(iter, previous_question_1=None, previous_question_2=None):
    # Initial Seed Prompts for Proposer
    seed_prompts = [
    # 1. Sequence & Array Logic
    "two pointers and array partitioning",
    "prefix sums and subarray properties",
    "sliding window over arrays or strings",

    # 2. String Manipulation & Parsing
    "string parsing and pattern matching",
    "run-length encoding and string compression",

    # 3. Intervals & Ordering
    "interval merging and coverage",
    "custom sorting order and rank transformations",

    # 4. Hash Tables & Frequencies
    "frequency counting and majority element logic",
    "hash mapping and duplicate tracking",

    # 5. Algorithmic Decision Making
    "greedy choice and local optimization",
    "binary search on answer ranges",
    "1D dynamic programming and state transitions",

    # 6. Extras
    "grid movement with directional turns, obstacles, or energy costs",
    "resource allocation or budget-constrained selection",
]

    current_seed = seed_prompts[iter % len(seed_prompts)]

    if previous_question_1 is not None and previous_question_2 is not None:
        proposer_messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert programming problem creator. "
                    "Make the new problem distinct in logic and structure from the two "
                    "previous problems shown in this conversation, without making it overly "
                    "obscure or ambiguous."
                    "You must always structure your response using exact tags: "
                    "[PROBLEM] followed by the problem description, and "
                    "[TESTS] followed by a Python code block containing up to 3 assertions "
                    "that call `solution(...)`. No other code is allowed other than assertions."
                    f"""
### Evaluation Criteria:
- Executability, your assert statements should be executable
- Correctness, your assert statements should be correct for the problem proposed
- Difficulty in solving the coding problem. Focus on either algorithmic reasoning or logic complexity.
- Creativity, the problem needs to be sufficiently different from the provided reference problem
"""
                )
            },
            # Reference problem
            {
                "role": "user",
                "content": "Create an algorithmic coding problem based on this topic:\nTopic: Create any algorithmic problem."
            },
            {
                "role": "assistant",
                "content": (
                    "[PROBLEM]\n"
                    f"{previous_question_1[0]}\n\n"
                    "[TESTS]\n"
                    "```python\n"
                    f"{previous_question_1[1]}\n"
                    "```"
                )
            },
            {
                "role": "user",
                "content": "Create an algorithmic coding problem based on this topic:\nTopic: Create any algorithmic problem."
            },
            {
                "role": "assistant",
                "content": (
                    "[PROBLEM]\n"
                    f"{previous_question_2[0]}\n\n"
                    "[TESTS]\n"
                    "```python\n"
                    f"{previous_question_2[1]}\n"
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
    else:
        proposer_messages = [
        {
            "role": "system",
            "content": (
                "You are an expert programming problem creator. "
                "You must always structure your response using exact tags: "
                "[PROBLEM] followed by the problem description, and "
                "[TESTS] followed by a Python code block containing up to 3 assertions "
                "that call `solution(...)`. No other code is allowed other than assertions."
                """
### Evaluation Criteria:
- Executability, your assert statements should be executable
- Correctness, your assert statements should be correct for the problem proposed
- Difficulty in solving the coding problem. Focus on either algorithmic reasoning or logic complexity.
- Creativity, the problem needs to be sufficiently different from the provided reference problem
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
                "Write a function `solution(s, k)` that takes a string `s` and a positive integer `k`.\n\n"
                "Split `s` into blocks of size `2k` from left to right. In each block:\n"
                "- Reverse the first `k` characters.\n"
                "- Leave the remaining characters in that block unchanged.\n"
                "- If fewer than `k` characters remain at the end of `s`, reverse all of them.\n\n"
                "Constraints:\n"
                "- `1 <= len(s) <= 10^4\n"
                "- `1 <= k <= 10^4\n"
                "- Target `O(n)` time.\n\n"
                "[TESTS]\n"
                "```python\n"
                "assert solution(\"abcdefg\", 2) == \"bacdfeg\"\n"
                "assert solution(\"abcd\", 2) == \"bacd\"\n"
                "assert solution(\"a\", 5) == \"a\"\n"
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
                "Write a function `solution(nums)` that takes a list of integers `nums` "
                "and returns the alternating sum of its elements: add elements at even "
                "indices (0, 2, 4, ...) and subtract elements at odd indices (1, 3, 5, ...). "
                "Return 0 if `nums` is empty.\n\n"
                "Constraints:\n"
                "- 0 <= len(nums) <= 10^4\n"
                "- -10^4 <= nums[i] <= 10^4\n"
                "- Aim for O(n) time.\n\n"
                "[TESTS]\n"
                "```python\n"
                "assert solution([1, 2, 3, 4]) == -2\n"
                "assert solution([10, 5, 2]) == 7\n"
                "assert solution([]) == 0\n"
                "```"
            ),
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
- Name the entry function `solution` (e.g., `def solution(...): ...`)
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
- The snippet should end with a return statement from the main function `solution(...)`, DO NOT ADD ANYTHING AFTER FINAL RETURN STATEMENT"
- Add comments using # to support logic flow of code, AVOID chat response style output
- Do not return prose or example print calls.

### Formatting:
- Format your code with:
  def solution(...):
      # your code here
      return ...
  
### Example Format:
def solution(name: str, info: dict):
    # code logic here
    return result

### Evaluation Criteria:
- Executability, your code should be executable when given arguments into function solution(...)
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
                "def solution(string):\n"
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
                "def solution(array):\n"
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
                    f"Write a Python function to solve this problem:\n{problem_desc}\n\n. Accept only one string as input and return one string as output."
                )
            }
        ]

    return answerer_messages


def get_good_probs():
    proposer_good_probs = [
        # 1. Longest alternating / sign-flipping or parity-alternating subarray length
        (
            "Write a function `solution(nums)` that returns the length of the longest contiguous "
            "subarray in `nums` where adjacent elements strictly alternate in parity (even then odd, "
            "or odd then even). Return `0` if `nums` is empty.\n",
            "assert solution([2, 1, 4, 3, 6]) == 5\n"
            "assert solution([1, 3, 5, 2, 4]) == 2\n"
            "assert solution([]) == 0",
        ),
        # 2. Subarray count with target prefix sum
        (
            "Write a function `solution(nums, target)` that returns the number of contiguous "
            "subarrays in `nums` whose elements sum to `target`. Elements may be negative, zero, "
            "or positive.\n",
            "assert solution([1, 1, 1], 2) == 2\n"
            "assert solution([1, -1, 1, -1], 0) == 4\n"
            "assert solution([], 0) == 0",
        ),
        # 3. Minimum window / segment length reaching at least target sum
        (
            "Write a function `solution(nums, target)` that returns the minimum length of a "
            "contiguous subarray of non-negative integers in `nums` whose sum is greater than or "
            "equal to `target`. Return `0` if no such subarray exists.\n",
            "assert solution([2, 3, 1, 2, 4, 3], 7) == 2\n"
            "assert solution([1, 4, 4], 4) == 1\n"
            "assert solution([1, 1, 1], 10) == 0",
        ),
        # 4. Run-length / balanced bracket or token depth / interval overlap count
        (
            "Write a function `solution(s)` that returns the maximum nesting depth of balanced "
            "parentheses `(` and `)` in `s`, ignoring other characters. If `s` is unbalanced at "
            "any point or has unmatched opening parentheses at the end, return `-1`.\n",
            "assert solution(\"(1+(2*3)+((8)/4))+1\") == 3\n"
            "assert solution(\"(()))(\") == -1\n"
            "assert solution(\"abc\") == 0",
        ),
    ]

    return proposer_good_probs