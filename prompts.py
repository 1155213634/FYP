def get_proposer_prompt(iter):
    # Initial Seed Prompts for Proposer
    seed_prompts = [
        "Create an algorithmic problem involving arrays and target sums.",
        "Create an algorithmic problem involving string reversal or parsing."
    ]

    current_seed = seed_prompts[iter % len(seed_prompts)]

    proposer_messages = [
            {
                "role": "system", 
                "content": (
                    "You are an expert programming problem creator. "
                    "FORMAT REQUIREMENT: You must always output problems using the exact tags: "
                    "[PROBLEM] followed by the description, and [TESTS] followed by executable Python assertions. Assertions use function solution(args)\n"
                    "Use 4 space character for indentation, and indent once for each assertion, \\n to separate each assertion, output 2-3 assertions"
                    "Example:\n"
                    "[PROBLEM] Detailed problem description\n"
                    "[TESTS]```python\n"
                    "assert solution(args) == expected\n"
                    "```"
                )
            },
            {
                "role": "user", 
                "content": (
                    f"Create an algorithmic coding problem based on this topic:\nTopic: {current_seed}\n\n"
                    "Strictly adhere to this layout:\n"
                    "[PROBLEM] Detailed problem description\n"
                    "[TESTS]```python\n"
                    "assert solution(args) == expected\n"
                    "```"
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
- Ensure execution completes within 10 seconds on a modern CPU
- All imports and class definitions should be at the very top of the code snippet
- The snippet should end with a return statement from the main function `f(...)`, DO NOT ADD ANYTHING AFTER FINAL RETURN STATEMENT"

### Formatting:
- Format your code with:
  def f(...):
      # your code here
      return ...
  
### Example Format:
def f(name: str, info: dict):
    # code logic here
    return result

"""
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