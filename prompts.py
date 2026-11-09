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
                "content": (
                    "You are a expert programmer."
                    "FORMAT REQUIREMENT: Output ONLY valid Python code. The code should have only one function solution(). Do not include any conversational filler."
                    "Use 4 space character for indentation, \\n to separate each line"
                )
            },
            {
                "role": "user", 
                "content": (
                    f"Write a Python function to solve this problem:\n{problem_desc}\n\n"
                    "Strictly adhere to this layout:\n"
                    "```python\n"
                    "def solution(args):\n"
                    "\\t# Working Code\n"
                    "```"
                )
            }
        ]

    return answerer_messages