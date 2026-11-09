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