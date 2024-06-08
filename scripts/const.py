# Inspired/adapted from https://github.com/Intelligent-CAT-Lab/CodeMind/blob/main/scripts/create_prompt_der.py
prompt_codellama = """[INST]Your task is to write a Python function to solve a programming problem.
The Python code must be between [PYTHON] and [/PYTHON] tags. Only return the function. You are prohibited from adding any natural language explanation or assertion to your response.

Problem: 
```
INSTRUCTION_PLACEHOLDER
```
[/INST]"""

prompt_magic = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
@@ Instruction
Write a Python solution to the following problem:
```
INSTRUCTION_PLACEHOLDER
```

Enclose your solution in ```python and ```. Only return the function. You are prohibited from adding any natural language explanation or assertion to your response.

@@ Response"""

prompt_deepseek = """You are an expert Python programmer.Your task is to write a Python function to solve a programming problem.
### Instruction
Write a Python solution to the following problem:
```
INSTRUCTION_PLACEHOLDER
```

Enclose your solution in ```python and ```. Only print the function. You are prohibited from adding any natural language explanation or assertion to your response.
### Response"""

prompt_gpt = """Your task is to write a Python function to solve a programming problem.
[Formatting Instructions]
- Start your code with ```python
- End your code with ```
- Only return the function. You are prohibited from adding any natural language explanation or assertion to your response.

Problem:
```
INSTRUCTION_PLACEHOLDER
```"""

prompt_gemma="""<start_of_turn>user
You are an expert Python programmer.Your task is to write a Python function to solve a programming problem.
### Instruction
Write a Python solution to the following problem:
```
INSTRUCTION_PLACEHOLDER
```

Enclose your solution in ```python and ```. Only print the function. You are prohibited from adding any natural language explanation or assertion to your response.<end_of_turn>
<start_of_turn>model"""
