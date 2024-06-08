# Adapted from ClassEval prompts
prompt_codellama_c = """[INST]Below is an instruction that describes a task. Write a response that appropriately completes the request.

Instruction: Please complete the `METHOD_NAME` method in the following class `CLASS_NAME`. Assume that all functions from the class `CLASS_NAME` except `METHOD_NAME` are already implemented, thus you call them without implementing them in your answer.
Please only return the completed method `METHOD_NAME` enclosed in [PYTHON] and [/PYTHON]. Do not return any other code in your answer. Do not print any explanation, comment or example in your answer.
```
CLASS_CODE
DOCSTRING
```

Response:[/INST]
[PYTHON]
def METHOD_SIGNATURE:"""

prompt_magic_c = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

@@ Instruction
Please complete the `METHOD_NAME` method in the following class `CLASS_NAME`. Assume that all functions from the class `CLASS_NAME` except `METHOD_NAME` are already implemented, thus you call them without implementing them in your answer.
Please only return the completed method `METHOD_NAME` enclosed in ```python and ```. Do not return any other code in your answer. Do not print any explanation, comment or example in your answer.
```
CLASS_CODE
DOCSTRING
```

@@ Response
```python
def METHOD_SIGNATURE:"""

prompt_deepseek_c = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction
Please complete the `METHOD_NAME` method in the following class `CLASS_NAME`. Assume that all functions from the class `CLASS_NAME` except `METHOD_NAME` are already implemented, thus you call them without implementing them in your answer.
Only return the completed method `METHOD_NAME` enclosed in ```python and ```. Do not return any other code in your answer. Do not print any explanation, comment or example in your answer.
```
CLASS_CODE
DOCSTRING
```

### Response
```python
def METHOD_SIGNATURE:"""

prompt_gpt_c = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

Problem:
Please complete the `METHOD_NAME` method in the following class `CLASS_NAME`. Assume that all functions from the class `CLASS_NAME` except `METHOD_NAME` are already implemented, thus you call them without implementing them in your answer.
Only return the completed method `METHOD_NAME` enclosed in ```python and ```. Do not return any other code in your answer. Do not print any explanation, comment or example in your answer.
```
CLASS_CODE
DOCSTRING
```

Reponse:
```python
def METHOD_SIGNATURE:"""

prompt_gemma_c="""<start_of_turn>user
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction
Please complete the `METHOD_NAME` method in the following class `CLASS_NAME`. Assume that all functions from the class `CLASS_NAME` except `METHOD_NAME` are already implemented, thus you call them without implementing them in your answer.
Only return the completed method `METHOD_NAME` enclosed in ```python and ```. Do not return any other code in your answer. Do not print any explanation, comment or example in your answer.
```
CLASS_CODE
DOCSTRING
```

### Response<end_of_turn>
<start_of_turn>model
```python
def METHOD_SIGNATURE:"""
