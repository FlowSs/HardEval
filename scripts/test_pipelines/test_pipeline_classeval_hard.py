import shutil
import time
import importlib
from func_timeout import func_set_timeout
import unittest
import json
import re
import os
import pandas as pd
import ast
import tqdm
import shutil
import argparse

def gen_py_file(test_code_name, code_list, test_code, tmp_dir):
    cnt = 0
    for code_snippet in code_list:
            test_code_py = code_snippet + '\n' + test_code
            with open(os.path.join(tmp_dir, test_code_name + '_' + str(cnt) + '.py'), 'w', encoding='utf-8') as f:
                f.write(test_code_py)
            cnt += 1

def prepare_code(dat, skeleton, method_names, method_signatures, model):
        
    samples = []
    code_answer = []
    wt = 0 # wrong tag, i.e., if the LLM forgot to enclose between correct tags
    bf = 0 # if synthax error, i.e. AST can't parse
    c = 0

    for key in dat:
        samples.append(dict(task_id=f'ClassEval/{int(key)}'))  
        code_answer.append(dict(task_id=f'ClassEval/{int(key)}'))
        for level in dat[key]:
            for ind, sol in enumerate(dat[key][level]): 
                c += 1
                regex_code = r'```python((.|\n)*?)```' if model != 'llama' else r'\[PYTHON\]((.|\n)*?)\[/PYTHON\]'
                answer_code = None
                    
                try:

                    if model == 'gpt':
                        answer_code = re.findall(regex_code, sol)[0]
                        answer_code = [e for e in answer_code if 'import ' not in e and '@staticmethod' not in e]
                        tmp = "".join(answer_code)
                        if 'def ' + method_signatures[int(key)] in tmp:
                            answer_code = tmp.strip()
                        else:
                            if len(tmp) > 1 and not(tmp[1].isspace()):
                                tmp = '\n' + '    ' + tmp.replace('\n', '\n    ')
                            answer_code = 'def ' + method_signatures[int(key)] +':' + tmp
                        
                    else:
                        answer_code = re.findall(regex_code, sol)[1][0].strip()

                    try:
                        if model == 'gemma':
                            answer_code = answer_code.replace('<start_of_turn>model', '')
                        # Function body should be first
                        parsed_code = ast.parse(answer_code).body[0]
                        if isinstance(parsed_code, ast.FunctionDef):
                           func_body = parsed_code.body
                           # Getting docstring to be removed, if any
                           docstring = ast.get_docstring(parsed_code)
                        else:
                            raise Exception()
                    except Exception as e:

                        func_body = 'Pass'
                        docstring = None
                        bf += 1  
                        
                except Exception as e:
                    if answer_code is None:
                        wt += 1
                        answer_code = None
                        func_body = 'Pass'
                        docstring = None
                                                                  
                    
                # Get the oracle code
                oracle_code = ast.parse(skeleton[int(key)])
                # Oracle code with injected generated function
                modified_code = ast.unparse(replace_body_function(oracle_code, method_names[int(key)], func_body))
                
                if docstring is not None:
                   # Removing docstring as it is sometimes badly formatted when injected bad, which could cause issue down the line
                   regex_docstring = r'\"\"\"((.|\n)*?)\"\"\"'
                   modified_code = re.sub(regex_docstring, '', modified_code)
                
                # Adding the code
                samples[-1][f"{level}_{ind}"] = modified_code
                code_answer[-1][f"{level}_{ind}"] = answer_code
        
    # this includes codes for which the LLM did not include in between tags or invalid synthax
    print("Number of code samples badly formatted: ", bf)
    print("Number of code samples with wrong tags: ", wt)
    print("Total code samples parsed: ", c)
    return samples, code_answer

@func_set_timeout(5)
def run_unit_test(test_code, test_name, tmp_dir):
    
    module = importlib.import_module(tmp_dir[:-1] + '.' + test_code)
    test_suite = unittest.TestLoader().loadTestsFromTestCase(getattr(module, test_name))
    with open(os.path.join('log_data.log'), 'a', encoding='utf-8') as f:
        test_result = unittest.TextTestRunner(stream = f).run(test_suite)
    

    return test_result

def test(code_num, test_code_name, answer_code_list, test_name, tmp):

    result = {}
    result['level 1'] = []
    result['level 2'] = []
    result['level 3'] = []

    fail_run = 0
    for i in range(code_num):
        level = None
        if i < 30:
           level = 'level 1'
        elif 30 <= i < 60:
           level = 'level 2'
        else:
           level = 'level 3'
        
        try:                
            res = run_unit_test(test_code_name + '_' + str(i), test_name, tmp)
            if res.testsRun == 0:
                raise Exception("No test ran")
            result[level].append((answer_code_list[i], True if len(res.errors) == 0 and len(res.failures) == 0 and res.testsRun != 0 else False))
        except BaseException as e:
            print("Test failed for {}".format(test_code_name))
            print("Reason: ", e)
            fail_run += 1
            result[level].append((answer_code_list[i], False))

    return result, fail_run

def test_pipeline(samples_list, code_answer_list, test_list, tmp):

    result_dict = {}
    # get test code and generate py file
    for task in samples_list:
        test_code = test_list[int(task['task_id'].split('/')[-1])]
        task_code_list = [task[k] for k in task if k != 'task_id']
        gen_py_file(str(task['task_id'].split('/')[-1]), task_code_list, test_code, tmp)
    
    fail_run = 0
    # run unit test
    print("Running test ...")
    for task in tqdm.tqdm(code_answer_list):
        answer_code_list = [task[k] for k in task if k != 'task_id']

        try:
            regex_test = r'class (.*)\(.'
            test_name = re.findall(regex_test, test_list[int(task['task_id'].split('/')[-1])])[0]
            print(test_name)
            result, fr = test(len(task_code_list), str(task["task_id"].split('/')[-1]), answer_code_list, test_name, tmp)
            fail_run += fr
            result_dict[str(task['task_id'].split('/')[-1])] = result
        except BaseException as e:
            print(str(task["task_id"].split('/')[-1]))
            print(e)
            raise Exception("This should not happen.")

    print("Failed run: ", fail_run)
    return result_dict

def save_result(model_name, dataset, sub_benchmark, result):
        out = f'results_{dataset}_{sub_benchmark}_{model_name}_eval.json'
        save_path = os.path.join(out)
        with open(os.path.join('..', '..', 'data', 'sub_benchmarks', 'classeval_hard', 'post_test', save_path), 'w') as f:
            json.dump(result, f, indent=4, sort_keys=True)

def tear_down(tmp_dir):
        shutil.rmtree(tmp_dir)
                    
def replace_body_function(current_ast, current_method_name, replacement_body):
    for sub_ast in current_ast.body:
     if isinstance(sub_ast, ast.ClassDef):
        for function in sub_ast.body:

            # If comments or in class assignation, pass, as it doesn't have 'name' attribute
            if isinstance(function, ast.Expr) or isinstance(function, ast.Assign):
                continue
            if function.name == current_method_name:
                function.body = replacement_body if replacement_body != 'Pass' else [ast.Pass()]
                return current_ast

def main(dataset, model, sub_benchmark):
  tmp_dir ='tmp/'
  if os.path.isdir(tmp_dir):
        raise Exception("Temporary directory '{}' already exist. Delete it before running the test".format(tmp_dir))
  else:
        os.mkdir(tmp_dir)

  data = pd.read_csv(os.path.join('..', '..', 'data', 'sub_benchmarks', 'classeval_hard', f'prompts_generated_{dataset}_{sub_benchmark}.csv'), sep=';', index_col=0)
  with open(os.path.join('..', '..', 'data', 'sub_benchmarks', 'classeval_hard','raw', f'results_{dataset}_{sub_benchmark}_{model}.json')) as f:
      res = json.load(f)
  samples, code_answer = prepare_code(res, data['skeleton'].values, data['method_name'].values, data['method_signature'].values, model)
  result_dict = test_pipeline(samples, code_answer, data['test'].values, tmp_dir)
  # save result
  save_result(model, dataset, sub_benchmark, result_dict)
  time.sleep(2)
  tear_down(tmp_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default=None)
    parser.add_argument('-d', '--dataset', default='classeval')
    parser.add_argument('-s', '--sub_benchmark', default=None)
    args = parser.parse_args()
    if args.sub_benchmark is None:
        raise Exception("Please also provide a sub_benchmark identifier")
    else:
        main(args.dataset, args.model, args.sub_benchmark)