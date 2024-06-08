import shutil
import time
import importlib
from func_timeout import func_set_timeout
import unittest
import json
import re
import os
import ast
import tqdm
import shutil


# Test Pipeline inspired by the pipeline from ClassEval repository
class AutoTest:

    def __init__(self, eval_data_name, model_name, dataset):
        self.eval_data = self.get_eval_data(eval_data_name)
        self.model_name = model_name
        self.dataset = dataset
        self.tmp_dir ='tmp/'

        if os.path.exists(os.path.join(model_name + "_log_data.log")):
            os.remove(os.path.join(model_name + "_log_data.log"))

        if os.path.isdir(self.tmp_dir):
            raise Exception("Temporary directory '{}' already exist. Delete it before running the test".format(self.tmp_dir))
        else:
            os.mkdir(self.tmp_dir)

    def get_eval_data(self, eval_data_name):
        eval_data = {}
        with open(eval_data_name, encoding='utf-8') as file:
            data = json.load(file)
        for item in data:
            eval_data[item['task_id']] = item
        return eval_data

    def gen_py_file(self, test_code_name, code_list, test_code):
        cnt = 0
        for code_snippet in code_list:
            test_code_py = code_snippet + '\n' + test_code
            with open(os.path.join(self.tmp_dir, test_code_name + '_' + str(cnt) + '.py'), 'w', encoding='utf-8') as f:
                f.write(test_code_py)
            cnt += 1

    def gen_code_list(self, generated_codes, sampled_tasks):
        code_list, code_for_sim_list, method_list = {}, {}, {}
        wt = 0 # wrong tag, i.e., if the LLM forgot to enclose between correct tags
        bf = 0 # if synthax error, i.e. AST can't parse
        c = 0

        for (_, row), ind in zip(sampled_tasks.iterrows(), generated_codes):
            if 'greedy' in self.model_name:
                c += 1
                # Regex to parse the obtained code
                regex_code = r'```python((.|\n)*?)```' if 'llama' not in self.model_name else r'\[PYTHON\]((.|\n)*?)\[/PYTHON\]'
                answer_code = None
                code = generated_codes[ind]
                
                try:

                    if 'gpt' in self.model_name:
                        answer_code = re.findall(regex_code, code)[0]
                        answer_code = [e for e in answer_code if 'import ' not in e and '@staticmethod' not in e]
                        tmp = "".join(answer_code)
                        if 'def ' + row['method_signature'] in tmp:
                            answer_code = tmp.strip()
                        else:
                            if len(tmp) > 1 and not(tmp[1].isspace()):
                                tmp = '\n' + '    ' + tmp.replace('\n', '\n    ')
                            answer_code = 'def ' + row['method_signature'] +':' + tmp
                        
                    else:
                        answer_code = re.findall(regex_code, code)[1][0].strip()
                    
                    code_for_sim_list['ClassEval_' + row['class_id']] = answer_code
                    try:
                        if 'gemma' in self.model_name:
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
                        code_for_sim_list['ClassEval_' + row['class_id']] = answer_code
                
                # Get the oracle code
                oracle_code = ast.parse(self.eval_data['ClassEval_' + row['class_id'].split('_')[0]]['solution_code'])
                # Oracle code with injected generated function
                modified_code = ast.unparse(self.replace_body_function(oracle_code, row['method_name'], func_body))
                
                if docstring is not None:
                   # Removing docstring as it is sometimes badly formatted when injected bad, which could cause issue down the line
                   regex_docstring = r'\"\"\"((.|\n)*?)\"\"\"'
                   modified_code = re.sub(regex_docstring, '', modified_code)
                
                # Adding the code
                code_list['ClassEval_' + row['class_id']] = modified_code
                # Method name to only run relevant test in function 'test'
                method_list['ClassEval_' + row['class_id']] = row['method_name']
                

            else:
                code_for_sim_list['ClassEval_' + row['class_id']] = {}
                code_list['ClassEval_' + row['class_id']] = []
                for level in generated_codes[ind]:
                    if level != 'original prompt':
                        code_for_sim_list['ClassEval_' + row['class_id']][level] = []
                        for _, code in enumerate(generated_codes[ind][level]):
                            c += 1
                            regex_code = r'```python((.|\n)*?)```' if self.model_name != 'llama' else r'\[PYTHON\]((.|\n)*?)\[/PYTHON\]'
                            answer_code = None
                            
                            try:

                                if self.model_name == 'gpt':
                                    answer_code = re.findall(regex_code, code)[0]
                                    answer_code = [e for e in answer_code if 'import ' not in e and '@staticmethod' not in e]
                                    tmp = "".join(answer_code)
                                    if 'def ' + row['method_signature'] in tmp:
                                        answer_code = tmp.strip()
                                    else:
                                        if len(tmp) > 1 and not(tmp[1].isspace()):
                                            tmp = '\n' + '    ' + tmp.replace('\n', '\n    ')
                                        answer_code = 'def ' + row['method_signature'] +':' + tmp
                                    
                                else:
                                    answer_code = re.findall(regex_code, code)[1][0].strip()

                                code_for_sim_list['ClassEval_' + row['class_id']][level].append(answer_code)
                                try:
                                    if self.model_name == 'gemma':
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
                                    code_for_sim_list['ClassEval_' + row['class_id']][level].append(answer_code)
                                                                              
                                
                            # Get the oracle code
                            oracle_code = ast.parse(self.eval_data['ClassEval_' + row['class_id'].split('_')[0]]['solution_code'])
                            # Oracle code with injected generated function
                            modified_code = ast.unparse(self.replace_body_function(oracle_code, row['method_name'], func_body))
                            
                            if docstring is not None:
                               # Removing docstring as it is sometimes badly formatted when injected bad, which could cause issue down the line
                               regex_docstring = r'\"\"\"((.|\n)*?)\"\"\"'
                               modified_code = re.sub(regex_docstring, '', modified_code)
                            
                            # Adding the code
                            code_list['ClassEval_' + row['class_id']].append(modified_code)
                
                # Method name to only run relevant test in function 'test'
                method_list['ClassEval_' + row['class_id']] = row['method_name']
        
        # this includes codes for which the LLM did not include in between tags or invalid synthax
        # they will be considered as 'invalid' when running the test
        print("Number of code samples badly formatted: ", bf)
        print("Number of code samples with wrong tags: ", wt)
        print("Total code samples parsed: ", c)
        
        return code_list, method_list, code_for_sim_list

    @func_set_timeout(5)
    def run_unit_test(self, test_code, test_class):
        
        module = importlib.import_module(self.tmp_dir[:-1] + '.' + test_code)
        log_path = os.path.join(self.model_name + "_log_data.log")
        
        with open(log_path, 'a', encoding='utf-8') as f:
            test_suite = unittest.TestLoader().loadTestsFromTestCase(getattr(module, test_class))
            test_result = unittest.TextTestRunner(stream = f).run(test_suite)        

        return test_result

    def test(self, code_num, test_code_name, test_class, code_list):

        if 'greedy' in self.model_name:
            result = None
        else:
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
                res = self.run_unit_test(test_code_name + '_' + str(i), test_class)
                if res.testsRun == 0:
                    raise Exception("No test ran")
                if 'greedy' in self.model_name:
                    result = (code_list[i], True if len(res.errors) == 0 and len(res.failures) == 0 and res.testsRun != 0 else False)
                else:
                    result[level].append((code_list[level][i % 30], True if len(res.errors) == 0 and len(res.failures) == 0 and res.testsRun != 0 else False))
            except BaseException as e:
                print("Test failed for {} and test_class {}".format(test_code_name, test_class))
                print("Reason: ", e)
                fail_run += 1
                if 'greedy' in self.model_name:
                    result = (code_list[i], False)
                else:
                    result[level].append((code_list[level][i % 30], False))

        return result, fail_run

    def save_result(self, result):
        out = f'results_{self.dataset}_{self.model_name}_eval.json'
        with open(os.path.join('..', '..', 'data', 'classeval', 'post_test', out), 'w') as f:
            json.dump(result, f, indent=4, sort_keys=True)

    def test_pipeline(self, generated_codes, sampled_tasks):

        result_dict = {}
        # get generate code list
        code_list, method_list, code_for_sim_list = self.gen_code_list(generated_codes, sampled_tasks)
        # get test code and generate py file
        for task_id in code_list:
            test_code = self.eval_data["_".join(task_id.split('_')[:2])]['test']
            task_code_list = code_list[task_id] if 'greedy' not in self.model_name else [code_list[task_id]]
            self.gen_py_file(task_id, task_code_list, test_code)
        
        fail_run = 0
        # run unit test
        print("Running test ...")
        for task_id in tqdm.tqdm(code_list):
            task_code_list = code_list[task_id] if 'greedy' not in self.model_name else [code_list[task_id]]
            method_name = method_list[task_id]
            code = code_for_sim_list[task_id] if 'greedy' not in self.model_name else [code_for_sim_list[task_id]]
            try:
                test_class = None
                for ele in self.eval_data["_".join(task_id.split('_')[:2])]['methods_info']:
                    if ele['method_name'] == method_name:
                        test_class = ele["test_class"]
                        break

                if test_class is None:
                    raise Exception("Test code for method {} of class id {} not found".format(method_name, task_id))

                result, fr = self.test(len(task_code_list), task_id, test_class, code)
                fail_run += fr
                result_dict[task_id] = result
            except BaseException as e:
                print(method_name, task_id)
                print(e)
                raise Exception("This should not occur.")
        
        print("Failed run: ", fail_run)
        # save result
        self.save_result(result_dict)
        time.sleep(2)
        self.tear_down()

    def tear_down(self):
        shutil.rmtree(self.tmp_dir)

    def replace_body_function(self, current_ast, current_method_name, replacement_body):
        for sub_ast in current_ast.body:
         if isinstance(sub_ast, ast.ClassDef):
            for function in sub_ast.body:

                # If comments or in class assignation, pass, as it doesn't have 'name' attribute
                if isinstance(function, ast.Expr) or isinstance(function, ast.Assign):
                    continue
                if function.name == current_method_name:
                    function.body = replacement_body if replacement_body != 'Pass' else [ast.Pass()]
                    return current_ast
