import argparse
import os
import pandas as pd
from func_timeout import func_set_timeout
import unittest
import json
import re
import tqdm
import importlib
import time
import shutil

def prepare_code(dataset, model, sub_benchmark):
  with open(os.path.join('..', '..', 'data', 'sub_benchmarks', 'humanevalplus_hard', 'raw', f'results_{dataset}_{sub_benchmark}_{model}.json')) as f:
    dat = json.load(f)
  
  reg_str = "```python((.|\n)*?)```" if model != 'llama' else "\[PYTHON\]((.|\n)*?)\[/PYTHON\]"
  if dataset == 'humanevalplus':
      samples = []
      for key in dat:
        samples.append(dict(task_id=f'HumanEval/{int(key)}'))   
        for k in dat[key]:
          for ind, sol in enumerate(dat[key][k]):         
            samples[-1][f"{k}_{ind}"] = []
            code = re.findall(reg_str, sol) 
            
            if len(code) < 2 and model != 'gpt':
              samples[-1][f"{k}_{ind}"] = '' 
              continue
            elif model == 'gpt' and len(code) == 0:
              samples[-1][f"{k}_{ind}"]= ''
              continue
            
            if model != 'gpt':
                raw_code = code[1][0]
            else:
                raw_code = code[0][0]
            screened_lines = []
            lines = raw_code.split('\n')
            for line in lines:
              if line.startswith('assert') or line.startswith("#") or line.startswith("print"):
                continue
              screened_lines.append(line)
                
            processed_code = '\n'.join(screened_lines)
            samples[-1][f"{k}_{ind}"] = processed_code
      return samples
  else:
    raise NotImplementedError

def gen_py_file(test_code_name, code_list, test_code, tmp_dir):
        cnt = 0
        for code_snippet in code_list:
            test_code_py = code_snippet + '\n' + test_code
            with open(os.path.join(tmp_dir, test_code_name + '_' + str(cnt) + '.py'), 'w', encoding='utf-8') as f:
                f.write(test_code_py)
            cnt += 1

@func_set_timeout(10)
def run_unit_test(test_code, tmp_dir):
    
    module = importlib.import_module(tmp_dir[:-1] + '.' + test_code)
    test_suite = unittest.TestLoader().loadTestsFromModule(module)
    with open(os.path.join('log_data.log'), 'a', encoding='utf-8') as f:
      test_result = unittest.TextTestRunner(f).run(test_suite)    

    return test_result

def test(code_num, test_code_name, task_code_list, tmp):

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
            res = run_unit_test(test_code_name + '_' + str(i), tmp)
            result[level].append((task_code_list[i], True if len(res.errors) == 0 and len(res.failures) == 0 and res.testsRun != 0 else False))
        except BaseException as e:
            print("Test failed for {}".format(test_code_name))
            print("Reason: ", e)
            fail_run += 1
            result[level].append((task_code_list[i], False))
    
    return result, fail_run

def test_pipeline(samples_list, data, tmp):

        result_dict = {}
        
        # get test code and generate py file
        for task in samples_list:
            test_code = data.iloc[int(task["task_id"].split('/')[-1])]['test']
            task_code_list = [task[k] for k in task if k != 'task_id']
            gen_py_file(str(task["task_id"].split('/')[-1]), task_code_list, test_code, tmp)
        fail_run = 0
        # run unit test
        print("Running test ...")
        for task in tqdm.tqdm(samples_list):
            task_code_list = [task[k] for k in task if k != 'task_id']
            try:
                result, fr = test(len(task_code_list), str(task["task_id"].split('/')[-1]), task_code_list, tmp)
                fail_run += fr
                result_dict[str(task["task_id"].split('/')[-1])] = result
            except Exception as e:
                print(str(task["task_id"].split('/')[-1]))
                print(e)
                raise Exception("This should not occur.")
        
        print("Failed run: ", fail_run)
        return result_dict

def save_result(model_name, dataset, sub_benchmark, result):
        out = f'results_{dataset}_{sub_benchmark}_{model_name}_eval.json'
        save_path = os.path.join(out)
        with open(os.path.join('..', '..', 'data', 'sub_benchmarks', 'humanevalplus_hard', 'post_test', save_path), 'w') as f:
            json.dump(result, f, indent=4, sort_keys=True)

def tear_down(tmp_dir):
        shutil.rmtree(tmp_dir)

def main(dataset, model, sub_benchmark):
  tmp_dir ='tmp/'
  if os.path.isdir(tmp_dir):
        raise Exception("Temporary directory '{}' already exist. Delete it before running the test".format(tmp_dir))
  else:
        os.mkdir(tmp_dir)

  data = pd.read_csv(os.path.join('..', '..', 'data', 'sub_benchmarks', 'humanevalplus_hard', f'prompts_generated_{dataset}_{sub_benchmark}.csv'), sep=';', index_col=0)
  samples = prepare_code(dataset, model, sub_benchmark)
  result_dict = test_pipeline(samples, data, tmp_dir)
  # save result
  save_result(model, dataset, sub_benchmark, result_dict)
  time.sleep(2)
  tear_down(tmp_dir)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--model', default=None)
  parser.add_argument('-d', '--dataset', default='humanevalplus')
  parser.add_argument('-s', '--sub_benchmark', default=None)
  args = parser.parse_args()
  if args.sub_benchmark is None:
    raise Exception("Please also provide a sub_benchmark identifier")
  else:
    main(args.dataset, args.model, args.sub_benchmark)
