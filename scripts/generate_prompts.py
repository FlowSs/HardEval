import os
import json
import time
from openai import OpenAI
import tqdm
import pandas as pd
import re
import argparse
from datasets import load_dataset

def preprocess_prompt( d_name, entry):
  """Return the oracle function signature, function code and original docstring to feed it to the prompt template.

  Args:
      d_name (str): dataset name
      entry (DataFrame): the task to consider, i.e. row from our dataframe

  Returns:
      the signature, code and original docstring of the task
  """
  
  if 'humanevalplus' in d_name:
   if 'sub_benchmark' in d_name:
    signature = entry['signature']
    function = entry['code']
    prompt = entry['original prompt']
   else:
    signature = re.findall(r"def (.*)\s?:\n", entry['prompt'])[-1]
    # Maybe overly complex, but it works!
    function = re.findall(r"^((.|\n)+:)\n\s+(?=(\"\"\"|\'\'\'))", entry['prompt'])[0][0] + entry['canonical_solution']
    prompt =  re.findall(r"(?:\"\"\"|\'\'\')((.|\n)+)(?:\"\"\"|\'\'\')", entry['prompt'])[0][0]
    prompt = "\n".join([re.sub(r'^\s+', '', e) for e in prompt.split('\n')])
  
  elif 'ClassEval' in d_name:
   signature = entry['method_signature']
   function = entry['method_code']
   prompt = entry['original prompt']


  return signature, function, prompt

def generate_instruction_prompts(dataset_name, prompt, function=None, entry=None):
  """Generate the level prompt adapted to the task to feed to GPT4 (see the template in 'data/')

  Args:
      dataset_name
      prompt: the docstring of the task
      function: the task oracle code
      entry: the task as a row from the pandas DataFrame

  Returns:
      the template prompt for level adapted to the current task
  """
  if dataset_name == 'humanevalplus':
    mod_prompt_instruction = custom_prompt_all.replace('INSERT_FUNCTION', function)
    mod_prompt_instruction = mod_prompt_instruction.replace("INSERT_PROMPT", prompt)
  elif dataset_name == 'ClassEval':
    mod_prompt_instruction = custom_prompt_all.replace('INSERT_METHOD_NAME', entry['method_name'])
    mod_prompt_instruction = mod_prompt_instruction.replace("INSERT_CLASS_CODE", entry['class_code'])
    mod_prompt_instruction = mod_prompt_instruction.replace("INSERT_DOCSTRING", prompt)
 
  return mod_prompt_instruction

def generate_rephrase_prompts(prompt, function):
  """Generate the rephrase prompt adapted to the task to feed to GPT4 (see the template in 'data/')

  Args:
      prompt: the docstring of the task
      function: the task oracle code

  Returns:
      _type_: _description_
  """
  mod_prompt_rephrase = custom_rephrase.replace('INSERT_FUNCTION', function)
  mod_prompt_rephrase = mod_prompt_rephrase.replace("INSERT_PROMPT", prompt)
  
  return mod_prompt_rephrase


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--dataset', default=None)
  parser.add_argument('-s', '--sub_benchmark', default=None)
  parser.add_argument('--levels', action='store_true')
  parser.add_argument('--rephrase', action='store_true')
  args = parser.parse_args()

  if args.levels and args.rephrase:
    raise Exception("Can't have levels and rephrase at the same time! Need to check the levels first...")
  elif not(args.levels) and not(args.rephrase):
    raise Exception("The script needs one transformation to apply, either levels or rephrase but both can not be false")

  dataset_name = args.dataset
  if dataset_name not in ['ClassEval', 'humanevalplus']:
    raise Exception(f"Dataset {dataset_name} not recognised")

  if args.sub_benchmark is not None and not(os.path.exists(f'prompts_generated_{dataset_name}_{args.sub_benchmark}.csv')):
    raise Exception(f"Sub benchmark: prompts_generated_{dataset_name}_{args.sub_benchmark}.csv does not exist")

  file_to_save = f'prompts_generated_{dataset_name}_{args.sub_benchmark}.json' if args.sub_benchmark is not None \
    else f'prompts_generated_{dataset_name}.json'
  
  print(f"Output will be saved in {file_to_save}")

  # Needs a file name 'key.json' to store the API keys for OpenAI
  if not(os.path.exists('key.json')):
    raise Exception(f'You need to provide a key.json file containing the OpenAI key in order to use GPT4')
  
  with open('key.json', 'r') as f:
    keys = json.load(f)
  OPENAI_API_KEY = keys['OPENAI']

  # Getting custom prompts
  with open(os.path.join('..', 'data', 'prompt_templates', f'template_prompt_{dataset_name.lower()}.txt'), 'r') as f:
    custom_prompt_all = f.readlines()
  custom_prompt_all = "".join(custom_prompt_all)

  with open(os.path.join('..', 'data', 'prompt_templates','template_prompt_rephrase.txt'), 'r') as f:
    custom_rephrase = f.readlines()
  custom_rephrase = "".join(custom_rephrase)

  # Loading dataset
  if args.sub_benchmark is not None:
    dataset = pd.read_csv(os.path.join('..', 'data', 'sub_benchmarks', f'{dataset_name}_hard', \
                                        f'prompts_generated_{dataset_name}_{args.sub_benchmark}.csv', sep=';'))
  elif dataset_name == 'humanevalplus':
    dataset = load_dataset("evalplus/humanevalplus")['test'].to_pandas()
  elif dataset_name == 'ClassEval':
    dataset = pd.read_csv(os.path.join('..', 'data', 'sub_benchmarks', 'classeval','ClassEval_datasets.csv', sep=";"))


  # Loading the .json if we already did partial calculation
  # Otherwise, initialize an empty dict
  if os.path.exists(os.path.join('..', 'data', f'{dataset_name}' if args.sub_benchmark is None \
                                 else os.path.join('sub_benchmarks', f'{dataset_name}'), file_to_save)):
    with open(os.path.join('..', 'data', f'{dataset_name}' if args.sub_benchmark is None \
                                 else os.path.join('sub_benchmarks', f'{dataset_name}'), file_to_save), 'r') as f:
      dat = json.load(f)
  else:
    dat = {}
  # Setting up OpenAI client
  client = OpenAI(api_key=OPENAI_API_KEY)
  model_generate = 'gpt-4-turbo'
  skipped = 0

  # For all tasks
  for ind in tqdm.tqdm(range(0, len(dataset))):
      
      # Getting different part of the prompt
      signature, function, prompt = preprocess_prompt(dataset_name if args.sub_benchmark is None \
                                                       else dataset_name + '_sub_benchmark', dataset.iloc[ind])
      
      start_time = time.time()

      ###########################
      # Generating Prompt Level #
      ###########################
      if args.levels:

        # If the prompt for this task has already been generated
        if (dataset_name == 'humanevalplus' and str(dataset.iloc[ind]['task_id']).split('/')[-1] in dat) or\
          (dataset_name == 'ClassEval' and str(dataset.iloc[ind]['class_id']) in dat):
          continue  
        
        mod_prompt_instruction = generate_instruction_prompts(dataset_name, prompt, function, entry=dataset.iloc[ind])
        levels_list = [[], [], []]
        inference_not_done = True
        number_of_trials = 0
        error_message = []
        
        # While inference is not completed or if we haven't tried at least 3 times (i.e. to avoid server error)
        while inference_not_done and number_of_trials != 3:
            try:
                completion = client.chat.completions.create(
                  model=model_generate,
                  messages=[
                    {"role": "user", "content": mod_prompt_instruction}
                  ],
                  temperature=0.8
                )
                out = completion.choices[0].message.content
                inference_not_done = False

            # in case something wrong happen, retry
            except Exception as e:
                error_message.append(e)
                time.sleep(2)
                number_of_trials += 1
        
        # Logging the output in case of errors (to know which one to rerun)
        # Otherwise, adding the generated code to be saved
        if number_of_trials == 3:
          print(f"** LOG ** Number of trials exceeded for function ({ind}) {signature} with the last error messages being {error_message}")
          raise Exception()
        else:
          reg_str1 = "<first>((.|\n)*?)</first>"
          reg_str2 = "<second>((.|\n)*?)</second>"
          reg_str3 = "<third>((.|\n)*?)</third>"
          
          level1 = re.findall(reg_str1, out)[0][0]
          level2 = re.findall(reg_str2, out)[0][0]
          level3 = re.findall(reg_str3, out)[0][0]
          
          levels_list[0].append(level1)
          levels_list[1].append(level2)
          levels_list[2].append(level3)     
        
        print("Generated instruction: ", time.time() - start_time)

      ###########################
      # Reprhasing Prompt Level #
      ###########################
      else:
        
        if dataset_name == 'humanevalplus':
          # If we already rephrased this task
          if len(dat[str(dataset.iloc[ind]['task_id']).split('/')[-1]]['level 1']) > 1 and \
              len(dat[str(dataset.iloc[ind]['task_id']).split('/')[-1]]['level 2']) > 1 and \
              len(dat[str(dataset.iloc[ind]['task_id']).split('/')[-1]]['level 3']) > 1:
            print(f"Already rephrased task {str(dataset.iloc[ind]['task_id']).split('/')[-1]}, skipping...")
            continue
          
          levels_list = [ [ dat[str(dataset.iloc[ind]['task_id']).split('/')[-1]]['level 1'][p] for p in dat[str(dataset.iloc[ind]['task_id']).split('/')[-1]]['level 1']], \
            [ dat[str(dataset.iloc[ind]['task_id']).split('/')[-1]]['level 2'][p] for p in dat[str(dataset.iloc[ind]['task_id']).split('/')[-1]]['level 2'] ], \
            [ dat[str(dataset.iloc[ind]['task_id']).split('/')[-1]]['level 3'][p] for p in dat[str(dataset.iloc[ind]['task_id']).split('/')[-1]]['level 3'] ] ]
        
        elif dataset_name == 'ClassEval':
          # If we already rephrased this task
          if len(dat[str(dataset.iloc[ind]['class_id'])]['level 1']) > 1 and \
              len(dat[str(dataset.iloc[ind]['class_id'])]['level 2']) > 1 and \
              len(dat[str(dataset.iloc[ind]['class_id'])]['level 3']) > 1:
            print(f"Already rephrased task {dataset.iloc[ind]['class_id']}, skipping...")
            continue

          levels_list = [ [ dat[str(dataset.iloc[ind]['class_id'])]['level 1'][p] for p in dat[str(dataset.iloc[ind]['class_id'])]['level 1']], \
            [ dat[str(dataset.iloc[ind]['class_id'])]['level 2'][p] for p in dat[str(dataset.iloc[ind]['class_id'])]['level 2']], \
            [ dat[str(dataset.iloc[ind]['class_id'])]['level 3'][p] for p in dat[str(dataset.iloc[ind]['class_id'])]['level 3']] ]

        for ii, inp in enumerate(levels_list):
          # This is used to check, within a task, if we already rephrased certain level
          if len(levels_list[ii]) > 1:
            print(f"Level {ii + 1} already rephrased, skipping")
            continue
          
          mod_prompt_rephrase = generate_rephrase_prompts(inp[0], function)
          
          inference_not_done = True
          number_of_trials = 0
          error_message = []
          
          # Second, rephrase the instruction to add diversity
          # While inference is not completed or if we haven't tried at least 3 times (i.e. to avoid server error)
          while inference_not_done and number_of_trials != 3:
              try:
                  
                  completion = client.chat.completions.create(
                    model=model_generate,
                    messages=[
                      {"role": "user", "content": mod_prompt_rephrase}
                    ],
                  )
                  out = completion.choices[0].message.content
                  inference_not_done = False
                  
              # in case something wrong happen, retry
              except Exception as e:
                  error_message.append(e)
                  time.sleep(1)
                  number_of_trials += 1
          # Logging the output in case of errors (to know which one to rerun)
          # Otherwise, adding the generated code to be saved
          if number_of_trials == 3:
            print(f"** LOG ** Number of trials exceeded for function ({ind}) {signature} with the last error messages being {error_message}")
            raise Exception()
          else:
            # Safeguard, sometimes GPT4 enclose the list in ```python ... ```.            
            try:
              reg_str = "```python((.|\n)*?)```"
              output = re.findall(reg_str, out)[0][0]
            except:
              output = out
            
            # Sometimes, it also names the list
            if len(output.split('= [')) == 2:
              output = '[' + output.split('= [')[1]
            elif len(output.split('= [')) > 2:
              raise Exception("error on split")

            output = eval(output)
            
            levels_list[ii] = levels_list[ii] + output

        print("Rephrased instruction ", time.time() - start_time)
      
      
      
      # Writing the total output
      if dataset_name == 'humanevalplus':
        if args.sub_benchmark is None:
          dat[str(dataset.iloc[ind]['task_id'].split('/')[1])] = {
            'signature': signature,
            'code': function,
            'original prompt': prompt,
            'level 1': levels_list[0],
            'level 2': levels_list[1],
            'level 3': levels_list[2],
          }
        else:
          dat[str(dataset.iloc[ind]['task_id'])] = {
            'signature': signature,
            'code': function,
            'original prompt': prompt,
            'level 1': levels_list[0],
            'level 2': levels_list[1],
            'level 3': levels_list[2],
          }
      elif dataset_name == 'ClassEval':      
        dat[str(dataset.iloc[ind]['class_id'])] = {
          'class_name': dataset.iloc[ind]['class_name'],
          'class_code': dataset.iloc[ind]['class_code'],
          'class_text': dataset.iloc[ind]['class_text'],
          'method_signature': signature,
          'method_name': dataset.iloc[ind]['method_name'],
          'method_code': function,
          'method_params': dataset.iloc[ind]['method_params'],
          'original prompt': prompt,
          'level 1': levels_list[0],
          'level 2': levels_list[1],
          'level 3': levels_list[2],
        }

      # putting it at the root of 'script' in the replication package to avoid mistakes
      with open(file_to_save, 'w') as f:
        json.dump(dat, f)