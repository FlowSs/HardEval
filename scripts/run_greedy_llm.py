import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI

import tqdm
import torch
import pandas as pd
import argparse
import json

import time

from const import *
from const_class import *

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('-m', '--model', default=None)
   parser.add_argument('-d', '--dataset', default=None)
   parser.add_argument('-s', '--sub_benchmark', default=None)
   args = parser.parse_args()

   if args.dataset not in ['humanevalplus', 'ClassEval']:
      raise Exception(f"Dataset {args.dataset} not recognised")

   # Needs a file name 'key.json' to store the API keys for OpenAI
   if not(os.path.exists('key.json')):
    raise Exception(f'You need to provide a key.json file containing the OpenAI key in order to use GPT4')
  
   with open('key.json', 'r') as f:
    keys = json.load(f)
   
   model_name = args.model 
   if 'gpt' in args.model:
      client = OpenAI(api_key=keys['OPENAI'])
   else:
      tokenizer = AutoTokenizer.from_pretrained(model_name, token=keys['HF_TOKEN'] )
      if 'Llama-3' in args.model or 'gemma' in args.model:
         model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto', token=keys['HF_TOKEN'])
      else:
         model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, token=keys['HF_TOKEN']).to('cuda')

   file_data = f'prompts_generated_{args.dataset}.csv' if args.sub_benchmark is None else f'prompts_generated_{args.dataset}_{args.sub_benchmark}.csv'
   file_data = f'prompts_generated_{args.dataset}.csv' if args.sub_benchmark is None else f'prompts_generated_{args.dataset}_{args.sub_benchmark}.csv'
   dat = pd.read_csv(os.path.join('..', 'data', f'{args.dataset}' if args.sub_benchmark is None \
                                 else os.path.join('sub_benchmarks', f'{args.dataset.lower()}_hard'), file_data), \
                     sep=";", index_col=0 if args.dataset == 'humanevalplus' else None)

   if 'Magicoder' in args.model:
      prompt_base = prompt_magic if args.dataset =='humanevalplus' else prompt_magic_c
      model_name = 'magicoder'
   elif 'deepseek' in args.model:
      prompt_base = prompt_deepseek if args.dataset =='humanevalplus' else prompt_deepseek_c
      model_name = 'deepseek'
   elif 'llama' in args.model:
      prompt_base = prompt_codellama if args.dataset =='humanevalplus' else prompt_codellama_c
      model_name = 'llama'
   elif 'gpt' in args.model:
      prompt_base = prompt_gpt if args.dataset =='humanevalplus' else prompt_gpt_c
      model_name = 'gpt'
   elif 'gemma' in args.model:
      prompt_base = prompt_gemma if args.dataset =='humanevalplus' else prompt_gemma_c
      model_name = 'gemma'
   else:
      raise Exception('model not recognised')

   file_output = f'results_{args.dataset}_{model_name}_greedy.json' if args.sub_benchmark is None else f'results_{args.dataset}_{args.sub_benchmark}_{model_name}_greedy.json'
   print(f"Running on dataset: {args.dataset} and model: {model_name} in Greedy Mode")

   if os.path.exists(os.path.join('..', 'data', f'{args.dataset}' if args.sub_benchmark is None \
                                 else os.path.join('sub_benchmarks', f'{args.dataset.lower()}_hard'), 'raw', file_output)):
      with open(os.path.join('..', 'data', f'{args.dataset}' if args.sub_benchmark is None \
                                 else os.path.join('sub_benchmarks', f'{args.dataset.lower()}_hard'), 'raw', file_output), 'r') as f:
         all_tasks = json.load(f)
   else:
      all_tasks = {}

   for task_id in tqdm.tqdm(dat.index.to_numpy()):
        if str(task_id) in all_tasks:
           print('Skipped')
           continue

        # Setting base prompt
        if args.dataset == 'humanevalplus':
          signature = dat.loc[task_id]['signature']
          prompt_base_plus = prompt_base.replace('CODE_PLACEHOLDER', signature)
        elif args.dataset == 'ClassEval':
          class_name = dat.loc[task_id]['class_name']
          method_name = dat.loc[task_id]['method_name']
          class_text = dat.loc[task_id]['class_text']
          method_param = dat.loc[task_id]['method_params']
          signature = dat.loc[task_id]['method_signature']
          
          prompt_base_plus = prompt_base.replace('METHOD_NAME', method_name)
          prompt_base_plus = prompt_base_plus.replace('CLASS_NAME', class_name)
          prompt_base_plus = prompt_base_plus.replace('CLASS_CODE', class_text.strip())
          prompt_base_plus = prompt_base_plus.replace('METHOD_SIGNATURE', signature.strip())
        
        # We use the first level 1 prompt, i.e. without rephrasing prompt
        # It's the closest to the orginal benchmark prompt
        prompt = eval(dat.loc[task_id]['level 1'])[0]
              
        if args.dataset == 'humanevalplus':
            input_prompt = prompt_base_plus.replace('INSTRUCTION_PLACEHOLDER', prompt)
        elif args.dataset == 'ClassEval':
            # Get the indentation of the code
            indent = class_text.split('pass')[-2].split(":\n")[-1]
            prompt_ = f"\n{indent}" + f"\n{indent}".join(prompt.split("\n"))
            if method_param.strip() == '':
                 to_replace = indent + '\"\"\"' + prompt_ + '\n' + indent + '\"\"\"'
            else:
                 to_replace = indent + '\"\"\"' + prompt_ + '\n' + indent + f"\n{indent}".join(method_param.split("\n")) + '\"\"\"'
            input_prompt = prompt_base_plus.replace('DOCSTRING', to_replace).strip()

        if model_name != 'gpt':
               tokenized_text = tokenizer([input_prompt], return_tensors='pt').to('cuda')

               out = model.generate(**tokenized_text,\
                                        max_new_tokens=1024,
                                        do_sample=False,
                                        pad_token_id = tokenizer.eos_token_id
                                      )
               output = tokenizer.decode(out[0])
        else:
               inference_not_done = True
               number_of_trials = 0
               error_message = []
      
               # While inference is not completed or if we haven't tried at least 3 times (i.e. to avoid server error)
               while inference_not_done and number_of_trials != 3:
                     try:
                      # Max tokens is 1024 and temperature is left as default (1.0 I think?)
                      completion = client.chat.completions.create(
                        model='gpt-3.5-turbo-0125',
                        messages=[
                         {"role": "system", "content": input_prompt}
                        ],
                        max_tokens=1024,
                        temperature=0.0,
                        seed=0,
                        n=1
                      )
                      output = completion.choices[0].message.content 
                      inference_not_done = False
                     # in case something wrong happen, retry
                     except Exception as e:
                      error_message.append(e)
                      time.sleep(2)
                      number_of_trials += 1


        all_tasks[str(task_id)] = output
        # Writing in the script/ directory just in case. Actual data are in 'data/'
        with open(file_output, 'w') as f:
             json.dump(all_tasks, f)
