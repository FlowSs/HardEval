import json
import pandas as pd
from codebleu import calc_codebleu
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sub_benchmark', default=None)
args = parser.parse_args()

res_model_list = []

for model in ['deepseek', 'magicoder', 'llama', 'gpt', 'gemma']:
 file_to_load = f'results_humanevalplus_{args.sub_benchmark}_{model}_eval.json'
 with open(os.path.join('..', '..', 'data', 'sub_benchmarks', 'humanevalplus_hard', 'post_test', file_to_load), 'r') as f:
    res_model_list.append(json.load(f))

dataset_to_load = os.path.join('..', '..', 'data', 'sub_benchmarks', 'humanevalplus_hard', f'prompts_generated_humanevalplus_{args.sub_benchmark}.csv')
prompt = pd.read_csv(dataset_to_load, sep=";", index_col=0)
print("Building tokenized corpus and correct code corpus...")
correct_code_dict = {}

for res_list_list in res_model_list:
 for ind_key in prompt.index.values:
     correct_code_dict[ind_key] = [prompt.iloc[ind_key]['code']]
     for key in res_list_list[str(ind_key)]:
         for ind_seed in range(len(res_list_list[str(ind_key)][key])):
             if res_list_list[str(ind_key)][key][ind_seed][1] == True:
                 correct_code_dict[ind_key].append(res_list_list[str(ind_key)][key][ind_seed][0])

print("Finished building correct code corpus!")
print("Calculating CodeBLEU metric for each fragments...")
for p, model in enumerate(['deepseek', 'magicoder', 'llama', 'gpt', 'gemma']):
 file_to_save = f'results_humanevalplus_{args.sub_benchmark}_{model}_sim.json'
 avg_dict = {}
 for ind_key in prompt.index.values:
    avg_dict[int(ind_key)] = {}
    print("For Model {}, Running samples of task ======= {} ({}) =======".format(model, prompt.iloc[ind_key]['signature'], prompt.index[ind_key]))
    for key in res_model_list[p][str(ind_key)]:
        avg_dict[int(ind_key)][key] = []
        for ind_seed in range(len(res_model_list[p][str(ind_key)][key])):
             if res_model_list[p][str(ind_key)][key][ind_seed][1] == True:
                 max_sim = 1.0
             # If code is empty
             elif 'def' not in res_model_list[p][str(ind_key)][key][ind_seed][0]:
                 max_sim = 0
             else:
                 max_sim = 0
                 for i in range(len(correct_code_dict[ind_key])):
                     sc = calc_codebleu([correct_code_dict[ind_key][i]], [res_model_list[p][str(ind_key)][key][ind_seed][0]], lang="python", weights=(0.1, 0.1, 0.4, 0.4), tokenizer=None)
                     # Happens when the code is empty. Then sim should be globally 0
                     if sc['dataflow_match_score'] == 0:
                       continue
                     if sc['codebleu'] > max_sim:
                           max_sim = sc['codebleu']

             avg_dict[int(ind_key)][key].append(max_sim) 

 with open(os.path.join('..', '..', 'data', 'sub_benchmarks', 'humanevalplus_hard', 'sim', file_to_save), 'w') as f:
   json.dump(avg_dict, f)
