import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import pandas as pd
import scienceplots
from scipy.stats import spearmanr
from radon.complexity import cc_visit

plt.style.use(['science'])
plt.rcParams.update({
    "font.size":30})          # specify font size here

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default=None)
parser.add_argument('-s', '--sub_benchmark', default=None)
args = parser.parse_args()


dataset_to_load = pd.read_csv(f'prompts_generated_{args.dataset}_{args.sub_benchmark}.csv' if args.sub_benchmark is not None else f'prompts_generated_{args.dataset}.csv', sep=';')

final_score = np.zeros(len(dataset_to_load))
final_list = []
task_diff_1 = np.zeros(len(dataset_to_load))
task_diff_2 = np.zeros(len(dataset_to_load))
task_diff_3 = np.zeros(len(dataset_to_load))
models = ['deepseek', 'magicoder', 'gpt', 'gemma', 'llama']
for sc in models:
    score_file = f'score_{args.dataset}_{args.sub_benchmark}_{sc}.npy' if args.sub_benchmark is not None else f'score_{args.dataset}_{sc}.npy'
    scores = np.load(os.path.join('scores', score_file), allow_pickle=True)
    final_score += np.array([s[0] for s in scores])
    final_list.append([s[0] for s in scores])
    task_diff_1 += np.array([s[3] for s in scores])
    task_diff_2 += np.array([s[2] for s in scores])
    task_diff_3 += np.array([s[1] for s in scores])

final_score /= len(models)
task_diff_1 /= len(models)
task_diff_2 /= len(models)
task_diff_3 /= len(models)

loc_code = []
for ind, row in dataset_to_load.iterrows():
    #loc_code.append(len([e for e in row['code'].split('\n') if e.strip() != '']))
    c = None
    for ele in cc_visit(row['code']):
        if ele.name == row['signature'].split('(')[0]:
            c = ele.complexity
            break
    #print(c)
    loc_code.append(c)

"""
loc_code = np.array(loc_code)
hard_taks = [1, 39, 50, 64, 65, 67, 76, 86, 91, 93, 109, 113, 115, 119, 120, 124, 125, 127, 129, 130, 131, 132, 133, 134, 137, 140, 141, 145]
plt.scatter(loc_code, final_score)
plt.scatter(loc_code[hard_taks], final_score[hard_taks], color='red')
plt.xlabel('LOC')
plt.ylabel('Difficulty score')
plt.show()
"""
fig = plt.figure(figsize=(14, 7))
a = plt.hist(final_score, bins=np.arange(0, 1.1, 0.1), label='Average LLMs', linewidth=3,  density=True, histtype='step',
                           cumulative=True)
print(a)
print(len(np.where(final_score < 0.4)[0])/len(final_score))
print(len(np.where(final_score >= 0.5)[0])/len(final_score))
for i in range(len(models)):
    a = plt.hist(final_list[i], bins=np.arange(0, 1.1, 0.1), label=f'{models[i]}', linewidth=3, density=True, histtype='step',
                           cumulative=True)
    print("Models : ", models[i])
    print(a)
    print(len(np.where(np.array(final_list[i]) < 0.4)[0])/len(final_list[i]))
    print(len(np.where(np.array(final_list[i]) >= 0.5)[0])/len(final_list[i]))
easy_all = set(np.arange(0, len(dataset_to_load)))
for i in range(len(models)):
    easy_all = easy_all & set(np.where(np.array(final_list[i]) < 0.5)[0])
print(len(easy_all)/len(dataset_to_load))
#plt.title("Over all LLMs.\nHistogram of number of tasks with a difficulty score in a given range.")
plt.xlabel('Task difficulty score')
plt.ylabel('Frequency of tasks')
plt.legend(loc='lower right', bbox_to_anchor=(0.9,-0.01))
plt.tight_layout()
plt.savefig('cum_humanevalplus.pdf', dpi=800)
plt.show()

task_indices = np.where(final_score >= 0.5)[0]
print("Task IDs with a score higher than 0.5: ", list(task_indices))
print("Number: ", len(task_indices))
print("Obtained score per level: ")
for i in task_indices:
    print(f"For task {i}, Overall score is {final_score[i]}, Level 1 is {task_diff_1[i]}, Level 2 is {task_diff_2[i]} and Level 3 is {task_diff_3[i]}")
