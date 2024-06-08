import numpy as np
import os
import argparse
import pandas as pd

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

task_indices = np.where(final_score >= 0.0)[0]
print("Task IDs with a score higher than 0.5: ", list(task_indices))
print("Number: ", len(task_indices))
print("Obtained score per level: ")
for i in task_indices:
    print(f"For task {i}, Overall score is {final_score[i]}, Level 1 is {task_diff_1[i]}, Level 2 is {task_diff_2[i]} and Level 3 is {task_diff_3[i]}")
