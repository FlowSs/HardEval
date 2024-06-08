import json
import pandas as pd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default=None)
parser.add_argument('-d', '--dataset', default=None)
parser.add_argument('-s', '--sub_benchmark', default=None)
args = parser.parse_args()

model = args.model
dataset = args.dataset
sub_benchmark = args.sub_benchmark

print(f"STATS FOR {model.upper()}")
eval_file = f'post_test/results_{dataset}_{sub_benchmark}_{model}_eval.json' if sub_benchmark is not None else f'post_test/results_{dataset}_{model}_eval.json'
sim_file = f'sim/results_{dataset}_{sub_benchmark}_{model}_sim.json' if sub_benchmark is not None else f'sim/results_{dataset}_{model}_sim.json'
with open(eval_file, 'r') as f:
    res_list_list = json.load(f)
with open(sim_file, 'r') as f:
    avg_list_list = json.load(f)  

x_level1 = [np.concatenate([[res_list[key][i][1] for i in range(len(res_list[key]))] for key in res_list if 'level 1' in key]) for (key, res_list) in res_list_list.items()]
x_level2 = [np.concatenate([[res_list[key][i][1] for i in range(len(res_list[key]))] for key in res_list if 'level 2' in key]) for (key, res_list) in res_list_list.items()]
x_level3 = [np.concatenate([[res_list[key][i][1] for i in range(len(res_list[key]))] for key in res_list if 'level 3' in key]) for (key, res_list) in res_list_list.items()]

x_level1_sim = [np.concatenate([[avg_list[key][i] for i in range(len(avg_list[key]))] for key in avg_list if 'level 1' in key]) for (key, avg_list) in avg_list_list.items()]
x_level2_sim = [np.concatenate([[avg_list[key][i] for i in range(len(avg_list[key]))] for key in avg_list if 'level 2' in key]) for (key, avg_list) in avg_list_list.items()]
x_level3_sim = [np.concatenate([[avg_list[key][i] for i in range(len(avg_list[key]))] for key in avg_list if 'level 3' in key]) for (key, avg_list) in avg_list_list.items()]

mean_x_level1_tot = np.mean([(x_level1[i] + x_level1_sim[i])/2 for i in range(len(x_level1_sim))], axis=-1)
mean_x_level2_tot = np.mean([(x_level2[i] + x_level2_sim[i])/2 for i in range(len(x_level2_sim))], axis=-1)
mean_x_level3_tot = np.mean([(x_level3[i] + x_level3_sim[i])/2 for i in range(len(x_level3_sim))], axis=-1)

print("############################################")
print("Calculating difficulty score per task")
score = []
weights = [0.2, 0.3, 0.5]

for i in range(len(mean_x_level1)):
    score.append((1 - np.sum(np.array(weights) * np.array([mean_x_level3_tot[i], mean_x_level2_tot[i], mean_x_level1_tot[i]])), mean_x_level3_tot[i], mean_x_level2_tot[i], mean_x_level1_tot[i]) )

score_file = f'{dataset}_{sub_benchmark}_{model}' if sub_benchmark is not None else f'{dataset}_{model}'
np.save(os.path.join('scores', f'score_{score_file}.npy'), np.array(score, dtype=object))

task_indices = np.where(np.array([s[0] for s in score]) > 0.5)[0]
print("Task IDs with a score higher than 0.5: ", list(task_indices))
print("Number: ", len(task_indices))
print("Obtained score per level: ")
for i in task_indices:
    print(f"For task {i}, Overall score is {score[i][0]}, Level 1 is {score[i][3]}, Level 2 is {score[i][2]} and Level 3 is {score[i][1]}")

