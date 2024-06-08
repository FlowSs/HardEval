import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau
import argparse
import os

def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default=None)
parser.add_argument('-d', '--dataset', default=None)
parser.add_argument('-s', '--sub_benchmark', default=None)
args = parser.parse_args()

model = args.model
dataset = args.dataset
sub_benchmark = args.sub_benchmark

print(f"STATS FOR {model.upper()}")
eval_file = f'results_{dataset}_{sub_benchmark}_{model}_eval.json' if sub_benchmark is not None else f'results_{dataset}_{model}_eval.json'
sim_file = f'results_{dataset}_{sub_benchmark}_{model}_sim.json' if sub_benchmark is not None else f'results_{dataset}_{model}_sim.json'
with open(eval_file, 'r') as f:
    res_list_list = json.load(f)
with open(sim_file, 'r') as f:
    avg_list_list = json.load(f)  

x_original = [np.concatenate([[res_list[key][i][1] for i in range(len(res_list[key]))] for key in res_list if 'original' in key]) for (key, res_list) in res_list_list.items()]
x_level1 = [np.concatenate([[res_list[key][i][1] for i in range(len(res_list[key]))] for key in res_list if 'level 1' in key]) for (key, res_list) in res_list_list.items()]
x_level2 = [np.concatenate([[res_list[key][i][1] for i in range(len(res_list[key]))] for key in res_list if 'level 2' in key]) for (key, res_list) in res_list_list.items()]
x_level3 = [np.concatenate([[res_list[key][i][1] for i in range(len(res_list[key]))] for key in res_list if 'level 3' in key]) for (key, res_list) in res_list_list.items()]

x_original_sim = [np.concatenate([[avg_list[key][i] for i in range(len(avg_list[key]))] for key in avg_list if 'original' in key]) for (key, avg_list) in avg_list_list.items()]
x_level1_sim = [np.concatenate([[avg_list[key][i] for i in range(len(avg_list[key]))] for key in avg_list if 'level 1' in key]) for (key, avg_list) in avg_list_list.items()]
x_level2_sim = [np.concatenate([[avg_list[key][i] for i in range(len(avg_list[key]))] for key in avg_list if 'level 2' in key]) for (key, avg_list) in avg_list_list.items()]
x_level3_sim = [np.concatenate([[avg_list[key][i] for i in range(len(avg_list[key]))] for key in avg_list if 'level 3' in key]) for (key, avg_list) in avg_list_list.items()]


score_per_task_level1 = np.array([(x_level1[i] + x_level1_sim[i])/2 for i in range(len(x_level1_sim))])
score_per_task_level1 = np.array([np.mean(score_per_task_level1[:,i*5:i*5+5], axis=-1) for i in range(6)]).T
score_per_task_level2 = np.array([(x_level2[i] + x_level2_sim[i])/2 for i in range(len(x_level2_sim))])
score_per_task_level2 = np.array([np.mean(score_per_task_level2[:,i*5:i*5+5], axis=-1) for i in range(6)]).T
score_per_task_level3 = np.array([(x_level3[i] + x_level3_sim[i])/2 for i in range(len(x_level3_sim))])
score_per_task_level3 = np.array([np.mean(score_per_task_level3[:,i*5:i*5+5], axis=-1) for i in range(6)]).T

score_per_task_level1_test = np.array([x_level1[i] for i in range(len(x_level1_sim))])
score_per_task_level1_test = np.array([np.mean(score_per_task_level1_test[:,i*5:i*5+5], axis=-1) for i in range(6)]).T
score_per_task_level2_test = np.array([x_level2[i] for i in range(len(x_level2_sim))])
score_per_task_level2_test = np.array([np.mean(score_per_task_level2_test[:,i*5:i*5+5], axis=-1) for i in range(6)]).T
score_per_task_level3_test = np.array([x_level3[i] for i in range(len(x_level3_sim))])
score_per_task_level3_test = np.array([np.mean(score_per_task_level3_test[:,i*5:i*5+5], axis=-1) for i in range(6)]).T

score_per_task_level1_sim = np.array([x_level1_sim[i] for i in range(len(x_level1_sim))])
score_per_task_level1_sim = np.array([np.mean(score_per_task_level1_sim[:,i*5:i*5+5], axis=-1) for i in range(6)]).T
score_per_task_level2_sim = np.array([x_level2_sim[i] for i in range(len(x_level2_sim))])
score_per_task_level2_sim = np.array([np.mean(score_per_task_level2_sim[:,i*5:i*5+5], axis=-1) for i in range(6)]).T
score_per_task_level3_sim = np.array([x_level3_sim[i] for i in range(len(x_level3_sim))])
score_per_task_level3_sim = np.array([np.mean(score_per_task_level3_sim[:,i*5:i*5+5], axis=-1) for i in range(6)]).T

mean_x_original_tot = np.mean([(x_original[i] + x_original_sim[i])/2 for i in range(len(x_original_sim))], axis=-1)
mean_x_level1_tot = np.mean([(x_level1[i] + x_level1_sim[i])/2 for i in range(len(x_level1_sim))], axis=-1)
mean_x_level2_tot = np.mean([(x_level2[i] + x_level2_sim[i])/2 for i in range(len(x_level2_sim))], axis=-1)
mean_x_level3_tot = np.mean([(x_level3[i] + x_level3_sim[i])/2 for i in range(len(x_level3_sim))], axis=-1)

mean_x_original_sim = np.mean(x_original_sim, axis=-1)
mean_x_level1_sim = np.mean(x_level1_sim, axis=-1)
mean_x_level2_sim = np.mean(x_level2_sim, axis=-1)
mean_x_level3_sim = np.mean(x_level3_sim, axis=-1)

mean_x_original = np.mean(x_original, axis=-1)
mean_x_level1 = np.mean(x_level1, axis=-1)
mean_x_level2 = np.mean(x_level2, axis=-1)
mean_x_level3 = np.mean(x_level3, axis=-1)

fig, axs = plt.subplots(2, 2, figsize=(15, 7))
lns1=axs[0][0].plot(np.arange(len(x_original)), mean_x_original_tot, marker='+', markersize=10, c='blue', label='original')
lns2=axs[0][1].plot(np.arange(len(x_level1)), mean_x_level1_tot, marker='+', markersize=10, c='orange', label='level 1')
lns3=axs[1][0].plot(np.arange(len(x_level2)), mean_x_level2_tot, marker='+', markersize=10, c='green', label='level 2')
lns4=axs[1][1].plot(np.arange(len(x_level3)), mean_x_level3_tot, marker='+', markersize=10, c='red', label='level 3')

lns1=axs[0][0].plot(np.arange(len(x_original)), mean_x_original_sim, c='blue', linestyle='dashed')
lns2=axs[0][1].plot(np.arange(len(x_level1)), mean_x_level1_sim, c='orange',  linestyle='dashed')
lns3=axs[1][0].plot(np.arange(len(x_level2)), mean_x_level2_sim,  c='green', linestyle='dashed')
lns4=axs[1][1].plot(np.arange(len(x_level3)), mean_x_level3_sim,c='red', linestyle='dashed')

lns1=axs[0][0].plot(np.arange(len(x_original)), mean_x_original,c='blue', linestyle='dotted')
lns2=axs[0][1].plot(np.arange(len(x_level1)), mean_x_level1, c='orange', linestyle='dotted')
lns3=axs[1][0].plot(np.arange(len(x_level2)), mean_x_level2, c='green', linestyle='dotted')
lns4=axs[1][1].plot(np.arange(len(x_level3)), mean_x_level3, c='red', linestyle='dotted')
#plt.plot(np.arange(len(x_level3)), (mean_x_level1 + mean_x_level2 + mean_x_level3)/3, marker='+', markersize=10)
for i in range(2):
 for j in range(2):
   axs[i][j].set_xticks(np.arange(len(x_original)))
   axs[i][j].set_xlabel('Task id')
   axs[i][j].set_ylim(0, 1.05)

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels)
#plt.hlines(0.8, 0, len(x_original), linestyles='dashed', color='black')
fig.text(0.04, 0.5, '(Plain) Average score per task,\n(Dashed) Average similarity to correct code,\n(Dotted) Average number of correct codes', va='center', rotation='vertical')
#plt.show()
plt.close()

print("############################################")
print("Avg Pass@1 for each level:")
print("Original prompt: ", np.mean([pass_at_k(len(c), sum(c), 1) for c in x_original]))
print("Level 1: ", np.mean([pass_at_k(len(c), sum(c), 1) for i, c in enumerate(x_level1)]))
print("Level 2: ", np.mean([pass_at_k(len(c), sum(c), 1) for i, c in enumerate(x_level2)]))
print("Level 3: ", np.mean([pass_at_k(len(c), sum(c), 1) for i, c in enumerate(x_level3)]))
#assert 1 == 0
print("############################################")
print("Correlation of avg number of correct samples and level of prompts")
print(kendalltau([mean_x_level3, mean_x_level2, mean_x_level1], [2]*len(mean_x_level3) + [1]*len(mean_x_level2) + [0]*len(mean_x_level1)))
print("Correlation of avg similarity of samples and level of prompts")
print(kendalltau([mean_x_level3_sim, mean_x_level2_sim, mean_x_level1_sim], [2]*len(mean_x_level3_sim) + [1]*len(mean_x_level2_sim) + [0]*len(mean_x_level1_sim)))
print("Correlation of avg difficulty score (# correct sample + similarity) of samples and level of prompts")
print(kendalltau([mean_x_level3_tot, mean_x_level2_tot, mean_x_level1_tot], [2]*len(mean_x_level3_tot) + [1]*len(mean_x_level2_tot) + [0]*len(mean_x_level1_tot)))

print("############################################")
print("Calculating difficulty score per task")
score = []
weights = [0.2, 0.3, 0.5]

for i in range(len(mean_x_original)):
    score.append((1 - np.sum(np.array(weights) * np.array([mean_x_level3_tot[i], mean_x_level2_tot[i], mean_x_level1_tot[i]])), mean_x_level3_tot[i], mean_x_level2_tot[i], mean_x_level1_tot[i]) )

score_per_task, score_per_task_test, score_per_task_sim = [], [], []
for i in range(len(score_per_task_level1)):
    score_per_task.append( (score_per_task_level3[i], score_per_task_level2[i], score_per_task_level1[i]) )
    score_per_task_test.append( (score_per_task_level3_test[i], score_per_task_level2_test[i], score_per_task_level1_test[i]) )
    score_per_task_sim.append( (score_per_task_level3_sim[i], score_per_task_level2_sim[i], score_per_task_level1_sim[i]) )

fig = plt.figure(figsize=(14, 7))
plt.hist([s[0] for s in score], bins=np.arange(0, 1.1, 0.1), label='Weighted Average', histtype=u'step', linewidth=3)
plt.hist([s[3] for s in score], bins=np.arange(0, 1.1, 0.1), label='Level 1 only', histtype=u'step', linewidth=1.2)
plt.hist([s[2] for s in score], bins=np.arange(0, 1.1, 0.1), label='Level 2 only', histtype=u'step', linewidth=1.2)
plt.hist([s[1] for s in score], bins=np.arange(0, 1.1, 0.1), label='Level 3 only', histtype=u'step', linewidth=1.2)
plt.title(f"For model {model.upper()}.\nHistogram of number of tasks with a difficulty score in a given range.\nBins range from, for example, [0.4, 0.5[")
plt.xlabel('Task difficulty score')
plt.ylabel('Frequency')
plt.legend(loc='upper left')
#plt.show()
score_file = f'{dataset}_{sub_benchmark}_{model}' if sub_benchmark is not None else f'{dataset}_{model}'
np.save(os.path.join('scores', f'score_{score_file}.npy'), np.array(score, dtype=object))
np.save(os.path.join('scores',f'score_per_task_{score_file}.npy'), np.array(score_per_task, dtype=object))
np.save(os.path.join('scores',f'score_per_task_test_{score_file}.npy'), np.array(score_per_task_test, dtype=object))
np.save(os.path.join('scores',f'score_per_task_sim_{score_file}.npy'), np.array(score_per_task_sim, dtype=object))

task_indices = np.where(np.array([s[0] for s in score]) > 0.5)[0]
print("Task IDs with a score higher than 0.5: ", list(task_indices))
print("Number: ", len(task_indices))
task_indices = np.where(np.array([1 - (np.mean(s[0]) * 0.2 + np.mean(s[1]) * 0.3 + np.mean(s[2]) * 0.5) for s in score_per_task_test]) > 0.5)[0]
#print(np.array([1 - (np.mean(s[0]) * 0.2 + np.mean(s[1]) * 0.3 + np.mean(s[2]) * 0.5) for s in score_per_task_test]))
print("Task IDs with a score higher than 0.5 (test metric): ", list(task_indices))
print("Number: ", len(task_indices))
#print("Obtained score per level: ")
#for i in task_indices:
#    print(f"For task {i}, Overall score is {score[i][0]}, Level 1 is {score[i][3]}, Level 2 is {score[i][2]} and Level 3 is {score[i][1]}")

