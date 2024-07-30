import numpy as np
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import argparse

def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def get_per_llm_score(model, sub_benchmark):
    print(f"STATS FOR {model.upper()}")
    eval_file = f'results_ClassEval_{sub_benchmark}_{model}_eval.json'
    sim_file = f'results_ClassEval_{sub_benchmark}_{model}_sim.json'
    with open(os.path.join('..', '..', 'data', 'sub_benchmarks', 'classeval_hard', 'post_test', eval_file), 'r') as f:
        res_list_list = json.load(f)
    with open(os.path.join('..', '..', 'data', 'sub_benchmarks', 'classeval_hard', 'sim', sim_file), 'r') as f:
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

    score_per_task_level1 = np.array([(x_level1[i] + x_level1_sim[i])/2 for i in range(len(x_level1_sim))])
    score_per_task_level1 = np.array([np.mean(score_per_task_level1[:,i*5:i*5+5], axis=-1) for i in range(6)]).T
    score_per_task_level2 = np.array([(x_level2[i] + x_level2_sim[i])/2 for i in range(len(x_level2_sim))])
    score_per_task_level2 = np.array([np.mean(score_per_task_level2[:,i*5:i*5+5], axis=-1) for i in range(6)]).T
    score_per_task_level3 = np.array([(x_level3[i] + x_level3_sim[i])/2 for i in range(len(x_level3_sim))])
    score_per_task_level3 = np.array([np.mean(score_per_task_level3[:,i*5:i*5+5], axis=-1) for i in range(6)]).T

    print("############################################")
    print("Calculating difficulty score per task")
    score, score_per_task = [], []
    weights = [0.15, 0.225, 0.625]

    for i in range(len(mean_x_level1_tot)):
        score.append((1 - np.sum(np.array(weights) * np.array([mean_x_level3_tot[i], mean_x_level2_tot[i], mean_x_level1_tot[i]])), mean_x_level3_tot[i], mean_x_level2_tot[i], mean_x_level1_tot[i]) )
        score_per_task.append( (score_per_task_level3[i], score_per_task_level2[i], score_per_task_level1[i]) )

    # Uncomment to save
    # score_file = f'ClassEval_{sub_benchmark}_{model}'
    # np.save(os.path.join('..', '..', 'data', 'sub_benchmarks', 'classeval_hard', 'scores', f'score_{score_file}.npy'), np.array(score, dtype=object))
    # np.save(os.path.join('..', '..', 'data', 'sub_benchmarks', 'classeval_hard', 'scores', f'score_oer_task_{score_file}.npy'), np.array(score_per_task, dtype=object))

    task_indices = np.where(np.array([s[0] for s in score]) > 0.5)[0]
    print("Task IDs with a score higher than 0.5: ", list(task_indices))
    print("Number: ", len(task_indices))
    #print("Obtained score per level: ")
    #for i in task_indices:
    #    print(f"For task {i}, Overall score is {score[i][0]}, Level 1 is {score[i][3]}, Level 2 is {score[i][2]} and Level 3 is {score[i][1]}")
    
    return np.array(score)


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sub_benchmark', default=None)
args = parser.parse_args()

final_score = np.zeros(5)
final_list = []
task_diff_1 = np.zeros(5)
task_diff_2 = np.zeros(5)
task_diff_3 = np.zeros(5)
models = ['deepseek', 'magicoder', 'gpt', 'gemma', 'llama']
for sc in models:
    score_file = os.path.join('..', '..', 'data', 'sub_benchmarks', 'classeval_hard', 'scores', f'score_ClassEval_{args.sub_benchmark}_{sc}.npy')
    scores = get_per_llm_score(sc, args.sub_benchmark)

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
#print("Task IDs with a score higher than 0.5: ", list(task_indices))
#print("Number: ", len(task_indices))
print("Obtained score per level: ")
for i in task_indices:
    print(f"For task {i}, Overall score is {final_score[i]}, Level 1 is {task_diff_1[i]}, Level 2 is {task_diff_2[i]} and Level 3 is {task_diff_3[i]}")
