import numpy as np
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science'])
plt.rcParams.update({
    "font.size":30})

def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def get_per_llm_score(model, weights=[0.15, 0.225, 0.625]):
    assert sum(weights) == 1.0, "Weights for context information does not sum to 1"
    
    eval_file = f'results_humanevalplus_{model}_eval.json'
    sim_file = f'results_humanevalplus_{model}_sim.json'
    with open(os.path.join('..', '..', 'data', 'humanevalplus', 'post_test', eval_file), 'r') as f:
        res_list_list = json.load(f)
    with open(os.path.join('..', '..', 'data', 'humanevalplus', 'sim', sim_file), 'r') as f:
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

    
    score, score_per_task = [], []
    
    for i in range(len(mean_x_level1_tot)):
        score.append((1 - np.sum(np.array(weights) * np.array([mean_x_level3_tot[i], mean_x_level2_tot[i], mean_x_level1_tot[i]])), mean_x_level3_tot[i], mean_x_level2_tot[i], mean_x_level1_tot[i]) )
        score_per_task.append( (score_per_task_level3[i], score_per_task_level2[i], score_per_task_level1[i]) )

    # Uncomment to save
    score_file = f'humanevalplus_{model}'
    np.save(os.path.join('..', '..', 'data', 'humanevalplus', 'scores', f'score_{score_file}.npy'), np.array(score, dtype=object))
    np.save(os.path.join('..', '..', 'data', 'humanevalplus', 'scores', f'score_per_task_{score_file}.npy'), np.array(score_per_task, dtype=object))

    task_indices = np.where(np.array([s[0] for s in score]) > 0.5)[0]
    
    print(f"STATS FOR {model.upper()}")
    print("############################################")
    print("Avg Pass@1 for each level:")
    print("Level 1: ", np.mean([pass_at_k(len(c), sum(c), 1) for _, c in enumerate(x_level1)]))
    print("Level 2: ", np.mean([pass_at_k(len(c), sum(c), 1) for _, c in enumerate(x_level2)]))
    print("Level 3: ", np.mean([pass_at_k(len(c), sum(c), 1) for _, c in enumerate(x_level3)]))

    print("############################################")
    print("Calculating difficulty score per task")
    print("Task IDs with a score higher than 0.5: ", list(task_indices))
    print("Number: ", len(task_indices))
    #print("Obtained score per level: ")
    #for i in task_indices:
    #    print(f"For task {i}, Overall score is {score[i][0]}, Level 1 is {score[i][3]}, Level 2 is {score[i][2]} and Level 3 is {score[i][1]}")
    
    return np.array(score)

final_score = np.zeros(164)
final_list = []
task_diff_1 = np.zeros(164)
task_diff_2 = np.zeros(164)
task_diff_3 = np.zeros(164)
models = ['deepseek', 'magicoder', 'gpt', 'gemma', 'llama']
for sc in models:
    score_file = os.path.join('..', '..', 'data', 'humanevalplus', 'scores', f'score_humanevalplus_{sc}.npy')
    scores = get_per_llm_score(sc)

    final_score += np.array([s[0] for s in scores])
    final_list.append([s[0] for s in scores])
    task_diff_1 += np.array([s[3] for s in scores])
    task_diff_2 += np.array([s[2] for s in scores])
    task_diff_3 += np.array([s[1] for s in scores])

final_score /= len(models)
task_diff_1 /= len(models)
task_diff_2 /= len(models)
task_diff_3 /= len(models)

print("Pct of task below 0.1: ", np.mean(final_score <= 0.1))
print("Pct of task below 0.4: ", np.mean(final_score <= 0.4))
print("Pct of task above 0.5: ", np.mean(final_score >= 0.5))

fig = plt.figure(figsize=(14, 7))
a = plt.hist(final_score, bins=np.arange(0, 1.01, 0.01), label='Average LLMs', linewidth=3,  density=True, histtype='step',
                        cumulative=True)

for i in range(len(models)):
    a = plt.hist(final_list[i], bins=np.arange(0, 1.01, 0.01), label=f'{models[i]}', linewidth=3, density=True, histtype='step',
                        cumulative=True)
    
plt.xlabel('Task difficulty score')
plt.ylabel('Frequency of tasks')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.yticks([0.25, 0.5, 0.75, 1.0])
plt.xlim([0, 1])
plt.legend(loc='lower right', bbox_to_anchor=(0.9,-0.01))
plt.minorticks_off()
plt.grid(True)
plt.tight_layout()
plt.show()

task_indices = np.where(final_score >= 0.5)[0]
print("Task IDs with a score higher than 0.5: ", list(task_indices))
print("Number: ", len(task_indices))
#print("Obtained score per level: ")
#for i in task_indices:
#    print(f"For task {i}, Overall score is {final_score[i]}, Level 1 is {task_diff_1[i]}, Level 2 is {task_diff_2[i]} and Level 3 is {task_diff_3[i]}")
