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

def get_per_llm_score(model, weights=[0.5, 0.3, 0.2]):
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
    
    score = []
    
    for i in range(len(mean_x_level1_tot)):
        score.append((1 - np.sum(np.array(weights) * np.array([mean_x_level3_tot[i], mean_x_level2_tot[i], mean_x_level1_tot[i]])), mean_x_level3_tot[i], mean_x_level2_tot[i], mean_x_level1_tot[i]) )
    
    return np.array(score)

glob_fin = []
# List of weights
alpha = np.array([0.33] + list(np.arange(0.35, 1.05, 0.05)))
alpha_inv = 1 - alpha
beta = 0.6 * alpha_inv
gamma = 0.4 * alpha_inv
weights = [[np.round(gamma[i], 2), np.round(beta[i], 2), np.round(alpha[i], 2)] for i in range(len(alpha))]
print(weights)

for w in weights:
    final_score = np.zeros(164)
    final_list = []
    task_diff_1 = np.zeros(164)
    task_diff_2 = np.zeros(164)
    task_diff_3 = np.zeros(164)
    models = ['deepseek', 'magicoder', 'gpt', 'gemma', 'llama']
    for sc in models:
        score_file = os.path.join('..', '..', 'data', 'humanevalplus', 'scores', f'score_humanevalplus_{sc}.npy')
        scores = get_per_llm_score(sc, w)

        final_score += np.array([s[0] for s in scores])
        final_list.append([s[0] for s in scores])
        task_diff_1 += np.array([s[3] for s in scores])
        task_diff_2 += np.array([s[2] for s in scores])
        task_diff_3 += np.array([s[1] for s in scores])

    final_score /= len(models)
    task_diff_1 /= len(models)
    task_diff_2 /= len(models)
    task_diff_3 /= len(models)
    glob_fin.append(final_score)

    task_indices = np.where(final_score >= 0.5)[0]
    print("Task IDs with a score higher than 0.5: ", list(task_indices))
    print("Number: ", len(task_indices))

s_list, m_list, w_list = [], [], []

for i, (w, g) in enumerate(zip(weights, glob_fin)):
    
    from scipy.stats import skew
    s = skew(g)
    s_list.append(s)
    m_list.append(np.std(g))
    w_list.append(w[2])
    #import seaborn as sns    
    #sns.kdeplot(g, label=round(w[2],2), clip=(0.0, 1.0))
#plt.xlabel('Task difficulty score')
#plt.legend()
#plt.show()

s_list_new = np.array(s_list)
m_list_new = np.array(m_list)
s_list_new = (s_list_new - min(s_list_new))/(max(s_list_new) - min(s_list_new))
m_list_new = (m_list_new - min(m_list_new))/(max(m_list_new) - min(m_list_new))

for w, s, k in zip(w_list, s_list_new, m_list_new):
    print(w)
    print("Dist to origin: ", np.linalg.norm([s, k]))

fig, ax = plt.subplots(figsize=(15, 7))
ax.annotate(r'$\alpha = 1/3$', (0.27, 0.173), xycoords='data', fontsize=25)
ax.annotate(r'$\alpha = 1$', (0.025, 0.22), xycoords='data', fontsize=25)
ax.scatter(s_list, m_list)
plt.xlim([0., 0.3])

plt.ylabel('Std $\\sigma$')
plt.xlabel(r'Skew $b_1$')
plt.grid(True)
plt.tight_layout()
plt.show()
