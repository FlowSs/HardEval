import numpy as np
import argparse
import os
from scipy.stats import spearmanr

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default=None)
args = parser.parse_args()

if args.dataset == 'humanevalplus':
    data_size = 164
elif args.dataset == 'ClassEval':
    data_size = 200
else:
    raise Exception(f"Dataset {args.dataset} not recognised")

std_per_task_lv1, std_per_task_lv2, std_per_task_lv3 = [], [], []
std_per_task_lv1_hard, std_per_task_lv2_hard, std_per_task_lv3_hard = [], [], []
std_per_task_lv1_easy, std_per_task_lv2_easy, std_per_task_lv3_easy = [], [], []

res_lv1, res_lv2, res_lv3 = np.zeros((data_size, 6)), np.zeros((data_size, 6)), np.zeros((data_size, 6))
res_lv1_test, res_lv2_test, res_lv3_test = np.zeros((data_size, 6)), np.zeros((data_size, 6)), np.zeros((data_size, 6))
res_lv1_sim, res_lv2_sim, res_lv3_sim = np.zeros((data_size, 6)), np.zeros((data_size, 6)), np.zeros((data_size, 6))

models_list = ["llama", "deepseek", "magicoder", "gemma", "gpt"]

def average_spearmanr(res_tot):
    spearman_rhos = np.array([spearmanr(res_tot[i], [0]*6 + [1]*6 + [2]*6).statistic for i in range(res_tot.shape[0])])
    # Spearman -> Person 
    r = 2*np.sin((1/6) * np.pi * np.array([k for k in spearman_rhos if str(k) != 'nan'])) # Removing task where all scores is 1 (i.e. perfect score) which leads to 'nan' for correlation
    # Pearson -> Fisher-Z
    z_r = 0.5 * np.log((1 + r)/(1 - r))
    # Average
    z_r_mean = np.mean(z_r)
    # Fisher-Z -> Pearson
    r_mean = (np.exp(2*z_r_mean) - 1)/(np.exp(2*z_r_mean) + 1)
    # Pearson -> Spearman
    spearman_rho_mean = 6*np.arcsin(r_mean/2)/np.pi
    return spearman_rho_mean

for model in models_list:
    print(f"\n******** STATS FOR {model.upper()} ********")
    score_file = f'{args.dataset}_{model}'
    dat = np.load(os.path.join('..', 'data', args.dataset.lower(), 'scores', f'score_per_task_{score_file}.npy'), allow_pickle=True)
    
    res_lv1_tmp = np.array([dat[i][2] for i in range(len(dat))], dtype=float)  
    res_lv1 += res_lv1_tmp

    res_lv2_tmp = np.array([dat[i][1] for i in range(len(dat))], dtype=float)  
    res_lv2 += res_lv2_tmp

    res_lv3_tmp = np.array([dat[i][0] for i in range(len(dat))], dtype=float)  
    res_lv3 += res_lv3_tmp
    print("Metrics: Composite")
    print("Correlation Across Task: ", spearmanr(np.concatenate((np.hstack(res_lv1_tmp), np.hstack(res_lv2_tmp), np.hstack(res_lv3_tmp)), axis=0), \
     [0] * (res_lv1_tmp.shape[0] * res_lv1_tmp.shape[1]) + [1] * (res_lv1_tmp.shape[0] * res_lv1_tmp.shape[1]) + [2] * (res_lv1_tmp.shape[0] * res_lv1_tmp.shape[1]))) #spearmanr(np.concatenate((mean_per_task_lv1, mean_per_task_lv2, mean_per_task_lv3), axis=0), [0] * len(mean_per_task_lv1) + [1] * len(mean_per_task_lv2) + [2] * len(mean_per_task_lv3)))
    
    res_tot = np.concatenate((res_lv1_tmp, res_lv2_tmp, res_lv3_tmp), axis=1)
    spearman_rho_mean = average_spearmanr(res_tot)
    print("Correlation Across Task (accounting for variability): ",spearman_rho_mean)

res_lv1 /= 5
res_lv2 /= 5
res_lv3 /= 5

print(f"\n******** STATS GLOBAL ********")
print("Metrics: Composite")
print("Correlation Across Task: ", spearmanr(np.concatenate((np.hstack(res_lv1), np.hstack(res_lv2), np.hstack(res_lv3)), axis=0), \
     [0] * (res_lv1.shape[0] * res_lv1.shape[1]) + [1] * (res_lv1.shape[0] * res_lv1.shape[1]) + [2] * (res_lv1.shape[0] * res_lv1.shape[1])))

res_tot = np.concatenate((res_lv1, res_lv2, res_lv3), axis=1)
spearman_rho_mean = average_spearmanr(res_tot)
print("Correlation Across Task (accounting for variability): ",spearman_rho_mean)