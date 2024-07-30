import json
import numpy as np
import pandas as pd
import argparse
import os

def jaccard(a, b):
    return len(set(a) & set(b))/len(set(a) | set(b))

def acc(sc, task):
    
    return np.mean(sc[task])

def acc_med(sc, task):
    
    return np.median(sc[task]).round(4)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default=None)
parser.add_argument('-w', '--weights', default=None, type=str, nargs='+')
args = parser.parse_args()

if args.dataset == 'humanevalplus':
    task_nb = 164
elif args.dataset == 'ClassEval':
    task_nb = 200
    sampled_tasks = pd.read_csv(os.path.join('..', 'data', 'classeval', 'sampled_tasks.csv'), sep=';')
else:
    raise Exception(f"Dataset {args.dataset} not recognised")

if args.weights is None or args.weights == ['0.15', '0.225', '0.625']:
    weights = ''
else:
    weights = '_' + args.weights[0] + '_' + args.weights[1] + '_' + args.weights[2]

models = ["llama", "magicoder", "deepseek", "gemma", "gpt"]
res = np.zeros((task_nb, len(models)))
sc_fin = np.zeros((task_nb, len(models)))

for ind, mod in enumerate(models):
    print(f"For model {mod}")
    with open(os.path.join('..', 'data', args.dataset.lower(), 'post_test', f'results_{args.dataset}_{mod}_greedy_eval.json')) as f:
        dat = json.load(f)
    res[:, ind] = np.array([dat[key][1] for key in dat]) if args.dataset == 'humanevalplus' else np.array([dat['ClassEval_' + key][1] for key in sampled_tasks['class_id'].values])
    sc_fin[:, ind] = np.load(os.path.join('..', 'data', args.dataset.lower(), 'scores', f'score_{args.dataset}_{mod}{weights}.npy'), allow_pickle=True)[:, 0]
    
    print("Acc on greedy: ", np.mean(res[:, ind]))
    curr_hard_comp = np.where(sc_fin[:,ind] > 0.5)[0]
    print("Pct of hard tasks: " + str(round(len(curr_hard_comp)/task_nb * 100, 2)) + '%')
    curr_easy_comp = list(set(np.arange(0, task_nb)) - set(curr_hard_comp))
    print("Composite: ")
    print("Hard")
    print("Overlap between hard task (greedy) and hard task (diff score): ", jaccard(curr_hard_comp, np.where(res[:, ind] == 0)[0]))
    print("Accuracy of greedy on DiffEval hard task: " + str(acc(res[:, ind], curr_hard_comp).round(5)* 100) + "%")
    print("Average diff score on Greedy hard task", acc_med(sc_fin[:, ind], np.where(res[:, ind] == 0)[0]))
    print(np.quantile(sc_fin[:, ind][np.where(res[:, ind] == 0)[0]], 0.25), np.quantile(sc_fin[:, ind][np.where(res[:, ind] == 0)[0]], 0.75))
    print("Easy")   
    print("Overlap between easy task (greedy) and easy task (diff score): ", jaccard(curr_easy_comp, np.where(res[:, ind] == 1)[0]))
    print("Accuracy of greedy on DiffEval easy task: " + str(acc(res[:, ind], curr_easy_comp).round(5)* 100) + "%")
    print("Average diff score on Greedy easy task", acc_med(sc_fin[:, ind], np.where(res[:, ind] == 1)[0]))
    print(np.quantile(sc_fin[:, ind][np.where(res[:, ind] == 1)[0]], 0.25), np.quantile(sc_fin[:, ind][np.where(res[:, ind] == 1)[0]], 0.75))
    print("###############################")

   
print("For Global")
hard_tasks_comp = np.where(np.mean(sc_fin, axis=1) > 0.5)[0]
print("Pct of hard tasks: " + str(round(len(hard_tasks_comp)/task_nb * 100, 2)) + '%')
print("Composite: ")
print("Hard")
print("Overlap between hard task (greedy - majority fails) and hard task (diff score): ", jaccard(hard_tasks_comp, np.where(np.sum(res, axis=1) < 3)[0]))
print("Accuracy of greedy on DiffEval hard task: " + str(acc(np.mean(res, axis=1), hard_tasks_comp).round(5) * 100) + "%")
print("Average diff score on Greedy hard task (majority fails)", acc_med(np.mean(sc_fin, axis=1), np.where(np.sum(res, axis=1) < 3)[0]))
print(np.quantile(np.mean(sc_fin, axis=1)[np.where(np.sum(res, axis=1) < 3)[0]], 0.25), np.quantile(np.mean(sc_fin, axis=1)[np.where(np.sum(res, axis=1) < 3)[0]], 0.75))
    
print("Easy")
easy_curr_comp = list(set(np.arange(0, task_nb)) - set(hard_tasks_comp))
print("Overlap between easy task (majority correct) and easy task (diff score): ", jaccard(easy_curr_comp, np.where(np.sum(res, axis=1) > 2)[0]))
print("Accuracy of greedy on DiffEval easy task: " + str(acc(np.mean(res, axis=1), easy_curr_comp).round(5) * 100) + "%")
print("Average diff score on Greedy easy task (majority correct)", acc_med(np.mean(sc_fin, axis=1), np.where(np.sum(res, axis=1) > 2)[0]))
print(np.quantile(np.mean(sc_fin, axis=1)[np.where(np.sum(res, axis=1) > 2)[0]], 0.25), np.quantile(np.mean(sc_fin, axis=1)[np.where(np.sum(res, axis=1) > 2)[0]], 0.75))

# import seaborn as sns
# import matplotlib.pyplot as plt
# corr = np.corrcoef(np.hstack((sc_fin, np.mean(sc_fin, axis=1)[:,None])).T)
# f, ax = plt.subplots(figsize=(11, 9))
# sns.heatmap(corr, label=list(models) + ['Overall'], annot=True, \
# xticklabels=list(models) + ['Overall'], yticklabels=list(models) + ['Overall'], cmap='Reds', mask=np.triu(np.ones_like(corr, dtype=bool)))

# plt.show()

# f, ax = plt.subplots(figsize=(11, 9))

# print((np.sum(res, axis=1) > 3).shape)
# corr = np.zeros((len(models)+1, len(models)+1))
# res = np.hstack((res, (np.sum(res, axis=1) > 2)[:,None]))
# for i in range(len(models)):
#     for j in range(i, len(models)+1):
#         if i == j:
#             continue
#         corr[i, j] = np.mean(1 - res[:, i].astype(bool) ^ res[:,j].astype(bool))
# sns.heatmap(corr, label=list(models) + ['Overall'], annot=True, \
#  xticklabels=list(models) + ['Overall'], yticklabels=list(models) + ['Overall'])

# plt.show()

# u, c = np.unique(np.sum(res, axis=1), return_counts=True)
# print(u, c)
# plt.bar([0, 1, 2, 3, 4, 5], c/res.shape[0])
# plt.show()