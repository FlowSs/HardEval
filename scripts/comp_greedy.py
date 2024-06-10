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
args = parser.parse_args()

if args.dataset == 'humanevalplus':
    task_nb = 164
elif args.dataset == 'ClassEval':
    task_nb = 200
    sampled_tasks = pd.read_csv(os.path.join('..', 'data', 'classeval', 'sampled_tasks.csv'), sep=';')
else:
    raise Exception(f"Dataset {args.dataset} not recognised")

res = np.zeros((task_nb, 5))
sc_fin = np.zeros((task_nb, 5))

for ind, mod in enumerate(["llama", "deepseek", "magicoder", "gemma", "gpt"]):
    print(f"For model {mod}")
    with open(os.path.join('..', 'data', args.dataset.lower(), 'post_test', f'results_{args.dataset}_{mod}_greedy_eval.json')) as f:
        dat = json.load(f)
    res[:, ind] = np.array([dat[key][1] for key in dat]) if args.dataset == 'humanevalplus' else np.array([dat['ClassEval_' + key][1] for key in sampled_tasks['class_id'].values])
    sc_fin[:, ind] = np.load(os.path.join('..', 'data', args.dataset.lower(), 'scores', f'score_{args.dataset}_{mod}.npy'), allow_pickle=True)[:, 0]
    
    print("Acc on greedy: ", np.mean(res[:, ind]))
    curr_hard_comp = np.where(sc_fin[:,ind] > 0.5)[0]
    print("Nb of hard tasks: ", len(curr_hard_comp)/task_nb)
    curr_easy_comp = list(set(np.arange(0, task_nb)) - set(curr_hard_comp))
    print("Composite: ")
    print("Hard")
    print("Overlap between hard task (greedy) and hard task (diff score): ", jaccard(curr_hard_comp, np.where(res[:, ind] == 0)[0]))
    print("Accuracy of greedy on DiffEval hard task: " + str(acc(res[:, ind], curr_hard_comp).round(5)* 100) + "%")
    print("Average diff score on Greedy hard task", acc_med(sc_fin[:, ind], np.where(res[:, ind] == 0)[0]))
    print("Easy")   
    print("Overlap between easy task (greedy) and easy task (diff score): ", jaccard(curr_easy_comp, np.where(res[:, ind] == 1)[0]))
    print("Accuracy of greedy on DiffEval easy task: " + str(acc(res[:, ind], curr_easy_comp).round(5)* 100) + "%")
    print("Average diff score on Greedy easy task", acc_med(sc_fin[:, ind], np.where(res[:, ind] == 1)[0]))
    print("###############################")
    
print("For Global")
hard_tasks_comp = np.where(np.mean(sc_fin, axis=1) > 0.5)[0]
print("Nb of hard tasks: ", len(hard_tasks_comp)/task_nb)
print("Composite: ")
print("Hard")
print("Overlap between hard task (greedy - no model correct) and hard task (diff score): ", jaccard(hard_tasks_comp, np.where(np.sum(res, axis=1) == 0)[0]))
print("Accuracy of greedy on DiffEval hard task: " + str(acc(np.mean(res, axis=1), hard_tasks_comp).round(5) * 100) + "%")
print("Average diff score on Greedy hard task (all models fail)", acc_med(np.mean(sc_fin, axis=1), np.where(np.sum(res, axis=1) == 0)[0]))

print("Easy")
easy_curr_comp = list(set(np.arange(0, task_nb)) - set(hard_tasks_comp))
print("Overlap between easy task (greedy - all correct) and easy task (diff score): ", jaccard(easy_curr_comp, np.where(np.sum(res, axis=1) == 5)[0]))
print("Accuracy of greedy on DiffEval easy task: " + str(acc(np.mean(res, axis=1), easy_curr_comp).round(5) * 100) + "%")
print("Average diff score on Greedy easy task (all models correct)", acc_med(np.mean(sc_fin, axis=1), np.where(np.sum(res, axis=1) == 5)[0]))