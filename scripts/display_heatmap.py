import pandas as pd
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
import numpy as np

#plt.style.use(['science'])
plt.rcParams.update({
    "font.size":13,
    "text.usetex": True}) 

# Names of topics
# HumanEval+
names_humanevalplus = ["Counting/splitting words in strings (0)",
"Integer calculations with constraints (1)",
"Prime number calculations (2)",
"Sorting lists with constraints (3)",
"Special patterns in strings (4)",
"Strings to numbers (5)",
"Arithmetics with polynomials (6)",
"Encoding/Decoding strings (7)",
"Sequences (8)",
"Grid problems (9)",
"Numeric calculations (10)",
"Min \& Max in lists with constraints (11)",
"Nested parentheses/brackets (12)",
"Products with constraints (13)",
"Replacements in strings (14)",
"Duplicates in lists (15)",
"Coding-competition like problems (16)"]

# ClassEval
names_classeval = ["Games functionality (0)",
"Getters (1)",
"Condition checking (2)",
"Data structure calculations (3)",
"Strings processing (4)",
"Files processing (5)",
"Intergers maths (6)",
"Statistics (7)",
"SQL request (8)",
"Converting data formats (9)",
"Users' information processing (10)",
"Complex structure processing (11)",
"Data structure items removal (12)",
"Files reading (13)",
"Text encryption/decryption (14)",
"Data structure items additions (15)",
"Strings replacements (16)",
"Data structure information retrieval (17)",
"Temperature processing (18)",
"Strings assessment (19)",
"Data structure update (20)"]

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default=None)
args = parser.parse_args()

if args.dataset == 'humanevalplus':
    task_nb = 164
    names_col = names_humanevalplus
elif args.dataset == 'ClassEval':
    task_nb = 200
    names_col = names_classeval
else:
    raise Exception(f"Dataset {args.dataset} not recognised")

dat = pd.read_csv(os.path.join('..', 'data', args.dataset.lower(), f'topics_{args.dataset.lower()}.csv'), sep=';', index_col=0)

models = ['Global', 'Llama', 'Deepseek', 'MagiCoder', 'CodeGemma', 'GPT']
for ind, row in dat.iterrows():
    for mod in models:
        if str(row['LLM-Hard ({})'.format(mod)]) != 'nan':
            dat.loc[ind, 'Proportion LLM-Hard ({})'.format(mod)] = len(eval('[' + row['LLM-Hard ({})'.format(mod)] + ']'))/len(eval('[' + row['Task'] + ']'))
        else:
            dat.loc[ind, 'Proportion LLM-Hard ({})'.format(mod)] = 0


models = ['llama', 'deepseek', 'magicoder', 'gemma', 'gpt']
score = np.zeros((task_nb,))
set_ = None
for mod in models:
    tmp = np.load(os.path.join('..', 'data', args.dataset.lower(), 'scores', f'score_{args.dataset}_{mod}.npy'), allow_pickle=True)
    score += tmp[:,0].astype(float)
score /= 5

scores_list = []
scores_glob = []
topic_list = []
for ind, row in dat.iterrows():
    scores_list.extend(score[eval('[' + row['Task'] + ']')])
    topic_list.extend([ind]*len(eval('[' + row['Task'] + ']')))

scores_list = np.array(scores_list)
topic_list = np.array(topic_list)

hard = []
easy = []
for ind in range(len(scores_list)):
    if scores_list[ind] > 0.5:
        hard.append(ind)
    else:
        easy.append(ind)

print(dat)
fig, (ax,ax2) = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [7, 3]})
fig.subplots_adjust(wspace=0.1)
models = ['Llama', 'Deepseek', 'MagiCoder', 'CodeGemma', 'GPT', 'Global']
cols = ['Proportion LLM-Hard ({})'.format(mod) for mod in models]
heat = sns.heatmap(dat[cols], annot=True, cmap='Reds', ax=ax, cbar=False, yticklabels = names_col)
# Proportion
#ax.set_yticklabels(names_col, rotation=0)
ax.set_xticklabels(['CodeLlama', 'Deepseek', 'MagiCoder', 'CodeGemma', 'GPT', 'Global'], rotation=45, ha="right")
ax.set_ylabel('Topic')

ticks = ax.get_yticks() #np.array([t - 1 for t in ax.get_yticks()])
ax2.scatter(scores_list[hard], ticks[topic_list[hard]], color='red', marker='+',s=50)
ax2.scatter(scores_list[easy], ticks[topic_list[easy]], marker='+', s=50)
ax2.vlines(0.5, min(ticks)-0.05, max(ticks)+0.05, linestyle='dashed', color='black')
ax2.xaxis.tick_top()   
ax2.yaxis.tick_left()  
ax2.set_yticks(ax.get_yticks())
ax2.set_yticklabels(np.arange(0, len(names_col)))
ax2.yaxis.set_ticks_position('right')
ax2.set_xticks([0, 0.25, 0.5, 0.75])
ax2.set_xlabel('Difficulty Score')
ax2.set_ylim(ax2.get_ylim()[::-1])
ax2.grid(True, axis='y', linestyle='dashed', color='black')


plt.tight_layout()
plt.show()