import json
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', default=None)
args = parser.parse_args()

if not('humanevalplus' in args.file) and not('ClassEval' in args.file):
  raise Exception("Name of file should contain humanevalplus or ClassEval")

# helper function as read_json modify my index for ClassEval
def reformat_ClassEval_index(i):
  i = str(i)
  i = i[:2] + '_' + i[2:]
  return i 

with open(f'{args.file}.json') as f:
  dat = pd.read_json(f, orient='index')

dat.index = dat.index.map(reformat_ClassEval_index if 'ClassEval' in args.file else str)
dat.index.names = ['class_id'] if 'ClassEval' in args.file else ['task_id']

for index, d in dat.iterrows():
  if isinstance(d['level 3'], dict):
    d['level 3'] = [val for val in d['level 3'].values()]
    d['level 2'] = [val for val in d['level 2'].values()]
    d['level 1'] = [val for val in d['level 1'].values()]
dat.to_csv(f'{args.file}.csv', sep=";")
