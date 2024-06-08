import json
import pandas as pd
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', default=None)
args = parser.parse_args()

dat = pd.read_csv(f'{args.file}.csv', sep=";", index_col=0)
dat.index = dat.index.map(str)
for index, d in dat.iterrows():
	#if str(d['level 3']) == 'nan':
	#	continue
	d['level 3'] = {i: ele for i, ele in enumerate(eval(d['level 3']))}
	d['level 2'] = {i: ele for i, ele in enumerate(eval(d['level 2']))}
	d['level 1'] = {i: ele for i, ele in enumerate(eval(d['level 1']))}
dat.to_json(f'{args.file}.json', orient='index')
