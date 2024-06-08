import argparse
import json
import pandas as pd
from test_pipeline_classeval import AutoTest
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-so",
        "--source_model",
        type=str,
        default=None,
        help="name of the model to evaluate",
    )
    parser.add_argument(
        "-e",
        "--eval_data",
        type=str,
        default=os.path.join('..', '..', 'data', 'classeval', 'ClassEval_data.json'),
        help="ClassEval data",
    )
    parser.add_argument(
        "-st",
        "--sampled_tasks",
        type=str,
        default=os.path.join('..', '..', 'data', 'classeval', 'sampled_tasks.csv'),
        help="sampled_tasks",
    )
    args = parser.parse_args()

    AutoT = AutoTest(args.eval_data, args.source_model, 'ClassEval')
    
    with open(os.path.join('..', '..', 'data', 'classeval', 'raw', f'results_ClassEval_{args.source_model}.json')) as f:
        dat = json.load(f)

    sampled_tasks = pd.read_csv(args.sampled_tasks, sep=';')

    AutoT.test_pipeline(dat, sampled_tasks)