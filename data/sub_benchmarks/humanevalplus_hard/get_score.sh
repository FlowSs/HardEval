#!/bin/bash

dataset="humanevalplus"
sub_benchmark='replacement'
for model in llama gemma deepseek magicoder gpt
do
   python get_score.py -m $model -d $dataset -s $sub_benchmark
done
