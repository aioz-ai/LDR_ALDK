"""
Our utility for evaluating LiTS dataset
"""

import os
import argparse
import numpy as np
import re

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', type=str, default='1casLDR_ALDK_Liver')
args = parser.parse_args()

eval_results = os.listdir('evaluate')

full_metrics = list()
for eval_result in eval_results:
    if eval_result.find('lits') != -1 and eval_result.find(args.model_name) != -1:
        metrics, filename = list(), 'evaluate/' + eval_result
        for line in reversed(list(open(filename))):
            line = line.rstrip()
            if line == "Summary":
                full_metrics.append(metrics)
                print('=========================')
                break

            print(line)
            metrics.append(float(re.findall("\d+\.\d+", line)[0]) * 1.)


full_metrics = np.asarray(full_metrics, dtype=np.float)
final_eval_output = 'evaluate/' + args.model_name + '_lits.txt'

print("Jacob Det - Land dis - Jacc - Dice", file=open(final_eval_output, "w"))
print(np.mean(full_metrics, axis=0), file=open(final_eval_output, "a"))
