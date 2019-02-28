import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
import json
import pickle
import os

# 1 - 2


def main(args):

    difference = {}

    list_1 = os.listdir(args.attention_1)
    list_2 = os.listdir(args.attention_2)

    # first filter out non ".npy" files
    print(list_1[0][-3:-1])
    list_1 = [w for w in list_1 if w[-3:-1] == "np"]
    list_2 = [w for w in list_2 if w[-3:-1] == "np"]

    # find pairs
    full_list = []
    for f1 in list_1:
        for f2 in list_2:
            if f1[-13:-1] == f2[-13:-1]:
                full_list.append((f1, f2))
                continue

    for f1, f2 in full_list:
        difference[f1] = {}

        d1 = np.load(args.attention_1 + f1).item()
        d1 = [v for (k, v) in d1.items()]

        d2 = np.load(args.attention_2 + f2).item()
        d2 = [v for (k, v) in d2.items()]

        for idx in range(len(d1)):
            m1 = d1[idx]['rec_dec_06_att_weights']
            m2 = d2[idx]['rec_dec_06_att_weights']
            s1 = np.sum(m1.transpose()[-1])
            s2 = np.sum(m2.transpose()[-1])
            diff = s1-s2
            avg_diff = float(diff)/float(m1.shape[1])
            difference[f1][idx] = (diff, avg_diff)

    # Sort
    full_difference = []
    for f1, _ in full_list:
        for v, k in difference[f1].items():
            full_difference.append((f1[-13:-1] + "-----" + str(v), k))

    full_difference.sort(key=lambda x: x[1][1])
    print(full_difference)
    print([w[0] for w in full_difference[-20:]])

    print('Full len: ' + str(len(full_difference)))

    # Get total average
    total_sum = sum([v[1][1] for v in full_difference])
    total_avg = total_sum / len(full_difference)

    print("Total average: " + str(total_avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect distributions')
    parser.add_argument('attention_1', metavar='attention_1', type=str, help='path to attention folder 1 (transformer)')

    parser.add_argument('attention_2', metavar='attention_2', type=str, help='path to attention folder 2 (hmm)')
    args = parser.parse_args()
    main(args)