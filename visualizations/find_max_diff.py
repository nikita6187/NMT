import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
import json
import pickle
import os


def main(args):

    difference = {}

    list_1 = os.listdir(args.attention_1)

    for f in list_1:
        print(f)
        # TODO:check that ends with .npy
        
        d = np.load(args.attention_1).item()
        d = [v for (k, v) in d.items()]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect distributions')
    parser.add_argument('attention_1', metavar='attention_1', type=str, help='path to attention folder 1')

    parser.add_argument('attention_2', metavar='attention_2', type=str, help='path to attention folder 2')
    args = parser.parse_args()
    main(args)