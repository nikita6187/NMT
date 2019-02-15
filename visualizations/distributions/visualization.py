import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
import json


def main(args):

    data = []

    with open(args.p, 'r') as w:
        lines = w.readlines()

    for line, idx in zip(lines, range(len(lines))):
        # Search for softmax
        if re.search("Softmax:", line) and line.split("Softmax:")[1][0] is "[":
            s = line.split("Softmax:")[1]
            t = lines[idx + 1].split("Target:")[1]
            data.append((s, t))

    visualize_single_batch(data[-1][0], data[-1][1], args)


def visualize_single_batch(softmax, target, args):
    # Convert text to np arrays
    softmax = softmax.replace(" ", ", ").replace("]]", "]],").replace("]],]", "]]]")
    softmax = np.array(json.loads(softmax), dtype=float)

    target = target.replace("]", "],").replace("],],", "]]")
    target = np.array(json.loads(target), dtype=float)

    for idx in range(args.max_t):
        visualize_single_step(softmax[idx][0], target[idx][0], idx + 1, 1 + args.max_t)
    plt.show()


def visualize_single_step(softmax, target, idx, total_amount):
    plt.subplot(total_amount, 1, idx)
    plt.scatter(range(len(softmax)), softmax)
    plt.scatter([target], [softmax[int(target)]], linewidth=20, color="g")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect distributions')
    parser.add_argument('p', metavar='p', type=str,
                        help='path to logs')
    parser.add_argument('max_t', metavar='max_t', type=int, help='max timesteps to visualize', default=0)
    args = parser.parse_args()
    main(args)