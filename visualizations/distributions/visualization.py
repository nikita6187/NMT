import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
import json
import pickle




def main(args):

    # Get vocab info
    with open(args.vocab_file, 'rb') as w:
        dictionary = pickle.load(w)
    int_to_vocab = {dictionary[w]: w for w in dictionary.keys()}

    # Get data
    data = []

    with open(args.p, 'r') as w:
        lines = w.readlines()

    for line, idx in zip(lines, range(len(lines))):
        # Search for softmax
        if re.search("Softmax:", line) and line.split("Softmax:")[1][0] is "[":
            s = line.split("Softmax:")[1]
            t = lines[idx + 1].split("Target:")[1]
            data.append((s, t))

    visualize_single_batch(data[-1][0], data[-1][1], args, int_to_vocab)


def visualize_single_batch(softmax, target, args, int_to_vocab):
    # Convert text to np arrays
    softmax = softmax.replace(" ", ", ").replace("]]", "]],").replace("]],]", "]]]")
    softmax = np.array(json.loads(softmax), dtype=float)  # Of shape [I, 1, vocab_size]

    target = target.replace("]", "],").replace("],],", "]]")
    target = np.array(json.loads(target), dtype=float)  # Shape [I, 1]

    full_sentence = " ".join([int_to_vocab[int(t)] for t in target.squeeze(1).tolist()])

    total_amount_of_subplots = 1 + args.max_t

    plt.subplot(total_amount_of_subplots, 1, 1)
    plt.title(full_sentence)

    for idx in range(args.max_t):
        visualize_single_step(softmax[args.offset + idx][0], target[args.offset + idx][0], idx + 1,
                              total_amount_of_subplots, int_to_vocab)
    plt.show()


def visualize_single_step(softmax, target, idx, total_amount, int_to_vocab):
    # TODO: add 2nd best and vocab
    plt.subplot(total_amount, 1, idx)
    plt.plot(range(len(softmax)), softmax)
    plt.scatter([target], [softmax[int(target)]], linewidth=20, color="g")
    plt.text(target, softmax[int(target)], int_to_vocab[int(target)])

    alt_best = [(a, b) for a, b in zip(range(len(softmax)), softmax.tolist())]
    alt_best.remove((int(target), softmax[int(target)]))
    alt_best.sort(reverse=True, key=lambda tup: tup[1])
    alt_best = alt_best[0]
    print(alt_best)
    plt.text(alt_best[0], alt_best[1], "Alternative: " + int_to_vocab[alt_best[0]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect distributions')
    parser.add_argument('p', metavar='p', type=str,
                        help='path to logs')
    parser.add_argument('max_t', metavar='max_t', type=int, help='max timesteps to visualize', default=1)
    parser.add_argument('offset', metavar='offset', type=int, help='offset of timesteps to visualize. Timesteps' +
                                                                   ' visualized is [offset: offset + max_t]', default=0)
    parser.add_argument('vocab_file', metavar='vocab_file', type=str, help='Path to vocab pickle file', default=1)

    args = parser.parse_args()
    main(args)