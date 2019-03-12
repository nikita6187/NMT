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

    visualize_single_batch(data[args.global_time_step][0], data[args.global_time_step][1], args, int_to_vocab)


def visualize_single_batch(softmax, target, args, int_to_vocab):
    # Convert text to np arrays
    softmax = softmax.replace(" ", ", ").replace("]]", "]],").replace("]],]", "]]]")
    softmax = np.array(json.loads(softmax), dtype=float)  # Of shape [I, 1, vocab_size]

    target = target.replace("]", "],").replace("],],", "]]")
    target = np.array(json.loads(target), dtype=float)  # Shape [I, 1]

    full_sentence = " ".join([int_to_vocab[int(t)] for t in target.squeeze(1).tolist()])

    total_amount_of_subplots = 1 + args.max_t

    sqrt_amount_of_subplots = int(np.sqrt(total_amount_of_subplots))

    import math
    print(sqrt_amount_of_subplots)
    print(int(math.ceil(np.median(range(1, sqrt_amount_of_subplots)))))

    plt.subplot(sqrt_amount_of_subplots, sqrt_amount_of_subplots, int(math.ceil(np.median(range(1, sqrt_amount_of_subplots)))))
    plt.title(full_sentence)

    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(sqrt_amount_of_subplots, sqrt_amount_of_subplots)
    gs.update(wspace=0.2, hspace=1)

    for idx in range(min(args.max_t, len(target))):
        visualize_single_step(softmax[args.offset + idx][0], target[args.offset + idx][0], idx + 1,
                              sqrt_amount_of_subplots, int_to_vocab, gs)

    plt.show()


def visualize_single_step(softmax, target, idx, total_amount, int_to_vocab, gs):
    # TODO: add 2nd best and vocab
    #ax = plt.subplot(total_amount, total_amount, idx)
    ax = plt.subplot(gs[idx - 1])

    #ax.annotate("Text", (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')

    ax.plot(range(len(softmax)), softmax)
    ax.scatter([target], [softmax[int(target)]], linewidth=5, color="orange")  # TODO: smaller

    best = [(a, b) for a, b in zip(range(len(softmax)), softmax.tolist())]
    #alt_best.remove((int(target), softmax[int(target)]))
    best.sort(reverse=True, key=lambda tup: tup[1])
    tag = "Model prediction: " if best[0] != (int(target), softmax[int(target)]) else "2nd best: "
    d = best[0] if best[0] != (int(target), softmax[int(target)]) else best[1]

    tag_2 = "Model prediction: " if best[0] == (int(target), softmax[int(target)]) else "2nd best: "

    # TODO: visualize text so it doesn't collid
    # TODO: replace "alternative" with better

    #ax.text(target, softmax[int(target)], tag_2 + int_to_vocab[int(target)])
    plt.ylim(0, 1)
    ax.grid(True)

    #ax.text(d[0], d[1], tag + int_to_vocab[d[0]])
    s = 9
    ax.text(0, -0.55, "True Label: " + int_to_vocab[int(target)],
            transform=ax.transAxes, size=s)
    ax.text(0, -0.75,  "Model Prediction: " + int_to_vocab[best[0][0]], transform=ax.transAxes, size=s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect distributions')
    parser.add_argument('p', metavar='p', type=str,
                        help='path to logs')
    parser.add_argument('max_t', metavar='max_t', type=int, help='max timesteps to visualize', default=1)
    parser.add_argument('offset', metavar='offset', type=int, help='offset of timesteps to visualize. Timesteps' +
                                                                   ' visualized is [offset: offset + max_t]', default=0)

    parser.add_argument('global_time_step', metavar='global_time_step', type=int, help='global time step to visualize',
                        default=0)

    d_t = "/home/nikita/NMT/visualizations/attention_weights/vocab.de-en.en.pkl"
    parser.add_argument('--vocab_file', metavar='vocab_file', type=str,
                        help='Path to vocab pickle file of targets',
                        default=d_t,
                        required=False)

    args = parser.parse_args()
    main(args)