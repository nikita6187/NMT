import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
import json
import pickle
import os
import gzip


def main(args):

    with gzip.open(args.source_file) as w:
        source_lines = w.readlines()
    with open(args.target_file) as w:
        target_lines = w.readlines()

    list_1 = os.listdir(args.attention)
    list_1 = [w for w in list_1 if w[-3:-1] == "np"]

    list_1.sort()

    matrices = []

    # load ids
    with open(args.seq_ids, 'rb') as w:
        ids = [int(n) for n in w.readlines()]

    # Load all matrices
    for f in list_1:
        matrices.append(np.load(args.attention + f))

    # Merge those of same shape behind each other
    matrices_full = [[matrices[0]]]
    for m in range(1, len(matrices)):
        if matrices[m].shape == matrices_full[-1][-1].shape:
            matrices_full[-1].append(matrices[m])
        else:
            matrices_full.append([matrices[m]])

    matrices = [np.stack(m) for m in matrices_full]
    matrices = [np.squeeze(m, axis=3) for m in matrices]
    matrices = [np.transpose(m, axes=(2, 0, 1)) for m in matrices]  # Now each matrix is of shape [beam_size, I, J]
    del matrices_full

    # Select best/worst beam
    matrices_best_worst = []
    for m, idx in zip(matrices, range(len(matrices))):

        min_s = 9999999
        max_s = 0

        ret = [None, None]  # First entry best, second worst

        for i in range(m.shape[0]):
            s = np.sum(m[i][:, m.shape[2]-1])
            if s > max_s:
                max_s = s
                ret[1] = m[i]
            if s < min_s:
                min_s = s
                ret[0] = m[i]
        matrices_best_worst.append((ret, idx))

    # Visualize
    for (b, w), idx in matrices_best_worst:
        visualize(w, args, source_lines[ids[idx]], target_lines[idx], beam="Worst")
        visualize(b, args, source_lines[ids[idx]], target_lines[idx], beam="Best")


def visualize(att_weights, args, source, target, beam):
    np.set_printoptions(suppress=True)

    fig, ax = plt.subplots()
    ax.matshow(att_weights, cmap=plt.cm.Blues)

    # TODO get target labels from ref
    source = source.decode("utf-8")
    target = target

    source_split = source.split()
    target_split = target.split()

    ax.set_xticks(np.arange(len(source_split)))
    ax.set_yticks(np.arange(len(target_split)))

    ax.set_xticklabels(source_split)
    ax.set_yticklabels(target_split)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

    if args.show_labels:
        for i in range(att_weights.shape[0]):
            for j in range(att_weights.shape[1]):
                text = ax.text(j, i, '{0:.2f}'.format(att_weights[i, j]).rstrip("0"),
                               ha="center", va="center", color="black")

    plt.title(beam)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect distributions')
    parser.add_argument('attention', metavar='attention', type=str, help='path to attention folder')
    parser.add_argument('beam', metavar='beam', type=int, help='beam id to use')
    parser.add_argument('--show_labels', dest='show_labels', action='store_true')

    parser.add_argument('--seq_ids', metavar='seq_ids', type=str,
                        help='Seq ids',
                        default='/work/smt2/makarov/NMT/hmm-factorization/de-en/logs/transformer-hmm-no-linear-projection/seq_list',
                        required=False)

    parser.add_argument('--source_file', metavar='source_file', type=str,
                        help='Seq ids',
                        default='/work/smt3/bahar/expriments/wmt/2018/de-en/data/julian-data/dev.bpe/newstest2018_bpe-50k.de-en.de.gz',
                        required=False)
    parser.add_argument('--target_file', metavar='target_file', type=str,
                        help='Seq ids',
                        default='/work/smt2/makarov/NMT/hmm-factorization/de-en/logs/transformer-hmm-no-linear-projection/log/attention/search/refs_output',
                        required=False)

    args = parser.parse_args()
    main(args)
