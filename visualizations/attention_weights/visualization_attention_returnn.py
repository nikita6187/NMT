from os import terminal_size

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
import json
import pickle
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtrans


# default target vocab: /work/smt3/bahar/expriments/wmt/2018/de-en/data/julian-data/nn-vocabs/vocab.de-en.en.pkl
# default source vocab: /work/smt3/bahar/expriments/wmt/2018/de-en/data/julian-data/nn-vocabs/vocab.de-en.de.pkl

def main(args):

    # Get dictionaries
    with open(args.target_vocab_file, 'rb') as w:
        target_dictionary = pickle.load(w)
    target_int_to_vocab = {target_dictionary[w]: w for w in target_dictionary.keys()}

    with open(args.source_vocab_file, 'rb') as w:
        source_dictionary = pickle.load(w)
    source_int_to_vocab = {source_dictionary[w]: w for w in source_dictionary.keys()}

    # Visualize
    d = np.load(args.attention).item()
    print(len(d))
    d = [v for (k, v) in d.items()]
    print(len(d))
    print(list(d[args.t].keys()))
    l = "attention_score"
    #l = "posterior_attention"
    for k in list(d[args.t].keys()):
        if len(k) > len("rec_"):
            if k[:len("rec_")] == "rec_":
                l = k
                break

    print("Encoder len: " + str(d[args.t]['encoder_len']))
    print("Output len: " + str(d[args.t]['output_len']))

    d[args.t]['output'] = d[args.t]['output'][:d[args.t]['output_len']]
    #print(d[args.t].keys())
    target = [target_int_to_vocab[w] for w in d[args.t]['output']]  # was 'classes' or 'output'
    source = [source_int_to_vocab[w] for w in d[args.t]['data']]

    att_weights = d[args.t][l]  # TODO: assuming only 1 layer, [J, I, H]
    att_weights = att_weights[:d[args.t]['output_len'], :d[args.t]['encoder_len']]
    target_len = len(target)
    source_len = len(source)

    # Process att_weights
    # att_weights = np.squeeze(att_weights, axis=-1)

    if args.multihead:

        colours = ['black', 'darkblue', 'blue', 'royalblue', 'deepskyblue', 'turquoise', 'mediumspringgreen','green']

        fig = plt.figure(figsize=(len(source), len(target)))
        #plt.axis('off')
        gs1 = gridspec.GridSpec(len(target), len(source))
        gs1.update(wspace=0.05, hspace=0.1)

        print("att weights shape: " + str(att_weights.shape))
        print("source len: " + str(source_len))
        print("target len: " + str(target_len))

        i = 0

        for y in range(target_len):
            for x in range(source_len):

                i += 1
                print(str(i) + "/" + str(target_len * source_len))

                ax1 = plt.subplot(gs1[y, x])
                viz = att_weights[y, x]
                ax1.bar(height=viz, x=range(viz.shape[0]), width=0.5, color=colours)
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])
                ax1.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off',
                                left='off', labelleft='off')
                ax1.set_ylim((0, 1.0))

                # label y
                if ax1.is_first_col():
                    ax1.set_ylabel(target[y], fontsize=20, rotation=0, ha="right", rotation_mode="anchor")
                    ax1.yaxis.set_label_coords(0, 0.1)

                # label x
                if ax1.is_first_row():
                    ax1.set_title(source[x], fontsize=20, rotation=45, ha="left", rotation_mode="anchor")


    else:
        if len(att_weights.shape) == 3:
            att_weights = np.average(att_weights, axis=-1)  # [I, J, 1]

        fig, ax = plt.subplots()
        ax.matshow(att_weights, cmap=plt.cm.Blues, aspect=0.5)

        ax.set_xticks(np.arange(len(source)))
        ax.set_yticks(np.arange(len(target)))

        fig.tight_layout()

        ax.set_xticklabels(source, size=20)
        ax.set_yticklabels(target, size=20)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
        plt.margins(x=50)

        if args.show_labels:
            for i in range(len(target)):
                for j in range(len(source)):
                    text = ax.text(j, i, '{0:.2f}'.format(att_weights[i, j]).rstrip("0"),
                                   ha="center", va="center", color="black")

    #fig.subplots_adjust(top=0.8, left=0.1)
    if args.save_fig is None:
        plt.show()
    else:
        plt.savefig(args.save_fig, bbox_inches="tight")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inspect distributions')
    parser.add_argument('attention', metavar='attention', type=str, help='path to attention file')
    parser.add_argument('t', metavar='t', type=int, help='batch step to visualize')

    # '/work/smt3/bahar/expriments/wmt/2018/de-en/data/julian-data/nn-vocabs/vocab.de-en.en.pkl'
    #d_t = "/home/nikita/NMT/visualizations/attention_weights/vocab.de-en.en.pkl"
    d_t = "/u/bahar/workspace/wmt/2018/de-en-6M--2019-01-16/de-en-hmm--2018-01-16/dataset/target.vocab.pkl"

    # /work/smt3/bahar/expriments/wmt/2018/de-en/data/julian-data/nn-vocabs/vocab.de-en.de.pkl
    #d_s = "/home/nikita/NMT/visualizations/attention_weights/vocab.de-en.de.pkl"
    d_s = "/u/bahar/workspace/wmt/2018/de-en-6M--2019-01-16/de-en-hmm--2018-01-16/dataset/source.vocab.pkl"

    parser.add_argument('--target_vocab_file', metavar='target_vocab_file', type=str,
                        help='Path to vocab pickle file of targets',
                        default=d_t,
                        required=False)
    parser.add_argument('--source_vocab_file', metavar='source_vocab_file', type=str,
                        help='Path to vocab pickle file of source',
                        default=d_s,
                        required=False)
    parser.add_argument('--multihead', dest='multihead', action='store_true')
    parser.add_argument('--show_labels', dest='show_labels', action='store_true')
    parser.add_argument('--save_fig', metavar='save_fig', type=str,
                        help='Path to save figure',
                        default=None,
                        required=False)

    args = parser.parse_args()
    main(args)
