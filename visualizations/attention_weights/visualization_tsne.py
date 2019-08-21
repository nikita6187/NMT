
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
import json
import pickle
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtrans
import math
import matplotlib
from matplotlib import font_manager
import sklearn.manifold


def merge_smartly(m):
    ms = []
    for i in range(m.shape[0]):
        for h in range(m.shape[1]):
            ms.append(m[i, h])
    m = np.stack(ms, axis=0)
    return m


def merge_smartly_list(l, amount_of_heads):
    ls = [[c] * amount_of_heads for c in l]
    lx = []
    for lm in ls:
        lx.extend(lm)
    return lx


def arrows(m, split_dims=False):
    assert not split_dims, NotImplementedError

    new_m = []
    for row in range(m.shape[0] - 1):
        new_m.append((m[row], m[row + 1] - m[row]))
    return new_m


def main(args):

    # load in vocab
    with open(args.target_vocab_file, 'rb') as w:
        target_dictionary = pickle.load(w)
    target_int_to_vocab = {target_dictionary[w]: w for w in target_dictionary.keys()}

    with open(args.source_vocab_file, 'rb') as w:
        source_dictionary = pickle.load(w)
    source_int_to_vocab = {source_dictionary[w]: w for w in source_dictionary.keys()}

    # get words
    d = np.load(args.attention, allow_pickle=True).item()

    target = [target_int_to_vocab[w].replace("▁", "") for w in d[args.t]['output'][:d[args.t]["output_len"]]]  # was 'classes' or 'output'
    source = [source_int_to_vocab[w].replace("▁", "") for w in d[args.t]['data']]

    str_to_dis = " ".join(source) + "\n -> " + " ".join(target)

    # load in states
    encoder_states = d[args.t]["encoder"]  # [J, f]
    decoder_states = d[args.t]["decoder"]  # [I, f]

    print("Encoder shape: " + str(encoder_states.shape))
    print("Decoder shape: " + str(decoder_states.shape))

    # optionally split
    if args.split_dims:
        encoder_states = np.stack(np.split(encoder_states, args.heads, axis=-1), axis=1)
        decoder_states = np.stack(np.split(decoder_states, args.heads, axis=-1), axis=1)

        encoder_states = merge_smartly(encoder_states)  # TODO: debug
        decoder_states = merge_smartly(decoder_states)  # TODO: debug

        source = merge_smartly_list(source, amount_of_heads=args.heads)
        target = merge_smartly_list(target, amount_of_heads=args.heads)

        print("Split dims encoder shape: " + str(encoder_states.shape))
        print("Split dims decoder shape: " + str(decoder_states.shape))

    # apply tsne
    def dot(x, y):
        return (np.dot(x, y) + 1)/2

    point_to_split = encoder_states.shape[0]
    all_points = np.concatenate([encoder_states, decoder_states], axis=0)
    tsne = sklearn.manifold.TSNE(random_state=0, verbose=1, metric=dot, learning_rate=50, n_iter=2000)  # TODO: maybe use different metric (dot product)
    all_points_2d = tsne.fit_transform(all_points)

    encoder_states_2d = all_points_2d[:point_to_split]
    decoder_states_2d = all_points_2d[point_to_split:]

    # TODO: visualize
    fig, ax = plt.subplots()

    print(target)
    plt.title(str_to_dis)

    ax.scatter(encoder_states_2d[:, 0], encoder_states_2d[:, 1], c="green", label="Encoder")

    for i, txt in enumerate(source):
        if i < encoder_states_2d.shape[0]:
            ax.annotate(txt, (encoder_states_2d[i, 0], encoder_states_2d[i, 1]))

    ax.scatter(decoder_states_2d[:, 0], decoder_states_2d[:, 1], c="blue", label="Decoder")

    for i, txt in enumerate(target):
        if i < decoder_states_2d.shape[0]:
            ax.annotate(txt, (decoder_states_2d[i, 0], decoder_states_2d[i, 1]))

    if args.arrows:
        jump = encoder_states.shape[0] // args.heads if args.split_dims else 1

        # TODO: make for split_dims
        encoder_arrows = arrows(encoder_states_2d)

        for row in range(len(encoder_arrows)):
            ax.arrow(x=encoder_arrows[row][0][0], y=encoder_arrows[row][0][1],
                     dx=encoder_arrows[row][1][0], dy=encoder_arrows[row][1][1],
                     color="green")

        decoder_arrows = arrows(decoder_states_2d)

        for row in range(len(decoder_arrows)):
            ax.arrow(x=decoder_arrows[row][0][0], y=decoder_arrows[row][0][1],
                     dx=decoder_arrows[row][1][0], dy=decoder_arrows[row][1][1],
                     color="blue")



    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inspect distributions')
    parser.add_argument('attention', metavar='attention', type=str, help='path to attention file')
    parser.add_argument('t', metavar='t', type=int, help='batch step to visualize')

    # '/work/smt3/bahar/expriments/wmt/2018/de-en/data/julian-data/nn-vocabs/vocab.de-en.en.pkl'
    #d_t = "/home/nikita/NMT/visualizations/attention_weights/vocab.de-en.en.pkl"
    d_de_en_en = "/u/bahar/workspace/wmt/2018/de-en-6M--2019-01-16/de-en-hmm--2018-01-16/dataset/target.vocab.pkl"
    d_zh_en_en = "/work/smt3/bahar/expriments/wmt/2019/zh-en/data/vocab.zh-en.en.pkl"
    # /work/smt3/bahar/expriments/wmt/2018/de-en/data/julian-data/nn-vocabs/vocab.de-en.de.pkl
    #d_s = "/home/nikita/NMT/visualizations/attention_weights/vocab.de-en.de.pkl"
    d_de_en_de = "/u/bahar/workspace/wmt/2018/de-en-6M--2019-01-16/de-en-hmm--2018-01-16/dataset/source.vocab.pkl"
    d_zh_en_zh = "/work/smt3/bahar/expriments/wmt/2019/zh-en/data/vocab.zh-en.zh.pkl"

    parser.add_argument('--target_vocab_file', metavar='target_vocab_file', type=str,
                        help='Path to vocab pickle file of targets',
                        default=None,
                        required=False)
    parser.add_argument('--source_vocab_file', metavar='source_vocab_file', type=str,
                        help='Path to vocab pickle file of source',
                        default=None,
                        required=False)

    parser.add_argument('--save_fig', metavar='save_fig', type=str,
                        help='Path to save figure',
                        default=None,
                        required=False)

    parser.add_argument('--lp', metavar='lp', type=str,
                        help='Language pair',
                        default="de-en",
                        required=False)

    parser.add_argument('--heads', metavar='heads', type=int,
                        help='Nr of attention heads',
                        default=8,
                        required=False)

    parser.add_argument('--split_dims',
                        help='Whether to split dimensions',
                        default=False,
                        action='store_true',
                        required=False)

    parser.add_argument('--arrows',
                        help='Whether to visualize arrows',
                        default=False,
                        action='store_true',
                        required=False)


    args = parser.parse_args()

    # Lang
    if args.lp == "de-en":
        args.source_vocab_file = d_de_en_de
        args.target_vocab_file = d_de_en_en
    elif args.lp == "en-de":
        args.target_vocab_file = d_de_en_de
        args.source_vocab_file = d_de_en_en
    elif args.lp == "zh-en":
        args.target_vocab_file = d_zh_en_en
        args.source_vocab_file = d_zh_en_zh
    elif args.lp == "en-zh":
        args.source_vocab_file = d_zh_en_en
        args.target_vocab_file = d_zh_en_zh

    main(args)


