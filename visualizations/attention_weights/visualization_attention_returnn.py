# -*- coding: utf-8 -*-
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
import math
import matplotlib
from matplotlib import font_manager

#matplotlib.rc('font', **{'sans-serif' : 'Arial',
#                         'family' : 'sans-serif'})


# default target vocab: /work/smt3/bahar/expriments/wmt/2018/de-en/data/julian-data/nn-vocabs/vocab.de-en.en.pkl
# default source vocab: /work/smt3/bahar/expriments/wmt/2018/de-en/data/julian-data/nn-vocabs/vocab.de-en.de.pkl

def main(args):

    # Get dictionaries
    try:
        with open(args.target_vocab_file, 'rb') as w:
            target_dictionary = pickle.load(w)
    except:
        with open(args.target_vocab_file, 'r') as w:
            raw = w.read()
        raw = raw.replace("\'", "\"")
        target_dictionary = json.loads(raw)
    target_int_to_vocab = {target_dictionary[w]: w for w in target_dictionary.keys()}

    with open(args.source_vocab_file, 'rb') as w:
        source_dictionary = pickle.load(w)
    source_int_to_vocab = {source_dictionary[w]: w for w in source_dictionary.keys()}

    # Visualize
    d = np.load(args.attention, allow_pickle=True).item()
    print(len(d))
    d = [v for (k, v) in d.items()]
    print(len(d))
    print(list(d[args.t].keys()))

    if args.all_layers:
        l = []
        for k in list(d[args.t].keys()):
            if args.encoder:
                if "enc" in k and k != "encoder_len":
                    l.append(k)
            else:
                if len(k) > len("rec_"):
                    if k[:len("rec_")] == "rec_":
                            l.append(k)
        l.sort()
        print("Using layers: " + str(l))
    else:
        if args.layer_to_viz is None:
            l = "attention_score"
            for k in list(d[args.t].keys()):
                if len(k) > len("rec_"):
                    if k[:len("rec_")] == "rec_":
                        if args.encoder:
                            if "enc" in k:
                                l = k
                        else:
                            l = k
                        break
        else:
            l = args.layer_to_viz
        print("Using layer: " + str(l))

    print("Encoder len: " + str(d[args.t]['encoder_len']))
    print("Output len: " + str(d[args.t]['output_len']))

    d[args.t]['output'] = d[args.t]['output'][:d[args.t]['output_len']]
    target = [target_int_to_vocab[w].replace("▁", "") for w in d[args.t]['output']]  # was 'classes' or 'output'

    if args.asr:
        source = [str(i) for i in range(d[args.t]['encoder_len'])]
    else:
        source = [source_int_to_vocab[w].replace("▁", "") for w in d[args.t]['data']]

    if args.encoder:
        target = source

    print(source)
    print(target)

    if args.all_layers:
        att_weights = []
        for layer in l:
            if args.encoder:
                att_weights.append(d[args.t][layer][:d[args.t]['encoder_len'], :d[args.t]['encoder_len']])
                att_weights[-1] = np.transpose(att_weights[-1], axes=(1, 2, 0))
            else:
                att_weights.append(d[args.t][layer][:d[args.t]['output_len'], :d[args.t]['encoder_len']])
    else:
        att_weights = d[args.t][l]  # TODO: assuming only 1 layer, [J, I, H]
        print(att_weights.shape)
        if args.encoder:
            att_weights = np.transpose(att_weights, axes=(1, 2, 0))
            att_weights = att_weights[:d[args.t]['encoder_len'], :d[args.t]['encoder_len']]
        else:
            att_weights = att_weights[:d[args.t]['output_len'], :d[args.t]['encoder_len']]
    target_len = len(target)
    source_len = len(source)
    print("target len: " + str(target_len))

    if args.multihead:

        if args.all_layers:

            all_y = []

            for y in range(target_len):
                all_x = []
                for x in range(source_len):
                    all_heads = []
                    for layer in range(len(l)):
                        all_heads.append(att_weights[layer][y, x, :])
                    c = np.stack(all_heads, axis=0)
                    all_x.append(c)
                d = np.concatenate(all_x, axis=1)
                all_y.append(d)

            viz = np.concatenate(all_y, axis=0)
            print("Viz: " + str(viz.shape))

            fig, ax = plt.subplots()
            ax.matshow(viz, cmap=plt.cm.Blues_r, aspect=0.5, extent=[0, viz.shape[1], 0, viz.shape[0]])

            heads = att_weights[0].shape[-1]
            amount_layers = len(l)

            source = [[s] for s in source]
            new_source = []
            for s in source:
                text = s.copy()
                s = [""] * (int(math.floor((heads - 1)/2)))
                s.extend(text)
                s.extend([""] * int((math.ceil((heads - 1)/2))))
                new_source.append(s)
            source = new_source
            source = [item for sublist in source for item in sublist]

            target = [[s] for s in target]
            new_target = []
            for s in target:
                text = s.copy()
                s = [""] * (int(math.floor((amount_layers - 1)/2)))
                s.extend(text)
                s.extend([""] * int((math.ceil((amount_layers - 1)/2))))
                new_target.append(s)
            target = new_target
            target = [item for sublist in target for item in sublist]

            ax.set_xticks(np.arange(len(source)))
            ax.set_yticks(np.arange(len(target)))

            fig.tight_layout()

            ax.set_xticklabels(source, size=20 if not args.asr else 1)
            t = target.copy()
            t.reverse()
            ax.set_yticklabels(t, size=20)

            for y, idx in zip(target, range(len(target))):
                if idx % amount_layers == 0:
                    ax.axhline(idx, linestyle='-', color='k')

            for x, idx in zip(source, range(len(source))):
                if idx % heads == 0:
                    ax.axvline(idx, linestyle='-', color='k')

            #plt.grid(b=True, which='major', color='black', linestyle='-')
            plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
            plt.margins(x=50)

        else:

            colours = ['black', 'darkblue', 'blue', 'royalblue', 'deepskyblue', 'turquoise', 'mediumspringgreen',
                       'green']

            fig = plt.figure(figsize=(len(source), len(target)))
            gs1 = gridspec.GridSpec(len(target), len(source))
            gs1.update(wspace=0.05, hspace=0.1)

            if args.all_layers is False:
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

        #assert args.all_layers is False, "all_layers parameters can only be used if multihead set to true!"

        print("Vizualizing layer: " + str(l))
        fontP = font_manager.FontProperties(fname="/u/makarov/fonts/PingFang.ttc")
        #fontP.set_family('DejaVu Sans Mono')
        fontP.set_size(20)
        
        if args.all_layers:
            # Average over all layers
            att_weights = np.average(att_weights, axis=0)

        if len(att_weights.shape) == 3:
            att_weights = np.average(att_weights, axis=-1)  # [I, J, 1]

        fig, ax = plt.subplots()
        ax.matshow(att_weights, cmap=plt.cm.Blues, aspect=0.5)

        ax.set_xticks(np.arange(len(source)))
        ax.set_yticks(np.arange(len(target)))

        ax.set_yticklabels(target, fontproperties=fontP)  #size=20)
        if args.asr:
            fontA = font_manager.FontProperties(fname="/u/makarov/fonts/PingFang.ttc")
            fontA.set_size(2)
            ax.set_xticklabels(source, fontproperties=fontA)  # size=20)
        else:
            ax.set_xticklabels(source, fontproperties=fontP)  # size=20)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
        #plt.margins(x=100)

        fig.set_size_inches(0.8 * len(source) if not args.asr else 0.05 * len(source), 1.0 * len(target))
        fig.tight_layout()

        if args.show_labels:
            for i in range(len(target)):
                for j in range(len(source)):
                    text = ax.text(j, i, '{0:.2f}'.format(att_weights[i, j]).rstrip("0"),
                                   ha="center", va="center", color="black")

    #fig.subplots_adjust(top=0.8, left=0.1)
    if args.save_fig is None:
        plt.show()
    else:
        plt.savefig(args.save_fig, dpi=300)


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
    parser.add_argument('--multihead', dest='multihead', action='store_true')
    parser.add_argument('--all_layers', dest='all_layers', action='store_true')
    parser.add_argument('--show_labels', dest='show_labels', action='store_true')
    parser.add_argument('--save_fig', metavar='save_fig', type=str,
                        help='Path to save figure',
                        default=None,
                        required=False)

    parser.add_argument('--layer_to_viz', metavar='layer_to_viz', type=str,
                        help='layer_to_viz',
                        default=None,
                        required=False)

    parser.add_argument('--lp', metavar='lp', type=str,
                        help='Language pair',
                        default="de-en",
                        required=False)

    parser.add_argument('--encoder',
                        help='Whether to visualize only the encoder',
                        default=False,
                        action='store_true',
                        required=False)

    parser.add_argument('--asr',
                        help='When using asr mode the source seq is not shown',
                        default=False,
                        action='store_true',
                        required=False)

    args = parser.parse_args()

    # Lang
    if args.asr is False:
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
    else:
        args.source_vocab_file = d_de_en_de  # just some value

    main(args)
