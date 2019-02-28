import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
import json
import pickle


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
    d = [v for (k, v) in d.items()]
    att_weights = d[args.t]['rec_dec_06_att_weights']
    target = [target_int_to_vocab[w] for w in d[args.t]['classes']]
    source = [source_int_to_vocab[w] for w in d[args.t]['data']]

    np.set_printoptions(suppress=True)

    fig, ax = plt.subplots()
    ax.matshow(att_weights, cmap=plt.cm.Blues)
    ax.set_xticks(np.arange(len(source)))
    ax.set_yticks(np.arange(len(target)))

    ax.set_xticklabels(source)
    ax.set_yticklabels(target)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

    if args.show_labels:
        for i in range(len(target)):
            for j in range(len(source)):
                text = ax.text(j, i, '{0:.2f}'.format(att_weights[i, j]).rstrip("0"),
                               ha="center", va="center", color="black")

    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inspect distributions')
    parser.add_argument('attention', metavar='attention', type=str, help='path to attention file')
    parser.add_argument('t', metavar='t', type=int, help='time step to visualize')

    parser.add_argument('--target_vocab_file', metavar='target_vocab_file', type=str,
                        help='Path to vocab pickle file of targets',
                        default='/work/smt3/bahar/expriments/wmt/2018/de-en/data/julian-data/nn-vocabs/vocab.de-en.en.pkl',
                        required=False)
    parser.add_argument('--source_vocab_file', metavar='source_vocab_file', type=str,
                        help='Path to vocab pickle file of source',
                        default='/work/smt3/bahar/expriments/wmt/2018/de-en/data/julian-data/nn-vocabs/vocab.de-en.de.pkl',
                        required=False)

    parser.add_argument('--show_labels', dest='show_labels', action='store_true')

    args = parser.parse_args()
    main(args)