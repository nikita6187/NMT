import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
import json
import pickle


def main(args):

    # Get vocab info
    """
    with open(args.target_vocab_file, 'rb') as w:
        target_dictionary = pickle.load(w)
    target_int_to_vocab = {target_dictionary[w]: w for w in target_dictionary.keys()}
    """

    # TODO:get source vocab file

    # TODO: figure out how to get target

    # Open numpy file with attention wiehgt
    attention_weights = np.load(args.attention)  # Shape: [I, J, B, 1], assuming B=1
    attention_weights = np.squeeze(attention_weights, axis=-1)
    attention_weights = np.squeeze(attention_weights, axis=-1)  # Now shape [I, J]
    attention_weights = attention_weights.transpose()  # Now [J, I]

    # Open corresponding source
    source_ids = np.load(args.source)  # Shape [B, I, vocab_size], assuming B=1
    source_ids = np.squeeze(source_ids, axis=0)  # Remove B
    source_ids = np.argmax(source_ids, axis=1)  # Now shape = [I], each entry the id

    # Visaulize
    # TODO: check if in correct order

    # TODO: remove debug
    attention_weights[0] = np.zeros(attention_weights.shape[1])

    fig, ax = plt.subplots()
    im = ax.imshow(attention_weights.transpose(), interpolation="None", origin="upper")

    """
    ax.set_xticks(np.arange(len(farmers)))
    ax.set_yticks(np.arange(len(vegetables)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(farmers)
    ax.set_yticklabels(vegetables)
    

    for i in range(attention_weights.shape[0]):
        for j in range(attention_weights.shape[1]):
            text = ax.text(j, i, attention_weights[i, j],
                           ha="center", va="center", color="w")
    """
    fig.tight_layout()
    plt.grid(color='black', linestyle='-', linewidth=2)
    plt.ylabel("Target")
    plt.xlabel("Source")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.title("Attention weights")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect distributions')
    parser.add_argument('attention', metavar='attention', type=str, help='path to attention file')
    #parser.add_argument('target_vocab_file', metavar='target_vocab_file', type=str,help='Path to vocab pickle file of targets')

    parser.add_argument('source', metavar='source', type=str, help='path to source file')

    args = parser.parse_args()
    main(args)