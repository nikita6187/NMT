import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pickle
import re
import copy

np.set_printoptions(suppress=True)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def get_returnn_files(args):
    r_files = [f for f in os.listdir(args.attention) if "_ep" in f]  # Super hacky
    r_files.sort(key=natural_keys)
    return r_files


def dumpclean(obj, spec="average"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if hasattr(v, '__iter__'):
                if spec in str(k):
                    print(k)
                    dumpclean(v)
            else:
                if spec in str(k):
                    print('%s : %s' % (k, v))
    elif isinstance(obj, list):
        for v in obj:
            if hasattr(v, '__iter__'):
                dumpclean(v)
            else:
                print(v)
    else:
        print(obj)


def get_data(idx, layer, d, args):
    att_weights = d[idx][layer]
    att_weights = att_weights[:d[idx]['output_len'] -0 if args.with_eos else -1, :d[idx]['encoder_len']]  # [I, J (, H)]

    if len(att_weights.shape) == 3:
        # Multihead attention
        s = att_weights[:, :None if args.with_eos else -1, :]
        s = np.average(s, axis=-1)
    else:
        # Normal attention
        s = att_weights[:, :None if args.with_eos else -1]
    return s


def merge_bpes(str_list, axis, matrix, args):
    assert matrix.shape[axis] == len(str_list), "Merge bpes: matrix shape incorrect for axis and string"

    # Matrix of shape [I, J]
    if axis == 1:
        matrix = np.transpose(matrix, axes=(1, 0))  # In this case [J, I]

    # get which rows need to be merged
    rows_to_merge = []
    word_start = False
    sub_rows = []
    curr_str = ""
    new_str = []

    for i in range(len(str_list)):

        if word_start is False and "@@" not in str_list[i]:
            new_str.append(str_list[i])
            continue

        if word_start is False and str_list[i][-2:] == "@@":
            sub_rows = [i]
            curr_str = str_list[i][:-2]
            word_start = True
            continue

        if word_start is True and str_list[i][-2:] == "@@":
            curr_str += str_list[i][:-2]
            sub_rows.append(i)
            continue

        if word_start is True and str_list[i][-2:] != "@@":
            sub_rows.append(i)
            curr_str += str_list[i]
            new_str.append(curr_str)
            rows_to_merge.append(copy.copy(sub_rows))
            word_start = False
            continue

    # merge rows with appropriate strategy
    new_matrix = np.zeros((matrix.shape[0] - len(rows_to_merge), matrix.shape[1]))
    amount_merged = 0
    amount_non_merged = 0

    if not rows_to_merge:
        if axis == 1:
            matrix = np.transpose(matrix, axes=(1, 0))
        return matrix, new_str

    # Add in rows before first occurence
    for i in range(rows_to_merge[0][0]):
        new_matrix[i] = matrix[i]
        amount_non_merged += 1

    for sub_row, idx in zip(rows_to_merge, range(len(rows_to_merge))):  # Assume that they are ordered
        new_row = np.zeros_like(matrix[0])
        for i in sub_row:
            if args.merge_bpes == "max" and np.max(matrix[i]) > np.max(new_row):
                new_row = np.copy(matrix[i])
            if args.merge_bpes == "avg":
                new_row += matrix[i]
            if args.merge_bpes == "first" and i == sub_row[0]:
                new_row = np.copy(matrix[i])
            amount_merged += 1

        # avg new_row if needed
        if args.merge_bpes == "avg":
            new_row /= float(len(sub_row))

        new_matrix[sub_row[0] - amount_non_merged + 1] = new_row

        amount_non_merged += 1

        # Add in normal rows
        i = sub_row[-1] + 1
        while (idx == len(rows_to_merge) - 1 and i < matrix.shape[0]) or (i < matrix.shape[0] and i != rows_to_merge[idx+1][0]):
            new_matrix[i - amount_merged + 1] = matrix[i]
            amount_non_merged += 1
            i += 1

    # transpose back
    if axis == 1:
        new_matrix = np.transpose(new_matrix, axes=(1, 0))  # In this case [I, J]

    return new_matrix, new_str


def main(args):

    all_files = get_returnn_files(args=args)

    # Get a random file for meta data
    d = np.load(args.attention + "/" + all_files[0], allow_pickle=True).item()

    if args.layer_to_use:
        assert args.layer_to_use in d[0].keys(), "layer_to_use not in keys: " + str(d[0].keys())
        layer = args.layer_to_use
    else:
        # Get layers
        layers = []
        for k in list(d[0].keys()):
            if len(k) > len("rec_"):
                if k[:len("rec_")] == "rec_":
                    layers.append(k)
        layers.sort()
        if args.debug:
            print("Using layers: " + str(layers))

    del d

    # Get dictionaries
    with open(args.target_vocab_file, 'rb') as w:
        target_dictionary = pickle.load(w)
    target_int_to_vocab = {target_dictionary[w]: w for w in target_dictionary.keys()}

    with open(args.source_vocab_file, 'rb') as w:
        source_dictionary = pickle.load(w)
    source_int_to_vocab = {source_dictionary[w]: w for w in source_dictionary.keys()}

    # Data management
    data = []

    # Go through all files and get data
    for file, idx in zip(all_files, range(len(all_files))):
        d = np.load(args.attention + "/" + file, allow_pickle=True).item()

        if args.debug:
            print(str(idx) + "/" + str(len(all_files)))

        # For each sample in batch
        batch_size = len(d.keys())
        for idx in range(batch_size):

            if args.layer_to_use:
                s = get_data(idx, layer, d, args)
            else:
                # average over all layers
                s_s = []
                for layer in layers:
                    s_s.append(get_data(idx, layer, d, args))
                s = np.mean(s_s, axis=0)
            # s is [I, J]

            target_list = [target_int_to_vocab[w] for w in
                           d[idx]['output'][:None if args.with_eos else d[idx]['output_len'] - 1]]
            source_list = [source_int_to_vocab[w] for w in d[idx]['data'][:None if args.with_eos else -1]]

            if args.merge_bpes:
                # first merge columns
                s, source_list = merge_bpes(source_list, 1, s, args)
                s, target_list = merge_bpes(target_list, 0, s, args)

            # Data management
            peaked = np.argmax(s, axis=-1)
            alignment_list = []

            for i in range(peaked.shape[0]):
                alignment_list.append("S " + str(peaked[i]) + " " + str(i))

            data.append((d[idx]["tag"], source_list, target_list, alignment_list))

            if args.viz_step:
                if int(d[idx]["tag"][len("line-"):]) == args.viz_step - 1:
                    print("Visualizing step: " + str(d[idx]["tag"]))
                    print(data[-1])
                    fig, ax = plt.subplots()
                    # viz = np.put(np.zeros(shape=(len(target_list), len(source_list))), peaked, 1)
                    viz = np.zeros(shape=(peaked.shape[0], len(source_list)))
                    Y = np.arange(peaked.shape[0])[:]
                    viz[Y, peaked] = 1

                    ax.matshow(viz, cmap=plt.cm.Blues, aspect=0.5)

                    ax.set_xticks(np.arange(len(source_list)))
                    ax.set_yticks(np.arange(len(target_list)))

                    fig.tight_layout()

                    ax.set_xticklabels(source_list, size=20)
                    ax.set_yticklabels(target_list, size=20)

                    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
                    plt.margins(x=50)
                    plt.show()
                    # plt.savefig("./test.png", bbox_inches="tight")

        del d

    if args.debug:
        print(data)
    else:
        lines = []
        for dat in data:
            #lines.append(" ".join(dat[1]))
            st = ""
            for a in dat[3]:
                st += " " + a

            src = " ".join(dat[1])
            trgt = " ".join(dat[2])

            stc = src + " # " + trgt + " # alignment" + st if args.show_src_trgt else "# alignment" + st
            lines.append(stc)

        for line in lines:
            print(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect distributions')
    parser.add_argument('attention', metavar='attention', type=str, help='path to attention folder')

    parser.add_argument('--layer_to_use', metavar='--layer_to_use', type=str, default=None,
                        help='layer_to_use', required=False)

    parser.add_argument('--merge_bpes', metavar='--merge_bpes', type=str, default=None,
                        help='Merge bpes, either "max", "avg" or "first"', required=False)

    parser.add_argument('--with_eos', dest='with_eos', action='store_true', default=False,
                        required=False)

    parser.add_argument('--debug', dest='debug', action='store_true', default=False,
                        required=False)

    parser.add_argument('--show_src_trgt', dest='show_src_trgt', action='store_true', default=False,
                        required=False)

    parser.add_argument('--viz_step', metavar='viz_step', type=int, default=None,
                        help='which step to visualize in matplotlib', required=False)

    d_t = "/u/bahar/workspace/wmt/2018/de-en-6M--2019-01-16/de-en-hmm--2018-01-16/dataset/target.vocab.pkl"
    d_s = "/u/bahar/workspace/wmt/2018/de-en-6M--2019-01-16/de-en-hmm--2018-01-16/dataset/source.vocab.pkl"

    parser.add_argument('--target_vocab_file', metavar='target_vocab_file', type=str,
                        help='Path to vocab pickle file of targets',
                        default=d_t,
                        required=False)
    parser.add_argument('--source_vocab_file', metavar='source_vocab_file', type=str,
                        help='Path to vocab pickle file of source',
                        default=d_s,
                        required=False)

    args = parser.parse_args()
    main(args)
