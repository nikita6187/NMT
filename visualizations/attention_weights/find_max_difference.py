import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import re

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


def get_returnn_files(path):
    r_files = [path + f for f in os.listdir(path) if "_ep" in f]  # Super hacky
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


def get_data(path):
    all_files = get_returnn_files(path)

    # Get a random file for meta data
    d = np.load(all_files[0]).item()

    # Get layers
    layers = []
    for k in list(d[0].keys()):
        if len(k) > len("rec_"):
            if k[:len("rec_")] == "rec_":
                layers.append(k)
    if "posterior_attention" in d[0].keys():
        layers.append("posterior_attention")
    if "attention_score" in d[0].keys():
        layers.append("attention_score")
    layers.sort()
    print("Using layers: " + str(layers))

    del d

    # Data management
    data = {
        "eos_attendence": 0.0,
        "amount_of_attention_heads": 0,
        "non_monotonicity": 0.0,
        "amount_of_seqs": 0,
    }

    for layer in layers:
        data[layer + "_attendence"] = 0.0
        data[layer + "_amount_of_heads"] = 0
        data[layer + "_non_monotonicity"] = 0.0
        data[layer + "_eos_per_idx"] = []

    # Go through all files and get data
    for file, idx_out in zip(all_files, range(len(all_files))):
        d = np.load(file).item()

        print(str(idx_out) + "/" + str(len(all_files)))

        # For each sample in batch
        batch_size = len(d.keys())
        for idx in range(batch_size):

            data["amount_of_seqs"] += 1

            # Go over every layer
            for layer in layers:

                if layer not in d[idx].keys():
                    data[layer + "_eos_per_idx"].append((idx_out, idx, -1, file[-13:-1], 0))
                    continue

                att_weights = d[idx][layer]
                att_weights = att_weights[:d[idx]['output_len'], :d[idx]['encoder_len']]  # [I, J (, H)]

                if len(att_weights.shape) == 3:
                    # Multihead attention
                    s = att_weights[:, -2:, :]

                    non_mon = att_weights[:, :-1].copy()
                    for h in range(att_weights.shape[-1]):
                        np.fill_diagonal(non_mon[:, :, h], 0)
                else:
                    # Normal attention
                    s = att_weights[:, -2:]
                    non_mon = att_weights[:, :-1].copy()
                    np.fill_diagonal(non_mon, 0)

                # Data management
                data[layer + "_eos_per_idx"].append((idx_out, idx, np.sum(s), file[-13:-1], np.size(s)))

                data["eos_attendence"] += np.sum(s)
                data["amount_of_attention_heads"] += s.size
                data[layer + "_attendence"] += np.sum(s)
                data[layer + "_amount_of_heads"] += s.size

                data["non_monotonicity"] += np.sum(non_mon)
                data[layer + "_non_monotonicity"] += np.sum(non_mon)
        del d

    # Process and print data
    data["average_eos_attendence"] = data["eos_attendence"] / float(data["amount_of_attention_heads"])
    data["average_non_monotonicity"] = data["non_monotonicity"] / float(data["amount_of_attention_heads"])

    for layer in layers:
        data[layer + "_average_eos_attendence"] = data[layer + "_attendence"] / float(data[layer + "_amount_of_heads"])
        data[layer + "_average_non_monotonicity"] = data[layer + "_non_monotonicity"] / float(
            data[layer + "_amount_of_heads"])

    return data


def main(args):

    att_1 = get_data(args.attention_1)
    att_2 = get_data(args.attention_2)

    eos_diff = []
    eos_sum = 0.0

    layer_to_use_for_2 = args.layer_to_use2 if args.layer_to_use2 else args.layer_to_use

    for (idx_1_out, idx_1, sum_1, f1, size_1), (idx_2_out, idx_2, sum_2, f2, size_2) in zip(att_1[args.layer_to_use + "_eos_per_idx"],
                                                                    att_2[layer_to_use_for_2 + "_eos_per_idx"]):
        assert idx_1_out == idx_2_out and idx_1 == idx_2, "Order incorrect: " + str([idx_1_out, idx_2_out, idx_1, idx_2])

        if sum_1 == -1 or sum_2 == -1:
            continue

        if args.use_direct:
            eos_diff.append((f1, idx_1, sum_2/size_2, sum_2))
        else:
            eos_diff.append((f1, idx_1, (sum_1/size_1) - (sum_2/size_2), sum_2/size_2))
        eos_sum += sum_1 - sum_2

    eos_diff.sort(key=lambda x: x[2])

    print("------")
    if args.use_direct:
        print("Sorted attention_2 EOS")
    else:
        print("Sorted difference of attention_1 - attention_2 EOS: ")
    print(eos_diff)
    print("Sum of differences")
    print(eos_sum)
    print("Average differences")
    print(eos_sum / float(len(eos_diff)))
    if args.use_direct is False:
        print("Amount above 0: ")
        print(len([d for d in eos_diff if d[2] > 0]))
        print("Amount equal or below 0: ")
        print(len([d for d in eos_diff if d[2] <= 0]))

    print("Warning: EOS inlcudes also last token!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect distributions')
    parser.add_argument('attention_1', metavar='attention_1', type=str, help='path to attention 1 folder')
    parser.add_argument('attention_2', metavar='attention_2', type=str, help='path to attention 2 folder')

    parser.add_argument('layer_to_use', metavar='layer_to_use', type=str,
                        help='layer_to_use')

    parser.add_argument('--layer_to_use2', metavar='--layer_to_use2', type=str,
                        help='layer_to_use for second attention', default=None)

    parser.add_argument('--use_direct', dest='use_direct', action='store_true')

    args = parser.parse_args()
    main(args)
