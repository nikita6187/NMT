import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import scipy.spatial.distance

np.set_printoptions(suppress=True)


def get_returnn_files(args):
    return [f for f in os.listdir(args.attention) if "_ep" in f]  # Super hacky


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


def main(args):

    all_files = get_returnn_files(args=args)

    # Get a random file for meta data
    d = np.load(all_files[0], allow_pickle=True).item()

    # Get layers
    layers = []
    print(d[0].keys())
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
        "std_deviation": 0,
        "entropy": 0,
        "distance": 0.0,
    }

    for layer in layers:
        data[layer + "_attendence"] = 0.0
        data[layer + "_amount_of_heads"] = 0
        data[layer + "_non_monotonicity"] = 0.0
        data[layer + "_std"] = 0.0
        data[layer + "_entropy"] = 0.0
        data[layer + "_distance"] = 0.0

    # Go through all files and get data
    for file, idx in zip(all_files, range(len(all_files))):
        d = np.load(file, allow_pickle=True).item()

        print(str(idx) + "/" + str(len(all_files)))

        # For each sample in batch
        batch_size = len(d.keys())
        for idx in range(batch_size):

            data["amount_of_seqs"] += 1

            # Go over every layer
            for layer in layers:
                if layer in d[idx]:
                    att_weights = d[idx][layer]
                    att_weights = att_weights[:d[idx]['output_len'], :d[idx]['encoder_len']]  # [I, J (, H)]

                    eos_offset = -1

                    if len(att_weights.shape) == 3:
                        # Multihead attention
                        s = att_weights[:, eos_offset if not args.eos_minus_2 else -2:, :]

                        # Get deviation across J
                        std = np.std(att_weights, axis=0)
                        std = np.mean(std)

                        non_mon = att_weights[:, :eos_offset if not args.eos_minus_2 else -2].copy()
                        for h in range(att_weights.shape[-1]):
                            np.fill_diagonal(non_mon[:, :, h], 0)
                    else:
                        # Normal attention
                        std = np.std(att_weights, axis=0)
                        std = np.mean(std)

                        s = att_weights[:, eos_offset if not args.eos_minus_2 else -2:]
                        non_mon = att_weights[:, :eos_offset if not args.eos_minus_2 else -2].copy()
                        np.fill_diagonal(non_mon, 0)

                    # Data management
                    data["eos_attendence"] += np.sum(s)
                    data["amount_of_attention_heads"] += s.size
                    data["entropy"] += np.sum(-np.log(s) * s)

                    dist = []
                    if len(att_weights.shape) == 3:
                        # do for all heads
                        for h in range(att_weights.shape[-1]):
                            dis = scipy.spatial.distance.cdist(att_weights[:, :, h], att_weights[:, :, h])
                            dis = dis[~np.eye(dis.shape[0], dtype=bool)].reshape(dis.shape[0], -1)
                            dis = np.average(dis)
                            dist.append(dis)
                    else:
                        dis = scipy.spatial.distance.cdist(att_weights, att_weights)
                        dis = dis[~np.eye(dis.shape[0], dtype=bool)].reshape(dis.shape[0], -1)
                        dis = np.average(dis)
                        dist = [dis]

                    full_dis = sum(dist)/len(dist)
                    data["distance"] += full_dis

                    data[layer + "_attendence"] += np.sum(s)
                    data[layer + "_amount_of_heads"] += s.size

                    data["non_monotonicity"] += np.sum(non_mon)
                    data[layer + "_non_monotonicity"] += np.sum(non_mon)

                    data[layer + "_std"] += std

                    data[layer + "_entropy"] += np.sum(-np.log(s) * s)
                    data[layer + "_distance"] += full_dis

        del d

    # Process and print data
    data["average_eos_attendence"] = data["eos_attendence"] / float(data["amount_of_attention_heads"])
    data["average_non_monotonicity"] = data["non_monotonicity"] / float(data["amount_of_attention_heads"])
    data["entropy"] = data["entropy"] / float(data["amount_of_attention_heads"])
    data["distance"] = data["distance"] / float(data["amount_of_attention_heads"])
    full_std = 0.0

    for layer in layers:
        data[layer + "_average_eos_attendence"] = data[layer + "_attendence"] / float(data[layer + "_amount_of_heads"])
        data[layer + "_entropy"] = data[layer + "_entropy"] / float(data[layer + "_amount_of_heads"])
        data[layer + "_distance"] = data[layer + "_distance"] / float(data[layer + "_amount_of_heads"])

        # data[layer + "_average_non_monotonicity"] = data[layer + "_non_monotonicity"] / float(data[layer + "_amount_of_heads"])

        if layer == "rec_dec_06_att_weights" or layer == "posterior_attention" or layer == "attention_score":
            data[layer + "_average_std"] = data[layer + "_std"] / float(data["amount_of_seqs"])
        full_std += data[layer + "_std"] / float(data["amount_of_seqs"])

    data["average_std"] = full_std / float(len(layers))

    # TODO: average attendence to all? something like that

    dumpclean(data)
    dumpclean(data, spec="entropy")

    np.set_printoptions(suppress=True)
    dumpclean(data, spec="distance")
    if args.eos_minus_2:
        print("Warning: EOS inlcudes also last token!!")
    if args.eos:
        print("WARNING: EOS also examined!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect distributions')
    parser.add_argument('attention', metavar='attention', type=str, help='path to attention folder')

    parser.add_argument('--eos_minus_2', dest='eos_minus_2', action="store_true",
                        help='When examining eos, whether to also use the symbol before EOS')

    parser.add_argument('--eos', dest='eos', action="store_true",
                        help='To also look at EOS')

    args = parser.parse_args()
    main(args)
