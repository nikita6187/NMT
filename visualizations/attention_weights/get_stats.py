import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

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
    d = np.load(all_files[0]).item()

    # Get layers
    layers = []
    for k in list(d[0].keys()):
        if len(k) > len("rec_"):
            if k[:len("rec_")] == "rec_":
                layers.append(k)
    layers.sort()
    print("Using layers: " + str(layers))

    del d

    # TODO: non-monotonicity check excluding EOS

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

    # Go through all files and get data
    for file, idx in zip(all_files, range(len(all_files))):
        d = np.load(file).item()

        print(str(idx) + "/" + str(len(all_files)))

        # For each sample in batch
        batch_size = len(d.keys())
        for idx in range(batch_size):

            data["amount_of_seqs"] += 1

            # Go over every layer
            for layer in layers:

                att_weights = d[idx][layer]
                att_weights = att_weights[:d[idx]['output_len'], :d[idx]['encoder_len']]  # [I, J (, H)]

                if len(att_weights.shape) == 3:
                    # Multihead attention
                    s = att_weights[:, -1, :]

                    non_mon = att_weights[:, :-1].copy()
                    for h in range(att_weights.shape[-1]):
                        np.fill_diagonal(non_mon[:, :, h], 0)
                else:
                    # Normal attention
                    s = att_weights[:, -1]
                    non_mon = att_weights[:, :-1].copy()
                    np.fill_diagonal(non_mon, 0)

                # Data management
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
        data[layer + "_average_non_monotonicity"] = data[layer + "_non_monotonicity"] / float(data[layer + "_amount_of_heads"])

    dumpclean(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect distributions')
    parser.add_argument('attention', metavar='attention', type=str, help='path to attention folder')

    args = parser.parse_args()
    main(args)
