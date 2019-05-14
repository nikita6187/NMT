import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def get_returnn_files(args):
    return [f for f in os.listdir(args.attention) if "_ep" in f]  # Super hacky


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

    # Data management
    data = {
        "eos_attendence": 0.0,
        "amount_of_attention_heads": 0,
            }

    # Go through all files and get data
    for file, idx in zip(all_files, range(len(all_files))):
        d = np.load(file).item()

        print(str(idx) + "/" + str(len(all_files)))

        # For each sample in batch
        batch_size = len(d.keys())  # TODO: check
        for idx in range(batch_size):

            # Go over every layer
            for layer in layers:

                att_weights = d[idx][layer]
                att_weights = att_weights[:d[idx]['output_len'], :d[idx]['encoder_len']]  # [I, J (, H)]

                if len(att_weights.shape) == 3:
                    # Multihead attention
                    s = att_weights[:, -1, :]
                else:
                    # Normal attention
                    s = att_weights[:, -1]
                data["eos_attendence"] += np.sum(s)
                data["amount_of_attention_heads"] += s.size

        del d

    # Process and print data
    data["average_eos_attendence"] = data["eos_attendence"] / float(data["amount_of_attention_heads"])

    print(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect distributions')
    parser.add_argument('attention', metavar='attention', type=str, help='path to attention folder')

    args = parser.parse_args()
    main(args)
