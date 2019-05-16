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

    assert args.layer_to_use in d[0].keys(), "layer_to_use not in keys: " + str(d[0].keys())

    del d

    # Data management
    data = []

    layer = args.layer_to_use

    # Go through all files and get data
    for file, idx in zip(all_files, range(len(all_files))):
        d = np.load(file).item()

        if args.debug:
            print(str(idx) + "/" + str(len(all_files)))

        # For each sample in batch
        batch_size = len(d.keys())
        for idx in range(batch_size):

            att_weights = d[idx][layer]
            att_weights = att_weights[:d[idx]['output_len'], :d[idx]['encoder_len']]  # [I, J (, H)]

            if len(att_weights.shape) == 3:
                # Multihead attention
                s = att_weights[:, :None if args.with_eos else -1, :]
                s = np.average(s, axis=-1)
            else:
                # Normal attention
                s = att_weights[:, :None if args.with_eos else -1]

            # Data management
            peaked = np.argmax(s, axis=-1)
            alignment_list = []

            for i in range(peaked.shape[0]):
                alignment_list.append("S " + str(peaked[i]) + " " + str(i))

            data.append((d[idx]["tag"], alignment_list))

        del d

    # TODO: export data
    data.sort()

    if args.debug:
        print(data)
    else:
        lines = []
        for dat in data:
            #lines.append(" ".join(dat[1]))
            st = ""
            for a in dat[1]:
                st += " " + a
            lines.append(st)

        for line in lines:
            print("# alignment" + line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect distributions')
    parser.add_argument('attention', metavar='attention', type=str, help='path to attention folder')

    parser.add_argument('layer_to_use', metavar='layer_to_use', type=str,
                        help='layer_to_use')

    parser.add_argument('--with_eos', dest='with_eos', action='store_true', default=False,
                        required=False)

    parser.add_argument('--debug', dest='debug', action='store_true', default=False,
                        required=False)

    args = parser.parse_args()
    main(args)
