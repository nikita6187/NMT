import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pickle

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

    # Get dictionaries
    with open(args.target_vocab_file, 'rb') as w:
        target_dictionary = pickle.load(w)
    target_int_to_vocab = {target_dictionary[w]: w for w in target_dictionary.keys()}

    with open(args.source_vocab_file, 'rb') as w:
        source_dictionary = pickle.load(w)
    source_int_to_vocab = {source_dictionary[w]: w for w in source_dictionary.keys()}

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

            target_list = [target_int_to_vocab[w] for w in d[idx]['output'][:None if args.with_eos else d[idx]['output_len'] - 1]]
            source_list = [source_int_to_vocab[w] for w in d[idx]['data'][:None if args.with_eos else -1]]

            for i in range(peaked.shape[0]):
                alignment_list.append("S " + str(peaked[i]) + " " + str(i))

            data.append((d[idx]["tag"], source_list, target_list, alignment_list))

            if args.viz_step:
                if int(d[idx]["tag"][len("line-"):]) == args.viz_step:
                    print("Visualizing step: " + str(d[idx]["tag"]))
                    print(data[-1])
                    fig, ax = plt.subplots()
                    #viz = np.put(np.zeros(shape=(len(target_list), len(source_list))), peaked, 1)
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
                    #plt.show()
                    plt.savefig("./test.png", bbox_inches="tight")

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

    parser.add_argument('layer_to_use', metavar='layer_to_use', type=str,
                        help='layer_to_use')

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
