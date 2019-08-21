import os
import argparse
from os import listdir
from os.path import isfile, join
import numpy as np


def get_max_batch(args):
    all_files = [f for f in listdir(args.folder) if isfile(join(args.folder, f))]
    max_batch = -1
    for file in all_files:
        if len(file) >= len("_attention.npy") and file[-len("_attention.npy"):] == "_attention.npy":
            batch = int(file.split("_")[0])
            if batch > max_batch:
                max_batch = batch
    assert max_batch != -1, "Did not find max batch size!"
    print("max batch: " + str(max_batch))
    return max_batch


def get_max_time(batch, args):
    all_files = [f for f in listdir(args.folder) if isfile(join(args.folder, f)) and f.split("_")[0] == str(batch)]
    assert all_files is not [], "Getting batch files is going wrong"

    max_time = -1
    for file in all_files:
        if len(file) >= len("_attention.npy") and file[-len("_attention.npy"):] == "_attention.npy":
            time = int(file.split("_")[-2])
            if time > max_time:
                max_time = time
    assert max_time != -1, "Did not get max_time"
    print("Max time for batch nr" + str(batch) + " is: " + str(max_time))
    return max_time


def get_returnn_epoch(args):
    all_files = [f for f in listdir(args.folder) if isfile(join(args.folder, f))
                 and f[:len(args.r_prefix)] == args.r_prefix]
    assert all_files is not None, "Did not find any files with this prefix!"
    epoch = int(all_files[0].split("_")[1][len("ep"):])
    print("Returnn epoch: " + str(epoch))
    return epoch


def main(args):
    # TODO: go over all hmm_fac_fo files in folder, and convert them to the appropriate format as in the normal
    # get-attention-weights.py file

    args.folder = os.path.dirname(args.folder)

    # TODO: find max batch
    max_batch = get_max_batch(args)
    r_epoch = get_returnn_epoch(args)
    curr_seq = 0


    # Go over all batches
    for batch in range(1, max_batch + 1):
        batch_size = 0

        print(batch)

        for enc_idx in range(args.enc_depth):
            enc = args.enc_name % enc_idx

            # TODO: then load all files for a batch
            if os.path.isfile(args.folder + "/" + str(batch) + "_" + str(enc) + "_attention.npy"):
                x = np.load(args.folder + "/" + str(batch) + "_" + str(enc) + "_attention.npy", allow_pickle=True).item()
            else:
                continue
            # TODO: get correct size
            # TODO: handle search
            tensor = x["attention_tensor"]  # [B, H, I, J]
            #tensor = np.squeeze(tensor, axis=-1)  # [I, J, B]
            if args.do_search:
                tensor = np.transpose(tensor, axes=(1, 0))
                tensor = tensor[::args.beam_size]
                tensor = np.transpose(tensor, axes=(1, 0))

            batch_size = tensor.shape[0]

            # TODO: merge along time axis

            testy = True

            # TODO: load corresponding npy file from returnn file
            r_fname = args.folder + "/" + args.r_prefix + "_ep" + str("{:03d}").format(r_epoch) + "_data_" + str(curr_seq) + "_" \
                         + str(curr_seq + batch_size - 1) + ".npy"
            try:
                r_data = np.load(r_fname, allow_pickle=True).item()

                if testy:
                    assert r_data is not None, "r_data is None!"

                    for tensor_batch in range(batch_size):
                        s = tensor[tensor_batch]
                        r_data[tensor_batch][str(enc)] = s

                # TODO: save in returnn format
                np.save(r_fname, r_data)
            except:
                print(r_fname)

        if batch_size is not 0:
            curr_seq += batch_size


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inspect distributions')
    parser.add_argument('folder', metavar='folder', type=str, help='path to attention folder')
    parser.add_argument('r_prefix', metavar='r_prefix', type=str, help='returnn files prefix')

    parser.add_argument('--enc_name', metavar='enc_name', type=str, default="enc_%02d_self_att_att", help='encoder names')
    parser.add_argument('--enc_depth', metavar='enc_depth', type=int, default=6,
                        help='encoder depth')

    parser.add_argument('--do_search', default=False, action='store_true', help='set this if data gotten using search')
    parser.add_argument('--beam_size', default=12, type=int)

    args = parser.parse_args()
    main(args)
