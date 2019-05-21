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
            batch = int(file.split("_")[-3])
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
        # TODO: then for every batch find max time
        max_time = get_max_time(batch=batch, args=args)
        arrays = []
        arrays_post = []
        batch_size = 0
        for time in range(max_time + 1):
            # TODO: then load all files for a batch
            if os.path.isfile(args.folder + "/" + str(batch) + "_" + str(time) + "_attention.npy"):
                x = np.load(args.folder + "/" + str(batch) + "_" + str(time) + "_attention.npy").item()
            else:
                arrays.append(np.zeros(10))
                arrays_post.append(np.zeros(10))
                continue
            # TODO: get correct size
            # TODO: handle search
            tensor = x["attention_tensor"]  # [J, B, 1]
            tensor_post = x["posterior_attention"]
            tensor = np.squeeze(tensor, axis=-1)  # [J, B]
            tensor_post = np.squeeze(tensor_post, axis=-1)  # [J, B]
            if args.do_search:
                tensor = np.transpose(tensor, axes=(1, 0))
                tensor = tensor[::args.beam_size]
                tensor = np.transpose(tensor, axes=(1, 0))

                tensor_post = np.transpose(tensor_post, axes=(1, 0))
                tensor_post = tensor_post[::args.beam_size]
                tensor_post = np.transpose(tensor_post, axes=(1, 0))
            if batch_size == 0:
                batch_size = tensor.shape[1]
            arrays.append(tensor)  # Each of shape [J, B]
            arrays_post.append(tensor_post)  # Each of shape [J, B]

        # TODO: merge along time axis
        testy = min([s.shape == arrays[0].shape for s in arrays])
        print(testy)

        # TODO: load corresponding npy file from returnn file
        r_fname = args.folder + "/" + args.r_prefix + "_ep" + str("{:03d}").format(r_epoch) + "_data_" + str(curr_seq) + "_" \
                     + str(curr_seq + batch_size - 1) + ".npy"
        r_data = np.load(r_fname).item()

        if testy:
            full_tensor = np.stack(arrays, axis=0)  # Of shape [I, J, B]
            full_tensor = np.transpose(full_tensor, axes=(2, 0, 1))  # [B, I, J]

            full_tensor_post = np.stack(arrays_post, axis=0)  # Of shape [I, J, B]
            full_tensor_post = np.transpose(full_tensor_post, axes=(2, 0, 1))  # [B, I, J]

            assert r_data is not None, "r_data is None!"

            for tensor_batch in range(batch_size):
                s = full_tensor[tensor_batch]
                r_data[tensor_batch]["attention_score"] = s
                r_data[tensor_batch]["posterior_attention"] = full_tensor_post[tensor_batch]

            # TODO: save in returnn format
            np.save(r_fname, r_data)

        curr_seq += batch_size


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inspect distributions')
    parser.add_argument('folder', metavar='folder', type=str, help='path to attention folder')
    parser.add_argument('r_prefix', metavar='r_prefix', type=str, help='returnn files prefix')
    parser.add_argument('--do_search', default=False, action='store_true', help='set this if data gotten using search')
    parser.add_argument('--beam_size', default=12, type=int)

    args = parser.parse_args()
    main(args)
