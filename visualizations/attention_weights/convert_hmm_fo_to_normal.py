import os
import argparse


def get_max_batch(args):
    all_files = [f for f in listdir(args.folder) if isfile(join(args.folder, f))]
    max_batch = -1
    for file in all_files:
        if len(file) >= len("_attention.npy") and file[-len("_attention.npy"):] == "_attention.npy":
            file = os.path.dirname(file)
            batch = int(file.split("_")[-3])
            if batch > max_batch:
                max_batch = batch
    assert max_batch != -1, "Did not find max batch size!"
    return max_batch


def get_max_time(batch, args):
    all_files = [f for f in listdir(args.folder) if isfile(join(args.folder, f)) and int(f.splt("_")[0]) == batch]
    assert all_files is not None, "Getting batch files is going wrong"

    max_time = -1
    for file in all_files:
        if len(file) >= len("_attention.npy") and file[-len("_attention.npy"):] == "_attention.npy":
            file = os.path.dirname(file)
            time = int(file.split("_")[-2])
            if time > max_time:
                max_time = time
    assert max_time != -1, "Did not get max_time"
    return max_time


def get_returnn_epoch(args):
    all_files = [f for f in listdir(args.folder) if isfile(join(args.folder, f)) and f[:len(args.r_prefix)] == r_prefix]
    assert all_files is not None, "Did not find any files with this prefix!"
    epoch = int(all_files[0].split("_")[1])
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
    for batch in range(max_batch + 1):
        # TODO: then for every batch find max time
        max_time = get_max_time(batch=batch, args=args)
        arrays = []
        for time in range(max_time + 1):
            # TODO: then load all files for a batch
            x = np.load(args.folder + "/" + str(batch) + "_" + str(time) + "_attention.npy").item()
            tensor = x["tensor"]  # [B, J, 1]
            tensor = np.squeeze(tensor, axis=-1)  # [B, J]
            arrays.append(tensor)  # Each of shape [B, J]

        # TODO: merge along time axis
        full_tensor = np.stack(arrays, axis=0)  # Of shape [I, B, J]
        full_tensor = np.transpose(full_tensor, axes=(1, 0, 2))  # [B, I, J]
        batch_size = full_tensor.shape[0]

        # TODO: load corresponding npy file from returnn file
        r_fname = args.folder + "/" + args.r_prefix + "_ep" + str(r_epoch) + "_data_" + str(curr_seq) + "_" \
                     + str(curr_seq + batch_size) + ".npy"
        r_data = np.load(r_fname).item()

        assert r_data is not None, "r_data is None!"

        for tensor_batch in range(batch_size):
            r_data[tensor_batch]["output"] = full_tensor[tensor_batch]

        # TODO: save in returnn format
        np.save(r_fname, r_data)

        curr_seq += batch_size


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inspect distributions')
    parser.add_argument('folder', metavar='folder', type=str, help='path to attention folder')
    parser.add_argument('r_prefix', metavar='r_prefix', type=str, help='returnn files prefix')

    args = parser.parse_args()
    main(args)
