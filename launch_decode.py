import argparse
import os
import subprocess
import shlex
from os import listdir
from os.path import isfile, join
import json

# Usage: python3 launch_config.py <path to config>


def main(args):

    args.p = os.path.abspath(args.p)

    # Get all folders:
    all_log_dirs = [args.p + '/logs/' + x + '/' for x in os.listdir(args.p + "/logs")]

    # Get all configs, these are only file names
    all_configs = [f for f in listdir(args.p) if isfile(join(args.p, f)) is True and f[-len("config"):] == "config"]

    for config_path in all_configs:
        # only consider if its got log dir
        config_log_dir = args.p + "/logs/" + config_path[:-len(".config")] + "/"
        print(config_log_dir)
        if config_log_dir in all_log_dirs:
            if os.path.isdir(config_log_dir + "/net-model/"):
                launch_single(args, config_log_dir, args.p + "/" + config_path)


def isint(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def launch_single(args, model_dir, config_path):

    search_dir = model_dir + "search/"
    search_dir_year = search_dir + args.year + "/"
    search_dir_year_beam = search_dir_year + "beam" + args.beam + "/"

    if not os.path.isdir(search_dir):
        print('Making logs folder: ' + str(search_dir))
        os.mkdir(search_dir)

    if not os.path.isdir(search_dir_year):
        print('Making logs folder: ' + str(search_dir_year))
        os.mkdir(search_dir_year)

    if not os.path.isdir(search_dir_year_beam):
        print('Making logs folder: ' + str(search_dir_year_beam))
        os.mkdir(search_dir_year_beam)

    with open(model_dir + "newbob.data") as json_file:
        data = json_file.readlines()
    data = [x for x in data if "dev_score" in x]
    data = [x.split()[1] for x in data]
    data = [float(x.split(",")[0]) for x in data]

    # TODO: check by list of existing models
    all_available_epochs = [int(f[len("network."):-len(".meta")]) for f in listdir(model_dir + "/net-model/")
                            if isfile(join(model_dir + "/net-model/", f)) is True and
                            f[-len("meta"):] == "meta" and
                            f[:len("network")] == "network" and
                            isint(f[len("network."):-len(".meta")])]
    print(all_available_epochs)

    data = list(zip(data, range(len(data))))  # now tuple of (dev_score, epoch)

    data = [d for d in data if d[1] in all_available_epochs]

    data.sort(key=lambda x: x[0])
    epochs_to_launch = data[:args.amount_of_epochs_to_try]

    for dev_score, epoch in epochs_to_launch:
        # TODO: make folder for epoch
        epoch = str(epoch)
        print("Picking epoch " + epoch + " with dev_score " + str(dev_score))
        search_dir_year_beam_epoch = search_dir_year_beam + epoch + "/"

        # Only launch if it hasn't been done before for these settings
        if not os.path.isdir(search_dir_year_beam_epoch):
            print('Making logs folder: ' + str(search_dir_year_beam_epoch))
            os.mkdir(search_dir_year_beam_epoch)
            # Launching of config
            path_to_runner = "/work/smt2/makarov/NMT/decode.sh"
            launch_command = "qsub -l gpu=1 -l h_rt=1:00:00 -l num_proc=5 -l h_vmem=10G -m abe -cwd {} {} {} {} {} {}"
            launch_command = launch_command.format(path_to_runner, args.year, config_path, epoch, args.beam,
                                                   search_dir_year_beam_epoch)

            # launch_command = shlex.split(launch_command)
            print('Running: ' + str(launch_command) + ' from ' + model_dir)

            # subprocess.Popen(launch_command, cwd=config_dir)
            subprocess.Popen(launch_command, cwd=model_dir, shell=True)
            print('Launched!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch config')

    parser.add_argument('p', metavar='p', type=str,
                        help='path to config folder')

    parser.add_argument('year', metavar='year', type=str,
                        help='Newstest year to use')
    parser.add_argument('beam', metavar='beam', type=str,
                        help='beam_size to use')

    parser.add_argument('amount_of_epochs_to_try', metavar='amount_of_epochs_to_try', type=int,
                        help='amount of epochs to launch from newbob')

    parser.add_argument('--memory', metavar='memory', type=str,
                        help='Max memory needed',
                        default="30",
                        required=False)

    args = parser.parse_args()
    main(args)

