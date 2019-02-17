import argparse
import os
import subprocess


def main(args):
    args.p = os.path.abspath(args.p)

    # Get all folders:
    all_dirs = [args.p + '/' + x + '/' for x in os.listdir(args.p)]

    full_out = "{0:<60} {1:<15} {2:<15} {3:<22} {4:<36} {5:<12}".format("Name           ",
                                                                "Current Epoch", "Epoch Time", "Learning Rate",
                                                                str("Last convergences"), "Last Time")
    print(full_out)

    # Iterate through all folders and get data
    for dir in all_dirs:

        # First check if its got everything we need
        if os.path.isdir(dir + 'log') is False or os.path.exists(dir + 'newbob.data') is False:
            continue

        # First get newbob data
        com = "grep dev_score " + dir + "newbob.data | tail -3"
        pipe = subprocess.Popen(com, shell=True, stdout=subprocess.PIPE)
        raw_output = str(pipe.communicate()[0])
        output = raw_output.split(",")
        last_convergences = list(reversed([o.split()[1] for o in output[:-1] if "dev_score" in o]))

        # Get lr
        com = "grep learningRate " + dir + "newbob.data | tail -1"
        lr = subprocess.Popen(com, shell=True, stdout=subprocess.PIPE)
        lr_pre = str(lr.communicate()[0])
        lr = lr_pre.split('=')[1].split(',')[0]

        # Get current epoch
        curr_epoch = str(lr_pre.split(":")[0]).split("'")[1]

        # Get last epoch time
        epoch_time = subprocess.Popen("grep train " + dir + "log/crnn.train.log |  grep finished | tail -1",
                                      shell=True, stdout=subprocess.PIPE)
        epoch_time = str(epoch_time.communicate()[0])
        epoch_time = epoch_time.split()[7]

        # Get name
        name = os.path.basename(os.path.normpath(dir))

        # Get last time something was changed
        change_time = subprocess.Popen("ls -l " + dir + "log/crnn.train.log",
                                      shell=True, stdout=subprocess.PIPE)
        change_time = str(change_time.communicate()[0])
        change_time = str(change_time.split()[5:8])

        # print
        data = (name, curr_epoch, epoch_time, lr, str(last_convergences))

        full_out = "{0:<60} {1:<15} {2:<15} {3:<50} {4:<36} {5:<12}".format(name, curr_epoch, epoch_time, lr,
                                                                    str(last_convergences), change_time)
        print(full_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch config')
    parser.add_argument('p', metavar='p', type=str,
                        help='path to all logs')
    args = parser.parse_args()
    main(args)


