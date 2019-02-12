import argparse
import os
import subprocess


def main(args):
    args.p = os.path.abspath(args.p)

    # Get all folders:
    all_dirs = [args.p + '/' + x + '/' for x in os.listdir(args.p)]

    # Iterate through all folders and get data
    for dir in all_dirs:

        # First check if its got everything we need
        if os.path.isdir(dir + 'log') is False or os.path.exists(dir + 'newbob.data') is False:
            continue

        # First get newbob data
        com = "grep dev_score " + dir + "newbob.data | tail -3"
        pipe = subprocess.Popen(com, shell=True)
        raw_output = str(pipe.communicate()[0])
        print(raw_output)
        output = raw_output.split(":")
        print(output)
        print(len(output))
        last_convergences = [o.split()[1] for o in output[:-1]]

        print(str(last_convergences))

        # Get lr
        lr = subprocess.getoutput("grep learningRate" + dir + "newbob.data | tail -1")

        print(lr)

        lr = lr.split('=')[1].split(',')[0]

        print(lr)

        # Get last epoch time
        epoch_time = subprocess.getoutput("grep train " + dir + "log/crnn.train.log |  grep finished | tail -1")

        epoch_time = epoch_time.split()[7]

        print(epoch_time)

        # Get name
        name = os.path.basename(os.path.normpath(dir))

        # print
        print(name + " taking: " + epoch_time + " lr: " + lr + " convergence: " + str(last_convergences))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch config')
    parser.add_argument('p', metavar='p', type=str,
                        help='path to all logs')
    args = parser.parse_args()
    main(args)


