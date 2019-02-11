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

        print(dir)

        # First get newbob data
        raw_output = subprocess.getoutput("grep dev_score " + dir + "newbob.data | tail -3")
        print(raw_output)
        raw_output = raw_output.split('\n')
        last_convergences = [o.split()[1] for o in raw_output]

        # Get lr
        lr = subprocess.check_output("grep learningRate newbob.data | tail -1".split(), cwd=dir, shell=True)
        lr = lr.split('=')[1].split(',')[0]

        # Get last epoch time
        epoch_time = subprocess.check_output("grep train log/crnn.train.log |  grep finished | tail -1".split(),
                                             cwd=dir, shell=True)
        epoch_time = epoch_time.split()[7]

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


