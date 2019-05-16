import argparse
import os
import subprocess


def main(args):
    args.p = os.path.abspath(args.p)

    # Get all folders:
    all_dirs = [args.p + '/' + x + '/' for x in os.listdir(args.p)]

    full_out = "{0:<60} {1:<15} {2:<15} {3:<22} {4:<36} {5:<15} {6:<7} {7:<7} {8:<5} {9:<5}".format("Name           ",
                                                                "Current Epoch", "Epoch Time", "Learning Rate",
                                                                str("Last convergences"), "Last Time", "FER", "Memory",
                                                                                             "Bleu18", "Bleu17")
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
        last_convergences = list(reversed([o.split()[1][0:5] for o in output[:-1] if "dev_score" in o]))

        # Get lr
        com = "grep learningRate " + dir + "newbob.data | tail -1"
        lr = subprocess.Popen(com, shell=True, stdout=subprocess.PIPE)
        lr_pre = str(lr.communicate()[0])
        lr = lr_pre.split('=')[1].split(',')[0]

        # Get fer
        com = "grep output_prob " + dir + "newbob.data | tail -1"
        fer = subprocess.Popen(com, shell=True, stdout=subprocess.PIPE)
        fer_pre = str(fer.communicate()[0])
        fer = fer_pre.split(':')[1].split(',')[0][0:6]

        # Get current epoch
        curr_epoch = str(lr_pre.split(":")[0]).split("'")[1]

        # Get last epoch time
        epoch_time = subprocess.Popen("grep train " + dir + "log/crnn.train.log |  grep finished | tail -1",
                                      shell=True, stdout=subprocess.PIPE)
        epoch_time = str(epoch_time.communicate()[0])
        if len(epoch_time.split()) >= 7:
            epoch_time = epoch_time.split()[7]
        else:
            epoch_time = ""
        # Get name
        name = os.path.basename(os.path.normpath(dir))

        # Get last time something was changed
        change_time = subprocess.Popen("ls -l " + dir + "log/crnn.train.log",
                                      shell=True, stdout=subprocess.PIPE)
        change_time = str(change_time.communicate()[0])
        change_time = "/".join(change_time.split()[5:8])

        # Get memory usage
        mem_usage = subprocess.Popen("grep mem_usage " + dir + "log/crnn.train.log | tail -1",
                                      shell=True, stdout=subprocess.PIPE)
        mem_usage = str(mem_usage.communicate()[0])
        if len(mem_usage.split()) >= 18:
            mem_usage = mem_usage.split()[18][:-1]
        else:
            mem_usage = ""

        # Get best 2018 bleu if possible
        if os.path.isdir(dir + "search/2018/beam12/"):
            bleu18 = subprocess.Popen("python3 /work/smt2/makarov/NMT/run_nested_program.py " + dir
                                    + "search/2018/beam12/" + " tail -3 {} | grep newstest2018",
                                          shell=True, stdout=subprocess.PIPE)
            bleu18 = bleu18.communicate()[0].decode("utf-8")
            bleus18 = bleu18.split("\n")

            if len(bleus18) > 1:
                bleus18 = [b for b in bleus18 if "\t" in b]
                bleus18 = [b.split("\t")[1] for b in bleus18]
                bleus18 = [b.split(" ")[0] for b in bleus18]
                bleus18 = [float(b) for b in bleus18]
                if len(bleus18) > 0:
                    max_bleu18 = max(bleus18)
                else:
                    max_bleu18 = 0.0
            else:
                max_bleu18 = 0.0
        else:
            max_bleu18 = 0.0

        # Get best 2017 bleu if possible
        if os.path.isdir(dir + "search/2017/beam12/"):
            bleu17 = subprocess.Popen("python3 /work/smt2/makarov/NMT/run_nested_program.py " + dir
                                      + "search/2017/beam12/" + " tail -3 {} | grep newstest2017",
                                      shell=True, stdout=subprocess.PIPE)
            bleu17 = bleu17.communicate()[0].decode("utf-8")
            bleus17 = bleu17.split("\n")

            if len(bleus17) > 1:
                bleus17 = [b for b in bleus17 if "\t" in b]
                bleus17 = [b.split("\t")[1] for b in bleus17]
                bleus17 = [b.split(" ")[0] for b in bleus17]
                bleus17 = [float(b) for b in bleus17]
                if len(bleus17) > 0:
                    max_bleu17 = max(bleus17)
                else:
                    max_bleu17 = 0.0
            else:
                max_bleu17 = 0.0
        else:
            max_bleu17 = 0.0

        # print
        data = (name, curr_epoch, epoch_time, lr, str(last_convergences))

        full_out = "{0:<60} {1:<15} {2:<15} {3:<22} {4:<36} {5:<15} {6:<7} {7:<7} {8:<5} {9:<5}".format(name, curr_epoch, epoch_time, lr,
                                                                    str(last_convergences), change_time, fer, mem_usage, max_bleu18,
                                                                                                 max_bleu17)
        print(full_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch config')
    parser.add_argument('p', metavar='p', type=str,
                        help='path to all logs')
    args = parser.parse_args()
    main(args)


