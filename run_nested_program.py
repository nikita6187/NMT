import argparse
import os
import subprocess
import sys
from os import listdir
from os.path import isfile, join

# Applies program iteratively over all files (nested) given a path
# python3 run_nested_program.py <path to start off> <program, use {} where file will go>


def main(path, program):
    all_dirs = [path + '/' + x for x in os.listdir(path) if os.path.isdir(os.path.join(path,x))]
    all_files = [f for f in listdir(path) if isfile(join(path, f)) is True]

    # Apply program to all files
    for file in all_files:
        file = path + "/" + file
        program_to_run = program.format(file)
        subprocess.Popen(program_to_run, cwd=path, shell=True)

    # Then go to all dirs
    for dire in all_dirs:
        main(path=dire, program=program)


if __name__ == '__main__':
    path = os.path.abspath(sys.argv[1])
    program = ' '.join(sys.argv[2:])
    main(path, program)


