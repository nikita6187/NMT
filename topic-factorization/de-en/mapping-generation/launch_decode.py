import argparse
import os
import subprocess
import shlex


# Usage: python3 launch_decode.py <config path>
# Note: launch from same folder as config

# TODO: put this into root folder

def main(args):
    # TODO: get last
    # TODO: launch from directory where /log and /net-model are subdirs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch config')
    parser.add_argument('p', metavar='p', type=str,
                        help='path to config that you want to launch')
    args = parser.parse_args()
    main(args)