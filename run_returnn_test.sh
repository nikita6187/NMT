#!/bin/bash

export PATH=~/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/cuda-9.0/extras/CUPTI/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/cuda-9.0/lib64

source /u/makarov/rimes-testing/ENV/bin/activate

python3 /u/bahar/src/python/returnn-22Jun18/returnn/rnn.py $1


