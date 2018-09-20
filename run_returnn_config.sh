#!/bin/bash

export PATH=/u/makarov/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/makarov/cuda-9.0/extras/CUPTI/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/makarov/cuda-9.0/lib64


source /u/makarov/rimes-testing/ENV/bin/activate

python3 /u/makarov/returnn/rnn.py $1


