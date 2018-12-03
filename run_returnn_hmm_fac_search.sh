#!/bin/bash

export PATH=/u/makarov/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/makarov/cuda-9.0/extras/CUPTI/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/makarov/cuda-9.0/lib64

source /u/makarov/rimes-testing/ENV/bin/activate

# Usage: sh run_returnn_hmm_fac_search <Config> <Epoch Nr> <Output> <Log>

python3 /u/makarov/returnn-hmm-fac/rnn.py $1 ++load_epoch $2 ++device 'gpu' --task 'search' ++search_data 'config:dev' ++beam_size '12' ++need_data 'False' ++max_seq_length '0' ++search_output_file $3 &> $4
