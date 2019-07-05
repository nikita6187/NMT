#!/bin/bash

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64:/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda/extras/CUPTI/lib64/:/usr/local/cuda-9.1/extras/CUPTI/lib64/:/usr/local/cuda-9.0/extras/CUPTI/lib64:/usr/lib/nvidia-418
#export LD_LIBRARY_PATH="/usr/local/lib::/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda-9.0/lib64:/usr/lib/nvidia-390:/opt/rbi/sge/lib/glinux:/opt/rbi/sge/lib/glinux"

source /u/bahar/settings/python3-returnn-tf1.9/bin/activate

python3 ~/returnn-vanilla/tools/get-attention-weights.py /u/bahar/workspace/asr/librispeech/test-20190121/config-train/base2.smlp2.specaug.datarndperm_noscale.bs18k.curric3.retrain1.config --epoch 250 --data 'config:get_dataset("'$1'")' --dump_dir ~/makarov/NMT/hmm-factorization/experiments/asr2019/lstm/forward-$1 --layers "att_weights" --rec_layer "output" --batch_size 4000 --tf_log_dir /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/lstm/forward-$1/tf_log_dir --reset_seq_ordering sorted_reverse
