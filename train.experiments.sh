#!/bin/bash


export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64:/usr/local/cudnn-9.0-v7.1/lib64

source /u/bahar/settings/python3-returnn-tf1.9/bin/activate



python3 /work/smt2/makarov/returnn-hmm/rnn.py $1


