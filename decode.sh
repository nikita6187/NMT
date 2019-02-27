#!/bin/sh

# usage: bash decode.sh <year> <config> <epoch> <beam_size> <output_folder>

YEAR=$1
CONFIG=$2
EPOCH=$3
BEAM_SIZE=$4
OUTPUT_FOLDER=$5

# First run returnn
python3 /work/smt2/makarov/returnn-hmm/rnn.py ++load_epoch ${EPOCH} --task search ++search_data config:newstest${YEAR} ++beam_size ${BEAM_SIZE} ++need_data False ++max_seq_length 0 ++search_output_file ${OUTPUT_FOLDER}scoring_${YEAR}.bpe ++batch_size 2000


