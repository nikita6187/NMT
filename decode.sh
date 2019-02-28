#!/bin/sh

# usage: bash decode.sh <year> <config> <epoch> <beam_size> <output_folder>
# NOTE: cwd should be the log directory (with /log and /net-model subdirs)

YEAR=$1
CONFIG=$2
EPOCH=$3
BEAM_SIZE=$4
OUTPUT_FOLDER=$5

# First run returnn
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64:/usr/local/cudnn-9.0-v7.1/lib64
source /u/bahar/settings/python3-returnn-tf1.9/bin/activate
python3 /work/smt2/makarov/returnn-hmm/rnn.py ${CONFIG} ++load_epoch ${EPOCH} ++device gpu --task search ++search_data config:newstest${YEAR} ++beam_size ${BEAM_SIZE} ++need_data False ++max_seq_length 0 ++search_output_file ${OUTPUT_FOLDER}scoring_${YEAR}_beam${BEAM_SIZE}.bpe ++batch_size 1000 ++log ${OUTPUT_FOLDER}search_log_${YEAR}_beam${BEAM_SIZE}.log
deactivate

# Post processing
cat ${OUTPUT_FOLDER}scoring_${YEAR}_beam${BEAM_SIZE}.bpe | sed 's/@@ //g' | /u/bahar/tools/postprocessing/pp.sh > ${OUTPUT_FOLDER}hyp_${YEAR}_beam${BEAM_SIZE}.pp

# Scoring
/u/bahar/bin/score-deen.sh ${YEAR} ${OUTPUT_FOLDER}hyp_${YEAR}_beam${BEAM_SIZE}.pp > ${OUTPUT_FOLDER}eval_${YEAR}_beam${BEAM_SIZE}.txt

