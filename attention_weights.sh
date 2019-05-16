YEAR=$1
CONFIG=$2
EPOCH=$3
BEAM_SIZE=$4
OUTPUT_FOLDER=$5
EXTRA_OPTION=$6

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64:/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda/extras/CUPTI/lib64/:/usr/local/cuda-9.1/extras/CUPTI/lib64/:/usr/local/cuda-9.0/extras/CUPTI/lib64

source /u/bahar/settings/python3-returnn-tf1.9/bin/activate


python3 /work/smt2/makarov/returnn-hmm/tools/get-attention-weights.py ${CONFIG} --epoch ${EPOCH} --data config:newstest${YEAR} --dump_dir ${OUTPUT_FOLDER} --layers "dec_02_att_weights" --layers "dec_01_att_weights" --layers "dec_03_att_weights" --layers "dec_04_att_weights" --layers "dec_05_att_weights" --layers "dec_06_att_weights" --rec_layer "output" --batch_size 600 ${EXTRA_OPTION}


