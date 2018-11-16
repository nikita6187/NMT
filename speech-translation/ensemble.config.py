#!crnn/rnn.py


import os
from subprocess import check_output
import numpy

my_dir = os.path.dirname(os.path.abspath(__file__))

debug_mode = False
if config.has("beam_size"):
    beam_size = config.int("beam_size", 0)
    print("** beam_size %i" % beam_size)
else:
    beam_size = 12

# task
use_tensorflow = True
task = "train"
device = "cpu"
multiprocessing = True
update_on_device = True

extern_data = {"data": {"shape": None, "sparse": True, "dtype": "int32", "dim": 22321, "available_for_inference": True},
                "classes": [22460, 1],
                "sparse_inputs": {"shape": (None, 20), "sparse": True, "dtype": "int32", "dim": 22321, "available_for_inference": True},
                "sparse_weights": {"shape": (None, 20), "sparse": False, "dtype": "float32", "dim": 20, "available_for_inference": True},
                "classes_cns": [22460, 1]}

num_inputs = extern_data["data"]["dim"]

num_seqs = {'train': 370945, 'dev': 888}
EpochSplit = 2
SeqOrderTrainBins = num_seqs["train"] // 1
TrainSeqOrder = "laplace:%i" % SeqOrderTrainBins
if debug_mode:
    TrainSeqOrder = "default"

def get_translation_dataset(data):
    epochSplit = {"train": EpochSplit}.get(data, 1)
    return {
        "class": "TranslationDataset",
        "path": "./dataset",
        "file_postfix": data,
        "source_postfix": " </S>",
        "target_postfix": " </S>",
        'unknown_label' : "<UNK>",
        "partition_epoch": epochSplit,
        "seq_ordering": {"train": TrainSeqOrder, "dev": "sorted"}.get(data, "default"),
        "estimated_num_seqs": (num_seqs.get(data, None) // epochSplit) if data in num_seqs else None}

def get_cn_dataset(data):
    epochSplit = {"train": EpochSplit}.get(data, 1)
    return {
        "class": "ConfusionNetworkDataset",
        "path": "./dataset.cns",
        "file_postfix": data,
        "source_postfix": " </S>",
        "target_postfix": " </S>",
        'unknown_label' : "<UNK>",
        "partition_epoch": epochSplit,
        "seq_ordering": {"train": TrainSeqOrder, "dev": "sorted"}.get(data, "default"),
        "estimated_num_seqs": (num_seqs.get(data, None) // epochSplit) if data in num_seqs else None,
        "max_density": 20}

#train = get_translation_dataset("train")
dev = get_translation_dataset("dev")
dev_cns = get_cn_dataset("dev")
#test_data = get_translation_dataset("test")
evaldata_translation = get_translation_dataset("evaldata")
evaldata_cns = get_cn_dataset("evaldata")

evaldata = {"class": "MetaDataset", "seq_list_file": "./dataset.cns/seq_list_file", "seq_lens_file": "", #"dataset/seq_lens.json",
    "datasets": {"translation": evaldata_translation, "cns": evaldata_cns},
    "data_map": {"data": ("translation", "data"),
                 "classes": ("translation", "classes"),
                 "sparse_inputs": ("cns", "sparse_inputs"),
                 "sparse_weights": ("cns", "sparse_weights"),
                 "classes_cns": ("cns", "classes")},  # not used, expected to be the same as "classes"
    "data_dims": {"data": [22321, 1],
                  "classes": [22460, 1],
                  "sparse_inputs": [22321, 1],
                  "sparse_weights": [20, 2],
                  "classes_cns": [22460, 1]}
}

cache_size = "0"
window = 1

# network
# (also defined by num_inputs & num_outputs)
network = {

#model 0
"source_embed": {"class": "linear", "activation": None, "with_bias": False, "n_out": 620, "from": ["data"]},

"lstm0_fw" : { "class": "rec", "unit": "nativelstm2", "n_out" : 1000, "direction": 1, "from": ["source_embed"] },
"lstm0_bw" : { "class": "rec", "unit": "nativelstm2", "n_out" : 1000, "direction": -1, "from": ["source_embed"] },

"lstm1_fw" : { "class": "rec", "unit": "nativelstm2", "n_out" : 1000, "direction": 1, "from": ["lstm0_fw", "lstm0_bw"] },
"lstm1_bw" : { "class": "rec", "unit": "nativelstm2", "n_out" : 1000, "direction": -1, "from": ["lstm0_fw", "lstm0_bw"] },

"lstm2_fw" : { "class": "rec", "unit": "nativelstm2", "n_out" : 1000, "direction": 1, "from": ["lstm1_fw", "lstm1_bw"] },
"lstm2_bw" : { "class": "rec", "unit": "nativelstm2", "n_out" : 1000, "direction": -1, "from": ["lstm1_fw", "lstm1_bw"] },

"lstm3_fw" : { "class": "rec", "unit": "nativelstm2", "n_out" : 1000, "direction": 1, "from": ["lstm2_fw", "lstm2_bw"] },
"lstm3_bw" : { "class": "rec", "unit": "nativelstm2", "n_out" : 1000, "direction": -1, "from": ["lstm2_fw", "lstm2_bw"] },

"encoder": {"class": "copy", "from": ["lstm3_fw", "lstm3_bw"]},
"enc_ctx": {"class": "linear", "activation": None, "with_bias": True, "from": ["encoder"], "n_out": 1000}, 
"inv_fertility": {"class": "linear", "activation": "sigmoid", "with_bias": False, "from": ["encoder"], "n_out": 1},

#model 1
"model1_source_embed": {'activation': None, 'class': 'linear', "from": ["data:sparse_inputs"], 'n_out': 620, 'with_bias': False},
"model1_source_embed_combine": {'class': "dot", "from": ["model1_source_embed", "data:sparse_weights"], "red1": -2, "var1": -1, "red2": -1, "var2": None, "add_var2_if_empty": False},

"model1_lstm0_fw" : { "class": "rec", "unit": "nativelstm2", "n_out" : 1000, "direction": 1, "from": ["model1_source_embed_combine"] },
"model1_lstm0_bw" : { "class": "rec", "unit": "nativelstm2", "n_out" : 1000, "direction": -1, "from": ["model1_source_embed_combine"] },

"model1_lstm1_fw" : { "class": "rec", "unit": "nativelstm2", "n_out" : 1000, "direction": 1, "from": ["model1_lstm0_fw", "model1_lstm0_bw"] },
"model1_lstm1_bw" : { "class": "rec", "unit": "nativelstm2", "n_out" : 1000, "direction": -1, "from": ["model1_lstm0_fw", "model1_lstm0_bw"] },

"model1_lstm2_fw" : { "class": "rec", "unit": "nativelstm2", "n_out" : 1000, "direction": 1, "from": ["model1_lstm1_fw", "model1_lstm1_bw"] },
"model1_lstm2_bw" : { "class": "rec", "unit": "nativelstm2", "n_out" : 1000, "direction": -1, "from": ["model1_lstm1_fw", "model1_lstm1_bw"] },

"model1_lstm3_fw" : { "class": "rec", "unit": "nativelstm2", "n_out" : 1000, "direction": 1, "from": ["model1_lstm2_fw", "model1_lstm2_bw"] },
"model1_lstm3_bw" : { "class": "rec", "unit": "nativelstm2", "n_out" : 1000, "direction": -1, "from": ["model1_lstm2_fw", "model1_lstm2_bw"] },

"model1_encoder": {"class": "copy", "from": ["model1_lstm3_fw", "model1_lstm3_bw"]},
"model1_enc_ctx": {"class": "linear", "activation": None, "with_bias": True, "from": ["model1_encoder"], "n_out": 1000}, 
"model1_inv_fertility": {"class": "linear", "activation": "sigmoid", "with_bias": False, "from": ["model1_encoder"], "n_out": 1},

"output": {"class": "rec", "from": [], "unit": {
# output of the ensemble
    'output': {'class': 'choice', 'target': 'classes', 'beam_size': beam_size, 'from': ["ens_output_prob"], "initial_output": 0},
     "ens_output_prob": {"class": "eval", "from": ["output_prob", "model1_output_prob" ],
                        "eval": "source(0)*0.5  + 0.5 * source(1) "},
    "end": {"class": "compare", "from": ["output"], "value": 0},
#model 0
    'target_embed': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'], "n_out": 621, "initial_output": 0},  # feedback_input
    "weight_feedback": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev:accum_att_weights"], "n_out": 1000},
    "prev_s_state": {"class": "get_last_hidden_state", "from": ["prev:s"], "n_out": 2000},
    "prev_s_transformed": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev_s_state"], "n_out": 1000},
    "energy_in": {"class": "combine", "kind": "add", "from": ["base:enc_ctx", "weight_feedback", "prev_s_transformed"], "n_out": 1000},
    "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
    "energy": {"class": "linear", "activation": None, "with_bias": False, "from": ["energy_tanh"], "n_out": 1},  # (B, enc-T, 1)
    "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, 1)
    "accum_att_weights": {"class": "eval", "from": ["prev:accum_att_weights", "att_weights", "base:inv_fertility"],
        "eval": "source(0) + source(1) * source(2) * 0.5", "out_type": {"dim": 1, "shape": (None, 1)}},
    "att": {"class": "generic_attention", "weights": "att_weights", "base": "base:encoder"},
    "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["target_embed", "att"], "n_out": 1000},  # transform
    "readout_in": {"class": "linear", "from": ["prev:s", "prev:target_embed", "att"], "activation": None, "n_out": 1000},  # merge + post_merge bias
    "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
    "output_prob": {
        "class": "softmax", "from": ["readout"], "dropout": 0.3,
        "target": "classes", "loss": "ce", "loss_opts": {"label_smoothing": 0.1}
        },
#model 1
    'model1_target_embed': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'], "n_out": 621, "initial_output": 0},  # feedback_input
    "model1_weight_feedback": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev:model1_accum_att_weights"], "n_out": 1000},
    "model1_prev_s_state": {"class": "get_last_hidden_state", "from": ["prev:model1_s"], "n_out": 2000},
    "model1_prev_s_transformed": {"class": "linear", "activation": None, "with_bias": False, "from": ["model1_prev_s_state"], "n_out": 1000},
    "model1_energy_in": {"class": "combine", "kind": "add", "from": ["base:model1_enc_ctx", "model1_weight_feedback", "model1_prev_s_transformed"], "n_out": 1000},
    "model1_energy_tanh": {"class": "activation", "activation": "tanh", "from": ["model1_energy_in"]},
    "model1_energy": {"class": "linear", "activation": None, "with_bias": False, "from": ["model1_energy_tanh"], "n_out": 1},  # (B, enc-T, 1)
    "model1_att_weights": {"class": "softmax_over_spatial", "from": ["model1_energy"]},  # (B, enc-T, 1)
    "model1_accum_att_weights": {"class": "eval", "from": ["prev:model1_accum_att_weights", "model1_att_weights", "base:model1_inv_fertility"],
        "eval": "source(0) + source(1) * source(2) * 0.5", "out_type": {"dim": 1, "shape": (None, 1)}},
    "model1_att": {"class": "generic_attention", "weights": "model1_att_weights", "base": "base:model1_encoder"},
    "model1_s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["model1_target_embed", "model1_att"], "n_out": 1000},  # transform
    "model1_readout_in": {"class": "linear", "from": ["prev:model1_s", "prev:model1_target_embed", "model1_att"], "activation": None, "n_out": 1000},  # merge + post_merge bias
    "model1_readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["model1_readout_in"]},
    "model1_output_prob": {
        "class": "softmax", "from": ["model1_readout"], "dropout": 0.3,
        "target": "classes", "loss": "ce", "loss_opts": {"label_smoothing": 0.1}
        },        
                
}, "target": "classes", "max_seq_len": "max_len_from('base:encoder') * 3"},

"decision": {
    "class": "decide", "from": ["output"], "loss": "edit_distance", "target": "classes",
    "loss_opts": {
        #"debug_print": True
        }
    }
}


search_output_layer = "decision"
debug_print_layer_output_template = True

# models for ensembling
ens_model_file_1 = "./net-model.cns/network.025"
ens_model_prefix_1 = "model1_"

# model 0 is loaded as usual
preload_from_files = {
"model1" : {"filename": ens_model_file_1, "prefix": ens_model_prefix_1},
}

# trainer
batching = "random"
batch_size = 4000
max_seqs = 100
max_seq_length = 75
#chunking = ""  # no chunking
truncation = -1
num_epochs = 1000
model = "net-model/network"
#import_model_train_epoch1= "net-model-v2/network.176"
cleanup_old_models = True
#pretrain = {"output_layers": ["encoder"], "input_layers": ["source_embed"], "repetitions": 5}
gradient_clip = 0
#gradient_clip_global_norm = 1.0
adam = True
optimizer_epsilon = 1e-8
#debug_add_check_numerics_ops = True
debug_add_check_numerics_on_output = True
tf_log_memory_usage = True
gradient_noise = 0.0
learning_rate = 0.0001
learning_rate_control = "newbob_multi_epoch"
#learning_rate_control_error_measure = "dev_score_output"
learning_rate_control_relative_error_relative_lr = True
learning_rate_control_min_num_epochs_per_new_lr = 3
newbob_multi_num_epochs = 6
newbob_multi_update_interval = 1
newbob_learning_rate_decay = 0.9
learning_rate_file = "newbob.data"

log_verbosity = 5
