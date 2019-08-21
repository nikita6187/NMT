+------- PROLOGUE SCRIPT -----------------------------------------------
|
| Job ID ...........: 9506002
| Started at .......: Tue Jul  2 12:20:18 CEST 2019
| Execution host ...: cluster-cn-244
| Cluster queue ....: 4-GPU-1080
| Script ...........: /var/spool/sge/cluster-cn-244/job_scripts/9506002
| > #!/bin/bash
| > 
| > export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64:/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda/extras/CUPTI/lib64/:/usr/local/cuda-9.1/extras/CUPTI/lib64/:/usr/local/cuda-9.0/extras/CUPTI/lib64:/usr/lib/nvidia-418
| > #export LD_LIBRARY_PATH="/usr/local/lib::/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda-9.0/lib64:/usr/lib/nvidia-390:/opt/rbi/sge/lib/glinux:/opt/rbi/sge/lib/glinux"
| > 
| > source /u/bahar/settings/python3-returnn-tf1.9/bin/activate
| > 
| > python3 ~/returnn-parnia/tools/get-attention-weights.py /u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3-prior-k6.config --epoch 560 --data 'config:get_dataset("'$1'")' --dump_dir ~/makarov/NMT/hmm-factorization/experiments/asr2019/transformer-prior/forward-$1 --layers "dec_02_att_weights" --layers "dec_01_att_weights" --layers "dec_03_att_weights" --layers "dec_04_att_weights" --layers "dec_05_att_weights" --layers "dec_06_att_weights" --layers "dec_07_att_weights" --layers "dec_08_att_weights" --layers "dec_09_att_weights" --layers "dec_10_att_weights" --layers "dec_11_att_weights" --layers "dec_12_att_weights" --rec_layer "output" --batch_size 4000 --tf_log_dir /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/transformer-prior/forward-$1/tf_log_dir --reset_seq_ordering sorted_reverse
|
+------- PROLOGUE SCRIPT -----------------------------------------------
RETURNN get-attention-weights starting up.
RETURNN starting up, version 20181022.053313--git-dfada43-dirty, date/time 2019-07-02-12-20-21 (UTC+0200), pid 25556, cwd /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3-prior-k6, Python /u/bahar/settings/python3-returnn-tf1.9/bin/python3
RETURNN config: /u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3-prior-k6.config
RETURNN command line options: ()
Hostname: cluster-cn-244
TensorFlow: 1.10.0 (v1.10.0-0-g656e7a2b34) (<site-package> in /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow)
Error while getting SGE num_proc: FileNotFoundError(2, "No such file or directory: 'qstat'")
Setup TF inter and intra global thread pools, num_threads None, session opts {'device_count': {'GPU': 0}, 'log_device_placement': False}.
CUDA_VISIBLE_DEVICES is set to '1'.
Collecting TensorFlow device list...
Local devices available to TensorFlow:
  1/2: name: "/device:CPU:0"
       device_type: "CPU"
       memory_limit: 268435456
       locality {
       }
       incarnation: 15137455021153769521
  2/2: name: "/device:GPU:0"
       device_type: "GPU"
       memory_limit: 10911236096
       locality {
         bus_id: 1
         links {
         }
       }
       incarnation: 6352221750325888957
       physical_device_desc: "device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1"
Using gpu device 1: GeForce GTX 1080 Ti
Device not set explicitly, and we found a GPU, which we will use.
Model file prefix: /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3-prior-k6/net-model/network
NOTE: We will use 'sorted_reverse' seq ordering.
Setup tf.Session with options {'device_count': {'GPU': 1}, 'log_device_placement': False} ...
layer root/'data' output: Data(name='data', shape=(None, 40))
layer root/'source' output: Data(name='source_output', shape=(None, 40))
layer root/'lstm0_fw' output: Data(name='lstm0_fw_output', shape=(None, 1024), batch_dim_axis=1)
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9506002.1.4-GPU-1080/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.cc -o /var/tmp/9506002.1.4-GPU-1080/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9506002.1.4-GPU-1080/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.cc -o /var/tmp/9506002.1.4-GPU-1080/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
layer root/'lstm0_bw' output: Data(name='lstm0_bw_output', shape=(None, 1024), batch_dim_axis=1)
layer root/'lstm0_pool' output: Data(name='lstm0_pool_output', shape=(None, 2048))
layer root/'lstm1_fw' output: Data(name='lstm1_fw_output', shape=(None, 1024), batch_dim_axis=1)
layer root/'lstm1_bw' output: Data(name='lstm1_bw_output', shape=(None, 1024), batch_dim_axis=1)
layer root/'lstm1_pool' output: Data(name='lstm1_pool_output', shape=(None, 2048))
layer root/'source_embed_raw' output: Data(name='source_embed_raw_output', shape=(None, 512))
layer root/'source_embed_weighted' output: Data(name='source_embed_weighted_output', shape=(None, 512))
layer root/'source_embed' output: Data(name='source_embed_output', shape=(None, 512))
layer root/'enc_01_self_att_laynorm' output: Data(name='enc_01_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_01_self_att_att' output: Data(name='enc_01_self_att_att_output', shape=(None, 512))
layer root/'enc_01_self_att_lin' output: Data(name='enc_01_self_att_lin_output', shape=(None, 512))
layer root/'enc_01_self_att_drop' output: Data(name='enc_01_self_att_drop_output', shape=(None, 512))
layer root/'enc_01_self_att_out' output: Data(name='enc_01_self_att_out_output', shape=(None, 512))
layer root/'enc_01_ff_laynorm' output: Data(name='enc_01_ff_laynorm_output', shape=(None, 512))
layer root/'enc_01_ff_conv1' output: Data(name='enc_01_ff_conv1_output', shape=(None, 2048))
layer root/'enc_01_ff_conv2' output: Data(name='enc_01_ff_conv2_output', shape=(None, 512))
layer root/'enc_01_ff_drop' output: Data(name='enc_01_ff_drop_output', shape=(None, 512))
layer root/'enc_01_ff_out' output: Data(name='enc_01_ff_out_output', shape=(None, 512))
layer root/'enc_01' output: Data(name='enc_01_output', shape=(None, 512))
layer root/'enc_02_self_att_laynorm' output: Data(name='enc_02_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_02_self_att_att' output: Data(name='enc_02_self_att_att_output', shape=(None, 512))
layer root/'enc_02_self_att_lin' output: Data(name='enc_02_self_att_lin_output', shape=(None, 512))
layer root/'enc_02_self_att_drop' output: Data(name='enc_02_self_att_drop_output', shape=(None, 512))
layer root/'enc_02_self_att_out' output: Data(name='enc_02_self_att_out_output', shape=(None, 512))
layer root/'enc_02_ff_laynorm' output: Data(name='enc_02_ff_laynorm_output', shape=(None, 512))
layer root/'enc_02_ff_conv1' output: Data(name='enc_02_ff_conv1_output', shape=(None, 2048))
layer root/'enc_02_ff_conv2' output: Data(name='enc_02_ff_conv2_output', shape=(None, 512))
layer root/'enc_02_ff_drop' output: Data(name='enc_02_ff_drop_output', shape=(None, 512))
layer root/'enc_02_ff_out' output: Data(name='enc_02_ff_out_output', shape=(None, 512))
layer root/'enc_02' output: Data(name='enc_02_output', shape=(None, 512))
layer root/'enc_03_self_att_laynorm' output: Data(name='enc_03_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_03_self_att_att' output: Data(name='enc_03_self_att_att_output', shape=(None, 512))
layer root/'enc_03_self_att_lin' output: Data(name='enc_03_self_att_lin_output', shape=(None, 512))
layer root/'enc_03_self_att_drop' output: Data(name='enc_03_self_att_drop_output', shape=(None, 512))
layer root/'enc_03_self_att_out' output: Data(name='enc_03_self_att_out_output', shape=(None, 512))
layer root/'enc_03_ff_laynorm' output: Data(name='enc_03_ff_laynorm_output', shape=(None, 512))
layer root/'enc_03_ff_conv1' output: Data(name='enc_03_ff_conv1_output', shape=(None, 2048))
layer root/'enc_03_ff_conv2' output: Data(name='enc_03_ff_conv2_output', shape=(None, 512))
layer root/'enc_03_ff_drop' output: Data(name='enc_03_ff_drop_output', shape=(None, 512))
layer root/'enc_03_ff_out' output: Data(name='enc_03_ff_out_output', shape=(None, 512))
layer root/'enc_03' output: Data(name='enc_03_output', shape=(None, 512))
layer root/'enc_04_self_att_laynorm' output: Data(name='enc_04_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_04_self_att_att' output: Data(name='enc_04_self_att_att_output', shape=(None, 512))
layer root/'enc_04_self_att_lin' output: Data(name='enc_04_self_att_lin_output', shape=(None, 512))
layer root/'enc_04_self_att_drop' output: Data(name='enc_04_self_att_drop_output', shape=(None, 512))
layer root/'enc_04_self_att_out' output: Data(name='enc_04_self_att_out_output', shape=(None, 512))
layer root/'enc_04_ff_laynorm' output: Data(name='enc_04_ff_laynorm_output', shape=(None, 512))
layer root/'enc_04_ff_conv1' output: Data(name='enc_04_ff_conv1_output', shape=(None, 2048))
layer root/'enc_04_ff_conv2' output: Data(name='enc_04_ff_conv2_output', shape=(None, 512))
layer root/'enc_04_ff_drop' output: Data(name='enc_04_ff_drop_output', shape=(None, 512))
layer root/'enc_04_ff_out' output: Data(name='enc_04_ff_out_output', shape=(None, 512))
layer root/'enc_04' output: Data(name='enc_04_output', shape=(None, 512))
layer root/'enc_05_self_att_laynorm' output: Data(name='enc_05_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_05_self_att_att' output: Data(name='enc_05_self_att_att_output', shape=(None, 512))
layer root/'enc_05_self_att_lin' output: Data(name='enc_05_self_att_lin_output', shape=(None, 512))
layer root/'enc_05_self_att_drop' output: Data(name='enc_05_self_att_drop_output', shape=(None, 512))
layer root/'enc_05_self_att_out' output: Data(name='enc_05_self_att_out_output', shape=(None, 512))
layer root/'enc_05_ff_laynorm' output: Data(name='enc_05_ff_laynorm_output', shape=(None, 512))
layer root/'enc_05_ff_conv1' output: Data(name='enc_05_ff_conv1_output', shape=(None, 2048))
layer root/'enc_05_ff_conv2' output: Data(name='enc_05_ff_conv2_output', shape=(None, 512))
layer root/'enc_05_ff_drop' output: Data(name='enc_05_ff_drop_output', shape=(None, 512))
layer root/'enc_05_ff_out' output: Data(name='enc_05_ff_out_output', shape=(None, 512))
layer root/'enc_05' output: Data(name='enc_05_output', shape=(None, 512))
layer root/'enc_06_self_att_laynorm' output: Data(name='enc_06_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_06_self_att_att' output: Data(name='enc_06_self_att_att_output', shape=(None, 512))
layer root/'enc_06_self_att_lin' output: Data(name='enc_06_self_att_lin_output', shape=(None, 512))
layer root/'enc_06_self_att_drop' output: Data(name='enc_06_self_att_drop_output', shape=(None, 512))
layer root/'enc_06_self_att_out' output: Data(name='enc_06_self_att_out_output', shape=(None, 512))
layer root/'enc_06_ff_laynorm' output: Data(name='enc_06_ff_laynorm_output', shape=(None, 512))
layer root/'enc_06_ff_conv1' output: Data(name='enc_06_ff_conv1_output', shape=(None, 2048))
layer root/'enc_06_ff_conv2' output: Data(name='enc_06_ff_conv2_output', shape=(None, 512))
layer root/'enc_06_ff_drop' output: Data(name='enc_06_ff_drop_output', shape=(None, 512))
layer root/'enc_06_ff_out' output: Data(name='enc_06_ff_out_output', shape=(None, 512))
layer root/'enc_06' output: Data(name='enc_06_output', shape=(None, 512))
layer root/'enc_07_self_att_laynorm' output: Data(name='enc_07_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_07_self_att_att' output: Data(name='enc_07_self_att_att_output', shape=(None, 512))
layer root/'enc_07_self_att_lin' output: Data(name='enc_07_self_att_lin_output', shape=(None, 512))
layer root/'enc_07_self_att_drop' output: Data(name='enc_07_self_att_drop_output', shape=(None, 512))
layer root/'enc_07_self_att_out' output: Data(name='enc_07_self_att_out_output', shape=(None, 512))
layer root/'enc_07_ff_laynorm' output: Data(name='enc_07_ff_laynorm_output', shape=(None, 512))
layer root/'enc_07_ff_conv1' output: Data(name='enc_07_ff_conv1_output', shape=(None, 2048))
layer root/'enc_07_ff_conv2' output: Data(name='enc_07_ff_conv2_output', shape=(None, 512))
layer root/'enc_07_ff_drop' output: Data(name='enc_07_ff_drop_output', shape=(None, 512))
layer root/'enc_07_ff_out' output: Data(name='enc_07_ff_out_output', shape=(None, 512))
layer root/'enc_07' output: Data(name='enc_07_output', shape=(None, 512))
layer root/'enc_08_self_att_laynorm' output: Data(name='enc_08_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_08_self_att_att' output: Data(name='enc_08_self_att_att_output', shape=(None, 512))
layer root/'enc_08_self_att_lin' output: Data(name='enc_08_self_att_lin_output', shape=(None, 512))
layer root/'enc_08_self_att_drop' output: Data(name='enc_08_self_att_drop_output', shape=(None, 512))
layer root/'enc_08_self_att_out' output: Data(name='enc_08_self_att_out_output', shape=(None, 512))
layer root/'enc_08_ff_laynorm' output: Data(name='enc_08_ff_laynorm_output', shape=(None, 512))
layer root/'enc_08_ff_conv1' output: Data(name='enc_08_ff_conv1_output', shape=(None, 2048))
layer root/'enc_08_ff_conv2' output: Data(name='enc_08_ff_conv2_output', shape=(None, 512))
layer root/'enc_08_ff_drop' output: Data(name='enc_08_ff_drop_output', shape=(None, 512))
layer root/'enc_08_ff_out' output: Data(name='enc_08_ff_out_output', shape=(None, 512))
layer root/'enc_08' output: Data(name='enc_08_output', shape=(None, 512))
layer root/'enc_09_self_att_laynorm' output: Data(name='enc_09_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_09_self_att_att' output: Data(name='enc_09_self_att_att_output', shape=(None, 512))
layer root/'enc_09_self_att_lin' output: Data(name='enc_09_self_att_lin_output', shape=(None, 512))
layer root/'enc_09_self_att_drop' output: Data(name='enc_09_self_att_drop_output', shape=(None, 512))
layer root/'enc_09_self_att_out' output: Data(name='enc_09_self_att_out_output', shape=(None, 512))
layer root/'enc_09_ff_laynorm' output: Data(name='enc_09_ff_laynorm_output', shape=(None, 512))
layer root/'enc_09_ff_conv1' output: Data(name='enc_09_ff_conv1_output', shape=(None, 2048))
layer root/'enc_09_ff_conv2' output: Data(name='enc_09_ff_conv2_output', shape=(None, 512))
layer root/'enc_09_ff_drop' output: Data(name='enc_09_ff_drop_output', shape=(None, 512))
layer root/'enc_09_ff_out' output: Data(name='enc_09_ff_out_output', shape=(None, 512))
layer root/'enc_09' output: Data(name='enc_09_output', shape=(None, 512))
layer root/'enc_10_self_att_laynorm' output: Data(name='enc_10_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_10_self_att_att' output: Data(name='enc_10_self_att_att_output', shape=(None, 512))
layer root/'enc_10_self_att_lin' output: Data(name='enc_10_self_att_lin_output', shape=(None, 512))
layer root/'enc_10_self_att_drop' output: Data(name='enc_10_self_att_drop_output', shape=(None, 512))
layer root/'enc_10_self_att_out' output: Data(name='enc_10_self_att_out_output', shape=(None, 512))
layer root/'enc_10_ff_laynorm' output: Data(name='enc_10_ff_laynorm_output', shape=(None, 512))
layer root/'enc_10_ff_conv1' output: Data(name='enc_10_ff_conv1_output', shape=(None, 2048))
layer root/'enc_10_ff_conv2' output: Data(name='enc_10_ff_conv2_output', shape=(None, 512))
layer root/'enc_10_ff_drop' output: Data(name='enc_10_ff_drop_output', shape=(None, 512))
layer root/'enc_10_ff_out' output: Data(name='enc_10_ff_out_output', shape=(None, 512))
layer root/'enc_10' output: Data(name='enc_10_output', shape=(None, 512))
layer root/'enc_11_self_att_laynorm' output: Data(name='enc_11_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_11_self_att_att' output: Data(name='enc_11_self_att_att_output', shape=(None, 512))
layer root/'enc_11_self_att_lin' output: Data(name='enc_11_self_att_lin_output', shape=(None, 512))
layer root/'enc_11_self_att_drop' output: Data(name='enc_11_self_att_drop_output', shape=(None, 512))
layer root/'enc_11_self_att_out' output: Data(name='enc_11_self_att_out_output', shape=(None, 512))
layer root/'enc_11_ff_laynorm' output: Data(name='enc_11_ff_laynorm_output', shape=(None, 512))
layer root/'enc_11_ff_conv1' output: Data(name='enc_11_ff_conv1_output', shape=(None, 2048))
layer root/'enc_11_ff_conv2' output: Data(name='enc_11_ff_conv2_output', shape=(None, 512))
layer root/'enc_11_ff_drop' output: Data(name='enc_11_ff_drop_output', shape=(None, 512))
layer root/'enc_11_ff_out' output: Data(name='enc_11_ff_out_output', shape=(None, 512))
layer root/'enc_11' output: Data(name='enc_11_output', shape=(None, 512))
layer root/'enc_12_self_att_laynorm' output: Data(name='enc_12_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_12_self_att_att' output: Data(name='enc_12_self_att_att_output', shape=(None, 512))
layer root/'enc_12_self_att_lin' output: Data(name='enc_12_self_att_lin_output', shape=(None, 512))
layer root/'enc_12_self_att_drop' output: Data(name='enc_12_self_att_drop_output', shape=(None, 512))
layer root/'enc_12_self_att_out' output: Data(name='enc_12_self_att_out_output', shape=(None, 512))
layer root/'enc_12_ff_laynorm' output: Data(name='enc_12_ff_laynorm_output', shape=(None, 512))
layer root/'enc_12_ff_conv1' output: Data(name='enc_12_ff_conv1_output', shape=(None, 2048))
layer root/'enc_12_ff_conv2' output: Data(name='enc_12_ff_conv2_output', shape=(None, 512))
layer root/'enc_12_ff_drop' output: Data(name='enc_12_ff_drop_output', shape=(None, 512))
layer root/'enc_12_ff_out' output: Data(name='enc_12_ff_out_output', shape=(None, 512))
layer root/'enc_12' output: Data(name='enc_12_output', shape=(None, 512))
layer root/'encoder' output: Data(name='encoder_output', shape=(None, 512))
layer root/'ctc' output: Data(name='ctc_output', shape=(None, 10026))
layer root/'dec_11_att_key0' output: Data(name='dec_11_att_key0_output', shape=(None, 512))
layer root/'dec_11_att_key' output: Data(name='dec_11_att_key_output', shape=(None, 8, 64))
layer root/'dec_07_att_value0' output: Data(name='dec_07_att_value0_output', shape=(None, 512))
layer root/'dec_07_att_value' output: Data(name='dec_07_att_value_output', shape=(None, 8, 64))
layer root/'dec_01_att_value0' output: Data(name='dec_01_att_value0_output', shape=(None, 512))
layer root/'dec_01_att_value' output: Data(name='dec_01_att_value_output', shape=(None, 8, 64))
layer root/'dec_10_att_value0' output: Data(name='dec_10_att_value0_output', shape=(None, 512))
layer root/'dec_10_att_value' output: Data(name='dec_10_att_value_output', shape=(None, 8, 64))
layer root/'dec_08_att_value0' output: Data(name='dec_08_att_value0_output', shape=(None, 512))
layer root/'dec_08_att_value' output: Data(name='dec_08_att_value_output', shape=(None, 8, 64))
layer root/'dec_06_att_value0' output: Data(name='dec_06_att_value0_output', shape=(None, 512))
layer root/'dec_06_att_value' output: Data(name='dec_06_att_value_output', shape=(None, 8, 64))
layer root/'dec_12_att_key0' output: Data(name='dec_12_att_key0_output', shape=(None, 512))
layer root/'dec_12_att_key' output: Data(name='dec_12_att_key_output', shape=(None, 8, 64))
layer root/'dec_12_att_value0' output: Data(name='dec_12_att_value0_output', shape=(None, 512))
layer root/'dec_12_att_value' output: Data(name='dec_12_att_value_output', shape=(None, 8, 64))
layer root/'dec_10_att_key0' output: Data(name='dec_10_att_key0_output', shape=(None, 512))
layer root/'dec_10_att_key' output: Data(name='dec_10_att_key_output', shape=(None, 8, 64))
layer root/'dec_04_att_key0' output: Data(name='dec_04_att_key0_output', shape=(None, 512))
layer root/'dec_04_att_key' output: Data(name='dec_04_att_key_output', shape=(None, 8, 64))
layer root/'dec_03_att_value0' output: Data(name='dec_03_att_value0_output', shape=(None, 512))
layer root/'dec_03_att_value' output: Data(name='dec_03_att_value_output', shape=(None, 8, 64))
layer root/'dec_01_att_key0' output: Data(name='dec_01_att_key0_output', shape=(None, 512))
layer root/'dec_01_att_key' output: Data(name='dec_01_att_key_output', shape=(None, 8, 64))
layer root/'dec_09_att_value0' output: Data(name='dec_09_att_value0_output', shape=(None, 512))
layer root/'dec_09_att_value' output: Data(name='dec_09_att_value_output', shape=(None, 8, 64))
layer root/'dec_03_att_key0' output: Data(name='dec_03_att_key0_output', shape=(None, 512))
layer root/'dec_03_att_key' output: Data(name='dec_03_att_key_output', shape=(None, 8, 64))
layer root/'dec_04_att_value0' output: Data(name='dec_04_att_value0_output', shape=(None, 512))
layer root/'dec_04_att_value' output: Data(name='dec_04_att_value_output', shape=(None, 8, 64))
layer root/'dec_02_att_value0' output: Data(name='dec_02_att_value0_output', shape=(None, 512))
layer root/'dec_02_att_value' output: Data(name='dec_02_att_value_output', shape=(None, 8, 64))
layer root/'dec_08_att_key0' output: Data(name='dec_08_att_key0_output', shape=(None, 512))
layer root/'dec_08_att_key' output: Data(name='dec_08_att_key_output', shape=(None, 8, 64))
layer root/'dec_05_att_key0' output: Data(name='dec_05_att_key0_output', shape=(None, 512))
layer root/'dec_05_att_key' output: Data(name='dec_05_att_key_output', shape=(None, 8, 64))
layer root/'dec_05_att_value0' output: Data(name='dec_05_att_value0_output', shape=(None, 512))
layer root/'dec_05_att_value' output: Data(name='dec_05_att_value_output', shape=(None, 8, 64))
layer root/'dec_02_att_key0' output: Data(name='dec_02_att_key0_output', shape=(None, 512))
layer root/'dec_02_att_key' output: Data(name='dec_02_att_key_output', shape=(None, 8, 64))
layer root/'dec_11_att_value0' output: Data(name='dec_11_att_value0_output', shape=(None, 512))
layer root/'dec_11_att_value' output: Data(name='dec_11_att_value_output', shape=(None, 8, 64))
layer root/'dec_07_att_key0' output: Data(name='dec_07_att_key0_output', shape=(None, 512))
layer root/'dec_07_att_key' output: Data(name='dec_07_att_key_output', shape=(None, 8, 64))
layer root/'dec_06_att_key0' output: Data(name='dec_06_att_key0_output', shape=(None, 512))
layer root/'dec_06_att_key' output: Data(name='dec_06_att_key_output', shape=(None, 8, 64))
layer root/'dec_09_att_key0' output: Data(name='dec_09_att_key0_output', shape=(None, 512))
layer root/'dec_09_att_key' output: Data(name='dec_09_att_key_output', shape=(None, 8, 64))
layer root/'output' output: Data(name='output_output', shape=(None,), dtype='int32', sparse=True, dim=10025, batch_dim_axis=1)
Rec layer sub net:
  Input layers moved out of loop: (#: 0)
    None
  Output layers moved out of loop: (#: 273)
    output_prob
    decoder_int
    decoder
    dec_12
    dec_12_ff_out
    dec_12_ff_drop
    dec_12_ff_conv2
    dec_12_ff_conv1
    dec_12_ff_laynorm
    dec_12_att_out
    dec_12_att_drop
    dec_12_att_lin
    dec_12_att_att
    dec_12_att0
    dec_12_att_weights_drop
    dec_12_att_weights
    dec_12_att_energy
    dec_12_att_query
    dec_12_att_query0
    dec_12_att_laynorm
    dec_12_self_att_out
    dec_12_self_att_drop
    dec_12_self_att_lin
    dec_12_self_att_att
    dec_12_self_att_laynorm
    dec_11
    dec_11_ff_out
    dec_11_ff_drop
    dec_11_ff_conv2
    dec_11_ff_conv1
    dec_11_ff_laynorm
    dec_11_att_out
    dec_11_att_drop
    dec_11_att_lin
    dec_11_att_att
    dec_11_att0
    dec_11_att_weights_drop
    dec_11_att_weights
    dec_11_att_energy
    dec_11_att_query
    dec_11_att_query0
    dec_11_att_laynorm
    dec_11_self_att_out
    dec_11_self_att_drop
    dec_11_self_att_lin
    dec_11_self_att_att
    dec_11_self_att_laynorm
    dec_10
    dec_10_ff_out
    dec_10_ff_drop
    dec_10_ff_conv2
    dec_10_ff_conv1
    dec_10_ff_laynorm
    dec_10_att_out
    dec_10_att_drop
    dec_10_att_lin
    dec_10_att_att
    dec_10_att0
    dec_10_att_weights_drop
    dec_10_att_weights
    dec_10_att_energy
    dec_10_att_query
    dec_10_att_query0
    dec_10_att_laynorm
    dec_10_self_att_out
    dec_10_self_att_drop
    dec_10_self_att_lin
    dec_10_self_att_att
    dec_10_self_att_laynorm
    dec_09
    dec_09_ff_out
    dec_09_ff_drop
    dec_09_ff_conv2
    dec_09_ff_conv1
    dec_09_ff_laynorm
    dec_09_att_out
    dec_09_att_drop
    dec_09_att_lin
    dec_09_att_att
    dec_09_att0
    dec_09_att_weights_drop
    dec_09_att_weights
    dec_09_att_energy
    dec_09_att_query
    dec_09_att_query0
    dec_09_att_laynorm
    dec_09_self_att_out
    dec_09_self_att_drop
    dec_09_self_att_lin
    dec_09_self_att_att
    dec_09_self_att_laynorm
    dec_08
    dec_08_ff_out
    dec_08_ff_drop
    dec_08_ff_conv2
    dec_08_ff_conv1
    dec_08_ff_laynorm
    dec_08_att_out
    dec_08_att_drop
    dec_08_att_lin
    dec_08_att_att
    dec_08_att0
    dec_08_att_weights_drop
    dec_08_att_weights
    dec_08_att_energy
    dec_08_att_query
    dec_08_att_query0
    dec_08_att_laynorm
    dec_08_self_att_out
    dec_08_self_att_drop
    dec_08_self_att_lin
    dec_08_self_att_att
    dec_08_self_att_laynorm
    dec_07
    dec_07_ff_out
    dec_07_ff_drop
    dec_07_ff_conv2
    dec_07_ff_conv1
    dec_07_ff_laynorm
    dec_07_att_out
    dec_07_att_drop
    dec_07_att_lin
    dec_07_att_att
    dec_07_att0
    dec_07_att_weights_drop
    dec_07_att_weights
    dec_07_att_energy
    dec_07_att_query
    dec_07_att_query0
    dec_07_att_laynorm
    dec_07_self_att_out
    dec_07_self_att_drop
    dec_07_self_att_lin
    dec_07_self_att_att
    dec_07_self_att_laynorm
    dec_06
    dec_06_ff_out
    dec_06_ff_drop
    dec_06_ff_conv2
    dec_06_ff_conv1
    dec_06_ff_laynorm
    dec_06_att_out
    dec_06_att_drop
    dec_06_att_lin
    dec_06_att_att
    dec_06_att0
    dec_06_att_weights_drop
    dec_06_att_weights
    dec_06_att_energy
    dec_06_att_query
    dec_06_att_query0
    dec_06_att_laynorm
    dec_06_self_att_out
    dec_06_self_att_drop
    dec_06_self_att_lin
    dec_06_self_att_att
    dec_06_self_att_laynorm
    dec_05
    dec_05_ff_out
    dec_05_ff_drop
    dec_05_ff_conv2
    dec_05_ff_conv1
    dec_05_ff_laynorm
    dec_05_att_out
    dec_05_att_drop
    dec_05_att_lin
    dec_05_att_att
    dec_05_att0
    dec_05_att_weights_drop
    dec_05_att_weights
    dec_05_att_energy
    dec_05_att_query
    dec_05_att_query0
    dec_05_att_laynorm
    dec_05_self_att_out
    dec_05_self_att_drop
    dec_05_self_att_lin
    dec_05_self_att_att
    dec_05_self_att_laynorm
    dec_04
    dec_04_ff_out
    dec_04_ff_drop
    dec_04_ff_conv2
    dec_04_ff_conv1
    dec_04_ff_laynorm
    dec_04_att_out
    dec_04_att_drop
    dec_04_att_lin
    dec_04_att_att
    dec_04_att0
    dec_04_att_weights_drop
    dec_04_att_weights
    dec_04_att_energy
    dec_04_att_query
    dec_04_att_query0
    dec_04_att_laynorm
    dec_04_self_att_out
    dec_04_self_att_drop
    dec_04_self_att_lin
    dec_04_self_att_att
    dec_04_self_att_laynorm
    dec_03
    dec_03_ff_out
    dec_03_ff_drop
    dec_03_ff_conv2
    dec_03_ff_conv1
    dec_03_ff_laynorm
    dec_03_att_out
    dec_03_att_drop
    dec_03_att_lin
    dec_03_att_att
    dec_03_att0
    dec_03_att_weights_drop
    dec_03_att_weights
    dec_03_att_energy
    dec_03_att_query
    dec_03_att_query0
    dec_03_att_laynorm
    dec_03_self_att_out
    dec_03_self_att_drop
    dec_03_self_att_lin
    dec_03_self_att_att
    dec_03_self_att_laynorm
    dec_02
    dec_02_ff_out
    dec_02_ff_drop
    dec_02_ff_conv2
    dec_02_ff_conv1
    dec_02_ff_laynorm
    dec_02_att_out
    dec_02_att_drop
    dec_02_att_lin
    dec_02_att_att
    dec_02_att0
    dec_02_att_weights_drop
    dec_02_att_weights
    dec_02_att_energy
    dec_02_att_query
    dec_02_att_query0
    dec_02_att_laynorm
    dec_02_self_att_out
    dec_02_self_att_drop
    dec_02_self_att_lin
    dec_02_self_att_att
    dec_02_self_att_laynorm
    dec_01
    dec_01_ff_out
    dec_01_ff_drop
    dec_01_ff_conv2
    dec_01_ff_conv1
    dec_01_ff_laynorm
    dec_01_att_out
    dec_01_att_drop
    dec_01_att_lin
    dec_01_att_att
    dec_01_att0
    dec_01_att_weights_drop
    dec_01_att_weights
    dec_01_att_energy
    dec_01_att_query
    dec_01_att_query0
    dec_01_att_laynorm
    dec_01_self_att_out
    dec_01_self_att_drop
    dec_01_self_att_lin
    dec_01_self_att_att
    dec_01_self_att_laynorm
    target_embed
    target_embed_weighted
    encoder_int
    prev_outputs_int
    target_embed_raw
    output
  Layers in loop: (#: 0)
    None
  Unused layers: (#: 1)
    end
layer root/output:rec-subnet-output/'output' output: Data(name='classes', shape=(None,), dtype='int32', sparse=True, dim=10025)
layer root/output:rec-subnet-output/'prev:output' output: Data(name='classes', shape=(None,), dtype='int32', sparse=True, dim=10025, batch_dim_axis=1)
layer root/output:rec-subnet-output/'target_embed_raw' output: Data(name='target_embed_raw_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'target_embed_weighted' output: Data(name='target_embed_weighted_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'target_embed' output: Data(name='target_embed_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_self_att_laynorm' output: Data(name='dec_01_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_self_att_att' output: Data(name='dec_01_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_01_self_att_lin' output: Data(name='dec_01_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_01_self_att_drop' output: Data(name='dec_01_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_01_self_att_out' output: Data(name='dec_01_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_att_laynorm' output: Data(name='dec_01_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_att_query0' output: Data(name='dec_01_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_att_query' output: Data(name='dec_01_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_att_energy' output: Data(name='dec_01_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_01_att_weights' output: Data(name='dec_01_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_01_att_weights_drop' output: Data(name='dec_01_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_01_att0' output: Data(name='dec_01_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_01_att_att' output: Data(name='dec_01_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_01_att_lin' output: Data(name='dec_01_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_01_att_drop' output: Data(name='dec_01_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_01_att_out' output: Data(name='dec_01_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_ff_laynorm' output: Data(name='dec_01_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_ff_conv1' output: Data(name='dec_01_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_ff_conv2' output: Data(name='dec_01_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_ff_drop' output: Data(name='dec_01_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_ff_out' output: Data(name='dec_01_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01' output: Data(name='dec_01_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_self_att_laynorm' output: Data(name='dec_02_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_self_att_att' output: Data(name='dec_02_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_02_self_att_lin' output: Data(name='dec_02_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_02_self_att_drop' output: Data(name='dec_02_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_02_self_att_out' output: Data(name='dec_02_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_att_laynorm' output: Data(name='dec_02_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_att_query0' output: Data(name='dec_02_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_att_query' output: Data(name='dec_02_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_att_energy' output: Data(name='dec_02_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_02_att_weights' output: Data(name='dec_02_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_02_att_weights_drop' output: Data(name='dec_02_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_02_att0' output: Data(name='dec_02_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_02_att_att' output: Data(name='dec_02_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_02_att_lin' output: Data(name='dec_02_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_02_att_drop' output: Data(name='dec_02_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_02_att_out' output: Data(name='dec_02_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_ff_laynorm' output: Data(name='dec_02_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_ff_conv1' output: Data(name='dec_02_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_ff_conv2' output: Data(name='dec_02_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_ff_drop' output: Data(name='dec_02_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_ff_out' output: Data(name='dec_02_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02' output: Data(name='dec_02_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_self_att_laynorm' output: Data(name='dec_03_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_self_att_att' output: Data(name='dec_03_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_03_self_att_lin' output: Data(name='dec_03_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_03_self_att_drop' output: Data(name='dec_03_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_03_self_att_out' output: Data(name='dec_03_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_att_laynorm' output: Data(name='dec_03_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_att_query0' output: Data(name='dec_03_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_att_query' output: Data(name='dec_03_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_att_energy' output: Data(name='dec_03_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_03_att_weights' output: Data(name='dec_03_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_03_att_weights_drop' output: Data(name='dec_03_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_03_att0' output: Data(name='dec_03_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_03_att_att' output: Data(name='dec_03_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_03_att_lin' output: Data(name='dec_03_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_03_att_drop' output: Data(name='dec_03_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_03_att_out' output: Data(name='dec_03_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_ff_laynorm' output: Data(name='dec_03_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_ff_conv1' output: Data(name='dec_03_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_ff_conv2' output: Data(name='dec_03_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_ff_drop' output: Data(name='dec_03_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_ff_out' output: Data(name='dec_03_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03' output: Data(name='dec_03_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_self_att_laynorm' output: Data(name='dec_04_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_self_att_att' output: Data(name='dec_04_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_04_self_att_lin' output: Data(name='dec_04_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_04_self_att_drop' output: Data(name='dec_04_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_04_self_att_out' output: Data(name='dec_04_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_att_laynorm' output: Data(name='dec_04_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_att_query0' output: Data(name='dec_04_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_att_query' output: Data(name='dec_04_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_att_energy' output: Data(name='dec_04_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_04_att_weights' output: Data(name='dec_04_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_04_att_weights_drop' output: Data(name='dec_04_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_04_att0' output: Data(name='dec_04_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_04_att_att' output: Data(name='dec_04_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_04_att_lin' output: Data(name='dec_04_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_04_att_drop' output: Data(name='dec_04_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_04_att_out' output: Data(name='dec_04_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_ff_laynorm' output: Data(name='dec_04_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_ff_conv1' output: Data(name='dec_04_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_ff_conv2' output: Data(name='dec_04_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_ff_drop' output: Data(name='dec_04_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_ff_out' output: Data(name='dec_04_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04' output: Data(name='dec_04_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_self_att_laynorm' output: Data(name='dec_05_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_self_att_att' output: Data(name='dec_05_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_05_self_att_lin' output: Data(name='dec_05_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_05_self_att_drop' output: Data(name='dec_05_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_05_self_att_out' output: Data(name='dec_05_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_att_laynorm' output: Data(name='dec_05_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_att_query0' output: Data(name='dec_05_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_att_query' output: Data(name='dec_05_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_att_energy' output: Data(name='dec_05_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_05_att_weights' output: Data(name='dec_05_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_05_att_weights_drop' output: Data(name='dec_05_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_05_att0' output: Data(name='dec_05_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_05_att_att' output: Data(name='dec_05_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_05_att_lin' output: Data(name='dec_05_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_05_att_drop' output: Data(name='dec_05_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_05_att_out' output: Data(name='dec_05_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_ff_laynorm' output: Data(name='dec_05_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_ff_conv1' output: Data(name='dec_05_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_ff_conv2' output: Data(name='dec_05_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_ff_drop' output: Data(name='dec_05_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_ff_out' output: Data(name='dec_05_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05' output: Data(name='dec_05_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_self_att_laynorm' output: Data(name='dec_06_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_self_att_att' output: Data(name='dec_06_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_06_self_att_lin' output: Data(name='dec_06_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_06_self_att_drop' output: Data(name='dec_06_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_06_self_att_out' output: Data(name='dec_06_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_att_laynorm' output: Data(name='dec_06_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_att_query0' output: Data(name='dec_06_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_att_query' output: Data(name='dec_06_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_att_energy' output: Data(name='dec_06_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_06_att_weights' output: Data(name='dec_06_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_06_att_weights_drop' output: Data(name='dec_06_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_06_att0' output: Data(name='dec_06_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_06_att_att' output: Data(name='dec_06_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_06_att_lin' output: Data(name='dec_06_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_06_att_drop' output: Data(name='dec_06_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_06_att_out' output: Data(name='dec_06_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_ff_laynorm' output: Data(name='dec_06_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_ff_conv1' output: Data(name='dec_06_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_ff_conv2' output: Data(name='dec_06_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_ff_drop' output: Data(name='dec_06_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_ff_out' output: Data(name='dec_06_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06' output: Data(name='dec_06_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_self_att_laynorm' output: Data(name='dec_07_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_self_att_att' output: Data(name='dec_07_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_07_self_att_lin' output: Data(name='dec_07_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_07_self_att_drop' output: Data(name='dec_07_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_07_self_att_out' output: Data(name='dec_07_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_att_laynorm' output: Data(name='dec_07_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_att_query0' output: Data(name='dec_07_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_att_query' output: Data(name='dec_07_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_att_energy' output: Data(name='dec_07_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_07_att_weights' output: Data(name='dec_07_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_07_att_weights_drop' output: Data(name='dec_07_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_07_att0' output: Data(name='dec_07_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_07_att_att' output: Data(name='dec_07_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_07_att_lin' output: Data(name='dec_07_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_07_att_drop' output: Data(name='dec_07_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_07_att_out' output: Data(name='dec_07_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_ff_laynorm' output: Data(name='dec_07_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_ff_conv1' output: Data(name='dec_07_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_ff_conv2' output: Data(name='dec_07_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_ff_drop' output: Data(name='dec_07_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_ff_out' output: Data(name='dec_07_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07' output: Data(name='dec_07_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_self_att_laynorm' output: Data(name='dec_08_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_self_att_att' output: Data(name='dec_08_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_08_self_att_lin' output: Data(name='dec_08_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_08_self_att_drop' output: Data(name='dec_08_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_08_self_att_out' output: Data(name='dec_08_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_att_laynorm' output: Data(name='dec_08_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_att_query0' output: Data(name='dec_08_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_att_query' output: Data(name='dec_08_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_att_energy' output: Data(name='dec_08_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_08_att_weights' output: Data(name='dec_08_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_08_att_weights_drop' output: Data(name='dec_08_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_08_att0' output: Data(name='dec_08_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_08_att_att' output: Data(name='dec_08_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_08_att_lin' output: Data(name='dec_08_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_08_att_drop' output: Data(name='dec_08_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_08_att_out' output: Data(name='dec_08_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_ff_laynorm' output: Data(name='dec_08_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_ff_conv1' output: Data(name='dec_08_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_ff_conv2' output: Data(name='dec_08_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_ff_drop' output: Data(name='dec_08_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_ff_out' output: Data(name='dec_08_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08' output: Data(name='dec_08_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_self_att_laynorm' output: Data(name='dec_09_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_self_att_att' output: Data(name='dec_09_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_09_self_att_lin' output: Data(name='dec_09_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_09_self_att_drop' output: Data(name='dec_09_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_09_self_att_out' output: Data(name='dec_09_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_att_laynorm' output: Data(name='dec_09_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_att_query0' output: Data(name='dec_09_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_att_query' output: Data(name='dec_09_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_att_energy' output: Data(name='dec_09_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_09_att_weights' output: Data(name='dec_09_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_09_att_weights_drop' output: Data(name='dec_09_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_09_att0' output: Data(name='dec_09_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_09_att_att' output: Data(name='dec_09_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_09_att_lin' output: Data(name='dec_09_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_09_att_drop' output: Data(name='dec_09_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_09_att_out' output: Data(name='dec_09_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_ff_laynorm' output: Data(name='dec_09_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_ff_conv1' output: Data(name='dec_09_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_ff_conv2' output: Data(name='dec_09_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_ff_drop' output: Data(name='dec_09_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_ff_out' output: Data(name='dec_09_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09' output: Data(name='dec_09_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_self_att_laynorm' output: Data(name='dec_10_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_self_att_att' output: Data(name='dec_10_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_10_self_att_lin' output: Data(name='dec_10_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_10_self_att_drop' output: Data(name='dec_10_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_10_self_att_out' output: Data(name='dec_10_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_att_laynorm' output: Data(name='dec_10_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_att_query0' output: Data(name='dec_10_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_att_query' output: Data(name='dec_10_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_att_energy' output: Data(name='dec_10_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_10_att_weights' output: Data(name='dec_10_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_10_att_weights_drop' output: Data(name='dec_10_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_10_att0' output: Data(name='dec_10_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_10_att_att' output: Data(name='dec_10_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_10_att_lin' output: Data(name='dec_10_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_10_att_drop' output: Data(name='dec_10_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_10_att_out' output: Data(name='dec_10_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_ff_laynorm' output: Data(name='dec_10_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_ff_conv1' output: Data(name='dec_10_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_ff_conv2' output: Data(name='dec_10_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_ff_drop' output: Data(name='dec_10_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_ff_out' output: Data(name='dec_10_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10' output: Data(name='dec_10_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_self_att_laynorm' output: Data(name='dec_11_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_self_att_att' output: Data(name='dec_11_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_11_self_att_lin' output: Data(name='dec_11_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_11_self_att_drop' output: Data(name='dec_11_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_11_self_att_out' output: Data(name='dec_11_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_att_laynorm' output: Data(name='dec_11_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_att_query0' output: Data(name='dec_11_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_att_query' output: Data(name='dec_11_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_att_energy' output: Data(name='dec_11_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_11_att_weights' output: Data(name='dec_11_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_11_att_weights_drop' output: Data(name='dec_11_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_11_att0' output: Data(name='dec_11_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_11_att_att' output: Data(name='dec_11_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_11_att_lin' output: Data(name='dec_11_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_11_att_drop' output: Data(name='dec_11_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_11_att_out' output: Data(name='dec_11_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_ff_laynorm' output: Data(name='dec_11_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_ff_conv1' output: Data(name='dec_11_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_ff_conv2' output: Data(name='dec_11_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_ff_drop' output: Data(name='dec_11_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_ff_out' output: Data(name='dec_11_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11' output: Data(name='dec_11_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_self_att_laynorm' output: Data(name='dec_12_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_self_att_att' output: Data(name='dec_12_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_12_self_att_lin' output: Data(name='dec_12_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_12_self_att_drop' output: Data(name='dec_12_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_12_self_att_out' output: Data(name='dec_12_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_att_laynorm' output: Data(name='dec_12_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_att_query0' output: Data(name='dec_12_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_att_query' output: Data(name='dec_12_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_att_energy' output: Data(name='dec_12_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_12_att_weights' output: Data(name='dec_12_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'encoder_int' output: Data(name='encoder_int_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_12_att_weights_drop' output: Data(name='dec_12_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_12_att0' output: Data(name='dec_12_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_12_att_att' output: Data(name='dec_12_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_12_att_lin' output: Data(name='dec_12_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_12_att_drop' output: Data(name='dec_12_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_12_att_out' output: Data(name='dec_12_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_ff_laynorm' output: Data(name='dec_12_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_ff_conv1' output: Data(name='dec_12_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_ff_conv2' output: Data(name='dec_12_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_ff_drop' output: Data(name='dec_12_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_ff_out' output: Data(name='dec_12_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12' output: Data(name='dec_12_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'decoder' output: Data(name='decoder_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'decoder_int' output: Data(name='decoder_int_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'prev:target_embed_raw' output: Data(name='target_embed_raw_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'prev_outputs_int' output: Data(name='prev_outputs_int_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'output_prob' output: Data(name='dec_12_att_weights_output', shape=(None, 10025), batch_dim_axis=1)
layer root/'decision' output: Data(name='output_output', shape=(None,), dtype='int32', sparse=True, dim=10025, batch_dim_axis=1)
Warning: using numerical unstable sparse Cross-Entropy loss calculation
Network layer topology:
  extern data: classes: Data(shape=(None,), dtype='int32', sparse=True, dim=10025, available_for_inference=False), data: Data(shape=(None, 40))
  used data keys: ['classes', 'data']
  layer softmax 'ctc' #: 10026
  layer source 'data' #: 40
  layer split_dims 'dec_01_att_key' #: 64
  layer linear 'dec_01_att_key0' #: 512
  layer split_dims 'dec_01_att_value' #: 64
  layer linear 'dec_01_att_value0' #: 512
  layer split_dims 'dec_02_att_key' #: 64
  layer linear 'dec_02_att_key0' #: 512
  layer split_dims 'dec_02_att_value' #: 64
  layer linear 'dec_02_att_value0' #: 512
  layer split_dims 'dec_03_att_key' #: 64
  layer linear 'dec_03_att_key0' #: 512
  layer split_dims 'dec_03_att_value' #: 64
  layer linear 'dec_03_att_value0' #: 512
  layer split_dims 'dec_04_att_key' #: 64
  layer linear 'dec_04_att_key0' #: 512
  layer split_dims 'dec_04_att_value' #: 64
  layer linear 'dec_04_att_value0' #: 512
  layer split_dims 'dec_05_att_key' #: 64
  layer linear 'dec_05_att_key0' #: 512
  layer split_dims 'dec_05_att_value' #: 64
  layer linear 'dec_05_att_value0' #: 512
  layer split_dims 'dec_06_att_key' #: 64
  layer linear 'dec_06_att_key0' #: 512
  layer split_dims 'dec_06_att_value' #: 64
  layer linear 'dec_06_att_value0' #: 512
  layer split_dims 'dec_07_att_key' #: 64
  layer linear 'dec_07_att_key0' #: 512
  layer split_dims 'dec_07_att_value' #: 64
  layer linear 'dec_07_att_value0' #: 512
  layer split_dims 'dec_08_att_key' #: 64
  layer linear 'dec_08_att_key0' #: 512
  layer split_dims 'dec_08_att_value' #: 64
  layer linear 'dec_08_att_value0' #: 512
  layer split_dims 'dec_09_att_key' #: 64
  layer linear 'dec_09_att_key0' #: 512
  layer split_dims 'dec_09_att_value' #: 64
  layer linear 'dec_09_att_value0' #: 512
  layer split_dims 'dec_10_att_key' #: 64
  layer linear 'dec_10_att_key0' #: 512
  layer split_dims 'dec_10_att_value' #: 64
  layer linear 'dec_10_att_value0' #: 512
  layer split_dims 'dec_11_att_key' #: 64
  layer linear 'dec_11_att_key0' #: 512
  layer split_dims 'dec_11_att_value' #: 64
  layer linear 'dec_11_att_value0' #: 512
  layer split_dims 'dec_12_att_key' #: 64
  layer linear 'dec_12_att_key0' #: 512
  layer split_dims 'dec_12_att_value' #: 64
  layer linear 'dec_12_att_value0' #: 512
  layer decide 'decision' #: 10025
  layer copy 'enc_01' #: 512
  layer linear 'enc_01_ff_conv1' #: 2048
  layer linear 'enc_01_ff_conv2' #: 512
  layer dropout 'enc_01_ff_drop' #: 512
  layer layer_norm 'enc_01_ff_laynorm' #: 512
  layer combine 'enc_01_ff_out' #: 512
  layer self_attention 'enc_01_self_att_att' #: 512
  layer dropout 'enc_01_self_att_drop' #: 512
  layer layer_norm 'enc_01_self_att_laynorm' #: 512
  layer linear 'enc_01_self_att_lin' #: 512
  layer combine 'enc_01_self_att_out' #: 512
  layer copy 'enc_02' #: 512
  layer linear 'enc_02_ff_conv1' #: 2048
  layer linear 'enc_02_ff_conv2' #: 512
  layer dropout 'enc_02_ff_drop' #: 512
  layer layer_norm 'enc_02_ff_laynorm' #: 512
  layer combine 'enc_02_ff_out' #: 512
  layer self_attention 'enc_02_self_att_att' #: 512
  layer dropout 'enc_02_self_att_drop' #: 512
  layer layer_norm 'enc_02_self_att_laynorm' #: 512
  layer linear 'enc_02_self_att_lin' #: 512
  layer combine 'enc_02_self_att_out' #: 512
  layer copy 'enc_03' #: 512
  layer linear 'enc_03_ff_conv1' #: 2048
  layer linear 'enc_03_ff_conv2' #: 512
  layer dropout 'enc_03_ff_drop' #: 512
  layer layer_norm 'enc_03_ff_laynorm' #: 512
  layer combine 'enc_03_ff_out' #: 512
  layer self_attention 'enc_03_self_att_att' #: 512
  layer dropout 'enc_03_self_att_drop' #: 512
  layer layer_norm 'enc_03_self_att_laynorm' #: 512
  layer linear 'enc_03_self_att_lin' #: 512
  layer combine 'enc_03_self_att_out' #: 512
  layer copy 'enc_04' #: 512
  layer linear 'enc_04_ff_conv1' #: 2048
  layer linear 'enc_04_ff_conv2' #: 512
  layer dropout 'enc_04_ff_drop' #: 512
  layer layer_norm 'enc_04_ff_laynorm' #: 512
  layer combine 'enc_04_ff_out' #: 512
  layer self_attention 'enc_04_self_att_att' #: 512
  layer dropout 'enc_04_self_att_drop' #: 512
  layer layer_norm 'enc_04_self_att_laynorm' #: 512
  layer linear 'enc_04_self_att_lin' #: 512
  layer combine 'enc_04_self_att_out' #: 512
  layer copy 'enc_05' #: 512
  layer linear 'enc_05_ff_conv1' #: 2048
  layer linear 'enc_05_ff_conv2' #: 512
  layer dropout 'enc_05_ff_drop' #: 512
  layer layer_norm 'enc_05_ff_laynorm' #: 512
  layer combine 'enc_05_ff_out' #: 512
  layer self_attention 'enc_05_self_att_att' #: 512
  layer dropout 'enc_05_self_att_drop' #: 512
  layer layer_norm 'enc_05_self_att_laynorm' #: 512
  layer linear 'enc_05_self_att_lin' #: 512
  layer combine 'enc_05_self_att_out' #: 512
  layer copy 'enc_06' #: 512
  layer linear 'enc_06_ff_conv1' #: 2048
  layer linear 'enc_06_ff_conv2' #: 512
  layer dropout 'enc_06_ff_drop' #: 512
  layer layer_norm 'enc_06_ff_laynorm' #: 512
  layer combine 'enc_06_ff_out' #: 512
  layer self_attention 'enc_06_self_att_att' #: 512
  layer dropout 'enc_06_self_att_drop' #: 512
  layer layer_norm 'enc_06_self_att_laynorm' #: 512
  layer linear 'enc_06_self_att_lin' #: 512
  layer combine 'enc_06_self_att_out' #: 512
  layer copy 'enc_07' #: 512
  layer linear 'enc_07_ff_conv1' #: 2048
  layer linear 'enc_07_ff_conv2' #: 512
  layer dropout 'enc_07_ff_drop' #: 512
  layer layer_norm 'enc_07_ff_laynorm' #: 512
  layer combine 'enc_07_ff_out' #: 512
  layer self_attention 'enc_07_self_att_att' #: 512
  layer dropout 'enc_07_self_att_drop' #: 512
  layer layer_norm 'enc_07_self_att_laynorm' #: 512
  layer linear 'enc_07_self_att_lin' #: 512
  layer combine 'enc_07_self_att_out' #: 512
  layer copy 'enc_08' #: 512
  layer linear 'enc_08_ff_conv1' #: 2048
  layer linear 'enc_08_ff_conv2' #: 512
  layer dropout 'enc_08_ff_drop' #: 512
  layer layer_norm 'enc_08_ff_laynorm' #: 512
  layer combine 'enc_08_ff_out' #: 512
  layer self_attention 'enc_08_self_att_att' #: 512
  layer dropout 'enc_08_self_att_drop' #: 512
  layer layer_norm 'enc_08_self_att_laynorm' #: 512
  layer linear 'enc_08_self_att_lin' #: 512
  layer combine 'enc_08_self_att_out' #: 512
  layer copy 'enc_09' #: 512
  layer linear 'enc_09_ff_conv1' #: 2048
  layer linear 'enc_09_ff_conv2' #: 512
  layer dropout 'enc_09_ff_drop' #: 512
  layer layer_norm 'enc_09_ff_laynorm' #: 512
  layer combine 'enc_09_ff_out' #: 512
  layer self_attention 'enc_09_self_att_att' #: 512
  layer dropout 'enc_09_self_att_drop' #: 512
  layer layer_norm 'enc_09_self_att_laynorm' #: 512
  layer linear 'enc_09_self_att_lin' #: 512
  layer combine 'enc_09_self_att_out' #: 512
  layer copy 'enc_10' #: 512
  layer linear 'enc_10_ff_conv1' #: 2048
  layer linear 'enc_10_ff_conv2' #: 512
  layer dropout 'enc_10_ff_drop' #: 512
  layer layer_norm 'enc_10_ff_laynorm' #: 512
  layer combine 'enc_10_ff_out' #: 512
  layer self_attention 'enc_10_self_att_att' #: 512
  layer dropout 'enc_10_self_att_drop' #: 512
  layer layer_norm 'enc_10_self_att_laynorm' #: 512
  layer linear 'enc_10_self_att_lin' #: 512
  layer combine 'enc_10_self_att_out' #: 512
  layer copy 'enc_11' #: 512
  layer linear 'enc_11_ff_conv1' #: 2048
  layer linear 'enc_11_ff_conv2' #: 512
  layer dropout 'enc_11_ff_drop' #: 512
  layer layer_norm 'enc_11_ff_laynorm' #: 512
  layer combine 'enc_11_ff_out' #: 512
  layer self_attention 'enc_11_self_att_att' #: 512
  layer dropout 'enc_11_self_att_drop' #: 512
  layer layer_norm 'enc_11_self_att_laynorm' #: 512
  layer linear 'enc_11_self_att_lin' #: 512
  layer combine 'enc_11_self_att_out' #: 512
  layer copy 'enc_12' #: 512
  layer linear 'enc_12_ff_conv1' #: 2048
  layer linear 'enc_12_ff_conv2' #: 512
  layer dropout 'enc_12_ff_drop' #: 512
  layer layer_norm 'enc_12_ff_laynorm' #: 512
  layer combine 'enc_12_ff_out' #: 512
  layer self_attention 'enc_12_self_att_att' #: 512
  layer dropout 'enc_12_self_att_drop' #: 512
  layer layer_norm 'enc_12_self_att_laynorm' #: 512
  layer linear 'enc_12_self_att_lin' #: 512
  layer combine 'enc_12_self_att_out' #: 512
  layer layer_norm 'encoder' #: 512
  layer rec 'lstm0_bw' #: 1024
  layer rec 'lstm0_fw' #: 1024
  layer pool 'lstm0_pool' #: 2048
  layer rec 'lstm1_bw' #: 1024
  layer rec 'lstm1_fw' #: 1024
  layer pool 'lstm1_pool' #: 2048
  layer rec 'output' #: 10025
  layer eval 'source' #: 40
  layer dropout 'source_embed' #: 512
  layer linear 'source_embed_raw' #: 512
  layer eval 'source_embed_weighted' #: 512
net params #: 139347754
net trainable params: [<tf.Variable 'ctc/W:0' shape=(512, 10026) dtype=float32_ref>, <tf.Variable 'ctc/b:0' shape=(10026,) dtype=float32_ref>, <tf.Variable 'dec_01_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_01_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_02_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_02_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_03_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_03_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_04_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_04_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_05_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_05_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_06_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_06_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_07_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_07_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_08_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_08_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_09_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_09_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_10_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_10_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_11_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_11_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_12_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_12_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_01_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_01_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_01_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_01_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_01_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_01_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_01_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_01_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_01_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_01_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_02_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_02_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_02_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_02_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_02_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_02_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_02_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_02_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_02_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_02_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_03_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_03_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_03_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_03_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_03_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_03_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_03_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_03_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_03_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_03_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_04_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_04_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_04_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_04_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_04_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_04_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_04_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_04_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_04_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_04_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_05_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_05_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_05_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_05_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_05_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_05_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_05_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_05_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_05_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_05_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_06_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_06_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_06_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_06_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_06_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_06_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_06_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_06_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_06_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_06_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_07_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_07_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_07_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_07_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_07_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_07_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_07_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_07_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_07_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_07_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_08_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_08_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_08_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_08_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_08_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_08_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_08_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_08_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_08_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_08_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_09_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_09_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_09_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_09_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_09_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_09_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_09_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_09_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_09_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_09_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_10_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_10_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_10_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_10_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_10_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_10_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_10_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_10_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_10_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_10_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_11_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_11_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_11_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_11_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_11_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_11_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_11_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_11_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_11_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_11_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_12_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_12_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_12_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_12_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_12_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_12_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_12_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_12_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_12_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_12_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'encoder/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'encoder/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'lstm0_bw/rec/W:0' shape=(40, 4096) dtype=float32_ref>, <tf.Variable 'lstm0_bw/rec/W_re:0' shape=(1024, 4096) dtype=float32_ref>, <tf.Variable 'lstm0_bw/rec/b:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'lstm0_fw/rec/W:0' shape=(40, 4096) dtype=float32_ref>, <tf.Variable 'lstm0_fw/rec/W_re:0' shape=(1024, 4096) dtype=float32_ref>, <tf.Variable 'lstm0_fw/rec/b:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'lstm1_bw/rec/W:0' shape=(2048, 4096) dtype=float32_ref>, <tf.Variable 'lstm1_bw/rec/W_re:0' shape=(1024, 4096) dtype=float32_ref>, <tf.Variable 'lstm1_bw/rec/b:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'lstm1_fw/rec/W:0' shape=(2048, 4096) dtype=float32_ref>, <tf.Variable 'lstm1_fw/rec/W_re:0' shape=(1024, 4096) dtype=float32_ref>, <tf.Variable 'lstm1_fw/rec/b:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/decoder/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/decoder/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/decoder_int/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/encoder_int/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/output_prob/dense/kernel:0' shape=(512, 10025) dtype=float32_ref>, <tf.Variable 'output/rec/prev_outputs_int/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/target_embed_raw/W:0' shape=(10025, 512) dtype=float32_ref>, <tf.Variable 'source_embed_raw/W:0' shape=(2048, 512) dtype=float32_ref>]
loading weights from /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3-prior-k6/net-model/network.560
Unhandled exception <class 'tensorflow.python.framework.errors_impl.PermissionDeniedError'> in thread <_MainThread(MainThread, started 47792568716288)>, proc 25556.

Thread current, main, <_MainThread(MainThread, started 47792568716288)>:
(Excluded thread.)

That were all threads.
+------- EPILOGUE SCRIPT -----------------------------------------------
|
| Job ID ..............: 9506002
| Stopped at ..........: Tue Jul  2 12:22:03 CEST 2019
| Resources requested .: h_rss=4G,h_vmem=1536G,num_proc=5,gpu=1,scratch_free=5G,h_fsize=20G,pxe=ubuntu_16.04,h_rt=7200,s_core=0
| Resources used ......: cpu=00:01:21, mem=52.74176 GB s, io=1.52469 GB, vmem=1.533G, maxvmem=1.807G, last_file_cache=102M, last_rss=2M, max-cache=1.458G
| Memory used .........: 1.558G / 4.000G (38.9%)
| Total time used .....: 0:01:44
|
+------- EPILOGUE SCRIPT -----------------------------------------------
+------- PROLOGUE SCRIPT -----------------------------------------------
|
| Job ID ...........: 9506015
| Started at .......: Tue Jul  2 12:24:18 CEST 2019
| Execution host ...: cluster-cn-244
| Cluster queue ....: 4-GPU-1080
| Script ...........: /var/spool/sge/cluster-cn-244/job_scripts/9506015
| > #!/bin/bash
| > 
| > export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64:/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda/extras/CUPTI/lib64/:/usr/local/cuda-9.1/extras/CUPTI/lib64/:/usr/local/cuda-9.0/extras/CUPTI/lib64:/usr/lib/nvidia-418
| > #export LD_LIBRARY_PATH="/usr/local/lib::/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda-9.0/lib64:/usr/lib/nvidia-390:/opt/rbi/sge/lib/glinux:/opt/rbi/sge/lib/glinux"
| > 
| > source /u/bahar/settings/python3-returnn-tf1.9/bin/activate
| > 
| > python3 ~/returnn-parnia/tools/get-attention-weights.py /u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3-prior-k6.config --epoch 560 --data 'config:get_dataset("'$1'")' --dump_dir ~/makarov/NMT/hmm-factorization/experiments/asr2019/transformer-prior/forward-$1 --layers "dec_02_att_weights" --layers "dec_01_att_weights" --layers "dec_03_att_weights" --layers "dec_04_att_weights" --layers "dec_05_att_weights" --layers "dec_06_att_weights" --layers "dec_07_att_weights" --layers "dec_08_att_weights" --layers "dec_09_att_weights" --layers "dec_10_att_weights" --layers "dec_11_att_weights" --layers "dec_12_att_weights" --rec_layer "output" --batch_size 4000 --tf_log_dir /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/transformer-prior/forward-$1/tf_log_dir --reset_seq_ordering sorted_reverse
|
+------- PROLOGUE SCRIPT -----------------------------------------------
RETURNN get-attention-weights starting up.
RETURNN starting up, version 20181022.053313--git-dfada43-dirty, date/time 2019-07-02-12-24-21 (UTC+0200), pid 28051, cwd /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3-prior-k6, Python /u/bahar/settings/python3-returnn-tf1.9/bin/python3
RETURNN config: /u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3-prior-k6.config
RETURNN command line options: ()
Hostname: cluster-cn-244
TensorFlow: 1.10.0 (v1.10.0-0-g656e7a2b34) (<site-package> in /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow)
Error while getting SGE num_proc: FileNotFoundError(2, "No such file or directory: 'qstat'")
Setup TF inter and intra global thread pools, num_threads None, session opts {'device_count': {'GPU': 0}, 'log_device_placement': False}.
CUDA_VISIBLE_DEVICES is set to '1'.
Collecting TensorFlow device list...
Local devices available to TensorFlow:
  1/2: name: "/device:CPU:0"
       device_type: "CPU"
       memory_limit: 268435456
       locality {
       }
       incarnation: 4683561085477603244
  2/2: name: "/device:GPU:0"
       device_type: "GPU"
       memory_limit: 10911236096
       locality {
         bus_id: 1
         links {
         }
       }
       incarnation: 14008534111316914888
       physical_device_desc: "device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1"
Using gpu device 1: GeForce GTX 1080 Ti
Device not set explicitly, and we found a GPU, which we will use.
Model file prefix: /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3-prior-k6/net-model/network
NOTE: We will use 'sorted_reverse' seq ordering.
Setup tf.Session with options {'device_count': {'GPU': 1}, 'log_device_placement': False} ...
layer root/'data' output: Data(name='data', shape=(None, 40))
layer root/'source' output: Data(name='source_output', shape=(None, 40))
layer root/'lstm0_fw' output: Data(name='lstm0_fw_output', shape=(None, 1024), batch_dim_axis=1)
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9506015.1.4-GPU-1080/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.cc -o /var/tmp/9506015.1.4-GPU-1080/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9506015.1.4-GPU-1080/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.cc -o /var/tmp/9506015.1.4-GPU-1080/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
layer root/'lstm0_bw' output: Data(name='lstm0_bw_output', shape=(None, 1024), batch_dim_axis=1)
layer root/'lstm0_pool' output: Data(name='lstm0_pool_output', shape=(None, 2048))
layer root/'lstm1_fw' output: Data(name='lstm1_fw_output', shape=(None, 1024), batch_dim_axis=1)
layer root/'lstm1_bw' output: Data(name='lstm1_bw_output', shape=(None, 1024), batch_dim_axis=1)
layer root/'lstm1_pool' output: Data(name='lstm1_pool_output', shape=(None, 2048))
layer root/'source_embed_raw' output: Data(name='source_embed_raw_output', shape=(None, 512))
layer root/'source_embed_weighted' output: Data(name='source_embed_weighted_output', shape=(None, 512))
layer root/'source_embed' output: Data(name='source_embed_output', shape=(None, 512))
layer root/'enc_01_self_att_laynorm' output: Data(name='enc_01_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_01_self_att_att' output: Data(name='enc_01_self_att_att_output', shape=(None, 512))
layer root/'enc_01_self_att_lin' output: Data(name='enc_01_self_att_lin_output', shape=(None, 512))
layer root/'enc_01_self_att_drop' output: Data(name='enc_01_self_att_drop_output', shape=(None, 512))
layer root/'enc_01_self_att_out' output: Data(name='enc_01_self_att_out_output', shape=(None, 512))
layer root/'enc_01_ff_laynorm' output: Data(name='enc_01_ff_laynorm_output', shape=(None, 512))
layer root/'enc_01_ff_conv1' output: Data(name='enc_01_ff_conv1_output', shape=(None, 2048))
layer root/'enc_01_ff_conv2' output: Data(name='enc_01_ff_conv2_output', shape=(None, 512))
layer root/'enc_01_ff_drop' output: Data(name='enc_01_ff_drop_output', shape=(None, 512))
layer root/'enc_01_ff_out' output: Data(name='enc_01_ff_out_output', shape=(None, 512))
layer root/'enc_01' output: Data(name='enc_01_output', shape=(None, 512))
layer root/'enc_02_self_att_laynorm' output: Data(name='enc_02_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_02_self_att_att' output: Data(name='enc_02_self_att_att_output', shape=(None, 512))
layer root/'enc_02_self_att_lin' output: Data(name='enc_02_self_att_lin_output', shape=(None, 512))
layer root/'enc_02_self_att_drop' output: Data(name='enc_02_self_att_drop_output', shape=(None, 512))
layer root/'enc_02_self_att_out' output: Data(name='enc_02_self_att_out_output', shape=(None, 512))
layer root/'enc_02_ff_laynorm' output: Data(name='enc_02_ff_laynorm_output', shape=(None, 512))
layer root/'enc_02_ff_conv1' output: Data(name='enc_02_ff_conv1_output', shape=(None, 2048))
layer root/'enc_02_ff_conv2' output: Data(name='enc_02_ff_conv2_output', shape=(None, 512))
layer root/'enc_02_ff_drop' output: Data(name='enc_02_ff_drop_output', shape=(None, 512))
layer root/'enc_02_ff_out' output: Data(name='enc_02_ff_out_output', shape=(None, 512))
layer root/'enc_02' output: Data(name='enc_02_output', shape=(None, 512))
layer root/'enc_03_self_att_laynorm' output: Data(name='enc_03_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_03_self_att_att' output: Data(name='enc_03_self_att_att_output', shape=(None, 512))
layer root/'enc_03_self_att_lin' output: Data(name='enc_03_self_att_lin_output', shape=(None, 512))
layer root/'enc_03_self_att_drop' output: Data(name='enc_03_self_att_drop_output', shape=(None, 512))
layer root/'enc_03_self_att_out' output: Data(name='enc_03_self_att_out_output', shape=(None, 512))
layer root/'enc_03_ff_laynorm' output: Data(name='enc_03_ff_laynorm_output', shape=(None, 512))
layer root/'enc_03_ff_conv1' output: Data(name='enc_03_ff_conv1_output', shape=(None, 2048))
layer root/'enc_03_ff_conv2' output: Data(name='enc_03_ff_conv2_output', shape=(None, 512))
layer root/'enc_03_ff_drop' output: Data(name='enc_03_ff_drop_output', shape=(None, 512))
layer root/'enc_03_ff_out' output: Data(name='enc_03_ff_out_output', shape=(None, 512))
layer root/'enc_03' output: Data(name='enc_03_output', shape=(None, 512))
layer root/'enc_04_self_att_laynorm' output: Data(name='enc_04_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_04_self_att_att' output: Data(name='enc_04_self_att_att_output', shape=(None, 512))
layer root/'enc_04_self_att_lin' output: Data(name='enc_04_self_att_lin_output', shape=(None, 512))
layer root/'enc_04_self_att_drop' output: Data(name='enc_04_self_att_drop_output', shape=(None, 512))
layer root/'enc_04_self_att_out' output: Data(name='enc_04_self_att_out_output', shape=(None, 512))
layer root/'enc_04_ff_laynorm' output: Data(name='enc_04_ff_laynorm_output', shape=(None, 512))
layer root/'enc_04_ff_conv1' output: Data(name='enc_04_ff_conv1_output', shape=(None, 2048))
layer root/'enc_04_ff_conv2' output: Data(name='enc_04_ff_conv2_output', shape=(None, 512))
layer root/'enc_04_ff_drop' output: Data(name='enc_04_ff_drop_output', shape=(None, 512))
layer root/'enc_04_ff_out' output: Data(name='enc_04_ff_out_output', shape=(None, 512))
layer root/'enc_04' output: Data(name='enc_04_output', shape=(None, 512))
layer root/'enc_05_self_att_laynorm' output: Data(name='enc_05_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_05_self_att_att' output: Data(name='enc_05_self_att_att_output', shape=(None, 512))
layer root/'enc_05_self_att_lin' output: Data(name='enc_05_self_att_lin_output', shape=(None, 512))
layer root/'enc_05_self_att_drop' output: Data(name='enc_05_self_att_drop_output', shape=(None, 512))
layer root/'enc_05_self_att_out' output: Data(name='enc_05_self_att_out_output', shape=(None, 512))
layer root/'enc_05_ff_laynorm' output: Data(name='enc_05_ff_laynorm_output', shape=(None, 512))
layer root/'enc_05_ff_conv1' output: Data(name='enc_05_ff_conv1_output', shape=(None, 2048))
layer root/'enc_05_ff_conv2' output: Data(name='enc_05_ff_conv2_output', shape=(None, 512))
layer root/'enc_05_ff_drop' output: Data(name='enc_05_ff_drop_output', shape=(None, 512))
layer root/'enc_05_ff_out' output: Data(name='enc_05_ff_out_output', shape=(None, 512))
layer root/'enc_05' output: Data(name='enc_05_output', shape=(None, 512))
layer root/'enc_06_self_att_laynorm' output: Data(name='enc_06_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_06_self_att_att' output: Data(name='enc_06_self_att_att_output', shape=(None, 512))
layer root/'enc_06_self_att_lin' output: Data(name='enc_06_self_att_lin_output', shape=(None, 512))
layer root/'enc_06_self_att_drop' output: Data(name='enc_06_self_att_drop_output', shape=(None, 512))
layer root/'enc_06_self_att_out' output: Data(name='enc_06_self_att_out_output', shape=(None, 512))
layer root/'enc_06_ff_laynorm' output: Data(name='enc_06_ff_laynorm_output', shape=(None, 512))
layer root/'enc_06_ff_conv1' output: Data(name='enc_06_ff_conv1_output', shape=(None, 2048))
layer root/'enc_06_ff_conv2' output: Data(name='enc_06_ff_conv2_output', shape=(None, 512))
layer root/'enc_06_ff_drop' output: Data(name='enc_06_ff_drop_output', shape=(None, 512))
layer root/'enc_06_ff_out' output: Data(name='enc_06_ff_out_output', shape=(None, 512))
layer root/'enc_06' output: Data(name='enc_06_output', shape=(None, 512))
layer root/'enc_07_self_att_laynorm' output: Data(name='enc_07_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_07_self_att_att' output: Data(name='enc_07_self_att_att_output', shape=(None, 512))
layer root/'enc_07_self_att_lin' output: Data(name='enc_07_self_att_lin_output', shape=(None, 512))
layer root/'enc_07_self_att_drop' output: Data(name='enc_07_self_att_drop_output', shape=(None, 512))
layer root/'enc_07_self_att_out' output: Data(name='enc_07_self_att_out_output', shape=(None, 512))
layer root/'enc_07_ff_laynorm' output: Data(name='enc_07_ff_laynorm_output', shape=(None, 512))
layer root/'enc_07_ff_conv1' output: Data(name='enc_07_ff_conv1_output', shape=(None, 2048))
layer root/'enc_07_ff_conv2' output: Data(name='enc_07_ff_conv2_output', shape=(None, 512))
layer root/'enc_07_ff_drop' output: Data(name='enc_07_ff_drop_output', shape=(None, 512))
layer root/'enc_07_ff_out' output: Data(name='enc_07_ff_out_output', shape=(None, 512))
layer root/'enc_07' output: Data(name='enc_07_output', shape=(None, 512))
layer root/'enc_08_self_att_laynorm' output: Data(name='enc_08_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_08_self_att_att' output: Data(name='enc_08_self_att_att_output', shape=(None, 512))
layer root/'enc_08_self_att_lin' output: Data(name='enc_08_self_att_lin_output', shape=(None, 512))
layer root/'enc_08_self_att_drop' output: Data(name='enc_08_self_att_drop_output', shape=(None, 512))
layer root/'enc_08_self_att_out' output: Data(name='enc_08_self_att_out_output', shape=(None, 512))
layer root/'enc_08_ff_laynorm' output: Data(name='enc_08_ff_laynorm_output', shape=(None, 512))
layer root/'enc_08_ff_conv1' output: Data(name='enc_08_ff_conv1_output', shape=(None, 2048))
layer root/'enc_08_ff_conv2' output: Data(name='enc_08_ff_conv2_output', shape=(None, 512))
layer root/'enc_08_ff_drop' output: Data(name='enc_08_ff_drop_output', shape=(None, 512))
layer root/'enc_08_ff_out' output: Data(name='enc_08_ff_out_output', shape=(None, 512))
layer root/'enc_08' output: Data(name='enc_08_output', shape=(None, 512))
layer root/'enc_09_self_att_laynorm' output: Data(name='enc_09_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_09_self_att_att' output: Data(name='enc_09_self_att_att_output', shape=(None, 512))
layer root/'enc_09_self_att_lin' output: Data(name='enc_09_self_att_lin_output', shape=(None, 512))
layer root/'enc_09_self_att_drop' output: Data(name='enc_09_self_att_drop_output', shape=(None, 512))
layer root/'enc_09_self_att_out' output: Data(name='enc_09_self_att_out_output', shape=(None, 512))
layer root/'enc_09_ff_laynorm' output: Data(name='enc_09_ff_laynorm_output', shape=(None, 512))
layer root/'enc_09_ff_conv1' output: Data(name='enc_09_ff_conv1_output', shape=(None, 2048))
layer root/'enc_09_ff_conv2' output: Data(name='enc_09_ff_conv2_output', shape=(None, 512))
layer root/'enc_09_ff_drop' output: Data(name='enc_09_ff_drop_output', shape=(None, 512))
layer root/'enc_09_ff_out' output: Data(name='enc_09_ff_out_output', shape=(None, 512))
layer root/'enc_09' output: Data(name='enc_09_output', shape=(None, 512))
layer root/'enc_10_self_att_laynorm' output: Data(name='enc_10_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_10_self_att_att' output: Data(name='enc_10_self_att_att_output', shape=(None, 512))
layer root/'enc_10_self_att_lin' output: Data(name='enc_10_self_att_lin_output', shape=(None, 512))
layer root/'enc_10_self_att_drop' output: Data(name='enc_10_self_att_drop_output', shape=(None, 512))
layer root/'enc_10_self_att_out' output: Data(name='enc_10_self_att_out_output', shape=(None, 512))
layer root/'enc_10_ff_laynorm' output: Data(name='enc_10_ff_laynorm_output', shape=(None, 512))
layer root/'enc_10_ff_conv1' output: Data(name='enc_10_ff_conv1_output', shape=(None, 2048))
layer root/'enc_10_ff_conv2' output: Data(name='enc_10_ff_conv2_output', shape=(None, 512))
layer root/'enc_10_ff_drop' output: Data(name='enc_10_ff_drop_output', shape=(None, 512))
layer root/'enc_10_ff_out' output: Data(name='enc_10_ff_out_output', shape=(None, 512))
layer root/'enc_10' output: Data(name='enc_10_output', shape=(None, 512))
layer root/'enc_11_self_att_laynorm' output: Data(name='enc_11_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_11_self_att_att' output: Data(name='enc_11_self_att_att_output', shape=(None, 512))
layer root/'enc_11_self_att_lin' output: Data(name='enc_11_self_att_lin_output', shape=(None, 512))
layer root/'enc_11_self_att_drop' output: Data(name='enc_11_self_att_drop_output', shape=(None, 512))
layer root/'enc_11_self_att_out' output: Data(name='enc_11_self_att_out_output', shape=(None, 512))
layer root/'enc_11_ff_laynorm' output: Data(name='enc_11_ff_laynorm_output', shape=(None, 512))
layer root/'enc_11_ff_conv1' output: Data(name='enc_11_ff_conv1_output', shape=(None, 2048))
layer root/'enc_11_ff_conv2' output: Data(name='enc_11_ff_conv2_output', shape=(None, 512))
layer root/'enc_11_ff_drop' output: Data(name='enc_11_ff_drop_output', shape=(None, 512))
layer root/'enc_11_ff_out' output: Data(name='enc_11_ff_out_output', shape=(None, 512))
layer root/'enc_11' output: Data(name='enc_11_output', shape=(None, 512))
layer root/'enc_12_self_att_laynorm' output: Data(name='enc_12_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_12_self_att_att' output: Data(name='enc_12_self_att_att_output', shape=(None, 512))
layer root/'enc_12_self_att_lin' output: Data(name='enc_12_self_att_lin_output', shape=(None, 512))
layer root/'enc_12_self_att_drop' output: Data(name='enc_12_self_att_drop_output', shape=(None, 512))
layer root/'enc_12_self_att_out' output: Data(name='enc_12_self_att_out_output', shape=(None, 512))
layer root/'enc_12_ff_laynorm' output: Data(name='enc_12_ff_laynorm_output', shape=(None, 512))
layer root/'enc_12_ff_conv1' output: Data(name='enc_12_ff_conv1_output', shape=(None, 2048))
layer root/'enc_12_ff_conv2' output: Data(name='enc_12_ff_conv2_output', shape=(None, 512))
layer root/'enc_12_ff_drop' output: Data(name='enc_12_ff_drop_output', shape=(None, 512))
layer root/'enc_12_ff_out' output: Data(name='enc_12_ff_out_output', shape=(None, 512))
layer root/'enc_12' output: Data(name='enc_12_output', shape=(None, 512))
layer root/'encoder' output: Data(name='encoder_output', shape=(None, 512))
layer root/'ctc' output: Data(name='ctc_output', shape=(None, 10026))
layer root/'dec_02_att_key0' output: Data(name='dec_02_att_key0_output', shape=(None, 512))
layer root/'dec_02_att_key' output: Data(name='dec_02_att_key_output', shape=(None, 8, 64))
layer root/'dec_03_att_key0' output: Data(name='dec_03_att_key0_output', shape=(None, 512))
layer root/'dec_03_att_key' output: Data(name='dec_03_att_key_output', shape=(None, 8, 64))
layer root/'dec_10_att_key0' output: Data(name='dec_10_att_key0_output', shape=(None, 512))
layer root/'dec_10_att_key' output: Data(name='dec_10_att_key_output', shape=(None, 8, 64))
layer root/'dec_06_att_key0' output: Data(name='dec_06_att_key0_output', shape=(None, 512))
layer root/'dec_06_att_key' output: Data(name='dec_06_att_key_output', shape=(None, 8, 64))
layer root/'dec_01_att_value0' output: Data(name='dec_01_att_value0_output', shape=(None, 512))
layer root/'dec_01_att_value' output: Data(name='dec_01_att_value_output', shape=(None, 8, 64))
layer root/'dec_08_att_value0' output: Data(name='dec_08_att_value0_output', shape=(None, 512))
layer root/'dec_08_att_value' output: Data(name='dec_08_att_value_output', shape=(None, 8, 64))
layer root/'dec_09_att_value0' output: Data(name='dec_09_att_value0_output', shape=(None, 512))
layer root/'dec_09_att_value' output: Data(name='dec_09_att_value_output', shape=(None, 8, 64))
layer root/'dec_07_att_value0' output: Data(name='dec_07_att_value0_output', shape=(None, 512))
layer root/'dec_07_att_value' output: Data(name='dec_07_att_value_output', shape=(None, 8, 64))
layer root/'dec_09_att_key0' output: Data(name='dec_09_att_key0_output', shape=(None, 512))
layer root/'dec_09_att_key' output: Data(name='dec_09_att_key_output', shape=(None, 8, 64))
layer root/'dec_03_att_value0' output: Data(name='dec_03_att_value0_output', shape=(None, 512))
layer root/'dec_03_att_value' output: Data(name='dec_03_att_value_output', shape=(None, 8, 64))
layer root/'dec_05_att_value0' output: Data(name='dec_05_att_value0_output', shape=(None, 512))
layer root/'dec_05_att_value' output: Data(name='dec_05_att_value_output', shape=(None, 8, 64))
layer root/'dec_05_att_key0' output: Data(name='dec_05_att_key0_output', shape=(None, 512))
layer root/'dec_05_att_key' output: Data(name='dec_05_att_key_output', shape=(None, 8, 64))
layer root/'dec_02_att_value0' output: Data(name='dec_02_att_value0_output', shape=(None, 512))
layer root/'dec_02_att_value' output: Data(name='dec_02_att_value_output', shape=(None, 8, 64))
layer root/'dec_07_att_key0' output: Data(name='dec_07_att_key0_output', shape=(None, 512))
layer root/'dec_07_att_key' output: Data(name='dec_07_att_key_output', shape=(None, 8, 64))
layer root/'dec_04_att_value0' output: Data(name='dec_04_att_value0_output', shape=(None, 512))
layer root/'dec_04_att_value' output: Data(name='dec_04_att_value_output', shape=(None, 8, 64))
layer root/'dec_10_att_value0' output: Data(name='dec_10_att_value0_output', shape=(None, 512))
layer root/'dec_10_att_value' output: Data(name='dec_10_att_value_output', shape=(None, 8, 64))
layer root/'dec_11_att_key0' output: Data(name='dec_11_att_key0_output', shape=(None, 512))
layer root/'dec_11_att_key' output: Data(name='dec_11_att_key_output', shape=(None, 8, 64))
layer root/'dec_12_att_value0' output: Data(name='dec_12_att_value0_output', shape=(None, 512))
layer root/'dec_12_att_value' output: Data(name='dec_12_att_value_output', shape=(None, 8, 64))
layer root/'dec_08_att_key0' output: Data(name='dec_08_att_key0_output', shape=(None, 512))
layer root/'dec_08_att_key' output: Data(name='dec_08_att_key_output', shape=(None, 8, 64))
layer root/'dec_11_att_value0' output: Data(name='dec_11_att_value0_output', shape=(None, 512))
layer root/'dec_11_att_value' output: Data(name='dec_11_att_value_output', shape=(None, 8, 64))
layer root/'dec_06_att_value0' output: Data(name='dec_06_att_value0_output', shape=(None, 512))
layer root/'dec_06_att_value' output: Data(name='dec_06_att_value_output', shape=(None, 8, 64))
layer root/'dec_04_att_key0' output: Data(name='dec_04_att_key0_output', shape=(None, 512))
layer root/'dec_04_att_key' output: Data(name='dec_04_att_key_output', shape=(None, 8, 64))
layer root/'dec_01_att_key0' output: Data(name='dec_01_att_key0_output', shape=(None, 512))
layer root/'dec_01_att_key' output: Data(name='dec_01_att_key_output', shape=(None, 8, 64))
layer root/'dec_12_att_key0' output: Data(name='dec_12_att_key0_output', shape=(None, 512))
layer root/'dec_12_att_key' output: Data(name='dec_12_att_key_output', shape=(None, 8, 64))
layer root/'output' output: Data(name='output_output', shape=(None,), dtype='int32', sparse=True, dim=10025, batch_dim_axis=1)
Rec layer sub net:
  Input layers moved out of loop: (#: 0)
    None
  Output layers moved out of loop: (#: 273)
    output_prob
    decoder_int
    decoder
    dec_12
    dec_12_ff_out
    dec_12_ff_drop
    dec_12_ff_conv2
    dec_12_ff_conv1
    dec_12_ff_laynorm
    dec_12_att_out
    dec_12_att_drop
    dec_12_att_lin
    dec_12_att_att
    dec_12_att0
    dec_12_att_weights_drop
    dec_12_att_weights
    dec_12_att_energy
    dec_12_att_query
    dec_12_att_query0
    dec_12_att_laynorm
    dec_12_self_att_out
    dec_12_self_att_drop
    dec_12_self_att_lin
    dec_12_self_att_att
    dec_12_self_att_laynorm
    dec_11
    dec_11_ff_out
    dec_11_ff_drop
    dec_11_ff_conv2
    dec_11_ff_conv1
    dec_11_ff_laynorm
    dec_11_att_out
    dec_11_att_drop
    dec_11_att_lin
    dec_11_att_att
    dec_11_att0
    dec_11_att_weights_drop
    dec_11_att_weights
    dec_11_att_energy
    dec_11_att_query
    dec_11_att_query0
    dec_11_att_laynorm
    dec_11_self_att_out
    dec_11_self_att_drop
    dec_11_self_att_lin
    dec_11_self_att_att
    dec_11_self_att_laynorm
    dec_10
    dec_10_ff_out
    dec_10_ff_drop
    dec_10_ff_conv2
    dec_10_ff_conv1
    dec_10_ff_laynorm
    dec_10_att_out
    dec_10_att_drop
    dec_10_att_lin
    dec_10_att_att
    dec_10_att0
    dec_10_att_weights_drop
    dec_10_att_weights
    dec_10_att_energy
    dec_10_att_query
    dec_10_att_query0
    dec_10_att_laynorm
    dec_10_self_att_out
    dec_10_self_att_drop
    dec_10_self_att_lin
    dec_10_self_att_att
    dec_10_self_att_laynorm
    dec_09
    dec_09_ff_out
    dec_09_ff_drop
    dec_09_ff_conv2
    dec_09_ff_conv1
    dec_09_ff_laynorm
    dec_09_att_out
    dec_09_att_drop
    dec_09_att_lin
    dec_09_att_att
    dec_09_att0
    dec_09_att_weights_drop
    dec_09_att_weights
    dec_09_att_energy
    dec_09_att_query
    dec_09_att_query0
    dec_09_att_laynorm
    dec_09_self_att_out
    dec_09_self_att_drop
    dec_09_self_att_lin
    dec_09_self_att_att
    dec_09_self_att_laynorm
    dec_08
    dec_08_ff_out
    dec_08_ff_drop
    dec_08_ff_conv2
    dec_08_ff_conv1
    dec_08_ff_laynorm
    dec_08_att_out
    dec_08_att_drop
    dec_08_att_lin
    dec_08_att_att
    dec_08_att0
    dec_08_att_weights_drop
    dec_08_att_weights
    dec_08_att_energy
    dec_08_att_query
    dec_08_att_query0
    dec_08_att_laynorm
    dec_08_self_att_out
    dec_08_self_att_drop
    dec_08_self_att_lin
    dec_08_self_att_att
    dec_08_self_att_laynorm
    dec_07
    dec_07_ff_out
    dec_07_ff_drop
    dec_07_ff_conv2
    dec_07_ff_conv1
    dec_07_ff_laynorm
    dec_07_att_out
    dec_07_att_drop
    dec_07_att_lin
    dec_07_att_att
    dec_07_att0
    dec_07_att_weights_drop
    dec_07_att_weights
    dec_07_att_energy
    dec_07_att_query
    dec_07_att_query0
    dec_07_att_laynorm
    dec_07_self_att_out
    dec_07_self_att_drop
    dec_07_self_att_lin
    dec_07_self_att_att
    dec_07_self_att_laynorm
    dec_06
    dec_06_ff_out
    dec_06_ff_drop
    dec_06_ff_conv2
    dec_06_ff_conv1
    dec_06_ff_laynorm
    dec_06_att_out
    dec_06_att_drop
    dec_06_att_lin
    dec_06_att_att
    dec_06_att0
    dec_06_att_weights_drop
    dec_06_att_weights
    dec_06_att_energy
    dec_06_att_query
    dec_06_att_query0
    dec_06_att_laynorm
    dec_06_self_att_out
    dec_06_self_att_drop
    dec_06_self_att_lin
    dec_06_self_att_att
    dec_06_self_att_laynorm
    dec_05
    dec_05_ff_out
    dec_05_ff_drop
    dec_05_ff_conv2
    dec_05_ff_conv1
    dec_05_ff_laynorm
    dec_05_att_out
    dec_05_att_drop
    dec_05_att_lin
    dec_05_att_att
    dec_05_att0
    dec_05_att_weights_drop
    dec_05_att_weights
    dec_05_att_energy
    dec_05_att_query
    dec_05_att_query0
    dec_05_att_laynorm
    dec_05_self_att_out
    dec_05_self_att_drop
    dec_05_self_att_lin
    dec_05_self_att_att
    dec_05_self_att_laynorm
    dec_04
    dec_04_ff_out
    dec_04_ff_drop
    dec_04_ff_conv2
    dec_04_ff_conv1
    dec_04_ff_laynorm
    dec_04_att_out
    dec_04_att_drop
    dec_04_att_lin
    dec_04_att_att
    dec_04_att0
    dec_04_att_weights_drop
    dec_04_att_weights
    dec_04_att_energy
    dec_04_att_query
    dec_04_att_query0
    dec_04_att_laynorm
    dec_04_self_att_out
    dec_04_self_att_drop
    dec_04_self_att_lin
    dec_04_self_att_att
    dec_04_self_att_laynorm
    dec_03
    dec_03_ff_out
    dec_03_ff_drop
    dec_03_ff_conv2
    dec_03_ff_conv1
    dec_03_ff_laynorm
    dec_03_att_out
    dec_03_att_drop
    dec_03_att_lin
    dec_03_att_att
    dec_03_att0
    dec_03_att_weights_drop
    dec_03_att_weights
    dec_03_att_energy
    dec_03_att_query
    dec_03_att_query0
    dec_03_att_laynorm
    dec_03_self_att_out
    dec_03_self_att_drop
    dec_03_self_att_lin
    dec_03_self_att_att
    dec_03_self_att_laynorm
    dec_02
    dec_02_ff_out
    dec_02_ff_drop
    dec_02_ff_conv2
    dec_02_ff_conv1
    dec_02_ff_laynorm
    dec_02_att_out
    dec_02_att_drop
    dec_02_att_lin
    dec_02_att_att
    dec_02_att0
    dec_02_att_weights_drop
    dec_02_att_weights
    dec_02_att_energy
    dec_02_att_query
    dec_02_att_query0
    dec_02_att_laynorm
    dec_02_self_att_out
    dec_02_self_att_drop
    dec_02_self_att_lin
    dec_02_self_att_att
    dec_02_self_att_laynorm
    dec_01
    dec_01_ff_out
    dec_01_ff_drop
    dec_01_ff_conv2
    dec_01_ff_conv1
    dec_01_ff_laynorm
    dec_01_att_out
    dec_01_att_drop
    dec_01_att_lin
    dec_01_att_att
    dec_01_att0
    dec_01_att_weights_drop
    dec_01_att_weights
    dec_01_att_energy
    dec_01_att_query
    dec_01_att_query0
    dec_01_att_laynorm
    dec_01_self_att_out
    dec_01_self_att_drop
    dec_01_self_att_lin
    dec_01_self_att_att
    dec_01_self_att_laynorm
    target_embed
    target_embed_weighted
    encoder_int
    prev_outputs_int
    target_embed_raw
    output
  Layers in loop: (#: 0)
    None
  Unused layers: (#: 1)
    end
layer root/output:rec-subnet-output/'output' output: Data(name='classes', shape=(None,), dtype='int32', sparse=True, dim=10025)
layer root/output:rec-subnet-output/'prev:output' output: Data(name='classes', shape=(None,), dtype='int32', sparse=True, dim=10025, batch_dim_axis=1)
layer root/output:rec-subnet-output/'target_embed_raw' output: Data(name='target_embed_raw_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'target_embed_weighted' output: Data(name='target_embed_weighted_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'target_embed' output: Data(name='target_embed_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_self_att_laynorm' output: Data(name='dec_01_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_self_att_att' output: Data(name='dec_01_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_01_self_att_lin' output: Data(name='dec_01_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_01_self_att_drop' output: Data(name='dec_01_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_01_self_att_out' output: Data(name='dec_01_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_att_laynorm' output: Data(name='dec_01_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_att_query0' output: Data(name='dec_01_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_att_query' output: Data(name='dec_01_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_att_energy' output: Data(name='dec_01_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_01_att_weights' output: Data(name='dec_01_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_01_att_weights_drop' output: Data(name='dec_01_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_01_att0' output: Data(name='dec_01_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_01_att_att' output: Data(name='dec_01_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_01_att_lin' output: Data(name='dec_01_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_01_att_drop' output: Data(name='dec_01_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_01_att_out' output: Data(name='dec_01_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_ff_laynorm' output: Data(name='dec_01_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_ff_conv1' output: Data(name='dec_01_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_ff_conv2' output: Data(name='dec_01_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_ff_drop' output: Data(name='dec_01_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_ff_out' output: Data(name='dec_01_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01' output: Data(name='dec_01_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_self_att_laynorm' output: Data(name='dec_02_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_self_att_att' output: Data(name='dec_02_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_02_self_att_lin' output: Data(name='dec_02_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_02_self_att_drop' output: Data(name='dec_02_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_02_self_att_out' output: Data(name='dec_02_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_att_laynorm' output: Data(name='dec_02_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_att_query0' output: Data(name='dec_02_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_att_query' output: Data(name='dec_02_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_att_energy' output: Data(name='dec_02_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_02_att_weights' output: Data(name='dec_02_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_02_att_weights_drop' output: Data(name='dec_02_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_02_att0' output: Data(name='dec_02_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_02_att_att' output: Data(name='dec_02_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_02_att_lin' output: Data(name='dec_02_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_02_att_drop' output: Data(name='dec_02_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_02_att_out' output: Data(name='dec_02_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_ff_laynorm' output: Data(name='dec_02_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_ff_conv1' output: Data(name='dec_02_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_ff_conv2' output: Data(name='dec_02_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_ff_drop' output: Data(name='dec_02_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_ff_out' output: Data(name='dec_02_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02' output: Data(name='dec_02_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_self_att_laynorm' output: Data(name='dec_03_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_self_att_att' output: Data(name='dec_03_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_03_self_att_lin' output: Data(name='dec_03_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_03_self_att_drop' output: Data(name='dec_03_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_03_self_att_out' output: Data(name='dec_03_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_att_laynorm' output: Data(name='dec_03_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_att_query0' output: Data(name='dec_03_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_att_query' output: Data(name='dec_03_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_att_energy' output: Data(name='dec_03_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_03_att_weights' output: Data(name='dec_03_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_03_att_weights_drop' output: Data(name='dec_03_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_03_att0' output: Data(name='dec_03_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_03_att_att' output: Data(name='dec_03_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_03_att_lin' output: Data(name='dec_03_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_03_att_drop' output: Data(name='dec_03_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_03_att_out' output: Data(name='dec_03_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_ff_laynorm' output: Data(name='dec_03_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_ff_conv1' output: Data(name='dec_03_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_ff_conv2' output: Data(name='dec_03_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_ff_drop' output: Data(name='dec_03_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_ff_out' output: Data(name='dec_03_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03' output: Data(name='dec_03_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_self_att_laynorm' output: Data(name='dec_04_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_self_att_att' output: Data(name='dec_04_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_04_self_att_lin' output: Data(name='dec_04_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_04_self_att_drop' output: Data(name='dec_04_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_04_self_att_out' output: Data(name='dec_04_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_att_laynorm' output: Data(name='dec_04_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_att_query0' output: Data(name='dec_04_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_att_query' output: Data(name='dec_04_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_att_energy' output: Data(name='dec_04_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_04_att_weights' output: Data(name='dec_04_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_04_att_weights_drop' output: Data(name='dec_04_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_04_att0' output: Data(name='dec_04_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_04_att_att' output: Data(name='dec_04_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_04_att_lin' output: Data(name='dec_04_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_04_att_drop' output: Data(name='dec_04_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_04_att_out' output: Data(name='dec_04_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_ff_laynorm' output: Data(name='dec_04_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_ff_conv1' output: Data(name='dec_04_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_ff_conv2' output: Data(name='dec_04_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_ff_drop' output: Data(name='dec_04_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_ff_out' output: Data(name='dec_04_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04' output: Data(name='dec_04_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_self_att_laynorm' output: Data(name='dec_05_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_self_att_att' output: Data(name='dec_05_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_05_self_att_lin' output: Data(name='dec_05_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_05_self_att_drop' output: Data(name='dec_05_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_05_self_att_out' output: Data(name='dec_05_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_att_laynorm' output: Data(name='dec_05_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_att_query0' output: Data(name='dec_05_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_att_query' output: Data(name='dec_05_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_att_energy' output: Data(name='dec_05_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_05_att_weights' output: Data(name='dec_05_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_05_att_weights_drop' output: Data(name='dec_05_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_05_att0' output: Data(name='dec_05_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_05_att_att' output: Data(name='dec_05_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_05_att_lin' output: Data(name='dec_05_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_05_att_drop' output: Data(name='dec_05_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_05_att_out' output: Data(name='dec_05_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_ff_laynorm' output: Data(name='dec_05_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_ff_conv1' output: Data(name='dec_05_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_ff_conv2' output: Data(name='dec_05_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_ff_drop' output: Data(name='dec_05_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_ff_out' output: Data(name='dec_05_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05' output: Data(name='dec_05_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_self_att_laynorm' output: Data(name='dec_06_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_self_att_att' output: Data(name='dec_06_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_06_self_att_lin' output: Data(name='dec_06_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_06_self_att_drop' output: Data(name='dec_06_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_06_self_att_out' output: Data(name='dec_06_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_att_laynorm' output: Data(name='dec_06_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_att_query0' output: Data(name='dec_06_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_att_query' output: Data(name='dec_06_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_att_energy' output: Data(name='dec_06_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_06_att_weights' output: Data(name='dec_06_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_06_att_weights_drop' output: Data(name='dec_06_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_06_att0' output: Data(name='dec_06_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_06_att_att' output: Data(name='dec_06_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_06_att_lin' output: Data(name='dec_06_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_06_att_drop' output: Data(name='dec_06_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_06_att_out' output: Data(name='dec_06_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_ff_laynorm' output: Data(name='dec_06_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_ff_conv1' output: Data(name='dec_06_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_ff_conv2' output: Data(name='dec_06_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_ff_drop' output: Data(name='dec_06_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_ff_out' output: Data(name='dec_06_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06' output: Data(name='dec_06_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_self_att_laynorm' output: Data(name='dec_07_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_self_att_att' output: Data(name='dec_07_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_07_self_att_lin' output: Data(name='dec_07_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_07_self_att_drop' output: Data(name='dec_07_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_07_self_att_out' output: Data(name='dec_07_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_att_laynorm' output: Data(name='dec_07_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_att_query0' output: Data(name='dec_07_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_att_query' output: Data(name='dec_07_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_att_energy' output: Data(name='dec_07_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_07_att_weights' output: Data(name='dec_07_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_07_att_weights_drop' output: Data(name='dec_07_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_07_att0' output: Data(name='dec_07_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_07_att_att' output: Data(name='dec_07_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_07_att_lin' output: Data(name='dec_07_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_07_att_drop' output: Data(name='dec_07_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_07_att_out' output: Data(name='dec_07_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_ff_laynorm' output: Data(name='dec_07_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_ff_conv1' output: Data(name='dec_07_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_ff_conv2' output: Data(name='dec_07_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_ff_drop' output: Data(name='dec_07_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_ff_out' output: Data(name='dec_07_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07' output: Data(name='dec_07_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_self_att_laynorm' output: Data(name='dec_08_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_self_att_att' output: Data(name='dec_08_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_08_self_att_lin' output: Data(name='dec_08_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_08_self_att_drop' output: Data(name='dec_08_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_08_self_att_out' output: Data(name='dec_08_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_att_laynorm' output: Data(name='dec_08_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_att_query0' output: Data(name='dec_08_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_att_query' output: Data(name='dec_08_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_att_energy' output: Data(name='dec_08_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_08_att_weights' output: Data(name='dec_08_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_08_att_weights_drop' output: Data(name='dec_08_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_08_att0' output: Data(name='dec_08_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_08_att_att' output: Data(name='dec_08_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_08_att_lin' output: Data(name='dec_08_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_08_att_drop' output: Data(name='dec_08_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_08_att_out' output: Data(name='dec_08_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_ff_laynorm' output: Data(name='dec_08_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_ff_conv1' output: Data(name='dec_08_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_ff_conv2' output: Data(name='dec_08_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_ff_drop' output: Data(name='dec_08_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_ff_out' output: Data(name='dec_08_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08' output: Data(name='dec_08_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_self_att_laynorm' output: Data(name='dec_09_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_self_att_att' output: Data(name='dec_09_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_09_self_att_lin' output: Data(name='dec_09_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_09_self_att_drop' output: Data(name='dec_09_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_09_self_att_out' output: Data(name='dec_09_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_att_laynorm' output: Data(name='dec_09_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_att_query0' output: Data(name='dec_09_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_att_query' output: Data(name='dec_09_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_att_energy' output: Data(name='dec_09_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_09_att_weights' output: Data(name='dec_09_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_09_att_weights_drop' output: Data(name='dec_09_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_09_att0' output: Data(name='dec_09_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_09_att_att' output: Data(name='dec_09_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_09_att_lin' output: Data(name='dec_09_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_09_att_drop' output: Data(name='dec_09_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_09_att_out' output: Data(name='dec_09_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_ff_laynorm' output: Data(name='dec_09_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_ff_conv1' output: Data(name='dec_09_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_ff_conv2' output: Data(name='dec_09_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_ff_drop' output: Data(name='dec_09_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_ff_out' output: Data(name='dec_09_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09' output: Data(name='dec_09_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_self_att_laynorm' output: Data(name='dec_10_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_self_att_att' output: Data(name='dec_10_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_10_self_att_lin' output: Data(name='dec_10_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_10_self_att_drop' output: Data(name='dec_10_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_10_self_att_out' output: Data(name='dec_10_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_att_laynorm' output: Data(name='dec_10_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_att_query0' output: Data(name='dec_10_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_att_query' output: Data(name='dec_10_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_att_energy' output: Data(name='dec_10_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_10_att_weights' output: Data(name='dec_10_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_10_att_weights_drop' output: Data(name='dec_10_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_10_att0' output: Data(name='dec_10_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_10_att_att' output: Data(name='dec_10_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_10_att_lin' output: Data(name='dec_10_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_10_att_drop' output: Data(name='dec_10_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_10_att_out' output: Data(name='dec_10_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_ff_laynorm' output: Data(name='dec_10_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_ff_conv1' output: Data(name='dec_10_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_ff_conv2' output: Data(name='dec_10_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_ff_drop' output: Data(name='dec_10_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_ff_out' output: Data(name='dec_10_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10' output: Data(name='dec_10_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_self_att_laynorm' output: Data(name='dec_11_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_self_att_att' output: Data(name='dec_11_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_11_self_att_lin' output: Data(name='dec_11_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_11_self_att_drop' output: Data(name='dec_11_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_11_self_att_out' output: Data(name='dec_11_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_att_laynorm' output: Data(name='dec_11_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_att_query0' output: Data(name='dec_11_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_att_query' output: Data(name='dec_11_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_att_energy' output: Data(name='dec_11_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_11_att_weights' output: Data(name='dec_11_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_11_att_weights_drop' output: Data(name='dec_11_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_11_att0' output: Data(name='dec_11_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_11_att_att' output: Data(name='dec_11_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_11_att_lin' output: Data(name='dec_11_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_11_att_drop' output: Data(name='dec_11_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_11_att_out' output: Data(name='dec_11_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_ff_laynorm' output: Data(name='dec_11_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_ff_conv1' output: Data(name='dec_11_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_ff_conv2' output: Data(name='dec_11_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_ff_drop' output: Data(name='dec_11_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_ff_out' output: Data(name='dec_11_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11' output: Data(name='dec_11_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_self_att_laynorm' output: Data(name='dec_12_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_self_att_att' output: Data(name='dec_12_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_12_self_att_lin' output: Data(name='dec_12_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_12_self_att_drop' output: Data(name='dec_12_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_12_self_att_out' output: Data(name='dec_12_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_att_laynorm' output: Data(name='dec_12_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_att_query0' output: Data(name='dec_12_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_att_query' output: Data(name='dec_12_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_att_energy' output: Data(name='dec_12_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_12_att_weights' output: Data(name='dec_12_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'encoder_int' output: Data(name='encoder_int_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_12_att_weights_drop' output: Data(name='dec_12_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_12_att0' output: Data(name='dec_12_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_12_att_att' output: Data(name='dec_12_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_12_att_lin' output: Data(name='dec_12_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_12_att_drop' output: Data(name='dec_12_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_12_att_out' output: Data(name='dec_12_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_ff_laynorm' output: Data(name='dec_12_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_ff_conv1' output: Data(name='dec_12_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_ff_conv2' output: Data(name='dec_12_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_ff_drop' output: Data(name='dec_12_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_ff_out' output: Data(name='dec_12_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12' output: Data(name='dec_12_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'decoder' output: Data(name='decoder_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'decoder_int' output: Data(name='decoder_int_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'prev:target_embed_raw' output: Data(name='target_embed_raw_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'prev_outputs_int' output: Data(name='prev_outputs_int_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'output_prob' output: Data(name='dec_12_att_weights_output', shape=(None, 10025), batch_dim_axis=1)
layer root/'decision' output: Data(name='output_output', shape=(None,), dtype='int32', sparse=True, dim=10025, batch_dim_axis=1)
Warning: using numerical unstable sparse Cross-Entropy loss calculation
Network layer topology:
  extern data: classes: Data(shape=(None,), dtype='int32', sparse=True, dim=10025, available_for_inference=False), data: Data(shape=(None, 40))
  used data keys: ['classes', 'data']
  layer softmax 'ctc' #: 10026
  layer source 'data' #: 40
  layer split_dims 'dec_01_att_key' #: 64
  layer linear 'dec_01_att_key0' #: 512
  layer split_dims 'dec_01_att_value' #: 64
  layer linear 'dec_01_att_value0' #: 512
  layer split_dims 'dec_02_att_key' #: 64
  layer linear 'dec_02_att_key0' #: 512
  layer split_dims 'dec_02_att_value' #: 64
  layer linear 'dec_02_att_value0' #: 512
  layer split_dims 'dec_03_att_key' #: 64
  layer linear 'dec_03_att_key0' #: 512
  layer split_dims 'dec_03_att_value' #: 64
  layer linear 'dec_03_att_value0' #: 512
  layer split_dims 'dec_04_att_key' #: 64
  layer linear 'dec_04_att_key0' #: 512
  layer split_dims 'dec_04_att_value' #: 64
  layer linear 'dec_04_att_value0' #: 512
  layer split_dims 'dec_05_att_key' #: 64
  layer linear 'dec_05_att_key0' #: 512
  layer split_dims 'dec_05_att_value' #: 64
  layer linear 'dec_05_att_value0' #: 512
  layer split_dims 'dec_06_att_key' #: 64
  layer linear 'dec_06_att_key0' #: 512
  layer split_dims 'dec_06_att_value' #: 64
  layer linear 'dec_06_att_value0' #: 512
  layer split_dims 'dec_07_att_key' #: 64
  layer linear 'dec_07_att_key0' #: 512
  layer split_dims 'dec_07_att_value' #: 64
  layer linear 'dec_07_att_value0' #: 512
  layer split_dims 'dec_08_att_key' #: 64
  layer linear 'dec_08_att_key0' #: 512
  layer split_dims 'dec_08_att_value' #: 64
  layer linear 'dec_08_att_value0' #: 512
  layer split_dims 'dec_09_att_key' #: 64
  layer linear 'dec_09_att_key0' #: 512
  layer split_dims 'dec_09_att_value' #: 64
  layer linear 'dec_09_att_value0' #: 512
  layer split_dims 'dec_10_att_key' #: 64
  layer linear 'dec_10_att_key0' #: 512
  layer split_dims 'dec_10_att_value' #: 64
  layer linear 'dec_10_att_value0' #: 512
  layer split_dims 'dec_11_att_key' #: 64
  layer linear 'dec_11_att_key0' #: 512
  layer split_dims 'dec_11_att_value' #: 64
  layer linear 'dec_11_att_value0' #: 512
  layer split_dims 'dec_12_att_key' #: 64
  layer linear 'dec_12_att_key0' #: 512
  layer split_dims 'dec_12_att_value' #: 64
  layer linear 'dec_12_att_value0' #: 512
  layer decide 'decision' #: 10025
  layer copy 'enc_01' #: 512
  layer linear 'enc_01_ff_conv1' #: 2048
  layer linear 'enc_01_ff_conv2' #: 512
  layer dropout 'enc_01_ff_drop' #: 512
  layer layer_norm 'enc_01_ff_laynorm' #: 512
  layer combine 'enc_01_ff_out' #: 512
  layer self_attention 'enc_01_self_att_att' #: 512
  layer dropout 'enc_01_self_att_drop' #: 512
  layer layer_norm 'enc_01_self_att_laynorm' #: 512
  layer linear 'enc_01_self_att_lin' #: 512
  layer combine 'enc_01_self_att_out' #: 512
  layer copy 'enc_02' #: 512
  layer linear 'enc_02_ff_conv1' #: 2048
  layer linear 'enc_02_ff_conv2' #: 512
  layer dropout 'enc_02_ff_drop' #: 512
  layer layer_norm 'enc_02_ff_laynorm' #: 512
  layer combine 'enc_02_ff_out' #: 512
  layer self_attention 'enc_02_self_att_att' #: 512
  layer dropout 'enc_02_self_att_drop' #: 512
  layer layer_norm 'enc_02_self_att_laynorm' #: 512
  layer linear 'enc_02_self_att_lin' #: 512
  layer combine 'enc_02_self_att_out' #: 512
  layer copy 'enc_03' #: 512
  layer linear 'enc_03_ff_conv1' #: 2048
  layer linear 'enc_03_ff_conv2' #: 512
  layer dropout 'enc_03_ff_drop' #: 512
  layer layer_norm 'enc_03_ff_laynorm' #: 512
  layer combine 'enc_03_ff_out' #: 512
  layer self_attention 'enc_03_self_att_att' #: 512
  layer dropout 'enc_03_self_att_drop' #: 512
  layer layer_norm 'enc_03_self_att_laynorm' #: 512
  layer linear 'enc_03_self_att_lin' #: 512
  layer combine 'enc_03_self_att_out' #: 512
  layer copy 'enc_04' #: 512
  layer linear 'enc_04_ff_conv1' #: 2048
  layer linear 'enc_04_ff_conv2' #: 512
  layer dropout 'enc_04_ff_drop' #: 512
  layer layer_norm 'enc_04_ff_laynorm' #: 512
  layer combine 'enc_04_ff_out' #: 512
  layer self_attention 'enc_04_self_att_att' #: 512
  layer dropout 'enc_04_self_att_drop' #: 512
  layer layer_norm 'enc_04_self_att_laynorm' #: 512
  layer linear 'enc_04_self_att_lin' #: 512
  layer combine 'enc_04_self_att_out' #: 512
  layer copy 'enc_05' #: 512
  layer linear 'enc_05_ff_conv1' #: 2048
  layer linear 'enc_05_ff_conv2' #: 512
  layer dropout 'enc_05_ff_drop' #: 512
  layer layer_norm 'enc_05_ff_laynorm' #: 512
  layer combine 'enc_05_ff_out' #: 512
  layer self_attention 'enc_05_self_att_att' #: 512
  layer dropout 'enc_05_self_att_drop' #: 512
  layer layer_norm 'enc_05_self_att_laynorm' #: 512
  layer linear 'enc_05_self_att_lin' #: 512
  layer combine 'enc_05_self_att_out' #: 512
  layer copy 'enc_06' #: 512
  layer linear 'enc_06_ff_conv1' #: 2048
  layer linear 'enc_06_ff_conv2' #: 512
  layer dropout 'enc_06_ff_drop' #: 512
  layer layer_norm 'enc_06_ff_laynorm' #: 512
  layer combine 'enc_06_ff_out' #: 512
  layer self_attention 'enc_06_self_att_att' #: 512
  layer dropout 'enc_06_self_att_drop' #: 512
  layer layer_norm 'enc_06_self_att_laynorm' #: 512
  layer linear 'enc_06_self_att_lin' #: 512
  layer combine 'enc_06_self_att_out' #: 512
  layer copy 'enc_07' #: 512
  layer linear 'enc_07_ff_conv1' #: 2048
  layer linear 'enc_07_ff_conv2' #: 512
  layer dropout 'enc_07_ff_drop' #: 512
  layer layer_norm 'enc_07_ff_laynorm' #: 512
  layer combine 'enc_07_ff_out' #: 512
  layer self_attention 'enc_07_self_att_att' #: 512
  layer dropout 'enc_07_self_att_drop' #: 512
  layer layer_norm 'enc_07_self_att_laynorm' #: 512
  layer linear 'enc_07_self_att_lin' #: 512
  layer combine 'enc_07_self_att_out' #: 512
  layer copy 'enc_08' #: 512
  layer linear 'enc_08_ff_conv1' #: 2048
  layer linear 'enc_08_ff_conv2' #: 512
  layer dropout 'enc_08_ff_drop' #: 512
  layer layer_norm 'enc_08_ff_laynorm' #: 512
  layer combine 'enc_08_ff_out' #: 512
  layer self_attention 'enc_08_self_att_att' #: 512
  layer dropout 'enc_08_self_att_drop' #: 512
  layer layer_norm 'enc_08_self_att_laynorm' #: 512
  layer linear 'enc_08_self_att_lin' #: 512
  layer combine 'enc_08_self_att_out' #: 512
  layer copy 'enc_09' #: 512
  layer linear 'enc_09_ff_conv1' #: 2048
  layer linear 'enc_09_ff_conv2' #: 512
  layer dropout 'enc_09_ff_drop' #: 512
  layer layer_norm 'enc_09_ff_laynorm' #: 512
  layer combine 'enc_09_ff_out' #: 512
  layer self_attention 'enc_09_self_att_att' #: 512
  layer dropout 'enc_09_self_att_drop' #: 512
  layer layer_norm 'enc_09_self_att_laynorm' #: 512
  layer linear 'enc_09_self_att_lin' #: 512
  layer combine 'enc_09_self_att_out' #: 512
  layer copy 'enc_10' #: 512
  layer linear 'enc_10_ff_conv1' #: 2048
  layer linear 'enc_10_ff_conv2' #: 512
  layer dropout 'enc_10_ff_drop' #: 512
  layer layer_norm 'enc_10_ff_laynorm' #: 512
  layer combine 'enc_10_ff_out' #: 512
  layer self_attention 'enc_10_self_att_att' #: 512
  layer dropout 'enc_10_self_att_drop' #: 512
  layer layer_norm 'enc_10_self_att_laynorm' #: 512
  layer linear 'enc_10_self_att_lin' #: 512
  layer combine 'enc_10_self_att_out' #: 512
  layer copy 'enc_11' #: 512
  layer linear 'enc_11_ff_conv1' #: 2048
  layer linear 'enc_11_ff_conv2' #: 512
  layer dropout 'enc_11_ff_drop' #: 512
  layer layer_norm 'enc_11_ff_laynorm' #: 512
  layer combine 'enc_11_ff_out' #: 512
  layer self_attention 'enc_11_self_att_att' #: 512
  layer dropout 'enc_11_self_att_drop' #: 512
  layer layer_norm 'enc_11_self_att_laynorm' #: 512
  layer linear 'enc_11_self_att_lin' #: 512
  layer combine 'enc_11_self_att_out' #: 512
  layer copy 'enc_12' #: 512
  layer linear 'enc_12_ff_conv1' #: 2048
  layer linear 'enc_12_ff_conv2' #: 512
  layer dropout 'enc_12_ff_drop' #: 512
  layer layer_norm 'enc_12_ff_laynorm' #: 512
  layer combine 'enc_12_ff_out' #: 512
  layer self_attention 'enc_12_self_att_att' #: 512
  layer dropout 'enc_12_self_att_drop' #: 512
  layer layer_norm 'enc_12_self_att_laynorm' #: 512
  layer linear 'enc_12_self_att_lin' #: 512
  layer combine 'enc_12_self_att_out' #: 512
  layer layer_norm 'encoder' #: 512
  layer rec 'lstm0_bw' #: 1024
  layer rec 'lstm0_fw' #: 1024
  layer pool 'lstm0_pool' #: 2048
  layer rec 'lstm1_bw' #: 1024
  layer rec 'lstm1_fw' #: 1024
  layer pool 'lstm1_pool' #: 2048
  layer rec 'output' #: 10025
  layer eval 'source' #: 40
  layer dropout 'source_embed' #: 512
  layer linear 'source_embed_raw' #: 512
  layer eval 'source_embed_weighted' #: 512
net params #: 139347754
net trainable params: [<tf.Variable 'ctc/W:0' shape=(512, 10026) dtype=float32_ref>, <tf.Variable 'ctc/b:0' shape=(10026,) dtype=float32_ref>, <tf.Variable 'dec_01_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_01_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_02_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_02_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_03_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_03_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_04_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_04_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_05_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_05_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_06_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_06_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_07_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_07_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_08_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_08_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_09_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_09_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_10_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_10_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_11_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_11_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_12_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_12_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_01_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_01_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_01_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_01_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_01_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_01_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_01_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_01_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_01_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_01_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_02_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_02_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_02_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_02_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_02_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_02_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_02_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_02_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_02_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_02_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_03_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_03_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_03_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_03_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_03_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_03_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_03_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_03_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_03_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_03_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_04_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_04_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_04_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_04_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_04_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_04_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_04_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_04_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_04_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_04_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_05_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_05_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_05_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_05_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_05_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_05_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_05_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_05_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_05_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_05_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_06_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_06_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_06_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_06_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_06_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_06_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_06_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_06_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_06_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_06_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_07_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_07_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_07_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_07_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_07_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_07_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_07_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_07_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_07_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_07_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_08_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_08_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_08_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_08_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_08_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_08_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_08_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_08_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_08_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_08_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_09_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_09_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_09_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_09_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_09_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_09_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_09_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_09_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_09_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_09_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_10_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_10_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_10_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_10_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_10_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_10_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_10_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_10_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_10_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_10_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_11_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_11_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_11_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_11_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_11_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_11_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_11_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_11_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_11_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_11_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_12_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_12_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_12_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_12_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_12_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_12_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_12_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_12_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_12_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_12_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'encoder/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'encoder/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'lstm0_bw/rec/W:0' shape=(40, 4096) dtype=float32_ref>, <tf.Variable 'lstm0_bw/rec/W_re:0' shape=(1024, 4096) dtype=float32_ref>, <tf.Variable 'lstm0_bw/rec/b:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'lstm0_fw/rec/W:0' shape=(40, 4096) dtype=float32_ref>, <tf.Variable 'lstm0_fw/rec/W_re:0' shape=(1024, 4096) dtype=float32_ref>, <tf.Variable 'lstm0_fw/rec/b:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'lstm1_bw/rec/W:0' shape=(2048, 4096) dtype=float32_ref>, <tf.Variable 'lstm1_bw/rec/W_re:0' shape=(1024, 4096) dtype=float32_ref>, <tf.Variable 'lstm1_bw/rec/b:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'lstm1_fw/rec/W:0' shape=(2048, 4096) dtype=float32_ref>, <tf.Variable 'lstm1_fw/rec/W_re:0' shape=(1024, 4096) dtype=float32_ref>, <tf.Variable 'lstm1_fw/rec/b:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/decoder/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/decoder/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/decoder_int/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/encoder_int/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/output_prob/dense/kernel:0' shape=(512, 10025) dtype=float32_ref>, <tf.Variable 'output/rec/prev_outputs_int/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/target_embed_raw/W:0' shape=(10025, 512) dtype=float32_ref>, <tf.Variable 'source_embed_raw/W:0' shape=(2048, 512) dtype=float32_ref>]
loading weights from /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3-prior-k6/net-model/network.560
TF: log_dir: /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/transformer-prior/forward-dev-clean/tf_log_dir/prefix:dev-clean-560-2019-07-02-10-24-19
Note: There are still these uninitialized variables: ['learning_rate:0']
att-weights epoch 560, step 0, max_size:classes 122, max_size:data 3215, mem_usage:GPU:0 1.0GB, num_seqs 1, 7.196 sec/step, elapsed 0:00:13, exp. remaining 0:56:04, complete 0.41%
att-weights epoch 560, step 1, max_size:classes 111, max_size:data 3249, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.992 sec/step, elapsed 0:00:18, exp. remaining 1:09:37, complete 0.44%
att-weights epoch 560, step 2, max_size:classes 103, max_size:data 3265, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.867 sec/step, elapsed 0:00:20, exp. remaining 1:11:20, complete 0.48%
att-weights epoch 560, step 3, max_size:classes 102, max_size:data 3232, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.748 sec/step, elapsed 0:00:22, exp. remaining 1:11:58, complete 0.52%
att-weights epoch 560, step 4, max_size:classes 100, max_size:data 3165, mem_usage:GPU:0 1.0GB, num_seqs 1, 2.018 sec/step, elapsed 0:00:24, exp. remaining 1:13:23, complete 0.55%
att-weights epoch 560, step 5, max_size:classes 98, max_size:data 3245, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.646 sec/step, elapsed 0:00:26, exp. remaining 1:13:31, complete 0.59%
att-weights epoch 560, step 6, max_size:classes 99, max_size:data 3206, mem_usage:GPU:0 1.0GB, num_seqs 1, 2.054 sec/step, elapsed 0:00:28, exp. remaining 1:15:00, complete 0.63%
att-weights epoch 560, step 7, max_size:classes 90, max_size:data 2895, mem_usage:GPU:0 1.0GB, num_seqs 1, 4.264 sec/step, elapsed 0:00:32, exp. remaining 1:21:54, complete 0.67%
att-weights epoch 560, step 8, max_size:classes 98, max_size:data 3171, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.469 sec/step, elapsed 0:00:34, exp. remaining 1:21:08, complete 0.70%
att-weights epoch 560, step 9, max_size:classes 94, max_size:data 3138, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.658 sec/step, elapsed 0:00:36, exp. remaining 1:20:56, complete 0.74%
att-weights epoch 560, step 10, max_size:classes 97, max_size:data 2880, mem_usage:GPU:0 1.0GB, num_seqs 1, 2.777 sec/step, elapsed 0:00:39, exp. remaining 1:23:02, complete 0.78%
att-weights epoch 560, step 11, max_size:classes 80, max_size:data 2413, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.051 sec/step, elapsed 0:00:40, exp. remaining 1:21:31, complete 0.81%
att-weights epoch 560, step 12, max_size:classes 86, max_size:data 2789, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.243 sec/step, elapsed 0:00:41, exp. remaining 1:20:21, complete 0.85%
att-weights epoch 560, step 13, max_size:classes 91, max_size:data 2858, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.312 sec/step, elapsed 0:00:42, exp. remaining 1:19:25, complete 0.89%
att-weights epoch 560, step 14, max_size:classes 85, max_size:data 2437, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.068 sec/step, elapsed 0:00:43, exp. remaining 1:18:07, complete 0.92%
att-weights epoch 560, step 15, max_size:classes 87, max_size:data 2345, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.904 sec/step, elapsed 0:00:44, exp. remaining 1:16:39, complete 0.96%
att-weights epoch 560, step 16, max_size:classes 78, max_size:data 2269, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.887 sec/step, elapsed 0:00:45, exp. remaining 1:15:15, complete 1.00%
att-weights epoch 560, step 17, max_size:classes 81, max_size:data 2404, mem_usage:GPU:0 1.0GB, num_seqs 1, 3.995 sec/step, elapsed 0:00:49, exp. remaining 1:18:53, complete 1.04%
att-weights epoch 560, step 18, max_size:classes 77, max_size:data 2800, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.188 sec/step, elapsed 0:00:50, exp. remaining 1:17:58, complete 1.07%
att-weights epoch 560, step 19, max_size:classes 76, max_size:data 2440, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.119 sec/step, elapsed 0:00:51, exp. remaining 1:17:00, complete 1.11%
att-weights epoch 560, step 20, max_size:classes 82, max_size:data 2377, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.363 sec/step, elapsed 0:00:53, exp. remaining 1:16:27, complete 1.15%
att-weights epoch 560, step 21, max_size:classes 86, max_size:data 2663, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.486 sec/step, elapsed 0:00:54, exp. remaining 1:16:06, complete 1.18%
att-weights epoch 560, step 22, max_size:classes 89, max_size:data 2841, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.278 sec/step, elapsed 0:00:55, exp. remaining 1:15:30, complete 1.22%
att-weights epoch 560, step 23, max_size:classes 82, max_size:data 2907, mem_usage:GPU:0 1.0GB, num_seqs 1, 3.599 sec/step, elapsed 0:00:59, exp. remaining 1:17:57, complete 1.26%
att-weights epoch 560, step 24, max_size:classes 79, max_size:data 2453, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.125 sec/step, elapsed 0:01:00, exp. remaining 1:17:08, complete 1.29%
att-weights epoch 560, step 25, max_size:classes 71, max_size:data 2287, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.879 sec/step, elapsed 0:01:01, exp. remaining 1:16:03, complete 1.33%
att-weights epoch 560, step 26, max_size:classes 77, max_size:data 2308, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.978 sec/step, elapsed 0:01:02, exp. remaining 1:15:08, complete 1.37%
att-weights epoch 560, step 27, max_size:classes 81, max_size:data 2282, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.972 sec/step, elapsed 0:01:03, exp. remaining 1:14:16, complete 1.41%
att-weights epoch 560, step 28, max_size:classes 81, max_size:data 2100, mem_usage:GPU:0 1.0GB, num_seqs 1, 4.050 sec/step, elapsed 0:01:07, exp. remaining 1:16:57, complete 1.44%
att-weights epoch 560, step 29, max_size:classes 94, max_size:data 2941, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.225 sec/step, elapsed 0:01:08, exp. remaining 1:16:21, complete 1.48%
att-weights epoch 560, step 30, max_size:classes 73, max_size:data 2089, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.898 sec/step, elapsed 0:01:09, exp. remaining 1:15:26, complete 1.52%
att-weights epoch 560, step 31, max_size:classes 73, max_size:data 2300, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.924 sec/step, elapsed 0:01:10, exp. remaining 1:14:35, complete 1.55%
att-weights epoch 560, step 32, max_size:classes 76, max_size:data 2194, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.990 sec/step, elapsed 0:01:11, exp. remaining 1:13:51, complete 1.59%
att-weights epoch 560, step 33, max_size:classes 75, max_size:data 2297, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.883 sec/step, elapsed 0:01:12, exp. remaining 1:13:02, complete 1.63%
att-weights epoch 560, step 34, max_size:classes 89, max_size:data 2481, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.289 sec/step, elapsed 0:01:13, exp. remaining 1:12:39, complete 1.66%
att-weights epoch 560, step 35, max_size:classes 60, max_size:data 2342, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.940 sec/step, elapsed 0:01:14, exp. remaining 1:11:57, complete 1.70%
att-weights epoch 560, step 36, max_size:classes 72, max_size:data 1928, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.999 sec/step, elapsed 0:01:15, exp. remaining 1:09:49, complete 1.78%
att-weights epoch 560, step 37, max_size:classes 66, max_size:data 2352, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.213 sec/step, elapsed 0:01:16, exp. remaining 1:09:28, complete 1.81%
att-weights epoch 560, step 38, max_size:classes 66, max_size:data 1990, mem_usage:GPU:0 1.0GB, num_seqs 1, 2.227 sec/step, elapsed 0:01:19, exp. remaining 1:08:37, complete 1.89%
att-weights epoch 560, step 39, max_size:classes 78, max_size:data 2234, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.112 sec/step, elapsed 0:01:20, exp. remaining 1:06:55, complete 1.96%
att-weights epoch 560, step 40, max_size:classes 81, max_size:data 2048, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.019 sec/step, elapsed 0:01:21, exp. remaining 1:06:29, complete 2.00%
att-weights epoch 560, step 41, max_size:classes 71, max_size:data 1781, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.789 sec/step, elapsed 0:01:22, exp. remaining 1:05:53, complete 2.03%
att-weights epoch 560, step 42, max_size:classes 67, max_size:data 2118, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.037 sec/step, elapsed 0:01:23, exp. remaining 1:04:19, complete 2.11%
att-weights epoch 560, step 43, max_size:classes 66, max_size:data 1992, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.987 sec/step, elapsed 0:01:24, exp. remaining 1:03:57, complete 2.15%
att-weights epoch 560, step 44, max_size:classes 67, max_size:data 2065, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.069 sec/step, elapsed 0:01:25, exp. remaining 1:02:33, complete 2.22%
att-weights epoch 560, step 45, max_size:classes 78, max_size:data 1944, mem_usage:GPU:0 1.0GB, num_seqs 2, 3.621 sec/step, elapsed 0:01:28, exp. remaining 1:04:07, complete 2.26%
att-weights epoch 560, step 46, max_size:classes 71, max_size:data 2042, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.048 sec/step, elapsed 0:01:29, exp. remaining 1:02:46, complete 2.33%
att-weights epoch 560, step 47, max_size:classes 64, max_size:data 1818, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.384 sec/step, elapsed 0:01:31, exp. remaining 1:01:43, complete 2.40%
att-weights epoch 560, step 48, max_size:classes 70, max_size:data 1979, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.536 sec/step, elapsed 0:01:32, exp. remaining 1:00:51, complete 2.48%
att-weights epoch 560, step 49, max_size:classes 62, max_size:data 2712, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.175 sec/step, elapsed 0:01:33, exp. remaining 0:59:47, complete 2.55%
att-weights epoch 560, step 50, max_size:classes 69, max_size:data 2238, mem_usage:GPU:0 1.0GB, num_seqs 1, 3.313 sec/step, elapsed 0:01:37, exp. remaining 1:00:59, complete 2.59%
att-weights epoch 560, step 51, max_size:classes 67, max_size:data 1936, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.853 sec/step, elapsed 0:01:39, exp. remaining 1:00:22, complete 2.66%
att-weights epoch 560, step 52, max_size:classes 64, max_size:data 2163, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.239 sec/step, elapsed 0:01:40, exp. remaining 1:00:16, complete 2.70%
att-weights epoch 560, step 53, max_size:classes 67, max_size:data 1913, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.521 sec/step, elapsed 0:01:41, exp. remaining 1:00:20, complete 2.74%
att-weights epoch 560, step 54, max_size:classes 66, max_size:data 2073, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.940 sec/step, elapsed 0:01:42, exp. remaining 1:00:03, complete 2.77%
att-weights epoch 560, step 55, max_size:classes 66, max_size:data 1718, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.327 sec/step, elapsed 0:01:44, exp. remaining 0:59:12, complete 2.85%
att-weights epoch 560, step 56, max_size:classes 64, max_size:data 1855, mem_usage:GPU:0 1.0GB, num_seqs 2, 3.323 sec/step, elapsed 0:01:47, exp. remaining 0:59:30, complete 2.92%
att-weights epoch 560, step 57, max_size:classes 68, max_size:data 1633, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.292 sec/step, elapsed 0:01:48, exp. remaining 0:59:26, complete 2.96%
att-weights epoch 560, step 58, max_size:classes 65, max_size:data 1720, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.277 sec/step, elapsed 0:01:50, exp. remaining 0:59:22, complete 3.00%
att-weights epoch 560, step 59, max_size:classes 59, max_size:data 2040, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.916 sec/step, elapsed 0:01:50, exp. remaining 0:59:07, complete 3.03%
att-weights epoch 560, step 60, max_size:classes 64, max_size:data 1647, mem_usage:GPU:0 1.0GB, num_seqs 2, 3.226 sec/step, elapsed 0:01:54, exp. remaining 1:00:05, complete 3.07%
att-weights epoch 560, step 61, max_size:classes 60, max_size:data 1636, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.856 sec/step, elapsed 0:01:55, exp. remaining 0:59:47, complete 3.11%
att-weights epoch 560, step 62, max_size:classes 67, max_size:data 2318, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.291 sec/step, elapsed 0:01:56, exp. remaining 0:59:43, complete 3.14%
att-weights epoch 560, step 63, max_size:classes 54, max_size:data 2054, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.961 sec/step, elapsed 0:01:57, exp. remaining 0:58:47, complete 3.22%
att-weights epoch 560, step 64, max_size:classes 61, max_size:data 1758, mem_usage:GPU:0 1.0GB, num_seqs 2, 4.689 sec/step, elapsed 0:02:02, exp. remaining 0:59:43, complete 3.29%
att-weights epoch 560, step 65, max_size:classes 63, max_size:data 1969, mem_usage:GPU:0 1.0GB, num_seqs 2, 3.892 sec/step, elapsed 0:02:05, exp. remaining 1:00:13, complete 3.37%
att-weights epoch 560, step 66, max_size:classes 64, max_size:data 1563, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.682 sec/step, elapsed 0:02:06, exp. remaining 0:59:12, complete 3.44%
att-weights epoch 560, step 67, max_size:classes 60, max_size:data 2087, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.794 sec/step, elapsed 0:02:07, exp. remaining 0:58:16, complete 3.51%
att-weights epoch 560, step 68, max_size:classes 55, max_size:data 1896, mem_usage:GPU:0 1.0GB, num_seqs 1, 3.398 sec/step, elapsed 0:02:10, exp. remaining 0:59:11, complete 3.55%
att-weights epoch 560, step 69, max_size:classes 60, max_size:data 2691, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.571 sec/step, elapsed 0:02:12, exp. remaining 0:59:15, complete 3.59%
att-weights epoch 560, step 70, max_size:classes 66, max_size:data 2057, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.175 sec/step, elapsed 0:02:13, exp. remaining 0:58:32, complete 3.66%
att-weights epoch 560, step 71, max_size:classes 54, max_size:data 2032, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.225 sec/step, elapsed 0:02:14, exp. remaining 0:57:51, complete 3.74%
att-weights epoch 560, step 72, max_size:classes 64, max_size:data 1642, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.148 sec/step, elapsed 0:02:15, exp. remaining 0:57:10, complete 3.81%
att-weights epoch 560, step 73, max_size:classes 61, max_size:data 1911, mem_usage:GPU:0 1.0GB, num_seqs 2, 3.164 sec/step, elapsed 0:02:19, exp. remaining 0:57:55, complete 3.85%
att-weights epoch 560, step 74, max_size:classes 62, max_size:data 1930, mem_usage:GPU:0 1.0GB, num_seqs 2, 3.378 sec/step, elapsed 0:02:22, exp. remaining 0:58:44, complete 3.88%
att-weights epoch 560, step 75, max_size:classes 62, max_size:data 1788, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.200 sec/step, elapsed 0:02:23, exp. remaining 0:58:05, complete 3.96%
att-weights epoch 560, step 76, max_size:classes 59, max_size:data 1671, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.188 sec/step, elapsed 0:02:24, exp. remaining 0:58:00, complete 4.00%
att-weights epoch 560, step 77, max_size:classes 61, max_size:data 1887, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.873 sec/step, elapsed 0:02:25, exp. remaining 0:57:47, complete 4.03%
att-weights epoch 560, step 78, max_size:classes 69, max_size:data 2029, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.550 sec/step, elapsed 0:02:27, exp. remaining 0:57:18, complete 4.11%
att-weights epoch 560, step 79, max_size:classes 65, max_size:data 1733, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.495 sec/step, elapsed 0:02:28, exp. remaining 0:56:49, complete 4.18%
att-weights epoch 560, step 80, max_size:classes 62, max_size:data 1778, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.471 sec/step, elapsed 0:02:30, exp. remaining 0:56:51, complete 4.22%
att-weights epoch 560, step 81, max_size:classes 65, max_size:data 1766, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.296 sec/step, elapsed 0:02:32, exp. remaining 0:57:12, complete 4.25%
att-weights epoch 560, step 82, max_size:classes 53, max_size:data 1463, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.681 sec/step, elapsed 0:02:33, exp. remaining 0:56:26, complete 4.33%
att-weights epoch 560, step 83, max_size:classes 69, max_size:data 2096, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.038 sec/step, elapsed 0:02:34, exp. remaining 0:55:49, complete 4.40%
att-weights epoch 560, step 84, max_size:classes 59, max_size:data 1629, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.225 sec/step, elapsed 0:02:35, exp. remaining 0:55:17, complete 4.48%
att-weights epoch 560, step 85, max_size:classes 68, max_size:data 1529, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.968 sec/step, elapsed 0:02:36, exp. remaining 0:54:41, complete 4.55%
att-weights epoch 560, step 86, max_size:classes 55, max_size:data 2056, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.940 sec/step, elapsed 0:02:37, exp. remaining 0:54:05, complete 4.62%
att-weights epoch 560, step 87, max_size:classes 59, max_size:data 1778, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.182 sec/step, elapsed 0:02:38, exp. remaining 0:53:36, complete 4.70%
att-weights epoch 560, step 88, max_size:classes 63, max_size:data 1832, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.453 sec/step, elapsed 0:02:40, exp. remaining 0:53:12, complete 4.77%
att-weights epoch 560, step 89, max_size:classes 56, max_size:data 1655, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.710 sec/step, elapsed 0:02:40, exp. remaining 0:53:01, complete 4.81%
att-weights epoch 560, step 90, max_size:classes 56, max_size:data 2470, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.289 sec/step, elapsed 0:02:42, exp. remaining 0:53:00, complete 4.85%
att-weights epoch 560, step 91, max_size:classes 59, max_size:data 1756, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.379 sec/step, elapsed 0:02:44, exp. remaining 0:52:56, complete 4.92%
att-weights epoch 560, step 92, max_size:classes 61, max_size:data 1845, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.276 sec/step, elapsed 0:02:45, exp. remaining 0:52:56, complete 4.96%
att-weights epoch 560, step 93, max_size:classes 53, max_size:data 1747, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.333 sec/step, elapsed 0:02:48, exp. remaining 0:52:51, complete 5.03%
att-weights epoch 560, step 94, max_size:classes 53, max_size:data 1723, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.171 sec/step, elapsed 0:02:49, exp. remaining 0:52:49, complete 5.07%
att-weights epoch 560, step 95, max_size:classes 58, max_size:data 1593, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.339 sec/step, elapsed 0:02:50, exp. remaining 0:52:49, complete 5.11%
att-weights epoch 560, step 96, max_size:classes 62, max_size:data 1939, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.362 sec/step, elapsed 0:02:51, exp. remaining 0:52:27, complete 5.18%
att-weights epoch 560, step 97, max_size:classes 60, max_size:data 1912, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.991 sec/step, elapsed 0:02:53, exp. remaining 0:52:16, complete 5.25%
att-weights epoch 560, step 98, max_size:classes 52, max_size:data 1629, mem_usage:GPU:0 1.0GB, num_seqs 1, 2.287 sec/step, elapsed 0:02:56, exp. remaining 0:52:10, complete 5.33%
att-weights epoch 560, step 99, max_size:classes 55, max_size:data 2083, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.026 sec/step, elapsed 0:02:57, exp. remaining 0:51:43, complete 5.40%
att-weights epoch 560, step 100, max_size:classes 62, max_size:data 1587, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.193 sec/step, elapsed 0:02:58, exp. remaining 0:51:19, complete 5.48%
att-weights epoch 560, step 101, max_size:classes 61, max_size:data 2101, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.890 sec/step, elapsed 0:02:59, exp. remaining 0:50:51, complete 5.55%
att-weights epoch 560, step 102, max_size:classes 55, max_size:data 1729, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.230 sec/step, elapsed 0:03:01, exp. remaining 0:50:46, complete 5.62%
att-weights epoch 560, step 103, max_size:classes 58, max_size:data 2075, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.931 sec/step, elapsed 0:03:02, exp. remaining 0:50:19, complete 5.70%
att-weights epoch 560, step 104, max_size:classes 54, max_size:data 2023, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.176 sec/step, elapsed 0:03:03, exp. remaining 0:49:58, complete 5.77%
att-weights epoch 560, step 105, max_size:classes 53, max_size:data 1786, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.451 sec/step, elapsed 0:03:05, exp. remaining 0:49:41, complete 5.85%
att-weights epoch 560, step 106, max_size:classes 63, max_size:data 1779, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.253 sec/step, elapsed 0:03:06, exp. remaining 0:49:21, complete 5.92%
att-weights epoch 560, step 107, max_size:classes 58, max_size:data 1767, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.896 sec/step, elapsed 0:03:08, exp. remaining 0:49:12, complete 5.99%
att-weights epoch 560, step 108, max_size:classes 54, max_size:data 1691, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.658 sec/step, elapsed 0:03:09, exp. remaining 0:48:59, complete 6.07%
att-weights epoch 560, step 109, max_size:classes 56, max_size:data 1715, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.310 sec/step, elapsed 0:03:11, exp. remaining 0:48:42, complete 6.14%
att-weights epoch 560, step 110, max_size:classes 61, max_size:data 1673, mem_usage:GPU:0 1.0GB, num_seqs 2, 7.459 sec/step, elapsed 0:03:18, exp. remaining 0:49:57, complete 6.22%
att-weights epoch 560, step 111, max_size:classes 54, max_size:data 1830, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.513 sec/step, elapsed 0:03:20, exp. remaining 0:49:42, complete 6.29%
att-weights epoch 560, step 112, max_size:classes 56, max_size:data 1814, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.564 sec/step, elapsed 0:03:21, exp. remaining 0:49:28, complete 6.36%
att-weights epoch 560, step 113, max_size:classes 55, max_size:data 1835, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.535 sec/step, elapsed 0:03:23, exp. remaining 0:49:14, complete 6.44%
att-weights epoch 560, step 114, max_size:classes 55, max_size:data 1641, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.162 sec/step, elapsed 0:03:24, exp. remaining 0:48:55, complete 6.51%
att-weights epoch 560, step 115, max_size:classes 49, max_size:data 1522, mem_usage:GPU:0 1.0GB, num_seqs 2, 12.307 sec/step, elapsed 0:03:36, exp. remaining 0:51:14, complete 6.59%
att-weights epoch 560, step 116, max_size:classes 49, max_size:data 1447, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.085 sec/step, elapsed 0:03:37, exp. remaining 0:50:53, complete 6.66%
att-weights epoch 560, step 117, max_size:classes 56, max_size:data 1834, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.238 sec/step, elapsed 0:03:39, exp. remaining 0:50:34, complete 6.73%
att-weights epoch 560, step 118, max_size:classes 55, max_size:data 1733, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.726 sec/step, elapsed 0:03:41, exp. remaining 0:50:36, complete 6.81%
att-weights epoch 560, step 119, max_size:classes 58, max_size:data 1553, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.188 sec/step, elapsed 0:03:42, exp. remaining 0:50:17, complete 6.88%
att-weights epoch 560, step 120, max_size:classes 57, max_size:data 1574, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.373 sec/step, elapsed 0:03:44, exp. remaining 0:50:01, complete 6.96%
+------- EPILOGUE SCRIPT -----------------------------------------------
|
| Job ID ..............: 9506015
| Stopped at ..........: Tue Jul  2 12:29:51 CEST 2019
| Resources requested .: gpu=1,h_rt=7200,h_rss=4G,h_vmem=1536G,s_core=0,pxe=ubuntu_16.04,num_proc=5,h_fsize=20G,scratch_free=5G
| Resources used ......: cpu=00:07:42, mem=1447.11793 GB s, io=3.57179 GB, vmem=4.349G, maxvmem=4.360G, last_file_cache=104K, last_rss=2M, max-cache=4.000G
| Memory used .........: 4.000G / 4.000G (100.0%)
| Total time used .....: 0:05:33
|
+------- EPILOGUE SCRIPT -----------------------------------------------
+------- PROLOGUE SCRIPT -----------------------------------------------
|
| Job ID ...........: 9506021
| Started at .......: Tue Jul  2 12:35:21 CEST 2019
| Execution host ...: cluster-cn-248
| Cluster queue ....: 4-GPU-1080
| Script ...........: /var/spool/sge/cluster-cn-248/job_scripts/9506021
| > #!/bin/bash
| > 
| > export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64:/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda/extras/CUPTI/lib64/:/usr/local/cuda-9.1/extras/CUPTI/lib64/:/usr/local/cuda-9.0/extras/CUPTI/lib64:/usr/lib/nvidia-418
| > #export LD_LIBRARY_PATH="/usr/local/lib::/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda-9.0/lib64:/usr/lib/nvidia-390:/opt/rbi/sge/lib/glinux:/opt/rbi/sge/lib/glinux"
| > 
| > source /u/bahar/settings/python3-returnn-tf1.9/bin/activate
| > 
| > python3 ~/returnn-parnia/tools/get-attention-weights.py /u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3-prior-k6.config --epoch 560 --data 'config:get_dataset("'$1'")' --dump_dir ~/makarov/NMT/hmm-factorization/experiments/asr2019/transformer-prior/forward-$1 --layers "dec_02_att_weights" --layers "dec_01_att_weights" --layers "dec_03_att_weights" --layers "dec_04_att_weights" --layers "dec_05_att_weights" --layers "dec_06_att_weights" --layers "dec_07_att_weights" --layers "dec_08_att_weights" --layers "dec_09_att_weights" --layers "dec_10_att_weights" --layers "dec_11_att_weights" --layers "dec_12_att_weights" --rec_layer "output" --batch_size 4000 --tf_log_dir /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/transformer-prior/forward-$1/tf_log_dir --reset_seq_ordering sorted_reverse
|
+------- PROLOGUE SCRIPT -----------------------------------------------
RETURNN get-attention-weights starting up.
RETURNN starting up, version 20181022.053313--git-dfada43-dirty, date/time 2019-07-02-12-35-24 (UTC+0200), pid 1271, cwd /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3-prior-k6, Python /u/bahar/settings/python3-returnn-tf1.9/bin/python3
RETURNN config: /u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3-prior-k6.config
RETURNN command line options: ()
Hostname: cluster-cn-248
TensorFlow: 1.10.0 (v1.10.0-0-g656e7a2b34) (<site-package> in /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow)
Error while getting SGE num_proc: FileNotFoundError(2, "No such file or directory: 'qstat'")
Setup TF inter and intra global thread pools, num_threads None, session opts {'log_device_placement': False, 'device_count': {'GPU': 0}}.
CUDA_VISIBLE_DEVICES is set to '2'.
Collecting TensorFlow device list...
Local devices available to TensorFlow:
  1/2: name: "/device:CPU:0"
       device_type: "CPU"
       memory_limit: 268435456
       locality {
       }
       incarnation: 14871484372561417758
  2/2: name: "/device:GPU:0"
       device_type: "GPU"
       memory_limit: 10911236096
       locality {
         bus_id: 2
         numa_node: 1
         links {
         }
       }
       incarnation: 18398298760414255527
       physical_device_desc: "device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:81:00.0, compute capability: 6.1"
Using gpu device 2: GeForce GTX 1080 Ti
Device not set explicitly, and we found a GPU, which we will use.
Model file prefix: /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3-prior-k6/net-model/network
NOTE: We will use 'sorted_reverse' seq ordering.
Setup tf.Session with options {'log_device_placement': False, 'device_count': {'GPU': 1}} ...
layer root/'data' output: Data(name='data', shape=(None, 40))
layer root/'source' output: Data(name='source_output', shape=(None, 40))
layer root/'lstm0_fw' output: Data(name='lstm0_fw_output', shape=(None, 1024), batch_dim_axis=1)
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9506021.1.4-GPU-1080/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.cc -o /var/tmp/9506021.1.4-GPU-1080/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9506021.1.4-GPU-1080/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.cc -o /var/tmp/9506021.1.4-GPU-1080/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
layer root/'lstm0_bw' output: Data(name='lstm0_bw_output', shape=(None, 1024), batch_dim_axis=1)
layer root/'lstm0_pool' output: Data(name='lstm0_pool_output', shape=(None, 2048))
layer root/'lstm1_fw' output: Data(name='lstm1_fw_output', shape=(None, 1024), batch_dim_axis=1)
layer root/'lstm1_bw' output: Data(name='lstm1_bw_output', shape=(None, 1024), batch_dim_axis=1)
layer root/'lstm1_pool' output: Data(name='lstm1_pool_output', shape=(None, 2048))
layer root/'source_embed_raw' output: Data(name='source_embed_raw_output', shape=(None, 512))
layer root/'source_embed_weighted' output: Data(name='source_embed_weighted_output', shape=(None, 512))
layer root/'source_embed' output: Data(name='source_embed_output', shape=(None, 512))
layer root/'enc_01_self_att_laynorm' output: Data(name='enc_01_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_01_self_att_att' output: Data(name='enc_01_self_att_att_output', shape=(None, 512))
layer root/'enc_01_self_att_lin' output: Data(name='enc_01_self_att_lin_output', shape=(None, 512))
layer root/'enc_01_self_att_drop' output: Data(name='enc_01_self_att_drop_output', shape=(None, 512))
layer root/'enc_01_self_att_out' output: Data(name='enc_01_self_att_out_output', shape=(None, 512))
layer root/'enc_01_ff_laynorm' output: Data(name='enc_01_ff_laynorm_output', shape=(None, 512))
layer root/'enc_01_ff_conv1' output: Data(name='enc_01_ff_conv1_output', shape=(None, 2048))
layer root/'enc_01_ff_conv2' output: Data(name='enc_01_ff_conv2_output', shape=(None, 512))
layer root/'enc_01_ff_drop' output: Data(name='enc_01_ff_drop_output', shape=(None, 512))
layer root/'enc_01_ff_out' output: Data(name='enc_01_ff_out_output', shape=(None, 512))
layer root/'enc_01' output: Data(name='enc_01_output', shape=(None, 512))
layer root/'enc_02_self_att_laynorm' output: Data(name='enc_02_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_02_self_att_att' output: Data(name='enc_02_self_att_att_output', shape=(None, 512))
layer root/'enc_02_self_att_lin' output: Data(name='enc_02_self_att_lin_output', shape=(None, 512))
layer root/'enc_02_self_att_drop' output: Data(name='enc_02_self_att_drop_output', shape=(None, 512))
layer root/'enc_02_self_att_out' output: Data(name='enc_02_self_att_out_output', shape=(None, 512))
layer root/'enc_02_ff_laynorm' output: Data(name='enc_02_ff_laynorm_output', shape=(None, 512))
layer root/'enc_02_ff_conv1' output: Data(name='enc_02_ff_conv1_output', shape=(None, 2048))
layer root/'enc_02_ff_conv2' output: Data(name='enc_02_ff_conv2_output', shape=(None, 512))
layer root/'enc_02_ff_drop' output: Data(name='enc_02_ff_drop_output', shape=(None, 512))
layer root/'enc_02_ff_out' output: Data(name='enc_02_ff_out_output', shape=(None, 512))
layer root/'enc_02' output: Data(name='enc_02_output', shape=(None, 512))
layer root/'enc_03_self_att_laynorm' output: Data(name='enc_03_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_03_self_att_att' output: Data(name='enc_03_self_att_att_output', shape=(None, 512))
layer root/'enc_03_self_att_lin' output: Data(name='enc_03_self_att_lin_output', shape=(None, 512))
layer root/'enc_03_self_att_drop' output: Data(name='enc_03_self_att_drop_output', shape=(None, 512))
layer root/'enc_03_self_att_out' output: Data(name='enc_03_self_att_out_output', shape=(None, 512))
layer root/'enc_03_ff_laynorm' output: Data(name='enc_03_ff_laynorm_output', shape=(None, 512))
layer root/'enc_03_ff_conv1' output: Data(name='enc_03_ff_conv1_output', shape=(None, 2048))
layer root/'enc_03_ff_conv2' output: Data(name='enc_03_ff_conv2_output', shape=(None, 512))
layer root/'enc_03_ff_drop' output: Data(name='enc_03_ff_drop_output', shape=(None, 512))
layer root/'enc_03_ff_out' output: Data(name='enc_03_ff_out_output', shape=(None, 512))
layer root/'enc_03' output: Data(name='enc_03_output', shape=(None, 512))
layer root/'enc_04_self_att_laynorm' output: Data(name='enc_04_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_04_self_att_att' output: Data(name='enc_04_self_att_att_output', shape=(None, 512))
layer root/'enc_04_self_att_lin' output: Data(name='enc_04_self_att_lin_output', shape=(None, 512))
layer root/'enc_04_self_att_drop' output: Data(name='enc_04_self_att_drop_output', shape=(None, 512))
layer root/'enc_04_self_att_out' output: Data(name='enc_04_self_att_out_output', shape=(None, 512))
layer root/'enc_04_ff_laynorm' output: Data(name='enc_04_ff_laynorm_output', shape=(None, 512))
layer root/'enc_04_ff_conv1' output: Data(name='enc_04_ff_conv1_output', shape=(None, 2048))
layer root/'enc_04_ff_conv2' output: Data(name='enc_04_ff_conv2_output', shape=(None, 512))
layer root/'enc_04_ff_drop' output: Data(name='enc_04_ff_drop_output', shape=(None, 512))
layer root/'enc_04_ff_out' output: Data(name='enc_04_ff_out_output', shape=(None, 512))
layer root/'enc_04' output: Data(name='enc_04_output', shape=(None, 512))
layer root/'enc_05_self_att_laynorm' output: Data(name='enc_05_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_05_self_att_att' output: Data(name='enc_05_self_att_att_output', shape=(None, 512))
layer root/'enc_05_self_att_lin' output: Data(name='enc_05_self_att_lin_output', shape=(None, 512))
layer root/'enc_05_self_att_drop' output: Data(name='enc_05_self_att_drop_output', shape=(None, 512))
layer root/'enc_05_self_att_out' output: Data(name='enc_05_self_att_out_output', shape=(None, 512))
layer root/'enc_05_ff_laynorm' output: Data(name='enc_05_ff_laynorm_output', shape=(None, 512))
layer root/'enc_05_ff_conv1' output: Data(name='enc_05_ff_conv1_output', shape=(None, 2048))
layer root/'enc_05_ff_conv2' output: Data(name='enc_05_ff_conv2_output', shape=(None, 512))
layer root/'enc_05_ff_drop' output: Data(name='enc_05_ff_drop_output', shape=(None, 512))
layer root/'enc_05_ff_out' output: Data(name='enc_05_ff_out_output', shape=(None, 512))
layer root/'enc_05' output: Data(name='enc_05_output', shape=(None, 512))
layer root/'enc_06_self_att_laynorm' output: Data(name='enc_06_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_06_self_att_att' output: Data(name='enc_06_self_att_att_output', shape=(None, 512))
layer root/'enc_06_self_att_lin' output: Data(name='enc_06_self_att_lin_output', shape=(None, 512))
layer root/'enc_06_self_att_drop' output: Data(name='enc_06_self_att_drop_output', shape=(None, 512))
layer root/'enc_06_self_att_out' output: Data(name='enc_06_self_att_out_output', shape=(None, 512))
layer root/'enc_06_ff_laynorm' output: Data(name='enc_06_ff_laynorm_output', shape=(None, 512))
layer root/'enc_06_ff_conv1' output: Data(name='enc_06_ff_conv1_output', shape=(None, 2048))
layer root/'enc_06_ff_conv2' output: Data(name='enc_06_ff_conv2_output', shape=(None, 512))
layer root/'enc_06_ff_drop' output: Data(name='enc_06_ff_drop_output', shape=(None, 512))
layer root/'enc_06_ff_out' output: Data(name='enc_06_ff_out_output', shape=(None, 512))
layer root/'enc_06' output: Data(name='enc_06_output', shape=(None, 512))
layer root/'enc_07_self_att_laynorm' output: Data(name='enc_07_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_07_self_att_att' output: Data(name='enc_07_self_att_att_output', shape=(None, 512))
layer root/'enc_07_self_att_lin' output: Data(name='enc_07_self_att_lin_output', shape=(None, 512))
layer root/'enc_07_self_att_drop' output: Data(name='enc_07_self_att_drop_output', shape=(None, 512))
layer root/'enc_07_self_att_out' output: Data(name='enc_07_self_att_out_output', shape=(None, 512))
layer root/'enc_07_ff_laynorm' output: Data(name='enc_07_ff_laynorm_output', shape=(None, 512))
layer root/'enc_07_ff_conv1' output: Data(name='enc_07_ff_conv1_output', shape=(None, 2048))
layer root/'enc_07_ff_conv2' output: Data(name='enc_07_ff_conv2_output', shape=(None, 512))
layer root/'enc_07_ff_drop' output: Data(name='enc_07_ff_drop_output', shape=(None, 512))
layer root/'enc_07_ff_out' output: Data(name='enc_07_ff_out_output', shape=(None, 512))
layer root/'enc_07' output: Data(name='enc_07_output', shape=(None, 512))
layer root/'enc_08_self_att_laynorm' output: Data(name='enc_08_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_08_self_att_att' output: Data(name='enc_08_self_att_att_output', shape=(None, 512))
layer root/'enc_08_self_att_lin' output: Data(name='enc_08_self_att_lin_output', shape=(None, 512))
layer root/'enc_08_self_att_drop' output: Data(name='enc_08_self_att_drop_output', shape=(None, 512))
layer root/'enc_08_self_att_out' output: Data(name='enc_08_self_att_out_output', shape=(None, 512))
layer root/'enc_08_ff_laynorm' output: Data(name='enc_08_ff_laynorm_output', shape=(None, 512))
layer root/'enc_08_ff_conv1' output: Data(name='enc_08_ff_conv1_output', shape=(None, 2048))
layer root/'enc_08_ff_conv2' output: Data(name='enc_08_ff_conv2_output', shape=(None, 512))
layer root/'enc_08_ff_drop' output: Data(name='enc_08_ff_drop_output', shape=(None, 512))
layer root/'enc_08_ff_out' output: Data(name='enc_08_ff_out_output', shape=(None, 512))
layer root/'enc_08' output: Data(name='enc_08_output', shape=(None, 512))
layer root/'enc_09_self_att_laynorm' output: Data(name='enc_09_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_09_self_att_att' output: Data(name='enc_09_self_att_att_output', shape=(None, 512))
layer root/'enc_09_self_att_lin' output: Data(name='enc_09_self_att_lin_output', shape=(None, 512))
layer root/'enc_09_self_att_drop' output: Data(name='enc_09_self_att_drop_output', shape=(None, 512))
layer root/'enc_09_self_att_out' output: Data(name='enc_09_self_att_out_output', shape=(None, 512))
layer root/'enc_09_ff_laynorm' output: Data(name='enc_09_ff_laynorm_output', shape=(None, 512))
layer root/'enc_09_ff_conv1' output: Data(name='enc_09_ff_conv1_output', shape=(None, 2048))
layer root/'enc_09_ff_conv2' output: Data(name='enc_09_ff_conv2_output', shape=(None, 512))
layer root/'enc_09_ff_drop' output: Data(name='enc_09_ff_drop_output', shape=(None, 512))
layer root/'enc_09_ff_out' output: Data(name='enc_09_ff_out_output', shape=(None, 512))
layer root/'enc_09' output: Data(name='enc_09_output', shape=(None, 512))
layer root/'enc_10_self_att_laynorm' output: Data(name='enc_10_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_10_self_att_att' output: Data(name='enc_10_self_att_att_output', shape=(None, 512))
layer root/'enc_10_self_att_lin' output: Data(name='enc_10_self_att_lin_output', shape=(None, 512))
layer root/'enc_10_self_att_drop' output: Data(name='enc_10_self_att_drop_output', shape=(None, 512))
layer root/'enc_10_self_att_out' output: Data(name='enc_10_self_att_out_output', shape=(None, 512))
layer root/'enc_10_ff_laynorm' output: Data(name='enc_10_ff_laynorm_output', shape=(None, 512))
layer root/'enc_10_ff_conv1' output: Data(name='enc_10_ff_conv1_output', shape=(None, 2048))
layer root/'enc_10_ff_conv2' output: Data(name='enc_10_ff_conv2_output', shape=(None, 512))
layer root/'enc_10_ff_drop' output: Data(name='enc_10_ff_drop_output', shape=(None, 512))
layer root/'enc_10_ff_out' output: Data(name='enc_10_ff_out_output', shape=(None, 512))
layer root/'enc_10' output: Data(name='enc_10_output', shape=(None, 512))
layer root/'enc_11_self_att_laynorm' output: Data(name='enc_11_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_11_self_att_att' output: Data(name='enc_11_self_att_att_output', shape=(None, 512))
layer root/'enc_11_self_att_lin' output: Data(name='enc_11_self_att_lin_output', shape=(None, 512))
layer root/'enc_11_self_att_drop' output: Data(name='enc_11_self_att_drop_output', shape=(None, 512))
layer root/'enc_11_self_att_out' output: Data(name='enc_11_self_att_out_output', shape=(None, 512))
layer root/'enc_11_ff_laynorm' output: Data(name='enc_11_ff_laynorm_output', shape=(None, 512))
layer root/'enc_11_ff_conv1' output: Data(name='enc_11_ff_conv1_output', shape=(None, 2048))
layer root/'enc_11_ff_conv2' output: Data(name='enc_11_ff_conv2_output', shape=(None, 512))
layer root/'enc_11_ff_drop' output: Data(name='enc_11_ff_drop_output', shape=(None, 512))
layer root/'enc_11_ff_out' output: Data(name='enc_11_ff_out_output', shape=(None, 512))
layer root/'enc_11' output: Data(name='enc_11_output', shape=(None, 512))
layer root/'enc_12_self_att_laynorm' output: Data(name='enc_12_self_att_laynorm_output', shape=(None, 512))
layer root/'enc_12_self_att_att' output: Data(name='enc_12_self_att_att_output', shape=(None, 512))
layer root/'enc_12_self_att_lin' output: Data(name='enc_12_self_att_lin_output', shape=(None, 512))
layer root/'enc_12_self_att_drop' output: Data(name='enc_12_self_att_drop_output', shape=(None, 512))
layer root/'enc_12_self_att_out' output: Data(name='enc_12_self_att_out_output', shape=(None, 512))
layer root/'enc_12_ff_laynorm' output: Data(name='enc_12_ff_laynorm_output', shape=(None, 512))
layer root/'enc_12_ff_conv1' output: Data(name='enc_12_ff_conv1_output', shape=(None, 2048))
layer root/'enc_12_ff_conv2' output: Data(name='enc_12_ff_conv2_output', shape=(None, 512))
layer root/'enc_12_ff_drop' output: Data(name='enc_12_ff_drop_output', shape=(None, 512))
layer root/'enc_12_ff_out' output: Data(name='enc_12_ff_out_output', shape=(None, 512))
layer root/'enc_12' output: Data(name='enc_12_output', shape=(None, 512))
layer root/'encoder' output: Data(name='encoder_output', shape=(None, 512))
layer root/'ctc' output: Data(name='ctc_output', shape=(None, 10026))
layer root/'dec_11_att_key0' output: Data(name='dec_11_att_key0_output', shape=(None, 512))
layer root/'dec_11_att_key' output: Data(name='dec_11_att_key_output', shape=(None, 8, 64))
layer root/'dec_02_att_key0' output: Data(name='dec_02_att_key0_output', shape=(None, 512))
layer root/'dec_02_att_key' output: Data(name='dec_02_att_key_output', shape=(None, 8, 64))
layer root/'dec_05_att_value0' output: Data(name='dec_05_att_value0_output', shape=(None, 512))
layer root/'dec_05_att_value' output: Data(name='dec_05_att_value_output', shape=(None, 8, 64))
layer root/'dec_06_att_key0' output: Data(name='dec_06_att_key0_output', shape=(None, 512))
layer root/'dec_06_att_key' output: Data(name='dec_06_att_key_output', shape=(None, 8, 64))
layer root/'dec_04_att_value0' output: Data(name='dec_04_att_value0_output', shape=(None, 512))
layer root/'dec_04_att_value' output: Data(name='dec_04_att_value_output', shape=(None, 8, 64))
layer root/'dec_07_att_value0' output: Data(name='dec_07_att_value0_output', shape=(None, 512))
layer root/'dec_07_att_value' output: Data(name='dec_07_att_value_output', shape=(None, 8, 64))
layer root/'dec_05_att_key0' output: Data(name='dec_05_att_key0_output', shape=(None, 512))
layer root/'dec_05_att_key' output: Data(name='dec_05_att_key_output', shape=(None, 8, 64))
layer root/'dec_10_att_value0' output: Data(name='dec_10_att_value0_output', shape=(None, 512))
layer root/'dec_10_att_value' output: Data(name='dec_10_att_value_output', shape=(None, 8, 64))
layer root/'dec_07_att_key0' output: Data(name='dec_07_att_key0_output', shape=(None, 512))
layer root/'dec_07_att_key' output: Data(name='dec_07_att_key_output', shape=(None, 8, 64))
layer root/'dec_06_att_value0' output: Data(name='dec_06_att_value0_output', shape=(None, 512))
layer root/'dec_06_att_value' output: Data(name='dec_06_att_value_output', shape=(None, 8, 64))
layer root/'dec_09_att_key0' output: Data(name='dec_09_att_key0_output', shape=(None, 512))
layer root/'dec_09_att_key' output: Data(name='dec_09_att_key_output', shape=(None, 8, 64))
layer root/'dec_09_att_value0' output: Data(name='dec_09_att_value0_output', shape=(None, 512))
layer root/'dec_09_att_value' output: Data(name='dec_09_att_value_output', shape=(None, 8, 64))
layer root/'dec_03_att_value0' output: Data(name='dec_03_att_value0_output', shape=(None, 512))
layer root/'dec_03_att_value' output: Data(name='dec_03_att_value_output', shape=(None, 8, 64))
layer root/'dec_04_att_key0' output: Data(name='dec_04_att_key0_output', shape=(None, 512))
layer root/'dec_04_att_key' output: Data(name='dec_04_att_key_output', shape=(None, 8, 64))
layer root/'dec_11_att_value0' output: Data(name='dec_11_att_value0_output', shape=(None, 512))
layer root/'dec_11_att_value' output: Data(name='dec_11_att_value_output', shape=(None, 8, 64))
layer root/'dec_03_att_key0' output: Data(name='dec_03_att_key0_output', shape=(None, 512))
layer root/'dec_03_att_key' output: Data(name='dec_03_att_key_output', shape=(None, 8, 64))
layer root/'dec_12_att_value0' output: Data(name='dec_12_att_value0_output', shape=(None, 512))
layer root/'dec_12_att_value' output: Data(name='dec_12_att_value_output', shape=(None, 8, 64))
layer root/'dec_02_att_value0' output: Data(name='dec_02_att_value0_output', shape=(None, 512))
layer root/'dec_02_att_value' output: Data(name='dec_02_att_value_output', shape=(None, 8, 64))
layer root/'dec_08_att_key0' output: Data(name='dec_08_att_key0_output', shape=(None, 512))
layer root/'dec_08_att_key' output: Data(name='dec_08_att_key_output', shape=(None, 8, 64))
layer root/'dec_10_att_key0' output: Data(name='dec_10_att_key0_output', shape=(None, 512))
layer root/'dec_10_att_key' output: Data(name='dec_10_att_key_output', shape=(None, 8, 64))
layer root/'dec_12_att_key0' output: Data(name='dec_12_att_key0_output', shape=(None, 512))
layer root/'dec_12_att_key' output: Data(name='dec_12_att_key_output', shape=(None, 8, 64))
layer root/'dec_01_att_key0' output: Data(name='dec_01_att_key0_output', shape=(None, 512))
layer root/'dec_01_att_key' output: Data(name='dec_01_att_key_output', shape=(None, 8, 64))
layer root/'dec_08_att_value0' output: Data(name='dec_08_att_value0_output', shape=(None, 512))
layer root/'dec_08_att_value' output: Data(name='dec_08_att_value_output', shape=(None, 8, 64))
layer root/'dec_01_att_value0' output: Data(name='dec_01_att_value0_output', shape=(None, 512))
layer root/'dec_01_att_value' output: Data(name='dec_01_att_value_output', shape=(None, 8, 64))
layer root/'output' output: Data(name='output_output', shape=(None,), dtype='int32', sparse=True, dim=10025, batch_dim_axis=1)
Rec layer sub net:
  Input layers moved out of loop: (#: 0)
    None
  Output layers moved out of loop: (#: 273)
    output_prob
    decoder_int
    decoder
    dec_12
    dec_12_ff_out
    dec_12_ff_drop
    dec_12_ff_conv2
    dec_12_ff_conv1
    dec_12_ff_laynorm
    dec_12_att_out
    dec_12_att_drop
    dec_12_att_lin
    dec_12_att_att
    dec_12_att0
    dec_12_att_weights_drop
    dec_12_att_weights
    dec_12_att_energy
    dec_12_att_query
    dec_12_att_query0
    dec_12_att_laynorm
    dec_12_self_att_out
    dec_12_self_att_drop
    dec_12_self_att_lin
    dec_12_self_att_att
    dec_12_self_att_laynorm
    dec_11
    dec_11_ff_out
    dec_11_ff_drop
    dec_11_ff_conv2
    dec_11_ff_conv1
    dec_11_ff_laynorm
    dec_11_att_out
    dec_11_att_drop
    dec_11_att_lin
    dec_11_att_att
    dec_11_att0
    dec_11_att_weights_drop
    dec_11_att_weights
    dec_11_att_energy
    dec_11_att_query
    dec_11_att_query0
    dec_11_att_laynorm
    dec_11_self_att_out
    dec_11_self_att_drop
    dec_11_self_att_lin
    dec_11_self_att_att
    dec_11_self_att_laynorm
    dec_10
    dec_10_ff_out
    dec_10_ff_drop
    dec_10_ff_conv2
    dec_10_ff_conv1
    dec_10_ff_laynorm
    dec_10_att_out
    dec_10_att_drop
    dec_10_att_lin
    dec_10_att_att
    dec_10_att0
    dec_10_att_weights_drop
    dec_10_att_weights
    dec_10_att_energy
    dec_10_att_query
    dec_10_att_query0
    dec_10_att_laynorm
    dec_10_self_att_out
    dec_10_self_att_drop
    dec_10_self_att_lin
    dec_10_self_att_att
    dec_10_self_att_laynorm
    dec_09
    dec_09_ff_out
    dec_09_ff_drop
    dec_09_ff_conv2
    dec_09_ff_conv1
    dec_09_ff_laynorm
    dec_09_att_out
    dec_09_att_drop
    dec_09_att_lin
    dec_09_att_att
    dec_09_att0
    dec_09_att_weights_drop
    dec_09_att_weights
    dec_09_att_energy
    dec_09_att_query
    dec_09_att_query0
    dec_09_att_laynorm
    dec_09_self_att_out
    dec_09_self_att_drop
    dec_09_self_att_lin
    dec_09_self_att_att
    dec_09_self_att_laynorm
    dec_08
    dec_08_ff_out
    dec_08_ff_drop
    dec_08_ff_conv2
    dec_08_ff_conv1
    dec_08_ff_laynorm
    dec_08_att_out
    dec_08_att_drop
    dec_08_att_lin
    dec_08_att_att
    dec_08_att0
    dec_08_att_weights_drop
    dec_08_att_weights
    dec_08_att_energy
    dec_08_att_query
    dec_08_att_query0
    dec_08_att_laynorm
    dec_08_self_att_out
    dec_08_self_att_drop
    dec_08_self_att_lin
    dec_08_self_att_att
    dec_08_self_att_laynorm
    dec_07
    dec_07_ff_out
    dec_07_ff_drop
    dec_07_ff_conv2
    dec_07_ff_conv1
    dec_07_ff_laynorm
    dec_07_att_out
    dec_07_att_drop
    dec_07_att_lin
    dec_07_att_att
    dec_07_att0
    dec_07_att_weights_drop
    dec_07_att_weights
    dec_07_att_energy
    dec_07_att_query
    dec_07_att_query0
    dec_07_att_laynorm
    dec_07_self_att_out
    dec_07_self_att_drop
    dec_07_self_att_lin
    dec_07_self_att_att
    dec_07_self_att_laynorm
    dec_06
    dec_06_ff_out
    dec_06_ff_drop
    dec_06_ff_conv2
    dec_06_ff_conv1
    dec_06_ff_laynorm
    dec_06_att_out
    dec_06_att_drop
    dec_06_att_lin
    dec_06_att_att
    dec_06_att0
    dec_06_att_weights_drop
    dec_06_att_weights
    dec_06_att_energy
    dec_06_att_query
    dec_06_att_query0
    dec_06_att_laynorm
    dec_06_self_att_out
    dec_06_self_att_drop
    dec_06_self_att_lin
    dec_06_self_att_att
    dec_06_self_att_laynorm
    dec_05
    dec_05_ff_out
    dec_05_ff_drop
    dec_05_ff_conv2
    dec_05_ff_conv1
    dec_05_ff_laynorm
    dec_05_att_out
    dec_05_att_drop
    dec_05_att_lin
    dec_05_att_att
    dec_05_att0
    dec_05_att_weights_drop
    dec_05_att_weights
    dec_05_att_energy
    dec_05_att_query
    dec_05_att_query0
    dec_05_att_laynorm
    dec_05_self_att_out
    dec_05_self_att_drop
    dec_05_self_att_lin
    dec_05_self_att_att
    dec_05_self_att_laynorm
    dec_04
    dec_04_ff_out
    dec_04_ff_drop
    dec_04_ff_conv2
    dec_04_ff_conv1
    dec_04_ff_laynorm
    dec_04_att_out
    dec_04_att_drop
    dec_04_att_lin
    dec_04_att_att
    dec_04_att0
    dec_04_att_weights_drop
    dec_04_att_weights
    dec_04_att_energy
    dec_04_att_query
    dec_04_att_query0
    dec_04_att_laynorm
    dec_04_self_att_out
    dec_04_self_att_drop
    dec_04_self_att_lin
    dec_04_self_att_att
    dec_04_self_att_laynorm
    dec_03
    dec_03_ff_out
    dec_03_ff_drop
    dec_03_ff_conv2
    dec_03_ff_conv1
    dec_03_ff_laynorm
    dec_03_att_out
    dec_03_att_drop
    dec_03_att_lin
    dec_03_att_att
    dec_03_att0
    dec_03_att_weights_drop
    dec_03_att_weights
    dec_03_att_energy
    dec_03_att_query
    dec_03_att_query0
    dec_03_att_laynorm
    dec_03_self_att_out
    dec_03_self_att_drop
    dec_03_self_att_lin
    dec_03_self_att_att
    dec_03_self_att_laynorm
    dec_02
    dec_02_ff_out
    dec_02_ff_drop
    dec_02_ff_conv2
    dec_02_ff_conv1
    dec_02_ff_laynorm
    dec_02_att_out
    dec_02_att_drop
    dec_02_att_lin
    dec_02_att_att
    dec_02_att0
    dec_02_att_weights_drop
    dec_02_att_weights
    dec_02_att_energy
    dec_02_att_query
    dec_02_att_query0
    dec_02_att_laynorm
    dec_02_self_att_out
    dec_02_self_att_drop
    dec_02_self_att_lin
    dec_02_self_att_att
    dec_02_self_att_laynorm
    dec_01
    dec_01_ff_out
    dec_01_ff_drop
    dec_01_ff_conv2
    dec_01_ff_conv1
    dec_01_ff_laynorm
    dec_01_att_out
    dec_01_att_drop
    dec_01_att_lin
    dec_01_att_att
    dec_01_att0
    dec_01_att_weights_drop
    dec_01_att_weights
    dec_01_att_energy
    dec_01_att_query
    dec_01_att_query0
    dec_01_att_laynorm
    dec_01_self_att_out
    dec_01_self_att_drop
    dec_01_self_att_lin
    dec_01_self_att_att
    dec_01_self_att_laynorm
    target_embed
    target_embed_weighted
    encoder_int
    prev_outputs_int
    target_embed_raw
    output
  Layers in loop: (#: 0)
    None
  Unused layers: (#: 1)
    end
layer root/output:rec-subnet-output/'output' output: Data(name='classes', shape=(None,), dtype='int32', sparse=True, dim=10025)
layer root/output:rec-subnet-output/'prev:output' output: Data(name='classes', shape=(None,), dtype='int32', sparse=True, dim=10025, batch_dim_axis=1)
layer root/output:rec-subnet-output/'target_embed_raw' output: Data(name='target_embed_raw_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'target_embed_weighted' output: Data(name='target_embed_weighted_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'target_embed' output: Data(name='target_embed_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_self_att_laynorm' output: Data(name='dec_01_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_self_att_att' output: Data(name='dec_01_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_01_self_att_lin' output: Data(name='dec_01_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_01_self_att_drop' output: Data(name='dec_01_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_01_self_att_out' output: Data(name='dec_01_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_att_laynorm' output: Data(name='dec_01_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_att_query0' output: Data(name='dec_01_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_att_query' output: Data(name='dec_01_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_att_energy' output: Data(name='dec_01_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_01_att_weights' output: Data(name='dec_01_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_01_att_weights_drop' output: Data(name='dec_01_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_01_att0' output: Data(name='dec_01_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_01_att_att' output: Data(name='dec_01_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_01_att_lin' output: Data(name='dec_01_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_01_att_drop' output: Data(name='dec_01_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_01_att_out' output: Data(name='dec_01_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_ff_laynorm' output: Data(name='dec_01_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_ff_conv1' output: Data(name='dec_01_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_ff_conv2' output: Data(name='dec_01_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_ff_drop' output: Data(name='dec_01_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01_ff_out' output: Data(name='dec_01_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_01' output: Data(name='dec_01_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_self_att_laynorm' output: Data(name='dec_02_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_self_att_att' output: Data(name='dec_02_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_02_self_att_lin' output: Data(name='dec_02_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_02_self_att_drop' output: Data(name='dec_02_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_02_self_att_out' output: Data(name='dec_02_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_att_laynorm' output: Data(name='dec_02_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_att_query0' output: Data(name='dec_02_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_att_query' output: Data(name='dec_02_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_att_energy' output: Data(name='dec_02_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_02_att_weights' output: Data(name='dec_02_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_02_att_weights_drop' output: Data(name='dec_02_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_02_att0' output: Data(name='dec_02_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_02_att_att' output: Data(name='dec_02_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_02_att_lin' output: Data(name='dec_02_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_02_att_drop' output: Data(name='dec_02_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_02_att_out' output: Data(name='dec_02_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_ff_laynorm' output: Data(name='dec_02_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_ff_conv1' output: Data(name='dec_02_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_ff_conv2' output: Data(name='dec_02_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_ff_drop' output: Data(name='dec_02_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02_ff_out' output: Data(name='dec_02_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_02' output: Data(name='dec_02_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_self_att_laynorm' output: Data(name='dec_03_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_self_att_att' output: Data(name='dec_03_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_03_self_att_lin' output: Data(name='dec_03_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_03_self_att_drop' output: Data(name='dec_03_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_03_self_att_out' output: Data(name='dec_03_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_att_laynorm' output: Data(name='dec_03_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_att_query0' output: Data(name='dec_03_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_att_query' output: Data(name='dec_03_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_att_energy' output: Data(name='dec_03_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_03_att_weights' output: Data(name='dec_03_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_03_att_weights_drop' output: Data(name='dec_03_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_03_att0' output: Data(name='dec_03_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_03_att_att' output: Data(name='dec_03_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_03_att_lin' output: Data(name='dec_03_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_03_att_drop' output: Data(name='dec_03_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_03_att_out' output: Data(name='dec_03_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_ff_laynorm' output: Data(name='dec_03_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_ff_conv1' output: Data(name='dec_03_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_ff_conv2' output: Data(name='dec_03_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_ff_drop' output: Data(name='dec_03_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03_ff_out' output: Data(name='dec_03_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_03' output: Data(name='dec_03_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_self_att_laynorm' output: Data(name='dec_04_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_self_att_att' output: Data(name='dec_04_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_04_self_att_lin' output: Data(name='dec_04_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_04_self_att_drop' output: Data(name='dec_04_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_04_self_att_out' output: Data(name='dec_04_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_att_laynorm' output: Data(name='dec_04_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_att_query0' output: Data(name='dec_04_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_att_query' output: Data(name='dec_04_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_att_energy' output: Data(name='dec_04_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_04_att_weights' output: Data(name='dec_04_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_04_att_weights_drop' output: Data(name='dec_04_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_04_att0' output: Data(name='dec_04_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_04_att_att' output: Data(name='dec_04_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_04_att_lin' output: Data(name='dec_04_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_04_att_drop' output: Data(name='dec_04_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_04_att_out' output: Data(name='dec_04_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_ff_laynorm' output: Data(name='dec_04_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_ff_conv1' output: Data(name='dec_04_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_ff_conv2' output: Data(name='dec_04_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_ff_drop' output: Data(name='dec_04_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04_ff_out' output: Data(name='dec_04_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_04' output: Data(name='dec_04_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_self_att_laynorm' output: Data(name='dec_05_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_self_att_att' output: Data(name='dec_05_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_05_self_att_lin' output: Data(name='dec_05_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_05_self_att_drop' output: Data(name='dec_05_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_05_self_att_out' output: Data(name='dec_05_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_att_laynorm' output: Data(name='dec_05_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_att_query0' output: Data(name='dec_05_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_att_query' output: Data(name='dec_05_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_att_energy' output: Data(name='dec_05_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_05_att_weights' output: Data(name='dec_05_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_05_att_weights_drop' output: Data(name='dec_05_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_05_att0' output: Data(name='dec_05_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_05_att_att' output: Data(name='dec_05_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_05_att_lin' output: Data(name='dec_05_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_05_att_drop' output: Data(name='dec_05_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_05_att_out' output: Data(name='dec_05_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_ff_laynorm' output: Data(name='dec_05_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_ff_conv1' output: Data(name='dec_05_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_ff_conv2' output: Data(name='dec_05_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_ff_drop' output: Data(name='dec_05_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05_ff_out' output: Data(name='dec_05_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_05' output: Data(name='dec_05_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_self_att_laynorm' output: Data(name='dec_06_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_self_att_att' output: Data(name='dec_06_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_06_self_att_lin' output: Data(name='dec_06_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_06_self_att_drop' output: Data(name='dec_06_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_06_self_att_out' output: Data(name='dec_06_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_att_laynorm' output: Data(name='dec_06_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_att_query0' output: Data(name='dec_06_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_att_query' output: Data(name='dec_06_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_att_energy' output: Data(name='dec_06_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_06_att_weights' output: Data(name='dec_06_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_06_att_weights_drop' output: Data(name='dec_06_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_06_att0' output: Data(name='dec_06_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_06_att_att' output: Data(name='dec_06_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_06_att_lin' output: Data(name='dec_06_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_06_att_drop' output: Data(name='dec_06_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_06_att_out' output: Data(name='dec_06_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_ff_laynorm' output: Data(name='dec_06_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_ff_conv1' output: Data(name='dec_06_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_ff_conv2' output: Data(name='dec_06_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_ff_drop' output: Data(name='dec_06_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06_ff_out' output: Data(name='dec_06_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_06' output: Data(name='dec_06_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_self_att_laynorm' output: Data(name='dec_07_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_self_att_att' output: Data(name='dec_07_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_07_self_att_lin' output: Data(name='dec_07_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_07_self_att_drop' output: Data(name='dec_07_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_07_self_att_out' output: Data(name='dec_07_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_att_laynorm' output: Data(name='dec_07_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_att_query0' output: Data(name='dec_07_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_att_query' output: Data(name='dec_07_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_att_energy' output: Data(name='dec_07_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_07_att_weights' output: Data(name='dec_07_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_07_att_weights_drop' output: Data(name='dec_07_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_07_att0' output: Data(name='dec_07_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_07_att_att' output: Data(name='dec_07_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_07_att_lin' output: Data(name='dec_07_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_07_att_drop' output: Data(name='dec_07_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_07_att_out' output: Data(name='dec_07_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_ff_laynorm' output: Data(name='dec_07_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_ff_conv1' output: Data(name='dec_07_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_ff_conv2' output: Data(name='dec_07_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_ff_drop' output: Data(name='dec_07_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07_ff_out' output: Data(name='dec_07_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_07' output: Data(name='dec_07_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_self_att_laynorm' output: Data(name='dec_08_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_self_att_att' output: Data(name='dec_08_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_08_self_att_lin' output: Data(name='dec_08_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_08_self_att_drop' output: Data(name='dec_08_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_08_self_att_out' output: Data(name='dec_08_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_att_laynorm' output: Data(name='dec_08_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_att_query0' output: Data(name='dec_08_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_att_query' output: Data(name='dec_08_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_att_energy' output: Data(name='dec_08_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_08_att_weights' output: Data(name='dec_08_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_08_att_weights_drop' output: Data(name='dec_08_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_08_att0' output: Data(name='dec_08_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_08_att_att' output: Data(name='dec_08_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_08_att_lin' output: Data(name='dec_08_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_08_att_drop' output: Data(name='dec_08_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_08_att_out' output: Data(name='dec_08_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_ff_laynorm' output: Data(name='dec_08_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_ff_conv1' output: Data(name='dec_08_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_ff_conv2' output: Data(name='dec_08_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_ff_drop' output: Data(name='dec_08_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08_ff_out' output: Data(name='dec_08_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_08' output: Data(name='dec_08_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_self_att_laynorm' output: Data(name='dec_09_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_self_att_att' output: Data(name='dec_09_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_09_self_att_lin' output: Data(name='dec_09_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_09_self_att_drop' output: Data(name='dec_09_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_09_self_att_out' output: Data(name='dec_09_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_att_laynorm' output: Data(name='dec_09_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_att_query0' output: Data(name='dec_09_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_att_query' output: Data(name='dec_09_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_att_energy' output: Data(name='dec_09_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_09_att_weights' output: Data(name='dec_09_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_09_att_weights_drop' output: Data(name='dec_09_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_09_att0' output: Data(name='dec_09_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_09_att_att' output: Data(name='dec_09_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_09_att_lin' output: Data(name='dec_09_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_09_att_drop' output: Data(name='dec_09_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_09_att_out' output: Data(name='dec_09_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_ff_laynorm' output: Data(name='dec_09_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_ff_conv1' output: Data(name='dec_09_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_ff_conv2' output: Data(name='dec_09_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_ff_drop' output: Data(name='dec_09_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09_ff_out' output: Data(name='dec_09_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_09' output: Data(name='dec_09_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_self_att_laynorm' output: Data(name='dec_10_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_self_att_att' output: Data(name='dec_10_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_10_self_att_lin' output: Data(name='dec_10_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_10_self_att_drop' output: Data(name='dec_10_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_10_self_att_out' output: Data(name='dec_10_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_att_laynorm' output: Data(name='dec_10_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_att_query0' output: Data(name='dec_10_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_att_query' output: Data(name='dec_10_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_att_energy' output: Data(name='dec_10_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_10_att_weights' output: Data(name='dec_10_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_10_att_weights_drop' output: Data(name='dec_10_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_10_att0' output: Data(name='dec_10_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_10_att_att' output: Data(name='dec_10_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_10_att_lin' output: Data(name='dec_10_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_10_att_drop' output: Data(name='dec_10_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_10_att_out' output: Data(name='dec_10_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_ff_laynorm' output: Data(name='dec_10_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_ff_conv1' output: Data(name='dec_10_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_ff_conv2' output: Data(name='dec_10_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_ff_drop' output: Data(name='dec_10_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10_ff_out' output: Data(name='dec_10_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_10' output: Data(name='dec_10_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_self_att_laynorm' output: Data(name='dec_11_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_self_att_att' output: Data(name='dec_11_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_11_self_att_lin' output: Data(name='dec_11_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_11_self_att_drop' output: Data(name='dec_11_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_11_self_att_out' output: Data(name='dec_11_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_att_laynorm' output: Data(name='dec_11_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_att_query0' output: Data(name='dec_11_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_att_query' output: Data(name='dec_11_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_att_energy' output: Data(name='dec_11_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_11_att_weights' output: Data(name='dec_11_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_11_att_weights_drop' output: Data(name='dec_11_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_11_att0' output: Data(name='dec_11_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_11_att_att' output: Data(name='dec_11_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_11_att_lin' output: Data(name='dec_11_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_11_att_drop' output: Data(name='dec_11_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_11_att_out' output: Data(name='dec_11_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_ff_laynorm' output: Data(name='dec_11_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_ff_conv1' output: Data(name='dec_11_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_ff_conv2' output: Data(name='dec_11_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_ff_drop' output: Data(name='dec_11_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11_ff_out' output: Data(name='dec_11_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_11' output: Data(name='dec_11_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_self_att_laynorm' output: Data(name='dec_12_self_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_self_att_att' output: Data(name='dec_12_self_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_12_self_att_lin' output: Data(name='dec_12_self_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_12_self_att_drop' output: Data(name='dec_12_self_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_12_self_att_out' output: Data(name='dec_12_self_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_att_laynorm' output: Data(name='dec_12_att_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_att_query0' output: Data(name='dec_12_att_query0_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_att_query' output: Data(name='dec_12_att_query_output', shape=(None, 8, 64), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_att_energy' output: Data(name='dec_12_att_energy_output', shape=(8, None, None), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_12_att_weights' output: Data(name='dec_12_att_weights_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'encoder_int' output: Data(name='encoder_int_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_12_att_weights_drop' output: Data(name='dec_12_att_weights_drop_output', shape=(None, 8, None))
layer root/output:rec-subnet-output/'dec_12_att0' output: Data(name='dec_12_att0_output', shape=(8, None, 64), time_dim_axis=2)
layer root/output:rec-subnet-output/'dec_12_att_att' output: Data(name='dec_12_att_att_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_12_att_lin' output: Data(name='dec_12_att_lin_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_12_att_drop' output: Data(name='dec_12_att_drop_output', shape=(None, 512))
layer root/output:rec-subnet-output/'dec_12_att_out' output: Data(name='dec_12_att_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_ff_laynorm' output: Data(name='dec_12_ff_laynorm_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_ff_conv1' output: Data(name='dec_12_ff_conv1_output', shape=(None, 2048), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_ff_conv2' output: Data(name='dec_12_ff_conv2_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_ff_drop' output: Data(name='dec_12_ff_drop_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12_ff_out' output: Data(name='dec_12_ff_out_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'dec_12' output: Data(name='dec_12_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'decoder' output: Data(name='decoder_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'decoder_int' output: Data(name='decoder_int_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'prev:target_embed_raw' output: Data(name='target_embed_raw_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'prev_outputs_int' output: Data(name='prev_outputs_int_output', shape=(None, 512), batch_dim_axis=1)
layer root/output:rec-subnet-output/'output_prob' output: Data(name='dec_12_att_weights_output', shape=(None, 10025), batch_dim_axis=1)
layer root/'decision' output: Data(name='output_output', shape=(None,), dtype='int32', sparse=True, dim=10025, batch_dim_axis=1)
Warning: using numerical unstable sparse Cross-Entropy loss calculation
Network layer topology:
  extern data: classes: Data(shape=(None,), dtype='int32', sparse=True, dim=10025, available_for_inference=False), data: Data(shape=(None, 40))
  used data keys: ['classes', 'data']
  layer softmax 'ctc' #: 10026
  layer source 'data' #: 40
  layer split_dims 'dec_01_att_key' #: 64
  layer linear 'dec_01_att_key0' #: 512
  layer split_dims 'dec_01_att_value' #: 64
  layer linear 'dec_01_att_value0' #: 512
  layer split_dims 'dec_02_att_key' #: 64
  layer linear 'dec_02_att_key0' #: 512
  layer split_dims 'dec_02_att_value' #: 64
  layer linear 'dec_02_att_value0' #: 512
  layer split_dims 'dec_03_att_key' #: 64
  layer linear 'dec_03_att_key0' #: 512
  layer split_dims 'dec_03_att_value' #: 64
  layer linear 'dec_03_att_value0' #: 512
  layer split_dims 'dec_04_att_key' #: 64
  layer linear 'dec_04_att_key0' #: 512
  layer split_dims 'dec_04_att_value' #: 64
  layer linear 'dec_04_att_value0' #: 512
  layer split_dims 'dec_05_att_key' #: 64
  layer linear 'dec_05_att_key0' #: 512
  layer split_dims 'dec_05_att_value' #: 64
  layer linear 'dec_05_att_value0' #: 512
  layer split_dims 'dec_06_att_key' #: 64
  layer linear 'dec_06_att_key0' #: 512
  layer split_dims 'dec_06_att_value' #: 64
  layer linear 'dec_06_att_value0' #: 512
  layer split_dims 'dec_07_att_key' #: 64
  layer linear 'dec_07_att_key0' #: 512
  layer split_dims 'dec_07_att_value' #: 64
  layer linear 'dec_07_att_value0' #: 512
  layer split_dims 'dec_08_att_key' #: 64
  layer linear 'dec_08_att_key0' #: 512
  layer split_dims 'dec_08_att_value' #: 64
  layer linear 'dec_08_att_value0' #: 512
  layer split_dims 'dec_09_att_key' #: 64
  layer linear 'dec_09_att_key0' #: 512
  layer split_dims 'dec_09_att_value' #: 64
  layer linear 'dec_09_att_value0' #: 512
  layer split_dims 'dec_10_att_key' #: 64
  layer linear 'dec_10_att_key0' #: 512
  layer split_dims 'dec_10_att_value' #: 64
  layer linear 'dec_10_att_value0' #: 512
  layer split_dims 'dec_11_att_key' #: 64
  layer linear 'dec_11_att_key0' #: 512
  layer split_dims 'dec_11_att_value' #: 64
  layer linear 'dec_11_att_value0' #: 512
  layer split_dims 'dec_12_att_key' #: 64
  layer linear 'dec_12_att_key0' #: 512
  layer split_dims 'dec_12_att_value' #: 64
  layer linear 'dec_12_att_value0' #: 512
  layer decide 'decision' #: 10025
  layer copy 'enc_01' #: 512
  layer linear 'enc_01_ff_conv1' #: 2048
  layer linear 'enc_01_ff_conv2' #: 512
  layer dropout 'enc_01_ff_drop' #: 512
  layer layer_norm 'enc_01_ff_laynorm' #: 512
  layer combine 'enc_01_ff_out' #: 512
  layer self_attention 'enc_01_self_att_att' #: 512
  layer dropout 'enc_01_self_att_drop' #: 512
  layer layer_norm 'enc_01_self_att_laynorm' #: 512
  layer linear 'enc_01_self_att_lin' #: 512
  layer combine 'enc_01_self_att_out' #: 512
  layer copy 'enc_02' #: 512
  layer linear 'enc_02_ff_conv1' #: 2048
  layer linear 'enc_02_ff_conv2' #: 512
  layer dropout 'enc_02_ff_drop' #: 512
  layer layer_norm 'enc_02_ff_laynorm' #: 512
  layer combine 'enc_02_ff_out' #: 512
  layer self_attention 'enc_02_self_att_att' #: 512
  layer dropout 'enc_02_self_att_drop' #: 512
  layer layer_norm 'enc_02_self_att_laynorm' #: 512
  layer linear 'enc_02_self_att_lin' #: 512
  layer combine 'enc_02_self_att_out' #: 512
  layer copy 'enc_03' #: 512
  layer linear 'enc_03_ff_conv1' #: 2048
  layer linear 'enc_03_ff_conv2' #: 512
  layer dropout 'enc_03_ff_drop' #: 512
  layer layer_norm 'enc_03_ff_laynorm' #: 512
  layer combine 'enc_03_ff_out' #: 512
  layer self_attention 'enc_03_self_att_att' #: 512
  layer dropout 'enc_03_self_att_drop' #: 512
  layer layer_norm 'enc_03_self_att_laynorm' #: 512
  layer linear 'enc_03_self_att_lin' #: 512
  layer combine 'enc_03_self_att_out' #: 512
  layer copy 'enc_04' #: 512
  layer linear 'enc_04_ff_conv1' #: 2048
  layer linear 'enc_04_ff_conv2' #: 512
  layer dropout 'enc_04_ff_drop' #: 512
  layer layer_norm 'enc_04_ff_laynorm' #: 512
  layer combine 'enc_04_ff_out' #: 512
  layer self_attention 'enc_04_self_att_att' #: 512
  layer dropout 'enc_04_self_att_drop' #: 512
  layer layer_norm 'enc_04_self_att_laynorm' #: 512
  layer linear 'enc_04_self_att_lin' #: 512
  layer combine 'enc_04_self_att_out' #: 512
  layer copy 'enc_05' #: 512
  layer linear 'enc_05_ff_conv1' #: 2048
  layer linear 'enc_05_ff_conv2' #: 512
  layer dropout 'enc_05_ff_drop' #: 512
  layer layer_norm 'enc_05_ff_laynorm' #: 512
  layer combine 'enc_05_ff_out' #: 512
  layer self_attention 'enc_05_self_att_att' #: 512
  layer dropout 'enc_05_self_att_drop' #: 512
  layer layer_norm 'enc_05_self_att_laynorm' #: 512
  layer linear 'enc_05_self_att_lin' #: 512
  layer combine 'enc_05_self_att_out' #: 512
  layer copy 'enc_06' #: 512
  layer linear 'enc_06_ff_conv1' #: 2048
  layer linear 'enc_06_ff_conv2' #: 512
  layer dropout 'enc_06_ff_drop' #: 512
  layer layer_norm 'enc_06_ff_laynorm' #: 512
  layer combine 'enc_06_ff_out' #: 512
  layer self_attention 'enc_06_self_att_att' #: 512
  layer dropout 'enc_06_self_att_drop' #: 512
  layer layer_norm 'enc_06_self_att_laynorm' #: 512
  layer linear 'enc_06_self_att_lin' #: 512
  layer combine 'enc_06_self_att_out' #: 512
  layer copy 'enc_07' #: 512
  layer linear 'enc_07_ff_conv1' #: 2048
  layer linear 'enc_07_ff_conv2' #: 512
  layer dropout 'enc_07_ff_drop' #: 512
  layer layer_norm 'enc_07_ff_laynorm' #: 512
  layer combine 'enc_07_ff_out' #: 512
  layer self_attention 'enc_07_self_att_att' #: 512
  layer dropout 'enc_07_self_att_drop' #: 512
  layer layer_norm 'enc_07_self_att_laynorm' #: 512
  layer linear 'enc_07_self_att_lin' #: 512
  layer combine 'enc_07_self_att_out' #: 512
  layer copy 'enc_08' #: 512
  layer linear 'enc_08_ff_conv1' #: 2048
  layer linear 'enc_08_ff_conv2' #: 512
  layer dropout 'enc_08_ff_drop' #: 512
  layer layer_norm 'enc_08_ff_laynorm' #: 512
  layer combine 'enc_08_ff_out' #: 512
  layer self_attention 'enc_08_self_att_att' #: 512
  layer dropout 'enc_08_self_att_drop' #: 512
  layer layer_norm 'enc_08_self_att_laynorm' #: 512
  layer linear 'enc_08_self_att_lin' #: 512
  layer combine 'enc_08_self_att_out' #: 512
  layer copy 'enc_09' #: 512
  layer linear 'enc_09_ff_conv1' #: 2048
  layer linear 'enc_09_ff_conv2' #: 512
  layer dropout 'enc_09_ff_drop' #: 512
  layer layer_norm 'enc_09_ff_laynorm' #: 512
  layer combine 'enc_09_ff_out' #: 512
  layer self_attention 'enc_09_self_att_att' #: 512
  layer dropout 'enc_09_self_att_drop' #: 512
  layer layer_norm 'enc_09_self_att_laynorm' #: 512
  layer linear 'enc_09_self_att_lin' #: 512
  layer combine 'enc_09_self_att_out' #: 512
  layer copy 'enc_10' #: 512
  layer linear 'enc_10_ff_conv1' #: 2048
  layer linear 'enc_10_ff_conv2' #: 512
  layer dropout 'enc_10_ff_drop' #: 512
  layer layer_norm 'enc_10_ff_laynorm' #: 512
  layer combine 'enc_10_ff_out' #: 512
  layer self_attention 'enc_10_self_att_att' #: 512
  layer dropout 'enc_10_self_att_drop' #: 512
  layer layer_norm 'enc_10_self_att_laynorm' #: 512
  layer linear 'enc_10_self_att_lin' #: 512
  layer combine 'enc_10_self_att_out' #: 512
  layer copy 'enc_11' #: 512
  layer linear 'enc_11_ff_conv1' #: 2048
  layer linear 'enc_11_ff_conv2' #: 512
  layer dropout 'enc_11_ff_drop' #: 512
  layer layer_norm 'enc_11_ff_laynorm' #: 512
  layer combine 'enc_11_ff_out' #: 512
  layer self_attention 'enc_11_self_att_att' #: 512
  layer dropout 'enc_11_self_att_drop' #: 512
  layer layer_norm 'enc_11_self_att_laynorm' #: 512
  layer linear 'enc_11_self_att_lin' #: 512
  layer combine 'enc_11_self_att_out' #: 512
  layer copy 'enc_12' #: 512
  layer linear 'enc_12_ff_conv1' #: 2048
  layer linear 'enc_12_ff_conv2' #: 512
  layer dropout 'enc_12_ff_drop' #: 512
  layer layer_norm 'enc_12_ff_laynorm' #: 512
  layer combine 'enc_12_ff_out' #: 512
  layer self_attention 'enc_12_self_att_att' #: 512
  layer dropout 'enc_12_self_att_drop' #: 512
  layer layer_norm 'enc_12_self_att_laynorm' #: 512
  layer linear 'enc_12_self_att_lin' #: 512
  layer combine 'enc_12_self_att_out' #: 512
  layer layer_norm 'encoder' #: 512
  layer rec 'lstm0_bw' #: 1024
  layer rec 'lstm0_fw' #: 1024
  layer pool 'lstm0_pool' #: 2048
  layer rec 'lstm1_bw' #: 1024
  layer rec 'lstm1_fw' #: 1024
  layer pool 'lstm1_pool' #: 2048
  layer rec 'output' #: 10025
  layer eval 'source' #: 40
  layer dropout 'source_embed' #: 512
  layer linear 'source_embed_raw' #: 512
  layer eval 'source_embed_weighted' #: 512
net params #: 139347754
net trainable params: [<tf.Variable 'ctc/W:0' shape=(512, 10026) dtype=float32_ref>, <tf.Variable 'ctc/b:0' shape=(10026,) dtype=float32_ref>, <tf.Variable 'dec_01_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_01_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_02_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_02_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_03_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_03_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_04_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_04_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_05_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_05_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_06_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_06_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_07_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_07_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_08_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_08_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_09_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_09_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_10_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_10_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_11_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_11_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_12_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_12_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_01_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_01_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_01_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_01_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_01_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_01_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_01_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_01_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_01_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_01_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_02_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_02_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_02_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_02_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_02_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_02_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_02_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_02_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_02_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_02_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_03_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_03_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_03_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_03_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_03_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_03_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_03_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_03_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_03_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_03_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_04_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_04_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_04_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_04_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_04_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_04_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_04_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_04_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_04_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_04_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_05_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_05_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_05_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_05_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_05_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_05_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_05_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_05_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_05_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_05_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_06_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_06_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_06_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_06_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_06_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_06_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_06_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_06_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_06_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_06_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_07_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_07_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_07_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_07_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_07_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_07_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_07_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_07_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_07_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_07_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_08_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_08_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_08_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_08_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_08_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_08_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_08_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_08_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_08_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_08_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_09_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_09_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_09_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_09_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_09_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_09_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_09_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_09_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_09_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_09_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_10_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_10_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_10_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_10_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_10_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_10_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_10_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_10_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_10_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_10_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_11_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_11_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_11_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_11_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_11_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_11_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_11_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_11_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_11_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_11_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_12_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_12_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_12_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_12_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_12_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_12_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_12_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_12_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_12_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_12_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'encoder/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'encoder/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'lstm0_bw/rec/W:0' shape=(40, 4096) dtype=float32_ref>, <tf.Variable 'lstm0_bw/rec/W_re:0' shape=(1024, 4096) dtype=float32_ref>, <tf.Variable 'lstm0_bw/rec/b:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'lstm0_fw/rec/W:0' shape=(40, 4096) dtype=float32_ref>, <tf.Variable 'lstm0_fw/rec/W_re:0' shape=(1024, 4096) dtype=float32_ref>, <tf.Variable 'lstm0_fw/rec/b:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'lstm1_bw/rec/W:0' shape=(2048, 4096) dtype=float32_ref>, <tf.Variable 'lstm1_bw/rec/W_re:0' shape=(1024, 4096) dtype=float32_ref>, <tf.Variable 'lstm1_bw/rec/b:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'lstm1_fw/rec/W:0' shape=(2048, 4096) dtype=float32_ref>, <tf.Variable 'lstm1_fw/rec/W_re:0' shape=(1024, 4096) dtype=float32_ref>, <tf.Variable 'lstm1_fw/rec/b:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/decoder/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/decoder/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/decoder_int/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/encoder_int/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/output_prob/dense/kernel:0' shape=(512, 10025) dtype=float32_ref>, <tf.Variable 'output/rec/prev_outputs_int/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/target_embed_raw/W:0' shape=(10025, 512) dtype=float32_ref>, <tf.Variable 'source_embed_raw/W:0' shape=(2048, 512) dtype=float32_ref>]
loading weights from /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3-prior-k6/net-model/network.560
TF: log_dir: /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/transformer-prior/forward-dev-clean/tf_log_dir/prefix:dev-clean-560-2019-07-02-10-35-22
Note: There are still these uninitialized variables: ['learning_rate:0']
att-weights epoch 560, step 0, max_size:classes 122, max_size:data 3215, mem_usage:GPU:0 0.9GB, num_seqs 1, 6.282 sec/step, elapsed 0:00:14, exp. remaining 0:57:07, complete 0.41%
att-weights epoch 560, step 1, max_size:classes 111, max_size:data 3249, mem_usage:GPU:0 0.9GB, num_seqs 1, 5.912 sec/step, elapsed 0:00:20, exp. remaining 1:14:50, complete 0.44%
att-weights epoch 560, step 2, max_size:classes 103, max_size:data 3265, mem_usage:GPU:0 0.9GB, num_seqs 1, 2.366 sec/step, elapsed 0:00:22, exp. remaining 1:17:26, complete 0.48%
att-weights epoch 560, step 3, max_size:classes 102, max_size:data 3232, mem_usage:GPU:0 0.9GB, num_seqs 1, 3.066 sec/step, elapsed 0:00:25, exp. remaining 1:21:48, complete 0.52%
att-weights epoch 560, step 4, max_size:classes 100, max_size:data 3165, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.807 sec/step, elapsed 0:00:27, exp. remaining 1:21:59, complete 0.55%
att-weights epoch 560, step 5, max_size:classes 98, max_size:data 3245, mem_usage:GPU:0 0.9GB, num_seqs 1, 4.819 sec/step, elapsed 0:00:32, exp. remaining 1:30:24, complete 0.59%
att-weights epoch 560, step 6, max_size:classes 99, max_size:data 3206, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.664 sec/step, elapsed 0:00:34, exp. remaining 1:29:40, complete 0.63%
att-weights epoch 560, step 7, max_size:classes 90, max_size:data 2895, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.650 sec/step, elapsed 0:00:35, exp. remaining 1:29:00, complete 0.67%
att-weights epoch 560, step 8, max_size:classes 98, max_size:data 3171, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.782 sec/step, elapsed 0:00:37, exp. remaining 1:28:36, complete 0.70%
att-weights epoch 560, step 9, max_size:classes 94, max_size:data 3138, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.543 sec/step, elapsed 0:00:39, exp. remaining 1:27:52, complete 0.74%
att-weights epoch 560, step 10, max_size:classes 97, max_size:data 2880, mem_usage:GPU:0 0.9GB, num_seqs 1, 2.919 sec/step, elapsed 0:00:42, exp. remaining 1:30:55, complete 0.78%
att-weights epoch 560, step 11, max_size:classes 80, max_size:data 2413, mem_usage:GPU:0 0.9GB, num_seqs 1, 7.826 sec/step, elapsed 0:00:50, exp. remaining 1:42:59, complete 0.81%
att-weights epoch 560, step 12, max_size:classes 86, max_size:data 2789, mem_usage:GPU:0 0.9GB, num_seqs 1, 5.858 sec/step, elapsed 0:00:56, exp. remaining 1:49:51, complete 0.85%
att-weights epoch 560, step 13, max_size:classes 91, max_size:data 2858, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.395 sec/step, elapsed 0:00:57, exp. remaining 1:47:49, complete 0.89%
att-weights epoch 560, step 14, max_size:classes 85, max_size:data 2437, mem_usage:GPU:0 0.9GB, num_seqs 1, 13.235 sec/step, elapsed 0:01:11, exp. remaining 2:07:07, complete 0.92%
att-weights epoch 560, step 15, max_size:classes 87, max_size:data 2345, mem_usage:GPU:0 0.9GB, num_seqs 1, 19.874 sec/step, elapsed 0:01:31, exp. remaining 2:36:17, complete 0.96%
att-weights epoch 560, step 16, max_size:classes 78, max_size:data 2269, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.174 sec/step, elapsed 0:01:32, exp. remaining 2:32:23, complete 1.00%
att-weights epoch 560, step 17, max_size:classes 81, max_size:data 2404, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.072 sec/step, elapsed 0:01:33, exp. remaining 2:28:35, complete 1.04%
att-weights epoch 560, step 18, max_size:classes 77, max_size:data 2800, mem_usage:GPU:0 0.9GB, num_seqs 1, 2.411 sec/step, elapsed 0:01:35, exp. remaining 2:27:07, complete 1.07%
att-weights epoch 560, step 19, max_size:classes 76, max_size:data 2440, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.289 sec/step, elapsed 0:01:37, exp. remaining 2:24:05, complete 1.11%
att-weights epoch 560, step 20, max_size:classes 82, max_size:data 2377, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.882 sec/step, elapsed 0:01:38, exp. remaining 2:22:05, complete 1.15%
att-weights epoch 560, step 21, max_size:classes 86, max_size:data 2663, mem_usage:GPU:0 0.9GB, num_seqs 1, 14.798 sec/step, elapsed 0:01:53, exp. remaining 2:38:15, complete 1.18%
att-weights epoch 560, step 22, max_size:classes 89, max_size:data 2841, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.920 sec/step, elapsed 0:01:55, exp. remaining 2:35:59, complete 1.22%
att-weights epoch 560, step 23, max_size:classes 82, max_size:data 2907, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.389 sec/step, elapsed 0:01:57, exp. remaining 2:33:10, complete 1.26%
att-weights epoch 560, step 24, max_size:classes 79, max_size:data 2453, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.131 sec/step, elapsed 0:01:58, exp. remaining 2:30:10, complete 1.29%
att-weights epoch 560, step 25, max_size:classes 71, max_size:data 2287, mem_usage:GPU:0 0.9GB, num_seqs 1, 4.086 sec/step, elapsed 0:02:02, exp. remaining 2:30:59, complete 1.33%
att-weights epoch 560, step 26, max_size:classes 77, max_size:data 2308, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.976 sec/step, elapsed 0:02:03, exp. remaining 2:28:02, complete 1.37%
att-weights epoch 560, step 27, max_size:classes 81, max_size:data 2282, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.040 sec/step, elapsed 0:02:04, exp. remaining 2:25:18, complete 1.41%
att-weights epoch 560, step 28, max_size:classes 81, max_size:data 2100, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.988 sec/step, elapsed 0:02:05, exp. remaining 2:22:38, complete 1.44%
att-weights epoch 560, step 29, max_size:classes 94, max_size:data 2941, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.341 sec/step, elapsed 0:02:06, exp. remaining 2:20:31, complete 1.48%
att-weights epoch 560, step 30, max_size:classes 73, max_size:data 2089, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.997 sec/step, elapsed 0:02:07, exp. remaining 2:18:07, complete 1.52%
att-weights epoch 560, step 31, max_size:classes 73, max_size:data 2300, mem_usage:GPU:0 0.9GB, num_seqs 1, 3.549 sec/step, elapsed 0:02:11, exp. remaining 2:18:31, complete 1.55%
att-weights epoch 560, step 32, max_size:classes 76, max_size:data 2194, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.983 sec/step, elapsed 0:02:12, exp. remaining 2:16:16, complete 1.59%
att-weights epoch 560, step 33, max_size:classes 75, max_size:data 2297, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.053 sec/step, elapsed 0:02:13, exp. remaining 2:14:11, complete 1.63%
att-weights epoch 560, step 34, max_size:classes 89, max_size:data 2481, mem_usage:GPU:0 0.9GB, num_seqs 1, 3.981 sec/step, elapsed 0:02:17, exp. remaining 2:15:04, complete 1.66%
att-weights epoch 560, step 35, max_size:classes 60, max_size:data 2342, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.100 sec/step, elapsed 0:02:18, exp. remaining 2:13:08, complete 1.70%
att-weights epoch 560, step 36, max_size:classes 72, max_size:data 1928, mem_usage:GPU:0 0.9GB, num_seqs 1, 2.416 sec/step, elapsed 0:02:20, exp. remaining 2:09:44, complete 1.78%
att-weights epoch 560, step 37, max_size:classes 66, max_size:data 2352, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.095 sec/step, elapsed 0:02:21, exp. remaining 2:08:01, complete 1.81%
att-weights epoch 560, step 38, max_size:classes 66, max_size:data 1990, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.777 sec/step, elapsed 0:02:22, exp. remaining 2:03:35, complete 1.89%
att-weights epoch 560, step 39, max_size:classes 78, max_size:data 2234, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.943 sec/step, elapsed 0:02:23, exp. remaining 1:59:37, complete 1.96%
att-weights epoch 560, step 40, max_size:classes 81, max_size:data 2048, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.031 sec/step, elapsed 0:02:24, exp. remaining 1:58:12, complete 2.00%
att-weights epoch 560, step 41, max_size:classes 71, max_size:data 1781, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.736 sec/step, elapsed 0:02:25, exp. remaining 1:56:36, complete 2.03%
att-weights epoch 560, step 42, max_size:classes 67, max_size:data 2118, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.058 sec/step, elapsed 0:02:26, exp. remaining 1:53:14, complete 2.11%
att-weights epoch 560, step 43, max_size:classes 66, max_size:data 1992, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.820 sec/step, elapsed 0:02:27, exp. remaining 1:51:52, complete 2.15%
att-weights epoch 560, step 44, max_size:classes 67, max_size:data 2065, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.963 sec/step, elapsed 0:02:28, exp. remaining 1:48:46, complete 2.22%
att-weights epoch 560, step 45, max_size:classes 78, max_size:data 1944, mem_usage:GPU:0 0.9GB, num_seqs 2, 4.457 sec/step, elapsed 0:02:32, exp. remaining 1:50:09, complete 2.26%
att-weights epoch 560, step 46, max_size:classes 71, max_size:data 2042, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.029 sec/step, elapsed 0:02:33, exp. remaining 1:47:18, complete 2.33%
att-weights epoch 560, step 47, max_size:classes 64, max_size:data 1818, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.499 sec/step, elapsed 0:02:35, exp. remaining 1:44:56, complete 2.40%
att-weights epoch 560, step 48, max_size:classes 70, max_size:data 1979, mem_usage:GPU:0 0.9GB, num_seqs 2, 5.590 sec/step, elapsed 0:02:40, exp. remaining 1:45:23, complete 2.48%
att-weights epoch 560, step 49, max_size:classes 62, max_size:data 2712, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.287 sec/step, elapsed 0:02:42, exp. remaining 1:43:04, complete 2.55%
att-weights epoch 560, step 50, max_size:classes 69, max_size:data 2238, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.950 sec/step, elapsed 0:02:42, exp. remaining 1:42:10, complete 2.59%
att-weights epoch 560, step 51, max_size:classes 67, max_size:data 1936, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.777 sec/step, elapsed 0:02:44, exp. remaining 1:40:20, complete 2.66%
att-weights epoch 560, step 52, max_size:classes 64, max_size:data 2163, mem_usage:GPU:0 0.9GB, num_seqs 1, 2.782 sec/step, elapsed 0:02:47, exp. remaining 1:40:35, complete 2.70%
att-weights epoch 560, step 53, max_size:classes 67, max_size:data 1913, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.701 sec/step, elapsed 0:02:49, exp. remaining 1:40:12, complete 2.74%
att-weights epoch 560, step 54, max_size:classes 66, max_size:data 2073, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.011 sec/step, elapsed 0:02:50, exp. remaining 1:39:25, complete 2.77%
att-weights epoch 560, step 55, max_size:classes 66, max_size:data 1718, mem_usage:GPU:0 0.9GB, num_seqs 2, 4.470 sec/step, elapsed 0:02:54, exp. remaining 1:39:18, complete 2.85%
att-weights epoch 560, step 56, max_size:classes 64, max_size:data 1855, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.832 sec/step, elapsed 0:02:56, exp. remaining 1:37:44, complete 2.92%
att-weights epoch 560, step 57, max_size:classes 68, max_size:data 1633, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.845 sec/step, elapsed 0:02:58, exp. remaining 1:37:29, complete 2.96%
att-weights epoch 560, step 58, max_size:classes 65, max_size:data 1720, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.085 sec/step, elapsed 0:03:00, exp. remaining 1:37:22, complete 3.00%
att-weights epoch 560, step 59, max_size:classes 59, max_size:data 2040, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.188 sec/step, elapsed 0:03:01, exp. remaining 1:36:46, complete 3.03%
att-weights epoch 560, step 60, max_size:classes 64, max_size:data 1647, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.968 sec/step, elapsed 0:03:03, exp. remaining 1:36:36, complete 3.07%
att-weights epoch 560, step 61, max_size:classes 60, max_size:data 1636, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.184 sec/step, elapsed 0:03:04, exp. remaining 1:36:02, complete 3.11%
att-weights epoch 560, step 62, max_size:classes 67, max_size:data 2318, mem_usage:GPU:0 0.9GB, num_seqs 1, 3.734 sec/step, elapsed 0:03:08, exp. remaining 1:36:49, complete 3.14%
att-weights epoch 560, step 63, max_size:classes 54, max_size:data 2054, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.115 sec/step, elapsed 0:03:09, exp. remaining 1:35:04, complete 3.22%
att-weights epoch 560, step 64, max_size:classes 61, max_size:data 1758, mem_usage:GPU:0 0.9GB, num_seqs 2, 3.378 sec/step, elapsed 0:03:13, exp. remaining 1:34:31, complete 3.29%
att-weights epoch 560, step 65, max_size:classes 63, max_size:data 1969, mem_usage:GPU:0 0.9GB, num_seqs 2, 18.290 sec/step, elapsed 0:03:31, exp. remaining 1:41:07, complete 3.37%
att-weights epoch 560, step 66, max_size:classes 64, max_size:data 1563, mem_usage:GPU:0 0.9GB, num_seqs 1, 7.008 sec/step, elapsed 0:03:38, exp. remaining 1:42:09, complete 3.44%
att-weights epoch 560, step 67, max_size:classes 60, max_size:data 2087, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.838 sec/step, elapsed 0:03:39, exp. remaining 1:40:18, complete 3.51%
att-weights epoch 560, step 68, max_size:classes 55, max_size:data 1896, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.897 sec/step, elapsed 0:03:40, exp. remaining 1:39:38, complete 3.55%
att-weights epoch 560, step 69, max_size:classes 60, max_size:data 2691, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.360 sec/step, elapsed 0:03:41, exp. remaining 1:39:10, complete 3.59%
att-weights epoch 560, step 70, max_size:classes 66, max_size:data 2057, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.369 sec/step, elapsed 0:03:42, exp. remaining 1:37:42, complete 3.66%
att-weights epoch 560, step 71, max_size:classes 54, max_size:data 2032, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.852 sec/step, elapsed 0:03:43, exp. remaining 1:36:03, complete 3.74%
att-weights epoch 560, step 72, max_size:classes 64, max_size:data 1642, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.055 sec/step, elapsed 0:03:45, exp. remaining 1:34:59, complete 3.81%
att-weights epoch 560, step 73, max_size:classes 61, max_size:data 1911, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.586 sec/step, elapsed 0:03:47, exp. remaining 1:34:41, complete 3.85%
att-weights epoch 560, step 74, max_size:classes 62, max_size:data 1930, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.682 sec/step, elapsed 0:03:50, exp. remaining 1:34:52, complete 3.88%
att-weights epoch 560, step 75, max_size:classes 62, max_size:data 1788, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.344 sec/step, elapsed 0:03:51, exp. remaining 1:33:33, complete 3.96%
att-weights epoch 560, step 76, max_size:classes 59, max_size:data 1671, mem_usage:GPU:0 0.9GB, num_seqs 2, 4.580 sec/step, elapsed 0:03:55, exp. remaining 1:34:29, complete 4.00%
att-weights epoch 560, step 77, max_size:classes 61, max_size:data 1887, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.942 sec/step, elapsed 0:03:56, exp. remaining 1:33:58, complete 4.03%
att-weights epoch 560, step 78, max_size:classes 69, max_size:data 2029, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.832 sec/step, elapsed 0:03:57, exp. remaining 1:32:31, complete 4.11%
att-weights epoch 560, step 79, max_size:classes 65, max_size:data 1733, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.406 sec/step, elapsed 0:03:59, exp. remaining 1:31:21, complete 4.18%
att-weights epoch 560, step 80, max_size:classes 62, max_size:data 1778, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.474 sec/step, elapsed 0:04:00, exp. remaining 1:31:04, complete 4.22%
att-weights epoch 560, step 81, max_size:classes 65, max_size:data 1766, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.503 sec/step, elapsed 0:04:02, exp. remaining 1:30:49, complete 4.25%
att-weights epoch 560, step 82, max_size:classes 53, max_size:data 1463, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.934 sec/step, elapsed 0:04:03, exp. remaining 1:29:32, complete 4.33%
att-weights epoch 560, step 83, max_size:classes 69, max_size:data 2096, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.072 sec/step, elapsed 0:04:04, exp. remaining 1:28:21, complete 4.40%
att-weights epoch 560, step 84, max_size:classes 59, max_size:data 1629, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.217 sec/step, elapsed 0:04:05, exp. remaining 1:27:15, complete 4.48%
att-weights epoch 560, step 85, max_size:classes 68, max_size:data 1529, mem_usage:GPU:0 0.9GB, num_seqs 1, 4.156 sec/step, elapsed 0:04:09, exp. remaining 1:27:13, complete 4.55%
att-weights epoch 560, step 86, max_size:classes 55, max_size:data 2056, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.906 sec/step, elapsed 0:04:10, exp. remaining 1:26:04, complete 4.62%
att-weights epoch 560, step 87, max_size:classes 59, max_size:data 1778, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.193 sec/step, elapsed 0:04:12, exp. remaining 1:25:23, complete 4.70%
att-weights epoch 560, step 88, max_size:classes 63, max_size:data 1832, mem_usage:GPU:0 0.9GB, num_seqs 2, 8.977 sec/step, elapsed 0:04:21, exp. remaining 1:26:59, complete 4.77%
att-weights epoch 560, step 89, max_size:classes 56, max_size:data 1655, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.816 sec/step, elapsed 0:04:22, exp. remaining 1:26:33, complete 4.81%
att-weights epoch 560, step 90, max_size:classes 56, max_size:data 2470, mem_usage:GPU:0 0.9GB, num_seqs 1, 6.545 sec/step, elapsed 0:04:28, exp. remaining 1:28:00, complete 4.85%
att-weights epoch 560, step 91, max_size:classes 59, max_size:data 1756, mem_usage:GPU:0 0.9GB, num_seqs 2, 4.627 sec/step, elapsed 0:04:33, exp. remaining 1:28:06, complete 4.92%
att-weights epoch 560, step 92, max_size:classes 61, max_size:data 1845, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.241 sec/step, elapsed 0:04:34, exp. remaining 1:27:48, complete 4.96%
att-weights epoch 560, step 93, max_size:classes 53, max_size:data 1747, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.212 sec/step, elapsed 0:04:36, exp. remaining 1:26:50, complete 5.03%
att-weights epoch 560, step 94, max_size:classes 53, max_size:data 1723, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.548 sec/step, elapsed 0:04:37, exp. remaining 1:26:39, complete 5.07%
att-weights epoch 560, step 95, max_size:classes 58, max_size:data 1593, mem_usage:GPU:0 0.9GB, num_seqs 2, 45.805 sec/step, elapsed 0:05:23, exp. remaining 1:40:11, complete 5.11%
att-weights epoch 560, step 96, max_size:classes 62, max_size:data 1939, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.058 sec/step, elapsed 0:05:25, exp. remaining 1:39:18, complete 5.18%
att-weights epoch 560, step 97, max_size:classes 60, max_size:data 1912, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.089 sec/step, elapsed 0:05:27, exp. remaining 1:38:27, complete 5.25%
att-weights epoch 560, step 98, max_size:classes 52, max_size:data 1629, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.938 sec/step, elapsed 0:05:28, exp. remaining 1:37:17, complete 5.33%
att-weights epoch 560, step 99, max_size:classes 55, max_size:data 2083, mem_usage:GPU:0 0.9GB, num_seqs 1, 10.286 sec/step, elapsed 0:05:38, exp. remaining 1:38:53, complete 5.40%
att-weights epoch 560, step 100, max_size:classes 62, max_size:data 1587, mem_usage:GPU:0 0.9GB, num_seqs 2, 7.768 sec/step, elapsed 0:05:46, exp. remaining 1:39:42, complete 5.48%
att-weights epoch 560, step 101, max_size:classes 61, max_size:data 2101, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.684 sec/step, elapsed 0:05:48, exp. remaining 1:38:46, complete 5.55%
att-weights epoch 560, step 102, max_size:classes 55, max_size:data 1729, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.257 sec/step, elapsed 0:05:50, exp. remaining 1:38:02, complete 5.62%
att-weights epoch 560, step 103, max_size:classes 58, max_size:data 2075, mem_usage:GPU:0 0.9GB, num_seqs 1, 15.929 sec/step, elapsed 0:06:06, exp. remaining 1:41:04, complete 5.70%
att-weights epoch 560, step 104, max_size:classes 54, max_size:data 2023, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.991 sec/step, elapsed 0:06:07, exp. remaining 1:39:58, complete 5.77%
att-weights epoch 560, step 105, max_size:classes 53, max_size:data 1786, mem_usage:GPU:0 0.9GB, num_seqs 2, 9.819 sec/step, elapsed 0:06:17, exp. remaining 1:41:16, complete 5.85%
att-weights epoch 560, step 106, max_size:classes 63, max_size:data 1779, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.636 sec/step, elapsed 0:06:18, exp. remaining 1:40:21, complete 5.92%
att-weights epoch 560, step 107, max_size:classes 58, max_size:data 1767, mem_usage:GPU:0 0.9GB, num_seqs 2, 3.807 sec/step, elapsed 0:06:22, exp. remaining 1:40:02, complete 5.99%
att-weights epoch 560, step 108, max_size:classes 54, max_size:data 1691, mem_usage:GPU:0 0.9GB, num_seqs 2, 5.310 sec/step, elapsed 0:06:27, exp. remaining 1:40:06, complete 6.07%
att-weights epoch 560, step 109, max_size:classes 56, max_size:data 1715, mem_usage:GPU:0 0.9GB, num_seqs 2, 4.906 sec/step, elapsed 0:06:32, exp. remaining 1:40:04, complete 6.14%
att-weights epoch 560, step 110, max_size:classes 61, max_size:data 1673, mem_usage:GPU:0 0.9GB, num_seqs 2, 3.277 sec/step, elapsed 0:06:36, exp. remaining 1:39:37, complete 6.22%
att-weights epoch 560, step 111, max_size:classes 54, max_size:data 1830, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.449 sec/step, elapsed 0:06:37, exp. remaining 1:38:44, complete 6.29%
att-weights epoch 560, step 112, max_size:classes 56, max_size:data 1814, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.667 sec/step, elapsed 0:06:39, exp. remaining 1:37:55, complete 6.36%
att-weights epoch 560, step 113, max_size:classes 55, max_size:data 1835, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.775 sec/step, elapsed 0:06:42, exp. remaining 1:37:23, complete 6.44%
att-weights epoch 560, step 114, max_size:classes 55, max_size:data 1641, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.349 sec/step, elapsed 0:06:43, exp. remaining 1:36:32, complete 6.51%
att-weights epoch 560, step 115, max_size:classes 49, max_size:data 1522, mem_usage:GPU:0 0.9GB, num_seqs 2, 3.207 sec/step, elapsed 0:06:46, exp. remaining 1:36:08, complete 6.59%
att-weights epoch 560, step 116, max_size:classes 49, max_size:data 1447, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.253 sec/step, elapsed 0:06:47, exp. remaining 1:35:17, complete 6.66%
att-weights epoch 560, step 117, max_size:classes 56, max_size:data 1834, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.270 sec/step, elapsed 0:06:49, exp. remaining 1:34:27, complete 6.73%
att-weights epoch 560, step 118, max_size:classes 55, max_size:data 1733, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.411 sec/step, elapsed 0:06:50, exp. remaining 1:33:40, complete 6.81%
att-weights epoch 560, step 119, max_size:classes 58, max_size:data 1553, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.529 sec/step, elapsed 0:06:52, exp. remaining 1:32:56, complete 6.88%
att-weights epoch 560, step 120, max_size:classes 57, max_size:data 1574, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.677 sec/step, elapsed 0:06:53, exp. remaining 1:32:15, complete 6.96%
att-weights epoch 560, step 121, max_size:classes 53, max_size:data 1440, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.160 sec/step, elapsed 0:06:54, exp. remaining 1:31:27, complete 7.03%
att-weights epoch 560, step 122, max_size:classes 60, max_size:data 1541, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.399 sec/step, elapsed 0:06:57, exp. remaining 1:30:57, complete 7.10%
att-weights epoch 560, step 123, max_size:classes 53, max_size:data 1710, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.245 sec/step, elapsed 0:06:58, exp. remaining 1:30:13, complete 7.18%
att-weights epoch 560, step 124, max_size:classes 58, max_size:data 1448, mem_usage:GPU:0 0.9GB, num_seqs 2, 3.208 sec/step, elapsed 0:07:01, exp. remaining 1:29:54, complete 7.25%
att-weights epoch 560, step 125, max_size:classes 50, max_size:data 1578, mem_usage:GPU:0 0.9GB, num_seqs 2, 4.628 sec/step, elapsed 0:07:06, exp. remaining 1:29:54, complete 7.33%
att-weights epoch 560, step 126, max_size:classes 53, max_size:data 1692, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.240 sec/step, elapsed 0:07:07, exp. remaining 1:29:11, complete 7.40%
att-weights epoch 560, step 127, max_size:classes 52, max_size:data 1571, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.111 sec/step, elapsed 0:07:08, exp. remaining 1:28:28, complete 7.47%
att-weights epoch 560, step 128, max_size:classes 54, max_size:data 1587, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.254 sec/step, elapsed 0:07:10, exp. remaining 1:27:47, complete 7.55%
att-weights epoch 560, step 129, max_size:classes 51, max_size:data 1523, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.066 sec/step, elapsed 0:07:11, exp. remaining 1:27:05, complete 7.62%
att-weights epoch 560, step 130, max_size:classes 50, max_size:data 1668, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.256 sec/step, elapsed 0:07:12, exp. remaining 1:26:25, complete 7.70%
att-weights epoch 560, step 131, max_size:classes 55, max_size:data 1492, mem_usage:GPU:0 0.9GB, num_seqs 2, 3.132 sec/step, elapsed 0:07:15, exp. remaining 1:26:09, complete 7.77%
att-weights epoch 560, step 132, max_size:classes 47, max_size:data 1534, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.950 sec/step, elapsed 0:07:18, exp. remaining 1:25:51, complete 7.84%
att-weights epoch 560, step 133, max_size:classes 50, max_size:data 1675, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.315 sec/step, elapsed 0:07:19, exp. remaining 1:25:14, complete 7.92%
att-weights epoch 560, step 134, max_size:classes 53, max_size:data 1433, mem_usage:GPU:0 0.9GB, num_seqs 2, 3.765 sec/step, elapsed 0:07:23, exp. remaining 1:25:06, complete 7.99%
att-weights epoch 560, step 135, max_size:classes 58, max_size:data 1680, mem_usage:GPU:0 0.9GB, num_seqs 2, 6.771 sec/step, elapsed 0:07:30, exp. remaining 1:25:32, complete 8.07%
att-weights epoch 560, step 136, max_size:classes 46, max_size:data 1676, mem_usage:GPU:0 0.9GB, num_seqs 2, 5.000 sec/step, elapsed 0:07:35, exp. remaining 1:25:38, complete 8.14%
att-weights epoch 560, step 137, max_size:classes 49, max_size:data 1587, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.126 sec/step, elapsed 0:07:36, exp. remaining 1:25:00, complete 8.21%
att-weights epoch 560, step 138, max_size:classes 56, max_size:data 1744, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.186 sec/step, elapsed 0:07:37, exp. remaining 1:24:24, complete 8.29%
att-weights epoch 560, step 139, max_size:classes 56, max_size:data 1516, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.095 sec/step, elapsed 0:07:38, exp. remaining 1:23:47, complete 8.36%
att-weights epoch 560, step 140, max_size:classes 48, max_size:data 1495, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.147 sec/step, elapsed 0:07:39, exp. remaining 1:23:11, complete 8.44%
att-weights epoch 560, step 141, max_size:classes 48, max_size:data 1596, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.356 sec/step, elapsed 0:07:41, exp. remaining 1:22:38, complete 8.51%
att-weights epoch 560, step 142, max_size:classes 52, max_size:data 1537, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.232 sec/step, elapsed 0:07:42, exp. remaining 1:22:05, complete 8.58%
att-weights epoch 560, step 143, max_size:classes 46, max_size:data 1493, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.195 sec/step, elapsed 0:07:43, exp. remaining 1:21:31, complete 8.66%
att-weights epoch 560, step 144, max_size:classes 46, max_size:data 1555, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.098 sec/step, elapsed 0:07:44, exp. remaining 1:20:57, complete 8.73%
att-weights epoch 560, step 145, max_size:classes 47, max_size:data 1416, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.968 sec/step, elapsed 0:07:45, exp. remaining 1:20:23, complete 8.81%
att-weights epoch 560, step 146, max_size:classes 48, max_size:data 1499, mem_usage:GPU:0 0.9GB, num_seqs 2, 4.386 sec/step, elapsed 0:07:50, exp. remaining 1:20:24, complete 8.88%
att-weights epoch 560, step 147, max_size:classes 51, max_size:data 1555, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.397 sec/step, elapsed 0:07:51, exp. remaining 1:19:54, complete 8.95%
att-weights epoch 560, step 148, max_size:classes 43, max_size:data 1255, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.305 sec/step, elapsed 0:07:52, exp. remaining 1:19:24, complete 9.03%
att-weights epoch 560, step 149, max_size:classes 47, max_size:data 1494, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.149 sec/step, elapsed 0:07:53, exp. remaining 1:18:53, complete 9.10%
att-weights epoch 560, step 150, max_size:classes 52, max_size:data 1416, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.975 sec/step, elapsed 0:07:54, exp. remaining 1:18:21, complete 9.17%
att-weights epoch 560, step 151, max_size:classes 46, max_size:data 1539, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.037 sec/step, elapsed 0:07:55, exp. remaining 1:17:49, complete 9.25%
att-weights epoch 560, step 152, max_size:classes 54, max_size:data 1448, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.988 sec/step, elapsed 0:07:56, exp. remaining 1:17:18, complete 9.32%
att-weights epoch 560, step 153, max_size:classes 45, max_size:data 1415, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.904 sec/step, elapsed 0:07:57, exp. remaining 1:16:47, complete 9.40%
att-weights epoch 560, step 154, max_size:classes 50, max_size:data 1465, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.367 sec/step, elapsed 0:07:59, exp. remaining 1:16:20, complete 9.47%
att-weights epoch 560, step 155, max_size:classes 46, max_size:data 1766, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.345 sec/step, elapsed 0:08:00, exp. remaining 1:15:53, complete 9.54%
att-weights epoch 560, step 156, max_size:classes 47, max_size:data 1809, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.639 sec/step, elapsed 0:08:02, exp. remaining 1:15:11, complete 9.66%
att-weights epoch 560, step 157, max_size:classes 50, max_size:data 1435, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.966 sec/step, elapsed 0:08:03, exp. remaining 1:14:23, complete 9.77%
att-weights epoch 560, step 158, max_size:classes 41, max_size:data 1401, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.196 sec/step, elapsed 0:08:04, exp. remaining 1:13:57, complete 9.84%
att-weights epoch 560, step 159, max_size:classes 49, max_size:data 1553, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.000 sec/step, elapsed 0:08:05, exp. remaining 1:13:29, complete 9.91%
att-weights epoch 560, step 160, max_size:classes 49, max_size:data 1196, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.058 sec/step, elapsed 0:08:06, exp. remaining 1:13:03, complete 9.99%
att-weights epoch 560, step 161, max_size:classes 47, max_size:data 1632, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.305 sec/step, elapsed 0:08:07, exp. remaining 1:12:38, complete 10.06%
att-weights epoch 560, step 162, max_size:classes 49, max_size:data 1557, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.189 sec/step, elapsed 0:08:08, exp. remaining 1:12:14, complete 10.14%
att-weights epoch 560, step 163, max_size:classes 47, max_size:data 1603, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.443 sec/step, elapsed 0:08:10, exp. remaining 1:11:51, complete 10.21%
att-weights epoch 560, step 164, max_size:classes 53, max_size:data 1677, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.119 sec/step, elapsed 0:08:11, exp. remaining 1:11:27, complete 10.28%
att-weights epoch 560, step 165, max_size:classes 50, max_size:data 1280, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.761 sec/step, elapsed 0:08:13, exp. remaining 1:10:51, complete 10.40%
att-weights epoch 560, step 166, max_size:classes 50, max_size:data 1330, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.235 sec/step, elapsed 0:08:15, exp. remaining 1:10:36, complete 10.47%
att-weights epoch 560, step 167, max_size:classes 43, max_size:data 1449, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.203 sec/step, elapsed 0:08:16, exp. remaining 1:10:13, complete 10.54%
att-weights epoch 560, step 168, max_size:classes 49, max_size:data 1584, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.231 sec/step, elapsed 0:08:17, exp. remaining 1:09:35, complete 10.65%
att-weights epoch 560, step 169, max_size:classes 49, max_size:data 1657, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.321 sec/step, elapsed 0:08:19, exp. remaining 1:09:13, complete 10.73%
att-weights epoch 560, step 170, max_size:classes 47, max_size:data 1361, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.339 sec/step, elapsed 0:08:20, exp. remaining 1:08:52, complete 10.80%
att-weights epoch 560, step 171, max_size:classes 45, max_size:data 1326, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.397 sec/step, elapsed 0:08:21, exp. remaining 1:08:17, complete 10.91%
att-weights epoch 560, step 172, max_size:classes 48, max_size:data 1405, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.777 sec/step, elapsed 0:08:23, exp. remaining 1:08:00, complete 10.99%
att-weights epoch 560, step 173, max_size:classes 47, max_size:data 1496, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.863 sec/step, elapsed 0:08:26, exp. remaining 1:07:37, complete 11.10%
att-weights epoch 560, step 174, max_size:classes 50, max_size:data 1327, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.499 sec/step, elapsed 0:08:28, exp. remaining 1:07:04, complete 11.21%
att-weights epoch 560, step 175, max_size:classes 47, max_size:data 1317, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.326 sec/step, elapsed 0:08:29, exp. remaining 1:06:30, complete 11.32%
att-weights epoch 560, step 176, max_size:classes 53, max_size:data 1431, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.382 sec/step, elapsed 0:08:30, exp. remaining 1:05:57, complete 11.43%
att-weights epoch 560, step 177, max_size:classes 43, max_size:data 1239, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.412 sec/step, elapsed 0:08:32, exp. remaining 1:05:25, complete 11.54%
att-weights epoch 560, step 178, max_size:classes 42, max_size:data 1140, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.185 sec/step, elapsed 0:08:33, exp. remaining 1:05:06, complete 11.62%
att-weights epoch 560, step 179, max_size:classes 44, max_size:data 1350, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.609 sec/step, elapsed 0:08:35, exp. remaining 1:04:50, complete 11.69%
att-weights epoch 560, step 180, max_size:classes 48, max_size:data 1319, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.433 sec/step, elapsed 0:08:36, exp. remaining 1:04:19, complete 11.80%
att-weights epoch 560, step 181, max_size:classes 47, max_size:data 1588, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.572 sec/step, elapsed 0:08:38, exp. remaining 1:03:50, complete 11.91%
att-weights epoch 560, step 182, max_size:classes 44, max_size:data 1320, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.408 sec/step, elapsed 0:08:39, exp. remaining 1:03:33, complete 11.99%
att-weights epoch 560, step 183, max_size:classes 47, max_size:data 1305, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.522 sec/step, elapsed 0:08:40, exp. remaining 1:03:18, complete 12.06%
att-weights epoch 560, step 184, max_size:classes 46, max_size:data 1199, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.421 sec/step, elapsed 0:08:42, exp. remaining 1:03:02, complete 12.13%
att-weights epoch 560, step 185, max_size:classes 46, max_size:data 1170, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.718 sec/step, elapsed 0:08:44, exp. remaining 1:02:48, complete 12.21%
att-weights epoch 560, step 186, max_size:classes 43, max_size:data 1198, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.517 sec/step, elapsed 0:08:45, exp. remaining 1:02:20, complete 12.32%
att-weights epoch 560, step 187, max_size:classes 40, max_size:data 1194, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.073 sec/step, elapsed 0:08:46, exp. remaining 1:01:50, complete 12.43%
att-weights epoch 560, step 188, max_size:classes 44, max_size:data 1512, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.025 sec/step, elapsed 0:08:47, exp. remaining 1:01:32, complete 12.50%
att-weights epoch 560, step 189, max_size:classes 46, max_size:data 1230, mem_usage:GPU:0 0.9GB, num_seqs 3, 3.218 sec/step, elapsed 0:08:50, exp. remaining 1:01:17, complete 12.62%
att-weights epoch 560, step 190, max_size:classes 48, max_size:data 1226, mem_usage:GPU:0 0.9GB, num_seqs 3, 4.487 sec/step, elapsed 0:08:55, exp. remaining 1:01:23, complete 12.69%
att-weights epoch 560, step 191, max_size:classes 47, max_size:data 1341, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.017 sec/step, elapsed 0:08:56, exp. remaining 1:00:54, complete 12.80%
att-weights epoch 560, step 192, max_size:classes 43, max_size:data 1687, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.043 sec/step, elapsed 0:08:57, exp. remaining 1:00:37, complete 12.87%
att-weights epoch 560, step 193, max_size:classes 48, max_size:data 1398, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.014 sec/step, elapsed 0:08:58, exp. remaining 1:00:20, complete 12.95%
att-weights epoch 560, step 194, max_size:classes 41, max_size:data 1399, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.369 sec/step, elapsed 0:08:59, exp. remaining 1:00:05, complete 13.02%
att-weights epoch 560, step 195, max_size:classes 43, max_size:data 1246, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.389 sec/step, elapsed 0:09:01, exp. remaining 0:59:51, complete 13.10%
att-weights epoch 560, step 196, max_size:classes 47, max_size:data 1167, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.314 sec/step, elapsed 0:09:02, exp. remaining 0:59:25, complete 13.21%
att-weights epoch 560, step 197, max_size:classes 46, max_size:data 1519, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.027 sec/step, elapsed 0:09:03, exp. remaining 0:58:57, complete 13.32%
att-weights epoch 560, step 198, max_size:classes 46, max_size:data 1277, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.297 sec/step, elapsed 0:09:04, exp. remaining 0:58:43, complete 13.39%
att-weights epoch 560, step 199, max_size:classes 51, max_size:data 1352, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.378 sec/step, elapsed 0:09:06, exp. remaining 0:58:30, complete 13.47%
att-weights epoch 560, step 200, max_size:classes 47, max_size:data 1181, mem_usage:GPU:0 0.9GB, num_seqs 3, 3.592 sec/step, elapsed 0:09:09, exp. remaining 0:58:30, complete 13.54%
att-weights epoch 560, step 201, max_size:classes 41, max_size:data 1175, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.996 sec/step, elapsed 0:09:10, exp. remaining 0:58:04, complete 13.65%
att-weights epoch 560, step 202, max_size:classes 39, max_size:data 1393, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.081 sec/step, elapsed 0:09:11, exp. remaining 0:57:49, complete 13.73%
att-weights epoch 560, step 203, max_size:classes 40, max_size:data 1140, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.070 sec/step, elapsed 0:09:13, exp. remaining 0:57:34, complete 13.80%
att-weights epoch 560, step 204, max_size:classes 37, max_size:data 1444, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.906 sec/step, elapsed 0:09:13, exp. remaining 0:57:08, complete 13.91%
att-weights epoch 560, step 205, max_size:classes 42, max_size:data 1295, mem_usage:GPU:0 0.9GB, num_seqs 3, 3.733 sec/step, elapsed 0:09:17, exp. remaining 0:56:59, complete 14.02%
att-weights epoch 560, step 206, max_size:classes 45, max_size:data 1285, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.731 sec/step, elapsed 0:09:19, exp. remaining 0:56:49, complete 14.10%
att-weights epoch 560, step 207, max_size:classes 41, max_size:data 1469, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.336 sec/step, elapsed 0:09:21, exp. remaining 0:56:42, complete 14.17%
att-weights epoch 560, step 208, max_size:classes 46, max_size:data 1237, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.059 sec/step, elapsed 0:09:22, exp. remaining 0:56:18, complete 14.28%
att-weights epoch 560, step 209, max_size:classes 38, max_size:data 1334, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.451 sec/step, elapsed 0:09:24, exp. remaining 0:55:56, complete 14.39%
att-weights epoch 560, step 210, max_size:classes 43, max_size:data 1232, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.392 sec/step, elapsed 0:09:25, exp. remaining 0:55:44, complete 14.47%
att-weights epoch 560, step 211, max_size:classes 53, max_size:data 1579, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.187 sec/step, elapsed 0:09:26, exp. remaining 0:55:21, complete 14.58%
att-weights epoch 560, step 212, max_size:classes 50, max_size:data 1600, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.230 sec/step, elapsed 0:09:28, exp. remaining 0:54:59, complete 14.69%
att-weights epoch 560, step 213, max_size:classes 41, max_size:data 1198, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.608 sec/step, elapsed 0:09:29, exp. remaining 0:54:39, complete 14.80%
att-weights epoch 560, step 214, max_size:classes 40, max_size:data 1203, mem_usage:GPU:0 0.9GB, num_seqs 3, 6.231 sec/step, elapsed 0:09:35, exp. remaining 0:54:56, complete 14.87%
att-weights epoch 560, step 215, max_size:classes 41, max_size:data 1374, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.053 sec/step, elapsed 0:09:36, exp. remaining 0:54:43, complete 14.95%
att-weights epoch 560, step 216, max_size:classes 40, max_size:data 1422, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.132 sec/step, elapsed 0:09:38, exp. remaining 0:54:21, complete 15.06%
att-weights epoch 560, step 217, max_size:classes 40, max_size:data 1188, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.024 sec/step, elapsed 0:09:40, exp. remaining 0:54:13, complete 15.13%
att-weights epoch 560, step 218, max_size:classes 42, max_size:data 1082, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.377 sec/step, elapsed 0:09:41, exp. remaining 0:53:53, complete 15.24%
att-weights epoch 560, step 219, max_size:classes 43, max_size:data 1361, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.930 sec/step, elapsed 0:09:42, exp. remaining 0:53:30, complete 15.35%
att-weights epoch 560, step 220, max_size:classes 46, max_size:data 1312, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.610 sec/step, elapsed 0:09:44, exp. remaining 0:53:21, complete 15.43%
att-weights epoch 560, step 221, max_size:classes 40, max_size:data 1319, mem_usage:GPU:0 0.9GB, num_seqs 3, 7.771 sec/step, elapsed 0:09:51, exp. remaining 0:53:45, complete 15.50%
att-weights epoch 560, step 222, max_size:classes 39, max_size:data 1158, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.524 sec/step, elapsed 0:09:53, exp. remaining 0:53:26, complete 15.61%
att-weights epoch 560, step 223, max_size:classes 41, max_size:data 1195, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.203 sec/step, elapsed 0:09:54, exp. remaining 0:53:06, complete 15.72%
att-weights epoch 560, step 224, max_size:classes 39, max_size:data 1544, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.300 sec/step, elapsed 0:09:55, exp. remaining 0:52:46, complete 15.83%
att-weights epoch 560, step 225, max_size:classes 42, max_size:data 1010, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.503 sec/step, elapsed 0:09:57, exp. remaining 0:52:28, complete 15.95%
att-weights epoch 560, step 226, max_size:classes 38, max_size:data 1579, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.045 sec/step, elapsed 0:09:58, exp. remaining 0:52:08, complete 16.06%
att-weights epoch 560, step 227, max_size:classes 38, max_size:data 1325, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.758 sec/step, elapsed 0:10:00, exp. remaining 0:51:51, complete 16.17%
att-weights epoch 560, step 228, max_size:classes 41, max_size:data 1032, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.475 sec/step, elapsed 0:10:01, exp. remaining 0:51:34, complete 16.28%
att-weights epoch 560, step 229, max_size:classes 36, max_size:data 1369, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.911 sec/step, elapsed 0:10:02, exp. remaining 0:51:13, complete 16.39%
att-weights epoch 560, step 230, max_size:classes 39, max_size:data 1456, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.191 sec/step, elapsed 0:10:03, exp. remaining 0:50:55, complete 16.50%
att-weights epoch 560, step 231, max_size:classes 38, max_size:data 1249, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.431 sec/step, elapsed 0:10:05, exp. remaining 0:50:45, complete 16.57%
att-weights epoch 560, step 232, max_size:classes 41, max_size:data 1230, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.390 sec/step, elapsed 0:10:06, exp. remaining 0:50:28, complete 16.69%
att-weights epoch 560, step 233, max_size:classes 36, max_size:data 1061, mem_usage:GPU:0 0.9GB, num_seqs 3, 4.362 sec/step, elapsed 0:10:10, exp. remaining 0:50:26, complete 16.80%
att-weights epoch 560, step 234, max_size:classes 40, max_size:data 1223, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.770 sec/step, elapsed 0:10:12, exp. remaining 0:50:10, complete 16.91%
att-weights epoch 560, step 235, max_size:classes 43, max_size:data 1068, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.463 sec/step, elapsed 0:10:14, exp. remaining 0:50:02, complete 16.98%
att-weights epoch 560, step 236, max_size:classes 39, max_size:data 1282, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.634 sec/step, elapsed 0:10:15, exp. remaining 0:49:54, complete 17.06%
att-weights epoch 560, step 237, max_size:classes 44, max_size:data 1278, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.993 sec/step, elapsed 0:10:17, exp. remaining 0:49:40, complete 17.17%
att-weights epoch 560, step 238, max_size:classes 39, max_size:data 1180, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.608 sec/step, elapsed 0:10:19, exp. remaining 0:49:25, complete 17.28%
att-weights epoch 560, step 239, max_size:classes 36, max_size:data 1151, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.610 sec/step, elapsed 0:10:20, exp. remaining 0:49:10, complete 17.39%
att-weights epoch 560, step 240, max_size:classes 40, max_size:data 1476, mem_usage:GPU:0 0.9GB, num_seqs 2, 7.092 sec/step, elapsed 0:10:28, exp. remaining 0:49:21, complete 17.50%
att-weights epoch 560, step 241, max_size:classes 40, max_size:data 1017, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.429 sec/step, elapsed 0:10:29, exp. remaining 0:49:05, complete 17.61%
att-weights epoch 560, step 242, max_size:classes 38, max_size:data 1216, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.127 sec/step, elapsed 0:10:30, exp. remaining 0:48:47, complete 17.72%
att-weights epoch 560, step 243, max_size:classes 37, max_size:data 1221, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.522 sec/step, elapsed 0:10:32, exp. remaining 0:48:32, complete 17.83%
att-weights epoch 560, step 244, max_size:classes 37, max_size:data 1074, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.941 sec/step, elapsed 0:10:33, exp. remaining 0:48:22, complete 17.91%
att-weights epoch 560, step 245, max_size:classes 38, max_size:data 1409, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.082 sec/step, elapsed 0:10:34, exp. remaining 0:48:05, complete 18.02%
att-weights epoch 560, step 246, max_size:classes 38, max_size:data 1042, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.320 sec/step, elapsed 0:10:35, exp. remaining 0:47:50, complete 18.13%
att-weights epoch 560, step 247, max_size:classes 39, max_size:data 1126, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.287 sec/step, elapsed 0:10:36, exp. remaining 0:47:34, complete 18.24%
att-weights epoch 560, step 248, max_size:classes 39, max_size:data 1221, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.506 sec/step, elapsed 0:10:38, exp. remaining 0:47:20, complete 18.35%
att-weights epoch 560, step 249, max_size:classes 40, max_size:data 1042, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.350 sec/step, elapsed 0:10:39, exp. remaining 0:47:05, complete 18.46%
att-weights epoch 560, step 250, max_size:classes 37, max_size:data 1124, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.234 sec/step, elapsed 0:10:40, exp. remaining 0:46:49, complete 18.57%
att-weights epoch 560, step 251, max_size:classes 38, max_size:data 1127, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.350 sec/step, elapsed 0:10:42, exp. remaining 0:46:35, complete 18.68%
att-weights epoch 560, step 252, max_size:classes 42, max_size:data 1101, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.314 sec/step, elapsed 0:10:43, exp. remaining 0:46:20, complete 18.79%
att-weights epoch 560, step 253, max_size:classes 41, max_size:data 1442, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.107 sec/step, elapsed 0:10:44, exp. remaining 0:46:05, complete 18.90%
att-weights epoch 560, step 254, max_size:classes 35, max_size:data 1062, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.321 sec/step, elapsed 0:10:45, exp. remaining 0:45:50, complete 19.02%
att-weights epoch 560, step 255, max_size:classes 33, max_size:data 1097, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.394 sec/step, elapsed 0:10:47, exp. remaining 0:45:37, complete 19.13%
att-weights epoch 560, step 256, max_size:classes 37, max_size:data 1030, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.372 sec/step, elapsed 0:10:48, exp. remaining 0:45:23, complete 19.24%
att-weights epoch 560, step 257, max_size:classes 36, max_size:data 1247, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.807 sec/step, elapsed 0:10:50, exp. remaining 0:45:11, complete 19.35%
att-weights epoch 560, step 258, max_size:classes 39, max_size:data 1114, mem_usage:GPU:0 0.9GB, num_seqs 3, 6.422 sec/step, elapsed 0:10:56, exp. remaining 0:45:18, complete 19.46%
att-weights epoch 560, step 259, max_size:classes 32, max_size:data 1299, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.732 sec/step, elapsed 0:10:58, exp. remaining 0:45:06, complete 19.57%
att-weights epoch 560, step 260, max_size:classes 38, max_size:data 1090, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.679 sec/step, elapsed 0:11:00, exp. remaining 0:44:54, complete 19.68%
att-weights epoch 560, step 261, max_size:classes 39, max_size:data 1119, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.557 sec/step, elapsed 0:11:01, exp. remaining 0:44:42, complete 19.79%
att-weights epoch 560, step 262, max_size:classes 39, max_size:data 1205, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.581 sec/step, elapsed 0:11:03, exp. remaining 0:44:30, complete 19.90%
att-weights epoch 560, step 263, max_size:classes 39, max_size:data 1021, mem_usage:GPU:0 0.9GB, num_seqs 3, 6.189 sec/step, elapsed 0:11:09, exp. remaining 0:44:42, complete 19.98%
att-weights epoch 560, step 264, max_size:classes 39, max_size:data 1039, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.639 sec/step, elapsed 0:11:11, exp. remaining 0:44:36, complete 20.05%
att-weights epoch 560, step 265, max_size:classes 37, max_size:data 1031, mem_usage:GPU:0 0.9GB, num_seqs 3, 6.763 sec/step, elapsed 0:11:18, exp. remaining 0:44:45, complete 20.16%
att-weights epoch 560, step 266, max_size:classes 36, max_size:data 1107, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.157 sec/step, elapsed 0:11:20, exp. remaining 0:44:35, complete 20.27%
att-weights epoch 560, step 267, max_size:classes 35, max_size:data 1025, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.864 sec/step, elapsed 0:11:23, exp. remaining 0:44:28, complete 20.38%
att-weights epoch 560, step 268, max_size:classes 36, max_size:data 1143, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.331 sec/step, elapsed 0:11:25, exp. remaining 0:44:12, complete 20.53%
att-weights epoch 560, step 269, max_size:classes 40, max_size:data 1129, mem_usage:GPU:0 0.9GB, num_seqs 3, 9.214 sec/step, elapsed 0:11:34, exp. remaining 0:44:30, complete 20.64%
att-weights epoch 560, step 270, max_size:classes 37, max_size:data 995, mem_usage:GPU:0 0.9GB, num_seqs 3, 7.934 sec/step, elapsed 0:11:42, exp. remaining 0:44:42, complete 20.75%
att-weights epoch 560, step 271, max_size:classes 33, max_size:data 1148, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.144 sec/step, elapsed 0:11:44, exp. remaining 0:44:32, complete 20.87%
att-weights epoch 560, step 272, max_size:classes 38, max_size:data 985, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.476 sec/step, elapsed 0:11:46, exp. remaining 0:44:26, complete 20.94%
att-weights epoch 560, step 273, max_size:classes 37, max_size:data 1382, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.795 sec/step, elapsed 0:11:48, exp. remaining 0:44:15, complete 21.05%
att-weights epoch 560, step 274, max_size:classes 40, max_size:data 1047, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.330 sec/step, elapsed 0:11:49, exp. remaining 0:44:02, complete 21.16%
att-weights epoch 560, step 275, max_size:classes 37, max_size:data 1082, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.308 sec/step, elapsed 0:11:50, exp. remaining 0:43:50, complete 21.27%
att-weights epoch 560, step 276, max_size:classes 38, max_size:data 1196, mem_usage:GPU:0 0.9GB, num_seqs 3, 25.320 sec/step, elapsed 0:12:16, exp. remaining 0:45:06, complete 21.38%
att-weights epoch 560, step 277, max_size:classes 36, max_size:data 977, mem_usage:GPU:0 0.9GB, num_seqs 4, 17.094 sec/step, elapsed 0:12:33, exp. remaining 0:45:50, complete 21.49%
att-weights epoch 560, step 278, max_size:classes 36, max_size:data 1026, mem_usage:GPU:0 0.9GB, num_seqs 3, 70.984 sec/step, elapsed 0:13:44, exp. remaining 0:49:50, complete 21.61%
att-weights epoch 560, step 279, max_size:classes 36, max_size:data 1071, mem_usage:GPU:0 0.9GB, num_seqs 3, 27.219 sec/step, elapsed 0:14:11, exp. remaining 0:51:08, complete 21.72%
att-weights epoch 560, step 280, max_size:classes 33, max_size:data 1195, mem_usage:GPU:0 0.9GB, num_seqs 3, 54.375 sec/step, elapsed 0:15:05, exp. remaining 0:54:03, complete 21.83%
att-weights epoch 560, step 281, max_size:classes 36, max_size:data 1415, mem_usage:GPU:0 0.9GB, num_seqs 2, 7.125 sec/step, elapsed 0:15:12, exp. remaining 0:54:08, complete 21.94%
att-weights epoch 560, step 282, max_size:classes 34, max_size:data 1105, mem_usage:GPU:0 0.9GB, num_seqs 3, 15.969 sec/step, elapsed 0:15:28, exp. remaining 0:54:43, complete 22.05%
att-weights epoch 560, step 283, max_size:classes 41, max_size:data 1216, mem_usage:GPU:0 0.9GB, num_seqs 3, 5.752 sec/step, elapsed 0:15:34, exp. remaining 0:54:35, complete 22.20%
att-weights epoch 560, step 284, max_size:classes 35, max_size:data 1162, mem_usage:GPU:0 0.9GB, num_seqs 3, 5.895 sec/step, elapsed 0:15:40, exp. remaining 0:54:35, complete 22.31%
att-weights epoch 560, step 285, max_size:classes 35, max_size:data 1086, mem_usage:GPU:0 0.9GB, num_seqs 3, 9.257 sec/step, elapsed 0:15:49, exp. remaining 0:54:46, complete 22.42%
att-weights epoch 560, step 286, max_size:classes 39, max_size:data 1281, mem_usage:GPU:0 0.9GB, num_seqs 3, 4.791 sec/step, elapsed 0:15:54, exp. remaining 0:54:42, complete 22.53%
att-weights epoch 560, step 287, max_size:classes 35, max_size:data 1253, mem_usage:GPU:0 0.9GB, num_seqs 3, 8.088 sec/step, elapsed 0:16:02, exp. remaining 0:54:42, complete 22.68%
att-weights epoch 560, step 288, max_size:classes 32, max_size:data 1089, mem_usage:GPU:0 0.9GB, num_seqs 3, 8.258 sec/step, elapsed 0:16:10, exp. remaining 0:54:49, complete 22.79%
att-weights epoch 560, step 289, max_size:classes 34, max_size:data 1174, mem_usage:GPU:0 0.9GB, num_seqs 3, 4.787 sec/step, elapsed 0:16:15, exp. remaining 0:54:44, complete 22.90%
att-weights epoch 560, step 290, max_size:classes 36, max_size:data 1166, mem_usage:GPU:0 0.9GB, num_seqs 3, 5.029 sec/step, elapsed 0:16:20, exp. remaining 0:54:41, complete 23.01%
att-weights epoch 560, step 291, max_size:classes 36, max_size:data 1074, mem_usage:GPU:0 0.9GB, num_seqs 3, 3.575 sec/step, elapsed 0:16:24, exp. remaining 0:54:32, complete 23.12%
att-weights epoch 560, step 292, max_size:classes 37, max_size:data 936, mem_usage:GPU:0 0.9GB, num_seqs 4, 5.960 sec/step, elapsed 0:16:30, exp. remaining 0:54:31, complete 23.23%
att-weights epoch 560, step 293, max_size:classes 35, max_size:data 1147, mem_usage:GPU:0 0.9GB, num_seqs 3, 3.384 sec/step, elapsed 0:16:33, exp. remaining 0:54:22, complete 23.34%
att-weights epoch 560, step 294, max_size:classes 36, max_size:data 1178, mem_usage:GPU:0 0.9GB, num_seqs 3, 4.373 sec/step, elapsed 0:16:37, exp. remaining 0:54:16, complete 23.46%
att-weights epoch 560, step 295, max_size:classes 33, max_size:data 1079, mem_usage:GPU:0 0.9GB, num_seqs 3, 4.980 sec/step, elapsed 0:16:42, exp. remaining 0:54:12, complete 23.57%
att-weights epoch 560, step 296, max_size:classes 36, max_size:data 1000, mem_usage:GPU:0 0.9GB, num_seqs 4, 6.267 sec/step, elapsed 0:16:49, exp. remaining 0:54:06, complete 23.71%
att-weights epoch 560, step 297, max_size:classes 36, max_size:data 1028, mem_usage:GPU:0 0.9GB, num_seqs 3, 10.696 sec/step, elapsed 0:16:59, exp. remaining 0:54:14, complete 23.86%
att-weights epoch 560, step 298, max_size:classes 33, max_size:data 1009, mem_usage:GPU:0 0.9GB, num_seqs 3, 3.773 sec/step, elapsed 0:17:03, exp. remaining 0:54:06, complete 23.97%
att-weights epoch 560, step 299, max_size:classes 36, max_size:data 1113, mem_usage:GPU:0 0.9GB, num_seqs 3, 3.866 sec/step, elapsed 0:17:07, exp. remaining 0:53:59, complete 24.08%
att-weights epoch 560, step 300, max_size:classes 37, max_size:data 1033, mem_usage:GPU:0 0.9GB, num_seqs 3, 8.502 sec/step, elapsed 0:17:16, exp. remaining 0:54:06, complete 24.20%
att-weights epoch 560, step 301, max_size:classes 35, max_size:data 1207, mem_usage:GPU:0 0.9GB, num_seqs 3, 14.809 sec/step, elapsed 0:17:30, exp. remaining 0:54:26, complete 24.34%
att-weights epoch 560, step 302, max_size:classes 35, max_size:data 1272, mem_usage:GPU:0 0.9GB, num_seqs 3, 14.269 sec/step, elapsed 0:17:45, exp. remaining 0:54:50, complete 24.45%
att-weights epoch 560, step 303, max_size:classes 34, max_size:data 1048, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.896 sec/step, elapsed 0:17:47, exp. remaining 0:54:36, complete 24.57%
att-weights epoch 560, step 304, max_size:classes 35, max_size:data 1012, mem_usage:GPU:0 0.9GB, num_seqs 3, 6.581 sec/step, elapsed 0:17:53, exp. remaining 0:54:30, complete 24.71%
att-weights epoch 560, step 305, max_size:classes 33, max_size:data 983, mem_usage:GPU:0 0.9GB, num_seqs 4, 5.950 sec/step, elapsed 0:17:59, exp. remaining 0:54:29, complete 24.82%
att-weights epoch 560, step 306, max_size:classes 38, max_size:data 959, mem_usage:GPU:0 0.9GB, num_seqs 4, 10.270 sec/step, elapsed 0:18:09, exp. remaining 0:54:41, complete 24.94%
att-weights epoch 560, step 307, max_size:classes 36, max_size:data 1060, mem_usage:GPU:0 0.9GB, num_seqs 3, 11.802 sec/step, elapsed 0:18:21, exp. remaining 0:54:56, complete 25.05%
att-weights epoch 560, step 308, max_size:classes 34, max_size:data 1031, mem_usage:GPU:0 0.9GB, num_seqs 3, 8.664 sec/step, elapsed 0:18:30, exp. remaining 0:55:03, complete 25.16%
att-weights epoch 560, step 309, max_size:classes 31, max_size:data 1115, mem_usage:GPU:0 0.9GB, num_seqs 3, 10.824 sec/step, elapsed 0:18:41, exp. remaining 0:55:15, complete 25.27%
att-weights epoch 560, step 310, max_size:classes 32, max_size:data 951, mem_usage:GPU:0 0.9GB, num_seqs 4, 6.049 sec/step, elapsed 0:18:47, exp. remaining 0:55:14, complete 25.38%
att-weights epoch 560, step 311, max_size:classes 35, max_size:data 1201, mem_usage:GPU:0 0.9GB, num_seqs 3, 4.924 sec/step, elapsed 0:18:52, exp. remaining 0:55:02, complete 25.53%
att-weights epoch 560, step 312, max_size:classes 30, max_size:data 1077, mem_usage:GPU:0 0.9GB, num_seqs 3, 3.173 sec/step, elapsed 0:18:55, exp. remaining 0:54:46, complete 25.68%
att-weights epoch 560, step 313, max_size:classes 34, max_size:data 862, mem_usage:GPU:0 0.9GB, num_seqs 4, 8.885 sec/step, elapsed 0:19:04, exp. remaining 0:54:46, complete 25.82%
att-weights epoch 560, step 314, max_size:classes 32, max_size:data 873, mem_usage:GPU:0 0.9GB, num_seqs 3, 3.725 sec/step, elapsed 0:19:07, exp. remaining 0:54:38, complete 25.93%
att-weights epoch 560, step 315, max_size:classes 32, max_size:data 1028, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.347 sec/step, elapsed 0:19:09, exp. remaining 0:54:23, complete 26.05%
att-weights epoch 560, step 316, max_size:classes 30, max_size:data 898, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.097 sec/step, elapsed 0:19:10, exp. remaining 0:54:07, complete 26.16%
att-weights epoch 560, step 317, max_size:classes 33, max_size:data 1116, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.585 sec/step, elapsed 0:19:11, exp. remaining 0:53:53, complete 26.27%
att-weights epoch 560, step 318, max_size:classes 35, max_size:data 1019, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.257 sec/step, elapsed 0:19:13, exp. remaining 0:53:38, complete 26.38%
att-weights epoch 560, step 319, max_size:classes 29, max_size:data 1125, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.183 sec/step, elapsed 0:19:14, exp. remaining 0:53:17, complete 26.53%
att-weights epoch 560, step 320, max_size:classes 35, max_size:data 938, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.006 sec/step, elapsed 0:19:16, exp. remaining 0:53:05, complete 26.64%
att-weights epoch 560, step 321, max_size:classes 37, max_size:data 910, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.687 sec/step, elapsed 0:19:18, exp. remaining 0:52:51, complete 26.75%
att-weights epoch 560, step 322, max_size:classes 34, max_size:data 883, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.932 sec/step, elapsed 0:19:20, exp. remaining 0:52:33, complete 26.90%
att-weights epoch 560, step 323, max_size:classes 38, max_size:data 1218, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.578 sec/step, elapsed 0:19:21, exp. remaining 0:52:13, complete 27.04%
att-weights epoch 560, step 324, max_size:classes 32, max_size:data 1008, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.894 sec/step, elapsed 0:19:23, exp. remaining 0:51:55, complete 27.19%
att-weights epoch 560, step 325, max_size:classes 31, max_size:data 1022, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.344 sec/step, elapsed 0:19:24, exp. remaining 0:51:35, complete 27.34%
att-weights epoch 560, step 326, max_size:classes 35, max_size:data 1019, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.634 sec/step, elapsed 0:19:26, exp. remaining 0:51:17, complete 27.49%
att-weights epoch 560, step 327, max_size:classes 31, max_size:data 1123, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.513 sec/step, elapsed 0:19:28, exp. remaining 0:50:58, complete 27.64%
att-weights epoch 560, step 328, max_size:classes 33, max_size:data 919, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.857 sec/step, elapsed 0:19:29, exp. remaining 0:50:40, complete 27.78%
att-weights epoch 560, step 329, max_size:classes 33, max_size:data 1066, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.464 sec/step, elapsed 0:19:31, exp. remaining 0:50:27, complete 27.89%
att-weights epoch 560, step 330, max_size:classes 35, max_size:data 1098, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.764 sec/step, elapsed 0:19:33, exp. remaining 0:50:15, complete 28.01%
att-weights epoch 560, step 331, max_size:classes 30, max_size:data 975, mem_usage:GPU:0 0.9GB, num_seqs 4, 6.399 sec/step, elapsed 0:19:39, exp. remaining 0:50:15, complete 28.12%
att-weights epoch 560, step 332, max_size:classes 32, max_size:data 910, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.115 sec/step, elapsed 0:19:41, exp. remaining 0:49:58, complete 28.26%
att-weights epoch 560, step 333, max_size:classes 36, max_size:data 918, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.764 sec/step, elapsed 0:19:44, exp. remaining 0:49:44, complete 28.41%
att-weights epoch 560, step 334, max_size:classes 33, max_size:data 993, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.688 sec/step, elapsed 0:19:47, exp. remaining 0:49:29, complete 28.56%
att-weights epoch 560, step 335, max_size:classes 34, max_size:data 991, mem_usage:GPU:0 0.9GB, num_seqs 4, 4.776 sec/step, elapsed 0:19:51, exp. remaining 0:49:19, complete 28.71%
att-weights epoch 560, step 336, max_size:classes 31, max_size:data 954, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.692 sec/step, elapsed 0:19:54, exp. remaining 0:49:10, complete 28.82%
att-weights epoch 560, step 337, max_size:classes 30, max_size:data 935, mem_usage:GPU:0 0.9GB, num_seqs 4, 16.713 sec/step, elapsed 0:20:11, exp. remaining 0:49:30, complete 28.97%
att-weights epoch 560, step 338, max_size:classes 30, max_size:data 1120, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.475 sec/step, elapsed 0:20:13, exp. remaining 0:49:15, complete 29.12%
att-weights epoch 560, step 339, max_size:classes 31, max_size:data 1152, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.492 sec/step, elapsed 0:20:15, exp. remaining 0:48:57, complete 29.26%
att-weights epoch 560, step 340, max_size:classes 31, max_size:data 1034, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.411 sec/step, elapsed 0:20:16, exp. remaining 0:48:45, complete 29.37%
att-weights epoch 560, step 341, max_size:classes 31, max_size:data 977, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.385 sec/step, elapsed 0:20:18, exp. remaining 0:48:27, complete 29.52%
att-weights epoch 560, step 342, max_size:classes 36, max_size:data 963, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.990 sec/step, elapsed 0:20:20, exp. remaining 0:48:11, complete 29.67%
att-weights epoch 560, step 343, max_size:classes 30, max_size:data 924, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.648 sec/step, elapsed 0:20:21, exp. remaining 0:48:00, complete 29.78%
att-weights epoch 560, step 344, max_size:classes 29, max_size:data 953, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.206 sec/step, elapsed 0:20:23, exp. remaining 0:47:45, complete 29.93%
att-weights epoch 560, step 345, max_size:classes 33, max_size:data 1010, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.272 sec/step, elapsed 0:20:25, exp. remaining 0:47:33, complete 30.04%
att-weights epoch 560, step 346, max_size:classes 33, max_size:data 852, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.686 sec/step, elapsed 0:20:26, exp. remaining 0:47:22, complete 30.15%
att-weights epoch 560, step 347, max_size:classes 31, max_size:data 856, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.808 sec/step, elapsed 0:20:28, exp. remaining 0:47:11, complete 30.26%
att-weights epoch 560, step 348, max_size:classes 30, max_size:data 883, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.608 sec/step, elapsed 0:20:30, exp. remaining 0:46:55, complete 30.41%
att-weights epoch 560, step 349, max_size:classes 31, max_size:data 1062, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.757 sec/step, elapsed 0:20:32, exp. remaining 0:46:39, complete 30.56%
att-weights epoch 560, step 350, max_size:classes 33, max_size:data 892, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.839 sec/step, elapsed 0:20:33, exp. remaining 0:46:29, complete 30.67%
att-weights epoch 560, step 351, max_size:classes 27, max_size:data 912, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.575 sec/step, elapsed 0:20:35, exp. remaining 0:46:13, complete 30.82%
att-weights epoch 560, step 352, max_size:classes 28, max_size:data 1007, mem_usage:GPU:0 0.9GB, num_seqs 3, 6.008 sec/step, elapsed 0:20:41, exp. remaining 0:46:07, complete 30.97%
att-weights epoch 560, step 353, max_size:classes 32, max_size:data 954, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.187 sec/step, elapsed 0:20:43, exp. remaining 0:45:53, complete 31.11%
att-weights epoch 560, step 354, max_size:classes 28, max_size:data 820, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.496 sec/step, elapsed 0:20:45, exp. remaining 0:45:42, complete 31.22%
att-weights epoch 560, step 355, max_size:classes 30, max_size:data 1149, mem_usage:GPU:0 0.9GB, num_seqs 3, 5.859 sec/step, elapsed 0:20:51, exp. remaining 0:45:36, complete 31.37%
att-weights epoch 560, step 356, max_size:classes 31, max_size:data 1015, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.081 sec/step, elapsed 0:20:53, exp. remaining 0:45:22, complete 31.52%
att-weights epoch 560, step 357, max_size:classes 31, max_size:data 969, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.924 sec/step, elapsed 0:20:55, exp. remaining 0:45:07, complete 31.67%
att-weights epoch 560, step 358, max_size:classes 29, max_size:data 869, mem_usage:GPU:0 0.9GB, num_seqs 4, 4.140 sec/step, elapsed 0:20:59, exp. remaining 0:44:58, complete 31.82%
att-weights epoch 560, step 359, max_size:classes 29, max_size:data 1056, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.630 sec/step, elapsed 0:21:00, exp. remaining 0:44:43, complete 31.96%
att-weights epoch 560, step 360, max_size:classes 31, max_size:data 883, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.475 sec/step, elapsed 0:21:02, exp. remaining 0:44:28, complete 32.11%
att-weights epoch 560, step 361, max_size:classes 36, max_size:data 982, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.739 sec/step, elapsed 0:21:04, exp. remaining 0:44:18, complete 32.22%
att-weights epoch 560, step 362, max_size:classes 30, max_size:data 890, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.280 sec/step, elapsed 0:21:06, exp. remaining 0:44:09, complete 32.33%
att-weights epoch 560, step 363, max_size:classes 29, max_size:data 1029, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.588 sec/step, elapsed 0:21:07, exp. remaining 0:43:55, complete 32.48%
att-weights epoch 560, step 364, max_size:classes 31, max_size:data 734, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.731 sec/step, elapsed 0:21:09, exp. remaining 0:43:41, complete 32.63%
att-weights epoch 560, step 365, max_size:classes 28, max_size:data 922, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.873 sec/step, elapsed 0:21:11, exp. remaining 0:43:27, complete 32.78%
att-weights epoch 560, step 366, max_size:classes 30, max_size:data 926, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.648 sec/step, elapsed 0:21:13, exp. remaining 0:43:13, complete 32.93%
att-weights epoch 560, step 367, max_size:classes 32, max_size:data 880, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.549 sec/step, elapsed 0:21:14, exp. remaining 0:42:59, complete 33.07%
att-weights epoch 560, step 368, max_size:classes 30, max_size:data 782, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.750 sec/step, elapsed 0:21:16, exp. remaining 0:42:45, complete 33.22%
att-weights epoch 560, step 369, max_size:classes 30, max_size:data 936, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.718 sec/step, elapsed 0:21:19, exp. remaining 0:42:34, complete 33.37%
att-weights epoch 560, step 370, max_size:classes 34, max_size:data 919, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.625 sec/step, elapsed 0:21:20, exp. remaining 0:42:20, complete 33.52%
att-weights epoch 560, step 371, max_size:classes 28, max_size:data 1001, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.315 sec/step, elapsed 0:21:22, exp. remaining 0:42:06, complete 33.67%
att-weights epoch 560, step 372, max_size:classes 28, max_size:data 833, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.352 sec/step, elapsed 0:21:23, exp. remaining 0:41:52, complete 33.81%
att-weights epoch 560, step 373, max_size:classes 30, max_size:data 947, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.841 sec/step, elapsed 0:21:25, exp. remaining 0:41:39, complete 33.96%
att-weights epoch 560, step 374, max_size:classes 31, max_size:data 930, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.858 sec/step, elapsed 0:21:28, exp. remaining 0:41:32, complete 34.07%
att-weights epoch 560, step 375, max_size:classes 27, max_size:data 954, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.883 sec/step, elapsed 0:21:30, exp. remaining 0:41:19, complete 34.22%
att-weights epoch 560, step 376, max_size:classes 29, max_size:data 886, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.886 sec/step, elapsed 0:21:31, exp. remaining 0:41:07, complete 34.37%
att-weights epoch 560, step 377, max_size:classes 27, max_size:data 986, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.215 sec/step, elapsed 0:21:34, exp. remaining 0:40:55, complete 34.52%
att-weights epoch 560, step 378, max_size:classes 32, max_size:data 808, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.740 sec/step, elapsed 0:21:35, exp. remaining 0:40:42, complete 34.67%
att-weights epoch 560, step 379, max_size:classes 30, max_size:data 845, mem_usage:GPU:0 0.9GB, num_seqs 4, 7.447 sec/step, elapsed 0:21:43, exp. remaining 0:40:40, complete 34.81%
att-weights epoch 560, step 380, max_size:classes 28, max_size:data 976, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.115 sec/step, elapsed 0:21:45, exp. remaining 0:40:32, complete 34.92%
att-weights epoch 560, step 381, max_size:classes 32, max_size:data 869, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.154 sec/step, elapsed 0:21:47, exp. remaining 0:40:20, complete 35.07%
att-weights epoch 560, step 382, max_size:classes 31, max_size:data 929, mem_usage:GPU:0 0.9GB, num_seqs 4, 9.234 sec/step, elapsed 0:21:56, exp. remaining 0:40:22, complete 35.22%
att-weights epoch 560, step 383, max_size:classes 29, max_size:data 1024, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.524 sec/step, elapsed 0:21:59, exp. remaining 0:40:07, complete 35.41%
att-weights epoch 560, step 384, max_size:classes 27, max_size:data 970, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.755 sec/step, elapsed 0:22:01, exp. remaining 0:39:54, complete 35.55%
att-weights epoch 560, step 385, max_size:classes 29, max_size:data 985, mem_usage:GPU:0 0.9GB, num_seqs 4, 6.911 sec/step, elapsed 0:22:08, exp. remaining 0:39:51, complete 35.70%
att-weights epoch 560, step 386, max_size:classes 29, max_size:data 859, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.643 sec/step, elapsed 0:22:09, exp. remaining 0:39:39, complete 35.85%
att-weights epoch 560, step 387, max_size:classes 30, max_size:data 864, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.318 sec/step, elapsed 0:22:11, exp. remaining 0:39:32, complete 35.96%
att-weights epoch 560, step 388, max_size:classes 28, max_size:data 884, mem_usage:GPU:0 0.9GB, num_seqs 4, 6.510 sec/step, elapsed 0:22:18, exp. remaining 0:39:32, complete 36.07%
att-weights epoch 560, step 389, max_size:classes 30, max_size:data 1108, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.511 sec/step, elapsed 0:22:20, exp. remaining 0:39:23, complete 36.18%
att-weights epoch 560, step 390, max_size:classes 28, max_size:data 827, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.120 sec/step, elapsed 0:22:22, exp. remaining 0:39:12, complete 36.33%
att-weights epoch 560, step 391, max_size:classes 28, max_size:data 884, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.765 sec/step, elapsed 0:22:23, exp. remaining 0:39:00, complete 36.48%
att-weights epoch 560, step 392, max_size:classes 29, max_size:data 767, mem_usage:GPU:0 0.9GB, num_seqs 5, 3.833 sec/step, elapsed 0:22:27, exp. remaining 0:38:48, complete 36.66%
att-weights epoch 560, step 393, max_size:classes 29, max_size:data 820, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.626 sec/step, elapsed 0:22:30, exp. remaining 0:38:38, complete 36.81%
att-weights epoch 560, step 394, max_size:classes 30, max_size:data 888, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.715 sec/step, elapsed 0:22:32, exp. remaining 0:38:26, complete 36.96%
att-weights epoch 560, step 395, max_size:classes 30, max_size:data 896, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.419 sec/step, elapsed 0:22:34, exp. remaining 0:38:15, complete 37.11%
att-weights epoch 560, step 396, max_size:classes 28, max_size:data 761, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.279 sec/step, elapsed 0:22:35, exp. remaining 0:37:59, complete 37.29%
att-weights epoch 560, step 397, max_size:classes 30, max_size:data 1015, mem_usage:GPU:0 0.9GB, num_seqs 3, 9.581 sec/step, elapsed 0:22:45, exp. remaining 0:38:01, complete 37.44%
att-weights epoch 560, step 398, max_size:classes 29, max_size:data 1067, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.743 sec/step, elapsed 0:22:47, exp. remaining 0:37:49, complete 37.59%
att-weights epoch 560, step 399, max_size:classes 30, max_size:data 845, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.530 sec/step, elapsed 0:22:49, exp. remaining 0:37:39, complete 37.74%
att-weights epoch 560, step 400, max_size:classes 29, max_size:data 836, mem_usage:GPU:0 0.9GB, num_seqs 4, 19.117 sec/step, elapsed 0:23:08, exp. remaining 0:37:57, complete 37.88%
att-weights epoch 560, step 401, max_size:classes 28, max_size:data 786, mem_usage:GPU:0 0.9GB, num_seqs 5, 7.233 sec/step, elapsed 0:23:15, exp. remaining 0:37:51, complete 38.07%
att-weights epoch 560, step 402, max_size:classes 27, max_size:data 936, mem_usage:GPU:0 0.9GB, num_seqs 4, 5.112 sec/step, elapsed 0:23:21, exp. remaining 0:37:41, complete 38.25%
att-weights epoch 560, step 403, max_size:classes 26, max_size:data 863, mem_usage:GPU:0 0.9GB, num_seqs 4, 6.824 sec/step, elapsed 0:23:27, exp. remaining 0:37:38, complete 38.40%
att-weights epoch 560, step 404, max_size:classes 26, max_size:data 949, mem_usage:GPU:0 0.9GB, num_seqs 4, 6.706 sec/step, elapsed 0:23:34, exp. remaining 0:37:34, complete 38.55%
att-weights epoch 560, step 405, max_size:classes 27, max_size:data 790, mem_usage:GPU:0 0.9GB, num_seqs 5, 3.780 sec/step, elapsed 0:23:38, exp. remaining 0:37:26, complete 38.70%
att-weights epoch 560, step 406, max_size:classes 27, max_size:data 874, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.960 sec/step, elapsed 0:23:40, exp. remaining 0:37:16, complete 38.85%
att-weights epoch 560, step 407, max_size:classes 27, max_size:data 845, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.922 sec/step, elapsed 0:23:42, exp. remaining 0:37:05, complete 38.99%
att-weights epoch 560, step 408, max_size:classes 28, max_size:data 912, mem_usage:GPU:0 0.9GB, num_seqs 4, 4.929 sec/step, elapsed 0:23:47, exp. remaining 0:36:59, complete 39.14%
att-weights epoch 560, step 409, max_size:classes 27, max_size:data 826, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.889 sec/step, elapsed 0:23:49, exp. remaining 0:36:44, complete 39.33%
att-weights epoch 560, step 410, max_size:classes 27, max_size:data 787, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.238 sec/step, elapsed 0:23:51, exp. remaining 0:36:34, complete 39.47%
att-weights epoch 560, step 411, max_size:classes 27, max_size:data 780, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.193 sec/step, elapsed 0:23:53, exp. remaining 0:36:24, complete 39.62%
att-weights epoch 560, step 412, max_size:classes 29, max_size:data 803, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.927 sec/step, elapsed 0:23:55, exp. remaining 0:36:13, complete 39.77%
att-weights epoch 560, step 413, max_size:classes 28, max_size:data 718, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.396 sec/step, elapsed 0:23:57, exp. remaining 0:36:00, complete 39.96%
att-weights epoch 560, step 414, max_size:classes 26, max_size:data 943, mem_usage:GPU:0 0.9GB, num_seqs 4, 7.194 sec/step, elapsed 0:24:05, exp. remaining 0:35:58, complete 40.10%
att-weights epoch 560, step 415, max_size:classes 28, max_size:data 802, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.616 sec/step, elapsed 0:24:06, exp. remaining 0:35:47, complete 40.25%
att-weights epoch 560, step 416, max_size:classes 29, max_size:data 720, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.777 sec/step, elapsed 0:24:08, exp. remaining 0:35:33, complete 40.44%
att-weights epoch 560, step 417, max_size:classes 31, max_size:data 829, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.133 sec/step, elapsed 0:24:10, exp. remaining 0:35:23, complete 40.58%
att-weights epoch 560, step 418, max_size:classes 28, max_size:data 783, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.527 sec/step, elapsed 0:24:13, exp. remaining 0:35:14, complete 40.73%
att-weights epoch 560, step 419, max_size:classes 27, max_size:data 872, mem_usage:GPU:0 0.9GB, num_seqs 4, 5.510 sec/step, elapsed 0:24:18, exp. remaining 0:35:06, complete 40.92%
att-weights epoch 560, step 420, max_size:classes 27, max_size:data 902, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.835 sec/step, elapsed 0:24:20, exp. remaining 0:34:56, complete 41.07%
att-weights epoch 560, step 421, max_size:classes 26, max_size:data 807, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.822 sec/step, elapsed 0:24:22, exp. remaining 0:34:42, complete 41.25%
att-weights epoch 560, step 422, max_size:classes 29, max_size:data 773, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.050 sec/step, elapsed 0:24:24, exp. remaining 0:34:29, complete 41.44%
att-weights epoch 560, step 423, max_size:classes 25, max_size:data 736, mem_usage:GPU:0 0.9GB, num_seqs 4, 4.526 sec/step, elapsed 0:24:28, exp. remaining 0:34:23, complete 41.58%
att-weights epoch 560, step 424, max_size:classes 27, max_size:data 929, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.078 sec/step, elapsed 0:24:30, exp. remaining 0:34:13, complete 41.73%
att-weights epoch 560, step 425, max_size:classes 28, max_size:data 772, mem_usage:GPU:0 0.9GB, num_seqs 5, 3.224 sec/step, elapsed 0:24:34, exp. remaining 0:34:05, complete 41.88%
att-weights epoch 560, step 426, max_size:classes 31, max_size:data 759, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.096 sec/step, elapsed 0:24:36, exp. remaining 0:33:53, complete 42.06%
att-weights epoch 560, step 427, max_size:classes 26, max_size:data 875, mem_usage:GPU:0 0.9GB, num_seqs 4, 6.494 sec/step, elapsed 0:24:42, exp. remaining 0:33:46, complete 42.25%
att-weights epoch 560, step 428, max_size:classes 28, max_size:data 798, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.023 sec/step, elapsed 0:24:44, exp. remaining 0:33:37, complete 42.40%
att-weights epoch 560, step 429, max_size:classes 25, max_size:data 821, mem_usage:GPU:0 0.9GB, num_seqs 4, 6.269 sec/step, elapsed 0:24:51, exp. remaining 0:33:30, complete 42.58%
att-weights epoch 560, step 430, max_size:classes 26, max_size:data 699, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.557 sec/step, elapsed 0:24:52, exp. remaining 0:33:17, complete 42.77%
att-weights epoch 560, step 431, max_size:classes 31, max_size:data 753, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.848 sec/step, elapsed 0:24:54, exp. remaining 0:33:07, complete 42.92%
att-weights epoch 560, step 432, max_size:classes 26, max_size:data 786, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.638 sec/step, elapsed 0:24:56, exp. remaining 0:32:58, complete 43.06%
att-weights epoch 560, step 433, max_size:classes 25, max_size:data 958, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.221 sec/step, elapsed 0:24:57, exp. remaining 0:32:47, complete 43.21%
att-weights epoch 560, step 434, max_size:classes 22, max_size:data 842, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.598 sec/step, elapsed 0:24:58, exp. remaining 0:32:35, complete 43.40%
att-weights epoch 560, step 435, max_size:classes 27, max_size:data 785, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.012 sec/step, elapsed 0:25:00, exp. remaining 0:32:26, complete 43.54%
att-weights epoch 560, step 436, max_size:classes 26, max_size:data 794, mem_usage:GPU:0 0.9GB, num_seqs 5, 3.953 sec/step, elapsed 0:25:04, exp. remaining 0:32:19, complete 43.69%
att-weights epoch 560, step 437, max_size:classes 25, max_size:data 801, mem_usage:GPU:0 0.9GB, num_seqs 4, 3.543 sec/step, elapsed 0:25:08, exp. remaining 0:32:09, complete 43.88%
att-weights epoch 560, step 438, max_size:classes 25, max_size:data 769, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.325 sec/step, elapsed 0:25:10, exp. remaining 0:32:00, complete 44.03%
att-weights epoch 560, step 439, max_size:classes 26, max_size:data 737, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.053 sec/step, elapsed 0:25:12, exp. remaining 0:31:51, complete 44.17%
att-weights epoch 560, step 440, max_size:classes 25, max_size:data 853, mem_usage:GPU:0 0.9GB, num_seqs 4, 3.277 sec/step, elapsed 0:25:16, exp. remaining 0:31:41, complete 44.36%
att-weights epoch 560, step 441, max_size:classes 27, max_size:data 660, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.553 sec/step, elapsed 0:25:17, exp. remaining 0:31:29, complete 44.54%
att-weights epoch 560, step 442, max_size:classes 24, max_size:data 848, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.644 sec/step, elapsed 0:25:19, exp. remaining 0:31:17, complete 44.73%
att-weights epoch 560, step 443, max_size:classes 24, max_size:data 694, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.661 sec/step, elapsed 0:25:21, exp. remaining 0:31:05, complete 44.91%
att-weights epoch 560, step 444, max_size:classes 25, max_size:data 823, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.725 sec/step, elapsed 0:25:22, exp. remaining 0:30:56, complete 45.06%
att-weights epoch 560, step 445, max_size:classes 27, max_size:data 801, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.656 sec/step, elapsed 0:25:24, exp. remaining 0:30:44, complete 45.25%
att-weights epoch 560, step 446, max_size:classes 24, max_size:data 738, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.679 sec/step, elapsed 0:25:26, exp. remaining 0:30:35, complete 45.39%
att-weights epoch 560, step 447, max_size:classes 22, max_size:data 806, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.538 sec/step, elapsed 0:25:27, exp. remaining 0:30:26, complete 45.54%
att-weights epoch 560, step 448, max_size:classes 27, max_size:data 925, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.874 sec/step, elapsed 0:25:29, exp. remaining 0:30:18, complete 45.69%
att-weights epoch 560, step 449, max_size:classes 24, max_size:data 679, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.883 sec/step, elapsed 0:25:31, exp. remaining 0:30:09, complete 45.84%
att-weights epoch 560, step 450, max_size:classes 24, max_size:data 750, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.019 sec/step, elapsed 0:25:33, exp. remaining 0:29:58, complete 46.02%
att-weights epoch 560, step 451, max_size:classes 22, max_size:data 764, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.286 sec/step, elapsed 0:25:35, exp. remaining 0:29:45, complete 46.24%
att-weights epoch 560, step 452, max_size:classes 22, max_size:data 743, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.931 sec/step, elapsed 0:25:37, exp. remaining 0:29:34, complete 46.43%
att-weights epoch 560, step 453, max_size:classes 23, max_size:data 863, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.322 sec/step, elapsed 0:25:38, exp. remaining 0:29:22, complete 46.61%
att-weights epoch 560, step 454, max_size:classes 22, max_size:data 779, mem_usage:GPU:0 0.9GB, num_seqs 5, 4.407 sec/step, elapsed 0:25:43, exp. remaining 0:29:14, complete 46.80%
att-weights epoch 560, step 455, max_size:classes 25, max_size:data 874, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.805 sec/step, elapsed 0:25:46, exp. remaining 0:29:09, complete 46.91%
att-weights epoch 560, step 456, max_size:classes 25, max_size:data 963, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.823 sec/step, elapsed 0:25:48, exp. remaining 0:29:05, complete 47.02%
att-weights epoch 560, step 457, max_size:classes 25, max_size:data 808, mem_usage:GPU:0 0.9GB, num_seqs 4, 6.590 sec/step, elapsed 0:25:55, exp. remaining 0:28:59, complete 47.21%
att-weights epoch 560, step 458, max_size:classes 23, max_size:data 898, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.020 sec/step, elapsed 0:25:57, exp. remaining 0:28:49, complete 47.39%
att-weights epoch 560, step 459, max_size:classes 23, max_size:data 683, mem_usage:GPU:0 0.9GB, num_seqs 5, 8.674 sec/step, elapsed 0:26:06, exp. remaining 0:28:48, complete 47.54%
att-weights epoch 560, step 460, max_size:classes 23, max_size:data 599, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.076 sec/step, elapsed 0:26:08, exp. remaining 0:28:37, complete 47.72%
att-weights epoch 560, step 461, max_size:classes 28, max_size:data 758, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.341 sec/step, elapsed 0:26:10, exp. remaining 0:28:27, complete 47.91%
att-weights epoch 560, step 462, max_size:classes 22, max_size:data 725, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.180 sec/step, elapsed 0:26:12, exp. remaining 0:28:19, complete 48.06%
att-weights epoch 560, step 463, max_size:classes 25, max_size:data 686, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.819 sec/step, elapsed 0:26:14, exp. remaining 0:28:09, complete 48.24%
att-weights epoch 560, step 464, max_size:classes 24, max_size:data 831, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.804 sec/step, elapsed 0:26:16, exp. remaining 0:27:58, complete 48.43%
att-weights epoch 560, step 465, max_size:classes 23, max_size:data 1043, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.897 sec/step, elapsed 0:26:18, exp. remaining 0:27:48, complete 48.61%
att-weights epoch 560, step 466, max_size:classes 23, max_size:data 738, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.085 sec/step, elapsed 0:26:20, exp. remaining 0:27:38, complete 48.80%
att-weights epoch 560, step 467, max_size:classes 23, max_size:data 767, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.671 sec/step, elapsed 0:26:22, exp. remaining 0:27:27, complete 48.98%
att-weights epoch 560, step 468, max_size:classes 23, max_size:data 814, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.613 sec/step, elapsed 0:26:23, exp. remaining 0:27:14, complete 49.20%
att-weights epoch 560, step 469, max_size:classes 25, max_size:data 711, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.595 sec/step, elapsed 0:26:26, exp. remaining 0:27:05, complete 49.39%
att-weights epoch 560, step 470, max_size:classes 22, max_size:data 755, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.699 sec/step, elapsed 0:26:28, exp. remaining 0:26:57, complete 49.54%
att-weights epoch 560, step 471, max_size:classes 23, max_size:data 826, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.928 sec/step, elapsed 0:26:29, exp. remaining 0:26:47, complete 49.72%
att-weights epoch 560, step 472, max_size:classes 26, max_size:data 697, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.704 sec/step, elapsed 0:26:31, exp. remaining 0:26:37, complete 49.91%
att-weights epoch 560, step 473, max_size:classes 24, max_size:data 672, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.721 sec/step, elapsed 0:26:33, exp. remaining 0:26:25, complete 50.13%
att-weights epoch 560, step 474, max_size:classes 26, max_size:data 671, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.776 sec/step, elapsed 0:26:35, exp. remaining 0:26:15, complete 50.31%
att-weights epoch 560, step 475, max_size:classes 22, max_size:data 695, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.059 sec/step, elapsed 0:26:37, exp. remaining 0:26:05, complete 50.50%
att-weights epoch 560, step 476, max_size:classes 24, max_size:data 785, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.915 sec/step, elapsed 0:26:39, exp. remaining 0:25:55, complete 50.68%
att-weights epoch 560, step 477, max_size:classes 25, max_size:data 640, mem_usage:GPU:0 0.9GB, num_seqs 6, 5.352 sec/step, elapsed 0:26:44, exp. remaining 0:25:49, complete 50.87%
att-weights epoch 560, step 478, max_size:classes 24, max_size:data 753, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.651 sec/step, elapsed 0:26:46, exp. remaining 0:25:39, complete 51.05%
att-weights epoch 560, step 479, max_size:classes 21, max_size:data 818, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.509 sec/step, elapsed 0:26:47, exp. remaining 0:25:29, complete 51.24%
att-weights epoch 560, step 480, max_size:classes 24, max_size:data 730, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.806 sec/step, elapsed 0:26:49, exp. remaining 0:25:22, complete 51.39%
att-weights epoch 560, step 481, max_size:classes 23, max_size:data 673, mem_usage:GPU:0 0.9GB, num_seqs 5, 3.491 sec/step, elapsed 0:26:52, exp. remaining 0:25:14, complete 51.57%
att-weights epoch 560, step 482, max_size:classes 21, max_size:data 661, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.080 sec/step, elapsed 0:26:55, exp. remaining 0:25:05, complete 51.76%
att-weights epoch 560, step 483, max_size:classes 23, max_size:data 745, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.035 sec/step, elapsed 0:26:57, exp. remaining 0:24:58, complete 51.91%
att-weights epoch 560, step 484, max_size:classes 23, max_size:data 674, mem_usage:GPU:0 0.9GB, num_seqs 5, 10.402 sec/step, elapsed 0:27:07, exp. remaining 0:24:56, complete 52.09%
att-weights epoch 560, step 485, max_size:classes 21, max_size:data 688, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.155 sec/step, elapsed 0:27:09, exp. remaining 0:24:45, complete 52.31%
att-weights epoch 560, step 486, max_size:classes 21, max_size:data 714, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.119 sec/step, elapsed 0:27:11, exp. remaining 0:24:36, complete 52.50%
att-weights epoch 560, step 487, max_size:classes 21, max_size:data 691, mem_usage:GPU:0 0.9GB, num_seqs 5, 4.850 sec/step, elapsed 0:27:16, exp. remaining 0:24:29, complete 52.68%
att-weights epoch 560, step 488, max_size:classes 22, max_size:data 720, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.369 sec/step, elapsed 0:27:17, exp. remaining 0:24:18, complete 52.90%
att-weights epoch 560, step 489, max_size:classes 21, max_size:data 987, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.765 sec/step, elapsed 0:27:19, exp. remaining 0:24:06, complete 53.13%
att-weights epoch 560, step 490, max_size:classes 23, max_size:data 669, mem_usage:GPU:0 0.9GB, num_seqs 5, 4.883 sec/step, elapsed 0:27:24, exp. remaining 0:24:00, complete 53.31%
att-weights epoch 560, step 491, max_size:classes 23, max_size:data 794, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.723 sec/step, elapsed 0:27:26, exp. remaining 0:23:51, complete 53.50%
att-weights epoch 560, step 492, max_size:classes 22, max_size:data 859, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.799 sec/step, elapsed 0:27:28, exp. remaining 0:23:42, complete 53.68%
att-weights epoch 560, step 493, max_size:classes 21, max_size:data 681, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.126 sec/step, elapsed 0:27:30, exp. remaining 0:23:33, complete 53.87%
att-weights epoch 560, step 494, max_size:classes 21, max_size:data 622, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.628 sec/step, elapsed 0:27:32, exp. remaining 0:23:23, complete 54.09%
att-weights epoch 560, step 495, max_size:classes 21, max_size:data 798, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.042 sec/step, elapsed 0:27:34, exp. remaining 0:23:12, complete 54.31%
att-weights epoch 560, step 496, max_size:classes 22, max_size:data 699, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.890 sec/step, elapsed 0:27:36, exp. remaining 0:23:03, complete 54.50%
att-weights epoch 560, step 497, max_size:classes 21, max_size:data 602, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.674 sec/step, elapsed 0:27:38, exp. remaining 0:22:52, complete 54.72%
att-weights epoch 560, step 498, max_size:classes 22, max_size:data 612, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.215 sec/step, elapsed 0:27:40, exp. remaining 0:22:42, complete 54.94%
att-weights epoch 560, step 499, max_size:classes 19, max_size:data 789, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.781 sec/step, elapsed 0:27:42, exp. remaining 0:22:33, complete 55.12%
att-weights epoch 560, step 500, max_size:classes 22, max_size:data 738, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.588 sec/step, elapsed 0:27:44, exp. remaining 0:22:24, complete 55.31%
att-weights epoch 560, step 501, max_size:classes 21, max_size:data 585, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.582 sec/step, elapsed 0:27:45, exp. remaining 0:22:13, complete 55.53%
att-weights epoch 560, step 502, max_size:classes 23, max_size:data 693, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.034 sec/step, elapsed 0:27:47, exp. remaining 0:22:05, complete 55.72%
att-weights epoch 560, step 503, max_size:classes 21, max_size:data 587, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.949 sec/step, elapsed 0:27:49, exp. remaining 0:21:55, complete 55.94%
att-weights epoch 560, step 504, max_size:classes 20, max_size:data 602, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.946 sec/step, elapsed 0:27:51, exp. remaining 0:21:44, complete 56.16%
att-weights epoch 560, step 505, max_size:classes 21, max_size:data 698, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.997 sec/step, elapsed 0:27:53, exp. remaining 0:21:34, complete 56.38%
att-weights epoch 560, step 506, max_size:classes 24, max_size:data 563, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.768 sec/step, elapsed 0:27:55, exp. remaining 0:21:24, complete 56.60%
att-weights epoch 560, step 507, max_size:classes 23, max_size:data 585, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.189 sec/step, elapsed 0:27:57, exp. remaining 0:21:12, complete 56.86%
att-weights epoch 560, step 508, max_size:classes 22, max_size:data 598, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.938 sec/step, elapsed 0:27:59, exp. remaining 0:21:02, complete 57.08%
att-weights epoch 560, step 509, max_size:classes 19, max_size:data 686, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.549 sec/step, elapsed 0:28:01, exp. remaining 0:20:54, complete 57.27%
att-weights epoch 560, step 510, max_size:classes 19, max_size:data 660, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.850 sec/step, elapsed 0:28:02, exp. remaining 0:20:46, complete 57.45%
att-weights epoch 560, step 511, max_size:classes 21, max_size:data 672, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.756 sec/step, elapsed 0:28:04, exp. remaining 0:20:36, complete 57.68%
att-weights epoch 560, step 512, max_size:classes 20, max_size:data 598, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.072 sec/step, elapsed 0:28:06, exp. remaining 0:20:26, complete 57.90%
att-weights epoch 560, step 513, max_size:classes 23, max_size:data 601, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.271 sec/step, elapsed 0:28:09, exp. remaining 0:20:17, complete 58.12%
att-weights epoch 560, step 514, max_size:classes 23, max_size:data 621, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.881 sec/step, elapsed 0:28:10, exp. remaining 0:20:07, complete 58.34%
att-weights epoch 560, step 515, max_size:classes 21, max_size:data 648, mem_usage:GPU:0 0.9GB, num_seqs 6, 5.057 sec/step, elapsed 0:28:15, exp. remaining 0:19:59, complete 58.56%
att-weights epoch 560, step 516, max_size:classes 21, max_size:data 563, mem_usage:GPU:0 0.9GB, num_seqs 7, 5.158 sec/step, elapsed 0:28:21, exp. remaining 0:19:52, complete 58.79%
att-weights epoch 560, step 517, max_size:classes 20, max_size:data 574, mem_usage:GPU:0 0.9GB, num_seqs 6, 19.872 sec/step, elapsed 0:28:40, exp. remaining 0:19:57, complete 58.97%
att-weights epoch 560, step 518, max_size:classes 22, max_size:data 605, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.978 sec/step, elapsed 0:28:42, exp. remaining 0:19:45, complete 59.23%
att-weights epoch 560, step 519, max_size:classes 19, max_size:data 687, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.895 sec/step, elapsed 0:28:44, exp. remaining 0:19:36, complete 59.45%
att-weights epoch 560, step 520, max_size:classes 19, max_size:data 654, mem_usage:GPU:0 0.9GB, num_seqs 6, 8.941 sec/step, elapsed 0:28:53, exp. remaining 0:19:29, complete 59.71%
att-weights epoch 560, step 521, max_size:classes 20, max_size:data 579, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.744 sec/step, elapsed 0:28:55, exp. remaining 0:19:18, complete 59.97%
att-weights epoch 560, step 522, max_size:classes 19, max_size:data 623, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.041 sec/step, elapsed 0:28:57, exp. remaining 0:19:10, complete 60.16%
att-weights epoch 560, step 523, max_size:classes 19, max_size:data 651, mem_usage:GPU:0 0.9GB, num_seqs 6, 8.594 sec/step, elapsed 0:29:06, exp. remaining 0:19:05, complete 60.38%
att-weights epoch 560, step 524, max_size:classes 21, max_size:data 653, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.059 sec/step, elapsed 0:29:08, exp. remaining 0:18:56, complete 60.60%
att-weights epoch 560, step 525, max_size:classes 19, max_size:data 612, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.031 sec/step, elapsed 0:29:10, exp. remaining 0:18:45, complete 60.86%
att-weights epoch 560, step 526, max_size:classes 20, max_size:data 692, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.872 sec/step, elapsed 0:29:12, exp. remaining 0:18:36, complete 61.08%
att-weights epoch 560, step 527, max_size:classes 19, max_size:data 524, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.460 sec/step, elapsed 0:29:14, exp. remaining 0:18:29, complete 61.27%
att-weights epoch 560, step 528, max_size:classes 21, max_size:data 588, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.977 sec/step, elapsed 0:29:16, exp. remaining 0:18:20, complete 61.49%
att-weights epoch 560, step 529, max_size:classes 19, max_size:data 557, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.280 sec/step, elapsed 0:29:18, exp. remaining 0:18:09, complete 61.75%
att-weights epoch 560, step 530, max_size:classes 18, max_size:data 543, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.803 sec/step, elapsed 0:29:21, exp. remaining 0:17:59, complete 62.01%
att-weights epoch 560, step 531, max_size:classes 16, max_size:data 678, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.805 sec/step, elapsed 0:29:23, exp. remaining 0:17:48, complete 62.26%
att-weights epoch 560, step 532, max_size:classes 22, max_size:data 591, mem_usage:GPU:0 0.9GB, num_seqs 6, 8.026 sec/step, elapsed 0:29:31, exp. remaining 0:17:41, complete 62.52%
att-weights epoch 560, step 533, max_size:classes 19, max_size:data 612, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.315 sec/step, elapsed 0:29:33, exp. remaining 0:17:31, complete 62.78%
att-weights epoch 560, step 534, max_size:classes 20, max_size:data 542, mem_usage:GPU:0 0.9GB, num_seqs 7, 5.795 sec/step, elapsed 0:29:39, exp. remaining 0:17:24, complete 63.00%
att-weights epoch 560, step 535, max_size:classes 19, max_size:data 656, mem_usage:GPU:0 0.9GB, num_seqs 6, 3.136 sec/step, elapsed 0:29:42, exp. remaining 0:17:15, complete 63.26%
att-weights epoch 560, step 536, max_size:classes 20, max_size:data 728, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.914 sec/step, elapsed 0:29:44, exp. remaining 0:17:06, complete 63.49%
att-weights epoch 560, step 537, max_size:classes 22, max_size:data 658, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.181 sec/step, elapsed 0:29:46, exp. remaining 0:16:54, complete 63.78%
att-weights epoch 560, step 538, max_size:classes 21, max_size:data 494, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.925 sec/step, elapsed 0:29:48, exp. remaining 0:16:46, complete 64.00%
att-weights epoch 560, step 539, max_size:classes 18, max_size:data 536, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.074 sec/step, elapsed 0:29:50, exp. remaining 0:16:37, complete 64.22%
att-weights epoch 560, step 540, max_size:classes 17, max_size:data 554, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.850 sec/step, elapsed 0:29:52, exp. remaining 0:16:28, complete 64.45%
att-weights epoch 560, step 541, max_size:classes 20, max_size:data 543, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.891 sec/step, elapsed 0:29:54, exp. remaining 0:16:20, complete 64.67%
att-weights epoch 560, step 542, max_size:classes 18, max_size:data 569, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.228 sec/step, elapsed 0:29:56, exp. remaining 0:16:12, complete 64.89%
att-weights epoch 560, step 543, max_size:classes 20, max_size:data 572, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.562 sec/step, elapsed 0:29:58, exp. remaining 0:16:03, complete 65.11%
att-weights epoch 560, step 544, max_size:classes 18, max_size:data 522, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.691 sec/step, elapsed 0:30:01, exp. remaining 0:15:52, complete 65.41%
att-weights epoch 560, step 545, max_size:classes 15, max_size:data 652, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.620 sec/step, elapsed 0:30:02, exp. remaining 0:15:45, complete 65.59%
att-weights epoch 560, step 546, max_size:classes 16, max_size:data 470, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.305 sec/step, elapsed 0:30:05, exp. remaining 0:15:35, complete 65.85%
att-weights epoch 560, step 547, max_size:classes 17, max_size:data 587, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.940 sec/step, elapsed 0:30:06, exp. remaining 0:15:29, complete 66.04%
att-weights epoch 560, step 548, max_size:classes 19, max_size:data 660, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.122 sec/step, elapsed 0:30:09, exp. remaining 0:15:21, complete 66.26%
att-weights epoch 560, step 549, max_size:classes 16, max_size:data 627, mem_usage:GPU:0 0.9GB, num_seqs 6, 4.589 sec/step, elapsed 0:30:13, exp. remaining 0:15:11, complete 66.56%
att-weights epoch 560, step 550, max_size:classes 21, max_size:data 603, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.984 sec/step, elapsed 0:30:15, exp. remaining 0:15:01, complete 66.81%
att-weights epoch 560, step 551, max_size:classes 18, max_size:data 579, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.875 sec/step, elapsed 0:30:17, exp. remaining 0:14:50, complete 67.11%
att-weights epoch 560, step 552, max_size:classes 17, max_size:data 601, mem_usage:GPU:0 0.9GB, num_seqs 6, 6.043 sec/step, elapsed 0:30:23, exp. remaining 0:14:43, complete 67.37%
att-weights epoch 560, step 553, max_size:classes 18, max_size:data 494, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.192 sec/step, elapsed 0:30:25, exp. remaining 0:14:33, complete 67.63%
att-weights epoch 560, step 554, max_size:classes 18, max_size:data 677, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.871 sec/step, elapsed 0:30:27, exp. remaining 0:14:25, complete 67.85%
att-weights epoch 560, step 555, max_size:classes 16, max_size:data 566, mem_usage:GPU:0 0.9GB, num_seqs 7, 3.410 sec/step, elapsed 0:30:31, exp. remaining 0:14:18, complete 68.07%
att-weights epoch 560, step 556, max_size:classes 16, max_size:data 702, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.949 sec/step, elapsed 0:30:32, exp. remaining 0:14:09, complete 68.33%
att-weights epoch 560, step 557, max_size:classes 17, max_size:data 630, mem_usage:GPU:0 0.9GB, num_seqs 6, 5.006 sec/step, elapsed 0:30:38, exp. remaining 0:14:03, complete 68.55%
att-weights epoch 560, step 558, max_size:classes 16, max_size:data 482, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.232 sec/step, elapsed 0:30:40, exp. remaining 0:13:54, complete 68.81%
att-weights epoch 560, step 559, max_size:classes 16, max_size:data 523, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.133 sec/step, elapsed 0:30:42, exp. remaining 0:13:44, complete 69.07%
att-weights epoch 560, step 560, max_size:classes 17, max_size:data 492, mem_usage:GPU:0 0.9GB, num_seqs 8, 7.077 sec/step, elapsed 0:30:49, exp. remaining 0:13:39, complete 69.29%
att-weights epoch 560, step 561, max_size:classes 16, max_size:data 545, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.321 sec/step, elapsed 0:30:51, exp. remaining 0:13:29, complete 69.59%
att-weights epoch 560, step 562, max_size:classes 20, max_size:data 485, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.252 sec/step, elapsed 0:30:54, exp. remaining 0:13:18, complete 69.89%
att-weights epoch 560, step 563, max_size:classes 16, max_size:data 635, mem_usage:GPU:0 0.9GB, num_seqs 6, 11.993 sec/step, elapsed 0:31:06, exp. remaining 0:13:14, complete 70.14%
att-weights epoch 560, step 564, max_size:classes 17, max_size:data 644, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.478 sec/step, elapsed 0:31:08, exp. remaining 0:13:05, complete 70.40%
att-weights epoch 560, step 565, max_size:classes 19, max_size:data 495, mem_usage:GPU:0 0.9GB, num_seqs 7, 14.133 sec/step, elapsed 0:31:22, exp. remaining 0:13:00, complete 70.70%
att-weights epoch 560, step 566, max_size:classes 16, max_size:data 593, mem_usage:GPU:0 0.9GB, num_seqs 6, 3.032 sec/step, elapsed 0:31:25, exp. remaining 0:12:51, complete 70.96%
att-weights epoch 560, step 567, max_size:classes 15, max_size:data 517, mem_usage:GPU:0 0.9GB, num_seqs 7, 7.164 sec/step, elapsed 0:31:32, exp. remaining 0:12:43, complete 71.25%
att-weights epoch 560, step 568, max_size:classes 16, max_size:data 543, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.975 sec/step, elapsed 0:31:34, exp. remaining 0:12:33, complete 71.55%
att-weights epoch 560, step 569, max_size:classes 16, max_size:data 585, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.994 sec/step, elapsed 0:31:36, exp. remaining 0:12:24, complete 71.81%
att-weights epoch 560, step 570, max_size:classes 18, max_size:data 492, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.646 sec/step, elapsed 0:31:39, exp. remaining 0:12:14, complete 72.11%
att-weights epoch 560, step 571, max_size:classes 15, max_size:data 488, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.394 sec/step, elapsed 0:31:41, exp. remaining 0:12:04, complete 72.40%
att-weights epoch 560, step 572, max_size:classes 16, max_size:data 549, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.815 sec/step, elapsed 0:31:43, exp. remaining 0:11:52, complete 72.77%
att-weights epoch 560, step 573, max_size:classes 16, max_size:data 541, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.726 sec/step, elapsed 0:31:45, exp. remaining 0:11:42, complete 73.07%
att-weights epoch 560, step 574, max_size:classes 15, max_size:data 488, mem_usage:GPU:0 0.9GB, num_seqs 8, 1.908 sec/step, elapsed 0:31:47, exp. remaining 0:11:32, complete 73.36%
att-weights epoch 560, step 575, max_size:classes 15, max_size:data 513, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.294 sec/step, elapsed 0:31:49, exp. remaining 0:11:24, complete 73.62%
att-weights epoch 560, step 576, max_size:classes 16, max_size:data 440, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.190 sec/step, elapsed 0:31:51, exp. remaining 0:11:14, complete 73.92%
att-weights epoch 560, step 577, max_size:classes 16, max_size:data 497, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.299 sec/step, elapsed 0:31:54, exp. remaining 0:11:06, complete 74.18%
att-weights epoch 560, step 578, max_size:classes 14, max_size:data 522, mem_usage:GPU:0 0.9GB, num_seqs 7, 3.215 sec/step, elapsed 0:31:57, exp. remaining 0:10:57, complete 74.47%
att-weights epoch 560, step 579, max_size:classes 18, max_size:data 463, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.138 sec/step, elapsed 0:31:59, exp. remaining 0:10:47, complete 74.77%
att-weights epoch 560, step 580, max_size:classes 15, max_size:data 458, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.200 sec/step, elapsed 0:32:01, exp. remaining 0:10:37, complete 75.10%
att-weights epoch 560, step 581, max_size:classes 14, max_size:data 397, mem_usage:GPU:0 0.9GB, num_seqs 10, 2.457 sec/step, elapsed 0:32:04, exp. remaining 0:10:29, complete 75.36%
att-weights epoch 560, step 582, max_size:classes 16, max_size:data 475, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.113 sec/step, elapsed 0:32:06, exp. remaining 0:10:19, complete 75.66%
att-weights epoch 560, step 583, max_size:classes 14, max_size:data 482, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.221 sec/step, elapsed 0:32:08, exp. remaining 0:10:11, complete 75.92%
att-weights epoch 560, step 584, max_size:classes 14, max_size:data 553, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.622 sec/step, elapsed 0:32:10, exp. remaining 0:10:02, complete 76.21%
att-weights epoch 560, step 585, max_size:classes 15, max_size:data 457, mem_usage:GPU:0 0.9GB, num_seqs 8, 1.821 sec/step, elapsed 0:32:11, exp. remaining 0:09:53, complete 76.51%
att-weights epoch 560, step 586, max_size:classes 14, max_size:data 517, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.803 sec/step, elapsed 0:32:13, exp. remaining 0:09:45, complete 76.77%
att-weights epoch 560, step 587, max_size:classes 13, max_size:data 493, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.120 sec/step, elapsed 0:32:15, exp. remaining 0:09:37, complete 77.03%
att-weights epoch 560, step 588, max_size:classes 14, max_size:data 490, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.364 sec/step, elapsed 0:32:18, exp. remaining 0:09:28, complete 77.32%
att-weights epoch 560, step 589, max_size:classes 17, max_size:data 419, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.357 sec/step, elapsed 0:32:20, exp. remaining 0:09:20, complete 77.58%
att-weights epoch 560, step 590, max_size:classes 13, max_size:data 508, mem_usage:GPU:0 0.9GB, num_seqs 7, 5.868 sec/step, elapsed 0:32:26, exp. remaining 0:09:12, complete 77.88%
att-weights epoch 560, step 591, max_size:classes 14, max_size:data 485, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.456 sec/step, elapsed 0:32:28, exp. remaining 0:09:04, complete 78.17%
att-weights epoch 560, step 592, max_size:classes 15, max_size:data 539, mem_usage:GPU:0 0.9GB, num_seqs 7, 4.132 sec/step, elapsed 0:32:32, exp. remaining 0:08:54, complete 78.51%
att-weights epoch 560, step 593, max_size:classes 15, max_size:data 470, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.521 sec/step, elapsed 0:32:35, exp. remaining 0:08:46, complete 78.80%
att-weights epoch 560, step 594, max_size:classes 15, max_size:data 474, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.200 sec/step, elapsed 0:32:37, exp. remaining 0:08:37, complete 79.10%
att-weights epoch 560, step 595, max_size:classes 14, max_size:data 481, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.007 sec/step, elapsed 0:32:39, exp. remaining 0:08:26, complete 79.47%
att-weights epoch 560, step 596, max_size:classes 17, max_size:data 571, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.193 sec/step, elapsed 0:32:41, exp. remaining 0:08:16, complete 79.80%
att-weights epoch 560, step 597, max_size:classes 14, max_size:data 483, mem_usage:GPU:0 0.9GB, num_seqs 8, 4.844 sec/step, elapsed 0:32:46, exp. remaining 0:08:08, complete 80.10%
att-weights epoch 560, step 598, max_size:classes 17, max_size:data 530, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.674 sec/step, elapsed 0:32:49, exp. remaining 0:07:59, complete 80.43%
att-weights epoch 560, step 599, max_size:classes 14, max_size:data 477, mem_usage:GPU:0 0.9GB, num_seqs 8, 6.688 sec/step, elapsed 0:32:56, exp. remaining 0:07:51, complete 80.73%
att-weights epoch 560, step 600, max_size:classes 14, max_size:data 468, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.441 sec/step, elapsed 0:32:58, exp. remaining 0:07:41, complete 81.10%
att-weights epoch 560, step 601, max_size:classes 13, max_size:data 432, mem_usage:GPU:0 0.9GB, num_seqs 9, 12.708 sec/step, elapsed 0:33:11, exp. remaining 0:07:34, complete 81.43%
att-weights epoch 560, step 602, max_size:classes 14, max_size:data 471, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.207 sec/step, elapsed 0:33:13, exp. remaining 0:07:23, complete 81.80%
att-weights epoch 560, step 603, max_size:classes 13, max_size:data 488, mem_usage:GPU:0 0.9GB, num_seqs 8, 1.918 sec/step, elapsed 0:33:15, exp. remaining 0:07:13, complete 82.17%
att-weights epoch 560, step 604, max_size:classes 14, max_size:data 389, mem_usage:GPU:0 0.9GB, num_seqs 10, 2.499 sec/step, elapsed 0:33:17, exp. remaining 0:07:03, complete 82.50%
att-weights epoch 560, step 605, max_size:classes 13, max_size:data 406, mem_usage:GPU:0 0.9GB, num_seqs 9, 13.471 sec/step, elapsed 0:33:31, exp. remaining 0:06:56, complete 82.83%
att-weights epoch 560, step 606, max_size:classes 13, max_size:data 486, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.149 sec/step, elapsed 0:33:33, exp. remaining 0:06:47, complete 83.17%
att-weights epoch 560, step 607, max_size:classes 13, max_size:data 423, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.475 sec/step, elapsed 0:33:35, exp. remaining 0:06:39, complete 83.46%
att-weights epoch 560, step 608, max_size:classes 13, max_size:data 454, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.519 sec/step, elapsed 0:33:38, exp. remaining 0:06:30, complete 83.80%
att-weights epoch 560, step 609, max_size:classes 12, max_size:data 391, mem_usage:GPU:0 0.9GB, num_seqs 10, 4.487 sec/step, elapsed 0:33:42, exp. remaining 0:06:20, complete 84.17%
att-weights epoch 560, step 610, max_size:classes 12, max_size:data 423, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.160 sec/step, elapsed 0:33:45, exp. remaining 0:06:10, complete 84.54%
att-weights epoch 560, step 611, max_size:classes 12, max_size:data 368, mem_usage:GPU:0 0.9GB, num_seqs 10, 2.538 sec/step, elapsed 0:33:47, exp. remaining 0:06:02, complete 84.83%
att-weights epoch 560, step 612, max_size:classes 14, max_size:data 368, mem_usage:GPU:0 0.9GB, num_seqs 10, 2.548 sec/step, elapsed 0:33:50, exp. remaining 0:05:52, complete 85.20%
att-weights epoch 560, step 613, max_size:classes 12, max_size:data 429, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.295 sec/step, elapsed 0:33:52, exp. remaining 0:05:43, complete 85.53%
att-weights epoch 560, step 614, max_size:classes 12, max_size:data 427, mem_usage:GPU:0 0.9GB, num_seqs 9, 10.893 sec/step, elapsed 0:34:03, exp. remaining 0:05:37, complete 85.83%
att-weights epoch 560, step 615, max_size:classes 12, max_size:data 426, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.316 sec/step, elapsed 0:34:05, exp. remaining 0:05:28, complete 86.16%
att-weights epoch 560, step 616, max_size:classes 11, max_size:data 471, mem_usage:GPU:0 0.9GB, num_seqs 8, 1.950 sec/step, elapsed 0:34:07, exp. remaining 0:05:18, complete 86.53%
att-weights epoch 560, step 617, max_size:classes 11, max_size:data 418, mem_usage:GPU:0 0.9GB, num_seqs 9, 6.207 sec/step, elapsed 0:34:13, exp. remaining 0:05:10, complete 86.87%
att-weights epoch 560, step 618, max_size:classes 13, max_size:data 388, mem_usage:GPU:0 0.9GB, num_seqs 10, 2.568 sec/step, elapsed 0:34:16, exp. remaining 0:05:02, complete 87.16%
att-weights epoch 560, step 619, max_size:classes 11, max_size:data 380, mem_usage:GPU:0 0.9GB, num_seqs 10, 2.529 sec/step, elapsed 0:34:18, exp. remaining 0:04:55, complete 87.46%
att-weights epoch 560, step 620, max_size:classes 13, max_size:data 468, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.224 sec/step, elapsed 0:34:21, exp. remaining 0:04:45, complete 87.83%
att-weights epoch 560, step 621, max_size:classes 12, max_size:data 386, mem_usage:GPU:0 0.9GB, num_seqs 10, 2.473 sec/step, elapsed 0:34:23, exp. remaining 0:04:35, complete 88.24%
att-weights epoch 560, step 622, max_size:classes 10, max_size:data 401, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.667 sec/step, elapsed 0:34:26, exp. remaining 0:04:25, complete 88.61%
att-weights epoch 560, step 623, max_size:classes 11, max_size:data 498, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.262 sec/step, elapsed 0:34:28, exp. remaining 0:04:17, complete 88.94%
att-weights epoch 560, step 624, max_size:classes 12, max_size:data 442, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.128 sec/step, elapsed 0:34:30, exp. remaining 0:04:06, complete 89.35%
att-weights epoch 560, step 625, max_size:classes 12, max_size:data 400, mem_usage:GPU:0 0.9GB, num_seqs 10, 3.329 sec/step, elapsed 0:34:34, exp. remaining 0:03:58, complete 89.68%
att-weights epoch 560, step 626, max_size:classes 12, max_size:data 346, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.288 sec/step, elapsed 0:34:36, exp. remaining 0:03:49, complete 90.05%
att-weights epoch 560, step 627, max_size:classes 13, max_size:data 482, mem_usage:GPU:0 0.9GB, num_seqs 8, 1.954 sec/step, elapsed 0:34:38, exp. remaining 0:03:39, complete 90.46%
att-weights epoch 560, step 628, max_size:classes 11, max_size:data 458, mem_usage:GPU:0 0.9GB, num_seqs 8, 1.808 sec/step, elapsed 0:34:40, exp. remaining 0:03:30, complete 90.83%
att-weights epoch 560, step 629, max_size:classes 11, max_size:data 398, mem_usage:GPU:0 0.9GB, num_seqs 10, 2.472 sec/step, elapsed 0:34:42, exp. remaining 0:03:21, complete 91.19%
att-weights epoch 560, step 630, max_size:classes 10, max_size:data 338, mem_usage:GPU:0 0.9GB, num_seqs 11, 2.809 sec/step, elapsed 0:34:45, exp. remaining 0:03:11, complete 91.60%
att-weights epoch 560, step 631, max_size:classes 12, max_size:data 371, mem_usage:GPU:0 0.9GB, num_seqs 10, 5.108 sec/step, elapsed 0:34:50, exp. remaining 0:03:02, complete 91.97%
att-weights epoch 560, step 632, max_size:classes 11, max_size:data 403, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.531 sec/step, elapsed 0:34:53, exp. remaining 0:02:53, complete 92.34%
att-weights epoch 560, step 633, max_size:classes 13, max_size:data 351, mem_usage:GPU:0 0.9GB, num_seqs 11, 11.419 sec/step, elapsed 0:35:04, exp. remaining 0:02:44, complete 92.75%
att-weights epoch 560, step 634, max_size:classes 9, max_size:data 434, mem_usage:GPU:0 0.9GB, num_seqs 9, 10.365 sec/step, elapsed 0:35:14, exp. remaining 0:02:35, complete 93.16%
att-weights epoch 560, step 635, max_size:classes 10, max_size:data 371, mem_usage:GPU:0 0.9GB, num_seqs 10, 11.146 sec/step, elapsed 0:35:25, exp. remaining 0:02:25, complete 93.60%
att-weights epoch 560, step 636, max_size:classes 12, max_size:data 345, mem_usage:GPU:0 0.9GB, num_seqs 11, 2.509 sec/step, elapsed 0:35:28, exp. remaining 0:02:15, complete 94.01%
att-weights epoch 560, step 637, max_size:classes 9, max_size:data 378, mem_usage:GPU:0 0.9GB, num_seqs 10, 2.535 sec/step, elapsed 0:35:31, exp. remaining 0:02:05, complete 94.45%
att-weights epoch 560, step 638, max_size:classes 11, max_size:data 370, mem_usage:GPU:0 0.9GB, num_seqs 10, 4.514 sec/step, elapsed 0:35:35, exp. remaining 0:01:55, complete 94.86%
att-weights epoch 560, step 639, max_size:classes 11, max_size:data 359, mem_usage:GPU:0 0.9GB, num_seqs 11, 2.924 sec/step, elapsed 0:35:38, exp. remaining 0:01:46, complete 95.26%
att-weights epoch 560, step 640, max_size:classes 10, max_size:data 372, mem_usage:GPU:0 0.9GB, num_seqs 10, 2.287 sec/step, elapsed 0:35:40, exp. remaining 0:01:36, complete 95.67%
att-weights epoch 560, step 641, max_size:classes 9, max_size:data 367, mem_usage:GPU:0 0.9GB, num_seqs 10, 9.868 sec/step, elapsed 0:35:50, exp. remaining 0:01:26, complete 96.12%
att-weights epoch 560, step 642, max_size:classes 10, max_size:data 360, mem_usage:GPU:0 0.9GB, num_seqs 11, 2.822 sec/step, elapsed 0:35:53, exp. remaining 0:01:18, complete 96.49%
att-weights epoch 560, step 643, max_size:classes 9, max_size:data 336, mem_usage:GPU:0 0.9GB, num_seqs 11, 2.530 sec/step, elapsed 0:35:55, exp. remaining 0:01:07, complete 96.97%
att-weights epoch 560, step 644, max_size:classes 9, max_size:data 309, mem_usage:GPU:0 0.9GB, num_seqs 12, 3.230 sec/step, elapsed 0:35:59, exp. remaining 0:00:57, complete 97.41%
att-weights epoch 560, step 645, max_size:classes 10, max_size:data 335, mem_usage:GPU:0 0.9GB, num_seqs 11, 2.689 sec/step, elapsed 0:36:01, exp. remaining 0:00:47, complete 97.85%
att-weights epoch 560, step 646, max_size:classes 10, max_size:data 311, mem_usage:GPU:0 0.9GB, num_seqs 12, 2.718 sec/step, elapsed 0:36:04, exp. remaining 0:00:36, complete 98.34%
att-weights epoch 560, step 647, max_size:classes 8, max_size:data 344, mem_usage:GPU:0 0.9GB, num_seqs 11, 2.927 sec/step, elapsed 0:36:07, exp. remaining 0:00:25, complete 98.85%
att-weights epoch 560, step 648, max_size:classes 10, max_size:data 277, mem_usage:GPU:0 0.9GB, num_seqs 11, 1.652 sec/step, elapsed 0:36:09, exp. remaining 0:00:14, complete 99.33%
att-weights epoch 560, step 649, max_size:classes 10, max_size:data 362, mem_usage:GPU:0 0.9GB, num_seqs 11, 1.023 sec/step, elapsed 0:36:10, exp. remaining 0:00:04, complete 99.82%
att-weights epoch 560, step 650, max_size:classes 8, max_size:data 331, mem_usage:GPU:0 0.9GB, num_seqs 12, 1.137 sec/step, elapsed 0:36:11, exp. remaining 0:00:04, complete 99.82%
att-weights epoch 560, step 651, max_size:classes 7, max_size:data 366, mem_usage:GPU:0 0.9GB, num_seqs 10, 0.887 sec/step, elapsed 0:36:12, exp. remaining 0:00:04, complete 99.82%
att-weights epoch 560, step 652, max_size:classes 11, max_size:data 289, mem_usage:GPU:0 0.9GB, num_seqs 13, 1.065 sec/step, elapsed 0:36:13, exp. remaining 0:00:04, complete 99.82%
att-weights epoch 560, step 653, max_size:classes 11, max_size:data 292, mem_usage:GPU:0 0.9GB, num_seqs 12, 1.098 sec/step, elapsed 0:36:14, exp. remaining 0:00:04, complete 99.82%
att-weights epoch 560, step 654, max_size:classes 11, max_size:data 329, mem_usage:GPU:0 0.9GB, num_seqs 12, 0.952 sec/step, elapsed 0:36:15, exp. remaining 0:00:04, complete 99.82%
att-weights epoch 560, step 655, max_size:classes 8, max_size:data 288, mem_usage:GPU:0 0.9GB, num_seqs 13, 0.982 sec/step, elapsed 0:36:16, exp. remaining 0:00:04, complete 99.82%
att-weights epoch 560, step 656, max_size:classes 8, max_size:data 281, mem_usage:GPU:0 0.9GB, num_seqs 14, 0.994 sec/step, elapsed 0:36:17, exp. remaining 0:00:04, complete 99.82%
att-weights epoch 560, step 657, max_size:classes 6, max_size:data 304, mem_usage:GPU:0 0.9GB, num_seqs 13, 0.961 sec/step, elapsed 0:36:18, exp. remaining 0:00:04, complete 99.82%
att-weights epoch 560, step 658, max_size:classes 4, max_size:data 288, mem_usage:GPU:0 0.9GB, num_seqs 13, 0.869 sec/step, elapsed 0:36:19, exp. remaining 0:00:04, complete 99.82%
att-weights epoch 560, step 659, max_size:classes 4, max_size:data 178, mem_usage:GPU:0 0.9GB, num_seqs 6, 0.341 sec/step, elapsed 0:36:19, exp. remaining 0:00:04, complete 99.82%
Stats:
  mem_usage:GPU:0: Stats(mean=0.9GB, std_dev=0.0B, min=0.9GB, max=0.9GB, num_seqs=660, avg_data_len=1)
att-weights epoch 560, finished after 660 steps, 0:36:19 elapsed (22.0% computing time)
Layer 'dec_02_att_weights' Stats:
  2703 seqs, 102444704 total frames, 37900.371439 average frames
  Mean: 0.005538597682513847
  Std dev: 0.06652690714796147
  Min/max: 0.0 / 1.0
Layer 'dec_01_att_weights' Stats:
  2703 seqs, 102444704 total frames, 37900.371439 average frames
  Mean: 0.005538597675159589
  Std dev: 0.06784411531276431
  Min/max: 0.0 / 1.0
Layer 'dec_03_att_weights' Stats:
  2703 seqs, 102444704 total frames, 37900.371439 average frames
  Mean: 0.005538597667488878
  Std dev: 0.0567917298545559
  Min/max: 0.0 / 1.0
Layer 'dec_04_att_weights' Stats:
  2703 seqs, 102444704 total frames, 37900.371439 average frames
  Mean: 0.005538597675960206
  Std dev: 0.06541223178476314
  Min/max: 0.0 / 1.0
Layer 'dec_05_att_weights' Stats:
  2703 seqs, 102444704 total frames, 37900.371439 average frames
  Mean: 0.005538597679870032
  Std dev: 0.04882894601096932
  Min/max: 0.0 / 1.0
Layer 'dec_06_att_weights' Stats:
  2703 seqs, 102444704 total frames, 37900.371439 average frames
  Mean: 0.005538597677933737
  Std dev: 0.059030147981879444
  Min/max: 0.0 / 1.0
Layer 'dec_07_att_weights' Stats:
  2703 seqs, 102444704 total frames, 37900.371439 average frames
  Mean: 0.005538597676835271
  Std dev: 0.050605499844402234
  Min/max: 0.0 / 1.0
Layer 'dec_08_att_weights' Stats:
  2703 seqs, 102444704 total frames, 37900.371439 average frames
  Mean: 0.005538597693219386
  Std dev: 0.06267316889079952
  Min/max: 0.0 / 1.0
Layer 'dec_09_att_weights' Stats:
  2703 seqs, 102444704 total frames, 37900.371439 average frames
  Mean: 0.005538597679832804
  Std dev: 0.07420827004662686
  Min/max: 0.0 / 1.0
Layer 'dec_10_att_weights' Stats:
  2703 seqs, 102444704 total frames, 37900.371439 average frames
  Mean: 0.0055385976849342345
  Std dev: 0.07420965637587705
  Min/max: 0.0 / 1.0
Layer 'dec_11_att_weights' Stats:
  2703 seqs, 102444704 total frames, 37900.371439 average frames
  Mean: 0.005538597679423205
  Std dev: 0.07419887393118761
  Min/max: 0.0 / 1.0
Layer 'dec_12_att_weights' Stats:
  2703 seqs, 102444704 total frames, 37900.371439 average frames
  Mean: 0.005538597676481527
  Std dev: 0.07271670245538356
  Min/max: 0.0 / 1.0
Quitting
+------- EPILOGUE SCRIPT -----------------------------------------------
|
| Job ID ..............: 9506021
| Stopped at ..........: Tue Jul  2 13:13:21 CEST 2019
| Resources requested .: scratch_free=5G,h_fsize=20G,num_proc=5,pxe=ubuntu_16.04,h_vmem=1536G,s_core=0,h_rss=8G,h_rt=7200,gpu=1
| Resources used ......: cpu=00:54:19, mem=14573.12824 GB s, io=12.96986 GB, vmem=4.799G, maxvmem=4.802G, last_file_cache=3.580G, last_rss=3M, max-cache=4.420G
| Memory used .........: 8.000G / 8.000G (100.0%)
| Total time used .....: 0:38:03
|
+------- EPILOGUE SCRIPT -----------------------------------------------
