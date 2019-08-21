+------- PROLOGUE SCRIPT -----------------------------------------------
|
| Job ID ...........: 9507830
| Started at .......: Wed Jul  3 10:01:49 CEST 2019
| Execution host ...: cluster-cn-216
| Cluster queue ....: 3-GPU-1080
| Script ...........: /var/spool/sge/cluster-cn-216/job_scripts/9507830
| > #!/bin/bash
| > 
| > export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64:/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda/extras/CUPTI/lib64/:/usr/local/cuda-9.1/extras/CUPTI/lib64/:/usr/local/cuda-9.0/extras/CUPTI/lib64:/usr/lib/nvidia-418
| > #export LD_LIBRARY_PATH="/usr/local/lib::/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda-9.0/lib64:/usr/lib/nvidia-390:/opt/rbi/sge/lib/glinux:/opt/rbi/sge/lib/glinux"
| > 
| > source /u/bahar/settings/python3-returnn-tf1.9/bin/activate
| > 
| > python3 ~/returnn-parnia/tools/get-attention-weights.py //u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc.nospecaug.config --epoch 481 --data 'config:get_dataset("'$1'")' --dump_dir ~/makarov/NMT/hmm-factorization/experiments/asr2019/transformer/forward-$1 --layers "dec_02_att_weights" --layers "dec_01_att_weights" --layers "dec_03_att_weights" --layers "dec_04_att_weights" --layers "dec_05_att_weights" --layers "dec_06_att_weights" --layers "dec_07_att_weights" --layers "dec_08_att_weights" --layers "dec_09_att_weights" --layers "dec_10_att_weights" --layers "dec_11_att_weights" --layers "dec_12_att_weights" --rec_layer "output" --batch_size 4000 --tf_log_dir /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/transformer/forward-$1/tf_log_dir --reset_seq_ordering sorted_reverse
|
+------- PROLOGUE SCRIPT -----------------------------------------------
RETURNN get-attention-weights starting up.
RETURNN starting up, version 20181022.053313--git-dfada43-dirty, date/time 2019-07-03-10-01-53 (UTC+0200), pid 22080, cwd /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc, Python /u/bahar/settings/python3-returnn-tf1.9/bin/python3
RETURNN config: //u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc.nospecaug.config
RETURNN command line options: ()
Hostname: cluster-cn-216
TensorFlow: 1.10.0 (v1.10.0-0-g656e7a2b34) (<site-package> in /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow)
Error while getting SGE num_proc: FileNotFoundError(2, "No such file or directory: 'qstat'")
Setup TF inter and intra global thread pools, num_threads None, session opts {'device_count': {'GPU': 0}, 'log_device_placement': False}.
CUDA_VISIBLE_DEVICES is set to '0'.
Collecting TensorFlow device list...
Local devices available to TensorFlow:
  1/2: name: "/device:CPU:0"
       device_type: "CPU"
       memory_limit: 268435456
       locality {
       }
       incarnation: 16327554349481351336
  2/2: name: "/device:GPU:0"
       device_type: "GPU"
       memory_limit: 10911236096
       locality {
         bus_id: 1
         links {
         }
       }
       incarnation: 6563626076326418031
       physical_device_desc: "device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1"
Using gpu device 0: GeForce GTX 1080 Ti
Device not set explicitly, and we found a GPU, which we will use.
Model file prefix: /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc/net-model/network
NOTE: We will use 'sorted_reverse' seq ordering.
Setup tf.Session with options {'device_count': {'GPU': 1}, 'log_device_placement': False} ...
layer root/'data' output: Data(name='data', shape=(None, 40))
layer root/'source' output: Data(name='source_output', shape=(None, 40))
layer root/'lstm0_fw' output: Data(name='lstm0_fw_output', shape=(None, 1024), batch_dim_axis=1)
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9507830.1.3-GPU-1080/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.cc -o /var/tmp/9507830.1.3-GPU-1080/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9507830.1.3-GPU-1080/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.cc -o /var/tmp/9507830.1.3-GPU-1080/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
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
layer root/'dec_02_att_value0' output: Data(name='dec_02_att_value0_output', shape=(None, 512))
layer root/'dec_02_att_value' output: Data(name='dec_02_att_value_output', shape=(None, 8, 64))
layer root/'dec_09_att_key0' output: Data(name='dec_09_att_key0_output', shape=(None, 512))
layer root/'dec_09_att_key' output: Data(name='dec_09_att_key_output', shape=(None, 8, 64))
layer root/'dec_02_att_key0' output: Data(name='dec_02_att_key0_output', shape=(None, 512))
layer root/'dec_02_att_key' output: Data(name='dec_02_att_key_output', shape=(None, 8, 64))
layer root/'dec_12_att_value0' output: Data(name='dec_12_att_value0_output', shape=(None, 512))
layer root/'dec_12_att_value' output: Data(name='dec_12_att_value_output', shape=(None, 8, 64))
layer root/'dec_12_att_key0' output: Data(name='dec_12_att_key0_output', shape=(None, 512))
layer root/'dec_12_att_key' output: Data(name='dec_12_att_key_output', shape=(None, 8, 64))
layer root/'dec_11_att_value0' output: Data(name='dec_11_att_value0_output', shape=(None, 512))
layer root/'dec_11_att_value' output: Data(name='dec_11_att_value_output', shape=(None, 8, 64))
layer root/'dec_10_att_key0' output: Data(name='dec_10_att_key0_output', shape=(None, 512))
layer root/'dec_10_att_key' output: Data(name='dec_10_att_key_output', shape=(None, 8, 64))
layer root/'dec_03_att_value0' output: Data(name='dec_03_att_value0_output', shape=(None, 512))
layer root/'dec_03_att_value' output: Data(name='dec_03_att_value_output', shape=(None, 8, 64))
layer root/'dec_04_att_value0' output: Data(name='dec_04_att_value0_output', shape=(None, 512))
layer root/'dec_04_att_value' output: Data(name='dec_04_att_value_output', shape=(None, 8, 64))
layer root/'dec_01_att_key0' output: Data(name='dec_01_att_key0_output', shape=(None, 512))
layer root/'dec_01_att_key' output: Data(name='dec_01_att_key_output', shape=(None, 8, 64))
layer root/'dec_06_att_key0' output: Data(name='dec_06_att_key0_output', shape=(None, 512))
layer root/'dec_06_att_key' output: Data(name='dec_06_att_key_output', shape=(None, 8, 64))
layer root/'dec_10_att_value0' output: Data(name='dec_10_att_value0_output', shape=(None, 512))
layer root/'dec_10_att_value' output: Data(name='dec_10_att_value_output', shape=(None, 8, 64))
layer root/'dec_04_att_key0' output: Data(name='dec_04_att_key0_output', shape=(None, 512))
layer root/'dec_04_att_key' output: Data(name='dec_04_att_key_output', shape=(None, 8, 64))
layer root/'dec_07_att_value0' output: Data(name='dec_07_att_value0_output', shape=(None, 512))
layer root/'dec_07_att_value' output: Data(name='dec_07_att_value_output', shape=(None, 8, 64))
layer root/'dec_03_att_key0' output: Data(name='dec_03_att_key0_output', shape=(None, 512))
layer root/'dec_03_att_key' output: Data(name='dec_03_att_key_output', shape=(None, 8, 64))
layer root/'dec_06_att_value0' output: Data(name='dec_06_att_value0_output', shape=(None, 512))
layer root/'dec_06_att_value' output: Data(name='dec_06_att_value_output', shape=(None, 8, 64))
layer root/'dec_05_att_value0' output: Data(name='dec_05_att_value0_output', shape=(None, 512))
layer root/'dec_05_att_value' output: Data(name='dec_05_att_value_output', shape=(None, 8, 64))
layer root/'dec_01_att_value0' output: Data(name='dec_01_att_value0_output', shape=(None, 512))
layer root/'dec_01_att_value' output: Data(name='dec_01_att_value_output', shape=(None, 8, 64))
layer root/'dec_08_att_value0' output: Data(name='dec_08_att_value0_output', shape=(None, 512))
layer root/'dec_08_att_value' output: Data(name='dec_08_att_value_output', shape=(None, 8, 64))
layer root/'dec_08_att_key0' output: Data(name='dec_08_att_key0_output', shape=(None, 512))
layer root/'dec_08_att_key' output: Data(name='dec_08_att_key_output', shape=(None, 8, 64))
layer root/'dec_09_att_value0' output: Data(name='dec_09_att_value0_output', shape=(None, 512))
layer root/'dec_09_att_value' output: Data(name='dec_09_att_value_output', shape=(None, 8, 64))
layer root/'dec_05_att_key0' output: Data(name='dec_05_att_key0_output', shape=(None, 512))
layer root/'dec_05_att_key' output: Data(name='dec_05_att_key_output', shape=(None, 8, 64))
layer root/'dec_07_att_key0' output: Data(name='dec_07_att_key0_output', shape=(None, 512))
layer root/'dec_07_att_key' output: Data(name='dec_07_att_key_output', shape=(None, 8, 64))
layer root/'output' output: Data(name='output_output', shape=(None,), dtype='int32', sparse=True, dim=10025, batch_dim_axis=1)
Rec layer sub net:
  Input layers moved out of loop: (#: 0)
    None
  Output layers moved out of loop: (#: 270)
    output_prob
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
layer root/output:rec-subnet-output/'output_prob' output: Data(name='output_prob_output', shape=(None, 10025), batch_dim_axis=1)
layer root/'decision' output: Data(name='output_output', shape=(None,), dtype='int32', sparse=True, dim=10025, batch_dim_axis=1)
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
net params #: 138571347
net trainable params: [<tf.Variable 'ctc/W:0' shape=(512, 10026) dtype=float32_ref>, <tf.Variable 'ctc/b:0' shape=(10026,) dtype=float32_ref>, <tf.Variable 'dec_01_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_01_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_02_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_02_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_03_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_03_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_04_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_04_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_05_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_05_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_06_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_06_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_07_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_07_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_08_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_08_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_09_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_09_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_10_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_10_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_11_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_11_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_12_att_key0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'dec_12_att_value0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_01_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_01_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_01_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_01_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_01_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_01_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_01_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_01_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_01_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_01_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_02_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_02_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_02_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_02_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_02_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_02_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_02_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_02_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_02_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_02_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_03_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_03_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_03_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_03_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_03_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_03_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_03_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_03_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_03_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_03_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_04_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_04_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_04_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_04_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_04_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_04_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_04_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_04_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_04_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_04_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_05_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_05_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_05_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_05_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_05_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_05_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_05_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_05_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_05_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_05_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_06_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_06_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_06_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_06_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_06_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_06_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_06_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_06_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_06_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_06_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_07_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_07_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_07_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_07_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_07_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_07_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_07_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_07_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_07_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_07_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_08_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_08_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_08_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_08_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_08_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_08_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_08_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_08_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_08_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_08_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_09_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_09_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_09_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_09_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_09_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_09_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_09_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_09_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_09_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_09_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_10_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_10_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_10_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_10_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_10_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_10_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_10_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_10_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_10_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_10_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_11_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_11_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_11_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_11_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_11_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_11_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_11_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_11_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_11_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_11_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'enc_12_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'enc_12_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'enc_12_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'enc_12_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_12_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_12_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_12_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'enc_12_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_12_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'enc_12_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'encoder/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'encoder/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'lstm0_bw/rec/W:0' shape=(40, 4096) dtype=float32_ref>, <tf.Variable 'lstm0_bw/rec/W_re:0' shape=(1024, 4096) dtype=float32_ref>, <tf.Variable 'lstm0_bw/rec/b:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'lstm0_fw/rec/W:0' shape=(40, 4096) dtype=float32_ref>, <tf.Variable 'lstm0_fw/rec/W_re:0' shape=(1024, 4096) dtype=float32_ref>, <tf.Variable 'lstm0_fw/rec/b:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'lstm1_bw/rec/W:0' shape=(2048, 4096) dtype=float32_ref>, <tf.Variable 'lstm1_bw/rec/W_re:0' shape=(1024, 4096) dtype=float32_ref>, <tf.Variable 'lstm1_bw/rec/b:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'lstm1_fw/rec/W:0' shape=(2048, 4096) dtype=float32_ref>, <tf.Variable 'lstm1_fw/rec/W_re:0' shape=(1024, 4096) dtype=float32_ref>, <tf.Variable 'lstm1_fw/rec/b:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_01_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_02_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_03_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_04_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_05_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_06_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_07_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_08_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_09_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_10_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_11_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_att_query0/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_ff_conv1/W:0' shape=(512, 2048) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_ff_conv1/b:0' shape=(2048,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_ff_conv2/W:0' shape=(2048, 512) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_ff_conv2/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_ff_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_ff_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_self_att_att/QKV:0' shape=(512, 1536) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_self_att_laynorm/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_self_att_laynorm/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/dec_12_self_att_lin/W:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'output/rec/decoder/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/decoder/scale:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'output/rec/output_prob/W:0' shape=(512, 10025) dtype=float32_ref>, <tf.Variable 'output/rec/output_prob/b:0' shape=(10025,) dtype=float32_ref>, <tf.Variable 'output/rec/target_embed_raw/W:0' shape=(10025, 512) dtype=float32_ref>, <tf.Variable 'source_embed_raw/W:0' shape=(2048, 512) dtype=float32_ref>]
loading weights from /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc/net-model/network.481
TF: log_dir: /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/transformer/forward-dev-other/tf_log_dir/prefix:dev-other-481-2019-07-03-08-01-51
Note: There are still these uninitialized variables: ['learning_rate:0']
att-weights epoch 481, step 0, max_size:classes 94, max_size:data 3206, mem_usage:GPU:0 801.3MB, num_seqs 1, 13.941 sec/step, elapsed 0:00:19, exp. remaining 1:26:19, complete 0.38%
att-weights epoch 481, step 1, max_size:classes 93, max_size:data 3516, mem_usage:GPU:0 801.3MB, num_seqs 1, 3.039 sec/step, elapsed 0:00:27, exp. remaining 1:48:35, complete 0.42%
att-weights epoch 481, step 2, max_size:classes 83, max_size:data 3322, mem_usage:GPU:0 801.3MB, num_seqs 1, 2.267 sec/step, elapsed 0:00:30, exp. remaining 1:50:33, complete 0.45%
att-weights epoch 481, step 3, max_size:classes 90, max_size:data 2413, mem_usage:GPU:0 801.3MB, num_seqs 1, 2.136 sec/step, elapsed 0:00:37, exp. remaining 2:06:31, complete 0.49%
att-weights epoch 481, step 4, max_size:classes 79, max_size:data 2611, mem_usage:GPU:0 801.3MB, num_seqs 1, 2.456 sec/step, elapsed 0:00:40, exp. remaining 2:08:07, complete 0.52%
att-weights epoch 481, step 5, max_size:classes 84, max_size:data 3350, mem_usage:GPU:0 801.3MB, num_seqs 1, 4.560 sec/step, elapsed 0:00:46, exp. remaining 2:18:01, complete 0.56%
att-weights epoch 481, step 6, max_size:classes 78, max_size:data 2592, mem_usage:GPU:0 801.3MB, num_seqs 1, 3.445 sec/step, elapsed 0:00:50, exp. remaining 2:20:23, complete 0.59%
att-weights epoch 481, step 7, max_size:classes 80, max_size:data 2232, mem_usage:GPU:0 801.3MB, num_seqs 1, 1.271 sec/step, elapsed 0:00:51, exp. remaining 2:16:51, complete 0.63%
att-weights epoch 481, step 8, max_size:classes 83, max_size:data 2463, mem_usage:GPU:0 801.3MB, num_seqs 1, 2.618 sec/step, elapsed 0:00:54, exp. remaining 2:17:00, complete 0.66%
att-weights epoch 481, step 9, max_size:classes 89, max_size:data 2577, mem_usage:GPU:0 801.3MB, num_seqs 1, 1.488 sec/step, elapsed 0:00:57, exp. remaining 2:15:10, complete 0.70%
att-weights epoch 481, step 10, max_size:classes 92, max_size:data 2618, mem_usage:GPU:0 801.3MB, num_seqs 1, 1.205 sec/step, elapsed 0:00:58, exp. remaining 2:12:40, complete 0.73%
att-weights epoch 481, step 11, max_size:classes 86, max_size:data 3177, mem_usage:GPU:0 801.3MB, num_seqs 1, 1.880 sec/step, elapsed 0:01:01, exp. remaining 2:11:28, complete 0.77%
att-weights epoch 481, step 12, max_size:classes 82, max_size:data 2822, mem_usage:GPU:0 801.3MB, num_seqs 1, 1.768 sec/step, elapsed 0:01:02, exp. remaining 2:09:21, complete 0.80%
att-weights epoch 481, step 13, max_size:classes 73, max_size:data 2951, mem_usage:GPU:0 801.3MB, num_seqs 1, 2.037 sec/step, elapsed 0:01:04, exp. remaining 2:07:56, complete 0.84%
att-weights epoch 481, step 14, max_size:classes 72, max_size:data 2463, mem_usage:GPU:0 801.3MB, num_seqs 1, 1.330 sec/step, elapsed 0:01:06, exp. remaining 2:05:17, complete 0.87%
att-weights epoch 481, step 15, max_size:classes 76, max_size:data 2211, mem_usage:GPU:0 801.3MB, num_seqs 1, 1.680 sec/step, elapsed 0:01:07, exp. remaining 2:03:29, complete 0.91%
att-weights epoch 481, step 16, max_size:classes 78, max_size:data 2964, mem_usage:GPU:0 801.3MB, num_seqs 1, 1.383 sec/step, elapsed 0:01:09, exp. remaining 2:01:18, complete 0.94%
att-weights epoch 481, step 17, max_size:classes 76, max_size:data 2016, mem_usage:GPU:0 801.3MB, num_seqs 1, 4.685 sec/step, elapsed 0:01:13, exp. remaining 2:04:53, complete 0.98%
att-weights epoch 481, step 18, max_size:classes 87, max_size:data 2598, mem_usage:GPU:0 801.3MB, num_seqs 1, 2.625 sec/step, elapsed 0:01:16, exp. remaining 2:04:49, complete 1.01%
att-weights epoch 481, step 19, max_size:classes 74, max_size:data 2578, mem_usage:GPU:0 801.3MB, num_seqs 1, 7.461 sec/step, elapsed 0:01:24, exp. remaining 2:12:22, complete 1.05%
att-weights epoch 481, step 20, max_size:classes 110, max_size:data 2909, mem_usage:GPU:0 801.3MB, num_seqs 1, 1.796 sec/step, elapsed 0:01:25, exp. remaining 2:10:47, complete 1.08%
att-weights epoch 481, step 21, max_size:classes 70, max_size:data 1753, mem_usage:GPU:0 801.3MB, num_seqs 1, 1.604 sec/step, elapsed 0:01:27, exp. remaining 2:09:01, complete 1.12%
att-weights epoch 481, step 22, max_size:classes 77, max_size:data 2778, mem_usage:GPU:0 801.3MB, num_seqs 1, 1.426 sec/step, elapsed 0:01:28, exp. remaining 2:07:07, complete 1.15%
att-weights epoch 481, step 23, max_size:classes 83, max_size:data 2699, mem_usage:GPU:0 801.3MB, num_seqs 1, 2.055 sec/step, elapsed 0:01:30, exp. remaining 2:06:11, complete 1.19%
att-weights epoch 481, step 24, max_size:classes 63, max_size:data 3014, mem_usage:GPU:0 801.3MB, num_seqs 1, 3.919 sec/step, elapsed 0:01:34, exp. remaining 2:07:49, complete 1.22%
att-weights epoch 481, step 25, max_size:classes 63, max_size:data 3232, mem_usage:GPU:0 801.3MB, num_seqs 1, 2.664 sec/step, elapsed 0:01:37, exp. remaining 2:04:12, complete 1.29%
att-weights epoch 481, step 26, max_size:classes 62, max_size:data 1935, mem_usage:GPU:0 801.3MB, num_seqs 1, 2.037 sec/step, elapsed 0:01:39, exp. remaining 2:00:13, complete 1.36%
att-weights epoch 481, step 27, max_size:classes 69, max_size:data 2024, mem_usage:GPU:0 801.3MB, num_seqs 1, 6.457 sec/step, elapsed 0:01:46, exp. remaining 2:01:41, complete 1.43%
att-weights epoch 481, step 28, max_size:classes 66, max_size:data 1976, mem_usage:GPU:0 801.3MB, num_seqs 1, 2.509 sec/step, elapsed 0:01:48, exp. remaining 2:01:33, complete 1.47%
att-weights epoch 481, step 29, max_size:classes 69, max_size:data 2385, mem_usage:GPU:0 801.3MB, num_seqs 1, 1.804 sec/step, elapsed 0:01:50, exp. remaining 2:00:39, complete 1.50%
att-weights epoch 481, step 30, max_size:classes 58, max_size:data 2818, mem_usage:GPU:0 801.3MB, num_seqs 1, 2.157 sec/step, elapsed 0:01:52, exp. remaining 2:00:11, complete 1.54%
att-weights epoch 481, step 31, max_size:classes 60, max_size:data 1849, mem_usage:GPU:0 801.3MB, num_seqs 1, 1.566 sec/step, elapsed 0:01:54, exp. remaining 1:59:06, complete 1.57%
att-weights epoch 481, step 32, max_size:classes 67, max_size:data 2141, mem_usage:GPU:0 801.3MB, num_seqs 1, 1.483 sec/step, elapsed 0:01:55, exp. remaining 1:57:59, complete 1.61%
att-weights epoch 481, step 33, max_size:classes 61, max_size:data 2119, mem_usage:GPU:0 801.3MB, num_seqs 1, 1.221 sec/step, elapsed 0:01:56, exp. remaining 1:56:39, complete 1.64%
att-weights epoch 481, step 34, max_size:classes 64, max_size:data 1820, mem_usage:GPU:0 801.3MB, num_seqs 2, 2.189 sec/step, elapsed 0:01:58, exp. remaining 1:53:55, complete 1.71%
att-weights epoch 481, step 35, max_size:classes 64, max_size:data 1807, mem_usage:GPU:0 801.3MB, num_seqs 2, 2.654 sec/step, elapsed 0:02:01, exp. remaining 1:51:48, complete 1.78%
att-weights epoch 481, step 36, max_size:classes 66, max_size:data 1775, mem_usage:GPU:0 801.3MB, num_seqs 2, 3.057 sec/step, elapsed 0:02:04, exp. remaining 1:50:13, complete 1.85%
att-weights epoch 481, step 37, max_size:classes 61, max_size:data 1991, mem_usage:GPU:0 801.3MB, num_seqs 1, 2.218 sec/step, elapsed 0:02:06, exp. remaining 1:48:01, complete 1.92%
att-weights epoch 481, step 38, max_size:classes 63, max_size:data 2176, mem_usage:GPU:0 801.3MB, num_seqs 1, 6.691 sec/step, elapsed 0:02:13, exp. remaining 1:49:39, complete 1.99%
att-weights epoch 481, step 39, max_size:classes 65, max_size:data 1634, mem_usage:GPU:0 801.3MB, num_seqs 1, 2.128 sec/step, elapsed 0:02:15, exp. remaining 1:47:33, complete 2.06%
att-weights epoch 481, step 40, max_size:classes 58, max_size:data 2835, mem_usage:GPU:0 801.3MB, num_seqs 1, 3.716 sec/step, elapsed 0:02:19, exp. remaining 1:46:47, complete 2.13%
att-weights epoch 481, step 41, max_size:classes 60, max_size:data 1838, mem_usage:GPU:0 801.3MB, num_seqs 1, 11.726 sec/step, elapsed 0:02:31, exp. remaining 1:52:01, complete 2.20%
att-weights epoch 481, step 42, max_size:classes 68, max_size:data 2190, mem_usage:GPU:0 801.3MB, num_seqs 1, 2.556 sec/step, elapsed 0:02:33, exp. remaining 1:50:19, complete 2.27%
att-weights epoch 481, step 43, max_size:classes 63, max_size:data 1632, mem_usage:GPU:0 801.3MB, num_seqs 2, 2.716 sec/step, elapsed 0:02:36, exp. remaining 1:50:32, complete 2.30%
att-weights epoch 481, step 44, max_size:classes 59, max_size:data 1934, mem_usage:GPU:0 801.3MB, num_seqs 2, 6.316 sec/step, elapsed 0:02:42, exp. remaining 1:53:14, complete 2.34%
att-weights epoch 481, step 45, max_size:classes 64, max_size:data 1987, mem_usage:GPU:0 801.3MB, num_seqs 2, 3.919 sec/step, elapsed 0:02:46, exp. remaining 1:52:32, complete 2.41%
att-weights epoch 481, step 46, max_size:classes 62, max_size:data 1990, mem_usage:GPU:0 801.3MB, num_seqs 2, 6.578 sec/step, elapsed 0:02:53, exp. remaining 1:53:35, complete 2.48%
att-weights epoch 481, step 47, max_size:classes 59, max_size:data 1627, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.798 sec/step, elapsed 0:02:55, exp. remaining 1:51:33, complete 2.55%
att-weights epoch 481, step 48, max_size:classes 58, max_size:data 1779, mem_usage:GPU:0 801.3MB, num_seqs 2, 5.472 sec/step, elapsed 0:03:00, exp. remaining 1:51:53, complete 2.62%
att-weights epoch 481, step 49, max_size:classes 55, max_size:data 1813, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.188 sec/step, elapsed 0:03:01, exp. remaining 1:51:06, complete 2.65%
att-weights epoch 481, step 50, max_size:classes 58, max_size:data 1785, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.316 sec/step, elapsed 0:03:03, exp. remaining 1:50:25, complete 2.69%
att-weights epoch 481, step 51, max_size:classes 60, max_size:data 1824, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.227 sec/step, elapsed 0:03:04, exp. remaining 1:49:41, complete 2.72%
att-weights epoch 481, step 52, max_size:classes 57, max_size:data 1478, mem_usage:GPU:0 801.3MB, num_seqs 1, 0.635 sec/step, elapsed 0:03:04, exp. remaining 1:47:14, complete 2.79%
att-weights epoch 481, step 53, max_size:classes 64, max_size:data 2023, mem_usage:GPU:0 801.3MB, num_seqs 1, 0.792 sec/step, elapsed 0:03:05, exp. remaining 1:46:20, complete 2.83%
att-weights epoch 481, step 54, max_size:classes 62, max_size:data 1990, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.433 sec/step, elapsed 0:03:07, exp. remaining 1:44:30, complete 2.90%
att-weights epoch 481, step 55, max_size:classes 56, max_size:data 1850, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.174 sec/step, elapsed 0:03:08, exp. remaining 1:43:52, complete 2.93%
att-weights epoch 481, step 56, max_size:classes 54, max_size:data 1942, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.308 sec/step, elapsed 0:03:09, exp. remaining 1:43:19, complete 2.97%
att-weights epoch 481, step 57, max_size:classes 61, max_size:data 1607, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.186 sec/step, elapsed 0:03:10, exp. remaining 1:41:30, complete 3.04%
att-weights epoch 481, step 58, max_size:classes 60, max_size:data 1529, mem_usage:GPU:0 801.3MB, num_seqs 1, 0.729 sec/step, elapsed 0:03:11, exp. remaining 1:39:32, complete 3.11%
att-weights epoch 481, step 59, max_size:classes 62, max_size:data 2044, mem_usage:GPU:0 801.3MB, num_seqs 1, 0.800 sec/step, elapsed 0:03:12, exp. remaining 1:37:40, complete 3.18%
att-weights epoch 481, step 60, max_size:classes 55, max_size:data 2007, mem_usage:GPU:0 801.3MB, num_seqs 1, 0.764 sec/step, elapsed 0:03:13, exp. remaining 1:35:53, complete 3.25%
att-weights epoch 481, step 61, max_size:classes 51, max_size:data 1820, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.064 sec/step, elapsed 0:03:14, exp. remaining 1:34:19, complete 3.32%
att-weights epoch 481, step 62, max_size:classes 62, max_size:data 2501, mem_usage:GPU:0 801.3MB, num_seqs 1, 1.652 sec/step, elapsed 0:03:15, exp. remaining 1:33:05, complete 3.39%
att-weights epoch 481, step 63, max_size:classes 59, max_size:data 1603, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.156 sec/step, elapsed 0:03:16, exp. remaining 1:31:41, complete 3.46%
att-weights epoch 481, step 64, max_size:classes 55, max_size:data 1503, mem_usage:GPU:0 801.3MB, num_seqs 1, 0.645 sec/step, elapsed 0:03:17, exp. remaining 1:30:06, complete 3.53%
att-weights epoch 481, step 65, max_size:classes 58, max_size:data 2102, mem_usage:GPU:0 801.3MB, num_seqs 1, 0.881 sec/step, elapsed 0:03:18, exp. remaining 1:28:41, complete 3.60%
att-weights epoch 481, step 66, max_size:classes 56, max_size:data 1594, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.049 sec/step, elapsed 0:03:19, exp. remaining 1:27:23, complete 3.67%
att-weights epoch 481, step 67, max_size:classes 57, max_size:data 1603, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.225 sec/step, elapsed 0:03:20, exp. remaining 1:26:13, complete 3.74%
att-weights epoch 481, step 68, max_size:classes 57, max_size:data 1912, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.154 sec/step, elapsed 0:03:21, exp. remaining 1:25:03, complete 3.81%
att-weights epoch 481, step 69, max_size:classes 50, max_size:data 1795, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.139 sec/step, elapsed 0:03:23, exp. remaining 1:23:56, complete 3.88%
att-weights epoch 481, step 70, max_size:classes 55, max_size:data 1574, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.168 sec/step, elapsed 0:03:24, exp. remaining 1:23:38, complete 3.91%
att-weights epoch 481, step 71, max_size:classes 53, max_size:data 1496, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.974 sec/step, elapsed 0:03:25, exp. remaining 1:23:16, complete 3.95%
att-weights epoch 481, step 72, max_size:classes 58, max_size:data 1601, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.043 sec/step, elapsed 0:03:26, exp. remaining 1:22:10, complete 4.02%
att-weights epoch 481, step 73, max_size:classes 51, max_size:data 1366, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.891 sec/step, elapsed 0:03:27, exp. remaining 1:21:03, complete 4.09%
att-weights epoch 481, step 74, max_size:classes 55, max_size:data 1479, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.004 sec/step, elapsed 0:03:28, exp. remaining 1:20:01, complete 4.16%
att-weights epoch 481, step 75, max_size:classes 48, max_size:data 1657, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.906 sec/step, elapsed 0:03:29, exp. remaining 1:18:59, complete 4.22%
att-weights epoch 481, step 76, max_size:classes 53, max_size:data 1560, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.047 sec/step, elapsed 0:03:30, exp. remaining 1:18:02, complete 4.29%
att-weights epoch 481, step 77, max_size:classes 58, max_size:data 1576, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.056 sec/step, elapsed 0:03:31, exp. remaining 1:17:07, complete 4.36%
att-weights epoch 481, step 78, max_size:classes 54, max_size:data 1657, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.078 sec/step, elapsed 0:03:32, exp. remaining 1:16:14, complete 4.43%
att-weights epoch 481, step 79, max_size:classes 50, max_size:data 1349, mem_usage:GPU:0 801.3MB, num_seqs 1, 0.517 sec/step, elapsed 0:03:32, exp. remaining 1:15:10, complete 4.50%
att-weights epoch 481, step 80, max_size:classes 53, max_size:data 2082, mem_usage:GPU:0 801.3MB, num_seqs 1, 0.833 sec/step, elapsed 0:03:33, exp. remaining 1:14:16, complete 4.57%
att-weights epoch 481, step 81, max_size:classes 54, max_size:data 1891, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.481 sec/step, elapsed 0:03:35, exp. remaining 1:14:11, complete 4.61%
att-weights epoch 481, step 82, max_size:classes 54, max_size:data 1364, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.971 sec/step, elapsed 0:03:36, exp. remaining 1:13:21, complete 4.68%
att-weights epoch 481, step 83, max_size:classes 45, max_size:data 1750, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.931 sec/step, elapsed 0:03:36, exp. remaining 1:12:32, complete 4.75%
att-weights epoch 481, step 84, max_size:classes 50, max_size:data 1418, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.898 sec/step, elapsed 0:03:37, exp. remaining 1:11:44, complete 4.82%
att-weights epoch 481, step 85, max_size:classes 61, max_size:data 1682, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.157 sec/step, elapsed 0:03:39, exp. remaining 1:11:01, complete 4.89%
att-weights epoch 481, step 86, max_size:classes 54, max_size:data 1357, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.990 sec/step, elapsed 0:03:40, exp. remaining 1:10:17, complete 4.96%
att-weights epoch 481, step 87, max_size:classes 55, max_size:data 1520, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.083 sec/step, elapsed 0:03:41, exp. remaining 1:09:36, complete 5.03%
att-weights epoch 481, step 88, max_size:classes 50, max_size:data 1434, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.970 sec/step, elapsed 0:03:42, exp. remaining 1:08:54, complete 5.10%
att-weights epoch 481, step 89, max_size:classes 52, max_size:data 1789, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.993 sec/step, elapsed 0:03:43, exp. remaining 1:08:13, complete 5.17%
att-weights epoch 481, step 90, max_size:classes 59, max_size:data 2713, mem_usage:GPU:0 801.3MB, num_seqs 1, 1.087 sec/step, elapsed 0:03:44, exp. remaining 1:07:35, complete 5.24%
att-weights epoch 481, step 91, max_size:classes 52, max_size:data 1987, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.156 sec/step, elapsed 0:03:45, exp. remaining 1:07:00, complete 5.31%
att-weights epoch 481, step 92, max_size:classes 51, max_size:data 1756, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.455 sec/step, elapsed 0:03:46, exp. remaining 1:06:30, complete 5.38%
att-weights epoch 481, step 93, max_size:classes 49, max_size:data 1426, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.055 sec/step, elapsed 0:03:47, exp. remaining 1:06:21, complete 5.41%
att-weights epoch 481, step 94, max_size:classes 48, max_size:data 1450, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.008 sec/step, elapsed 0:03:48, exp. remaining 1:05:45, complete 5.48%
att-weights epoch 481, step 95, max_size:classes 49, max_size:data 1606, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.014 sec/step, elapsed 0:03:49, exp. remaining 1:05:10, complete 5.55%
att-weights epoch 481, step 96, max_size:classes 49, max_size:data 1593, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.060 sec/step, elapsed 0:03:50, exp. remaining 1:04:36, complete 5.62%
att-weights epoch 481, step 97, max_size:classes 54, max_size:data 1701, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.092 sec/step, elapsed 0:03:52, exp. remaining 1:04:04, complete 5.69%
att-weights epoch 481, step 98, max_size:classes 47, max_size:data 1596, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.927 sec/step, elapsed 0:03:52, exp. remaining 1:03:30, complete 5.76%
att-weights epoch 481, step 99, max_size:classes 49, max_size:data 1878, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.102 sec/step, elapsed 0:03:54, exp. remaining 1:02:59, complete 5.83%
att-weights epoch 481, step 100, max_size:classes 55, max_size:data 1407, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.985 sec/step, elapsed 0:03:55, exp. remaining 1:02:27, complete 5.90%
att-weights epoch 481, step 101, max_size:classes 48, max_size:data 1524, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.933 sec/step, elapsed 0:03:55, exp. remaining 1:01:56, complete 5.97%
att-weights epoch 481, step 102, max_size:classes 52, max_size:data 2377, mem_usage:GPU:0 801.3MB, num_seqs 1, 0.883 sec/step, elapsed 0:03:56, exp. remaining 1:01:24, complete 6.04%
att-weights epoch 481, step 103, max_size:classes 52, max_size:data 1682, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.034 sec/step, elapsed 0:03:57, exp. remaining 1:00:55, complete 6.11%
att-weights epoch 481, step 104, max_size:classes 50, max_size:data 1310, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.960 sec/step, elapsed 0:03:58, exp. remaining 1:00:25, complete 6.18%
att-weights epoch 481, step 105, max_size:classes 46, max_size:data 1659, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.942 sec/step, elapsed 0:03:59, exp. remaining 0:59:56, complete 6.25%
att-weights epoch 481, step 106, max_size:classes 56, max_size:data 1578, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.047 sec/step, elapsed 0:04:00, exp. remaining 0:59:29, complete 6.32%
att-weights epoch 481, step 107, max_size:classes 49, max_size:data 1712, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.972 sec/step, elapsed 0:04:01, exp. remaining 0:59:02, complete 6.39%
att-weights epoch 481, step 108, max_size:classes 49, max_size:data 1813, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.970 sec/step, elapsed 0:04:02, exp. remaining 0:58:35, complete 6.46%
att-weights epoch 481, step 109, max_size:classes 48, max_size:data 1637, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.934 sec/step, elapsed 0:04:03, exp. remaining 0:58:08, complete 6.53%
att-weights epoch 481, step 110, max_size:classes 52, max_size:data 1820, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.145 sec/step, elapsed 0:04:04, exp. remaining 0:57:45, complete 6.60%
att-weights epoch 481, step 111, max_size:classes 43, max_size:data 1717, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.097 sec/step, elapsed 0:04:05, exp. remaining 0:57:22, complete 6.67%
att-weights epoch 481, step 112, max_size:classes 44, max_size:data 1363, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.833 sec/step, elapsed 0:04:06, exp. remaining 0:56:55, complete 6.74%
att-weights epoch 481, step 113, max_size:classes 51, max_size:data 1431, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.931 sec/step, elapsed 0:04:07, exp. remaining 0:56:30, complete 6.81%
att-weights epoch 481, step 114, max_size:classes 48, max_size:data 1879, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.979 sec/step, elapsed 0:04:08, exp. remaining 0:56:06, complete 6.88%
att-weights epoch 481, step 115, max_size:classes 50, max_size:data 1384, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.005 sec/step, elapsed 0:04:09, exp. remaining 0:55:44, complete 6.95%
att-weights epoch 481, step 116, max_size:classes 44, max_size:data 1359, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.873 sec/step, elapsed 0:04:10, exp. remaining 0:55:19, complete 7.02%
att-weights epoch 481, step 117, max_size:classes 47, max_size:data 1672, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.951 sec/step, elapsed 0:04:11, exp. remaining 0:54:57, complete 7.09%
att-weights epoch 481, step 118, max_size:classes 45, max_size:data 1593, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.941 sec/step, elapsed 0:04:12, exp. remaining 0:54:34, complete 7.16%
att-weights epoch 481, step 119, max_size:classes 45, max_size:data 1746, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.880 sec/step, elapsed 0:04:13, exp. remaining 0:54:11, complete 7.23%
att-weights epoch 481, step 120, max_size:classes 47, max_size:data 1635, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.905 sec/step, elapsed 0:04:14, exp. remaining 0:53:49, complete 7.30%
att-weights epoch 481, step 121, max_size:classes 48, max_size:data 1607, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.991 sec/step, elapsed 0:04:15, exp. remaining 0:53:29, complete 7.37%
att-weights epoch 481, step 122, max_size:classes 59, max_size:data 1396, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.997 sec/step, elapsed 0:04:16, exp. remaining 0:53:09, complete 7.44%
att-weights epoch 481, step 123, max_size:classes 46, max_size:data 1434, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.955 sec/step, elapsed 0:04:17, exp. remaining 0:52:49, complete 7.51%
att-weights epoch 481, step 124, max_size:classes 39, max_size:data 1269, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.830 sec/step, elapsed 0:04:18, exp. remaining 0:52:11, complete 7.61%
att-weights epoch 481, step 125, max_size:classes 46, max_size:data 1530, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.919 sec/step, elapsed 0:04:18, exp. remaining 0:51:52, complete 7.68%
att-weights epoch 481, step 126, max_size:classes 39, max_size:data 1782, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.909 sec/step, elapsed 0:04:19, exp. remaining 0:51:32, complete 7.75%
att-weights epoch 481, step 127, max_size:classes 43, max_size:data 1598, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.911 sec/step, elapsed 0:04:20, exp. remaining 0:51:13, complete 7.82%
att-weights epoch 481, step 128, max_size:classes 47, max_size:data 1236, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.980 sec/step, elapsed 0:04:21, exp. remaining 0:50:55, complete 7.89%
att-weights epoch 481, step 129, max_size:classes 43, max_size:data 1534, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.873 sec/step, elapsed 0:04:22, exp. remaining 0:50:36, complete 7.96%
att-weights epoch 481, step 130, max_size:classes 47, max_size:data 1391, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.783 sec/step, elapsed 0:04:23, exp. remaining 0:50:02, complete 8.07%
att-weights epoch 481, step 131, max_size:classes 43, max_size:data 1494, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.850 sec/step, elapsed 0:04:24, exp. remaining 0:49:44, complete 8.14%
att-weights epoch 481, step 132, max_size:classes 45, max_size:data 1531, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.876 sec/step, elapsed 0:04:25, exp. remaining 0:49:26, complete 8.21%
att-weights epoch 481, step 133, max_size:classes 41, max_size:data 1304, mem_usage:GPU:0 801.3MB, num_seqs 3, 1.102 sec/step, elapsed 0:04:26, exp. remaining 0:49:11, complete 8.28%
att-weights epoch 481, step 134, max_size:classes 45, max_size:data 1371, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.959 sec/step, elapsed 0:04:27, exp. remaining 0:48:54, complete 8.34%
att-weights epoch 481, step 135, max_size:classes 48, max_size:data 1380, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.911 sec/step, elapsed 0:04:28, exp. remaining 0:48:38, complete 8.41%
att-weights epoch 481, step 136, max_size:classes 45, max_size:data 1664, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.953 sec/step, elapsed 0:04:29, exp. remaining 0:48:09, complete 8.52%
att-weights epoch 481, step 137, max_size:classes 41, max_size:data 1115, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.688 sec/step, elapsed 0:04:29, exp. remaining 0:47:38, complete 8.62%
att-weights epoch 481, step 138, max_size:classes 41, max_size:data 1760, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.914 sec/step, elapsed 0:04:30, exp. remaining 0:47:10, complete 8.73%
att-weights epoch 481, step 139, max_size:classes 44, max_size:data 1268, mem_usage:GPU:0 801.3MB, num_seqs 3, 1.173 sec/step, elapsed 0:04:31, exp. remaining 0:46:57, complete 8.80%
att-weights epoch 481, step 140, max_size:classes 41, max_size:data 1466, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.967 sec/step, elapsed 0:04:32, exp. remaining 0:46:43, complete 8.87%
att-weights epoch 481, step 141, max_size:classes 41, max_size:data 1377, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.920 sec/step, elapsed 0:04:33, exp. remaining 0:46:28, complete 8.94%
att-weights epoch 481, step 142, max_size:classes 47, max_size:data 1233, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.917 sec/step, elapsed 0:04:34, exp. remaining 0:46:14, complete 9.01%
att-weights epoch 481, step 143, max_size:classes 40, max_size:data 1558, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.855 sec/step, elapsed 0:04:35, exp. remaining 0:45:47, complete 9.11%
att-weights epoch 481, step 144, max_size:classes 38, max_size:data 1389, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.845 sec/step, elapsed 0:04:36, exp. remaining 0:45:21, complete 9.22%
att-weights epoch 481, step 145, max_size:classes 42, max_size:data 1079, mem_usage:GPU:0 801.3MB, num_seqs 3, 0.927 sec/step, elapsed 0:04:37, exp. remaining 0:45:08, complete 9.29%
att-weights epoch 481, step 146, max_size:classes 41, max_size:data 1246, mem_usage:GPU:0 801.3MB, num_seqs 3, 1.185 sec/step, elapsed 0:04:38, exp. remaining 0:44:46, complete 9.39%
att-weights epoch 481, step 147, max_size:classes 42, max_size:data 1182, mem_usage:GPU:0 801.3MB, num_seqs 3, 0.979 sec/step, elapsed 0:04:39, exp. remaining 0:44:33, complete 9.46%
att-weights epoch 481, step 148, max_size:classes 40, max_size:data 1375, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.746 sec/step, elapsed 0:04:40, exp. remaining 0:44:08, complete 9.57%
att-weights epoch 481, step 149, max_size:classes 41, max_size:data 974, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.681 sec/step, elapsed 0:04:40, exp. remaining 0:43:53, complete 9.64%
att-weights epoch 481, step 150, max_size:classes 38, max_size:data 1528, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.918 sec/step, elapsed 0:04:41, exp. remaining 0:43:41, complete 9.71%
att-weights epoch 481, step 151, max_size:classes 43, max_size:data 1376, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.841 sec/step, elapsed 0:04:42, exp. remaining 0:43:28, complete 9.78%
att-weights epoch 481, step 152, max_size:classes 40, max_size:data 1006, mem_usage:GPU:0 801.3MB, num_seqs 3, 1.072 sec/step, elapsed 0:04:43, exp. remaining 0:43:07, complete 9.88%
att-weights epoch 481, step 153, max_size:classes 43, max_size:data 1126, mem_usage:GPU:0 801.3MB, num_seqs 3, 1.399 sec/step, elapsed 0:04:45, exp. remaining 0:42:49, complete 9.99%
att-weights epoch 481, step 154, max_size:classes 45, max_size:data 1416, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.846 sec/step, elapsed 0:04:45, exp. remaining 0:42:27, complete 10.09%
att-weights epoch 481, step 155, max_size:classes 39, max_size:data 1275, mem_usage:GPU:0 801.3MB, num_seqs 3, 1.178 sec/step, elapsed 0:04:47, exp. remaining 0:42:09, complete 10.20%
att-weights epoch 481, step 156, max_size:classes 39, max_size:data 1359, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.960 sec/step, elapsed 0:04:48, exp. remaining 0:41:58, complete 10.27%
att-weights epoch 481, step 157, max_size:classes 44, max_size:data 1266, mem_usage:GPU:0 801.3MB, num_seqs 3, 1.441 sec/step, elapsed 0:04:49, exp. remaining 0:41:51, complete 10.34%
att-weights epoch 481, step 158, max_size:classes 40, max_size:data 1008, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.811 sec/step, elapsed 0:04:50, exp. remaining 0:41:40, complete 10.41%
att-weights epoch 481, step 159, max_size:classes 43, max_size:data 1530, mem_usage:GPU:0 801.3MB, num_seqs 2, 5.609 sec/step, elapsed 0:04:55, exp. remaining 0:42:09, complete 10.47%
att-weights epoch 481, step 160, max_size:classes 33, max_size:data 1605, mem_usage:GPU:0 801.3MB, num_seqs 2, 2.121 sec/step, elapsed 0:04:58, exp. remaining 0:42:08, complete 10.54%
att-weights epoch 481, step 161, max_size:classes 46, max_size:data 1117, mem_usage:GPU:0 801.3MB, num_seqs 3, 1.309 sec/step, elapsed 0:04:59, exp. remaining 0:42:01, complete 10.61%
att-weights epoch 481, step 162, max_size:classes 41, max_size:data 1300, mem_usage:GPU:0 801.3MB, num_seqs 3, 1.034 sec/step, elapsed 0:05:00, exp. remaining 0:41:42, complete 10.72%
att-weights epoch 481, step 163, max_size:classes 41, max_size:data 1281, mem_usage:GPU:0 801.3MB, num_seqs 3, 12.554 sec/step, elapsed 0:05:12, exp. remaining 0:42:58, complete 10.82%
att-weights epoch 481, step 164, max_size:classes 41, max_size:data 1307, mem_usage:GPU:0 801.3MB, num_seqs 3, 1.369 sec/step, elapsed 0:05:14, exp. remaining 0:42:51, complete 10.89%
att-weights epoch 481, step 165, max_size:classes 41, max_size:data 1429, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.565 sec/step, elapsed 0:05:15, exp. remaining 0:42:36, complete 11.00%
att-weights epoch 481, step 166, max_size:classes 38, max_size:data 1771, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.919 sec/step, elapsed 0:05:16, exp. remaining 0:42:16, complete 11.10%
att-weights epoch 481, step 167, max_size:classes 42, max_size:data 1241, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.765 sec/step, elapsed 0:05:17, exp. remaining 0:42:04, complete 11.17%
att-weights epoch 481, step 168, max_size:classes 43, max_size:data 1530, mem_usage:GPU:0 801.3MB, num_seqs 2, 6.522 sec/step, elapsed 0:05:24, exp. remaining 0:42:38, complete 11.24%
att-weights epoch 481, step 169, max_size:classes 42, max_size:data 1191, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.891 sec/step, elapsed 0:05:25, exp. remaining 0:42:19, complete 11.35%
att-weights epoch 481, step 170, max_size:classes 43, max_size:data 1350, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.810 sec/step, elapsed 0:05:25, exp. remaining 0:42:07, complete 11.42%
att-weights epoch 481, step 171, max_size:classes 38, max_size:data 1203, mem_usage:GPU:0 801.3MB, num_seqs 3, 1.253 sec/step, elapsed 0:05:27, exp. remaining 0:42:00, complete 11.49%
att-weights epoch 481, step 172, max_size:classes 41, max_size:data 1228, mem_usage:GPU:0 801.3MB, num_seqs 3, 1.086 sec/step, elapsed 0:05:28, exp. remaining 0:41:51, complete 11.56%
att-weights epoch 481, step 173, max_size:classes 42, max_size:data 1591, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.866 sec/step, elapsed 0:05:29, exp. remaining 0:41:32, complete 11.66%
att-weights epoch 481, step 174, max_size:classes 43, max_size:data 1327, mem_usage:GPU:0 801.3MB, num_seqs 3, 1.776 sec/step, elapsed 0:05:30, exp. remaining 0:41:28, complete 11.73%
att-weights epoch 481, step 175, max_size:classes 41, max_size:data 1307, mem_usage:GPU:0 801.3MB, num_seqs 3, 1.626 sec/step, elapsed 0:05:32, exp. remaining 0:41:24, complete 11.80%
att-weights epoch 481, step 176, max_size:classes 37, max_size:data 1059, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.785 sec/step, elapsed 0:05:33, exp. remaining 0:41:05, complete 11.91%
att-weights epoch 481, step 177, max_size:classes 41, max_size:data 1392, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.884 sec/step, elapsed 0:05:34, exp. remaining 0:40:47, complete 12.01%
att-weights epoch 481, step 178, max_size:classes 37, max_size:data 1081, mem_usage:GPU:0 801.3MB, num_seqs 3, 1.093 sec/step, elapsed 0:05:35, exp. remaining 0:40:39, complete 12.08%
att-weights epoch 481, step 179, max_size:classes 38, max_size:data 1461, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.890 sec/step, elapsed 0:05:36, exp. remaining 0:40:21, complete 12.19%
att-weights epoch 481, step 180, max_size:classes 44, max_size:data 1108, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.835 sec/step, elapsed 0:05:36, exp. remaining 0:40:04, complete 12.29%
att-weights epoch 481, step 181, max_size:classes 41, max_size:data 1351, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.893 sec/step, elapsed 0:05:37, exp. remaining 0:39:47, complete 12.40%
att-weights epoch 481, step 182, max_size:classes 38, max_size:data 1161, mem_usage:GPU:0 801.3MB, num_seqs 3, 1.147 sec/step, elapsed 0:05:38, exp. remaining 0:39:32, complete 12.50%
att-weights epoch 481, step 183, max_size:classes 41, max_size:data 1526, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.997 sec/step, elapsed 0:05:39, exp. remaining 0:39:24, complete 12.57%
att-weights epoch 481, step 184, max_size:classes 40, max_size:data 1952, mem_usage:GPU:0 801.3MB, num_seqs 2, 1.106 sec/step, elapsed 0:05:41, exp. remaining 0:39:09, complete 12.67%
att-weights epoch 481, step 185, max_size:classes 42, max_size:data 1272, mem_usage:GPU:0 801.3MB, num_seqs 3, 2.186 sec/step, elapsed 0:05:43, exp. remaining 0:39:02, complete 12.78%
att-weights epoch 481, step 186, max_size:classes 34, max_size:data 1238, mem_usage:GPU:0 801.3MB, num_seqs 3, 1.659 sec/step, elapsed 0:05:44, exp. remaining 0:38:59, complete 12.85%
att-weights epoch 481, step 187, max_size:classes 47, max_size:data 1351, mem_usage:GPU:0 801.3MB, num_seqs 2, 0.869 sec/step, elapsed 0:05:45, exp. remaining 0:38:43, complete 12.95%
att-weights epoch 481, step 188, max_size:classes 38, max_size:data 1263, mem_usage:GPU:0 801.3MB, num_seqs 3, 1.139 sec/step, elapsed 0:05:46, exp. remaining 0:38:22, complete 13.09%
att-weights epoch 481, step 189, max_size:classes 38, max_size:data 1061, mem_usage:GPU:0 801.3MB, num_seqs 3, 1.024 sec/step, elapsed 0:05:47, exp. remaining 0:38:15, complete 13.16%
att-weights epoch 481, step 190, max_size:classes 37, max_size:data 1254, mem_usage:GPU:0 801.3MB, num_seqs 3, 1.078 sec/step, elapsed 0:05:49, exp. remaining 0:38:01, complete 13.27%
att-weights epoch 481, step 191, max_size:classes 41, max_size:data 1299, mem_usage:GPU:0 0.8GB, num_seqs 3, 3.338 sec/step, elapsed 0:05:52, exp. remaining 0:38:02, complete 13.37%
att-weights epoch 481, step 192, max_size:classes 38, max_size:data 1447, mem_usage:GPU:0 0.8GB, num_seqs 2, 0.891 sec/step, elapsed 0:05:53, exp. remaining 0:37:47, complete 13.48%
att-weights epoch 481, step 193, max_size:classes 37, max_size:data 1163, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.535 sec/step, elapsed 0:05:54, exp. remaining 0:37:37, complete 13.58%
att-weights epoch 481, step 194, max_size:classes 43, max_size:data 1024, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.114 sec/step, elapsed 0:05:55, exp. remaining 0:37:24, complete 13.69%
att-weights epoch 481, step 195, max_size:classes 33, max_size:data 1395, mem_usage:GPU:0 0.8GB, num_seqs 2, 0.869 sec/step, elapsed 0:05:56, exp. remaining 0:37:10, complete 13.79%
att-weights epoch 481, step 196, max_size:classes 37, max_size:data 1269, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.243 sec/step, elapsed 0:05:58, exp. remaining 0:36:51, complete 13.93%
att-weights epoch 481, step 197, max_size:classes 35, max_size:data 912, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.454 sec/step, elapsed 0:05:59, exp. remaining 0:36:48, complete 14.00%
att-weights epoch 481, step 198, max_size:classes 35, max_size:data 1369, mem_usage:GPU:0 0.8GB, num_seqs 2, 0.865 sec/step, elapsed 0:06:00, exp. remaining 0:36:34, complete 14.11%
att-weights epoch 481, step 199, max_size:classes 33, max_size:data 972, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.213 sec/step, elapsed 0:06:01, exp. remaining 0:36:16, complete 14.25%
att-weights epoch 481, step 200, max_size:classes 36, max_size:data 1295, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.112 sec/step, elapsed 0:06:02, exp. remaining 0:36:04, complete 14.35%
att-weights epoch 481, step 201, max_size:classes 37, max_size:data 1160, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.118 sec/step, elapsed 0:06:03, exp. remaining 0:35:52, complete 14.46%
att-weights epoch 481, step 202, max_size:classes 35, max_size:data 1027, mem_usage:GPU:0 0.8GB, num_seqs 3, 7.828 sec/step, elapsed 0:06:11, exp. remaining 0:36:26, complete 14.53%
att-weights epoch 481, step 203, max_size:classes 36, max_size:data 1230, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.106 sec/step, elapsed 0:06:12, exp. remaining 0:36:15, complete 14.63%
att-weights epoch 481, step 204, max_size:classes 36, max_size:data 1042, mem_usage:GPU:0 0.8GB, num_seqs 3, 2.821 sec/step, elapsed 0:06:15, exp. remaining 0:36:19, complete 14.70%
att-weights epoch 481, step 205, max_size:classes 37, max_size:data 968, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.506 sec/step, elapsed 0:06:17, exp. remaining 0:36:09, complete 14.80%
att-weights epoch 481, step 206, max_size:classes 33, max_size:data 1366, mem_usage:GPU:0 0.8GB, num_seqs 2, 0.915 sec/step, elapsed 0:06:17, exp. remaining 0:35:57, complete 14.91%
att-weights epoch 481, step 207, max_size:classes 34, max_size:data 1093, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.879 sec/step, elapsed 0:06:19, exp. remaining 0:35:44, complete 15.05%
att-weights epoch 481, step 208, max_size:classes 36, max_size:data 994, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.385 sec/step, elapsed 0:06:21, exp. remaining 0:35:34, complete 15.15%
att-weights epoch 481, step 209, max_size:classes 40, max_size:data 1010, mem_usage:GPU:0 0.8GB, num_seqs 3, 0.958 sec/step, elapsed 0:06:22, exp. remaining 0:35:22, complete 15.26%
att-weights epoch 481, step 210, max_size:classes 38, max_size:data 1276, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.195 sec/step, elapsed 0:06:23, exp. remaining 0:35:12, complete 15.36%
att-weights epoch 481, step 211, max_size:classes 35, max_size:data 1355, mem_usage:GPU:0 0.8GB, num_seqs 2, 0.901 sec/step, elapsed 0:06:24, exp. remaining 0:35:00, complete 15.47%
att-weights epoch 481, step 212, max_size:classes 33, max_size:data 1069, mem_usage:GPU:0 0.8GB, num_seqs 3, 0.986 sec/step, elapsed 0:06:25, exp. remaining 0:34:48, complete 15.57%
att-weights epoch 481, step 213, max_size:classes 40, max_size:data 1448, mem_usage:GPU:0 0.8GB, num_seqs 2, 0.901 sec/step, elapsed 0:06:26, exp. remaining 0:34:37, complete 15.68%
att-weights epoch 481, step 214, max_size:classes 38, max_size:data 1323, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.734 sec/step, elapsed 0:06:27, exp. remaining 0:34:30, complete 15.78%
att-weights epoch 481, step 215, max_size:classes 34, max_size:data 1177, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.276 sec/step, elapsed 0:06:29, exp. remaining 0:34:20, complete 15.89%
att-weights epoch 481, step 216, max_size:classes 37, max_size:data 982, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.384 sec/step, elapsed 0:06:30, exp. remaining 0:34:11, complete 15.99%
att-weights epoch 481, step 217, max_size:classes 34, max_size:data 1000, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.005 sec/step, elapsed 0:06:31, exp. remaining 0:34:01, complete 16.10%
att-weights epoch 481, step 218, max_size:classes 34, max_size:data 1072, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.168 sec/step, elapsed 0:06:32, exp. remaining 0:33:51, complete 16.20%
att-weights epoch 481, step 219, max_size:classes 36, max_size:data 1197, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.246 sec/step, elapsed 0:06:34, exp. remaining 0:33:42, complete 16.31%
att-weights epoch 481, step 220, max_size:classes 33, max_size:data 1013, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.098 sec/step, elapsed 0:06:35, exp. remaining 0:33:27, complete 16.45%
att-weights epoch 481, step 221, max_size:classes 36, max_size:data 930, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.086 sec/step, elapsed 0:06:36, exp. remaining 0:33:17, complete 16.55%
att-weights epoch 481, step 222, max_size:classes 35, max_size:data 1123, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.075 sec/step, elapsed 0:06:37, exp. remaining 0:33:03, complete 16.69%
att-weights epoch 481, step 223, max_size:classes 33, max_size:data 1042, mem_usage:GPU:0 0.8GB, num_seqs 3, 0.922 sec/step, elapsed 0:06:38, exp. remaining 0:32:52, complete 16.79%
att-weights epoch 481, step 224, max_size:classes 33, max_size:data 1186, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.058 sec/step, elapsed 0:06:39, exp. remaining 0:32:43, complete 16.90%
att-weights epoch 481, step 225, max_size:classes 35, max_size:data 1170, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.011 sec/step, elapsed 0:06:40, exp. remaining 0:32:33, complete 17.00%
att-weights epoch 481, step 226, max_size:classes 37, max_size:data 1025, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.016 sec/step, elapsed 0:06:41, exp. remaining 0:32:24, complete 17.11%
att-weights epoch 481, step 227, max_size:classes 32, max_size:data 1013, mem_usage:GPU:0 0.8GB, num_seqs 3, 0.998 sec/step, elapsed 0:06:42, exp. remaining 0:32:14, complete 17.21%
att-weights epoch 481, step 228, max_size:classes 35, max_size:data 1177, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.019 sec/step, elapsed 0:06:43, exp. remaining 0:32:00, complete 17.35%
att-weights epoch 481, step 229, max_size:classes 35, max_size:data 955, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.685 sec/step, elapsed 0:06:44, exp. remaining 0:31:54, complete 17.46%
att-weights epoch 481, step 230, max_size:classes 31, max_size:data 1265, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.071 sec/step, elapsed 0:06:46, exp. remaining 0:31:45, complete 17.56%
att-weights epoch 481, step 231, max_size:classes 37, max_size:data 981, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.243 sec/step, elapsed 0:06:47, exp. remaining 0:31:33, complete 17.70%
att-weights epoch 481, step 232, max_size:classes 35, max_size:data 1093, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.062 sec/step, elapsed 0:06:48, exp. remaining 0:31:24, complete 17.81%
att-weights epoch 481, step 233, max_size:classes 33, max_size:data 1208, mem_usage:GPU:0 0.8GB, num_seqs 3, 0.960 sec/step, elapsed 0:06:49, exp. remaining 0:31:20, complete 17.88%
att-weights epoch 481, step 234, max_size:classes 37, max_size:data 1028, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.226 sec/step, elapsed 0:06:50, exp. remaining 0:31:08, complete 18.02%
att-weights epoch 481, step 235, max_size:classes 35, max_size:data 1012, mem_usage:GPU:0 0.8GB, num_seqs 3, 0.878 sec/step, elapsed 0:06:51, exp. remaining 0:30:58, complete 18.12%
att-weights epoch 481, step 236, max_size:classes 31, max_size:data 1283, mem_usage:GPU:0 0.8GB, num_seqs 3, 0.984 sec/step, elapsed 0:06:52, exp. remaining 0:30:45, complete 18.26%
att-weights epoch 481, step 237, max_size:classes 34, max_size:data 963, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.309 sec/step, elapsed 0:06:53, exp. remaining 0:30:38, complete 18.37%
att-weights epoch 481, step 238, max_size:classes 33, max_size:data 1087, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.046 sec/step, elapsed 0:06:54, exp. remaining 0:30:30, complete 18.47%
att-weights epoch 481, step 239, max_size:classes 38, max_size:data 1071, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.000 sec/step, elapsed 0:06:55, exp. remaining 0:30:22, complete 18.58%
att-weights epoch 481, step 240, max_size:classes 34, max_size:data 931, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.248 sec/step, elapsed 0:06:57, exp. remaining 0:30:15, complete 18.68%
att-weights epoch 481, step 241, max_size:classes 29, max_size:data 1052, mem_usage:GPU:0 0.8GB, num_seqs 3, 0.923 sec/step, elapsed 0:06:57, exp. remaining 0:30:02, complete 18.82%
att-weights epoch 481, step 242, max_size:classes 34, max_size:data 1371, mem_usage:GPU:0 0.8GB, num_seqs 2, 0.757 sec/step, elapsed 0:06:58, exp. remaining 0:29:53, complete 18.92%
att-weights epoch 481, step 243, max_size:classes 36, max_size:data 982, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.289 sec/step, elapsed 0:06:59, exp. remaining 0:29:47, complete 19.03%
att-weights epoch 481, step 244, max_size:classes 34, max_size:data 1332, mem_usage:GPU:0 0.8GB, num_seqs 3, 0.967 sec/step, elapsed 0:07:00, exp. remaining 0:29:39, complete 19.13%
att-weights epoch 481, step 245, max_size:classes 31, max_size:data 969, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.073 sec/step, elapsed 0:07:02, exp. remaining 0:29:31, complete 19.24%
att-weights epoch 481, step 246, max_size:classes 33, max_size:data 1179, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.114 sec/step, elapsed 0:07:03, exp. remaining 0:29:24, complete 19.34%
att-weights epoch 481, step 247, max_size:classes 32, max_size:data 1085, mem_usage:GPU:0 0.8GB, num_seqs 3, 0.914 sec/step, elapsed 0:07:04, exp. remaining 0:29:16, complete 19.45%
att-weights epoch 481, step 248, max_size:classes 28, max_size:data 1044, mem_usage:GPU:0 0.8GB, num_seqs 3, 0.853 sec/step, elapsed 0:07:04, exp. remaining 0:29:04, complete 19.59%
att-weights epoch 481, step 249, max_size:classes 34, max_size:data 1326, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.041 sec/step, elapsed 0:07:05, exp. remaining 0:28:53, complete 19.73%
att-weights epoch 481, step 250, max_size:classes 34, max_size:data 879, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.065 sec/step, elapsed 0:07:07, exp. remaining 0:28:46, complete 19.83%
att-weights epoch 481, step 251, max_size:classes 32, max_size:data 1032, mem_usage:GPU:0 0.8GB, num_seqs 3, 0.896 sec/step, elapsed 0:07:07, exp. remaining 0:28:38, complete 19.94%
att-weights epoch 481, step 252, max_size:classes 33, max_size:data 1070, mem_usage:GPU:0 0.8GB, num_seqs 3, 0.965 sec/step, elapsed 0:07:08, exp. remaining 0:28:31, complete 20.04%
att-weights epoch 481, step 253, max_size:classes 30, max_size:data 958, mem_usage:GPU:0 0.8GB, num_seqs 3, 0.945 sec/step, elapsed 0:07:09, exp. remaining 0:28:23, complete 20.15%
att-weights epoch 481, step 254, max_size:classes 29, max_size:data 1214, mem_usage:GPU:0 0.8GB, num_seqs 3, 0.902 sec/step, elapsed 0:07:10, exp. remaining 0:28:16, complete 20.25%
att-weights epoch 481, step 255, max_size:classes 33, max_size:data 1108, mem_usage:GPU:0 0.8GB, num_seqs 3, 0.903 sec/step, elapsed 0:07:11, exp. remaining 0:28:08, complete 20.36%
att-weights epoch 481, step 256, max_size:classes 30, max_size:data 1229, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.078 sec/step, elapsed 0:07:12, exp. remaining 0:28:02, complete 20.46%
att-weights epoch 481, step 257, max_size:classes 34, max_size:data 884, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.306 sec/step, elapsed 0:07:14, exp. remaining 0:27:52, complete 20.60%
att-weights epoch 481, step 258, max_size:classes 34, max_size:data 907, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.430 sec/step, elapsed 0:07:15, exp. remaining 0:27:44, complete 20.74%
att-weights epoch 481, step 259, max_size:classes 32, max_size:data 1249, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.287 sec/step, elapsed 0:07:16, exp. remaining 0:27:38, complete 20.84%
att-weights epoch 481, step 260, max_size:classes 31, max_size:data 1104, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.214 sec/step, elapsed 0:07:17, exp. remaining 0:27:32, complete 20.95%
att-weights epoch 481, step 261, max_size:classes 33, max_size:data 1026, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.529 sec/step, elapsed 0:07:19, exp. remaining 0:27:27, complete 21.05%
att-weights epoch 481, step 262, max_size:classes 39, max_size:data 1026, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.191 sec/step, elapsed 0:07:20, exp. remaining 0:27:18, complete 21.19%
att-weights epoch 481, step 263, max_size:classes 34, max_size:data 1010, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.037 sec/step, elapsed 0:07:21, exp. remaining 0:27:12, complete 21.30%
att-weights epoch 481, step 264, max_size:classes 30, max_size:data 1215, mem_usage:GPU:0 0.8GB, num_seqs 3, 0.992 sec/step, elapsed 0:07:22, exp. remaining 0:27:05, complete 21.40%
att-weights epoch 481, step 265, max_size:classes 30, max_size:data 1177, mem_usage:GPU:0 0.8GB, num_seqs 3, 0.937 sec/step, elapsed 0:07:23, exp. remaining 0:26:55, complete 21.54%
att-weights epoch 481, step 266, max_size:classes 36, max_size:data 848, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.430 sec/step, elapsed 0:07:25, exp. remaining 0:26:47, complete 21.68%
att-weights epoch 481, step 267, max_size:classes 30, max_size:data 906, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.398 sec/step, elapsed 0:07:26, exp. remaining 0:26:42, complete 21.79%
att-weights epoch 481, step 268, max_size:classes 31, max_size:data 1044, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.976 sec/step, elapsed 0:07:28, exp. remaining 0:26:39, complete 21.89%
att-weights epoch 481, step 269, max_size:classes 27, max_size:data 1224, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.538 sec/step, elapsed 0:07:29, exp. remaining 0:26:29, complete 22.07%
att-weights epoch 481, step 270, max_size:classes 29, max_size:data 1077, mem_usage:GPU:0 0.8GB, num_seqs 3, 4.740 sec/step, elapsed 0:07:34, exp. remaining 0:26:32, complete 22.21%
att-weights epoch 481, step 271, max_size:classes 31, max_size:data 868, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.285 sec/step, elapsed 0:07:36, exp. remaining 0:26:27, complete 22.31%
att-weights epoch 481, step 272, max_size:classes 32, max_size:data 1176, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.105 sec/step, elapsed 0:07:37, exp. remaining 0:26:18, complete 22.45%
att-weights epoch 481, step 273, max_size:classes 34, max_size:data 1218, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.076 sec/step, elapsed 0:07:38, exp. remaining 0:26:13, complete 22.56%
att-weights epoch 481, step 274, max_size:classes 32, max_size:data 845, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.494 sec/step, elapsed 0:07:39, exp. remaining 0:26:05, complete 22.70%
att-weights epoch 481, step 275, max_size:classes 32, max_size:data 865, mem_usage:GPU:0 0.8GB, num_seqs 4, 4.035 sec/step, elapsed 0:07:43, exp. remaining 0:26:10, complete 22.80%
att-weights epoch 481, step 276, max_size:classes 31, max_size:data 1180, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.465 sec/step, elapsed 0:07:45, exp. remaining 0:26:05, complete 22.91%
att-weights epoch 481, step 277, max_size:classes 31, max_size:data 1005, mem_usage:GPU:0 0.8GB, num_seqs 3, 0.858 sec/step, elapsed 0:07:46, exp. remaining 0:25:56, complete 23.04%
att-weights epoch 481, step 278, max_size:classes 29, max_size:data 793, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.247 sec/step, elapsed 0:07:47, exp. remaining 0:25:48, complete 23.18%
att-weights epoch 481, step 279, max_size:classes 29, max_size:data 994, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.141 sec/step, elapsed 0:07:48, exp. remaining 0:25:42, complete 23.29%
att-weights epoch 481, step 280, max_size:classes 29, max_size:data 1115, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.022 sec/step, elapsed 0:07:49, exp. remaining 0:25:37, complete 23.39%
att-weights epoch 481, step 281, max_size:classes 29, max_size:data 949, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.108 sec/step, elapsed 0:07:50, exp. remaining 0:25:31, complete 23.50%
att-weights epoch 481, step 282, max_size:classes 32, max_size:data 1248, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.028 sec/step, elapsed 0:07:51, exp. remaining 0:25:23, complete 23.64%
att-weights epoch 481, step 283, max_size:classes 28, max_size:data 858, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.036 sec/step, elapsed 0:07:52, exp. remaining 0:25:15, complete 23.78%
att-weights epoch 481, step 284, max_size:classes 31, max_size:data 1176, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.040 sec/step, elapsed 0:07:53, exp. remaining 0:25:06, complete 23.92%
att-weights epoch 481, step 285, max_size:classes 30, max_size:data 1074, mem_usage:GPU:0 0.8GB, num_seqs 3, 0.783 sec/step, elapsed 0:07:54, exp. remaining 0:25:00, complete 24.02%
att-weights epoch 481, step 286, max_size:classes 28, max_size:data 829, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.157 sec/step, elapsed 0:07:55, exp. remaining 0:24:55, complete 24.13%
att-weights epoch 481, step 287, max_size:classes 29, max_size:data 788, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.338 sec/step, elapsed 0:07:56, exp. remaining 0:24:51, complete 24.23%
att-weights epoch 481, step 288, max_size:classes 34, max_size:data 993, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.288 sec/step, elapsed 0:07:58, exp. remaining 0:24:44, complete 24.37%
att-weights epoch 481, step 289, max_size:classes 30, max_size:data 1088, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.004 sec/step, elapsed 0:07:59, exp. remaining 0:24:35, complete 24.51%
att-weights epoch 481, step 290, max_size:classes 28, max_size:data 1039, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.020 sec/step, elapsed 0:08:00, exp. remaining 0:24:28, complete 24.65%
att-weights epoch 481, step 291, max_size:classes 29, max_size:data 923, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.104 sec/step, elapsed 0:08:01, exp. remaining 0:24:20, complete 24.79%
att-weights epoch 481, step 292, max_size:classes 30, max_size:data 881, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.410 sec/step, elapsed 0:08:02, exp. remaining 0:24:13, complete 24.93%
att-weights epoch 481, step 293, max_size:classes 32, max_size:data 824, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.475 sec/step, elapsed 0:08:04, exp. remaining 0:24:07, complete 25.07%
att-weights epoch 481, step 294, max_size:classes 30, max_size:data 1012, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.044 sec/step, elapsed 0:08:05, exp. remaining 0:23:59, complete 25.21%
att-weights epoch 481, step 295, max_size:classes 29, max_size:data 956, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.058 sec/step, elapsed 0:08:06, exp. remaining 0:23:54, complete 25.31%
att-weights epoch 481, step 296, max_size:classes 29, max_size:data 1138, mem_usage:GPU:0 0.8GB, num_seqs 3, 10.772 sec/step, elapsed 0:08:17, exp. remaining 0:24:18, complete 25.42%
att-weights epoch 481, step 297, max_size:classes 28, max_size:data 908, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.134 sec/step, elapsed 0:08:18, exp. remaining 0:24:13, complete 25.52%
att-weights epoch 481, step 298, max_size:classes 29, max_size:data 836, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.145 sec/step, elapsed 0:08:19, exp. remaining 0:24:03, complete 25.70%
att-weights epoch 481, step 299, max_size:classes 29, max_size:data 886, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.125 sec/step, elapsed 0:08:20, exp. remaining 0:23:59, complete 25.80%
att-weights epoch 481, step 300, max_size:classes 32, max_size:data 991, mem_usage:GPU:0 0.8GB, num_seqs 4, 5.985 sec/step, elapsed 0:08:26, exp. remaining 0:24:05, complete 25.94%
att-weights epoch 481, step 301, max_size:classes 28, max_size:data 916, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.005 sec/step, elapsed 0:08:27, exp. remaining 0:23:58, complete 26.08%
att-weights epoch 481, step 302, max_size:classes 32, max_size:data 835, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.262 sec/step, elapsed 0:08:28, exp. remaining 0:23:51, complete 26.22%
att-weights epoch 481, step 303, max_size:classes 26, max_size:data 885, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.376 sec/step, elapsed 0:08:30, exp. remaining 0:23:45, complete 26.36%
att-weights epoch 481, step 304, max_size:classes 27, max_size:data 918, mem_usage:GPU:0 0.8GB, num_seqs 3, 0.889 sec/step, elapsed 0:08:31, exp. remaining 0:23:37, complete 26.50%
att-weights epoch 481, step 305, max_size:classes 30, max_size:data 1127, mem_usage:GPU:0 0.8GB, num_seqs 3, 0.936 sec/step, elapsed 0:08:31, exp. remaining 0:23:27, complete 26.68%
att-weights epoch 481, step 306, max_size:classes 33, max_size:data 1053, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.108 sec/step, elapsed 0:08:33, exp. remaining 0:23:20, complete 26.82%
att-weights epoch 481, step 307, max_size:classes 28, max_size:data 783, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.476 sec/step, elapsed 0:08:34, exp. remaining 0:23:14, complete 26.96%
att-weights epoch 481, step 308, max_size:classes 30, max_size:data 1033, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.028 sec/step, elapsed 0:08:35, exp. remaining 0:23:07, complete 27.09%
att-weights epoch 481, step 309, max_size:classes 32, max_size:data 909, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.302 sec/step, elapsed 0:08:36, exp. remaining 0:23:01, complete 27.23%
att-weights epoch 481, step 310, max_size:classes 26, max_size:data 812, mem_usage:GPU:0 0.8GB, num_seqs 4, 7.043 sec/step, elapsed 0:08:43, exp. remaining 0:23:10, complete 27.37%
att-weights epoch 481, step 311, max_size:classes 30, max_size:data 960, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.205 sec/step, elapsed 0:08:45, exp. remaining 0:23:03, complete 27.51%
att-weights epoch 481, step 312, max_size:classes 31, max_size:data 895, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.362 sec/step, elapsed 0:08:46, exp. remaining 0:22:57, complete 27.65%
att-weights epoch 481, step 313, max_size:classes 29, max_size:data 973, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.323 sec/step, elapsed 0:08:47, exp. remaining 0:22:48, complete 27.83%
att-weights epoch 481, step 314, max_size:classes 29, max_size:data 709, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.344 sec/step, elapsed 0:08:49, exp. remaining 0:22:42, complete 27.97%
att-weights epoch 481, step 315, max_size:classes 27, max_size:data 898, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.062 sec/step, elapsed 0:08:50, exp. remaining 0:22:36, complete 28.11%
att-weights epoch 481, step 316, max_size:classes 25, max_size:data 801, mem_usage:GPU:0 0.8GB, num_seqs 4, 0.984 sec/step, elapsed 0:08:51, exp. remaining 0:22:29, complete 28.25%
att-weights epoch 481, step 317, max_size:classes 26, max_size:data 957, mem_usage:GPU:0 0.8GB, num_seqs 4, 3.432 sec/step, elapsed 0:08:54, exp. remaining 0:22:31, complete 28.35%
att-weights epoch 481, step 318, max_size:classes 28, max_size:data 834, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.183 sec/step, elapsed 0:08:55, exp. remaining 0:22:27, complete 28.46%
att-weights epoch 481, step 319, max_size:classes 26, max_size:data 909, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.044 sec/step, elapsed 0:08:56, exp. remaining 0:22:20, complete 28.60%
att-weights epoch 481, step 320, max_size:classes 27, max_size:data 913, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.064 sec/step, elapsed 0:08:57, exp. remaining 0:22:14, complete 28.74%
att-weights epoch 481, step 321, max_size:classes 29, max_size:data 942, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.331 sec/step, elapsed 0:08:59, exp. remaining 0:22:08, complete 28.88%
att-weights epoch 481, step 322, max_size:classes 28, max_size:data 764, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.545 sec/step, elapsed 0:09:00, exp. remaining 0:22:05, complete 28.98%
att-weights epoch 481, step 323, max_size:classes 28, max_size:data 952, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.227 sec/step, elapsed 0:09:02, exp. remaining 0:21:59, complete 29.12%
att-weights epoch 481, step 324, max_size:classes 26, max_size:data 942, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.138 sec/step, elapsed 0:09:03, exp. remaining 0:21:55, complete 29.22%
att-weights epoch 481, step 325, max_size:classes 27, max_size:data 845, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.244 sec/step, elapsed 0:09:04, exp. remaining 0:21:49, complete 29.36%
att-weights epoch 481, step 326, max_size:classes 26, max_size:data 636, mem_usage:GPU:0 0.8GB, num_seqs 3, 14.848 sec/step, elapsed 0:09:19, exp. remaining 0:22:16, complete 29.50%
att-weights epoch 481, step 327, max_size:classes 26, max_size:data 1029, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.166 sec/step, elapsed 0:09:20, exp. remaining 0:22:07, complete 29.68%
att-weights epoch 481, step 328, max_size:classes 26, max_size:data 917, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.092 sec/step, elapsed 0:09:21, exp. remaining 0:21:59, complete 29.85%
att-weights epoch 481, step 329, max_size:classes 30, max_size:data 780, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.203 sec/step, elapsed 0:09:22, exp. remaining 0:21:51, complete 30.03%
att-weights epoch 481, step 330, max_size:classes 27, max_size:data 864, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.486 sec/step, elapsed 0:09:24, exp. remaining 0:21:46, complete 30.17%
att-weights epoch 481, step 331, max_size:classes 27, max_size:data 1036, mem_usage:GPU:0 0.8GB, num_seqs 3, 2.759 sec/step, elapsed 0:09:27, exp. remaining 0:21:41, complete 30.34%
att-weights epoch 481, step 332, max_size:classes 26, max_size:data 716, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.699 sec/step, elapsed 0:09:28, exp. remaining 0:21:34, complete 30.52%
att-weights epoch 481, step 333, max_size:classes 24, max_size:data 1075, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.088 sec/step, elapsed 0:09:29, exp. remaining 0:21:28, complete 30.66%
att-weights epoch 481, step 334, max_size:classes 30, max_size:data 876, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.133 sec/step, elapsed 0:09:30, exp. remaining 0:21:22, complete 30.80%
att-weights epoch 481, step 335, max_size:classes 25, max_size:data 887, mem_usage:GPU:0 0.8GB, num_seqs 4, 12.083 sec/step, elapsed 0:09:43, exp. remaining 0:21:41, complete 30.94%
att-weights epoch 481, step 336, max_size:classes 25, max_size:data 741, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.404 sec/step, elapsed 0:09:44, exp. remaining 0:21:36, complete 31.08%
att-weights epoch 481, step 337, max_size:classes 29, max_size:data 725, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.386 sec/step, elapsed 0:09:45, exp. remaining 0:21:30, complete 31.22%
att-weights epoch 481, step 338, max_size:classes 29, max_size:data 785, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.322 sec/step, elapsed 0:09:47, exp. remaining 0:21:25, complete 31.35%
att-weights epoch 481, step 339, max_size:classes 28, max_size:data 956, mem_usage:GPU:0 0.8GB, num_seqs 4, 14.933 sec/step, elapsed 0:10:02, exp. remaining 0:21:47, complete 31.53%
att-weights epoch 481, step 340, max_size:classes 26, max_size:data 758, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.670 sec/step, elapsed 0:10:03, exp. remaining 0:21:42, complete 31.67%
att-weights epoch 481, step 341, max_size:classes 27, max_size:data 785, mem_usage:GPU:0 0.8GB, num_seqs 5, 2.868 sec/step, elapsed 0:10:06, exp. remaining 0:21:40, complete 31.81%
att-weights epoch 481, step 342, max_size:classes 28, max_size:data 896, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.308 sec/step, elapsed 0:10:07, exp. remaining 0:21:34, complete 31.95%
att-weights epoch 481, step 343, max_size:classes 25, max_size:data 995, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.433 sec/step, elapsed 0:10:09, exp. remaining 0:21:29, complete 32.09%
att-weights epoch 481, step 344, max_size:classes 26, max_size:data 922, mem_usage:GPU:0 0.8GB, num_seqs 4, 2.992 sec/step, elapsed 0:10:12, exp. remaining 0:21:25, complete 32.26%
att-weights epoch 481, step 345, max_size:classes 24, max_size:data 885, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.798 sec/step, elapsed 0:10:14, exp. remaining 0:21:19, complete 32.44%
att-weights epoch 481, step 346, max_size:classes 29, max_size:data 850, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.367 sec/step, elapsed 0:10:15, exp. remaining 0:21:13, complete 32.58%
att-weights epoch 481, step 347, max_size:classes 26, max_size:data 896, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.466 sec/step, elapsed 0:10:16, exp. remaining 0:21:08, complete 32.72%
att-weights epoch 481, step 348, max_size:classes 28, max_size:data 724, mem_usage:GPU:0 0.8GB, num_seqs 5, 5.187 sec/step, elapsed 0:10:22, exp. remaining 0:21:11, complete 32.86%
att-weights epoch 481, step 349, max_size:classes 27, max_size:data 858, mem_usage:GPU:0 0.8GB, num_seqs 4, 3.691 sec/step, elapsed 0:10:25, exp. remaining 0:21:10, complete 33.00%
att-weights epoch 481, step 350, max_size:classes 25, max_size:data 922, mem_usage:GPU:0 0.8GB, num_seqs 4, 15.653 sec/step, elapsed 0:10:41, exp. remaining 0:21:34, complete 33.14%
att-weights epoch 481, step 351, max_size:classes 26, max_size:data 687, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.617 sec/step, elapsed 0:10:43, exp. remaining 0:21:27, complete 33.31%
att-weights epoch 481, step 352, max_size:classes 31, max_size:data 863, mem_usage:GPU:0 0.8GB, num_seqs 4, 7.868 sec/step, elapsed 0:10:51, exp. remaining 0:21:33, complete 33.48%
att-weights epoch 481, step 353, max_size:classes 26, max_size:data 669, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.346 sec/step, elapsed 0:10:52, exp. remaining 0:21:27, complete 33.62%
att-weights epoch 481, step 354, max_size:classes 26, max_size:data 688, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.282 sec/step, elapsed 0:10:53, exp. remaining 0:21:20, complete 33.80%
att-weights epoch 481, step 355, max_size:classes 32, max_size:data 887, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.129 sec/step, elapsed 0:10:54, exp. remaining 0:21:12, complete 33.97%
att-weights epoch 481, step 356, max_size:classes 28, max_size:data 941, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.074 sec/step, elapsed 0:10:55, exp. remaining 0:21:06, complete 34.11%
att-weights epoch 481, step 357, max_size:classes 25, max_size:data 811, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.116 sec/step, elapsed 0:10:56, exp. remaining 0:20:59, complete 34.29%
att-weights epoch 481, step 358, max_size:classes 27, max_size:data 928, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.198 sec/step, elapsed 0:10:58, exp. remaining 0:20:53, complete 34.43%
att-weights epoch 481, step 359, max_size:classes 25, max_size:data 987, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.188 sec/step, elapsed 0:10:59, exp. remaining 0:20:46, complete 34.60%
att-weights epoch 481, step 360, max_size:classes 24, max_size:data 697, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.246 sec/step, elapsed 0:11:00, exp. remaining 0:20:37, complete 34.81%
att-weights epoch 481, step 361, max_size:classes 24, max_size:data 795, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.391 sec/step, elapsed 0:11:01, exp. remaining 0:20:30, complete 34.99%
att-weights epoch 481, step 362, max_size:classes 23, max_size:data 912, mem_usage:GPU:0 0.8GB, num_seqs 4, 0.997 sec/step, elapsed 0:11:02, exp. remaining 0:20:24, complete 35.13%
att-weights epoch 481, step 363, max_size:classes 24, max_size:data 721, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.982 sec/step, elapsed 0:11:04, exp. remaining 0:20:18, complete 35.30%
att-weights epoch 481, step 364, max_size:classes 26, max_size:data 769, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.596 sec/step, elapsed 0:11:06, exp. remaining 0:20:14, complete 35.44%
att-weights epoch 481, step 365, max_size:classes 26, max_size:data 824, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.286 sec/step, elapsed 0:11:07, exp. remaining 0:20:09, complete 35.58%
att-weights epoch 481, step 366, max_size:classes 24, max_size:data 738, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.335 sec/step, elapsed 0:11:09, exp. remaining 0:20:02, complete 35.75%
att-weights epoch 481, step 367, max_size:classes 24, max_size:data 827, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.131 sec/step, elapsed 0:11:10, exp. remaining 0:19:57, complete 35.89%
att-weights epoch 481, step 368, max_size:classes 26, max_size:data 777, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.188 sec/step, elapsed 0:11:11, exp. remaining 0:19:50, complete 36.07%
att-weights epoch 481, step 369, max_size:classes 29, max_size:data 665, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.791 sec/step, elapsed 0:11:13, exp. remaining 0:19:44, complete 36.24%
att-weights epoch 481, step 370, max_size:classes 26, max_size:data 708, mem_usage:GPU:0 0.8GB, num_seqs 5, 3.313 sec/step, elapsed 0:11:16, exp. remaining 0:19:43, complete 36.38%
att-weights epoch 481, step 371, max_size:classes 27, max_size:data 878, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.154 sec/step, elapsed 0:11:17, exp. remaining 0:19:36, complete 36.56%
att-weights epoch 481, step 372, max_size:classes 25, max_size:data 660, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.412 sec/step, elapsed 0:11:19, exp. remaining 0:19:29, complete 36.73%
att-weights epoch 481, step 373, max_size:classes 26, max_size:data 771, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.253 sec/step, elapsed 0:11:20, exp. remaining 0:19:23, complete 36.91%
att-weights epoch 481, step 374, max_size:classes 24, max_size:data 915, mem_usage:GPU:0 0.8GB, num_seqs 4, 4.229 sec/step, elapsed 0:11:24, exp. remaining 0:19:21, complete 37.08%
att-weights epoch 481, step 375, max_size:classes 24, max_size:data 601, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.253 sec/step, elapsed 0:11:25, exp. remaining 0:19:16, complete 37.22%
att-weights epoch 481, step 376, max_size:classes 23, max_size:data 842, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.201 sec/step, elapsed 0:11:27, exp. remaining 0:19:08, complete 37.43%
att-weights epoch 481, step 377, max_size:classes 24, max_size:data 745, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.401 sec/step, elapsed 0:11:28, exp. remaining 0:19:04, complete 37.57%
att-weights epoch 481, step 378, max_size:classes 24, max_size:data 798, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.452 sec/step, elapsed 0:11:29, exp. remaining 0:18:58, complete 37.74%
att-weights epoch 481, step 379, max_size:classes 25, max_size:data 911, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.197 sec/step, elapsed 0:11:31, exp. remaining 0:18:51, complete 37.92%
att-weights epoch 481, step 380, max_size:classes 23, max_size:data 760, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.234 sec/step, elapsed 0:11:32, exp. remaining 0:18:43, complete 38.13%
att-weights epoch 481, step 381, max_size:classes 24, max_size:data 762, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.352 sec/step, elapsed 0:11:33, exp. remaining 0:18:37, complete 38.30%
att-weights epoch 481, step 382, max_size:classes 24, max_size:data 711, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.321 sec/step, elapsed 0:11:35, exp. remaining 0:18:31, complete 38.48%
att-weights epoch 481, step 383, max_size:classes 28, max_size:data 635, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.320 sec/step, elapsed 0:11:36, exp. remaining 0:18:25, complete 38.65%
att-weights epoch 481, step 384, max_size:classes 26, max_size:data 849, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.126 sec/step, elapsed 0:11:37, exp. remaining 0:18:18, complete 38.83%
att-weights epoch 481, step 385, max_size:classes 24, max_size:data 628, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.606 sec/step, elapsed 0:11:39, exp. remaining 0:18:15, complete 38.97%
att-weights epoch 481, step 386, max_size:classes 23, max_size:data 819, mem_usage:GPU:0 0.8GB, num_seqs 4, 0.971 sec/step, elapsed 0:11:40, exp. remaining 0:18:08, complete 39.14%
att-weights epoch 481, step 387, max_size:classes 24, max_size:data 701, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.242 sec/step, elapsed 0:11:41, exp. remaining 0:18:00, complete 39.35%
att-weights epoch 481, step 388, max_size:classes 23, max_size:data 681, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.268 sec/step, elapsed 0:11:42, exp. remaining 0:17:56, complete 39.49%
att-weights epoch 481, step 389, max_size:classes 24, max_size:data 577, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.402 sec/step, elapsed 0:11:44, exp. remaining 0:17:49, complete 39.70%
att-weights epoch 481, step 390, max_size:classes 24, max_size:data 765, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.328 sec/step, elapsed 0:11:45, exp. remaining 0:17:43, complete 39.87%
att-weights epoch 481, step 391, max_size:classes 24, max_size:data 746, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.014 sec/step, elapsed 0:11:46, exp. remaining 0:17:37, complete 40.05%
att-weights epoch 481, step 392, max_size:classes 25, max_size:data 697, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.253 sec/step, elapsed 0:11:47, exp. remaining 0:17:31, complete 40.22%
att-weights epoch 481, step 393, max_size:classes 26, max_size:data 676, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.230 sec/step, elapsed 0:11:48, exp. remaining 0:17:27, complete 40.36%
att-weights epoch 481, step 394, max_size:classes 29, max_size:data 844, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.102 sec/step, elapsed 0:11:49, exp. remaining 0:17:19, complete 40.57%
att-weights epoch 481, step 395, max_size:classes 21, max_size:data 754, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.409 sec/step, elapsed 0:11:51, exp. remaining 0:17:14, complete 40.75%
att-weights epoch 481, step 396, max_size:classes 24, max_size:data 626, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.545 sec/step, elapsed 0:11:52, exp. remaining 0:17:07, complete 40.96%
att-weights epoch 481, step 397, max_size:classes 22, max_size:data 827, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.117 sec/step, elapsed 0:11:54, exp. remaining 0:17:01, complete 41.13%
att-weights epoch 481, step 398, max_size:classes 24, max_size:data 627, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.443 sec/step, elapsed 0:11:55, exp. remaining 0:16:56, complete 41.31%
att-weights epoch 481, step 399, max_size:classes 27, max_size:data 786, mem_usage:GPU:0 0.8GB, num_seqs 5, 2.053 sec/step, elapsed 0:11:57, exp. remaining 0:16:53, complete 41.45%
att-weights epoch 481, step 400, max_size:classes 23, max_size:data 796, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.376 sec/step, elapsed 0:11:58, exp. remaining 0:16:48, complete 41.62%
att-weights epoch 481, step 401, max_size:classes 22, max_size:data 763, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.626 sec/step, elapsed 0:12:00, exp. remaining 0:16:42, complete 41.83%
att-weights epoch 481, step 402, max_size:classes 25, max_size:data 861, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.122 sec/step, elapsed 0:12:01, exp. remaining 0:16:36, complete 42.00%
att-weights epoch 481, step 403, max_size:classes 22, max_size:data 664, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.435 sec/step, elapsed 0:12:03, exp. remaining 0:16:31, complete 42.18%
att-weights epoch 481, step 404, max_size:classes 24, max_size:data 754, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.244 sec/step, elapsed 0:12:04, exp. remaining 0:16:27, complete 42.32%
att-weights epoch 481, step 405, max_size:classes 26, max_size:data 627, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.368 sec/step, elapsed 0:12:05, exp. remaining 0:16:20, complete 42.53%
att-weights epoch 481, step 406, max_size:classes 23, max_size:data 706, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.288 sec/step, elapsed 0:12:06, exp. remaining 0:16:15, complete 42.70%
att-weights epoch 481, step 407, max_size:classes 22, max_size:data 750, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.598 sec/step, elapsed 0:12:08, exp. remaining 0:16:10, complete 42.88%
att-weights epoch 481, step 408, max_size:classes 25, max_size:data 809, mem_usage:GPU:0 0.8GB, num_seqs 4, 0.897 sec/step, elapsed 0:12:09, exp. remaining 0:16:04, complete 43.05%
att-weights epoch 481, step 409, max_size:classes 20, max_size:data 705, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.097 sec/step, elapsed 0:12:10, exp. remaining 0:15:58, complete 43.26%
att-weights epoch 481, step 410, max_size:classes 21, max_size:data 657, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.367 sec/step, elapsed 0:12:11, exp. remaining 0:15:54, complete 43.40%
att-weights epoch 481, step 411, max_size:classes 24, max_size:data 759, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.222 sec/step, elapsed 0:12:13, exp. remaining 0:15:49, complete 43.58%
att-weights epoch 481, step 412, max_size:classes 22, max_size:data 767, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.373 sec/step, elapsed 0:12:14, exp. remaining 0:15:45, complete 43.72%
att-weights epoch 481, step 413, max_size:classes 22, max_size:data 864, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.007 sec/step, elapsed 0:12:15, exp. remaining 0:15:39, complete 43.92%
att-weights epoch 481, step 414, max_size:classes 23, max_size:data 655, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.407 sec/step, elapsed 0:12:16, exp. remaining 0:15:32, complete 44.13%
att-weights epoch 481, step 415, max_size:classes 21, max_size:data 776, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.168 sec/step, elapsed 0:12:18, exp. remaining 0:15:27, complete 44.31%
att-weights epoch 481, step 416, max_size:classes 23, max_size:data 773, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.232 sec/step, elapsed 0:12:19, exp. remaining 0:15:22, complete 44.48%
att-weights epoch 481, step 417, max_size:classes 22, max_size:data 723, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.301 sec/step, elapsed 0:12:20, exp. remaining 0:15:16, complete 44.69%
att-weights epoch 481, step 418, max_size:classes 25, max_size:data 639, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.353 sec/step, elapsed 0:12:22, exp. remaining 0:15:11, complete 44.87%
att-weights epoch 481, step 419, max_size:classes 23, max_size:data 946, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.066 sec/step, elapsed 0:12:23, exp. remaining 0:15:06, complete 45.04%
att-weights epoch 481, step 420, max_size:classes 26, max_size:data 768, mem_usage:GPU:0 0.8GB, num_seqs 5, 2.380 sec/step, elapsed 0:12:25, exp. remaining 0:15:01, complete 45.25%
att-weights epoch 481, step 421, max_size:classes 21, max_size:data 815, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.121 sec/step, elapsed 0:12:26, exp. remaining 0:14:56, complete 45.43%
att-weights epoch 481, step 422, max_size:classes 23, max_size:data 623, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.368 sec/step, elapsed 0:12:27, exp. remaining 0:14:52, complete 45.60%
att-weights epoch 481, step 423, max_size:classes 22, max_size:data 636, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.455 sec/step, elapsed 0:12:29, exp. remaining 0:14:47, complete 45.78%
att-weights epoch 481, step 424, max_size:classes 20, max_size:data 630, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.404 sec/step, elapsed 0:12:30, exp. remaining 0:14:43, complete 45.95%
att-weights epoch 481, step 425, max_size:classes 21, max_size:data 722, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.333 sec/step, elapsed 0:12:32, exp. remaining 0:14:38, complete 46.12%
att-weights epoch 481, step 426, max_size:classes 21, max_size:data 647, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.539 sec/step, elapsed 0:12:33, exp. remaining 0:14:32, complete 46.33%
att-weights epoch 481, step 427, max_size:classes 22, max_size:data 707, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.273 sec/step, elapsed 0:12:34, exp. remaining 0:14:28, complete 46.51%
att-weights epoch 481, step 428, max_size:classes 21, max_size:data 735, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.191 sec/step, elapsed 0:12:36, exp. remaining 0:14:21, complete 46.75%
att-weights epoch 481, step 429, max_size:classes 24, max_size:data 598, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.332 sec/step, elapsed 0:12:37, exp. remaining 0:14:16, complete 46.93%
att-weights epoch 481, step 430, max_size:classes 20, max_size:data 770, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.355 sec/step, elapsed 0:12:38, exp. remaining 0:14:12, complete 47.10%
att-weights epoch 481, step 431, max_size:classes 23, max_size:data 696, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.191 sec/step, elapsed 0:12:40, exp. remaining 0:14:09, complete 47.21%
att-weights epoch 481, step 432, max_size:classes 19, max_size:data 708, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.313 sec/step, elapsed 0:12:41, exp. remaining 0:14:04, complete 47.42%
att-weights epoch 481, step 433, max_size:classes 20, max_size:data 652, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.445 sec/step, elapsed 0:12:42, exp. remaining 0:14:00, complete 47.59%
att-weights epoch 481, step 434, max_size:classes 21, max_size:data 700, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.454 sec/step, elapsed 0:12:44, exp. remaining 0:13:55, complete 47.77%
att-weights epoch 481, step 435, max_size:classes 22, max_size:data 621, mem_usage:GPU:0 0.8GB, num_seqs 6, 4.496 sec/step, elapsed 0:12:48, exp. remaining 0:13:52, complete 48.01%
att-weights epoch 481, step 436, max_size:classes 21, max_size:data 776, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.198 sec/step, elapsed 0:12:49, exp. remaining 0:13:45, complete 48.25%
att-weights epoch 481, step 437, max_size:classes 20, max_size:data 547, mem_usage:GPU:0 0.8GB, num_seqs 7, 2.475 sec/step, elapsed 0:12:52, exp. remaining 0:13:41, complete 48.46%
att-weights epoch 481, step 438, max_size:classes 21, max_size:data 777, mem_usage:GPU:0 0.8GB, num_seqs 5, 3.107 sec/step, elapsed 0:12:55, exp. remaining 0:13:40, complete 48.60%
att-weights epoch 481, step 439, max_size:classes 21, max_size:data 653, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.377 sec/step, elapsed 0:12:56, exp. remaining 0:13:34, complete 48.81%
att-weights epoch 481, step 440, max_size:classes 22, max_size:data 1093, mem_usage:GPU:0 0.8GB, num_seqs 3, 0.981 sec/step, elapsed 0:12:57, exp. remaining 0:13:28, complete 49.02%
att-weights epoch 481, step 441, max_size:classes 20, max_size:data 522, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.705 sec/step, elapsed 0:12:59, exp. remaining 0:13:25, complete 49.20%
att-weights epoch 481, step 442, max_size:classes 20, max_size:data 606, mem_usage:GPU:0 0.8GB, num_seqs 5, 2.104 sec/step, elapsed 0:13:01, exp. remaining 0:13:20, complete 49.41%
att-weights epoch 481, step 443, max_size:classes 22, max_size:data 751, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.319 sec/step, elapsed 0:13:03, exp. remaining 0:13:15, complete 49.62%
att-weights epoch 481, step 444, max_size:classes 21, max_size:data 566, mem_usage:GPU:0 0.8GB, num_seqs 7, 3.111 sec/step, elapsed 0:13:06, exp. remaining 0:13:11, complete 49.83%
att-weights epoch 481, step 445, max_size:classes 18, max_size:data 570, mem_usage:GPU:0 0.8GB, num_seqs 7, 1.632 sec/step, elapsed 0:13:07, exp. remaining 0:13:06, complete 50.03%
att-weights epoch 481, step 446, max_size:classes 21, max_size:data 604, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.356 sec/step, elapsed 0:13:09, exp. remaining 0:13:02, complete 50.21%
att-weights epoch 481, step 447, max_size:classes 20, max_size:data 814, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.224 sec/step, elapsed 0:13:10, exp. remaining 0:12:56, complete 50.45%
att-weights epoch 481, step 448, max_size:classes 19, max_size:data 592, mem_usage:GPU:0 0.8GB, num_seqs 6, 5.061 sec/step, elapsed 0:13:15, exp. remaining 0:12:54, complete 50.66%
att-weights epoch 481, step 449, max_size:classes 20, max_size:data 620, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.744 sec/step, elapsed 0:13:17, exp. remaining 0:12:49, complete 50.87%
att-weights epoch 481, step 450, max_size:classes 20, max_size:data 697, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.380 sec/step, elapsed 0:13:18, exp. remaining 0:12:44, complete 51.08%
att-weights epoch 481, step 451, max_size:classes 21, max_size:data 641, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.543 sec/step, elapsed 0:13:20, exp. remaining 0:12:39, complete 51.29%
att-weights epoch 481, step 452, max_size:classes 19, max_size:data 591, mem_usage:GPU:0 0.8GB, num_seqs 6, 19.357 sec/step, elapsed 0:13:39, exp. remaining 0:12:52, complete 51.47%
att-weights epoch 481, step 453, max_size:classes 22, max_size:data 654, mem_usage:GPU:0 0.8GB, num_seqs 6, 2.377 sec/step, elapsed 0:13:41, exp. remaining 0:12:49, complete 51.64%
att-weights epoch 481, step 454, max_size:classes 20, max_size:data 599, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.386 sec/step, elapsed 0:13:43, exp. remaining 0:12:45, complete 51.82%
att-weights epoch 481, step 455, max_size:classes 23, max_size:data 677, mem_usage:GPU:0 0.8GB, num_seqs 5, 5.830 sec/step, elapsed 0:13:49, exp. remaining 0:12:44, complete 52.03%
att-weights epoch 481, step 456, max_size:classes 20, max_size:data 558, mem_usage:GPU:0 0.8GB, num_seqs 7, 1.938 sec/step, elapsed 0:13:50, exp. remaining 0:12:39, complete 52.23%
att-weights epoch 481, step 457, max_size:classes 17, max_size:data 612, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.390 sec/step, elapsed 0:13:52, exp. remaining 0:12:35, complete 52.41%
att-weights epoch 481, step 458, max_size:classes 20, max_size:data 643, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.442 sec/step, elapsed 0:13:53, exp. remaining 0:12:30, complete 52.62%
att-weights epoch 481, step 459, max_size:classes 21, max_size:data 585, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.325 sec/step, elapsed 0:13:55, exp. remaining 0:12:25, complete 52.83%
att-weights epoch 481, step 460, max_size:classes 20, max_size:data 606, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.385 sec/step, elapsed 0:13:56, exp. remaining 0:12:20, complete 53.04%
att-weights epoch 481, step 461, max_size:classes 20, max_size:data 681, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.134 sec/step, elapsed 0:13:57, exp. remaining 0:12:14, complete 53.28%
att-weights epoch 481, step 462, max_size:classes 24, max_size:data 620, mem_usage:GPU:0 0.8GB, num_seqs 5, 0.764 sec/step, elapsed 0:13:58, exp. remaining 0:12:08, complete 53.49%
att-weights epoch 481, step 463, max_size:classes 19, max_size:data 683, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.256 sec/step, elapsed 0:13:59, exp. remaining 0:12:03, complete 53.70%
att-weights epoch 481, step 464, max_size:classes 20, max_size:data 530, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.328 sec/step, elapsed 0:14:01, exp. remaining 0:12:02, complete 53.81%
att-weights epoch 481, step 465, max_size:classes 20, max_size:data 596, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.381 sec/step, elapsed 0:14:02, exp. remaining 0:11:56, complete 54.05%
att-weights epoch 481, step 466, max_size:classes 19, max_size:data 670, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.115 sec/step, elapsed 0:14:03, exp. remaining 0:11:50, complete 54.29%
att-weights epoch 481, step 467, max_size:classes 20, max_size:data 592, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.171 sec/step, elapsed 0:14:04, exp. remaining 0:11:45, complete 54.50%
att-weights epoch 481, step 468, max_size:classes 19, max_size:data 645, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.339 sec/step, elapsed 0:14:06, exp. remaining 0:11:40, complete 54.71%
att-weights epoch 481, step 469, max_size:classes 22, max_size:data 573, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.282 sec/step, elapsed 0:14:07, exp. remaining 0:11:36, complete 54.89%
att-weights epoch 481, step 470, max_size:classes 18, max_size:data 497, mem_usage:GPU:0 0.8GB, num_seqs 7, 1.346 sec/step, elapsed 0:14:08, exp. remaining 0:11:31, complete 55.10%
att-weights epoch 481, step 471, max_size:classes 21, max_size:data 622, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.375 sec/step, elapsed 0:14:10, exp. remaining 0:11:25, complete 55.34%
att-weights epoch 481, step 472, max_size:classes 19, max_size:data 625, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.460 sec/step, elapsed 0:14:11, exp. remaining 0:11:21, complete 55.55%
att-weights epoch 481, step 473, max_size:classes 20, max_size:data 1012, mem_usage:GPU:0 0.8GB, num_seqs 3, 1.022 sec/step, elapsed 0:14:12, exp. remaining 0:11:17, complete 55.73%
att-weights epoch 481, step 474, max_size:classes 18, max_size:data 554, mem_usage:GPU:0 0.8GB, num_seqs 7, 1.575 sec/step, elapsed 0:14:14, exp. remaining 0:11:11, complete 55.97%
att-weights epoch 481, step 475, max_size:classes 20, max_size:data 548, mem_usage:GPU:0 0.8GB, num_seqs 7, 1.796 sec/step, elapsed 0:14:15, exp. remaining 0:11:07, complete 56.18%
att-weights epoch 481, step 476, max_size:classes 23, max_size:data 616, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.371 sec/step, elapsed 0:14:17, exp. remaining 0:11:03, complete 56.35%
att-weights epoch 481, step 477, max_size:classes 18, max_size:data 606, mem_usage:GPU:0 0.8GB, num_seqs 6, 10.597 sec/step, elapsed 0:14:27, exp. remaining 0:11:07, complete 56.53%
att-weights epoch 481, step 478, max_size:classes 20, max_size:data 686, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.257 sec/step, elapsed 0:14:29, exp. remaining 0:11:02, complete 56.74%
att-weights epoch 481, step 479, max_size:classes 18, max_size:data 619, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.138 sec/step, elapsed 0:14:30, exp. remaining 0:10:57, complete 56.95%
att-weights epoch 481, step 480, max_size:classes 16, max_size:data 548, mem_usage:GPU:0 0.8GB, num_seqs 7, 1.351 sec/step, elapsed 0:14:31, exp. remaining 0:10:52, complete 57.19%
att-weights epoch 481, step 481, max_size:classes 18, max_size:data 578, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.335 sec/step, elapsed 0:14:32, exp. remaining 0:10:48, complete 57.37%
att-weights epoch 481, step 482, max_size:classes 18, max_size:data 693, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.131 sec/step, elapsed 0:14:34, exp. remaining 0:10:44, complete 57.54%
att-weights epoch 481, step 483, max_size:classes 18, max_size:data 523, mem_usage:GPU:0 0.8GB, num_seqs 7, 1.449 sec/step, elapsed 0:14:35, exp. remaining 0:10:39, complete 57.79%
att-weights epoch 481, step 484, max_size:classes 21, max_size:data 616, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.249 sec/step, elapsed 0:14:36, exp. remaining 0:10:34, complete 58.03%
att-weights epoch 481, step 485, max_size:classes 19, max_size:data 654, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.000 sec/step, elapsed 0:14:37, exp. remaining 0:10:29, complete 58.24%
att-weights epoch 481, step 486, max_size:classes 18, max_size:data 713, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.202 sec/step, elapsed 0:14:38, exp. remaining 0:10:23, complete 58.52%
att-weights epoch 481, step 487, max_size:classes 17, max_size:data 513, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.202 sec/step, elapsed 0:14:40, exp. remaining 0:10:20, complete 58.66%
att-weights epoch 481, step 488, max_size:classes 19, max_size:data 604, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.412 sec/step, elapsed 0:14:41, exp. remaining 0:10:15, complete 58.87%
att-weights epoch 481, step 489, max_size:classes 19, max_size:data 567, mem_usage:GPU:0 0.8GB, num_seqs 7, 1.453 sec/step, elapsed 0:14:43, exp. remaining 0:10:11, complete 59.08%
att-weights epoch 481, step 490, max_size:classes 17, max_size:data 711, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.042 sec/step, elapsed 0:14:44, exp. remaining 0:10:05, complete 59.36%
att-weights epoch 481, step 491, max_size:classes 17, max_size:data 696, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.115 sec/step, elapsed 0:14:45, exp. remaining 0:10:00, complete 59.57%
att-weights epoch 481, step 492, max_size:classes 18, max_size:data 525, mem_usage:GPU:0 0.8GB, num_seqs 7, 1.606 sec/step, elapsed 0:14:46, exp. remaining 0:09:56, complete 59.78%
att-weights epoch 481, step 493, max_size:classes 16, max_size:data 517, mem_usage:GPU:0 0.8GB, num_seqs 7, 6.334 sec/step, elapsed 0:14:53, exp. remaining 0:09:55, complete 59.99%
att-weights epoch 481, step 494, max_size:classes 17, max_size:data 656, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.209 sec/step, elapsed 0:14:54, exp. remaining 0:09:50, complete 60.23%
att-weights epoch 481, step 495, max_size:classes 17, max_size:data 500, mem_usage:GPU:0 0.8GB, num_seqs 8, 1.788 sec/step, elapsed 0:14:56, exp. remaining 0:09:47, complete 60.41%
att-weights epoch 481, step 496, max_size:classes 17, max_size:data 805, mem_usage:GPU:0 0.8GB, num_seqs 4, 0.912 sec/step, elapsed 0:14:57, exp. remaining 0:09:43, complete 60.58%
att-weights epoch 481, step 497, max_size:classes 15, max_size:data 607, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.072 sec/step, elapsed 0:14:58, exp. remaining 0:09:39, complete 60.79%
att-weights epoch 481, step 498, max_size:classes 21, max_size:data 612, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.416 sec/step, elapsed 0:14:59, exp. remaining 0:09:36, complete 60.93%
att-weights epoch 481, step 499, max_size:classes 18, max_size:data 498, mem_usage:GPU:0 0.8GB, num_seqs 8, 1.539 sec/step, elapsed 0:15:01, exp. remaining 0:09:34, complete 61.07%
att-weights epoch 481, step 500, max_size:classes 16, max_size:data 524, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.162 sec/step, elapsed 0:15:02, exp. remaining 0:09:30, complete 61.28%
att-weights epoch 481, step 501, max_size:classes 16, max_size:data 662, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.198 sec/step, elapsed 0:15:03, exp. remaining 0:09:25, complete 61.49%
att-weights epoch 481, step 502, max_size:classes 17, max_size:data 609, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.379 sec/step, elapsed 0:15:04, exp. remaining 0:09:20, complete 61.77%
att-weights epoch 481, step 503, max_size:classes 18, max_size:data 546, mem_usage:GPU:0 0.8GB, num_seqs 7, 1.804 sec/step, elapsed 0:15:06, exp. remaining 0:09:16, complete 61.98%
att-weights epoch 481, step 504, max_size:classes 17, max_size:data 612, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.123 sec/step, elapsed 0:15:07, exp. remaining 0:09:11, complete 62.19%
att-weights epoch 481, step 505, max_size:classes 16, max_size:data 710, mem_usage:GPU:0 0.8GB, num_seqs 5, 7.036 sec/step, elapsed 0:15:14, exp. remaining 0:09:10, complete 62.43%
att-weights epoch 481, step 506, max_size:classes 15, max_size:data 595, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.194 sec/step, elapsed 0:15:15, exp. remaining 0:09:04, complete 62.71%
att-weights epoch 481, step 507, max_size:classes 17, max_size:data 533, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.068 sec/step, elapsed 0:15:17, exp. remaining 0:09:00, complete 62.92%
att-weights epoch 481, step 508, max_size:classes 17, max_size:data 821, mem_usage:GPU:0 0.8GB, num_seqs 4, 1.329 sec/step, elapsed 0:15:18, exp. remaining 0:08:56, complete 63.13%
att-weights epoch 481, step 509, max_size:classes 19, max_size:data 601, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.317 sec/step, elapsed 0:15:19, exp. remaining 0:08:49, complete 63.44%
att-weights epoch 481, step 510, max_size:classes 18, max_size:data 584, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.382 sec/step, elapsed 0:15:21, exp. remaining 0:08:45, complete 63.69%
att-weights epoch 481, step 511, max_size:classes 18, max_size:data 478, mem_usage:GPU:0 0.8GB, num_seqs 8, 1.528 sec/step, elapsed 0:15:22, exp. remaining 0:08:40, complete 63.93%
att-weights epoch 481, step 512, max_size:classes 16, max_size:data 556, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.294 sec/step, elapsed 0:15:23, exp. remaining 0:08:34, complete 64.21%
att-weights epoch 481, step 513, max_size:classes 17, max_size:data 616, mem_usage:GPU:0 0.8GB, num_seqs 6, 23.419 sec/step, elapsed 0:15:47, exp. remaining 0:08:43, complete 64.42%
att-weights epoch 481, step 514, max_size:classes 16, max_size:data 561, mem_usage:GPU:0 0.8GB, num_seqs 7, 1.628 sec/step, elapsed 0:15:48, exp. remaining 0:08:38, complete 64.66%
att-weights epoch 481, step 515, max_size:classes 19, max_size:data 487, mem_usage:GPU:0 0.8GB, num_seqs 8, 1.492 sec/step, elapsed 0:15:50, exp. remaining 0:08:33, complete 64.94%
att-weights epoch 481, step 516, max_size:classes 16, max_size:data 636, mem_usage:GPU:0 0.8GB, num_seqs 6, 22.932 sec/step, elapsed 0:16:13, exp. remaining 0:08:39, complete 65.19%
att-weights epoch 481, step 517, max_size:classes 17, max_size:data 659, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.660 sec/step, elapsed 0:16:15, exp. remaining 0:08:35, complete 65.40%
att-weights epoch 481, step 518, max_size:classes 16, max_size:data 418, mem_usage:GPU:0 0.8GB, num_seqs 9, 9.052 sec/step, elapsed 0:16:24, exp. remaining 0:08:34, complete 65.68%
att-weights epoch 481, step 519, max_size:classes 17, max_size:data 546, mem_usage:GPU:0 0.8GB, num_seqs 7, 1.765 sec/step, elapsed 0:16:25, exp. remaining 0:08:30, complete 65.89%
att-weights epoch 481, step 520, max_size:classes 16, max_size:data 564, mem_usage:GPU:0 0.8GB, num_seqs 7, 1.663 sec/step, elapsed 0:16:27, exp. remaining 0:08:25, complete 66.13%
att-weights epoch 481, step 521, max_size:classes 16, max_size:data 492, mem_usage:GPU:0 0.8GB, num_seqs 8, 4.351 sec/step, elapsed 0:16:31, exp. remaining 0:08:24, complete 66.31%
att-weights epoch 481, step 522, max_size:classes 15, max_size:data 606, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.331 sec/step, elapsed 0:16:33, exp. remaining 0:08:19, complete 66.52%
att-weights epoch 481, step 523, max_size:classes 14, max_size:data 555, mem_usage:GPU:0 0.8GB, num_seqs 7, 1.386 sec/step, elapsed 0:16:34, exp. remaining 0:08:15, complete 66.76%
att-weights epoch 481, step 524, max_size:classes 17, max_size:data 455, mem_usage:GPU:0 0.8GB, num_seqs 8, 1.537 sec/step, elapsed 0:16:36, exp. remaining 0:08:09, complete 67.04%
att-weights epoch 481, step 525, max_size:classes 22, max_size:data 520, mem_usage:GPU:0 0.8GB, num_seqs 7, 1.355 sec/step, elapsed 0:16:37, exp. remaining 0:08:05, complete 67.28%
att-weights epoch 481, step 526, max_size:classes 20, max_size:data 587, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.246 sec/step, elapsed 0:16:38, exp. remaining 0:08:00, complete 67.53%
att-weights epoch 481, step 527, max_size:classes 16, max_size:data 464, mem_usage:GPU:0 0.8GB, num_seqs 8, 1.563 sec/step, elapsed 0:16:40, exp. remaining 0:07:55, complete 67.77%
att-weights epoch 481, step 528, max_size:classes 16, max_size:data 572, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.220 sec/step, elapsed 0:16:41, exp. remaining 0:07:51, complete 67.98%
att-weights epoch 481, step 529, max_size:classes 15, max_size:data 502, mem_usage:GPU:0 0.8GB, num_seqs 7, 1.522 sec/step, elapsed 0:16:43, exp. remaining 0:07:45, complete 68.30%
att-weights epoch 481, step 530, max_size:classes 21, max_size:data 679, mem_usage:GPU:0 0.8GB, num_seqs 5, 1.174 sec/step, elapsed 0:16:44, exp. remaining 0:07:40, complete 68.58%
att-weights epoch 481, step 531, max_size:classes 17, max_size:data 648, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.429 sec/step, elapsed 0:16:45, exp. remaining 0:07:35, complete 68.82%
att-weights epoch 481, step 532, max_size:classes 15, max_size:data 505, mem_usage:GPU:0 0.8GB, num_seqs 7, 7.593 sec/step, elapsed 0:16:53, exp. remaining 0:07:33, complete 69.06%
att-weights epoch 481, step 533, max_size:classes 14, max_size:data 462, mem_usage:GPU:0 0.8GB, num_seqs 8, 1.603 sec/step, elapsed 0:16:54, exp. remaining 0:07:29, complete 69.31%
att-weights epoch 481, step 534, max_size:classes 22, max_size:data 539, mem_usage:GPU:0 0.8GB, num_seqs 7, 1.487 sec/step, elapsed 0:16:56, exp. remaining 0:07:23, complete 69.62%
att-weights epoch 481, step 535, max_size:classes 16, max_size:data 515, mem_usage:GPU:0 0.8GB, num_seqs 7, 1.266 sec/step, elapsed 0:16:57, exp. remaining 0:07:18, complete 69.90%
att-weights epoch 481, step 536, max_size:classes 16, max_size:data 557, mem_usage:GPU:0 0.8GB, num_seqs 7, 6.016 sec/step, elapsed 0:17:03, exp. remaining 0:07:14, complete 70.18%
att-weights epoch 481, step 537, max_size:classes 15, max_size:data 611, mem_usage:GPU:0 0.8GB, num_seqs 6, 1.159 sec/step, elapsed 0:17:04, exp. remaining 0:07:09, complete 70.46%
att-weights epoch 481, step 538, max_size:classes 16, max_size:data 423, mem_usage:GPU:0 0.8GB, num_seqs 9, 1.930 sec/step, elapsed 0:17:06, exp. remaining 0:07:04, complete 70.74%
att-weights epoch 481, step 539, max_size:classes 14, max_size:data 471, mem_usage:GPU:0 0.8GB, num_seqs 8, 1.507 sec/step, elapsed 0:17:08, exp. remaining 0:06:59, complete 71.02%
att-weights epoch 481, step 540, max_size:classes 14, max_size:data 520, mem_usage:GPU:0 0.8GB, num_seqs 7, 1.541 sec/step, elapsed 0:17:09, exp. remaining 0:06:54, complete 71.30%
att-weights epoch 481, step 541, max_size:classes 15, max_size:data 507, mem_usage:GPU:0 0.8GB, num_seqs 7, 1.605 sec/step, elapsed 0:17:11, exp. remaining 0:06:49, complete 71.58%
att-weights epoch 481, step 542, max_size:classes 15, max_size:data 528, mem_usage:GPU:0 0.8GB, num_seqs 7, 1.344 sec/step, elapsed 0:17:12, exp. remaining 0:06:43, complete 71.89%
att-weights epoch 481, step 543, max_size:classes 14, max_size:data 402, mem_usage:GPU:0 0.8GB, num_seqs 9, 1.407 sec/step, elapsed 0:17:14, exp. remaining 0:06:37, complete 72.24%
att-weights epoch 481, step 544, max_size:classes 15, max_size:data 487, mem_usage:GPU:0 0.8GB, num_seqs 8, 1.718 sec/step, elapsed 0:17:15, exp. remaining 0:06:32, complete 72.52%
att-weights epoch 481, step 545, max_size:classes 16, max_size:data 466, mem_usage:GPU:0 0.8GB, num_seqs 8, 1.376 sec/step, elapsed 0:17:17, exp. remaining 0:06:28, complete 72.77%
att-weights epoch 481, step 546, max_size:classes 15, max_size:data 485, mem_usage:GPU:0 0.8GB, num_seqs 8, 1.539 sec/step, elapsed 0:17:18, exp. remaining 0:06:23, complete 73.04%
att-weights epoch 481, step 547, max_size:classes 14, max_size:data 490, mem_usage:GPU:0 0.8GB, num_seqs 8, 1.384 sec/step, elapsed 0:17:20, exp. remaining 0:06:17, complete 73.36%
att-weights epoch 481, step 548, max_size:classes 14, max_size:data 488, mem_usage:GPU:0 0.8GB, num_seqs 8, 1.419 sec/step, elapsed 0:17:21, exp. remaining 0:06:12, complete 73.67%
att-weights epoch 481, step 549, max_size:classes 14, max_size:data 500, mem_usage:GPU:0 0.8GB, num_seqs 8, 1.472 sec/step, elapsed 0:17:23, exp. remaining 0:06:08, complete 73.92%
att-weights epoch 481, step 550, max_size:classes 13, max_size:data 493, mem_usage:GPU:0 0.8GB, num_seqs 8, 1.562 sec/step, elapsed 0:17:24, exp. remaining 0:06:03, complete 74.16%
att-weights epoch 481, step 551, max_size:classes 16, max_size:data 415, mem_usage:GPU:0 0.8GB, num_seqs 9, 1.971 sec/step, elapsed 0:17:26, exp. remaining 0:05:57, complete 74.55%
att-weights epoch 481, step 552, max_size:classes 13, max_size:data 398, mem_usage:GPU:0 0.8GB, num_seqs 10, 1.821 sec/step, elapsed 0:17:28, exp. remaining 0:05:52, complete 74.83%
att-weights epoch 481, step 553, max_size:classes 14, max_size:data 475, mem_usage:GPU:0 0.8GB, num_seqs 8, 1.361 sec/step, elapsed 0:17:29, exp. remaining 0:05:47, complete 75.14%
att-weights epoch 481, step 554, max_size:classes 15, max_size:data 562, mem_usage:GPU:0 0.8GB, num_seqs 7, 1.351 sec/step, elapsed 0:17:31, exp. remaining 0:05:41, complete 75.45%
att-weights epoch 481, step 555, max_size:classes 14, max_size:data 463, mem_usage:GPU:0 0.8GB, num_seqs 8, 1.688 sec/step, elapsed 0:17:32, exp. remaining 0:05:37, complete 75.70%
att-weights epoch 481, step 556, max_size:classes 14, max_size:data 396, mem_usage:GPU:0 0.8GB, num_seqs 9, 3.043 sec/step, elapsed 0:17:35, exp. remaining 0:05:33, complete 75.98%
att-weights epoch 481, step 557, max_size:classes 13, max_size:data 440, mem_usage:GPU:0 0.8GB, num_seqs 9, 1.841 sec/step, elapsed 0:17:37, exp. remaining 0:05:28, complete 76.29%
att-weights epoch 481, step 558, max_size:classes 15, max_size:data 460, mem_usage:GPU:0 0.8GB, num_seqs 7, 7.804 sec/step, elapsed 0:17:45, exp. remaining 0:05:25, complete 76.61%
att-weights epoch 481, step 559, max_size:classes 16, max_size:data 534, mem_usage:GPU:0 0.8GB, num_seqs 7, 1.536 sec/step, elapsed 0:17:47, exp. remaining 0:05:20, complete 76.92%
att-weights epoch 481, step 560, max_size:classes 14, max_size:data 362, mem_usage:GPU:0 0.8GB, num_seqs 11, 2.236 sec/step, elapsed 0:17:49, exp. remaining 0:05:14, complete 77.27%
att-weights epoch 481, step 561, max_size:classes 13, max_size:data 445, mem_usage:GPU:0 0.8GB, num_seqs 8, 1.447 sec/step, elapsed 0:17:50, exp. remaining 0:05:09, complete 77.58%
att-weights epoch 481, step 562, max_size:classes 13, max_size:data 433, mem_usage:GPU:0 0.8GB, num_seqs 9, 1.616 sec/step, elapsed 0:17:52, exp. remaining 0:05:05, complete 77.83%
att-weights epoch 481, step 563, max_size:classes 13, max_size:data 444, mem_usage:GPU:0 0.8GB, num_seqs 9, 1.589 sec/step, elapsed 0:17:53, exp. remaining 0:05:01, complete 78.07%
att-weights epoch 481, step 564, max_size:classes 13, max_size:data 539, mem_usage:GPU:0 0.8GB, num_seqs 7, 1.163 sec/step, elapsed 0:17:55, exp. remaining 0:04:58, complete 78.28%
att-weights epoch 481, step 565, max_size:classes 17, max_size:data 496, mem_usage:GPU:0 0.8GB, num_seqs 8, 1.889 sec/step, elapsed 0:17:56, exp. remaining 0:04:53, complete 78.60%
att-weights epoch 481, step 566, max_size:classes 13, max_size:data 415, mem_usage:GPU:0 0.8GB, num_seqs 9, 1.533 sec/step, elapsed 0:17:58, exp. remaining 0:04:48, complete 78.88%
att-weights epoch 481, step 567, max_size:classes 13, max_size:data 426, mem_usage:GPU:0 0.8GB, num_seqs 9, 2.167 sec/step, elapsed 0:18:00, exp. remaining 0:04:44, complete 79.16%
att-weights epoch 481, step 568, max_size:classes 13, max_size:data 407, mem_usage:GPU:0 0.8GB, num_seqs 9, 15.160 sec/step, elapsed 0:18:15, exp. remaining 0:04:43, complete 79.47%
att-weights epoch 481, step 569, max_size:classes 14, max_size:data 368, mem_usage:GPU:0 0.8GB, num_seqs 10, 2.294 sec/step, elapsed 0:18:18, exp. remaining 0:04:37, complete 79.82%
att-weights epoch 481, step 570, max_size:classes 14, max_size:data 391, mem_usage:GPU:0 0.8GB, num_seqs 9, 2.319 sec/step, elapsed 0:18:20, exp. remaining 0:04:32, complete 80.13%
att-weights epoch 481, step 571, max_size:classes 13, max_size:data 539, mem_usage:GPU:0 0.8GB, num_seqs 7, 20.892 sec/step, elapsed 0:18:41, exp. remaining 0:04:31, complete 80.48%
att-weights epoch 481, step 572, max_size:classes 12, max_size:data 472, mem_usage:GPU:0 0.8GB, num_seqs 7, 1.351 sec/step, elapsed 0:18:42, exp. remaining 0:04:27, complete 80.76%
att-weights epoch 481, step 573, max_size:classes 13, max_size:data 600, mem_usage:GPU:0 0.8GB, num_seqs 6, 12.723 sec/step, elapsed 0:18:55, exp. remaining 0:04:26, complete 81.01%
att-weights epoch 481, step 574, max_size:classes 13, max_size:data 426, mem_usage:GPU:0 0.8GB, num_seqs 9, 1.748 sec/step, elapsed 0:18:57, exp. remaining 0:04:21, complete 81.28%
att-weights epoch 481, step 575, max_size:classes 11, max_size:data 469, mem_usage:GPU:0 0.8GB, num_seqs 8, 1.683 sec/step, elapsed 0:18:58, exp. remaining 0:04:17, complete 81.56%
att-weights epoch 481, step 576, max_size:classes 13, max_size:data 491, mem_usage:GPU:0 0.8GB, num_seqs 8, 1.579 sec/step, elapsed 0:19:00, exp. remaining 0:04:11, complete 81.91%
att-weights epoch 481, step 577, max_size:classes 13, max_size:data 424, mem_usage:GPU:0 0.8GB, num_seqs 9, 5.031 sec/step, elapsed 0:19:05, exp. remaining 0:04:05, complete 82.37%
att-weights epoch 481, step 578, max_size:classes 17, max_size:data 398, mem_usage:GPU:0 0.8GB, num_seqs 10, 1.481 sec/step, elapsed 0:19:06, exp. remaining 0:03:59, complete 82.72%
att-weights epoch 481, step 579, max_size:classes 14, max_size:data 415, mem_usage:GPU:0 0.8GB, num_seqs 9, 1.283 sec/step, elapsed 0:19:08, exp. remaining 0:03:54, complete 83.03%
att-weights epoch 481, step 580, max_size:classes 14, max_size:data 375, mem_usage:GPU:0 0.8GB, num_seqs 10, 1.690 sec/step, elapsed 0:19:09, exp. remaining 0:03:49, complete 83.34%
att-weights epoch 481, step 581, max_size:classes 11, max_size:data 403, mem_usage:GPU:0 0.8GB, num_seqs 8, 1.235 sec/step, elapsed 0:19:11, exp. remaining 0:03:44, complete 83.66%
att-weights epoch 481, step 582, max_size:classes 14, max_size:data 505, mem_usage:GPU:0 0.8GB, num_seqs 7, 1.552 sec/step, elapsed 0:19:12, exp. remaining 0:03:38, complete 84.04%
att-weights epoch 481, step 583, max_size:classes 12, max_size:data 462, mem_usage:GPU:0 0.8GB, num_seqs 8, 1.507 sec/step, elapsed 0:19:14, exp. remaining 0:03:33, complete 84.39%
att-weights epoch 481, step 584, max_size:classes 13, max_size:data 496, mem_usage:GPU:0 0.8GB, num_seqs 8, 1.397 sec/step, elapsed 0:19:15, exp. remaining 0:03:27, complete 84.78%
att-weights epoch 481, step 585, max_size:classes 13, max_size:data 384, mem_usage:GPU:0 0.8GB, num_seqs 10, 1.457 sec/step, elapsed 0:19:17, exp. remaining 0:03:22, complete 85.09%
att-weights epoch 481, step 586, max_size:classes 11, max_size:data 291, mem_usage:GPU:0 0.8GB, num_seqs 13, 1.896 sec/step, elapsed 0:19:18, exp. remaining 0:03:17, complete 85.44%
att-weights epoch 481, step 587, max_size:classes 12, max_size:data 392, mem_usage:GPU:0 0.8GB, num_seqs 10, 1.548 sec/step, elapsed 0:19:20, exp. remaining 0:03:11, complete 85.82%
att-weights epoch 481, step 588, max_size:classes 12, max_size:data 435, mem_usage:GPU:0 0.8GB, num_seqs 9, 1.289 sec/step, elapsed 0:19:21, exp. remaining 0:03:06, complete 86.17%
att-weights epoch 481, step 589, max_size:classes 12, max_size:data 423, mem_usage:GPU:0 0.8GB, num_seqs 9, 2.585 sec/step, elapsed 0:19:24, exp. remaining 0:03:01, complete 86.49%
att-weights epoch 481, step 590, max_size:classes 13, max_size:data 405, mem_usage:GPU:0 0.8GB, num_seqs 9, 1.474 sec/step, elapsed 0:19:25, exp. remaining 0:02:57, complete 86.77%
att-weights epoch 481, step 591, max_size:classes 12, max_size:data 348, mem_usage:GPU:0 0.8GB, num_seqs 11, 1.729 sec/step, elapsed 0:19:27, exp. remaining 0:02:52, complete 87.15%
att-weights epoch 481, step 592, max_size:classes 12, max_size:data 372, mem_usage:GPU:0 0.8GB, num_seqs 10, 1.759 sec/step, elapsed 0:19:29, exp. remaining 0:02:48, complete 87.40%
att-weights epoch 481, step 593, max_size:classes 12, max_size:data 356, mem_usage:GPU:0 0.8GB, num_seqs 11, 2.116 sec/step, elapsed 0:19:31, exp. remaining 0:02:44, complete 87.71%
att-weights epoch 481, step 594, max_size:classes 12, max_size:data 422, mem_usage:GPU:0 0.8GB, num_seqs 9, 1.514 sec/step, elapsed 0:19:32, exp. remaining 0:02:38, complete 88.09%
att-weights epoch 481, step 595, max_size:classes 12, max_size:data 399, mem_usage:GPU:0 0.8GB, num_seqs 10, 3.699 sec/step, elapsed 0:19:36, exp. remaining 0:02:34, complete 88.41%
att-weights epoch 481, step 596, max_size:classes 13, max_size:data 358, mem_usage:GPU:0 0.8GB, num_seqs 11, 2.446 sec/step, elapsed 0:19:39, exp. remaining 0:02:28, complete 88.83%
att-weights epoch 481, step 597, max_size:classes 10, max_size:data 377, mem_usage:GPU:0 0.8GB, num_seqs 10, 1.730 sec/step, elapsed 0:19:40, exp. remaining 0:02:23, complete 89.14%
att-weights epoch 481, step 598, max_size:classes 11, max_size:data 388, mem_usage:GPU:0 0.8GB, num_seqs 9, 1.580 sec/step, elapsed 0:19:42, exp. remaining 0:02:19, complete 89.42%
att-weights epoch 481, step 599, max_size:classes 9, max_size:data 478, mem_usage:GPU:0 0.8GB, num_seqs 8, 1.394 sec/step, elapsed 0:19:43, exp. remaining 0:02:13, complete 89.87%
att-weights epoch 481, step 600, max_size:classes 13, max_size:data 331, mem_usage:GPU:0 0.8GB, num_seqs 11, 1.643 sec/step, elapsed 0:19:45, exp. remaining 0:02:06, complete 90.33%
att-weights epoch 481, step 601, max_size:classes 10, max_size:data 513, mem_usage:GPU:0 0.8GB, num_seqs 7, 1.252 sec/step, elapsed 0:19:46, exp. remaining 0:02:01, complete 90.71%
att-weights epoch 481, step 602, max_size:classes 14, max_size:data 420, mem_usage:GPU:0 0.8GB, num_seqs 9, 1.310 sec/step, elapsed 0:19:48, exp. remaining 0:01:56, complete 91.06%
att-weights epoch 481, step 603, max_size:classes 11, max_size:data 347, mem_usage:GPU:0 0.8GB, num_seqs 11, 1.404 sec/step, elapsed 0:19:49, exp. remaining 0:01:50, complete 91.48%
att-weights epoch 481, step 604, max_size:classes 11, max_size:data 434, mem_usage:GPU:0 0.8GB, num_seqs 9, 1.375 sec/step, elapsed 0:19:50, exp. remaining 0:01:45, complete 91.86%
att-weights epoch 481, step 605, max_size:classes 13, max_size:data 332, mem_usage:GPU:0 0.8GB, num_seqs 12, 1.822 sec/step, elapsed 0:19:52, exp. remaining 0:01:39, complete 92.28%
att-weights epoch 481, step 606, max_size:classes 11, max_size:data 359, mem_usage:GPU:0 0.8GB, num_seqs 9, 1.479 sec/step, elapsed 0:19:54, exp. remaining 0:01:34, complete 92.67%
att-weights epoch 481, step 607, max_size:classes 10, max_size:data 448, mem_usage:GPU:0 0.8GB, num_seqs 8, 1.309 sec/step, elapsed 0:19:55, exp. remaining 0:01:29, complete 93.05%
att-weights epoch 481, step 608, max_size:classes 11, max_size:data 297, mem_usage:GPU:0 0.8GB, num_seqs 13, 1.768 sec/step, elapsed 0:19:57, exp. remaining 0:01:23, complete 93.47%
att-weights epoch 481, step 609, max_size:classes 16, max_size:data 296, mem_usage:GPU:0 0.8GB, num_seqs 13, 3.062 sec/step, elapsed 0:20:00, exp. remaining 0:01:18, complete 93.85%
att-weights epoch 481, step 610, max_size:classes 9, max_size:data 339, mem_usage:GPU:0 0.8GB, num_seqs 11, 1.608 sec/step, elapsed 0:20:01, exp. remaining 0:01:13, complete 94.24%
att-weights epoch 481, step 611, max_size:classes 9, max_size:data 385, mem_usage:GPU:0 0.8GB, num_seqs 10, 1.294 sec/step, elapsed 0:20:03, exp. remaining 0:01:08, complete 94.59%
att-weights epoch 481, step 612, max_size:classes 9, max_size:data 319, mem_usage:GPU:0 0.8GB, num_seqs 12, 2.490 sec/step, elapsed 0:20:05, exp. remaining 0:01:03, complete 95.01%
att-weights epoch 481, step 613, max_size:classes 11, max_size:data 349, mem_usage:GPU:0 0.8GB, num_seqs 11, 1.837 sec/step, elapsed 0:20:07, exp. remaining 0:00:58, complete 95.39%
att-weights epoch 481, step 614, max_size:classes 10, max_size:data 328, mem_usage:GPU:0 0.8GB, num_seqs 12, 1.933 sec/step, elapsed 0:20:09, exp. remaining 0:00:52, complete 95.84%
att-weights epoch 481, step 615, max_size:classes 10, max_size:data 319, mem_usage:GPU:0 0.8GB, num_seqs 11, 1.493 sec/step, elapsed 0:20:10, exp. remaining 0:00:46, complete 96.26%
att-weights epoch 481, step 616, max_size:classes 9, max_size:data 337, mem_usage:GPU:0 0.8GB, num_seqs 11, 1.452 sec/step, elapsed 0:20:12, exp. remaining 0:00:40, complete 96.79%
att-weights epoch 481, step 617, max_size:classes 9, max_size:data 323, mem_usage:GPU:0 0.8GB, num_seqs 12, 1.678 sec/step, elapsed 0:20:14, exp. remaining 0:00:35, complete 97.17%
att-weights epoch 481, step 618, max_size:classes 11, max_size:data 351, mem_usage:GPU:0 0.8GB, num_seqs 11, 1.839 sec/step, elapsed 0:20:15, exp. remaining 0:00:30, complete 97.52%
att-weights epoch 481, step 619, max_size:classes 9, max_size:data 348, mem_usage:GPU:0 0.8GB, num_seqs 11, 1.810 sec/step, elapsed 0:20:17, exp. remaining 0:00:25, complete 97.94%
att-weights epoch 481, step 620, max_size:classes 8, max_size:data 386, mem_usage:GPU:0 0.8GB, num_seqs 10, 2.293 sec/step, elapsed 0:20:19, exp. remaining 0:00:19, complete 98.43%
att-weights epoch 481, step 621, max_size:classes 8, max_size:data 327, mem_usage:GPU:0 0.8GB, num_seqs 12, 1.809 sec/step, elapsed 0:20:21, exp. remaining 0:00:13, complete 98.92%
att-weights epoch 481, step 622, max_size:classes 9, max_size:data 340, mem_usage:GPU:0 0.8GB, num_seqs 11, 1.096 sec/step, elapsed 0:20:22, exp. remaining 0:00:07, complete 99.37%
att-weights epoch 481, step 623, max_size:classes 7, max_size:data 303, mem_usage:GPU:0 0.8GB, num_seqs 13, 1.995 sec/step, elapsed 0:20:24, exp. remaining 0:00:01, complete 99.90%
att-weights epoch 481, step 624, max_size:classes 9, max_size:data 331, mem_usage:GPU:0 0.8GB, num_seqs 12, 1.123 sec/step, elapsed 0:20:26, exp. remaining 0:00:01, complete 99.90%
att-weights epoch 481, step 625, max_size:classes 7, max_size:data 248, mem_usage:GPU:0 0.8GB, num_seqs 15, 1.319 sec/step, elapsed 0:20:27, exp. remaining 0:00:01, complete 99.90%
att-weights epoch 481, step 626, max_size:classes 7, max_size:data 342, mem_usage:GPU:0 0.8GB, num_seqs 11, 0.933 sec/step, elapsed 0:20:28, exp. remaining 0:00:01, complete 99.90%
att-weights epoch 481, step 627, max_size:classes 7, max_size:data 369, mem_usage:GPU:0 0.8GB, num_seqs 10, 1.413 sec/step, elapsed 0:20:29, exp. remaining 0:00:01, complete 99.90%
att-weights epoch 481, step 628, max_size:classes 6, max_size:data 317, mem_usage:GPU:0 0.8GB, num_seqs 12, 1.029 sec/step, elapsed 0:20:30, exp. remaining 0:00:01, complete 99.90%
att-weights epoch 481, step 629, max_size:classes 6, max_size:data 278, mem_usage:GPU:0 0.8GB, num_seqs 14, 1.432 sec/step, elapsed 0:20:32, exp. remaining 0:00:01, complete 99.90%
att-weights epoch 481, step 630, max_size:classes 6, max_size:data 282, mem_usage:GPU:0 0.8GB, num_seqs 14, 2.178 sec/step, elapsed 0:20:34, exp. remaining 0:00:01, complete 99.90%
att-weights epoch 481, step 631, max_size:classes 5, max_size:data 287, mem_usage:GPU:0 0.8GB, num_seqs 13, 1.122 sec/step, elapsed 0:20:35, exp. remaining 0:00:01, complete 99.90%
att-weights epoch 481, step 632, max_size:classes 7, max_size:data 253, mem_usage:GPU:0 0.8GB, num_seqs 15, 1.181 sec/step, elapsed 0:20:36, exp. remaining 0:00:01, complete 99.90%
att-weights epoch 481, step 633, max_size:classes 3, max_size:data 135, mem_usage:GPU:0 0.8GB, num_seqs 4, 0.849 sec/step, elapsed 0:20:37, exp. remaining 0:00:01, complete 99.90%
Stats:
  mem_usage:GPU:0: Stats(mean=0.8GB, std_dev=19.4MB, min=801.3MB, max=0.8GB, num_seqs=634, avg_data_len=1)
att-weights epoch 481, finished after 634 steps, 0:20:37 elapsed (27.3% computing time)
Layer 'dec_02_att_weights' Stats:
  2864 seqs, 88282648 total frames, 30824.946927 average frames
  Mean: 0.006072314465764884
  Std dev: 0.042536799900763554
  Min/max: 0.0 / 1.0
Layer 'dec_01_att_weights' Stats:
  2864 seqs, 88282648 total frames, 30824.946927 average frames
  Mean: 0.006072314457468553
  Std dev: 0.07535685406878191
  Min/max: 0.0 / 1.0
Layer 'dec_03_att_weights' Stats:
  2864 seqs, 88282648 total frames, 30824.946927 average frames
  Mean: 0.006072314448307995
  Std dev: 0.049085054278321265
  Min/max: 0.0 / 0.9999995
Layer 'dec_04_att_weights' Stats:
  2864 seqs, 88282648 total frames, 30824.946927 average frames
  Mean: 0.006072314453320367
  Std dev: 0.046239342205011184
  Min/max: 0.0 / 0.9999949
Layer 'dec_05_att_weights' Stats:
  2864 seqs, 88282648 total frames, 30824.946927 average frames
  Mean: 0.006072314453406818
  Std dev: 0.07362049250936413
  Min/max: 0.0 / 1.0
Layer 'dec_06_att_weights' Stats:
  2864 seqs, 88282648 total frames, 30824.946927 average frames
  Mean: 0.006072314453903707
  Std dev: 0.06897310893259738
  Min/max: 0.0 / 1.0
Layer 'dec_07_att_weights' Stats:
  2864 seqs, 88282648 total frames, 30824.946927 average frames
  Mean: 0.006072314451656791
  Std dev: 0.07296020934523063
  Min/max: 0.0 / 1.0
Layer 'dec_08_att_weights' Stats:
  2864 seqs, 88282648 total frames, 30824.946927 average frames
  Mean: 0.006072314455351244
  Std dev: 0.07156136993328814
  Min/max: 0.0 / 1.0
Layer 'dec_09_att_weights' Stats:
  2864 seqs, 88282648 total frames, 30824.946927 average frames
  Mean: 0.0060723144576413784
  Std dev: 0.07425808692111643
  Min/max: 0.0 / 1.0
Layer 'dec_10_att_weights' Stats:
  2864 seqs, 88282648 total frames, 30824.946927 average frames
  Mean: 0.006072314451203101
  Std dev: 0.07287552827139789
  Min/max: 0.0 / 1.0
Layer 'dec_11_att_weights' Stats:
  2864 seqs, 88282648 total frames, 30824.946927 average frames
  Mean: 0.006072314443857371
  Std dev: 0.07114473511907633
  Min/max: 0.0 / 1.0
Layer 'dec_12_att_weights' Stats:
  2864 seqs, 88282648 total frames, 30824.946927 average frames
  Mean: 0.006072314446838874
  Std dev: 0.07017381395782606
  Min/max: 0.0 / 1.0
Quitting
+------- EPILOGUE SCRIPT -----------------------------------------------
|
| Job ID ..............: 9507830
| Stopped at ..........: Wed Jul  3 10:24:39 CEST 2019
| Resources requested .: h_rss=8G,h_rt=7200,gpu=1,h_vmem=1536G,s_core=0,pxe=ubuntu_16.04,num_proc=5,scratch_free=5G,h_fsize=20G
| Resources used ......: cpu=00:36:37, mem=7858.24678 GB s, io=10.84972 GB, vmem=3.842G, maxvmem=3.856G, last_file_cache=3.929G, last_rss=2M, max-cache=3.440G
| Memory used .........: 7.368G / 8.000G (92.1%)
| Total time used .....: 0:22:50
|
+------- EPILOGUE SCRIPT -----------------------------------------------
