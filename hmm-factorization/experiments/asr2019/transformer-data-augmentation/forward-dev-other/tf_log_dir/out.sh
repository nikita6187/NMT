+------- PROLOGUE SCRIPT -----------------------------------------------
|
| Job ID ...........: 9491080
| Started at .......: Sun Jun 30 12:29:54 CEST 2019
| Execution host ...: cluster-cn-235
| Cluster queue ....: 4-GPU-1080
| Script ...........: /var/spool/sge/cluster-cn-235/job_scripts/9491080
| > #!/bin/bash
| > 
| > 
| > export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64:/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda/extras/CUPTI/lib64/:/usr/local/cuda-9.1/extras/CUPTI/lib64/:/usr/local/cuda-9.0/extras/CUPTI/lib64:/usr/lib/nvidia-418
| > #export LD_LIBRARY_PATH="/usr/local/lib::/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda-9.0/lib64:/usr/lib/nvidia-390:/opt/rbi/sge/lib/glinux:/opt/rbi/sge/lib/glinux"
| > 
| > source /u/bahar/settings/python3-returnn-tf1.9/bin/activate
| > 
| > 
| > python3 ~/returnn-parnia/tools/get-attention-weights.py /u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc.config --epoch 481 --data 'config:get_dataset("dev-other")' --dump_dir ~/makarov/NMT/hmm-factorization/experiments/asr2019/transformer/forward --layers "dec_02_att_weights" --layers "dec_01_att_weights" --layers "dec_03_att_weights" --layers "dec_04_att_weights" --layers "dec_05_att_weights" --layers "dec_06_att_weights" --layers "dec_07_att_weights" --layers "dec_08_att_weights" --layers "dec_09_att_weights" --layers "dec_10_att_weights" --layers "dec_11_att_weights" --layers "dec_12_att_weights" --rec_layer "output" --batch_size 4000 --tf_log_dir /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/transformer/forward/tf_log_dir --reset_seq_ordering sorted_reverse
|
+------- PROLOGUE SCRIPT -----------------------------------------------
RETURNN get-attention-weights starting up.
RETURNN starting up, version 20181022.053313--git-dfada43-dirty, date/time 2019-06-30-12-30-02 (UTC+0200), pid 31683, cwd /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc, Python /u/bahar/settings/python3-returnn-tf1.9/bin/python3
RETURNN config: /u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc.config
RETURNN command line options: ()
Hostname: cluster-cn-235
TensorFlow: 1.10.0 (v1.10.0-0-g656e7a2b34) (<site-package> in /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow)
Error while getting SGE num_proc: FileNotFoundError(2, "No such file or directory: 'qstat'")
Setup TF inter and intra global thread pools, num_threads None, session opts {'log_device_placement': False, 'device_count': {'GPU': 0}}.
CUDA_VISIBLE_DEVICES is set to '0'.
Collecting TensorFlow device list...
Local devices available to TensorFlow:
  1/2: name: "/device:CPU:0"
       device_type: "CPU"
       memory_limit: 268435456
       locality {
       }
       incarnation: 18322056001686618588
  2/2: name: "/device:GPU:0"
       device_type: "GPU"
       memory_limit: 10911236096
       locality {
         bus_id: 1
         links {
         }
       }
       incarnation: 46370892098926036
       physical_device_desc: "device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1"
Using gpu device 0: GeForce GTX 1080 Ti
Device not set explicitly, and we found a GPU, which we will use.
Model file prefix: /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc/net-model/network
NOTE: We will use 'sorted_reverse' seq ordering.
Setup tf.Session with options {'log_device_placement': False, 'device_count': {'GPU': 1}} ...
layer root/'data' output: Data(name='data', shape=(None, 40))
layer root/'source' output: Data(name='source_output', shape=(None, 40))
layer root/'lstm0_fw' output: Data(name='lstm0_fw_output', shape=(None, 1024), batch_dim_axis=1)
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9491080.1.4-GPU-1080/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.cc -o /var/tmp/9491080.1.4-GPU-1080/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9491080.1.4-GPU-1080/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.cc -o /var/tmp/9491080.1.4-GPU-1080/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
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
layer root/'dec_09_att_value0' output: Data(name='dec_09_att_value0_output', shape=(None, 512))
layer root/'dec_09_att_value' output: Data(name='dec_09_att_value_output', shape=(None, 8, 64))
layer root/'dec_12_att_key0' output: Data(name='dec_12_att_key0_output', shape=(None, 512))
layer root/'dec_12_att_key' output: Data(name='dec_12_att_key_output', shape=(None, 8, 64))
layer root/'dec_06_att_value0' output: Data(name='dec_06_att_value0_output', shape=(None, 512))
layer root/'dec_06_att_value' output: Data(name='dec_06_att_value_output', shape=(None, 8, 64))
layer root/'dec_10_att_key0' output: Data(name='dec_10_att_key0_output', shape=(None, 512))
layer root/'dec_10_att_key' output: Data(name='dec_10_att_key_output', shape=(None, 8, 64))
layer root/'dec_11_att_value0' output: Data(name='dec_11_att_value0_output', shape=(None, 512))
layer root/'dec_11_att_value' output: Data(name='dec_11_att_value_output', shape=(None, 8, 64))
layer root/'dec_02_att_key0' output: Data(name='dec_02_att_key0_output', shape=(None, 512))
layer root/'dec_02_att_key' output: Data(name='dec_02_att_key_output', shape=(None, 8, 64))
layer root/'dec_05_att_value0' output: Data(name='dec_05_att_value0_output', shape=(None, 512))
layer root/'dec_05_att_value' output: Data(name='dec_05_att_value_output', shape=(None, 8, 64))
layer root/'dec_04_att_key0' output: Data(name='dec_04_att_key0_output', shape=(None, 512))
layer root/'dec_04_att_key' output: Data(name='dec_04_att_key_output', shape=(None, 8, 64))
layer root/'dec_07_att_value0' output: Data(name='dec_07_att_value0_output', shape=(None, 512))
layer root/'dec_07_att_value' output: Data(name='dec_07_att_value_output', shape=(None, 8, 64))
layer root/'dec_10_att_value0' output: Data(name='dec_10_att_value0_output', shape=(None, 512))
layer root/'dec_10_att_value' output: Data(name='dec_10_att_value_output', shape=(None, 8, 64))
layer root/'dec_11_att_key0' output: Data(name='dec_11_att_key0_output', shape=(None, 512))
layer root/'dec_11_att_key' output: Data(name='dec_11_att_key_output', shape=(None, 8, 64))
layer root/'dec_04_att_value0' output: Data(name='dec_04_att_value0_output', shape=(None, 512))
layer root/'dec_04_att_value' output: Data(name='dec_04_att_value_output', shape=(None, 8, 64))
layer root/'dec_03_att_key0' output: Data(name='dec_03_att_key0_output', shape=(None, 512))
layer root/'dec_03_att_key' output: Data(name='dec_03_att_key_output', shape=(None, 8, 64))
layer root/'dec_12_att_value0' output: Data(name='dec_12_att_value0_output', shape=(None, 512))
layer root/'dec_12_att_value' output: Data(name='dec_12_att_value_output', shape=(None, 8, 64))
layer root/'dec_01_att_value0' output: Data(name='dec_01_att_value0_output', shape=(None, 512))
layer root/'dec_01_att_value' output: Data(name='dec_01_att_value_output', shape=(None, 8, 64))
layer root/'dec_01_att_key0' output: Data(name='dec_01_att_key0_output', shape=(None, 512))
layer root/'dec_01_att_key' output: Data(name='dec_01_att_key_output', shape=(None, 8, 64))
layer root/'dec_02_att_value0' output: Data(name='dec_02_att_value0_output', shape=(None, 512))
layer root/'dec_02_att_value' output: Data(name='dec_02_att_value_output', shape=(None, 8, 64))
layer root/'dec_08_att_key0' output: Data(name='dec_08_att_key0_output', shape=(None, 512))
layer root/'dec_08_att_key' output: Data(name='dec_08_att_key_output', shape=(None, 8, 64))
layer root/'dec_07_att_key0' output: Data(name='dec_07_att_key0_output', shape=(None, 512))
layer root/'dec_07_att_key' output: Data(name='dec_07_att_key_output', shape=(None, 8, 64))
layer root/'dec_08_att_value0' output: Data(name='dec_08_att_value0_output', shape=(None, 512))
layer root/'dec_08_att_value' output: Data(name='dec_08_att_value_output', shape=(None, 8, 64))
layer root/'dec_05_att_key0' output: Data(name='dec_05_att_key0_output', shape=(None, 512))
layer root/'dec_05_att_key' output: Data(name='dec_05_att_key_output', shape=(None, 8, 64))
layer root/'dec_03_att_value0' output: Data(name='dec_03_att_value0_output', shape=(None, 512))
layer root/'dec_03_att_value' output: Data(name='dec_03_att_value_output', shape=(None, 8, 64))
layer root/'dec_06_att_key0' output: Data(name='dec_06_att_key0_output', shape=(None, 512))
layer root/'dec_06_att_key' output: Data(name='dec_06_att_key_output', shape=(None, 8, 64))
layer root/'dec_09_att_key0' output: Data(name='dec_09_att_key0_output', shape=(None, 512))
layer root/'dec_09_att_key' output: Data(name='dec_09_att_key_output', shape=(None, 8, 64))
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
TF: log_dir: /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/transformer/forward/tf_log_dir/prefix:dev-other-481-2019-06-30-10-29-57
Note: There are still these uninitialized variables: ['learning_rate:0']
att-weights epoch 481, step 0, max_size:classes 94, max_size:data 3206, mem_usage:GPU:0 0.9GB, num_seqs 1, 19.630 sec/step, elapsed 0:00:32, exp. remaining 2:20:11, complete 0.38%
att-weights epoch 481, step 1, max_size:classes 93, max_size:data 3516, mem_usage:GPU:0 0.9GB, num_seqs 1, 12.627 sec/step, elapsed 0:00:45, exp. remaining 2:59:14, complete 0.42%
att-weights epoch 481, step 2, max_size:classes 83, max_size:data 3322, mem_usage:GPU:0 0.9GB, num_seqs 1, 28.936 sec/step, elapsed 0:01:14, exp. remaining 4:31:14, complete 0.45%
att-weights epoch 481, step 3, max_size:classes 90, max_size:data 2413, mem_usage:GPU:0 0.9GB, num_seqs 1, 16.816 sec/step, elapsed 0:01:31, exp. remaining 5:08:56, complete 0.49%
att-weights epoch 481, step 4, max_size:classes 79, max_size:data 2611, mem_usage:GPU:0 0.9GB, num_seqs 1, 7.868 sec/step, elapsed 0:01:38, exp. remaining 5:13:16, complete 0.52%
att-weights epoch 481, step 5, max_size:classes 84, max_size:data 3350, mem_usage:GPU:0 0.9GB, num_seqs 1, 14.441 sec/step, elapsed 0:01:53, exp. remaining 5:36:30, complete 0.56%
att-weights epoch 481, step 6, max_size:classes 78, max_size:data 2592, mem_usage:GPU:0 0.9GB, num_seqs 1, 21.978 sec/step, elapsed 0:02:15, exp. remaining 6:18:00, complete 0.59%
att-weights epoch 481, step 7, max_size:classes 80, max_size:data 2232, mem_usage:GPU:0 0.9GB, num_seqs 1, 7.999 sec/step, elapsed 0:02:23, exp. remaining 6:18:10, complete 0.63%
att-weights epoch 481, step 8, max_size:classes 83, max_size:data 2463, mem_usage:GPU:0 0.9GB, num_seqs 1, 24.177 sec/step, elapsed 0:02:47, exp. remaining 6:58:45, complete 0.66%
att-weights epoch 481, step 9, max_size:classes 89, max_size:data 2577, mem_usage:GPU:0 0.9GB, num_seqs 1, 19.033 sec/step, elapsed 0:03:06, exp. remaining 7:22:51, complete 0.70%
att-weights epoch 481, step 10, max_size:classes 92, max_size:data 2618, mem_usage:GPU:0 0.9GB, num_seqs 1, 3.100 sec/step, elapsed 0:03:10, exp. remaining 7:08:43, complete 0.73%
att-weights epoch 481, step 11, max_size:classes 86, max_size:data 3177, mem_usage:GPU:0 0.9GB, num_seqs 1, 20.799 sec/step, elapsed 0:03:30, exp. remaining 7:33:56, complete 0.77%
att-weights epoch 481, step 12, max_size:classes 82, max_size:data 2822, mem_usage:GPU:0 0.9GB, num_seqs 1, 21.532 sec/step, elapsed 0:03:52, exp. remaining 7:58:23, complete 0.80%
att-weights epoch 481, step 13, max_size:classes 73, max_size:data 2951, mem_usage:GPU:0 0.9GB, num_seqs 1, 5.645 sec/step, elapsed 0:03:58, exp. remaining 7:49:26, complete 0.84%
att-weights epoch 481, step 14, max_size:classes 72, max_size:data 2463, mem_usage:GPU:0 0.9GB, num_seqs 1, 7.473 sec/step, elapsed 0:04:05, exp. remaining 7:44:38, complete 0.87%
att-weights epoch 481, step 15, max_size:classes 76, max_size:data 2211, mem_usage:GPU:0 0.9GB, num_seqs 1, 2.986 sec/step, elapsed 0:04:08, exp. remaining 7:32:03, complete 0.91%
att-weights epoch 481, step 16, max_size:classes 78, max_size:data 2964, mem_usage:GPU:0 0.9GB, num_seqs 1, 6.733 sec/step, elapsed 0:04:15, exp. remaining 7:26:57, complete 0.94%
att-weights epoch 481, step 17, max_size:classes 76, max_size:data 2016, mem_usage:GPU:0 0.9GB, num_seqs 1, 3.558 sec/step, elapsed 0:04:18, exp. remaining 7:16:50, complete 0.98%
att-weights epoch 481, step 18, max_size:classes 87, max_size:data 2598, mem_usage:GPU:0 0.9GB, num_seqs 1, 22.529 sec/step, elapsed 0:04:41, exp. remaining 7:38:20, complete 1.01%
att-weights epoch 481, step 19, max_size:classes 74, max_size:data 2578, mem_usage:GPU:0 0.9GB, num_seqs 1, 7.384 sec/step, elapsed 0:04:48, exp. remaining 7:34:32, complete 1.05%
att-weights epoch 481, step 20, max_size:classes 110, max_size:data 2909, mem_usage:GPU:0 0.9GB, num_seqs 1, 23.304 sec/step, elapsed 0:05:12, exp. remaining 7:55:12, complete 1.08%
att-weights epoch 481, step 21, max_size:classes 70, max_size:data 1753, mem_usage:GPU:0 0.9GB, num_seqs 1, 6.834 sec/step, elapsed 0:05:18, exp. remaining 7:50:17, complete 1.12%
att-weights epoch 481, step 22, max_size:classes 77, max_size:data 2778, mem_usage:GPU:0 0.9GB, num_seqs 1, 6.269 sec/step, elapsed 0:05:25, exp. remaining 7:44:50, complete 1.15%
att-weights epoch 481, step 23, max_size:classes 83, max_size:data 2699, mem_usage:GPU:0 0.9GB, num_seqs 1, 7.382 sec/step, elapsed 0:05:32, exp. remaining 7:41:15, complete 1.19%
att-weights epoch 481, step 24, max_size:classes 63, max_size:data 3014, mem_usage:GPU:0 0.9GB, num_seqs 1, 8.972 sec/step, elapsed 0:05:41, exp. remaining 7:40:00, complete 1.22%
att-weights epoch 481, step 25, max_size:classes 63, max_size:data 3232, mem_usage:GPU:0 0.9GB, num_seqs 1, 14.068 sec/step, elapsed 0:05:55, exp. remaining 7:32:44, complete 1.29%
att-weights epoch 481, step 26, max_size:classes 62, max_size:data 1935, mem_usage:GPU:0 0.9GB, num_seqs 1, 9.026 sec/step, elapsed 0:06:04, exp. remaining 7:20:07, complete 1.36%
att-weights epoch 481, step 27, max_size:classes 69, max_size:data 2024, mem_usage:GPU:0 0.9GB, num_seqs 1, 5.278 sec/step, elapsed 0:06:09, exp. remaining 7:04:25, complete 1.43%
att-weights epoch 481, step 28, max_size:classes 66, max_size:data 1976, mem_usage:GPU:0 0.9GB, num_seqs 1, 7.911 sec/step, elapsed 0:06:17, exp. remaining 7:03:01, complete 1.47%
att-weights epoch 481, step 29, max_size:classes 69, max_size:data 2385, mem_usage:GPU:0 0.9GB, num_seqs 1, 5.433 sec/step, elapsed 0:06:23, exp. remaining 6:58:59, complete 1.50%
att-weights epoch 481, step 30, max_size:classes 58, max_size:data 2818, mem_usage:GPU:0 0.9GB, num_seqs 1, 5.766 sec/step, elapsed 0:06:28, exp. remaining 6:55:28, complete 1.54%
att-weights epoch 481, step 31, max_size:classes 60, max_size:data 1849, mem_usage:GPU:0 0.9GB, num_seqs 1, 6.365 sec/step, elapsed 0:06:35, exp. remaining 6:52:45, complete 1.57%
att-weights epoch 481, step 32, max_size:classes 67, max_size:data 2141, mem_usage:GPU:0 0.9GB, num_seqs 1, 3.241 sec/step, elapsed 0:06:38, exp. remaining 6:46:56, complete 1.61%
att-weights epoch 481, step 33, max_size:classes 61, max_size:data 2119, mem_usage:GPU:0 0.9GB, num_seqs 1, 8.302 sec/step, elapsed 0:06:46, exp. remaining 6:46:26, complete 1.64%
att-weights epoch 481, step 34, max_size:classes 64, max_size:data 1820, mem_usage:GPU:0 0.9GB, num_seqs 2, 9.865 sec/step, elapsed 0:06:56, exp. remaining 6:39:01, complete 1.71%
att-weights epoch 481, step 35, max_size:classes 64, max_size:data 1807, mem_usage:GPU:0 0.9GB, num_seqs 2, 30.508 sec/step, elapsed 0:07:27, exp. remaining 6:51:08, complete 1.78%
att-weights epoch 481, step 36, max_size:classes 66, max_size:data 1775, mem_usage:GPU:0 0.9GB, num_seqs 2, 10.680 sec/step, elapsed 0:07:37, exp. remaining 6:44:47, complete 1.85%
att-weights epoch 481, step 37, max_size:classes 61, max_size:data 1991, mem_usage:GPU:0 0.9GB, num_seqs 1, 3.554 sec/step, elapsed 0:07:41, exp. remaining 6:32:49, complete 1.92%
att-weights epoch 481, step 38, max_size:classes 63, max_size:data 2176, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.199 sec/step, elapsed 0:07:42, exp. remaining 6:19:45, complete 1.99%
att-weights epoch 481, step 39, max_size:classes 65, max_size:data 1634, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.926 sec/step, elapsed 0:07:43, exp. remaining 6:07:21, complete 2.06%
att-weights epoch 481, step 40, max_size:classes 58, max_size:data 2835, mem_usage:GPU:0 0.9GB, num_seqs 1, 10.886 sec/step, elapsed 0:07:54, exp. remaining 6:03:23, complete 2.13%
att-weights epoch 481, step 41, max_size:classes 60, max_size:data 1838, mem_usage:GPU:0 0.9GB, num_seqs 1, 3.347 sec/step, elapsed 0:07:57, exp. remaining 5:54:05, complete 2.20%
att-weights epoch 481, step 42, max_size:classes 68, max_size:data 2190, mem_usage:GPU:0 0.9GB, num_seqs 1, 2.516 sec/step, elapsed 0:08:00, exp. remaining 5:44:45, complete 2.27%
att-weights epoch 481, step 43, max_size:classes 63, max_size:data 1632, mem_usage:GPU:0 0.9GB, num_seqs 2, 17.658 sec/step, elapsed 0:08:18, exp. remaining 5:51:53, complete 2.30%
att-weights epoch 481, step 44, max_size:classes 59, max_size:data 1934, mem_usage:GPU:0 0.9GB, num_seqs 2, 18.221 sec/step, elapsed 0:08:36, exp. remaining 5:59:11, complete 2.34%
att-weights epoch 481, step 45, max_size:classes 64, max_size:data 1987, mem_usage:GPU:0 0.9GB, num_seqs 2, 7.133 sec/step, elapsed 0:08:43, exp. remaining 5:53:20, complete 2.41%
att-weights epoch 481, step 46, max_size:classes 62, max_size:data 1990, mem_usage:GPU:0 0.9GB, num_seqs 2, 13.403 sec/step, elapsed 0:08:56, exp. remaining 5:51:56, complete 2.48%
att-weights epoch 481, step 47, max_size:classes 59, max_size:data 1627, mem_usage:GPU:0 0.9GB, num_seqs 2, 6.818 sec/step, elapsed 0:09:03, exp. remaining 5:46:23, complete 2.55%
att-weights epoch 481, step 48, max_size:classes 58, max_size:data 1779, mem_usage:GPU:0 0.9GB, num_seqs 2, 7.168 sec/step, elapsed 0:09:10, exp. remaining 5:41:21, complete 2.62%
att-weights epoch 481, step 49, max_size:classes 55, max_size:data 1813, mem_usage:GPU:0 0.9GB, num_seqs 2, 10.304 sec/step, elapsed 0:09:21, exp. remaining 5:43:02, complete 2.65%
att-weights epoch 481, step 50, max_size:classes 58, max_size:data 1785, mem_usage:GPU:0 0.9GB, num_seqs 2, 20.490 sec/step, elapsed 0:09:41, exp. remaining 5:50:50, complete 2.69%
att-weights epoch 481, step 51, max_size:classes 60, max_size:data 1824, mem_usage:GPU:0 0.9GB, num_seqs 2, 28.367 sec/step, elapsed 0:10:09, exp. remaining 6:03:06, complete 2.72%
att-weights epoch 481, step 52, max_size:classes 57, max_size:data 1478, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.284 sec/step, elapsed 0:10:11, exp. remaining 5:54:30, complete 2.79%
att-weights epoch 481, step 53, max_size:classes 64, max_size:data 2023, mem_usage:GPU:0 0.9GB, num_seqs 1, 3.380 sec/step, elapsed 0:10:14, exp. remaining 5:51:57, complete 2.83%
att-weights epoch 481, step 54, max_size:classes 62, max_size:data 1990, mem_usage:GPU:0 0.9GB, num_seqs 2, 26.820 sec/step, elapsed 0:10:41, exp. remaining 5:58:12, complete 2.90%
att-weights epoch 481, step 55, max_size:classes 56, max_size:data 1850, mem_usage:GPU:0 0.9GB, num_seqs 2, 8.912 sec/step, elapsed 0:10:50, exp. remaining 5:58:43, complete 2.93%
att-weights epoch 481, step 56, max_size:classes 54, max_size:data 1942, mem_usage:GPU:0 0.9GB, num_seqs 2, 17.843 sec/step, elapsed 0:11:08, exp. remaining 6:04:06, complete 2.97%
att-weights epoch 481, step 57, max_size:classes 61, max_size:data 1607, mem_usage:GPU:0 0.9GB, num_seqs 2, 15.553 sec/step, elapsed 0:11:23, exp. remaining 6:03:45, complete 3.04%
att-weights epoch 481, step 58, max_size:classes 60, max_size:data 1529, mem_usage:GPU:0 0.9GB, num_seqs 1, 12.873 sec/step, elapsed 0:11:36, exp. remaining 6:02:00, complete 3.11%
att-weights epoch 481, step 59, max_size:classes 62, max_size:data 2044, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.985 sec/step, elapsed 0:11:38, exp. remaining 5:54:48, complete 3.18%
att-weights epoch 481, step 60, max_size:classes 55, max_size:data 2007, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.584 sec/step, elapsed 0:11:40, exp. remaining 5:47:42, complete 3.25%
att-weights epoch 481, step 61, max_size:classes 51, max_size:data 1820, mem_usage:GPU:0 0.9GB, num_seqs 2, 9.039 sec/step, elapsed 0:11:49, exp. remaining 5:44:32, complete 3.32%
att-weights epoch 481, step 62, max_size:classes 62, max_size:data 2501, mem_usage:GPU:0 0.9GB, num_seqs 1, 26.500 sec/step, elapsed 0:12:15, exp. remaining 5:49:47, complete 3.39%
att-weights epoch 481, step 63, max_size:classes 59, max_size:data 1603, mem_usage:GPU:0 0.9GB, num_seqs 2, 11.710 sec/step, elapsed 0:12:27, exp. remaining 5:47:55, complete 3.46%
att-weights epoch 481, step 64, max_size:classes 55, max_size:data 1503, mem_usage:GPU:0 0.9GB, num_seqs 1, 2.504 sec/step, elapsed 0:12:29, exp. remaining 5:41:56, complete 3.53%
att-weights epoch 481, step 65, max_size:classes 58, max_size:data 2102, mem_usage:GPU:0 0.9GB, num_seqs 1, 6.235 sec/step, elapsed 0:12:36, exp. remaining 5:37:50, complete 3.60%
att-weights epoch 481, step 66, max_size:classes 56, max_size:data 1594, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.119 sec/step, elapsed 0:12:38, exp. remaining 5:32:05, complete 3.67%
att-weights epoch 481, step 67, max_size:classes 57, max_size:data 1603, mem_usage:GPU:0 0.9GB, num_seqs 2, 15.697 sec/step, elapsed 0:12:54, exp. remaining 5:32:23, complete 3.74%
att-weights epoch 481, step 68, max_size:classes 57, max_size:data 1912, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.505 sec/step, elapsed 0:12:56, exp. remaining 5:27:06, complete 3.81%
att-weights epoch 481, step 69, max_size:classes 50, max_size:data 1795, mem_usage:GPU:0 0.9GB, num_seqs 2, 5.909 sec/step, elapsed 0:13:02, exp. remaining 5:23:25, complete 3.88%
att-weights epoch 481, step 70, max_size:classes 55, max_size:data 1574, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.436 sec/step, elapsed 0:13:04, exp. remaining 5:21:25, complete 3.91%
att-weights epoch 481, step 71, max_size:classes 53, max_size:data 1496, mem_usage:GPU:0 0.9GB, num_seqs 2, 3.841 sec/step, elapsed 0:13:08, exp. remaining 5:20:01, complete 3.95%
att-weights epoch 481, step 72, max_size:classes 58, max_size:data 1601, mem_usage:GPU:0 0.9GB, num_seqs 2, 10.323 sec/step, elapsed 0:13:19, exp. remaining 5:18:20, complete 4.02%
att-weights epoch 481, step 73, max_size:classes 51, max_size:data 1366, mem_usage:GPU:0 0.9GB, num_seqs 2, 16.692 sec/step, elapsed 0:13:35, exp. remaining 5:19:11, complete 4.09%
att-weights epoch 481, step 74, max_size:classes 55, max_size:data 1479, mem_usage:GPU:0 0.9GB, num_seqs 2, 5.652 sec/step, elapsed 0:13:41, exp. remaining 5:15:46, complete 4.16%
att-weights epoch 481, step 75, max_size:classes 48, max_size:data 1657, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.558 sec/step, elapsed 0:13:43, exp. remaining 5:11:18, complete 4.22%
att-weights epoch 481, step 76, max_size:classes 53, max_size:data 1560, mem_usage:GPU:0 0.9GB, num_seqs 2, 9.549 sec/step, elapsed 0:13:53, exp. remaining 5:09:33, complete 4.29%
att-weights epoch 481, step 77, max_size:classes 58, max_size:data 1576, mem_usage:GPU:0 0.9GB, num_seqs 2, 6.629 sec/step, elapsed 0:14:00, exp. remaining 5:06:48, complete 4.36%
att-weights epoch 481, step 78, max_size:classes 54, max_size:data 1657, mem_usage:GPU:0 0.9GB, num_seqs 2, 6.236 sec/step, elapsed 0:14:06, exp. remaining 5:03:59, complete 4.43%
att-weights epoch 481, step 79, max_size:classes 50, max_size:data 1349, mem_usage:GPU:0 0.9GB, num_seqs 1, 2.665 sec/step, elapsed 0:14:09, exp. remaining 5:00:00, complete 4.50%
att-weights epoch 481, step 80, max_size:classes 53, max_size:data 2082, mem_usage:GPU:0 0.9GB, num_seqs 1, 3.184 sec/step, elapsed 0:14:12, exp. remaining 4:56:19, complete 4.57%
att-weights epoch 481, step 81, max_size:classes 54, max_size:data 1891, mem_usage:GPU:0 0.9GB, num_seqs 2, 25.562 sec/step, elapsed 0:14:37, exp. remaining 5:02:47, complete 4.61%
att-weights epoch 481, step 82, max_size:classes 54, max_size:data 1364, mem_usage:GPU:0 0.9GB, num_seqs 2, 17.103 sec/step, elapsed 0:14:54, exp. remaining 5:03:51, complete 4.68%
att-weights epoch 481, step 83, max_size:classes 45, max_size:data 1750, mem_usage:GPU:0 0.9GB, num_seqs 2, 9.535 sec/step, elapsed 0:15:04, exp. remaining 5:02:21, complete 4.75%
att-weights epoch 481, step 84, max_size:classes 50, max_size:data 1418, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.784 sec/step, elapsed 0:15:06, exp. remaining 4:58:20, complete 4.82%
att-weights epoch 481, step 85, max_size:classes 61, max_size:data 1682, mem_usage:GPU:0 0.9GB, num_seqs 2, 3.686 sec/step, elapsed 0:15:09, exp. remaining 4:55:03, complete 4.89%
att-weights epoch 481, step 86, max_size:classes 54, max_size:data 1357, mem_usage:GPU:0 0.9GB, num_seqs 2, 12.785 sec/step, elapsed 0:15:22, exp. remaining 4:54:46, complete 4.96%
att-weights epoch 481, step 87, max_size:classes 55, max_size:data 1520, mem_usage:GPU:0 0.9GB, num_seqs 2, 9.846 sec/step, elapsed 0:15:32, exp. remaining 4:53:34, complete 5.03%
att-weights epoch 481, step 88, max_size:classes 50, max_size:data 1434, mem_usage:GPU:0 0.9GB, num_seqs 2, 5.009 sec/step, elapsed 0:15:37, exp. remaining 4:50:53, complete 5.10%
att-weights epoch 481, step 89, max_size:classes 52, max_size:data 1789, mem_usage:GPU:0 0.9GB, num_seqs 2, 14.434 sec/step, elapsed 0:15:51, exp. remaining 4:51:09, complete 5.17%
att-weights epoch 481, step 90, max_size:classes 59, max_size:data 2713, mem_usage:GPU:0 0.9GB, num_seqs 1, 17.610 sec/step, elapsed 0:16:09, exp. remaining 4:52:22, complete 5.24%
att-weights epoch 481, step 91, max_size:classes 52, max_size:data 1987, mem_usage:GPU:0 0.9GB, num_seqs 2, 13.107 sec/step, elapsed 0:16:22, exp. remaining 4:52:13, complete 5.31%
att-weights epoch 481, step 92, max_size:classes 51, max_size:data 1756, mem_usage:GPU:0 0.9GB, num_seqs 2, 8.048 sec/step, elapsed 0:16:30, exp. remaining 4:50:34, complete 5.38%
att-weights epoch 481, step 93, max_size:classes 49, max_size:data 1426, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.219 sec/step, elapsed 0:16:32, exp. remaining 4:49:14, complete 5.41%
att-weights epoch 481, step 94, max_size:classes 48, max_size:data 1450, mem_usage:GPU:0 0.9GB, num_seqs 2, 9.439 sec/step, elapsed 0:16:42, exp. remaining 4:48:03, complete 5.48%
att-weights epoch 481, step 95, max_size:classes 49, max_size:data 1606, mem_usage:GPU:0 0.9GB, num_seqs 2, 8.459 sec/step, elapsed 0:16:50, exp. remaining 4:46:37, complete 5.55%
att-weights epoch 481, step 96, max_size:classes 49, max_size:data 1593, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.516 sec/step, elapsed 0:16:52, exp. remaining 4:43:16, complete 5.62%
att-weights epoch 481, step 97, max_size:classes 54, max_size:data 1701, mem_usage:GPU:0 0.9GB, num_seqs 2, 4.338 sec/step, elapsed 0:16:56, exp. remaining 4:40:47, complete 5.69%
att-weights epoch 481, step 98, max_size:classes 47, max_size:data 1596, mem_usage:GPU:0 0.9GB, num_seqs 2, 8.157 sec/step, elapsed 0:17:04, exp. remaining 4:39:24, complete 5.76%
att-weights epoch 481, step 99, max_size:classes 49, max_size:data 1878, mem_usage:GPU:0 0.9GB, num_seqs 2, 8.080 sec/step, elapsed 0:17:12, exp. remaining 4:38:01, complete 5.83%
att-weights epoch 481, step 100, max_size:classes 55, max_size:data 1407, mem_usage:GPU:0 0.9GB, num_seqs 2, 17.431 sec/step, elapsed 0:17:30, exp. remaining 4:39:10, complete 5.90%
att-weights epoch 481, step 101, max_size:classes 48, max_size:data 1524, mem_usage:GPU:0 0.9GB, num_seqs 2, 13.455 sec/step, elapsed 0:17:43, exp. remaining 4:39:13, complete 5.97%
att-weights epoch 481, step 102, max_size:classes 52, max_size:data 2377, mem_usage:GPU:0 0.9GB, num_seqs 1, 19.941 sec/step, elapsed 0:18:03, exp. remaining 4:40:58, complete 6.04%
att-weights epoch 481, step 103, max_size:classes 52, max_size:data 1682, mem_usage:GPU:0 0.9GB, num_seqs 2, 6.705 sec/step, elapsed 0:18:10, exp. remaining 4:39:16, complete 6.11%
att-weights epoch 481, step 104, max_size:classes 50, max_size:data 1310, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.543 sec/step, elapsed 0:18:12, exp. remaining 4:36:17, complete 6.18%
att-weights epoch 481, step 105, max_size:classes 46, max_size:data 1659, mem_usage:GPU:0 0.9GB, num_seqs 2, 4.137 sec/step, elapsed 0:18:16, exp. remaining 4:34:02, complete 6.25%
att-weights epoch 481, step 106, max_size:classes 56, max_size:data 1578, mem_usage:GPU:0 0.9GB, num_seqs 2, 18.323 sec/step, elapsed 0:18:34, exp. remaining 4:35:20, complete 6.32%
att-weights epoch 481, step 107, max_size:classes 49, max_size:data 1712, mem_usage:GPU:0 0.9GB, num_seqs 2, 5.311 sec/step, elapsed 0:18:39, exp. remaining 4:33:25, complete 6.39%
att-weights epoch 481, step 108, max_size:classes 49, max_size:data 1813, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.606 sec/step, elapsed 0:18:42, exp. remaining 4:30:53, complete 6.46%
att-weights epoch 481, step 109, max_size:classes 48, max_size:data 1637, mem_usage:GPU:0 0.9GB, num_seqs 2, 6.976 sec/step, elapsed 0:18:49, exp. remaining 4:29:27, complete 6.53%
att-weights epoch 481, step 110, max_size:classes 52, max_size:data 1820, mem_usage:GPU:0 0.9GB, num_seqs 2, 6.432 sec/step, elapsed 0:18:55, exp. remaining 4:27:55, complete 6.60%
att-weights epoch 481, step 111, max_size:classes 43, max_size:data 1717, mem_usage:GPU:0 0.9GB, num_seqs 2, 5.513 sec/step, elapsed 0:19:01, exp. remaining 4:26:12, complete 6.67%
att-weights epoch 481, step 112, max_size:classes 44, max_size:data 1363, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.211 sec/step, elapsed 0:19:02, exp. remaining 4:23:32, complete 6.74%
att-weights epoch 481, step 113, max_size:classes 51, max_size:data 1431, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.350 sec/step, elapsed 0:19:03, exp. remaining 4:20:56, complete 6.81%
att-weights epoch 481, step 114, max_size:classes 48, max_size:data 1879, mem_usage:GPU:0 0.9GB, num_seqs 2, 17.025 sec/step, elapsed 0:19:20, exp. remaining 4:21:56, complete 6.88%
att-weights epoch 481, step 115, max_size:classes 50, max_size:data 1384, mem_usage:GPU:0 0.9GB, num_seqs 2, 3.610 sec/step, elapsed 0:19:24, exp. remaining 4:19:55, complete 6.95%
att-weights epoch 481, step 116, max_size:classes 44, max_size:data 1359, mem_usage:GPU:0 0.9GB, num_seqs 2, 7.051 sec/step, elapsed 0:19:31, exp. remaining 4:18:42, complete 7.02%
att-weights epoch 481, step 117, max_size:classes 47, max_size:data 1672, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.936 sec/step, elapsed 0:19:33, exp. remaining 4:16:23, complete 7.09%
att-weights epoch 481, step 118, max_size:classes 45, max_size:data 1593, mem_usage:GPU:0 0.9GB, num_seqs 2, 4.860 sec/step, elapsed 0:19:38, exp. remaining 4:14:44, complete 7.16%
att-weights epoch 481, step 119, max_size:classes 45, max_size:data 1746, mem_usage:GPU:0 0.9GB, num_seqs 2, 3.095 sec/step, elapsed 0:19:41, exp. remaining 4:12:45, complete 7.23%
att-weights epoch 481, step 120, max_size:classes 47, max_size:data 1635, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.389 sec/step, elapsed 0:19:42, exp. remaining 4:10:26, complete 7.30%
att-weights epoch 481, step 121, max_size:classes 48, max_size:data 1607, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.496 sec/step, elapsed 0:19:44, exp. remaining 4:08:11, complete 7.37%
att-weights epoch 481, step 122, max_size:classes 59, max_size:data 1396, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.832 sec/step, elapsed 0:19:47, exp. remaining 4:06:15, complete 7.44%
att-weights epoch 481, step 123, max_size:classes 46, max_size:data 1434, mem_usage:GPU:0 0.9GB, num_seqs 2, 12.858 sec/step, elapsed 0:20:00, exp. remaining 4:06:25, complete 7.51%
att-weights epoch 481, step 124, max_size:classes 39, max_size:data 1269, mem_usage:GPU:0 0.9GB, num_seqs 2, 4.639 sec/step, elapsed 0:20:04, exp. remaining 4:03:42, complete 7.61%
att-weights epoch 481, step 125, max_size:classes 46, max_size:data 1530, mem_usage:GPU:0 0.9GB, num_seqs 2, 9.514 sec/step, elapsed 0:20:14, exp. remaining 4:03:12, complete 7.68%
att-weights epoch 481, step 126, max_size:classes 39, max_size:data 1782, mem_usage:GPU:0 0.9GB, num_seqs 2, 4.815 sec/step, elapsed 0:20:19, exp. remaining 4:01:47, complete 7.75%
att-weights epoch 481, step 127, max_size:classes 43, max_size:data 1598, mem_usage:GPU:0 0.9GB, num_seqs 2, 4.611 sec/step, elapsed 0:20:23, exp. remaining 4:00:21, complete 7.82%
att-weights epoch 481, step 128, max_size:classes 47, max_size:data 1236, mem_usage:GPU:0 0.9GB, num_seqs 2, 14.191 sec/step, elapsed 0:20:37, exp. remaining 4:00:48, complete 7.89%
att-weights epoch 481, step 129, max_size:classes 43, max_size:data 1534, mem_usage:GPU:0 0.9GB, num_seqs 2, 4.962 sec/step, elapsed 0:20:42, exp. remaining 3:59:28, complete 7.96%
att-weights epoch 481, step 130, max_size:classes 47, max_size:data 1391, mem_usage:GPU:0 0.9GB, num_seqs 2, 6.921 sec/step, elapsed 0:20:49, exp. remaining 3:57:24, complete 8.07%
att-weights epoch 481, step 131, max_size:classes 43, max_size:data 1494, mem_usage:GPU:0 0.9GB, num_seqs 2, 9.279 sec/step, elapsed 0:20:59, exp. remaining 3:56:56, complete 8.14%
att-weights epoch 481, step 132, max_size:classes 45, max_size:data 1531, mem_usage:GPU:0 0.9GB, num_seqs 2, 11.172 sec/step, elapsed 0:21:10, exp. remaining 3:56:49, complete 8.21%
att-weights epoch 481, step 133, max_size:classes 41, max_size:data 1304, mem_usage:GPU:0 0.9GB, num_seqs 3, 12.288 sec/step, elapsed 0:21:22, exp. remaining 3:56:55, complete 8.28%
att-weights epoch 481, step 134, max_size:classes 45, max_size:data 1371, mem_usage:GPU:0 0.9GB, num_seqs 2, 3.073 sec/step, elapsed 0:21:25, exp. remaining 3:55:19, complete 8.34%
att-weights epoch 481, step 135, max_size:classes 48, max_size:data 1380, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.775 sec/step, elapsed 0:21:28, exp. remaining 3:53:41, complete 8.41%
att-weights epoch 481, step 136, max_size:classes 45, max_size:data 1664, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.397 sec/step, elapsed 0:21:30, exp. remaining 3:50:59, complete 8.52%
att-weights epoch 481, step 137, max_size:classes 41, max_size:data 1115, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.680 sec/step, elapsed 0:21:32, exp. remaining 3:48:13, complete 8.62%
att-weights epoch 481, step 138, max_size:classes 41, max_size:data 1760, mem_usage:GPU:0 0.9GB, num_seqs 2, 3.007 sec/step, elapsed 0:21:35, exp. remaining 3:45:44, complete 8.73%
att-weights epoch 481, step 139, max_size:classes 44, max_size:data 1268, mem_usage:GPU:0 0.9GB, num_seqs 3, 12.666 sec/step, elapsed 0:21:48, exp. remaining 3:45:58, complete 8.80%
att-weights epoch 481, step 140, max_size:classes 41, max_size:data 1466, mem_usage:GPU:0 0.9GB, num_seqs 2, 12.618 sec/step, elapsed 0:22:00, exp. remaining 3:46:10, complete 8.87%
att-weights epoch 481, step 141, max_size:classes 41, max_size:data 1377, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.434 sec/step, elapsed 0:22:03, exp. remaining 3:44:39, complete 8.94%
att-weights epoch 481, step 142, max_size:classes 47, max_size:data 1233, mem_usage:GPU:0 0.9GB, num_seqs 2, 6.621 sec/step, elapsed 0:22:09, exp. remaining 3:43:51, complete 9.01%
att-weights epoch 481, step 143, max_size:classes 40, max_size:data 1558, mem_usage:GPU:0 0.9GB, num_seqs 2, 3.617 sec/step, elapsed 0:22:13, exp. remaining 3:41:37, complete 9.11%
att-weights epoch 481, step 144, max_size:classes 38, max_size:data 1389, mem_usage:GPU:0 0.9GB, num_seqs 2, 3.187 sec/step, elapsed 0:22:16, exp. remaining 3:39:23, complete 9.22%
att-weights epoch 481, step 145, max_size:classes 42, max_size:data 1079, mem_usage:GPU:0 0.9GB, num_seqs 3, 6.155 sec/step, elapsed 0:22:22, exp. remaining 3:38:34, complete 9.29%
att-weights epoch 481, step 146, max_size:classes 41, max_size:data 1246, mem_usage:GPU:0 0.9GB, num_seqs 3, 4.819 sec/step, elapsed 0:22:27, exp. remaining 3:36:39, complete 9.39%
att-weights epoch 481, step 147, max_size:classes 42, max_size:data 1182, mem_usage:GPU:0 0.9GB, num_seqs 3, 9.036 sec/step, elapsed 0:22:36, exp. remaining 3:36:20, complete 9.46%
att-weights epoch 481, step 148, max_size:classes 40, max_size:data 1375, mem_usage:GPU:0 0.9GB, num_seqs 2, 11.180 sec/step, elapsed 0:22:47, exp. remaining 3:35:28, complete 9.57%
att-weights epoch 481, step 149, max_size:classes 41, max_size:data 974, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.417 sec/step, elapsed 0:22:50, exp. remaining 3:34:07, complete 9.64%
att-weights epoch 481, step 150, max_size:classes 38, max_size:data 1528, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.379 sec/step, elapsed 0:22:52, exp. remaining 3:32:47, complete 9.71%
att-weights epoch 481, step 151, max_size:classes 43, max_size:data 1376, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.581 sec/step, elapsed 0:22:55, exp. remaining 3:31:30, complete 9.78%
att-weights epoch 481, step 152, max_size:classes 40, max_size:data 1006, mem_usage:GPU:0 0.9GB, num_seqs 3, 3.554 sec/step, elapsed 0:22:58, exp. remaining 3:29:33, complete 9.88%
att-weights epoch 481, step 153, max_size:classes 43, max_size:data 1126, mem_usage:GPU:0 0.9GB, num_seqs 3, 5.499 sec/step, elapsed 0:23:04, exp. remaining 3:27:57, complete 9.99%
att-weights epoch 481, step 154, max_size:classes 45, max_size:data 1416, mem_usage:GPU:0 0.9GB, num_seqs 2, 15.168 sec/step, elapsed 0:23:19, exp. remaining 3:27:48, complete 10.09%
att-weights epoch 481, step 155, max_size:classes 39, max_size:data 1275, mem_usage:GPU:0 0.9GB, num_seqs 3, 10.977 sec/step, elapsed 0:23:30, exp. remaining 3:27:02, complete 10.20%
att-weights epoch 481, step 156, max_size:classes 39, max_size:data 1359, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.844 sec/step, elapsed 0:23:32, exp. remaining 3:25:44, complete 10.27%
att-weights epoch 481, step 157, max_size:classes 44, max_size:data 1266, mem_usage:GPU:0 0.9GB, num_seqs 3, 4.281 sec/step, elapsed 0:23:36, exp. remaining 3:24:48, complete 10.34%
att-weights epoch 481, step 158, max_size:classes 40, max_size:data 1008, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.110 sec/step, elapsed 0:23:38, exp. remaining 3:23:35, complete 10.41%
att-weights epoch 481, step 159, max_size:classes 43, max_size:data 1530, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.936 sec/step, elapsed 0:23:40, exp. remaining 3:22:20, complete 10.47%
att-weights epoch 481, step 160, max_size:classes 33, max_size:data 1605, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.073 sec/step, elapsed 0:23:42, exp. remaining 3:21:08, complete 10.54%
att-weights epoch 481, step 161, max_size:classes 46, max_size:data 1117, mem_usage:GPU:0 0.9GB, num_seqs 3, 5.865 sec/step, elapsed 0:23:48, exp. remaining 3:20:29, complete 10.61%
att-weights epoch 481, step 162, max_size:classes 41, max_size:data 1300, mem_usage:GPU:0 0.9GB, num_seqs 3, 10.945 sec/step, elapsed 0:23:59, exp. remaining 3:19:48, complete 10.72%
att-weights epoch 481, step 163, max_size:classes 41, max_size:data 1281, mem_usage:GPU:0 0.9GB, num_seqs 3, 8.833 sec/step, elapsed 0:24:08, exp. remaining 3:18:51, complete 10.82%
att-weights epoch 481, step 164, max_size:classes 41, max_size:data 1307, mem_usage:GPU:0 0.9GB, num_seqs 3, 6.380 sec/step, elapsed 0:24:14, exp. remaining 3:18:18, complete 10.89%
att-weights epoch 481, step 165, max_size:classes 41, max_size:data 1429, mem_usage:GPU:0 0.9GB, num_seqs 2, 3.215 sec/step, elapsed 0:24:17, exp. remaining 3:16:36, complete 11.00%
att-weights epoch 481, step 166, max_size:classes 38, max_size:data 1771, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.815 sec/step, elapsed 0:24:19, exp. remaining 3:14:46, complete 11.10%
att-weights epoch 481, step 167, max_size:classes 42, max_size:data 1241, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.192 sec/step, elapsed 0:24:21, exp. remaining 3:13:41, complete 11.17%
att-weights epoch 481, step 168, max_size:classes 43, max_size:data 1530, mem_usage:GPU:0 0.9GB, num_seqs 2, 3.128 sec/step, elapsed 0:24:24, exp. remaining 3:12:45, complete 11.24%
att-weights epoch 481, step 169, max_size:classes 42, max_size:data 1191, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.059 sec/step, elapsed 0:24:27, exp. remaining 3:11:00, complete 11.35%
att-weights epoch 481, step 170, max_size:classes 43, max_size:data 1350, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.765 sec/step, elapsed 0:24:28, exp. remaining 3:09:55, complete 11.42%
att-weights epoch 481, step 171, max_size:classes 38, max_size:data 1203, mem_usage:GPU:0 0.9GB, num_seqs 3, 3.123 sec/step, elapsed 0:24:31, exp. remaining 3:09:01, complete 11.49%
att-weights epoch 481, step 172, max_size:classes 41, max_size:data 1228, mem_usage:GPU:0 0.9GB, num_seqs 3, 3.005 sec/step, elapsed 0:24:34, exp. remaining 3:08:06, complete 11.56%
att-weights epoch 481, step 173, max_size:classes 42, max_size:data 1591, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.266 sec/step, elapsed 0:24:36, exp. remaining 3:06:21, complete 11.66%
att-weights epoch 481, step 174, max_size:classes 43, max_size:data 1327, mem_usage:GPU:0 0.9GB, num_seqs 3, 3.191 sec/step, elapsed 0:24:39, exp. remaining 3:05:30, complete 11.73%
att-weights epoch 481, step 175, max_size:classes 41, max_size:data 1307, mem_usage:GPU:0 0.9GB, num_seqs 3, 3.092 sec/step, elapsed 0:24:42, exp. remaining 3:04:39, complete 11.80%
att-weights epoch 481, step 176, max_size:classes 37, max_size:data 1059, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.818 sec/step, elapsed 0:24:45, exp. remaining 3:03:09, complete 11.91%
att-weights epoch 481, step 177, max_size:classes 41, max_size:data 1392, mem_usage:GPU:0 0.9GB, num_seqs 2, 9.640 sec/step, elapsed 0:24:54, exp. remaining 3:02:31, complete 12.01%
att-weights epoch 481, step 178, max_size:classes 37, max_size:data 1081, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.449 sec/step, elapsed 0:24:56, exp. remaining 3:01:29, complete 12.08%
att-weights epoch 481, step 179, max_size:classes 38, max_size:data 1461, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.096 sec/step, elapsed 0:24:58, exp. remaining 2:59:58, complete 12.19%
att-weights epoch 481, step 180, max_size:classes 44, max_size:data 1108, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.167 sec/step, elapsed 0:24:59, exp. remaining 2:58:22, complete 12.29%
att-weights epoch 481, step 181, max_size:classes 41, max_size:data 1351, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.409 sec/step, elapsed 0:25:01, exp. remaining 2:56:48, complete 12.40%
att-weights epoch 481, step 182, max_size:classes 38, max_size:data 1161, mem_usage:GPU:0 0.9GB, num_seqs 3, 5.322 sec/step, elapsed 0:25:06, exp. remaining 2:55:44, complete 12.50%
att-weights epoch 481, step 183, max_size:classes 41, max_size:data 1526, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.451 sec/step, elapsed 0:25:07, exp. remaining 2:54:47, complete 12.57%
att-weights epoch 481, step 184, max_size:classes 40, max_size:data 1952, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.170 sec/step, elapsed 0:25:09, exp. remaining 2:53:16, complete 12.67%
att-weights epoch 481, step 185, max_size:classes 42, max_size:data 1272, mem_usage:GPU:0 0.9GB, num_seqs 3, 4.176 sec/step, elapsed 0:25:13, exp. remaining 2:52:07, complete 12.78%
att-weights epoch 481, step 186, max_size:classes 34, max_size:data 1238, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.939 sec/step, elapsed 0:25:15, exp. remaining 2:51:16, complete 12.85%
att-weights epoch 481, step 187, max_size:classes 47, max_size:data 1351, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.170 sec/step, elapsed 0:25:16, exp. remaining 2:49:49, complete 12.95%
att-weights epoch 481, step 188, max_size:classes 38, max_size:data 1263, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.548 sec/step, elapsed 0:25:18, exp. remaining 2:48:01, complete 13.09%
att-weights epoch 481, step 189, max_size:classes 38, max_size:data 1061, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.158 sec/step, elapsed 0:25:21, exp. remaining 2:47:13, complete 13.16%
att-weights epoch 481, step 190, max_size:classes 37, max_size:data 1254, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.681 sec/step, elapsed 0:25:22, exp. remaining 2:45:53, complete 13.27%
att-weights epoch 481, step 191, max_size:classes 41, max_size:data 1299, mem_usage:GPU:0 0.9GB, num_seqs 3, 4.549 sec/step, elapsed 0:25:27, exp. remaining 2:44:53, complete 13.37%
att-weights epoch 481, step 192, max_size:classes 38, max_size:data 1447, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.239 sec/step, elapsed 0:25:28, exp. remaining 2:43:32, complete 13.48%
att-weights epoch 481, step 193, max_size:classes 37, max_size:data 1163, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.978 sec/step, elapsed 0:25:30, exp. remaining 2:42:17, complete 13.58%
att-weights epoch 481, step 194, max_size:classes 43, max_size:data 1024, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.537 sec/step, elapsed 0:25:31, exp. remaining 2:41:00, complete 13.69%
att-weights epoch 481, step 195, max_size:classes 33, max_size:data 1395, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.988 sec/step, elapsed 0:25:32, exp. remaining 2:39:42, complete 13.79%
att-weights epoch 481, step 196, max_size:classes 37, max_size:data 1269, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.808 sec/step, elapsed 0:25:35, exp. remaining 2:38:08, complete 13.93%
att-weights epoch 481, step 197, max_size:classes 35, max_size:data 912, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.111 sec/step, elapsed 0:25:37, exp. remaining 2:37:26, complete 14.00%
att-weights epoch 481, step 198, max_size:classes 35, max_size:data 1369, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.249 sec/step, elapsed 0:25:39, exp. remaining 2:36:12, complete 14.11%
att-weights epoch 481, step 199, max_size:classes 33, max_size:data 972, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.370 sec/step, elapsed 0:25:40, exp. remaining 2:34:33, complete 14.25%
att-weights epoch 481, step 200, max_size:classes 36, max_size:data 1295, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.084 sec/step, elapsed 0:25:42, exp. remaining 2:33:26, complete 14.35%
att-weights epoch 481, step 201, max_size:classes 37, max_size:data 1160, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.998 sec/step, elapsed 0:25:45, exp. remaining 2:32:26, complete 14.46%
att-weights epoch 481, step 202, max_size:classes 35, max_size:data 1027, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.713 sec/step, elapsed 0:25:47, exp. remaining 2:31:45, complete 14.53%
att-weights epoch 481, step 203, max_size:classes 36, max_size:data 1230, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.342 sec/step, elapsed 0:25:49, exp. remaining 2:30:42, complete 14.63%
att-weights epoch 481, step 204, max_size:classes 36, max_size:data 1042, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.139 sec/step, elapsed 0:25:51, exp. remaining 2:30:04, complete 14.70%
att-weights epoch 481, step 205, max_size:classes 37, max_size:data 968, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.206 sec/step, elapsed 0:25:54, exp. remaining 2:29:02, complete 14.80%
att-weights epoch 481, step 206, max_size:classes 33, max_size:data 1366, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.282 sec/step, elapsed 0:25:55, exp. remaining 2:27:56, complete 14.91%
att-weights epoch 481, step 207, max_size:classes 34, max_size:data 1093, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.998 sec/step, elapsed 0:25:58, exp. remaining 2:26:36, complete 15.05%
att-weights epoch 481, step 208, max_size:classes 36, max_size:data 994, mem_usage:GPU:0 0.9GB, num_seqs 4, 9.257 sec/step, elapsed 0:26:07, exp. remaining 2:26:16, complete 15.15%
att-weights epoch 481, step 209, max_size:classes 40, max_size:data 1010, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.273 sec/step, elapsed 0:26:09, exp. remaining 2:25:18, complete 15.26%
att-weights epoch 481, step 210, max_size:classes 38, max_size:data 1276, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.503 sec/step, elapsed 0:26:12, exp. remaining 2:24:22, complete 15.36%
att-weights epoch 481, step 211, max_size:classes 35, max_size:data 1355, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.161 sec/step, elapsed 0:26:13, exp. remaining 2:23:19, complete 15.47%
att-weights epoch 481, step 212, max_size:classes 33, max_size:data 1069, mem_usage:GPU:0 0.9GB, num_seqs 3, 3.539 sec/step, elapsed 0:26:17, exp. remaining 2:22:29, complete 15.57%
att-weights epoch 481, step 213, max_size:classes 40, max_size:data 1448, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.386 sec/step, elapsed 0:26:18, exp. remaining 2:21:29, complete 15.68%
att-weights epoch 481, step 214, max_size:classes 38, max_size:data 1323, mem_usage:GPU:0 0.9GB, num_seqs 3, 4.477 sec/step, elapsed 0:26:22, exp. remaining 2:20:46, complete 15.78%
att-weights epoch 481, step 215, max_size:classes 34, max_size:data 1177, mem_usage:GPU:0 0.9GB, num_seqs 3, 3.123 sec/step, elapsed 0:26:26, exp. remaining 2:19:57, complete 15.89%
att-weights epoch 481, step 216, max_size:classes 37, max_size:data 982, mem_usage:GPU:0 0.9GB, num_seqs 4, 6.841 sec/step, elapsed 0:26:32, exp. remaining 2:19:27, complete 15.99%
att-weights epoch 481, step 217, max_size:classes 34, max_size:data 1000, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.533 sec/step, elapsed 0:26:35, exp. remaining 2:18:36, complete 16.10%
att-weights epoch 481, step 218, max_size:classes 34, max_size:data 1072, mem_usage:GPU:0 0.9GB, num_seqs 3, 4.402 sec/step, elapsed 0:26:39, exp. remaining 2:17:54, complete 16.20%
att-weights epoch 481, step 219, max_size:classes 36, max_size:data 1197, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.628 sec/step, elapsed 0:26:41, exp. remaining 2:16:59, complete 16.31%
att-weights epoch 481, step 220, max_size:classes 33, max_size:data 1013, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.848 sec/step, elapsed 0:26:43, exp. remaining 2:15:45, complete 16.45%
att-weights epoch 481, step 221, max_size:classes 36, max_size:data 930, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.697 sec/step, elapsed 0:26:44, exp. remaining 2:14:52, complete 16.55%
att-weights epoch 481, step 222, max_size:classes 35, max_size:data 1123, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.697 sec/step, elapsed 0:26:47, exp. remaining 2:13:44, complete 16.69%
att-weights epoch 481, step 223, max_size:classes 33, max_size:data 1042, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.694 sec/step, elapsed 0:26:50, exp. remaining 2:12:58, complete 16.79%
att-weights epoch 481, step 224, max_size:classes 33, max_size:data 1186, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.414 sec/step, elapsed 0:26:52, exp. remaining 2:12:10, complete 16.90%
att-weights epoch 481, step 225, max_size:classes 35, max_size:data 1170, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.516 sec/step, elapsed 0:26:54, exp. remaining 2:11:19, complete 17.00%
att-weights epoch 481, step 226, max_size:classes 37, max_size:data 1025, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.603 sec/step, elapsed 0:26:55, exp. remaining 2:10:28, complete 17.11%
att-weights epoch 481, step 227, max_size:classes 32, max_size:data 1013, mem_usage:GPU:0 0.9GB, num_seqs 3, 4.144 sec/step, elapsed 0:27:00, exp. remaining 2:09:51, complete 17.21%
att-weights epoch 481, step 228, max_size:classes 35, max_size:data 1177, mem_usage:GPU:0 0.9GB, num_seqs 3, 3.018 sec/step, elapsed 0:27:03, exp. remaining 2:08:49, complete 17.35%
att-weights epoch 481, step 229, max_size:classes 35, max_size:data 955, mem_usage:GPU:0 0.9GB, num_seqs 4, 3.215 sec/step, elapsed 0:27:06, exp. remaining 2:08:09, complete 17.46%
att-weights epoch 481, step 230, max_size:classes 31, max_size:data 1265, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.775 sec/step, elapsed 0:27:09, exp. remaining 2:07:26, complete 17.56%
att-weights epoch 481, step 231, max_size:classes 37, max_size:data 981, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.111 sec/step, elapsed 0:27:11, exp. remaining 2:06:23, complete 17.70%
att-weights epoch 481, step 232, max_size:classes 35, max_size:data 1093, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.678 sec/step, elapsed 0:27:12, exp. remaining 2:05:36, complete 17.81%
att-weights epoch 481, step 233, max_size:classes 33, max_size:data 1208, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.727 sec/step, elapsed 0:27:14, exp. remaining 2:05:08, complete 17.88%
att-weights epoch 481, step 234, max_size:classes 37, max_size:data 1028, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.545 sec/step, elapsed 0:27:16, exp. remaining 2:04:05, complete 18.02%
att-weights epoch 481, step 235, max_size:classes 35, max_size:data 1012, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.430 sec/step, elapsed 0:27:17, exp. remaining 2:03:19, complete 18.12%
att-weights epoch 481, step 236, max_size:classes 31, max_size:data 1283, mem_usage:GPU:0 0.9GB, num_seqs 3, 4.218 sec/step, elapsed 0:27:21, exp. remaining 2:02:28, complete 18.26%
att-weights epoch 481, step 237, max_size:classes 34, max_size:data 963, mem_usage:GPU:0 0.9GB, num_seqs 4, 3.577 sec/step, elapsed 0:27:25, exp. remaining 2:01:53, complete 18.37%
att-weights epoch 481, step 238, max_size:classes 33, max_size:data 1087, mem_usage:GPU:0 0.9GB, num_seqs 3, 7.278 sec/step, elapsed 0:27:32, exp. remaining 2:01:34, complete 18.47%
att-weights epoch 481, step 239, max_size:classes 38, max_size:data 1071, mem_usage:GPU:0 0.9GB, num_seqs 3, 6.953 sec/step, elapsed 0:27:39, exp. remaining 2:01:14, complete 18.58%
att-weights epoch 481, step 240, max_size:classes 34, max_size:data 931, mem_usage:GPU:0 0.9GB, num_seqs 4, 3.983 sec/step, elapsed 0:27:43, exp. remaining 2:00:42, complete 18.68%
att-weights epoch 481, step 241, max_size:classes 29, max_size:data 1052, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.862 sec/step, elapsed 0:27:46, exp. remaining 1:59:48, complete 18.82%
att-weights epoch 481, step 242, max_size:classes 34, max_size:data 1371, mem_usage:GPU:0 0.9GB, num_seqs 2, 4.477 sec/step, elapsed 0:27:50, exp. remaining 1:59:18, complete 18.92%
att-weights epoch 481, step 243, max_size:classes 36, max_size:data 982, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.919 sec/step, elapsed 0:27:52, exp. remaining 1:58:38, complete 19.03%
att-weights epoch 481, step 244, max_size:classes 34, max_size:data 1332, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.678 sec/step, elapsed 0:27:54, exp. remaining 1:57:56, complete 19.13%
att-weights epoch 481, step 245, max_size:classes 31, max_size:data 969, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.789 sec/step, elapsed 0:27:57, exp. remaining 1:57:21, complete 19.24%
att-weights epoch 481, step 246, max_size:classes 33, max_size:data 1179, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.581 sec/step, elapsed 0:27:58, exp. remaining 1:56:40, complete 19.34%
att-weights epoch 481, step 247, max_size:classes 32, max_size:data 1085, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.532 sec/step, elapsed 0:28:00, exp. remaining 1:56:00, complete 19.45%
att-weights epoch 481, step 248, max_size:classes 28, max_size:data 1044, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.424 sec/step, elapsed 0:28:01, exp. remaining 1:55:04, complete 19.59%
att-weights epoch 481, step 249, max_size:classes 34, max_size:data 1326, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.359 sec/step, elapsed 0:28:03, exp. remaining 1:54:09, complete 19.73%
att-weights epoch 481, step 250, max_size:classes 34, max_size:data 879, mem_usage:GPU:0 0.9GB, num_seqs 4, 3.133 sec/step, elapsed 0:28:06, exp. remaining 1:53:36, complete 19.83%
att-weights epoch 481, step 251, max_size:classes 32, max_size:data 1032, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.354 sec/step, elapsed 0:28:07, exp. remaining 1:52:57, complete 19.94%
att-weights epoch 481, step 252, max_size:classes 33, max_size:data 1070, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.377 sec/step, elapsed 0:28:09, exp. remaining 1:52:18, complete 20.04%
att-weights epoch 481, step 253, max_size:classes 30, max_size:data 958, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.251 sec/step, elapsed 0:28:10, exp. remaining 1:51:39, complete 20.15%
att-weights epoch 481, step 254, max_size:classes 29, max_size:data 1214, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.031 sec/step, elapsed 0:28:12, exp. remaining 1:51:04, complete 20.25%
att-weights epoch 481, step 255, max_size:classes 33, max_size:data 1108, mem_usage:GPU:0 0.9GB, num_seqs 3, 3.656 sec/step, elapsed 0:28:16, exp. remaining 1:50:35, complete 20.36%
att-weights epoch 481, step 256, max_size:classes 30, max_size:data 1229, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.624 sec/step, elapsed 0:28:17, exp. remaining 1:49:59, complete 20.46%
att-weights epoch 481, step 257, max_size:classes 34, max_size:data 884, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.615 sec/step, elapsed 0:28:19, exp. remaining 1:49:09, complete 20.60%
att-weights epoch 481, step 258, max_size:classes 34, max_size:data 907, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.904 sec/step, elapsed 0:28:21, exp. remaining 1:48:21, complete 20.74%
att-weights epoch 481, step 259, max_size:classes 32, max_size:data 1249, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.393 sec/step, elapsed 0:28:22, exp. remaining 1:47:45, complete 20.84%
att-weights epoch 481, step 260, max_size:classes 31, max_size:data 1104, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.504 sec/step, elapsed 0:28:24, exp. remaining 1:47:09, complete 20.95%
att-weights epoch 481, step 261, max_size:classes 33, max_size:data 1026, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.409 sec/step, elapsed 0:28:25, exp. remaining 1:46:34, complete 21.05%
att-weights epoch 481, step 262, max_size:classes 39, max_size:data 1026, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.903 sec/step, elapsed 0:28:28, exp. remaining 1:45:52, complete 21.19%
att-weights epoch 481, step 263, max_size:classes 34, max_size:data 1010, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.401 sec/step, elapsed 0:28:29, exp. remaining 1:45:17, complete 21.30%
att-weights epoch 481, step 264, max_size:classes 30, max_size:data 1215, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.669 sec/step, elapsed 0:28:31, exp. remaining 1:44:44, complete 21.40%
att-weights epoch 481, step 265, max_size:classes 30, max_size:data 1177, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.293 sec/step, elapsed 0:28:32, exp. remaining 1:43:57, complete 21.54%
att-weights epoch 481, step 266, max_size:classes 36, max_size:data 848, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.584 sec/step, elapsed 0:28:34, exp. remaining 1:43:12, complete 21.68%
att-weights epoch 481, step 267, max_size:classes 30, max_size:data 906, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.120 sec/step, elapsed 0:28:36, exp. remaining 1:42:41, complete 21.79%
att-weights epoch 481, step 268, max_size:classes 31, max_size:data 1044, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.510 sec/step, elapsed 0:28:37, exp. remaining 1:42:09, complete 21.89%
att-weights epoch 481, step 269, max_size:classes 27, max_size:data 1224, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.870 sec/step, elapsed 0:28:39, exp. remaining 1:41:13, complete 22.07%
att-weights epoch 481, step 270, max_size:classes 29, max_size:data 1077, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.480 sec/step, elapsed 0:28:41, exp. remaining 1:40:29, complete 22.21%
att-weights epoch 481, step 271, max_size:classes 31, max_size:data 868, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.609 sec/step, elapsed 0:28:42, exp. remaining 1:39:59, complete 22.31%
att-weights epoch 481, step 272, max_size:classes 32, max_size:data 1176, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.457 sec/step, elapsed 0:28:44, exp. remaining 1:39:16, complete 22.45%
att-weights epoch 481, step 273, max_size:classes 34, max_size:data 1218, mem_usage:GPU:0 0.9GB, num_seqs 3, 3.091 sec/step, elapsed 0:28:47, exp. remaining 1:38:51, complete 22.56%
att-weights epoch 481, step 274, max_size:classes 32, max_size:data 845, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.761 sec/step, elapsed 0:28:49, exp. remaining 1:38:10, complete 22.70%
att-weights epoch 481, step 275, max_size:classes 32, max_size:data 865, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.470 sec/step, elapsed 0:28:50, exp. remaining 1:37:40, complete 22.80%
att-weights epoch 481, step 276, max_size:classes 31, max_size:data 1180, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.587 sec/step, elapsed 0:28:52, exp. remaining 1:37:10, complete 22.91%
att-weights epoch 481, step 277, max_size:classes 31, max_size:data 1005, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.391 sec/step, elapsed 0:28:53, exp. remaining 1:36:29, complete 23.04%
att-weights epoch 481, step 278, max_size:classes 29, max_size:data 793, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.787 sec/step, elapsed 0:28:55, exp. remaining 1:35:50, complete 23.18%
att-weights epoch 481, step 279, max_size:classes 29, max_size:data 994, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.383 sec/step, elapsed 0:28:56, exp. remaining 1:35:20, complete 23.29%
att-weights epoch 481, step 280, max_size:classes 29, max_size:data 1115, mem_usage:GPU:0 0.9GB, num_seqs 3, 5.153 sec/step, elapsed 0:29:02, exp. remaining 1:35:04, complete 23.39%
att-weights epoch 481, step 281, max_size:classes 29, max_size:data 949, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.415 sec/step, elapsed 0:29:03, exp. remaining 1:34:35, complete 23.50%
att-weights epoch 481, step 282, max_size:classes 32, max_size:data 1248, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.378 sec/step, elapsed 0:29:04, exp. remaining 1:33:56, complete 23.64%
att-weights epoch 481, step 283, max_size:classes 28, max_size:data 858, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.378 sec/step, elapsed 0:29:06, exp. remaining 1:33:17, complete 23.78%
att-weights epoch 481, step 284, max_size:classes 31, max_size:data 1176, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.771 sec/step, elapsed 0:29:07, exp. remaining 1:32:40, complete 23.92%
att-weights epoch 481, step 285, max_size:classes 30, max_size:data 1074, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.115 sec/step, elapsed 0:29:09, exp. remaining 1:32:11, complete 24.02%
att-weights epoch 481, step 286, max_size:classes 28, max_size:data 829, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.466 sec/step, elapsed 0:29:10, exp. remaining 1:31:44, complete 24.13%
att-weights epoch 481, step 287, max_size:classes 29, max_size:data 788, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.935 sec/step, elapsed 0:29:12, exp. remaining 1:31:19, complete 24.23%
att-weights epoch 481, step 288, max_size:classes 34, max_size:data 993, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.372 sec/step, elapsed 0:29:13, exp. remaining 1:30:42, complete 24.37%
att-weights epoch 481, step 289, max_size:classes 30, max_size:data 1088, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.555 sec/step, elapsed 0:29:15, exp. remaining 1:30:06, complete 24.51%
att-weights epoch 481, step 290, max_size:classes 28, max_size:data 1039, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.424 sec/step, elapsed 0:29:16, exp. remaining 1:29:30, complete 24.65%
att-weights epoch 481, step 291, max_size:classes 29, max_size:data 923, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.849 sec/step, elapsed 0:29:18, exp. remaining 1:28:55, complete 24.79%
att-weights epoch 481, step 292, max_size:classes 30, max_size:data 881, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.929 sec/step, elapsed 0:29:20, exp. remaining 1:28:21, complete 24.93%
att-weights epoch 481, step 293, max_size:classes 32, max_size:data 824, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.489 sec/step, elapsed 0:29:22, exp. remaining 1:27:46, complete 25.07%
att-weights epoch 481, step 294, max_size:classes 30, max_size:data 1012, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.056 sec/step, elapsed 0:29:23, exp. remaining 1:27:10, complete 25.21%
att-weights epoch 481, step 295, max_size:classes 29, max_size:data 956, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.047 sec/step, elapsed 0:29:24, exp. remaining 1:26:45, complete 25.31%
att-weights epoch 481, step 296, max_size:classes 29, max_size:data 1138, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.652 sec/step, elapsed 0:29:25, exp. remaining 1:26:21, complete 25.42%
att-weights epoch 481, step 297, max_size:classes 28, max_size:data 908, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.372 sec/step, elapsed 0:29:27, exp. remaining 1:25:56, complete 25.52%
att-weights epoch 481, step 298, max_size:classes 29, max_size:data 836, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.507 sec/step, elapsed 0:29:28, exp. remaining 1:25:13, complete 25.70%
att-weights epoch 481, step 299, max_size:classes 29, max_size:data 886, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.638 sec/step, elapsed 0:29:30, exp. remaining 1:24:50, complete 25.80%
att-weights epoch 481, step 300, max_size:classes 32, max_size:data 991, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.618 sec/step, elapsed 0:29:31, exp. remaining 1:24:18, complete 25.94%
att-weights epoch 481, step 301, max_size:classes 28, max_size:data 916, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.516 sec/step, elapsed 0:29:33, exp. remaining 1:23:46, complete 26.08%
att-weights epoch 481, step 302, max_size:classes 32, max_size:data 835, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.624 sec/step, elapsed 0:29:35, exp. remaining 1:23:14, complete 26.22%
att-weights epoch 481, step 303, max_size:classes 26, max_size:data 885, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.767 sec/step, elapsed 0:29:36, exp. remaining 1:22:43, complete 26.36%
att-weights epoch 481, step 304, max_size:classes 27, max_size:data 918, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.409 sec/step, elapsed 0:29:38, exp. remaining 1:22:11, complete 26.50%
att-weights epoch 481, step 305, max_size:classes 30, max_size:data 1127, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.570 sec/step, elapsed 0:29:39, exp. remaining 1:21:32, complete 26.68%
att-weights epoch 481, step 306, max_size:classes 33, max_size:data 1053, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.485 sec/step, elapsed 0:29:41, exp. remaining 1:21:01, complete 26.82%
att-weights epoch 481, step 307, max_size:classes 28, max_size:data 783, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.957 sec/step, elapsed 0:29:43, exp. remaining 1:20:32, complete 26.96%
att-weights epoch 481, step 308, max_size:classes 30, max_size:data 1033, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.396 sec/step, elapsed 0:29:44, exp. remaining 1:20:02, complete 27.09%
att-weights epoch 481, step 309, max_size:classes 32, max_size:data 909, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.596 sec/step, elapsed 0:29:46, exp. remaining 1:19:32, complete 27.23%
att-weights epoch 481, step 310, max_size:classes 26, max_size:data 812, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.589 sec/step, elapsed 0:29:47, exp. remaining 1:19:03, complete 27.37%
att-weights epoch 481, step 311, max_size:classes 30, max_size:data 960, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.508 sec/step, elapsed 0:29:50, exp. remaining 1:18:36, complete 27.51%
att-weights epoch 481, step 312, max_size:classes 31, max_size:data 895, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.701 sec/step, elapsed 0:29:52, exp. remaining 1:18:08, complete 27.65%
att-weights epoch 481, step 313, max_size:classes 29, max_size:data 973, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.540 sec/step, elapsed 0:29:53, exp. remaining 1:17:31, complete 27.83%
att-weights epoch 481, step 314, max_size:classes 29, max_size:data 709, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.796 sec/step, elapsed 0:29:55, exp. remaining 1:17:04, complete 27.97%
att-weights epoch 481, step 315, max_size:classes 27, max_size:data 898, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.289 sec/step, elapsed 0:29:56, exp. remaining 1:16:35, complete 28.11%
att-weights epoch 481, step 316, max_size:classes 25, max_size:data 801, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.323 sec/step, elapsed 0:29:58, exp. remaining 1:16:07, complete 28.25%
att-weights epoch 481, step 317, max_size:classes 26, max_size:data 957, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.689 sec/step, elapsed 0:29:59, exp. remaining 1:15:48, complete 28.35%
att-weights epoch 481, step 318, max_size:classes 28, max_size:data 834, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.380 sec/step, elapsed 0:30:01, exp. remaining 1:15:28, complete 28.46%
att-weights epoch 481, step 319, max_size:classes 26, max_size:data 909, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.733 sec/step, elapsed 0:30:02, exp. remaining 1:15:01, complete 28.60%
att-weights epoch 481, step 320, max_size:classes 27, max_size:data 913, mem_usage:GPU:0 0.9GB, num_seqs 4, 3.134 sec/step, elapsed 0:30:06, exp. remaining 1:14:38, complete 28.74%
att-weights epoch 481, step 321, max_size:classes 29, max_size:data 942, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.041 sec/step, elapsed 0:30:08, exp. remaining 1:14:13, complete 28.88%
att-weights epoch 481, step 322, max_size:classes 28, max_size:data 764, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.700 sec/step, elapsed 0:30:09, exp. remaining 1:13:55, complete 28.98%
att-weights epoch 481, step 323, max_size:classes 28, max_size:data 952, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.295 sec/step, elapsed 0:30:12, exp. remaining 1:13:30, complete 29.12%
att-weights epoch 481, step 324, max_size:classes 26, max_size:data 942, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.448 sec/step, elapsed 0:30:13, exp. remaining 1:13:11, complete 29.22%
att-weights epoch 481, step 325, max_size:classes 27, max_size:data 845, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.513 sec/step, elapsed 0:30:15, exp. remaining 1:12:45, complete 29.36%
att-weights epoch 481, step 326, max_size:classes 26, max_size:data 636, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.160 sec/step, elapsed 0:30:16, exp. remaining 1:12:19, complete 29.50%
att-weights epoch 481, step 327, max_size:classes 26, max_size:data 1029, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.455 sec/step, elapsed 0:30:17, exp. remaining 1:11:46, complete 29.68%
att-weights epoch 481, step 328, max_size:classes 26, max_size:data 917, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.581 sec/step, elapsed 0:30:19, exp. remaining 1:11:14, complete 29.85%
att-weights epoch 481, step 329, max_size:classes 30, max_size:data 780, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.529 sec/step, elapsed 0:30:20, exp. remaining 1:10:42, complete 30.03%
att-weights epoch 481, step 330, max_size:classes 27, max_size:data 864, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.555 sec/step, elapsed 0:30:22, exp. remaining 1:10:18, complete 30.17%
att-weights epoch 481, step 331, max_size:classes 27, max_size:data 1036, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.381 sec/step, elapsed 0:30:23, exp. remaining 1:09:46, complete 30.34%
att-weights epoch 481, step 332, max_size:classes 26, max_size:data 716, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.346 sec/step, elapsed 0:30:25, exp. remaining 1:09:15, complete 30.52%
att-weights epoch 481, step 333, max_size:classes 24, max_size:data 1075, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.452 sec/step, elapsed 0:30:26, exp. remaining 1:08:51, complete 30.66%
att-weights epoch 481, step 334, max_size:classes 30, max_size:data 876, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.541 sec/step, elapsed 0:30:28, exp. remaining 1:08:27, complete 30.80%
att-weights epoch 481, step 335, max_size:classes 25, max_size:data 887, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.447 sec/step, elapsed 0:30:29, exp. remaining 1:08:04, complete 30.94%
att-weights epoch 481, step 336, max_size:classes 25, max_size:data 741, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.516 sec/step, elapsed 0:30:30, exp. remaining 1:07:41, complete 31.08%
att-weights epoch 481, step 337, max_size:classes 29, max_size:data 725, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.962 sec/step, elapsed 0:30:32, exp. remaining 1:07:19, complete 31.22%
att-weights epoch 481, step 338, max_size:classes 29, max_size:data 785, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.826 sec/step, elapsed 0:30:34, exp. remaining 1:06:56, complete 31.35%
att-weights epoch 481, step 339, max_size:classes 28, max_size:data 956, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.516 sec/step, elapsed 0:30:37, exp. remaining 1:06:29, complete 31.53%
att-weights epoch 481, step 340, max_size:classes 26, max_size:data 758, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.896 sec/step, elapsed 0:30:39, exp. remaining 1:06:08, complete 31.67%
att-weights epoch 481, step 341, max_size:classes 27, max_size:data 785, mem_usage:GPU:0 0.9GB, num_seqs 5, 3.014 sec/step, elapsed 0:30:42, exp. remaining 1:05:49, complete 31.81%
att-weights epoch 481, step 342, max_size:classes 28, max_size:data 896, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.696 sec/step, elapsed 0:30:43, exp. remaining 1:05:27, complete 31.95%
att-weights epoch 481, step 343, max_size:classes 25, max_size:data 995, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.420 sec/step, elapsed 0:30:46, exp. remaining 1:05:07, complete 32.09%
att-weights epoch 481, step 344, max_size:classes 26, max_size:data 922, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.594 sec/step, elapsed 0:30:47, exp. remaining 1:04:39, complete 32.26%
att-weights epoch 481, step 345, max_size:classes 24, max_size:data 885, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.468 sec/step, elapsed 0:30:49, exp. remaining 1:04:12, complete 32.44%
att-weights epoch 481, step 346, max_size:classes 29, max_size:data 850, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.401 sec/step, elapsed 0:30:50, exp. remaining 1:03:50, complete 32.58%
att-weights epoch 481, step 347, max_size:classes 26, max_size:data 896, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.427 sec/step, elapsed 0:30:52, exp. remaining 1:03:29, complete 32.72%
att-weights epoch 481, step 348, max_size:classes 28, max_size:data 724, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.539 sec/step, elapsed 0:30:53, exp. remaining 1:03:08, complete 32.86%
att-weights epoch 481, step 349, max_size:classes 27, max_size:data 858, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.471 sec/step, elapsed 0:30:55, exp. remaining 1:02:47, complete 33.00%
att-weights epoch 481, step 350, max_size:classes 25, max_size:data 922, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.394 sec/step, elapsed 0:30:56, exp. remaining 1:02:26, complete 33.14%
att-weights epoch 481, step 351, max_size:classes 26, max_size:data 687, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.327 sec/step, elapsed 0:30:57, exp. remaining 1:01:59, complete 33.31%
att-weights epoch 481, step 352, max_size:classes 31, max_size:data 863, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.662 sec/step, elapsed 0:30:59, exp. remaining 1:01:34, complete 33.48%
att-weights epoch 481, step 353, max_size:classes 26, max_size:data 669, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.833 sec/step, elapsed 0:31:01, exp. remaining 1:01:14, complete 33.62%
att-weights epoch 481, step 354, max_size:classes 26, max_size:data 688, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.638 sec/step, elapsed 0:31:03, exp. remaining 1:00:49, complete 33.80%
att-weights epoch 481, step 355, max_size:classes 32, max_size:data 887, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.463 sec/step, elapsed 0:31:04, exp. remaining 1:00:23, complete 33.97%
att-weights epoch 481, step 356, max_size:classes 28, max_size:data 941, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.457 sec/step, elapsed 0:31:06, exp. remaining 1:00:04, complete 34.11%
att-weights epoch 481, step 357, max_size:classes 25, max_size:data 811, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.226 sec/step, elapsed 0:31:07, exp. remaining 0:59:38, complete 34.29%
att-weights epoch 481, step 358, max_size:classes 27, max_size:data 928, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.726 sec/step, elapsed 0:31:08, exp. remaining 0:59:19, complete 34.43%
att-weights epoch 481, step 359, max_size:classes 25, max_size:data 987, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.982 sec/step, elapsed 0:31:10, exp. remaining 0:58:56, complete 34.60%
att-weights epoch 481, step 360, max_size:classes 24, max_size:data 697, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.705 sec/step, elapsed 0:31:12, exp. remaining 0:58:26, complete 34.81%
att-weights epoch 481, step 361, max_size:classes 24, max_size:data 795, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.840 sec/step, elapsed 0:31:14, exp. remaining 0:58:03, complete 34.99%
att-weights epoch 481, step 362, max_size:classes 23, max_size:data 912, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.233 sec/step, elapsed 0:31:15, exp. remaining 0:57:44, complete 35.13%
att-weights epoch 481, step 363, max_size:classes 24, max_size:data 721, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.743 sec/step, elapsed 0:31:17, exp. remaining 0:57:21, complete 35.30%
att-weights epoch 481, step 364, max_size:classes 26, max_size:data 769, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.806 sec/step, elapsed 0:31:19, exp. remaining 0:57:03, complete 35.44%
att-weights epoch 481, step 365, max_size:classes 26, max_size:data 824, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.529 sec/step, elapsed 0:31:20, exp. remaining 0:56:45, complete 35.58%
att-weights epoch 481, step 366, max_size:classes 24, max_size:data 738, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.927 sec/step, elapsed 0:31:22, exp. remaining 0:56:23, complete 35.75%
att-weights epoch 481, step 367, max_size:classes 24, max_size:data 827, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.308 sec/step, elapsed 0:31:24, exp. remaining 0:56:04, complete 35.89%
att-weights epoch 481, step 368, max_size:classes 26, max_size:data 777, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.565 sec/step, elapsed 0:31:25, exp. remaining 0:55:42, complete 36.07%
att-weights epoch 481, step 369, max_size:classes 29, max_size:data 665, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.974 sec/step, elapsed 0:31:27, exp. remaining 0:55:20, complete 36.24%
att-weights epoch 481, step 370, max_size:classes 26, max_size:data 708, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.660 sec/step, elapsed 0:31:29, exp. remaining 0:55:03, complete 36.38%
att-weights epoch 481, step 371, max_size:classes 27, max_size:data 878, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.001 sec/step, elapsed 0:31:31, exp. remaining 0:54:42, complete 36.56%
att-weights epoch 481, step 372, max_size:classes 25, max_size:data 660, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.783 sec/step, elapsed 0:31:33, exp. remaining 0:54:20, complete 36.73%
att-weights epoch 481, step 373, max_size:classes 26, max_size:data 771, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.420 sec/step, elapsed 0:31:34, exp. remaining 0:53:58, complete 36.91%
att-weights epoch 481, step 374, max_size:classes 24, max_size:data 915, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.023 sec/step, elapsed 0:31:36, exp. remaining 0:53:37, complete 37.08%
att-weights epoch 481, step 375, max_size:classes 24, max_size:data 601, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.635 sec/step, elapsed 0:31:38, exp. remaining 0:53:21, complete 37.22%
att-weights epoch 481, step 376, max_size:classes 23, max_size:data 842, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.516 sec/step, elapsed 0:31:39, exp. remaining 0:52:55, complete 37.43%
att-weights epoch 481, step 377, max_size:classes 24, max_size:data 745, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.868 sec/step, elapsed 0:31:41, exp. remaining 0:52:39, complete 37.57%
att-weights epoch 481, step 378, max_size:classes 24, max_size:data 798, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.081 sec/step, elapsed 0:31:43, exp. remaining 0:52:19, complete 37.74%
att-weights epoch 481, step 379, max_size:classes 25, max_size:data 911, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.660 sec/step, elapsed 0:31:45, exp. remaining 0:51:59, complete 37.92%
att-weights epoch 481, step 380, max_size:classes 23, max_size:data 760, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.584 sec/step, elapsed 0:31:46, exp. remaining 0:51:34, complete 38.13%
att-weights epoch 481, step 381, max_size:classes 24, max_size:data 762, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.875 sec/step, elapsed 0:31:48, exp. remaining 0:51:14, complete 38.30%
att-weights epoch 481, step 382, max_size:classes 24, max_size:data 711, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.635 sec/step, elapsed 0:31:50, exp. remaining 0:50:54, complete 38.48%
att-weights epoch 481, step 383, max_size:classes 28, max_size:data 635, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.617 sec/step, elapsed 0:31:51, exp. remaining 0:50:34, complete 38.65%
att-weights epoch 481, step 384, max_size:classes 26, max_size:data 849, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.615 sec/step, elapsed 0:31:53, exp. remaining 0:50:14, complete 38.83%
att-weights epoch 481, step 385, max_size:classes 24, max_size:data 628, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.200 sec/step, elapsed 0:31:55, exp. remaining 0:50:00, complete 38.97%
att-weights epoch 481, step 386, max_size:classes 23, max_size:data 819, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.422 sec/step, elapsed 0:31:57, exp. remaining 0:49:40, complete 39.14%
att-weights epoch 481, step 387, max_size:classes 24, max_size:data 701, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.998 sec/step, elapsed 0:31:59, exp. remaining 0:49:17, complete 39.35%
att-weights epoch 481, step 388, max_size:classes 23, max_size:data 681, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.730 sec/step, elapsed 0:32:00, exp. remaining 0:49:03, complete 39.49%
att-weights epoch 481, step 389, max_size:classes 24, max_size:data 577, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.546 sec/step, elapsed 0:32:02, exp. remaining 0:48:40, complete 39.70%
att-weights epoch 481, step 390, max_size:classes 24, max_size:data 765, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.612 sec/step, elapsed 0:32:04, exp. remaining 0:48:21, complete 39.87%
att-weights epoch 481, step 391, max_size:classes 24, max_size:data 746, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.579 sec/step, elapsed 0:32:05, exp. remaining 0:48:02, complete 40.05%
att-weights epoch 481, step 392, max_size:classes 25, max_size:data 697, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.667 sec/step, elapsed 0:32:08, exp. remaining 0:47:45, complete 40.22%
att-weights epoch 481, step 393, max_size:classes 26, max_size:data 676, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.306 sec/step, elapsed 0:32:10, exp. remaining 0:47:32, complete 40.36%
att-weights epoch 481, step 394, max_size:classes 29, max_size:data 844, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.904 sec/step, elapsed 0:32:12, exp. remaining 0:47:10, complete 40.57%
att-weights epoch 481, step 395, max_size:classes 21, max_size:data 754, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.750 sec/step, elapsed 0:32:14, exp. remaining 0:46:52, complete 40.75%
att-weights epoch 481, step 396, max_size:classes 24, max_size:data 626, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.918 sec/step, elapsed 0:32:16, exp. remaining 0:46:31, complete 40.96%
att-weights epoch 481, step 397, max_size:classes 22, max_size:data 827, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.271 sec/step, elapsed 0:32:17, exp. remaining 0:46:13, complete 41.13%
att-weights epoch 481, step 398, max_size:classes 24, max_size:data 627, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.898 sec/step, elapsed 0:32:19, exp. remaining 0:45:55, complete 41.31%
att-weights epoch 481, step 399, max_size:classes 27, max_size:data 786, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.372 sec/step, elapsed 0:32:21, exp. remaining 0:45:43, complete 41.45%
att-weights epoch 481, step 400, max_size:classes 23, max_size:data 796, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.825 sec/step, elapsed 0:32:23, exp. remaining 0:45:26, complete 41.62%
att-weights epoch 481, step 401, max_size:classes 22, max_size:data 763, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.714 sec/step, elapsed 0:32:25, exp. remaining 0:45:05, complete 41.83%
att-weights epoch 481, step 402, max_size:classes 25, max_size:data 861, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.408 sec/step, elapsed 0:32:26, exp. remaining 0:44:47, complete 42.00%
att-weights epoch 481, step 403, max_size:classes 22, max_size:data 664, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.759 sec/step, elapsed 0:32:28, exp. remaining 0:44:31, complete 42.18%
att-weights epoch 481, step 404, max_size:classes 24, max_size:data 754, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.806 sec/step, elapsed 0:32:30, exp. remaining 0:44:18, complete 42.32%
att-weights epoch 481, step 405, max_size:classes 26, max_size:data 627, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.977 sec/step, elapsed 0:32:32, exp. remaining 0:43:58, complete 42.53%
att-weights epoch 481, step 406, max_size:classes 23, max_size:data 706, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.025 sec/step, elapsed 0:32:34, exp. remaining 0:43:42, complete 42.70%
att-weights epoch 481, step 407, max_size:classes 22, max_size:data 750, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.706 sec/step, elapsed 0:32:35, exp. remaining 0:43:25, complete 42.88%
att-weights epoch 481, step 408, max_size:classes 25, max_size:data 809, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.505 sec/step, elapsed 0:32:37, exp. remaining 0:43:09, complete 43.05%
att-weights epoch 481, step 409, max_size:classes 20, max_size:data 705, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.484 sec/step, elapsed 0:32:38, exp. remaining 0:42:49, complete 43.26%
att-weights epoch 481, step 410, max_size:classes 21, max_size:data 657, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.602 sec/step, elapsed 0:32:40, exp. remaining 0:42:36, complete 43.40%
att-weights epoch 481, step 411, max_size:classes 24, max_size:data 759, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.862 sec/step, elapsed 0:32:42, exp. remaining 0:42:21, complete 43.58%
att-weights epoch 481, step 412, max_size:classes 22, max_size:data 767, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.877 sec/step, elapsed 0:32:44, exp. remaining 0:42:09, complete 43.72%
att-weights epoch 481, step 413, max_size:classes 22, max_size:data 864, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.446 sec/step, elapsed 0:32:45, exp. remaining 0:41:49, complete 43.92%
att-weights epoch 481, step 414, max_size:classes 23, max_size:data 655, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.748 sec/step, elapsed 0:32:47, exp. remaining 0:41:30, complete 44.13%
att-weights epoch 481, step 415, max_size:classes 21, max_size:data 776, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.908 sec/step, elapsed 0:32:49, exp. remaining 0:41:15, complete 44.31%
att-weights epoch 481, step 416, max_size:classes 23, max_size:data 773, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.613 sec/step, elapsed 0:32:51, exp. remaining 0:40:59, complete 44.48%
att-weights epoch 481, step 417, max_size:classes 22, max_size:data 723, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.718 sec/step, elapsed 0:32:52, exp. remaining 0:40:41, complete 44.69%
att-weights epoch 481, step 418, max_size:classes 25, max_size:data 639, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.389 sec/step, elapsed 0:32:55, exp. remaining 0:40:27, complete 44.87%
att-weights epoch 481, step 419, max_size:classes 23, max_size:data 946, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.853 sec/step, elapsed 0:32:57, exp. remaining 0:40:12, complete 45.04%
att-weights epoch 481, step 420, max_size:classes 26, max_size:data 768, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.601 sec/step, elapsed 0:32:58, exp. remaining 0:39:53, complete 45.25%
att-weights epoch 481, step 421, max_size:classes 21, max_size:data 815, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.529 sec/step, elapsed 0:33:00, exp. remaining 0:39:38, complete 45.43%
att-weights epoch 481, step 422, max_size:classes 23, max_size:data 623, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.712 sec/step, elapsed 0:33:01, exp. remaining 0:39:24, complete 45.60%
att-weights epoch 481, step 423, max_size:classes 22, max_size:data 636, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.618 sec/step, elapsed 0:33:04, exp. remaining 0:39:10, complete 45.78%
att-weights epoch 481, step 424, max_size:classes 20, max_size:data 630, mem_usage:GPU:0 0.9GB, num_seqs 5, 3.834 sec/step, elapsed 0:33:08, exp. remaining 0:38:58, complete 45.95%
att-weights epoch 481, step 425, max_size:classes 21, max_size:data 722, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.656 sec/step, elapsed 0:33:09, exp. remaining 0:38:44, complete 46.12%
att-weights epoch 481, step 426, max_size:classes 21, max_size:data 647, mem_usage:GPU:0 0.9GB, num_seqs 6, 12.835 sec/step, elapsed 0:33:22, exp. remaining 0:38:39, complete 46.33%
att-weights epoch 481, step 427, max_size:classes 22, max_size:data 707, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.854 sec/step, elapsed 0:33:24, exp. remaining 0:38:25, complete 46.51%
att-weights epoch 481, step 428, max_size:classes 21, max_size:data 735, mem_usage:GPU:0 0.9GB, num_seqs 5, 3.561 sec/step, elapsed 0:33:28, exp. remaining 0:38:07, complete 46.75%
att-weights epoch 481, step 429, max_size:classes 24, max_size:data 598, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.537 sec/step, elapsed 0:33:29, exp. remaining 0:37:52, complete 46.93%
att-weights epoch 481, step 430, max_size:classes 20, max_size:data 770, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.874 sec/step, elapsed 0:33:31, exp. remaining 0:37:39, complete 47.10%
att-weights epoch 481, step 431, max_size:classes 23, max_size:data 696, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.756 sec/step, elapsed 0:33:33, exp. remaining 0:37:31, complete 47.21%
att-weights epoch 481, step 432, max_size:classes 19, max_size:data 708, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.776 sec/step, elapsed 0:33:35, exp. remaining 0:37:14, complete 47.42%
att-weights epoch 481, step 433, max_size:classes 20, max_size:data 652, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.901 sec/step, elapsed 0:33:37, exp. remaining 0:37:01, complete 47.59%
att-weights epoch 481, step 434, max_size:classes 21, max_size:data 700, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.941 sec/step, elapsed 0:33:39, exp. remaining 0:36:47, complete 47.77%
att-weights epoch 481, step 435, max_size:classes 22, max_size:data 621, mem_usage:GPU:0 0.9GB, num_seqs 6, 3.301 sec/step, elapsed 0:33:42, exp. remaining 0:36:29, complete 48.01%
att-weights epoch 481, step 436, max_size:classes 21, max_size:data 776, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.875 sec/step, elapsed 0:33:44, exp. remaining 0:36:10, complete 48.25%
att-weights epoch 481, step 437, max_size:classes 20, max_size:data 547, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.373 sec/step, elapsed 0:33:46, exp. remaining 0:35:55, complete 48.46%
att-weights epoch 481, step 438, max_size:classes 21, max_size:data 777, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.890 sec/step, elapsed 0:33:48, exp. remaining 0:35:45, complete 48.60%
att-weights epoch 481, step 439, max_size:classes 21, max_size:data 653, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.622 sec/step, elapsed 0:33:50, exp. remaining 0:35:28, complete 48.81%
att-weights epoch 481, step 440, max_size:classes 22, max_size:data 1093, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.565 sec/step, elapsed 0:33:51, exp. remaining 0:35:12, complete 49.02%
att-weights epoch 481, step 441, max_size:classes 20, max_size:data 522, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.862 sec/step, elapsed 0:33:53, exp. remaining 0:34:59, complete 49.20%
att-weights epoch 481, step 442, max_size:classes 20, max_size:data 606, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.499 sec/step, elapsed 0:33:55, exp. remaining 0:34:43, complete 49.41%
att-weights epoch 481, step 443, max_size:classes 22, max_size:data 751, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.855 sec/step, elapsed 0:33:56, exp. remaining 0:34:28, complete 49.62%
att-weights epoch 481, step 444, max_size:classes 21, max_size:data 566, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.021 sec/step, elapsed 0:33:58, exp. remaining 0:34:13, complete 49.83%
att-weights epoch 481, step 445, max_size:classes 18, max_size:data 570, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.064 sec/step, elapsed 0:34:00, exp. remaining 0:33:58, complete 50.03%
att-weights epoch 481, step 446, max_size:classes 21, max_size:data 604, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.861 sec/step, elapsed 0:34:02, exp. remaining 0:33:45, complete 50.21%
att-weights epoch 481, step 447, max_size:classes 20, max_size:data 814, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.439 sec/step, elapsed 0:34:04, exp. remaining 0:33:27, complete 50.45%
att-weights epoch 481, step 448, max_size:classes 19, max_size:data 592, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.976 sec/step, elapsed 0:34:07, exp. remaining 0:33:13, complete 50.66%
att-weights epoch 481, step 449, max_size:classes 20, max_size:data 620, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.888 sec/step, elapsed 0:34:09, exp. remaining 0:32:58, complete 50.87%
att-weights epoch 481, step 450, max_size:classes 20, max_size:data 697, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.483 sec/step, elapsed 0:34:10, exp. remaining 0:32:43, complete 51.08%
att-weights epoch 481, step 451, max_size:classes 21, max_size:data 641, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.011 sec/step, elapsed 0:34:12, exp. remaining 0:32:29, complete 51.29%
att-weights epoch 481, step 452, max_size:classes 19, max_size:data 591, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.749 sec/step, elapsed 0:34:14, exp. remaining 0:32:17, complete 51.47%
att-weights epoch 481, step 453, max_size:classes 22, max_size:data 654, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.180 sec/step, elapsed 0:34:16, exp. remaining 0:32:05, complete 51.64%
att-weights epoch 481, step 454, max_size:classes 20, max_size:data 599, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.846 sec/step, elapsed 0:34:18, exp. remaining 0:31:54, complete 51.82%
att-weights epoch 481, step 455, max_size:classes 23, max_size:data 677, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.689 sec/step, elapsed 0:34:20, exp. remaining 0:31:39, complete 52.03%
att-weights epoch 481, step 456, max_size:classes 20, max_size:data 558, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.141 sec/step, elapsed 0:34:22, exp. remaining 0:31:25, complete 52.23%
att-weights epoch 481, step 457, max_size:classes 17, max_size:data 612, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.678 sec/step, elapsed 0:34:23, exp. remaining 0:31:14, complete 52.41%
att-weights epoch 481, step 458, max_size:classes 20, max_size:data 643, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.208 sec/step, elapsed 0:34:26, exp. remaining 0:31:00, complete 52.62%
att-weights epoch 481, step 459, max_size:classes 21, max_size:data 585, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.040 sec/step, elapsed 0:34:28, exp. remaining 0:30:46, complete 52.83%
att-weights epoch 481, step 460, max_size:classes 20, max_size:data 606, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.900 sec/step, elapsed 0:34:30, exp. remaining 0:30:32, complete 53.04%
att-weights epoch 481, step 461, max_size:classes 20, max_size:data 681, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.867 sec/step, elapsed 0:34:31, exp. remaining 0:30:16, complete 53.28%
att-weights epoch 481, step 462, max_size:classes 24, max_size:data 620, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.033 sec/step, elapsed 0:34:32, exp. remaining 0:30:02, complete 53.49%
att-weights epoch 481, step 463, max_size:classes 19, max_size:data 683, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.787 sec/step, elapsed 0:34:34, exp. remaining 0:29:48, complete 53.70%
att-weights epoch 481, step 464, max_size:classes 20, max_size:data 530, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.925 sec/step, elapsed 0:34:36, exp. remaining 0:29:42, complete 53.81%
att-weights epoch 481, step 465, max_size:classes 20, max_size:data 596, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.610 sec/step, elapsed 0:34:38, exp. remaining 0:29:26, complete 54.05%
att-weights epoch 481, step 466, max_size:classes 19, max_size:data 670, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.731 sec/step, elapsed 0:34:40, exp. remaining 0:29:10, complete 54.29%
att-weights epoch 481, step 467, max_size:classes 20, max_size:data 592, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.885 sec/step, elapsed 0:34:41, exp. remaining 0:28:57, complete 54.50%
att-weights epoch 481, step 468, max_size:classes 19, max_size:data 645, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.123 sec/step, elapsed 0:34:44, exp. remaining 0:28:44, complete 54.71%
att-weights epoch 481, step 469, max_size:classes 22, max_size:data 573, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.820 sec/step, elapsed 0:34:45, exp. remaining 0:28:34, complete 54.89%
att-weights epoch 481, step 470, max_size:classes 18, max_size:data 497, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.864 sec/step, elapsed 0:34:47, exp. remaining 0:28:21, complete 55.10%
att-weights epoch 481, step 471, max_size:classes 21, max_size:data 622, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.656 sec/step, elapsed 0:34:49, exp. remaining 0:28:05, complete 55.34%
att-weights epoch 481, step 472, max_size:classes 19, max_size:data 625, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.914 sec/step, elapsed 0:34:51, exp. remaining 0:27:53, complete 55.55%
att-weights epoch 481, step 473, max_size:classes 20, max_size:data 1012, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.535 sec/step, elapsed 0:34:52, exp. remaining 0:27:42, complete 55.73%
att-weights epoch 481, step 474, max_size:classes 18, max_size:data 554, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.786 sec/step, elapsed 0:34:54, exp. remaining 0:27:27, complete 55.97%
att-weights epoch 481, step 475, max_size:classes 20, max_size:data 548, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.677 sec/step, elapsed 0:34:56, exp. remaining 0:27:15, complete 56.18%
att-weights epoch 481, step 476, max_size:classes 23, max_size:data 616, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.737 sec/step, elapsed 0:34:58, exp. remaining 0:27:04, complete 56.35%
att-weights epoch 481, step 477, max_size:classes 18, max_size:data 606, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.904 sec/step, elapsed 0:34:59, exp. remaining 0:26:54, complete 56.53%
att-weights epoch 481, step 478, max_size:classes 20, max_size:data 686, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.889 sec/step, elapsed 0:35:01, exp. remaining 0:26:42, complete 56.74%
att-weights epoch 481, step 479, max_size:classes 18, max_size:data 619, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.763 sec/step, elapsed 0:35:03, exp. remaining 0:26:30, complete 56.95%
att-weights epoch 481, step 480, max_size:classes 16, max_size:data 548, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.823 sec/step, elapsed 0:35:05, exp. remaining 0:26:15, complete 57.19%
att-weights epoch 481, step 481, max_size:classes 18, max_size:data 578, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.798 sec/step, elapsed 0:35:07, exp. remaining 0:26:05, complete 57.37%
att-weights epoch 481, step 482, max_size:classes 18, max_size:data 693, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.646 sec/step, elapsed 0:35:08, exp. remaining 0:25:56, complete 57.54%
att-weights epoch 481, step 483, max_size:classes 18, max_size:data 523, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.101 sec/step, elapsed 0:35:10, exp. remaining 0:25:42, complete 57.79%
att-weights epoch 481, step 484, max_size:classes 21, max_size:data 616, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.964 sec/step, elapsed 0:35:12, exp. remaining 0:25:28, complete 58.03%
att-weights epoch 481, step 485, max_size:classes 19, max_size:data 654, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.287 sec/step, elapsed 0:35:14, exp. remaining 0:25:15, complete 58.24%
att-weights epoch 481, step 486, max_size:classes 18, max_size:data 713, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.760 sec/step, elapsed 0:35:15, exp. remaining 0:24:59, complete 58.52%
att-weights epoch 481, step 487, max_size:classes 17, max_size:data 513, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.539 sec/step, elapsed 0:35:17, exp. remaining 0:24:52, complete 58.66%
att-weights epoch 481, step 488, max_size:classes 19, max_size:data 604, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.144 sec/step, elapsed 0:35:19, exp. remaining 0:24:40, complete 58.87%
att-weights epoch 481, step 489, max_size:classes 19, max_size:data 567, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.069 sec/step, elapsed 0:35:21, exp. remaining 0:24:29, complete 59.08%
att-weights epoch 481, step 490, max_size:classes 17, max_size:data 711, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.562 sec/step, elapsed 0:35:23, exp. remaining 0:24:13, complete 59.36%
att-weights epoch 481, step 491, max_size:classes 17, max_size:data 696, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.950 sec/step, elapsed 0:35:25, exp. remaining 0:24:02, complete 59.57%
att-weights epoch 481, step 492, max_size:classes 18, max_size:data 525, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.993 sec/step, elapsed 0:35:27, exp. remaining 0:23:51, complete 59.78%
att-weights epoch 481, step 493, max_size:classes 16, max_size:data 517, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.965 sec/step, elapsed 0:35:29, exp. remaining 0:23:40, complete 59.99%
att-weights epoch 481, step 494, max_size:classes 17, max_size:data 656, mem_usage:GPU:0 0.9GB, num_seqs 6, 5.036 sec/step, elapsed 0:35:34, exp. remaining 0:23:29, complete 60.23%
att-weights epoch 481, step 495, max_size:classes 17, max_size:data 500, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.106 sec/step, elapsed 0:35:36, exp. remaining 0:23:20, complete 60.41%
att-weights epoch 481, step 496, max_size:classes 17, max_size:data 805, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.306 sec/step, elapsed 0:35:37, exp. remaining 0:23:11, complete 60.58%
att-weights epoch 481, step 497, max_size:classes 15, max_size:data 607, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.864 sec/step, elapsed 0:35:39, exp. remaining 0:23:00, complete 60.79%
att-weights epoch 481, step 498, max_size:classes 21, max_size:data 612, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.895 sec/step, elapsed 0:35:41, exp. remaining 0:22:53, complete 60.93%
att-weights epoch 481, step 499, max_size:classes 18, max_size:data 498, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.143 sec/step, elapsed 0:35:43, exp. remaining 0:22:46, complete 61.07%
att-weights epoch 481, step 500, max_size:classes 16, max_size:data 524, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.014 sec/step, elapsed 0:35:45, exp. remaining 0:22:35, complete 61.28%
att-weights epoch 481, step 501, max_size:classes 16, max_size:data 662, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.034 sec/step, elapsed 0:35:47, exp. remaining 0:22:25, complete 61.49%
att-weights epoch 481, step 502, max_size:classes 17, max_size:data 609, mem_usage:GPU:0 0.9GB, num_seqs 6, 3.432 sec/step, elapsed 0:35:51, exp. remaining 0:22:11, complete 61.77%
att-weights epoch 481, step 503, max_size:classes 18, max_size:data 546, mem_usage:GPU:0 0.9GB, num_seqs 7, 7.397 sec/step, elapsed 0:35:58, exp. remaining 0:22:04, complete 61.98%
att-weights epoch 481, step 504, max_size:classes 17, max_size:data 612, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.496 sec/step, elapsed 0:36:00, exp. remaining 0:21:54, complete 62.19%
att-weights epoch 481, step 505, max_size:classes 16, max_size:data 710, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.990 sec/step, elapsed 0:36:02, exp. remaining 0:21:41, complete 62.43%
att-weights epoch 481, step 506, max_size:classes 15, max_size:data 595, mem_usage:GPU:0 0.9GB, num_seqs 6, 4.136 sec/step, elapsed 0:36:07, exp. remaining 0:21:28, complete 62.71%
att-weights epoch 481, step 507, max_size:classes 17, max_size:data 533, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.624 sec/step, elapsed 0:36:08, exp. remaining 0:21:18, complete 62.92%
att-weights epoch 481, step 508, max_size:classes 17, max_size:data 821, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.932 sec/step, elapsed 0:36:10, exp. remaining 0:21:07, complete 63.13%
att-weights epoch 481, step 509, max_size:classes 19, max_size:data 601, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.995 sec/step, elapsed 0:36:13, exp. remaining 0:20:52, complete 63.44%
att-weights epoch 481, step 510, max_size:classes 18, max_size:data 584, mem_usage:GPU:0 0.9GB, num_seqs 6, 3.671 sec/step, elapsed 0:36:17, exp. remaining 0:20:41, complete 63.69%
att-weights epoch 481, step 511, max_size:classes 18, max_size:data 478, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.344 sec/step, elapsed 0:36:19, exp. remaining 0:20:29, complete 63.93%
att-weights epoch 481, step 512, max_size:classes 16, max_size:data 556, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.829 sec/step, elapsed 0:36:21, exp. remaining 0:20:15, complete 64.21%
att-weights epoch 481, step 513, max_size:classes 17, max_size:data 616, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.191 sec/step, elapsed 0:36:23, exp. remaining 0:20:06, complete 64.42%
att-weights epoch 481, step 514, max_size:classes 16, max_size:data 561, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.895 sec/step, elapsed 0:36:25, exp. remaining 0:19:54, complete 64.66%
att-weights epoch 481, step 515, max_size:classes 19, max_size:data 487, mem_usage:GPU:0 0.9GB, num_seqs 8, 1.862 sec/step, elapsed 0:36:27, exp. remaining 0:19:40, complete 64.94%
att-weights epoch 481, step 516, max_size:classes 16, max_size:data 636, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.211 sec/step, elapsed 0:36:29, exp. remaining 0:19:29, complete 65.19%
att-weights epoch 481, step 517, max_size:classes 17, max_size:data 659, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.072 sec/step, elapsed 0:36:31, exp. remaining 0:19:19, complete 65.40%
att-weights epoch 481, step 518, max_size:classes 16, max_size:data 418, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.102 sec/step, elapsed 0:36:33, exp. remaining 0:19:06, complete 65.68%
att-weights epoch 481, step 519, max_size:classes 17, max_size:data 546, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.022 sec/step, elapsed 0:36:35, exp. remaining 0:18:56, complete 65.89%
att-weights epoch 481, step 520, max_size:classes 16, max_size:data 564, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.039 sec/step, elapsed 0:36:37, exp. remaining 0:18:45, complete 66.13%
att-weights epoch 481, step 521, max_size:classes 16, max_size:data 492, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.632 sec/step, elapsed 0:36:40, exp. remaining 0:18:38, complete 66.31%
att-weights epoch 481, step 522, max_size:classes 15, max_size:data 606, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.243 sec/step, elapsed 0:36:42, exp. remaining 0:18:28, complete 66.52%
att-weights epoch 481, step 523, max_size:classes 14, max_size:data 555, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.956 sec/step, elapsed 0:36:44, exp. remaining 0:18:17, complete 66.76%
att-weights epoch 481, step 524, max_size:classes 17, max_size:data 455, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.353 sec/step, elapsed 0:36:47, exp. remaining 0:18:05, complete 67.04%
att-weights epoch 481, step 525, max_size:classes 22, max_size:data 520, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.142 sec/step, elapsed 0:36:49, exp. remaining 0:17:54, complete 67.28%
att-weights epoch 481, step 526, max_size:classes 20, max_size:data 587, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.900 sec/step, elapsed 0:36:51, exp. remaining 0:17:43, complete 67.53%
att-weights epoch 481, step 527, max_size:classes 16, max_size:data 464, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.377 sec/step, elapsed 0:36:53, exp. remaining 0:17:32, complete 67.77%
att-weights epoch 481, step 528, max_size:classes 16, max_size:data 572, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.074 sec/step, elapsed 0:36:55, exp. remaining 0:17:23, complete 67.98%
att-weights epoch 481, step 529, max_size:classes 15, max_size:data 502, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.998 sec/step, elapsed 0:36:57, exp. remaining 0:17:09, complete 68.30%
att-weights epoch 481, step 530, max_size:classes 21, max_size:data 679, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.913 sec/step, elapsed 0:36:59, exp. remaining 0:16:57, complete 68.58%
att-weights epoch 481, step 531, max_size:classes 17, max_size:data 648, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.903 sec/step, elapsed 0:37:01, exp. remaining 0:16:46, complete 68.82%
att-weights epoch 481, step 532, max_size:classes 15, max_size:data 505, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.426 sec/step, elapsed 0:37:03, exp. remaining 0:16:36, complete 69.06%
att-weights epoch 481, step 533, max_size:classes 14, max_size:data 462, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.336 sec/step, elapsed 0:37:06, exp. remaining 0:16:25, complete 69.31%
att-weights epoch 481, step 534, max_size:classes 22, max_size:data 539, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.144 sec/step, elapsed 0:37:08, exp. remaining 0:16:12, complete 69.62%
att-weights epoch 481, step 535, max_size:classes 16, max_size:data 515, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.461 sec/step, elapsed 0:37:10, exp. remaining 0:16:00, complete 69.90%
att-weights epoch 481, step 536, max_size:classes 16, max_size:data 557, mem_usage:GPU:0 0.9GB, num_seqs 7, 14.544 sec/step, elapsed 0:37:25, exp. remaining 0:15:53, complete 70.18%
att-weights epoch 481, step 537, max_size:classes 15, max_size:data 611, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.036 sec/step, elapsed 0:37:27, exp. remaining 0:15:42, complete 70.46%
att-weights epoch 481, step 538, max_size:classes 16, max_size:data 423, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.220 sec/step, elapsed 0:37:29, exp. remaining 0:15:30, complete 70.74%
att-weights epoch 481, step 539, max_size:classes 14, max_size:data 471, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.132 sec/step, elapsed 0:37:31, exp. remaining 0:15:18, complete 71.02%
att-weights epoch 481, step 540, max_size:classes 14, max_size:data 520, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.885 sec/step, elapsed 0:37:34, exp. remaining 0:15:07, complete 71.30%
att-weights epoch 481, step 541, max_size:classes 15, max_size:data 507, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.210 sec/step, elapsed 0:37:36, exp. remaining 0:14:56, complete 71.58%
att-weights epoch 481, step 542, max_size:classes 15, max_size:data 528, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.303 sec/step, elapsed 0:37:39, exp. remaining 0:14:43, complete 71.89%
att-weights epoch 481, step 543, max_size:classes 14, max_size:data 402, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.504 sec/step, elapsed 0:37:41, exp. remaining 0:14:28, complete 72.24%
att-weights epoch 481, step 544, max_size:classes 15, max_size:data 487, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.288 sec/step, elapsed 0:37:43, exp. remaining 0:14:17, complete 72.52%
att-weights epoch 481, step 545, max_size:classes 16, max_size:data 466, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.354 sec/step, elapsed 0:37:46, exp. remaining 0:14:08, complete 72.77%
att-weights epoch 481, step 546, max_size:classes 15, max_size:data 485, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.151 sec/step, elapsed 0:37:48, exp. remaining 0:13:57, complete 73.04%
att-weights epoch 481, step 547, max_size:classes 14, max_size:data 490, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.116 sec/step, elapsed 0:37:50, exp. remaining 0:13:44, complete 73.36%
att-weights epoch 481, step 548, max_size:classes 14, max_size:data 488, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.004 sec/step, elapsed 0:37:52, exp. remaining 0:13:32, complete 73.67%
att-weights epoch 481, step 549, max_size:classes 14, max_size:data 500, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.539 sec/step, elapsed 0:37:55, exp. remaining 0:13:22, complete 73.92%
att-weights epoch 481, step 550, max_size:classes 13, max_size:data 493, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.399 sec/step, elapsed 0:37:57, exp. remaining 0:13:13, complete 74.16%
att-weights epoch 481, step 551, max_size:classes 16, max_size:data 415, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.245 sec/step, elapsed 0:37:59, exp. remaining 0:12:58, complete 74.55%
att-weights epoch 481, step 552, max_size:classes 13, max_size:data 398, mem_usage:GPU:0 0.9GB, num_seqs 10, 2.143 sec/step, elapsed 0:38:01, exp. remaining 0:12:47, complete 74.83%
att-weights epoch 481, step 553, max_size:classes 14, max_size:data 475, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.356 sec/step, elapsed 0:38:04, exp. remaining 0:12:35, complete 75.14%
att-weights epoch 481, step 554, max_size:classes 15, max_size:data 562, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.106 sec/step, elapsed 0:38:06, exp. remaining 0:12:23, complete 75.45%
att-weights epoch 481, step 555, max_size:classes 14, max_size:data 463, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.374 sec/step, elapsed 0:38:08, exp. remaining 0:12:14, complete 75.70%
att-weights epoch 481, step 556, max_size:classes 14, max_size:data 396, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.438 sec/step, elapsed 0:38:11, exp. remaining 0:12:04, complete 75.98%
att-weights epoch 481, step 557, max_size:classes 13, max_size:data 440, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.245 sec/step, elapsed 0:38:13, exp. remaining 0:11:52, complete 76.29%
att-weights epoch 481, step 558, max_size:classes 15, max_size:data 460, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.331 sec/step, elapsed 0:38:15, exp. remaining 0:11:41, complete 76.61%
att-weights epoch 481, step 559, max_size:classes 16, max_size:data 534, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.185 sec/step, elapsed 0:38:17, exp. remaining 0:11:29, complete 76.92%
att-weights epoch 481, step 560, max_size:classes 14, max_size:data 362, mem_usage:GPU:0 0.9GB, num_seqs 11, 2.294 sec/step, elapsed 0:38:20, exp. remaining 0:11:16, complete 77.27%
att-weights epoch 481, step 561, max_size:classes 13, max_size:data 445, mem_usage:GPU:0 0.9GB, num_seqs 8, 1.715 sec/step, elapsed 0:38:21, exp. remaining 0:11:05, complete 77.58%
att-weights epoch 481, step 562, max_size:classes 13, max_size:data 433, mem_usage:GPU:0 0.9GB, num_seqs 9, 1.996 sec/step, elapsed 0:38:23, exp. remaining 0:10:56, complete 77.83%
att-weights epoch 481, step 563, max_size:classes 13, max_size:data 444, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.161 sec/step, elapsed 0:38:26, exp. remaining 0:10:47, complete 78.07%
att-weights epoch 481, step 564, max_size:classes 13, max_size:data 539, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.807 sec/step, elapsed 0:38:27, exp. remaining 0:10:40, complete 78.28%
att-weights epoch 481, step 565, max_size:classes 17, max_size:data 496, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.020 sec/step, elapsed 0:38:29, exp. remaining 0:10:29, complete 78.60%
att-weights epoch 481, step 566, max_size:classes 13, max_size:data 415, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.211 sec/step, elapsed 0:38:32, exp. remaining 0:10:19, complete 78.88%
att-weights epoch 481, step 567, max_size:classes 13, max_size:data 426, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.669 sec/step, elapsed 0:38:34, exp. remaining 0:10:09, complete 79.16%
att-weights epoch 481, step 568, max_size:classes 13, max_size:data 407, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.325 sec/step, elapsed 0:38:37, exp. remaining 0:09:58, complete 79.47%
att-weights epoch 481, step 569, max_size:classes 14, max_size:data 368, mem_usage:GPU:0 0.9GB, num_seqs 10, 2.527 sec/step, elapsed 0:38:39, exp. remaining 0:09:46, complete 79.82%
att-weights epoch 481, step 570, max_size:classes 14, max_size:data 391, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.191 sec/step, elapsed 0:38:41, exp. remaining 0:09:35, complete 80.13%
att-weights epoch 481, step 571, max_size:classes 13, max_size:data 539, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.226 sec/step, elapsed 0:38:44, exp. remaining 0:09:23, complete 80.48%
att-weights epoch 481, step 572, max_size:classes 12, max_size:data 472, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.055 sec/step, elapsed 0:38:46, exp. remaining 0:09:14, complete 80.76%
att-weights epoch 481, step 573, max_size:classes 13, max_size:data 600, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.021 sec/step, elapsed 0:38:48, exp. remaining 0:09:05, complete 81.01%
att-weights epoch 481, step 574, max_size:classes 13, max_size:data 426, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.032 sec/step, elapsed 0:38:50, exp. remaining 0:08:56, complete 81.28%
att-weights epoch 481, step 575, max_size:classes 11, max_size:data 469, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.665 sec/step, elapsed 0:38:52, exp. remaining 0:08:47, complete 81.56%
att-weights epoch 481, step 576, max_size:classes 13, max_size:data 491, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.423 sec/step, elapsed 0:38:55, exp. remaining 0:08:35, complete 81.91%
att-weights epoch 481, step 577, max_size:classes 13, max_size:data 424, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.336 sec/step, elapsed 0:38:57, exp. remaining 0:08:20, complete 82.37%
att-weights epoch 481, step 578, max_size:classes 17, max_size:data 398, mem_usage:GPU:0 0.9GB, num_seqs 10, 2.622 sec/step, elapsed 0:39:00, exp. remaining 0:08:08, complete 82.72%
att-weights epoch 481, step 579, max_size:classes 14, max_size:data 415, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.329 sec/step, elapsed 0:39:02, exp. remaining 0:07:58, complete 83.03%
att-weights epoch 481, step 580, max_size:classes 14, max_size:data 375, mem_usage:GPU:0 0.9GB, num_seqs 10, 2.454 sec/step, elapsed 0:39:04, exp. remaining 0:07:48, complete 83.34%
att-weights epoch 481, step 581, max_size:classes 11, max_size:data 403, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.581 sec/step, elapsed 0:39:07, exp. remaining 0:07:38, complete 83.66%
att-weights epoch 481, step 582, max_size:classes 14, max_size:data 505, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.528 sec/step, elapsed 0:39:10, exp. remaining 0:07:26, complete 84.04%
att-weights epoch 481, step 583, max_size:classes 12, max_size:data 462, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.188 sec/step, elapsed 0:39:12, exp. remaining 0:07:15, complete 84.39%
att-weights epoch 481, step 584, max_size:classes 13, max_size:data 496, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.370 sec/step, elapsed 0:39:14, exp. remaining 0:07:02, complete 84.78%
att-weights epoch 481, step 585, max_size:classes 13, max_size:data 384, mem_usage:GPU:0 0.9GB, num_seqs 10, 2.609 sec/step, elapsed 0:39:17, exp. remaining 0:06:53, complete 85.09%
att-weights epoch 481, step 586, max_size:classes 11, max_size:data 291, mem_usage:GPU:0 0.9GB, num_seqs 13, 3.023 sec/step, elapsed 0:39:20, exp. remaining 0:06:42, complete 85.44%
att-weights epoch 481, step 587, max_size:classes 12, max_size:data 392, mem_usage:GPU:0 0.9GB, num_seqs 10, 2.363 sec/step, elapsed 0:39:22, exp. remaining 0:06:30, complete 85.82%
att-weights epoch 481, step 588, max_size:classes 12, max_size:data 435, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.779 sec/step, elapsed 0:39:25, exp. remaining 0:06:19, complete 86.17%
att-weights epoch 481, step 589, max_size:classes 12, max_size:data 423, mem_usage:GPU:0 0.9GB, num_seqs 9, 3.705 sec/step, elapsed 0:39:29, exp. remaining 0:06:10, complete 86.49%
att-weights epoch 481, step 590, max_size:classes 13, max_size:data 405, mem_usage:GPU:0 0.9GB, num_seqs 9, 1.965 sec/step, elapsed 0:39:31, exp. remaining 0:06:01, complete 86.77%
att-weights epoch 481, step 591, max_size:classes 12, max_size:data 348, mem_usage:GPU:0 0.9GB, num_seqs 11, 2.354 sec/step, elapsed 0:39:33, exp. remaining 0:05:49, complete 87.15%
att-weights epoch 481, step 592, max_size:classes 12, max_size:data 372, mem_usage:GPU:0 0.9GB, num_seqs 10, 2.891 sec/step, elapsed 0:39:36, exp. remaining 0:05:42, complete 87.40%
att-weights epoch 481, step 593, max_size:classes 12, max_size:data 356, mem_usage:GPU:0 0.9GB, num_seqs 11, 2.543 sec/step, elapsed 0:39:38, exp. remaining 0:05:33, complete 87.71%
att-weights epoch 481, step 594, max_size:classes 12, max_size:data 422, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.571 sec/step, elapsed 0:39:41, exp. remaining 0:05:21, complete 88.09%
att-weights epoch 481, step 595, max_size:classes 12, max_size:data 399, mem_usage:GPU:0 0.9GB, num_seqs 10, 2.600 sec/step, elapsed 0:39:44, exp. remaining 0:05:12, complete 88.41%
att-weights epoch 481, step 596, max_size:classes 13, max_size:data 358, mem_usage:GPU:0 0.9GB, num_seqs 11, 2.466 sec/step, elapsed 0:39:46, exp. remaining 0:05:00, complete 88.83%
att-weights epoch 481, step 597, max_size:classes 10, max_size:data 377, mem_usage:GPU:0 0.9GB, num_seqs 10, 2.829 sec/step, elapsed 0:39:49, exp. remaining 0:04:51, complete 89.14%
att-weights epoch 481, step 598, max_size:classes 11, max_size:data 388, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.578 sec/step, elapsed 0:39:51, exp. remaining 0:04:42, complete 89.42%
att-weights epoch 481, step 599, max_size:classes 9, max_size:data 478, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.514 sec/step, elapsed 0:39:54, exp. remaining 0:04:29, complete 89.87%
att-weights epoch 481, step 600, max_size:classes 13, max_size:data 331, mem_usage:GPU:0 0.9GB, num_seqs 11, 2.566 sec/step, elapsed 0:39:57, exp. remaining 0:04:16, complete 90.33%
att-weights epoch 481, step 601, max_size:classes 10, max_size:data 513, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.437 sec/step, elapsed 0:39:59, exp. remaining 0:04:05, complete 90.71%
att-weights epoch 481, step 602, max_size:classes 14, max_size:data 420, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.301 sec/step, elapsed 0:40:01, exp. remaining 0:03:55, complete 91.06%
att-weights epoch 481, step 603, max_size:classes 11, max_size:data 347, mem_usage:GPU:0 0.9GB, num_seqs 11, 2.902 sec/step, elapsed 0:40:04, exp. remaining 0:03:43, complete 91.48%
att-weights epoch 481, step 604, max_size:classes 11, max_size:data 434, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.642 sec/step, elapsed 0:40:07, exp. remaining 0:03:33, complete 91.86%
att-weights epoch 481, step 605, max_size:classes 13, max_size:data 332, mem_usage:GPU:0 0.9GB, num_seqs 12, 3.066 sec/step, elapsed 0:40:10, exp. remaining 0:03:21, complete 92.28%
att-weights epoch 481, step 606, max_size:classes 11, max_size:data 359, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.403 sec/step, elapsed 0:40:12, exp. remaining 0:03:10, complete 92.67%
att-weights epoch 481, step 607, max_size:classes 10, max_size:data 448, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.196 sec/step, elapsed 0:40:14, exp. remaining 0:03:00, complete 93.05%
att-weights epoch 481, step 608, max_size:classes 11, max_size:data 297, mem_usage:GPU:0 0.9GB, num_seqs 13, 2.978 sec/step, elapsed 0:40:17, exp. remaining 0:02:48, complete 93.47%
att-weights epoch 481, step 609, max_size:classes 16, max_size:data 296, mem_usage:GPU:0 0.9GB, num_seqs 13, 4.782 sec/step, elapsed 0:40:22, exp. remaining 0:02:38, complete 93.85%
att-weights epoch 481, step 610, max_size:classes 9, max_size:data 339, mem_usage:GPU:0 0.9GB, num_seqs 11, 3.593 sec/step, elapsed 0:40:26, exp. remaining 0:02:28, complete 94.24%
att-weights epoch 481, step 611, max_size:classes 9, max_size:data 385, mem_usage:GPU:0 0.9GB, num_seqs 10, 1.966 sec/step, elapsed 0:40:28, exp. remaining 0:02:18, complete 94.59%
att-weights epoch 481, step 612, max_size:classes 9, max_size:data 319, mem_usage:GPU:0 0.9GB, num_seqs 12, 2.456 sec/step, elapsed 0:40:30, exp. remaining 0:02:07, complete 95.01%
att-weights epoch 481, step 613, max_size:classes 11, max_size:data 349, mem_usage:GPU:0 0.9GB, num_seqs 11, 2.777 sec/step, elapsed 0:40:33, exp. remaining 0:01:57, complete 95.39%
att-weights epoch 481, step 614, max_size:classes 10, max_size:data 328, mem_usage:GPU:0 0.9GB, num_seqs 12, 2.911 sec/step, elapsed 0:40:36, exp. remaining 0:01:45, complete 95.84%
att-weights epoch 481, step 615, max_size:classes 10, max_size:data 319, mem_usage:GPU:0 0.9GB, num_seqs 11, 2.504 sec/step, elapsed 0:40:38, exp. remaining 0:01:34, complete 96.26%
att-weights epoch 481, step 616, max_size:classes 9, max_size:data 337, mem_usage:GPU:0 0.9GB, num_seqs 11, 2.112 sec/step, elapsed 0:40:41, exp. remaining 0:01:21, complete 96.79%
att-weights epoch 481, step 617, max_size:classes 9, max_size:data 323, mem_usage:GPU:0 0.9GB, num_seqs 12, 2.822 sec/step, elapsed 0:40:43, exp. remaining 0:01:11, complete 97.17%
att-weights epoch 481, step 618, max_size:classes 11, max_size:data 351, mem_usage:GPU:0 0.9GB, num_seqs 11, 3.232 sec/step, elapsed 0:40:47, exp. remaining 0:01:02, complete 97.52%
att-weights epoch 481, step 619, max_size:classes 9, max_size:data 348, mem_usage:GPU:0 0.9GB, num_seqs 11, 3.238 sec/step, elapsed 0:40:50, exp. remaining 0:00:51, complete 97.94%
att-weights epoch 481, step 620, max_size:classes 8, max_size:data 386, mem_usage:GPU:0 0.9GB, num_seqs 10, 2.910 sec/step, elapsed 0:40:53, exp. remaining 0:00:39, complete 98.43%
att-weights epoch 481, step 621, max_size:classes 8, max_size:data 327, mem_usage:GPU:0 0.9GB, num_seqs 12, 3.131 sec/step, elapsed 0:40:56, exp. remaining 0:00:26, complete 98.92%
att-weights epoch 481, step 622, max_size:classes 9, max_size:data 340, mem_usage:GPU:0 0.9GB, num_seqs 11, 1.136 sec/step, elapsed 0:40:57, exp. remaining 0:00:15, complete 99.37%
att-weights epoch 481, step 623, max_size:classes 7, max_size:data 303, mem_usage:GPU:0 0.9GB, num_seqs 13, 0.941 sec/step, elapsed 0:40:58, exp. remaining 0:00:02, complete 99.90%
att-weights epoch 481, step 624, max_size:classes 9, max_size:data 331, mem_usage:GPU:0 0.9GB, num_seqs 12, 0.915 sec/step, elapsed 0:40:59, exp. remaining 0:00:02, complete 99.90%
att-weights epoch 481, step 625, max_size:classes 7, max_size:data 248, mem_usage:GPU:0 0.9GB, num_seqs 15, 1.134 sec/step, elapsed 0:41:00, exp. remaining 0:00:02, complete 99.90%
att-weights epoch 481, step 626, max_size:classes 7, max_size:data 342, mem_usage:GPU:0 0.9GB, num_seqs 11, 0.925 sec/step, elapsed 0:41:01, exp. remaining 0:00:02, complete 99.90%
att-weights epoch 481, step 627, max_size:classes 7, max_size:data 369, mem_usage:GPU:0 0.9GB, num_seqs 10, 0.713 sec/step, elapsed 0:41:02, exp. remaining 0:00:02, complete 99.90%
att-weights epoch 481, step 628, max_size:classes 6, max_size:data 317, mem_usage:GPU:0 0.9GB, num_seqs 12, 0.900 sec/step, elapsed 0:41:03, exp. remaining 0:00:02, complete 99.90%
att-weights epoch 481, step 629, max_size:classes 6, max_size:data 278, mem_usage:GPU:0 0.9GB, num_seqs 14, 1.090 sec/step, elapsed 0:41:04, exp. remaining 0:00:02, complete 99.90%
att-weights epoch 481, step 630, max_size:classes 6, max_size:data 282, mem_usage:GPU:0 0.9GB, num_seqs 14, 1.050 sec/step, elapsed 0:41:05, exp. remaining 0:00:02, complete 99.90%
att-weights epoch 481, step 631, max_size:classes 5, max_size:data 287, mem_usage:GPU:0 0.9GB, num_seqs 13, 0.934 sec/step, elapsed 0:41:06, exp. remaining 0:00:02, complete 99.90%
att-weights epoch 481, step 632, max_size:classes 7, max_size:data 253, mem_usage:GPU:0 0.9GB, num_seqs 15, 1.065 sec/step, elapsed 0:41:07, exp. remaining 0:00:02, complete 99.90%
att-weights epoch 481, step 633, max_size:classes 3, max_size:data 135, mem_usage:GPU:0 0.9GB, num_seqs 4, 0.312 sec/step, elapsed 0:41:07, exp. remaining 0:00:02, complete 99.90%
Stats:
  mem_usage:GPU:0: Stats(mean=0.9GB, std_dev=0.0B, min=0.9GB, max=0.9GB, num_seqs=634, avg_data_len=1)
att-weights epoch 481, finished after 634 steps, 0:41:07 elapsed (29.5% computing time)
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
| Job ID ..............: 9491080
| Stopped at ..........: Sun Jun 30 13:14:53 CEST 2019
| Resources requested .: s_core=0,pxe=ubuntu_16.04,h_vmem=1536G,gpu=1,h_rt=7200,h_rss=4G,h_fsize=20G,scratch_free=5G,num_proc=5
| Resources used ......: cpu=01:15:44, mem=16800.15896 GB s, io=10.61092 GB, vmem=4.162G, maxvmem=4.184G, last_file_cache=248M, last_rss=3M, max-cache=3.758G
| Memory used .........: 4.000G / 4.000G (100.0%)
| Total time used .....: 0:44:59
|
+------- EPILOGUE SCRIPT -----------------------------------------------
