+------- PROLOGUE SCRIPT -----------------------------------------------
|
| Job ID ...........: 9506153
| Started at .......: Tue Jul  2 13:42:58 CEST 2019
| Execution host ...: cluster-cn-242
| Cluster queue ....: 4-GPU-1080
| Script ...........: /var/spool/sge/cluster-cn-242/job_scripts/9506153
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
RETURNN starting up, version 20181022.053313--git-dfada43-dirty, date/time 2019-07-02-13-43-17 (UTC+0200), pid 24358, cwd /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3-prior-k6, Python /u/bahar/settings/python3-returnn-tf1.9/bin/python3
RETURNN config: /u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3-prior-k6.config
RETURNN command line options: ()
Hostname: cluster-cn-242
TensorFlow: 1.10.0 (v1.10.0-0-g656e7a2b34) (<site-package> in /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow)
Error while getting SGE num_proc: FileNotFoundError(2, "No such file or directory: 'qstat'")
Setup TF inter and intra global thread pools, num_threads None, session opts {'device_count': {'GPU': 0}, 'log_device_placement': False}.
CUDA_VISIBLE_DEVICES is set to '2'.
Collecting TensorFlow device list...
Local devices available to TensorFlow:
  1/2: name: "/device:CPU:0"
       device_type: "CPU"
       memory_limit: 268435456
       locality {
       }
       incarnation: 3917583291381990150
  2/2: name: "/device:GPU:0"
       device_type: "GPU"
       memory_limit: 10911236096
       locality {
         bus_id: 2
         numa_node: 1
         links {
         }
       }
       incarnation: 10864521132473668120
       physical_device_desc: "device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:81:00.0, compute capability: 6.1"
Using gpu device 2: GeForce GTX 1080 Ti
Device not set explicitly, and we found a GPU, which we will use.
Model file prefix: /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3-prior-k6/net-model/network
NOTE: We will use 'sorted_reverse' seq ordering.
Setup tf.Session with options {'device_count': {'GPU': 1}, 'log_device_placement': False} ...
layer root/'data' output: Data(name='data', shape=(None, 40))
layer root/'source' output: Data(name='source_output', shape=(None, 40))
layer root/'lstm0_fw' output: Data(name='lstm0_fw_output', shape=(None, 1024), batch_dim_axis=1)
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9506153.1.4-GPU-1080/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.cc -o /var/tmp/9506153.1.4-GPU-1080/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9506153.1.4-GPU-1080/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.cc -o /var/tmp/9506153.1.4-GPU-1080/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
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
layer root/'dec_08_att_value0' output: Data(name='dec_08_att_value0_output', shape=(None, 512))
layer root/'dec_08_att_value' output: Data(name='dec_08_att_value_output', shape=(None, 8, 64))
layer root/'dec_06_att_value0' output: Data(name='dec_06_att_value0_output', shape=(None, 512))
layer root/'dec_06_att_value' output: Data(name='dec_06_att_value_output', shape=(None, 8, 64))
layer root/'dec_02_att_value0' output: Data(name='dec_02_att_value0_output', shape=(None, 512))
layer root/'dec_02_att_value' output: Data(name='dec_02_att_value_output', shape=(None, 8, 64))
layer root/'dec_10_att_value0' output: Data(name='dec_10_att_value0_output', shape=(None, 512))
layer root/'dec_10_att_value' output: Data(name='dec_10_att_value_output', shape=(None, 8, 64))
layer root/'dec_03_att_value0' output: Data(name='dec_03_att_value0_output', shape=(None, 512))
layer root/'dec_03_att_value' output: Data(name='dec_03_att_value_output', shape=(None, 8, 64))
layer root/'dec_12_att_key0' output: Data(name='dec_12_att_key0_output', shape=(None, 512))
layer root/'dec_12_att_key' output: Data(name='dec_12_att_key_output', shape=(None, 8, 64))
layer root/'dec_03_att_key0' output: Data(name='dec_03_att_key0_output', shape=(None, 512))
layer root/'dec_03_att_key' output: Data(name='dec_03_att_key_output', shape=(None, 8, 64))
layer root/'dec_05_att_key0' output: Data(name='dec_05_att_key0_output', shape=(None, 512))
layer root/'dec_05_att_key' output: Data(name='dec_05_att_key_output', shape=(None, 8, 64))
layer root/'dec_08_att_key0' output: Data(name='dec_08_att_key0_output', shape=(None, 512))
layer root/'dec_08_att_key' output: Data(name='dec_08_att_key_output', shape=(None, 8, 64))
layer root/'dec_01_att_value0' output: Data(name='dec_01_att_value0_output', shape=(None, 512))
layer root/'dec_01_att_value' output: Data(name='dec_01_att_value_output', shape=(None, 8, 64))
layer root/'dec_04_att_key0' output: Data(name='dec_04_att_key0_output', shape=(None, 512))
layer root/'dec_04_att_key' output: Data(name='dec_04_att_key_output', shape=(None, 8, 64))
layer root/'dec_01_att_key0' output: Data(name='dec_01_att_key0_output', shape=(None, 512))
layer root/'dec_01_att_key' output: Data(name='dec_01_att_key_output', shape=(None, 8, 64))
layer root/'dec_11_att_key0' output: Data(name='dec_11_att_key0_output', shape=(None, 512))
layer root/'dec_11_att_key' output: Data(name='dec_11_att_key_output', shape=(None, 8, 64))
layer root/'dec_10_att_key0' output: Data(name='dec_10_att_key0_output', shape=(None, 512))
layer root/'dec_10_att_key' output: Data(name='dec_10_att_key_output', shape=(None, 8, 64))
layer root/'dec_07_att_value0' output: Data(name='dec_07_att_value0_output', shape=(None, 512))
layer root/'dec_07_att_value' output: Data(name='dec_07_att_value_output', shape=(None, 8, 64))
layer root/'dec_04_att_value0' output: Data(name='dec_04_att_value0_output', shape=(None, 512))
layer root/'dec_04_att_value' output: Data(name='dec_04_att_value_output', shape=(None, 8, 64))
layer root/'dec_12_att_value0' output: Data(name='dec_12_att_value0_output', shape=(None, 512))
layer root/'dec_12_att_value' output: Data(name='dec_12_att_value_output', shape=(None, 8, 64))
layer root/'dec_02_att_key0' output: Data(name='dec_02_att_key0_output', shape=(None, 512))
layer root/'dec_02_att_key' output: Data(name='dec_02_att_key_output', shape=(None, 8, 64))
layer root/'dec_06_att_key0' output: Data(name='dec_06_att_key0_output', shape=(None, 512))
layer root/'dec_06_att_key' output: Data(name='dec_06_att_key_output', shape=(None, 8, 64))
layer root/'dec_05_att_value0' output: Data(name='dec_05_att_value0_output', shape=(None, 512))
layer root/'dec_05_att_value' output: Data(name='dec_05_att_value_output', shape=(None, 8, 64))
layer root/'dec_11_att_value0' output: Data(name='dec_11_att_value0_output', shape=(None, 512))
layer root/'dec_11_att_value' output: Data(name='dec_11_att_value_output', shape=(None, 8, 64))
layer root/'dec_09_att_value0' output: Data(name='dec_09_att_value0_output', shape=(None, 512))
layer root/'dec_09_att_value' output: Data(name='dec_09_att_value_output', shape=(None, 8, 64))
layer root/'dec_07_att_key0' output: Data(name='dec_07_att_key0_output', shape=(None, 512))
layer root/'dec_07_att_key' output: Data(name='dec_07_att_key_output', shape=(None, 8, 64))
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
TF: log_dir: /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/transformer-prior/forward-test-other/tf_log_dir/prefix:test-other-560-2019-07-02-11-43-00
Note: There are still these uninitialized variables: ['learning_rate:0']
att-weights epoch 560, step 0, max_size:classes 137, max_size:data 3452, mem_usage:GPU:0 1.0GB, num_seqs 1, 18.625 sec/step, elapsed 0:00:23, exp. remaining 1:46:21, complete 0.37%
att-weights epoch 560, step 1, max_size:classes 98, max_size:data 2953, mem_usage:GPU:0 1.0GB, num_seqs 1, 6.986 sec/step, elapsed 0:00:35, exp. remaining 2:22:20, complete 0.41%
att-weights epoch 560, step 2, max_size:classes 98, max_size:data 3365, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.782 sec/step, elapsed 0:00:37, exp. remaining 2:18:54, complete 0.44%
att-weights epoch 560, step 3, max_size:classes 89, max_size:data 2804, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.492 sec/step, elapsed 0:00:38, exp. remaining 2:14:21, complete 0.48%
att-weights epoch 560, step 4, max_size:classes 103, max_size:data 3373, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.729 sec/step, elapsed 0:00:40, exp. remaining 2:11:05, complete 0.51%
att-weights epoch 560, step 5, max_size:classes 95, max_size:data 3114, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.507 sec/step, elapsed 0:00:41, exp. remaining 2:07:45, complete 0.54%
att-weights epoch 560, step 6, max_size:classes 103, max_size:data 2961, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.444 sec/step, elapsed 0:00:43, exp. remaining 2:04:31, complete 0.58%
att-weights epoch 560, step 7, max_size:classes 89, max_size:data 2838, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.541 sec/step, elapsed 0:00:45, exp. remaining 2:01:54, complete 0.61%
att-weights epoch 560, step 8, max_size:classes 94, max_size:data 2277, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.105 sec/step, elapsed 0:00:46, exp. remaining 1:58:39, complete 0.65%
att-weights epoch 560, step 9, max_size:classes 79, max_size:data 3353, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.590 sec/step, elapsed 0:00:47, exp. remaining 1:51:02, complete 0.71%
att-weights epoch 560, step 10, max_size:classes 87, max_size:data 3095, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.478 sec/step, elapsed 0:00:49, exp. remaining 1:49:19, complete 0.75%
att-weights epoch 560, step 11, max_size:classes 98, max_size:data 3296, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.698 sec/step, elapsed 0:00:51, exp. remaining 1:48:42, complete 0.78%
att-weights epoch 560, step 12, max_size:classes 91, max_size:data 2540, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.241 sec/step, elapsed 0:00:52, exp. remaining 1:46:38, complete 0.82%
att-weights epoch 560, step 13, max_size:classes 86, max_size:data 2716, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.335 sec/step, elapsed 0:00:54, exp. remaining 1:44:56, complete 0.85%
att-weights epoch 560, step 14, max_size:classes 83, max_size:data 2145, mem_usage:GPU:0 1.0GB, num_seqs 1, 2.461 sec/step, elapsed 0:00:56, exp. remaining 1:45:28, complete 0.88%
att-weights epoch 560, step 15, max_size:classes 74, max_size:data 2312, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.009 sec/step, elapsed 0:00:57, exp. remaining 1:43:20, complete 0.92%
att-weights epoch 560, step 16, max_size:classes 70, max_size:data 2275, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.067 sec/step, elapsed 0:00:58, exp. remaining 1:41:28, complete 0.95%
att-weights epoch 560, step 17, max_size:classes 73, max_size:data 2610, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.138 sec/step, elapsed 0:00:59, exp. remaining 1:39:50, complete 0.99%
att-weights epoch 560, step 18, max_size:classes 76, max_size:data 1892, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.852 sec/step, elapsed 0:01:01, exp. remaining 1:39:28, complete 1.02%
att-weights epoch 560, step 19, max_size:classes 75, max_size:data 2350, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.122 sec/step, elapsed 0:01:02, exp. remaining 1:37:59, complete 1.05%
att-weights epoch 560, step 20, max_size:classes 83, max_size:data 2302, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.088 sec/step, elapsed 0:01:03, exp. remaining 1:33:34, complete 1.12%
att-weights epoch 560, step 21, max_size:classes 75, max_size:data 2230, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.102 sec/step, elapsed 0:01:04, exp. remaining 1:32:22, complete 1.16%
att-weights epoch 560, step 22, max_size:classes 78, max_size:data 1853, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.823 sec/step, elapsed 0:01:05, exp. remaining 1:30:50, complete 1.19%
att-weights epoch 560, step 23, max_size:classes 74, max_size:data 2043, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.113 sec/step, elapsed 0:01:06, exp. remaining 1:29:46, complete 1.22%
att-weights epoch 560, step 24, max_size:classes 65, max_size:data 2213, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.956 sec/step, elapsed 0:01:07, exp. remaining 1:28:34, complete 1.26%
att-weights epoch 560, step 25, max_size:classes 67, max_size:data 2336, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.952 sec/step, elapsed 0:01:08, exp. remaining 1:27:25, complete 1.29%
att-weights epoch 560, step 26, max_size:classes 71, max_size:data 2120, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.006 sec/step, elapsed 0:01:09, exp. remaining 1:26:24, complete 1.33%
att-weights epoch 560, step 27, max_size:classes 69, max_size:data 1907, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.183 sec/step, elapsed 0:01:10, exp. remaining 1:25:38, complete 1.36%
att-weights epoch 560, step 28, max_size:classes 68, max_size:data 2022, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.796 sec/step, elapsed 0:01:11, exp. remaining 1:24:27, complete 1.40%
att-weights epoch 560, step 29, max_size:classes 63, max_size:data 1564, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.715 sec/step, elapsed 0:01:14, exp. remaining 1:25:32, complete 1.43%
att-weights epoch 560, step 30, max_size:classes 66, max_size:data 2154, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.903 sec/step, elapsed 0:01:15, exp. remaining 1:24:32, complete 1.46%
att-weights epoch 560, step 31, max_size:classes 69, max_size:data 1730, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.824 sec/step, elapsed 0:01:16, exp. remaining 1:23:29, complete 1.50%
att-weights epoch 560, step 32, max_size:classes 90, max_size:data 2030, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.170 sec/step, elapsed 0:01:17, exp. remaining 1:22:51, complete 1.53%
att-weights epoch 560, step 33, max_size:classes 68, max_size:data 1874, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.971 sec/step, elapsed 0:01:18, exp. remaining 1:20:16, complete 1.60%
att-weights epoch 560, step 34, max_size:classes 62, max_size:data 2075, mem_usage:GPU:0 1.0GB, num_seqs 1, 2.556 sec/step, elapsed 0:01:20, exp. remaining 1:21:08, complete 1.63%
att-weights epoch 560, step 35, max_size:classes 74, max_size:data 1982, mem_usage:GPU:0 1.0GB, num_seqs 1, 9.107 sec/step, elapsed 0:01:29, exp. remaining 1:28:25, complete 1.67%
att-weights epoch 560, step 36, max_size:classes 68, max_size:data 2276, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.313 sec/step, elapsed 0:01:31, exp. remaining 1:26:07, complete 1.74%
att-weights epoch 560, step 37, max_size:classes 67, max_size:data 2127, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.296 sec/step, elapsed 0:01:32, exp. remaining 1:25:38, complete 1.77%
att-weights epoch 560, step 38, max_size:classes 65, max_size:data 2039, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.104 sec/step, elapsed 0:01:33, exp. remaining 1:25:00, complete 1.80%
att-weights epoch 560, step 39, max_size:classes 65, max_size:data 1559, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.092 sec/step, elapsed 0:01:34, exp. remaining 1:24:22, complete 1.84%
att-weights epoch 560, step 40, max_size:classes 66, max_size:data 2304, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.043 sec/step, elapsed 0:01:35, exp. remaining 1:23:43, complete 1.87%
att-weights epoch 560, step 41, max_size:classes 58, max_size:data 2136, mem_usage:GPU:0 1.0GB, num_seqs 1, 5.058 sec/step, elapsed 0:01:40, exp. remaining 1:24:59, complete 1.94%
att-weights epoch 560, step 42, max_size:classes 67, max_size:data 1908, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.742 sec/step, elapsed 0:01:42, exp. remaining 1:24:56, complete 1.97%
att-weights epoch 560, step 43, max_size:classes 72, max_size:data 2457, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.411 sec/step, elapsed 0:01:44, exp. remaining 1:23:10, complete 2.04%
att-weights epoch 560, step 44, max_size:classes 66, max_size:data 2009, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.100 sec/step, elapsed 0:01:45, exp. remaining 1:22:39, complete 2.08%
att-weights epoch 560, step 45, max_size:classes 62, max_size:data 1674, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.516 sec/step, elapsed 0:01:46, exp. remaining 1:21:07, complete 2.14%
att-weights epoch 560, step 46, max_size:classes 67, max_size:data 2085, mem_usage:GPU:0 1.0GB, num_seqs 1, 6.081 sec/step, elapsed 0:01:52, exp. remaining 1:23:03, complete 2.21%
att-weights epoch 560, step 47, max_size:classes 64, max_size:data 1931, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.894 sec/step, elapsed 0:01:53, exp. remaining 1:21:09, complete 2.28%
att-weights epoch 560, step 48, max_size:classes 62, max_size:data 2066, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.039 sec/step, elapsed 0:01:54, exp. remaining 1:19:28, complete 2.35%
att-weights epoch 560, step 49, max_size:classes 60, max_size:data 2001, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.672 sec/step, elapsed 0:01:55, exp. remaining 1:17:38, complete 2.42%
att-weights epoch 560, step 50, max_size:classes 64, max_size:data 1869, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.439 sec/step, elapsed 0:01:56, exp. remaining 1:17:29, complete 2.45%
att-weights epoch 560, step 51, max_size:classes 63, max_size:data 2170, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.069 sec/step, elapsed 0:01:57, exp. remaining 1:17:05, complete 2.48%
att-weights epoch 560, step 52, max_size:classes 60, max_size:data 1923, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.534 sec/step, elapsed 0:01:59, exp. remaining 1:15:57, complete 2.55%
att-weights epoch 560, step 53, max_size:classes 57, max_size:data 2141, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.917 sec/step, elapsed 0:02:00, exp. remaining 1:14:30, complete 2.62%
att-weights epoch 560, step 54, max_size:classes 59, max_size:data 1729, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.306 sec/step, elapsed 0:02:01, exp. remaining 1:13:21, complete 2.69%
att-weights epoch 560, step 55, max_size:classes 59, max_size:data 1827, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.505 sec/step, elapsed 0:02:03, exp. remaining 1:12:23, complete 2.76%
att-weights epoch 560, step 56, max_size:classes 67, max_size:data 1811, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.375 sec/step, elapsed 0:02:04, exp. remaining 1:12:16, complete 2.79%
att-weights epoch 560, step 57, max_size:classes 64, max_size:data 1941, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.257 sec/step, elapsed 0:02:05, exp. remaining 1:11:13, complete 2.86%
att-weights epoch 560, step 58, max_size:classes 67, max_size:data 1699, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.125 sec/step, elapsed 0:02:06, exp. remaining 1:10:59, complete 2.89%
att-weights epoch 560, step 59, max_size:classes 59, max_size:data 1264, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.546 sec/step, elapsed 0:02:07, exp. remaining 1:10:26, complete 2.93%
att-weights epoch 560, step 60, max_size:classes 71, max_size:data 2293, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.045 sec/step, elapsed 0:02:08, exp. remaining 1:10:10, complete 2.96%
att-weights epoch 560, step 61, max_size:classes 65, max_size:data 1922, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.283 sec/step, elapsed 0:02:09, exp. remaining 1:10:02, complete 2.99%
att-weights epoch 560, step 62, max_size:classes 55, max_size:data 1785, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.427 sec/step, elapsed 0:02:11, exp. remaining 1:09:11, complete 3.06%
att-weights epoch 560, step 63, max_size:classes 60, max_size:data 1703, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.371 sec/step, elapsed 0:02:12, exp. remaining 1:08:21, complete 3.13%
att-weights epoch 560, step 64, max_size:classes 61, max_size:data 1814, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.039 sec/step, elapsed 0:02:14, exp. remaining 1:07:52, complete 3.20%
att-weights epoch 560, step 65, max_size:classes 59, max_size:data 2068, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.054 sec/step, elapsed 0:02:15, exp. remaining 1:06:56, complete 3.27%
att-weights epoch 560, step 66, max_size:classes 59, max_size:data 1474, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.886 sec/step, elapsed 0:02:17, exp. remaining 1:06:26, complete 3.33%
att-weights epoch 560, step 67, max_size:classes 56, max_size:data 1337, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.722 sec/step, elapsed 0:02:18, exp. remaining 1:05:24, complete 3.40%
att-weights epoch 560, step 68, max_size:classes 60, max_size:data 2178, mem_usage:GPU:0 1.0GB, num_seqs 1, 14.887 sec/step, elapsed 0:02:33, exp. remaining 1:10:58, complete 3.47%
att-weights epoch 560, step 69, max_size:classes 66, max_size:data 1807, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.064 sec/step, elapsed 0:02:34, exp. remaining 1:10:02, complete 3.54%
att-weights epoch 560, step 70, max_size:classes 56, max_size:data 2126, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.003 sec/step, elapsed 0:02:35, exp. remaining 1:09:07, complete 3.61%
att-weights epoch 560, step 71, max_size:classes 56, max_size:data 1737, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.457 sec/step, elapsed 0:02:36, exp. remaining 1:08:26, complete 3.67%
att-weights epoch 560, step 72, max_size:classes 55, max_size:data 1613, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.608 sec/step, elapsed 0:02:38, exp. remaining 1:07:49, complete 3.74%
att-weights epoch 560, step 73, max_size:classes 62, max_size:data 1674, mem_usage:GPU:0 1.0GB, num_seqs 2, 5.508 sec/step, elapsed 0:02:43, exp. remaining 1:08:53, complete 3.81%
att-weights epoch 560, step 74, max_size:classes 57, max_size:data 1748, mem_usage:GPU:0 1.0GB, num_seqs 2, 6.090 sec/step, elapsed 0:02:49, exp. remaining 1:10:08, complete 3.88%
att-weights epoch 560, step 75, max_size:classes 56, max_size:data 1904, mem_usage:GPU:0 1.0GB, num_seqs 2, 36.236 sec/step, elapsed 0:03:26, exp. remaining 1:23:35, complete 3.95%
att-weights epoch 560, step 76, max_size:classes 53, max_size:data 1554, mem_usage:GPU:0 1.0GB, num_seqs 2, 17.150 sec/step, elapsed 0:03:43, exp. remaining 1:28:56, complete 4.01%
att-weights epoch 560, step 77, max_size:classes 55, max_size:data 1737, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.908 sec/step, elapsed 0:03:46, exp. remaining 1:28:32, complete 4.08%
att-weights epoch 560, step 78, max_size:classes 49, max_size:data 1887, mem_usage:GPU:0 1.0GB, num_seqs 2, 9.590 sec/step, elapsed 0:03:55, exp. remaining 1:30:43, complete 4.15%
att-weights epoch 560, step 79, max_size:classes 53, max_size:data 1824, mem_usage:GPU:0 1.0GB, num_seqs 2, 6.021 sec/step, elapsed 0:04:01, exp. remaining 1:31:28, complete 4.22%
att-weights epoch 560, step 80, max_size:classes 62, max_size:data 1552, mem_usage:GPU:0 1.0GB, num_seqs 2, 25.262 sec/step, elapsed 0:04:27, exp. remaining 1:39:21, complete 4.29%
att-weights epoch 560, step 81, max_size:classes 57, max_size:data 1992, mem_usage:GPU:0 1.0GB, num_seqs 2, 35.566 sec/step, elapsed 0:05:02, exp. remaining 1:50:45, complete 4.36%
att-weights epoch 560, step 82, max_size:classes 60, max_size:data 1831, mem_usage:GPU:0 1.0GB, num_seqs 2, 7.181 sec/step, elapsed 0:05:09, exp. remaining 1:51:33, complete 4.42%
att-weights epoch 560, step 83, max_size:classes 58, max_size:data 1855, mem_usage:GPU:0 1.0GB, num_seqs 2, 57.054 sec/step, elapsed 0:06:06, exp. remaining 2:11:02, complete 4.46%
att-weights epoch 560, step 84, max_size:classes 48, max_size:data 1706, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.965 sec/step, elapsed 0:06:09, exp. remaining 2:11:03, complete 4.49%
att-weights epoch 560, step 85, max_size:classes 55, max_size:data 1676, mem_usage:GPU:0 1.0GB, num_seqs 2, 18.722 sec/step, elapsed 0:06:28, exp. remaining 2:15:32, complete 4.56%
att-weights epoch 560, step 86, max_size:classes 54, max_size:data 1499, mem_usage:GPU:0 1.0GB, num_seqs 2, 12.627 sec/step, elapsed 0:06:41, exp. remaining 2:17:47, complete 4.63%
att-weights epoch 560, step 87, max_size:classes 52, max_size:data 1607, mem_usage:GPU:0 1.0GB, num_seqs 2, 8.665 sec/step, elapsed 0:06:49, exp. remaining 2:18:37, complete 4.70%
att-weights epoch 560, step 88, max_size:classes 51, max_size:data 1855, mem_usage:GPU:0 1.0GB, num_seqs 2, 3.549 sec/step, elapsed 0:06:53, exp. remaining 2:17:44, complete 4.76%
att-weights epoch 560, step 89, max_size:classes 55, max_size:data 1675, mem_usage:GPU:0 1.0GB, num_seqs 2, 24.044 sec/step, elapsed 0:07:17, exp. remaining 2:23:35, complete 4.83%
att-weights epoch 560, step 90, max_size:classes 48, max_size:data 1570, mem_usage:GPU:0 1.0GB, num_seqs 2, 7.003 sec/step, elapsed 0:07:24, exp. remaining 2:23:47, complete 4.90%
att-weights epoch 560, step 91, max_size:classes 48, max_size:data 1874, mem_usage:GPU:0 1.0GB, num_seqs 2, 14.486 sec/step, elapsed 0:07:38, exp. remaining 2:26:20, complete 4.97%
att-weights epoch 560, step 92, max_size:classes 48, max_size:data 1686, mem_usage:GPU:0 1.0GB, num_seqs 1, 14.245 sec/step, elapsed 0:07:53, exp. remaining 2:28:43, complete 5.04%
att-weights epoch 560, step 93, max_size:classes 50, max_size:data 2031, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.533 sec/step, elapsed 0:07:54, exp. remaining 2:28:09, complete 5.07%
att-weights epoch 560, step 94, max_size:classes 61, max_size:data 1966, mem_usage:GPU:0 1.0GB, num_seqs 2, 9.369 sec/step, elapsed 0:08:04, exp. remaining 2:30:01, complete 5.10%
att-weights epoch 560, step 95, max_size:classes 52, max_size:data 1728, mem_usage:GPU:0 1.0GB, num_seqs 2, 16.562 sec/step, elapsed 0:08:20, exp. remaining 2:33:00, complete 5.17%
att-weights epoch 560, step 96, max_size:classes 51, max_size:data 1732, mem_usage:GPU:0 1.0GB, num_seqs 2, 11.623 sec/step, elapsed 0:08:32, exp. remaining 2:34:24, complete 5.24%
att-weights epoch 560, step 97, max_size:classes 49, max_size:data 1607, mem_usage:GPU:0 1.0GB, num_seqs 2, 15.596 sec/step, elapsed 0:08:47, exp. remaining 2:36:57, complete 5.31%
att-weights epoch 560, step 98, max_size:classes 54, max_size:data 1646, mem_usage:GPU:0 1.0GB, num_seqs 2, 7.473 sec/step, elapsed 0:08:55, exp. remaining 2:36:00, complete 5.41%
att-weights epoch 560, step 99, max_size:classes 49, max_size:data 1706, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.499 sec/step, elapsed 0:08:56, exp. remaining 2:34:23, complete 5.48%
att-weights epoch 560, step 100, max_size:classes 45, max_size:data 1517, mem_usage:GPU:0 1.0GB, num_seqs 2, 14.839 sec/step, elapsed 0:09:11, exp. remaining 2:36:36, complete 5.55%
att-weights epoch 560, step 101, max_size:classes 48, max_size:data 1634, mem_usage:GPU:0 1.0GB, num_seqs 2, 6.329 sec/step, elapsed 0:09:18, exp. remaining 2:36:22, complete 5.61%
att-weights epoch 560, step 102, max_size:classes 51, max_size:data 1594, mem_usage:GPU:0 1.0GB, num_seqs 1, 5.179 sec/step, elapsed 0:09:23, exp. remaining 2:35:48, complete 5.68%
att-weights epoch 560, step 103, max_size:classes 53, max_size:data 2006, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.054 sec/step, elapsed 0:09:24, exp. remaining 2:34:08, complete 5.75%
att-weights epoch 560, step 104, max_size:classes 50, max_size:data 1370, mem_usage:GPU:0 1.0GB, num_seqs 2, 4.416 sec/step, elapsed 0:09:28, exp. remaining 2:33:25, complete 5.82%
att-weights epoch 560, step 105, max_size:classes 46, max_size:data 1455, mem_usage:GPU:0 1.0GB, num_seqs 2, 18.230 sec/step, elapsed 0:09:46, exp. remaining 2:36:24, complete 5.89%
att-weights epoch 560, step 106, max_size:classes 48, max_size:data 1673, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.479 sec/step, elapsed 0:09:48, exp. remaining 2:34:53, complete 5.95%
att-weights epoch 560, step 107, max_size:classes 52, max_size:data 1326, mem_usage:GPU:0 1.0GB, num_seqs 3, 2.516 sec/step, elapsed 0:09:50, exp. remaining 2:33:41, complete 6.02%
att-weights epoch 560, step 108, max_size:classes 50, max_size:data 1498, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.109 sec/step, elapsed 0:09:52, exp. remaining 2:32:08, complete 6.09%
att-weights epoch 560, step 109, max_size:classes 47, max_size:data 1258, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.237 sec/step, elapsed 0:09:54, exp. remaining 2:30:55, complete 6.16%
att-weights epoch 560, step 110, max_size:classes 54, max_size:data 1845, mem_usage:GPU:0 1.0GB, num_seqs 2, 5.457 sec/step, elapsed 0:09:59, exp. remaining 2:30:32, complete 6.23%
att-weights epoch 560, step 111, max_size:classes 49, max_size:data 1461, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.931 sec/step, elapsed 0:10:01, exp. remaining 2:29:16, complete 6.29%
att-weights epoch 560, step 112, max_size:classes 49, max_size:data 1588, mem_usage:GPU:0 1.0GB, num_seqs 2, 6.984 sec/step, elapsed 0:10:08, exp. remaining 2:28:26, complete 6.40%
att-weights epoch 560, step 113, max_size:classes 54, max_size:data 1573, mem_usage:GPU:0 1.0GB, num_seqs 2, 10.709 sec/step, elapsed 0:10:19, exp. remaining 2:29:21, complete 6.46%
att-weights epoch 560, step 114, max_size:classes 48, max_size:data 1601, mem_usage:GPU:0 1.0GB, num_seqs 2, 5.382 sec/step, elapsed 0:10:24, exp. remaining 2:28:58, complete 6.53%
att-weights epoch 560, step 115, max_size:classes 55, max_size:data 1652, mem_usage:GPU:0 1.0GB, num_seqs 2, 7.189 sec/step, elapsed 0:10:31, exp. remaining 2:29:01, complete 6.60%
att-weights epoch 560, step 116, max_size:classes 48, max_size:data 1308, mem_usage:GPU:0 1.0GB, num_seqs 2, 12.946 sec/step, elapsed 0:10:44, exp. remaining 2:30:25, complete 6.67%
att-weights epoch 560, step 117, max_size:classes 47, max_size:data 1462, mem_usage:GPU:0 1.0GB, num_seqs 2, 16.701 sec/step, elapsed 0:11:01, exp. remaining 2:32:38, complete 6.74%
att-weights epoch 560, step 118, max_size:classes 48, max_size:data 1716, mem_usage:GPU:0 1.0GB, num_seqs 2, 9.435 sec/step, elapsed 0:11:11, exp. remaining 2:33:09, complete 6.81%
att-weights epoch 560, step 119, max_size:classes 48, max_size:data 1471, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.599 sec/step, elapsed 0:11:13, exp. remaining 2:32:07, complete 6.87%
att-weights epoch 560, step 120, max_size:classes 54, max_size:data 1882, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.847 sec/step, elapsed 0:11:16, exp. remaining 2:31:09, complete 6.94%
att-weights epoch 560, step 121, max_size:classes 52, max_size:data 1285, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.376 sec/step, elapsed 0:11:17, exp. remaining 2:29:52, complete 7.01%
att-weights epoch 560, step 122, max_size:classes 44, max_size:data 1997, mem_usage:GPU:0 1.0GB, num_seqs 2, 21.804 sec/step, elapsed 0:11:39, exp. remaining 2:33:06, complete 7.08%
att-weights epoch 560, step 123, max_size:classes 48, max_size:data 1695, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.209 sec/step, elapsed 0:11:40, exp. remaining 2:31:47, complete 7.15%
att-weights epoch 560, step 124, max_size:classes 53, max_size:data 1792, mem_usage:GPU:0 1.0GB, num_seqs 2, 9.165 sec/step, elapsed 0:11:50, exp. remaining 2:32:13, complete 7.21%
att-weights epoch 560, step 125, max_size:classes 48, max_size:data 1449, mem_usage:GPU:0 1.0GB, num_seqs 2, 22.719 sec/step, elapsed 0:12:12, exp. remaining 2:35:30, complete 7.28%
att-weights epoch 560, step 126, max_size:classes 48, max_size:data 1516, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.260 sec/step, elapsed 0:12:14, exp. remaining 2:34:13, complete 7.35%
att-weights epoch 560, step 127, max_size:classes 65, max_size:data 1781, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.559 sec/step, elapsed 0:12:15, exp. remaining 2:33:01, complete 7.42%
att-weights epoch 560, step 128, max_size:classes 43, max_size:data 1534, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.044 sec/step, elapsed 0:12:16, exp. remaining 2:31:43, complete 7.49%
att-weights epoch 560, step 129, max_size:classes 44, max_size:data 1660, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.516 sec/step, elapsed 0:12:18, exp. remaining 2:30:33, complete 7.55%
att-weights epoch 560, step 130, max_size:classes 51, max_size:data 1670, mem_usage:GPU:0 1.0GB, num_seqs 2, 10.069 sec/step, elapsed 0:12:28, exp. remaining 2:31:08, complete 7.62%
att-weights epoch 560, step 131, max_size:classes 52, max_size:data 1941, mem_usage:GPU:0 1.0GB, num_seqs 2, 16.663 sec/step, elapsed 0:12:44, exp. remaining 2:33:01, complete 7.69%
att-weights epoch 560, step 132, max_size:classes 49, max_size:data 1864, mem_usage:GPU:0 1.0GB, num_seqs 2, 21.485 sec/step, elapsed 0:13:06, exp. remaining 2:35:49, complete 7.76%
att-weights epoch 560, step 133, max_size:classes 49, max_size:data 1311, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.437 sec/step, elapsed 0:13:07, exp. remaining 2:34:38, complete 7.83%
att-weights epoch 560, step 134, max_size:classes 48, max_size:data 1426, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.087 sec/step, elapsed 0:13:08, exp. remaining 2:33:24, complete 7.89%
att-weights epoch 560, step 135, max_size:classes 45, max_size:data 1368, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.392 sec/step, elapsed 0:13:10, exp. remaining 2:32:15, complete 7.96%
att-weights epoch 560, step 136, max_size:classes 48, max_size:data 1527, mem_usage:GPU:0 1.0GB, num_seqs 2, 6.210 sec/step, elapsed 0:13:16, exp. remaining 2:32:02, complete 8.03%
att-weights epoch 560, step 137, max_size:classes 51, max_size:data 1157, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.912 sec/step, elapsed 0:13:18, exp. remaining 2:31:00, complete 8.10%
att-weights epoch 560, step 138, max_size:classes 45, max_size:data 1407, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.434 sec/step, elapsed 0:13:19, exp. remaining 2:29:54, complete 8.17%
att-weights epoch 560, step 139, max_size:classes 47, max_size:data 1681, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.860 sec/step, elapsed 0:13:21, exp. remaining 2:28:54, complete 8.23%
att-weights epoch 560, step 140, max_size:classes 49, max_size:data 1462, mem_usage:GPU:0 1.0GB, num_seqs 2, 7.115 sec/step, elapsed 0:13:28, exp. remaining 2:28:13, complete 8.34%
att-weights epoch 560, step 141, max_size:classes 52, max_size:data 1474, mem_usage:GPU:0 1.0GB, num_seqs 2, 7.647 sec/step, elapsed 0:13:36, exp. remaining 2:28:18, complete 8.40%
att-weights epoch 560, step 142, max_size:classes 50, max_size:data 1672, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.040 sec/step, elapsed 0:13:38, exp. remaining 2:27:22, complete 8.47%
att-weights epoch 560, step 143, max_size:classes 45, max_size:data 1631, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.260 sec/step, elapsed 0:13:39, exp. remaining 2:26:18, complete 8.54%
att-weights epoch 560, step 144, max_size:classes 46, max_size:data 1300, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.903 sec/step, elapsed 0:13:40, exp. remaining 2:25:12, complete 8.61%
att-weights epoch 560, step 145, max_size:classes 46, max_size:data 1694, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.050 sec/step, elapsed 0:13:41, exp. remaining 2:23:31, complete 8.71%
att-weights epoch 560, step 146, max_size:classes 50, max_size:data 1727, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.132 sec/step, elapsed 0:13:42, exp. remaining 2:22:30, complete 8.78%
att-weights epoch 560, step 147, max_size:classes 50, max_size:data 1384, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.204 sec/step, elapsed 0:13:44, exp. remaining 2:21:30, complete 8.85%
att-weights epoch 560, step 148, max_size:classes 41, max_size:data 1592, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.080 sec/step, elapsed 0:13:45, exp. remaining 2:20:30, complete 8.91%
att-weights epoch 560, step 149, max_size:classes 47, max_size:data 1314, mem_usage:GPU:0 1.0GB, num_seqs 3, 3.557 sec/step, elapsed 0:13:48, exp. remaining 2:19:56, complete 8.98%
att-weights epoch 560, step 150, max_size:classes 40, max_size:data 1227, mem_usage:GPU:0 1.0GB, num_seqs 2, 8.615 sec/step, elapsed 0:13:57, exp. remaining 2:20:13, complete 9.05%
att-weights epoch 560, step 151, max_size:classes 47, max_size:data 1693, mem_usage:GPU:0 1.0GB, num_seqs 2, 18.801 sec/step, elapsed 0:14:16, exp. remaining 2:21:37, complete 9.15%
att-weights epoch 560, step 152, max_size:classes 50, max_size:data 1357, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.236 sec/step, elapsed 0:14:17, exp. remaining 2:20:40, complete 9.22%
att-weights epoch 560, step 153, max_size:classes 42, max_size:data 1406, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.967 sec/step, elapsed 0:14:18, exp. remaining 2:19:08, complete 9.32%
att-weights epoch 560, step 154, max_size:classes 40, max_size:data 1263, mem_usage:GPU:0 1.0GB, num_seqs 3, 7.294 sec/step, elapsed 0:14:25, exp. remaining 2:19:11, complete 9.39%
att-weights epoch 560, step 155, max_size:classes 41, max_size:data 1413, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.377 sec/step, elapsed 0:14:27, exp. remaining 2:18:28, complete 9.46%
att-weights epoch 560, step 156, max_size:classes 44, max_size:data 1729, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.419 sec/step, elapsed 0:14:29, exp. remaining 2:17:36, complete 9.53%
att-weights epoch 560, step 157, max_size:classes 47, max_size:data 1353, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.128 sec/step, elapsed 0:14:30, exp. remaining 2:16:42, complete 9.60%
att-weights epoch 560, step 158, max_size:classes 46, max_size:data 1807, mem_usage:GPU:0 1.0GB, num_seqs 2, 7.300 sec/step, elapsed 0:14:37, exp. remaining 2:16:46, complete 9.66%
att-weights epoch 560, step 159, max_size:classes 46, max_size:data 1480, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.289 sec/step, elapsed 0:14:40, exp. remaining 2:16:04, complete 9.73%
att-weights epoch 560, step 160, max_size:classes 48, max_size:data 1285, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.698 sec/step, elapsed 0:14:41, exp. remaining 2:15:16, complete 9.80%
att-weights epoch 560, step 161, max_size:classes 42, max_size:data 1387, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.002 sec/step, elapsed 0:14:42, exp. remaining 2:14:24, complete 9.87%
att-weights epoch 560, step 162, max_size:classes 41, max_size:data 1326, mem_usage:GPU:0 1.0GB, num_seqs 3, 2.607 sec/step, elapsed 0:14:45, exp. remaining 2:13:46, complete 9.94%
att-weights epoch 560, step 163, max_size:classes 44, max_size:data 1361, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.178 sec/step, elapsed 0:14:46, exp. remaining 2:12:56, complete 10.00%
att-weights epoch 560, step 164, max_size:classes 44, max_size:data 1354, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.072 sec/step, elapsed 0:14:47, exp. remaining 2:11:36, complete 10.11%
att-weights epoch 560, step 165, max_size:classes 45, max_size:data 1700, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.374 sec/step, elapsed 0:14:49, exp. remaining 2:10:49, complete 10.17%
att-weights epoch 560, step 166, max_size:classes 47, max_size:data 1425, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.084 sec/step, elapsed 0:14:50, exp. remaining 2:10:01, complete 10.24%
att-weights epoch 560, step 167, max_size:classes 50, max_size:data 1689, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.983 sec/step, elapsed 0:14:51, exp. remaining 2:08:43, complete 10.34%
att-weights epoch 560, step 168, max_size:classes 41, max_size:data 1363, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.088 sec/step, elapsed 0:14:52, exp. remaining 2:07:29, complete 10.45%
att-weights epoch 560, step 169, max_size:classes 55, max_size:data 1361, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.224 sec/step, elapsed 0:14:53, exp. remaining 2:06:44, complete 10.51%
att-weights epoch 560, step 170, max_size:classes 37, max_size:data 1332, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.339 sec/step, elapsed 0:14:54, exp. remaining 2:06:00, complete 10.58%
att-weights epoch 560, step 171, max_size:classes 41, max_size:data 1590, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.298 sec/step, elapsed 0:14:56, exp. remaining 2:05:17, complete 10.65%
att-weights epoch 560, step 172, max_size:classes 44, max_size:data 1429, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.062 sec/step, elapsed 0:14:57, exp. remaining 2:04:06, complete 10.75%
att-weights epoch 560, step 173, max_size:classes 52, max_size:data 1322, mem_usage:GPU:0 1.0GB, num_seqs 3, 9.257 sec/step, elapsed 0:15:06, exp. remaining 2:04:04, complete 10.85%
att-weights epoch 560, step 174, max_size:classes 45, max_size:data 1114, mem_usage:GPU:0 1.0GB, num_seqs 2, 6.621 sec/step, elapsed 0:15:13, exp. remaining 2:03:40, complete 10.96%
att-weights epoch 560, step 175, max_size:classes 40, max_size:data 1727, mem_usage:GPU:0 1.0GB, num_seqs 2, 9.765 sec/step, elapsed 0:15:22, exp. remaining 2:04:07, complete 11.02%
att-weights epoch 560, step 176, max_size:classes 45, max_size:data 1307, mem_usage:GPU:0 1.0GB, num_seqs 3, 23.232 sec/step, elapsed 0:15:45, exp. remaining 2:05:56, complete 11.13%
att-weights epoch 560, step 177, max_size:classes 42, max_size:data 1297, mem_usage:GPU:0 1.0GB, num_seqs 3, 9.167 sec/step, elapsed 0:15:55, exp. remaining 2:05:51, complete 11.23%
att-weights epoch 560, step 178, max_size:classes 42, max_size:data 1575, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.685 sec/step, elapsed 0:15:56, exp. remaining 2:05:13, complete 11.30%
att-weights epoch 560, step 179, max_size:classes 46, max_size:data 1223, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.441 sec/step, elapsed 0:15:58, exp. remaining 2:04:34, complete 11.36%
att-weights epoch 560, step 180, max_size:classes 46, max_size:data 1538, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.522 sec/step, elapsed 0:16:00, exp. remaining 2:04:03, complete 11.43%
att-weights epoch 560, step 181, max_size:classes 41, max_size:data 1182, mem_usage:GPU:0 1.0GB, num_seqs 3, 3.270 sec/step, elapsed 0:16:04, exp. remaining 2:03:38, complete 11.50%
att-weights epoch 560, step 182, max_size:classes 42, max_size:data 1135, mem_usage:GPU:0 1.0GB, num_seqs 3, 22.938 sec/step, elapsed 0:16:27, exp. remaining 2:05:19, complete 11.60%
att-weights epoch 560, step 183, max_size:classes 45, max_size:data 1333, mem_usage:GPU:0 1.0GB, num_seqs 3, 2.494 sec/step, elapsed 0:16:29, exp. remaining 2:04:49, complete 11.67%
att-weights epoch 560, step 184, max_size:classes 49, max_size:data 1580, mem_usage:GPU:0 1.0GB, num_seqs 2, 44.656 sec/step, elapsed 0:17:14, exp. remaining 2:09:35, complete 11.74%
att-weights epoch 560, step 185, max_size:classes 42, max_size:data 1126, mem_usage:GPU:0 1.0GB, num_seqs 3, 2.321 sec/step, elapsed 0:17:16, exp. remaining 2:09:02, complete 11.81%
att-weights epoch 560, step 186, max_size:classes 42, max_size:data 1170, mem_usage:GPU:0 1.0GB, num_seqs 3, 2.205 sec/step, elapsed 0:17:18, exp. remaining 2:08:28, complete 11.87%
att-weights epoch 560, step 187, max_size:classes 38, max_size:data 1559, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.104 sec/step, elapsed 0:17:19, exp. remaining 2:07:22, complete 11.98%
att-weights epoch 560, step 188, max_size:classes 40, max_size:data 1344, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.289 sec/step, elapsed 0:17:21, exp. remaining 2:06:18, complete 12.08%
att-weights epoch 560, step 189, max_size:classes 41, max_size:data 1310, mem_usage:GPU:0 1.0GB, num_seqs 2, 17.527 sec/step, elapsed 0:17:38, exp. remaining 2:07:36, complete 12.15%
att-weights epoch 560, step 190, max_size:classes 40, max_size:data 1655, mem_usage:GPU:0 1.0GB, num_seqs 2, 9.305 sec/step, elapsed 0:17:47, exp. remaining 2:07:54, complete 12.22%
att-weights epoch 560, step 191, max_size:classes 42, max_size:data 1259, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.497 sec/step, elapsed 0:17:49, exp. remaining 2:07:17, complete 12.28%
att-weights epoch 560, step 192, max_size:classes 39, max_size:data 1455, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.457 sec/step, elapsed 0:17:50, exp. remaining 2:06:39, complete 12.35%
att-weights epoch 560, step 193, max_size:classes 39, max_size:data 1109, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.171 sec/step, elapsed 0:17:52, exp. remaining 2:06:00, complete 12.42%
att-weights epoch 560, step 194, max_size:classes 44, max_size:data 1406, mem_usage:GPU:0 1.0GB, num_seqs 2, 7.240 sec/step, elapsed 0:17:59, exp. remaining 2:05:40, complete 12.52%
att-weights epoch 560, step 195, max_size:classes 40, max_size:data 1377, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.145 sec/step, elapsed 0:18:00, exp. remaining 2:04:38, complete 12.62%
att-weights epoch 560, step 196, max_size:classes 40, max_size:data 1249, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.428 sec/step, elapsed 0:18:01, exp. remaining 2:03:39, complete 12.73%
att-weights epoch 560, step 197, max_size:classes 42, max_size:data 1316, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.331 sec/step, elapsed 0:18:03, exp. remaining 2:02:41, complete 12.83%
att-weights epoch 560, step 198, max_size:classes 42, max_size:data 1311, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.171 sec/step, elapsed 0:18:04, exp. remaining 2:01:42, complete 12.93%
att-weights epoch 560, step 199, max_size:classes 38, max_size:data 1509, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.305 sec/step, elapsed 0:18:05, exp. remaining 2:00:45, complete 13.03%
att-weights epoch 560, step 200, max_size:classes 35, max_size:data 1361, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.058 sec/step, elapsed 0:18:06, exp. remaining 1:59:47, complete 13.13%
att-weights epoch 560, step 201, max_size:classes 40, max_size:data 1408, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.126 sec/step, elapsed 0:18:07, exp. remaining 1:58:51, complete 13.24%
att-weights epoch 560, step 202, max_size:classes 42, max_size:data 1378, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.231 sec/step, elapsed 0:18:09, exp. remaining 1:57:56, complete 13.34%
att-weights epoch 560, step 203, max_size:classes 38, max_size:data 1314, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.644 sec/step, elapsed 0:18:10, exp. remaining 1:57:04, complete 13.44%
att-weights epoch 560, step 204, max_size:classes 36, max_size:data 1184, mem_usage:GPU:0 1.0GB, num_seqs 3, 3.549 sec/step, elapsed 0:18:14, exp. remaining 1:56:26, complete 13.54%
att-weights epoch 560, step 205, max_size:classes 35, max_size:data 1293, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.630 sec/step, elapsed 0:18:15, exp. remaining 1:55:36, complete 13.64%
att-weights epoch 560, step 206, max_size:classes 38, max_size:data 1163, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.659 sec/step, elapsed 0:18:17, exp. remaining 1:54:47, complete 13.75%
att-weights epoch 560, step 207, max_size:classes 47, max_size:data 1236, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.380 sec/step, elapsed 0:18:18, exp. remaining 1:53:56, complete 13.85%
att-weights epoch 560, step 208, max_size:classes 39, max_size:data 1167, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.204 sec/step, elapsed 0:18:20, exp. remaining 1:53:06, complete 13.95%
att-weights epoch 560, step 209, max_size:classes 39, max_size:data 1145, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.470 sec/step, elapsed 0:18:21, exp. remaining 1:52:36, complete 14.02%
att-weights epoch 560, step 210, max_size:classes 38, max_size:data 1074, mem_usage:GPU:0 1.0GB, num_seqs 3, 6.192 sec/step, elapsed 0:18:27, exp. remaining 1:52:36, complete 14.09%
att-weights epoch 560, step 211, max_size:classes 39, max_size:data 1304, mem_usage:GPU:0 1.0GB, num_seqs 3, 6.581 sec/step, elapsed 0:18:34, exp. remaining 1:52:19, complete 14.19%
att-weights epoch 560, step 212, max_size:classes 43, max_size:data 1169, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.407 sec/step, elapsed 0:18:35, exp. remaining 1:51:50, complete 14.26%
att-weights epoch 560, step 213, max_size:classes 35, max_size:data 1014, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.463 sec/step, elapsed 0:18:37, exp. remaining 1:51:22, complete 14.32%
att-weights epoch 560, step 214, max_size:classes 41, max_size:data 1244, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.360 sec/step, elapsed 0:18:38, exp. remaining 1:50:35, complete 14.43%
att-weights epoch 560, step 215, max_size:classes 34, max_size:data 1162, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.511 sec/step, elapsed 0:18:40, exp. remaining 1:49:49, complete 14.53%
att-weights epoch 560, step 216, max_size:classes 36, max_size:data 1233, mem_usage:GPU:0 1.0GB, num_seqs 3, 3.932 sec/step, elapsed 0:18:44, exp. remaining 1:49:18, complete 14.63%
att-weights epoch 560, step 217, max_size:classes 36, max_size:data 1226, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.631 sec/step, elapsed 0:18:45, exp. remaining 1:48:52, complete 14.70%
att-weights epoch 560, step 218, max_size:classes 43, max_size:data 1368, mem_usage:GPU:0 1.0GB, num_seqs 2, 3.869 sec/step, elapsed 0:18:49, exp. remaining 1:48:39, complete 14.77%
att-weights epoch 560, step 219, max_size:classes 42, max_size:data 1496, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.250 sec/step, elapsed 0:18:50, exp. remaining 1:47:54, complete 14.87%
att-weights epoch 560, step 220, max_size:classes 42, max_size:data 993, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.634 sec/step, elapsed 0:18:52, exp. remaining 1:47:11, complete 14.97%
att-weights epoch 560, step 221, max_size:classes 38, max_size:data 1313, mem_usage:GPU:0 1.0GB, num_seqs 2, 10.027 sec/step, elapsed 0:19:02, exp. remaining 1:47:17, complete 15.07%
att-weights epoch 560, step 222, max_size:classes 37, max_size:data 1372, mem_usage:GPU:0 1.0GB, num_seqs 2, 6.147 sec/step, elapsed 0:19:08, exp. remaining 1:47:00, complete 15.18%
att-weights epoch 560, step 223, max_size:classes 40, max_size:data 1028, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.486 sec/step, elapsed 0:19:10, exp. remaining 1:46:18, complete 15.28%
att-weights epoch 560, step 224, max_size:classes 36, max_size:data 1208, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.106 sec/step, elapsed 0:19:11, exp. remaining 1:45:51, complete 15.35%
att-weights epoch 560, step 225, max_size:classes 36, max_size:data 1169, mem_usage:GPU:0 1.0GB, num_seqs 3, 2.775 sec/step, elapsed 0:19:14, exp. remaining 1:45:16, complete 15.45%
att-weights epoch 560, step 226, max_size:classes 37, max_size:data 1149, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.539 sec/step, elapsed 0:19:15, exp. remaining 1:44:52, complete 15.52%
att-weights epoch 560, step 227, max_size:classes 37, max_size:data 1361, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.306 sec/step, elapsed 0:19:16, exp. remaining 1:44:10, complete 15.62%
att-weights epoch 560, step 228, max_size:classes 36, max_size:data 1053, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.547 sec/step, elapsed 0:19:18, exp. remaining 1:43:14, complete 15.75%
att-weights epoch 560, step 229, max_size:classes 35, max_size:data 1023, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.414 sec/step, elapsed 0:19:19, exp. remaining 1:42:35, complete 15.86%
att-weights epoch 560, step 230, max_size:classes 37, max_size:data 1057, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.449 sec/step, elapsed 0:19:21, exp. remaining 1:41:55, complete 15.96%
att-weights epoch 560, step 231, max_size:classes 35, max_size:data 1209, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.934 sec/step, elapsed 0:19:23, exp. remaining 1:41:19, complete 16.06%
att-weights epoch 560, step 232, max_size:classes 34, max_size:data 1023, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.284 sec/step, elapsed 0:19:24, exp. remaining 1:40:25, complete 16.20%
att-weights epoch 560, step 233, max_size:classes 39, max_size:data 1542, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.287 sec/step, elapsed 0:19:25, exp. remaining 1:39:47, complete 16.30%
att-weights epoch 560, step 234, max_size:classes 34, max_size:data 1155, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.451 sec/step, elapsed 0:19:27, exp. remaining 1:39:10, complete 16.40%
att-weights epoch 560, step 235, max_size:classes 31, max_size:data 1425, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.049 sec/step, elapsed 0:19:28, exp. remaining 1:38:31, complete 16.50%
att-weights epoch 560, step 236, max_size:classes 35, max_size:data 1073, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.012 sec/step, elapsed 0:19:29, exp. remaining 1:37:38, complete 16.64%
att-weights epoch 560, step 237, max_size:classes 35, max_size:data 992, mem_usage:GPU:0 1.0GB, num_seqs 4, 2.138 sec/step, elapsed 0:19:31, exp. remaining 1:37:06, complete 16.74%
att-weights epoch 560, step 238, max_size:classes 34, max_size:data 1110, mem_usage:GPU:0 1.0GB, num_seqs 3, 3.179 sec/step, elapsed 0:19:34, exp. remaining 1:36:53, complete 16.81%
att-weights epoch 560, step 239, max_size:classes 36, max_size:data 813, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.552 sec/step, elapsed 0:19:36, exp. remaining 1:36:05, complete 16.94%
att-weights epoch 560, step 240, max_size:classes 37, max_size:data 1236, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.887 sec/step, elapsed 0:19:38, exp. remaining 1:35:32, complete 17.05%
att-weights epoch 560, step 241, max_size:classes 37, max_size:data 941, mem_usage:GPU:0 1.0GB, num_seqs 4, 2.110 sec/step, elapsed 0:19:40, exp. remaining 1:35:01, complete 17.15%
att-weights epoch 560, step 242, max_size:classes 35, max_size:data 1091, mem_usage:GPU:0 1.0GB, num_seqs 3, 2.112 sec/step, elapsed 0:19:42, exp. remaining 1:34:31, complete 17.25%
att-weights epoch 560, step 243, max_size:classes 35, max_size:data 1181, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.987 sec/step, elapsed 0:19:44, exp. remaining 1:34:13, complete 17.32%
att-weights epoch 560, step 244, max_size:classes 34, max_size:data 1062, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.184 sec/step, elapsed 0:19:45, exp. remaining 1:33:39, complete 17.42%
att-weights epoch 560, step 245, max_size:classes 39, max_size:data 967, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.393 sec/step, elapsed 0:19:46, exp. remaining 1:33:06, complete 17.52%
att-weights epoch 560, step 246, max_size:classes 33, max_size:data 1087, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.441 sec/step, elapsed 0:19:48, exp. remaining 1:32:33, complete 17.63%
att-weights epoch 560, step 247, max_size:classes 35, max_size:data 1351, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.316 sec/step, elapsed 0:19:49, exp. remaining 1:32:13, complete 17.69%
att-weights epoch 560, step 248, max_size:classes 43, max_size:data 955, mem_usage:GPU:0 1.0GB, num_seqs 4, 3.424 sec/step, elapsed 0:19:53, exp. remaining 1:31:51, complete 17.80%
att-weights epoch 560, step 249, max_size:classes 35, max_size:data 885, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.597 sec/step, elapsed 0:19:54, exp. remaining 1:31:20, complete 17.90%
att-weights epoch 560, step 250, max_size:classes 30, max_size:data 1070, mem_usage:GPU:0 1.0GB, num_seqs 3, 2.914 sec/step, elapsed 0:19:57, exp. remaining 1:30:55, complete 18.00%
att-weights epoch 560, step 251, max_size:classes 36, max_size:data 1093, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.591 sec/step, elapsed 0:19:59, exp. remaining 1:30:25, complete 18.10%
att-weights epoch 560, step 252, max_size:classes 35, max_size:data 1501, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.829 sec/step, elapsed 0:20:01, exp. remaining 1:30:00, complete 18.20%
att-weights epoch 560, step 253, max_size:classes 33, max_size:data 1073, mem_usage:GPU:0 1.0GB, num_seqs 3, 3.024 sec/step, elapsed 0:20:04, exp. remaining 1:29:37, complete 18.31%
att-weights epoch 560, step 254, max_size:classes 34, max_size:data 1194, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.586 sec/step, elapsed 0:20:06, exp. remaining 1:29:08, complete 18.41%
att-weights epoch 560, step 255, max_size:classes 34, max_size:data 1070, mem_usage:GPU:0 1.0GB, num_seqs 3, 11.704 sec/step, elapsed 0:20:18, exp. remaining 1:29:23, complete 18.51%
att-weights epoch 560, step 256, max_size:classes 40, max_size:data 1349, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.347 sec/step, elapsed 0:20:19, exp. remaining 1:28:53, complete 18.61%
att-weights epoch 560, step 257, max_size:classes 34, max_size:data 1244, mem_usage:GPU:0 1.0GB, num_seqs 3, 7.325 sec/step, elapsed 0:20:26, exp. remaining 1:28:49, complete 18.71%
att-weights epoch 560, step 258, max_size:classes 32, max_size:data 1103, mem_usage:GPU:0 1.0GB, num_seqs 3, 6.855 sec/step, elapsed 0:20:33, exp. remaining 1:28:31, complete 18.85%
att-weights epoch 560, step 259, max_size:classes 37, max_size:data 892, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.318 sec/step, elapsed 0:20:35, exp. remaining 1:27:50, complete 18.99%
att-weights epoch 560, step 260, max_size:classes 35, max_size:data 1179, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.679 sec/step, elapsed 0:20:36, exp. remaining 1:27:22, complete 19.09%
att-weights epoch 560, step 261, max_size:classes 34, max_size:data 1249, mem_usage:GPU:0 1.0GB, num_seqs 3, 3.833 sec/step, elapsed 0:20:40, exp. remaining 1:27:04, complete 19.19%
att-weights epoch 560, step 262, max_size:classes 33, max_size:data 1049, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.355 sec/step, elapsed 0:20:41, exp. remaining 1:26:35, complete 19.29%
att-weights epoch 560, step 263, max_size:classes 37, max_size:data 1002, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.444 sec/step, elapsed 0:20:43, exp. remaining 1:26:07, complete 19.39%
att-weights epoch 560, step 264, max_size:classes 36, max_size:data 1111, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.557 sec/step, elapsed 0:20:44, exp. remaining 1:25:29, complete 19.53%
att-weights epoch 560, step 265, max_size:classes 31, max_size:data 813, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.213 sec/step, elapsed 0:20:46, exp. remaining 1:25:01, complete 19.63%
att-weights epoch 560, step 266, max_size:classes 34, max_size:data 1082, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.599 sec/step, elapsed 0:20:47, exp. remaining 1:24:35, complete 19.73%
att-weights epoch 560, step 267, max_size:classes 38, max_size:data 921, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.982 sec/step, elapsed 0:20:49, exp. remaining 1:24:10, complete 19.84%
att-weights epoch 560, step 268, max_size:classes 31, max_size:data 973, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.845 sec/step, elapsed 0:20:51, exp. remaining 1:23:45, complete 19.94%
att-weights epoch 560, step 269, max_size:classes 36, max_size:data 1282, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.643 sec/step, elapsed 0:20:53, exp. remaining 1:23:09, complete 20.07%
att-weights epoch 560, step 270, max_size:classes 34, max_size:data 1149, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.268 sec/step, elapsed 0:20:54, exp. remaining 1:22:32, complete 20.21%
att-weights epoch 560, step 271, max_size:classes 33, max_size:data 1180, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.450 sec/step, elapsed 0:20:56, exp. remaining 1:22:07, complete 20.31%
att-weights epoch 560, step 272, max_size:classes 37, max_size:data 1031, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.939 sec/step, elapsed 0:20:57, exp. remaining 1:21:43, complete 20.42%
att-weights epoch 560, step 273, max_size:classes 32, max_size:data 995, mem_usage:GPU:0 1.0GB, num_seqs 4, 2.464 sec/step, elapsed 0:21:00, exp. remaining 1:21:12, complete 20.55%
att-weights epoch 560, step 274, max_size:classes 30, max_size:data 1052, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.348 sec/step, elapsed 0:21:01, exp. remaining 1:20:47, complete 20.65%
att-weights epoch 560, step 275, max_size:classes 32, max_size:data 1096, mem_usage:GPU:0 1.0GB, num_seqs 3, 4.891 sec/step, elapsed 0:21:06, exp. remaining 1:20:26, complete 20.79%
att-weights epoch 560, step 276, max_size:classes 31, max_size:data 954, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.586 sec/step, elapsed 0:21:08, exp. remaining 1:20:02, complete 20.89%
att-weights epoch 560, step 277, max_size:classes 33, max_size:data 1193, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.214 sec/step, elapsed 0:21:09, exp. remaining 1:19:27, complete 21.03%
att-weights epoch 560, step 278, max_size:classes 30, max_size:data 869, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.883 sec/step, elapsed 0:21:11, exp. remaining 1:18:55, complete 21.16%
att-weights epoch 560, step 279, max_size:classes 34, max_size:data 946, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.494 sec/step, elapsed 0:21:12, exp. remaining 1:18:42, complete 21.23%
att-weights epoch 560, step 280, max_size:classes 31, max_size:data 1039, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.469 sec/step, elapsed 0:21:14, exp. remaining 1:18:18, complete 21.33%
att-weights epoch 560, step 281, max_size:classes 36, max_size:data 1290, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.721 sec/step, elapsed 0:21:16, exp. remaining 1:17:56, complete 21.44%
att-weights epoch 560, step 282, max_size:classes 32, max_size:data 969, mem_usage:GPU:0 1.0GB, num_seqs 4, 4.273 sec/step, elapsed 0:21:20, exp. remaining 1:17:34, complete 21.57%
att-weights epoch 560, step 283, max_size:classes 30, max_size:data 1056, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.574 sec/step, elapsed 0:21:21, exp. remaining 1:17:12, complete 21.67%
att-weights epoch 560, step 284, max_size:classes 30, max_size:data 1000, mem_usage:GPU:0 1.0GB, num_seqs 4, 2.101 sec/step, elapsed 0:21:23, exp. remaining 1:16:43, complete 21.81%
att-weights epoch 560, step 285, max_size:classes 32, max_size:data 1021, mem_usage:GPU:0 1.0GB, num_seqs 3, 3.959 sec/step, elapsed 0:21:27, exp. remaining 1:16:29, complete 21.91%
att-weights epoch 560, step 286, max_size:classes 37, max_size:data 953, mem_usage:GPU:0 1.0GB, num_seqs 4, 8.399 sec/step, elapsed 0:21:36, exp. remaining 1:16:23, complete 22.05%
att-weights epoch 560, step 287, max_size:classes 36, max_size:data 930, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.893 sec/step, elapsed 0:21:38, exp. remaining 1:15:53, complete 22.18%
att-weights epoch 560, step 288, max_size:classes 35, max_size:data 1458, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.099 sec/step, elapsed 0:21:39, exp. remaining 1:15:30, complete 22.29%
att-weights epoch 560, step 289, max_size:classes 31, max_size:data 1017, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.528 sec/step, elapsed 0:21:40, exp. remaining 1:15:09, complete 22.39%
att-weights epoch 560, step 290, max_size:classes 31, max_size:data 1164, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.467 sec/step, elapsed 0:21:42, exp. remaining 1:14:48, complete 22.49%
att-weights epoch 560, step 291, max_size:classes 30, max_size:data 957, mem_usage:GPU:0 1.0GB, num_seqs 4, 7.249 sec/step, elapsed 0:21:49, exp. remaining 1:14:38, complete 22.63%
att-weights epoch 560, step 292, max_size:classes 33, max_size:data 1015, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.631 sec/step, elapsed 0:21:51, exp. remaining 1:14:09, complete 22.76%
att-weights epoch 560, step 293, max_size:classes 31, max_size:data 997, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.715 sec/step, elapsed 0:21:52, exp. remaining 1:13:49, complete 22.86%
att-weights epoch 560, step 294, max_size:classes 30, max_size:data 1153, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.326 sec/step, elapsed 0:21:54, exp. remaining 1:13:28, complete 22.97%
att-weights epoch 560, step 295, max_size:classes 34, max_size:data 926, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.443 sec/step, elapsed 0:21:55, exp. remaining 1:12:59, complete 23.10%
att-weights epoch 560, step 296, max_size:classes 35, max_size:data 979, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.459 sec/step, elapsed 0:21:57, exp. remaining 1:12:30, complete 23.24%
att-weights epoch 560, step 297, max_size:classes 31, max_size:data 849, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.174 sec/step, elapsed 0:21:58, exp. remaining 1:12:09, complete 23.34%
att-weights epoch 560, step 298, max_size:classes 29, max_size:data 1096, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.154 sec/step, elapsed 0:21:59, exp. remaining 1:11:48, complete 23.44%
att-weights epoch 560, step 299, max_size:classes 35, max_size:data 1121, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.500 sec/step, elapsed 0:22:00, exp. remaining 1:11:29, complete 23.55%
att-weights epoch 560, step 300, max_size:classes 32, max_size:data 962, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.821 sec/step, elapsed 0:22:02, exp. remaining 1:11:10, complete 23.65%
att-weights epoch 560, step 301, max_size:classes 36, max_size:data 998, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.933 sec/step, elapsed 0:22:04, exp. remaining 1:10:45, complete 23.78%
att-weights epoch 560, step 302, max_size:classes 30, max_size:data 1071, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.614 sec/step, elapsed 0:22:06, exp. remaining 1:10:18, complete 23.92%
att-weights epoch 560, step 303, max_size:classes 28, max_size:data 1084, mem_usage:GPU:0 1.0GB, num_seqs 3, 2.589 sec/step, elapsed 0:22:08, exp. remaining 1:09:55, complete 24.06%
att-weights epoch 560, step 304, max_size:classes 30, max_size:data 777, mem_usage:GPU:0 1.0GB, num_seqs 4, 5.530 sec/step, elapsed 0:22:14, exp. remaining 1:09:41, complete 24.19%
att-weights epoch 560, step 305, max_size:classes 30, max_size:data 863, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.758 sec/step, elapsed 0:22:16, exp. remaining 1:09:23, complete 24.29%
att-weights epoch 560, step 306, max_size:classes 29, max_size:data 1031, mem_usage:GPU:0 1.0GB, num_seqs 3, 2.049 sec/step, elapsed 0:22:18, exp. remaining 1:09:07, complete 24.40%
att-weights epoch 560, step 307, max_size:classes 29, max_size:data 701, mem_usage:GPU:0 1.0GB, num_seqs 3, 3.577 sec/step, elapsed 0:22:21, exp. remaining 1:08:55, complete 24.50%
att-weights epoch 560, step 308, max_size:classes 28, max_size:data 1025, mem_usage:GPU:0 1.0GB, num_seqs 3, 2.450 sec/step, elapsed 0:22:24, exp. remaining 1:08:32, complete 24.63%
att-weights epoch 560, step 309, max_size:classes 41, max_size:data 1070, mem_usage:GPU:0 1.0GB, num_seqs 3, 2.436 sec/step, elapsed 0:22:26, exp. remaining 1:08:17, complete 24.74%
att-weights epoch 560, step 310, max_size:classes 27, max_size:data 871, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.646 sec/step, elapsed 0:22:28, exp. remaining 1:08:00, complete 24.84%
att-weights epoch 560, step 311, max_size:classes 30, max_size:data 836, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.483 sec/step, elapsed 0:22:29, exp. remaining 1:07:42, complete 24.94%
att-weights epoch 560, step 312, max_size:classes 32, max_size:data 892, mem_usage:GPU:0 1.0GB, num_seqs 4, 4.135 sec/step, elapsed 0:22:33, exp. remaining 1:07:25, complete 25.08%
att-weights epoch 560, step 313, max_size:classes 33, max_size:data 981, mem_usage:GPU:0 1.0GB, num_seqs 4, 3.529 sec/step, elapsed 0:22:37, exp. remaining 1:07:14, complete 25.18%
att-weights epoch 560, step 314, max_size:classes 32, max_size:data 1045, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.359 sec/step, elapsed 0:22:38, exp. remaining 1:06:56, complete 25.28%
att-weights epoch 560, step 315, max_size:classes 31, max_size:data 917, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.947 sec/step, elapsed 0:22:40, exp. remaining 1:06:40, complete 25.38%
att-weights epoch 560, step 316, max_size:classes 32, max_size:data 1086, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.520 sec/step, elapsed 0:22:42, exp. remaining 1:06:23, complete 25.48%
att-weights epoch 560, step 317, max_size:classes 32, max_size:data 904, mem_usage:GPU:0 1.0GB, num_seqs 4, 3.422 sec/step, elapsed 0:22:45, exp. remaining 1:06:12, complete 25.59%
att-weights epoch 560, step 318, max_size:classes 31, max_size:data 1076, mem_usage:GPU:0 1.0GB, num_seqs 3, 2.571 sec/step, elapsed 0:22:48, exp. remaining 1:05:51, complete 25.72%
att-weights epoch 560, step 319, max_size:classes 28, max_size:data 1303, mem_usage:GPU:0 1.0GB, num_seqs 3, 2.498 sec/step, elapsed 0:22:50, exp. remaining 1:05:30, complete 25.86%
att-weights epoch 560, step 320, max_size:classes 33, max_size:data 1099, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.456 sec/step, elapsed 0:22:52, exp. remaining 1:05:06, complete 26.00%
att-weights epoch 560, step 321, max_size:classes 31, max_size:data 930, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.845 sec/step, elapsed 0:22:54, exp. remaining 1:04:44, complete 26.13%
att-weights epoch 560, step 322, max_size:classes 31, max_size:data 1021, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.330 sec/step, elapsed 0:22:55, exp. remaining 1:04:27, complete 26.23%
att-weights epoch 560, step 323, max_size:classes 30, max_size:data 1012, mem_usage:GPU:0 1.0GB, num_seqs 3, 4.777 sec/step, elapsed 0:23:00, exp. remaining 1:04:14, complete 26.37%
att-weights epoch 560, step 324, max_size:classes 28, max_size:data 827, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.041 sec/step, elapsed 0:23:01, exp. remaining 1:03:50, complete 26.51%
att-weights epoch 560, step 325, max_size:classes 30, max_size:data 1072, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.399 sec/step, elapsed 0:23:02, exp. remaining 1:03:27, complete 26.64%
att-weights epoch 560, step 326, max_size:classes 27, max_size:data 1066, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.659 sec/step, elapsed 0:23:04, exp. remaining 1:03:12, complete 26.74%
att-weights epoch 560, step 327, max_size:classes 28, max_size:data 998, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.640 sec/step, elapsed 0:23:05, exp. remaining 1:02:50, complete 26.88%
att-weights epoch 560, step 328, max_size:classes 28, max_size:data 874, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.634 sec/step, elapsed 0:23:07, exp. remaining 1:02:22, complete 27.05%
att-weights epoch 560, step 329, max_size:classes 30, max_size:data 998, mem_usage:GPU:0 1.0GB, num_seqs 4, 4.488 sec/step, elapsed 0:23:12, exp. remaining 1:02:08, complete 27.19%
att-weights epoch 560, step 330, max_size:classes 30, max_size:data 902, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.576 sec/step, elapsed 0:23:13, exp. remaining 1:01:40, complete 27.36%
att-weights epoch 560, step 331, max_size:classes 31, max_size:data 1085, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.142 sec/step, elapsed 0:23:14, exp. remaining 1:01:18, complete 27.49%
att-weights epoch 560, step 332, max_size:classes 28, max_size:data 949, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.825 sec/step, elapsed 0:23:16, exp. remaining 1:00:58, complete 27.63%
att-weights epoch 560, step 333, max_size:classes 25, max_size:data 918, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.530 sec/step, elapsed 0:23:18, exp. remaining 1:00:43, complete 27.73%
att-weights epoch 560, step 334, max_size:classes 32, max_size:data 875, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.663 sec/step, elapsed 0:23:19, exp. remaining 1:00:23, complete 27.87%
att-weights epoch 560, step 335, max_size:classes 29, max_size:data 1096, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.410 sec/step, elapsed 0:23:21, exp. remaining 1:00:02, complete 28.00%
att-weights epoch 560, step 336, max_size:classes 33, max_size:data 816, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.311 sec/step, elapsed 0:23:22, exp. remaining 0:59:41, complete 28.14%
att-weights epoch 560, step 337, max_size:classes 26, max_size:data 792, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.707 sec/step, elapsed 0:23:24, exp. remaining 0:59:22, complete 28.27%
att-weights epoch 560, step 338, max_size:classes 31, max_size:data 946, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.549 sec/step, elapsed 0:23:25, exp. remaining 0:59:02, complete 28.41%
att-weights epoch 560, step 339, max_size:classes 26, max_size:data 772, mem_usage:GPU:0 1.0GB, num_seqs 5, 9.303 sec/step, elapsed 0:23:35, exp. remaining 0:59:02, complete 28.55%
att-weights epoch 560, step 340, max_size:classes 29, max_size:data 836, mem_usage:GPU:0 1.0GB, num_seqs 4, 2.091 sec/step, elapsed 0:23:37, exp. remaining 0:58:49, complete 28.65%
att-weights epoch 560, step 341, max_size:classes 29, max_size:data 881, mem_usage:GPU:0 1.0GB, num_seqs 4, 2.020 sec/step, elapsed 0:23:39, exp. remaining 0:58:31, complete 28.79%
att-weights epoch 560, step 342, max_size:classes 25, max_size:data 1075, mem_usage:GPU:0 1.0GB, num_seqs 3, 6.863 sec/step, elapsed 0:23:46, exp. remaining 0:58:30, complete 28.89%
att-weights epoch 560, step 343, max_size:classes 28, max_size:data 870, mem_usage:GPU:0 1.0GB, num_seqs 4, 3.833 sec/step, elapsed 0:23:49, exp. remaining 0:58:16, complete 29.02%
att-weights epoch 560, step 344, max_size:classes 26, max_size:data 864, mem_usage:GPU:0 1.0GB, num_seqs 4, 9.176 sec/step, elapsed 0:23:59, exp. remaining 0:58:10, complete 29.19%
att-weights epoch 560, step 345, max_size:classes 27, max_size:data 857, mem_usage:GPU:0 1.0GB, num_seqs 4, 4.316 sec/step, elapsed 0:24:03, exp. remaining 0:57:57, complete 29.33%
att-weights epoch 560, step 346, max_size:classes 29, max_size:data 967, mem_usage:GPU:0 1.0GB, num_seqs 4, 4.889 sec/step, elapsed 0:24:08, exp. remaining 0:57:46, complete 29.47%
att-weights epoch 560, step 347, max_size:classes 27, max_size:data 907, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.860 sec/step, elapsed 0:24:10, exp. remaining 0:57:28, complete 29.60%
att-weights epoch 560, step 348, max_size:classes 25, max_size:data 993, mem_usage:GPU:0 1.0GB, num_seqs 4, 3.487 sec/step, elapsed 0:24:13, exp. remaining 0:57:09, complete 29.77%
att-weights epoch 560, step 349, max_size:classes 27, max_size:data 1174, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.321 sec/step, elapsed 0:24:15, exp. remaining 0:56:49, complete 29.91%
att-weights epoch 560, step 350, max_size:classes 26, max_size:data 963, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.800 sec/step, elapsed 0:24:16, exp. remaining 0:56:32, complete 30.04%
att-weights epoch 560, step 351, max_size:classes 25, max_size:data 1114, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.461 sec/step, elapsed 0:24:18, exp. remaining 0:56:13, complete 30.18%
att-weights epoch 560, step 352, max_size:classes 27, max_size:data 903, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.568 sec/step, elapsed 0:24:19, exp. remaining 0:55:55, complete 30.32%
att-weights epoch 560, step 353, max_size:classes 31, max_size:data 769, mem_usage:GPU:0 1.0GB, num_seqs 5, 4.632 sec/step, elapsed 0:24:24, exp. remaining 0:55:39, complete 30.49%
att-weights epoch 560, step 354, max_size:classes 34, max_size:data 937, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.235 sec/step, elapsed 0:24:25, exp. remaining 0:55:20, complete 30.62%
att-weights epoch 560, step 355, max_size:classes 30, max_size:data 992, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.499 sec/step, elapsed 0:24:27, exp. remaining 0:55:02, complete 30.76%
att-weights epoch 560, step 356, max_size:classes 24, max_size:data 869, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.687 sec/step, elapsed 0:24:28, exp. remaining 0:54:50, complete 30.86%
att-weights epoch 560, step 357, max_size:classes 27, max_size:data 760, mem_usage:GPU:0 1.0GB, num_seqs 5, 5.652 sec/step, elapsed 0:24:34, exp. remaining 0:54:42, complete 31.00%
att-weights epoch 560, step 358, max_size:classes 27, max_size:data 905, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.704 sec/step, elapsed 0:24:36, exp. remaining 0:54:20, complete 31.17%
att-weights epoch 560, step 359, max_size:classes 24, max_size:data 886, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.606 sec/step, elapsed 0:24:37, exp. remaining 0:54:03, complete 31.30%
att-weights epoch 560, step 360, max_size:classes 28, max_size:data 804, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.304 sec/step, elapsed 0:24:39, exp. remaining 0:53:40, complete 31.47%
att-weights epoch 560, step 361, max_size:classes 25, max_size:data 847, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.420 sec/step, elapsed 0:24:40, exp. remaining 0:53:23, complete 31.61%
att-weights epoch 560, step 362, max_size:classes 28, max_size:data 787, mem_usage:GPU:0 1.0GB, num_seqs 5, 2.129 sec/step, elapsed 0:24:42, exp. remaining 0:53:07, complete 31.75%
att-weights epoch 560, step 363, max_size:classes 27, max_size:data 941, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.487 sec/step, elapsed 0:24:44, exp. remaining 0:52:51, complete 31.88%
att-weights epoch 560, step 364, max_size:classes 26, max_size:data 680, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.392 sec/step, elapsed 0:24:45, exp. remaining 0:52:29, complete 32.05%
att-weights epoch 560, step 365, max_size:classes 27, max_size:data 1070, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.555 sec/step, elapsed 0:24:47, exp. remaining 0:52:13, complete 32.19%
att-weights epoch 560, step 366, max_size:classes 25, max_size:data 869, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.805 sec/step, elapsed 0:24:48, exp. remaining 0:51:52, complete 32.36%
att-weights epoch 560, step 367, max_size:classes 23, max_size:data 739, mem_usage:GPU:0 1.0GB, num_seqs 5, 2.413 sec/step, elapsed 0:24:51, exp. remaining 0:51:38, complete 32.49%
att-weights epoch 560, step 368, max_size:classes 31, max_size:data 836, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.661 sec/step, elapsed 0:24:53, exp. remaining 0:51:22, complete 32.63%
att-weights epoch 560, step 369, max_size:classes 24, max_size:data 645, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.654 sec/step, elapsed 0:24:54, exp. remaining 0:51:02, complete 32.80%
att-weights epoch 560, step 370, max_size:classes 27, max_size:data 862, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.440 sec/step, elapsed 0:24:56, exp. remaining 0:50:46, complete 32.94%
att-weights epoch 560, step 371, max_size:classes 25, max_size:data 821, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.368 sec/step, elapsed 0:24:57, exp. remaining 0:50:30, complete 33.07%
att-weights epoch 560, step 372, max_size:classes 26, max_size:data 804, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.541 sec/step, elapsed 0:24:59, exp. remaining 0:50:14, complete 33.21%
att-weights epoch 560, step 373, max_size:classes 27, max_size:data 768, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.576 sec/step, elapsed 0:25:00, exp. remaining 0:49:59, complete 33.34%
att-weights epoch 560, step 374, max_size:classes 26, max_size:data 819, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.334 sec/step, elapsed 0:25:01, exp. remaining 0:49:44, complete 33.48%
att-weights epoch 560, step 375, max_size:classes 25, max_size:data 731, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.703 sec/step, elapsed 0:25:03, exp. remaining 0:49:29, complete 33.62%
att-weights epoch 560, step 376, max_size:classes 30, max_size:data 895, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.810 sec/step, elapsed 0:25:05, exp. remaining 0:49:14, complete 33.75%
att-weights epoch 560, step 377, max_size:classes 26, max_size:data 948, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.497 sec/step, elapsed 0:25:06, exp. remaining 0:48:59, complete 33.89%
att-weights epoch 560, step 378, max_size:classes 26, max_size:data 754, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.599 sec/step, elapsed 0:25:08, exp. remaining 0:48:36, complete 34.09%
att-weights epoch 560, step 379, max_size:classes 25, max_size:data 735, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.387 sec/step, elapsed 0:25:09, exp. remaining 0:48:21, complete 34.23%
att-weights epoch 560, step 380, max_size:classes 29, max_size:data 808, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.361 sec/step, elapsed 0:25:11, exp. remaining 0:48:06, complete 34.37%
att-weights epoch 560, step 381, max_size:classes 22, max_size:data 815, mem_usage:GPU:0 1.0GB, num_seqs 4, 3.791 sec/step, elapsed 0:25:15, exp. remaining 0:47:56, complete 34.50%
att-weights epoch 560, step 382, max_size:classes 25, max_size:data 848, mem_usage:GPU:0 1.0GB, num_seqs 4, 15.540 sec/step, elapsed 0:25:30, exp. remaining 0:48:08, complete 34.64%
att-weights epoch 560, step 383, max_size:classes 24, max_size:data 741, mem_usage:GPU:0 1.0GB, num_seqs 4, 2.144 sec/step, elapsed 0:25:32, exp. remaining 0:47:50, complete 34.81%
att-weights epoch 560, step 384, max_size:classes 26, max_size:data 879, mem_usage:GPU:0 1.0GB, num_seqs 4, 13.249 sec/step, elapsed 0:25:46, exp. remaining 0:47:53, complete 34.98%
att-weights epoch 560, step 385, max_size:classes 26, max_size:data 903, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.816 sec/step, elapsed 0:25:47, exp. remaining 0:47:35, complete 35.15%
att-weights epoch 560, step 386, max_size:classes 26, max_size:data 849, mem_usage:GPU:0 1.0GB, num_seqs 4, 4.401 sec/step, elapsed 0:25:52, exp. remaining 0:47:27, complete 35.28%
att-weights epoch 560, step 387, max_size:classes 23, max_size:data 653, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.300 sec/step, elapsed 0:25:53, exp. remaining 0:47:08, complete 35.45%
att-weights epoch 560, step 388, max_size:classes 26, max_size:data 704, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.190 sec/step, elapsed 0:25:54, exp. remaining 0:46:53, complete 35.59%
att-weights epoch 560, step 389, max_size:classes 24, max_size:data 889, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.513 sec/step, elapsed 0:25:56, exp. remaining 0:46:43, complete 35.69%
att-weights epoch 560, step 390, max_size:classes 23, max_size:data 745, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.235 sec/step, elapsed 0:25:57, exp. remaining 0:46:29, complete 35.83%
att-weights epoch 560, step 391, max_size:classes 24, max_size:data 837, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.313 sec/step, elapsed 0:25:58, exp. remaining 0:46:11, complete 36.00%
att-weights epoch 560, step 392, max_size:classes 27, max_size:data 736, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.413 sec/step, elapsed 0:26:00, exp. remaining 0:45:57, complete 36.13%
att-weights epoch 560, step 393, max_size:classes 25, max_size:data 784, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.965 sec/step, elapsed 0:26:02, exp. remaining 0:45:48, complete 36.24%
att-weights epoch 560, step 394, max_size:classes 25, max_size:data 710, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.658 sec/step, elapsed 0:26:03, exp. remaining 0:45:39, complete 36.34%
att-weights epoch 560, step 395, max_size:classes 25, max_size:data 872, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.284 sec/step, elapsed 0:26:05, exp. remaining 0:45:17, complete 36.54%
att-weights epoch 560, step 396, max_size:classes 25, max_size:data 796, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.710 sec/step, elapsed 0:26:06, exp. remaining 0:45:04, complete 36.68%
att-weights epoch 560, step 397, max_size:classes 24, max_size:data 854, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.132 sec/step, elapsed 0:26:07, exp. remaining 0:44:51, complete 36.82%
att-weights epoch 560, step 398, max_size:classes 24, max_size:data 1166, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.360 sec/step, elapsed 0:26:09, exp. remaining 0:44:37, complete 36.95%
att-weights epoch 560, step 399, max_size:classes 22, max_size:data 962, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.608 sec/step, elapsed 0:26:10, exp. remaining 0:44:28, complete 37.05%
att-weights epoch 560, step 400, max_size:classes 27, max_size:data 797, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.721 sec/step, elapsed 0:26:12, exp. remaining 0:44:12, complete 37.22%
att-weights epoch 560, step 401, max_size:classes 24, max_size:data 847, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.635 sec/step, elapsed 0:26:14, exp. remaining 0:43:59, complete 37.36%
att-weights epoch 560, step 402, max_size:classes 25, max_size:data 1160, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.532 sec/step, elapsed 0:26:15, exp. remaining 0:43:43, complete 37.53%
att-weights epoch 560, step 403, max_size:classes 24, max_size:data 1026, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.321 sec/step, elapsed 0:26:17, exp. remaining 0:43:26, complete 37.70%
att-weights epoch 560, step 404, max_size:classes 23, max_size:data 666, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.905 sec/step, elapsed 0:26:19, exp. remaining 0:43:10, complete 37.87%
att-weights epoch 560, step 405, max_size:classes 23, max_size:data 842, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.179 sec/step, elapsed 0:26:20, exp. remaining 0:42:57, complete 38.01%
att-weights epoch 560, step 406, max_size:classes 29, max_size:data 894, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.858 sec/step, elapsed 0:26:22, exp. remaining 0:42:45, complete 38.14%
att-weights epoch 560, step 407, max_size:classes 27, max_size:data 678, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.749 sec/step, elapsed 0:26:23, exp. remaining 0:42:33, complete 38.28%
att-weights epoch 560, step 408, max_size:classes 23, max_size:data 1045, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.459 sec/step, elapsed 0:26:25, exp. remaining 0:42:17, complete 38.45%
att-weights epoch 560, step 409, max_size:classes 24, max_size:data 776, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.679 sec/step, elapsed 0:26:26, exp. remaining 0:42:02, complete 38.62%
att-weights epoch 560, step 410, max_size:classes 26, max_size:data 848, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.752 sec/step, elapsed 0:26:28, exp. remaining 0:41:47, complete 38.79%
att-weights epoch 560, step 411, max_size:classes 23, max_size:data 700, mem_usage:GPU:0 1.0GB, num_seqs 5, 2.105 sec/step, elapsed 0:26:30, exp. remaining 0:41:36, complete 38.92%
att-weights epoch 560, step 412, max_size:classes 23, max_size:data 747, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.656 sec/step, elapsed 0:26:32, exp. remaining 0:41:17, complete 39.13%
att-weights epoch 560, step 413, max_size:classes 25, max_size:data 708, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.476 sec/step, elapsed 0:26:33, exp. remaining 0:40:58, complete 39.33%
att-weights epoch 560, step 414, max_size:classes 22, max_size:data 756, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.573 sec/step, elapsed 0:26:35, exp. remaining 0:40:43, complete 39.50%
att-weights epoch 560, step 415, max_size:classes 24, max_size:data 809, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.567 sec/step, elapsed 0:26:37, exp. remaining 0:40:28, complete 39.67%
att-weights epoch 560, step 416, max_size:classes 23, max_size:data 902, mem_usage:GPU:0 1.0GB, num_seqs 4, 2.039 sec/step, elapsed 0:26:39, exp. remaining 0:40:14, complete 39.84%
att-weights epoch 560, step 417, max_size:classes 23, max_size:data 774, mem_usage:GPU:0 1.0GB, num_seqs 5, 3.048 sec/step, elapsed 0:26:42, exp. remaining 0:40:05, complete 39.98%
att-weights epoch 560, step 418, max_size:classes 23, max_size:data 799, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.701 sec/step, elapsed 0:26:43, exp. remaining 0:39:47, complete 40.18%
att-weights epoch 560, step 419, max_size:classes 26, max_size:data 690, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.541 sec/step, elapsed 0:26:45, exp. remaining 0:39:29, complete 40.39%
att-weights epoch 560, step 420, max_size:classes 23, max_size:data 852, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.606 sec/step, elapsed 0:26:47, exp. remaining 0:39:15, complete 40.56%
att-weights epoch 560, step 421, max_size:classes 26, max_size:data 648, mem_usage:GPU:0 1.0GB, num_seqs 6, 3.377 sec/step, elapsed 0:26:50, exp. remaining 0:39:06, complete 40.69%
att-weights epoch 560, step 422, max_size:classes 24, max_size:data 592, mem_usage:GPU:0 1.0GB, num_seqs 6, 2.772 sec/step, elapsed 0:26:53, exp. remaining 0:38:54, complete 40.86%
att-weights epoch 560, step 423, max_size:classes 27, max_size:data 762, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.554 sec/step, elapsed 0:26:54, exp. remaining 0:38:37, complete 41.07%
att-weights epoch 560, step 424, max_size:classes 21, max_size:data 712, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.674 sec/step, elapsed 0:26:56, exp. remaining 0:38:23, complete 41.24%
att-weights epoch 560, step 425, max_size:classes 23, max_size:data 707, mem_usage:GPU:0 1.0GB, num_seqs 5, 2.375 sec/step, elapsed 0:26:58, exp. remaining 0:38:13, complete 41.37%
att-weights epoch 560, step 426, max_size:classes 25, max_size:data 832, mem_usage:GPU:0 1.0GB, num_seqs 4, 5.804 sec/step, elapsed 0:27:04, exp. remaining 0:38:05, complete 41.54%
att-weights epoch 560, step 427, max_size:classes 21, max_size:data 595, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.514 sec/step, elapsed 0:27:06, exp. remaining 0:37:52, complete 41.71%
att-weights epoch 560, step 428, max_size:classes 24, max_size:data 611, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.315 sec/step, elapsed 0:27:07, exp. remaining 0:37:38, complete 41.88%
att-weights epoch 560, step 429, max_size:classes 23, max_size:data 796, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.923 sec/step, elapsed 0:27:09, exp. remaining 0:37:31, complete 41.99%
att-weights epoch 560, step 430, max_size:classes 22, max_size:data 902, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.989 sec/step, elapsed 0:27:11, exp. remaining 0:37:24, complete 42.09%
att-weights epoch 560, step 431, max_size:classes 22, max_size:data 703, mem_usage:GPU:0 1.0GB, num_seqs 5, 2.457 sec/step, elapsed 0:27:13, exp. remaining 0:37:15, complete 42.23%
att-weights epoch 560, step 432, max_size:classes 21, max_size:data 602, mem_usage:GPU:0 1.0GB, num_seqs 6, 2.133 sec/step, elapsed 0:27:15, exp. remaining 0:37:02, complete 42.40%
att-weights epoch 560, step 433, max_size:classes 22, max_size:data 754, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.531 sec/step, elapsed 0:27:17, exp. remaining 0:36:49, complete 42.57%
att-weights epoch 560, step 434, max_size:classes 23, max_size:data 804, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.599 sec/step, elapsed 0:27:19, exp. remaining 0:36:36, complete 42.74%
att-weights epoch 560, step 435, max_size:classes 22, max_size:data 562, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.451 sec/step, elapsed 0:27:20, exp. remaining 0:36:23, complete 42.91%
att-weights epoch 560, step 436, max_size:classes 22, max_size:data 703, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.681 sec/step, elapsed 0:27:22, exp. remaining 0:36:13, complete 43.04%
att-weights epoch 560, step 437, max_size:classes 21, max_size:data 713, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.592 sec/step, elapsed 0:27:23, exp. remaining 0:36:00, complete 43.21%
att-weights epoch 560, step 438, max_size:classes 21, max_size:data 817, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.215 sec/step, elapsed 0:27:25, exp. remaining 0:35:49, complete 43.35%
att-weights epoch 560, step 439, max_size:classes 21, max_size:data 1124, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.770 sec/step, elapsed 0:27:26, exp. remaining 0:35:37, complete 43.52%
att-weights epoch 560, step 440, max_size:classes 19, max_size:data 857, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.218 sec/step, elapsed 0:27:28, exp. remaining 0:35:27, complete 43.65%
att-weights epoch 560, step 441, max_size:classes 25, max_size:data 789, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.608 sec/step, elapsed 0:27:29, exp. remaining 0:35:08, complete 43.89%
att-weights epoch 560, step 442, max_size:classes 23, max_size:data 728, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.768 sec/step, elapsed 0:27:31, exp. remaining 0:34:59, complete 44.03%
att-weights epoch 560, step 443, max_size:classes 21, max_size:data 701, mem_usage:GPU:0 1.0GB, num_seqs 5, 2.761 sec/step, elapsed 0:27:34, exp. remaining 0:34:48, complete 44.20%
att-weights epoch 560, step 444, max_size:classes 22, max_size:data 690, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.651 sec/step, elapsed 0:27:35, exp. remaining 0:34:36, complete 44.37%
att-weights epoch 560, step 445, max_size:classes 23, max_size:data 803, mem_usage:GPU:0 1.0GB, num_seqs 4, 2.109 sec/step, elapsed 0:27:37, exp. remaining 0:34:18, complete 44.61%
att-weights epoch 560, step 446, max_size:classes 22, max_size:data 773, mem_usage:GPU:0 1.0GB, num_seqs 5, 2.377 sec/step, elapsed 0:27:40, exp. remaining 0:34:07, complete 44.78%
att-weights epoch 560, step 447, max_size:classes 21, max_size:data 859, mem_usage:GPU:0 1.0GB, num_seqs 4, 2.077 sec/step, elapsed 0:27:42, exp. remaining 0:33:58, complete 44.91%
att-weights epoch 560, step 448, max_size:classes 20, max_size:data 658, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.495 sec/step, elapsed 0:27:43, exp. remaining 0:33:46, complete 45.08%
att-weights epoch 560, step 449, max_size:classes 21, max_size:data 908, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.674 sec/step, elapsed 0:27:45, exp. remaining 0:33:32, complete 45.29%
att-weights epoch 560, step 450, max_size:classes 20, max_size:data 524, mem_usage:GPU:0 1.0GB, num_seqs 7, 4.849 sec/step, elapsed 0:27:50, exp. remaining 0:33:24, complete 45.46%
att-weights epoch 560, step 451, max_size:classes 19, max_size:data 846, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.557 sec/step, elapsed 0:27:51, exp. remaining 0:33:12, complete 45.63%
att-weights epoch 560, step 452, max_size:classes 21, max_size:data 691, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.216 sec/step, elapsed 0:27:53, exp. remaining 0:33:00, complete 45.80%
att-weights epoch 560, step 453, max_size:classes 22, max_size:data 773, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.856 sec/step, elapsed 0:27:55, exp. remaining 0:32:48, complete 45.97%
att-weights epoch 560, step 454, max_size:classes 25, max_size:data 570, mem_usage:GPU:0 1.0GB, num_seqs 7, 2.863 sec/step, elapsed 0:27:57, exp. remaining 0:32:41, complete 46.10%
att-weights epoch 560, step 455, max_size:classes 26, max_size:data 649, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.957 sec/step, elapsed 0:27:59, exp. remaining 0:32:27, complete 46.31%
att-weights epoch 560, step 456, max_size:classes 23, max_size:data 839, mem_usage:GPU:0 1.0GB, num_seqs 4, 2.018 sec/step, elapsed 0:28:01, exp. remaining 0:32:19, complete 46.44%
att-weights epoch 560, step 457, max_size:classes 21, max_size:data 673, mem_usage:GPU:0 1.0GB, num_seqs 5, 2.019 sec/step, elapsed 0:28:03, exp. remaining 0:32:05, complete 46.65%
att-weights epoch 560, step 458, max_size:classes 23, max_size:data 638, mem_usage:GPU:0 1.0GB, num_seqs 6, 6.085 sec/step, elapsed 0:28:09, exp. remaining 0:31:56, complete 46.85%
att-weights epoch 560, step 459, max_size:classes 22, max_size:data 657, mem_usage:GPU:0 1.0GB, num_seqs 5, 2.156 sec/step, elapsed 0:28:12, exp. remaining 0:31:43, complete 47.06%
att-weights epoch 560, step 460, max_size:classes 20, max_size:data 709, mem_usage:GPU:0 1.0GB, num_seqs 5, 3.367 sec/step, elapsed 0:28:15, exp. remaining 0:31:31, complete 47.26%
att-weights epoch 560, step 461, max_size:classes 19, max_size:data 710, mem_usage:GPU:0 1.0GB, num_seqs 5, 12.107 sec/step, elapsed 0:28:27, exp. remaining 0:31:27, complete 47.50%
att-weights epoch 560, step 462, max_size:classes 21, max_size:data 684, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.703 sec/step, elapsed 0:28:29, exp. remaining 0:31:13, complete 47.70%
att-weights epoch 560, step 463, max_size:classes 20, max_size:data 978, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.547 sec/step, elapsed 0:28:30, exp. remaining 0:31:00, complete 47.91%
att-weights epoch 560, step 464, max_size:classes 22, max_size:data 654, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.804 sec/step, elapsed 0:28:32, exp. remaining 0:30:52, complete 48.04%
att-weights epoch 560, step 465, max_size:classes 19, max_size:data 847, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.441 sec/step, elapsed 0:28:34, exp. remaining 0:30:41, complete 48.21%
att-weights epoch 560, step 466, max_size:classes 18, max_size:data 654, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.485 sec/step, elapsed 0:28:35, exp. remaining 0:30:30, complete 48.38%
att-weights epoch 560, step 467, max_size:classes 21, max_size:data 651, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.779 sec/step, elapsed 0:28:37, exp. remaining 0:30:19, complete 48.55%
att-weights epoch 560, step 468, max_size:classes 20, max_size:data 624, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.902 sec/step, elapsed 0:28:39, exp. remaining 0:30:11, complete 48.69%
att-weights epoch 560, step 469, max_size:classes 23, max_size:data 624, mem_usage:GPU:0 1.0GB, num_seqs 6, 2.000 sec/step, elapsed 0:28:41, exp. remaining 0:30:01, complete 48.86%
att-weights epoch 560, step 470, max_size:classes 22, max_size:data 555, mem_usage:GPU:0 1.0GB, num_seqs 7, 2.337 sec/step, elapsed 0:28:43, exp. remaining 0:29:51, complete 49.03%
att-weights epoch 560, step 471, max_size:classes 22, max_size:data 626, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.473 sec/step, elapsed 0:28:45, exp. remaining 0:29:38, complete 49.23%
att-weights epoch 560, step 472, max_size:classes 19, max_size:data 591, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.539 sec/step, elapsed 0:28:46, exp. remaining 0:29:23, complete 49.47%
att-weights epoch 560, step 473, max_size:classes 17, max_size:data 815, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.728 sec/step, elapsed 0:28:48, exp. remaining 0:29:15, complete 49.61%
att-weights epoch 560, step 474, max_size:classes 20, max_size:data 677, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.696 sec/step, elapsed 0:28:50, exp. remaining 0:29:05, complete 49.78%
att-weights epoch 560, step 475, max_size:classes 20, max_size:data 680, mem_usage:GPU:0 1.0GB, num_seqs 5, 2.331 sec/step, elapsed 0:28:52, exp. remaining 0:28:53, complete 49.98%
att-weights epoch 560, step 476, max_size:classes 20, max_size:data 586, mem_usage:GPU:0 1.0GB, num_seqs 5, 2.293 sec/step, elapsed 0:28:54, exp. remaining 0:28:41, complete 50.19%
att-weights epoch 560, step 477, max_size:classes 22, max_size:data 809, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.549 sec/step, elapsed 0:28:56, exp. remaining 0:28:31, complete 50.36%
att-weights epoch 560, step 478, max_size:classes 20, max_size:data 552, mem_usage:GPU:0 1.0GB, num_seqs 5, 9.565 sec/step, elapsed 0:29:05, exp. remaining 0:28:24, complete 50.60%
att-weights epoch 560, step 479, max_size:classes 21, max_size:data 701, mem_usage:GPU:0 1.0GB, num_seqs 5, 2.520 sec/step, elapsed 0:29:08, exp. remaining 0:28:15, complete 50.77%
att-weights epoch 560, step 480, max_size:classes 20, max_size:data 591, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.799 sec/step, elapsed 0:29:10, exp. remaining 0:28:03, complete 50.97%
att-weights epoch 560, step 481, max_size:classes 18, max_size:data 549, mem_usage:GPU:0 1.0GB, num_seqs 7, 5.079 sec/step, elapsed 0:29:15, exp. remaining 0:27:56, complete 51.14%
att-weights epoch 560, step 482, max_size:classes 21, max_size:data 810, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.569 sec/step, elapsed 0:29:16, exp. remaining 0:27:47, complete 51.31%
att-weights epoch 560, step 483, max_size:classes 20, max_size:data 703, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.928 sec/step, elapsed 0:29:18, exp. remaining 0:27:35, complete 51.51%
att-weights epoch 560, step 484, max_size:classes 18, max_size:data 614, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.983 sec/step, elapsed 0:29:20, exp. remaining 0:27:23, complete 51.72%
att-weights epoch 560, step 485, max_size:classes 19, max_size:data 600, mem_usage:GPU:0 1.0GB, num_seqs 6, 4.799 sec/step, elapsed 0:29:25, exp. remaining 0:27:14, complete 51.92%
att-weights epoch 560, step 486, max_size:classes 21, max_size:data 683, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.923 sec/step, elapsed 0:29:27, exp. remaining 0:27:03, complete 52.13%
att-weights epoch 560, step 487, max_size:classes 22, max_size:data 563, mem_usage:GPU:0 1.0GB, num_seqs 7, 6.088 sec/step, elapsed 0:29:33, exp. remaining 0:26:55, complete 52.33%
att-weights epoch 560, step 488, max_size:classes 20, max_size:data 685, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.891 sec/step, elapsed 0:29:35, exp. remaining 0:26:41, complete 52.57%
att-weights epoch 560, step 489, max_size:classes 18, max_size:data 643, mem_usage:GPU:0 1.0GB, num_seqs 6, 2.240 sec/step, elapsed 0:29:37, exp. remaining 0:26:26, complete 52.84%
att-weights epoch 560, step 490, max_size:classes 17, max_size:data 684, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.909 sec/step, elapsed 0:29:39, exp. remaining 0:26:15, complete 53.05%
att-weights epoch 560, step 491, max_size:classes 18, max_size:data 741, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.664 sec/step, elapsed 0:29:41, exp. remaining 0:26:03, complete 53.25%
att-weights epoch 560, step 492, max_size:classes 18, max_size:data 647, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.881 sec/step, elapsed 0:29:43, exp. remaining 0:25:52, complete 53.45%
att-weights epoch 560, step 493, max_size:classes 19, max_size:data 633, mem_usage:GPU:0 1.0GB, num_seqs 6, 2.244 sec/step, elapsed 0:29:45, exp. remaining 0:25:41, complete 53.66%
att-weights epoch 560, step 494, max_size:classes 19, max_size:data 613, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.990 sec/step, elapsed 0:29:47, exp. remaining 0:25:30, complete 53.86%
att-weights epoch 560, step 495, max_size:classes 20, max_size:data 631, mem_usage:GPU:0 1.0GB, num_seqs 6, 2.028 sec/step, elapsed 0:29:49, exp. remaining 0:25:22, complete 54.03%
att-weights epoch 560, step 496, max_size:classes 18, max_size:data 580, mem_usage:GPU:0 1.0GB, num_seqs 6, 7.873 sec/step, elapsed 0:29:57, exp. remaining 0:25:18, complete 54.20%
att-weights epoch 560, step 497, max_size:classes 17, max_size:data 556, mem_usage:GPU:0 1.0GB, num_seqs 7, 3.728 sec/step, elapsed 0:30:00, exp. remaining 0:25:09, complete 54.41%
att-weights epoch 560, step 498, max_size:classes 20, max_size:data 480, mem_usage:GPU:0 1.0GB, num_seqs 8, 2.388 sec/step, elapsed 0:30:03, exp. remaining 0:24:56, complete 54.64%
att-weights epoch 560, step 499, max_size:classes 19, max_size:data 558, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.870 sec/step, elapsed 0:30:05, exp. remaining 0:24:43, complete 54.88%
att-weights epoch 560, step 500, max_size:classes 21, max_size:data 651, mem_usage:GPU:0 1.0GB, num_seqs 6, 3.666 sec/step, elapsed 0:30:08, exp. remaining 0:24:34, complete 55.09%
att-weights epoch 560, step 501, max_size:classes 21, max_size:data 554, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.787 sec/step, elapsed 0:30:10, exp. remaining 0:24:22, complete 55.32%
att-weights epoch 560, step 502, max_size:classes 19, max_size:data 611, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.630 sec/step, elapsed 0:30:12, exp. remaining 0:24:11, complete 55.53%
att-weights epoch 560, step 503, max_size:classes 20, max_size:data 606, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.897 sec/step, elapsed 0:30:14, exp. remaining 0:24:02, complete 55.70%
att-weights epoch 560, step 504, max_size:classes 17, max_size:data 601, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.570 sec/step, elapsed 0:30:15, exp. remaining 0:23:54, complete 55.87%
att-weights epoch 560, step 505, max_size:classes 17, max_size:data 773, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.998 sec/step, elapsed 0:30:17, exp. remaining 0:23:41, complete 56.11%
att-weights epoch 560, step 506, max_size:classes 17, max_size:data 574, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.841 sec/step, elapsed 0:30:19, exp. remaining 0:23:33, complete 56.28%
att-weights epoch 560, step 507, max_size:classes 18, max_size:data 551, mem_usage:GPU:0 1.0GB, num_seqs 7, 2.122 sec/step, elapsed 0:30:21, exp. remaining 0:23:21, complete 56.52%
att-weights epoch 560, step 508, max_size:classes 18, max_size:data 486, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.984 sec/step, elapsed 0:30:23, exp. remaining 0:23:09, complete 56.75%
att-weights epoch 560, step 509, max_size:classes 17, max_size:data 643, mem_usage:GPU:0 1.0GB, num_seqs 6, 2.316 sec/step, elapsed 0:30:25, exp. remaining 0:22:59, complete 56.96%
att-weights epoch 560, step 510, max_size:classes 19, max_size:data 522, mem_usage:GPU:0 1.0GB, num_seqs 7, 2.374 sec/step, elapsed 0:30:28, exp. remaining 0:22:48, complete 57.20%
att-weights epoch 560, step 511, max_size:classes 22, max_size:data 621, mem_usage:GPU:0 1.0GB, num_seqs 6, 3.346 sec/step, elapsed 0:30:31, exp. remaining 0:22:37, complete 57.43%
att-weights epoch 560, step 512, max_size:classes 17, max_size:data 579, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.751 sec/step, elapsed 0:30:33, exp. remaining 0:22:25, complete 57.67%
att-weights epoch 560, step 513, max_size:classes 18, max_size:data 750, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.865 sec/step, elapsed 0:30:35, exp. remaining 0:22:13, complete 57.91%
att-weights epoch 560, step 514, max_size:classes 18, max_size:data 567, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.881 sec/step, elapsed 0:30:37, exp. remaining 0:22:04, complete 58.12%
att-weights epoch 560, step 515, max_size:classes 18, max_size:data 680, mem_usage:GPU:0 1.0GB, num_seqs 5, 2.455 sec/step, elapsed 0:30:39, exp. remaining 0:21:54, complete 58.32%
att-weights epoch 560, step 516, max_size:classes 17, max_size:data 520, mem_usage:GPU:0 1.0GB, num_seqs 7, 2.574 sec/step, elapsed 0:30:42, exp. remaining 0:21:47, complete 58.49%
att-weights epoch 560, step 517, max_size:classes 18, max_size:data 571, mem_usage:GPU:0 1.0GB, num_seqs 7, 2.357 sec/step, elapsed 0:30:44, exp. remaining 0:21:34, complete 58.76%
att-weights epoch 560, step 518, max_size:classes 16, max_size:data 661, mem_usage:GPU:0 1.0GB, num_seqs 6, 3.018 sec/step, elapsed 0:30:47, exp. remaining 0:21:25, complete 58.97%
att-weights epoch 560, step 519, max_size:classes 15, max_size:data 510, mem_usage:GPU:0 1.0GB, num_seqs 7, 11.071 sec/step, elapsed 0:30:58, exp. remaining 0:21:22, complete 59.17%
att-weights epoch 560, step 520, max_size:classes 17, max_size:data 546, mem_usage:GPU:0 1.0GB, num_seqs 7, 8.043 sec/step, elapsed 0:31:06, exp. remaining 0:21:15, complete 59.41%
att-weights epoch 560, step 521, max_size:classes 19, max_size:data 535, mem_usage:GPU:0 1.0GB, num_seqs 7, 3.159 sec/step, elapsed 0:31:09, exp. remaining 0:21:08, complete 59.58%
att-weights epoch 560, step 522, max_size:classes 17, max_size:data 551, mem_usage:GPU:0 1.0GB, num_seqs 7, 5.134 sec/step, elapsed 0:31:15, exp. remaining 0:21:01, complete 59.78%
att-weights epoch 560, step 523, max_size:classes 17, max_size:data 627, mem_usage:GPU:0 1.0GB, num_seqs 6, 4.752 sec/step, elapsed 0:31:19, exp. remaining 0:20:52, complete 60.02%
att-weights epoch 560, step 524, max_size:classes 16, max_size:data 653, mem_usage:GPU:0 1.0GB, num_seqs 6, 10.588 sec/step, elapsed 0:31:30, exp. remaining 0:20:50, complete 60.19%
att-weights epoch 560, step 525, max_size:classes 17, max_size:data 776, mem_usage:GPU:0 1.0GB, num_seqs 5, 2.609 sec/step, elapsed 0:31:32, exp. remaining 0:20:36, complete 60.50%
att-weights epoch 560, step 526, max_size:classes 19, max_size:data 457, mem_usage:GPU:0 1.0GB, num_seqs 8, 13.004 sec/step, elapsed 0:31:45, exp. remaining 0:20:30, complete 60.77%
att-weights epoch 560, step 527, max_size:classes 14, max_size:data 498, mem_usage:GPU:0 1.0GB, num_seqs 6, 2.569 sec/step, elapsed 0:31:48, exp. remaining 0:20:19, complete 61.01%
att-weights epoch 560, step 528, max_size:classes 16, max_size:data 645, mem_usage:GPU:0 1.0GB, num_seqs 6, 9.263 sec/step, elapsed 0:31:57, exp. remaining 0:20:17, complete 61.18%
att-weights epoch 560, step 529, max_size:classes 16, max_size:data 516, mem_usage:GPU:0 1.0GB, num_seqs 7, 3.074 sec/step, elapsed 0:32:00, exp. remaining 0:20:06, complete 61.42%
att-weights epoch 560, step 530, max_size:classes 16, max_size:data 682, mem_usage:GPU:0 1.0GB, num_seqs 5, 2.617 sec/step, elapsed 0:32:03, exp. remaining 0:19:56, complete 61.65%
att-weights epoch 560, step 531, max_size:classes 16, max_size:data 618, mem_usage:GPU:0 1.0GB, num_seqs 6, 6.771 sec/step, elapsed 0:32:10, exp. remaining 0:19:46, complete 61.93%
att-weights epoch 560, step 532, max_size:classes 16, max_size:data 534, mem_usage:GPU:0 1.0GB, num_seqs 7, 2.572 sec/step, elapsed 0:32:12, exp. remaining 0:19:36, complete 62.16%
att-weights epoch 560, step 533, max_size:classes 15, max_size:data 698, mem_usage:GPU:0 1.0GB, num_seqs 5, 3.052 sec/step, elapsed 0:32:15, exp. remaining 0:19:23, complete 62.47%
att-weights epoch 560, step 534, max_size:classes 20, max_size:data 435, mem_usage:GPU:0 1.0GB, num_seqs 9, 6.977 sec/step, elapsed 0:32:22, exp. remaining 0:19:17, complete 62.67%
att-weights epoch 560, step 535, max_size:classes 18, max_size:data 486, mem_usage:GPU:0 1.0GB, num_seqs 8, 5.102 sec/step, elapsed 0:32:27, exp. remaining 0:19:08, complete 62.91%
att-weights epoch 560, step 536, max_size:classes 17, max_size:data 441, mem_usage:GPU:0 1.0GB, num_seqs 7, 4.671 sec/step, elapsed 0:32:32, exp. remaining 0:18:59, complete 63.15%
att-weights epoch 560, step 537, max_size:classes 19, max_size:data 675, mem_usage:GPU:0 1.0GB, num_seqs 5, 2.048 sec/step, elapsed 0:32:34, exp. remaining 0:18:47, complete 63.42%
att-weights epoch 560, step 538, max_size:classes 16, max_size:data 526, mem_usage:GPU:0 1.0GB, num_seqs 7, 3.050 sec/step, elapsed 0:32:37, exp. remaining 0:18:35, complete 63.70%
att-weights epoch 560, step 539, max_size:classes 16, max_size:data 538, mem_usage:GPU:0 1.0GB, num_seqs 7, 2.913 sec/step, elapsed 0:32:40, exp. remaining 0:18:24, complete 63.97%
att-weights epoch 560, step 540, max_size:classes 17, max_size:data 477, mem_usage:GPU:0 1.0GB, num_seqs 8, 2.964 sec/step, elapsed 0:32:43, exp. remaining 0:18:16, complete 64.17%
att-weights epoch 560, step 541, max_size:classes 17, max_size:data 558, mem_usage:GPU:0 1.0GB, num_seqs 7, 3.263 sec/step, elapsed 0:32:46, exp. remaining 0:18:03, complete 64.48%
att-weights epoch 560, step 542, max_size:classes 15, max_size:data 444, mem_usage:GPU:0 1.0GB, num_seqs 9, 6.892 sec/step, elapsed 0:32:53, exp. remaining 0:17:52, complete 64.78%
att-weights epoch 560, step 543, max_size:classes 15, max_size:data 662, mem_usage:GPU:0 1.0GB, num_seqs 6, 3.660 sec/step, elapsed 0:32:57, exp. remaining 0:17:43, complete 65.02%
att-weights epoch 560, step 544, max_size:classes 15, max_size:data 531, mem_usage:GPU:0 1.0GB, num_seqs 7, 3.853 sec/step, elapsed 0:33:01, exp. remaining 0:17:33, complete 65.29%
att-weights epoch 560, step 545, max_size:classes 16, max_size:data 571, mem_usage:GPU:0 1.0GB, num_seqs 7, 14.207 sec/step, elapsed 0:33:15, exp. remaining 0:17:29, complete 65.53%
att-weights epoch 560, step 546, max_size:classes 17, max_size:data 464, mem_usage:GPU:0 1.0GB, num_seqs 8, 2.842 sec/step, elapsed 0:33:18, exp. remaining 0:17:18, complete 65.80%
att-weights epoch 560, step 547, max_size:classes 16, max_size:data 478, mem_usage:GPU:0 1.0GB, num_seqs 8, 38.841 sec/step, elapsed 0:33:57, exp. remaining 0:17:25, complete 66.08%
att-weights epoch 560, step 548, max_size:classes 16, max_size:data 486, mem_usage:GPU:0 1.0GB, num_seqs 8, 18.290 sec/step, elapsed 0:34:15, exp. remaining 0:17:22, complete 66.35%
att-weights epoch 560, step 549, max_size:classes 16, max_size:data 585, mem_usage:GPU:0 1.0GB, num_seqs 6, 2.722 sec/step, elapsed 0:34:18, exp. remaining 0:17:11, complete 66.62%
att-weights epoch 560, step 550, max_size:classes 16, max_size:data 438, mem_usage:GPU:0 1.0GB, num_seqs 9, 3.156 sec/step, elapsed 0:34:21, exp. remaining 0:16:58, complete 66.93%
att-weights epoch 560, step 551, max_size:classes 14, max_size:data 431, mem_usage:GPU:0 1.0GB, num_seqs 9, 3.047 sec/step, elapsed 0:34:24, exp. remaining 0:16:49, complete 67.17%
att-weights epoch 560, step 552, max_size:classes 17, max_size:data 523, mem_usage:GPU:0 1.0GB, num_seqs 7, 2.598 sec/step, elapsed 0:34:27, exp. remaining 0:16:38, complete 67.44%
att-weights epoch 560, step 553, max_size:classes 15, max_size:data 470, mem_usage:GPU:0 1.0GB, num_seqs 8, 7.643 sec/step, elapsed 0:34:34, exp. remaining 0:16:30, complete 67.68%
att-weights epoch 560, step 554, max_size:classes 19, max_size:data 558, mem_usage:GPU:0 1.0GB, num_seqs 7, 2.246 sec/step, elapsed 0:34:36, exp. remaining 0:16:21, complete 67.91%
att-weights epoch 560, step 555, max_size:classes 15, max_size:data 451, mem_usage:GPU:0 1.0GB, num_seqs 8, 2.368 sec/step, elapsed 0:34:39, exp. remaining 0:16:10, complete 68.19%
att-weights epoch 560, step 556, max_size:classes 14, max_size:data 456, mem_usage:GPU:0 1.0GB, num_seqs 8, 5.775 sec/step, elapsed 0:34:45, exp. remaining 0:16:00, complete 68.46%
att-weights epoch 560, step 557, max_size:classes 15, max_size:data 480, mem_usage:GPU:0 1.0GB, num_seqs 8, 8.547 sec/step, elapsed 0:34:53, exp. remaining 0:15:53, complete 68.70%
att-weights epoch 560, step 558, max_size:classes 14, max_size:data 465, mem_usage:GPU:0 1.0GB, num_seqs 8, 2.039 sec/step, elapsed 0:34:55, exp. remaining 0:15:45, complete 68.90%
att-weights epoch 560, step 559, max_size:classes 16, max_size:data 432, mem_usage:GPU:0 1.0GB, num_seqs 9, 2.559 sec/step, elapsed 0:34:58, exp. remaining 0:15:36, complete 69.14%
att-weights epoch 560, step 560, max_size:classes 17, max_size:data 536, mem_usage:GPU:0 1.0GB, num_seqs 7, 2.358 sec/step, elapsed 0:35:00, exp. remaining 0:15:27, complete 69.38%
att-weights epoch 560, step 561, max_size:classes 14, max_size:data 478, mem_usage:GPU:0 1.0GB, num_seqs 8, 12.790 sec/step, elapsed 0:35:13, exp. remaining 0:15:22, complete 69.62%
att-weights epoch 560, step 562, max_size:classes 13, max_size:data 471, mem_usage:GPU:0 1.0GB, num_seqs 7, 2.338 sec/step, elapsed 0:35:15, exp. remaining 0:15:11, complete 69.89%
att-weights epoch 560, step 563, max_size:classes 17, max_size:data 526, mem_usage:GPU:0 1.0GB, num_seqs 7, 2.273 sec/step, elapsed 0:35:17, exp. remaining 0:15:02, complete 70.13%
att-weights epoch 560, step 564, max_size:classes 14, max_size:data 486, mem_usage:GPU:0 1.0GB, num_seqs 8, 2.202 sec/step, elapsed 0:35:20, exp. remaining 0:14:52, complete 70.36%
att-weights epoch 560, step 565, max_size:classes 17, max_size:data 468, mem_usage:GPU:0 1.0GB, num_seqs 8, 2.549 sec/step, elapsed 0:35:22, exp. remaining 0:14:40, complete 70.67%
att-weights epoch 560, step 566, max_size:classes 14, max_size:data 389, mem_usage:GPU:0 1.0GB, num_seqs 7, 2.242 sec/step, elapsed 0:35:24, exp. remaining 0:14:31, complete 70.91%
att-weights epoch 560, step 567, max_size:classes 15, max_size:data 574, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.847 sec/step, elapsed 0:35:26, exp. remaining 0:14:21, complete 71.18%
att-weights epoch 560, step 568, max_size:classes 16, max_size:data 520, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.988 sec/step, elapsed 0:35:28, exp. remaining 0:14:07, complete 71.52%
att-weights epoch 560, step 569, max_size:classes 14, max_size:data 384, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.663 sec/step, elapsed 0:35:30, exp. remaining 0:13:58, complete 71.76%
att-weights epoch 560, step 570, max_size:classes 14, max_size:data 557, mem_usage:GPU:0 1.0GB, num_seqs 7, 2.778 sec/step, elapsed 0:35:33, exp. remaining 0:13:48, complete 72.03%
att-weights epoch 560, step 571, max_size:classes 15, max_size:data 457, mem_usage:GPU:0 1.0GB, num_seqs 8, 2.041 sec/step, elapsed 0:35:35, exp. remaining 0:13:36, complete 72.34%
att-weights epoch 560, step 572, max_size:classes 15, max_size:data 514, mem_usage:GPU:0 1.0GB, num_seqs 7, 2.276 sec/step, elapsed 0:35:37, exp. remaining 0:13:26, complete 72.61%
att-weights epoch 560, step 573, max_size:classes 13, max_size:data 516, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.769 sec/step, elapsed 0:35:39, exp. remaining 0:13:16, complete 72.88%
att-weights epoch 560, step 574, max_size:classes 13, max_size:data 418, mem_usage:GPU:0 1.0GB, num_seqs 9, 2.013 sec/step, elapsed 0:35:41, exp. remaining 0:13:05, complete 73.15%
att-weights epoch 560, step 575, max_size:classes 14, max_size:data 535, mem_usage:GPU:0 1.0GB, num_seqs 7, 2.148 sec/step, elapsed 0:35:43, exp. remaining 0:12:55, complete 73.43%
att-weights epoch 560, step 576, max_size:classes 13, max_size:data 476, mem_usage:GPU:0 1.0GB, num_seqs 8, 2.663 sec/step, elapsed 0:35:46, exp. remaining 0:12:47, complete 73.66%
att-weights epoch 560, step 577, max_size:classes 13, max_size:data 370, mem_usage:GPU:0 1.0GB, num_seqs 10, 2.795 sec/step, elapsed 0:35:48, exp. remaining 0:12:38, complete 73.90%
att-weights epoch 560, step 578, max_size:classes 15, max_size:data 533, mem_usage:GPU:0 1.0GB, num_seqs 7, 11.089 sec/step, elapsed 0:36:00, exp. remaining 0:12:32, complete 74.17%
att-weights epoch 560, step 579, max_size:classes 14, max_size:data 486, mem_usage:GPU:0 1.0GB, num_seqs 8, 3.741 sec/step, elapsed 0:36:03, exp. remaining 0:12:24, complete 74.41%
att-weights epoch 560, step 580, max_size:classes 15, max_size:data 426, mem_usage:GPU:0 1.0GB, num_seqs 9, 2.610 sec/step, elapsed 0:36:06, exp. remaining 0:12:12, complete 74.72%
att-weights epoch 560, step 581, max_size:classes 17, max_size:data 446, mem_usage:GPU:0 1.0GB, num_seqs 8, 2.049 sec/step, elapsed 0:36:08, exp. remaining 0:12:03, complete 74.99%
att-weights epoch 560, step 582, max_size:classes 13, max_size:data 431, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.984 sec/step, elapsed 0:36:10, exp. remaining 0:11:53, complete 75.26%
att-weights epoch 560, step 583, max_size:classes 14, max_size:data 461, mem_usage:GPU:0 1.0GB, num_seqs 8, 2.282 sec/step, elapsed 0:36:12, exp. remaining 0:11:42, complete 75.57%
att-weights epoch 560, step 584, max_size:classes 12, max_size:data 433, mem_usage:GPU:0 1.0GB, num_seqs 8, 2.640 sec/step, elapsed 0:36:15, exp. remaining 0:11:34, complete 75.81%
att-weights epoch 560, step 585, max_size:classes 12, max_size:data 494, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.830 sec/step, elapsed 0:36:17, exp. remaining 0:11:23, complete 76.11%
att-weights epoch 560, step 586, max_size:classes 12, max_size:data 522, mem_usage:GPU:0 1.0GB, num_seqs 7, 2.143 sec/step, elapsed 0:36:19, exp. remaining 0:11:12, complete 76.42%
att-weights epoch 560, step 587, max_size:classes 13, max_size:data 385, mem_usage:GPU:0 1.0GB, num_seqs 8, 2.218 sec/step, elapsed 0:36:21, exp. remaining 0:11:04, complete 76.66%
att-weights epoch 560, step 588, max_size:classes 12, max_size:data 566, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.769 sec/step, elapsed 0:36:23, exp. remaining 0:10:54, complete 76.93%
att-weights epoch 560, step 589, max_size:classes 12, max_size:data 402, mem_usage:GPU:0 1.0GB, num_seqs 9, 2.288 sec/step, elapsed 0:36:25, exp. remaining 0:10:44, complete 77.24%
att-weights epoch 560, step 590, max_size:classes 12, max_size:data 486, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.953 sec/step, elapsed 0:36:27, exp. remaining 0:10:34, complete 77.51%
att-weights epoch 560, step 591, max_size:classes 12, max_size:data 460, mem_usage:GPU:0 1.0GB, num_seqs 8, 2.729 sec/step, elapsed 0:36:30, exp. remaining 0:10:25, complete 77.78%
att-weights epoch 560, step 592, max_size:classes 13, max_size:data 412, mem_usage:GPU:0 1.0GB, num_seqs 9, 7.125 sec/step, elapsed 0:36:37, exp. remaining 0:10:19, complete 78.02%
att-weights epoch 560, step 593, max_size:classes 13, max_size:data 570, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.933 sec/step, elapsed 0:36:39, exp. remaining 0:10:11, complete 78.26%
att-weights epoch 560, step 594, max_size:classes 15, max_size:data 442, mem_usage:GPU:0 1.0GB, num_seqs 9, 2.137 sec/step, elapsed 0:36:41, exp. remaining 0:10:01, complete 78.53%
att-weights epoch 560, step 595, max_size:classes 12, max_size:data 414, mem_usage:GPU:0 1.0GB, num_seqs 9, 2.308 sec/step, elapsed 0:36:43, exp. remaining 0:09:51, complete 78.84%
att-weights epoch 560, step 596, max_size:classes 13, max_size:data 512, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.804 sec/step, elapsed 0:36:45, exp. remaining 0:09:42, complete 79.11%
att-weights epoch 560, step 597, max_size:classes 14, max_size:data 484, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.952 sec/step, elapsed 0:36:47, exp. remaining 0:09:31, complete 79.45%
att-weights epoch 560, step 598, max_size:classes 12, max_size:data 414, mem_usage:GPU:0 1.0GB, num_seqs 9, 2.437 sec/step, elapsed 0:36:49, exp. remaining 0:09:20, complete 79.76%
att-weights epoch 560, step 599, max_size:classes 13, max_size:data 449, mem_usage:GPU:0 1.0GB, num_seqs 8, 2.250 sec/step, elapsed 0:36:52, exp. remaining 0:09:10, complete 80.06%
att-weights epoch 560, step 600, max_size:classes 12, max_size:data 453, mem_usage:GPU:0 1.0GB, num_seqs 8, 2.497 sec/step, elapsed 0:36:54, exp. remaining 0:09:01, complete 80.37%
att-weights epoch 560, step 601, max_size:classes 15, max_size:data 473, mem_usage:GPU:0 1.0GB, num_seqs 7, 2.232 sec/step, elapsed 0:36:56, exp. remaining 0:08:49, complete 80.71%
att-weights epoch 560, step 602, max_size:classes 12, max_size:data 513, mem_usage:GPU:0 1.0GB, num_seqs 7, 2.208 sec/step, elapsed 0:36:59, exp. remaining 0:08:38, complete 81.05%
att-weights epoch 560, step 603, max_size:classes 12, max_size:data 454, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.958 sec/step, elapsed 0:37:01, exp. remaining 0:08:27, complete 81.39%
att-weights epoch 560, step 604, max_size:classes 16, max_size:data 338, mem_usage:GPU:0 1.0GB, num_seqs 9, 3.743 sec/step, elapsed 0:37:04, exp. remaining 0:08:17, complete 81.73%
att-weights epoch 560, step 605, max_size:classes 12, max_size:data 464, mem_usage:GPU:0 1.0GB, num_seqs 8, 2.107 sec/step, elapsed 0:37:06, exp. remaining 0:08:07, complete 82.03%
att-weights epoch 560, step 606, max_size:classes 12, max_size:data 369, mem_usage:GPU:0 1.0GB, num_seqs 10, 2.467 sec/step, elapsed 0:37:09, exp. remaining 0:07:57, complete 82.37%
att-weights epoch 560, step 607, max_size:classes 13, max_size:data 419, mem_usage:GPU:0 1.0GB, num_seqs 9, 2.523 sec/step, elapsed 0:37:11, exp. remaining 0:07:47, complete 82.68%
att-weights epoch 560, step 608, max_size:classes 11, max_size:data 311, mem_usage:GPU:0 1.0GB, num_seqs 9, 2.059 sec/step, elapsed 0:37:14, exp. remaining 0:07:36, complete 83.02%
att-weights epoch 560, step 609, max_size:classes 12, max_size:data 441, mem_usage:GPU:0 1.0GB, num_seqs 9, 2.623 sec/step, elapsed 0:37:16, exp. remaining 0:07:26, complete 83.36%
att-weights epoch 560, step 610, max_size:classes 11, max_size:data 395, mem_usage:GPU:0 1.0GB, num_seqs 10, 2.599 sec/step, elapsed 0:37:19, exp. remaining 0:07:17, complete 83.67%
att-weights epoch 560, step 611, max_size:classes 12, max_size:data 394, mem_usage:GPU:0 1.0GB, num_seqs 10, 2.569 sec/step, elapsed 0:37:21, exp. remaining 0:07:07, complete 83.97%
att-weights epoch 560, step 612, max_size:classes 11, max_size:data 397, mem_usage:GPU:0 1.0GB, num_seqs 10, 3.753 sec/step, elapsed 0:37:25, exp. remaining 0:06:57, complete 84.31%
att-weights epoch 560, step 613, max_size:classes 11, max_size:data 388, mem_usage:GPU:0 1.0GB, num_seqs 10, 2.728 sec/step, elapsed 0:37:28, exp. remaining 0:06:47, complete 84.65%
att-weights epoch 560, step 614, max_size:classes 11, max_size:data 440, mem_usage:GPU:0 1.0GB, num_seqs 9, 4.368 sec/step, elapsed 0:37:32, exp. remaining 0:06:37, complete 84.99%
att-weights epoch 560, step 615, max_size:classes 11, max_size:data 372, mem_usage:GPU:0 1.0GB, num_seqs 10, 2.736 sec/step, elapsed 0:37:35, exp. remaining 0:06:28, complete 85.30%
att-weights epoch 560, step 616, max_size:classes 11, max_size:data 402, mem_usage:GPU:0 1.0GB, num_seqs 9, 2.585 sec/step, elapsed 0:37:37, exp. remaining 0:06:17, complete 85.68%
att-weights epoch 560, step 617, max_size:classes 12, max_size:data 391, mem_usage:GPU:0 1.0GB, num_seqs 10, 2.937 sec/step, elapsed 0:37:40, exp. remaining 0:06:04, complete 86.12%
att-weights epoch 560, step 618, max_size:classes 12, max_size:data 393, mem_usage:GPU:0 1.0GB, num_seqs 10, 2.469 sec/step, elapsed 0:37:43, exp. remaining 0:05:54, complete 86.46%
att-weights epoch 560, step 619, max_size:classes 14, max_size:data 401, mem_usage:GPU:0 1.0GB, num_seqs 9, 1.949 sec/step, elapsed 0:37:45, exp. remaining 0:05:44, complete 86.80%
att-weights epoch 560, step 620, max_size:classes 9, max_size:data 407, mem_usage:GPU:0 1.0GB, num_seqs 9, 2.374 sec/step, elapsed 0:37:47, exp. remaining 0:05:33, complete 87.17%
att-weights epoch 560, step 621, max_size:classes 11, max_size:data 399, mem_usage:GPU:0 1.0GB, num_seqs 10, 2.746 sec/step, elapsed 0:37:50, exp. remaining 0:05:25, complete 87.44%
att-weights epoch 560, step 622, max_size:classes 12, max_size:data 384, mem_usage:GPU:0 1.0GB, num_seqs 10, 2.393 sec/step, elapsed 0:37:52, exp. remaining 0:05:16, complete 87.78%
att-weights epoch 560, step 623, max_size:classes 12, max_size:data 369, mem_usage:GPU:0 1.0GB, num_seqs 10, 2.071 sec/step, elapsed 0:37:54, exp. remaining 0:05:05, complete 88.16%
att-weights epoch 560, step 624, max_size:classes 12, max_size:data 415, mem_usage:GPU:0 1.0GB, num_seqs 9, 2.534 sec/step, elapsed 0:37:57, exp. remaining 0:04:55, complete 88.50%
att-weights epoch 560, step 625, max_size:classes 10, max_size:data 343, mem_usage:GPU:0 1.0GB, num_seqs 11, 2.812 sec/step, elapsed 0:38:00, exp. remaining 0:04:45, complete 88.87%
att-weights epoch 560, step 626, max_size:classes 11, max_size:data 300, mem_usage:GPU:0 1.0GB, num_seqs 13, 9.646 sec/step, elapsed 0:38:09, exp. remaining 0:04:34, complete 89.28%
att-weights epoch 560, step 627, max_size:classes 10, max_size:data 369, mem_usage:GPU:0 1.0GB, num_seqs 10, 2.702 sec/step, elapsed 0:38:12, exp. remaining 0:04:25, complete 89.62%
att-weights epoch 560, step 628, max_size:classes 12, max_size:data 376, mem_usage:GPU:0 1.0GB, num_seqs 10, 2.395 sec/step, elapsed 0:38:15, exp. remaining 0:04:14, complete 90.03%
att-weights epoch 560, step 629, max_size:classes 9, max_size:data 337, mem_usage:GPU:0 1.0GB, num_seqs 11, 2.634 sec/step, elapsed 0:38:17, exp. remaining 0:04:03, complete 90.40%
att-weights epoch 560, step 630, max_size:classes 9, max_size:data 490, mem_usage:GPU:0 1.0GB, num_seqs 8, 2.494 sec/step, elapsed 0:38:20, exp. remaining 0:03:55, complete 90.71%
att-weights epoch 560, step 631, max_size:classes 10, max_size:data 392, mem_usage:GPU:0 1.0GB, num_seqs 10, 2.869 sec/step, elapsed 0:38:23, exp. remaining 0:03:44, complete 91.12%
att-weights epoch 560, step 632, max_size:classes 11, max_size:data 348, mem_usage:GPU:0 1.0GB, num_seqs 11, 2.179 sec/step, elapsed 0:38:25, exp. remaining 0:03:33, complete 91.53%
att-weights epoch 560, step 633, max_size:classes 10, max_size:data 376, mem_usage:GPU:0 1.0GB, num_seqs 10, 8.248 sec/step, elapsed 0:38:33, exp. remaining 0:03:22, complete 91.94%
att-weights epoch 560, step 634, max_size:classes 10, max_size:data 357, mem_usage:GPU:0 1.0GB, num_seqs 11, 2.363 sec/step, elapsed 0:38:35, exp. remaining 0:03:13, complete 92.28%
att-weights epoch 560, step 635, max_size:classes 9, max_size:data 326, mem_usage:GPU:0 1.0GB, num_seqs 12, 2.527 sec/step, elapsed 0:38:38, exp. remaining 0:03:03, complete 92.65%
att-weights epoch 560, step 636, max_size:classes 9, max_size:data 381, mem_usage:GPU:0 1.0GB, num_seqs 10, 2.615 sec/step, elapsed 0:38:40, exp. remaining 0:02:54, complete 92.99%
att-weights epoch 560, step 637, max_size:classes 10, max_size:data 314, mem_usage:GPU:0 1.0GB, num_seqs 12, 2.960 sec/step, elapsed 0:38:43, exp. remaining 0:02:46, complete 93.33%
att-weights epoch 560, step 638, max_size:classes 10, max_size:data 356, mem_usage:GPU:0 1.0GB, num_seqs 11, 16.231 sec/step, elapsed 0:39:00, exp. remaining 0:02:35, complete 93.77%
att-weights epoch 560, step 639, max_size:classes 9, max_size:data 429, mem_usage:GPU:0 1.0GB, num_seqs 9, 17.311 sec/step, elapsed 0:39:17, exp. remaining 0:02:24, complete 94.22%
att-weights epoch 560, step 640, max_size:classes 10, max_size:data 310, mem_usage:GPU:0 1.0GB, num_seqs 12, 3.409 sec/step, elapsed 0:39:20, exp. remaining 0:02:14, complete 94.62%
att-weights epoch 560, step 641, max_size:classes 10, max_size:data 319, mem_usage:GPU:0 1.0GB, num_seqs 12, 2.986 sec/step, elapsed 0:39:23, exp. remaining 0:02:04, complete 95.00%
att-weights epoch 560, step 642, max_size:classes 9, max_size:data 331, mem_usage:GPU:0 1.0GB, num_seqs 12, 2.426 sec/step, elapsed 0:39:26, exp. remaining 0:01:53, complete 95.41%
att-weights epoch 560, step 643, max_size:classes 10, max_size:data 389, mem_usage:GPU:0 1.0GB, num_seqs 10, 2.304 sec/step, elapsed 0:39:28, exp. remaining 0:01:42, complete 95.85%
att-weights epoch 560, step 644, max_size:classes 12, max_size:data 352, mem_usage:GPU:0 1.0GB, num_seqs 11, 2.617 sec/step, elapsed 0:39:31, exp. remaining 0:01:33, complete 96.22%
att-weights epoch 560, step 645, max_size:classes 9, max_size:data 350, mem_usage:GPU:0 1.0GB, num_seqs 10, 7.141 sec/step, elapsed 0:39:38, exp. remaining 0:01:22, complete 96.67%
att-weights epoch 560, step 646, max_size:classes 9, max_size:data 380, mem_usage:GPU:0 1.0GB, num_seqs 10, 2.363 sec/step, elapsed 0:39:40, exp. remaining 0:01:09, complete 97.18%
att-weights epoch 560, step 647, max_size:classes 8, max_size:data 297, mem_usage:GPU:0 1.0GB, num_seqs 13, 3.108 sec/step, elapsed 0:39:43, exp. remaining 0:01:00, complete 97.52%
att-weights epoch 560, step 648, max_size:classes 9, max_size:data 291, mem_usage:GPU:0 1.0GB, num_seqs 13, 3.341 sec/step, elapsed 0:39:47, exp. remaining 0:00:52, complete 97.86%
att-weights epoch 560, step 649, max_size:classes 10, max_size:data 264, mem_usage:GPU:0 1.0GB, num_seqs 12, 2.152 sec/step, elapsed 0:39:49, exp. remaining 0:00:40, complete 98.33%
att-weights epoch 560, step 650, max_size:classes 9, max_size:data 341, mem_usage:GPU:0 1.0GB, num_seqs 11, 2.276 sec/step, elapsed 0:39:51, exp. remaining 0:00:31, complete 98.71%
att-weights epoch 560, step 651, max_size:classes 8, max_size:data 325, mem_usage:GPU:0 1.0GB, num_seqs 12, 2.692 sec/step, elapsed 0:39:54, exp. remaining 0:00:20, complete 99.15%
att-weights epoch 560, step 652, max_size:classes 8, max_size:data 286, mem_usage:GPU:0 1.0GB, num_seqs 13, 1.445 sec/step, elapsed 0:39:55, exp. remaining 0:00:09, complete 99.63%
att-weights epoch 560, step 653, max_size:classes 10, max_size:data 334, mem_usage:GPU:0 1.0GB, num_seqs 11, 0.933 sec/step, elapsed 0:39:56, exp. remaining 0:00:09, complete 99.63%
att-weights epoch 560, step 654, max_size:classes 8, max_size:data 304, mem_usage:GPU:0 1.0GB, num_seqs 13, 1.039 sec/step, elapsed 0:39:57, exp. remaining 0:00:09, complete 99.63%
att-weights epoch 560, step 655, max_size:classes 9, max_size:data 260, mem_usage:GPU:0 1.0GB, num_seqs 15, 1.204 sec/step, elapsed 0:39:58, exp. remaining 0:00:09, complete 99.63%
att-weights epoch 560, step 656, max_size:classes 7, max_size:data 272, mem_usage:GPU:0 1.0GB, num_seqs 10, 0.655 sec/step, elapsed 0:39:59, exp. remaining 0:00:09, complete 99.63%
att-weights epoch 560, step 657, max_size:classes 8, max_size:data 377, mem_usage:GPU:0 1.0GB, num_seqs 10, 0.763 sec/step, elapsed 0:40:00, exp. remaining 0:00:09, complete 99.63%
att-weights epoch 560, step 658, max_size:classes 7, max_size:data 283, mem_usage:GPU:0 1.0GB, num_seqs 14, 1.183 sec/step, elapsed 0:40:01, exp. remaining 0:00:09, complete 99.63%
att-weights epoch 560, step 659, max_size:classes 8, max_size:data 340, mem_usage:GPU:0 1.0GB, num_seqs 11, 1.037 sec/step, elapsed 0:40:02, exp. remaining 0:00:09, complete 99.63%
att-weights epoch 560, step 660, max_size:classes 6, max_size:data 289, mem_usage:GPU:0 1.0GB, num_seqs 13, 0.963 sec/step, elapsed 0:40:03, exp. remaining 0:00:09, complete 99.63%
att-weights epoch 560, step 661, max_size:classes 5, max_size:data 272, mem_usage:GPU:0 1.0GB, num_seqs 14, 1.129 sec/step, elapsed 0:40:04, exp. remaining 0:00:09, complete 99.63%
att-weights epoch 560, step 662, max_size:classes 5, max_size:data 323, mem_usage:GPU:0 1.0GB, num_seqs 12, 1.021 sec/step, elapsed 0:40:05, exp. remaining 0:00:09, complete 99.63%
Stats:
  mem_usage:GPU:0: Stats(mean=1.0GB, std_dev=0.0B, min=1.0GB, max=1.0GB, num_seqs=663, avg_data_len=1)
att-weights epoch 560, finished after 663 steps, 0:40:05 elapsed (27.7% computing time)
Layer 'dec_02_att_weights' Stats:
  2939 seqs, 94097088 total frames, 32016.702280 average frames
  Mean: 0.005875463426341238
  Std dev: 0.06792592613795424
  Min/max: 0.0 / 1.0
Layer 'dec_01_att_weights' Stats:
  2939 seqs, 94097088 total frames, 32016.702280 average frames
  Mean: 0.005875463429665519
  Std dev: 0.0694710437990086
  Min/max: 0.0 / 1.0
Layer 'dec_03_att_weights' Stats:
  2939 seqs, 94097088 total frames, 32016.702280 average frames
  Mean: 0.005875463431449302
  Std dev: 0.05750864642349959
  Min/max: 0.0 / 1.0
Layer 'dec_04_att_weights' Stats:
  2939 seqs, 94097088 total frames, 32016.702280 average frames
  Mean: 0.005875463440489707
  Std dev: 0.06698838358667435
  Min/max: 0.0 / 1.0
Layer 'dec_05_att_weights' Stats:
  2939 seqs, 94097088 total frames, 32016.702280 average frames
  Mean: 0.0058754634305574
  Std dev: 0.04977696275403289
  Min/max: 0.0 / 1.0
Layer 'dec_06_att_weights' Stats:
  2939 seqs, 94097088 total frames, 32016.702280 average frames
  Mean: 0.005875463429746622
  Std dev: 0.06063279376671837
  Min/max: 0.0 / 1.0
Layer 'dec_07_att_weights' Stats:
  2939 seqs, 94097088 total frames, 32016.702280 average frames
  Mean: 0.005875463439759991
  Std dev: 0.05209164785614527
  Min/max: 0.0 / 1.0
Layer 'dec_08_att_weights' Stats:
  2939 seqs, 94097088 total frames, 32016.702280 average frames
  Mean: 0.0058754634248412594
  Std dev: 0.06426800708834597
  Min/max: 0.0 / 1.0
Layer 'dec_09_att_weights' Stats:
  2939 seqs, 94097088 total frames, 32016.702280 average frames
  Mean: 0.0058754634350573634
  Std dev: 0.07641915772072543
  Min/max: 0.0 / 1.0
Layer 'dec_10_att_weights' Stats:
  2939 seqs, 94097088 total frames, 32016.702280 average frames
  Mean: 0.005875463436192477
  Std dev: 0.0764178238612876
  Min/max: 0.0 / 1.0
Layer 'dec_11_att_weights' Stats:
  2939 seqs, 94097088 total frames, 32016.702280 average frames
  Mean: 0.005875463433476306
  Std dev: 0.07641049527073235
  Min/max: 0.0 / 1.0
Layer 'dec_12_att_weights' Stats:
  2939 seqs, 94097088 total frames, 32016.702280 average frames
  Mean: 0.005875463431692542
  Std dev: 0.07462016221060196
  Min/max: 0.0 / 1.0
Quitting
+------- EPILOGUE SCRIPT -----------------------------------------------
|
| Job ID ..............: 9506153
| Stopped at ..........: Tue Jul  2 14:29:28 CEST 2019
| Resources requested .: h_rss=8G,h_rt=7200,gpu=1,pxe=ubuntu_16.04,h_vmem=1536G,s_core=0,num_proc=5,scratch_free=5G,h_fsize=20G
| Resources used ......: cpu=01:09:58, mem=16949.11398 GB s, io=12.13433 GB, vmem=4.578G, maxvmem=4.596G, last_file_cache=3.807G, last_rss=3M, max-cache=4.193G
| Memory used .........: 8.000G / 8.000G (100.0%)
| Total time used .....: 0:46:30
|
+------- EPILOGUE SCRIPT -----------------------------------------------
