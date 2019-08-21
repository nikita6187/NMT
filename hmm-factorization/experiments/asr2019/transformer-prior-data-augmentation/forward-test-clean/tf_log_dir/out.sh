+------- PROLOGUE SCRIPT -----------------------------------------------
|
| Job ID ...........: 9506152
| Started at .......: Tue Jul  2 13:42:59 CEST 2019
| Execution host ...: cluster-cn-216
| Cluster queue ....: 3-GPU-1080
| Script ...........: /var/spool/sge/cluster-cn-216/job_scripts/9506152
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
RETURNN starting up, version 20181022.053313--git-dfada43-dirty, date/time 2019-07-02-13-43-08 (UTC+0200), pid 15136, cwd /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3-prior-k6, Python /u/bahar/settings/python3-returnn-tf1.9/bin/python3
RETURNN config: /u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3-prior-k6.config
RETURNN command line options: ()
Hostname: cluster-cn-216
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
       incarnation: 9644640494688639351
  2/2: name: "/device:GPU:0"
       device_type: "GPU"
       memory_limit: 10911236096
       locality {
         bus_id: 1
         links {
         }
       }
       incarnation: 17302626731486625650
       physical_device_desc: "device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1"
Using gpu device 0: GeForce GTX 1080 Ti
Device not set explicitly, and we found a GPU, which we will use.
Model file prefix: /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3-prior-k6/net-model/network
NOTE: We will use 'sorted_reverse' seq ordering.
Setup tf.Session with options {'log_device_placement': False, 'device_count': {'GPU': 1}} ...
layer root/'data' output: Data(name='data', shape=(None, 40))
layer root/'source' output: Data(name='source_output', shape=(None, 40))
layer root/'lstm0_fw' output: Data(name='lstm0_fw_output', shape=(None, 1024), batch_dim_axis=1)
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9506152.1.3-GPU-1080/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.cc -o /var/tmp/9506152.1.3-GPU-1080/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9506152.1.3-GPU-1080/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.cc -o /var/tmp/9506152.1.3-GPU-1080/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
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
layer root/'dec_01_att_key0' output: Data(name='dec_01_att_key0_output', shape=(None, 512))
layer root/'dec_01_att_key' output: Data(name='dec_01_att_key_output', shape=(None, 8, 64))
layer root/'dec_06_att_value0' output: Data(name='dec_06_att_value0_output', shape=(None, 512))
layer root/'dec_06_att_value' output: Data(name='dec_06_att_value_output', shape=(None, 8, 64))
layer root/'dec_08_att_key0' output: Data(name='dec_08_att_key0_output', shape=(None, 512))
layer root/'dec_08_att_key' output: Data(name='dec_08_att_key_output', shape=(None, 8, 64))
layer root/'dec_07_att_key0' output: Data(name='dec_07_att_key0_output', shape=(None, 512))
layer root/'dec_07_att_key' output: Data(name='dec_07_att_key_output', shape=(None, 8, 64))
layer root/'dec_09_att_value0' output: Data(name='dec_09_att_value0_output', shape=(None, 512))
layer root/'dec_09_att_value' output: Data(name='dec_09_att_value_output', shape=(None, 8, 64))
layer root/'dec_04_att_value0' output: Data(name='dec_04_att_value0_output', shape=(None, 512))
layer root/'dec_04_att_value' output: Data(name='dec_04_att_value_output', shape=(None, 8, 64))
layer root/'dec_01_att_value0' output: Data(name='dec_01_att_value0_output', shape=(None, 512))
layer root/'dec_01_att_value' output: Data(name='dec_01_att_value_output', shape=(None, 8, 64))
layer root/'dec_03_att_value0' output: Data(name='dec_03_att_value0_output', shape=(None, 512))
layer root/'dec_03_att_value' output: Data(name='dec_03_att_value_output', shape=(None, 8, 64))
layer root/'dec_05_att_key0' output: Data(name='dec_05_att_key0_output', shape=(None, 512))
layer root/'dec_05_att_key' output: Data(name='dec_05_att_key_output', shape=(None, 8, 64))
layer root/'dec_11_att_key0' output: Data(name='dec_11_att_key0_output', shape=(None, 512))
layer root/'dec_11_att_key' output: Data(name='dec_11_att_key_output', shape=(None, 8, 64))
layer root/'dec_06_att_key0' output: Data(name='dec_06_att_key0_output', shape=(None, 512))
layer root/'dec_06_att_key' output: Data(name='dec_06_att_key_output', shape=(None, 8, 64))
layer root/'dec_02_att_key0' output: Data(name='dec_02_att_key0_output', shape=(None, 512))
layer root/'dec_02_att_key' output: Data(name='dec_02_att_key_output', shape=(None, 8, 64))
layer root/'dec_10_att_key0' output: Data(name='dec_10_att_key0_output', shape=(None, 512))
layer root/'dec_10_att_key' output: Data(name='dec_10_att_key_output', shape=(None, 8, 64))
layer root/'dec_03_att_key0' output: Data(name='dec_03_att_key0_output', shape=(None, 512))
layer root/'dec_03_att_key' output: Data(name='dec_03_att_key_output', shape=(None, 8, 64))
layer root/'dec_12_att_value0' output: Data(name='dec_12_att_value0_output', shape=(None, 512))
layer root/'dec_12_att_value' output: Data(name='dec_12_att_value_output', shape=(None, 8, 64))
layer root/'dec_11_att_value0' output: Data(name='dec_11_att_value0_output', shape=(None, 512))
layer root/'dec_11_att_value' output: Data(name='dec_11_att_value_output', shape=(None, 8, 64))
layer root/'dec_02_att_value0' output: Data(name='dec_02_att_value0_output', shape=(None, 512))
layer root/'dec_02_att_value' output: Data(name='dec_02_att_value_output', shape=(None, 8, 64))
layer root/'dec_10_att_value0' output: Data(name='dec_10_att_value0_output', shape=(None, 512))
layer root/'dec_10_att_value' output: Data(name='dec_10_att_value_output', shape=(None, 8, 64))
layer root/'dec_05_att_value0' output: Data(name='dec_05_att_value0_output', shape=(None, 512))
layer root/'dec_05_att_value' output: Data(name='dec_05_att_value_output', shape=(None, 8, 64))
layer root/'dec_09_att_key0' output: Data(name='dec_09_att_key0_output', shape=(None, 512))
layer root/'dec_09_att_key' output: Data(name='dec_09_att_key_output', shape=(None, 8, 64))
layer root/'dec_04_att_key0' output: Data(name='dec_04_att_key0_output', shape=(None, 512))
layer root/'dec_04_att_key' output: Data(name='dec_04_att_key_output', shape=(None, 8, 64))
layer root/'dec_12_att_key0' output: Data(name='dec_12_att_key0_output', shape=(None, 512))
layer root/'dec_12_att_key' output: Data(name='dec_12_att_key_output', shape=(None, 8, 64))
layer root/'dec_08_att_value0' output: Data(name='dec_08_att_value0_output', shape=(None, 512))
layer root/'dec_08_att_value' output: Data(name='dec_08_att_value_output', shape=(None, 8, 64))
layer root/'dec_07_att_value0' output: Data(name='dec_07_att_value0_output', shape=(None, 512))
layer root/'dec_07_att_value' output: Data(name='dec_07_att_value_output', shape=(None, 8, 64))
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
TF: log_dir: /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/transformer-prior/forward-test-clean/tf_log_dir/prefix:test-clean-560-2019-07-02-11-43-01
Note: There are still these uninitialized variables: ['learning_rate:0']
att-weights epoch 560, step 0, max_size:classes 120, max_size:data 3392, mem_usage:GPU:0 0.9GB, num_seqs 1, 11.003 sec/step, elapsed 0:00:18, exp. remaining 1:11:15, complete 0.42%
att-weights epoch 560, step 1, max_size:classes 103, max_size:data 2512, mem_usage:GPU:0 0.9GB, num_seqs 1, 2.273 sec/step, elapsed 0:00:20, exp. remaining 1:14:17, complete 0.46%
att-weights epoch 560, step 2, max_size:classes 101, max_size:data 2843, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.473 sec/step, elapsed 0:00:22, exp. remaining 1:14:03, complete 0.50%
att-weights epoch 560, step 3, max_size:classes 93, max_size:data 2961, mem_usage:GPU:0 0.9GB, num_seqs 1, 5.899 sec/step, elapsed 0:00:28, exp. remaining 1:27:24, complete 0.53%
att-weights epoch 560, step 4, max_size:classes 97, max_size:data 3166, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.721 sec/step, elapsed 0:00:43, exp. remaining 2:07:10, complete 0.57%
att-weights epoch 560, step 5, max_size:classes 89, max_size:data 2542, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.294 sec/step, elapsed 0:00:45, exp. remaining 2:03:07, complete 0.61%
att-weights epoch 560, step 6, max_size:classes 90, max_size:data 3062, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.505 sec/step, elapsed 0:00:47, exp. remaining 2:00:08, complete 0.65%
att-weights epoch 560, step 7, max_size:classes 91, max_size:data 2915, mem_usage:GPU:0 0.9GB, num_seqs 1, 11.777 sec/step, elapsed 0:00:58, exp. remaining 2:22:07, complete 0.69%
att-weights epoch 560, step 8, max_size:classes 101, max_size:data 2842, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.645 sec/step, elapsed 0:01:03, exp. remaining 2:26:00, complete 0.73%
att-weights epoch 560, step 9, max_size:classes 87, max_size:data 3496, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.703 sec/step, elapsed 0:01:05, exp. remaining 2:22:46, complete 0.76%
att-weights epoch 560, step 10, max_size:classes 86, max_size:data 2828, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.540 sec/step, elapsed 0:01:07, exp. remaining 2:19:33, complete 0.80%
att-weights epoch 560, step 11, max_size:classes 89, max_size:data 2713, mem_usage:GPU:0 0.9GB, num_seqs 1, 3.300 sec/step, elapsed 0:01:11, exp. remaining 2:20:02, complete 0.84%
att-weights epoch 560, step 12, max_size:classes 84, max_size:data 2486, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.100 sec/step, elapsed 0:01:12, exp. remaining 2:15:58, complete 0.88%
att-weights epoch 560, step 13, max_size:classes 81, max_size:data 2565, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.172 sec/step, elapsed 0:01:13, exp. remaining 2:12:22, complete 0.92%
att-weights epoch 560, step 14, max_size:classes 79, max_size:data 2446, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.069 sec/step, elapsed 0:01:14, exp. remaining 2:08:52, complete 0.95%
att-weights epoch 560, step 15, max_size:classes 85, max_size:data 2387, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.147 sec/step, elapsed 0:01:15, exp. remaining 2:05:46, complete 0.99%
att-weights epoch 560, step 16, max_size:classes 93, max_size:data 3278, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.489 sec/step, elapsed 0:01:17, exp. remaining 2:03:27, complete 1.03%
att-weights epoch 560, step 17, max_size:classes 85, max_size:data 2334, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.235 sec/step, elapsed 0:01:18, exp. remaining 2:00:54, complete 1.07%
att-weights epoch 560, step 18, max_size:classes 77, max_size:data 2858, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.212 sec/step, elapsed 0:01:19, exp. remaining 1:58:30, complete 1.11%
att-weights epoch 560, step 19, max_size:classes 76, max_size:data 2102, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.054 sec/step, elapsed 0:01:20, exp. remaining 1:56:01, complete 1.15%
att-weights epoch 560, step 20, max_size:classes 78, max_size:data 2307, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.047 sec/step, elapsed 0:01:21, exp. remaining 1:53:41, complete 1.18%
att-weights epoch 560, step 21, max_size:classes 73, max_size:data 3289, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.409 sec/step, elapsed 0:01:23, exp. remaining 1:52:00, complete 1.22%
att-weights epoch 560, step 22, max_size:classes 67, max_size:data 2019, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.848 sec/step, elapsed 0:01:23, exp. remaining 1:49:40, complete 1.26%
att-weights epoch 560, step 23, max_size:classes 76, max_size:data 2599, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.087 sec/step, elapsed 0:01:25, exp. remaining 1:47:47, complete 1.30%
att-weights epoch 560, step 24, max_size:classes 77, max_size:data 2004, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.894 sec/step, elapsed 0:01:25, exp. remaining 1:45:46, complete 1.34%
att-weights epoch 560, step 25, max_size:classes 71, max_size:data 2550, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.995 sec/step, elapsed 0:01:26, exp. remaining 1:43:59, complete 1.37%
att-weights epoch 560, step 26, max_size:classes 67, max_size:data 2375, mem_usage:GPU:0 0.9GB, num_seqs 1, 2.244 sec/step, elapsed 0:01:29, exp. remaining 1:43:44, complete 1.41%
att-weights epoch 560, step 27, max_size:classes 75, max_size:data 3082, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.221 sec/step, elapsed 0:01:30, exp. remaining 1:42:21, complete 1.45%
att-weights epoch 560, step 28, max_size:classes 83, max_size:data 2615, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.196 sec/step, elapsed 0:01:31, exp. remaining 1:41:01, complete 1.49%
att-weights epoch 560, step 29, max_size:classes 79, max_size:data 2612, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.204 sec/step, elapsed 0:01:32, exp. remaining 1:39:44, complete 1.53%
att-weights epoch 560, step 30, max_size:classes 70, max_size:data 2595, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.127 sec/step, elapsed 0:01:33, exp. remaining 1:38:27, complete 1.56%
att-weights epoch 560, step 31, max_size:classes 72, max_size:data 2221, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.925 sec/step, elapsed 0:01:34, exp. remaining 1:37:01, complete 1.60%
att-weights epoch 560, step 32, max_size:classes 67, max_size:data 2332, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.953 sec/step, elapsed 0:01:35, exp. remaining 1:35:41, complete 1.64%
att-weights epoch 560, step 33, max_size:classes 67, max_size:data 2285, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.947 sec/step, elapsed 0:01:36, exp. remaining 1:34:24, complete 1.68%
att-weights epoch 560, step 34, max_size:classes 71, max_size:data 3162, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.299 sec/step, elapsed 0:01:38, exp. remaining 1:33:30, complete 1.72%
att-weights epoch 560, step 35, max_size:classes 76, max_size:data 2174, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.169 sec/step, elapsed 0:01:39, exp. remaining 1:32:31, complete 1.76%
att-weights epoch 560, step 36, max_size:classes 74, max_size:data 1888, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.900 sec/step, elapsed 0:01:40, exp. remaining 1:31:20, complete 1.79%
att-weights epoch 560, step 37, max_size:classes 83, max_size:data 2327, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.982 sec/step, elapsed 0:01:41, exp. remaining 1:30:17, complete 1.83%
att-weights epoch 560, step 38, max_size:classes 80, max_size:data 3005, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.326 sec/step, elapsed 0:01:42, exp. remaining 1:29:34, complete 1.87%
att-weights epoch 560, step 39, max_size:classes 74, max_size:data 2229, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.949 sec/step, elapsed 0:01:43, exp. remaining 1:28:33, complete 1.91%
att-weights epoch 560, step 40, max_size:classes 73, max_size:data 2842, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.270 sec/step, elapsed 0:01:44, exp. remaining 1:27:51, complete 1.95%
att-weights epoch 560, step 41, max_size:classes 66, max_size:data 2753, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.177 sec/step, elapsed 0:01:45, exp. remaining 1:27:06, complete 1.98%
att-weights epoch 560, step 42, max_size:classes 72, max_size:data 2151, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.934 sec/step, elapsed 0:01:46, exp. remaining 1:26:10, complete 2.02%
att-weights epoch 560, step 43, max_size:classes 64, max_size:data 1948, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.869 sec/step, elapsed 0:01:47, exp. remaining 1:25:14, complete 2.06%
att-weights epoch 560, step 44, max_size:classes 72, max_size:data 2026, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.803 sec/step, elapsed 0:01:48, exp. remaining 1:24:16, complete 2.10%
att-weights epoch 560, step 45, max_size:classes 67, max_size:data 2086, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.794 sec/step, elapsed 0:01:49, exp. remaining 1:21:51, complete 2.18%
att-weights epoch 560, step 46, max_size:classes 74, max_size:data 2210, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.938 sec/step, elapsed 0:01:50, exp. remaining 1:21:06, complete 2.21%
att-weights epoch 560, step 47, max_size:classes 76, max_size:data 2217, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.936 sec/step, elapsed 0:01:51, exp. remaining 1:20:22, complete 2.25%
att-weights epoch 560, step 48, max_size:classes 60, max_size:data 2455, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.998 sec/step, elapsed 0:01:52, exp. remaining 1:19:42, complete 2.29%
att-weights epoch 560, step 49, max_size:classes 65, max_size:data 2179, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.847 sec/step, elapsed 0:01:52, exp. remaining 1:18:58, complete 2.33%
att-weights epoch 560, step 50, max_size:classes 67, max_size:data 2038, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.848 sec/step, elapsed 0:01:53, exp. remaining 1:18:15, complete 2.37%
att-weights epoch 560, step 51, max_size:classes 64, max_size:data 2449, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.900 sec/step, elapsed 0:01:54, exp. remaining 1:17:35, complete 2.40%
att-weights epoch 560, step 52, max_size:classes 75, max_size:data 2810, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.107 sec/step, elapsed 0:01:55, exp. remaining 1:17:05, complete 2.44%
att-weights epoch 560, step 53, max_size:classes 69, max_size:data 2617, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.033 sec/step, elapsed 0:01:56, exp. remaining 1:16:32, complete 2.48%
att-weights epoch 560, step 54, max_size:classes 68, max_size:data 1992, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.498 sec/step, elapsed 0:01:58, exp. remaining 1:16:19, complete 2.52%
att-weights epoch 560, step 55, max_size:classes 66, max_size:data 1674, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.776 sec/step, elapsed 0:01:59, exp. remaining 1:15:38, complete 2.56%
att-weights epoch 560, step 56, max_size:classes 62, max_size:data 2091, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.947 sec/step, elapsed 0:02:00, exp. remaining 1:15:05, complete 2.60%
att-weights epoch 560, step 57, max_size:classes 64, max_size:data 1903, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.952 sec/step, elapsed 0:02:01, exp. remaining 1:14:34, complete 2.63%
att-weights epoch 560, step 58, max_size:classes 68, max_size:data 2119, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.915 sec/step, elapsed 0:02:01, exp. remaining 1:14:01, complete 2.67%
att-weights epoch 560, step 59, max_size:classes 65, max_size:data 2152, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.841 sec/step, elapsed 0:02:02, exp. remaining 1:12:24, complete 2.75%
att-weights epoch 560, step 60, max_size:classes 74, max_size:data 2149, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.864 sec/step, elapsed 0:02:03, exp. remaining 1:11:53, complete 2.79%
att-weights epoch 560, step 61, max_size:classes 64, max_size:data 2135, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.952 sec/step, elapsed 0:02:04, exp. remaining 1:10:27, complete 2.86%
att-weights epoch 560, step 62, max_size:classes 64, max_size:data 1653, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.844 sec/step, elapsed 0:02:05, exp. remaining 1:09:58, complete 2.90%
att-weights epoch 560, step 63, max_size:classes 56, max_size:data 2250, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.899 sec/step, elapsed 0:02:06, exp. remaining 1:09:32, complete 2.94%
att-weights epoch 560, step 64, max_size:classes 61, max_size:data 2448, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.080 sec/step, elapsed 0:02:07, exp. remaining 1:09:12, complete 2.98%
att-weights epoch 560, step 65, max_size:classes 66, max_size:data 2057, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.934 sec/step, elapsed 0:02:08, exp. remaining 1:08:48, complete 3.02%
att-weights epoch 560, step 66, max_size:classes 67, max_size:data 1853, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.756 sec/step, elapsed 0:02:09, exp. remaining 1:08:19, complete 3.05%
att-weights epoch 560, step 67, max_size:classes 68, max_size:data 2016, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.807 sec/step, elapsed 0:02:09, exp. remaining 1:07:52, complete 3.09%
att-weights epoch 560, step 68, max_size:classes 63, max_size:data 1971, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.615 sec/step, elapsed 0:02:11, exp. remaining 1:07:51, complete 3.13%
att-weights epoch 560, step 69, max_size:classes 67, max_size:data 2076, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.869 sec/step, elapsed 0:02:12, exp. remaining 1:06:37, complete 3.21%
att-weights epoch 560, step 70, max_size:classes 68, max_size:data 1758, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.519 sec/step, elapsed 0:02:13, exp. remaining 1:06:33, complete 3.24%
att-weights epoch 560, step 71, max_size:classes 67, max_size:data 2036, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.981 sec/step, elapsed 0:02:14, exp. remaining 1:06:14, complete 3.28%
att-weights epoch 560, step 72, max_size:classes 57, max_size:data 2015, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.759 sec/step, elapsed 0:02:15, exp. remaining 1:05:49, complete 3.32%
att-weights epoch 560, step 73, max_size:classes 65, max_size:data 1995, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.795 sec/step, elapsed 0:02:16, exp. remaining 1:05:26, complete 3.36%
att-weights epoch 560, step 74, max_size:classes 60, max_size:data 2719, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.986 sec/step, elapsed 0:02:17, exp. remaining 1:05:08, complete 3.40%
att-weights epoch 560, step 75, max_size:classes 55, max_size:data 2029, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.798 sec/step, elapsed 0:02:18, exp. remaining 1:04:46, complete 3.44%
att-weights epoch 560, step 76, max_size:classes 53, max_size:data 2237, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.850 sec/step, elapsed 0:02:19, exp. remaining 1:04:25, complete 3.47%
att-weights epoch 560, step 77, max_size:classes 57, max_size:data 2054, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.800 sec/step, elapsed 0:02:19, exp. remaining 1:03:21, complete 3.55%
att-weights epoch 560, step 78, max_size:classes 67, max_size:data 1864, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.332 sec/step, elapsed 0:02:21, exp. remaining 1:03:15, complete 3.59%
att-weights epoch 560, step 79, max_size:classes 62, max_size:data 1823, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.875 sec/step, elapsed 0:02:22, exp. remaining 1:02:56, complete 3.63%
att-weights epoch 560, step 80, max_size:classes 58, max_size:data 2002, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.831 sec/step, elapsed 0:02:22, exp. remaining 1:02:37, complete 3.66%
att-weights epoch 560, step 81, max_size:classes 61, max_size:data 1542, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.625 sec/step, elapsed 0:02:23, exp. remaining 1:02:14, complete 3.70%
att-weights epoch 560, step 82, max_size:classes 59, max_size:data 2013, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.812 sec/step, elapsed 0:02:24, exp. remaining 1:01:16, complete 3.78%
att-weights epoch 560, step 83, max_size:classes 61, max_size:data 2002, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.816 sec/step, elapsed 0:02:25, exp. remaining 1:00:58, complete 3.82%
att-weights epoch 560, step 84, max_size:classes 59, max_size:data 2066, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.798 sec/step, elapsed 0:02:25, exp. remaining 1:00:03, complete 3.89%
att-weights epoch 560, step 85, max_size:classes 66, max_size:data 2001, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.078 sec/step, elapsed 0:02:27, exp. remaining 0:59:53, complete 3.93%
att-weights epoch 560, step 86, max_size:classes 62, max_size:data 1962, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.434 sec/step, elapsed 0:02:28, exp. remaining 0:59:52, complete 3.97%
att-weights epoch 560, step 87, max_size:classes 56, max_size:data 1629, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.753 sec/step, elapsed 0:02:29, exp. remaining 0:58:59, complete 4.05%
att-weights epoch 560, step 88, max_size:classes 61, max_size:data 2006, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.923 sec/step, elapsed 0:02:30, exp. remaining 0:58:13, complete 4.12%
att-weights epoch 560, step 89, max_size:classes 62, max_size:data 1623, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.680 sec/step, elapsed 0:02:30, exp. remaining 0:57:22, complete 4.20%
att-weights epoch 560, step 90, max_size:classes 62, max_size:data 2258, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.864 sec/step, elapsed 0:02:31, exp. remaining 0:56:37, complete 4.27%
att-weights epoch 560, step 91, max_size:classes 61, max_size:data 1813, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.366 sec/step, elapsed 0:02:33, exp. remaining 0:56:36, complete 4.31%
att-weights epoch 560, step 92, max_size:classes 69, max_size:data 2368, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.053 sec/step, elapsed 0:02:34, exp. remaining 0:56:28, complete 4.35%
att-weights epoch 560, step 93, max_size:classes 60, max_size:data 1797, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.183 sec/step, elapsed 0:02:35, exp. remaining 0:55:52, complete 4.43%
att-weights epoch 560, step 94, max_size:classes 59, max_size:data 1987, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.824 sec/step, elapsed 0:02:36, exp. remaining 0:55:40, complete 4.47%
att-weights epoch 560, step 95, max_size:classes 63, max_size:data 2106, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.871 sec/step, elapsed 0:02:37, exp. remaining 0:55:00, complete 4.54%
att-weights epoch 560, step 96, max_size:classes 60, max_size:data 1992, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.523 sec/step, elapsed 0:02:38, exp. remaining 0:54:34, complete 4.62%
att-weights epoch 560, step 97, max_size:classes 59, max_size:data 1992, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.371 sec/step, elapsed 0:02:39, exp. remaining 0:54:06, complete 4.69%
att-weights epoch 560, step 98, max_size:classes 60, max_size:data 1981, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.346 sec/step, elapsed 0:02:41, exp. remaining 0:53:38, complete 4.77%
att-weights epoch 560, step 99, max_size:classes 53, max_size:data 1657, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.316 sec/step, elapsed 0:02:42, exp. remaining 0:53:38, complete 4.81%
att-weights epoch 560, step 100, max_size:classes 57, max_size:data 2047, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.918 sec/step, elapsed 0:02:43, exp. remaining 0:53:03, complete 4.89%
att-weights epoch 560, step 101, max_size:classes 55, max_size:data 2016, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.877 sec/step, elapsed 0:02:44, exp. remaining 0:52:28, complete 4.96%
att-weights epoch 560, step 102, max_size:classes 54, max_size:data 1798, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.155 sec/step, elapsed 0:02:45, exp. remaining 0:52:00, complete 5.04%
att-weights epoch 560, step 103, max_size:classes 64, max_size:data 2088, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.910 sec/step, elapsed 0:02:46, exp. remaining 0:51:28, complete 5.11%
att-weights epoch 560, step 104, max_size:classes 64, max_size:data 1641, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.218 sec/step, elapsed 0:02:47, exp. remaining 0:51:26, complete 5.15%
att-weights epoch 560, step 105, max_size:classes 57, max_size:data 1674, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.139 sec/step, elapsed 0:02:48, exp. remaining 0:51:23, complete 5.19%
att-weights epoch 560, step 106, max_size:classes 58, max_size:data 1832, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.612 sec/step, elapsed 0:02:50, exp. remaining 0:51:28, complete 5.23%
att-weights epoch 560, step 107, max_size:classes 60, max_size:data 1907, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.765 sec/step, elapsed 0:02:52, exp. remaining 0:51:13, complete 5.31%
att-weights epoch 560, step 108, max_size:classes 59, max_size:data 2126, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.934 sec/step, elapsed 0:02:53, exp. remaining 0:50:43, complete 5.38%
att-weights epoch 560, step 109, max_size:classes 62, max_size:data 1775, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.336 sec/step, elapsed 0:02:54, exp. remaining 0:50:21, complete 5.46%
att-weights epoch 560, step 110, max_size:classes 57, max_size:data 1910, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.524 sec/step, elapsed 0:02:55, exp. remaining 0:50:03, complete 5.53%
att-weights epoch 560, step 111, max_size:classes 56, max_size:data 1723, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.304 sec/step, elapsed 0:02:57, exp. remaining 0:50:04, complete 5.57%
att-weights epoch 560, step 112, max_size:classes 60, max_size:data 2000, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.762 sec/step, elapsed 0:02:59, exp. remaining 0:50:12, complete 5.61%
att-weights epoch 560, step 113, max_size:classes 51, max_size:data 2034, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.899 sec/step, elapsed 0:02:59, exp. remaining 0:49:44, complete 5.69%
att-weights epoch 560, step 114, max_size:classes 53, max_size:data 1609, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.793 sec/step, elapsed 0:03:00, exp. remaining 0:49:15, complete 5.76%
att-weights epoch 560, step 115, max_size:classes 58, max_size:data 2099, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.725 sec/step, elapsed 0:03:02, exp. remaining 0:49:22, complete 5.80%
att-weights epoch 560, step 116, max_size:classes 57, max_size:data 1842, mem_usage:GPU:0 0.9GB, num_seqs 2, 4.658 sec/step, elapsed 0:03:07, exp. remaining 0:49:56, complete 5.88%
att-weights epoch 560, step 117, max_size:classes 58, max_size:data 1721, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.650 sec/step, elapsed 0:03:08, exp. remaining 0:49:41, complete 5.95%
att-weights epoch 560, step 118, max_size:classes 53, max_size:data 1701, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.478 sec/step, elapsed 0:03:10, exp. remaining 0:49:44, complete 5.99%
att-weights epoch 560, step 119, max_size:classes 60, max_size:data 1503, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.409 sec/step, elapsed 0:03:11, exp. remaining 0:49:46, complete 6.03%
att-weights epoch 560, step 120, max_size:classes 53, max_size:data 1565, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.759 sec/step, elapsed 0:03:12, exp. remaining 0:49:18, complete 6.11%
att-weights epoch 560, step 121, max_size:classes 64, max_size:data 2540, mem_usage:GPU:0 0.9GB, num_seqs 1, 3.382 sec/step, elapsed 0:03:15, exp. remaining 0:49:31, complete 6.18%
att-weights epoch 560, step 122, max_size:classes 58, max_size:data 1800, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.848 sec/step, elapsed 0:03:17, exp. remaining 0:49:20, complete 6.26%
att-weights epoch 560, step 123, max_size:classes 54, max_size:data 1685, mem_usage:GPU:0 0.9GB, num_seqs 2, 3.026 sec/step, elapsed 0:03:20, exp. remaining 0:49:27, complete 6.34%
att-weights epoch 560, step 124, max_size:classes 54, max_size:data 2083, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.909 sec/step, elapsed 0:03:21, exp. remaining 0:49:02, complete 6.41%
att-weights epoch 560, step 125, max_size:classes 55, max_size:data 1634, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.292 sec/step, elapsed 0:03:22, exp. remaining 0:49:03, complete 6.45%
att-weights epoch 560, step 126, max_size:classes 58, max_size:data 1637, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.213 sec/step, elapsed 0:03:24, exp. remaining 0:49:02, complete 6.49%
att-weights epoch 560, step 127, max_size:classes 57, max_size:data 2351, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.952 sec/step, elapsed 0:03:25, exp. remaining 0:48:39, complete 6.56%
att-weights epoch 560, step 128, max_size:classes 56, max_size:data 2246, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.888 sec/step, elapsed 0:03:25, exp. remaining 0:48:15, complete 6.64%
att-weights epoch 560, step 129, max_size:classes 56, max_size:data 1765, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.289 sec/step, elapsed 0:03:27, exp. remaining 0:47:58, complete 6.72%
att-weights epoch 560, step 130, max_size:classes 56, max_size:data 1672, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.216 sec/step, elapsed 0:03:28, exp. remaining 0:47:40, complete 6.79%
att-weights epoch 560, step 131, max_size:classes 59, max_size:data 1815, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.203 sec/step, elapsed 0:03:29, exp. remaining 0:47:22, complete 6.87%
att-weights epoch 560, step 132, max_size:classes 53, max_size:data 1795, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.381 sec/step, elapsed 0:03:31, exp. remaining 0:47:07, complete 6.95%
att-weights epoch 560, step 133, max_size:classes 54, max_size:data 1648, mem_usage:GPU:0 0.9GB, num_seqs 2, 4.939 sec/step, elapsed 0:03:36, exp. remaining 0:47:39, complete 7.02%
att-weights epoch 560, step 134, max_size:classes 51, max_size:data 1416, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.787 sec/step, elapsed 0:03:36, exp. remaining 0:47:17, complete 7.10%
att-weights epoch 560, step 135, max_size:classes 58, max_size:data 2251, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.976 sec/step, elapsed 0:03:37, exp. remaining 0:46:57, complete 7.18%
att-weights epoch 560, step 136, max_size:classes 61, max_size:data 1652, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.160 sec/step, elapsed 0:03:38, exp. remaining 0:46:40, complete 7.25%
att-weights epoch 560, step 137, max_size:classes 59, max_size:data 1673, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.129 sec/step, elapsed 0:03:40, exp. remaining 0:46:22, complete 7.33%
att-weights epoch 560, step 138, max_size:classes 55, max_size:data 1680, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.237 sec/step, elapsed 0:03:41, exp. remaining 0:46:07, complete 7.40%
att-weights epoch 560, step 139, max_size:classes 52, max_size:data 1404, mem_usage:GPU:0 0.9GB, num_seqs 2, 3.721 sec/step, elapsed 0:03:45, exp. remaining 0:46:23, complete 7.48%
att-weights epoch 560, step 140, max_size:classes 53, max_size:data 1877, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.239 sec/step, elapsed 0:03:46, exp. remaining 0:46:07, complete 7.56%
att-weights epoch 560, step 141, max_size:classes 56, max_size:data 1855, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.499 sec/step, elapsed 0:03:47, exp. remaining 0:45:56, complete 7.63%
att-weights epoch 560, step 142, max_size:classes 54, max_size:data 1711, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.229 sec/step, elapsed 0:03:48, exp. remaining 0:45:41, complete 7.71%
att-weights epoch 560, step 143, max_size:classes 53, max_size:data 1996, mem_usage:GPU:0 0.9GB, num_seqs 2, 11.318 sec/step, elapsed 0:04:00, exp. remaining 0:47:26, complete 7.79%
att-weights epoch 560, step 144, max_size:classes 52, max_size:data 1901, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.402 sec/step, elapsed 0:04:01, exp. remaining 0:47:12, complete 7.86%
att-weights epoch 560, step 145, max_size:classes 49, max_size:data 1739, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.040 sec/step, elapsed 0:04:02, exp. remaining 0:46:55, complete 7.94%
att-weights epoch 560, step 146, max_size:classes 53, max_size:data 1512, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.182 sec/step, elapsed 0:04:03, exp. remaining 0:46:39, complete 8.02%
att-weights epoch 560, step 147, max_size:classes 50, max_size:data 1536, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.100 sec/step, elapsed 0:04:05, exp. remaining 0:46:37, complete 8.05%
att-weights epoch 560, step 148, max_size:classes 56, max_size:data 1901, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.378 sec/step, elapsed 0:04:06, exp. remaining 0:46:24, complete 8.13%
att-weights epoch 560, step 149, max_size:classes 52, max_size:data 1483, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.077 sec/step, elapsed 0:04:07, exp. remaining 0:46:08, complete 8.21%
att-weights epoch 560, step 150, max_size:classes 55, max_size:data 1605, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.171 sec/step, elapsed 0:04:08, exp. remaining 0:45:53, complete 8.28%
att-weights epoch 560, step 151, max_size:classes 56, max_size:data 1623, mem_usage:GPU:0 0.9GB, num_seqs 2, 9.312 sec/step, elapsed 0:04:17, exp. remaining 0:47:08, complete 8.36%
att-weights epoch 560, step 152, max_size:classes 49, max_size:data 1577, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.529 sec/step, elapsed 0:04:19, exp. remaining 0:46:57, complete 8.44%
att-weights epoch 560, step 153, max_size:classes 56, max_size:data 1658, mem_usage:GPU:0 0.9GB, num_seqs 2, 18.813 sec/step, elapsed 0:04:38, exp. remaining 0:49:51, complete 8.51%
att-weights epoch 560, step 154, max_size:classes 50, max_size:data 1440, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.110 sec/step, elapsed 0:04:39, exp. remaining 0:49:34, complete 8.59%
att-weights epoch 560, step 155, max_size:classes 56, max_size:data 1855, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.292 sec/step, elapsed 0:04:40, exp. remaining 0:49:19, complete 8.66%
att-weights epoch 560, step 156, max_size:classes 50, max_size:data 2145, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.815 sec/step, elapsed 0:04:41, exp. remaining 0:48:59, complete 8.74%
att-weights epoch 560, step 157, max_size:classes 54, max_size:data 1890, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.473 sec/step, elapsed 0:04:43, exp. remaining 0:48:47, complete 8.82%
att-weights epoch 560, step 158, max_size:classes 52, max_size:data 1482, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.038 sec/step, elapsed 0:04:44, exp. remaining 0:48:43, complete 8.85%
att-weights epoch 560, step 159, max_size:classes 50, max_size:data 1423, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.030 sec/step, elapsed 0:04:45, exp. remaining 0:48:40, complete 8.89%
att-weights epoch 560, step 160, max_size:classes 52, max_size:data 1460, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.060 sec/step, elapsed 0:04:46, exp. remaining 0:48:24, complete 8.97%
att-weights epoch 560, step 161, max_size:classes 49, max_size:data 1432, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.072 sec/step, elapsed 0:04:47, exp. remaining 0:48:08, complete 9.05%
att-weights epoch 560, step 162, max_size:classes 51, max_size:data 1789, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.359 sec/step, elapsed 0:04:48, exp. remaining 0:47:55, complete 9.12%
att-weights epoch 560, step 163, max_size:classes 48, max_size:data 1659, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.159 sec/step, elapsed 0:04:49, exp. remaining 0:47:40, complete 9.20%
att-weights epoch 560, step 164, max_size:classes 50, max_size:data 1451, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.980 sec/step, elapsed 0:04:50, exp. remaining 0:47:23, complete 9.27%
att-weights epoch 560, step 165, max_size:classes 52, max_size:data 1468, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.041 sec/step, elapsed 0:04:51, exp. remaining 0:47:08, complete 9.35%
att-weights epoch 560, step 166, max_size:classes 50, max_size:data 1677, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.411 sec/step, elapsed 0:04:53, exp. remaining 0:46:56, complete 9.43%
att-weights epoch 560, step 167, max_size:classes 50, max_size:data 1431, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.669 sec/step, elapsed 0:04:53, exp. remaining 0:46:38, complete 9.50%
att-weights epoch 560, step 168, max_size:classes 52, max_size:data 2065, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.911 sec/step, elapsed 0:04:54, exp. remaining 0:46:22, complete 9.58%
att-weights epoch 560, step 169, max_size:classes 48, max_size:data 1692, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.076 sec/step, elapsed 0:04:55, exp. remaining 0:46:07, complete 9.66%
att-weights epoch 560, step 170, max_size:classes 53, max_size:data 1797, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.148 sec/step, elapsed 0:04:56, exp. remaining 0:45:54, complete 9.73%
att-weights epoch 560, step 171, max_size:classes 46, max_size:data 1874, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.688 sec/step, elapsed 0:04:59, exp. remaining 0:45:55, complete 9.81%
att-weights epoch 560, step 172, max_size:classes 49, max_size:data 1573, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.216 sec/step, elapsed 0:05:00, exp. remaining 0:45:42, complete 9.89%
att-weights epoch 560, step 173, max_size:classes 50, max_size:data 1614, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.221 sec/step, elapsed 0:05:02, exp. remaining 0:45:30, complete 9.96%
att-weights epoch 560, step 174, max_size:classes 46, max_size:data 1732, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.296 sec/step, elapsed 0:05:03, exp. remaining 0:45:19, complete 10.04%
att-weights epoch 560, step 175, max_size:classes 47, max_size:data 1349, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.139 sec/step, elapsed 0:05:04, exp. remaining 0:45:06, complete 10.11%
att-weights epoch 560, step 176, max_size:classes 54, max_size:data 1529, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.119 sec/step, elapsed 0:05:05, exp. remaining 0:44:53, complete 10.19%
att-weights epoch 560, step 177, max_size:classes 55, max_size:data 1588, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.015 sec/step, elapsed 0:05:06, exp. remaining 0:44:40, complete 10.27%
att-weights epoch 560, step 178, max_size:classes 49, max_size:data 1504, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.164 sec/step, elapsed 0:05:07, exp. remaining 0:44:28, complete 10.34%
att-weights epoch 560, step 179, max_size:classes 49, max_size:data 1403, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.998 sec/step, elapsed 0:05:08, exp. remaining 0:44:15, complete 10.42%
att-weights epoch 560, step 180, max_size:classes 51, max_size:data 1370, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.942 sec/step, elapsed 0:05:09, exp. remaining 0:44:01, complete 10.50%
att-weights epoch 560, step 181, max_size:classes 44, max_size:data 1668, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.163 sec/step, elapsed 0:05:10, exp. remaining 0:43:50, complete 10.57%
att-weights epoch 560, step 182, max_size:classes 43, max_size:data 1783, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.505 sec/step, elapsed 0:05:12, exp. remaining 0:43:52, complete 10.61%
att-weights epoch 560, step 183, max_size:classes 46, max_size:data 1506, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.012 sec/step, elapsed 0:05:13, exp. remaining 0:43:39, complete 10.69%
att-weights epoch 560, step 184, max_size:classes 48, max_size:data 1390, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.982 sec/step, elapsed 0:05:14, exp. remaining 0:43:27, complete 10.76%
att-weights epoch 560, step 185, max_size:classes 47, max_size:data 1393, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.003 sec/step, elapsed 0:05:15, exp. remaining 0:43:14, complete 10.84%
att-weights epoch 560, step 186, max_size:classes 48, max_size:data 1573, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.205 sec/step, elapsed 0:05:17, exp. remaining 0:43:02, complete 10.95%
att-weights epoch 560, step 187, max_size:classes 50, max_size:data 1749, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.656 sec/step, elapsed 0:05:19, exp. remaining 0:42:55, complete 11.03%
att-weights epoch 560, step 188, max_size:classes 49, max_size:data 1502, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.262 sec/step, elapsed 0:05:20, exp. remaining 0:42:45, complete 11.11%
att-weights epoch 560, step 189, max_size:classes 50, max_size:data 1553, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.122 sec/step, elapsed 0:05:21, exp. remaining 0:42:35, complete 11.18%
att-weights epoch 560, step 190, max_size:classes 51, max_size:data 1460, mem_usage:GPU:0 0.9GB, num_seqs 2, 3.617 sec/step, elapsed 0:05:25, exp. remaining 0:42:44, complete 11.26%
att-weights epoch 560, step 191, max_size:classes 46, max_size:data 2147, mem_usage:GPU:0 0.9GB, num_seqs 1, 7.479 sec/step, elapsed 0:05:32, exp. remaining 0:43:23, complete 11.34%
att-weights epoch 560, step 192, max_size:classes 51, max_size:data 1499, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.174 sec/step, elapsed 0:05:33, exp. remaining 0:43:12, complete 11.41%
att-weights epoch 560, step 193, max_size:classes 50, max_size:data 1482, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.160 sec/step, elapsed 0:05:35, exp. remaining 0:43:02, complete 11.49%
att-weights epoch 560, step 194, max_size:classes 50, max_size:data 1441, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.149 sec/step, elapsed 0:05:36, exp. remaining 0:42:42, complete 11.60%
att-weights epoch 560, step 195, max_size:classes 48, max_size:data 1283, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.632 sec/step, elapsed 0:05:37, exp. remaining 0:42:35, complete 11.68%
att-weights epoch 560, step 196, max_size:classes 54, max_size:data 1322, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.076 sec/step, elapsed 0:05:39, exp. remaining 0:42:24, complete 11.76%
att-weights epoch 560, step 197, max_size:classes 47, max_size:data 1495, mem_usage:GPU:0 0.9GB, num_seqs 2, 5.187 sec/step, elapsed 0:05:44, exp. remaining 0:42:44, complete 11.83%
att-weights epoch 560, step 198, max_size:classes 52, max_size:data 1335, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.130 sec/step, elapsed 0:05:45, exp. remaining 0:42:34, complete 11.91%
att-weights epoch 560, step 199, max_size:classes 49, max_size:data 1389, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.986 sec/step, elapsed 0:05:46, exp. remaining 0:42:14, complete 12.02%
att-weights epoch 560, step 200, max_size:classes 43, max_size:data 1367, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.868 sec/step, elapsed 0:05:47, exp. remaining 0:42:02, complete 12.10%
att-weights epoch 560, step 201, max_size:classes 46, max_size:data 1414, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.101 sec/step, elapsed 0:05:48, exp. remaining 0:41:52, complete 12.18%
att-weights epoch 560, step 202, max_size:classes 50, max_size:data 1438, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.987 sec/step, elapsed 0:05:49, exp. remaining 0:41:41, complete 12.25%
att-weights epoch 560, step 203, max_size:classes 54, max_size:data 1245, mem_usage:GPU:0 0.9GB, num_seqs 3, 6.624 sec/step, elapsed 0:05:55, exp. remaining 0:42:02, complete 12.37%
att-weights epoch 560, step 204, max_size:classes 51, max_size:data 1381, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.037 sec/step, elapsed 0:05:56, exp. remaining 0:41:51, complete 12.44%
att-weights epoch 560, step 205, max_size:classes 45, max_size:data 1507, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.964 sec/step, elapsed 0:05:57, exp. remaining 0:41:40, complete 12.52%
att-weights epoch 560, step 206, max_size:classes 46, max_size:data 1390, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.964 sec/step, elapsed 0:05:58, exp. remaining 0:41:30, complete 12.60%
att-weights epoch 560, step 207, max_size:classes 44, max_size:data 1421, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.865 sec/step, elapsed 0:05:59, exp. remaining 0:41:19, complete 12.67%
att-weights epoch 560, step 208, max_size:classes 45, max_size:data 1315, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.310 sec/step, elapsed 0:06:01, exp. remaining 0:41:11, complete 12.75%
att-weights epoch 560, step 209, max_size:classes 41, max_size:data 1312, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.800 sec/step, elapsed 0:06:01, exp. remaining 0:40:59, complete 12.82%
att-weights epoch 560, step 210, max_size:classes 47, max_size:data 1507, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.000 sec/step, elapsed 0:06:02, exp. remaining 0:40:49, complete 12.90%
att-weights epoch 560, step 211, max_size:classes 48, max_size:data 1338, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.844 sec/step, elapsed 0:06:03, exp. remaining 0:40:38, complete 12.98%
att-weights epoch 560, step 212, max_size:classes 44, max_size:data 1296, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.231 sec/step, elapsed 0:06:04, exp. remaining 0:40:30, complete 13.05%
att-weights epoch 560, step 213, max_size:classes 44, max_size:data 1170, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.812 sec/step, elapsed 0:06:05, exp. remaining 0:40:19, complete 13.13%
att-weights epoch 560, step 214, max_size:classes 42, max_size:data 1627, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.025 sec/step, elapsed 0:06:06, exp. remaining 0:40:10, complete 13.21%
att-weights epoch 560, step 215, max_size:classes 44, max_size:data 1426, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.090 sec/step, elapsed 0:06:07, exp. remaining 0:40:01, complete 13.28%
att-weights epoch 560, step 216, max_size:classes 46, max_size:data 1608, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.034 sec/step, elapsed 0:06:08, exp. remaining 0:39:52, complete 13.36%
att-weights epoch 560, step 217, max_size:classes 42, max_size:data 1520, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.104 sec/step, elapsed 0:06:09, exp. remaining 0:39:36, complete 13.47%
att-weights epoch 560, step 218, max_size:classes 44, max_size:data 1699, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.061 sec/step, elapsed 0:06:11, exp. remaining 0:39:27, complete 13.55%
att-weights epoch 560, step 219, max_size:classes 43, max_size:data 1373, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.891 sec/step, elapsed 0:06:11, exp. remaining 0:39:17, complete 13.63%
att-weights epoch 560, step 220, max_size:classes 44, max_size:data 1546, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.958 sec/step, elapsed 0:06:12, exp. remaining 0:39:08, complete 13.70%
att-weights epoch 560, step 221, max_size:classes 41, max_size:data 1383, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.907 sec/step, elapsed 0:06:13, exp. remaining 0:38:59, complete 13.78%
att-weights epoch 560, step 222, max_size:classes 45, max_size:data 1464, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.993 sec/step, elapsed 0:06:14, exp. remaining 0:38:50, complete 13.85%
att-weights epoch 560, step 223, max_size:classes 52, max_size:data 1360, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.100 sec/step, elapsed 0:06:15, exp. remaining 0:38:42, complete 13.93%
att-weights epoch 560, step 224, max_size:classes 43, max_size:data 1365, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.095 sec/step, elapsed 0:06:17, exp. remaining 0:38:34, complete 14.01%
att-weights epoch 560, step 225, max_size:classes 46, max_size:data 1436, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.988 sec/step, elapsed 0:06:17, exp. remaining 0:38:25, complete 14.08%
att-weights epoch 560, step 226, max_size:classes 42, max_size:data 1276, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.292 sec/step, elapsed 0:06:19, exp. remaining 0:38:19, complete 14.16%
att-weights epoch 560, step 227, max_size:classes 51, max_size:data 1391, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.187 sec/step, elapsed 0:06:20, exp. remaining 0:38:12, complete 14.24%
att-weights epoch 560, step 228, max_size:classes 43, max_size:data 1285, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.006 sec/step, elapsed 0:06:21, exp. remaining 0:38:03, complete 14.31%
att-weights epoch 560, step 229, max_size:classes 39, max_size:data 1506, mem_usage:GPU:0 0.9GB, num_seqs 2, 15.175 sec/step, elapsed 0:06:36, exp. remaining 0:39:19, complete 14.39%
att-weights epoch 560, step 230, max_size:classes 43, max_size:data 1460, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.989 sec/step, elapsed 0:06:37, exp. remaining 0:39:11, complete 14.47%
att-weights epoch 560, step 231, max_size:classes 45, max_size:data 1775, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.180 sec/step, elapsed 0:06:38, exp. remaining 0:39:03, complete 14.54%
att-weights epoch 560, step 232, max_size:classes 41, max_size:data 1517, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.980 sec/step, elapsed 0:06:39, exp. remaining 0:38:55, complete 14.62%
att-weights epoch 560, step 233, max_size:classes 42, max_size:data 1468, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.994 sec/step, elapsed 0:06:40, exp. remaining 0:38:46, complete 14.69%
att-weights epoch 560, step 234, max_size:classes 43, max_size:data 1476, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.975 sec/step, elapsed 0:06:41, exp. remaining 0:38:31, complete 14.81%
att-weights epoch 560, step 235, max_size:classes 41, max_size:data 1247, mem_usage:GPU:0 0.9GB, num_seqs 2, 5.401 sec/step, elapsed 0:06:47, exp. remaining 0:38:48, complete 14.89%
att-weights epoch 560, step 236, max_size:classes 46, max_size:data 1579, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.362 sec/step, elapsed 0:06:48, exp. remaining 0:38:42, complete 14.96%
att-weights epoch 560, step 237, max_size:classes 45, max_size:data 1345, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.976 sec/step, elapsed 0:06:49, exp. remaining 0:38:26, complete 15.08%
att-weights epoch 560, step 238, max_size:classes 45, max_size:data 1144, mem_usage:GPU:0 0.9GB, num_seqs 2, 3.811 sec/step, elapsed 0:06:53, exp. remaining 0:38:27, complete 15.19%
att-weights epoch 560, step 239, max_size:classes 39, max_size:data 1546, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.192 sec/step, elapsed 0:06:54, exp. remaining 0:38:13, complete 15.31%
att-weights epoch 560, step 240, max_size:classes 39, max_size:data 1294, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.067 sec/step, elapsed 0:06:56, exp. remaining 0:38:05, complete 15.42%
att-weights epoch 560, step 241, max_size:classes 42, max_size:data 1363, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.306 sec/step, elapsed 0:06:57, exp. remaining 0:37:58, complete 15.50%
att-weights epoch 560, step 242, max_size:classes 42, max_size:data 1394, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.957 sec/step, elapsed 0:06:58, exp. remaining 0:37:50, complete 15.57%
att-weights epoch 560, step 243, max_size:classes 42, max_size:data 1317, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.377 sec/step, elapsed 0:07:00, exp. remaining 0:37:38, complete 15.69%
att-weights epoch 560, step 244, max_size:classes 45, max_size:data 1466, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.933 sec/step, elapsed 0:07:01, exp. remaining 0:37:24, complete 15.80%
att-weights epoch 560, step 245, max_size:classes 47, max_size:data 1727, mem_usage:GPU:0 0.9GB, num_seqs 2, 4.997 sec/step, elapsed 0:07:06, exp. remaining 0:37:31, complete 15.92%
att-weights epoch 560, step 246, max_size:classes 42, max_size:data 1327, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.353 sec/step, elapsed 0:07:07, exp. remaining 0:37:25, complete 15.99%
att-weights epoch 560, step 247, max_size:classes 48, max_size:data 1154, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.210 sec/step, elapsed 0:07:08, exp. remaining 0:37:19, complete 16.07%
att-weights epoch 560, step 248, max_size:classes 39, max_size:data 1312, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.205 sec/step, elapsed 0:07:09, exp. remaining 0:37:06, complete 16.18%
att-weights epoch 560, step 249, max_size:classes 42, max_size:data 1285, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.270 sec/step, elapsed 0:07:11, exp. remaining 0:36:54, complete 16.30%
att-weights epoch 560, step 250, max_size:classes 38, max_size:data 1150, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.799 sec/step, elapsed 0:07:12, exp. remaining 0:36:46, complete 16.37%
att-weights epoch 560, step 251, max_size:classes 42, max_size:data 1729, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.128 sec/step, elapsed 0:07:13, exp. remaining 0:36:33, complete 16.49%
att-weights epoch 560, step 252, max_size:classes 43, max_size:data 1320, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.252 sec/step, elapsed 0:07:14, exp. remaining 0:36:28, complete 16.56%
att-weights epoch 560, step 253, max_size:classes 41, max_size:data 1274, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.229 sec/step, elapsed 0:07:15, exp. remaining 0:36:22, complete 16.64%
att-weights epoch 560, step 254, max_size:classes 41, max_size:data 1183, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.443 sec/step, elapsed 0:07:17, exp. remaining 0:36:23, complete 16.68%
att-weights epoch 560, step 255, max_size:classes 45, max_size:data 1491, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.040 sec/step, elapsed 0:07:18, exp. remaining 0:36:22, complete 16.72%
att-weights epoch 560, step 256, max_size:classes 36, max_size:data 1407, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.955 sec/step, elapsed 0:07:19, exp. remaining 0:36:09, complete 16.83%
att-weights epoch 560, step 257, max_size:classes 43, max_size:data 1177, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.141 sec/step, elapsed 0:07:20, exp. remaining 0:35:57, complete 16.95%
att-weights epoch 560, step 258, max_size:classes 40, max_size:data 1216, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.221 sec/step, elapsed 0:07:21, exp. remaining 0:35:45, complete 17.06%
att-weights epoch 560, step 259, max_size:classes 40, max_size:data 1432, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.972 sec/step, elapsed 0:07:22, exp. remaining 0:35:39, complete 17.14%
att-weights epoch 560, step 260, max_size:classes 35, max_size:data 1278, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.202 sec/step, elapsed 0:07:23, exp. remaining 0:35:33, complete 17.21%
att-weights epoch 560, step 261, max_size:classes 39, max_size:data 1517, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.041 sec/step, elapsed 0:07:24, exp. remaining 0:35:27, complete 17.29%
att-weights epoch 560, step 262, max_size:classes 40, max_size:data 1480, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.554 sec/step, elapsed 0:07:26, exp. remaining 0:35:17, complete 17.40%
att-weights epoch 560, step 263, max_size:classes 40, max_size:data 1284, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.587 sec/step, elapsed 0:07:26, exp. remaining 0:35:03, complete 17.52%
att-weights epoch 560, step 264, max_size:classes 39, max_size:data 2057, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.789 sec/step, elapsed 0:07:27, exp. remaining 0:34:56, complete 17.60%
att-weights epoch 560, step 265, max_size:classes 42, max_size:data 1203, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.180 sec/step, elapsed 0:07:28, exp. remaining 0:34:50, complete 17.67%
att-weights epoch 560, step 266, max_size:classes 43, max_size:data 1256, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.145 sec/step, elapsed 0:07:29, exp. remaining 0:34:39, complete 17.79%
att-weights epoch 560, step 267, max_size:classes 40, max_size:data 1202, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.230 sec/step, elapsed 0:07:31, exp. remaining 0:34:29, complete 17.90%
att-weights epoch 560, step 268, max_size:classes 40, max_size:data 1392, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.926 sec/step, elapsed 0:07:32, exp. remaining 0:34:17, complete 18.02%
att-weights epoch 560, step 269, max_size:classes 46, max_size:data 1241, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.085 sec/step, elapsed 0:07:33, exp. remaining 0:34:06, complete 18.13%
att-weights epoch 560, step 270, max_size:classes 43, max_size:data 1740, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.073 sec/step, elapsed 0:07:34, exp. remaining 0:33:55, complete 18.24%
att-weights epoch 560, step 271, max_size:classes 39, max_size:data 977, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.032 sec/step, elapsed 0:07:35, exp. remaining 0:33:44, complete 18.36%
att-weights epoch 560, step 272, max_size:classes 38, max_size:data 1255, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.299 sec/step, elapsed 0:07:36, exp. remaining 0:33:34, complete 18.47%
att-weights epoch 560, step 273, max_size:classes 40, max_size:data 1530, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.916 sec/step, elapsed 0:07:37, exp. remaining 0:33:28, complete 18.55%
att-weights epoch 560, step 274, max_size:classes 35, max_size:data 1362, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.838 sec/step, elapsed 0:07:38, exp. remaining 0:33:17, complete 18.66%
att-weights epoch 560, step 275, max_size:classes 41, max_size:data 1170, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.125 sec/step, elapsed 0:07:39, exp. remaining 0:33:12, complete 18.74%
att-weights epoch 560, step 276, max_size:classes 42, max_size:data 1069, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.318 sec/step, elapsed 0:07:40, exp. remaining 0:33:02, complete 18.85%
att-weights epoch 560, step 277, max_size:classes 37, max_size:data 1234, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.277 sec/step, elapsed 0:07:42, exp. remaining 0:32:58, complete 18.93%
att-weights epoch 560, step 278, max_size:classes 42, max_size:data 1302, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.163 sec/step, elapsed 0:07:43, exp. remaining 0:32:48, complete 19.05%
att-weights epoch 560, step 279, max_size:classes 42, max_size:data 1146, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.516 sec/step, elapsed 0:07:44, exp. remaining 0:32:40, complete 19.16%
att-weights epoch 560, step 280, max_size:classes 42, max_size:data 1166, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.233 sec/step, elapsed 0:07:45, exp. remaining 0:32:31, complete 19.27%
att-weights epoch 560, step 281, max_size:classes 39, max_size:data 1190, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.282 sec/step, elapsed 0:07:47, exp. remaining 0:32:27, complete 19.35%
att-weights epoch 560, step 282, max_size:classes 39, max_size:data 1343, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.913 sec/step, elapsed 0:07:48, exp. remaining 0:32:21, complete 19.43%
att-weights epoch 560, step 283, max_size:classes 41, max_size:data 1094, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.171 sec/step, elapsed 0:07:49, exp. remaining 0:32:12, complete 19.54%
att-weights epoch 560, step 284, max_size:classes 38, max_size:data 1770, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.152 sec/step, elapsed 0:07:50, exp. remaining 0:32:07, complete 19.62%
att-weights epoch 560, step 285, max_size:classes 38, max_size:data 1250, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.314 sec/step, elapsed 0:07:51, exp. remaining 0:31:59, complete 19.73%
att-weights epoch 560, step 286, max_size:classes 40, max_size:data 1416, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.937 sec/step, elapsed 0:07:52, exp. remaining 0:31:49, complete 19.85%
att-weights epoch 560, step 287, max_size:classes 39, max_size:data 1170, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.284 sec/step, elapsed 0:07:54, exp. remaining 0:31:40, complete 19.96%
att-weights epoch 560, step 288, max_size:classes 40, max_size:data 1259, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.287 sec/step, elapsed 0:07:56, exp. remaining 0:31:36, complete 20.08%
att-weights epoch 560, step 289, max_size:classes 37, max_size:data 1309, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.174 sec/step, elapsed 0:07:58, exp. remaining 0:31:31, complete 20.19%
att-weights epoch 560, step 290, max_size:classes 39, max_size:data 1630, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.271 sec/step, elapsed 0:07:59, exp. remaining 0:31:22, complete 20.31%
att-weights epoch 560, step 291, max_size:classes 36, max_size:data 1397, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.983 sec/step, elapsed 0:08:00, exp. remaining 0:31:17, complete 20.38%
att-weights epoch 560, step 292, max_size:classes 37, max_size:data 994, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.183 sec/step, elapsed 0:08:01, exp. remaining 0:31:09, complete 20.50%
att-weights epoch 560, step 293, max_size:classes 41, max_size:data 1528, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.409 sec/step, elapsed 0:08:03, exp. remaining 0:31:01, complete 20.61%
att-weights epoch 560, step 294, max_size:classes 35, max_size:data 1141, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.247 sec/step, elapsed 0:08:04, exp. remaining 0:30:53, complete 20.73%
att-weights epoch 560, step 295, max_size:classes 37, max_size:data 1129, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.559 sec/step, elapsed 0:08:06, exp. remaining 0:30:46, complete 20.84%
att-weights epoch 560, step 296, max_size:classes 36, max_size:data 1057, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.385 sec/step, elapsed 0:08:07, exp. remaining 0:30:43, complete 20.92%
att-weights epoch 560, step 297, max_size:classes 38, max_size:data 1303, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.436 sec/step, elapsed 0:08:08, exp. remaining 0:30:40, complete 20.99%
att-weights epoch 560, step 298, max_size:classes 35, max_size:data 1184, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.100 sec/step, elapsed 0:08:10, exp. remaining 0:30:35, complete 21.07%
att-weights epoch 560, step 299, max_size:classes 38, max_size:data 1114, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.310 sec/step, elapsed 0:08:11, exp. remaining 0:30:28, complete 21.18%
att-weights epoch 560, step 300, max_size:classes 38, max_size:data 1400, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.061 sec/step, elapsed 0:08:12, exp. remaining 0:30:23, complete 21.26%
att-weights epoch 560, step 301, max_size:classes 37, max_size:data 1191, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.253 sec/step, elapsed 0:08:13, exp. remaining 0:30:20, complete 21.34%
att-weights epoch 560, step 302, max_size:classes 36, max_size:data 1304, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.468 sec/step, elapsed 0:08:15, exp. remaining 0:30:13, complete 21.45%
att-weights epoch 560, step 303, max_size:classes 39, max_size:data 1036, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.136 sec/step, elapsed 0:08:16, exp. remaining 0:30:05, complete 21.56%
att-weights epoch 560, step 304, max_size:classes 37, max_size:data 1122, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.148 sec/step, elapsed 0:08:17, exp. remaining 0:29:57, complete 21.68%
att-weights epoch 560, step 305, max_size:classes 39, max_size:data 1180, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.967 sec/step, elapsed 0:08:18, exp. remaining 0:29:48, complete 21.79%
att-weights epoch 560, step 306, max_size:classes 40, max_size:data 1468, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.881 sec/step, elapsed 0:08:19, exp. remaining 0:29:39, complete 21.91%
att-weights epoch 560, step 307, max_size:classes 38, max_size:data 1459, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.978 sec/step, elapsed 0:08:20, exp. remaining 0:29:35, complete 21.98%
att-weights epoch 560, step 308, max_size:classes 41, max_size:data 1263, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.222 sec/step, elapsed 0:08:21, exp. remaining 0:29:31, complete 22.06%
att-weights epoch 560, step 309, max_size:classes 34, max_size:data 1165, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.838 sec/step, elapsed 0:08:22, exp. remaining 0:29:22, complete 22.18%
att-weights epoch 560, step 310, max_size:classes 34, max_size:data 1363, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.018 sec/step, elapsed 0:08:23, exp. remaining 0:29:14, complete 22.29%
att-weights epoch 560, step 311, max_size:classes 35, max_size:data 1114, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.160 sec/step, elapsed 0:08:24, exp. remaining 0:29:07, complete 22.40%
att-weights epoch 560, step 312, max_size:classes 37, max_size:data 1214, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.822 sec/step, elapsed 0:08:26, exp. remaining 0:29:02, complete 22.52%
att-weights epoch 560, step 313, max_size:classes 36, max_size:data 1099, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.648 sec/step, elapsed 0:08:27, exp. remaining 0:28:56, complete 22.63%
att-weights epoch 560, step 314, max_size:classes 35, max_size:data 1193, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.192 sec/step, elapsed 0:08:29, exp. remaining 0:28:49, complete 22.75%
att-weights epoch 560, step 315, max_size:classes 33, max_size:data 1109, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.346 sec/step, elapsed 0:08:30, exp. remaining 0:28:38, complete 22.90%
att-weights epoch 560, step 316, max_size:classes 38, max_size:data 1266, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.180 sec/step, elapsed 0:08:31, exp. remaining 0:28:31, complete 23.02%
att-weights epoch 560, step 317, max_size:classes 37, max_size:data 1411, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.993 sec/step, elapsed 0:08:32, exp. remaining 0:28:23, complete 23.13%
att-weights epoch 560, step 318, max_size:classes 40, max_size:data 1290, mem_usage:GPU:0 0.9GB, num_seqs 3, 6.841 sec/step, elapsed 0:08:39, exp. remaining 0:28:35, complete 23.24%
att-weights epoch 560, step 319, max_size:classes 33, max_size:data 1018, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.148 sec/step, elapsed 0:08:41, exp. remaining 0:28:31, complete 23.36%
att-weights epoch 560, step 320, max_size:classes 42, max_size:data 1142, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.274 sec/step, elapsed 0:08:42, exp. remaining 0:28:24, complete 23.47%
att-weights epoch 560, step 321, max_size:classes 35, max_size:data 1074, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.079 sec/step, elapsed 0:08:44, exp. remaining 0:28:17, complete 23.59%
att-weights epoch 560, step 322, max_size:classes 33, max_size:data 1045, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.566 sec/step, elapsed 0:08:46, exp. remaining 0:28:15, complete 23.70%
att-weights epoch 560, step 323, max_size:classes 36, max_size:data 1122, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.192 sec/step, elapsed 0:08:47, exp. remaining 0:28:11, complete 23.78%
att-weights epoch 560, step 324, max_size:classes 36, max_size:data 970, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.417 sec/step, elapsed 0:08:49, exp. remaining 0:28:02, complete 23.93%
att-weights epoch 560, step 325, max_size:classes 34, max_size:data 1202, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.282 sec/step, elapsed 0:08:50, exp. remaining 0:27:55, complete 24.05%
att-weights epoch 560, step 326, max_size:classes 35, max_size:data 1115, mem_usage:GPU:0 0.9GB, num_seqs 3, 19.273 sec/step, elapsed 0:09:09, exp. remaining 0:28:45, complete 24.16%
att-weights epoch 560, step 327, max_size:classes 34, max_size:data 1098, mem_usage:GPU:0 0.9GB, num_seqs 3, 3.933 sec/step, elapsed 0:09:13, exp. remaining 0:28:47, complete 24.27%
att-weights epoch 560, step 328, max_size:classes 33, max_size:data 1092, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.429 sec/step, elapsed 0:09:16, exp. remaining 0:28:40, complete 24.43%
att-weights epoch 560, step 329, max_size:classes 34, max_size:data 1187, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.146 sec/step, elapsed 0:09:17, exp. remaining 0:28:29, complete 24.58%
att-weights epoch 560, step 330, max_size:classes 33, max_size:data 983, mem_usage:GPU:0 0.9GB, num_seqs 3, 4.226 sec/step, elapsed 0:09:21, exp. remaining 0:28:32, complete 24.69%
att-weights epoch 560, step 331, max_size:classes 33, max_size:data 1132, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.747 sec/step, elapsed 0:09:24, exp. remaining 0:28:26, complete 24.85%
att-weights epoch 560, step 332, max_size:classes 43, max_size:data 1543, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.042 sec/step, elapsed 0:09:25, exp. remaining 0:28:15, complete 25.00%
att-weights epoch 560, step 333, max_size:classes 38, max_size:data 925, mem_usage:GPU:0 0.9GB, num_seqs 4, 36.150 sec/step, elapsed 0:10:01, exp. remaining 0:29:53, complete 25.11%
att-weights epoch 560, step 334, max_size:classes 38, max_size:data 1012, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.065 sec/step, elapsed 0:10:03, exp. remaining 0:29:48, complete 25.23%
att-weights epoch 560, step 335, max_size:classes 31, max_size:data 1095, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.352 sec/step, elapsed 0:10:04, exp. remaining 0:29:41, complete 25.34%
att-weights epoch 560, step 336, max_size:classes 34, max_size:data 1155, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.165 sec/step, elapsed 0:10:06, exp. remaining 0:29:34, complete 25.46%
att-weights epoch 560, step 337, max_size:classes 34, max_size:data 968, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.271 sec/step, elapsed 0:10:07, exp. remaining 0:29:24, complete 25.61%
att-weights epoch 560, step 338, max_size:classes 35, max_size:data 938, mem_usage:GPU:0 0.9GB, num_seqs 4, 3.608 sec/step, elapsed 0:10:10, exp. remaining 0:29:20, complete 25.76%
att-weights epoch 560, step 339, max_size:classes 35, max_size:data 1149, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.106 sec/step, elapsed 0:10:12, exp. remaining 0:29:13, complete 25.88%
att-weights epoch 560, step 340, max_size:classes 33, max_size:data 957, mem_usage:GPU:0 0.9GB, num_seqs 4, 19.853 sec/step, elapsed 0:10:31, exp. remaining 0:29:59, complete 25.99%
att-weights epoch 560, step 341, max_size:classes 31, max_size:data 961, mem_usage:GPU:0 0.9GB, num_seqs 4, 4.880 sec/step, elapsed 0:10:36, exp. remaining 0:30:02, complete 26.11%
att-weights epoch 560, step 342, max_size:classes 37, max_size:data 1125, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.300 sec/step, elapsed 0:10:38, exp. remaining 0:29:51, complete 26.26%
att-weights epoch 560, step 343, max_size:classes 35, max_size:data 1203, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.034 sec/step, elapsed 0:10:39, exp. remaining 0:29:44, complete 26.37%
att-weights epoch 560, step 344, max_size:classes 30, max_size:data 1177, mem_usage:GPU:0 0.9GB, num_seqs 3, 0.877 sec/step, elapsed 0:10:39, exp. remaining 0:29:36, complete 26.49%
att-weights epoch 560, step 345, max_size:classes 32, max_size:data 1036, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.232 sec/step, elapsed 0:10:41, exp. remaining 0:29:29, complete 26.60%
att-weights epoch 560, step 346, max_size:classes 30, max_size:data 976, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.506 sec/step, elapsed 0:10:42, exp. remaining 0:29:26, complete 26.68%
att-weights epoch 560, step 347, max_size:classes 32, max_size:data 988, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.700 sec/step, elapsed 0:10:44, exp. remaining 0:29:17, complete 26.83%
att-weights epoch 560, step 348, max_size:classes 33, max_size:data 1056, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.090 sec/step, elapsed 0:10:45, exp. remaining 0:29:06, complete 26.98%
att-weights epoch 560, step 349, max_size:classes 33, max_size:data 1128, mem_usage:GPU:0 0.9GB, num_seqs 3, 0.988 sec/step, elapsed 0:10:46, exp. remaining 0:28:55, complete 27.14%
att-weights epoch 560, step 350, max_size:classes 32, max_size:data 1090, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.109 sec/step, elapsed 0:10:47, exp. remaining 0:28:45, complete 27.29%
att-weights epoch 560, step 351, max_size:classes 34, max_size:data 994, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.472 sec/step, elapsed 0:10:49, exp. remaining 0:28:39, complete 27.40%
att-weights epoch 560, step 352, max_size:classes 30, max_size:data 1058, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.014 sec/step, elapsed 0:10:50, exp. remaining 0:28:32, complete 27.52%
att-weights epoch 560, step 353, max_size:classes 35, max_size:data 1019, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.004 sec/step, elapsed 0:10:51, exp. remaining 0:28:21, complete 27.67%
att-weights epoch 560, step 354, max_size:classes 34, max_size:data 1140, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.065 sec/step, elapsed 0:10:52, exp. remaining 0:28:11, complete 27.82%
att-weights epoch 560, step 355, max_size:classes 31, max_size:data 1532, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.850 sec/step, elapsed 0:10:53, exp. remaining 0:28:04, complete 27.94%
att-weights epoch 560, step 356, max_size:classes 34, max_size:data 878, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.334 sec/step, elapsed 0:10:54, exp. remaining 0:27:58, complete 28.05%
att-weights epoch 560, step 357, max_size:classes 35, max_size:data 949, mem_usage:GPU:0 0.9GB, num_seqs 4, 3.072 sec/step, elapsed 0:10:57, exp. remaining 0:27:56, complete 28.17%
att-weights epoch 560, step 358, max_size:classes 33, max_size:data 920, mem_usage:GPU:0 0.9GB, num_seqs 4, 4.687 sec/step, elapsed 0:11:02, exp. remaining 0:27:55, complete 28.32%
att-weights epoch 560, step 359, max_size:classes 29, max_size:data 964, mem_usage:GPU:0 0.9GB, num_seqs 4, 7.893 sec/step, elapsed 0:11:10, exp. remaining 0:28:06, complete 28.44%
att-weights epoch 560, step 360, max_size:classes 33, max_size:data 1135, mem_usage:GPU:0 0.9GB, num_seqs 3, 10.106 sec/step, elapsed 0:11:20, exp. remaining 0:28:22, complete 28.55%
att-weights epoch 560, step 361, max_size:classes 31, max_size:data 1057, mem_usage:GPU:0 0.9GB, num_seqs 3, 8.266 sec/step, elapsed 0:11:28, exp. remaining 0:28:33, complete 28.66%
att-weights epoch 560, step 362, max_size:classes 36, max_size:data 970, mem_usage:GPU:0 0.9GB, num_seqs 4, 28.911 sec/step, elapsed 0:11:57, exp. remaining 0:29:31, complete 28.82%
att-weights epoch 560, step 363, max_size:classes 34, max_size:data 844, mem_usage:GPU:0 0.9GB, num_seqs 4, 9.721 sec/step, elapsed 0:12:07, exp. remaining 0:29:45, complete 28.93%
att-weights epoch 560, step 364, max_size:classes 30, max_size:data 1150, mem_usage:GPU:0 0.9GB, num_seqs 3, 5.454 sec/step, elapsed 0:12:12, exp. remaining 0:29:49, complete 29.05%
att-weights epoch 560, step 365, max_size:classes 31, max_size:data 956, mem_usage:GPU:0 0.9GB, num_seqs 3, 5.393 sec/step, elapsed 0:12:17, exp. remaining 0:29:52, complete 29.16%
att-weights epoch 560, step 366, max_size:classes 31, max_size:data 1058, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.783 sec/step, elapsed 0:12:20, exp. remaining 0:29:49, complete 29.27%
att-weights epoch 560, step 367, max_size:classes 30, max_size:data 875, mem_usage:GPU:0 0.9GB, num_seqs 4, 27.509 sec/step, elapsed 0:12:48, exp. remaining 0:30:42, complete 29.43%
att-weights epoch 560, step 368, max_size:classes 26, max_size:data 1004, mem_usage:GPU:0 0.9GB, num_seqs 3, 5.478 sec/step, elapsed 0:12:53, exp. remaining 0:30:41, complete 29.58%
att-weights epoch 560, step 369, max_size:classes 28, max_size:data 1027, mem_usage:GPU:0 0.9GB, num_seqs 3, 5.803 sec/step, elapsed 0:12:59, exp. remaining 0:30:42, complete 29.73%
att-weights epoch 560, step 370, max_size:classes 34, max_size:data 1008, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.342 sec/step, elapsed 0:13:00, exp. remaining 0:30:35, complete 29.85%
att-weights epoch 560, step 371, max_size:classes 31, max_size:data 905, mem_usage:GPU:0 0.9GB, num_seqs 4, 3.831 sec/step, elapsed 0:13:04, exp. remaining 0:30:34, complete 29.96%
att-weights epoch 560, step 372, max_size:classes 31, max_size:data 821, mem_usage:GPU:0 0.9GB, num_seqs 3, 3.466 sec/step, elapsed 0:13:08, exp. remaining 0:30:28, complete 30.11%
att-weights epoch 560, step 373, max_size:classes 31, max_size:data 1045, mem_usage:GPU:0 0.9GB, num_seqs 3, 10.367 sec/step, elapsed 0:13:18, exp. remaining 0:30:39, complete 30.27%
att-weights epoch 560, step 374, max_size:classes 30, max_size:data 988, mem_usage:GPU:0 0.9GB, num_seqs 3, 7.715 sec/step, elapsed 0:13:26, exp. remaining 0:30:47, complete 30.38%
att-weights epoch 560, step 375, max_size:classes 31, max_size:data 1131, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.096 sec/step, elapsed 0:13:27, exp. remaining 0:30:39, complete 30.50%
att-weights epoch 560, step 376, max_size:classes 31, max_size:data 854, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.410 sec/step, elapsed 0:13:28, exp. remaining 0:30:33, complete 30.61%
att-weights epoch 560, step 377, max_size:classes 30, max_size:data 931, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.207 sec/step, elapsed 0:13:29, exp. remaining 0:30:26, complete 30.73%
att-weights epoch 560, step 378, max_size:classes 33, max_size:data 861, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.202 sec/step, elapsed 0:13:31, exp. remaining 0:30:15, complete 30.88%
att-weights epoch 560, step 379, max_size:classes 32, max_size:data 1067, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.147 sec/step, elapsed 0:13:32, exp. remaining 0:30:08, complete 30.99%
att-weights epoch 560, step 380, max_size:classes 32, max_size:data 1134, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.151 sec/step, elapsed 0:13:33, exp. remaining 0:30:01, complete 31.11%
att-weights epoch 560, step 381, max_size:classes 32, max_size:data 964, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.432 sec/step, elapsed 0:13:34, exp. remaining 0:29:55, complete 31.22%
att-weights epoch 560, step 382, max_size:classes 30, max_size:data 881, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.646 sec/step, elapsed 0:13:36, exp. remaining 0:29:45, complete 31.37%
att-weights epoch 560, step 383, max_size:classes 29, max_size:data 1013, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.379 sec/step, elapsed 0:13:37, exp. remaining 0:29:36, complete 31.53%
att-weights epoch 560, step 384, max_size:classes 32, max_size:data 1032, mem_usage:GPU:0 0.9GB, num_seqs 3, 3.975 sec/step, elapsed 0:13:41, exp. remaining 0:29:32, complete 31.68%
att-weights epoch 560, step 385, max_size:classes 31, max_size:data 923, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.063 sec/step, elapsed 0:13:42, exp. remaining 0:29:22, complete 31.83%
att-weights epoch 560, step 386, max_size:classes 30, max_size:data 1059, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.209 sec/step, elapsed 0:13:44, exp. remaining 0:29:15, complete 31.95%
att-weights epoch 560, step 387, max_size:classes 30, max_size:data 929, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.639 sec/step, elapsed 0:13:45, exp. remaining 0:29:06, complete 32.10%
att-weights epoch 560, step 388, max_size:classes 27, max_size:data 1026, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.509 sec/step, elapsed 0:13:47, exp. remaining 0:28:57, complete 32.25%
att-weights epoch 560, step 389, max_size:classes 30, max_size:data 855, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.421 sec/step, elapsed 0:13:48, exp. remaining 0:28:48, complete 32.40%
att-weights epoch 560, step 390, max_size:classes 32, max_size:data 1040, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.186 sec/step, elapsed 0:13:49, exp. remaining 0:28:42, complete 32.52%
att-weights epoch 560, step 391, max_size:classes 32, max_size:data 989, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.508 sec/step, elapsed 0:13:51, exp. remaining 0:28:36, complete 32.63%
att-weights epoch 560, step 392, max_size:classes 31, max_size:data 904, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.281 sec/step, elapsed 0:13:52, exp. remaining 0:28:26, complete 32.79%
att-weights epoch 560, step 393, max_size:classes 33, max_size:data 826, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.572 sec/step, elapsed 0:13:54, exp. remaining 0:28:18, complete 32.94%
att-weights epoch 560, step 394, max_size:classes 28, max_size:data 842, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.420 sec/step, elapsed 0:13:55, exp. remaining 0:28:09, complete 33.09%
att-weights epoch 560, step 395, max_size:classes 27, max_size:data 1009, mem_usage:GPU:0 0.9GB, num_seqs 3, 0.865 sec/step, elapsed 0:13:56, exp. remaining 0:27:59, complete 33.24%
att-weights epoch 560, step 396, max_size:classes 28, max_size:data 913, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.358 sec/step, elapsed 0:13:57, exp. remaining 0:27:50, complete 33.40%
att-weights epoch 560, step 397, max_size:classes 26, max_size:data 922, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.440 sec/step, elapsed 0:13:59, exp. remaining 0:27:45, complete 33.51%
att-weights epoch 560, step 398, max_size:classes 29, max_size:data 801, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.184 sec/step, elapsed 0:14:00, exp. remaining 0:27:39, complete 33.63%
att-weights epoch 560, step 399, max_size:classes 26, max_size:data 784, mem_usage:GPU:0 0.9GB, num_seqs 3, 0.998 sec/step, elapsed 0:14:01, exp. remaining 0:27:29, complete 33.78%
att-weights epoch 560, step 400, max_size:classes 32, max_size:data 1038, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.210 sec/step, elapsed 0:14:02, exp. remaining 0:27:23, complete 33.89%
att-weights epoch 560, step 401, max_size:classes 27, max_size:data 853, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.202 sec/step, elapsed 0:14:03, exp. remaining 0:27:14, complete 34.05%
att-weights epoch 560, step 402, max_size:classes 29, max_size:data 845, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.439 sec/step, elapsed 0:14:05, exp. remaining 0:27:06, complete 34.20%
att-weights epoch 560, step 403, max_size:classes 32, max_size:data 907, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.973 sec/step, elapsed 0:14:07, exp. remaining 0:26:59, complete 34.35%
att-weights epoch 560, step 404, max_size:classes 29, max_size:data 957, mem_usage:GPU:0 0.9GB, num_seqs 4, 6.994 sec/step, elapsed 0:14:14, exp. remaining 0:27:07, complete 34.43%
att-weights epoch 560, step 405, max_size:classes 30, max_size:data 810, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.393 sec/step, elapsed 0:14:15, exp. remaining 0:27:04, complete 34.50%
att-weights epoch 560, step 406, max_size:classes 30, max_size:data 820, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.108 sec/step, elapsed 0:14:16, exp. remaining 0:26:58, complete 34.62%
att-weights epoch 560, step 407, max_size:classes 28, max_size:data 1001, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.176 sec/step, elapsed 0:14:18, exp. remaining 0:26:49, complete 34.77%
att-weights epoch 560, step 408, max_size:classes 27, max_size:data 923, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.219 sec/step, elapsed 0:14:19, exp. remaining 0:26:41, complete 34.92%
att-weights epoch 560, step 409, max_size:classes 30, max_size:data 1025, mem_usage:GPU:0 0.9GB, num_seqs 3, 0.930 sec/step, elapsed 0:14:20, exp. remaining 0:26:32, complete 35.08%
att-weights epoch 560, step 410, max_size:classes 29, max_size:data 923, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.545 sec/step, elapsed 0:14:21, exp. remaining 0:26:24, complete 35.23%
att-weights epoch 560, step 411, max_size:classes 30, max_size:data 900, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.482 sec/step, elapsed 0:14:23, exp. remaining 0:26:16, complete 35.38%
att-weights epoch 560, step 412, max_size:classes 26, max_size:data 919, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.354 sec/step, elapsed 0:14:24, exp. remaining 0:26:08, complete 35.53%
att-weights epoch 560, step 413, max_size:classes 28, max_size:data 781, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.776 sec/step, elapsed 0:14:25, exp. remaining 0:25:59, complete 35.69%
att-weights epoch 560, step 414, max_size:classes 25, max_size:data 1396, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.929 sec/step, elapsed 0:14:26, exp. remaining 0:25:50, complete 35.84%
att-weights epoch 560, step 415, max_size:classes 29, max_size:data 1010, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.009 sec/step, elapsed 0:14:27, exp. remaining 0:25:42, complete 35.99%
att-weights epoch 560, step 416, max_size:classes 28, max_size:data 851, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.306 sec/step, elapsed 0:14:28, exp. remaining 0:25:34, complete 36.15%
att-weights epoch 560, step 417, max_size:classes 25, max_size:data 841, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.432 sec/step, elapsed 0:14:29, exp. remaining 0:25:26, complete 36.30%
att-weights epoch 560, step 418, max_size:classes 29, max_size:data 862, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.334 sec/step, elapsed 0:14:31, exp. remaining 0:25:19, complete 36.45%
att-weights epoch 560, step 419, max_size:classes 28, max_size:data 963, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.260 sec/step, elapsed 0:14:32, exp. remaining 0:25:08, complete 36.64%
att-weights epoch 560, step 420, max_size:classes 27, max_size:data 785, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.061 sec/step, elapsed 0:14:33, exp. remaining 0:25:00, complete 36.79%
att-weights epoch 560, step 421, max_size:classes 27, max_size:data 832, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.472 sec/step, elapsed 0:14:35, exp. remaining 0:24:55, complete 36.91%
att-weights epoch 560, step 422, max_size:classes 26, max_size:data 863, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.343 sec/step, elapsed 0:14:36, exp. remaining 0:24:50, complete 37.02%
att-weights epoch 560, step 423, max_size:classes 30, max_size:data 893, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.127 sec/step, elapsed 0:14:38, exp. remaining 0:24:42, complete 37.21%
att-weights epoch 560, step 424, max_size:classes 27, max_size:data 824, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.171 sec/step, elapsed 0:14:39, exp. remaining 0:24:37, complete 37.33%
att-weights epoch 560, step 425, max_size:classes 25, max_size:data 870, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.289 sec/step, elapsed 0:14:41, exp. remaining 0:24:32, complete 37.44%
att-weights epoch 560, step 426, max_size:classes 31, max_size:data 887, mem_usage:GPU:0 0.9GB, num_seqs 4, 3.045 sec/step, elapsed 0:14:44, exp. remaining 0:24:25, complete 37.63%
att-weights epoch 560, step 427, max_size:classes 29, max_size:data 946, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.402 sec/step, elapsed 0:14:45, exp. remaining 0:24:17, complete 37.79%
att-weights epoch 560, step 428, max_size:classes 26, max_size:data 788, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.412 sec/step, elapsed 0:14:46, exp. remaining 0:24:08, complete 37.98%
att-weights epoch 560, step 429, max_size:classes 29, max_size:data 955, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.320 sec/step, elapsed 0:14:48, exp. remaining 0:24:01, complete 38.13%
att-weights epoch 560, step 430, max_size:classes 28, max_size:data 832, mem_usage:GPU:0 0.9GB, num_seqs 3, 0.897 sec/step, elapsed 0:14:49, exp. remaining 0:23:53, complete 38.28%
att-weights epoch 560, step 431, max_size:classes 29, max_size:data 1208, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.373 sec/step, elapsed 0:14:50, exp. remaining 0:23:46, complete 38.44%
att-weights epoch 560, step 432, max_size:classes 27, max_size:data 773, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.100 sec/step, elapsed 0:14:52, exp. remaining 0:23:40, complete 38.59%
att-weights epoch 560, step 433, max_size:classes 29, max_size:data 1077, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.158 sec/step, elapsed 0:14:53, exp. remaining 0:23:33, complete 38.74%
att-weights epoch 560, step 434, max_size:classes 27, max_size:data 1073, mem_usage:GPU:0 0.9GB, num_seqs 3, 8.564 sec/step, elapsed 0:15:02, exp. remaining 0:23:35, complete 38.93%
att-weights epoch 560, step 435, max_size:classes 30, max_size:data 761, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.906 sec/step, elapsed 0:15:04, exp. remaining 0:23:27, complete 39.12%
att-weights epoch 560, step 436, max_size:classes 26, max_size:data 958, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.166 sec/step, elapsed 0:15:05, exp. remaining 0:23:19, complete 39.27%
att-weights epoch 560, step 437, max_size:classes 26, max_size:data 747, mem_usage:GPU:0 0.9GB, num_seqs 5, 9.967 sec/step, elapsed 0:15:15, exp. remaining 0:23:26, complete 39.43%
att-weights epoch 560, step 438, max_size:classes 24, max_size:data 910, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.099 sec/step, elapsed 0:15:16, exp. remaining 0:23:19, complete 39.58%
att-weights epoch 560, step 439, max_size:classes 28, max_size:data 839, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.319 sec/step, elapsed 0:15:17, exp. remaining 0:23:09, complete 39.77%
att-weights epoch 560, step 440, max_size:classes 26, max_size:data 982, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.142 sec/step, elapsed 0:15:18, exp. remaining 0:23:02, complete 39.92%
att-weights epoch 560, step 441, max_size:classes 27, max_size:data 981, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.476 sec/step, elapsed 0:15:20, exp. remaining 0:22:56, complete 40.08%
att-weights epoch 560, step 442, max_size:classes 25, max_size:data 861, mem_usage:GPU:0 0.9GB, num_seqs 4, 3.204 sec/step, elapsed 0:15:23, exp. remaining 0:22:52, complete 40.23%
att-weights epoch 560, step 443, max_size:classes 29, max_size:data 792, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.440 sec/step, elapsed 0:15:25, exp. remaining 0:22:45, complete 40.38%
att-weights epoch 560, step 444, max_size:classes 26, max_size:data 791, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.330 sec/step, elapsed 0:15:26, exp. remaining 0:22:36, complete 40.57%
att-weights epoch 560, step 445, max_size:classes 25, max_size:data 720, mem_usage:GPU:0 0.9GB, num_seqs 4, 0.999 sec/step, elapsed 0:15:27, exp. remaining 0:22:27, complete 40.76%
att-weights epoch 560, step 446, max_size:classes 26, max_size:data 823, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.104 sec/step, elapsed 0:15:28, exp. remaining 0:22:20, complete 40.92%
att-weights epoch 560, step 447, max_size:classes 26, max_size:data 881, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.183 sec/step, elapsed 0:15:29, exp. remaining 0:22:14, complete 41.07%
att-weights epoch 560, step 448, max_size:classes 25, max_size:data 731, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.411 sec/step, elapsed 0:15:31, exp. remaining 0:22:09, complete 41.18%
att-weights epoch 560, step 449, max_size:classes 31, max_size:data 847, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.167 sec/step, elapsed 0:15:32, exp. remaining 0:22:03, complete 41.34%
att-weights epoch 560, step 450, max_size:classes 23, max_size:data 743, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.510 sec/step, elapsed 0:15:33, exp. remaining 0:21:56, complete 41.49%
att-weights epoch 560, step 451, max_size:classes 24, max_size:data 875, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.415 sec/step, elapsed 0:15:35, exp. remaining 0:21:50, complete 41.64%
att-weights epoch 560, step 452, max_size:classes 25, max_size:data 851, mem_usage:GPU:0 0.9GB, num_seqs 4, 7.710 sec/step, elapsed 0:15:42, exp. remaining 0:21:55, complete 41.76%
att-weights epoch 560, step 453, max_size:classes 28, max_size:data 771, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.602 sec/step, elapsed 0:15:44, exp. remaining 0:21:47, complete 41.95%
att-weights epoch 560, step 454, max_size:classes 23, max_size:data 699, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.214 sec/step, elapsed 0:15:45, exp. remaining 0:21:38, complete 42.14%
att-weights epoch 560, step 455, max_size:classes 26, max_size:data 875, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.326 sec/step, elapsed 0:15:47, exp. remaining 0:21:30, complete 42.33%
att-weights epoch 560, step 456, max_size:classes 27, max_size:data 804, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.299 sec/step, elapsed 0:15:48, exp. remaining 0:21:24, complete 42.48%
att-weights epoch 560, step 457, max_size:classes 27, max_size:data 1038, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.038 sec/step, elapsed 0:15:49, exp. remaining 0:21:15, complete 42.67%
att-weights epoch 560, step 458, max_size:classes 26, max_size:data 855, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.132 sec/step, elapsed 0:15:50, exp. remaining 0:21:05, complete 42.90%
att-weights epoch 560, step 459, max_size:classes 24, max_size:data 841, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.208 sec/step, elapsed 0:15:51, exp. remaining 0:20:56, complete 43.09%
att-weights epoch 560, step 460, max_size:classes 28, max_size:data 927, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.545 sec/step, elapsed 0:15:53, exp. remaining 0:20:49, complete 43.28%
att-weights epoch 560, step 461, max_size:classes 29, max_size:data 1133, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.268 sec/step, elapsed 0:15:54, exp. remaining 0:20:41, complete 43.47%
att-weights epoch 560, step 462, max_size:classes 25, max_size:data 752, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.194 sec/step, elapsed 0:15:55, exp. remaining 0:20:33, complete 43.66%
att-weights epoch 560, step 463, max_size:classes 24, max_size:data 701, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.501 sec/step, elapsed 0:15:57, exp. remaining 0:20:23, complete 43.89%
att-weights epoch 560, step 464, max_size:classes 23, max_size:data 737, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.402 sec/step, elapsed 0:15:58, exp. remaining 0:20:17, complete 44.05%
att-weights epoch 560, step 465, max_size:classes 23, max_size:data 880, mem_usage:GPU:0 0.9GB, num_seqs 4, 0.956 sec/step, elapsed 0:15:59, exp. remaining 0:20:09, complete 44.24%
att-weights epoch 560, step 466, max_size:classes 26, max_size:data 774, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.335 sec/step, elapsed 0:16:00, exp. remaining 0:20:01, complete 44.43%
att-weights epoch 560, step 467, max_size:classes 27, max_size:data 663, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.783 sec/step, elapsed 0:16:02, exp. remaining 0:19:56, complete 44.58%
att-weights epoch 560, step 468, max_size:classes 23, max_size:data 765, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.466 sec/step, elapsed 0:16:04, exp. remaining 0:19:49, complete 44.77%
att-weights epoch 560, step 469, max_size:classes 26, max_size:data 732, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.241 sec/step, elapsed 0:16:05, exp. remaining 0:19:39, complete 45.00%
att-weights epoch 560, step 470, max_size:classes 21, max_size:data 771, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.561 sec/step, elapsed 0:16:06, exp. remaining 0:19:32, complete 45.19%
att-weights epoch 560, step 471, max_size:classes 25, max_size:data 784, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.558 sec/step, elapsed 0:16:08, exp. remaining 0:19:25, complete 45.38%
att-weights epoch 560, step 472, max_size:classes 22, max_size:data 654, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.697 sec/step, elapsed 0:16:10, exp. remaining 0:19:16, complete 45.61%
att-weights epoch 560, step 473, max_size:classes 23, max_size:data 922, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.131 sec/step, elapsed 0:16:11, exp. remaining 0:19:09, complete 45.80%
att-weights epoch 560, step 474, max_size:classes 25, max_size:data 757, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.297 sec/step, elapsed 0:16:12, exp. remaining 0:19:02, complete 45.99%
att-weights epoch 560, step 475, max_size:classes 22, max_size:data 656, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.284 sec/step, elapsed 0:16:13, exp. remaining 0:18:56, complete 46.15%
att-weights epoch 560, step 476, max_size:classes 22, max_size:data 826, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.568 sec/step, elapsed 0:16:15, exp. remaining 0:18:51, complete 46.30%
att-weights epoch 560, step 477, max_size:classes 23, max_size:data 742, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.707 sec/step, elapsed 0:16:17, exp. remaining 0:18:43, complete 46.53%
att-weights epoch 560, step 478, max_size:classes 24, max_size:data 646, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.795 sec/step, elapsed 0:16:19, exp. remaining 0:18:34, complete 46.76%
att-weights epoch 560, step 479, max_size:classes 21, max_size:data 681, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.392 sec/step, elapsed 0:16:20, exp. remaining 0:18:27, complete 46.95%
att-weights epoch 560, step 480, max_size:classes 24, max_size:data 729, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.440 sec/step, elapsed 0:16:21, exp. remaining 0:18:19, complete 47.18%
att-weights epoch 560, step 481, max_size:classes 22, max_size:data 662, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.342 sec/step, elapsed 0:16:23, exp. remaining 0:18:14, complete 47.33%
att-weights epoch 560, step 482, max_size:classes 23, max_size:data 674, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.363 sec/step, elapsed 0:16:24, exp. remaining 0:18:09, complete 47.48%
att-weights epoch 560, step 483, max_size:classes 23, max_size:data 795, mem_usage:GPU:0 0.9GB, num_seqs 5, 3.129 sec/step, elapsed 0:16:27, exp. remaining 0:18:05, complete 47.63%
att-weights epoch 560, step 484, max_size:classes 23, max_size:data 878, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.202 sec/step, elapsed 0:16:28, exp. remaining 0:17:58, complete 47.82%
att-weights epoch 560, step 485, max_size:classes 23, max_size:data 810, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.100 sec/step, elapsed 0:16:30, exp. remaining 0:17:55, complete 47.94%
att-weights epoch 560, step 486, max_size:classes 23, max_size:data 652, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.952 sec/step, elapsed 0:16:32, exp. remaining 0:17:48, complete 48.17%
att-weights epoch 560, step 487, max_size:classes 22, max_size:data 664, mem_usage:GPU:0 0.9GB, num_seqs 6, 4.746 sec/step, elapsed 0:16:37, exp. remaining 0:17:45, complete 48.36%
att-weights epoch 560, step 488, max_size:classes 23, max_size:data 758, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.377 sec/step, elapsed 0:16:39, exp. remaining 0:17:40, complete 48.51%
att-weights epoch 560, step 489, max_size:classes 23, max_size:data 636, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.587 sec/step, elapsed 0:16:40, exp. remaining 0:17:35, complete 48.66%
att-weights epoch 560, step 490, max_size:classes 22, max_size:data 883, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.140 sec/step, elapsed 0:16:41, exp. remaining 0:17:28, complete 48.85%
att-weights epoch 560, step 491, max_size:classes 21, max_size:data 613, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.088 sec/step, elapsed 0:16:42, exp. remaining 0:17:21, complete 49.05%
att-weights epoch 560, step 492, max_size:classes 24, max_size:data 815, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.388 sec/step, elapsed 0:16:44, exp. remaining 0:17:17, complete 49.20%
att-weights epoch 560, step 493, max_size:classes 23, max_size:data 707, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.421 sec/step, elapsed 0:16:45, exp. remaining 0:17:10, complete 49.39%
att-weights epoch 560, step 494, max_size:classes 25, max_size:data 1083, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.171 sec/step, elapsed 0:16:46, exp. remaining 0:17:03, complete 49.58%
att-weights epoch 560, step 495, max_size:classes 24, max_size:data 629, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.949 sec/step, elapsed 0:16:48, exp. remaining 0:16:56, complete 49.81%
att-weights epoch 560, step 496, max_size:classes 22, max_size:data 662, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.542 sec/step, elapsed 0:16:50, exp. remaining 0:16:48, complete 50.04%
att-weights epoch 560, step 497, max_size:classes 24, max_size:data 747, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.377 sec/step, elapsed 0:16:51, exp. remaining 0:16:42, complete 50.23%
att-weights epoch 560, step 498, max_size:classes 23, max_size:data 871, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.142 sec/step, elapsed 0:16:52, exp. remaining 0:16:36, complete 50.42%
att-weights epoch 560, step 499, max_size:classes 21, max_size:data 680, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.211 sec/step, elapsed 0:16:54, exp. remaining 0:16:31, complete 50.57%
att-weights epoch 560, step 500, max_size:classes 19, max_size:data 756, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.531 sec/step, elapsed 0:16:55, exp. remaining 0:16:23, complete 50.80%
att-weights epoch 560, step 501, max_size:classes 22, max_size:data 852, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.352 sec/step, elapsed 0:16:56, exp. remaining 0:16:17, complete 50.99%
att-weights epoch 560, step 502, max_size:classes 22, max_size:data 700, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.378 sec/step, elapsed 0:16:58, exp. remaining 0:16:09, complete 51.22%
att-weights epoch 560, step 503, max_size:classes 22, max_size:data 761, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.308 sec/step, elapsed 0:16:59, exp. remaining 0:16:03, complete 51.41%
att-weights epoch 560, step 504, max_size:classes 22, max_size:data 637, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.617 sec/step, elapsed 0:17:01, exp. remaining 0:15:56, complete 51.64%
att-weights epoch 560, step 505, max_size:classes 23, max_size:data 636, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.664 sec/step, elapsed 0:17:02, exp. remaining 0:15:49, complete 51.87%
att-weights epoch 560, step 506, max_size:classes 23, max_size:data 691, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.327 sec/step, elapsed 0:17:04, exp. remaining 0:15:43, complete 52.06%
att-weights epoch 560, step 507, max_size:classes 21, max_size:data 777, mem_usage:GPU:0 0.9GB, num_seqs 5, 5.619 sec/step, elapsed 0:17:09, exp. remaining 0:15:41, complete 52.25%
att-weights epoch 560, step 508, max_size:classes 20, max_size:data 821, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.319 sec/step, elapsed 0:17:11, exp. remaining 0:15:35, complete 52.44%
att-weights epoch 560, step 509, max_size:classes 19, max_size:data 632, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.316 sec/step, elapsed 0:17:12, exp. remaining 0:15:29, complete 52.63%
att-weights epoch 560, step 510, max_size:classes 21, max_size:data 760, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.626 sec/step, elapsed 0:17:14, exp. remaining 0:15:22, complete 52.86%
att-weights epoch 560, step 511, max_size:classes 20, max_size:data 657, mem_usage:GPU:0 0.9GB, num_seqs 6, 5.869 sec/step, elapsed 0:17:20, exp. remaining 0:15:18, complete 53.09%
att-weights epoch 560, step 512, max_size:classes 21, max_size:data 754, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.559 sec/step, elapsed 0:17:21, exp. remaining 0:15:13, complete 53.28%
att-weights epoch 560, step 513, max_size:classes 21, max_size:data 665, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.642 sec/step, elapsed 0:17:23, exp. remaining 0:15:06, complete 53.51%
att-weights epoch 560, step 514, max_size:classes 22, max_size:data 554, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.474 sec/step, elapsed 0:17:24, exp. remaining 0:14:57, complete 53.78%
att-weights epoch 560, step 515, max_size:classes 22, max_size:data 780, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.694 sec/step, elapsed 0:17:26, exp. remaining 0:14:53, complete 53.93%
att-weights epoch 560, step 516, max_size:classes 21, max_size:data 702, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.383 sec/step, elapsed 0:17:27, exp. remaining 0:14:46, complete 54.16%
att-weights epoch 560, step 517, max_size:classes 20, max_size:data 676, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.439 sec/step, elapsed 0:17:29, exp. remaining 0:14:41, complete 54.35%
att-weights epoch 560, step 518, max_size:classes 22, max_size:data 757, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.756 sec/step, elapsed 0:17:30, exp. remaining 0:14:34, complete 54.58%
att-weights epoch 560, step 519, max_size:classes 21, max_size:data 580, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.789 sec/step, elapsed 0:17:32, exp. remaining 0:14:28, complete 54.81%
att-weights epoch 560, step 520, max_size:classes 24, max_size:data 647, mem_usage:GPU:0 0.9GB, num_seqs 6, 8.761 sec/step, elapsed 0:17:41, exp. remaining 0:14:28, complete 55.00%
att-weights epoch 560, step 521, max_size:classes 20, max_size:data 749, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.659 sec/step, elapsed 0:17:43, exp. remaining 0:14:21, complete 55.23%
att-weights epoch 560, step 522, max_size:classes 21, max_size:data 582, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.472 sec/step, elapsed 0:17:44, exp. remaining 0:14:15, complete 55.46%
att-weights epoch 560, step 523, max_size:classes 20, max_size:data 526, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.702 sec/step, elapsed 0:17:46, exp. remaining 0:14:08, complete 55.69%
att-weights epoch 560, step 524, max_size:classes 22, max_size:data 951, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.233 sec/step, elapsed 0:17:47, exp. remaining 0:14:03, complete 55.88%
att-weights epoch 560, step 525, max_size:classes 19, max_size:data 662, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.634 sec/step, elapsed 0:17:49, exp. remaining 0:13:56, complete 56.11%
att-weights epoch 560, step 526, max_size:classes 19, max_size:data 683, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.326 sec/step, elapsed 0:17:50, exp. remaining 0:13:49, complete 56.34%
att-weights epoch 560, step 527, max_size:classes 20, max_size:data 587, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.920 sec/step, elapsed 0:17:52, exp. remaining 0:13:44, complete 56.53%
att-weights epoch 560, step 528, max_size:classes 20, max_size:data 607, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.721 sec/step, elapsed 0:17:54, exp. remaining 0:13:39, complete 56.72%
att-weights epoch 560, step 529, max_size:classes 19, max_size:data 713, mem_usage:GPU:0 0.9GB, num_seqs 5, 3.582 sec/step, elapsed 0:17:57, exp. remaining 0:13:36, complete 56.91%
att-weights epoch 560, step 530, max_size:classes 20, max_size:data 637, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.766 sec/step, elapsed 0:17:59, exp. remaining 0:13:28, complete 57.18%
att-weights epoch 560, step 531, max_size:classes 19, max_size:data 646, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.677 sec/step, elapsed 0:18:01, exp. remaining 0:13:21, complete 57.44%
att-weights epoch 560, step 532, max_size:classes 19, max_size:data 590, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.433 sec/step, elapsed 0:18:03, exp. remaining 0:13:16, complete 57.63%
att-weights epoch 560, step 533, max_size:classes 20, max_size:data 723, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.611 sec/step, elapsed 0:18:05, exp. remaining 0:13:11, complete 57.82%
att-weights epoch 560, step 534, max_size:classes 20, max_size:data 606, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.682 sec/step, elapsed 0:18:07, exp. remaining 0:13:05, complete 58.05%
att-weights epoch 560, step 535, max_size:classes 21, max_size:data 643, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.432 sec/step, elapsed 0:18:09, exp. remaining 0:13:01, complete 58.24%
att-weights epoch 560, step 536, max_size:classes 19, max_size:data 672, mem_usage:GPU:0 0.9GB, num_seqs 5, 11.266 sec/step, elapsed 0:18:20, exp. remaining 0:13:00, complete 58.51%
att-weights epoch 560, step 537, max_size:classes 20, max_size:data 695, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.597 sec/step, elapsed 0:18:22, exp. remaining 0:12:55, complete 58.70%
att-weights epoch 560, step 538, max_size:classes 22, max_size:data 710, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.793 sec/step, elapsed 0:18:24, exp. remaining 0:12:50, complete 58.89%
att-weights epoch 560, step 539, max_size:classes 18, max_size:data 512, mem_usage:GPU:0 0.9GB, num_seqs 7, 5.446 sec/step, elapsed 0:18:29, exp. remaining 0:12:45, complete 59.16%
att-weights epoch 560, step 540, max_size:classes 18, max_size:data 559, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.772 sec/step, elapsed 0:18:31, exp. remaining 0:12:39, complete 59.39%
att-weights epoch 560, step 541, max_size:classes 18, max_size:data 533, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.620 sec/step, elapsed 0:18:32, exp. remaining 0:12:33, complete 59.62%
att-weights epoch 560, step 542, max_size:classes 18, max_size:data 727, mem_usage:GPU:0 0.9GB, num_seqs 5, 8.460 sec/step, elapsed 0:18:41, exp. remaining 0:12:33, complete 59.81%
att-weights epoch 560, step 543, max_size:classes 18, max_size:data 592, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.375 sec/step, elapsed 0:18:42, exp. remaining 0:12:26, complete 60.08%
att-weights epoch 560, step 544, max_size:classes 17, max_size:data 682, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.292 sec/step, elapsed 0:18:44, exp. remaining 0:12:19, complete 60.31%
att-weights epoch 560, step 545, max_size:classes 22, max_size:data 549, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.754 sec/step, elapsed 0:18:45, exp. remaining 0:12:15, complete 60.50%
att-weights epoch 560, step 546, max_size:classes 18, max_size:data 674, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.269 sec/step, elapsed 0:18:47, exp. remaining 0:12:07, complete 60.76%
att-weights epoch 560, step 547, max_size:classes 19, max_size:data 682, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.179 sec/step, elapsed 0:18:49, exp. remaining 0:12:02, complete 60.99%
att-weights epoch 560, step 548, max_size:classes 18, max_size:data 537, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.646 sec/step, elapsed 0:18:50, exp. remaining 0:11:56, complete 61.22%
att-weights epoch 560, step 549, max_size:classes 20, max_size:data 664, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.591 sec/step, elapsed 0:18:52, exp. remaining 0:11:51, complete 61.41%
att-weights epoch 560, step 550, max_size:classes 18, max_size:data 649, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.554 sec/step, elapsed 0:18:54, exp. remaining 0:11:45, complete 61.64%
att-weights epoch 560, step 551, max_size:classes 18, max_size:data 676, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.173 sec/step, elapsed 0:18:55, exp. remaining 0:11:39, complete 61.87%
att-weights epoch 560, step 552, max_size:classes 19, max_size:data 571, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.485 sec/step, elapsed 0:18:56, exp. remaining 0:11:33, complete 62.10%
att-weights epoch 560, step 553, max_size:classes 18, max_size:data 646, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.494 sec/step, elapsed 0:18:58, exp. remaining 0:11:26, complete 62.37%
att-weights epoch 560, step 554, max_size:classes 18, max_size:data 705, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.279 sec/step, elapsed 0:18:59, exp. remaining 0:11:20, complete 62.60%
att-weights epoch 560, step 555, max_size:classes 17, max_size:data 541, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.277 sec/step, elapsed 0:19:01, exp. remaining 0:11:14, complete 62.86%
att-weights epoch 560, step 556, max_size:classes 17, max_size:data 600, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.591 sec/step, elapsed 0:19:03, exp. remaining 0:11:07, complete 63.13%
att-weights epoch 560, step 557, max_size:classes 18, max_size:data 581, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.071 sec/step, elapsed 0:19:04, exp. remaining 0:11:00, complete 63.40%
att-weights epoch 560, step 558, max_size:classes 18, max_size:data 762, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.420 sec/step, elapsed 0:19:05, exp. remaining 0:10:54, complete 63.66%
att-weights epoch 560, step 559, max_size:classes 18, max_size:data 621, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.469 sec/step, elapsed 0:19:07, exp. remaining 0:10:49, complete 63.85%
att-weights epoch 560, step 560, max_size:classes 17, max_size:data 603, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.598 sec/step, elapsed 0:19:08, exp. remaining 0:10:42, complete 64.12%
att-weights epoch 560, step 561, max_size:classes 15, max_size:data 619, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.498 sec/step, elapsed 0:19:10, exp. remaining 0:10:37, complete 64.35%
att-weights epoch 560, step 562, max_size:classes 17, max_size:data 498, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.389 sec/step, elapsed 0:19:11, exp. remaining 0:10:30, complete 64.62%
att-weights epoch 560, step 563, max_size:classes 16, max_size:data 592, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.359 sec/step, elapsed 0:19:13, exp. remaining 0:10:24, complete 64.89%
att-weights epoch 560, step 564, max_size:classes 20, max_size:data 552, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.760 sec/step, elapsed 0:19:14, exp. remaining 0:10:18, complete 65.11%
att-weights epoch 560, step 565, max_size:classes 17, max_size:data 513, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.598 sec/step, elapsed 0:19:16, exp. remaining 0:10:12, complete 65.38%
att-weights epoch 560, step 566, max_size:classes 18, max_size:data 526, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.147 sec/step, elapsed 0:19:18, exp. remaining 0:10:05, complete 65.69%
att-weights epoch 560, step 567, max_size:classes 18, max_size:data 529, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.444 sec/step, elapsed 0:19:21, exp. remaining 0:09:59, complete 65.95%
att-weights epoch 560, step 568, max_size:classes 20, max_size:data 704, mem_usage:GPU:0 0.9GB, num_seqs 5, 5.442 sec/step, elapsed 0:19:26, exp. remaining 0:09:54, complete 66.26%
att-weights epoch 560, step 569, max_size:classes 22, max_size:data 564, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.774 sec/step, elapsed 0:19:28, exp. remaining 0:09:48, complete 66.49%
att-weights epoch 560, step 570, max_size:classes 17, max_size:data 621, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.616 sec/step, elapsed 0:19:30, exp. remaining 0:09:42, complete 66.76%
att-weights epoch 560, step 571, max_size:classes 17, max_size:data 563, mem_usage:GPU:0 0.9GB, num_seqs 7, 8.840 sec/step, elapsed 0:19:38, exp. remaining 0:09:39, complete 67.06%
att-weights epoch 560, step 572, max_size:classes 17, max_size:data 513, mem_usage:GPU:0 0.9GB, num_seqs 7, 7.051 sec/step, elapsed 0:19:45, exp. remaining 0:09:35, complete 67.33%
att-weights epoch 560, step 573, max_size:classes 18, max_size:data 629, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.752 sec/step, elapsed 0:19:47, exp. remaining 0:09:29, complete 67.60%
att-weights epoch 560, step 574, max_size:classes 17, max_size:data 561, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.881 sec/step, elapsed 0:19:49, exp. remaining 0:09:23, complete 67.86%
att-weights epoch 560, step 575, max_size:classes 18, max_size:data 500, mem_usage:GPU:0 0.9GB, num_seqs 8, 9.060 sec/step, elapsed 0:19:58, exp. remaining 0:09:21, complete 68.09%
att-weights epoch 560, step 576, max_size:classes 17, max_size:data 539, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.224 sec/step, elapsed 0:20:00, exp. remaining 0:09:16, complete 68.32%
att-weights epoch 560, step 577, max_size:classes 15, max_size:data 459, mem_usage:GPU:0 0.9GB, num_seqs 8, 1.702 sec/step, elapsed 0:20:02, exp. remaining 0:09:13, complete 68.47%
att-weights epoch 560, step 578, max_size:classes 17, max_size:data 590, mem_usage:GPU:0 0.9GB, num_seqs 6, 3.215 sec/step, elapsed 0:20:05, exp. remaining 0:09:07, complete 68.78%
att-weights epoch 560, step 579, max_size:classes 15, max_size:data 545, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.041 sec/step, elapsed 0:20:07, exp. remaining 0:09:01, complete 69.05%
att-weights epoch 560, step 580, max_size:classes 14, max_size:data 447, mem_usage:GPU:0 0.9GB, num_seqs 8, 23.662 sec/step, elapsed 0:20:31, exp. remaining 0:09:06, complete 69.27%
att-weights epoch 560, step 581, max_size:classes 17, max_size:data 513, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.677 sec/step, elapsed 0:20:33, exp. remaining 0:09:00, complete 69.54%
att-weights epoch 560, step 582, max_size:classes 16, max_size:data 563, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.537 sec/step, elapsed 0:20:34, exp. remaining 0:08:54, complete 69.77%
att-weights epoch 560, step 583, max_size:classes 16, max_size:data 514, mem_usage:GPU:0 0.9GB, num_seqs 7, 43.748 sec/step, elapsed 0:21:18, exp. remaining 0:09:07, complete 70.00%
att-weights epoch 560, step 584, max_size:classes 16, max_size:data 578, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.859 sec/step, elapsed 0:21:20, exp. remaining 0:09:02, complete 70.23%
att-weights epoch 560, step 585, max_size:classes 14, max_size:data 537, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.105 sec/step, elapsed 0:21:22, exp. remaining 0:08:55, complete 70.53%
att-weights epoch 560, step 586, max_size:classes 14, max_size:data 874, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.450 sec/step, elapsed 0:21:23, exp. remaining 0:08:50, complete 70.76%
att-weights epoch 560, step 587, max_size:classes 17, max_size:data 452, mem_usage:GPU:0 0.9GB, num_seqs 8, 27.648 sec/step, elapsed 0:21:51, exp. remaining 0:08:54, complete 71.03%
att-weights epoch 560, step 588, max_size:classes 18, max_size:data 413, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.926 sec/step, elapsed 0:21:53, exp. remaining 0:08:47, complete 71.34%
att-weights epoch 560, step 589, max_size:classes 16, max_size:data 616, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.561 sec/step, elapsed 0:21:54, exp. remaining 0:08:42, complete 71.56%
att-weights epoch 560, step 590, max_size:classes 17, max_size:data 470, mem_usage:GPU:0 0.9GB, num_seqs 7, 7.737 sec/step, elapsed 0:22:02, exp. remaining 0:08:39, complete 71.79%
att-weights epoch 560, step 591, max_size:classes 16, max_size:data 628, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.421 sec/step, elapsed 0:22:04, exp. remaining 0:08:33, complete 72.06%
att-weights epoch 560, step 592, max_size:classes 15, max_size:data 506, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.458 sec/step, elapsed 0:22:05, exp. remaining 0:08:27, complete 72.33%
att-weights epoch 560, step 593, max_size:classes 16, max_size:data 611, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.481 sec/step, elapsed 0:22:07, exp. remaining 0:08:21, complete 72.56%
att-weights epoch 560, step 594, max_size:classes 16, max_size:data 454, mem_usage:GPU:0 0.9GB, num_seqs 8, 1.451 sec/step, elapsed 0:22:08, exp. remaining 0:08:14, complete 72.86%
att-weights epoch 560, step 595, max_size:classes 16, max_size:data 591, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.336 sec/step, elapsed 0:22:09, exp. remaining 0:08:08, complete 73.13%
att-weights epoch 560, step 596, max_size:classes 15, max_size:data 523, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.506 sec/step, elapsed 0:22:11, exp. remaining 0:08:02, complete 73.40%
att-weights epoch 560, step 597, max_size:classes 14, max_size:data 459, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.030 sec/step, elapsed 0:22:13, exp. remaining 0:07:56, complete 73.66%
att-weights epoch 560, step 598, max_size:classes 16, max_size:data 656, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.350 sec/step, elapsed 0:22:14, exp. remaining 0:07:51, complete 73.89%
att-weights epoch 560, step 599, max_size:classes 15, max_size:data 578, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.475 sec/step, elapsed 0:22:16, exp. remaining 0:07:43, complete 74.24%
att-weights epoch 560, step 600, max_size:classes 15, max_size:data 473, mem_usage:GPU:0 0.9GB, num_seqs 7, 2.311 sec/step, elapsed 0:22:18, exp. remaining 0:07:38, complete 74.47%
att-weights epoch 560, step 601, max_size:classes 15, max_size:data 536, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.638 sec/step, elapsed 0:22:20, exp. remaining 0:07:33, complete 74.73%
att-weights epoch 560, step 602, max_size:classes 16, max_size:data 635, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.087 sec/step, elapsed 0:22:22, exp. remaining 0:07:26, complete 75.04%
att-weights epoch 560, step 603, max_size:classes 15, max_size:data 494, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.288 sec/step, elapsed 0:22:24, exp. remaining 0:07:21, complete 75.27%
att-weights epoch 560, step 604, max_size:classes 14, max_size:data 545, mem_usage:GPU:0 0.9GB, num_seqs 7, 6.510 sec/step, elapsed 0:22:31, exp. remaining 0:07:16, complete 75.57%
att-weights epoch 560, step 605, max_size:classes 13, max_size:data 449, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.201 sec/step, elapsed 0:22:32, exp. remaining 0:07:09, complete 75.88%
att-weights epoch 560, step 606, max_size:classes 16, max_size:data 528, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.536 sec/step, elapsed 0:22:33, exp. remaining 0:07:03, complete 76.18%
att-weights epoch 560, step 607, max_size:classes 14, max_size:data 624, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.421 sec/step, elapsed 0:22:35, exp. remaining 0:06:54, complete 76.56%
att-weights epoch 560, step 608, max_size:classes 14, max_size:data 419, mem_usage:GPU:0 0.9GB, num_seqs 9, 1.586 sec/step, elapsed 0:22:36, exp. remaining 0:06:49, complete 76.83%
att-weights epoch 560, step 609, max_size:classes 14, max_size:data 572, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.367 sec/step, elapsed 0:22:38, exp. remaining 0:06:41, complete 77.18%
att-weights epoch 560, step 610, max_size:classes 16, max_size:data 549, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.426 sec/step, elapsed 0:22:39, exp. remaining 0:06:35, complete 77.48%
att-weights epoch 560, step 611, max_size:classes 16, max_size:data 439, mem_usage:GPU:0 0.9GB, num_seqs 8, 1.533 sec/step, elapsed 0:22:41, exp. remaining 0:06:27, complete 77.82%
att-weights epoch 560, step 612, max_size:classes 15, max_size:data 593, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.330 sec/step, elapsed 0:22:42, exp. remaining 0:06:20, complete 78.17%
att-weights epoch 560, step 613, max_size:classes 14, max_size:data 466, mem_usage:GPU:0 0.9GB, num_seqs 8, 3.046 sec/step, elapsed 0:22:45, exp. remaining 0:06:13, complete 78.51%
att-weights epoch 560, step 614, max_size:classes 15, max_size:data 478, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.684 sec/step, elapsed 0:22:48, exp. remaining 0:06:08, complete 78.78%
att-weights epoch 560, step 615, max_size:classes 13, max_size:data 477, mem_usage:GPU:0 0.9GB, num_seqs 8, 5.168 sec/step, elapsed 0:22:53, exp. remaining 0:06:04, complete 79.05%
att-weights epoch 560, step 616, max_size:classes 13, max_size:data 391, mem_usage:GPU:0 0.9GB, num_seqs 10, 2.003 sec/step, elapsed 0:22:55, exp. remaining 0:05:57, complete 79.39%
att-weights epoch 560, step 617, max_size:classes 15, max_size:data 513, mem_usage:GPU:0 0.9GB, num_seqs 7, 16.181 sec/step, elapsed 0:23:11, exp. remaining 0:05:54, complete 79.69%
att-weights epoch 560, step 618, max_size:classes 13, max_size:data 422, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.073 sec/step, elapsed 0:23:13, exp. remaining 0:05:49, complete 79.96%
att-weights epoch 560, step 619, max_size:classes 13, max_size:data 452, mem_usage:GPU:0 0.9GB, num_seqs 8, 2.082 sec/step, elapsed 0:23:15, exp. remaining 0:05:43, complete 80.23%
att-weights epoch 560, step 620, max_size:classes 12, max_size:data 429, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.351 sec/step, elapsed 0:23:18, exp. remaining 0:05:37, complete 80.53%
att-weights epoch 560, step 621, max_size:classes 14, max_size:data 428, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.328 sec/step, elapsed 0:23:20, exp. remaining 0:05:31, complete 80.84%
att-weights epoch 560, step 622, max_size:classes 12, max_size:data 412, mem_usage:GPU:0 0.9GB, num_seqs 9, 1.620 sec/step, elapsed 0:23:22, exp. remaining 0:05:24, complete 81.18%
att-weights epoch 560, step 623, max_size:classes 12, max_size:data 403, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.463 sec/step, elapsed 0:23:23, exp. remaining 0:05:18, complete 81.49%
att-weights epoch 560, step 624, max_size:classes 12, max_size:data 508, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.361 sec/step, elapsed 0:23:24, exp. remaining 0:05:12, complete 81.79%
att-weights epoch 560, step 625, max_size:classes 12, max_size:data 415, mem_usage:GPU:0 0.9GB, num_seqs 9, 1.656 sec/step, elapsed 0:23:26, exp. remaining 0:05:07, complete 82.06%
att-weights epoch 560, step 626, max_size:classes 12, max_size:data 489, mem_usage:GPU:0 0.9GB, num_seqs 8, 1.674 sec/step, elapsed 0:23:28, exp. remaining 0:05:00, complete 82.40%
att-weights epoch 560, step 627, max_size:classes 13, max_size:data 428, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.492 sec/step, elapsed 0:23:29, exp. remaining 0:04:54, complete 82.71%
att-weights epoch 560, step 628, max_size:classes 13, max_size:data 515, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.174 sec/step, elapsed 0:23:30, exp. remaining 0:04:47, complete 83.09%
att-weights epoch 560, step 629, max_size:classes 12, max_size:data 472, mem_usage:GPU:0 0.9GB, num_seqs 8, 1.533 sec/step, elapsed 0:23:32, exp. remaining 0:04:40, complete 83.44%
att-weights epoch 560, step 630, max_size:classes 12, max_size:data 469, mem_usage:GPU:0 0.9GB, num_seqs 8, 1.776 sec/step, elapsed 0:23:34, exp. remaining 0:04:34, complete 83.74%
att-weights epoch 560, step 631, max_size:classes 12, max_size:data 359, mem_usage:GPU:0 0.9GB, num_seqs 9, 1.769 sec/step, elapsed 0:23:35, exp. remaining 0:04:26, complete 84.16%
att-weights epoch 560, step 632, max_size:classes 13, max_size:data 475, mem_usage:GPU:0 0.9GB, num_seqs 8, 1.541 sec/step, elapsed 0:23:37, exp. remaining 0:04:20, complete 84.47%
att-weights epoch 560, step 633, max_size:classes 11, max_size:data 360, mem_usage:GPU:0 0.9GB, num_seqs 8, 1.376 sec/step, elapsed 0:23:38, exp. remaining 0:04:14, complete 84.81%
att-weights epoch 560, step 634, max_size:classes 11, max_size:data 544, mem_usage:GPU:0 0.9GB, num_seqs 7, 1.275 sec/step, elapsed 0:23:40, exp. remaining 0:04:07, complete 85.15%
att-weights epoch 560, step 635, max_size:classes 12, max_size:data 426, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.780 sec/step, elapsed 0:23:42, exp. remaining 0:04:03, complete 85.38%
att-weights epoch 560, step 636, max_size:classes 15, max_size:data 471, mem_usage:GPU:0 0.9GB, num_seqs 8, 1.782 sec/step, elapsed 0:23:44, exp. remaining 0:03:57, complete 85.69%
att-weights epoch 560, step 637, max_size:classes 12, max_size:data 379, mem_usage:GPU:0 0.9GB, num_seqs 10, 3.493 sec/step, elapsed 0:23:48, exp. remaining 0:03:52, complete 85.99%
att-weights epoch 560, step 638, max_size:classes 11, max_size:data 330, mem_usage:GPU:0 0.9GB, num_seqs 9, 1.907 sec/step, elapsed 0:23:50, exp. remaining 0:03:45, complete 86.37%
att-weights epoch 560, step 639, max_size:classes 11, max_size:data 490, mem_usage:GPU:0 0.9GB, num_seqs 8, 1.747 sec/step, elapsed 0:23:51, exp. remaining 0:03:41, complete 86.60%
att-weights epoch 560, step 640, max_size:classes 12, max_size:data 339, mem_usage:GPU:0 0.9GB, num_seqs 11, 1.878 sec/step, elapsed 0:23:53, exp. remaining 0:03:33, complete 87.06%
att-weights epoch 560, step 641, max_size:classes 12, max_size:data 450, mem_usage:GPU:0 0.9GB, num_seqs 8, 1.749 sec/step, elapsed 0:23:55, exp. remaining 0:03:26, complete 87.44%
att-weights epoch 560, step 642, max_size:classes 13, max_size:data 374, mem_usage:GPU:0 0.9GB, num_seqs 9, 1.493 sec/step, elapsed 0:23:56, exp. remaining 0:03:20, complete 87.75%
att-weights epoch 560, step 643, max_size:classes 11, max_size:data 429, mem_usage:GPU:0 0.9GB, num_seqs 9, 1.625 sec/step, elapsed 0:23:58, exp. remaining 0:03:14, complete 88.09%
att-weights epoch 560, step 644, max_size:classes 11, max_size:data 612, mem_usage:GPU:0 0.9GB, num_seqs 6, 2.421 sec/step, elapsed 0:24:01, exp. remaining 0:03:08, complete 88.44%
att-weights epoch 560, step 645, max_size:classes 12, max_size:data 500, mem_usage:GPU:0 0.9GB, num_seqs 8, 3.396 sec/step, elapsed 0:24:04, exp. remaining 0:03:02, complete 88.78%
att-weights epoch 560, step 646, max_size:classes 12, max_size:data 467, mem_usage:GPU:0 0.9GB, num_seqs 8, 1.949 sec/step, elapsed 0:24:06, exp. remaining 0:02:55, complete 89.16%
att-weights epoch 560, step 647, max_size:classes 10, max_size:data 382, mem_usage:GPU:0 0.9GB, num_seqs 10, 2.525 sec/step, elapsed 0:24:08, exp. remaining 0:02:49, complete 89.50%
att-weights epoch 560, step 648, max_size:classes 9, max_size:data 657, mem_usage:GPU:0 0.9GB, num_seqs 6, 1.234 sec/step, elapsed 0:24:10, exp. remaining 0:02:42, complete 89.92%
att-weights epoch 560, step 649, max_size:classes 11, max_size:data 332, mem_usage:GPU:0 0.9GB, num_seqs 12, 12.618 sec/step, elapsed 0:24:22, exp. remaining 0:02:38, complete 90.23%
att-weights epoch 560, step 650, max_size:classes 12, max_size:data 390, mem_usage:GPU:0 0.9GB, num_seqs 10, 2.030 sec/step, elapsed 0:24:24, exp. remaining 0:02:33, complete 90.53%
att-weights epoch 560, step 651, max_size:classes 12, max_size:data 474, mem_usage:GPU:0 0.9GB, num_seqs 8, 12.047 sec/step, elapsed 0:24:36, exp. remaining 0:02:27, complete 90.92%
att-weights epoch 560, step 652, max_size:classes 9, max_size:data 403, mem_usage:GPU:0 0.9GB, num_seqs 9, 1.744 sec/step, elapsed 0:24:38, exp. remaining 0:02:21, complete 91.26%
att-weights epoch 560, step 653, max_size:classes 10, max_size:data 402, mem_usage:GPU:0 0.9GB, num_seqs 9, 2.035 sec/step, elapsed 0:24:40, exp. remaining 0:02:15, complete 91.60%
att-weights epoch 560, step 654, max_size:classes 10, max_size:data 410, mem_usage:GPU:0 0.9GB, num_seqs 9, 4.212 sec/step, elapsed 0:24:44, exp. remaining 0:02:09, complete 91.98%
att-weights epoch 560, step 655, max_size:classes 11, max_size:data 293, mem_usage:GPU:0 0.9GB, num_seqs 10, 1.586 sec/step, elapsed 0:24:46, exp. remaining 0:02:02, complete 92.40%
att-weights epoch 560, step 656, max_size:classes 10, max_size:data 426, mem_usage:GPU:0 0.9GB, num_seqs 9, 1.824 sec/step, elapsed 0:24:48, exp. remaining 0:01:54, complete 92.86%
att-weights epoch 560, step 657, max_size:classes 10, max_size:data 312, mem_usage:GPU:0 0.9GB, num_seqs 11, 2.453 sec/step, elapsed 0:24:50, exp. remaining 0:01:48, complete 93.21%
att-weights epoch 560, step 658, max_size:classes 9, max_size:data 384, mem_usage:GPU:0 0.9GB, num_seqs 8, 1.626 sec/step, elapsed 0:24:52, exp. remaining 0:01:42, complete 93.59%
att-weights epoch 560, step 659, max_size:classes 11, max_size:data 448, mem_usage:GPU:0 0.9GB, num_seqs 8, 1.566 sec/step, elapsed 0:24:53, exp. remaining 0:01:34, complete 94.05%
att-weights epoch 560, step 660, max_size:classes 11, max_size:data 389, mem_usage:GPU:0 0.9GB, num_seqs 10, 1.801 sec/step, elapsed 0:24:55, exp. remaining 0:01:27, complete 94.47%
att-weights epoch 560, step 661, max_size:classes 10, max_size:data 404, mem_usage:GPU:0 0.9GB, num_seqs 9, 1.556 sec/step, elapsed 0:24:57, exp. remaining 0:01:21, complete 94.85%
att-weights epoch 560, step 662, max_size:classes 10, max_size:data 432, mem_usage:GPU:0 0.9GB, num_seqs 9, 1.818 sec/step, elapsed 0:24:59, exp. remaining 0:01:15, complete 95.19%
att-weights epoch 560, step 663, max_size:classes 10, max_size:data 399, mem_usage:GPU:0 0.9GB, num_seqs 10, 1.992 sec/step, elapsed 0:25:01, exp. remaining 0:01:08, complete 95.65%
att-weights epoch 560, step 664, max_size:classes 10, max_size:data 353, mem_usage:GPU:0 0.9GB, num_seqs 11, 1.963 sec/step, elapsed 0:25:02, exp. remaining 0:01:01, complete 96.07%
att-weights epoch 560, step 665, max_size:classes 11, max_size:data 328, mem_usage:GPU:0 0.9GB, num_seqs 12, 2.255 sec/step, elapsed 0:25:05, exp. remaining 0:00:56, complete 96.41%
att-weights epoch 560, step 666, max_size:classes 8, max_size:data 412, mem_usage:GPU:0 0.9GB, num_seqs 9, 4.963 sec/step, elapsed 0:25:10, exp. remaining 0:00:48, complete 96.87%
att-weights epoch 560, step 667, max_size:classes 9, max_size:data 384, mem_usage:GPU:0 0.9GB, num_seqs 10, 1.867 sec/step, elapsed 0:25:12, exp. remaining 0:00:40, complete 97.37%
att-weights epoch 560, step 668, max_size:classes 9, max_size:data 309, mem_usage:GPU:0 0.9GB, num_seqs 12, 1.944 sec/step, elapsed 0:25:14, exp. remaining 0:00:34, complete 97.79%
att-weights epoch 560, step 669, max_size:classes 9, max_size:data 280, mem_usage:GPU:0 0.9GB, num_seqs 11, 1.650 sec/step, elapsed 0:25:15, exp. remaining 0:00:27, complete 98.21%
att-weights epoch 560, step 670, max_size:classes 10, max_size:data 388, mem_usage:GPU:0 0.9GB, num_seqs 10, 1.882 sec/step, elapsed 0:25:17, exp. remaining 0:00:19, complete 98.70%
att-weights epoch 560, step 671, max_size:classes 10, max_size:data 442, mem_usage:GPU:0 0.9GB, num_seqs 9, 1.653 sec/step, elapsed 0:25:19, exp. remaining 0:00:11, complete 99.24%
att-weights epoch 560, step 672, max_size:classes 9, max_size:data 327, mem_usage:GPU:0 0.9GB, num_seqs 12, 4.664 sec/step, elapsed 0:25:23, exp. remaining 0:00:04, complete 99.69%
att-weights epoch 560, step 673, max_size:classes 8, max_size:data 336, mem_usage:GPU:0 0.9GB, num_seqs 11, 1.700 sec/step, elapsed 0:25:25, exp. remaining 0:00:04, complete 99.69%
att-weights epoch 560, step 674, max_size:classes 8, max_size:data 405, mem_usage:GPU:0 0.9GB, num_seqs 9, 1.337 sec/step, elapsed 0:25:26, exp. remaining 0:00:04, complete 99.69%
att-weights epoch 560, step 675, max_size:classes 7, max_size:data 332, mem_usage:GPU:0 0.9GB, num_seqs 12, 5.000 sec/step, elapsed 0:25:31, exp. remaining 0:00:04, complete 99.69%
att-weights epoch 560, step 676, max_size:classes 9, max_size:data 301, mem_usage:GPU:0 0.9GB, num_seqs 13, 1.382 sec/step, elapsed 0:25:33, exp. remaining 0:00:04, complete 99.69%
att-weights epoch 560, step 677, max_size:classes 8, max_size:data 304, mem_usage:GPU:0 0.9GB, num_seqs 11, 0.988 sec/step, elapsed 0:25:34, exp. remaining 0:00:04, complete 99.69%
att-weights epoch 560, step 678, max_size:classes 7, max_size:data 337, mem_usage:GPU:0 0.9GB, num_seqs 11, 0.952 sec/step, elapsed 0:25:35, exp. remaining 0:00:04, complete 99.69%
att-weights epoch 560, step 679, max_size:classes 7, max_size:data 293, mem_usage:GPU:0 0.9GB, num_seqs 13, 1.573 sec/step, elapsed 0:25:36, exp. remaining 0:00:04, complete 99.69%
att-weights epoch 560, step 680, max_size:classes 6, max_size:data 268, mem_usage:GPU:0 0.9GB, num_seqs 14, 1.167 sec/step, elapsed 0:25:37, exp. remaining 0:00:04, complete 99.69%
att-weights epoch 560, step 681, max_size:classes 7, max_size:data 311, mem_usage:GPU:0 0.9GB, num_seqs 12, 1.177 sec/step, elapsed 0:25:39, exp. remaining 0:00:04, complete 99.69%
att-weights epoch 560, step 682, max_size:classes 4, max_size:data 310, mem_usage:GPU:0 0.9GB, num_seqs 9, 0.751 sec/step, elapsed 0:25:39, exp. remaining 0:00:04, complete 99.69%
Stats:
  mem_usage:GPU:0: Stats(mean=0.9GB, std_dev=0.0B, min=0.9GB, max=0.9GB, num_seqs=683, avg_data_len=1)
att-weights epoch 560, finished after 683 steps, 0:25:39 elapsed (24.2% computing time)
Layer 'dec_02_att_weights' Stats:
  2620 seqs, 103825928 total frames, 39628.216794 average frames
  Mean: 0.005243314563570105
  Std dev: 0.064714690113644
  Min/max: 0.0 / 1.0
Layer 'dec_01_att_weights' Stats:
  2620 seqs, 103825928 total frames, 39628.216794 average frames
  Mean: 0.005243314559840874
  Std dev: 0.06602755690080123
  Min/max: 0.0 / 1.0
Layer 'dec_03_att_weights' Stats:
  2620 seqs, 103825928 total frames, 39628.216794 average frames
  Mean: 0.005243314547459045
  Std dev: 0.05509198362677103
  Min/max: 0.0 / 1.0
Layer 'dec_04_att_weights' Stats:
  2620 seqs, 103825928 total frames, 39628.216794 average frames
  Mean: 0.005243314560979859
  Std dev: 0.06380857783721997
  Min/max: 0.0 / 1.0
Layer 'dec_05_att_weights' Stats:
  2620 seqs, 103825928 total frames, 39628.216794 average frames
  Mean: 0.0052433145590876865
  Std dev: 0.04738234002366064
  Min/max: 0.0 / 1.0
Layer 'dec_06_att_weights' Stats:
  2620 seqs, 103825928 total frames, 39628.216794 average frames
  Mean: 0.005243314558316095
  Std dev: 0.05738882448957763
  Min/max: 0.0 / 1.0
Layer 'dec_07_att_weights' Stats:
  2620 seqs, 103825928 total frames, 39628.216794 average frames
  Mean: 0.005243314557618024
  Std dev: 0.04908226366947897
  Min/max: 0.0 / 1.0
Layer 'dec_08_att_weights' Stats:
  2620 seqs, 103825928 total frames, 39628.216794 average frames
  Mean: 0.005243314561145169
  Std dev: 0.06083064491056315
  Min/max: 0.0 / 1.0
Layer 'dec_09_att_weights' Stats:
  2620 seqs, 103825928 total frames, 39628.216794 average frames
  Mean: 0.005243314557324085
  Std dev: 0.07221602721065086
  Min/max: 0.0 / 1.0
Layer 'dec_10_att_weights' Stats:
  2620 seqs, 103825928 total frames, 39628.216794 average frames
  Mean: 0.005243314559528562
  Std dev: 0.07221506561287987
  Min/max: 0.0 / 1.0
Layer 'dec_11_att_weights' Stats:
  2620 seqs, 103825928 total frames, 39628.216794 average frames
  Mean: 0.005243314564231438
  Std dev: 0.07220485593153754
  Min/max: 0.0 / 1.0
Layer 'dec_12_att_weights' Stats:
  2620 seqs, 103825928 total frames, 39628.216794 average frames
  Mean: 0.005243314557085278
  Std dev: 0.07076180252657492
  Min/max: 0.0 / 1.0
Quitting
+------- EPILOGUE SCRIPT -----------------------------------------------
|
| Job ID ..............: 9506152
| Stopped at ..........: Tue Jul  2 14:11:01 CEST 2019
| Resources requested .: scratch_free=5G,h_fsize=20G,num_proc=5,h_vmem=1536G,s_core=0,pxe=ubuntu_16.04,h_rss=8G,h_rt=7200,gpu=1
| Resources used ......: cpu=00:35:14, mem=8092.96678 GB s, io=11.97166 GB, vmem=4.146G, maxvmem=4.171G, last_file_cache=4.253G, last_rss=3M, max-cache=3.747G
| Memory used .........: 8.000G / 8.000G (100.0%)
| Total time used .....: 0:28:02
|
+------- EPILOGUE SCRIPT -----------------------------------------------
