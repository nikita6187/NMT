+------- PROLOGUE SCRIPT -----------------------------------------------
|
| Job ID ...........: 9505916
| Started at .......: Tue Jul  2 11:40:10 CEST 2019
| Execution host ...: cluster-cn-246
| Cluster queue ....: 4-GPU-1080
| Script ...........: /var/spool/sge/cluster-cn-246/job_scripts/9505916
| > #!/bin/bash
| > 
| > DATA=$1
| > 
| > export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64:/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda/extras/CUPTI/lib64/:/usr/local/cuda-9.1/extras/CUPTI/lib64/:/usr/local/cuda-9.0/extras/CUPTI/lib64:/usr/lib/nvidia-418
| > #export LD_LIBRARY_PATH="/usr/local/lib::/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda-9.0/lib64:/usr/lib/nvidia-390:/opt/rbi/sge/lib/glinux:/opt/rbi/sge/lib/glinux"
| > 
| > source /u/bahar/settings/python3-returnn-tf1.9/bin/activate
| > 
| > python3 ~/returnn-parnia/tools/get-attention-weights.py /u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc.config --epoch 481 --data 'config:get_dataset("${DATA}")' --dump_dir ~/makarov/NMT/hmm-factorization/experiments/asr2019/transformer/forward --layers "dec_02_att_weights" --layers "dec_01_att_weights" --layers "dec_03_att_weights" --layers "dec_04_att_weights" --layers "dec_05_att_weights" --layers "dec_06_att_weights" --layers "dec_07_att_weights" --layers "dec_08_att_weights" --layers "dec_09_att_weights" --layers "dec_10_att_weights" --layers "dec_11_att_weights" --layers "dec_12_att_weights" --rec_layer "output" --batch_size 600 --tf_log_dir /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/transformer/forward/tf_log_dir --reset_seq_ordering sorted_reverse
|
+------- PROLOGUE SCRIPT -----------------------------------------------
RETURNN get-attention-weights starting up.
RETURNN starting up, version 20181022.053313--git-dfada43-dirty, date/time 2019-07-02-11-40-12 (UTC+0200), pid 30450, cwd /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc, Python /u/bahar/settings/python3-returnn-tf1.9/bin/python3
RETURNN config: /u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc.config
RETURNN command line options: ()
Hostname: cluster-cn-246
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
       incarnation: 2823097897374161720
  2/2: name: "/device:GPU:0"
       device_type: "GPU"
       memory_limit: 10911236096
       locality {
         bus_id: 1
         links {
         }
       }
       incarnation: 9341818341909794250
       physical_device_desc: "device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1"
Using gpu device 1: GeForce GTX 1080 Ti
Device not set explicitly, and we found a GPU, which we will use.
Model file prefix: /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc/net-model/network
NOTE: We will use 'sorted_reverse' seq ordering.
Unhandled exception <class 'AssertionError'> in thread <_MainThread(MainThread, started 47852395011072)>, proc 30450.

Thread current, main, <_MainThread(MainThread, started 47852395011072)>:
(Excluded thread.)

That were all threads.
+------- EPILOGUE SCRIPT -----------------------------------------------
|
| Job ID ..............: 9505916
| Stopped at ..........: Tue Jul  2 11:40:43 CEST 2019
| Resources requested .: h_rss=4G,h_vmem=1536G,num_proc=5,scratch_free=5G,gpu=1,pxe=ubuntu_16.04,h_rt=7200,h_fsize=20G,s_core=0
| Resources used ......: cpu=00:00:25, mem=19.18538 GB s, io=0.02932 GB, vmem=1023.906M, maxvmem=1023.906M, last_file_cache=68K, last_rss=2M, max-cache=730M
| Memory used .........: 730M / 4.000G (17.8%)
| Total time used .....: 0:00:28
|
+------- EPILOGUE SCRIPT -----------------------------------------------
+------- PROLOGUE SCRIPT -----------------------------------------------
|
| Job ID ...........: 9505983
| Started at .......: Tue Jul  2 12:08:08 CEST 2019
| Execution host ...: cluster-cn-244
| Cluster queue ....: 4-GPU-1080
| Script ...........: /var/spool/sge/cluster-cn-244/job_scripts/9505983
| > #!/bin/bash
| > 
| > export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64:/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda/extras/CUPTI/lib64/:/usr/local/cuda-9.1/extras/CUPTI/lib64/:/usr/local/cuda-9.0/extras/CUPTI/lib64:/usr/lib/nvidia-418
| > #export LD_LIBRARY_PATH="/usr/local/lib::/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda-9.0/lib64:/usr/lib/nvidia-390:/opt/rbi/sge/lib/glinux:/opt/rbi/sge/lib/glinux"
| > 
| > source /u/bahar/settings/python3-returnn-tf1.9/bin/activate
| > 
| > python3 ~/returnn-parnia/tools/get-attention-weights.py /u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc.config --epoch 481 --data 'config:get_dataset("'$1'")' --dump_dir ~/makarov/NMT/hmm-factorization/experiments/asr2019/transformer/forward-$1 --layers "dec_02_att_weights" --layers "dec_01_att_weights" --layers "dec_03_att_weights" --layers "dec_04_att_weights" --layers "dec_05_att_weights" --layers "dec_06_att_weights" --layers "dec_07_att_weights" --layers "dec_08_att_weights" --layers "dec_09_att_weights" --layers "dec_10_att_weights" --layers "dec_11_att_weights" --layers "dec_12_att_weights" --rec_layer "output" --batch_size 600 --tf_log_dir /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/transformer/forward/tf_log_dir --reset_seq_ordering sorted_reverse
|
+------- PROLOGUE SCRIPT -----------------------------------------------
RETURNN get-attention-weights starting up.
RETURNN starting up, version 20181022.053313--git-dfada43-dirty, date/time 2019-07-02-12-08-24 (UTC+0200), pid 18977, cwd /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc, Python /u/bahar/settings/python3-returnn-tf1.9/bin/python3
RETURNN config: /u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc.config
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
       incarnation: 5919342370686817045
  2/2: name: "/device:GPU:0"
       device_type: "GPU"
       memory_limit: 10911236096
       locality {
         bus_id: 1
         links {
         }
       }
       incarnation: 8425478695778460751
       physical_device_desc: "device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1"
Using gpu device 1: GeForce GTX 1080 Ti
Device not set explicitly, and we found a GPU, which we will use.
Model file prefix: /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc/net-model/network
NOTE: We will use 'sorted_reverse' seq ordering.
Setup tf.Session with options {'device_count': {'GPU': 1}, 'log_device_placement': False} ...
layer root/'data' output: Data(name='data', shape=(None, 40))
layer root/'source' output: Data(name='source_output', shape=(None, 40))
layer root/'lstm0_fw' output: Data(name='lstm0_fw_output', shape=(None, 1024), batch_dim_axis=1)
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9505983.1.4-GPU-1080/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.cc -o /var/tmp/9505983.1.4-GPU-1080/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9505983.1.4-GPU-1080/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.cc -o /var/tmp/9505983.1.4-GPU-1080/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
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
layer root/'dec_12_att_value0' output: Data(name='dec_12_att_value0_output', shape=(None, 512))
layer root/'dec_12_att_value' output: Data(name='dec_12_att_value_output', shape=(None, 8, 64))
layer root/'dec_05_att_value0' output: Data(name='dec_05_att_value0_output', shape=(None, 512))
layer root/'dec_05_att_value' output: Data(name='dec_05_att_value_output', shape=(None, 8, 64))
layer root/'dec_06_att_value0' output: Data(name='dec_06_att_value0_output', shape=(None, 512))
layer root/'dec_06_att_value' output: Data(name='dec_06_att_value_output', shape=(None, 8, 64))
layer root/'dec_11_att_key0' output: Data(name='dec_11_att_key0_output', shape=(None, 512))
layer root/'dec_11_att_key' output: Data(name='dec_11_att_key_output', shape=(None, 8, 64))
layer root/'dec_06_att_key0' output: Data(name='dec_06_att_key0_output', shape=(None, 512))
layer root/'dec_06_att_key' output: Data(name='dec_06_att_key_output', shape=(None, 8, 64))
layer root/'dec_03_att_value0' output: Data(name='dec_03_att_value0_output', shape=(None, 512))
layer root/'dec_03_att_value' output: Data(name='dec_03_att_value_output', shape=(None, 8, 64))
layer root/'dec_07_att_key0' output: Data(name='dec_07_att_key0_output', shape=(None, 512))
layer root/'dec_07_att_key' output: Data(name='dec_07_att_key_output', shape=(None, 8, 64))
layer root/'dec_12_att_key0' output: Data(name='dec_12_att_key0_output', shape=(None, 512))
layer root/'dec_12_att_key' output: Data(name='dec_12_att_key_output', shape=(None, 8, 64))
layer root/'dec_01_att_key0' output: Data(name='dec_01_att_key0_output', shape=(None, 512))
layer root/'dec_01_att_key' output: Data(name='dec_01_att_key_output', shape=(None, 8, 64))
layer root/'dec_05_att_key0' output: Data(name='dec_05_att_key0_output', shape=(None, 512))
layer root/'dec_05_att_key' output: Data(name='dec_05_att_key_output', shape=(None, 8, 64))
layer root/'dec_08_att_value0' output: Data(name='dec_08_att_value0_output', shape=(None, 512))
layer root/'dec_08_att_value' output: Data(name='dec_08_att_value_output', shape=(None, 8, 64))
layer root/'dec_09_att_key0' output: Data(name='dec_09_att_key0_output', shape=(None, 512))
layer root/'dec_09_att_key' output: Data(name='dec_09_att_key_output', shape=(None, 8, 64))
layer root/'dec_11_att_value0' output: Data(name='dec_11_att_value0_output', shape=(None, 512))
layer root/'dec_11_att_value' output: Data(name='dec_11_att_value_output', shape=(None, 8, 64))
layer root/'dec_02_att_key0' output: Data(name='dec_02_att_key0_output', shape=(None, 512))
layer root/'dec_02_att_key' output: Data(name='dec_02_att_key_output', shape=(None, 8, 64))
layer root/'dec_10_att_key0' output: Data(name='dec_10_att_key0_output', shape=(None, 512))
layer root/'dec_10_att_key' output: Data(name='dec_10_att_key_output', shape=(None, 8, 64))
layer root/'dec_07_att_value0' output: Data(name='dec_07_att_value0_output', shape=(None, 512))
layer root/'dec_07_att_value' output: Data(name='dec_07_att_value_output', shape=(None, 8, 64))
layer root/'dec_10_att_value0' output: Data(name='dec_10_att_value0_output', shape=(None, 512))
layer root/'dec_10_att_value' output: Data(name='dec_10_att_value_output', shape=(None, 8, 64))
layer root/'dec_04_att_value0' output: Data(name='dec_04_att_value0_output', shape=(None, 512))
layer root/'dec_04_att_value' output: Data(name='dec_04_att_value_output', shape=(None, 8, 64))
layer root/'dec_04_att_key0' output: Data(name='dec_04_att_key0_output', shape=(None, 512))
layer root/'dec_04_att_key' output: Data(name='dec_04_att_key_output', shape=(None, 8, 64))
layer root/'dec_08_att_key0' output: Data(name='dec_08_att_key0_output', shape=(None, 512))
layer root/'dec_08_att_key' output: Data(name='dec_08_att_key_output', shape=(None, 8, 64))
layer root/'dec_02_att_value0' output: Data(name='dec_02_att_value0_output', shape=(None, 512))
layer root/'dec_02_att_value' output: Data(name='dec_02_att_value_output', shape=(None, 8, 64))
layer root/'dec_03_att_key0' output: Data(name='dec_03_att_key0_output', shape=(None, 512))
layer root/'dec_03_att_key' output: Data(name='dec_03_att_key_output', shape=(None, 8, 64))
layer root/'dec_09_att_value0' output: Data(name='dec_09_att_value0_output', shape=(None, 512))
layer root/'dec_09_att_value' output: Data(name='dec_09_att_value_output', shape=(None, 8, 64))
layer root/'dec_01_att_value0' output: Data(name='dec_01_att_value0_output', shape=(None, 512))
layer root/'dec_01_att_value' output: Data(name='dec_01_att_value_output', shape=(None, 8, 64))
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
+------- EPILOGUE SCRIPT -----------------------------------------------
|
| Job ID ..............: 9505983
| Stopped at ..........: Tue Jul  2 12:10:11 CEST 2019
| Resources requested .: h_fsize=20G,scratch_free=5G,num_proc=5,s_core=0,h_vmem=1536G,pxe=ubuntu_16.04,gpu=1,h_rt=7200,h_rss=4G
| Resources used ......: cpu=00:01:05, mem=38.32296 GB s, io=1.27655 GB, vmem=1.413G, maxvmem=1.934G, last_file_cache=1.729G, last_rss=2M, max-cache=1.226G
| Memory used .........: 2.956G / 4.000G (73.9%)
| Total time used .....: 0:02:03
|
+------- EPILOGUE SCRIPT -----------------------------------------------
+------- PROLOGUE SCRIPT -----------------------------------------------
|
| Job ID ...........: 9505995
| Started at .......: Tue Jul  2 12:13:08 CEST 2019
| Execution host ...: cluster-cn-240
| Cluster queue ....: 4-GPU-1080
| Script ...........: /var/spool/sge/cluster-cn-240/job_scripts/9505995
| > #!/bin/bash
| > 
| > export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64:/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda/extras/CUPTI/lib64/:/usr/local/cuda-9.1/extras/CUPTI/lib64/:/usr/local/cuda-9.0/extras/CUPTI/lib64:/usr/lib/nvidia-418
| > #export LD_LIBRARY_PATH="/usr/local/lib::/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda-9.0/lib64:/usr/lib/nvidia-390:/opt/rbi/sge/lib/glinux:/opt/rbi/sge/lib/glinux"
| > 
| > source /u/bahar/settings/python3-returnn-tf1.9/bin/activate
| > 
| > python3 ~/returnn-parnia/tools/get-attention-weights.py /u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc.config --epoch 481 --data 'config:get_dataset("'$1'")' --dump_dir ~/makarov/NMT/hmm-factorization/experiments/asr2019/transformer/forward-$1 --layers "dec_02_att_weights" --layers "dec_01_att_weights" --layers "dec_03_att_weights" --layers "dec_04_att_weights" --layers "dec_05_att_weights" --layers "dec_06_att_weights" --layers "dec_07_att_weights" --layers "dec_08_att_weights" --layers "dec_09_att_weights" --layers "dec_10_att_weights" --layers "dec_11_att_weights" --layers "dec_12_att_weights" --rec_layer "output" --batch_size 4000 --tf_log_dir /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/transformer/forward/tf_log_dir --reset_seq_ordering sorted_reverse
|
+------- PROLOGUE SCRIPT -----------------------------------------------
RETURNN get-attention-weights starting up.
RETURNN starting up, version 20181022.053313--git-dfada43-dirty, date/time 2019-07-02-12-13-11 (UTC+0200), pid 29913, cwd /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc, Python /u/bahar/settings/python3-returnn-tf1.9/bin/python3
RETURNN config: /u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc.config
RETURNN command line options: ()
Hostname: cluster-cn-240
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
       incarnation: 15265471555544704341
  2/2: name: "/device:GPU:0"
       device_type: "GPU"
       memory_limit: 10910862541
       locality {
         bus_id: 1
         links {
         }
       }
       incarnation: 17370998107423429171
       physical_device_desc: "device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1"
Using gpu device 0: GeForce GTX 1080 Ti
Device not set explicitly, and we found a GPU, which we will use.
Model file prefix: /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc/net-model/network
NOTE: We will use 'sorted_reverse' seq ordering.
Setup tf.Session with options {'device_count': {'GPU': 1}, 'log_device_placement': False} ...
layer root/'data' output: Data(name='data', shape=(None, 40))
layer root/'source' output: Data(name='source_output', shape=(None, 40))
layer root/'lstm0_fw' output: Data(name='lstm0_fw_output', shape=(None, 1024), batch_dim_axis=1)
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9505995.1.4-GPU-1080/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.cc -o /var/tmp/9505995.1.4-GPU-1080/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9505995.1.4-GPU-1080/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.cc -o /var/tmp/9505995.1.4-GPU-1080/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
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
layer root/'dec_12_att_value0' output: Data(name='dec_12_att_value0_output', shape=(None, 512))
layer root/'dec_12_att_value' output: Data(name='dec_12_att_value_output', shape=(None, 8, 64))
layer root/'dec_12_att_key0' output: Data(name='dec_12_att_key0_output', shape=(None, 512))
layer root/'dec_12_att_key' output: Data(name='dec_12_att_key_output', shape=(None, 8, 64))
layer root/'dec_03_att_value0' output: Data(name='dec_03_att_value0_output', shape=(None, 512))
layer root/'dec_03_att_value' output: Data(name='dec_03_att_value_output', shape=(None, 8, 64))
layer root/'dec_04_att_value0' output: Data(name='dec_04_att_value0_output', shape=(None, 512))
layer root/'dec_04_att_value' output: Data(name='dec_04_att_value_output', shape=(None, 8, 64))
layer root/'dec_09_att_key0' output: Data(name='dec_09_att_key0_output', shape=(None, 512))
layer root/'dec_09_att_key' output: Data(name='dec_09_att_key_output', shape=(None, 8, 64))
layer root/'dec_07_att_key0' output: Data(name='dec_07_att_key0_output', shape=(None, 512))
layer root/'dec_07_att_key' output: Data(name='dec_07_att_key_output', shape=(None, 8, 64))
layer root/'dec_10_att_key0' output: Data(name='dec_10_att_key0_output', shape=(None, 512))
layer root/'dec_10_att_key' output: Data(name='dec_10_att_key_output', shape=(None, 8, 64))
layer root/'dec_01_att_key0' output: Data(name='dec_01_att_key0_output', shape=(None, 512))
layer root/'dec_01_att_key' output: Data(name='dec_01_att_key_output', shape=(None, 8, 64))
layer root/'dec_02_att_key0' output: Data(name='dec_02_att_key0_output', shape=(None, 512))
layer root/'dec_02_att_key' output: Data(name='dec_02_att_key_output', shape=(None, 8, 64))
layer root/'dec_06_att_value0' output: Data(name='dec_06_att_value0_output', shape=(None, 512))
layer root/'dec_06_att_value' output: Data(name='dec_06_att_value_output', shape=(None, 8, 64))
layer root/'dec_05_att_value0' output: Data(name='dec_05_att_value0_output', shape=(None, 512))
layer root/'dec_05_att_value' output: Data(name='dec_05_att_value_output', shape=(None, 8, 64))
layer root/'dec_11_att_key0' output: Data(name='dec_11_att_key0_output', shape=(None, 512))
layer root/'dec_11_att_key' output: Data(name='dec_11_att_key_output', shape=(None, 8, 64))
layer root/'dec_06_att_key0' output: Data(name='dec_06_att_key0_output', shape=(None, 512))
layer root/'dec_06_att_key' output: Data(name='dec_06_att_key_output', shape=(None, 8, 64))
layer root/'dec_07_att_value0' output: Data(name='dec_07_att_value0_output', shape=(None, 512))
layer root/'dec_07_att_value' output: Data(name='dec_07_att_value_output', shape=(None, 8, 64))
layer root/'dec_02_att_value0' output: Data(name='dec_02_att_value0_output', shape=(None, 512))
layer root/'dec_02_att_value' output: Data(name='dec_02_att_value_output', shape=(None, 8, 64))
layer root/'dec_05_att_key0' output: Data(name='dec_05_att_key0_output', shape=(None, 512))
layer root/'dec_05_att_key' output: Data(name='dec_05_att_key_output', shape=(None, 8, 64))
layer root/'dec_03_att_key0' output: Data(name='dec_03_att_key0_output', shape=(None, 512))
layer root/'dec_03_att_key' output: Data(name='dec_03_att_key_output', shape=(None, 8, 64))
layer root/'dec_01_att_value0' output: Data(name='dec_01_att_value0_output', shape=(None, 512))
layer root/'dec_01_att_value' output: Data(name='dec_01_att_value_output', shape=(None, 8, 64))
layer root/'dec_08_att_key0' output: Data(name='dec_08_att_key0_output', shape=(None, 512))
layer root/'dec_08_att_key' output: Data(name='dec_08_att_key_output', shape=(None, 8, 64))
layer root/'dec_10_att_value0' output: Data(name='dec_10_att_value0_output', shape=(None, 512))
layer root/'dec_10_att_value' output: Data(name='dec_10_att_value_output', shape=(None, 8, 64))
layer root/'dec_09_att_value0' output: Data(name='dec_09_att_value0_output', shape=(None, 512))
layer root/'dec_09_att_value' output: Data(name='dec_09_att_value_output', shape=(None, 8, 64))
layer root/'dec_08_att_value0' output: Data(name='dec_08_att_value0_output', shape=(None, 512))
layer root/'dec_08_att_value' output: Data(name='dec_08_att_value_output', shape=(None, 8, 64))
layer root/'dec_11_att_value0' output: Data(name='dec_11_att_value0_output', shape=(None, 512))
layer root/'dec_11_att_value' output: Data(name='dec_11_att_value_output', shape=(None, 8, 64))
layer root/'dec_04_att_key0' output: Data(name='dec_04_att_key0_output', shape=(None, 512))
layer root/'dec_04_att_key' output: Data(name='dec_04_att_key_output', shape=(None, 8, 64))
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
  extern data: data: Data(shape=(None, 40)), classes: Data(shape=(None,), dtype='int32', sparse=True, dim=10025, available_for_inference=False)
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
TF: log_dir: /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/transformer/forward/tf_log_dir/prefix:test-other-481-2019-07-02-10-13-10
Note: There are still these uninitialized variables: ['learning_rate:0']
att-weights epoch 481, step 0, max_size:classes 137, max_size:data 3452, mem_usage:GPU:0 0.9GB, num_seqs 1, 3.611 sec/step, elapsed 0:00:09, exp. remaining 0:43:22, complete 0.37%
att-weights epoch 481, step 1, max_size:classes 98, max_size:data 2953, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.596 sec/step, elapsed 0:00:11, exp. remaining 0:46:30, complete 0.41%
att-weights epoch 481, step 2, max_size:classes 98, max_size:data 3365, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.709 sec/step, elapsed 0:00:13, exp. remaining 0:50:01, complete 0.44%
att-weights epoch 481, step 3, max_size:classes 89, max_size:data 2804, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.257 sec/step, elapsed 0:00:14, exp. remaining 0:51:00, complete 0.48%
att-weights epoch 481, step 4, max_size:classes 103, max_size:data 3373, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.608 sec/step, elapsed 0:00:16, exp. remaining 0:53:02, complete 0.51%
att-weights epoch 481, step 5, max_size:classes 95, max_size:data 3114, mem_usage:GPU:0 0.9GB, num_seqs 1, 3.727 sec/step, elapsed 0:00:20, exp. remaining 1:01:16, complete 0.54%
att-weights epoch 481, step 6, max_size:classes 103, max_size:data 2961, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.490 sec/step, elapsed 0:00:21, exp. remaining 1:02:03, complete 0.58%
att-weights epoch 481, step 7, max_size:classes 89, max_size:data 2838, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.399 sec/step, elapsed 0:00:23, exp. remaining 1:02:44, complete 0.61%
att-weights epoch 481, step 8, max_size:classes 94, max_size:data 2277, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.221 sec/step, elapsed 0:00:24, exp. remaining 1:02:41, complete 0.65%
att-weights epoch 481, step 9, max_size:classes 79, max_size:data 3353, mem_usage:GPU:0 0.9GB, num_seqs 1, 5.296 sec/step, elapsed 0:00:29, exp. remaining 1:09:07, complete 0.71%
att-weights epoch 481, step 10, max_size:classes 87, max_size:data 3095, mem_usage:GPU:0 0.9GB, num_seqs 1, 3.567 sec/step, elapsed 0:00:33, exp. remaining 1:13:56, complete 0.75%
att-weights epoch 481, step 11, max_size:classes 98, max_size:data 3296, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.741 sec/step, elapsed 0:00:39, exp. remaining 1:23:54, complete 0.78%
att-weights epoch 481, step 12, max_size:classes 91, max_size:data 2540, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.287 sec/step, elapsed 0:00:40, exp. remaining 1:22:59, complete 0.82%
att-weights epoch 481, step 13, max_size:classes 86, max_size:data 2716, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.249 sec/step, elapsed 0:00:42, exp. remaining 1:22:04, complete 0.85%
att-weights epoch 481, step 14, max_size:classes 83, max_size:data 2145, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.980 sec/step, elapsed 0:00:43, exp. remaining 1:20:43, complete 0.88%
att-weights epoch 481, step 15, max_size:classes 74, max_size:data 2312, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.050 sec/step, elapsed 0:00:44, exp. remaining 1:19:35, complete 0.92%
att-weights epoch 481, step 16, max_size:classes 70, max_size:data 2275, mem_usage:GPU:0 0.9GB, num_seqs 1, 2.106 sec/step, elapsed 0:00:46, exp. remaining 1:20:22, complete 0.95%
att-weights epoch 481, step 17, max_size:classes 73, max_size:data 2610, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.456 sec/step, elapsed 0:00:47, exp. remaining 1:20:01, complete 0.99%
att-weights epoch 481, step 18, max_size:classes 76, max_size:data 1892, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.480 sec/step, elapsed 0:00:49, exp. remaining 1:19:42, complete 1.02%
att-weights epoch 481, step 19, max_size:classes 75, max_size:data 2350, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.036 sec/step, elapsed 0:00:50, exp. remaining 1:18:44, complete 1.05%
att-weights epoch 481, step 20, max_size:classes 83, max_size:data 2302, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.073 sec/step, elapsed 0:00:51, exp. remaining 1:15:29, complete 1.12%
att-weights epoch 481, step 21, max_size:classes 75, max_size:data 2230, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.907 sec/step, elapsed 0:00:52, exp. remaining 1:14:32, complete 1.16%
att-weights epoch 481, step 22, max_size:classes 78, max_size:data 1853, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.810 sec/step, elapsed 0:00:53, exp. remaining 1:13:30, complete 1.19%
att-weights epoch 481, step 23, max_size:classes 74, max_size:data 2043, mem_usage:GPU:0 0.9GB, num_seqs 1, 4.850 sec/step, elapsed 0:00:58, exp. remaining 1:17:57, complete 1.22%
att-weights epoch 481, step 24, max_size:classes 65, max_size:data 2213, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.501 sec/step, elapsed 0:00:59, exp. remaining 1:17:47, complete 1.26%
att-weights epoch 481, step 25, max_size:classes 67, max_size:data 2336, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.118 sec/step, elapsed 0:01:00, exp. remaining 1:17:08, complete 1.29%
att-weights epoch 481, step 26, max_size:classes 71, max_size:data 2120, mem_usage:GPU:0 0.9GB, num_seqs 1, 3.614 sec/step, elapsed 0:01:04, exp. remaining 1:19:36, complete 1.33%
att-weights epoch 481, step 27, max_size:classes 69, max_size:data 1907, mem_usage:GPU:0 0.9GB, num_seqs 1, 4.715 sec/step, elapsed 0:01:08, exp. remaining 1:23:17, complete 1.36%
att-weights epoch 481, step 28, max_size:classes 68, max_size:data 2022, mem_usage:GPU:0 0.9GB, num_seqs 1, 9.233 sec/step, elapsed 0:01:18, exp. remaining 1:32:06, complete 1.40%
att-weights epoch 481, step 29, max_size:classes 63, max_size:data 1564, mem_usage:GPU:0 0.9GB, num_seqs 2, 31.295 sec/step, elapsed 0:01:49, exp. remaining 2:05:58, complete 1.43%
att-weights epoch 481, step 30, max_size:classes 66, max_size:data 2154, mem_usage:GPU:0 0.9GB, num_seqs 1, 7.037 sec/step, elapsed 0:01:56, exp. remaining 2:10:54, complete 1.46%
att-weights epoch 481, step 31, max_size:classes 69, max_size:data 1730, mem_usage:GPU:0 0.9GB, num_seqs 1, 2.257 sec/step, elapsed 0:01:58, exp. remaining 2:10:24, complete 1.50%
att-weights epoch 481, step 32, max_size:classes 90, max_size:data 2030, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.349 sec/step, elapsed 0:02:00, exp. remaining 2:08:54, complete 1.53%
att-weights epoch 481, step 33, max_size:classes 68, max_size:data 1874, mem_usage:GPU:0 0.9GB, num_seqs 1, 5.209 sec/step, elapsed 0:02:05, exp. remaining 2:08:40, complete 1.60%
att-weights epoch 481, step 34, max_size:classes 62, max_size:data 2075, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.337 sec/step, elapsed 0:02:06, exp. remaining 2:07:18, complete 1.63%
att-weights epoch 481, step 35, max_size:classes 74, max_size:data 1982, mem_usage:GPU:0 0.9GB, num_seqs 1, 12.772 sec/step, elapsed 0:02:19, exp. remaining 2:17:12, complete 1.67%
att-weights epoch 481, step 36, max_size:classes 68, max_size:data 2276, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.361 sec/step, elapsed 0:02:20, exp. remaining 2:13:01, complete 1.74%
att-weights epoch 481, step 37, max_size:classes 67, max_size:data 2127, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.089 sec/step, elapsed 0:02:22, exp. remaining 2:11:25, complete 1.77%
att-weights epoch 481, step 38, max_size:classes 65, max_size:data 2039, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.030 sec/step, elapsed 0:02:23, exp. remaining 2:09:50, complete 1.80%
att-weights epoch 481, step 39, max_size:classes 65, max_size:data 1559, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.893 sec/step, elapsed 0:02:23, exp. remaining 2:08:11, complete 1.84%
att-weights epoch 481, step 40, max_size:classes 66, max_size:data 2304, mem_usage:GPU:0 0.9GB, num_seqs 1, 7.102 sec/step, elapsed 0:02:31, exp. remaining 2:12:01, complete 1.87%
att-weights epoch 481, step 41, max_size:classes 58, max_size:data 2136, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.122 sec/step, elapsed 0:02:32, exp. remaining 2:08:14, complete 1.94%
att-weights epoch 481, step 42, max_size:classes 67, max_size:data 1908, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.731 sec/step, elapsed 0:02:33, exp. remaining 2:07:25, complete 1.97%
att-weights epoch 481, step 43, max_size:classes 72, max_size:data 2457, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.416 sec/step, elapsed 0:02:35, exp. remaining 2:04:13, complete 2.04%
att-weights epoch 481, step 44, max_size:classes 66, max_size:data 2009, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.035 sec/step, elapsed 0:02:36, exp. remaining 2:02:57, complete 2.08%
att-weights epoch 481, step 45, max_size:classes 62, max_size:data 1674, mem_usage:GPU:0 0.9GB, num_seqs 2, 17.438 sec/step, elapsed 0:02:53, exp. remaining 2:12:14, complete 2.14%
att-weights epoch 481, step 46, max_size:classes 67, max_size:data 2085, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.077 sec/step, elapsed 0:02:54, exp. remaining 2:08:52, complete 2.21%
att-weights epoch 481, step 47, max_size:classes 64, max_size:data 1931, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.048 sec/step, elapsed 0:02:55, exp. remaining 2:05:41, complete 2.28%
att-weights epoch 481, step 48, max_size:classes 62, max_size:data 2066, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.934 sec/step, elapsed 0:02:56, exp. remaining 2:02:36, complete 2.35%
att-weights epoch 481, step 49, max_size:classes 60, max_size:data 2001, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.907 sec/step, elapsed 0:02:57, exp. remaining 1:59:41, complete 2.42%
att-weights epoch 481, step 50, max_size:classes 64, max_size:data 1869, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.652 sec/step, elapsed 0:02:59, exp. remaining 1:59:04, complete 2.45%
att-weights epoch 481, step 51, max_size:classes 63, max_size:data 2170, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.007 sec/step, elapsed 0:03:00, exp. remaining 1:58:04, complete 2.48%
att-weights epoch 481, step 52, max_size:classes 60, max_size:data 1923, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.451 sec/step, elapsed 0:03:01, exp. remaining 1:55:45, complete 2.55%
att-weights epoch 481, step 53, max_size:classes 57, max_size:data 2141, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.905 sec/step, elapsed 0:03:02, exp. remaining 1:53:14, complete 2.62%
att-weights epoch 481, step 54, max_size:classes 59, max_size:data 1729, mem_usage:GPU:0 0.9GB, num_seqs 2, 3.893 sec/step, elapsed 0:03:06, exp. remaining 1:52:38, complete 2.69%
att-weights epoch 481, step 55, max_size:classes 59, max_size:data 1827, mem_usage:GPU:0 0.9GB, num_seqs 2, 4.038 sec/step, elapsed 0:03:10, exp. remaining 1:52:09, complete 2.76%
att-weights epoch 481, step 56, max_size:classes 67, max_size:data 1811, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.549 sec/step, elapsed 0:03:12, exp. remaining 1:51:39, complete 2.79%
att-weights epoch 481, step 57, max_size:classes 64, max_size:data 1941, mem_usage:GPU:0 0.9GB, num_seqs 2, 10.087 sec/step, elapsed 0:03:22, exp. remaining 1:54:38, complete 2.86%
att-weights epoch 481, step 58, max_size:classes 67, max_size:data 1699, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.527 sec/step, elapsed 0:03:23, exp. remaining 1:54:06, complete 2.89%
att-weights epoch 481, step 59, max_size:classes 59, max_size:data 1264, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.719 sec/step, elapsed 0:03:24, exp. remaining 1:53:08, complete 2.93%
att-weights epoch 481, step 60, max_size:classes 71, max_size:data 2293, mem_usage:GPU:0 0.9GB, num_seqs 1, 3.265 sec/step, elapsed 0:03:27, exp. remaining 1:53:34, complete 2.96%
att-weights epoch 481, step 61, max_size:classes 65, max_size:data 1922, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.105 sec/step, elapsed 0:03:29, exp. remaining 1:53:23, complete 2.99%
att-weights epoch 481, step 62, max_size:classes 55, max_size:data 1785, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.495 sec/step, elapsed 0:03:31, exp. remaining 1:51:34, complete 3.06%
att-weights epoch 481, step 63, max_size:classes 60, max_size:data 1703, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.357 sec/step, elapsed 0:03:32, exp. remaining 1:49:46, complete 3.13%
att-weights epoch 481, step 64, max_size:classes 61, max_size:data 1814, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.619 sec/step, elapsed 0:03:34, exp. remaining 1:48:10, complete 3.20%
att-weights epoch 481, step 65, max_size:classes 59, max_size:data 2068, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.254 sec/step, elapsed 0:03:35, exp. remaining 1:46:28, complete 3.27%
att-weights epoch 481, step 66, max_size:classes 59, max_size:data 1474, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.090 sec/step, elapsed 0:03:37, exp. remaining 1:45:14, complete 3.33%
att-weights epoch 481, step 67, max_size:classes 56, max_size:data 1337, mem_usage:GPU:0 0.9GB, num_seqs 1, 11.942 sec/step, elapsed 0:03:49, exp. remaining 1:48:42, complete 3.40%
att-weights epoch 481, step 68, max_size:classes 60, max_size:data 2178, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.882 sec/step, elapsed 0:03:50, exp. remaining 1:46:54, complete 3.47%
att-weights epoch 481, step 69, max_size:classes 66, max_size:data 1807, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.107 sec/step, elapsed 0:03:51, exp. remaining 1:45:17, complete 3.54%
att-weights epoch 481, step 70, max_size:classes 56, max_size:data 2126, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.996 sec/step, elapsed 0:03:52, exp. remaining 1:43:40, complete 3.61%
att-weights epoch 481, step 71, max_size:classes 56, max_size:data 1737, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.191 sec/step, elapsed 0:03:53, exp. remaining 1:42:11, complete 3.67%
att-weights epoch 481, step 72, max_size:classes 55, max_size:data 1613, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.144 sec/step, elapsed 0:03:55, exp. remaining 1:40:45, complete 3.74%
att-weights epoch 481, step 73, max_size:classes 62, max_size:data 1674, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.021 sec/step, elapsed 0:03:57, exp. remaining 1:39:44, complete 3.81%
att-weights epoch 481, step 74, max_size:classes 57, max_size:data 1748, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.389 sec/step, elapsed 0:03:58, exp. remaining 1:38:29, complete 3.88%
att-weights epoch 481, step 75, max_size:classes 56, max_size:data 1904, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.039 sec/step, elapsed 0:04:00, exp. remaining 1:37:33, complete 3.95%
att-weights epoch 481, step 76, max_size:classes 53, max_size:data 1554, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.829 sec/step, elapsed 0:04:02, exp. remaining 1:36:33, complete 4.01%
att-weights epoch 481, step 77, max_size:classes 55, max_size:data 1737, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.093 sec/step, elapsed 0:04:03, exp. remaining 1:35:19, complete 4.08%
att-weights epoch 481, step 78, max_size:classes 49, max_size:data 1887, mem_usage:GPU:0 0.9GB, num_seqs 2, 6.892 sec/step, elapsed 0:04:10, exp. remaining 1:36:20, complete 4.15%
att-weights epoch 481, step 79, max_size:classes 53, max_size:data 1824, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.389 sec/step, elapsed 0:04:11, exp. remaining 1:35:14, complete 4.22%
att-weights epoch 481, step 80, max_size:classes 62, max_size:data 1552, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.242 sec/step, elapsed 0:04:13, exp. remaining 1:34:30, complete 4.29%
att-weights epoch 481, step 81, max_size:classes 57, max_size:data 1992, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.423 sec/step, elapsed 0:04:16, exp. remaining 1:33:50, complete 4.36%
att-weights epoch 481, step 82, max_size:classes 60, max_size:data 1831, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.455 sec/step, elapsed 0:04:18, exp. remaining 1:33:13, complete 4.42%
att-weights epoch 481, step 83, max_size:classes 58, max_size:data 1855, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.315 sec/step, elapsed 0:04:20, exp. remaining 1:32:56, complete 4.46%
att-weights epoch 481, step 84, max_size:classes 48, max_size:data 1706, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.037 sec/step, elapsed 0:04:22, exp. remaining 1:32:55, complete 4.49%
att-weights epoch 481, step 85, max_size:classes 55, max_size:data 1676, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.556 sec/step, elapsed 0:04:23, exp. remaining 1:32:01, complete 4.56%
att-weights epoch 481, step 86, max_size:classes 54, max_size:data 1499, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.222 sec/step, elapsed 0:04:24, exp. remaining 1:31:01, complete 4.63%
att-weights epoch 481, step 87, max_size:classes 52, max_size:data 1607, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.059 sec/step, elapsed 0:04:26, exp. remaining 1:29:59, complete 4.70%
att-weights epoch 481, step 88, max_size:classes 51, max_size:data 1855, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.171 sec/step, elapsed 0:04:27, exp. remaining 1:29:02, complete 4.76%
att-weights epoch 481, step 89, max_size:classes 55, max_size:data 1675, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.300 sec/step, elapsed 0:04:28, exp. remaining 1:28:09, complete 4.83%
att-weights epoch 481, step 90, max_size:classes 48, max_size:data 1570, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.247 sec/step, elapsed 0:04:29, exp. remaining 1:27:16, complete 4.90%
att-weights epoch 481, step 91, max_size:classes 48, max_size:data 1874, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.211 sec/step, elapsed 0:04:30, exp. remaining 1:26:23, complete 4.97%
att-weights epoch 481, step 92, max_size:classes 48, max_size:data 1686, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.742 sec/step, elapsed 0:04:31, exp. remaining 1:25:24, complete 5.04%
att-weights epoch 481, step 93, max_size:classes 50, max_size:data 2031, mem_usage:GPU:0 0.9GB, num_seqs 1, 1.007 sec/step, elapsed 0:04:32, exp. remaining 1:25:06, complete 5.07%
att-weights epoch 481, step 94, max_size:classes 61, max_size:data 1966, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.481 sec/step, elapsed 0:04:34, exp. remaining 1:24:58, complete 5.10%
att-weights epoch 481, step 95, max_size:classes 52, max_size:data 1728, mem_usage:GPU:0 0.9GB, num_seqs 2, 13.947 sec/step, elapsed 0:04:48, exp. remaining 1:28:03, complete 5.17%
att-weights epoch 481, step 96, max_size:classes 51, max_size:data 1732, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.255 sec/step, elapsed 0:04:49, exp. remaining 1:27:14, complete 5.24%
att-weights epoch 481, step 97, max_size:classes 49, max_size:data 1607, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.168 sec/step, elapsed 0:04:50, exp. remaining 1:26:24, complete 5.31%
att-weights epoch 481, step 98, max_size:classes 54, max_size:data 1646, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.108 sec/step, elapsed 0:04:51, exp. remaining 1:25:00, complete 5.41%
att-weights epoch 481, step 99, max_size:classes 49, max_size:data 1706, mem_usage:GPU:0 0.9GB, num_seqs 2, 4.146 sec/step, elapsed 0:04:55, exp. remaining 1:25:04, complete 5.48%
att-weights epoch 481, step 100, max_size:classes 45, max_size:data 1517, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.964 sec/step, elapsed 0:04:56, exp. remaining 1:24:14, complete 5.55%
att-weights epoch 481, step 101, max_size:classes 48, max_size:data 1634, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.165 sec/step, elapsed 0:04:57, exp. remaining 1:23:29, complete 5.61%
att-weights epoch 481, step 102, max_size:classes 51, max_size:data 1594, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.802 sec/step, elapsed 0:04:58, exp. remaining 1:22:39, complete 5.68%
att-weights epoch 481, step 103, max_size:classes 53, max_size:data 2006, mem_usage:GPU:0 0.9GB, num_seqs 1, 0.899 sec/step, elapsed 0:04:59, exp. remaining 1:21:51, complete 5.75%
att-weights epoch 481, step 104, max_size:classes 50, max_size:data 1370, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.997 sec/step, elapsed 0:05:00, exp. remaining 1:21:07, complete 5.82%
att-weights epoch 481, step 105, max_size:classes 46, max_size:data 1455, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.030 sec/step, elapsed 0:05:01, exp. remaining 1:20:23, complete 5.89%
att-weights epoch 481, step 106, max_size:classes 48, max_size:data 1673, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.470 sec/step, elapsed 0:05:03, exp. remaining 1:19:48, complete 5.95%
att-weights epoch 481, step 107, max_size:classes 52, max_size:data 1326, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.432 sec/step, elapsed 0:05:04, exp. remaining 1:19:13, complete 6.02%
att-weights epoch 481, step 108, max_size:classes 50, max_size:data 1498, mem_usage:GPU:0 0.9GB, num_seqs 2, 4.790 sec/step, elapsed 0:05:09, exp. remaining 1:19:30, complete 6.09%
att-weights epoch 481, step 109, max_size:classes 47, max_size:data 1258, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.896 sec/step, elapsed 0:05:10, exp. remaining 1:18:48, complete 6.16%
att-weights epoch 481, step 110, max_size:classes 54, max_size:data 1845, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.582 sec/step, elapsed 0:05:11, exp. remaining 1:18:16, complete 6.23%
att-weights epoch 481, step 111, max_size:classes 49, max_size:data 1461, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.121 sec/step, elapsed 0:05:12, exp. remaining 1:17:39, complete 6.29%
att-weights epoch 481, step 112, max_size:classes 49, max_size:data 1588, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.089 sec/step, elapsed 0:05:14, exp. remaining 1:16:36, complete 6.40%
att-weights epoch 481, step 113, max_size:classes 54, max_size:data 1573, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.159 sec/step, elapsed 0:05:16, exp. remaining 1:16:15, complete 6.46%
att-weights epoch 481, step 114, max_size:classes 48, max_size:data 1601, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.939 sec/step, elapsed 0:05:17, exp. remaining 1:15:38, complete 6.53%
att-weights epoch 481, step 115, max_size:classes 55, max_size:data 1652, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.049 sec/step, elapsed 0:05:18, exp. remaining 1:15:02, complete 6.60%
att-weights epoch 481, step 116, max_size:classes 48, max_size:data 1308, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.834 sec/step, elapsed 0:05:19, exp. remaining 1:14:25, complete 6.67%
att-weights epoch 481, step 117, max_size:classes 47, max_size:data 1462, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.921 sec/step, elapsed 0:05:19, exp. remaining 1:13:49, complete 6.74%
att-weights epoch 481, step 118, max_size:classes 48, max_size:data 1716, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.208 sec/step, elapsed 0:05:21, exp. remaining 1:13:18, complete 6.81%
att-weights epoch 481, step 119, max_size:classes 48, max_size:data 1471, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.037 sec/step, elapsed 0:05:23, exp. remaining 1:12:59, complete 6.87%
att-weights epoch 481, step 120, max_size:classes 54, max_size:data 1882, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.342 sec/step, elapsed 0:05:24, exp. remaining 1:12:31, complete 6.94%
att-weights epoch 481, step 121, max_size:classes 52, max_size:data 1285, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.973 sec/step, elapsed 0:05:26, exp. remaining 1:12:12, complete 7.01%
att-weights epoch 481, step 122, max_size:classes 44, max_size:data 1997, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.228 sec/step, elapsed 0:05:27, exp. remaining 1:11:43, complete 7.08%
att-weights epoch 481, step 123, max_size:classes 48, max_size:data 1695, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.095 sec/step, elapsed 0:05:28, exp. remaining 1:11:13, complete 7.15%
att-weights epoch 481, step 124, max_size:classes 53, max_size:data 1792, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.136 sec/step, elapsed 0:05:30, exp. remaining 1:10:45, complete 7.21%
att-weights epoch 481, step 125, max_size:classes 48, max_size:data 1449, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.019 sec/step, elapsed 0:05:31, exp. remaining 1:10:15, complete 7.28%
att-weights epoch 481, step 126, max_size:classes 48, max_size:data 1516, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.970 sec/step, elapsed 0:05:32, exp. remaining 1:09:45, complete 7.35%
att-weights epoch 481, step 127, max_size:classes 65, max_size:data 1781, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.210 sec/step, elapsed 0:05:33, exp. remaining 1:09:19, complete 7.42%
att-weights epoch 481, step 128, max_size:classes 43, max_size:data 1534, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.590 sec/step, elapsed 0:05:35, exp. remaining 1:09:10, complete 7.49%
att-weights epoch 481, step 129, max_size:classes 44, max_size:data 1660, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.209 sec/step, elapsed 0:05:37, exp. remaining 1:08:44, complete 7.55%
att-weights epoch 481, step 130, max_size:classes 51, max_size:data 1670, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.291 sec/step, elapsed 0:05:38, exp. remaining 1:08:20, complete 7.62%
att-weights epoch 481, step 131, max_size:classes 52, max_size:data 1941, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.541 sec/step, elapsed 0:05:39, exp. remaining 1:07:59, complete 7.69%
att-weights epoch 481, step 132, max_size:classes 49, max_size:data 1864, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.156 sec/step, elapsed 0:05:41, exp. remaining 1:07:34, complete 7.76%
att-weights epoch 481, step 133, max_size:classes 49, max_size:data 1311, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.188 sec/step, elapsed 0:05:42, exp. remaining 1:07:10, complete 7.83%
att-weights epoch 481, step 134, max_size:classes 48, max_size:data 1426, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.186 sec/step, elapsed 0:05:43, exp. remaining 1:06:46, complete 7.89%
att-weights epoch 481, step 135, max_size:classes 45, max_size:data 1368, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.942 sec/step, elapsed 0:05:44, exp. remaining 1:06:20, complete 7.96%
att-weights epoch 481, step 136, max_size:classes 48, max_size:data 1527, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.138 sec/step, elapsed 0:05:45, exp. remaining 1:05:56, complete 8.03%
att-weights epoch 481, step 137, max_size:classes 51, max_size:data 1157, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.064 sec/step, elapsed 0:05:46, exp. remaining 1:05:32, complete 8.10%
att-weights epoch 481, step 138, max_size:classes 45, max_size:data 1407, mem_usage:GPU:0 0.9GB, num_seqs 2, 3.305 sec/step, elapsed 0:05:49, exp. remaining 1:05:34, complete 8.17%
att-weights epoch 481, step 139, max_size:classes 47, max_size:data 1681, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.370 sec/step, elapsed 0:05:51, exp. remaining 1:05:14, complete 8.23%
att-weights epoch 481, step 140, max_size:classes 49, max_size:data 1462, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.422 sec/step, elapsed 0:05:52, exp. remaining 1:04:37, complete 8.34%
att-weights epoch 481, step 141, max_size:classes 52, max_size:data 1474, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.283 sec/step, elapsed 0:05:53, exp. remaining 1:04:17, complete 8.40%
att-weights epoch 481, step 142, max_size:classes 50, max_size:data 1672, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.407 sec/step, elapsed 0:05:55, exp. remaining 1:03:58, complete 8.47%
att-weights epoch 481, step 143, max_size:classes 45, max_size:data 1631, mem_usage:GPU:0 0.9GB, num_seqs 2, 14.368 sec/step, elapsed 0:06:09, exp. remaining 1:05:59, complete 8.54%
att-weights epoch 481, step 144, max_size:classes 46, max_size:data 1300, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.223 sec/step, elapsed 0:06:10, exp. remaining 1:05:37, complete 8.61%
att-weights epoch 481, step 145, max_size:classes 46, max_size:data 1694, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.259 sec/step, elapsed 0:06:12, exp. remaining 1:05:00, complete 8.71%
att-weights epoch 481, step 146, max_size:classes 50, max_size:data 1727, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.021 sec/step, elapsed 0:06:13, exp. remaining 1:04:38, complete 8.78%
att-weights epoch 481, step 147, max_size:classes 50, max_size:data 1384, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.094 sec/step, elapsed 0:06:14, exp. remaining 1:04:16, complete 8.85%
att-weights epoch 481, step 148, max_size:classes 41, max_size:data 1592, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.186 sec/step, elapsed 0:06:15, exp. remaining 1:03:56, complete 8.91%
att-weights epoch 481, step 149, max_size:classes 47, max_size:data 1314, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.638 sec/step, elapsed 0:06:17, exp. remaining 1:03:41, complete 8.98%
att-weights epoch 481, step 150, max_size:classes 40, max_size:data 1227, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.911 sec/step, elapsed 0:06:18, exp. remaining 1:03:18, complete 9.05%
att-weights epoch 481, step 151, max_size:classes 47, max_size:data 1693, mem_usage:GPU:0 0.9GB, num_seqs 2, 11.048 sec/step, elapsed 0:06:29, exp. remaining 1:04:21, complete 9.15%
att-weights epoch 481, step 152, max_size:classes 50, max_size:data 1357, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.463 sec/step, elapsed 0:06:30, exp. remaining 1:04:04, complete 9.22%
att-weights epoch 481, step 153, max_size:classes 42, max_size:data 1406, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.019 sec/step, elapsed 0:06:31, exp. remaining 1:03:28, complete 9.32%
att-weights epoch 481, step 154, max_size:classes 40, max_size:data 1263, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.665 sec/step, elapsed 0:06:33, exp. remaining 1:03:13, complete 9.39%
att-weights epoch 481, step 155, max_size:classes 41, max_size:data 1413, mem_usage:GPU:0 0.9GB, num_seqs 2, 7.781 sec/step, elapsed 0:06:41, exp. remaining 1:03:58, complete 9.46%
att-weights epoch 481, step 156, max_size:classes 44, max_size:data 1729, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.522 sec/step, elapsed 0:06:42, exp. remaining 1:03:42, complete 9.53%
att-weights epoch 481, step 157, max_size:classes 47, max_size:data 1353, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.308 sec/step, elapsed 0:06:43, exp. remaining 1:03:24, complete 9.60%
att-weights epoch 481, step 158, max_size:classes 46, max_size:data 1807, mem_usage:GPU:0 0.9GB, num_seqs 2, 6.645 sec/step, elapsed 0:06:50, exp. remaining 1:03:57, complete 9.66%
att-weights epoch 481, step 159, max_size:classes 46, max_size:data 1480, mem_usage:GPU:0 0.9GB, num_seqs 2, 2.029 sec/step, elapsed 0:06:52, exp. remaining 1:03:46, complete 9.73%
att-weights epoch 481, step 160, max_size:classes 48, max_size:data 1285, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.637 sec/step, elapsed 0:06:54, exp. remaining 1:03:32, complete 9.80%
att-weights epoch 481, step 161, max_size:classes 42, max_size:data 1387, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.272 sec/step, elapsed 0:06:55, exp. remaining 1:03:14, complete 9.87%
att-weights epoch 481, step 162, max_size:classes 41, max_size:data 1326, mem_usage:GPU:0 0.9GB, num_seqs 3, 6.477 sec/step, elapsed 0:07:01, exp. remaining 1:03:44, complete 9.94%
att-weights epoch 481, step 163, max_size:classes 44, max_size:data 1361, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.245 sec/step, elapsed 0:07:03, exp. remaining 1:03:26, complete 10.00%
att-weights epoch 481, step 164, max_size:classes 44, max_size:data 1354, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.000 sec/step, elapsed 0:07:04, exp. remaining 1:02:53, complete 10.11%
att-weights epoch 481, step 165, max_size:classes 45, max_size:data 1700, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.214 sec/step, elapsed 0:07:05, exp. remaining 1:02:35, complete 10.17%
att-weights epoch 481, step 166, max_size:classes 47, max_size:data 1425, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.167 sec/step, elapsed 0:07:06, exp. remaining 1:02:18, complete 10.24%
att-weights epoch 481, step 167, max_size:classes 50, max_size:data 1689, mem_usage:GPU:0 0.9GB, num_seqs 2, 5.344 sec/step, elapsed 0:07:11, exp. remaining 1:02:23, complete 10.34%
att-weights epoch 481, step 168, max_size:classes 41, max_size:data 1363, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.113 sec/step, elapsed 0:07:12, exp. remaining 1:01:52, complete 10.45%
att-weights epoch 481, step 169, max_size:classes 55, max_size:data 1361, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.068 sec/step, elapsed 0:07:14, exp. remaining 1:01:34, complete 10.51%
att-weights epoch 481, step 170, max_size:classes 37, max_size:data 1332, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.020 sec/step, elapsed 0:07:15, exp. remaining 1:01:16, complete 10.58%
att-weights epoch 481, step 171, max_size:classes 41, max_size:data 1590, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.157 sec/step, elapsed 0:07:16, exp. remaining 1:00:59, complete 10.65%
att-weights epoch 481, step 172, max_size:classes 44, max_size:data 1429, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.273 sec/step, elapsed 0:07:17, exp. remaining 1:00:31, complete 10.75%
att-weights epoch 481, step 173, max_size:classes 52, max_size:data 1322, mem_usage:GPU:0 0.9GB, num_seqs 3, 5.659 sec/step, elapsed 0:07:23, exp. remaining 1:00:39, complete 10.85%
att-weights epoch 481, step 174, max_size:classes 45, max_size:data 1114, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.050 sec/step, elapsed 0:07:24, exp. remaining 1:00:10, complete 10.96%
att-weights epoch 481, step 175, max_size:classes 40, max_size:data 1727, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.343 sec/step, elapsed 0:07:25, exp. remaining 0:59:56, complete 11.02%
att-weights epoch 481, step 176, max_size:classes 45, max_size:data 1307, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.339 sec/step, elapsed 0:07:26, exp. remaining 0:59:29, complete 11.13%
att-weights epoch 481, step 177, max_size:classes 42, max_size:data 1297, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.223 sec/step, elapsed 0:07:28, exp. remaining 0:59:02, complete 11.23%
att-weights epoch 481, step 178, max_size:classes 42, max_size:data 1575, mem_usage:GPU:0 0.9GB, num_seqs 2, 3.968 sec/step, elapsed 0:07:32, exp. remaining 0:59:10, complete 11.30%
att-weights epoch 481, step 179, max_size:classes 46, max_size:data 1223, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.981 sec/step, elapsed 0:07:33, exp. remaining 0:58:53, complete 11.36%
att-weights epoch 481, step 180, max_size:classes 46, max_size:data 1538, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.298 sec/step, elapsed 0:07:34, exp. remaining 0:58:40, complete 11.43%
att-weights epoch 481, step 181, max_size:classes 41, max_size:data 1182, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.140 sec/step, elapsed 0:07:35, exp. remaining 0:58:25, complete 11.50%
att-weights epoch 481, step 182, max_size:classes 42, max_size:data 1135, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.202 sec/step, elapsed 0:07:36, exp. remaining 0:57:59, complete 11.60%
att-weights epoch 481, step 183, max_size:classes 45, max_size:data 1333, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.355 sec/step, elapsed 0:07:38, exp. remaining 0:57:46, complete 11.67%
att-weights epoch 481, step 184, max_size:classes 49, max_size:data 1580, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.067 sec/step, elapsed 0:07:39, exp. remaining 0:57:32, complete 11.74%
att-weights epoch 481, step 185, max_size:classes 42, max_size:data 1126, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.291 sec/step, elapsed 0:07:40, exp. remaining 0:57:19, complete 11.81%
att-weights epoch 481, step 186, max_size:classes 42, max_size:data 1170, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.460 sec/step, elapsed 0:07:41, exp. remaining 0:57:07, complete 11.87%
att-weights epoch 481, step 187, max_size:classes 38, max_size:data 1559, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.986 sec/step, elapsed 0:07:42, exp. remaining 0:56:41, complete 11.98%
att-weights epoch 481, step 188, max_size:classes 40, max_size:data 1344, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.873 sec/step, elapsed 0:07:43, exp. remaining 0:56:15, complete 12.08%
att-weights epoch 481, step 189, max_size:classes 41, max_size:data 1310, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.838 sec/step, elapsed 0:07:44, exp. remaining 0:56:00, complete 12.15%
att-weights epoch 481, step 190, max_size:classes 40, max_size:data 1655, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.078 sec/step, elapsed 0:07:45, exp. remaining 0:55:46, complete 12.22%
att-weights epoch 481, step 191, max_size:classes 42, max_size:data 1259, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.234 sec/step, elapsed 0:07:46, exp. remaining 0:55:34, complete 12.28%
att-weights epoch 481, step 192, max_size:classes 39, max_size:data 1455, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.076 sec/step, elapsed 0:07:47, exp. remaining 0:55:20, complete 12.35%
att-weights epoch 481, step 193, max_size:classes 39, max_size:data 1109, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.124 sec/step, elapsed 0:07:49, exp. remaining 0:55:08, complete 12.42%
att-weights epoch 481, step 194, max_size:classes 44, max_size:data 1406, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.196 sec/step, elapsed 0:07:50, exp. remaining 0:54:45, complete 12.52%
att-weights epoch 481, step 195, max_size:classes 40, max_size:data 1377, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.327 sec/step, elapsed 0:07:51, exp. remaining 0:54:24, complete 12.62%
att-weights epoch 481, step 196, max_size:classes 40, max_size:data 1249, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.563 sec/step, elapsed 0:07:53, exp. remaining 0:54:05, complete 12.73%
att-weights epoch 481, step 197, max_size:classes 42, max_size:data 1316, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.453 sec/step, elapsed 0:07:54, exp. remaining 0:53:45, complete 12.83%
att-weights epoch 481, step 198, max_size:classes 42, max_size:data 1311, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.209 sec/step, elapsed 0:07:55, exp. remaining 0:53:24, complete 12.93%
att-weights epoch 481, step 199, max_size:classes 38, max_size:data 1509, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.240 sec/step, elapsed 0:07:57, exp. remaining 0:53:03, complete 13.03%
att-weights epoch 481, step 200, max_size:classes 35, max_size:data 1361, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.119 sec/step, elapsed 0:07:58, exp. remaining 0:52:42, complete 13.13%
att-weights epoch 481, step 201, max_size:classes 40, max_size:data 1408, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.258 sec/step, elapsed 0:07:59, exp. remaining 0:52:23, complete 13.24%
att-weights epoch 481, step 202, max_size:classes 42, max_size:data 1378, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.109 sec/step, elapsed 0:08:00, exp. remaining 0:52:02, complete 13.34%
att-weights epoch 481, step 203, max_size:classes 38, max_size:data 1314, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.385 sec/step, elapsed 0:08:01, exp. remaining 0:51:44, complete 13.44%
att-weights epoch 481, step 204, max_size:classes 36, max_size:data 1184, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.707 sec/step, elapsed 0:08:04, exp. remaining 0:51:34, complete 13.54%
att-weights epoch 481, step 205, max_size:classes 35, max_size:data 1293, mem_usage:GPU:0 0.9GB, num_seqs 3, 5.045 sec/step, elapsed 0:08:09, exp. remaining 0:51:39, complete 13.64%
att-weights epoch 481, step 206, max_size:classes 38, max_size:data 1163, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.753 sec/step, elapsed 0:08:11, exp. remaining 0:51:23, complete 13.75%
att-weights epoch 481, step 207, max_size:classes 47, max_size:data 1236, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.370 sec/step, elapsed 0:08:12, exp. remaining 0:51:06, complete 13.85%
att-weights epoch 481, step 208, max_size:classes 39, max_size:data 1167, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.079 sec/step, elapsed 0:08:13, exp. remaining 0:50:46, complete 13.95%
att-weights epoch 481, step 209, max_size:classes 39, max_size:data 1145, mem_usage:GPU:0 0.9GB, num_seqs 3, 4.350 sec/step, elapsed 0:08:18, exp. remaining 0:50:56, complete 14.02%
att-weights epoch 481, step 210, max_size:classes 38, max_size:data 1074, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.098 sec/step, elapsed 0:08:19, exp. remaining 0:50:45, complete 14.09%
att-weights epoch 481, step 211, max_size:classes 39, max_size:data 1304, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.166 sec/step, elapsed 0:08:20, exp. remaining 0:50:27, complete 14.19%
att-weights epoch 481, step 212, max_size:classes 43, max_size:data 1169, mem_usage:GPU:0 0.9GB, num_seqs 3, 4.880 sec/step, elapsed 0:08:25, exp. remaining 0:50:39, complete 14.26%
att-weights epoch 481, step 213, max_size:classes 35, max_size:data 1014, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.386 sec/step, elapsed 0:08:26, exp. remaining 0:50:31, complete 14.32%
att-weights epoch 481, step 214, max_size:classes 41, max_size:data 1244, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.334 sec/step, elapsed 0:08:28, exp. remaining 0:50:14, complete 14.43%
att-weights epoch 481, step 215, max_size:classes 34, max_size:data 1162, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.233 sec/step, elapsed 0:08:29, exp. remaining 0:49:56, complete 14.53%
att-weights epoch 481, step 216, max_size:classes 36, max_size:data 1233, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.309 sec/step, elapsed 0:08:30, exp. remaining 0:49:39, complete 14.63%
att-weights epoch 481, step 217, max_size:classes 36, max_size:data 1226, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.930 sec/step, elapsed 0:08:33, exp. remaining 0:49:40, complete 14.70%
att-weights epoch 481, step 218, max_size:classes 43, max_size:data 1368, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.232 sec/step, elapsed 0:08:34, exp. remaining 0:49:31, complete 14.77%
att-weights epoch 481, step 219, max_size:classes 42, max_size:data 1496, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.206 sec/step, elapsed 0:08:36, exp. remaining 0:49:14, complete 14.87%
att-weights epoch 481, step 220, max_size:classes 42, max_size:data 993, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.367 sec/step, elapsed 0:08:37, exp. remaining 0:48:58, complete 14.97%
att-weights epoch 481, step 221, max_size:classes 38, max_size:data 1313, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.151 sec/step, elapsed 0:08:38, exp. remaining 0:48:41, complete 15.07%
att-weights epoch 481, step 222, max_size:classes 37, max_size:data 1372, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.951 sec/step, elapsed 0:08:39, exp. remaining 0:48:23, complete 15.18%
att-weights epoch 481, step 223, max_size:classes 40, max_size:data 1028, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.920 sec/step, elapsed 0:08:42, exp. remaining 0:48:17, complete 15.28%
att-weights epoch 481, step 224, max_size:classes 36, max_size:data 1208, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.199 sec/step, elapsed 0:08:43, exp. remaining 0:48:08, complete 15.35%
att-weights epoch 481, step 225, max_size:classes 36, max_size:data 1169, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.430 sec/step, elapsed 0:08:45, exp. remaining 0:47:54, complete 15.45%
att-weights epoch 481, step 226, max_size:classes 37, max_size:data 1149, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.142 sec/step, elapsed 0:08:46, exp. remaining 0:47:45, complete 15.52%
att-weights epoch 481, step 227, max_size:classes 37, max_size:data 1361, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.100 sec/step, elapsed 0:08:47, exp. remaining 0:47:29, complete 15.62%
att-weights epoch 481, step 228, max_size:classes 36, max_size:data 1053, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.128 sec/step, elapsed 0:08:48, exp. remaining 0:47:05, complete 15.75%
att-weights epoch 481, step 229, max_size:classes 35, max_size:data 1023, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.196 sec/step, elapsed 0:08:49, exp. remaining 0:46:50, complete 15.86%
att-weights epoch 481, step 230, max_size:classes 37, max_size:data 1057, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.346 sec/step, elapsed 0:08:50, exp. remaining 0:46:36, complete 15.96%
att-weights epoch 481, step 231, max_size:classes 35, max_size:data 1209, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.394 sec/step, elapsed 0:08:52, exp. remaining 0:46:22, complete 16.06%
att-weights epoch 481, step 232, max_size:classes 34, max_size:data 1023, mem_usage:GPU:0 0.9GB, num_seqs 3, 3.309 sec/step, elapsed 0:08:55, exp. remaining 0:46:11, complete 16.20%
att-weights epoch 481, step 233, max_size:classes 39, max_size:data 1542, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.262 sec/step, elapsed 0:08:56, exp. remaining 0:45:57, complete 16.30%
att-weights epoch 481, step 234, max_size:classes 34, max_size:data 1155, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.457 sec/step, elapsed 0:08:58, exp. remaining 0:45:44, complete 16.40%
att-weights epoch 481, step 235, max_size:classes 31, max_size:data 1425, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.205 sec/step, elapsed 0:08:59, exp. remaining 0:45:30, complete 16.50%
att-weights epoch 481, step 236, max_size:classes 35, max_size:data 1073, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.240 sec/step, elapsed 0:09:00, exp. remaining 0:45:09, complete 16.64%
att-weights epoch 481, step 237, max_size:classes 35, max_size:data 992, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.674 sec/step, elapsed 0:09:02, exp. remaining 0:44:58, complete 16.74%
att-weights epoch 481, step 238, max_size:classes 34, max_size:data 1110, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.284 sec/step, elapsed 0:09:03, exp. remaining 0:44:51, complete 16.81%
att-weights epoch 481, step 239, max_size:classes 36, max_size:data 813, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.200 sec/step, elapsed 0:09:05, exp. remaining 0:44:31, complete 16.94%
att-weights epoch 481, step 240, max_size:classes 37, max_size:data 1236, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.410 sec/step, elapsed 0:09:06, exp. remaining 0:44:19, complete 17.05%
att-weights epoch 481, step 241, max_size:classes 37, max_size:data 941, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.247 sec/step, elapsed 0:09:07, exp. remaining 0:44:06, complete 17.15%
att-weights epoch 481, step 242, max_size:classes 35, max_size:data 1091, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.236 sec/step, elapsed 0:09:08, exp. remaining 0:43:53, complete 17.25%
att-weights epoch 481, step 243, max_size:classes 35, max_size:data 1181, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.180 sec/step, elapsed 0:09:10, exp. remaining 0:43:46, complete 17.32%
att-weights epoch 481, step 244, max_size:classes 34, max_size:data 1062, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.371 sec/step, elapsed 0:09:11, exp. remaining 0:43:34, complete 17.42%
att-weights epoch 481, step 245, max_size:classes 39, max_size:data 967, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.387 sec/step, elapsed 0:09:12, exp. remaining 0:43:22, complete 17.52%
att-weights epoch 481, step 246, max_size:classes 33, max_size:data 1087, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.353 sec/step, elapsed 0:09:14, exp. remaining 0:43:10, complete 17.63%
att-weights epoch 481, step 247, max_size:classes 35, max_size:data 1351, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.378 sec/step, elapsed 0:09:15, exp. remaining 0:43:04, complete 17.69%
att-weights epoch 481, step 248, max_size:classes 43, max_size:data 955, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.918 sec/step, elapsed 0:09:17, exp. remaining 0:42:55, complete 17.80%
att-weights epoch 481, step 249, max_size:classes 35, max_size:data 885, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.499 sec/step, elapsed 0:09:19, exp. remaining 0:42:44, complete 17.90%
att-weights epoch 481, step 250, max_size:classes 30, max_size:data 1070, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.531 sec/step, elapsed 0:09:20, exp. remaining 0:42:33, complete 18.00%
att-weights epoch 481, step 251, max_size:classes 36, max_size:data 1093, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.366 sec/step, elapsed 0:09:21, exp. remaining 0:42:22, complete 18.10%
att-weights epoch 481, step 252, max_size:classes 35, max_size:data 1501, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.519 sec/step, elapsed 0:09:23, exp. remaining 0:42:11, complete 18.20%
att-weights epoch 481, step 253, max_size:classes 33, max_size:data 1073, mem_usage:GPU:0 0.9GB, num_seqs 3, 11.654 sec/step, elapsed 0:09:35, exp. remaining 0:42:46, complete 18.31%
att-weights epoch 481, step 254, max_size:classes 34, max_size:data 1194, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.542 sec/step, elapsed 0:09:36, exp. remaining 0:42:35, complete 18.41%
att-weights epoch 481, step 255, max_size:classes 34, max_size:data 1070, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.890 sec/step, elapsed 0:09:38, exp. remaining 0:42:26, complete 18.51%
att-weights epoch 481, step 256, max_size:classes 40, max_size:data 1349, mem_usage:GPU:0 0.9GB, num_seqs 2, 1.331 sec/step, elapsed 0:09:39, exp. remaining 0:42:15, complete 18.61%
att-weights epoch 481, step 257, max_size:classes 34, max_size:data 1244, mem_usage:GPU:0 0.9GB, num_seqs 3, 9.339 sec/step, elapsed 0:09:49, exp. remaining 0:42:39, complete 18.71%
att-weights epoch 481, step 258, max_size:classes 32, max_size:data 1103, mem_usage:GPU:0 0.9GB, num_seqs 3, 4.907 sec/step, elapsed 0:09:54, exp. remaining 0:42:37, complete 18.85%
att-weights epoch 481, step 259, max_size:classes 37, max_size:data 892, mem_usage:GPU:0 0.9GB, num_seqs 3, 5.918 sec/step, elapsed 0:10:00, exp. remaining 0:42:40, complete 18.99%
att-weights epoch 481, step 260, max_size:classes 35, max_size:data 1179, mem_usage:GPU:0 0.9GB, num_seqs 3, 3.001 sec/step, elapsed 0:10:03, exp. remaining 0:42:36, complete 19.09%
att-weights epoch 481, step 261, max_size:classes 34, max_size:data 1249, mem_usage:GPU:0 0.9GB, num_seqs 3, 4.707 sec/step, elapsed 0:10:07, exp. remaining 0:42:39, complete 19.19%
att-weights epoch 481, step 262, max_size:classes 33, max_size:data 1049, mem_usage:GPU:0 0.9GB, num_seqs 3, 3.743 sec/step, elapsed 0:10:11, exp. remaining 0:42:38, complete 19.29%
att-weights epoch 481, step 263, max_size:classes 37, max_size:data 1002, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.306 sec/step, elapsed 0:10:12, exp. remaining 0:42:27, complete 19.39%
att-weights epoch 481, step 264, max_size:classes 36, max_size:data 1111, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.490 sec/step, elapsed 0:10:14, exp. remaining 0:42:11, complete 19.53%
att-weights epoch 481, step 265, max_size:classes 31, max_size:data 813, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.244 sec/step, elapsed 0:10:15, exp. remaining 0:42:00, complete 19.63%
att-weights epoch 481, step 266, max_size:classes 34, max_size:data 1082, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.403 sec/step, elapsed 0:10:17, exp. remaining 0:41:49, complete 19.73%
att-weights epoch 481, step 267, max_size:classes 38, max_size:data 921, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.061 sec/step, elapsed 0:10:19, exp. remaining 0:41:41, complete 19.84%
att-weights epoch 481, step 268, max_size:classes 31, max_size:data 973, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.034 sec/step, elapsed 0:10:21, exp. remaining 0:41:34, complete 19.94%
att-weights epoch 481, step 269, max_size:classes 36, max_size:data 1282, mem_usage:GPU:0 0.9GB, num_seqs 3, 3.491 sec/step, elapsed 0:10:24, exp. remaining 0:41:26, complete 20.07%
att-weights epoch 481, step 270, max_size:classes 34, max_size:data 1149, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.538 sec/step, elapsed 0:10:26, exp. remaining 0:41:11, complete 20.21%
att-weights epoch 481, step 271, max_size:classes 33, max_size:data 1180, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.685 sec/step, elapsed 0:10:27, exp. remaining 0:41:02, complete 20.31%
att-weights epoch 481, step 272, max_size:classes 37, max_size:data 1031, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.601 sec/step, elapsed 0:10:29, exp. remaining 0:40:53, complete 20.42%
att-weights epoch 481, step 273, max_size:classes 32, max_size:data 995, mem_usage:GPU:0 0.9GB, num_seqs 4, 4.154 sec/step, elapsed 0:10:33, exp. remaining 0:40:49, complete 20.55%
att-weights epoch 481, step 274, max_size:classes 30, max_size:data 1052, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.591 sec/step, elapsed 0:10:35, exp. remaining 0:40:40, complete 20.65%
att-weights epoch 481, step 275, max_size:classes 32, max_size:data 1096, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.515 sec/step, elapsed 0:10:36, exp. remaining 0:40:25, complete 20.79%
att-weights epoch 481, step 276, max_size:classes 31, max_size:data 954, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.875 sec/step, elapsed 0:10:38, exp. remaining 0:40:18, complete 20.89%
att-weights epoch 481, step 277, max_size:classes 33, max_size:data 1193, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.423 sec/step, elapsed 0:10:40, exp. remaining 0:40:03, complete 21.03%
att-weights epoch 481, step 278, max_size:classes 30, max_size:data 869, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.610 sec/step, elapsed 0:10:42, exp. remaining 0:39:53, complete 21.16%
att-weights epoch 481, step 279, max_size:classes 34, max_size:data 946, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.640 sec/step, elapsed 0:10:44, exp. remaining 0:39:50, complete 21.23%
att-weights epoch 481, step 280, max_size:classes 31, max_size:data 1039, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.483 sec/step, elapsed 0:10:45, exp. remaining 0:39:41, complete 21.33%
att-weights epoch 481, step 281, max_size:classes 36, max_size:data 1290, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.442 sec/step, elapsed 0:10:47, exp. remaining 0:39:31, complete 21.44%
att-weights epoch 481, step 282, max_size:classes 32, max_size:data 969, mem_usage:GPU:0 0.9GB, num_seqs 4, 6.737 sec/step, elapsed 0:10:53, exp. remaining 0:39:37, complete 21.57%
att-weights epoch 481, step 283, max_size:classes 30, max_size:data 1056, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.392 sec/step, elapsed 0:10:55, exp. remaining 0:39:28, complete 21.67%
att-weights epoch 481, step 284, max_size:classes 30, max_size:data 1000, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.903 sec/step, elapsed 0:10:57, exp. remaining 0:39:16, complete 21.81%
att-weights epoch 481, step 285, max_size:classes 32, max_size:data 1021, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.476 sec/step, elapsed 0:10:58, exp. remaining 0:39:07, complete 21.91%
att-weights epoch 481, step 286, max_size:classes 37, max_size:data 953, mem_usage:GPU:0 0.9GB, num_seqs 4, 3.156 sec/step, elapsed 0:11:01, exp. remaining 0:38:59, complete 22.05%
att-weights epoch 481, step 287, max_size:classes 36, max_size:data 930, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.565 sec/step, elapsed 0:11:03, exp. remaining 0:38:47, complete 22.18%
att-weights epoch 481, step 288, max_size:classes 35, max_size:data 1458, mem_usage:GPU:0 0.9GB, num_seqs 2, 0.949 sec/step, elapsed 0:11:04, exp. remaining 0:38:36, complete 22.29%
att-weights epoch 481, step 289, max_size:classes 31, max_size:data 1017, mem_usage:GPU:0 0.9GB, num_seqs 3, 7.804 sec/step, elapsed 0:11:12, exp. remaining 0:38:50, complete 22.39%
att-weights epoch 481, step 290, max_size:classes 31, max_size:data 1164, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.586 sec/step, elapsed 0:11:13, exp. remaining 0:38:41, complete 22.49%
att-weights epoch 481, step 291, max_size:classes 30, max_size:data 957, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.479 sec/step, elapsed 0:11:15, exp. remaining 0:38:29, complete 22.63%
att-weights epoch 481, step 292, max_size:classes 33, max_size:data 1015, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.187 sec/step, elapsed 0:11:16, exp. remaining 0:38:15, complete 22.76%
att-weights epoch 481, step 293, max_size:classes 31, max_size:data 997, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.149 sec/step, elapsed 0:11:18, exp. remaining 0:38:09, complete 22.86%
att-weights epoch 481, step 294, max_size:classes 30, max_size:data 1153, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.495 sec/step, elapsed 0:11:20, exp. remaining 0:38:01, complete 22.97%
att-weights epoch 481, step 295, max_size:classes 34, max_size:data 926, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.391 sec/step, elapsed 0:11:21, exp. remaining 0:37:48, complete 23.10%
att-weights epoch 481, step 296, max_size:classes 35, max_size:data 979, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.797 sec/step, elapsed 0:11:23, exp. remaining 0:37:36, complete 23.24%
att-weights epoch 481, step 297, max_size:classes 31, max_size:data 849, mem_usage:GPU:0 0.9GB, num_seqs 3, 4.069 sec/step, elapsed 0:11:27, exp. remaining 0:37:37, complete 23.34%
att-weights epoch 481, step 298, max_size:classes 29, max_size:data 1096, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.235 sec/step, elapsed 0:11:28, exp. remaining 0:37:28, complete 23.44%
att-weights epoch 481, step 299, max_size:classes 35, max_size:data 1121, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.449 sec/step, elapsed 0:11:30, exp. remaining 0:37:20, complete 23.55%
att-weights epoch 481, step 300, max_size:classes 32, max_size:data 962, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.560 sec/step, elapsed 0:11:31, exp. remaining 0:37:12, complete 23.65%
att-weights epoch 481, step 301, max_size:classes 36, max_size:data 998, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.951 sec/step, elapsed 0:11:33, exp. remaining 0:37:02, complete 23.78%
att-weights epoch 481, step 302, max_size:classes 30, max_size:data 1071, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.392 sec/step, elapsed 0:11:34, exp. remaining 0:36:50, complete 23.92%
att-weights epoch 481, step 303, max_size:classes 28, max_size:data 1084, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.449 sec/step, elapsed 0:11:36, exp. remaining 0:36:38, complete 24.06%
att-weights epoch 481, step 304, max_size:classes 30, max_size:data 777, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.314 sec/step, elapsed 0:11:37, exp. remaining 0:36:26, complete 24.19%
att-weights epoch 481, step 305, max_size:classes 30, max_size:data 863, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.453 sec/step, elapsed 0:11:39, exp. remaining 0:36:18, complete 24.29%
att-weights epoch 481, step 306, max_size:classes 29, max_size:data 1031, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.359 sec/step, elapsed 0:11:40, exp. remaining 0:36:10, complete 24.40%
att-weights epoch 481, step 307, max_size:classes 29, max_size:data 701, mem_usage:GPU:0 0.9GB, num_seqs 3, 2.878 sec/step, elapsed 0:11:43, exp. remaining 0:36:07, complete 24.50%
att-weights epoch 481, step 308, max_size:classes 28, max_size:data 1025, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.872 sec/step, elapsed 0:11:45, exp. remaining 0:35:57, complete 24.63%
att-weights epoch 481, step 309, max_size:classes 41, max_size:data 1070, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.687 sec/step, elapsed 0:11:46, exp. remaining 0:35:50, complete 24.74%
att-weights epoch 481, step 310, max_size:classes 27, max_size:data 871, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.030 sec/step, elapsed 0:11:48, exp. remaining 0:35:45, complete 24.84%
att-weights epoch 481, step 311, max_size:classes 30, max_size:data 836, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.483 sec/step, elapsed 0:11:50, exp. remaining 0:35:38, complete 24.94%
att-weights epoch 481, step 312, max_size:classes 32, max_size:data 892, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.277 sec/step, elapsed 0:11:52, exp. remaining 0:35:29, complete 25.08%
att-weights epoch 481, step 313, max_size:classes 33, max_size:data 981, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.338 sec/step, elapsed 0:11:54, exp. remaining 0:35:21, complete 25.18%
att-weights epoch 481, step 314, max_size:classes 32, max_size:data 1045, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.233 sec/step, elapsed 0:11:55, exp. remaining 0:35:14, complete 25.28%
att-weights epoch 481, step 315, max_size:classes 31, max_size:data 917, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.035 sec/step, elapsed 0:11:56, exp. remaining 0:35:05, complete 25.38%
att-weights epoch 481, step 316, max_size:classes 32, max_size:data 1086, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.461 sec/step, elapsed 0:11:57, exp. remaining 0:34:58, complete 25.48%
att-weights epoch 481, step 317, max_size:classes 32, max_size:data 904, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.617 sec/step, elapsed 0:11:59, exp. remaining 0:34:52, complete 25.59%
att-weights epoch 481, step 318, max_size:classes 31, max_size:data 1076, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.401 sec/step, elapsed 0:12:00, exp. remaining 0:34:41, complete 25.72%
att-weights epoch 481, step 319, max_size:classes 28, max_size:data 1303, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.556 sec/step, elapsed 0:12:02, exp. remaining 0:34:31, complete 25.86%
att-weights epoch 481, step 320, max_size:classes 33, max_size:data 1099, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.247 sec/step, elapsed 0:12:03, exp. remaining 0:34:20, complete 26.00%
att-weights epoch 481, step 321, max_size:classes 31, max_size:data 930, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.593 sec/step, elapsed 0:12:05, exp. remaining 0:34:10, complete 26.13%
att-weights epoch 481, step 322, max_size:classes 31, max_size:data 1021, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.460 sec/step, elapsed 0:12:06, exp. remaining 0:34:03, complete 26.23%
att-weights epoch 481, step 323, max_size:classes 30, max_size:data 1012, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.446 sec/step, elapsed 0:12:08, exp. remaining 0:33:53, complete 26.37%
att-weights epoch 481, step 324, max_size:classes 28, max_size:data 827, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.184 sec/step, elapsed 0:12:09, exp. remaining 0:33:42, complete 26.51%
att-weights epoch 481, step 325, max_size:classes 30, max_size:data 1072, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.343 sec/step, elapsed 0:12:10, exp. remaining 0:33:31, complete 26.64%
att-weights epoch 481, step 326, max_size:classes 27, max_size:data 1066, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.736 sec/step, elapsed 0:12:12, exp. remaining 0:33:26, complete 26.74%
att-weights epoch 481, step 327, max_size:classes 28, max_size:data 998, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.574 sec/step, elapsed 0:12:13, exp. remaining 0:33:16, complete 26.88%
att-weights epoch 481, step 328, max_size:classes 28, max_size:data 874, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.763 sec/step, elapsed 0:12:15, exp. remaining 0:33:04, complete 27.05%
att-weights epoch 481, step 329, max_size:classes 30, max_size:data 998, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.956 sec/step, elapsed 0:12:17, exp. remaining 0:32:55, complete 27.19%
att-weights epoch 481, step 330, max_size:classes 30, max_size:data 902, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.793 sec/step, elapsed 0:12:19, exp. remaining 0:32:43, complete 27.36%
att-weights epoch 481, step 331, max_size:classes 31, max_size:data 1085, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.380 sec/step, elapsed 0:12:20, exp. remaining 0:32:33, complete 27.49%
att-weights epoch 481, step 332, max_size:classes 28, max_size:data 949, mem_usage:GPU:0 0.9GB, num_seqs 4, 3.400 sec/step, elapsed 0:12:24, exp. remaining 0:32:29, complete 27.63%
att-weights epoch 481, step 333, max_size:classes 25, max_size:data 918, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.991 sec/step, elapsed 0:12:26, exp. remaining 0:32:24, complete 27.73%
att-weights epoch 481, step 334, max_size:classes 32, max_size:data 875, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.964 sec/step, elapsed 0:12:28, exp. remaining 0:32:16, complete 27.87%
att-weights epoch 481, step 335, max_size:classes 29, max_size:data 1096, mem_usage:GPU:0 0.9GB, num_seqs 3, 3.664 sec/step, elapsed 0:12:31, exp. remaining 0:32:13, complete 28.00%
att-weights epoch 481, step 336, max_size:classes 33, max_size:data 816, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.508 sec/step, elapsed 0:12:33, exp. remaining 0:32:04, complete 28.14%
att-weights epoch 481, step 337, max_size:classes 26, max_size:data 792, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.386 sec/step, elapsed 0:12:35, exp. remaining 0:31:57, complete 28.27%
att-weights epoch 481, step 338, max_size:classes 31, max_size:data 946, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.678 sec/step, elapsed 0:12:37, exp. remaining 0:31:48, complete 28.41%
att-weights epoch 481, step 339, max_size:classes 26, max_size:data 772, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.980 sec/step, elapsed 0:12:39, exp. remaining 0:31:40, complete 28.55%
att-weights epoch 481, step 340, max_size:classes 29, max_size:data 836, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.447 sec/step, elapsed 0:12:41, exp. remaining 0:31:37, complete 28.65%
att-weights epoch 481, step 341, max_size:classes 29, max_size:data 881, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.717 sec/step, elapsed 0:12:43, exp. remaining 0:31:29, complete 28.79%
att-weights epoch 481, step 342, max_size:classes 25, max_size:data 1075, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.844 sec/step, elapsed 0:12:45, exp. remaining 0:31:24, complete 28.89%
att-weights epoch 481, step 343, max_size:classes 28, max_size:data 870, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.690 sec/step, elapsed 0:12:47, exp. remaining 0:31:16, complete 29.02%
att-weights epoch 481, step 344, max_size:classes 26, max_size:data 864, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.579 sec/step, elapsed 0:12:48, exp. remaining 0:31:04, complete 29.19%
att-weights epoch 481, step 345, max_size:classes 27, max_size:data 857, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.522 sec/step, elapsed 0:12:50, exp. remaining 0:30:55, complete 29.33%
att-weights epoch 481, step 346, max_size:classes 29, max_size:data 967, mem_usage:GPU:0 0.9GB, num_seqs 4, 4.170 sec/step, elapsed 0:12:54, exp. remaining 0:30:53, complete 29.47%
att-weights epoch 481, step 347, max_size:classes 27, max_size:data 907, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.591 sec/step, elapsed 0:12:56, exp. remaining 0:30:45, complete 29.60%
att-weights epoch 481, step 348, max_size:classes 25, max_size:data 993, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.421 sec/step, elapsed 0:12:57, exp. remaining 0:30:33, complete 29.77%
att-weights epoch 481, step 349, max_size:classes 27, max_size:data 1174, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.172 sec/step, elapsed 0:12:58, exp. remaining 0:30:24, complete 29.91%
att-weights epoch 481, step 350, max_size:classes 26, max_size:data 963, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.742 sec/step, elapsed 0:13:00, exp. remaining 0:30:16, complete 30.04%
att-weights epoch 481, step 351, max_size:classes 25, max_size:data 1114, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.388 sec/step, elapsed 0:13:01, exp. remaining 0:30:08, complete 30.18%
att-weights epoch 481, step 352, max_size:classes 27, max_size:data 903, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.376 sec/step, elapsed 0:13:03, exp. remaining 0:30:00, complete 30.32%
att-weights epoch 481, step 353, max_size:classes 31, max_size:data 769, mem_usage:GPU:0 0.9GB, num_seqs 5, 3.029 sec/step, elapsed 0:13:06, exp. remaining 0:29:52, complete 30.49%
att-weights epoch 481, step 354, max_size:classes 34, max_size:data 937, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.800 sec/step, elapsed 0:13:07, exp. remaining 0:29:45, complete 30.62%
att-weights epoch 481, step 355, max_size:classes 30, max_size:data 992, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.198 sec/step, elapsed 0:13:10, exp. remaining 0:29:38, complete 30.76%
att-weights epoch 481, step 356, max_size:classes 24, max_size:data 869, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.833 sec/step, elapsed 0:13:11, exp. remaining 0:29:34, complete 30.86%
att-weights epoch 481, step 357, max_size:classes 27, max_size:data 760, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.843 sec/step, elapsed 0:13:13, exp. remaining 0:29:27, complete 31.00%
att-weights epoch 481, step 358, max_size:classes 27, max_size:data 905, mem_usage:GPU:0 0.9GB, num_seqs 4, 2.883 sec/step, elapsed 0:13:16, exp. remaining 0:29:19, complete 31.17%
att-weights epoch 481, step 359, max_size:classes 24, max_size:data 886, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.741 sec/step, elapsed 0:13:18, exp. remaining 0:29:12, complete 31.30%
att-weights epoch 481, step 360, max_size:classes 28, max_size:data 804, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.582 sec/step, elapsed 0:13:20, exp. remaining 0:29:01, complete 31.47%
att-weights epoch 481, step 361, max_size:classes 25, max_size:data 847, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.474 sec/step, elapsed 0:13:21, exp. remaining 0:28:54, complete 31.61%
att-weights epoch 481, step 362, max_size:classes 28, max_size:data 787, mem_usage:GPU:0 0.9GB, num_seqs 5, 2.451 sec/step, elapsed 0:13:23, exp. remaining 0:28:48, complete 31.75%
att-weights epoch 481, step 363, max_size:classes 27, max_size:data 941, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.531 sec/step, elapsed 0:13:25, exp. remaining 0:28:41, complete 31.88%
att-weights epoch 481, step 364, max_size:classes 26, max_size:data 680, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.931 sec/step, elapsed 0:13:27, exp. remaining 0:28:31, complete 32.05%
att-weights epoch 481, step 365, max_size:classes 27, max_size:data 1070, mem_usage:GPU:0 0.9GB, num_seqs 3, 1.422 sec/step, elapsed 0:13:28, exp. remaining 0:28:24, complete 32.19%
att-weights epoch 481, step 366, max_size:classes 25, max_size:data 869, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.746 sec/step, elapsed 0:13:30, exp. remaining 0:28:14, complete 32.36%
att-weights epoch 481, step 367, max_size:classes 23, max_size:data 739, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.964 sec/step, elapsed 0:13:32, exp. remaining 0:28:08, complete 32.49%
att-weights epoch 481, step 368, max_size:classes 31, max_size:data 836, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.474 sec/step, elapsed 0:13:34, exp. remaining 0:28:00, complete 32.63%
att-weights epoch 481, step 369, max_size:classes 24, max_size:data 645, mem_usage:GPU:0 0.9GB, num_seqs 5, 1.627 sec/step, elapsed 0:13:35, exp. remaining 0:27:51, complete 32.80%
att-weights epoch 481, step 370, max_size:classes 27, max_size:data 862, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.527 sec/step, elapsed 0:13:37, exp. remaining 0:27:43, complete 32.94%
att-weights epoch 481, step 371, max_size:classes 25, max_size:data 821, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.334 sec/step, elapsed 0:13:38, exp. remaining 0:27:36, complete 33.07%
att-weights epoch 481, step 372, max_size:classes 26, max_size:data 804, mem_usage:GPU:0 0.9GB, num_seqs 4, 1.343 sec/step, elapsed 0:13:39, exp. remaining 0:27:28, complete 33.21%
+------- EPILOGUE SCRIPT -----------------------------------------------
|
| Job ID ..............: 9505995
| Stopped at ..........: Tue Jul  2 12:28:52 CEST 2019
| Resources requested .: s_core=0,pxe=ubuntu_16.04,h_vmem=1536G,gpu=1,h_rt=7200,h_rss=4G,h_fsize=20G,scratch_free=5G,num_proc=5
| Resources used ......: cpu=00:23:19, mem=5376.38132 GB s, io=7.24953 GB, vmem=4.375G, maxvmem=4.392G, last_file_cache=128K, last_rss=3M, max-cache=4.000G
| Memory used .........: 4.000G / 4.000G (100.0%)
| Total time used .....: 0:15:44
|
+------- EPILOGUE SCRIPT -----------------------------------------------
