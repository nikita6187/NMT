+------- PROLOGUE SCRIPT -----------------------------------------------
|
| Job ID ...........: 9505913
| Started at .......: Tue Jul  2 11:39:22 CEST 2019
| Execution host ...: cluster-cn-246
| Cluster queue ....: 4-GPU-1080
| Script ...........: /var/spool/sge/cluster-cn-246/job_scripts/9505913
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
RETURNN starting up, version 20181022.053313--git-dfada43-dirty, date/time 2019-07-02-11-39-25 (UTC+0200), pid 29439, cwd /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc, Python /u/bahar/settings/python3-returnn-tf1.9/bin/python3
RETURNN config: /u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc.config
RETURNN command line options: ()
Hostname: cluster-cn-246
TensorFlow: 1.10.0 (v1.10.0-0-g656e7a2b34) (<site-package> in /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow)
Error while getting SGE num_proc: FileNotFoundError(2, "No such file or directory: 'qstat'")
Setup TF inter and intra global thread pools, num_threads None, session opts {'log_device_placement': False, 'device_count': {'GPU': 0}}.
CUDA_VISIBLE_DEVICES is set to '1'.
Collecting TensorFlow device list...
Local devices available to TensorFlow:
  1/2: name: "/device:CPU:0"
       device_type: "CPU"
       memory_limit: 268435456
       locality {
       }
       incarnation: 12745687730772755032
  2/2: name: "/device:GPU:0"
       device_type: "GPU"
       memory_limit: 10911236096
       locality {
         bus_id: 1
         links {
         }
       }
       incarnation: 11110695114506379742
       physical_device_desc: "device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1"
Using gpu device 1: GeForce GTX 1080 Ti
Device not set explicitly, and we found a GPU, which we will use.
Model file prefix: /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc/net-model/network
NOTE: We will use 'sorted_reverse' seq ordering.
Unhandled exception <class 'AssertionError'> in thread <_MainThread(MainThread, started 47746112007168)>, proc 29439.

Thread current, main, <_MainThread(MainThread, started 47746112007168)>:
(Excluded thread.)

That were all threads.
+------- EPILOGUE SCRIPT -----------------------------------------------
|
| Job ID ..............: 9505913
| Stopped at ..........: Tue Jul  2 11:39:56 CEST 2019
| Resources requested .: h_rss=4G,h_rt=7200,gpu=1,h_vmem=1536G,s_core=0,pxe=ubuntu_16.04,num_proc=5,scratch_free=5G,h_fsize=20G
| Resources used ......: cpu=00:00:33, mem=24.17117 GB s, io=0.02961 GB, vmem=3.000M, maxvmem=1.004G, last_file_cache=15M, last_rss=3M, max-cache=730M
| Memory used .........: 744M / 4.000G (18.2%)
| Total time used .....: 0:00:34
|
+------- EPILOGUE SCRIPT -----------------------------------------------
+------- PROLOGUE SCRIPT -----------------------------------------------
|
| Job ID ...........: 9505970
| Started at .......: Tue Jul  2 12:03:18 CEST 2019
| Execution host ...: cluster-cn-248
| Cluster queue ....: 4-GPU-1080
| Script ...........: /var/spool/sge/cluster-cn-248/job_scripts/9505970
| > #!/bin/bash
| > 
| > export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64:/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda/extras/CUPTI/lib64/:/usr/local/cuda-9.1/extras/CUPTI/lib64/:/usr/local/cuda-9.0/extras/CUPTI/lib64:/usr/lib/nvidia-418
| > #export LD_LIBRARY_PATH="/usr/local/lib::/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda-9.0/lib64:/usr/lib/nvidia-390:/opt/rbi/sge/lib/glinux:/opt/rbi/sge/lib/glinux"
| > 
| > source /u/bahar/settings/python3-returnn-tf1.9/bin/activate
| > 
| > python3 ~/returnn-parnia/tools/get-attention-weights.py /u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc.config --epoch 481 --data 'config:get_dataset("'$1'")' --dump_dir ~/makarov/NMT/hmm-factorization/experiments/asr2019/transformer/forward --layers "dec_02_att_weights" --layers "dec_01_att_weights" --layers "dec_03_att_weights" --layers "dec_04_att_weights" --layers "dec_05_att_weights" --layers "dec_06_att_weights" --layers "dec_07_att_weights" --layers "dec_08_att_weights" --layers "dec_09_att_weights" --layers "dec_10_att_weights" --layers "dec_11_att_weights" --layers "dec_12_att_weights" --rec_layer "output" --batch_size 600 --tf_log_dir /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/transformer/forward/tf_log_dir --reset_seq_ordering sorted_reverse
|
+------- PROLOGUE SCRIPT -----------------------------------------------
+------- EPILOGUE SCRIPT -----------------------------------------------
|
| Job ID ..............: 9505970
| Stopped at ..........: Tue Jul  2 12:03:20 CEST 2019
| Resources requested .: scratch_free=5G,h_fsize=20G,num_proc=5,h_vmem=1536G,s_core=0,pxe=ubuntu_16.04,h_rss=4G,h_rt=7200,gpu=1
| Resources used ......: cpu=00:00:00, mem=0.00000 GB s, io=0.00000 GB, vmem=N/A, maxvmem=N/A, last_file_cache=56K, last_rss=3M, max-cache=73M
| Memory used .........: 73M / 4.000G (1.8%)
| Total time used .....: 0:00:02
|
+------- EPILOGUE SCRIPT -----------------------------------------------
+------- PROLOGUE SCRIPT -----------------------------------------------
|
| Job ID ...........: 9505982
| Started at .......: Tue Jul  2 12:06:41 CEST 2019
| Execution host ...: cluster-cn-216
| Cluster queue ....: 3-GPU-1080
| Script ...........: /var/spool/sge/cluster-cn-216/job_scripts/9505982
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
RETURNN starting up, version 20181022.053313--git-dfada43-dirty, date/time 2019-07-02-12-06-46 (UTC+0200), pid 14440, cwd /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc, Python /u/bahar/settings/python3-returnn-tf1.9/bin/python3
RETURNN config: /u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc.config
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
       incarnation: 17133444942322553506
  2/2: name: "/device:GPU:0"
       device_type: "GPU"
       memory_limit: 10911236096
       locality {
         bus_id: 1
         links {
         }
       }
       incarnation: 10657903856011714228
       physical_device_desc: "device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1"
Using gpu device 0: GeForce GTX 1080 Ti
Device not set explicitly, and we found a GPU, which we will use.
Model file prefix: /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc/net-model/network
NOTE: We will use 'sorted_reverse' seq ordering.
Setup tf.Session with options {'device_count': {'GPU': 1}, 'log_device_placement': False} ...
layer root/'data' output: Data(name='data', shape=(None, 40))
layer root/'source' output: Data(name='source_output', shape=(None, 40))
layer root/'lstm0_fw' output: Data(name='lstm0_fw_output', shape=(None, 1024), batch_dim_axis=1)
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9505982.1.3-GPU-1080/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.cc -o /var/tmp/9505982.1.3-GPU-1080/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9505982.1.3-GPU-1080/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.cc -o /var/tmp/9505982.1.3-GPU-1080/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
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
layer root/'dec_07_att_value0' output: Data(name='dec_07_att_value0_output', shape=(None, 512))
layer root/'dec_07_att_value' output: Data(name='dec_07_att_value_output', shape=(None, 8, 64))
layer root/'dec_11_att_key0' output: Data(name='dec_11_att_key0_output', shape=(None, 512))
layer root/'dec_11_att_key' output: Data(name='dec_11_att_key_output', shape=(None, 8, 64))
layer root/'dec_03_att_value0' output: Data(name='dec_03_att_value0_output', shape=(None, 512))
layer root/'dec_03_att_value' output: Data(name='dec_03_att_value_output', shape=(None, 8, 64))
layer root/'dec_02_att_key0' output: Data(name='dec_02_att_key0_output', shape=(None, 512))
layer root/'dec_02_att_key' output: Data(name='dec_02_att_key_output', shape=(None, 8, 64))
layer root/'dec_06_att_key0' output: Data(name='dec_06_att_key0_output', shape=(None, 512))
layer root/'dec_06_att_key' output: Data(name='dec_06_att_key_output', shape=(None, 8, 64))
layer root/'dec_03_att_key0' output: Data(name='dec_03_att_key0_output', shape=(None, 512))
layer root/'dec_03_att_key' output: Data(name='dec_03_att_key_output', shape=(None, 8, 64))
layer root/'dec_12_att_value0' output: Data(name='dec_12_att_value0_output', shape=(None, 512))
layer root/'dec_12_att_value' output: Data(name='dec_12_att_value_output', shape=(None, 8, 64))
layer root/'dec_05_att_key0' output: Data(name='dec_05_att_key0_output', shape=(None, 512))
layer root/'dec_05_att_key' output: Data(name='dec_05_att_key_output', shape=(None, 8, 64))
layer root/'dec_05_att_value0' output: Data(name='dec_05_att_value0_output', shape=(None, 512))
layer root/'dec_05_att_value' output: Data(name='dec_05_att_value_output', shape=(None, 8, 64))
layer root/'dec_08_att_value0' output: Data(name='dec_08_att_value0_output', shape=(None, 512))
layer root/'dec_08_att_value' output: Data(name='dec_08_att_value_output', shape=(None, 8, 64))
layer root/'dec_09_att_value0' output: Data(name='dec_09_att_value0_output', shape=(None, 512))
layer root/'dec_09_att_value' output: Data(name='dec_09_att_value_output', shape=(None, 8, 64))
layer root/'dec_07_att_key0' output: Data(name='dec_07_att_key0_output', shape=(None, 512))
layer root/'dec_07_att_key' output: Data(name='dec_07_att_key_output', shape=(None, 8, 64))
layer root/'dec_01_att_key0' output: Data(name='dec_01_att_key0_output', shape=(None, 512))
layer root/'dec_01_att_key' output: Data(name='dec_01_att_key_output', shape=(None, 8, 64))
layer root/'dec_10_att_key0' output: Data(name='dec_10_att_key0_output', shape=(None, 512))
layer root/'dec_10_att_key' output: Data(name='dec_10_att_key_output', shape=(None, 8, 64))
layer root/'dec_08_att_key0' output: Data(name='dec_08_att_key0_output', shape=(None, 512))
layer root/'dec_08_att_key' output: Data(name='dec_08_att_key_output', shape=(None, 8, 64))
layer root/'dec_04_att_key0' output: Data(name='dec_04_att_key0_output', shape=(None, 512))
layer root/'dec_04_att_key' output: Data(name='dec_04_att_key_output', shape=(None, 8, 64))
layer root/'dec_04_att_value0' output: Data(name='dec_04_att_value0_output', shape=(None, 512))
layer root/'dec_04_att_value' output: Data(name='dec_04_att_value_output', shape=(None, 8, 64))
layer root/'dec_10_att_value0' output: Data(name='dec_10_att_value0_output', shape=(None, 512))
layer root/'dec_10_att_value' output: Data(name='dec_10_att_value_output', shape=(None, 8, 64))
layer root/'dec_11_att_value0' output: Data(name='dec_11_att_value0_output', shape=(None, 512))
layer root/'dec_11_att_value' output: Data(name='dec_11_att_value_output', shape=(None, 8, 64))
layer root/'dec_09_att_key0' output: Data(name='dec_09_att_key0_output', shape=(None, 512))
layer root/'dec_09_att_key' output: Data(name='dec_09_att_key_output', shape=(None, 8, 64))
layer root/'dec_02_att_value0' output: Data(name='dec_02_att_value0_output', shape=(None, 512))
layer root/'dec_02_att_value' output: Data(name='dec_02_att_value_output', shape=(None, 8, 64))
layer root/'dec_06_att_value0' output: Data(name='dec_06_att_value0_output', shape=(None, 512))
layer root/'dec_06_att_value' output: Data(name='dec_06_att_value_output', shape=(None, 8, 64))
layer root/'dec_12_att_key0' output: Data(name='dec_12_att_key0_output', shape=(None, 512))
layer root/'dec_12_att_key' output: Data(name='dec_12_att_key_output', shape=(None, 8, 64))
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
TF: log_dir: /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/transformer/forward/tf_log_dir/prefix:test-clean-481-2019-07-02-10-06-44
warning: sequence length (3392) larger than limit (600)
warning: sequence length (2512) larger than limit (600)
warning: sequence length (2843) larger than limit (600)
warning: sequence length (2961) larger than limit (600)
warning: sequence length (3166) larger than limit (600)
warning: sequence length (2542) larger than limit (600)
warning: sequence length (3062) larger than limit (600)
warning: sequence length (2915) larger than limit (600)
warning: sequence length (2842) larger than limit (600)
warning: sequence length (3496) larger than limit (600)
warning: sequence length (2828) larger than limit (600)
warning: sequence length (2713) larger than limit (600)
Note: There are still these uninitialized variables: ['learning_rate:0']
warning: sequence length (2486) larger than limit (600)
att-weights epoch 481, step 0, max_size:classes 120, max_size:data 3392, mem_usage:GPU:0 1.0GB, num_seqs 1, 4.831 sec/step, elapsed 0:00:11, exp. remaining 0:45:53, complete 0.42%
warning: sequence length (2565) larger than limit (600)
att-weights epoch 481, step 1, max_size:classes 103, max_size:data 2512, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.400 sec/step, elapsed 0:00:13, exp. remaining 0:47:19, complete 0.46%
warning: sequence length (2446) larger than limit (600)
att-weights epoch 481, step 2, max_size:classes 101, max_size:data 2843, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.489 sec/step, elapsed 0:00:14, exp. remaining 0:48:53, complete 0.50%
warning: sequence length (2387) larger than limit (600)
att-weights epoch 481, step 3, max_size:classes 93, max_size:data 2961, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.762 sec/step, elapsed 0:00:16, exp. remaining 0:51:09, complete 0.53%
warning: sequence length (3278) larger than limit (600)
att-weights epoch 481, step 4, max_size:classes 97, max_size:data 3166, mem_usage:GPU:0 1.0GB, num_seqs 1, 6.811 sec/step, elapsed 0:00:23, exp. remaining 1:07:46, complete 0.57%
warning: sequence length (2334) larger than limit (600)
att-weights epoch 481, step 5, max_size:classes 89, max_size:data 2542, mem_usage:GPU:0 1.0GB, num_seqs 1, 19.916 sec/step, elapsed 0:00:48, exp. remaining 2:12:18, complete 0.61%
warning: sequence length (2858) larger than limit (600)
att-weights epoch 481, step 6, max_size:classes 90, max_size:data 3062, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.572 sec/step, elapsed 0:00:50, exp. remaining 2:08:51, complete 0.65%
warning: sequence length (2102) larger than limit (600)
+------- EPILOGUE SCRIPT -----------------------------------------------
|
| Job ID ..............: 9505982
| Stopped at ..........: Tue Jul  2 12:09:54 CEST 2019
| Resources requested .: pxe=ubuntu_16.04,h_vmem=1536G,s_core=0,h_rss=4G,h_rt=7200,gpu=1,scratch_free=5G,h_fsize=20G,num_proc=5
| Resources used ......: cpu=00:02:12, mem=171.54915 GB s, io=1.69913 GB, vmem=3.018G, maxvmem=3.868G, last_file_cache=1.759G, last_rss=3M, max-cache=1.656G
| Memory used .........: 3.414G / 4.000G (85.4%)
| Total time used .....: 0:03:15
|
+------- EPILOGUE SCRIPT -----------------------------------------------
+------- PROLOGUE SCRIPT -----------------------------------------------
|
| Job ID ...........: 9505996
| Started at .......: Tue Jul  2 12:13:18 CEST 2019
| Execution host ...: cluster-cn-244
| Cluster queue ....: 4-GPU-1080
| Script ...........: /var/spool/sge/cluster-cn-244/job_scripts/9505996
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
RETURNN starting up, version 20181022.053313--git-dfada43-dirty, date/time 2019-07-02-12-13-21 (UTC+0200), pid 21792, cwd /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc, Python /u/bahar/settings/python3-returnn-tf1.9/bin/python3
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
       incarnation: 14924608382727586234
  2/2: name: "/device:GPU:0"
       device_type: "GPU"
       memory_limit: 10911236096
       locality {
         bus_id: 1
         links {
         }
       }
       incarnation: 7601051809194232941
       physical_device_desc: "device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1"
Using gpu device 1: GeForce GTX 1080 Ti
Device not set explicitly, and we found a GPU, which we will use.
Model file prefix: /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc/net-model/network
NOTE: We will use 'sorted_reverse' seq ordering.
Setup tf.Session with options {'device_count': {'GPU': 1}, 'log_device_placement': False} ...
layer root/'data' output: Data(name='data', shape=(None, 40))
layer root/'source' output: Data(name='source_output', shape=(None, 40))
layer root/'lstm0_fw' output: Data(name='lstm0_fw_output', shape=(None, 1024), batch_dim_axis=1)
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9505996.1.4-GPU-1080/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.cc -o /var/tmp/9505996.1.4-GPU-1080/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9505996.1.4-GPU-1080/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.cc -o /var/tmp/9505996.1.4-GPU-1080/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
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
layer root/'dec_08_att_key0' output: Data(name='dec_08_att_key0_output', shape=(None, 512))
layer root/'dec_08_att_key' output: Data(name='dec_08_att_key_output', shape=(None, 8, 64))
layer root/'dec_09_att_value0' output: Data(name='dec_09_att_value0_output', shape=(None, 512))
layer root/'dec_09_att_value' output: Data(name='dec_09_att_value_output', shape=(None, 8, 64))
layer root/'dec_05_att_value0' output: Data(name='dec_05_att_value0_output', shape=(None, 512))
layer root/'dec_05_att_value' output: Data(name='dec_05_att_value_output', shape=(None, 8, 64))
layer root/'dec_05_att_key0' output: Data(name='dec_05_att_key0_output', shape=(None, 512))
layer root/'dec_05_att_key' output: Data(name='dec_05_att_key_output', shape=(None, 8, 64))
layer root/'dec_10_att_value0' output: Data(name='dec_10_att_value0_output', shape=(None, 512))
layer root/'dec_10_att_value' output: Data(name='dec_10_att_value_output', shape=(None, 8, 64))
layer root/'dec_12_att_value0' output: Data(name='dec_12_att_value0_output', shape=(None, 512))
layer root/'dec_12_att_value' output: Data(name='dec_12_att_value_output', shape=(None, 8, 64))
layer root/'dec_01_att_value0' output: Data(name='dec_01_att_value0_output', shape=(None, 512))
layer root/'dec_01_att_value' output: Data(name='dec_01_att_value_output', shape=(None, 8, 64))
layer root/'dec_03_att_value0' output: Data(name='dec_03_att_value0_output', shape=(None, 512))
layer root/'dec_03_att_value' output: Data(name='dec_03_att_value_output', shape=(None, 8, 64))
layer root/'dec_07_att_key0' output: Data(name='dec_07_att_key0_output', shape=(None, 512))
layer root/'dec_07_att_key' output: Data(name='dec_07_att_key_output', shape=(None, 8, 64))
layer root/'dec_09_att_key0' output: Data(name='dec_09_att_key0_output', shape=(None, 512))
layer root/'dec_09_att_key' output: Data(name='dec_09_att_key_output', shape=(None, 8, 64))
layer root/'dec_02_att_key0' output: Data(name='dec_02_att_key0_output', shape=(None, 512))
layer root/'dec_02_att_key' output: Data(name='dec_02_att_key_output', shape=(None, 8, 64))
layer root/'dec_01_att_key0' output: Data(name='dec_01_att_key0_output', shape=(None, 512))
layer root/'dec_01_att_key' output: Data(name='dec_01_att_key_output', shape=(None, 8, 64))
layer root/'dec_06_att_key0' output: Data(name='dec_06_att_key0_output', shape=(None, 512))
layer root/'dec_06_att_key' output: Data(name='dec_06_att_key_output', shape=(None, 8, 64))
layer root/'dec_10_att_key0' output: Data(name='dec_10_att_key0_output', shape=(None, 512))
layer root/'dec_10_att_key' output: Data(name='dec_10_att_key_output', shape=(None, 8, 64))
layer root/'dec_12_att_key0' output: Data(name='dec_12_att_key0_output', shape=(None, 512))
layer root/'dec_12_att_key' output: Data(name='dec_12_att_key_output', shape=(None, 8, 64))
layer root/'dec_07_att_value0' output: Data(name='dec_07_att_value0_output', shape=(None, 512))
layer root/'dec_07_att_value' output: Data(name='dec_07_att_value_output', shape=(None, 8, 64))
layer root/'dec_04_att_key0' output: Data(name='dec_04_att_key0_output', shape=(None, 512))
layer root/'dec_04_att_key' output: Data(name='dec_04_att_key_output', shape=(None, 8, 64))
layer root/'dec_06_att_value0' output: Data(name='dec_06_att_value0_output', shape=(None, 512))
layer root/'dec_06_att_value' output: Data(name='dec_06_att_value_output', shape=(None, 8, 64))
layer root/'dec_03_att_key0' output: Data(name='dec_03_att_key0_output', shape=(None, 512))
layer root/'dec_03_att_key' output: Data(name='dec_03_att_key_output', shape=(None, 8, 64))
layer root/'dec_02_att_value0' output: Data(name='dec_02_att_value0_output', shape=(None, 512))
layer root/'dec_02_att_value' output: Data(name='dec_02_att_value_output', shape=(None, 8, 64))
layer root/'dec_04_att_value0' output: Data(name='dec_04_att_value0_output', shape=(None, 512))
layer root/'dec_04_att_value' output: Data(name='dec_04_att_value_output', shape=(None, 8, 64))
layer root/'dec_11_att_key0' output: Data(name='dec_11_att_key0_output', shape=(None, 512))
layer root/'dec_11_att_key' output: Data(name='dec_11_att_key_output', shape=(None, 8, 64))
layer root/'dec_11_att_value0' output: Data(name='dec_11_att_value0_output', shape=(None, 512))
layer root/'dec_11_att_value' output: Data(name='dec_11_att_value_output', shape=(None, 8, 64))
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
TF: log_dir: /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/transformer/forward/tf_log_dir/prefix:test-clean-481-2019-07-02-10-13-19
Note: There are still these uninitialized variables: ['learning_rate:0']
att-weights epoch 481, step 0, max_size:classes 120, max_size:data 3392, mem_usage:GPU:0 1.0GB, num_seqs 1, 3.151 sec/step, elapsed 0:00:09, exp. remaining 0:39:24, complete 0.42%
att-weights epoch 481, step 1, max_size:classes 103, max_size:data 2512, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.838 sec/step, elapsed 0:00:11, exp. remaining 0:43:09, complete 0.46%
att-weights epoch 481, step 2, max_size:classes 101, max_size:data 2843, mem_usage:GPU:0 1.0GB, num_seqs 1, 2.112 sec/step, elapsed 0:00:14, exp. remaining 0:46:59, complete 0.50%
att-weights epoch 481, step 3, max_size:classes 93, max_size:data 2961, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.269 sec/step, elapsed 0:00:15, exp. remaining 0:47:57, complete 0.53%
att-weights epoch 481, step 4, max_size:classes 97, max_size:data 3166, mem_usage:GPU:0 1.0GB, num_seqs 1, 2.988 sec/step, elapsed 0:00:18, exp. remaining 0:53:43, complete 0.57%
att-weights epoch 481, step 5, max_size:classes 89, max_size:data 2542, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.047 sec/step, elapsed 0:00:19, exp. remaining 0:53:43, complete 0.61%
att-weights epoch 481, step 6, max_size:classes 90, max_size:data 3062, mem_usage:GPU:0 1.0GB, num_seqs 1, 5.613 sec/step, elapsed 0:00:25, exp. remaining 1:04:58, complete 0.65%
att-weights epoch 481, step 7, max_size:classes 91, max_size:data 2915, mem_usage:GPU:0 1.0GB, num_seqs 1, 2.152 sec/step, elapsed 0:00:27, exp. remaining 1:07:01, complete 0.69%
att-weights epoch 481, step 8, max_size:classes 101, max_size:data 2842, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.450 sec/step, elapsed 0:00:29, exp. remaining 1:07:08, complete 0.73%
att-weights epoch 481, step 9, max_size:classes 87, max_size:data 3496, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.227 sec/step, elapsed 0:00:30, exp. remaining 1:06:28, complete 0.76%
att-weights epoch 481, step 10, max_size:classes 86, max_size:data 2828, mem_usage:GPU:0 1.0GB, num_seqs 1, 3.896 sec/step, elapsed 0:00:34, exp. remaining 1:11:39, complete 0.80%
att-weights epoch 481, step 11, max_size:classes 89, max_size:data 2713, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.174 sec/step, elapsed 0:00:36, exp. remaining 1:10:52, complete 0.84%
att-weights epoch 481, step 12, max_size:classes 84, max_size:data 2486, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.984 sec/step, elapsed 0:00:36, exp. remaining 1:09:36, complete 0.88%
att-weights epoch 481, step 13, max_size:classes 81, max_size:data 2565, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.701 sec/step, elapsed 0:00:38, exp. remaining 1:09:45, complete 0.92%
att-weights epoch 481, step 14, max_size:classes 79, max_size:data 2446, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.882 sec/step, elapsed 0:00:40, exp. remaining 1:10:11, complete 0.95%
att-weights epoch 481, step 15, max_size:classes 85, max_size:data 2387, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.182 sec/step, elapsed 0:00:41, exp. remaining 1:09:26, complete 0.99%
att-weights epoch 481, step 16, max_size:classes 93, max_size:data 3278, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.496 sec/step, elapsed 0:00:43, exp. remaining 1:09:14, complete 1.03%
att-weights epoch 481, step 17, max_size:classes 85, max_size:data 2334, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.920 sec/step, elapsed 0:00:44, exp. remaining 1:08:09, complete 1.07%
att-weights epoch 481, step 18, max_size:classes 77, max_size:data 2858, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.975 sec/step, elapsed 0:00:45, exp. remaining 1:07:14, complete 1.11%
att-weights epoch 481, step 19, max_size:classes 76, max_size:data 2102, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.843 sec/step, elapsed 0:00:45, exp. remaining 1:06:10, complete 1.15%
att-weights epoch 481, step 20, max_size:classes 78, max_size:data 2307, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.929 sec/step, elapsed 0:00:46, exp. remaining 1:05:18, complete 1.18%
att-weights epoch 481, step 21, max_size:classes 73, max_size:data 3289, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.222 sec/step, elapsed 0:00:48, exp. remaining 1:04:53, complete 1.22%
att-weights epoch 481, step 22, max_size:classes 67, max_size:data 2019, mem_usage:GPU:0 1.0GB, num_seqs 1, 2.203 sec/step, elapsed 0:00:50, exp. remaining 1:05:47, complete 1.26%
att-weights epoch 481, step 23, max_size:classes 76, max_size:data 2599, mem_usage:GPU:0 1.0GB, num_seqs 1, 6.699 sec/step, elapsed 0:00:57, exp. remaining 1:12:19, complete 1.30%
att-weights epoch 481, step 24, max_size:classes 77, max_size:data 2004, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.924 sec/step, elapsed 0:00:57, exp. remaining 1:11:21, complete 1.34%
att-weights epoch 481, step 25, max_size:classes 71, max_size:data 2550, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.162 sec/step, elapsed 0:00:59, exp. remaining 1:10:44, complete 1.37%
att-weights epoch 481, step 26, max_size:classes 67, max_size:data 2375, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.038 sec/step, elapsed 0:01:00, exp. remaining 1:10:01, complete 1.41%
att-weights epoch 481, step 27, max_size:classes 75, max_size:data 3082, mem_usage:GPU:0 1.0GB, num_seqs 1, 3.369 sec/step, elapsed 0:01:03, exp. remaining 1:11:57, complete 1.45%
att-weights epoch 481, step 28, max_size:classes 83, max_size:data 2615, mem_usage:GPU:0 1.0GB, num_seqs 1, 2.267 sec/step, elapsed 0:01:05, exp. remaining 1:12:35, complete 1.49%
att-weights epoch 481, step 29, max_size:classes 79, max_size:data 2612, mem_usage:GPU:0 1.0GB, num_seqs 1, 3.426 sec/step, elapsed 0:01:09, exp. remaining 1:14:26, complete 1.53%
att-weights epoch 481, step 30, max_size:classes 70, max_size:data 2595, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.902 sec/step, elapsed 0:01:10, exp. remaining 1:13:32, complete 1.56%
att-weights epoch 481, step 31, max_size:classes 72, max_size:data 2221, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.895 sec/step, elapsed 0:01:11, exp. remaining 1:12:40, complete 1.60%
att-weights epoch 481, step 32, max_size:classes 67, max_size:data 2332, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.941 sec/step, elapsed 0:01:11, exp. remaining 1:11:53, complete 1.64%
att-weights epoch 481, step 33, max_size:classes 67, max_size:data 2285, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.837 sec/step, elapsed 0:01:12, exp. remaining 1:11:03, complete 1.68%
att-weights epoch 481, step 34, max_size:classes 71, max_size:data 3162, mem_usage:GPU:0 1.0GB, num_seqs 1, 2.687 sec/step, elapsed 0:01:15, exp. remaining 1:12:00, complete 1.72%
att-weights epoch 481, step 35, max_size:classes 76, max_size:data 2174, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.846 sec/step, elapsed 0:01:16, exp. remaining 1:11:12, complete 1.76%
att-weights epoch 481, step 36, max_size:classes 74, max_size:data 1888, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.864 sec/step, elapsed 0:01:17, exp. remaining 1:10:27, complete 1.79%
att-weights epoch 481, step 37, max_size:classes 83, max_size:data 2327, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.160 sec/step, elapsed 0:01:18, exp. remaining 1:09:59, complete 1.83%
att-weights epoch 481, step 38, max_size:classes 80, max_size:data 3005, mem_usage:GPU:0 1.0GB, num_seqs 1, 2.866 sec/step, elapsed 0:01:21, exp. remaining 1:11:02, complete 1.87%
att-weights epoch 481, step 39, max_size:classes 74, max_size:data 2229, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.705 sec/step, elapsed 0:01:22, exp. remaining 1:11:03, complete 1.91%
att-weights epoch 481, step 40, max_size:classes 73, max_size:data 2842, mem_usage:GPU:0 1.0GB, num_seqs 1, 4.662 sec/step, elapsed 0:01:27, exp. remaining 1:13:33, complete 1.95%
att-weights epoch 481, step 41, max_size:classes 66, max_size:data 2753, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.126 sec/step, elapsed 0:01:28, exp. remaining 1:13:02, complete 1.98%
att-weights epoch 481, step 42, max_size:classes 72, max_size:data 2151, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.058 sec/step, elapsed 0:01:29, exp. remaining 1:12:29, complete 2.02%
att-weights epoch 481, step 43, max_size:classes 64, max_size:data 1948, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.872 sec/step, elapsed 0:01:30, exp. remaining 1:11:48, complete 2.06%
att-weights epoch 481, step 44, max_size:classes 72, max_size:data 2026, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.834 sec/step, elapsed 0:01:31, exp. remaining 1:11:07, complete 2.10%
att-weights epoch 481, step 45, max_size:classes 67, max_size:data 2086, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.477 sec/step, elapsed 0:01:32, exp. remaining 1:09:40, complete 2.18%
att-weights epoch 481, step 46, max_size:classes 74, max_size:data 2210, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.959 sec/step, elapsed 0:01:33, exp. remaining 1:09:09, complete 2.21%
att-weights epoch 481, step 47, max_size:classes 76, max_size:data 2217, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.883 sec/step, elapsed 0:01:34, exp. remaining 1:08:36, complete 2.25%
att-weights epoch 481, step 48, max_size:classes 60, max_size:data 2455, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.870 sec/step, elapsed 0:01:35, exp. remaining 1:08:02, complete 2.29%
att-weights epoch 481, step 49, max_size:classes 65, max_size:data 2179, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.802 sec/step, elapsed 0:01:36, exp. remaining 1:07:28, complete 2.33%
att-weights epoch 481, step 50, max_size:classes 67, max_size:data 2038, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.932 sec/step, elapsed 0:01:37, exp. remaining 1:06:59, complete 2.37%
att-weights epoch 481, step 51, max_size:classes 64, max_size:data 2449, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.888 sec/step, elapsed 0:01:38, exp. remaining 1:06:30, complete 2.40%
att-weights epoch 481, step 52, max_size:classes 75, max_size:data 2810, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.995 sec/step, elapsed 0:01:39, exp. remaining 1:06:06, complete 2.44%
att-weights epoch 481, step 53, max_size:classes 69, max_size:data 2617, mem_usage:GPU:0 1.0GB, num_seqs 1, 19.274 sec/step, elapsed 0:01:58, exp. remaining 1:17:41, complete 2.48%
att-weights epoch 481, step 54, max_size:classes 68, max_size:data 1992, mem_usage:GPU:0 1.0GB, num_seqs 2, 17.789 sec/step, elapsed 0:02:16, exp. remaining 1:27:57, complete 2.52%
att-weights epoch 481, step 55, max_size:classes 66, max_size:data 1674, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.690 sec/step, elapsed 0:02:18, exp. remaining 1:27:41, complete 2.56%
att-weights epoch 481, step 56, max_size:classes 62, max_size:data 2091, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.863 sec/step, elapsed 0:02:18, exp. remaining 1:26:54, complete 2.60%
att-weights epoch 481, step 57, max_size:classes 64, max_size:data 1903, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.842 sec/step, elapsed 0:02:19, exp. remaining 1:26:07, complete 2.63%
att-weights epoch 481, step 58, max_size:classes 68, max_size:data 2119, mem_usage:GPU:0 1.0GB, num_seqs 1, 5.101 sec/step, elapsed 0:02:24, exp. remaining 1:27:57, complete 2.67%
att-weights epoch 481, step 59, max_size:classes 65, max_size:data 2152, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.325 sec/step, elapsed 0:02:26, exp. remaining 1:26:14, complete 2.75%
att-weights epoch 481, step 60, max_size:classes 74, max_size:data 2149, mem_usage:GPU:0 1.0GB, num_seqs 1, 12.333 sec/step, elapsed 0:02:38, exp. remaining 1:32:11, complete 2.79%
att-weights epoch 481, step 61, max_size:classes 64, max_size:data 2135, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.749 sec/step, elapsed 0:02:39, exp. remaining 1:30:05, complete 2.86%
att-weights epoch 481, step 62, max_size:classes 64, max_size:data 1653, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.771 sec/step, elapsed 0:02:40, exp. remaining 1:29:18, complete 2.90%
att-weights epoch 481, step 63, max_size:classes 56, max_size:data 2250, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.088 sec/step, elapsed 0:02:41, exp. remaining 1:28:42, complete 2.94%
att-weights epoch 481, step 64, max_size:classes 61, max_size:data 2448, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.932 sec/step, elapsed 0:02:42, exp. remaining 1:28:02, complete 2.98%
att-weights epoch 481, step 65, max_size:classes 66, max_size:data 2057, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.779 sec/step, elapsed 0:02:42, exp. remaining 1:27:18, complete 3.02%
att-weights epoch 481, step 66, max_size:classes 67, max_size:data 1853, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.769 sec/step, elapsed 0:02:43, exp. remaining 1:26:35, complete 3.05%
att-weights epoch 481, step 67, max_size:classes 68, max_size:data 2016, mem_usage:GPU:0 1.0GB, num_seqs 1, 7.131 sec/step, elapsed 0:02:50, exp. remaining 1:29:13, complete 3.09%
att-weights epoch 481, step 68, max_size:classes 63, max_size:data 1971, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.488 sec/step, elapsed 0:02:52, exp. remaining 1:28:51, complete 3.13%
att-weights epoch 481, step 69, max_size:classes 67, max_size:data 2076, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.353 sec/step, elapsed 0:02:53, exp. remaining 1:27:21, complete 3.21%
att-weights epoch 481, step 70, max_size:classes 68, max_size:data 1758, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.643 sec/step, elapsed 0:02:55, exp. remaining 1:27:06, complete 3.24%
att-weights epoch 481, step 71, max_size:classes 67, max_size:data 2036, mem_usage:GPU:0 1.0GB, num_seqs 1, 6.728 sec/step, elapsed 0:03:01, exp. remaining 1:29:22, complete 3.28%
att-weights epoch 481, step 72, max_size:classes 57, max_size:data 2015, mem_usage:GPU:0 1.0GB, num_seqs 1, 10.657 sec/step, elapsed 0:03:12, exp. remaining 1:33:28, complete 3.32%
att-weights epoch 481, step 73, max_size:classes 65, max_size:data 1995, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.513 sec/step, elapsed 0:03:14, exp. remaining 1:33:06, complete 3.36%
att-weights epoch 481, step 74, max_size:classes 60, max_size:data 2719, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.343 sec/step, elapsed 0:03:15, exp. remaining 1:32:39, complete 3.40%
att-weights epoch 481, step 75, max_size:classes 55, max_size:data 2029, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.341 sec/step, elapsed 0:03:16, exp. remaining 1:32:13, complete 3.44%
att-weights epoch 481, step 76, max_size:classes 53, max_size:data 2237, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.822 sec/step, elapsed 0:03:17, exp. remaining 1:31:33, complete 3.47%
att-weights epoch 481, step 77, max_size:classes 57, max_size:data 2054, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.875 sec/step, elapsed 0:03:18, exp. remaining 1:29:54, complete 3.55%
att-weights epoch 481, step 78, max_size:classes 67, max_size:data 1864, mem_usage:GPU:0 1.0GB, num_seqs 2, 3.790 sec/step, elapsed 0:03:22, exp. remaining 1:30:37, complete 3.59%
att-weights epoch 481, step 79, max_size:classes 62, max_size:data 1823, mem_usage:GPU:0 1.0GB, num_seqs 1, 3.807 sec/step, elapsed 0:03:26, exp. remaining 1:31:19, complete 3.63%
att-weights epoch 481, step 80, max_size:classes 58, max_size:data 2002, mem_usage:GPU:0 1.0GB, num_seqs 1, 2.828 sec/step, elapsed 0:03:28, exp. remaining 1:31:34, complete 3.66%
att-weights epoch 481, step 81, max_size:classes 61, max_size:data 1542, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.020 sec/step, elapsed 0:03:29, exp. remaining 1:31:02, complete 3.70%
att-weights epoch 481, step 82, max_size:classes 59, max_size:data 2013, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.221 sec/step, elapsed 0:03:31, exp. remaining 1:29:38, complete 3.78%
att-weights epoch 481, step 83, max_size:classes 61, max_size:data 2002, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.082 sec/step, elapsed 0:03:32, exp. remaining 1:29:09, complete 3.82%
att-weights epoch 481, step 84, max_size:classes 59, max_size:data 2066, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.000 sec/step, elapsed 0:03:33, exp. remaining 1:27:45, complete 3.89%
att-weights epoch 481, step 85, max_size:classes 66, max_size:data 2001, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.157 sec/step, elapsed 0:03:34, exp. remaining 1:27:20, complete 3.93%
att-weights epoch 481, step 86, max_size:classes 62, max_size:data 1962, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.708 sec/step, elapsed 0:03:36, exp. remaining 1:27:09, complete 3.97%
att-weights epoch 481, step 87, max_size:classes 56, max_size:data 1629, mem_usage:GPU:0 1.0GB, num_seqs 1, 5.799 sec/step, elapsed 0:03:41, exp. remaining 1:27:44, complete 4.05%
att-weights epoch 481, step 88, max_size:classes 61, max_size:data 2006, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.970 sec/step, elapsed 0:03:42, exp. remaining 1:26:25, complete 4.12%
att-weights epoch 481, step 89, max_size:classes 62, max_size:data 1623, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.713 sec/step, elapsed 0:03:43, exp. remaining 1:25:03, complete 4.20%
att-weights epoch 481, step 90, max_size:classes 62, max_size:data 2258, mem_usage:GPU:0 1.0GB, num_seqs 1, 3.659 sec/step, elapsed 0:03:47, exp. remaining 1:24:50, complete 4.27%
att-weights epoch 481, step 91, max_size:classes 61, max_size:data 1813, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.467 sec/step, elapsed 0:03:48, exp. remaining 1:24:35, complete 4.31%
att-weights epoch 481, step 92, max_size:classes 69, max_size:data 2368, mem_usage:GPU:0 1.0GB, num_seqs 1, 3.293 sec/step, elapsed 0:03:52, exp. remaining 1:25:01, complete 4.35%
att-weights epoch 481, step 93, max_size:classes 60, max_size:data 1797, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.653 sec/step, elapsed 0:03:53, exp. remaining 1:24:05, complete 4.43%
att-weights epoch 481, step 94, max_size:classes 59, max_size:data 1987, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.979 sec/step, elapsed 0:03:54, exp. remaining 1:23:41, complete 4.47%
att-weights epoch 481, step 95, max_size:classes 63, max_size:data 2106, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.986 sec/step, elapsed 0:03:55, exp. remaining 1:22:33, complete 4.54%
att-weights epoch 481, step 96, max_size:classes 60, max_size:data 1992, mem_usage:GPU:0 1.0GB, num_seqs 2, 13.761 sec/step, elapsed 0:04:09, exp. remaining 1:25:51, complete 4.62%
att-weights epoch 481, step 97, max_size:classes 59, max_size:data 1992, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.844 sec/step, elapsed 0:04:11, exp. remaining 1:25:01, complete 4.69%
att-weights epoch 481, step 98, max_size:classes 60, max_size:data 1981, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.338 sec/step, elapsed 0:04:12, exp. remaining 1:24:02, complete 4.77%
att-weights epoch 481, step 99, max_size:classes 53, max_size:data 1657, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.186 sec/step, elapsed 0:04:13, exp. remaining 1:23:44, complete 4.81%
att-weights epoch 481, step 100, max_size:classes 57, max_size:data 2047, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.809 sec/step, elapsed 0:04:14, exp. remaining 1:22:37, complete 4.89%
att-weights epoch 481, step 101, max_size:classes 55, max_size:data 2016, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.142 sec/step, elapsed 0:04:15, exp. remaining 1:21:39, complete 4.96%
att-weights epoch 481, step 102, max_size:classes 54, max_size:data 1798, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.313 sec/step, elapsed 0:04:17, exp. remaining 1:20:45, complete 5.04%
att-weights epoch 481, step 103, max_size:classes 64, max_size:data 2088, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.896 sec/step, elapsed 0:04:17, exp. remaining 1:19:46, complete 5.11%
att-weights epoch 481, step 104, max_size:classes 64, max_size:data 1641, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.216 sec/step, elapsed 0:04:19, exp. remaining 1:19:31, complete 5.15%
att-weights epoch 481, step 105, max_size:classes 57, max_size:data 1674, mem_usage:GPU:0 1.0GB, num_seqs 2, 4.692 sec/step, elapsed 0:04:23, exp. remaining 1:20:20, complete 5.19%
att-weights epoch 481, step 106, max_size:classes 58, max_size:data 1832, mem_usage:GPU:0 1.0GB, num_seqs 2, 3.103 sec/step, elapsed 0:04:27, exp. remaining 1:20:39, complete 5.23%
att-weights epoch 481, step 107, max_size:classes 60, max_size:data 1907, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.546 sec/step, elapsed 0:04:28, exp. remaining 1:19:53, complete 5.31%
att-weights epoch 481, step 108, max_size:classes 59, max_size:data 2126, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.873 sec/step, elapsed 0:04:29, exp. remaining 1:18:56, complete 5.38%
att-weights epoch 481, step 109, max_size:classes 62, max_size:data 1775, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.061 sec/step, elapsed 0:04:31, exp. remaining 1:18:22, complete 5.46%
att-weights epoch 481, step 110, max_size:classes 57, max_size:data 1910, mem_usage:GPU:0 1.0GB, num_seqs 2, 5.159 sec/step, elapsed 0:04:36, exp. remaining 1:18:42, complete 5.53%
att-weights epoch 481, step 111, max_size:classes 56, max_size:data 1723, mem_usage:GPU:0 1.0GB, num_seqs 2, 3.487 sec/step, elapsed 0:04:40, exp. remaining 1:19:06, complete 5.57%
att-weights epoch 481, step 112, max_size:classes 60, max_size:data 2000, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.584 sec/step, elapsed 0:04:41, exp. remaining 1:18:59, complete 5.61%
att-weights epoch 481, step 113, max_size:classes 51, max_size:data 2034, mem_usage:GPU:0 1.0GB, num_seqs 1, 3.250 sec/step, elapsed 0:04:44, exp. remaining 1:18:45, complete 5.69%
att-weights epoch 481, step 114, max_size:classes 53, max_size:data 1609, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.768 sec/step, elapsed 0:04:45, exp. remaining 1:17:52, complete 5.76%
att-weights epoch 481, step 115, max_size:classes 58, max_size:data 2099, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.775 sec/step, elapsed 0:04:46, exp. remaining 1:17:32, complete 5.80%
att-weights epoch 481, step 116, max_size:classes 57, max_size:data 1842, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.070 sec/step, elapsed 0:04:47, exp. remaining 1:16:45, complete 5.88%
att-weights epoch 481, step 117, max_size:classes 58, max_size:data 1721, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.025 sec/step, elapsed 0:04:48, exp. remaining 1:15:58, complete 5.95%
att-weights epoch 481, step 118, max_size:classes 53, max_size:data 1701, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.433 sec/step, elapsed 0:04:50, exp. remaining 1:15:50, complete 5.99%
att-weights epoch 481, step 119, max_size:classes 60, max_size:data 1503, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.184 sec/step, elapsed 0:04:51, exp. remaining 1:15:38, complete 6.03%
att-weights epoch 481, step 120, max_size:classes 53, max_size:data 1565, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.676 sec/step, elapsed 0:04:51, exp. remaining 1:14:48, complete 6.11%
att-weights epoch 481, step 121, max_size:classes 64, max_size:data 2540, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.120 sec/step, elapsed 0:04:53, exp. remaining 1:14:06, complete 6.18%
att-weights epoch 481, step 122, max_size:classes 58, max_size:data 1800, mem_usage:GPU:0 1.0GB, num_seqs 2, 9.570 sec/step, elapsed 0:05:02, exp. remaining 1:15:31, complete 6.26%
att-weights epoch 481, step 123, max_size:classes 54, max_size:data 1685, mem_usage:GPU:0 1.0GB, num_seqs 2, 7.261 sec/step, elapsed 0:05:09, exp. remaining 1:16:20, complete 6.34%
att-weights epoch 481, step 124, max_size:classes 54, max_size:data 2083, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.779 sec/step, elapsed 0:05:10, exp. remaining 1:15:33, complete 6.41%
att-weights epoch 481, step 125, max_size:classes 55, max_size:data 1634, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.133 sec/step, elapsed 0:05:11, exp. remaining 1:15:21, complete 6.45%
att-weights epoch 481, step 126, max_size:classes 58, max_size:data 1637, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.253 sec/step, elapsed 0:05:13, exp. remaining 1:15:11, complete 6.49%
att-weights epoch 481, step 127, max_size:classes 57, max_size:data 2351, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.947 sec/step, elapsed 0:05:13, exp. remaining 1:14:28, complete 6.56%
att-weights epoch 481, step 128, max_size:classes 56, max_size:data 2246, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.792 sec/step, elapsed 0:05:14, exp. remaining 1:13:44, complete 6.64%
+------- EPILOGUE SCRIPT -----------------------------------------------
|
| Job ID ..............: 9505996
| Stopped at ..........: Tue Jul  2 12:20:07 CEST 2019
| Resources requested .: h_rss=4G,h_vmem=1536G,num_proc=5,gpu=1,scratch_free=5G,pxe=ubuntu_16.04,h_fsize=20G,s_core=0,h_rt=7200
| Resources used ......: cpu=00:06:55, mem=1312.36229 GB s, io=3.60671 GB, vmem=4.305G, maxvmem=4.369G, last_file_cache=140K, last_rss=2M, max-cache=4.000G
| Memory used .........: 4.000G / 4.000G (100.0%)
| Total time used .....: 0:06:49
|
+------- EPILOGUE SCRIPT -----------------------------------------------
+------- PROLOGUE SCRIPT -----------------------------------------------
|
| Job ID ...........: 9506020
| Started at .......: Tue Jul  2 12:35:09 CEST 2019
| Execution host ...: cluster-cn-216
| Cluster queue ....: 3-GPU-1080
| Script ...........: /var/spool/sge/cluster-cn-216/job_scripts/9506020
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
RETURNN starting up, version 20181022.053313--git-dfada43-dirty, date/time 2019-07-02-12-35-13 (UTC+0200), pid 24934, cwd /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc, Python /u/bahar/settings/python3-returnn-tf1.9/bin/python3
RETURNN config: /u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc.config
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
       incarnation: 5766240606266365984
  2/2: name: "/device:GPU:0"
       device_type: "GPU"
       memory_limit: 10911236096
       locality {
         bus_id: 1
         links {
         }
       }
       incarnation: 15098534524566578204
       physical_device_desc: "device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1"
Using gpu device 0: GeForce GTX 1080 Ti
Device not set explicitly, and we found a GPU, which we will use.
Model file prefix: /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc/net-model/network
NOTE: We will use 'sorted_reverse' seq ordering.
Setup tf.Session with options {'device_count': {'GPU': 1}, 'log_device_placement': False} ...
layer root/'data' output: Data(name='data', shape=(None, 40))
layer root/'source' output: Data(name='source_output', shape=(None, 40))
layer root/'lstm0_fw' output: Data(name='lstm0_fw_output', shape=(None, 1024), batch_dim_axis=1)
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9506020.1.3-GPU-1080/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.cc -o /var/tmp/9506020.1.3-GPU-1080/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9506020.1.3-GPU-1080/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.cc -o /var/tmp/9506020.1.3-GPU-1080/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
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
layer root/'dec_07_att_key0' output: Data(name='dec_07_att_key0_output', shape=(None, 512))
layer root/'dec_07_att_key' output: Data(name='dec_07_att_key_output', shape=(None, 8, 64))
layer root/'dec_05_att_value0' output: Data(name='dec_05_att_value0_output', shape=(None, 512))
layer root/'dec_05_att_value' output: Data(name='dec_05_att_value_output', shape=(None, 8, 64))
layer root/'dec_03_att_key0' output: Data(name='dec_03_att_key0_output', shape=(None, 512))
layer root/'dec_03_att_key' output: Data(name='dec_03_att_key_output', shape=(None, 8, 64))
layer root/'dec_08_att_value0' output: Data(name='dec_08_att_value0_output', shape=(None, 512))
layer root/'dec_08_att_value' output: Data(name='dec_08_att_value_output', shape=(None, 8, 64))
layer root/'dec_01_att_key0' output: Data(name='dec_01_att_key0_output', shape=(None, 512))
layer root/'dec_01_att_key' output: Data(name='dec_01_att_key_output', shape=(None, 8, 64))
layer root/'dec_07_att_value0' output: Data(name='dec_07_att_value0_output', shape=(None, 512))
layer root/'dec_07_att_value' output: Data(name='dec_07_att_value_output', shape=(None, 8, 64))
layer root/'dec_03_att_value0' output: Data(name='dec_03_att_value0_output', shape=(None, 512))
layer root/'dec_03_att_value' output: Data(name='dec_03_att_value_output', shape=(None, 8, 64))
layer root/'dec_12_att_key0' output: Data(name='dec_12_att_key0_output', shape=(None, 512))
layer root/'dec_12_att_key' output: Data(name='dec_12_att_key_output', shape=(None, 8, 64))
layer root/'dec_06_att_key0' output: Data(name='dec_06_att_key0_output', shape=(None, 512))
layer root/'dec_06_att_key' output: Data(name='dec_06_att_key_output', shape=(None, 8, 64))
layer root/'dec_05_att_key0' output: Data(name='dec_05_att_key0_output', shape=(None, 512))
layer root/'dec_05_att_key' output: Data(name='dec_05_att_key_output', shape=(None, 8, 64))
layer root/'dec_06_att_value0' output: Data(name='dec_06_att_value0_output', shape=(None, 512))
layer root/'dec_06_att_value' output: Data(name='dec_06_att_value_output', shape=(None, 8, 64))
layer root/'dec_09_att_value0' output: Data(name='dec_09_att_value0_output', shape=(None, 512))
layer root/'dec_09_att_value' output: Data(name='dec_09_att_value_output', shape=(None, 8, 64))
layer root/'dec_01_att_value0' output: Data(name='dec_01_att_value0_output', shape=(None, 512))
layer root/'dec_01_att_value' output: Data(name='dec_01_att_value_output', shape=(None, 8, 64))
layer root/'dec_02_att_key0' output: Data(name='dec_02_att_key0_output', shape=(None, 512))
layer root/'dec_02_att_key' output: Data(name='dec_02_att_key_output', shape=(None, 8, 64))
layer root/'dec_09_att_key0' output: Data(name='dec_09_att_key0_output', shape=(None, 512))
layer root/'dec_09_att_key' output: Data(name='dec_09_att_key_output', shape=(None, 8, 64))
layer root/'dec_04_att_key0' output: Data(name='dec_04_att_key0_output', shape=(None, 512))
layer root/'dec_04_att_key' output: Data(name='dec_04_att_key_output', shape=(None, 8, 64))
layer root/'dec_11_att_value0' output: Data(name='dec_11_att_value0_output', shape=(None, 512))
layer root/'dec_11_att_value' output: Data(name='dec_11_att_value_output', shape=(None, 8, 64))
layer root/'dec_10_att_value0' output: Data(name='dec_10_att_value0_output', shape=(None, 512))
layer root/'dec_10_att_value' output: Data(name='dec_10_att_value_output', shape=(None, 8, 64))
layer root/'dec_12_att_value0' output: Data(name='dec_12_att_value0_output', shape=(None, 512))
layer root/'dec_12_att_value' output: Data(name='dec_12_att_value_output', shape=(None, 8, 64))
layer root/'dec_04_att_value0' output: Data(name='dec_04_att_value0_output', shape=(None, 512))
layer root/'dec_04_att_value' output: Data(name='dec_04_att_value_output', shape=(None, 8, 64))
layer root/'dec_10_att_key0' output: Data(name='dec_10_att_key0_output', shape=(None, 512))
layer root/'dec_10_att_key' output: Data(name='dec_10_att_key_output', shape=(None, 8, 64))
layer root/'dec_11_att_key0' output: Data(name='dec_11_att_key0_output', shape=(None, 512))
layer root/'dec_11_att_key' output: Data(name='dec_11_att_key_output', shape=(None, 8, 64))
layer root/'dec_02_att_value0' output: Data(name='dec_02_att_value0_output', shape=(None, 512))
layer root/'dec_02_att_value' output: Data(name='dec_02_att_value_output', shape=(None, 8, 64))
layer root/'dec_08_att_key0' output: Data(name='dec_08_att_key0_output', shape=(None, 512))
layer root/'dec_08_att_key' output: Data(name='dec_08_att_key_output', shape=(None, 8, 64))
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
TF: log_dir: /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/transformer/forward/tf_log_dir/prefix:test-clean-481-2019-07-02-10-35-10
Note: There are still these uninitialized variables: ['learning_rate:0']
att-weights epoch 481, step 0, max_size:classes 120, max_size:data 3392, mem_usage:GPU:0 1.0GB, num_seqs 1, 4.014 sec/step, elapsed 0:00:10, exp. remaining 0:43:07, complete 0.42%
att-weights epoch 481, step 1, max_size:classes 103, max_size:data 2512, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.318 sec/step, elapsed 0:00:12, exp. remaining 0:44:48, complete 0.46%
att-weights epoch 481, step 2, max_size:classes 101, max_size:data 2843, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.389 sec/step, elapsed 0:00:13, exp. remaining 0:46:28, complete 0.50%
att-weights epoch 481, step 3, max_size:classes 93, max_size:data 2961, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.266 sec/step, elapsed 0:00:15, exp. remaining 0:47:26, complete 0.53%
att-weights epoch 481, step 4, max_size:classes 97, max_size:data 3166, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.448 sec/step, elapsed 0:00:16, exp. remaining 0:48:44, complete 0.57%
att-weights epoch 481, step 5, max_size:classes 89, max_size:data 2542, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.136 sec/step, elapsed 0:00:18, exp. remaining 0:49:07, complete 0.61%
att-weights epoch 481, step 6, max_size:classes 90, max_size:data 3062, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.476 sec/step, elapsed 0:00:19, exp. remaining 0:50:20, complete 0.65%
att-weights epoch 481, step 7, max_size:classes 91, max_size:data 2915, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.203 sec/step, elapsed 0:00:21, exp. remaining 0:50:40, complete 0.69%
att-weights epoch 481, step 8, max_size:classes 101, max_size:data 2842, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.606 sec/step, elapsed 0:00:22, exp. remaining 0:51:55, complete 0.73%
att-weights epoch 481, step 9, max_size:classes 87, max_size:data 3496, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.893 sec/step, elapsed 0:00:24, exp. remaining 0:53:44, complete 0.76%
att-weights epoch 481, step 10, max_size:classes 86, max_size:data 2828, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.551 sec/step, elapsed 0:00:26, exp. remaining 0:54:44, complete 0.80%
att-weights epoch 481, step 11, max_size:classes 89, max_size:data 2713, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.213 sec/step, elapsed 0:00:27, exp. remaining 0:54:54, complete 0.84%
att-weights epoch 481, step 12, max_size:classes 84, max_size:data 2486, mem_usage:GPU:0 1.0GB, num_seqs 1, 5.931 sec/step, elapsed 0:00:33, exp. remaining 1:03:40, complete 0.88%
att-weights epoch 481, step 13, max_size:classes 81, max_size:data 2565, mem_usage:GPU:0 1.0GB, num_seqs 1, 5.136 sec/step, elapsed 0:00:38, exp. remaining 1:10:15, complete 0.92%
att-weights epoch 481, step 14, max_size:classes 79, max_size:data 2446, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.988 sec/step, elapsed 0:00:39, exp. remaining 1:09:08, complete 0.95%
att-weights epoch 481, step 15, max_size:classes 85, max_size:data 2387, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.046 sec/step, elapsed 0:00:41, exp. remaining 1:08:11, complete 0.99%
att-weights epoch 481, step 16, max_size:classes 93, max_size:data 3278, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.548 sec/step, elapsed 0:00:42, exp. remaining 1:08:07, complete 1.03%
att-weights epoch 481, step 17, max_size:classes 85, max_size:data 2334, mem_usage:GPU:0 1.0GB, num_seqs 1, 13.562 sec/step, elapsed 0:00:56, exp. remaining 1:26:35, complete 1.07%
att-weights epoch 481, step 18, max_size:classes 77, max_size:data 2858, mem_usage:GPU:0 1.0GB, num_seqs 1, 20.272 sec/step, elapsed 0:01:16, exp. remaining 1:53:45, complete 1.11%
att-weights epoch 481, step 19, max_size:classes 76, max_size:data 2102, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.016 sec/step, elapsed 0:01:17, exp. remaining 1:51:23, complete 1.15%
att-weights epoch 481, step 20, max_size:classes 78, max_size:data 2307, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.081 sec/step, elapsed 0:01:18, exp. remaining 1:49:15, complete 1.18%
att-weights epoch 481, step 21, max_size:classes 73, max_size:data 3289, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.321 sec/step, elapsed 0:01:19, exp. remaining 1:47:35, complete 1.22%
att-weights epoch 481, step 22, max_size:classes 67, max_size:data 2019, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.765 sec/step, elapsed 0:01:20, exp. remaining 1:45:17, complete 1.26%
att-weights epoch 481, step 23, max_size:classes 76, max_size:data 2599, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.986 sec/step, elapsed 0:01:21, exp. remaining 1:43:24, complete 1.30%
att-weights epoch 481, step 24, max_size:classes 77, max_size:data 2004, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.882 sec/step, elapsed 0:01:22, exp. remaining 1:41:30, complete 1.34%
att-weights epoch 481, step 25, max_size:classes 71, max_size:data 2550, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.006 sec/step, elapsed 0:01:23, exp. remaining 1:39:50, complete 1.37%
att-weights epoch 481, step 26, max_size:classes 67, max_size:data 2375, mem_usage:GPU:0 1.0GB, num_seqs 1, 9.807 sec/step, elapsed 0:01:33, exp. remaining 1:48:31, complete 1.41%
att-weights epoch 481, step 27, max_size:classes 75, max_size:data 3082, mem_usage:GPU:0 1.0GB, num_seqs 1, 5.943 sec/step, elapsed 0:01:39, exp. remaining 1:52:21, complete 1.45%
att-weights epoch 481, step 28, max_size:classes 83, max_size:data 2615, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.173 sec/step, elapsed 0:01:40, exp. remaining 1:50:43, complete 1.49%
att-weights epoch 481, step 29, max_size:classes 79, max_size:data 2612, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.079 sec/step, elapsed 0:01:41, exp. remaining 1:49:04, complete 1.53%
att-weights epoch 481, step 30, max_size:classes 70, max_size:data 2595, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.051 sec/step, elapsed 0:01:42, exp. remaining 1:47:28, complete 1.56%
att-weights epoch 481, step 31, max_size:classes 72, max_size:data 2221, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.907 sec/step, elapsed 0:01:43, exp. remaining 1:45:48, complete 1.60%
att-weights epoch 481, step 32, max_size:classes 67, max_size:data 2332, mem_usage:GPU:0 1.0GB, num_seqs 1, 4.021 sec/step, elapsed 0:01:47, exp. remaining 1:47:19, complete 1.64%
att-weights epoch 481, step 33, max_size:classes 67, max_size:data 2285, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.952 sec/step, elapsed 0:01:48, exp. remaining 1:45:46, complete 1.68%
att-weights epoch 481, step 34, max_size:classes 71, max_size:data 3162, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.241 sec/step, elapsed 0:01:49, exp. remaining 1:44:34, complete 1.72%
att-weights epoch 481, step 35, max_size:classes 76, max_size:data 2174, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.856 sec/step, elapsed 0:01:50, exp. remaining 1:43:03, complete 1.76%
att-weights epoch 481, step 36, max_size:classes 74, max_size:data 1888, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.789 sec/step, elapsed 0:01:51, exp. remaining 1:41:32, complete 1.79%
att-weights epoch 481, step 37, max_size:classes 83, max_size:data 2327, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.951 sec/step, elapsed 0:01:52, exp. remaining 1:40:14, complete 1.83%
att-weights epoch 481, step 38, max_size:classes 80, max_size:data 3005, mem_usage:GPU:0 1.0GB, num_seqs 1, 3.674 sec/step, elapsed 0:01:55, exp. remaining 1:41:22, complete 1.87%
att-weights epoch 481, step 39, max_size:classes 74, max_size:data 2229, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.049 sec/step, elapsed 0:01:56, exp. remaining 1:40:12, complete 1.91%
att-weights epoch 481, step 40, max_size:classes 73, max_size:data 2842, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.212 sec/step, elapsed 0:01:58, exp. remaining 1:39:13, complete 1.95%
att-weights epoch 481, step 41, max_size:classes 66, max_size:data 2753, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.075 sec/step, elapsed 0:01:59, exp. remaining 1:38:09, complete 1.98%
att-weights epoch 481, step 42, max_size:classes 72, max_size:data 2151, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.823 sec/step, elapsed 0:02:00, exp. remaining 1:36:55, complete 2.02%
att-weights epoch 481, step 43, max_size:classes 64, max_size:data 1948, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.870 sec/step, elapsed 0:02:00, exp. remaining 1:35:47, complete 2.06%
att-weights epoch 481, step 44, max_size:classes 72, max_size:data 2026, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.750 sec/step, elapsed 0:02:01, exp. remaining 1:34:35, complete 2.10%
att-weights epoch 481, step 45, max_size:classes 67, max_size:data 2086, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.837 sec/step, elapsed 0:02:02, exp. remaining 1:31:50, complete 2.18%
att-weights epoch 481, step 46, max_size:classes 74, max_size:data 2210, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.047 sec/step, elapsed 0:02:03, exp. remaining 1:30:59, complete 2.21%
att-weights epoch 481, step 47, max_size:classes 76, max_size:data 2217, mem_usage:GPU:0 1.0GB, num_seqs 1, 2.284 sec/step, elapsed 0:02:05, exp. remaining 1:31:03, complete 2.25%
att-weights epoch 481, step 48, max_size:classes 60, max_size:data 2455, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.182 sec/step, elapsed 0:02:07, exp. remaining 1:30:21, complete 2.29%
att-weights epoch 481, step 49, max_size:classes 65, max_size:data 2179, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.837 sec/step, elapsed 0:02:07, exp. remaining 1:29:25, complete 2.33%
att-weights epoch 481, step 50, max_size:classes 67, max_size:data 2038, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.763 sec/step, elapsed 0:02:08, exp. remaining 1:28:28, complete 2.37%
att-weights epoch 481, step 51, max_size:classes 64, max_size:data 2449, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.905 sec/step, elapsed 0:02:09, exp. remaining 1:27:38, complete 2.40%
att-weights epoch 481, step 52, max_size:classes 75, max_size:data 2810, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.083 sec/step, elapsed 0:02:10, exp. remaining 1:26:57, complete 2.44%
att-weights epoch 481, step 53, max_size:classes 69, max_size:data 2617, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.955 sec/step, elapsed 0:02:11, exp. remaining 1:26:13, complete 2.48%
att-weights epoch 481, step 54, max_size:classes 68, max_size:data 1992, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.421 sec/step, elapsed 0:02:13, exp. remaining 1:25:47, complete 2.52%
att-weights epoch 481, step 55, max_size:classes 66, max_size:data 1674, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.648 sec/step, elapsed 0:02:13, exp. remaining 1:24:53, complete 2.56%
att-weights epoch 481, step 56, max_size:classes 62, max_size:data 2091, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.817 sec/step, elapsed 0:02:14, exp. remaining 1:24:07, complete 2.60%
att-weights epoch 481, step 57, max_size:classes 64, max_size:data 1903, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.798 sec/step, elapsed 0:02:15, exp. remaining 1:23:21, complete 2.63%
att-weights epoch 481, step 58, max_size:classes 68, max_size:data 2119, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.831 sec/step, elapsed 0:02:16, exp. remaining 1:22:38, complete 2.67%
att-weights epoch 481, step 59, max_size:classes 65, max_size:data 2152, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.902 sec/step, elapsed 0:02:17, exp. remaining 1:20:49, complete 2.75%
att-weights epoch 481, step 60, max_size:classes 74, max_size:data 2149, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.945 sec/step, elapsed 0:02:17, exp. remaining 1:20:13, complete 2.79%
att-weights epoch 481, step 61, max_size:classes 64, max_size:data 2135, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.813 sec/step, elapsed 0:02:18, exp. remaining 1:18:29, complete 2.86%
att-weights epoch 481, step 62, max_size:classes 64, max_size:data 1653, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.625 sec/step, elapsed 0:02:19, exp. remaining 1:17:46, complete 2.90%
att-weights epoch 481, step 63, max_size:classes 56, max_size:data 2250, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.834 sec/step, elapsed 0:02:20, exp. remaining 1:17:11, complete 2.94%
att-weights epoch 481, step 64, max_size:classes 61, max_size:data 2448, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.907 sec/step, elapsed 0:02:21, exp. remaining 1:16:40, complete 2.98%
att-weights epoch 481, step 65, max_size:classes 66, max_size:data 2057, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.724 sec/step, elapsed 0:02:21, exp. remaining 1:16:03, complete 3.02%
att-weights epoch 481, step 66, max_size:classes 67, max_size:data 1853, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.760 sec/step, elapsed 0:02:22, exp. remaining 1:15:28, complete 3.05%
att-weights epoch 481, step 67, max_size:classes 68, max_size:data 2016, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.800 sec/step, elapsed 0:02:23, exp. remaining 1:14:56, complete 3.09%
att-weights epoch 481, step 68, max_size:classes 63, max_size:data 1971, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.146 sec/step, elapsed 0:02:24, exp. remaining 1:14:35, complete 3.13%
att-weights epoch 481, step 69, max_size:classes 67, max_size:data 2076, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.820 sec/step, elapsed 0:02:25, exp. remaining 1:13:10, complete 3.21%
att-weights epoch 481, step 70, max_size:classes 68, max_size:data 1758, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.092 sec/step, elapsed 0:02:26, exp. remaining 1:12:49, complete 3.24%
att-weights epoch 481, step 71, max_size:classes 67, max_size:data 2036, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.922 sec/step, elapsed 0:02:27, exp. remaining 1:12:23, complete 3.28%
att-weights epoch 481, step 72, max_size:classes 57, max_size:data 2015, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.782 sec/step, elapsed 0:02:28, exp. remaining 1:11:55, complete 3.32%
att-weights epoch 481, step 73, max_size:classes 65, max_size:data 1995, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.968 sec/step, elapsed 0:02:29, exp. remaining 1:11:32, complete 3.36%
att-weights epoch 481, step 74, max_size:classes 60, max_size:data 2719, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.981 sec/step, elapsed 0:02:30, exp. remaining 1:11:10, complete 3.40%
att-weights epoch 481, step 75, max_size:classes 55, max_size:data 2029, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.785 sec/step, elapsed 0:02:30, exp. remaining 1:10:43, complete 3.44%
att-weights epoch 481, step 76, max_size:classes 53, max_size:data 2237, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.799 sec/step, elapsed 0:02:31, exp. remaining 1:10:17, complete 3.47%
att-weights epoch 481, step 77, max_size:classes 57, max_size:data 2054, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.697 sec/step, elapsed 0:02:32, exp. remaining 1:09:02, complete 3.55%
att-weights epoch 481, step 78, max_size:classes 67, max_size:data 1864, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.117 sec/step, elapsed 0:02:33, exp. remaining 1:08:46, complete 3.59%
att-weights epoch 481, step 79, max_size:classes 62, max_size:data 1823, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.765 sec/step, elapsed 0:02:34, exp. remaining 1:08:21, complete 3.63%
att-weights epoch 481, step 80, max_size:classes 58, max_size:data 2002, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.817 sec/step, elapsed 0:02:35, exp. remaining 1:07:59, complete 3.66%
att-weights epoch 481, step 81, max_size:classes 61, max_size:data 1542, mem_usage:GPU:0 1.0GB, num_seqs 1, 3.569 sec/step, elapsed 0:02:38, exp. remaining 1:08:48, complete 3.70%
att-weights epoch 481, step 82, max_size:classes 59, max_size:data 2013, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.309 sec/step, elapsed 0:02:40, exp. remaining 1:07:54, complete 3.78%
att-weights epoch 481, step 83, max_size:classes 61, max_size:data 2002, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.773 sec/step, elapsed 0:02:40, exp. remaining 1:07:32, complete 3.82%
att-weights epoch 481, step 84, max_size:classes 59, max_size:data 2066, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.865 sec/step, elapsed 0:02:41, exp. remaining 1:06:30, complete 3.89%
att-weights epoch 481, step 85, max_size:classes 66, max_size:data 2001, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.985 sec/step, elapsed 0:02:42, exp. remaining 1:06:14, complete 3.93%
att-weights epoch 481, step 86, max_size:classes 62, max_size:data 1962, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.362 sec/step, elapsed 0:02:44, exp. remaining 1:06:07, complete 3.97%
att-weights epoch 481, step 87, max_size:classes 56, max_size:data 1629, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.783 sec/step, elapsed 0:02:44, exp. remaining 1:05:08, complete 4.05%
att-weights epoch 481, step 88, max_size:classes 61, max_size:data 2006, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.956 sec/step, elapsed 0:02:45, exp. remaining 1:04:15, complete 4.12%
att-weights epoch 481, step 89, max_size:classes 62, max_size:data 1623, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.762 sec/step, elapsed 0:02:46, exp. remaining 1:03:19, complete 4.20%
att-weights epoch 481, step 90, max_size:classes 62, max_size:data 2258, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.123 sec/step, elapsed 0:02:47, exp. remaining 1:02:34, complete 4.27%
att-weights epoch 481, step 91, max_size:classes 61, max_size:data 1813, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.667 sec/step, elapsed 0:02:49, exp. remaining 1:02:36, complete 4.31%
att-weights epoch 481, step 92, max_size:classes 69, max_size:data 2368, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.184 sec/step, elapsed 0:02:50, exp. remaining 1:02:27, complete 4.35%
att-weights epoch 481, step 93, max_size:classes 60, max_size:data 1797, mem_usage:GPU:0 1.0GB, num_seqs 2, 3.747 sec/step, elapsed 0:02:54, exp. remaining 1:02:41, complete 4.43%
att-weights epoch 481, step 94, max_size:classes 59, max_size:data 1987, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.816 sec/step, elapsed 0:02:55, exp. remaining 1:02:25, complete 4.47%
att-weights epoch 481, step 95, max_size:classes 63, max_size:data 2106, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.052 sec/step, elapsed 0:02:56, exp. remaining 1:01:41, complete 4.54%
att-weights epoch 481, step 96, max_size:classes 60, max_size:data 1992, mem_usage:GPU:0 1.0GB, num_seqs 2, 20.458 sec/step, elapsed 0:03:16, exp. remaining 1:07:39, complete 4.62%
att-weights epoch 481, step 97, max_size:classes 59, max_size:data 1992, mem_usage:GPU:0 1.0GB, num_seqs 2, 7.387 sec/step, elapsed 0:03:23, exp. remaining 1:09:00, complete 4.69%
att-weights epoch 481, step 98, max_size:classes 60, max_size:data 1981, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.311 sec/step, elapsed 0:03:25, exp. remaining 1:08:17, complete 4.77%
att-weights epoch 481, step 99, max_size:classes 53, max_size:data 1657, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.078 sec/step, elapsed 0:03:26, exp. remaining 1:08:04, complete 4.81%
att-weights epoch 481, step 100, max_size:classes 57, max_size:data 2047, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.011 sec/step, elapsed 0:03:27, exp. remaining 1:07:17, complete 4.89%
att-weights epoch 481, step 101, max_size:classes 55, max_size:data 2016, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.878 sec/step, elapsed 0:03:28, exp. remaining 1:06:28, complete 4.96%
att-weights epoch 481, step 102, max_size:classes 54, max_size:data 1798, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.370 sec/step, elapsed 0:03:30, exp. remaining 1:06:10, complete 5.04%
att-weights epoch 481, step 103, max_size:classes 64, max_size:data 2088, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.844 sec/step, elapsed 0:03:31, exp. remaining 1:05:23, complete 5.11%
att-weights epoch 481, step 104, max_size:classes 64, max_size:data 1641, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.299 sec/step, elapsed 0:03:32, exp. remaining 1:05:16, complete 5.15%
att-weights epoch 481, step 105, max_size:classes 57, max_size:data 1674, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.147 sec/step, elapsed 0:03:33, exp. remaining 1:05:07, complete 5.19%
att-weights epoch 481, step 106, max_size:classes 58, max_size:data 1832, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.223 sec/step, elapsed 0:03:35, exp. remaining 1:04:59, complete 5.23%
att-weights epoch 481, step 107, max_size:classes 60, max_size:data 1907, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.549 sec/step, elapsed 0:03:36, exp. remaining 1:04:27, complete 5.31%
att-weights epoch 481, step 108, max_size:classes 59, max_size:data 2126, mem_usage:GPU:0 1.0GB, num_seqs 1, 4.029 sec/step, elapsed 0:03:40, exp. remaining 1:04:40, complete 5.38%
att-weights epoch 481, step 109, max_size:classes 62, max_size:data 1775, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.172 sec/step, elapsed 0:03:41, exp. remaining 1:04:03, complete 5.46%
att-weights epoch 481, step 110, max_size:classes 57, max_size:data 1910, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.279 sec/step, elapsed 0:03:43, exp. remaining 1:03:29, complete 5.53%
att-weights epoch 481, step 111, max_size:classes 56, max_size:data 1723, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.400 sec/step, elapsed 0:03:44, exp. remaining 1:03:25, complete 5.57%
att-weights epoch 481, step 112, max_size:classes 60, max_size:data 2000, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.476 sec/step, elapsed 0:03:46, exp. remaining 1:03:22, complete 5.61%
att-weights epoch 481, step 113, max_size:classes 51, max_size:data 2034, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.720 sec/step, elapsed 0:03:46, exp. remaining 1:02:40, complete 5.69%
att-weights epoch 481, step 114, max_size:classes 53, max_size:data 1609, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.618 sec/step, elapsed 0:03:47, exp. remaining 1:01:58, complete 5.76%
att-weights epoch 481, step 115, max_size:classes 58, max_size:data 2099, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.823 sec/step, elapsed 0:03:48, exp. remaining 1:01:45, complete 5.80%
att-weights epoch 481, step 116, max_size:classes 57, max_size:data 1842, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.291 sec/step, elapsed 0:03:49, exp. remaining 1:01:15, complete 5.88%
att-weights epoch 481, step 117, max_size:classes 58, max_size:data 1721, mem_usage:GPU:0 1.0GB, num_seqs 2, 4.774 sec/step, elapsed 0:03:54, exp. remaining 1:01:40, complete 5.95%
att-weights epoch 481, step 118, max_size:classes 53, max_size:data 1701, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.060 sec/step, elapsed 0:03:55, exp. remaining 1:01:32, complete 5.99%
att-weights epoch 481, step 119, max_size:classes 60, max_size:data 1503, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.067 sec/step, elapsed 0:03:56, exp. remaining 1:01:23, complete 6.03%
att-weights epoch 481, step 120, max_size:classes 53, max_size:data 1565, mem_usage:GPU:0 1.0GB, num_seqs 1, 3.081 sec/step, elapsed 0:03:59, exp. remaining 1:01:22, complete 6.11%
att-weights epoch 481, step 121, max_size:classes 64, max_size:data 2540, mem_usage:GPU:0 1.0GB, num_seqs 1, 6.936 sec/step, elapsed 0:04:06, exp. remaining 1:02:19, complete 6.18%
att-weights epoch 481, step 122, max_size:classes 58, max_size:data 1800, mem_usage:GPU:0 1.0GB, num_seqs 2, 3.659 sec/step, elapsed 0:04:10, exp. remaining 1:02:25, complete 6.26%
att-weights epoch 481, step 123, max_size:classes 54, max_size:data 1685, mem_usage:GPU:0 1.0GB, num_seqs 2, 7.781 sec/step, elapsed 0:04:17, exp. remaining 1:03:32, complete 6.34%
att-weights epoch 481, step 124, max_size:classes 54, max_size:data 2083, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.030 sec/step, elapsed 0:04:18, exp. remaining 1:02:58, complete 6.41%
att-weights epoch 481, step 125, max_size:classes 55, max_size:data 1634, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.391 sec/step, elapsed 0:04:20, exp. remaining 1:02:55, complete 6.45%
att-weights epoch 481, step 126, max_size:classes 58, max_size:data 1637, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.382 sec/step, elapsed 0:04:21, exp. remaining 1:02:51, complete 6.49%
att-weights epoch 481, step 127, max_size:classes 57, max_size:data 2351, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.246 sec/step, elapsed 0:04:22, exp. remaining 1:02:22, complete 6.56%
att-weights epoch 481, step 128, max_size:classes 56, max_size:data 2246, mem_usage:GPU:0 1.0GB, num_seqs 1, 41.216 sec/step, elapsed 0:05:04, exp. remaining 1:11:15, complete 6.64%
att-weights epoch 481, step 129, max_size:classes 56, max_size:data 1765, mem_usage:GPU:0 1.0GB, num_seqs 2, 5.019 sec/step, elapsed 0:05:09, exp. remaining 1:11:33, complete 6.72%
att-weights epoch 481, step 130, max_size:classes 56, max_size:data 1672, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.420 sec/step, elapsed 0:05:10, exp. remaining 1:11:01, complete 6.79%
att-weights epoch 481, step 131, max_size:classes 59, max_size:data 1815, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.115 sec/step, elapsed 0:05:12, exp. remaining 1:10:38, complete 6.87%
att-weights epoch 481, step 132, max_size:classes 53, max_size:data 1795, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.367 sec/step, elapsed 0:05:14, exp. remaining 1:10:07, complete 6.95%
att-weights epoch 481, step 133, max_size:classes 54, max_size:data 1648, mem_usage:GPU:0 1.0GB, num_seqs 2, 16.609 sec/step, elapsed 0:05:30, exp. remaining 1:12:57, complete 7.02%
att-weights epoch 481, step 134, max_size:classes 51, max_size:data 1416, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.989 sec/step, elapsed 0:05:31, exp. remaining 1:12:20, complete 7.10%
att-weights epoch 481, step 135, max_size:classes 58, max_size:data 2251, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.478 sec/step, elapsed 0:05:33, exp. remaining 1:11:49, complete 7.18%
att-weights epoch 481, step 136, max_size:classes 61, max_size:data 1652, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.029 sec/step, elapsed 0:05:35, exp. remaining 1:11:26, complete 7.25%
att-weights epoch 481, step 137, max_size:classes 59, max_size:data 1673, mem_usage:GPU:0 1.0GB, num_seqs 2, 4.197 sec/step, elapsed 0:05:39, exp. remaining 1:11:31, complete 7.33%
att-weights epoch 481, step 138, max_size:classes 55, max_size:data 1680, mem_usage:GPU:0 1.0GB, num_seqs 2, 12.611 sec/step, elapsed 0:05:51, exp. remaining 1:13:21, complete 7.40%
att-weights epoch 481, step 139, max_size:classes 52, max_size:data 1404, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.029 sec/step, elapsed 0:05:53, exp. remaining 1:12:45, complete 7.48%
att-weights epoch 481, step 140, max_size:classes 53, max_size:data 1877, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.335 sec/step, elapsed 0:05:54, exp. remaining 1:12:14, complete 7.56%
att-weights epoch 481, step 141, max_size:classes 56, max_size:data 1855, mem_usage:GPU:0 1.0GB, num_seqs 2, 3.327 sec/step, elapsed 0:05:57, exp. remaining 1:12:08, complete 7.63%
att-weights epoch 481, step 142, max_size:classes 54, max_size:data 1711, mem_usage:GPU:0 1.0GB, num_seqs 2, 5.220 sec/step, elapsed 0:06:02, exp. remaining 1:12:24, complete 7.71%
att-weights epoch 481, step 143, max_size:classes 53, max_size:data 1996, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.707 sec/step, elapsed 0:06:04, exp. remaining 1:11:58, complete 7.79%
att-weights epoch 481, step 144, max_size:classes 52, max_size:data 1901, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.064 sec/step, elapsed 0:06:06, exp. remaining 1:11:36, complete 7.86%
att-weights epoch 481, step 145, max_size:classes 49, max_size:data 1739, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.169 sec/step, elapsed 0:06:07, exp. remaining 1:11:05, complete 7.94%
att-weights epoch 481, step 146, max_size:classes 53, max_size:data 1512, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.135 sec/step, elapsed 0:06:08, exp. remaining 1:10:34, complete 8.02%
att-weights epoch 481, step 147, max_size:classes 50, max_size:data 1536, mem_usage:GPU:0 1.0GB, num_seqs 2, 4.248 sec/step, elapsed 0:06:13, exp. remaining 1:11:01, complete 8.05%
att-weights epoch 481, step 148, max_size:classes 56, max_size:data 1901, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.203 sec/step, elapsed 0:06:14, exp. remaining 1:10:31, complete 8.13%
att-weights epoch 481, step 149, max_size:classes 52, max_size:data 1483, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.981 sec/step, elapsed 0:06:15, exp. remaining 1:09:59, complete 8.21%
att-weights epoch 481, step 150, max_size:classes 55, max_size:data 1605, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.056 sec/step, elapsed 0:06:16, exp. remaining 1:09:29, complete 8.28%
att-weights epoch 481, step 151, max_size:classes 56, max_size:data 1623, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.094 sec/step, elapsed 0:06:17, exp. remaining 1:08:59, complete 8.36%
att-weights epoch 481, step 152, max_size:classes 49, max_size:data 1577, mem_usage:GPU:0 1.0GB, num_seqs 2, 3.533 sec/step, elapsed 0:06:21, exp. remaining 1:08:56, complete 8.44%
att-weights epoch 481, step 153, max_size:classes 56, max_size:data 1658, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.106 sec/step, elapsed 0:06:22, exp. remaining 1:08:28, complete 8.51%
att-weights epoch 481, step 154, max_size:classes 50, max_size:data 1440, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.909 sec/step, elapsed 0:06:23, exp. remaining 1:07:58, complete 8.59%
att-weights epoch 481, step 155, max_size:classes 56, max_size:data 1855, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.266 sec/step, elapsed 0:06:24, exp. remaining 1:07:32, complete 8.66%
att-weights epoch 481, step 156, max_size:classes 50, max_size:data 2145, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.781 sec/step, elapsed 0:06:25, exp. remaining 1:07:01, complete 8.74%
att-weights epoch 481, step 157, max_size:classes 54, max_size:data 1890, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.019 sec/step, elapsed 0:06:26, exp. remaining 1:06:33, complete 8.82%
att-weights epoch 481, step 158, max_size:classes 52, max_size:data 1482, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.071 sec/step, elapsed 0:06:27, exp. remaining 1:06:26, complete 8.85%
att-weights epoch 481, step 159, max_size:classes 50, max_size:data 1423, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.134 sec/step, elapsed 0:06:28, exp. remaining 1:06:18, complete 8.89%
att-weights epoch 481, step 160, max_size:classes 52, max_size:data 1460, mem_usage:GPU:0 1.0GB, num_seqs 2, 3.341 sec/step, elapsed 0:06:31, exp. remaining 1:06:15, complete 8.97%
att-weights epoch 481, step 161, max_size:classes 49, max_size:data 1432, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.121 sec/step, elapsed 0:06:32, exp. remaining 1:05:50, complete 9.05%
att-weights epoch 481, step 162, max_size:classes 51, max_size:data 1789, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.017 sec/step, elapsed 0:06:33, exp. remaining 1:05:23, complete 9.12%
att-weights epoch 481, step 163, max_size:classes 48, max_size:data 1659, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.035 sec/step, elapsed 0:06:34, exp. remaining 1:04:58, complete 9.20%
att-weights epoch 481, step 164, max_size:classes 50, max_size:data 1451, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.043 sec/step, elapsed 0:06:35, exp. remaining 1:04:33, complete 9.27%
att-weights epoch 481, step 165, max_size:classes 52, max_size:data 1468, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.351 sec/step, elapsed 0:06:37, exp. remaining 1:04:11, complete 9.35%
att-weights epoch 481, step 166, max_size:classes 50, max_size:data 1677, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.295 sec/step, elapsed 0:06:38, exp. remaining 1:03:49, complete 9.43%
att-weights epoch 481, step 167, max_size:classes 50, max_size:data 1431, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.767 sec/step, elapsed 0:06:39, exp. remaining 1:03:22, complete 9.50%
att-weights epoch 481, step 168, max_size:classes 52, max_size:data 2065, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.792 sec/step, elapsed 0:06:40, exp. remaining 1:02:56, complete 9.58%
att-weights epoch 481, step 169, max_size:classes 48, max_size:data 1692, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.695 sec/step, elapsed 0:06:41, exp. remaining 1:02:39, complete 9.66%
att-weights epoch 481, step 170, max_size:classes 53, max_size:data 1797, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.049 sec/step, elapsed 0:06:42, exp. remaining 1:02:16, complete 9.73%
att-weights epoch 481, step 171, max_size:classes 46, max_size:data 1874, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.138 sec/step, elapsed 0:06:44, exp. remaining 1:01:55, complete 9.81%
att-weights epoch 481, step 172, max_size:classes 49, max_size:data 1573, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.162 sec/step, elapsed 0:06:45, exp. remaining 1:01:33, complete 9.89%
att-weights epoch 481, step 173, max_size:classes 50, max_size:data 1614, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.040 sec/step, elapsed 0:06:46, exp. remaining 1:01:11, complete 9.96%
att-weights epoch 481, step 174, max_size:classes 46, max_size:data 1732, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.979 sec/step, elapsed 0:06:47, exp. remaining 1:00:49, complete 10.04%
att-weights epoch 481, step 175, max_size:classes 47, max_size:data 1349, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.912 sec/step, elapsed 0:06:48, exp. remaining 1:00:27, complete 10.11%
att-weights epoch 481, step 176, max_size:classes 54, max_size:data 1529, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.973 sec/step, elapsed 0:06:49, exp. remaining 1:00:05, complete 10.19%
att-weights epoch 481, step 177, max_size:classes 55, max_size:data 1588, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.917 sec/step, elapsed 0:06:50, exp. remaining 0:59:43, complete 10.27%
att-weights epoch 481, step 178, max_size:classes 49, max_size:data 1504, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.911 sec/step, elapsed 0:06:50, exp. remaining 0:59:22, complete 10.34%
att-weights epoch 481, step 179, max_size:classes 49, max_size:data 1403, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.898 sec/step, elapsed 0:06:51, exp. remaining 0:59:00, complete 10.42%
att-weights epoch 481, step 180, max_size:classes 51, max_size:data 1370, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.912 sec/step, elapsed 0:06:52, exp. remaining 0:58:39, complete 10.50%
att-weights epoch 481, step 181, max_size:classes 44, max_size:data 1668, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.862 sec/step, elapsed 0:06:53, exp. remaining 0:58:18, complete 10.57%
att-weights epoch 481, step 182, max_size:classes 43, max_size:data 1783, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.919 sec/step, elapsed 0:06:54, exp. remaining 0:58:12, complete 10.61%
att-weights epoch 481, step 183, max_size:classes 46, max_size:data 1506, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.902 sec/step, elapsed 0:06:55, exp. remaining 0:57:51, complete 10.69%
att-weights epoch 481, step 184, max_size:classes 48, max_size:data 1390, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.990 sec/step, elapsed 0:06:56, exp. remaining 0:57:32, complete 10.76%
att-weights epoch 481, step 185, max_size:classes 47, max_size:data 1393, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.990 sec/step, elapsed 0:06:57, exp. remaining 0:57:13, complete 10.84%
att-weights epoch 481, step 186, max_size:classes 48, max_size:data 1573, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.967 sec/step, elapsed 0:06:58, exp. remaining 0:56:41, complete 10.95%
att-weights epoch 481, step 187, max_size:classes 50, max_size:data 1749, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.057 sec/step, elapsed 0:06:59, exp. remaining 0:56:23, complete 11.03%
att-weights epoch 481, step 188, max_size:classes 49, max_size:data 1502, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.984 sec/step, elapsed 0:07:00, exp. remaining 0:56:04, complete 11.11%
att-weights epoch 481, step 189, max_size:classes 50, max_size:data 1553, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.983 sec/step, elapsed 0:07:01, exp. remaining 0:55:46, complete 11.18%
att-weights epoch 481, step 190, max_size:classes 51, max_size:data 1460, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.018 sec/step, elapsed 0:07:02, exp. remaining 0:55:29, complete 11.26%
att-weights epoch 481, step 191, max_size:classes 46, max_size:data 2147, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.757 sec/step, elapsed 0:07:03, exp. remaining 0:55:10, complete 11.34%
att-weights epoch 481, step 192, max_size:classes 51, max_size:data 1499, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.998 sec/step, elapsed 0:07:04, exp. remaining 0:54:52, complete 11.41%
att-weights epoch 481, step 193, max_size:classes 50, max_size:data 1482, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.999 sec/step, elapsed 0:07:05, exp. remaining 0:54:35, complete 11.49%
att-weights epoch 481, step 194, max_size:classes 50, max_size:data 1441, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.242 sec/step, elapsed 0:07:06, exp. remaining 0:54:08, complete 11.60%
att-weights epoch 481, step 195, max_size:classes 48, max_size:data 1283, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.162 sec/step, elapsed 0:07:07, exp. remaining 0:53:53, complete 11.68%
att-weights epoch 481, step 196, max_size:classes 54, max_size:data 1322, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.010 sec/step, elapsed 0:07:08, exp. remaining 0:53:37, complete 11.76%
att-weights epoch 481, step 197, max_size:classes 47, max_size:data 1495, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.182 sec/step, elapsed 0:07:09, exp. remaining 0:53:22, complete 11.83%
att-weights epoch 481, step 198, max_size:classes 52, max_size:data 1335, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.820 sec/step, elapsed 0:07:10, exp. remaining 0:53:05, complete 11.91%
att-weights epoch 481, step 199, max_size:classes 49, max_size:data 1389, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.872 sec/step, elapsed 0:07:11, exp. remaining 0:52:37, complete 12.02%
att-weights epoch 481, step 200, max_size:classes 43, max_size:data 1367, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.884 sec/step, elapsed 0:07:12, exp. remaining 0:52:21, complete 12.10%
att-weights epoch 481, step 201, max_size:classes 46, max_size:data 1414, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.046 sec/step, elapsed 0:07:13, exp. remaining 0:52:06, complete 12.18%
att-weights epoch 481, step 202, max_size:classes 50, max_size:data 1438, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.867 sec/step, elapsed 0:07:14, exp. remaining 0:51:50, complete 12.25%
att-weights epoch 481, step 203, max_size:classes 54, max_size:data 1245, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.642 sec/step, elapsed 0:07:15, exp. remaining 0:51:29, complete 12.37%
att-weights epoch 481, step 204, max_size:classes 51, max_size:data 1381, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.932 sec/step, elapsed 0:07:16, exp. remaining 0:51:14, complete 12.44%
att-weights epoch 481, step 205, max_size:classes 45, max_size:data 1507, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.839 sec/step, elapsed 0:07:17, exp. remaining 0:50:58, complete 12.52%
att-weights epoch 481, step 206, max_size:classes 46, max_size:data 1390, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.944 sec/step, elapsed 0:07:18, exp. remaining 0:50:43, complete 12.60%
att-weights epoch 481, step 207, max_size:classes 44, max_size:data 1421, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.796 sec/step, elapsed 0:07:19, exp. remaining 0:50:28, complete 12.67%
att-weights epoch 481, step 208, max_size:classes 45, max_size:data 1315, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.252 sec/step, elapsed 0:07:20, exp. remaining 0:50:16, complete 12.75%
att-weights epoch 481, step 209, max_size:classes 41, max_size:data 1312, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.804 sec/step, elapsed 0:07:21, exp. remaining 0:50:01, complete 12.82%
att-weights epoch 481, step 210, max_size:classes 47, max_size:data 1507, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.870 sec/step, elapsed 0:07:22, exp. remaining 0:49:46, complete 12.90%
att-weights epoch 481, step 211, max_size:classes 48, max_size:data 1338, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.854 sec/step, elapsed 0:07:23, exp. remaining 0:49:32, complete 12.98%
att-weights epoch 481, step 212, max_size:classes 44, max_size:data 1296, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.266 sec/step, elapsed 0:07:24, exp. remaining 0:49:20, complete 13.05%
att-weights epoch 481, step 213, max_size:classes 44, max_size:data 1170, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.773 sec/step, elapsed 0:07:25, exp. remaining 0:49:05, complete 13.13%
att-weights epoch 481, step 214, max_size:classes 42, max_size:data 1627, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.065 sec/step, elapsed 0:07:26, exp. remaining 0:48:53, complete 13.21%
att-weights epoch 481, step 215, max_size:classes 44, max_size:data 1426, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.058 sec/step, elapsed 0:07:27, exp. remaining 0:48:40, complete 13.28%
att-weights epoch 481, step 216, max_size:classes 46, max_size:data 1608, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.960 sec/step, elapsed 0:07:28, exp. remaining 0:48:27, complete 13.36%
att-weights epoch 481, step 217, max_size:classes 42, max_size:data 1520, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.882 sec/step, elapsed 0:07:29, exp. remaining 0:48:05, complete 13.47%
att-weights epoch 481, step 218, max_size:classes 44, max_size:data 1699, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.937 sec/step, elapsed 0:07:30, exp. remaining 0:47:52, complete 13.55%
att-weights epoch 481, step 219, max_size:classes 43, max_size:data 1373, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.872 sec/step, elapsed 0:07:31, exp. remaining 0:47:39, complete 13.63%
att-weights epoch 481, step 220, max_size:classes 44, max_size:data 1546, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.905 sec/step, elapsed 0:07:31, exp. remaining 0:47:26, complete 13.70%
att-weights epoch 481, step 221, max_size:classes 41, max_size:data 1383, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.867 sec/step, elapsed 0:07:32, exp. remaining 0:47:13, complete 13.78%
att-weights epoch 481, step 222, max_size:classes 45, max_size:data 1464, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.910 sec/step, elapsed 0:07:33, exp. remaining 0:47:01, complete 13.85%
att-weights epoch 481, step 223, max_size:classes 52, max_size:data 1360, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.850 sec/step, elapsed 0:07:34, exp. remaining 0:46:48, complete 13.93%
att-weights epoch 481, step 224, max_size:classes 43, max_size:data 1365, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.954 sec/step, elapsed 0:07:35, exp. remaining 0:46:36, complete 14.01%
att-weights epoch 481, step 225, max_size:classes 46, max_size:data 1436, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.053 sec/step, elapsed 0:07:36, exp. remaining 0:46:25, complete 14.08%
att-weights epoch 481, step 226, max_size:classes 42, max_size:data 1276, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.351 sec/step, elapsed 0:07:37, exp. remaining 0:46:16, complete 14.16%
att-weights epoch 481, step 227, max_size:classes 51, max_size:data 1391, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.926 sec/step, elapsed 0:07:38, exp. remaining 0:46:04, complete 14.24%
att-weights epoch 481, step 228, max_size:classes 43, max_size:data 1285, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.759 sec/step, elapsed 0:07:39, exp. remaining 0:45:51, complete 14.31%
att-weights epoch 481, step 229, max_size:classes 39, max_size:data 1506, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.805 sec/step, elapsed 0:07:40, exp. remaining 0:45:39, complete 14.39%
att-weights epoch 481, step 230, max_size:classes 43, max_size:data 1460, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.890 sec/step, elapsed 0:07:41, exp. remaining 0:45:27, complete 14.47%
att-weights epoch 481, step 231, max_size:classes 45, max_size:data 1775, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.014 sec/step, elapsed 0:07:42, exp. remaining 0:45:16, complete 14.54%
att-weights epoch 481, step 232, max_size:classes 41, max_size:data 1517, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.917 sec/step, elapsed 0:07:43, exp. remaining 0:45:05, complete 14.62%
att-weights epoch 481, step 233, max_size:classes 42, max_size:data 1468, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.916 sec/step, elapsed 0:07:44, exp. remaining 0:44:54, complete 14.69%
att-weights epoch 481, step 234, max_size:classes 43, max_size:data 1476, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.910 sec/step, elapsed 0:07:45, exp. remaining 0:44:35, complete 14.81%
att-weights epoch 481, step 235, max_size:classes 41, max_size:data 1247, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.993 sec/step, elapsed 0:07:46, exp. remaining 0:44:24, complete 14.89%
att-weights epoch 481, step 236, max_size:classes 46, max_size:data 1579, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.434 sec/step, elapsed 0:07:47, exp. remaining 0:44:17, complete 14.96%
att-weights epoch 481, step 237, max_size:classes 45, max_size:data 1345, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.010 sec/step, elapsed 0:07:48, exp. remaining 0:43:59, complete 15.08%
att-weights epoch 481, step 238, max_size:classes 45, max_size:data 1144, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.971 sec/step, elapsed 0:07:49, exp. remaining 0:43:41, complete 15.19%
att-weights epoch 481, step 239, max_size:classes 39, max_size:data 1546, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.973 sec/step, elapsed 0:07:50, exp. remaining 0:43:23, complete 15.31%
att-weights epoch 481, step 240, max_size:classes 39, max_size:data 1294, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.760 sec/step, elapsed 0:07:51, exp. remaining 0:43:04, complete 15.42%
att-weights epoch 481, step 241, max_size:classes 42, max_size:data 1363, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.218 sec/step, elapsed 0:07:52, exp. remaining 0:42:56, complete 15.50%
att-weights epoch 481, step 242, max_size:classes 42, max_size:data 1394, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.079 sec/step, elapsed 0:07:53, exp. remaining 0:42:47, complete 15.57%
att-weights epoch 481, step 243, max_size:classes 42, max_size:data 1317, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.286 sec/step, elapsed 0:07:54, exp. remaining 0:42:31, complete 15.69%
att-weights epoch 481, step 244, max_size:classes 45, max_size:data 1466, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.021 sec/step, elapsed 0:07:55, exp. remaining 0:42:15, complete 15.80%
att-weights epoch 481, step 245, max_size:classes 47, max_size:data 1727, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.022 sec/step, elapsed 0:07:56, exp. remaining 0:41:59, complete 15.92%
att-weights epoch 481, step 246, max_size:classes 42, max_size:data 1327, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.687 sec/step, elapsed 0:07:58, exp. remaining 0:41:53, complete 15.99%
att-weights epoch 481, step 247, max_size:classes 48, max_size:data 1154, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.835 sec/step, elapsed 0:08:00, exp. remaining 0:41:49, complete 16.07%
att-weights epoch 481, step 248, max_size:classes 39, max_size:data 1312, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.264 sec/step, elapsed 0:08:01, exp. remaining 0:41:34, complete 16.18%
att-weights epoch 481, step 249, max_size:classes 42, max_size:data 1285, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.377 sec/step, elapsed 0:08:03, exp. remaining 0:41:20, complete 16.30%
att-weights epoch 481, step 250, max_size:classes 38, max_size:data 1150, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.913 sec/step, elapsed 0:08:03, exp. remaining 0:41:11, complete 16.37%
att-weights epoch 481, step 251, max_size:classes 42, max_size:data 1729, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.991 sec/step, elapsed 0:08:04, exp. remaining 0:40:56, complete 16.49%
att-weights epoch 481, step 252, max_size:classes 43, max_size:data 1320, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.710 sec/step, elapsed 0:08:06, exp. remaining 0:40:51, complete 16.56%
att-weights epoch 481, step 253, max_size:classes 41, max_size:data 1274, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.706 sec/step, elapsed 0:08:08, exp. remaining 0:40:46, complete 16.64%
att-weights epoch 481, step 254, max_size:classes 41, max_size:data 1183, mem_usage:GPU:0 1.0GB, num_seqs 3, 2.721 sec/step, elapsed 0:08:11, exp. remaining 0:40:53, complete 16.68%
att-weights epoch 481, step 255, max_size:classes 45, max_size:data 1491, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.030 sec/step, elapsed 0:08:12, exp. remaining 0:40:51, complete 16.72%
att-weights epoch 481, step 256, max_size:classes 36, max_size:data 1407, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.176 sec/step, elapsed 0:08:13, exp. remaining 0:40:37, complete 16.83%
att-weights epoch 481, step 257, max_size:classes 43, max_size:data 1177, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.126 sec/step, elapsed 0:08:14, exp. remaining 0:40:23, complete 16.95%
att-weights epoch 481, step 258, max_size:classes 40, max_size:data 1216, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.394 sec/step, elapsed 0:08:15, exp. remaining 0:40:10, complete 17.06%
att-weights epoch 481, step 259, max_size:classes 40, max_size:data 1432, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.905 sec/step, elapsed 0:08:16, exp. remaining 0:40:01, complete 17.14%
att-weights epoch 481, step 260, max_size:classes 35, max_size:data 1278, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.366 sec/step, elapsed 0:08:18, exp. remaining 0:39:55, complete 17.21%
att-weights epoch 481, step 261, max_size:classes 39, max_size:data 1517, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.953 sec/step, elapsed 0:08:19, exp. remaining 0:39:47, complete 17.29%
att-weights epoch 481, step 262, max_size:classes 40, max_size:data 1480, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.089 sec/step, elapsed 0:08:20, exp. remaining 0:39:33, complete 17.40%
att-weights epoch 481, step 263, max_size:classes 40, max_size:data 1284, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.603 sec/step, elapsed 0:08:20, exp. remaining 0:39:17, complete 17.52%
att-weights epoch 481, step 264, max_size:classes 39, max_size:data 2057, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.926 sec/step, elapsed 0:08:21, exp. remaining 0:39:09, complete 17.60%
att-weights epoch 481, step 265, max_size:classes 42, max_size:data 1203, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.239 sec/step, elapsed 0:08:22, exp. remaining 0:39:02, complete 17.67%
att-weights epoch 481, step 266, max_size:classes 43, max_size:data 1256, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.321 sec/step, elapsed 0:08:24, exp. remaining 0:38:50, complete 17.79%
att-weights epoch 481, step 267, max_size:classes 40, max_size:data 1202, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.195 sec/step, elapsed 0:08:25, exp. remaining 0:38:37, complete 17.90%
att-weights epoch 481, step 268, max_size:classes 40, max_size:data 1392, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.065 sec/step, elapsed 0:08:26, exp. remaining 0:38:24, complete 18.02%
att-weights epoch 481, step 269, max_size:classes 46, max_size:data 1241, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.126 sec/step, elapsed 0:08:27, exp. remaining 0:38:12, complete 18.13%
att-weights epoch 481, step 270, max_size:classes 43, max_size:data 1740, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.485 sec/step, elapsed 0:08:29, exp. remaining 0:38:01, complete 18.24%
att-weights epoch 481, step 271, max_size:classes 39, max_size:data 977, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.023 sec/step, elapsed 0:08:30, exp. remaining 0:37:48, complete 18.36%
att-weights epoch 481, step 272, max_size:classes 38, max_size:data 1255, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.201 sec/step, elapsed 0:08:31, exp. remaining 0:37:36, complete 18.47%
att-weights epoch 481, step 273, max_size:classes 40, max_size:data 1530, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.769 sec/step, elapsed 0:08:32, exp. remaining 0:37:28, complete 18.55%
att-weights epoch 481, step 274, max_size:classes 35, max_size:data 1362, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.050 sec/step, elapsed 0:08:33, exp. remaining 0:37:16, complete 18.66%
att-weights epoch 481, step 275, max_size:classes 41, max_size:data 1170, mem_usage:GPU:0 1.0GB, num_seqs 3, 2.680 sec/step, elapsed 0:08:35, exp. remaining 0:37:16, complete 18.74%
att-weights epoch 481, step 276, max_size:classes 42, max_size:data 1069, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.630 sec/step, elapsed 0:08:37, exp. remaining 0:37:06, complete 18.85%
att-weights epoch 481, step 277, max_size:classes 37, max_size:data 1234, mem_usage:GPU:0 1.0GB, num_seqs 3, 3.406 sec/step, elapsed 0:08:40, exp. remaining 0:37:10, complete 18.93%
att-weights epoch 481, step 278, max_size:classes 42, max_size:data 1302, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.227 sec/step, elapsed 0:08:42, exp. remaining 0:36:59, complete 19.05%
att-weights epoch 481, step 279, max_size:classes 42, max_size:data 1146, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.085 sec/step, elapsed 0:08:43, exp. remaining 0:36:47, complete 19.16%
att-weights epoch 481, step 280, max_size:classes 42, max_size:data 1166, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.030 sec/step, elapsed 0:08:44, exp. remaining 0:36:35, complete 19.27%
att-weights epoch 481, step 281, max_size:classes 39, max_size:data 1190, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.151 sec/step, elapsed 0:08:45, exp. remaining 0:36:29, complete 19.35%
att-weights epoch 481, step 282, max_size:classes 39, max_size:data 1343, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.800 sec/step, elapsed 0:08:46, exp. remaining 0:36:22, complete 19.43%
att-weights epoch 481, step 283, max_size:classes 41, max_size:data 1094, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.016 sec/step, elapsed 0:08:47, exp. remaining 0:36:10, complete 19.54%
att-weights epoch 481, step 284, max_size:classes 38, max_size:data 1770, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.942 sec/step, elapsed 0:08:48, exp. remaining 0:36:03, complete 19.62%
att-weights epoch 481, step 285, max_size:classes 38, max_size:data 1250, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.135 sec/step, elapsed 0:08:49, exp. remaining 0:35:52, complete 19.73%
att-weights epoch 481, step 286, max_size:classes 40, max_size:data 1416, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.028 sec/step, elapsed 0:08:50, exp. remaining 0:35:41, complete 19.85%
att-weights epoch 481, step 287, max_size:classes 39, max_size:data 1170, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.246 sec/step, elapsed 0:08:51, exp. remaining 0:35:31, complete 19.96%
att-weights epoch 481, step 288, max_size:classes 40, max_size:data 1259, mem_usage:GPU:0 1.0GB, num_seqs 3, 3.314 sec/step, elapsed 0:08:54, exp. remaining 0:35:29, complete 20.08%
att-weights epoch 481, step 289, max_size:classes 37, max_size:data 1309, mem_usage:GPU:0 1.0GB, num_seqs 3, 0.984 sec/step, elapsed 0:08:55, exp. remaining 0:35:17, complete 20.19%
att-weights epoch 481, step 290, max_size:classes 39, max_size:data 1630, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.267 sec/step, elapsed 0:08:57, exp. remaining 0:35:07, complete 20.31%
att-weights epoch 481, step 291, max_size:classes 36, max_size:data 1397, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.227 sec/step, elapsed 0:08:58, exp. remaining 0:35:02, complete 20.38%
att-weights epoch 481, step 292, max_size:classes 37, max_size:data 994, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.191 sec/step, elapsed 0:08:59, exp. remaining 0:34:52, complete 20.50%
att-weights epoch 481, step 293, max_size:classes 41, max_size:data 1528, mem_usage:GPU:0 1.0GB, num_seqs 2, 3.580 sec/step, elapsed 0:09:03, exp. remaining 0:34:51, complete 20.61%
att-weights epoch 481, step 294, max_size:classes 35, max_size:data 1141, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.031 sec/step, elapsed 0:09:04, exp. remaining 0:34:41, complete 20.73%
att-weights epoch 481, step 295, max_size:classes 37, max_size:data 1129, mem_usage:GPU:0 1.0GB, num_seqs 3, 2.571 sec/step, elapsed 0:09:06, exp. remaining 0:34:36, complete 20.84%
att-weights epoch 481, step 296, max_size:classes 36, max_size:data 1057, mem_usage:GPU:0 1.0GB, num_seqs 3, 0.976 sec/step, elapsed 0:09:07, exp. remaining 0:34:30, complete 20.92%
att-weights epoch 481, step 297, max_size:classes 38, max_size:data 1303, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.318 sec/step, elapsed 0:09:08, exp. remaining 0:34:26, complete 20.99%
att-weights epoch 481, step 298, max_size:classes 35, max_size:data 1184, mem_usage:GPU:0 1.0GB, num_seqs 3, 0.966 sec/step, elapsed 0:09:09, exp. remaining 0:34:20, complete 21.07%
att-weights epoch 481, step 299, max_size:classes 38, max_size:data 1114, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.052 sec/step, elapsed 0:09:10, exp. remaining 0:34:10, complete 21.18%
att-weights epoch 481, step 300, max_size:classes 38, max_size:data 1400, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.893 sec/step, elapsed 0:09:11, exp. remaining 0:34:04, complete 21.26%
att-weights epoch 481, step 301, max_size:classes 37, max_size:data 1191, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.149 sec/step, elapsed 0:09:13, exp. remaining 0:33:59, complete 21.34%
att-weights epoch 481, step 302, max_size:classes 36, max_size:data 1304, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.317 sec/step, elapsed 0:09:14, exp. remaining 0:33:50, complete 21.45%
att-weights epoch 481, step 303, max_size:classes 39, max_size:data 1036, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.059 sec/step, elapsed 0:09:15, exp. remaining 0:33:40, complete 21.56%
att-weights epoch 481, step 304, max_size:classes 37, max_size:data 1122, mem_usage:GPU:0 1.0GB, num_seqs 3, 5.436 sec/step, elapsed 0:09:20, exp. remaining 0:33:46, complete 21.68%
att-weights epoch 481, step 305, max_size:classes 39, max_size:data 1180, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.789 sec/step, elapsed 0:09:21, exp. remaining 0:33:35, complete 21.79%
att-weights epoch 481, step 306, max_size:classes 40, max_size:data 1468, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.019 sec/step, elapsed 0:09:22, exp. remaining 0:33:25, complete 21.91%
att-weights epoch 481, step 307, max_size:classes 38, max_size:data 1459, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.896 sec/step, elapsed 0:09:23, exp. remaining 0:33:19, complete 21.98%
att-weights epoch 481, step 308, max_size:classes 41, max_size:data 1263, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.291 sec/step, elapsed 0:09:24, exp. remaining 0:33:15, complete 22.06%
att-weights epoch 481, step 309, max_size:classes 34, max_size:data 1165, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.859 sec/step, elapsed 0:09:25, exp. remaining 0:33:05, complete 22.18%
att-weights epoch 481, step 310, max_size:classes 34, max_size:data 1363, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.017 sec/step, elapsed 0:09:26, exp. remaining 0:32:55, complete 22.29%
att-weights epoch 481, step 311, max_size:classes 35, max_size:data 1114, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.148 sec/step, elapsed 0:09:27, exp. remaining 0:32:46, complete 22.40%
att-weights epoch 481, step 312, max_size:classes 37, max_size:data 1214, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.264 sec/step, elapsed 0:09:29, exp. remaining 0:32:38, complete 22.52%
att-weights epoch 481, step 313, max_size:classes 36, max_size:data 1099, mem_usage:GPU:0 1.0GB, num_seqs 3, 7.572 sec/step, elapsed 0:09:36, exp. remaining 0:32:51, complete 22.63%
att-weights epoch 481, step 314, max_size:classes 35, max_size:data 1193, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.205 sec/step, elapsed 0:09:37, exp. remaining 0:32:42, complete 22.75%
att-weights epoch 481, step 315, max_size:classes 33, max_size:data 1109, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.083 sec/step, elapsed 0:09:39, exp. remaining 0:32:29, complete 22.90%
att-weights epoch 481, step 316, max_size:classes 38, max_size:data 1266, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.950 sec/step, elapsed 0:09:39, exp. remaining 0:32:19, complete 23.02%
att-weights epoch 481, step 317, max_size:classes 37, max_size:data 1411, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.954 sec/step, elapsed 0:09:40, exp. remaining 0:32:10, complete 23.13%
att-weights epoch 481, step 318, max_size:classes 40, max_size:data 1290, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.371 sec/step, elapsed 0:09:42, exp. remaining 0:32:02, complete 23.24%
att-weights epoch 481, step 319, max_size:classes 33, max_size:data 1018, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.066 sec/step, elapsed 0:09:43, exp. remaining 0:31:54, complete 23.36%
att-weights epoch 481, step 320, max_size:classes 42, max_size:data 1142, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.736 sec/step, elapsed 0:09:45, exp. remaining 0:31:47, complete 23.47%
att-weights epoch 481, step 321, max_size:classes 35, max_size:data 1074, mem_usage:GPU:0 1.0GB, num_seqs 3, 0.992 sec/step, elapsed 0:09:46, exp. remaining 0:31:38, complete 23.59%
att-weights epoch 481, step 322, max_size:classes 33, max_size:data 1045, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.148 sec/step, elapsed 0:09:47, exp. remaining 0:31:30, complete 23.70%
att-weights epoch 481, step 323, max_size:classes 36, max_size:data 1122, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.092 sec/step, elapsed 0:09:48, exp. remaining 0:31:25, complete 23.78%
att-weights epoch 481, step 324, max_size:classes 36, max_size:data 970, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.274 sec/step, elapsed 0:09:49, exp. remaining 0:31:14, complete 23.93%
att-weights epoch 481, step 325, max_size:classes 34, max_size:data 1202, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.097 sec/step, elapsed 0:09:50, exp. remaining 0:31:05, complete 24.05%
att-weights epoch 481, step 326, max_size:classes 35, max_size:data 1115, mem_usage:GPU:0 1.0GB, num_seqs 3, 4.915 sec/step, elapsed 0:09:55, exp. remaining 0:31:09, complete 24.16%
att-weights epoch 481, step 327, max_size:classes 34, max_size:data 1098, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.458 sec/step, elapsed 0:09:57, exp. remaining 0:31:02, complete 24.27%
att-weights epoch 481, step 328, max_size:classes 33, max_size:data 1092, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.110 sec/step, elapsed 0:09:58, exp. remaining 0:30:50, complete 24.43%
att-weights epoch 481, step 329, max_size:classes 34, max_size:data 1187, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.851 sec/step, elapsed 0:10:00, exp. remaining 0:30:41, complete 24.58%
att-weights epoch 481, step 330, max_size:classes 33, max_size:data 983, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.226 sec/step, elapsed 0:10:01, exp. remaining 0:30:33, complete 24.69%
att-weights epoch 481, step 331, max_size:classes 33, max_size:data 1132, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.396 sec/step, elapsed 0:10:02, exp. remaining 0:30:22, complete 24.85%
att-weights epoch 481, step 332, max_size:classes 43, max_size:data 1543, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.065 sec/step, elapsed 0:10:03, exp. remaining 0:30:11, complete 25.00%
att-weights epoch 481, step 333, max_size:classes 38, max_size:data 925, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.371 sec/step, elapsed 0:10:05, exp. remaining 0:30:04, complete 25.11%
att-weights epoch 481, step 334, max_size:classes 38, max_size:data 1012, mem_usage:GPU:0 1.0GB, num_seqs 3, 7.762 sec/step, elapsed 0:10:12, exp. remaining 0:30:16, complete 25.23%
att-weights epoch 481, step 335, max_size:classes 31, max_size:data 1095, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.049 sec/step, elapsed 0:10:13, exp. remaining 0:30:08, complete 25.34%
att-weights epoch 481, step 336, max_size:classes 34, max_size:data 1155, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.164 sec/step, elapsed 0:10:15, exp. remaining 0:30:00, complete 25.46%
att-weights epoch 481, step 337, max_size:classes 34, max_size:data 968, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.165 sec/step, elapsed 0:10:16, exp. remaining 0:29:49, complete 25.61%
att-weights epoch 481, step 338, max_size:classes 35, max_size:data 938, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.278 sec/step, elapsed 0:10:17, exp. remaining 0:29:39, complete 25.76%
att-weights epoch 481, step 339, max_size:classes 35, max_size:data 1149, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.115 sec/step, elapsed 0:10:18, exp. remaining 0:29:31, complete 25.88%
att-weights epoch 481, step 340, max_size:classes 33, max_size:data 957, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.215 sec/step, elapsed 0:10:19, exp. remaining 0:29:24, complete 25.99%
att-weights epoch 481, step 341, max_size:classes 31, max_size:data 961, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.274 sec/step, elapsed 0:10:21, exp. remaining 0:29:18, complete 26.11%
att-weights epoch 481, step 342, max_size:classes 37, max_size:data 1125, mem_usage:GPU:0 1.0GB, num_seqs 3, 0.988 sec/step, elapsed 0:10:22, exp. remaining 0:29:07, complete 26.26%
att-weights epoch 481, step 343, max_size:classes 35, max_size:data 1203, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.059 sec/step, elapsed 0:10:23, exp. remaining 0:28:59, complete 26.37%
att-weights epoch 481, step 344, max_size:classes 30, max_size:data 1177, mem_usage:GPU:0 1.0GB, num_seqs 3, 0.942 sec/step, elapsed 0:10:24, exp. remaining 0:28:52, complete 26.49%
att-weights epoch 481, step 345, max_size:classes 32, max_size:data 1036, mem_usage:GPU:0 1.0GB, num_seqs 3, 0.988 sec/step, elapsed 0:10:25, exp. remaining 0:28:44, complete 26.60%
att-weights epoch 481, step 346, max_size:classes 30, max_size:data 976, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.460 sec/step, elapsed 0:10:26, exp. remaining 0:28:41, complete 26.68%
att-weights epoch 481, step 347, max_size:classes 32, max_size:data 988, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.334 sec/step, elapsed 0:10:27, exp. remaining 0:28:32, complete 26.83%
att-weights epoch 481, step 348, max_size:classes 33, max_size:data 1056, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.216 sec/step, elapsed 0:10:29, exp. remaining 0:28:22, complete 26.98%
att-weights epoch 481, step 349, max_size:classes 33, max_size:data 1128, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.007 sec/step, elapsed 0:10:30, exp. remaining 0:28:11, complete 27.14%
att-weights epoch 481, step 350, max_size:classes 32, max_size:data 1090, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.203 sec/step, elapsed 0:10:31, exp. remaining 0:28:02, complete 27.29%
att-weights epoch 481, step 351, max_size:classes 34, max_size:data 994, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.499 sec/step, elapsed 0:10:32, exp. remaining 0:27:56, complete 27.40%
att-weights epoch 481, step 352, max_size:classes 30, max_size:data 1058, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.127 sec/step, elapsed 0:10:33, exp. remaining 0:27:49, complete 27.52%
att-weights epoch 481, step 353, max_size:classes 35, max_size:data 1019, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.041 sec/step, elapsed 0:10:35, exp. remaining 0:27:39, complete 27.67%
att-weights epoch 481, step 354, max_size:classes 34, max_size:data 1140, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.969 sec/step, elapsed 0:10:36, exp. remaining 0:27:32, complete 27.82%
att-weights epoch 481, step 355, max_size:classes 31, max_size:data 1532, mem_usage:GPU:0 1.0GB, num_seqs 2, 4.296 sec/step, elapsed 0:10:41, exp. remaining 0:27:34, complete 27.94%
att-weights epoch 481, step 356, max_size:classes 34, max_size:data 878, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.472 sec/step, elapsed 0:10:42, exp. remaining 0:27:28, complete 28.05%
att-weights epoch 481, step 357, max_size:classes 35, max_size:data 949, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.383 sec/step, elapsed 0:10:44, exp. remaining 0:27:22, complete 28.17%
att-weights epoch 481, step 358, max_size:classes 33, max_size:data 920, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.480 sec/step, elapsed 0:10:45, exp. remaining 0:27:14, complete 28.32%
att-weights epoch 481, step 359, max_size:classes 29, max_size:data 964, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.599 sec/step, elapsed 0:10:47, exp. remaining 0:27:08, complete 28.44%
att-weights epoch 481, step 360, max_size:classes 33, max_size:data 1135, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.260 sec/step, elapsed 0:10:48, exp. remaining 0:27:02, complete 28.55%
att-weights epoch 481, step 361, max_size:classes 31, max_size:data 1057, mem_usage:GPU:0 1.0GB, num_seqs 3, 6.136 sec/step, elapsed 0:10:54, exp. remaining 0:27:09, complete 28.66%
att-weights epoch 481, step 362, max_size:classes 36, max_size:data 970, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.479 sec/step, elapsed 0:10:56, exp. remaining 0:27:00, complete 28.82%
att-weights epoch 481, step 363, max_size:classes 34, max_size:data 844, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.159 sec/step, elapsed 0:10:57, exp. remaining 0:26:54, complete 28.93%
att-weights epoch 481, step 364, max_size:classes 30, max_size:data 1150, mem_usage:GPU:0 1.0GB, num_seqs 3, 5.595 sec/step, elapsed 0:11:02, exp. remaining 0:26:59, complete 29.05%
att-weights epoch 481, step 365, max_size:classes 31, max_size:data 956, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.133 sec/step, elapsed 0:11:04, exp. remaining 0:26:53, complete 29.16%
att-weights epoch 481, step 366, max_size:classes 31, max_size:data 1058, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.239 sec/step, elapsed 0:11:05, exp. remaining 0:26:47, complete 29.27%
att-weights epoch 481, step 367, max_size:classes 30, max_size:data 875, mem_usage:GPU:0 1.0GB, num_seqs 4, 2.117 sec/step, elapsed 0:11:07, exp. remaining 0:26:40, complete 29.43%
att-weights epoch 481, step 368, max_size:classes 26, max_size:data 1004, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.315 sec/step, elapsed 0:11:08, exp. remaining 0:26:31, complete 29.58%
att-weights epoch 481, step 369, max_size:classes 28, max_size:data 1027, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.159 sec/step, elapsed 0:11:09, exp. remaining 0:26:23, complete 29.73%
att-weights epoch 481, step 370, max_size:classes 34, max_size:data 1008, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.400 sec/step, elapsed 0:11:11, exp. remaining 0:26:17, complete 29.85%
att-weights epoch 481, step 371, max_size:classes 31, max_size:data 905, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.569 sec/step, elapsed 0:11:12, exp. remaining 0:26:12, complete 29.96%
att-weights epoch 481, step 372, max_size:classes 31, max_size:data 821, mem_usage:GPU:0 1.0GB, num_seqs 3, 14.743 sec/step, elapsed 0:11:27, exp. remaining 0:26:35, complete 30.11%
att-weights epoch 481, step 373, max_size:classes 31, max_size:data 1045, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.219 sec/step, elapsed 0:11:28, exp. remaining 0:26:26, complete 30.27%
att-weights epoch 481, step 374, max_size:classes 30, max_size:data 988, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.306 sec/step, elapsed 0:11:30, exp. remaining 0:26:21, complete 30.38%
att-weights epoch 481, step 375, max_size:classes 31, max_size:data 1131, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.095 sec/step, elapsed 0:11:31, exp. remaining 0:26:15, complete 30.50%
att-weights epoch 481, step 376, max_size:classes 31, max_size:data 854, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.524 sec/step, elapsed 0:11:32, exp. remaining 0:26:10, complete 30.61%
att-weights epoch 481, step 377, max_size:classes 30, max_size:data 931, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.213 sec/step, elapsed 0:11:33, exp. remaining 0:26:04, complete 30.73%
att-weights epoch 481, step 378, max_size:classes 33, max_size:data 861, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.228 sec/step, elapsed 0:11:35, exp. remaining 0:25:56, complete 30.88%
att-weights epoch 481, step 379, max_size:classes 32, max_size:data 1067, mem_usage:GPU:0 1.0GB, num_seqs 3, 2.293 sec/step, elapsed 0:11:37, exp. remaining 0:25:52, complete 30.99%
att-weights epoch 481, step 380, max_size:classes 32, max_size:data 1134, mem_usage:GPU:0 1.0GB, num_seqs 3, 24.249 sec/step, elapsed 0:12:01, exp. remaining 0:26:38, complete 31.11%
att-weights epoch 481, step 381, max_size:classes 32, max_size:data 964, mem_usage:GPU:0 1.0GB, num_seqs 4, 20.466 sec/step, elapsed 0:12:22, exp. remaining 0:27:15, complete 31.22%
att-weights epoch 481, step 382, max_size:classes 30, max_size:data 881, mem_usage:GPU:0 1.0GB, num_seqs 4, 56.521 sec/step, elapsed 0:13:18, exp. remaining 0:29:07, complete 31.37%
att-weights epoch 481, step 383, max_size:classes 29, max_size:data 1013, mem_usage:GPU:0 1.0GB, num_seqs 3, 34.725 sec/step, elapsed 0:13:53, exp. remaining 0:30:10, complete 31.53%
att-weights epoch 481, step 384, max_size:classes 32, max_size:data 1032, mem_usage:GPU:0 1.0GB, num_seqs 3, 6.388 sec/step, elapsed 0:13:59, exp. remaining 0:30:11, complete 31.68%
att-weights epoch 481, step 385, max_size:classes 31, max_size:data 923, mem_usage:GPU:0 1.0GB, num_seqs 3, 49.576 sec/step, elapsed 0:14:49, exp. remaining 0:31:44, complete 31.83%
att-weights epoch 481, step 386, max_size:classes 30, max_size:data 1059, mem_usage:GPU:0 1.0GB, num_seqs 3, 6.292 sec/step, elapsed 0:14:55, exp. remaining 0:31:48, complete 31.95%
att-weights epoch 481, step 387, max_size:classes 30, max_size:data 929, mem_usage:GPU:0 1.0GB, num_seqs 4, 23.684 sec/step, elapsed 0:15:19, exp. remaining 0:32:24, complete 32.10%
att-weights epoch 481, step 388, max_size:classes 27, max_size:data 1026, mem_usage:GPU:0 1.0GB, num_seqs 3, 5.071 sec/step, elapsed 0:15:24, exp. remaining 0:32:21, complete 32.25%
att-weights epoch 481, step 389, max_size:classes 30, max_size:data 855, mem_usage:GPU:0 1.0GB, num_seqs 3, 10.172 sec/step, elapsed 0:15:34, exp. remaining 0:32:29, complete 32.40%
att-weights epoch 481, step 390, max_size:classes 32, max_size:data 1040, mem_usage:GPU:0 1.0GB, num_seqs 3, 4.871 sec/step, elapsed 0:15:39, exp. remaining 0:32:29, complete 32.52%
att-weights epoch 481, step 391, max_size:classes 32, max_size:data 989, mem_usage:GPU:0 1.0GB, num_seqs 4, 8.262 sec/step, elapsed 0:15:47, exp. remaining 0:32:36, complete 32.63%
att-weights epoch 481, step 392, max_size:classes 31, max_size:data 904, mem_usage:GPU:0 1.0GB, num_seqs 4, 7.870 sec/step, elapsed 0:15:55, exp. remaining 0:32:39, complete 32.79%
att-weights epoch 481, step 393, max_size:classes 33, max_size:data 826, mem_usage:GPU:0 1.0GB, num_seqs 4, 6.316 sec/step, elapsed 0:16:01, exp. remaining 0:32:38, complete 32.94%
att-weights epoch 481, step 394, max_size:classes 28, max_size:data 842, mem_usage:GPU:0 1.0GB, num_seqs 4, 4.508 sec/step, elapsed 0:16:06, exp. remaining 0:32:34, complete 33.09%
att-weights epoch 481, step 395, max_size:classes 27, max_size:data 1009, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.980 sec/step, elapsed 0:16:08, exp. remaining 0:32:24, complete 33.24%
att-weights epoch 481, step 396, max_size:classes 28, max_size:data 913, mem_usage:GPU:0 1.0GB, num_seqs 4, 3.573 sec/step, elapsed 0:16:12, exp. remaining 0:32:18, complete 33.40%
att-weights epoch 481, step 397, max_size:classes 26, max_size:data 922, mem_usage:GPU:0 1.0GB, num_seqs 4, 6.031 sec/step, elapsed 0:16:18, exp. remaining 0:32:20, complete 33.51%
att-weights epoch 481, step 398, max_size:classes 29, max_size:data 801, mem_usage:GPU:0 1.0GB, num_seqs 4, 2.231 sec/step, elapsed 0:16:20, exp. remaining 0:32:15, complete 33.63%
att-weights epoch 481, step 399, max_size:classes 26, max_size:data 784, mem_usage:GPU:0 1.0GB, num_seqs 3, 3.859 sec/step, elapsed 0:16:24, exp. remaining 0:32:09, complete 33.78%
att-weights epoch 481, step 400, max_size:classes 32, max_size:data 1038, mem_usage:GPU:0 1.0GB, num_seqs 3, 4.118 sec/step, elapsed 0:16:28, exp. remaining 0:32:07, complete 33.89%
att-weights epoch 481, step 401, max_size:classes 27, max_size:data 853, mem_usage:GPU:0 1.0GB, num_seqs 4, 6.381 sec/step, elapsed 0:16:34, exp. remaining 0:32:06, complete 34.05%
att-weights epoch 481, step 402, max_size:classes 29, max_size:data 845, mem_usage:GPU:0 1.0GB, num_seqs 4, 11.028 sec/step, elapsed 0:16:45, exp. remaining 0:32:15, complete 34.20%
att-weights epoch 481, step 403, max_size:classes 32, max_size:data 907, mem_usage:GPU:0 1.0GB, num_seqs 4, 4.609 sec/step, elapsed 0:16:50, exp. remaining 0:32:10, complete 34.35%
att-weights epoch 481, step 404, max_size:classes 29, max_size:data 957, mem_usage:GPU:0 1.0GB, num_seqs 4, 7.683 sec/step, elapsed 0:16:57, exp. remaining 0:32:18, complete 34.43%
att-weights epoch 481, step 405, max_size:classes 30, max_size:data 810, mem_usage:GPU:0 1.0GB, num_seqs 4, 12.170 sec/step, elapsed 0:17:10, exp. remaining 0:32:35, complete 34.50%
att-weights epoch 481, step 406, max_size:classes 30, max_size:data 820, mem_usage:GPU:0 1.0GB, num_seqs 3, 5.902 sec/step, elapsed 0:17:16, exp. remaining 0:32:36, complete 34.62%
att-weights epoch 481, step 407, max_size:classes 28, max_size:data 1001, mem_usage:GPU:0 1.0GB, num_seqs 3, 13.482 sec/step, elapsed 0:17:29, exp. remaining 0:32:48, complete 34.77%
att-weights epoch 481, step 408, max_size:classes 27, max_size:data 923, mem_usage:GPU:0 1.0GB, num_seqs 4, 2.516 sec/step, elapsed 0:17:32, exp. remaining 0:32:40, complete 34.92%
att-weights epoch 481, step 409, max_size:classes 30, max_size:data 1025, mem_usage:GPU:0 1.0GB, num_seqs 3, 6.526 sec/step, elapsed 0:17:38, exp. remaining 0:32:39, complete 35.08%
att-weights epoch 481, step 410, max_size:classes 29, max_size:data 923, mem_usage:GPU:0 1.0GB, num_seqs 4, 5.897 sec/step, elapsed 0:17:44, exp. remaining 0:32:37, complete 35.23%
att-weights epoch 481, step 411, max_size:classes 30, max_size:data 900, mem_usage:GPU:0 1.0GB, num_seqs 4, 9.953 sec/step, elapsed 0:17:54, exp. remaining 0:32:42, complete 35.38%
att-weights epoch 481, step 412, max_size:classes 26, max_size:data 919, mem_usage:GPU:0 1.0GB, num_seqs 4, 6.912 sec/step, elapsed 0:18:01, exp. remaining 0:32:41, complete 35.53%
att-weights epoch 481, step 413, max_size:classes 28, max_size:data 781, mem_usage:GPU:0 1.0GB, num_seqs 2, 5.519 sec/step, elapsed 0:18:06, exp. remaining 0:32:38, complete 35.69%
att-weights epoch 481, step 414, max_size:classes 25, max_size:data 1396, mem_usage:GPU:0 1.0GB, num_seqs 2, 6.921 sec/step, elapsed 0:18:13, exp. remaining 0:32:38, complete 35.84%
att-weights epoch 481, step 415, max_size:classes 29, max_size:data 1010, mem_usage:GPU:0 1.0GB, num_seqs 3, 12.216 sec/step, elapsed 0:18:26, exp. remaining 0:32:46, complete 35.99%
att-weights epoch 481, step 416, max_size:classes 28, max_size:data 851, mem_usage:GPU:0 1.0GB, num_seqs 4, 5.777 sec/step, elapsed 0:18:31, exp. remaining 0:32:44, complete 36.15%
att-weights epoch 481, step 417, max_size:classes 25, max_size:data 841, mem_usage:GPU:0 1.0GB, num_seqs 4, 5.260 sec/step, elapsed 0:18:37, exp. remaining 0:32:40, complete 36.30%
att-weights epoch 481, step 418, max_size:classes 29, max_size:data 862, mem_usage:GPU:0 1.0GB, num_seqs 4, 3.976 sec/step, elapsed 0:18:41, exp. remaining 0:32:34, complete 36.45%
att-weights epoch 481, step 419, max_size:classes 28, max_size:data 963, mem_usage:GPU:0 1.0GB, num_seqs 4, 9.271 sec/step, elapsed 0:18:50, exp. remaining 0:32:34, complete 36.64%
att-weights epoch 481, step 420, max_size:classes 27, max_size:data 785, mem_usage:GPU:0 1.0GB, num_seqs 4, 3.613 sec/step, elapsed 0:18:53, exp. remaining 0:32:27, complete 36.79%
att-weights epoch 481, step 421, max_size:classes 27, max_size:data 832, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.127 sec/step, elapsed 0:18:55, exp. remaining 0:32:20, complete 36.91%
att-weights epoch 481, step 422, max_size:classes 26, max_size:data 863, mem_usage:GPU:0 1.0GB, num_seqs 4, 0.917 sec/step, elapsed 0:18:55, exp. remaining 0:32:12, complete 37.02%
att-weights epoch 481, step 423, max_size:classes 30, max_size:data 893, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.191 sec/step, elapsed 0:18:57, exp. remaining 0:31:58, complete 37.21%
att-weights epoch 481, step 424, max_size:classes 27, max_size:data 824, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.327 sec/step, elapsed 0:18:58, exp. remaining 0:31:51, complete 37.33%
att-weights epoch 481, step 425, max_size:classes 25, max_size:data 870, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.174 sec/step, elapsed 0:18:59, exp. remaining 0:31:44, complete 37.44%
att-weights epoch 481, step 426, max_size:classes 31, max_size:data 887, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.546 sec/step, elapsed 0:19:01, exp. remaining 0:31:31, complete 37.63%
att-weights epoch 481, step 427, max_size:classes 29, max_size:data 946, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.141 sec/step, elapsed 0:19:02, exp. remaining 0:31:20, complete 37.79%
att-weights epoch 481, step 428, max_size:classes 26, max_size:data 788, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.237 sec/step, elapsed 0:19:03, exp. remaining 0:31:07, complete 37.98%
att-weights epoch 481, step 429, max_size:classes 29, max_size:data 955, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.314 sec/step, elapsed 0:19:04, exp. remaining 0:30:57, complete 38.13%
att-weights epoch 481, step 430, max_size:classes 28, max_size:data 832, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.069 sec/step, elapsed 0:19:05, exp. remaining 0:30:47, complete 38.28%
att-weights epoch 481, step 431, max_size:classes 29, max_size:data 1208, mem_usage:GPU:0 1.0GB, num_seqs 3, 2.218 sec/step, elapsed 0:19:08, exp. remaining 0:30:39, complete 38.44%
att-weights epoch 481, step 432, max_size:classes 27, max_size:data 773, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.442 sec/step, elapsed 0:19:09, exp. remaining 0:30:29, complete 38.59%
att-weights epoch 481, step 433, max_size:classes 29, max_size:data 1077, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.160 sec/step, elapsed 0:19:10, exp. remaining 0:30:19, complete 38.74%
att-weights epoch 481, step 434, max_size:classes 27, max_size:data 1073, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.064 sec/step, elapsed 0:19:11, exp. remaining 0:30:06, complete 38.93%
att-weights epoch 481, step 435, max_size:classes 30, max_size:data 761, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.360 sec/step, elapsed 0:19:13, exp. remaining 0:29:54, complete 39.12%
att-weights epoch 481, step 436, max_size:classes 26, max_size:data 958, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.194 sec/step, elapsed 0:19:14, exp. remaining 0:29:44, complete 39.27%
att-weights epoch 481, step 437, max_size:classes 26, max_size:data 747, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.503 sec/step, elapsed 0:19:15, exp. remaining 0:29:35, complete 39.43%
att-weights epoch 481, step 438, max_size:classes 24, max_size:data 910, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.133 sec/step, elapsed 0:19:17, exp. remaining 0:29:26, complete 39.58%
att-weights epoch 481, step 439, max_size:classes 28, max_size:data 839, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.393 sec/step, elapsed 0:19:18, exp. remaining 0:29:14, complete 39.77%
att-weights epoch 481, step 440, max_size:classes 26, max_size:data 982, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.110 sec/step, elapsed 0:19:19, exp. remaining 0:29:04, complete 39.92%
att-weights epoch 481, step 441, max_size:classes 27, max_size:data 981, mem_usage:GPU:0 1.0GB, num_seqs 4, 5.282 sec/step, elapsed 0:19:24, exp. remaining 0:29:01, complete 40.08%
att-weights epoch 481, step 442, max_size:classes 25, max_size:data 861, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.211 sec/step, elapsed 0:19:26, exp. remaining 0:28:52, complete 40.23%
att-weights epoch 481, step 443, max_size:classes 29, max_size:data 792, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.451 sec/step, elapsed 0:19:27, exp. remaining 0:28:43, complete 40.38%
att-weights epoch 481, step 444, max_size:classes 26, max_size:data 791, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.401 sec/step, elapsed 0:19:28, exp. remaining 0:28:32, complete 40.57%
att-weights epoch 481, step 445, max_size:classes 25, max_size:data 720, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.335 sec/step, elapsed 0:19:30, exp. remaining 0:28:20, complete 40.76%
att-weights epoch 481, step 446, max_size:classes 26, max_size:data 823, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.323 sec/step, elapsed 0:19:31, exp. remaining 0:28:11, complete 40.92%
att-weights epoch 481, step 447, max_size:classes 26, max_size:data 881, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.481 sec/step, elapsed 0:19:33, exp. remaining 0:28:03, complete 41.07%
att-weights epoch 481, step 448, max_size:classes 25, max_size:data 731, mem_usage:GPU:0 1.0GB, num_seqs 5, 3.537 sec/step, elapsed 0:19:36, exp. remaining 0:28:00, complete 41.18%
att-weights epoch 481, step 449, max_size:classes 31, max_size:data 847, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.266 sec/step, elapsed 0:19:37, exp. remaining 0:27:51, complete 41.34%
att-weights epoch 481, step 450, max_size:classes 23, max_size:data 743, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.065 sec/step, elapsed 0:19:38, exp. remaining 0:27:42, complete 41.49%
att-weights epoch 481, step 451, max_size:classes 24, max_size:data 875, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.287 sec/step, elapsed 0:19:40, exp. remaining 0:27:34, complete 41.64%
att-weights epoch 481, step 452, max_size:classes 25, max_size:data 851, mem_usage:GPU:0 1.0GB, num_seqs 4, 16.102 sec/step, elapsed 0:19:56, exp. remaining 0:27:48, complete 41.76%
att-weights epoch 481, step 453, max_size:classes 28, max_size:data 771, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.290 sec/step, elapsed 0:19:57, exp. remaining 0:27:37, complete 41.95%
att-weights epoch 481, step 454, max_size:classes 23, max_size:data 699, mem_usage:GPU:0 1.0GB, num_seqs 5, 2.033 sec/step, elapsed 0:19:59, exp. remaining 0:27:27, complete 42.14%
att-weights epoch 481, step 455, max_size:classes 26, max_size:data 875, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.165 sec/step, elapsed 0:20:00, exp. remaining 0:27:16, complete 42.33%
att-weights epoch 481, step 456, max_size:classes 27, max_size:data 804, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.267 sec/step, elapsed 0:20:02, exp. remaining 0:27:07, complete 42.48%
att-weights epoch 481, step 457, max_size:classes 27, max_size:data 1038, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.160 sec/step, elapsed 0:20:03, exp. remaining 0:26:56, complete 42.67%
att-weights epoch 481, step 458, max_size:classes 26, max_size:data 855, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.314 sec/step, elapsed 0:20:04, exp. remaining 0:26:43, complete 42.90%
att-weights epoch 481, step 459, max_size:classes 24, max_size:data 841, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.235 sec/step, elapsed 0:20:05, exp. remaining 0:26:32, complete 43.09%
att-weights epoch 481, step 460, max_size:classes 28, max_size:data 927, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.557 sec/step, elapsed 0:20:07, exp. remaining 0:26:22, complete 43.28%
att-weights epoch 481, step 461, max_size:classes 29, max_size:data 1133, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.119 sec/step, elapsed 0:20:08, exp. remaining 0:26:11, complete 43.47%
att-weights epoch 481, step 462, max_size:classes 25, max_size:data 752, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.329 sec/step, elapsed 0:20:09, exp. remaining 0:26:00, complete 43.66%
att-weights epoch 481, step 463, max_size:classes 24, max_size:data 701, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.721 sec/step, elapsed 0:20:11, exp. remaining 0:25:48, complete 43.89%
att-weights epoch 481, step 464, max_size:classes 23, max_size:data 737, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.196 sec/step, elapsed 0:20:12, exp. remaining 0:25:40, complete 44.05%
att-weights epoch 481, step 465, max_size:classes 23, max_size:data 880, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.263 sec/step, elapsed 0:20:13, exp. remaining 0:25:30, complete 44.24%
att-weights epoch 481, step 466, max_size:classes 26, max_size:data 774, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.477 sec/step, elapsed 0:20:15, exp. remaining 0:25:20, complete 44.43%
att-weights epoch 481, step 467, max_size:classes 27, max_size:data 663, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.810 sec/step, elapsed 0:20:17, exp. remaining 0:25:13, complete 44.58%
att-weights epoch 481, step 468, max_size:classes 23, max_size:data 765, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.463 sec/step, elapsed 0:20:18, exp. remaining 0:25:03, complete 44.77%
att-weights epoch 481, step 469, max_size:classes 26, max_size:data 732, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.365 sec/step, elapsed 0:20:20, exp. remaining 0:24:51, complete 45.00%
att-weights epoch 481, step 470, max_size:classes 21, max_size:data 771, mem_usage:GPU:0 1.0GB, num_seqs 5, 5.975 sec/step, elapsed 0:20:26, exp. remaining 0:24:47, complete 45.19%
att-weights epoch 481, step 471, max_size:classes 25, max_size:data 784, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.380 sec/step, elapsed 0:20:27, exp. remaining 0:24:37, complete 45.38%
att-weights epoch 481, step 472, max_size:classes 22, max_size:data 654, mem_usage:GPU:0 1.0GB, num_seqs 6, 2.282 sec/step, elapsed 0:20:29, exp. remaining 0:24:26, complete 45.61%
att-weights epoch 481, step 473, max_size:classes 23, max_size:data 922, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.277 sec/step, elapsed 0:20:31, exp. remaining 0:24:16, complete 45.80%
att-weights epoch 481, step 474, max_size:classes 25, max_size:data 757, mem_usage:GPU:0 1.0GB, num_seqs 5, 5.707 sec/step, elapsed 0:20:36, exp. remaining 0:24:12, complete 45.99%
att-weights epoch 481, step 475, max_size:classes 22, max_size:data 656, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.846 sec/step, elapsed 0:20:38, exp. remaining 0:24:05, complete 46.15%
att-weights epoch 481, step 476, max_size:classes 22, max_size:data 826, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.406 sec/step, elapsed 0:20:39, exp. remaining 0:23:58, complete 46.30%
att-weights epoch 481, step 477, max_size:classes 23, max_size:data 742, mem_usage:GPU:0 1.0GB, num_seqs 5, 4.646 sec/step, elapsed 0:20:44, exp. remaining 0:23:50, complete 46.53%
att-weights epoch 481, step 478, max_size:classes 24, max_size:data 646, mem_usage:GPU:0 1.0GB, num_seqs 6, 2.189 sec/step, elapsed 0:20:46, exp. remaining 0:23:39, complete 46.76%
att-weights epoch 481, step 479, max_size:classes 21, max_size:data 681, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.164 sec/step, elapsed 0:20:47, exp. remaining 0:23:30, complete 46.95%
att-weights epoch 481, step 480, max_size:classes 24, max_size:data 729, mem_usage:GPU:0 1.0GB, num_seqs 5, 2.875 sec/step, elapsed 0:20:50, exp. remaining 0:23:20, complete 47.18%
att-weights epoch 481, step 481, max_size:classes 22, max_size:data 662, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.579 sec/step, elapsed 0:20:52, exp. remaining 0:23:13, complete 47.33%
att-weights epoch 481, step 482, max_size:classes 23, max_size:data 674, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.380 sec/step, elapsed 0:20:53, exp. remaining 0:23:06, complete 47.48%
att-weights epoch 481, step 483, max_size:classes 23, max_size:data 795, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.259 sec/step, elapsed 0:20:55, exp. remaining 0:22:59, complete 47.63%
att-weights epoch 481, step 484, max_size:classes 23, max_size:data 878, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.459 sec/step, elapsed 0:20:56, exp. remaining 0:22:50, complete 47.82%
att-weights epoch 481, step 485, max_size:classes 23, max_size:data 810, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.170 sec/step, elapsed 0:20:57, exp. remaining 0:22:45, complete 47.94%
att-weights epoch 481, step 486, max_size:classes 23, max_size:data 652, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.306 sec/step, elapsed 0:20:59, exp. remaining 0:22:34, complete 48.17%
att-weights epoch 481, step 487, max_size:classes 22, max_size:data 664, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.623 sec/step, elapsed 0:21:00, exp. remaining 0:22:26, complete 48.36%
att-weights epoch 481, step 488, max_size:classes 23, max_size:data 758, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.401 sec/step, elapsed 0:21:02, exp. remaining 0:22:19, complete 48.51%
att-weights epoch 481, step 489, max_size:classes 23, max_size:data 636, mem_usage:GPU:0 1.0GB, num_seqs 6, 2.822 sec/step, elapsed 0:21:04, exp. remaining 0:22:14, complete 48.66%
att-weights epoch 481, step 490, max_size:classes 22, max_size:data 883, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.622 sec/step, elapsed 0:21:06, exp. remaining 0:22:05, complete 48.85%
att-weights epoch 481, step 491, max_size:classes 21, max_size:data 613, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.156 sec/step, elapsed 0:21:07, exp. remaining 0:21:56, complete 49.05%
att-weights epoch 481, step 492, max_size:classes 24, max_size:data 815, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.189 sec/step, elapsed 0:21:08, exp. remaining 0:21:50, complete 49.20%
att-weights epoch 481, step 493, max_size:classes 23, max_size:data 707, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.504 sec/step, elapsed 0:21:10, exp. remaining 0:21:41, complete 49.39%
att-weights epoch 481, step 494, max_size:classes 25, max_size:data 1083, mem_usage:GPU:0 1.0GB, num_seqs 3, 2.460 sec/step, elapsed 0:21:12, exp. remaining 0:21:34, complete 49.58%
att-weights epoch 481, step 495, max_size:classes 24, max_size:data 629, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.878 sec/step, elapsed 0:21:14, exp. remaining 0:21:24, complete 49.81%
att-weights epoch 481, step 496, max_size:classes 22, max_size:data 662, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.216 sec/step, elapsed 0:21:15, exp. remaining 0:21:13, complete 50.04%
att-weights epoch 481, step 497, max_size:classes 24, max_size:data 747, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.457 sec/step, elapsed 0:21:17, exp. remaining 0:21:05, complete 50.23%
att-weights epoch 481, step 498, max_size:classes 23, max_size:data 871, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.809 sec/step, elapsed 0:21:19, exp. remaining 0:20:57, complete 50.42%
att-weights epoch 481, step 499, max_size:classes 21, max_size:data 680, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.172 sec/step, elapsed 0:21:20, exp. remaining 0:20:51, complete 50.57%
att-weights epoch 481, step 500, max_size:classes 19, max_size:data 756, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.326 sec/step, elapsed 0:21:21, exp. remaining 0:20:41, complete 50.80%
att-weights epoch 481, step 501, max_size:classes 22, max_size:data 852, mem_usage:GPU:0 1.0GB, num_seqs 4, 6.911 sec/step, elapsed 0:21:28, exp. remaining 0:20:38, complete 50.99%
att-weights epoch 481, step 502, max_size:classes 22, max_size:data 700, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.815 sec/step, elapsed 0:21:30, exp. remaining 0:20:28, complete 51.22%
att-weights epoch 481, step 503, max_size:classes 22, max_size:data 761, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.427 sec/step, elapsed 0:21:31, exp. remaining 0:20:20, complete 51.41%
att-weights epoch 481, step 504, max_size:classes 22, max_size:data 637, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.686 sec/step, elapsed 0:21:33, exp. remaining 0:20:11, complete 51.64%
att-weights epoch 481, step 505, max_size:classes 23, max_size:data 636, mem_usage:GPU:0 1.0GB, num_seqs 6, 9.406 sec/step, elapsed 0:21:42, exp. remaining 0:20:08, complete 51.87%
att-weights epoch 481, step 506, max_size:classes 23, max_size:data 691, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.695 sec/step, elapsed 0:21:44, exp. remaining 0:20:01, complete 52.06%
att-weights epoch 481, step 507, max_size:classes 21, max_size:data 777, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.285 sec/step, elapsed 0:21:45, exp. remaining 0:19:53, complete 52.25%
att-weights epoch 481, step 508, max_size:classes 20, max_size:data 821, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.071 sec/step, elapsed 0:21:46, exp. remaining 0:19:45, complete 52.44%
att-weights epoch 481, step 509, max_size:classes 19, max_size:data 632, mem_usage:GPU:0 1.0GB, num_seqs 6, 6.035 sec/step, elapsed 0:21:53, exp. remaining 0:19:41, complete 52.63%
att-weights epoch 481, step 510, max_size:classes 21, max_size:data 760, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.362 sec/step, elapsed 0:21:54, exp. remaining 0:19:32, complete 52.86%
att-weights epoch 481, step 511, max_size:classes 20, max_size:data 657, mem_usage:GPU:0 1.0GB, num_seqs 6, 2.097 sec/step, elapsed 0:21:56, exp. remaining 0:19:23, complete 53.09%
att-weights epoch 481, step 512, max_size:classes 21, max_size:data 754, mem_usage:GPU:0 1.0GB, num_seqs 5, 7.076 sec/step, elapsed 0:22:03, exp. remaining 0:19:20, complete 53.28%
att-weights epoch 481, step 513, max_size:classes 21, max_size:data 665, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.638 sec/step, elapsed 0:22:05, exp. remaining 0:19:11, complete 53.51%
att-weights epoch 481, step 514, max_size:classes 22, max_size:data 554, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.150 sec/step, elapsed 0:22:06, exp. remaining 0:18:59, complete 53.78%
att-weights epoch 481, step 515, max_size:classes 22, max_size:data 780, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.852 sec/step, elapsed 0:22:08, exp. remaining 0:18:54, complete 53.93%
att-weights epoch 481, step 516, max_size:classes 21, max_size:data 702, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.424 sec/step, elapsed 0:22:09, exp. remaining 0:18:45, complete 54.16%
att-weights epoch 481, step 517, max_size:classes 20, max_size:data 676, mem_usage:GPU:0 1.0GB, num_seqs 5, 3.112 sec/step, elapsed 0:22:12, exp. remaining 0:18:39, complete 54.35%
att-weights epoch 481, step 518, max_size:classes 22, max_size:data 757, mem_usage:GPU:0 1.0GB, num_seqs 5, 2.381 sec/step, elapsed 0:22:15, exp. remaining 0:18:31, complete 54.58%
att-weights epoch 481, step 519, max_size:classes 21, max_size:data 580, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.614 sec/step, elapsed 0:22:16, exp. remaining 0:18:22, complete 54.81%
att-weights epoch 481, step 520, max_size:classes 24, max_size:data 647, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.717 sec/step, elapsed 0:22:18, exp. remaining 0:18:15, complete 55.00%
att-weights epoch 481, step 521, max_size:classes 20, max_size:data 749, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.777 sec/step, elapsed 0:22:20, exp. remaining 0:18:06, complete 55.23%
att-weights epoch 481, step 522, max_size:classes 21, max_size:data 582, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.542 sec/step, elapsed 0:22:21, exp. remaining 0:17:57, complete 55.46%
att-weights epoch 481, step 523, max_size:classes 20, max_size:data 526, mem_usage:GPU:0 1.0GB, num_seqs 7, 9.583 sec/step, elapsed 0:22:31, exp. remaining 0:17:55, complete 55.69%
att-weights epoch 481, step 524, max_size:classes 22, max_size:data 951, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.343 sec/step, elapsed 0:22:32, exp. remaining 0:17:48, complete 55.88%
att-weights epoch 481, step 525, max_size:classes 19, max_size:data 662, mem_usage:GPU:0 1.0GB, num_seqs 6, 2.079 sec/step, elapsed 0:22:34, exp. remaining 0:17:39, complete 56.11%
att-weights epoch 481, step 526, max_size:classes 19, max_size:data 683, mem_usage:GPU:0 1.0GB, num_seqs 5, 19.154 sec/step, elapsed 0:22:53, exp. remaining 0:17:44, complete 56.34%
att-weights epoch 481, step 527, max_size:classes 20, max_size:data 587, mem_usage:GPU:0 1.0GB, num_seqs 6, 6.806 sec/step, elapsed 0:23:00, exp. remaining 0:17:41, complete 56.53%
att-weights epoch 481, step 528, max_size:classes 20, max_size:data 607, mem_usage:GPU:0 1.0GB, num_seqs 6, 2.128 sec/step, elapsed 0:23:02, exp. remaining 0:17:35, complete 56.72%
att-weights epoch 481, step 529, max_size:classes 19, max_size:data 713, mem_usage:GPU:0 1.0GB, num_seqs 5, 10.055 sec/step, elapsed 0:23:12, exp. remaining 0:17:34, complete 56.91%
att-weights epoch 481, step 530, max_size:classes 20, max_size:data 637, mem_usage:GPU:0 1.0GB, num_seqs 6, 6.668 sec/step, elapsed 0:23:19, exp. remaining 0:17:28, complete 57.18%
att-weights epoch 481, step 531, max_size:classes 19, max_size:data 646, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.652 sec/step, elapsed 0:23:21, exp. remaining 0:17:18, complete 57.44%
att-weights epoch 481, step 532, max_size:classes 19, max_size:data 590, mem_usage:GPU:0 1.0GB, num_seqs 6, 3.309 sec/step, elapsed 0:23:24, exp. remaining 0:17:12, complete 57.63%
att-weights epoch 481, step 533, max_size:classes 20, max_size:data 723, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.609 sec/step, elapsed 0:23:26, exp. remaining 0:17:05, complete 57.82%
att-weights epoch 481, step 534, max_size:classes 20, max_size:data 606, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.801 sec/step, elapsed 0:23:27, exp. remaining 0:16:57, complete 58.05%
att-weights epoch 481, step 535, max_size:classes 21, max_size:data 643, mem_usage:GPU:0 1.0GB, num_seqs 6, 4.402 sec/step, elapsed 0:23:32, exp. remaining 0:16:52, complete 58.24%
att-weights epoch 481, step 536, max_size:classes 19, max_size:data 672, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.128 sec/step, elapsed 0:23:33, exp. remaining 0:16:42, complete 58.51%
att-weights epoch 481, step 537, max_size:classes 20, max_size:data 695, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.172 sec/step, elapsed 0:23:34, exp. remaining 0:16:35, complete 58.70%
att-weights epoch 481, step 538, max_size:classes 22, max_size:data 710, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.528 sec/step, elapsed 0:23:36, exp. remaining 0:16:28, complete 58.89%
att-weights epoch 481, step 539, max_size:classes 18, max_size:data 512, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.461 sec/step, elapsed 0:23:37, exp. remaining 0:16:18, complete 59.16%
att-weights epoch 481, step 540, max_size:classes 18, max_size:data 559, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.652 sec/step, elapsed 0:23:39, exp. remaining 0:16:10, complete 59.39%
att-weights epoch 481, step 541, max_size:classes 18, max_size:data 533, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.819 sec/step, elapsed 0:23:41, exp. remaining 0:16:02, complete 59.62%
att-weights epoch 481, step 542, max_size:classes 18, max_size:data 727, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.373 sec/step, elapsed 0:23:42, exp. remaining 0:15:55, complete 59.81%
att-weights epoch 481, step 543, max_size:classes 18, max_size:data 592, mem_usage:GPU:0 1.0GB, num_seqs 6, 7.336 sec/step, elapsed 0:23:49, exp. remaining 0:15:50, complete 60.08%
att-weights epoch 481, step 544, max_size:classes 17, max_size:data 682, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.866 sec/step, elapsed 0:23:51, exp. remaining 0:15:42, complete 60.31%
att-weights epoch 481, step 545, max_size:classes 22, max_size:data 549, mem_usage:GPU:0 1.0GB, num_seqs 7, 3.141 sec/step, elapsed 0:23:54, exp. remaining 0:15:36, complete 60.50%
att-weights epoch 481, step 546, max_size:classes 18, max_size:data 674, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.607 sec/step, elapsed 0:23:56, exp. remaining 0:15:27, complete 60.76%
att-weights epoch 481, step 547, max_size:classes 19, max_size:data 682, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.664 sec/step, elapsed 0:23:58, exp. remaining 0:15:19, complete 60.99%
att-weights epoch 481, step 548, max_size:classes 18, max_size:data 537, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.843 sec/step, elapsed 0:23:59, exp. remaining 0:15:12, complete 61.22%
att-weights epoch 481, step 549, max_size:classes 20, max_size:data 664, mem_usage:GPU:0 1.0GB, num_seqs 6, 5.000 sec/step, elapsed 0:24:04, exp. remaining 0:15:07, complete 61.41%
att-weights epoch 481, step 550, max_size:classes 18, max_size:data 649, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.352 sec/step, elapsed 0:24:06, exp. remaining 0:15:00, complete 61.64%
att-weights epoch 481, step 551, max_size:classes 18, max_size:data 676, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.559 sec/step, elapsed 0:24:07, exp. remaining 0:14:52, complete 61.87%
att-weights epoch 481, step 552, max_size:classes 19, max_size:data 571, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.674 sec/step, elapsed 0:24:09, exp. remaining 0:14:44, complete 62.10%
att-weights epoch 481, step 553, max_size:classes 18, max_size:data 646, mem_usage:GPU:0 1.0GB, num_seqs 6, 4.591 sec/step, elapsed 0:24:14, exp. remaining 0:14:37, complete 62.37%
att-weights epoch 481, step 554, max_size:classes 18, max_size:data 705, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.673 sec/step, elapsed 0:24:15, exp. remaining 0:14:29, complete 62.60%
att-weights epoch 481, step 555, max_size:classes 17, max_size:data 541, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.865 sec/step, elapsed 0:24:17, exp. remaining 0:14:21, complete 62.86%
att-weights epoch 481, step 556, max_size:classes 17, max_size:data 600, mem_usage:GPU:0 1.0GB, num_seqs 6, 2.670 sec/step, elapsed 0:24:20, exp. remaining 0:14:12, complete 63.13%
att-weights epoch 481, step 557, max_size:classes 18, max_size:data 581, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.409 sec/step, elapsed 0:24:21, exp. remaining 0:14:03, complete 63.40%
att-weights epoch 481, step 558, max_size:classes 18, max_size:data 762, mem_usage:GPU:0 1.0GB, num_seqs 5, 5.705 sec/step, elapsed 0:24:27, exp. remaining 0:13:57, complete 63.66%
att-weights epoch 481, step 559, max_size:classes 18, max_size:data 621, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.675 sec/step, elapsed 0:24:29, exp. remaining 0:13:51, complete 63.85%
att-weights epoch 481, step 560, max_size:classes 17, max_size:data 603, mem_usage:GPU:0 1.0GB, num_seqs 6, 6.820 sec/step, elapsed 0:24:35, exp. remaining 0:13:45, complete 64.12%
att-weights epoch 481, step 561, max_size:classes 15, max_size:data 619, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.288 sec/step, elapsed 0:24:37, exp. remaining 0:13:38, complete 64.35%
att-weights epoch 481, step 562, max_size:classes 17, max_size:data 498, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.548 sec/step, elapsed 0:24:38, exp. remaining 0:13:29, complete 64.62%
att-weights epoch 481, step 563, max_size:classes 16, max_size:data 592, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.250 sec/step, elapsed 0:24:40, exp. remaining 0:13:20, complete 64.89%
att-weights epoch 481, step 564, max_size:classes 20, max_size:data 552, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.466 sec/step, elapsed 0:24:41, exp. remaining 0:13:13, complete 65.11%
att-weights epoch 481, step 565, max_size:classes 17, max_size:data 513, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.690 sec/step, elapsed 0:24:43, exp. remaining 0:13:05, complete 65.38%
att-weights epoch 481, step 566, max_size:classes 18, max_size:data 526, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.843 sec/step, elapsed 0:24:45, exp. remaining 0:12:55, complete 65.69%
att-weights epoch 481, step 567, max_size:classes 18, max_size:data 529, mem_usage:GPU:0 1.0GB, num_seqs 7, 4.531 sec/step, elapsed 0:24:49, exp. remaining 0:12:48, complete 65.95%
att-weights epoch 481, step 568, max_size:classes 20, max_size:data 704, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.434 sec/step, elapsed 0:24:51, exp. remaining 0:12:39, complete 66.26%
att-weights epoch 481, step 569, max_size:classes 22, max_size:data 564, mem_usage:GPU:0 1.0GB, num_seqs 7, 2.976 sec/step, elapsed 0:24:54, exp. remaining 0:12:33, complete 66.49%
att-weights epoch 481, step 570, max_size:classes 17, max_size:data 621, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.425 sec/step, elapsed 0:24:55, exp. remaining 0:12:24, complete 66.76%
att-weights epoch 481, step 571, max_size:classes 17, max_size:data 563, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.524 sec/step, elapsed 0:24:56, exp. remaining 0:12:15, complete 67.06%
att-weights epoch 481, step 572, max_size:classes 17, max_size:data 513, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.434 sec/step, elapsed 0:24:58, exp. remaining 0:12:07, complete 67.33%
att-weights epoch 481, step 573, max_size:classes 18, max_size:data 629, mem_usage:GPU:0 1.0GB, num_seqs 6, 2.918 sec/step, elapsed 0:25:01, exp. remaining 0:11:59, complete 67.60%
att-weights epoch 481, step 574, max_size:classes 17, max_size:data 561, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.568 sec/step, elapsed 0:25:02, exp. remaining 0:11:51, complete 67.86%
att-weights epoch 481, step 575, max_size:classes 18, max_size:data 500, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.344 sec/step, elapsed 0:25:04, exp. remaining 0:11:44, complete 68.09%
att-weights epoch 481, step 576, max_size:classes 17, max_size:data 539, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.283 sec/step, elapsed 0:25:05, exp. remaining 0:11:38, complete 68.32%
att-weights epoch 481, step 577, max_size:classes 15, max_size:data 459, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.747 sec/step, elapsed 0:25:07, exp. remaining 0:11:33, complete 68.47%
att-weights epoch 481, step 578, max_size:classes 17, max_size:data 590, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.507 sec/step, elapsed 0:25:08, exp. remaining 0:11:24, complete 68.78%
att-weights epoch 481, step 579, max_size:classes 15, max_size:data 545, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.349 sec/step, elapsed 0:25:10, exp. remaining 0:11:17, complete 69.05%
att-weights epoch 481, step 580, max_size:classes 14, max_size:data 447, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.520 sec/step, elapsed 0:25:11, exp. remaining 0:11:10, complete 69.27%
att-weights epoch 481, step 581, max_size:classes 17, max_size:data 513, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.525 sec/step, elapsed 0:25:13, exp. remaining 0:11:02, complete 69.54%
att-weights epoch 481, step 582, max_size:classes 16, max_size:data 563, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.592 sec/step, elapsed 0:25:14, exp. remaining 0:10:56, complete 69.77%
att-weights epoch 481, step 583, max_size:classes 16, max_size:data 514, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.630 sec/step, elapsed 0:25:16, exp. remaining 0:10:49, complete 70.00%
att-weights epoch 481, step 584, max_size:classes 16, max_size:data 578, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.214 sec/step, elapsed 0:25:17, exp. remaining 0:10:43, complete 70.23%
att-weights epoch 481, step 585, max_size:classes 14, max_size:data 537, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.272 sec/step, elapsed 0:25:18, exp. remaining 0:10:34, complete 70.53%
att-weights epoch 481, step 586, max_size:classes 14, max_size:data 874, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.130 sec/step, elapsed 0:25:20, exp. remaining 0:10:28, complete 70.76%
att-weights epoch 481, step 587, max_size:classes 17, max_size:data 452, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.772 sec/step, elapsed 0:25:21, exp. remaining 0:10:20, complete 71.03%
att-weights epoch 481, step 588, max_size:classes 18, max_size:data 413, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.513 sec/step, elapsed 0:25:23, exp. remaining 0:10:12, complete 71.34%
att-weights epoch 481, step 589, max_size:classes 16, max_size:data 616, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.405 sec/step, elapsed 0:25:24, exp. remaining 0:10:05, complete 71.56%
att-weights epoch 481, step 590, max_size:classes 17, max_size:data 470, mem_usage:GPU:0 1.0GB, num_seqs 7, 4.046 sec/step, elapsed 0:25:28, exp. remaining 0:10:00, complete 71.79%
att-weights epoch 481, step 591, max_size:classes 16, max_size:data 628, mem_usage:GPU:0 1.0GB, num_seqs 6, 2.186 sec/step, elapsed 0:25:30, exp. remaining 0:09:53, complete 72.06%
att-weights epoch 481, step 592, max_size:classes 15, max_size:data 506, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.270 sec/step, elapsed 0:25:32, exp. remaining 0:09:46, complete 72.33%
att-weights epoch 481, step 593, max_size:classes 16, max_size:data 611, mem_usage:GPU:0 1.0GB, num_seqs 6, 8.126 sec/step, elapsed 0:25:40, exp. remaining 0:09:42, complete 72.56%
att-weights epoch 481, step 594, max_size:classes 16, max_size:data 454, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.860 sec/step, elapsed 0:25:42, exp. remaining 0:09:34, complete 72.86%
att-weights epoch 481, step 595, max_size:classes 16, max_size:data 591, mem_usage:GPU:0 1.0GB, num_seqs 6, 8.393 sec/step, elapsed 0:25:50, exp. remaining 0:09:29, complete 73.13%
att-weights epoch 481, step 596, max_size:classes 15, max_size:data 523, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.394 sec/step, elapsed 0:25:51, exp. remaining 0:09:22, complete 73.40%
att-weights epoch 481, step 597, max_size:classes 14, max_size:data 459, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.628 sec/step, elapsed 0:25:53, exp. remaining 0:09:15, complete 73.66%
att-weights epoch 481, step 598, max_size:classes 16, max_size:data 656, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.123 sec/step, elapsed 0:25:54, exp. remaining 0:09:09, complete 73.89%
att-weights epoch 481, step 599, max_size:classes 15, max_size:data 578, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.338 sec/step, elapsed 0:25:56, exp. remaining 0:09:00, complete 74.24%
att-weights epoch 481, step 600, max_size:classes 15, max_size:data 473, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.576 sec/step, elapsed 0:25:57, exp. remaining 0:08:54, complete 74.47%
att-weights epoch 481, step 601, max_size:classes 15, max_size:data 536, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.270 sec/step, elapsed 0:25:58, exp. remaining 0:08:47, complete 74.73%
att-weights epoch 481, step 602, max_size:classes 16, max_size:data 635, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.413 sec/step, elapsed 0:26:00, exp. remaining 0:08:39, complete 75.04%
att-weights epoch 481, step 603, max_size:classes 15, max_size:data 494, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.789 sec/step, elapsed 0:26:02, exp. remaining 0:08:33, complete 75.27%
att-weights epoch 481, step 604, max_size:classes 14, max_size:data 545, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.753 sec/step, elapsed 0:26:03, exp. remaining 0:08:25, complete 75.57%
att-weights epoch 481, step 605, max_size:classes 13, max_size:data 449, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.523 sec/step, elapsed 0:26:05, exp. remaining 0:08:17, complete 75.88%
att-weights epoch 481, step 606, max_size:classes 16, max_size:data 528, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.333 sec/step, elapsed 0:26:06, exp. remaining 0:08:09, complete 76.18%
att-weights epoch 481, step 607, max_size:classes 14, max_size:data 624, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.269 sec/step, elapsed 0:26:08, exp. remaining 0:07:59, complete 76.56%
att-weights epoch 481, step 608, max_size:classes 14, max_size:data 419, mem_usage:GPU:0 1.0GB, num_seqs 9, 3.278 sec/step, elapsed 0:26:11, exp. remaining 0:07:53, complete 76.83%
att-weights epoch 481, step 609, max_size:classes 14, max_size:data 572, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.127 sec/step, elapsed 0:26:12, exp. remaining 0:07:45, complete 77.18%
att-weights epoch 481, step 610, max_size:classes 16, max_size:data 549, mem_usage:GPU:0 1.0GB, num_seqs 7, 2.702 sec/step, elapsed 0:26:15, exp. remaining 0:07:37, complete 77.48%
att-weights epoch 481, step 611, max_size:classes 16, max_size:data 439, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.638 sec/step, elapsed 0:26:16, exp. remaining 0:07:29, complete 77.82%
att-weights epoch 481, step 612, max_size:classes 15, max_size:data 593, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.178 sec/step, elapsed 0:26:17, exp. remaining 0:07:20, complete 78.17%
att-weights epoch 481, step 613, max_size:classes 14, max_size:data 466, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.447 sec/step, elapsed 0:26:19, exp. remaining 0:07:12, complete 78.51%
att-weights epoch 481, step 614, max_size:classes 15, max_size:data 478, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.482 sec/step, elapsed 0:26:20, exp. remaining 0:07:05, complete 78.78%
att-weights epoch 481, step 615, max_size:classes 13, max_size:data 477, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.775 sec/step, elapsed 0:26:22, exp. remaining 0:06:59, complete 79.05%
att-weights epoch 481, step 616, max_size:classes 13, max_size:data 391, mem_usage:GPU:0 1.0GB, num_seqs 10, 1.805 sec/step, elapsed 0:26:24, exp. remaining 0:06:51, complete 79.39%
att-weights epoch 481, step 617, max_size:classes 15, max_size:data 513, mem_usage:GPU:0 1.0GB, num_seqs 7, 4.874 sec/step, elapsed 0:26:29, exp. remaining 0:06:44, complete 79.69%
att-weights epoch 481, step 618, max_size:classes 13, max_size:data 422, mem_usage:GPU:0 1.0GB, num_seqs 9, 1.734 sec/step, elapsed 0:26:31, exp. remaining 0:06:38, complete 79.96%
att-weights epoch 481, step 619, max_size:classes 13, max_size:data 452, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.692 sec/step, elapsed 0:26:32, exp. remaining 0:06:32, complete 80.23%
att-weights epoch 481, step 620, max_size:classes 12, max_size:data 429, mem_usage:GPU:0 1.0GB, num_seqs 9, 1.749 sec/step, elapsed 0:26:34, exp. remaining 0:06:25, complete 80.53%
att-weights epoch 481, step 621, max_size:classes 14, max_size:data 428, mem_usage:GPU:0 1.0GB, num_seqs 9, 4.052 sec/step, elapsed 0:26:38, exp. remaining 0:06:18, complete 80.84%
att-weights epoch 481, step 622, max_size:classes 12, max_size:data 412, mem_usage:GPU:0 1.0GB, num_seqs 9, 1.783 sec/step, elapsed 0:26:40, exp. remaining 0:06:10, complete 81.18%
att-weights epoch 481, step 623, max_size:classes 12, max_size:data 403, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.480 sec/step, elapsed 0:26:41, exp. remaining 0:06:03, complete 81.49%
att-weights epoch 481, step 624, max_size:classes 12, max_size:data 508, mem_usage:GPU:0 1.0GB, num_seqs 7, 10.051 sec/step, elapsed 0:26:51, exp. remaining 0:05:58, complete 81.79%
att-weights epoch 481, step 625, max_size:classes 12, max_size:data 415, mem_usage:GPU:0 1.0GB, num_seqs 9, 2.129 sec/step, elapsed 0:26:54, exp. remaining 0:05:52, complete 82.06%
att-weights epoch 481, step 626, max_size:classes 12, max_size:data 489, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.858 sec/step, elapsed 0:26:55, exp. remaining 0:05:45, complete 82.40%
att-weights epoch 481, step 627, max_size:classes 13, max_size:data 428, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.520 sec/step, elapsed 0:26:57, exp. remaining 0:05:38, complete 82.71%
att-weights epoch 481, step 628, max_size:classes 13, max_size:data 515, mem_usage:GPU:0 1.0GB, num_seqs 7, 4.563 sec/step, elapsed 0:27:01, exp. remaining 0:05:30, complete 83.09%
att-weights epoch 481, step 629, max_size:classes 12, max_size:data 472, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.673 sec/step, elapsed 0:27:03, exp. remaining 0:05:22, complete 83.44%
att-weights epoch 481, step 630, max_size:classes 12, max_size:data 469, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.650 sec/step, elapsed 0:27:05, exp. remaining 0:05:15, complete 83.74%
att-weights epoch 481, step 631, max_size:classes 12, max_size:data 359, mem_usage:GPU:0 1.0GB, num_seqs 9, 4.834 sec/step, elapsed 0:27:10, exp. remaining 0:05:06, complete 84.16%
att-weights epoch 481, step 632, max_size:classes 13, max_size:data 475, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.545 sec/step, elapsed 0:27:11, exp. remaining 0:05:00, complete 84.47%
att-weights epoch 481, step 633, max_size:classes 11, max_size:data 360, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.297 sec/step, elapsed 0:27:12, exp. remaining 0:04:52, complete 84.81%
att-weights epoch 481, step 634, max_size:classes 11, max_size:data 544, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.569 sec/step, elapsed 0:27:14, exp. remaining 0:04:44, complete 85.15%
att-weights epoch 481, step 635, max_size:classes 12, max_size:data 426, mem_usage:GPU:0 1.0GB, num_seqs 9, 1.593 sec/step, elapsed 0:27:16, exp. remaining 0:04:40, complete 85.38%
att-weights epoch 481, step 636, max_size:classes 15, max_size:data 471, mem_usage:GPU:0 1.0GB, num_seqs 8, 2.108 sec/step, elapsed 0:27:18, exp. remaining 0:04:33, complete 85.69%
att-weights epoch 481, step 637, max_size:classes 12, max_size:data 379, mem_usage:GPU:0 1.0GB, num_seqs 10, 1.893 sec/step, elapsed 0:27:20, exp. remaining 0:04:27, complete 85.99%
att-weights epoch 481, step 638, max_size:classes 11, max_size:data 330, mem_usage:GPU:0 1.0GB, num_seqs 9, 1.516 sec/step, elapsed 0:27:21, exp. remaining 0:04:18, complete 86.37%
att-weights epoch 481, step 639, max_size:classes 11, max_size:data 490, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.420 sec/step, elapsed 0:27:23, exp. remaining 0:04:14, complete 86.60%
att-weights epoch 481, step 640, max_size:classes 12, max_size:data 339, mem_usage:GPU:0 1.0GB, num_seqs 11, 2.091 sec/step, elapsed 0:27:25, exp. remaining 0:04:04, complete 87.06%
att-weights epoch 481, step 641, max_size:classes 12, max_size:data 450, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.632 sec/step, elapsed 0:27:26, exp. remaining 0:03:56, complete 87.44%
att-weights epoch 481, step 642, max_size:classes 13, max_size:data 374, mem_usage:GPU:0 1.0GB, num_seqs 9, 1.690 sec/step, elapsed 0:27:28, exp. remaining 0:03:50, complete 87.75%
att-weights epoch 481, step 643, max_size:classes 11, max_size:data 429, mem_usage:GPU:0 1.0GB, num_seqs 9, 1.653 sec/step, elapsed 0:27:30, exp. remaining 0:03:43, complete 88.09%
att-weights epoch 481, step 644, max_size:classes 11, max_size:data 612, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.168 sec/step, elapsed 0:27:31, exp. remaining 0:03:35, complete 88.44%
att-weights epoch 481, step 645, max_size:classes 12, max_size:data 500, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.489 sec/step, elapsed 0:27:32, exp. remaining 0:03:28, complete 88.78%
att-weights epoch 481, step 646, max_size:classes 12, max_size:data 467, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.944 sec/step, elapsed 0:27:34, exp. remaining 0:03:21, complete 89.16%
att-weights epoch 481, step 647, max_size:classes 10, max_size:data 382, mem_usage:GPU:0 1.0GB, num_seqs 10, 1.584 sec/step, elapsed 0:27:36, exp. remaining 0:03:14, complete 89.50%
att-weights epoch 481, step 648, max_size:classes 9, max_size:data 657, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.031 sec/step, elapsed 0:27:37, exp. remaining 0:03:05, complete 89.92%
att-weights epoch 481, step 649, max_size:classes 11, max_size:data 332, mem_usage:GPU:0 1.0GB, num_seqs 12, 1.751 sec/step, elapsed 0:27:39, exp. remaining 0:02:59, complete 90.23%
att-weights epoch 481, step 650, max_size:classes 12, max_size:data 390, mem_usage:GPU:0 1.0GB, num_seqs 10, 1.600 sec/step, elapsed 0:27:40, exp. remaining 0:02:53, complete 90.53%
att-weights epoch 481, step 651, max_size:classes 12, max_size:data 474, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.794 sec/step, elapsed 0:27:42, exp. remaining 0:02:46, complete 90.92%
att-weights epoch 481, step 652, max_size:classes 9, max_size:data 403, mem_usage:GPU:0 1.0GB, num_seqs 9, 2.534 sec/step, elapsed 0:27:45, exp. remaining 0:02:39, complete 91.26%
att-weights epoch 481, step 653, max_size:classes 10, max_size:data 402, mem_usage:GPU:0 1.0GB, num_seqs 9, 1.564 sec/step, elapsed 0:27:46, exp. remaining 0:02:32, complete 91.60%
att-weights epoch 481, step 654, max_size:classes 10, max_size:data 410, mem_usage:GPU:0 1.0GB, num_seqs 9, 1.579 sec/step, elapsed 0:27:48, exp. remaining 0:02:25, complete 91.98%
att-weights epoch 481, step 655, max_size:classes 11, max_size:data 293, mem_usage:GPU:0 1.0GB, num_seqs 10, 1.244 sec/step, elapsed 0:27:49, exp. remaining 0:02:17, complete 92.40%
att-weights epoch 481, step 656, max_size:classes 10, max_size:data 426, mem_usage:GPU:0 1.0GB, num_seqs 9, 1.388 sec/step, elapsed 0:27:50, exp. remaining 0:02:08, complete 92.86%
att-weights epoch 481, step 657, max_size:classes 10, max_size:data 312, mem_usage:GPU:0 1.0GB, num_seqs 11, 1.943 sec/step, elapsed 0:27:52, exp. remaining 0:02:01, complete 93.21%
att-weights epoch 481, step 658, max_size:classes 9, max_size:data 384, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.418 sec/step, elapsed 0:27:54, exp. remaining 0:01:54, complete 93.59%
att-weights epoch 481, step 659, max_size:classes 11, max_size:data 448, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.272 sec/step, elapsed 0:27:55, exp. remaining 0:01:46, complete 94.05%
att-weights epoch 481, step 660, max_size:classes 11, max_size:data 389, mem_usage:GPU:0 1.0GB, num_seqs 10, 5.561 sec/step, elapsed 0:28:01, exp. remaining 0:01:38, complete 94.47%
att-weights epoch 481, step 661, max_size:classes 10, max_size:data 404, mem_usage:GPU:0 1.0GB, num_seqs 9, 2.131 sec/step, elapsed 0:28:03, exp. remaining 0:01:31, complete 94.85%
att-weights epoch 481, step 662, max_size:classes 10, max_size:data 432, mem_usage:GPU:0 1.0GB, num_seqs 9, 3.487 sec/step, elapsed 0:28:06, exp. remaining 0:01:25, complete 95.19%
att-weights epoch 481, step 663, max_size:classes 10, max_size:data 399, mem_usage:GPU:0 1.0GB, num_seqs 10, 20.006 sec/step, elapsed 0:28:26, exp. remaining 0:01:17, complete 95.65%
att-weights epoch 481, step 664, max_size:classes 10, max_size:data 353, mem_usage:GPU:0 1.0GB, num_seqs 11, 1.997 sec/step, elapsed 0:28:28, exp. remaining 0:01:09, complete 96.07%
att-weights epoch 481, step 665, max_size:classes 11, max_size:data 328, mem_usage:GPU:0 1.0GB, num_seqs 12, 9.992 sec/step, elapsed 0:28:38, exp. remaining 0:01:03, complete 96.41%
att-weights epoch 481, step 666, max_size:classes 8, max_size:data 412, mem_usage:GPU:0 1.0GB, num_seqs 9, 1.395 sec/step, elapsed 0:28:40, exp. remaining 0:00:55, complete 96.87%
att-weights epoch 481, step 667, max_size:classes 9, max_size:data 384, mem_usage:GPU:0 1.0GB, num_seqs 10, 1.735 sec/step, elapsed 0:28:41, exp. remaining 0:00:46, complete 97.37%
att-weights epoch 481, step 668, max_size:classes 9, max_size:data 309, mem_usage:GPU:0 1.0GB, num_seqs 12, 9.104 sec/step, elapsed 0:28:50, exp. remaining 0:00:39, complete 97.79%
att-weights epoch 481, step 669, max_size:classes 9, max_size:data 280, mem_usage:GPU:0 1.0GB, num_seqs 11, 1.614 sec/step, elapsed 0:28:52, exp. remaining 0:00:31, complete 98.21%
att-weights epoch 481, step 670, max_size:classes 10, max_size:data 388, mem_usage:GPU:0 1.0GB, num_seqs 10, 1.638 sec/step, elapsed 0:28:54, exp. remaining 0:00:22, complete 98.70%
att-weights epoch 481, step 671, max_size:classes 10, max_size:data 442, mem_usage:GPU:0 1.0GB, num_seqs 9, 1.542 sec/step, elapsed 0:28:55, exp. remaining 0:00:13, complete 99.24%
att-weights epoch 481, step 672, max_size:classes 9, max_size:data 327, mem_usage:GPU:0 1.0GB, num_seqs 12, 1.175 sec/step, elapsed 0:28:56, exp. remaining 0:00:05, complete 99.69%
att-weights epoch 481, step 673, max_size:classes 8, max_size:data 336, mem_usage:GPU:0 1.0GB, num_seqs 11, 1.482 sec/step, elapsed 0:28:58, exp. remaining 0:00:05, complete 99.69%
att-weights epoch 481, step 674, max_size:classes 8, max_size:data 405, mem_usage:GPU:0 1.0GB, num_seqs 9, 1.350 sec/step, elapsed 0:28:59, exp. remaining 0:00:05, complete 99.69%
att-weights epoch 481, step 675, max_size:classes 7, max_size:data 332, mem_usage:GPU:0 1.0GB, num_seqs 12, 1.513 sec/step, elapsed 0:29:01, exp. remaining 0:00:05, complete 99.69%
att-weights epoch 481, step 676, max_size:classes 9, max_size:data 301, mem_usage:GPU:0 1.0GB, num_seqs 13, 1.250 sec/step, elapsed 0:29:02, exp. remaining 0:00:05, complete 99.69%
att-weights epoch 481, step 677, max_size:classes 8, max_size:data 304, mem_usage:GPU:0 1.0GB, num_seqs 11, 1.098 sec/step, elapsed 0:29:03, exp. remaining 0:00:05, complete 99.69%
att-weights epoch 481, step 678, max_size:classes 7, max_size:data 337, mem_usage:GPU:0 1.0GB, num_seqs 11, 1.164 sec/step, elapsed 0:29:04, exp. remaining 0:00:05, complete 99.69%
att-weights epoch 481, step 679, max_size:classes 7, max_size:data 293, mem_usage:GPU:0 1.0GB, num_seqs 13, 1.712 sec/step, elapsed 0:29:06, exp. remaining 0:00:05, complete 99.69%
att-weights epoch 481, step 680, max_size:classes 6, max_size:data 268, mem_usage:GPU:0 1.0GB, num_seqs 14, 1.513 sec/step, elapsed 0:29:07, exp. remaining 0:00:05, complete 99.69%
att-weights epoch 481, step 681, max_size:classes 7, max_size:data 311, mem_usage:GPU:0 1.0GB, num_seqs 12, 1.101 sec/step, elapsed 0:29:09, exp. remaining 0:00:05, complete 99.69%
att-weights epoch 481, step 682, max_size:classes 4, max_size:data 310, mem_usage:GPU:0 1.0GB, num_seqs 9, 0.650 sec/step, elapsed 0:29:09, exp. remaining 0:00:05, complete 99.69%
Stats:
  mem_usage:GPU:0: Stats(mean=1.0GB, std_dev=0.0B, min=1.0GB, max=1.0GB, num_seqs=683, avg_data_len=1)
att-weights epoch 481, finished after 683 steps, 0:29:09 elapsed (20.2% computing time)
Layer 'dec_02_att_weights' Stats:
  2620 seqs, 103825928 total frames, 39628.216794 average frames
  Mean: 0.005243314557379197
  Std dev: 0.03989972525846154
  Min/max: 0.0 / 1.0
Layer 'dec_01_att_weights' Stats:
  2620 seqs, 103825928 total frames, 39628.216794 average frames
  Mean: 0.005243314556276944
  Std dev: 0.07017685052210659
  Min/max: 0.0 / 1.0
Layer 'dec_03_att_weights' Stats:
  2620 seqs, 103825928 total frames, 39628.216794 average frames
  Mean: 0.005243314551169924
  Std dev: 0.045612990217290074
  Min/max: 0.0 / 0.99999845
Layer 'dec_04_att_weights' Stats:
  2620 seqs, 103825928 total frames, 39628.216794 average frames
  Mean: 0.005243314556570896
  Std dev: 0.04217736225748746
  Min/max: 0.0 / 0.99997807
Layer 'dec_05_att_weights' Stats:
  2620 seqs, 103825928 total frames, 39628.216794 average frames
  Mean: 0.0052433145562769505
  Std dev: 0.06863625709934076
  Min/max: 0.0 / 1.0
Layer 'dec_06_att_weights' Stats:
  2620 seqs, 103825928 total frames, 39628.216794 average frames
  Mean: 0.005243314549277751
  Std dev: 0.06446082554817019
  Min/max: 0.0 / 1.0
Layer 'dec_07_att_weights' Stats:
  2620 seqs, 103825928 total frames, 39628.216794 average frames
  Mean: 0.0052433145651867435
  Std dev: 0.06804413116193106
  Min/max: 0.0 / 1.0
Layer 'dec_08_att_weights' Stats:
  2620 seqs, 103825928 total frames, 39628.216794 average frames
  Mean: 0.00524331455682809
  Std dev: 0.06687532565822238
  Min/max: 0.0 / 1.0
Layer 'dec_09_att_weights' Stats:
  2620 seqs, 103825928 total frames, 39628.216794 average frames
  Mean: 0.005243314561696316
  Std dev: 0.06916352876224433
  Min/max: 0.0 / 1.0
Layer 'dec_10_att_weights' Stats:
  2620 seqs, 103825928 total frames, 39628.216794 average frames
  Mean: 0.005243314561861635
  Std dev: 0.06779887492126115
  Min/max: 0.0 / 1.0
Layer 'dec_11_att_weights' Stats:
  2620 seqs, 103825928 total frames, 39628.216794 average frames
  Mean: 0.005243314554990997
  Std dev: 0.06628206334487449
  Min/max: 0.0 / 1.0
Layer 'dec_12_att_weights' Stats:
  2620 seqs, 103825928 total frames, 39628.216794 average frames
  Mean: 0.005243314557066908
  Std dev: 0.0656131699198549
  Min/max: 0.0 / 1.0
Quitting
+------- EPILOGUE SCRIPT -----------------------------------------------
|
| Job ID ..............: 9506020
| Stopped at ..........: Tue Jul  2 13:06:30 CEST 2019
| Resources requested .: scratch_free=5G,h_fsize=20G,num_proc=5,s_core=0,pxe=ubuntu_16.04,h_vmem=1536G,h_rss=8G,h_rt=7200,gpu=1
| Resources used ......: cpu=00:35:12, mem=7909.62281 GB s, io=11.63465 GB, vmem=4.041G, maxvmem=4.069G, last_file_cache=4.343G, last_rss=2M, max-cache=3.657G
| Memory used .........: 8.000G / 8.000G (100.0%)
| Total time used .....: 0:31:21
|
+------- EPILOGUE SCRIPT -----------------------------------------------
