+------- PROLOGUE SCRIPT -----------------------------------------------
|
| Job ID ...........: 9505912
| Started at .......: Tue Jul  2 11:38:22 CEST 2019
| Execution host ...: cluster-cn-248
| Cluster queue ....: 4-GPU-1080
| Script ...........: /var/spool/sge/cluster-cn-248/job_scripts/9505912
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
RETURNN starting up, version 20181022.053313--git-dfada43-dirty, date/time 2019-07-02-11-38-25 (UTC+0200), pid 4055, cwd /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc, Python /u/bahar/settings/python3-returnn-tf1.9/bin/python3
RETURNN config: /u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc.config
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
       incarnation: 1598750592866309513
  2/2: name: "/device:GPU:0"
       device_type: "GPU"
       memory_limit: 10911236096
       locality {
         bus_id: 2
         numa_node: 1
         links {
         }
       }
       incarnation: 8684506277494175504
       physical_device_desc: "device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:81:00.0, compute capability: 6.1"
Using gpu device 2: GeForce GTX 1080 Ti
Device not set explicitly, and we found a GPU, which we will use.
Model file prefix: /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc/net-model/network
NOTE: We will use 'sorted_reverse' seq ordering.
Unhandled exception <class 'AssertionError'> in thread <_MainThread(MainThread, started 47330658409472)>, proc 4055.

Thread current, main, <_MainThread(MainThread, started 47330658409472)>:
(Excluded thread.)

That were all threads.
+------- EPILOGUE SCRIPT -----------------------------------------------
|
| Job ID ..............: 9505912
| Stopped at ..........: Tue Jul  2 11:38:31 CEST 2019
| Resources requested .: h_fsize=20G,s_core=0,pxe=ubuntu_16.04,h_rt=7200,gpu=1,scratch_free=5G,num_proc=5,h_vmem=1536G,h_rss=4G
| Resources used ......: cpu=00:00:00, mem=0.00000 GB s, io=0.00000 GB, vmem=N/A, maxvmem=N/A, last_file_cache=13M, last_rss=3M, max-cache=737M
| Memory used .........: 750M / 4.000G (18.3%)
| Total time used .....: 0:00:09
|
+------- EPILOGUE SCRIPT -----------------------------------------------
+------- PROLOGUE SCRIPT -----------------------------------------------
|
| Job ID ...........: 9505968
| Started at .......: Tue Jul  2 12:01:42 CEST 2019
| Execution host ...: cluster-cn-262
| Cluster queue ....: 4-GPU-1080-128G
| Script ...........: /var/spool/sge/cluster-cn-262/job_scripts/9505968
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
RETURNN get-attention-weights starting up.
RETURNN starting up, version 20181022.053313--git-dfada43-dirty, date/time 2019-07-02-12-01-46 (UTC+0200), pid 22628, cwd /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc, Python /u/bahar/settings/python3-returnn-tf1.9/bin/python3
RETURNN config: /u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc.config
RETURNN command line options: ()
Hostname: cluster-cn-262
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
       incarnation: 327050562716946467
  2/2: name: "/device:GPU:0"
       device_type: "GPU"
       memory_limit: 10911236096
       locality {
         bus_id: 2
         numa_node: 1
         links {
         }
       }
       incarnation: 7330634697605228016
       physical_device_desc: "device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:81:00.0, compute capability: 6.1"
Using gpu device 2: GeForce GTX 1080 Ti
Device not set explicitly, and we found a GPU, which we will use.
Model file prefix: /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc/net-model/network
NOTE: We will use 'sorted_reverse' seq ordering.
Setup tf.Session with options {'log_device_placement': False, 'device_count': {'GPU': 1}} ...
layer root/'data' output: Data(name='data', shape=(None, 40))
layer root/'source' output: Data(name='source_output', shape=(None, 40))
layer root/'lstm0_fw' output: Data(name='lstm0_fw_output', shape=(None, 1024), batch_dim_axis=1)
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9505968.1.4-GPU-1080-128G/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.cc -o /var/tmp/9505968.1.4-GPU-1080-128G/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9505968.1.4-GPU-1080-128G/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.cc -o /var/tmp/9505968.1.4-GPU-1080-128G/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
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
layer root/'dec_04_att_key0' output: Data(name='dec_04_att_key0_output', shape=(None, 512))
layer root/'dec_04_att_key' output: Data(name='dec_04_att_key_output', shape=(None, 8, 64))
layer root/'dec_08_att_value0' output: Data(name='dec_08_att_value0_output', shape=(None, 512))
layer root/'dec_08_att_value' output: Data(name='dec_08_att_value_output', shape=(None, 8, 64))
layer root/'dec_06_att_key0' output: Data(name='dec_06_att_key0_output', shape=(None, 512))
layer root/'dec_06_att_key' output: Data(name='dec_06_att_key_output', shape=(None, 8, 64))
layer root/'dec_10_att_key0' output: Data(name='dec_10_att_key0_output', shape=(None, 512))
layer root/'dec_10_att_key' output: Data(name='dec_10_att_key_output', shape=(None, 8, 64))
layer root/'dec_04_att_value0' output: Data(name='dec_04_att_value0_output', shape=(None, 512))
layer root/'dec_04_att_value' output: Data(name='dec_04_att_value_output', shape=(None, 8, 64))
layer root/'dec_07_att_value0' output: Data(name='dec_07_att_value0_output', shape=(None, 512))
layer root/'dec_07_att_value' output: Data(name='dec_07_att_value_output', shape=(None, 8, 64))
layer root/'dec_11_att_key0' output: Data(name='dec_11_att_key0_output', shape=(None, 512))
layer root/'dec_11_att_key' output: Data(name='dec_11_att_key_output', shape=(None, 8, 64))
layer root/'dec_02_att_key0' output: Data(name='dec_02_att_key0_output', shape=(None, 512))
layer root/'dec_02_att_key' output: Data(name='dec_02_att_key_output', shape=(None, 8, 64))
layer root/'dec_07_att_key0' output: Data(name='dec_07_att_key0_output', shape=(None, 512))
layer root/'dec_07_att_key' output: Data(name='dec_07_att_key_output', shape=(None, 8, 64))
layer root/'dec_03_att_key0' output: Data(name='dec_03_att_key0_output', shape=(None, 512))
layer root/'dec_03_att_key' output: Data(name='dec_03_att_key_output', shape=(None, 8, 64))
layer root/'dec_12_att_key0' output: Data(name='dec_12_att_key0_output', shape=(None, 512))
layer root/'dec_12_att_key' output: Data(name='dec_12_att_key_output', shape=(None, 8, 64))
layer root/'dec_09_att_key0' output: Data(name='dec_09_att_key0_output', shape=(None, 512))
layer root/'dec_09_att_key' output: Data(name='dec_09_att_key_output', shape=(None, 8, 64))
layer root/'dec_10_att_value0' output: Data(name='dec_10_att_value0_output', shape=(None, 512))
layer root/'dec_10_att_value' output: Data(name='dec_10_att_value_output', shape=(None, 8, 64))
layer root/'dec_01_att_value0' output: Data(name='dec_01_att_value0_output', shape=(None, 512))
layer root/'dec_01_att_value' output: Data(name='dec_01_att_value_output', shape=(None, 8, 64))
layer root/'dec_06_att_value0' output: Data(name='dec_06_att_value0_output', shape=(None, 512))
layer root/'dec_06_att_value' output: Data(name='dec_06_att_value_output', shape=(None, 8, 64))
layer root/'dec_08_att_key0' output: Data(name='dec_08_att_key0_output', shape=(None, 512))
layer root/'dec_08_att_key' output: Data(name='dec_08_att_key_output', shape=(None, 8, 64))
layer root/'dec_12_att_value0' output: Data(name='dec_12_att_value0_output', shape=(None, 512))
layer root/'dec_12_att_value' output: Data(name='dec_12_att_value_output', shape=(None, 8, 64))
layer root/'dec_11_att_value0' output: Data(name='dec_11_att_value0_output', shape=(None, 512))
layer root/'dec_11_att_value' output: Data(name='dec_11_att_value_output', shape=(None, 8, 64))
layer root/'dec_09_att_value0' output: Data(name='dec_09_att_value0_output', shape=(None, 512))
layer root/'dec_09_att_value' output: Data(name='dec_09_att_value_output', shape=(None, 8, 64))
layer root/'dec_05_att_value0' output: Data(name='dec_05_att_value0_output', shape=(None, 512))
layer root/'dec_05_att_value' output: Data(name='dec_05_att_value_output', shape=(None, 8, 64))
layer root/'dec_03_att_value0' output: Data(name='dec_03_att_value0_output', shape=(None, 512))
layer root/'dec_03_att_value' output: Data(name='dec_03_att_value_output', shape=(None, 8, 64))
layer root/'dec_01_att_key0' output: Data(name='dec_01_att_key0_output', shape=(None, 512))
layer root/'dec_01_att_key' output: Data(name='dec_01_att_key_output', shape=(None, 8, 64))
layer root/'dec_02_att_value0' output: Data(name='dec_02_att_value0_output', shape=(None, 512))
layer root/'dec_02_att_value' output: Data(name='dec_02_att_value_output', shape=(None, 8, 64))
layer root/'dec_05_att_key0' output: Data(name='dec_05_att_key0_output', shape=(None, 512))
layer root/'dec_05_att_key' output: Data(name='dec_05_att_key_output', shape=(None, 8, 64))
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
+------- EPILOGUE SCRIPT -----------------------------------------------
|
| Job ID ..............: 9505968
| Stopped at ..........: Tue Jul  2 12:02:59 CEST 2019
| Resources requested .: h_vmem=1536G,h_rss=4G,scratch_free=5G,gpu=1,num_proc=5,h_fsize=20G,h_rt=7200,s_core=0,pxe=ubuntu_16.04
| Resources used ......: cpu=00:00:33, mem=17.37517 GB s, io=1.21346 GB, vmem=1.126G, maxvmem=1.940G, last_file_cache=116M, last_rss=3M, max-cache=1.464G
| Memory used .........: 1.577G / 4.000G (39.4%)
| Total time used .....: 0:01:41
|
+------- EPILOGUE SCRIPT -----------------------------------------------
+------- PROLOGUE SCRIPT -----------------------------------------------
|
| Job ID ...........: 9505979
| Started at .......: Tue Jul  2 12:06:18 CEST 2019
| Execution host ...: cluster-cn-240
| Cluster queue ....: 4-GPU-1080
| Script ...........: /var/spool/sge/cluster-cn-240/job_scripts/9505979
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
RETURNN starting up, version 20181022.053313--git-dfada43-dirty, date/time 2019-07-02-12-06-21 (UTC+0200), pid 26404, cwd /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc, Python /u/bahar/settings/python3-returnn-tf1.9/bin/python3
RETURNN config: /u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc.config
RETURNN command line options: ()
Hostname: cluster-cn-240
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
       incarnation: 14797591839462367839
  2/2: name: "/device:GPU:0"
       device_type: "GPU"
       memory_limit: 10910862541
       locality {
         bus_id: 1
         links {
         }
       }
       incarnation: 8383558901879264655
       physical_device_desc: "device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1"
Using gpu device 0: GeForce GTX 1080 Ti
Device not set explicitly, and we found a GPU, which we will use.
Model file prefix: /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc/net-model/network
NOTE: We will use 'sorted_reverse' seq ordering.
Setup tf.Session with options {'log_device_placement': False, 'device_count': {'GPU': 1}} ...
layer root/'data' output: Data(name='data', shape=(None, 40))
layer root/'source' output: Data(name='source_output', shape=(None, 40))
layer root/'lstm0_fw' output: Data(name='lstm0_fw_output', shape=(None, 1024), batch_dim_axis=1)
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9505979.1.4-GPU-1080/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.cc -o /var/tmp/9505979.1.4-GPU-1080/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9505979.1.4-GPU-1080/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.cc -o /var/tmp/9505979.1.4-GPU-1080/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
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
layer root/'dec_05_att_value0' output: Data(name='dec_05_att_value0_output', shape=(None, 512))
layer root/'dec_05_att_value' output: Data(name='dec_05_att_value_output', shape=(None, 8, 64))
layer root/'dec_06_att_value0' output: Data(name='dec_06_att_value0_output', shape=(None, 512))
layer root/'dec_06_att_value' output: Data(name='dec_06_att_value_output', shape=(None, 8, 64))
layer root/'dec_07_att_key0' output: Data(name='dec_07_att_key0_output', shape=(None, 512))
layer root/'dec_07_att_key' output: Data(name='dec_07_att_key_output', shape=(None, 8, 64))
layer root/'dec_10_att_key0' output: Data(name='dec_10_att_key0_output', shape=(None, 512))
layer root/'dec_10_att_key' output: Data(name='dec_10_att_key_output', shape=(None, 8, 64))
layer root/'dec_08_att_value0' output: Data(name='dec_08_att_value0_output', shape=(None, 512))
layer root/'dec_08_att_value' output: Data(name='dec_08_att_value_output', shape=(None, 8, 64))
layer root/'dec_12_att_value0' output: Data(name='dec_12_att_value0_output', shape=(None, 512))
layer root/'dec_12_att_value' output: Data(name='dec_12_att_value_output', shape=(None, 8, 64))
layer root/'dec_08_att_key0' output: Data(name='dec_08_att_key0_output', shape=(None, 512))
layer root/'dec_08_att_key' output: Data(name='dec_08_att_key_output', shape=(None, 8, 64))
layer root/'dec_01_att_value0' output: Data(name='dec_01_att_value0_output', shape=(None, 512))
layer root/'dec_01_att_value' output: Data(name='dec_01_att_value_output', shape=(None, 8, 64))
layer root/'dec_04_att_value0' output: Data(name='dec_04_att_value0_output', shape=(None, 512))
layer root/'dec_04_att_value' output: Data(name='dec_04_att_value_output', shape=(None, 8, 64))
layer root/'dec_04_att_key0' output: Data(name='dec_04_att_key0_output', shape=(None, 512))
layer root/'dec_04_att_key' output: Data(name='dec_04_att_key_output', shape=(None, 8, 64))
layer root/'dec_01_att_key0' output: Data(name='dec_01_att_key0_output', shape=(None, 512))
layer root/'dec_01_att_key' output: Data(name='dec_01_att_key_output', shape=(None, 8, 64))
layer root/'dec_06_att_key0' output: Data(name='dec_06_att_key0_output', shape=(None, 512))
layer root/'dec_06_att_key' output: Data(name='dec_06_att_key_output', shape=(None, 8, 64))
layer root/'dec_09_att_value0' output: Data(name='dec_09_att_value0_output', shape=(None, 512))
layer root/'dec_09_att_value' output: Data(name='dec_09_att_value_output', shape=(None, 8, 64))
layer root/'dec_05_att_key0' output: Data(name='dec_05_att_key0_output', shape=(None, 512))
layer root/'dec_05_att_key' output: Data(name='dec_05_att_key_output', shape=(None, 8, 64))
layer root/'dec_10_att_value0' output: Data(name='dec_10_att_value0_output', shape=(None, 512))
layer root/'dec_10_att_value' output: Data(name='dec_10_att_value_output', shape=(None, 8, 64))
layer root/'dec_03_att_value0' output: Data(name='dec_03_att_value0_output', shape=(None, 512))
layer root/'dec_03_att_value' output: Data(name='dec_03_att_value_output', shape=(None, 8, 64))
layer root/'dec_02_att_value0' output: Data(name='dec_02_att_value0_output', shape=(None, 512))
layer root/'dec_02_att_value' output: Data(name='dec_02_att_value_output', shape=(None, 8, 64))
layer root/'dec_11_att_key0' output: Data(name='dec_11_att_key0_output', shape=(None, 512))
layer root/'dec_11_att_key' output: Data(name='dec_11_att_key_output', shape=(None, 8, 64))
layer root/'dec_03_att_key0' output: Data(name='dec_03_att_key0_output', shape=(None, 512))
layer root/'dec_03_att_key' output: Data(name='dec_03_att_key_output', shape=(None, 8, 64))
layer root/'dec_02_att_key0' output: Data(name='dec_02_att_key0_output', shape=(None, 512))
layer root/'dec_02_att_key' output: Data(name='dec_02_att_key_output', shape=(None, 8, 64))
layer root/'dec_07_att_value0' output: Data(name='dec_07_att_value0_output', shape=(None, 512))
layer root/'dec_07_att_value' output: Data(name='dec_07_att_value_output', shape=(None, 8, 64))
layer root/'dec_12_att_key0' output: Data(name='dec_12_att_key0_output', shape=(None, 512))
layer root/'dec_12_att_key' output: Data(name='dec_12_att_key_output', shape=(None, 8, 64))
layer root/'dec_09_att_key0' output: Data(name='dec_09_att_key0_output', shape=(None, 512))
layer root/'dec_09_att_key' output: Data(name='dec_09_att_key_output', shape=(None, 8, 64))
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
TF: log_dir: /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/transformer/forward/tf_log_dir/prefix:dev-clean-481-2019-07-02-10-06-19
warning: sequence length (3215) larger than limit (600)
warning: sequence length (3249) larger than limit (600)
warning: sequence length (3265) larger than limit (600)
warning: sequence length (3232) larger than limit (600)
warning: sequence length (3165) larger than limit (600)
warning: sequence length (3245) larger than limit (600)
warning: sequence length (3206) larger than limit (600)
warning: sequence length (2895) larger than limit (600)
warning: sequence length (3171) larger than limit (600)
warning: sequence length (3138) larger than limit (600)
warning: sequence length (2880) larger than limit (600)
Note: There are still these uninitialized variables: ['learning_rate:0']
warning: sequence length (2413) larger than limit (600)
warning: sequence length (2789) larger than limit (600)
att-weights epoch 481, step 0, max_size:classes 122, max_size:data 3215, mem_usage:GPU:0 812.6MB, num_seqs 1, 10.119 sec/step, elapsed 0:00:16, exp. remaining 1:09:19, complete 0.41%
warning: sequence length (2858) larger than limit (600)
att-weights epoch 481, step 1, max_size:classes 111, max_size:data 3249, mem_usage:GPU:0 812.6MB, num_seqs 1, 8.696 sec/step, elapsed 0:00:38, exp. remaining 2:23:00, complete 0.44%
warning: sequence length (2437) larger than limit (600)
att-weights epoch 481, step 2, max_size:classes 103, max_size:data 3265, mem_usage:GPU:0 812.6MB, num_seqs 1, 1.291 sec/step, elapsed 0:00:39, exp. remaining 2:16:49, complete 0.48%
warning: sequence length (2345) larger than limit (600)
att-weights epoch 481, step 3, max_size:classes 102, max_size:data 3232, mem_usage:GPU:0 812.6MB, num_seqs 1, 1.269 sec/step, elapsed 0:00:40, exp. remaining 2:11:12, complete 0.52%
warning: sequence length (2269) larger than limit (600)
att-weights epoch 481, step 4, max_size:classes 100, max_size:data 3165, mem_usage:GPU:0 812.6MB, num_seqs 1, 2.546 sec/step, elapsed 0:00:43, exp. remaining 2:10:16, complete 0.55%
warning: sequence length (2404) larger than limit (600)
att-weights epoch 481, step 5, max_size:classes 98, max_size:data 3245, mem_usage:GPU:0 812.6MB, num_seqs 1, 5.942 sec/step, elapsed 0:00:49, exp. remaining 2:18:53, complete 0.59%
warning: sequence length (2800) larger than limit (600)
att-weights epoch 481, step 6, max_size:classes 99, max_size:data 3206, mem_usage:GPU:0 812.6MB, num_seqs 1, 1.277 sec/step, elapsed 0:00:50, exp. remaining 2:14:08, complete 0.63%
warning: sequence length (2440) larger than limit (600)
att-weights epoch 481, step 7, max_size:classes 90, max_size:data 2895, mem_usage:GPU:0 812.6MB, num_seqs 1, 1.421 sec/step, elapsed 0:00:52, exp. remaining 2:10:19, complete 0.67%
warning: sequence length (2377) larger than limit (600)
att-weights epoch 481, step 8, max_size:classes 98, max_size:data 3171, mem_usage:GPU:0 812.6MB, num_seqs 1, 3.382 sec/step, elapsed 0:00:55, exp. remaining 2:11:28, complete 0.70%
warning: sequence length (2663) larger than limit (600)
att-weights epoch 481, step 9, max_size:classes 94, max_size:data 3138, mem_usage:GPU:0 812.6MB, num_seqs 1, 1.296 sec/step, elapsed 0:00:57, exp. remaining 2:07:51, complete 0.74%
warning: sequence length (2841) larger than limit (600)
att-weights epoch 481, step 10, max_size:classes 97, max_size:data 2880, mem_usage:GPU:0 812.6MB, num_seqs 1, 1.167 sec/step, elapsed 0:00:58, exp. remaining 2:04:26, complete 0.78%
warning: sequence length (2907) larger than limit (600)
+------- EPILOGUE SCRIPT -----------------------------------------------
|
| Job ID ..............: 9505979
| Stopped at ..........: Tue Jul  2 12:09:51 CEST 2019
| Resources requested .: num_proc=5,scratch_free=5G,gpu=1,s_core=0,pxe=ubuntu_16.04,h_fsize=20G,h_rt=7200,h_rss=4G,h_vmem=1536G
| Resources used ......: cpu=00:02:32, mem=245.64883 GB s, io=1.92151 GB, vmem=2.958G, maxvmem=3.838G, last_file_cache=1.889G, last_rss=3M, max-cache=1.607G
| Memory used .........: 3.497G / 4.000G (87.4%)
| Total time used .....: 0:03:31
|
+------- EPILOGUE SCRIPT -----------------------------------------------
+------- PROLOGUE SCRIPT -----------------------------------------------
|
| Job ID ...........: 9505992
| Started at .......: Tue Jul  2 12:10:40 CEST 2019
| Execution host ...: cluster-cn-216
| Cluster queue ....: 3-GPU-1080
| Script ...........: /var/spool/sge/cluster-cn-216/job_scripts/9505992
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
RETURNN starting up, version 20181022.053313--git-dfada43-dirty, date/time 2019-07-02-12-10-44 (UTC+0200), pid 16739, cwd /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc, Python /u/bahar/settings/python3-returnn-tf1.9/bin/python3
RETURNN config: /u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc.config
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
       incarnation: 8502138118260999202
  2/2: name: "/device:GPU:0"
       device_type: "GPU"
       memory_limit: 10911236096
       locality {
         bus_id: 1
         links {
         }
       }
       incarnation: 6513728009298124783
       physical_device_desc: "device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1"
Using gpu device 0: GeForce GTX 1080 Ti
Device not set explicitly, and we found a GPU, which we will use.
Model file prefix: /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc/net-model/network
NOTE: We will use 'sorted_reverse' seq ordering.
Setup tf.Session with options {'log_device_placement': False, 'device_count': {'GPU': 1}} ...
layer root/'data' output: Data(name='data', shape=(None, 40))
layer root/'source' output: Data(name='source_output', shape=(None, 40))
layer root/'lstm0_fw' output: Data(name='lstm0_fw_output', shape=(None, 1024), batch_dim_axis=1)
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9505992.1.3-GPU-1080/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.cc -o /var/tmp/9505992.1.3-GPU-1080/makarov/returnn_tf_cache/ops/NativeLstm2/54c5e8856f/NativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9505992.1.3-GPU-1080/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.cc -o /var/tmp/9505992.1.3-GPU-1080/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef4d3ea334/GradOfNativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
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
layer root/'dec_02_att_value0' output: Data(name='dec_02_att_value0_output', shape=(None, 512))
layer root/'dec_02_att_value' output: Data(name='dec_02_att_value_output', shape=(None, 8, 64))
layer root/'dec_02_att_key0' output: Data(name='dec_02_att_key0_output', shape=(None, 512))
layer root/'dec_02_att_key' output: Data(name='dec_02_att_key_output', shape=(None, 8, 64))
layer root/'dec_04_att_value0' output: Data(name='dec_04_att_value0_output', shape=(None, 512))
layer root/'dec_04_att_value' output: Data(name='dec_04_att_value_output', shape=(None, 8, 64))
layer root/'dec_03_att_value0' output: Data(name='dec_03_att_value0_output', shape=(None, 512))
layer root/'dec_03_att_value' output: Data(name='dec_03_att_value_output', shape=(None, 8, 64))
layer root/'dec_01_att_key0' output: Data(name='dec_01_att_key0_output', shape=(None, 512))
layer root/'dec_01_att_key' output: Data(name='dec_01_att_key_output', shape=(None, 8, 64))
layer root/'dec_01_att_value0' output: Data(name='dec_01_att_value0_output', shape=(None, 512))
layer root/'dec_01_att_value' output: Data(name='dec_01_att_value_output', shape=(None, 8, 64))
layer root/'dec_05_att_key0' output: Data(name='dec_05_att_key0_output', shape=(None, 512))
layer root/'dec_05_att_key' output: Data(name='dec_05_att_key_output', shape=(None, 8, 64))
layer root/'dec_06_att_key0' output: Data(name='dec_06_att_key0_output', shape=(None, 512))
layer root/'dec_06_att_key' output: Data(name='dec_06_att_key_output', shape=(None, 8, 64))
layer root/'dec_03_att_key0' output: Data(name='dec_03_att_key0_output', shape=(None, 512))
layer root/'dec_03_att_key' output: Data(name='dec_03_att_key_output', shape=(None, 8, 64))
layer root/'dec_12_att_key0' output: Data(name='dec_12_att_key0_output', shape=(None, 512))
layer root/'dec_12_att_key' output: Data(name='dec_12_att_key_output', shape=(None, 8, 64))
layer root/'dec_07_att_value0' output: Data(name='dec_07_att_value0_output', shape=(None, 512))
layer root/'dec_07_att_value' output: Data(name='dec_07_att_value_output', shape=(None, 8, 64))
layer root/'dec_08_att_key0' output: Data(name='dec_08_att_key0_output', shape=(None, 512))
layer root/'dec_08_att_key' output: Data(name='dec_08_att_key_output', shape=(None, 8, 64))
layer root/'dec_06_att_value0' output: Data(name='dec_06_att_value0_output', shape=(None, 512))
layer root/'dec_06_att_value' output: Data(name='dec_06_att_value_output', shape=(None, 8, 64))
layer root/'dec_08_att_value0' output: Data(name='dec_08_att_value0_output', shape=(None, 512))
layer root/'dec_08_att_value' output: Data(name='dec_08_att_value_output', shape=(None, 8, 64))
layer root/'dec_10_att_key0' output: Data(name='dec_10_att_key0_output', shape=(None, 512))
layer root/'dec_10_att_key' output: Data(name='dec_10_att_key_output', shape=(None, 8, 64))
layer root/'dec_10_att_value0' output: Data(name='dec_10_att_value0_output', shape=(None, 512))
layer root/'dec_10_att_value' output: Data(name='dec_10_att_value_output', shape=(None, 8, 64))
layer root/'dec_05_att_value0' output: Data(name='dec_05_att_value0_output', shape=(None, 512))
layer root/'dec_05_att_value' output: Data(name='dec_05_att_value_output', shape=(None, 8, 64))
layer root/'dec_09_att_value0' output: Data(name='dec_09_att_value0_output', shape=(None, 512))
layer root/'dec_09_att_value' output: Data(name='dec_09_att_value_output', shape=(None, 8, 64))
layer root/'dec_11_att_key0' output: Data(name='dec_11_att_key0_output', shape=(None, 512))
layer root/'dec_11_att_key' output: Data(name='dec_11_att_key_output', shape=(None, 8, 64))
layer root/'dec_07_att_key0' output: Data(name='dec_07_att_key0_output', shape=(None, 512))
layer root/'dec_07_att_key' output: Data(name='dec_07_att_key_output', shape=(None, 8, 64))
layer root/'dec_04_att_key0' output: Data(name='dec_04_att_key0_output', shape=(None, 512))
layer root/'dec_04_att_key' output: Data(name='dec_04_att_key_output', shape=(None, 8, 64))
layer root/'dec_12_att_value0' output: Data(name='dec_12_att_value0_output', shape=(None, 512))
layer root/'dec_12_att_value' output: Data(name='dec_12_att_value_output', shape=(None, 8, 64))
layer root/'dec_11_att_value0' output: Data(name='dec_11_att_value0_output', shape=(None, 512))
layer root/'dec_11_att_value' output: Data(name='dec_11_att_value_output', shape=(None, 8, 64))
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
TF: log_dir: /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/transformer/forward/tf_log_dir/prefix:dev-clean-481-2019-07-02-10-10-42
Note: There are still these uninitialized variables: ['learning_rate:0']
att-weights epoch 481, step 0, max_size:classes 122, max_size:data 3215, mem_usage:GPU:0 1.0GB, num_seqs 1, 5.550 sec/step, elapsed 0:00:12, exp. remaining 0:49:21, complete 0.41%
att-weights epoch 481, step 1, max_size:classes 111, max_size:data 3249, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.532 sec/step, elapsed 0:00:13, exp. remaining 0:51:32, complete 0.44%
att-weights epoch 481, step 2, max_size:classes 103, max_size:data 3265, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.404 sec/step, elapsed 0:00:15, exp. remaining 0:52:53, complete 0.48%
att-weights epoch 481, step 3, max_size:classes 102, max_size:data 3232, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.445 sec/step, elapsed 0:00:16, exp. remaining 0:54:04, complete 0.52%
att-weights epoch 481, step 4, max_size:classes 100, max_size:data 3165, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.325 sec/step, elapsed 0:00:18, exp. remaining 0:54:46, complete 0.55%
att-weights epoch 481, step 5, max_size:classes 98, max_size:data 3245, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.397 sec/step, elapsed 0:00:19, exp. remaining 0:55:32, complete 0.59%
att-weights epoch 481, step 6, max_size:classes 99, max_size:data 3206, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.327 sec/step, elapsed 0:00:21, exp. remaining 0:56:10, complete 0.63%
att-weights epoch 481, step 7, max_size:classes 90, max_size:data 2895, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.184 sec/step, elapsed 0:00:22, exp. remaining 0:56:20, complete 0.67%
att-weights epoch 481, step 8, max_size:classes 98, max_size:data 3171, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.348 sec/step, elapsed 0:00:24, exp. remaining 0:56:48, complete 0.70%
att-weights epoch 481, step 9, max_size:classes 94, max_size:data 3138, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.320 sec/step, elapsed 0:00:25, exp. remaining 0:57:08, complete 0.74%
att-weights epoch 481, step 10, max_size:classes 97, max_size:data 2880, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.268 sec/step, elapsed 0:00:26, exp. remaining 0:57:21, complete 0.78%
att-weights epoch 481, step 11, max_size:classes 80, max_size:data 2413, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.906 sec/step, elapsed 0:00:27, exp. remaining 0:56:50, complete 0.81%
att-weights epoch 481, step 12, max_size:classes 86, max_size:data 2789, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.200 sec/step, elapsed 0:00:29, exp. remaining 0:56:40, complete 0.85%
att-weights epoch 481, step 13, max_size:classes 91, max_size:data 2858, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.228 sec/step, elapsed 0:00:30, exp. remaining 0:56:35, complete 0.89%
att-weights epoch 481, step 14, max_size:classes 85, max_size:data 2437, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.118 sec/step, elapsed 0:00:31, exp. remaining 0:56:17, complete 0.92%
att-weights epoch 481, step 15, max_size:classes 87, max_size:data 2345, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.832 sec/step, elapsed 0:00:33, exp. remaining 0:57:15, complete 0.96%
att-weights epoch 481, step 16, max_size:classes 78, max_size:data 2269, mem_usage:GPU:0 1.0GB, num_seqs 1, 3.746 sec/step, elapsed 0:00:37, exp. remaining 1:01:18, complete 1.00%
att-weights epoch 481, step 17, max_size:classes 81, max_size:data 2404, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.964 sec/step, elapsed 0:00:38, exp. remaining 1:00:37, complete 1.04%
att-weights epoch 481, step 18, max_size:classes 77, max_size:data 2800, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.120 sec/step, elapsed 0:00:39, exp. remaining 1:00:14, complete 1.07%
att-weights epoch 481, step 19, max_size:classes 76, max_size:data 2440, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.041 sec/step, elapsed 0:00:40, exp. remaining 0:59:45, complete 1.11%
att-weights epoch 481, step 20, max_size:classes 82, max_size:data 2377, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.921 sec/step, elapsed 0:00:41, exp. remaining 0:59:07, complete 1.15%
att-weights epoch 481, step 21, max_size:classes 86, max_size:data 2663, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.113 sec/step, elapsed 0:00:42, exp. remaining 0:58:48, complete 1.18%
att-weights epoch 481, step 22, max_size:classes 89, max_size:data 2841, mem_usage:GPU:0 1.0GB, num_seqs 1, 3.686 sec/step, elapsed 0:00:45, exp. remaining 1:01:58, complete 1.22%
att-weights epoch 481, step 23, max_size:classes 82, max_size:data 2907, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.192 sec/step, elapsed 0:00:47, exp. remaining 1:01:41, complete 1.26%
att-weights epoch 481, step 24, max_size:classes 79, max_size:data 2453, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.212 sec/step, elapsed 0:00:48, exp. remaining 1:01:26, complete 1.29%
att-weights epoch 481, step 25, max_size:classes 71, max_size:data 2287, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.874 sec/step, elapsed 0:00:49, exp. remaining 1:00:48, complete 1.33%
att-weights epoch 481, step 26, max_size:classes 77, max_size:data 2308, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.914 sec/step, elapsed 0:00:50, exp. remaining 1:00:14, complete 1.37%
att-weights epoch 481, step 27, max_size:classes 81, max_size:data 2282, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.888 sec/step, elapsed 0:00:51, exp. remaining 0:59:39, complete 1.41%
att-weights epoch 481, step 28, max_size:classes 81, max_size:data 2100, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.811 sec/step, elapsed 0:00:51, exp. remaining 0:59:02, complete 1.44%
att-weights epoch 481, step 29, max_size:classes 94, max_size:data 2941, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.318 sec/step, elapsed 0:00:53, exp. remaining 0:59:00, complete 1.48%
att-weights epoch 481, step 30, max_size:classes 73, max_size:data 2089, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.860 sec/step, elapsed 0:00:54, exp. remaining 0:58:28, complete 1.52%
att-weights epoch 481, step 31, max_size:classes 73, max_size:data 2300, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.009 sec/step, elapsed 0:00:55, exp. remaining 0:58:07, complete 1.55%
att-weights epoch 481, step 32, max_size:classes 76, max_size:data 2194, mem_usage:GPU:0 1.0GB, num_seqs 1, 4.967 sec/step, elapsed 0:01:00, exp. remaining 1:01:52, complete 1.59%
att-weights epoch 481, step 33, max_size:classes 75, max_size:data 2297, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.415 sec/step, elapsed 0:01:01, exp. remaining 1:01:52, complete 1.63%
att-weights epoch 481, step 34, max_size:classes 89, max_size:data 2481, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.470 sec/step, elapsed 0:01:02, exp. remaining 1:01:55, complete 1.66%
att-weights epoch 481, step 35, max_size:classes 60, max_size:data 2342, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.460 sec/step, elapsed 0:01:04, exp. remaining 1:01:57, complete 1.70%
att-weights epoch 481, step 36, max_size:classes 72, max_size:data 1928, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.883 sec/step, elapsed 0:01:05, exp. remaining 1:00:08, complete 1.78%
att-weights epoch 481, step 37, max_size:classes 66, max_size:data 2352, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.098 sec/step, elapsed 0:01:06, exp. remaining 0:59:53, complete 1.81%
att-weights epoch 481, step 38, max_size:classes 66, max_size:data 1990, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.731 sec/step, elapsed 0:01:07, exp. remaining 0:58:07, complete 1.89%
att-weights epoch 481, step 39, max_size:classes 78, max_size:data 2234, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.512 sec/step, elapsed 0:01:08, exp. remaining 0:57:09, complete 1.96%
att-weights epoch 481, step 40, max_size:classes 81, max_size:data 2048, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.038 sec/step, elapsed 0:01:09, exp. remaining 0:56:55, complete 2.00%
att-weights epoch 481, step 41, max_size:classes 71, max_size:data 1781, mem_usage:GPU:0 1.0GB, num_seqs 1, 2.872 sec/step, elapsed 0:01:12, exp. remaining 0:58:10, complete 2.03%
att-weights epoch 481, step 42, max_size:classes 67, max_size:data 2118, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.835 sec/step, elapsed 0:01:13, exp. remaining 0:56:44, complete 2.11%
att-weights epoch 481, step 43, max_size:classes 66, max_size:data 1992, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.803 sec/step, elapsed 0:01:14, exp. remaining 0:56:21, complete 2.15%
att-weights epoch 481, step 44, max_size:classes 67, max_size:data 2065, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.823 sec/step, elapsed 0:01:14, exp. remaining 0:55:02, complete 2.22%
att-weights epoch 481, step 45, max_size:classes 78, max_size:data 1944, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.571 sec/step, elapsed 0:01:16, exp. remaining 0:55:14, complete 2.26%
att-weights epoch 481, step 46, max_size:classes 71, max_size:data 2042, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.894 sec/step, elapsed 0:01:17, exp. remaining 0:54:04, complete 2.33%
att-weights epoch 481, step 47, max_size:classes 64, max_size:data 1818, mem_usage:GPU:0 1.0GB, num_seqs 2, 4.219 sec/step, elapsed 0:01:21, exp. remaining 0:55:13, complete 2.40%
att-weights epoch 481, step 48, max_size:classes 70, max_size:data 1979, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.309 sec/step, elapsed 0:01:22, exp. remaining 0:54:23, complete 2.48%
att-weights epoch 481, step 49, max_size:classes 62, max_size:data 2712, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.039 sec/step, elapsed 0:01:24, exp. remaining 0:53:26, complete 2.55%
att-weights epoch 481, step 50, max_size:classes 69, max_size:data 2238, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.857 sec/step, elapsed 0:01:24, exp. remaining 0:53:11, complete 2.59%
att-weights epoch 481, step 51, max_size:classes 67, max_size:data 1936, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.265 sec/step, elapsed 0:01:26, exp. remaining 0:52:27, complete 2.66%
att-weights epoch 481, step 52, max_size:classes 64, max_size:data 2163, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.766 sec/step, elapsed 0:01:26, exp. remaining 0:52:10, complete 2.70%
att-weights epoch 481, step 53, max_size:classes 67, max_size:data 1913, mem_usage:GPU:0 1.0GB, num_seqs 2, 4.649 sec/step, elapsed 0:01:31, exp. remaining 0:54:12, complete 2.74%
att-weights epoch 481, step 54, max_size:classes 66, max_size:data 2073, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.978 sec/step, elapsed 0:01:32, exp. remaining 0:54:01, complete 2.77%
att-weights epoch 481, step 55, max_size:classes 66, max_size:data 1718, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.254 sec/step, elapsed 0:01:33, exp. remaining 0:53:18, complete 2.85%
att-weights epoch 481, step 56, max_size:classes 64, max_size:data 1855, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.258 sec/step, elapsed 0:01:35, exp. remaining 0:52:36, complete 2.92%
att-weights epoch 481, step 57, max_size:classes 68, max_size:data 1633, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.179 sec/step, elapsed 0:01:36, exp. remaining 0:52:34, complete 2.96%
att-weights epoch 481, step 58, max_size:classes 65, max_size:data 1720, mem_usage:GPU:0 1.0GB, num_seqs 2, 8.895 sec/step, elapsed 0:01:45, exp. remaining 0:56:42, complete 3.00%
att-weights epoch 481, step 59, max_size:classes 59, max_size:data 2040, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.944 sec/step, elapsed 0:01:46, exp. remaining 0:56:29, complete 3.03%
att-weights epoch 481, step 60, max_size:classes 64, max_size:data 1647, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.219 sec/step, elapsed 0:01:48, exp. remaining 0:56:57, complete 3.07%
att-weights epoch 481, step 61, max_size:classes 60, max_size:data 1636, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.718 sec/step, elapsed 0:01:48, exp. remaining 0:56:38, complete 3.11%
att-weights epoch 481, step 62, max_size:classes 67, max_size:data 2318, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.129 sec/step, elapsed 0:01:50, exp. remaining 0:56:31, complete 3.14%
att-weights epoch 481, step 63, max_size:classes 54, max_size:data 2054, mem_usage:GPU:0 1.0GB, num_seqs 1, 7.862 sec/step, elapsed 0:01:57, exp. remaining 0:59:07, complete 3.22%
att-weights epoch 481, step 64, max_size:classes 61, max_size:data 1758, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.395 sec/step, elapsed 0:01:59, exp. remaining 0:58:26, complete 3.29%
att-weights epoch 481, step 65, max_size:classes 63, max_size:data 1969, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.416 sec/step, elapsed 0:02:00, exp. remaining 0:57:47, complete 3.37%
att-weights epoch 481, step 66, max_size:classes 64, max_size:data 1563, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.728 sec/step, elapsed 0:02:01, exp. remaining 0:56:50, complete 3.44%
att-weights epoch 481, step 67, max_size:classes 60, max_size:data 2087, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.932 sec/step, elapsed 0:02:03, exp. remaining 0:56:29, complete 3.51%
att-weights epoch 481, step 68, max_size:classes 55, max_size:data 1896, mem_usage:GPU:0 1.0GB, num_seqs 1, 7.500 sec/step, elapsed 0:02:10, exp. remaining 0:59:16, complete 3.55%
att-weights epoch 481, step 69, max_size:classes 60, max_size:data 2691, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.115 sec/step, elapsed 0:02:12, exp. remaining 0:59:08, complete 3.59%
att-weights epoch 481, step 70, max_size:classes 66, max_size:data 2057, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.837 sec/step, elapsed 0:02:12, exp. remaining 0:58:16, complete 3.66%
att-weights epoch 481, step 71, max_size:classes 54, max_size:data 2032, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.754 sec/step, elapsed 0:02:13, exp. remaining 0:57:23, complete 3.74%
att-weights epoch 481, step 72, max_size:classes 64, max_size:data 1642, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.088 sec/step, elapsed 0:02:14, exp. remaining 0:56:41, complete 3.81%
att-weights epoch 481, step 73, max_size:classes 61, max_size:data 1911, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.235 sec/step, elapsed 0:02:15, exp. remaining 0:56:38, complete 3.85%
att-weights epoch 481, step 74, max_size:classes 62, max_size:data 1930, mem_usage:GPU:0 1.0GB, num_seqs 2, 3.432 sec/step, elapsed 0:02:19, exp. remaining 0:57:29, complete 3.88%
att-weights epoch 481, step 75, max_size:classes 62, max_size:data 1788, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.192 sec/step, elapsed 0:02:20, exp. remaining 0:56:51, complete 3.96%
att-weights epoch 481, step 76, max_size:classes 59, max_size:data 1671, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.397 sec/step, elapsed 0:02:22, exp. remaining 0:56:52, complete 4.00%
att-weights epoch 481, step 77, max_size:classes 61, max_size:data 1887, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.813 sec/step, elapsed 0:02:22, exp. remaining 0:56:39, complete 4.03%
att-weights epoch 481, step 78, max_size:classes 69, max_size:data 2029, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.830 sec/step, elapsed 0:02:23, exp. remaining 0:55:54, complete 4.11%
att-weights epoch 481, step 79, max_size:classes 65, max_size:data 1733, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.140 sec/step, elapsed 0:02:24, exp. remaining 0:55:18, complete 4.18%
att-weights epoch 481, step 80, max_size:classes 62, max_size:data 1778, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.342 sec/step, elapsed 0:02:26, exp. remaining 0:55:19, complete 4.22%
att-weights epoch 481, step 81, max_size:classes 65, max_size:data 1766, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.397 sec/step, elapsed 0:02:27, exp. remaining 0:55:20, complete 4.25%
att-weights epoch 481, step 82, max_size:classes 53, max_size:data 1463, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.921 sec/step, elapsed 0:02:28, exp. remaining 0:54:41, complete 4.33%
att-weights epoch 481, step 83, max_size:classes 69, max_size:data 2096, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.834 sec/step, elapsed 0:02:29, exp. remaining 0:54:01, complete 4.40%
att-weights epoch 481, step 84, max_size:classes 59, max_size:data 1629, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.363 sec/step, elapsed 0:02:30, exp. remaining 0:53:34, complete 4.48%
att-weights epoch 481, step 85, max_size:classes 68, max_size:data 1529, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.757 sec/step, elapsed 0:02:31, exp. remaining 0:52:56, complete 4.55%
att-weights epoch 481, step 86, max_size:classes 55, max_size:data 2056, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.792 sec/step, elapsed 0:02:32, exp. remaining 0:52:19, complete 4.62%
att-weights epoch 481, step 87, max_size:classes 59, max_size:data 1778, mem_usage:GPU:0 1.0GB, num_seqs 2, 3.901 sec/step, elapsed 0:02:36, exp. remaining 0:52:46, complete 4.70%
att-weights epoch 481, step 88, max_size:classes 63, max_size:data 1832, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.163 sec/step, elapsed 0:02:38, exp. remaining 0:52:38, complete 4.77%
att-weights epoch 481, step 89, max_size:classes 56, max_size:data 1655, mem_usage:GPU:0 1.0GB, num_seqs 1, 1.686 sec/step, elapsed 0:02:39, exp. remaining 0:52:46, complete 4.81%
att-weights epoch 481, step 90, max_size:classes 56, max_size:data 2470, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.865 sec/step, elapsed 0:02:40, exp. remaining 0:52:37, complete 4.85%
att-weights epoch 481, step 91, max_size:classes 59, max_size:data 1756, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.258 sec/step, elapsed 0:02:42, exp. remaining 0:52:12, complete 4.92%
att-weights epoch 481, step 92, max_size:classes 61, max_size:data 1845, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.176 sec/step, elapsed 0:02:43, exp. remaining 0:52:10, complete 4.96%
att-weights epoch 481, step 93, max_size:classes 53, max_size:data 1747, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.115 sec/step, elapsed 0:02:44, exp. remaining 0:51:42, complete 5.03%
att-weights epoch 481, step 94, max_size:classes 53, max_size:data 1723, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.082 sec/step, elapsed 0:02:45, exp. remaining 0:51:39, complete 5.07%
att-weights epoch 481, step 95, max_size:classes 58, max_size:data 1593, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.162 sec/step, elapsed 0:02:46, exp. remaining 0:51:37, complete 5.11%
att-weights epoch 481, step 96, max_size:classes 62, max_size:data 1939, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.628 sec/step, elapsed 0:02:48, exp. remaining 0:51:20, complete 5.18%
att-weights epoch 481, step 97, max_size:classes 60, max_size:data 1912, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.081 sec/step, elapsed 0:02:50, exp. remaining 0:51:12, complete 5.25%
att-weights epoch 481, step 98, max_size:classes 52, max_size:data 1629, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.815 sec/step, elapsed 0:02:51, exp. remaining 0:50:41, complete 5.33%
att-weights epoch 481, step 99, max_size:classes 55, max_size:data 2083, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.778 sec/step, elapsed 0:02:51, exp. remaining 0:50:11, complete 5.40%
att-weights epoch 481, step 100, max_size:classes 62, max_size:data 1587, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.167 sec/step, elapsed 0:02:53, exp. remaining 0:49:48, complete 5.48%
att-weights epoch 481, step 101, max_size:classes 61, max_size:data 2101, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.915 sec/step, elapsed 0:02:54, exp. remaining 0:49:21, complete 5.55%
att-weights epoch 481, step 102, max_size:classes 55, max_size:data 1729, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.311 sec/step, elapsed 0:02:55, exp. remaining 0:49:02, complete 5.62%
att-weights epoch 481, step 103, max_size:classes 58, max_size:data 2075, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.908 sec/step, elapsed 0:02:56, exp. remaining 0:48:37, complete 5.70%
att-weights epoch 481, step 104, max_size:classes 54, max_size:data 2023, mem_usage:GPU:0 1.0GB, num_seqs 1, 0.721 sec/step, elapsed 0:02:56, exp. remaining 0:48:09, complete 5.77%
att-weights epoch 481, step 105, max_size:classes 53, max_size:data 1786, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.249 sec/step, elapsed 0:02:58, exp. remaining 0:47:50, complete 5.85%
att-weights epoch 481, step 106, max_size:classes 63, max_size:data 1779, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.235 sec/step, elapsed 0:02:59, exp. remaining 0:47:32, complete 5.92%
att-weights epoch 481, step 107, max_size:classes 58, max_size:data 1767, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.639 sec/step, elapsed 0:03:01, exp. remaining 0:47:20, complete 5.99%
att-weights epoch 481, step 108, max_size:classes 54, max_size:data 1691, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.206 sec/step, elapsed 0:03:03, exp. remaining 0:47:17, complete 6.07%
att-weights epoch 481, step 109, max_size:classes 56, max_size:data 1715, mem_usage:GPU:0 1.0GB, num_seqs 2, 6.291 sec/step, elapsed 0:03:09, exp. remaining 0:48:17, complete 6.14%
att-weights epoch 481, step 110, max_size:classes 61, max_size:data 1673, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.352 sec/step, elapsed 0:03:10, exp. remaining 0:48:01, complete 6.22%
att-weights epoch 481, step 111, max_size:classes 54, max_size:data 1830, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.288 sec/step, elapsed 0:03:12, exp. remaining 0:47:44, complete 6.29%
att-weights epoch 481, step 112, max_size:classes 56, max_size:data 1814, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.234 sec/step, elapsed 0:03:13, exp. remaining 0:47:27, complete 6.36%
att-weights epoch 481, step 113, max_size:classes 55, max_size:data 1835, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.452 sec/step, elapsed 0:03:14, exp. remaining 0:47:13, complete 6.44%
att-weights epoch 481, step 114, max_size:classes 55, max_size:data 1641, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.712 sec/step, elapsed 0:03:17, exp. remaining 0:47:17, complete 6.51%
att-weights epoch 481, step 115, max_size:classes 49, max_size:data 1522, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.956 sec/step, elapsed 0:03:18, exp. remaining 0:46:57, complete 6.59%
att-weights epoch 481, step 116, max_size:classes 49, max_size:data 1447, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.994 sec/step, elapsed 0:03:19, exp. remaining 0:46:37, complete 6.66%
att-weights epoch 481, step 117, max_size:classes 56, max_size:data 1834, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.128 sec/step, elapsed 0:03:20, exp. remaining 0:46:20, complete 6.73%
att-weights epoch 481, step 118, max_size:classes 55, max_size:data 1733, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.053 sec/step, elapsed 0:03:21, exp. remaining 0:46:02, complete 6.81%
att-weights epoch 481, step 119, max_size:classes 58, max_size:data 1553, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.074 sec/step, elapsed 0:03:22, exp. remaining 0:45:45, complete 6.88%
att-weights epoch 481, step 120, max_size:classes 57, max_size:data 1574, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.192 sec/step, elapsed 0:03:24, exp. remaining 0:45:29, complete 6.96%
att-weights epoch 481, step 121, max_size:classes 53, max_size:data 1440, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.913 sec/step, elapsed 0:03:24, exp. remaining 0:45:10, complete 7.03%
att-weights epoch 481, step 122, max_size:classes 60, max_size:data 1541, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.027 sec/step, elapsed 0:03:25, exp. remaining 0:44:54, complete 7.10%
att-weights epoch 481, step 123, max_size:classes 53, max_size:data 1710, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.144 sec/step, elapsed 0:03:27, exp. remaining 0:44:38, complete 7.18%
att-weights epoch 481, step 124, max_size:classes 58, max_size:data 1448, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.005 sec/step, elapsed 0:03:28, exp. remaining 0:44:22, complete 7.25%
att-weights epoch 481, step 125, max_size:classes 50, max_size:data 1578, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.979 sec/step, elapsed 0:03:29, exp. remaining 0:44:05, complete 7.33%
att-weights epoch 481, step 126, max_size:classes 53, max_size:data 1692, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.088 sec/step, elapsed 0:03:30, exp. remaining 0:43:50, complete 7.40%
att-weights epoch 481, step 127, max_size:classes 52, max_size:data 1571, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.158 sec/step, elapsed 0:03:31, exp. remaining 0:43:37, complete 7.47%
att-weights epoch 481, step 128, max_size:classes 54, max_size:data 1587, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.482 sec/step, elapsed 0:03:33, exp. remaining 0:43:39, complete 7.55%
att-weights epoch 481, step 129, max_size:classes 51, max_size:data 1523, mem_usage:GPU:0 1.0GB, num_seqs 2, 6.179 sec/step, elapsed 0:03:40, exp. remaining 0:44:27, complete 7.62%
att-weights epoch 481, step 130, max_size:classes 50, max_size:data 1668, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.365 sec/step, elapsed 0:03:41, exp. remaining 0:44:15, complete 7.70%
att-weights epoch 481, step 131, max_size:classes 55, max_size:data 1492, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.007 sec/step, elapsed 0:03:42, exp. remaining 0:44:00, complete 7.77%
att-weights epoch 481, step 132, max_size:classes 47, max_size:data 1534, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.931 sec/step, elapsed 0:03:43, exp. remaining 0:43:44, complete 7.84%
att-weights epoch 481, step 133, max_size:classes 50, max_size:data 1675, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.841 sec/step, elapsed 0:03:45, exp. remaining 0:43:39, complete 7.92%
att-weights epoch 481, step 134, max_size:classes 53, max_size:data 1433, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.024 sec/step, elapsed 0:03:46, exp. remaining 0:43:24, complete 7.99%
att-weights epoch 481, step 135, max_size:classes 58, max_size:data 1680, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.149 sec/step, elapsed 0:03:47, exp. remaining 0:43:11, complete 8.07%
att-weights epoch 481, step 136, max_size:classes 46, max_size:data 1676, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.197 sec/step, elapsed 0:03:48, exp. remaining 0:42:59, complete 8.14%
att-weights epoch 481, step 137, max_size:classes 49, max_size:data 1587, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.323 sec/step, elapsed 0:03:49, exp. remaining 0:42:49, complete 8.21%
att-weights epoch 481, step 138, max_size:classes 56, max_size:data 1744, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.086 sec/step, elapsed 0:03:50, exp. remaining 0:42:36, complete 8.29%
att-weights epoch 481, step 139, max_size:classes 56, max_size:data 1516, mem_usage:GPU:0 1.0GB, num_seqs 2, 36.341 sec/step, elapsed 0:04:27, exp. remaining 0:48:49, complete 8.36%
att-weights epoch 481, step 140, max_size:classes 48, max_size:data 1495, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.148 sec/step, elapsed 0:04:29, exp. remaining 0:48:45, complete 8.44%
att-weights epoch 481, step 141, max_size:classes 48, max_size:data 1596, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.122 sec/step, elapsed 0:04:30, exp. remaining 0:48:29, complete 8.51%
att-weights epoch 481, step 142, max_size:classes 52, max_size:data 1537, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.930 sec/step, elapsed 0:04:31, exp. remaining 0:48:11, complete 8.58%
att-weights epoch 481, step 143, max_size:classes 46, max_size:data 1493, mem_usage:GPU:0 1.0GB, num_seqs 2, 5.433 sec/step, elapsed 0:04:36, exp. remaining 0:48:42, complete 8.66%
att-weights epoch 481, step 144, max_size:classes 46, max_size:data 1555, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.129 sec/step, elapsed 0:04:38, exp. remaining 0:48:26, complete 8.73%
att-weights epoch 481, step 145, max_size:classes 47, max_size:data 1416, mem_usage:GPU:0 1.0GB, num_seqs 2, 12.946 sec/step, elapsed 0:04:51, exp. remaining 0:50:14, complete 8.81%
att-weights epoch 481, step 146, max_size:classes 48, max_size:data 1499, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.184 sec/step, elapsed 0:04:52, exp. remaining 0:49:58, complete 8.88%
att-weights epoch 481, step 147, max_size:classes 51, max_size:data 1555, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.308 sec/step, elapsed 0:04:53, exp. remaining 0:49:44, complete 8.95%
att-weights epoch 481, step 148, max_size:classes 43, max_size:data 1255, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.952 sec/step, elapsed 0:04:54, exp. remaining 0:49:27, complete 9.03%
att-weights epoch 481, step 149, max_size:classes 47, max_size:data 1494, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.936 sec/step, elapsed 0:04:55, exp. remaining 0:49:10, complete 9.10%
att-weights epoch 481, step 150, max_size:classes 52, max_size:data 1416, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.875 sec/step, elapsed 0:04:56, exp. remaining 0:48:52, complete 9.17%
att-weights epoch 481, step 151, max_size:classes 46, max_size:data 1539, mem_usage:GPU:0 1.0GB, num_seqs 2, 6.804 sec/step, elapsed 0:05:03, exp. remaining 0:49:33, complete 9.25%
att-weights epoch 481, step 152, max_size:classes 54, max_size:data 1448, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.272 sec/step, elapsed 0:05:04, exp. remaining 0:49:20, complete 9.32%
att-weights epoch 481, step 153, max_size:classes 45, max_size:data 1415, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.039 sec/step, elapsed 0:05:05, exp. remaining 0:49:04, complete 9.40%
att-weights epoch 481, step 154, max_size:classes 50, max_size:data 1465, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.392 sec/step, elapsed 0:05:06, exp. remaining 0:48:52, complete 9.47%
att-weights epoch 481, step 155, max_size:classes 46, max_size:data 1766, mem_usage:GPU:0 1.0GB, num_seqs 2, 7.847 sec/step, elapsed 0:05:14, exp. remaining 0:49:41, complete 9.54%
att-weights epoch 481, step 156, max_size:classes 47, max_size:data 1809, mem_usage:GPU:0 1.0GB, num_seqs 2, 10.417 sec/step, elapsed 0:05:25, exp. remaining 0:50:41, complete 9.66%
att-weights epoch 481, step 157, max_size:classes 50, max_size:data 1435, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.217 sec/step, elapsed 0:05:26, exp. remaining 0:50:14, complete 9.77%
att-weights epoch 481, step 158, max_size:classes 41, max_size:data 1401, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.048 sec/step, elapsed 0:05:27, exp. remaining 0:49:59, complete 9.84%
att-weights epoch 481, step 159, max_size:classes 49, max_size:data 1553, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.221 sec/step, elapsed 0:05:28, exp. remaining 0:49:45, complete 9.91%
att-weights epoch 481, step 160, max_size:classes 49, max_size:data 1196, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.947 sec/step, elapsed 0:05:29, exp. remaining 0:49:29, complete 9.99%
att-weights epoch 481, step 161, max_size:classes 47, max_size:data 1632, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.587 sec/step, elapsed 0:05:31, exp. remaining 0:49:19, complete 10.06%
att-weights epoch 481, step 162, max_size:classes 49, max_size:data 1557, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.184 sec/step, elapsed 0:05:32, exp. remaining 0:49:05, complete 10.14%
att-weights epoch 481, step 163, max_size:classes 47, max_size:data 1603, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.420 sec/step, elapsed 0:05:33, exp. remaining 0:48:54, complete 10.21%
att-weights epoch 481, step 164, max_size:classes 53, max_size:data 1677, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.385 sec/step, elapsed 0:05:35, exp. remaining 0:48:43, complete 10.28%
att-weights epoch 481, step 165, max_size:classes 50, max_size:data 1280, mem_usage:GPU:0 1.0GB, num_seqs 3, 7.039 sec/step, elapsed 0:05:42, exp. remaining 0:49:08, complete 10.40%
att-weights epoch 481, step 166, max_size:classes 50, max_size:data 1330, mem_usage:GPU:0 1.0GB, num_seqs 3, 2.260 sec/step, elapsed 0:05:44, exp. remaining 0:49:05, complete 10.47%
att-weights epoch 481, step 167, max_size:classes 43, max_size:data 1449, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.284 sec/step, elapsed 0:05:45, exp. remaining 0:48:52, complete 10.54%
att-weights epoch 481, step 168, max_size:classes 49, max_size:data 1584, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.439 sec/step, elapsed 0:05:47, exp. remaining 0:48:30, complete 10.65%
att-weights epoch 481, step 169, max_size:classes 49, max_size:data 1657, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.608 sec/step, elapsed 0:05:49, exp. remaining 0:48:29, complete 10.73%
att-weights epoch 481, step 170, max_size:classes 47, max_size:data 1361, mem_usage:GPU:0 1.0GB, num_seqs 2, 4.443 sec/step, elapsed 0:05:54, exp. remaining 0:48:44, complete 10.80%
att-weights epoch 481, step 171, max_size:classes 45, max_size:data 1326, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.117 sec/step, elapsed 0:05:55, exp. remaining 0:48:20, complete 10.91%
att-weights epoch 481, step 172, max_size:classes 48, max_size:data 1405, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.186 sec/step, elapsed 0:05:56, exp. remaining 0:48:07, complete 10.99%
att-weights epoch 481, step 173, max_size:classes 47, max_size:data 1496, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.889 sec/step, elapsed 0:05:59, exp. remaining 0:47:58, complete 11.10%
att-weights epoch 481, step 174, max_size:classes 50, max_size:data 1327, mem_usage:GPU:0 1.0GB, num_seqs 3, 2.340 sec/step, elapsed 0:06:01, exp. remaining 0:47:45, complete 11.21%
att-weights epoch 481, step 175, max_size:classes 47, max_size:data 1317, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.335 sec/step, elapsed 0:06:03, exp. remaining 0:47:23, complete 11.32%
att-weights epoch 481, step 176, max_size:classes 53, max_size:data 1431, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.279 sec/step, elapsed 0:06:04, exp. remaining 0:47:02, complete 11.43%
att-weights epoch 481, step 177, max_size:classes 43, max_size:data 1239, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.189 sec/step, elapsed 0:06:05, exp. remaining 0:46:41, complete 11.54%
att-weights epoch 481, step 178, max_size:classes 42, max_size:data 1140, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.048 sec/step, elapsed 0:06:06, exp. remaining 0:46:28, complete 11.62%
att-weights epoch 481, step 179, max_size:classes 44, max_size:data 1350, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.000 sec/step, elapsed 0:06:07, exp. remaining 0:46:16, complete 11.69%
att-weights epoch 481, step 180, max_size:classes 48, max_size:data 1319, mem_usage:GPU:0 1.0GB, num_seqs 3, 13.881 sec/step, elapsed 0:06:21, exp. remaining 0:47:30, complete 11.80%
att-weights epoch 481, step 181, max_size:classes 47, max_size:data 1588, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.115 sec/step, elapsed 0:06:22, exp. remaining 0:47:08, complete 11.91%
att-weights epoch 481, step 182, max_size:classes 44, max_size:data 1320, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.232 sec/step, elapsed 0:06:23, exp. remaining 0:46:58, complete 11.99%
att-weights epoch 481, step 183, max_size:classes 47, max_size:data 1305, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.080 sec/step, elapsed 0:06:24, exp. remaining 0:46:46, complete 12.06%
att-weights epoch 481, step 184, max_size:classes 46, max_size:data 1199, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.349 sec/step, elapsed 0:06:26, exp. remaining 0:46:36, complete 12.13%
att-weights epoch 481, step 185, max_size:classes 46, max_size:data 1170, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.188 sec/step, elapsed 0:06:27, exp. remaining 0:46:25, complete 12.21%
att-weights epoch 481, step 186, max_size:classes 43, max_size:data 1198, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.364 sec/step, elapsed 0:06:28, exp. remaining 0:46:06, complete 12.32%
att-weights epoch 481, step 187, max_size:classes 40, max_size:data 1194, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.906 sec/step, elapsed 0:06:29, exp. remaining 0:45:45, complete 12.43%
att-weights epoch 481, step 188, max_size:classes 44, max_size:data 1512, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.055 sec/step, elapsed 0:06:30, exp. remaining 0:45:34, complete 12.50%
att-weights epoch 481, step 189, max_size:classes 46, max_size:data 1230, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.241 sec/step, elapsed 0:06:31, exp. remaining 0:45:15, complete 12.62%
att-weights epoch 481, step 190, max_size:classes 48, max_size:data 1226, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.173 sec/step, elapsed 0:06:33, exp. remaining 0:45:05, complete 12.69%
att-weights epoch 481, step 191, max_size:classes 47, max_size:data 1341, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.867 sec/step, elapsed 0:06:34, exp. remaining 0:44:44, complete 12.80%
att-weights epoch 481, step 192, max_size:classes 43, max_size:data 1687, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.028 sec/step, elapsed 0:06:35, exp. remaining 0:44:33, complete 12.87%
att-weights epoch 481, step 193, max_size:classes 48, max_size:data 1398, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.008 sec/step, elapsed 0:06:36, exp. remaining 0:44:22, complete 12.95%
att-weights epoch 481, step 194, max_size:classes 41, max_size:data 1399, mem_usage:GPU:0 1.0GB, num_seqs 2, 2.510 sec/step, elapsed 0:06:38, exp. remaining 0:44:22, complete 13.02%
att-weights epoch 481, step 195, max_size:classes 43, max_size:data 1246, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.191 sec/step, elapsed 0:06:39, exp. remaining 0:44:12, complete 13.10%
att-weights epoch 481, step 196, max_size:classes 47, max_size:data 1167, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.247 sec/step, elapsed 0:06:41, exp. remaining 0:43:55, complete 13.21%
att-weights epoch 481, step 197, max_size:classes 46, max_size:data 1519, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.978 sec/step, elapsed 0:06:41, exp. remaining 0:43:36, complete 13.32%
att-weights epoch 481, step 198, max_size:classes 46, max_size:data 1277, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.479 sec/step, elapsed 0:06:43, exp. remaining 0:43:29, complete 13.39%
att-weights epoch 481, step 199, max_size:classes 51, max_size:data 1352, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.047 sec/step, elapsed 0:06:44, exp. remaining 0:43:19, complete 13.47%
att-weights epoch 481, step 200, max_size:classes 47, max_size:data 1181, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.163 sec/step, elapsed 0:06:45, exp. remaining 0:43:10, complete 13.54%
att-weights epoch 481, step 201, max_size:classes 41, max_size:data 1175, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.958 sec/step, elapsed 0:06:46, exp. remaining 0:42:52, complete 13.65%
att-weights epoch 481, step 202, max_size:classes 39, max_size:data 1393, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.903 sec/step, elapsed 0:06:47, exp. remaining 0:42:41, complete 13.73%
att-weights epoch 481, step 203, max_size:classes 40, max_size:data 1140, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.883 sec/step, elapsed 0:06:48, exp. remaining 0:42:31, complete 13.80%
att-weights epoch 481, step 204, max_size:classes 37, max_size:data 1444, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.180 sec/step, elapsed 0:06:49, exp. remaining 0:42:14, complete 13.91%
att-weights epoch 481, step 205, max_size:classes 42, max_size:data 1295, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.125 sec/step, elapsed 0:06:50, exp. remaining 0:41:58, complete 14.02%
att-weights epoch 481, step 206, max_size:classes 45, max_size:data 1285, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.532 sec/step, elapsed 0:06:52, exp. remaining 0:41:52, complete 14.10%
att-weights epoch 481, step 207, max_size:classes 41, max_size:data 1469, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.263 sec/step, elapsed 0:06:53, exp. remaining 0:41:44, complete 14.17%
att-weights epoch 481, step 208, max_size:classes 46, max_size:data 1237, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.956 sec/step, elapsed 0:06:54, exp. remaining 0:41:28, complete 14.28%
att-weights epoch 481, step 209, max_size:classes 38, max_size:data 1334, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.940 sec/step, elapsed 0:06:55, exp. remaining 0:41:11, complete 14.39%
att-weights epoch 481, step 210, max_size:classes 43, max_size:data 1232, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.141 sec/step, elapsed 0:06:56, exp. remaining 0:41:03, complete 14.47%
att-weights epoch 481, step 211, max_size:classes 53, max_size:data 1579, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.121 sec/step, elapsed 0:06:57, exp. remaining 0:40:47, complete 14.58%
att-weights epoch 481, step 212, max_size:classes 50, max_size:data 1600, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.096 sec/step, elapsed 0:06:58, exp. remaining 0:40:32, complete 14.69%
att-weights epoch 481, step 213, max_size:classes 41, max_size:data 1198, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.162 sec/step, elapsed 0:06:59, exp. remaining 0:40:17, complete 14.80%
att-weights epoch 481, step 214, max_size:classes 40, max_size:data 1203, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.201 sec/step, elapsed 0:07:01, exp. remaining 0:40:10, complete 14.87%
att-weights epoch 481, step 215, max_size:classes 41, max_size:data 1374, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.840 sec/step, elapsed 0:07:01, exp. remaining 0:40:01, complete 14.95%
att-weights epoch 481, step 216, max_size:classes 40, max_size:data 1422, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.042 sec/step, elapsed 0:07:03, exp. remaining 0:39:46, complete 15.06%
att-weights epoch 481, step 217, max_size:classes 40, max_size:data 1188, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.171 sec/step, elapsed 0:07:04, exp. remaining 0:39:39, complete 15.13%
att-weights epoch 481, step 218, max_size:classes 42, max_size:data 1082, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.150 sec/step, elapsed 0:07:05, exp. remaining 0:39:25, complete 15.24%
att-weights epoch 481, step 219, max_size:classes 43, max_size:data 1361, mem_usage:GPU:0 1.0GB, num_seqs 2, 9.625 sec/step, elapsed 0:07:14, exp. remaining 0:39:58, complete 15.35%
att-weights epoch 481, step 220, max_size:classes 46, max_size:data 1312, mem_usage:GPU:0 1.0GB, num_seqs 3, 5.129 sec/step, elapsed 0:07:20, exp. remaining 0:40:12, complete 15.43%
att-weights epoch 481, step 221, max_size:classes 40, max_size:data 1319, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.225 sec/step, elapsed 0:07:21, exp. remaining 0:40:05, complete 15.50%
att-weights epoch 481, step 222, max_size:classes 39, max_size:data 1158, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.237 sec/step, elapsed 0:07:22, exp. remaining 0:39:52, complete 15.61%
att-weights epoch 481, step 223, max_size:classes 41, max_size:data 1195, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.841 sec/step, elapsed 0:07:23, exp. remaining 0:39:36, complete 15.72%
att-weights epoch 481, step 224, max_size:classes 39, max_size:data 1544, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.844 sec/step, elapsed 0:07:24, exp. remaining 0:39:21, complete 15.83%
att-weights epoch 481, step 225, max_size:classes 42, max_size:data 1010, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.178 sec/step, elapsed 0:07:25, exp. remaining 0:39:08, complete 15.95%
att-weights epoch 481, step 226, max_size:classes 38, max_size:data 1579, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.976 sec/step, elapsed 0:07:26, exp. remaining 0:38:53, complete 16.06%
att-weights epoch 481, step 227, max_size:classes 38, max_size:data 1325, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.194 sec/step, elapsed 0:07:27, exp. remaining 0:38:41, complete 16.17%
att-weights epoch 481, step 228, max_size:classes 41, max_size:data 1032, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.159 sec/step, elapsed 0:07:28, exp. remaining 0:38:28, complete 16.28%
att-weights epoch 481, step 229, max_size:classes 36, max_size:data 1369, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.879 sec/step, elapsed 0:07:29, exp. remaining 0:38:13, complete 16.39%
att-weights epoch 481, step 230, max_size:classes 39, max_size:data 1456, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.826 sec/step, elapsed 0:07:30, exp. remaining 0:37:59, complete 16.50%
att-weights epoch 481, step 231, max_size:classes 38, max_size:data 1249, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.151 sec/step, elapsed 0:07:31, exp. remaining 0:37:53, complete 16.57%
att-weights epoch 481, step 232, max_size:classes 41, max_size:data 1230, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.148 sec/step, elapsed 0:07:32, exp. remaining 0:37:40, complete 16.69%
att-weights epoch 481, step 233, max_size:classes 36, max_size:data 1061, mem_usage:GPU:0 1.0GB, num_seqs 3, 0.924 sec/step, elapsed 0:07:33, exp. remaining 0:37:27, complete 16.80%
att-weights epoch 481, step 234, max_size:classes 40, max_size:data 1223, mem_usage:GPU:0 1.0GB, num_seqs 3, 0.978 sec/step, elapsed 0:07:34, exp. remaining 0:37:14, complete 16.91%
att-weights epoch 481, step 235, max_size:classes 43, max_size:data 1068, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.074 sec/step, elapsed 0:07:35, exp. remaining 0:37:08, complete 16.98%
att-weights epoch 481, step 236, max_size:classes 39, max_size:data 1282, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.046 sec/step, elapsed 0:07:36, exp. remaining 0:37:01, complete 17.06%
att-weights epoch 481, step 237, max_size:classes 44, max_size:data 1278, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.108 sec/step, elapsed 0:07:37, exp. remaining 0:36:49, complete 17.17%
att-weights epoch 481, step 238, max_size:classes 39, max_size:data 1180, mem_usage:GPU:0 1.0GB, num_seqs 3, 0.993 sec/step, elapsed 0:07:38, exp. remaining 0:36:37, complete 17.28%
att-weights epoch 481, step 239, max_size:classes 36, max_size:data 1151, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.138 sec/step, elapsed 0:07:40, exp. remaining 0:36:25, complete 17.39%
att-weights epoch 481, step 240, max_size:classes 40, max_size:data 1476, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.889 sec/step, elapsed 0:07:40, exp. remaining 0:36:13, complete 17.50%
att-weights epoch 481, step 241, max_size:classes 40, max_size:data 1017, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.137 sec/step, elapsed 0:07:42, exp. remaining 0:36:01, complete 17.61%
att-weights epoch 481, step 242, max_size:classes 38, max_size:data 1216, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.130 sec/step, elapsed 0:07:43, exp. remaining 0:35:50, complete 17.72%
att-weights epoch 481, step 243, max_size:classes 37, max_size:data 1221, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.078 sec/step, elapsed 0:07:44, exp. remaining 0:35:39, complete 17.83%
att-weights epoch 481, step 244, max_size:classes 37, max_size:data 1074, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.913 sec/step, elapsed 0:07:45, exp. remaining 0:35:32, complete 17.91%
att-weights epoch 481, step 245, max_size:classes 38, max_size:data 1409, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.013 sec/step, elapsed 0:07:46, exp. remaining 0:35:21, complete 18.02%
att-weights epoch 481, step 246, max_size:classes 38, max_size:data 1042, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.107 sec/step, elapsed 0:07:47, exp. remaining 0:35:10, complete 18.13%
att-weights epoch 481, step 247, max_size:classes 39, max_size:data 1126, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.179 sec/step, elapsed 0:07:48, exp. remaining 0:35:00, complete 18.24%
att-weights epoch 481, step 248, max_size:classes 39, max_size:data 1221, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.130 sec/step, elapsed 0:07:49, exp. remaining 0:34:49, complete 18.35%
att-weights epoch 481, step 249, max_size:classes 40, max_size:data 1042, mem_usage:GPU:0 1.0GB, num_seqs 3, 0.947 sec/step, elapsed 0:07:50, exp. remaining 0:34:38, complete 18.46%
att-weights epoch 481, step 250, max_size:classes 37, max_size:data 1124, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.154 sec/step, elapsed 0:07:51, exp. remaining 0:34:28, complete 18.57%
att-weights epoch 481, step 251, max_size:classes 38, max_size:data 1127, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.161 sec/step, elapsed 0:07:52, exp. remaining 0:34:18, complete 18.68%
att-weights epoch 481, step 252, max_size:classes 42, max_size:data 1101, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.139 sec/step, elapsed 0:07:54, exp. remaining 0:34:08, complete 18.79%
att-weights epoch 481, step 253, max_size:classes 41, max_size:data 1442, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.932 sec/step, elapsed 0:07:54, exp. remaining 0:33:57, complete 18.90%
att-weights epoch 481, step 254, max_size:classes 35, max_size:data 1062, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.261 sec/step, elapsed 0:07:56, exp. remaining 0:33:48, complete 19.02%
att-weights epoch 481, step 255, max_size:classes 33, max_size:data 1097, mem_usage:GPU:0 1.0GB, num_seqs 3, 0.972 sec/step, elapsed 0:07:57, exp. remaining 0:33:37, complete 19.13%
att-weights epoch 481, step 256, max_size:classes 37, max_size:data 1030, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.029 sec/step, elapsed 0:07:58, exp. remaining 0:33:27, complete 19.24%
att-weights epoch 481, step 257, max_size:classes 36, max_size:data 1247, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.207 sec/step, elapsed 0:07:59, exp. remaining 0:33:18, complete 19.35%
att-weights epoch 481, step 258, max_size:classes 39, max_size:data 1114, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.126 sec/step, elapsed 0:08:00, exp. remaining 0:33:08, complete 19.46%
att-weights epoch 481, step 259, max_size:classes 32, max_size:data 1299, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.094 sec/step, elapsed 0:08:01, exp. remaining 0:32:59, complete 19.57%
att-weights epoch 481, step 260, max_size:classes 38, max_size:data 1090, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.130 sec/step, elapsed 0:08:02, exp. remaining 0:32:50, complete 19.68%
att-weights epoch 481, step 261, max_size:classes 39, max_size:data 1119, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.087 sec/step, elapsed 0:08:03, exp. remaining 0:32:40, complete 19.79%
att-weights epoch 481, step 262, max_size:classes 39, max_size:data 1205, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.076 sec/step, elapsed 0:08:04, exp. remaining 0:32:31, complete 19.90%
att-weights epoch 481, step 263, max_size:classes 39, max_size:data 1021, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.055 sec/step, elapsed 0:08:06, exp. remaining 0:32:26, complete 19.98%
att-weights epoch 481, step 264, max_size:classes 39, max_size:data 1039, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.056 sec/step, elapsed 0:08:07, exp. remaining 0:32:21, complete 20.05%
att-weights epoch 481, step 265, max_size:classes 37, max_size:data 1031, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.072 sec/step, elapsed 0:08:08, exp. remaining 0:32:12, complete 20.16%
att-weights epoch 481, step 266, max_size:classes 36, max_size:data 1107, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.423 sec/step, elapsed 0:08:09, exp. remaining 0:32:05, complete 20.27%
att-weights epoch 481, step 267, max_size:classes 35, max_size:data 1025, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.021 sec/step, elapsed 0:08:10, exp. remaining 0:31:56, complete 20.38%
att-weights epoch 481, step 268, max_size:classes 36, max_size:data 1143, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.152 sec/step, elapsed 0:08:11, exp. remaining 0:31:43, complete 20.53%
att-weights epoch 481, step 269, max_size:classes 40, max_size:data 1129, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.174 sec/step, elapsed 0:08:12, exp. remaining 0:31:34, complete 20.64%
att-weights epoch 481, step 270, max_size:classes 37, max_size:data 995, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.025 sec/step, elapsed 0:08:13, exp. remaining 0:31:25, complete 20.75%
att-weights epoch 481, step 271, max_size:classes 33, max_size:data 1148, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.095 sec/step, elapsed 0:08:15, exp. remaining 0:31:17, complete 20.87%
att-weights epoch 481, step 272, max_size:classes 38, max_size:data 985, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.872 sec/step, elapsed 0:08:15, exp. remaining 0:31:12, complete 20.94%
att-weights epoch 481, step 273, max_size:classes 37, max_size:data 1382, mem_usage:GPU:0 1.0GB, num_seqs 2, 0.928 sec/step, elapsed 0:08:16, exp. remaining 0:31:03, complete 21.05%
att-weights epoch 481, step 274, max_size:classes 40, max_size:data 1047, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.076 sec/step, elapsed 0:08:17, exp. remaining 0:30:54, complete 21.16%
att-weights epoch 481, step 275, max_size:classes 37, max_size:data 1082, mem_usage:GPU:0 1.0GB, num_seqs 3, 3.150 sec/step, elapsed 0:08:21, exp. remaining 0:30:54, complete 21.27%
att-weights epoch 481, step 276, max_size:classes 38, max_size:data 1196, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.205 sec/step, elapsed 0:08:22, exp. remaining 0:30:46, complete 21.38%
att-weights epoch 481, step 277, max_size:classes 36, max_size:data 977, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.450 sec/step, elapsed 0:08:23, exp. remaining 0:30:39, complete 21.49%
att-weights epoch 481, step 278, max_size:classes 36, max_size:data 1026, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.275 sec/step, elapsed 0:08:24, exp. remaining 0:30:32, complete 21.61%
att-weights epoch 481, step 279, max_size:classes 36, max_size:data 1071, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.575 sec/step, elapsed 0:08:26, exp. remaining 0:30:26, complete 21.72%
att-weights epoch 481, step 280, max_size:classes 33, max_size:data 1195, mem_usage:GPU:0 1.0GB, num_seqs 3, 14.127 sec/step, elapsed 0:08:40, exp. remaining 0:31:04, complete 21.83%
att-weights epoch 481, step 281, max_size:classes 36, max_size:data 1415, mem_usage:GPU:0 1.0GB, num_seqs 2, 1.237 sec/step, elapsed 0:08:41, exp. remaining 0:30:57, complete 21.94%
att-weights epoch 481, step 282, max_size:classes 34, max_size:data 1105, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.098 sec/step, elapsed 0:08:43, exp. remaining 0:30:49, complete 22.05%
att-weights epoch 481, step 283, max_size:classes 41, max_size:data 1216, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.175 sec/step, elapsed 0:08:44, exp. remaining 0:30:37, complete 22.20%
att-weights epoch 481, step 284, max_size:classes 35, max_size:data 1162, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.154 sec/step, elapsed 0:08:45, exp. remaining 0:30:29, complete 22.31%
att-weights epoch 481, step 285, max_size:classes 35, max_size:data 1086, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.101 sec/step, elapsed 0:08:46, exp. remaining 0:30:21, complete 22.42%
att-weights epoch 481, step 286, max_size:classes 39, max_size:data 1281, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.184 sec/step, elapsed 0:08:47, exp. remaining 0:30:14, complete 22.53%
att-weights epoch 481, step 287, max_size:classes 35, max_size:data 1253, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.220 sec/step, elapsed 0:08:48, exp. remaining 0:30:03, complete 22.68%
att-weights epoch 481, step 288, max_size:classes 32, max_size:data 1089, mem_usage:GPU:0 1.0GB, num_seqs 3, 11.174 sec/step, elapsed 0:09:00, exp. remaining 0:30:29, complete 22.79%
att-weights epoch 481, step 289, max_size:classes 34, max_size:data 1174, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.152 sec/step, elapsed 0:09:01, exp. remaining 0:30:22, complete 22.90%
att-weights epoch 481, step 290, max_size:classes 36, max_size:data 1166, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.120 sec/step, elapsed 0:09:02, exp. remaining 0:30:14, complete 23.01%
att-weights epoch 481, step 291, max_size:classes 36, max_size:data 1074, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.002 sec/step, elapsed 0:09:03, exp. remaining 0:30:06, complete 23.12%
att-weights epoch 481, step 292, max_size:classes 37, max_size:data 936, mem_usage:GPU:0 1.0GB, num_seqs 4, 5.588 sec/step, elapsed 0:09:08, exp. remaining 0:30:13, complete 23.23%
att-weights epoch 481, step 293, max_size:classes 35, max_size:data 1147, mem_usage:GPU:0 1.0GB, num_seqs 3, 3.368 sec/step, elapsed 0:09:12, exp. remaining 0:30:13, complete 23.34%
att-weights epoch 481, step 294, max_size:classes 36, max_size:data 1178, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.433 sec/step, elapsed 0:09:13, exp. remaining 0:30:06, complete 23.46%
att-weights epoch 481, step 295, max_size:classes 33, max_size:data 1079, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.308 sec/step, elapsed 0:09:15, exp. remaining 0:30:00, complete 23.57%
att-weights epoch 481, step 296, max_size:classes 36, max_size:data 1000, mem_usage:GPU:0 1.0GB, num_seqs 4, 6.869 sec/step, elapsed 0:09:21, exp. remaining 0:30:07, complete 23.71%
att-weights epoch 481, step 297, max_size:classes 36, max_size:data 1028, mem_usage:GPU:0 1.0GB, num_seqs 3, 2.069 sec/step, elapsed 0:09:23, exp. remaining 0:29:59, complete 23.86%
att-weights epoch 481, step 298, max_size:classes 33, max_size:data 1009, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.183 sec/step, elapsed 0:09:25, exp. remaining 0:29:52, complete 23.97%
att-weights epoch 481, step 299, max_size:classes 36, max_size:data 1113, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.273 sec/step, elapsed 0:09:26, exp. remaining 0:29:45, complete 24.08%
att-weights epoch 481, step 300, max_size:classes 37, max_size:data 1033, mem_usage:GPU:0 1.0GB, num_seqs 3, 6.521 sec/step, elapsed 0:09:32, exp. remaining 0:29:55, complete 24.20%
att-weights epoch 481, step 301, max_size:classes 35, max_size:data 1207, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.356 sec/step, elapsed 0:09:34, exp. remaining 0:29:44, complete 24.34%
att-weights epoch 481, step 302, max_size:classes 35, max_size:data 1272, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.387 sec/step, elapsed 0:09:35, exp. remaining 0:29:38, complete 24.45%
att-weights epoch 481, step 303, max_size:classes 34, max_size:data 1048, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.037 sec/step, elapsed 0:09:36, exp. remaining 0:29:30, complete 24.57%
att-weights epoch 481, step 304, max_size:classes 35, max_size:data 1012, mem_usage:GPU:0 1.0GB, num_seqs 3, 0.929 sec/step, elapsed 0:09:37, exp. remaining 0:29:19, complete 24.71%
att-weights epoch 481, step 305, max_size:classes 33, max_size:data 983, mem_usage:GPU:0 1.0GB, num_seqs 4, 5.809 sec/step, elapsed 0:09:43, exp. remaining 0:29:26, complete 24.82%
att-weights epoch 481, step 306, max_size:classes 38, max_size:data 959, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.359 sec/step, elapsed 0:09:44, exp. remaining 0:29:20, complete 24.94%
att-weights epoch 481, step 307, max_size:classes 36, max_size:data 1060, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.005 sec/step, elapsed 0:09:45, exp. remaining 0:29:13, complete 25.05%
att-weights epoch 481, step 308, max_size:classes 34, max_size:data 1031, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.016 sec/step, elapsed 0:09:46, exp. remaining 0:29:05, complete 25.16%
att-weights epoch 481, step 309, max_size:classes 31, max_size:data 1115, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.047 sec/step, elapsed 0:09:47, exp. remaining 0:28:58, complete 25.27%
att-weights epoch 481, step 310, max_size:classes 32, max_size:data 951, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.472 sec/step, elapsed 0:09:49, exp. remaining 0:28:52, complete 25.38%
att-weights epoch 481, step 311, max_size:classes 35, max_size:data 1201, mem_usage:GPU:0 1.0GB, num_seqs 3, 4.792 sec/step, elapsed 0:09:54, exp. remaining 0:28:53, complete 25.53%
att-weights epoch 481, step 312, max_size:classes 30, max_size:data 1077, mem_usage:GPU:0 1.0GB, num_seqs 3, 0.865 sec/step, elapsed 0:09:55, exp. remaining 0:28:42, complete 25.68%
att-weights epoch 481, step 313, max_size:classes 34, max_size:data 862, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.176 sec/step, elapsed 0:09:56, exp. remaining 0:28:32, complete 25.82%
att-weights epoch 481, step 314, max_size:classes 32, max_size:data 873, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.195 sec/step, elapsed 0:09:57, exp. remaining 0:28:26, complete 25.93%
att-weights epoch 481, step 315, max_size:classes 32, max_size:data 1028, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.050 sec/step, elapsed 0:09:58, exp. remaining 0:28:19, complete 26.05%
att-weights epoch 481, step 316, max_size:classes 30, max_size:data 898, mem_usage:GPU:0 1.0GB, num_seqs 3, 4.306 sec/step, elapsed 0:10:02, exp. remaining 0:28:21, complete 26.16%
att-weights epoch 481, step 317, max_size:classes 33, max_size:data 1116, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.275 sec/step, elapsed 0:10:04, exp. remaining 0:28:15, complete 26.27%
att-weights epoch 481, step 318, max_size:classes 35, max_size:data 1019, mem_usage:GPU:0 1.0GB, num_seqs 3, 0.958 sec/step, elapsed 0:10:04, exp. remaining 0:28:08, complete 26.38%
att-weights epoch 481, step 319, max_size:classes 29, max_size:data 1125, mem_usage:GPU:0 1.0GB, num_seqs 3, 0.977 sec/step, elapsed 0:10:05, exp. remaining 0:27:58, complete 26.53%
att-weights epoch 481, step 320, max_size:classes 35, max_size:data 938, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.281 sec/step, elapsed 0:10:07, exp. remaining 0:27:52, complete 26.64%
att-weights epoch 481, step 321, max_size:classes 37, max_size:data 910, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.267 sec/step, elapsed 0:10:08, exp. remaining 0:27:46, complete 26.75%
att-weights epoch 481, step 322, max_size:classes 34, max_size:data 883, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.292 sec/step, elapsed 0:10:09, exp. remaining 0:27:37, complete 26.90%
att-weights epoch 481, step 323, max_size:classes 38, max_size:data 1218, mem_usage:GPU:0 1.0GB, num_seqs 3, 0.939 sec/step, elapsed 0:10:10, exp. remaining 0:27:27, complete 27.04%
att-weights epoch 481, step 324, max_size:classes 32, max_size:data 1008, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.498 sec/step, elapsed 0:10:12, exp. remaining 0:27:19, complete 27.19%
att-weights epoch 481, step 325, max_size:classes 31, max_size:data 1022, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.033 sec/step, elapsed 0:10:13, exp. remaining 0:27:09, complete 27.34%
att-weights epoch 481, step 326, max_size:classes 35, max_size:data 1019, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.162 sec/step, elapsed 0:10:14, exp. remaining 0:27:00, complete 27.49%
att-weights epoch 481, step 327, max_size:classes 31, max_size:data 1123, mem_usage:GPU:0 1.0GB, num_seqs 3, 0.962 sec/step, elapsed 0:10:15, exp. remaining 0:26:51, complete 27.64%
att-weights epoch 481, step 328, max_size:classes 33, max_size:data 919, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.114 sec/step, elapsed 0:10:16, exp. remaining 0:26:42, complete 27.78%
att-weights epoch 481, step 329, max_size:classes 33, max_size:data 1066, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.005 sec/step, elapsed 0:10:17, exp. remaining 0:26:36, complete 27.89%
att-weights epoch 481, step 330, max_size:classes 35, max_size:data 1098, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.061 sec/step, elapsed 0:10:18, exp. remaining 0:26:30, complete 28.01%
att-weights epoch 481, step 331, max_size:classes 30, max_size:data 975, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.382 sec/step, elapsed 0:10:19, exp. remaining 0:26:25, complete 28.12%
att-weights epoch 481, step 332, max_size:classes 32, max_size:data 910, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.332 sec/step, elapsed 0:10:21, exp. remaining 0:26:16, complete 28.26%
att-weights epoch 481, step 333, max_size:classes 36, max_size:data 918, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.612 sec/step, elapsed 0:10:22, exp. remaining 0:26:09, complete 28.41%
att-weights epoch 481, step 334, max_size:classes 33, max_size:data 993, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.251 sec/step, elapsed 0:10:24, exp. remaining 0:26:01, complete 28.56%
att-weights epoch 481, step 335, max_size:classes 34, max_size:data 991, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.457 sec/step, elapsed 0:10:25, exp. remaining 0:25:53, complete 28.71%
att-weights epoch 481, step 336, max_size:classes 31, max_size:data 954, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.230 sec/step, elapsed 0:10:26, exp. remaining 0:25:48, complete 28.82%
att-weights epoch 481, step 337, max_size:classes 30, max_size:data 935, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.402 sec/step, elapsed 0:10:28, exp. remaining 0:25:40, complete 28.97%
att-weights epoch 481, step 338, max_size:classes 30, max_size:data 1120, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.103 sec/step, elapsed 0:10:29, exp. remaining 0:25:32, complete 29.12%
att-weights epoch 481, step 339, max_size:classes 31, max_size:data 1152, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.105 sec/step, elapsed 0:10:30, exp. remaining 0:25:23, complete 29.26%
att-weights epoch 481, step 340, max_size:classes 31, max_size:data 1034, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.013 sec/step, elapsed 0:10:31, exp. remaining 0:25:18, complete 29.37%
att-weights epoch 481, step 341, max_size:classes 31, max_size:data 977, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.161 sec/step, elapsed 0:10:32, exp. remaining 0:25:10, complete 29.52%
att-weights epoch 481, step 342, max_size:classes 36, max_size:data 963, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.382 sec/step, elapsed 0:10:34, exp. remaining 0:25:02, complete 29.67%
att-weights epoch 481, step 343, max_size:classes 30, max_size:data 924, mem_usage:GPU:0 1.0GB, num_seqs 4, 2.247 sec/step, elapsed 0:10:36, exp. remaining 0:25:00, complete 29.78%
att-weights epoch 481, step 344, max_size:classes 29, max_size:data 953, mem_usage:GPU:0 1.0GB, num_seqs 4, 5.029 sec/step, elapsed 0:10:41, exp. remaining 0:25:01, complete 29.93%
att-weights epoch 481, step 345, max_size:classes 33, max_size:data 1010, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.346 sec/step, elapsed 0:10:42, exp. remaining 0:24:56, complete 30.04%
att-weights epoch 481, step 346, max_size:classes 33, max_size:data 852, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.445 sec/step, elapsed 0:10:44, exp. remaining 0:24:52, complete 30.15%
att-weights epoch 481, step 347, max_size:classes 31, max_size:data 856, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.456 sec/step, elapsed 0:10:45, exp. remaining 0:24:47, complete 30.26%
att-weights epoch 481, step 348, max_size:classes 30, max_size:data 883, mem_usage:GPU:0 1.0GB, num_seqs 4, 4.336 sec/step, elapsed 0:10:49, exp. remaining 0:24:47, complete 30.41%
att-weights epoch 481, step 349, max_size:classes 31, max_size:data 1062, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.170 sec/step, elapsed 0:10:51, exp. remaining 0:24:39, complete 30.56%
att-weights epoch 481, step 350, max_size:classes 33, max_size:data 892, mem_usage:GPU:0 1.0GB, num_seqs 4, 4.856 sec/step, elapsed 0:10:55, exp. remaining 0:24:42, complete 30.67%
att-weights epoch 481, step 351, max_size:classes 27, max_size:data 912, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.491 sec/step, elapsed 0:10:57, exp. remaining 0:24:35, complete 30.82%
att-weights epoch 481, step 352, max_size:classes 28, max_size:data 1007, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.168 sec/step, elapsed 0:10:58, exp. remaining 0:24:28, complete 30.97%
att-weights epoch 481, step 353, max_size:classes 32, max_size:data 954, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.643 sec/step, elapsed 0:11:00, exp. remaining 0:24:21, complete 31.11%
att-weights epoch 481, step 354, max_size:classes 28, max_size:data 820, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.005 sec/step, elapsed 0:11:01, exp. remaining 0:24:16, complete 31.22%
att-weights epoch 481, step 355, max_size:classes 30, max_size:data 1149, mem_usage:GPU:0 1.0GB, num_seqs 3, 3.242 sec/step, elapsed 0:11:04, exp. remaining 0:24:13, complete 31.37%
att-weights epoch 481, step 356, max_size:classes 31, max_size:data 1015, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.087 sec/step, elapsed 0:11:05, exp. remaining 0:24:05, complete 31.52%
att-weights epoch 481, step 357, max_size:classes 31, max_size:data 969, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.388 sec/step, elapsed 0:11:06, exp. remaining 0:23:59, complete 31.67%
att-weights epoch 481, step 358, max_size:classes 29, max_size:data 869, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.324 sec/step, elapsed 0:11:08, exp. remaining 0:23:52, complete 31.82%
att-weights epoch 481, step 359, max_size:classes 29, max_size:data 1056, mem_usage:GPU:0 1.0GB, num_seqs 3, 0.997 sec/step, elapsed 0:11:09, exp. remaining 0:23:44, complete 31.96%
att-weights epoch 481, step 360, max_size:classes 31, max_size:data 883, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.029 sec/step, elapsed 0:11:10, exp. remaining 0:23:37, complete 32.11%
att-weights epoch 481, step 361, max_size:classes 36, max_size:data 982, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.315 sec/step, elapsed 0:11:11, exp. remaining 0:23:32, complete 32.22%
att-weights epoch 481, step 362, max_size:classes 30, max_size:data 890, mem_usage:GPU:0 1.0GB, num_seqs 4, 2.497 sec/step, elapsed 0:11:14, exp. remaining 0:23:30, complete 32.33%
att-weights epoch 481, step 363, max_size:classes 29, max_size:data 1029, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.078 sec/step, elapsed 0:11:15, exp. remaining 0:23:23, complete 32.48%
att-weights epoch 481, step 364, max_size:classes 31, max_size:data 734, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.190 sec/step, elapsed 0:11:16, exp. remaining 0:23:16, complete 32.63%
att-weights epoch 481, step 365, max_size:classes 28, max_size:data 922, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.331 sec/step, elapsed 0:11:17, exp. remaining 0:23:09, complete 32.78%
att-weights epoch 481, step 366, max_size:classes 30, max_size:data 926, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.375 sec/step, elapsed 0:11:19, exp. remaining 0:23:03, complete 32.93%
att-weights epoch 481, step 367, max_size:classes 32, max_size:data 880, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.146 sec/step, elapsed 0:11:20, exp. remaining 0:22:56, complete 33.07%
att-weights epoch 481, step 368, max_size:classes 30, max_size:data 782, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.187 sec/step, elapsed 0:11:21, exp. remaining 0:22:49, complete 33.22%
att-weights epoch 481, step 369, max_size:classes 30, max_size:data 936, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.273 sec/step, elapsed 0:11:22, exp. remaining 0:22:43, complete 33.37%
att-weights epoch 481, step 370, max_size:classes 34, max_size:data 919, mem_usage:GPU:0 1.0GB, num_seqs 3, 0.996 sec/step, elapsed 0:11:23, exp. remaining 0:22:36, complete 33.52%
att-weights epoch 481, step 371, max_size:classes 28, max_size:data 1001, mem_usage:GPU:0 1.0GB, num_seqs 3, 3.014 sec/step, elapsed 0:11:26, exp. remaining 0:22:33, complete 33.67%
att-weights epoch 481, step 372, max_size:classes 28, max_size:data 833, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.129 sec/step, elapsed 0:11:27, exp. remaining 0:22:26, complete 33.81%
att-weights epoch 481, step 373, max_size:classes 30, max_size:data 947, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.384 sec/step, elapsed 0:11:29, exp. remaining 0:22:20, complete 33.96%
att-weights epoch 481, step 374, max_size:classes 31, max_size:data 930, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.180 sec/step, elapsed 0:11:30, exp. remaining 0:22:15, complete 34.07%
att-weights epoch 481, step 375, max_size:classes 27, max_size:data 954, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.333 sec/step, elapsed 0:11:31, exp. remaining 0:22:09, complete 34.22%
att-weights epoch 481, step 376, max_size:classes 29, max_size:data 886, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.112 sec/step, elapsed 0:11:32, exp. remaining 0:22:03, complete 34.37%
att-weights epoch 481, step 377, max_size:classes 27, max_size:data 986, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.435 sec/step, elapsed 0:11:34, exp. remaining 0:21:57, complete 34.52%
att-weights epoch 481, step 378, max_size:classes 32, max_size:data 808, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.311 sec/step, elapsed 0:11:35, exp. remaining 0:21:51, complete 34.67%
att-weights epoch 481, step 379, max_size:classes 30, max_size:data 845, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.146 sec/step, elapsed 0:11:36, exp. remaining 0:21:44, complete 34.81%
att-weights epoch 481, step 380, max_size:classes 28, max_size:data 976, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.267 sec/step, elapsed 0:11:38, exp. remaining 0:21:40, complete 34.92%
att-weights epoch 481, step 381, max_size:classes 32, max_size:data 869, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.334 sec/step, elapsed 0:11:39, exp. remaining 0:21:34, complete 35.07%
att-weights epoch 481, step 382, max_size:classes 31, max_size:data 929, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.161 sec/step, elapsed 0:11:40, exp. remaining 0:21:28, complete 35.22%
att-weights epoch 481, step 383, max_size:classes 29, max_size:data 1024, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.050 sec/step, elapsed 0:11:41, exp. remaining 0:21:19, complete 35.41%
att-weights epoch 481, step 384, max_size:classes 27, max_size:data 970, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.330 sec/step, elapsed 0:11:42, exp. remaining 0:21:14, complete 35.55%
att-weights epoch 481, step 385, max_size:classes 29, max_size:data 985, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.218 sec/step, elapsed 0:11:44, exp. remaining 0:21:08, complete 35.70%
att-weights epoch 481, step 386, max_size:classes 29, max_size:data 859, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.110 sec/step, elapsed 0:11:45, exp. remaining 0:21:01, complete 35.85%
att-weights epoch 481, step 387, max_size:classes 30, max_size:data 864, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.149 sec/step, elapsed 0:11:46, exp. remaining 0:20:57, complete 35.96%
att-weights epoch 481, step 388, max_size:classes 28, max_size:data 884, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.056 sec/step, elapsed 0:11:47, exp. remaining 0:20:53, complete 36.07%
att-weights epoch 481, step 389, max_size:classes 30, max_size:data 1108, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.029 sec/step, elapsed 0:11:48, exp. remaining 0:20:49, complete 36.18%
att-weights epoch 481, step 390, max_size:classes 28, max_size:data 827, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.415 sec/step, elapsed 0:11:49, exp. remaining 0:20:44, complete 36.33%
att-weights epoch 481, step 391, max_size:classes 28, max_size:data 884, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.234 sec/step, elapsed 0:11:51, exp. remaining 0:20:38, complete 36.48%
att-weights epoch 481, step 392, max_size:classes 29, max_size:data 767, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.460 sec/step, elapsed 0:11:52, exp. remaining 0:20:31, complete 36.66%
att-weights epoch 481, step 393, max_size:classes 29, max_size:data 820, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.112 sec/step, elapsed 0:11:53, exp. remaining 0:20:25, complete 36.81%
att-weights epoch 481, step 394, max_size:classes 30, max_size:data 888, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.312 sec/step, elapsed 0:11:54, exp. remaining 0:20:19, complete 36.96%
att-weights epoch 481, step 395, max_size:classes 30, max_size:data 896, mem_usage:GPU:0 1.0GB, num_seqs 4, 11.434 sec/step, elapsed 0:12:06, exp. remaining 0:20:31, complete 37.11%
att-weights epoch 481, step 396, max_size:classes 28, max_size:data 761, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.180 sec/step, elapsed 0:12:07, exp. remaining 0:20:23, complete 37.29%
att-weights epoch 481, step 397, max_size:classes 30, max_size:data 1015, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.325 sec/step, elapsed 0:12:08, exp. remaining 0:20:18, complete 37.44%
att-weights epoch 481, step 398, max_size:classes 29, max_size:data 1067, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.075 sec/step, elapsed 0:12:10, exp. remaining 0:20:12, complete 37.59%
att-weights epoch 481, step 399, max_size:classes 30, max_size:data 845, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.366 sec/step, elapsed 0:12:11, exp. remaining 0:20:06, complete 37.74%
att-weights epoch 481, step 400, max_size:classes 29, max_size:data 836, mem_usage:GPU:0 1.0GB, num_seqs 4, 3.884 sec/step, elapsed 0:12:15, exp. remaining 0:20:05, complete 37.88%
att-weights epoch 481, step 401, max_size:classes 28, max_size:data 786, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.531 sec/step, elapsed 0:12:16, exp. remaining 0:19:58, complete 38.07%
att-weights epoch 481, step 402, max_size:classes 27, max_size:data 936, mem_usage:GPU:0 1.0GB, num_seqs 4, 4.099 sec/step, elapsed 0:12:20, exp. remaining 0:19:55, complete 38.25%
att-weights epoch 481, step 403, max_size:classes 26, max_size:data 863, mem_usage:GPU:0 1.0GB, num_seqs 4, 4.294 sec/step, elapsed 0:12:25, exp. remaining 0:19:55, complete 38.40%
att-weights epoch 481, step 404, max_size:classes 26, max_size:data 949, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.160 sec/step, elapsed 0:12:26, exp. remaining 0:19:49, complete 38.55%
att-weights epoch 481, step 405, max_size:classes 27, max_size:data 790, mem_usage:GPU:0 1.0GB, num_seqs 5, 4.472 sec/step, elapsed 0:12:30, exp. remaining 0:19:49, complete 38.70%
att-weights epoch 481, step 406, max_size:classes 27, max_size:data 874, mem_usage:GPU:0 1.0GB, num_seqs 4, 3.482 sec/step, elapsed 0:12:34, exp. remaining 0:19:47, complete 38.85%
att-weights epoch 481, step 407, max_size:classes 27, max_size:data 845, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.327 sec/step, elapsed 0:12:35, exp. remaining 0:19:42, complete 38.99%
att-weights epoch 481, step 408, max_size:classes 28, max_size:data 912, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.229 sec/step, elapsed 0:12:36, exp. remaining 0:19:36, complete 39.14%
att-weights epoch 481, step 409, max_size:classes 27, max_size:data 826, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.172 sec/step, elapsed 0:12:38, exp. remaining 0:19:29, complete 39.33%
att-weights epoch 481, step 410, max_size:classes 27, max_size:data 787, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.396 sec/step, elapsed 0:12:39, exp. remaining 0:19:24, complete 39.47%
att-weights epoch 481, step 411, max_size:classes 27, max_size:data 780, mem_usage:GPU:0 1.0GB, num_seqs 5, 3.742 sec/step, elapsed 0:12:43, exp. remaining 0:19:22, complete 39.62%
att-weights epoch 481, step 412, max_size:classes 29, max_size:data 803, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.136 sec/step, elapsed 0:12:44, exp. remaining 0:19:17, complete 39.77%
att-weights epoch 481, step 413, max_size:classes 28, max_size:data 718, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.311 sec/step, elapsed 0:12:45, exp. remaining 0:19:10, complete 39.96%
att-weights epoch 481, step 414, max_size:classes 26, max_size:data 943, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.491 sec/step, elapsed 0:12:47, exp. remaining 0:19:05, complete 40.10%
att-weights epoch 481, step 415, max_size:classes 28, max_size:data 802, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.194 sec/step, elapsed 0:12:48, exp. remaining 0:19:00, complete 40.25%
att-weights epoch 481, step 416, max_size:classes 29, max_size:data 720, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.140 sec/step, elapsed 0:12:49, exp. remaining 0:18:53, complete 40.44%
att-weights epoch 481, step 417, max_size:classes 31, max_size:data 829, mem_usage:GPU:0 1.0GB, num_seqs 4, 2.384 sec/step, elapsed 0:12:51, exp. remaining 0:18:49, complete 40.58%
att-weights epoch 481, step 418, max_size:classes 28, max_size:data 783, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.369 sec/step, elapsed 0:12:53, exp. remaining 0:18:45, complete 40.73%
att-weights epoch 481, step 419, max_size:classes 27, max_size:data 872, mem_usage:GPU:0 1.0GB, num_seqs 4, 3.148 sec/step, elapsed 0:12:56, exp. remaining 0:18:41, complete 40.92%
att-weights epoch 481, step 420, max_size:classes 27, max_size:data 902, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.399 sec/step, elapsed 0:12:57, exp. remaining 0:18:36, complete 41.07%
att-weights epoch 481, step 421, max_size:classes 26, max_size:data 807, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.419 sec/step, elapsed 0:12:59, exp. remaining 0:18:29, complete 41.25%
att-weights epoch 481, step 422, max_size:classes 29, max_size:data 773, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.570 sec/step, elapsed 0:13:00, exp. remaining 0:18:23, complete 41.44%
att-weights epoch 481, step 423, max_size:classes 25, max_size:data 736, mem_usage:GPU:0 1.0GB, num_seqs 4, 3.638 sec/step, elapsed 0:13:04, exp. remaining 0:18:21, complete 41.58%
att-weights epoch 481, step 424, max_size:classes 27, max_size:data 929, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.806 sec/step, elapsed 0:13:06, exp. remaining 0:18:17, complete 41.73%
att-weights epoch 481, step 425, max_size:classes 28, max_size:data 772, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.321 sec/step, elapsed 0:13:07, exp. remaining 0:18:12, complete 41.88%
att-weights epoch 481, step 426, max_size:classes 31, max_size:data 759, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.818 sec/step, elapsed 0:13:09, exp. remaining 0:18:07, complete 42.06%
att-weights epoch 481, step 427, max_size:classes 26, max_size:data 875, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.532 sec/step, elapsed 0:13:10, exp. remaining 0:18:01, complete 42.25%
att-weights epoch 481, step 428, max_size:classes 28, max_size:data 798, mem_usage:GPU:0 1.0GB, num_seqs 5, 2.783 sec/step, elapsed 0:13:13, exp. remaining 0:17:58, complete 42.40%
att-weights epoch 481, step 429, max_size:classes 25, max_size:data 821, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.127 sec/step, elapsed 0:13:14, exp. remaining 0:17:51, complete 42.58%
att-weights epoch 481, step 430, max_size:classes 26, max_size:data 699, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.540 sec/step, elapsed 0:13:16, exp. remaining 0:17:45, complete 42.77%
att-weights epoch 481, step 431, max_size:classes 31, max_size:data 753, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.499 sec/step, elapsed 0:13:17, exp. remaining 0:17:41, complete 42.92%
att-weights epoch 481, step 432, max_size:classes 26, max_size:data 786, mem_usage:GPU:0 1.0GB, num_seqs 4, 6.600 sec/step, elapsed 0:13:24, exp. remaining 0:17:43, complete 43.06%
att-weights epoch 481, step 433, max_size:classes 25, max_size:data 958, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.193 sec/step, elapsed 0:13:25, exp. remaining 0:17:38, complete 43.21%
att-weights epoch 481, step 434, max_size:classes 22, max_size:data 842, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.097 sec/step, elapsed 0:13:26, exp. remaining 0:17:32, complete 43.40%
att-weights epoch 481, step 435, max_size:classes 27, max_size:data 785, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.648 sec/step, elapsed 0:13:28, exp. remaining 0:17:28, complete 43.54%
att-weights epoch 481, step 436, max_size:classes 26, max_size:data 794, mem_usage:GPU:0 1.0GB, num_seqs 5, 3.948 sec/step, elapsed 0:13:32, exp. remaining 0:17:26, complete 43.69%
att-weights epoch 481, step 437, max_size:classes 25, max_size:data 801, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.454 sec/step, elapsed 0:13:33, exp. remaining 0:17:20, complete 43.88%
att-weights epoch 481, step 438, max_size:classes 25, max_size:data 769, mem_usage:GPU:0 1.0GB, num_seqs 5, 2.414 sec/step, elapsed 0:13:36, exp. remaining 0:17:17, complete 44.03%
att-weights epoch 481, step 439, max_size:classes 26, max_size:data 737, mem_usage:GPU:0 1.0GB, num_seqs 5, 7.585 sec/step, elapsed 0:13:43, exp. remaining 0:17:21, complete 44.17%
att-weights epoch 481, step 440, max_size:classes 25, max_size:data 853, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.524 sec/step, elapsed 0:13:45, exp. remaining 0:17:15, complete 44.36%
att-weights epoch 481, step 441, max_size:classes 27, max_size:data 660, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.706 sec/step, elapsed 0:13:47, exp. remaining 0:17:09, complete 44.54%
att-weights epoch 481, step 442, max_size:classes 24, max_size:data 848, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.316 sec/step, elapsed 0:13:48, exp. remaining 0:17:03, complete 44.73%
att-weights epoch 481, step 443, max_size:classes 24, max_size:data 694, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.942 sec/step, elapsed 0:13:50, exp. remaining 0:16:58, complete 44.91%
att-weights epoch 481, step 444, max_size:classes 25, max_size:data 823, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.868 sec/step, elapsed 0:13:52, exp. remaining 0:16:54, complete 45.06%
att-weights epoch 481, step 445, max_size:classes 27, max_size:data 801, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.325 sec/step, elapsed 0:13:53, exp. remaining 0:16:48, complete 45.25%
att-weights epoch 481, step 446, max_size:classes 24, max_size:data 738, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.677 sec/step, elapsed 0:13:55, exp. remaining 0:16:44, complete 45.39%
att-weights epoch 481, step 447, max_size:classes 22, max_size:data 806, mem_usage:GPU:0 1.0GB, num_seqs 4, 3.735 sec/step, elapsed 0:13:58, exp. remaining 0:16:43, complete 45.54%
att-weights epoch 481, step 448, max_size:classes 27, max_size:data 925, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.422 sec/step, elapsed 0:14:00, exp. remaining 0:16:38, complete 45.69%
att-weights epoch 481, step 449, max_size:classes 24, max_size:data 679, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.399 sec/step, elapsed 0:14:01, exp. remaining 0:16:34, complete 45.84%
att-weights epoch 481, step 450, max_size:classes 24, max_size:data 750, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.564 sec/step, elapsed 0:14:03, exp. remaining 0:16:29, complete 46.02%
att-weights epoch 481, step 451, max_size:classes 22, max_size:data 764, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.580 sec/step, elapsed 0:14:04, exp. remaining 0:16:22, complete 46.24%
att-weights epoch 481, step 452, max_size:classes 22, max_size:data 743, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.402 sec/step, elapsed 0:14:06, exp. remaining 0:16:16, complete 46.43%
att-weights epoch 481, step 453, max_size:classes 23, max_size:data 863, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.090 sec/step, elapsed 0:14:07, exp. remaining 0:16:10, complete 46.61%
att-weights epoch 481, step 454, max_size:classes 22, max_size:data 779, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.208 sec/step, elapsed 0:14:08, exp. remaining 0:16:04, complete 46.80%
att-weights epoch 481, step 455, max_size:classes 25, max_size:data 874, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.396 sec/step, elapsed 0:14:09, exp. remaining 0:16:01, complete 46.91%
att-weights epoch 481, step 456, max_size:classes 25, max_size:data 963, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.118 sec/step, elapsed 0:14:11, exp. remaining 0:15:58, complete 47.02%
att-weights epoch 481, step 457, max_size:classes 25, max_size:data 808, mem_usage:GPU:0 1.0GB, num_seqs 4, 3.204 sec/step, elapsed 0:14:14, exp. remaining 0:15:55, complete 47.21%
att-weights epoch 481, step 458, max_size:classes 23, max_size:data 898, mem_usage:GPU:0 1.0GB, num_seqs 4, 2.357 sec/step, elapsed 0:14:16, exp. remaining 0:15:50, complete 47.39%
att-weights epoch 481, step 459, max_size:classes 23, max_size:data 683, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.321 sec/step, elapsed 0:14:17, exp. remaining 0:15:46, complete 47.54%
att-weights epoch 481, step 460, max_size:classes 23, max_size:data 599, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.918 sec/step, elapsed 0:14:19, exp. remaining 0:15:41, complete 47.72%
att-weights epoch 481, step 461, max_size:classes 28, max_size:data 758, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.485 sec/step, elapsed 0:14:21, exp. remaining 0:15:36, complete 47.91%
att-weights epoch 481, step 462, max_size:classes 22, max_size:data 725, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.958 sec/step, elapsed 0:14:23, exp. remaining 0:15:33, complete 48.06%
att-weights epoch 481, step 463, max_size:classes 25, max_size:data 686, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.193 sec/step, elapsed 0:14:24, exp. remaining 0:15:27, complete 48.24%
att-weights epoch 481, step 464, max_size:classes 24, max_size:data 831, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.207 sec/step, elapsed 0:14:25, exp. remaining 0:15:21, complete 48.43%
att-weights epoch 481, step 465, max_size:classes 23, max_size:data 1043, mem_usage:GPU:0 1.0GB, num_seqs 3, 1.081 sec/step, elapsed 0:14:26, exp. remaining 0:15:16, complete 48.61%
att-weights epoch 481, step 466, max_size:classes 23, max_size:data 738, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.198 sec/step, elapsed 0:14:28, exp. remaining 0:15:10, complete 48.80%
att-weights epoch 481, step 467, max_size:classes 23, max_size:data 767, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.379 sec/step, elapsed 0:14:29, exp. remaining 0:15:05, complete 48.98%
att-weights epoch 481, step 468, max_size:classes 23, max_size:data 814, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.087 sec/step, elapsed 0:14:30, exp. remaining 0:14:58, complete 49.20%
att-weights epoch 481, step 469, max_size:classes 25, max_size:data 711, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.197 sec/step, elapsed 0:14:31, exp. remaining 0:14:53, complete 49.39%
att-weights epoch 481, step 470, max_size:classes 22, max_size:data 755, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.320 sec/step, elapsed 0:14:32, exp. remaining 0:14:49, complete 49.54%
att-weights epoch 481, step 471, max_size:classes 23, max_size:data 826, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.034 sec/step, elapsed 0:14:34, exp. remaining 0:14:43, complete 49.72%
att-weights epoch 481, step 472, max_size:classes 26, max_size:data 697, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.310 sec/step, elapsed 0:14:35, exp. remaining 0:14:38, complete 49.91%
att-weights epoch 481, step 473, max_size:classes 24, max_size:data 672, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.391 sec/step, elapsed 0:14:36, exp. remaining 0:14:32, complete 50.13%
att-weights epoch 481, step 474, max_size:classes 26, max_size:data 671, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.363 sec/step, elapsed 0:14:38, exp. remaining 0:14:27, complete 50.31%
att-weights epoch 481, step 475, max_size:classes 22, max_size:data 695, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.272 sec/step, elapsed 0:14:39, exp. remaining 0:14:21, complete 50.50%
att-weights epoch 481, step 476, max_size:classes 24, max_size:data 785, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.377 sec/step, elapsed 0:14:40, exp. remaining 0:14:16, complete 50.68%
att-weights epoch 481, step 477, max_size:classes 25, max_size:data 640, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.906 sec/step, elapsed 0:14:42, exp. remaining 0:14:12, complete 50.87%
att-weights epoch 481, step 478, max_size:classes 24, max_size:data 753, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.281 sec/step, elapsed 0:14:43, exp. remaining 0:14:07, complete 51.05%
att-weights epoch 481, step 479, max_size:classes 21, max_size:data 818, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.344 sec/step, elapsed 0:14:45, exp. remaining 0:14:02, complete 51.24%
att-weights epoch 481, step 480, max_size:classes 24, max_size:data 730, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.686 sec/step, elapsed 0:14:46, exp. remaining 0:13:59, complete 51.39%
att-weights epoch 481, step 481, max_size:classes 23, max_size:data 673, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.470 sec/step, elapsed 0:14:48, exp. remaining 0:13:54, complete 51.57%
att-weights epoch 481, step 482, max_size:classes 21, max_size:data 661, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.553 sec/step, elapsed 0:14:49, exp. remaining 0:13:49, complete 51.76%
att-weights epoch 481, step 483, max_size:classes 23, max_size:data 745, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.365 sec/step, elapsed 0:14:51, exp. remaining 0:13:45, complete 51.91%
att-weights epoch 481, step 484, max_size:classes 23, max_size:data 674, mem_usage:GPU:0 1.0GB, num_seqs 5, 3.852 sec/step, elapsed 0:14:55, exp. remaining 0:13:43, complete 52.09%
att-weights epoch 481, step 485, max_size:classes 21, max_size:data 688, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.327 sec/step, elapsed 0:14:56, exp. remaining 0:13:37, complete 52.31%
att-weights epoch 481, step 486, max_size:classes 21, max_size:data 714, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.462 sec/step, elapsed 0:14:58, exp. remaining 0:13:32, complete 52.50%
att-weights epoch 481, step 487, max_size:classes 21, max_size:data 691, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.499 sec/step, elapsed 0:14:59, exp. remaining 0:13:27, complete 52.68%
att-weights epoch 481, step 488, max_size:classes 22, max_size:data 720, mem_usage:GPU:0 1.0GB, num_seqs 5, 3.803 sec/step, elapsed 0:15:03, exp. remaining 0:13:24, complete 52.90%
att-weights epoch 481, step 489, max_size:classes 21, max_size:data 987, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.149 sec/step, elapsed 0:15:04, exp. remaining 0:13:18, complete 53.13%
att-weights epoch 481, step 490, max_size:classes 23, max_size:data 669, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.964 sec/step, elapsed 0:15:06, exp. remaining 0:13:13, complete 53.31%
att-weights epoch 481, step 491, max_size:classes 23, max_size:data 794, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.744 sec/step, elapsed 0:15:08, exp. remaining 0:13:09, complete 53.50%
att-weights epoch 481, step 492, max_size:classes 22, max_size:data 859, mem_usage:GPU:0 1.0GB, num_seqs 4, 1.356 sec/step, elapsed 0:15:09, exp. remaining 0:13:04, complete 53.68%
att-weights epoch 481, step 493, max_size:classes 21, max_size:data 681, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.447 sec/step, elapsed 0:15:10, exp. remaining 0:13:00, complete 53.87%
att-weights epoch 481, step 494, max_size:classes 21, max_size:data 622, mem_usage:GPU:0 1.0GB, num_seqs 6, 2.843 sec/step, elapsed 0:15:13, exp. remaining 0:12:55, complete 54.09%
att-weights epoch 481, step 495, max_size:classes 21, max_size:data 798, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.530 sec/step, elapsed 0:15:15, exp. remaining 0:12:50, complete 54.31%
att-weights epoch 481, step 496, max_size:classes 22, max_size:data 699, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.475 sec/step, elapsed 0:15:16, exp. remaining 0:12:45, complete 54.50%
att-weights epoch 481, step 497, max_size:classes 21, max_size:data 602, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.739 sec/step, elapsed 0:15:18, exp. remaining 0:12:40, complete 54.72%
att-weights epoch 481, step 498, max_size:classes 22, max_size:data 612, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.926 sec/step, elapsed 0:15:20, exp. remaining 0:12:34, complete 54.94%
att-weights epoch 481, step 499, max_size:classes 19, max_size:data 789, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.381 sec/step, elapsed 0:15:21, exp. remaining 0:12:30, complete 55.12%
att-weights epoch 481, step 500, max_size:classes 22, max_size:data 738, mem_usage:GPU:0 1.0GB, num_seqs 5, 2.389 sec/step, elapsed 0:15:24, exp. remaining 0:12:26, complete 55.31%
att-weights epoch 481, step 501, max_size:classes 21, max_size:data 585, mem_usage:GPU:0 1.0GB, num_seqs 5, 2.032 sec/step, elapsed 0:15:26, exp. remaining 0:12:21, complete 55.53%
att-weights epoch 481, step 502, max_size:classes 23, max_size:data 693, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.387 sec/step, elapsed 0:15:27, exp. remaining 0:12:17, complete 55.72%
att-weights epoch 481, step 503, max_size:classes 21, max_size:data 587, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.453 sec/step, elapsed 0:15:29, exp. remaining 0:12:11, complete 55.94%
att-weights epoch 481, step 504, max_size:classes 20, max_size:data 602, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.464 sec/step, elapsed 0:15:30, exp. remaining 0:12:06, complete 56.16%
att-weights epoch 481, step 505, max_size:classes 21, max_size:data 698, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.651 sec/step, elapsed 0:15:32, exp. remaining 0:12:01, complete 56.38%
att-weights epoch 481, step 506, max_size:classes 24, max_size:data 563, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.451 sec/step, elapsed 0:15:33, exp. remaining 0:11:55, complete 56.60%
att-weights epoch 481, step 507, max_size:classes 23, max_size:data 585, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.249 sec/step, elapsed 0:15:34, exp. remaining 0:11:49, complete 56.86%
att-weights epoch 481, step 508, max_size:classes 22, max_size:data 598, mem_usage:GPU:0 1.0GB, num_seqs 5, 2.343 sec/step, elapsed 0:15:37, exp. remaining 0:11:44, complete 57.08%
att-weights epoch 481, step 509, max_size:classes 19, max_size:data 686, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.690 sec/step, elapsed 0:15:38, exp. remaining 0:11:40, complete 57.27%
att-weights epoch 481, step 510, max_size:classes 19, max_size:data 660, mem_usage:GPU:0 1.0GB, num_seqs 6, 2.025 sec/step, elapsed 0:15:41, exp. remaining 0:11:36, complete 57.45%
att-weights epoch 481, step 511, max_size:classes 21, max_size:data 672, mem_usage:GPU:0 1.0GB, num_seqs 5, 2.024 sec/step, elapsed 0:15:43, exp. remaining 0:11:32, complete 57.68%
att-weights epoch 481, step 512, max_size:classes 20, max_size:data 598, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.693 sec/step, elapsed 0:15:44, exp. remaining 0:11:26, complete 57.90%
att-weights epoch 481, step 513, max_size:classes 23, max_size:data 601, mem_usage:GPU:0 1.0GB, num_seqs 6, 3.040 sec/step, elapsed 0:15:47, exp. remaining 0:11:22, complete 58.12%
att-weights epoch 481, step 514, max_size:classes 23, max_size:data 621, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.430 sec/step, elapsed 0:15:49, exp. remaining 0:11:17, complete 58.34%
att-weights epoch 481, step 515, max_size:classes 21, max_size:data 648, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.547 sec/step, elapsed 0:15:50, exp. remaining 0:11:12, complete 58.56%
att-weights epoch 481, step 516, max_size:classes 21, max_size:data 563, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.623 sec/step, elapsed 0:15:52, exp. remaining 0:11:07, complete 58.79%
att-weights epoch 481, step 517, max_size:classes 20, max_size:data 574, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.572 sec/step, elapsed 0:15:53, exp. remaining 0:11:03, complete 58.97%
att-weights epoch 481, step 518, max_size:classes 22, max_size:data 605, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.656 sec/step, elapsed 0:15:55, exp. remaining 0:10:57, complete 59.23%
att-weights epoch 481, step 519, max_size:classes 19, max_size:data 687, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.435 sec/step, elapsed 0:15:57, exp. remaining 0:10:52, complete 59.45%
att-weights epoch 481, step 520, max_size:classes 19, max_size:data 654, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.652 sec/step, elapsed 0:15:58, exp. remaining 0:10:46, complete 59.71%
att-weights epoch 481, step 521, max_size:classes 20, max_size:data 579, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.452 sec/step, elapsed 0:16:00, exp. remaining 0:10:40, complete 59.97%
att-weights epoch 481, step 522, max_size:classes 19, max_size:data 623, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.660 sec/step, elapsed 0:16:01, exp. remaining 0:10:37, complete 60.16%
att-weights epoch 481, step 523, max_size:classes 19, max_size:data 651, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.435 sec/step, elapsed 0:16:03, exp. remaining 0:10:32, complete 60.38%
att-weights epoch 481, step 524, max_size:classes 21, max_size:data 653, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.398 sec/step, elapsed 0:16:04, exp. remaining 0:10:27, complete 60.60%
att-weights epoch 481, step 525, max_size:classes 19, max_size:data 612, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.667 sec/step, elapsed 0:16:06, exp. remaining 0:10:21, complete 60.86%
att-weights epoch 481, step 526, max_size:classes 20, max_size:data 692, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.169 sec/step, elapsed 0:16:07, exp. remaining 0:10:16, complete 61.08%
att-weights epoch 481, step 527, max_size:classes 19, max_size:data 524, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.650 sec/step, elapsed 0:16:09, exp. remaining 0:10:12, complete 61.27%
att-weights epoch 481, step 528, max_size:classes 21, max_size:data 588, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.250 sec/step, elapsed 0:16:10, exp. remaining 0:10:07, complete 61.49%
att-weights epoch 481, step 529, max_size:classes 19, max_size:data 557, mem_usage:GPU:0 1.0GB, num_seqs 7, 2.426 sec/step, elapsed 0:16:12, exp. remaining 0:10:02, complete 61.75%
att-weights epoch 481, step 530, max_size:classes 18, max_size:data 543, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.722 sec/step, elapsed 0:16:14, exp. remaining 0:09:57, complete 62.01%
att-weights epoch 481, step 531, max_size:classes 16, max_size:data 678, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.175 sec/step, elapsed 0:16:15, exp. remaining 0:09:51, complete 62.26%
att-weights epoch 481, step 532, max_size:classes 22, max_size:data 591, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.370 sec/step, elapsed 0:16:17, exp. remaining 0:09:45, complete 62.52%
att-weights epoch 481, step 533, max_size:classes 19, max_size:data 612, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.476 sec/step, elapsed 0:16:18, exp. remaining 0:09:40, complete 62.78%
att-weights epoch 481, step 534, max_size:classes 20, max_size:data 542, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.605 sec/step, elapsed 0:16:20, exp. remaining 0:09:35, complete 63.00%
att-weights epoch 481, step 535, max_size:classes 19, max_size:data 656, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.744 sec/step, elapsed 0:16:21, exp. remaining 0:09:30, complete 63.26%
att-weights epoch 481, step 536, max_size:classes 20, max_size:data 728, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.484 sec/step, elapsed 0:16:23, exp. remaining 0:09:25, complete 63.49%
att-weights epoch 481, step 537, max_size:classes 22, max_size:data 658, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.346 sec/step, elapsed 0:16:24, exp. remaining 0:09:19, complete 63.78%
att-weights epoch 481, step 538, max_size:classes 21, max_size:data 494, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.236 sec/step, elapsed 0:16:25, exp. remaining 0:09:14, complete 64.00%
att-weights epoch 481, step 539, max_size:classes 18, max_size:data 536, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.624 sec/step, elapsed 0:16:27, exp. remaining 0:09:10, complete 64.22%
att-weights epoch 481, step 540, max_size:classes 17, max_size:data 554, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.397 sec/step, elapsed 0:16:28, exp. remaining 0:09:05, complete 64.45%
att-weights epoch 481, step 541, max_size:classes 20, max_size:data 543, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.624 sec/step, elapsed 0:16:30, exp. remaining 0:09:01, complete 64.67%
att-weights epoch 481, step 542, max_size:classes 18, max_size:data 569, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.811 sec/step, elapsed 0:16:32, exp. remaining 0:08:56, complete 64.89%
att-weights epoch 481, step 543, max_size:classes 20, max_size:data 572, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.306 sec/step, elapsed 0:16:33, exp. remaining 0:08:52, complete 65.11%
att-weights epoch 481, step 544, max_size:classes 18, max_size:data 522, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.786 sec/step, elapsed 0:16:35, exp. remaining 0:08:46, complete 65.41%
att-weights epoch 481, step 545, max_size:classes 15, max_size:data 652, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.642 sec/step, elapsed 0:16:37, exp. remaining 0:08:43, complete 65.59%
att-weights epoch 481, step 546, max_size:classes 16, max_size:data 470, mem_usage:GPU:0 1.0GB, num_seqs 8, 7.901 sec/step, elapsed 0:16:45, exp. remaining 0:08:41, complete 65.85%
att-weights epoch 481, step 547, max_size:classes 17, max_size:data 587, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.626 sec/step, elapsed 0:16:46, exp. remaining 0:08:37, complete 66.04%
att-weights epoch 481, step 548, max_size:classes 19, max_size:data 660, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.510 sec/step, elapsed 0:16:48, exp. remaining 0:08:33, complete 66.26%
att-weights epoch 481, step 549, max_size:classes 16, max_size:data 627, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.433 sec/step, elapsed 0:16:49, exp. remaining 0:08:27, complete 66.56%
att-weights epoch 481, step 550, max_size:classes 21, max_size:data 603, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.293 sec/step, elapsed 0:16:50, exp. remaining 0:08:22, complete 66.81%
att-weights epoch 481, step 551, max_size:classes 18, max_size:data 579, mem_usage:GPU:0 1.0GB, num_seqs 6, 12.031 sec/step, elapsed 0:17:02, exp. remaining 0:08:21, complete 67.11%
att-weights epoch 481, step 552, max_size:classes 17, max_size:data 601, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.344 sec/step, elapsed 0:17:04, exp. remaining 0:08:16, complete 67.37%
att-weights epoch 481, step 553, max_size:classes 18, max_size:data 494, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.572 sec/step, elapsed 0:17:05, exp. remaining 0:08:11, complete 67.63%
att-weights epoch 481, step 554, max_size:classes 18, max_size:data 677, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.593 sec/step, elapsed 0:17:07, exp. remaining 0:08:06, complete 67.85%
att-weights epoch 481, step 555, max_size:classes 16, max_size:data 566, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.828 sec/step, elapsed 0:17:09, exp. remaining 0:08:02, complete 68.07%
att-weights epoch 481, step 556, max_size:classes 16, max_size:data 702, mem_usage:GPU:0 1.0GB, num_seqs 5, 1.492 sec/step, elapsed 0:17:10, exp. remaining 0:07:57, complete 68.33%
att-weights epoch 481, step 557, max_size:classes 17, max_size:data 630, mem_usage:GPU:0 1.0GB, num_seqs 6, 3.749 sec/step, elapsed 0:17:14, exp. remaining 0:07:54, complete 68.55%
att-weights epoch 481, step 558, max_size:classes 16, max_size:data 482, mem_usage:GPU:0 1.0GB, num_seqs 8, 12.708 sec/step, elapsed 0:17:27, exp. remaining 0:07:54, complete 68.81%
att-weights epoch 481, step 559, max_size:classes 16, max_size:data 523, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.462 sec/step, elapsed 0:17:28, exp. remaining 0:07:49, complete 69.07%
att-weights epoch 481, step 560, max_size:classes 17, max_size:data 492, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.456 sec/step, elapsed 0:17:30, exp. remaining 0:07:45, complete 69.29%
att-weights epoch 481, step 561, max_size:classes 16, max_size:data 545, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.324 sec/step, elapsed 0:17:31, exp. remaining 0:07:39, complete 69.59%
att-weights epoch 481, step 562, max_size:classes 20, max_size:data 485, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.502 sec/step, elapsed 0:17:33, exp. remaining 0:07:33, complete 69.89%
att-weights epoch 481, step 563, max_size:classes 16, max_size:data 635, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.385 sec/step, elapsed 0:17:34, exp. remaining 0:07:28, complete 70.14%
att-weights epoch 481, step 564, max_size:classes 17, max_size:data 644, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.312 sec/step, elapsed 0:17:35, exp. remaining 0:07:23, complete 70.40%
att-weights epoch 481, step 565, max_size:classes 19, max_size:data 495, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.346 sec/step, elapsed 0:17:37, exp. remaining 0:07:18, complete 70.70%
att-weights epoch 481, step 566, max_size:classes 16, max_size:data 593, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.405 sec/step, elapsed 0:17:38, exp. remaining 0:07:13, complete 70.96%
att-weights epoch 481, step 567, max_size:classes 15, max_size:data 517, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.417 sec/step, elapsed 0:17:39, exp. remaining 0:07:07, complete 71.25%
att-weights epoch 481, step 568, max_size:classes 16, max_size:data 543, mem_usage:GPU:0 1.0GB, num_seqs 7, 10.666 sec/step, elapsed 0:17:50, exp. remaining 0:07:05, complete 71.55%
att-weights epoch 481, step 569, max_size:classes 16, max_size:data 585, mem_usage:GPU:0 1.0GB, num_seqs 6, 1.157 sec/step, elapsed 0:17:51, exp. remaining 0:07:00, complete 71.81%
att-weights epoch 481, step 570, max_size:classes 18, max_size:data 492, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.800 sec/step, elapsed 0:17:53, exp. remaining 0:06:55, complete 72.11%
att-weights epoch 481, step 571, max_size:classes 15, max_size:data 488, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.724 sec/step, elapsed 0:17:55, exp. remaining 0:06:49, complete 72.40%
att-weights epoch 481, step 572, max_size:classes 16, max_size:data 549, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.307 sec/step, elapsed 0:17:56, exp. remaining 0:06:42, complete 72.77%
att-weights epoch 481, step 573, max_size:classes 16, max_size:data 541, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.539 sec/step, elapsed 0:17:58, exp. remaining 0:06:37, complete 73.07%
att-weights epoch 481, step 574, max_size:classes 15, max_size:data 488, mem_usage:GPU:0 1.0GB, num_seqs 8, 7.927 sec/step, elapsed 0:18:06, exp. remaining 0:06:34, complete 73.36%
att-weights epoch 481, step 575, max_size:classes 15, max_size:data 513, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.494 sec/step, elapsed 0:18:07, exp. remaining 0:06:29, complete 73.62%
att-weights epoch 481, step 576, max_size:classes 16, max_size:data 440, mem_usage:GPU:0 1.0GB, num_seqs 8, 2.881 sec/step, elapsed 0:18:10, exp. remaining 0:06:24, complete 73.92%
att-weights epoch 481, step 577, max_size:classes 16, max_size:data 497, mem_usage:GPU:0 1.0GB, num_seqs 8, 7.136 sec/step, elapsed 0:18:17, exp. remaining 0:06:22, complete 74.18%
att-weights epoch 481, step 578, max_size:classes 14, max_size:data 522, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.470 sec/step, elapsed 0:18:19, exp. remaining 0:06:16, complete 74.47%
att-weights epoch 481, step 579, max_size:classes 18, max_size:data 463, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.191 sec/step, elapsed 0:18:20, exp. remaining 0:06:11, complete 74.77%
att-weights epoch 481, step 580, max_size:classes 15, max_size:data 458, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.383 sec/step, elapsed 0:18:21, exp. remaining 0:06:05, complete 75.10%
att-weights epoch 481, step 581, max_size:classes 14, max_size:data 397, mem_usage:GPU:0 1.0GB, num_seqs 10, 1.706 sec/step, elapsed 0:18:23, exp. remaining 0:06:00, complete 75.36%
att-weights epoch 481, step 582, max_size:classes 16, max_size:data 475, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.408 sec/step, elapsed 0:18:24, exp. remaining 0:05:55, complete 75.66%
att-weights epoch 481, step 583, max_size:classes 14, max_size:data 482, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.442 sec/step, elapsed 0:18:26, exp. remaining 0:05:50, complete 75.92%
att-weights epoch 481, step 584, max_size:classes 14, max_size:data 553, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.609 sec/step, elapsed 0:18:27, exp. remaining 0:05:45, complete 76.21%
att-weights epoch 481, step 585, max_size:classes 15, max_size:data 457, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.670 sec/step, elapsed 0:18:29, exp. remaining 0:05:40, complete 76.51%
att-weights epoch 481, step 586, max_size:classes 14, max_size:data 517, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.292 sec/step, elapsed 0:18:30, exp. remaining 0:05:36, complete 76.77%
att-weights epoch 481, step 587, max_size:classes 13, max_size:data 493, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.584 sec/step, elapsed 0:18:32, exp. remaining 0:05:31, complete 77.03%
att-weights epoch 481, step 588, max_size:classes 14, max_size:data 490, mem_usage:GPU:0 1.0GB, num_seqs 8, 3.826 sec/step, elapsed 0:18:36, exp. remaining 0:05:27, complete 77.32%
att-weights epoch 481, step 589, max_size:classes 17, max_size:data 419, mem_usage:GPU:0 1.0GB, num_seqs 9, 1.678 sec/step, elapsed 0:18:37, exp. remaining 0:05:23, complete 77.58%
att-weights epoch 481, step 590, max_size:classes 13, max_size:data 508, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.102 sec/step, elapsed 0:18:38, exp. remaining 0:05:17, complete 77.88%
att-weights epoch 481, step 591, max_size:classes 14, max_size:data 485, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.223 sec/step, elapsed 0:18:40, exp. remaining 0:05:12, complete 78.17%
att-weights epoch 481, step 592, max_size:classes 15, max_size:data 539, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.442 sec/step, elapsed 0:18:41, exp. remaining 0:05:07, complete 78.51%
att-weights epoch 481, step 593, max_size:classes 15, max_size:data 470, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.464 sec/step, elapsed 0:18:43, exp. remaining 0:05:02, complete 78.80%
att-weights epoch 481, step 594, max_size:classes 15, max_size:data 474, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.582 sec/step, elapsed 0:18:44, exp. remaining 0:04:57, complete 79.10%
att-weights epoch 481, step 595, max_size:classes 14, max_size:data 481, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.404 sec/step, elapsed 0:18:46, exp. remaining 0:04:50, complete 79.47%
att-weights epoch 481, step 596, max_size:classes 17, max_size:data 571, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.537 sec/step, elapsed 0:18:47, exp. remaining 0:04:45, complete 79.80%
att-weights epoch 481, step 597, max_size:classes 14, max_size:data 483, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.509 sec/step, elapsed 0:18:49, exp. remaining 0:04:40, complete 80.10%
att-weights epoch 481, step 598, max_size:classes 17, max_size:data 530, mem_usage:GPU:0 1.0GB, num_seqs 7, 1.645 sec/step, elapsed 0:18:50, exp. remaining 0:04:35, complete 80.43%
att-weights epoch 481, step 599, max_size:classes 14, max_size:data 477, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.568 sec/step, elapsed 0:18:52, exp. remaining 0:04:30, complete 80.73%
att-weights epoch 481, step 600, max_size:classes 14, max_size:data 468, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.464 sec/step, elapsed 0:18:53, exp. remaining 0:04:24, complete 81.10%
att-weights epoch 481, step 601, max_size:classes 13, max_size:data 432, mem_usage:GPU:0 1.0GB, num_seqs 9, 1.752 sec/step, elapsed 0:18:55, exp. remaining 0:04:19, complete 81.43%
att-weights epoch 481, step 602, max_size:classes 14, max_size:data 471, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.561 sec/step, elapsed 0:18:57, exp. remaining 0:04:13, complete 81.80%
att-weights epoch 481, step 603, max_size:classes 13, max_size:data 488, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.468 sec/step, elapsed 0:18:58, exp. remaining 0:04:07, complete 82.17%
att-weights epoch 481, step 604, max_size:classes 14, max_size:data 389, mem_usage:GPU:0 1.0GB, num_seqs 10, 1.655 sec/step, elapsed 0:19:00, exp. remaining 0:04:01, complete 82.50%
att-weights epoch 481, step 605, max_size:classes 13, max_size:data 406, mem_usage:GPU:0 1.0GB, num_seqs 9, 3.715 sec/step, elapsed 0:19:03, exp. remaining 0:03:57, complete 82.83%
att-weights epoch 481, step 606, max_size:classes 13, max_size:data 486, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.349 sec/step, elapsed 0:19:05, exp. remaining 0:03:51, complete 83.17%
att-weights epoch 481, step 607, max_size:classes 13, max_size:data 423, mem_usage:GPU:0 1.0GB, num_seqs 9, 1.717 sec/step, elapsed 0:19:07, exp. remaining 0:03:47, complete 83.46%
att-weights epoch 481, step 608, max_size:classes 13, max_size:data 454, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.408 sec/step, elapsed 0:19:08, exp. remaining 0:03:42, complete 83.80%
att-weights epoch 481, step 609, max_size:classes 12, max_size:data 391, mem_usage:GPU:0 1.0GB, num_seqs 10, 1.523 sec/step, elapsed 0:19:09, exp. remaining 0:03:36, complete 84.17%
att-weights epoch 481, step 610, max_size:classes 12, max_size:data 423, mem_usage:GPU:0 1.0GB, num_seqs 9, 1.517 sec/step, elapsed 0:19:11, exp. remaining 0:03:30, complete 84.54%
att-weights epoch 481, step 611, max_size:classes 12, max_size:data 368, mem_usage:GPU:0 1.0GB, num_seqs 10, 1.867 sec/step, elapsed 0:19:13, exp. remaining 0:03:26, complete 84.83%
att-weights epoch 481, step 612, max_size:classes 14, max_size:data 368, mem_usage:GPU:0 1.0GB, num_seqs 10, 18.381 sec/step, elapsed 0:19:31, exp. remaining 0:03:23, complete 85.20%
att-weights epoch 481, step 613, max_size:classes 12, max_size:data 429, mem_usage:GPU:0 1.0GB, num_seqs 9, 2.131 sec/step, elapsed 0:19:33, exp. remaining 0:03:18, complete 85.53%
att-weights epoch 481, step 614, max_size:classes 12, max_size:data 427, mem_usage:GPU:0 1.0GB, num_seqs 9, 9.850 sec/step, elapsed 0:19:43, exp. remaining 0:03:15, complete 85.83%
att-weights epoch 481, step 615, max_size:classes 12, max_size:data 426, mem_usage:GPU:0 1.0GB, num_seqs 9, 1.817 sec/step, elapsed 0:19:45, exp. remaining 0:03:10, complete 86.16%
att-weights epoch 481, step 616, max_size:classes 11, max_size:data 471, mem_usage:GPU:0 1.0GB, num_seqs 8, 2.536 sec/step, elapsed 0:19:48, exp. remaining 0:03:04, complete 86.53%
att-weights epoch 481, step 617, max_size:classes 11, max_size:data 418, mem_usage:GPU:0 1.0GB, num_seqs 9, 4.568 sec/step, elapsed 0:19:52, exp. remaining 0:03:00, complete 86.87%
att-weights epoch 481, step 618, max_size:classes 13, max_size:data 388, mem_usage:GPU:0 1.0GB, num_seqs 10, 1.736 sec/step, elapsed 0:19:54, exp. remaining 0:02:55, complete 87.16%
att-weights epoch 481, step 619, max_size:classes 11, max_size:data 380, mem_usage:GPU:0 1.0GB, num_seqs 10, 2.502 sec/step, elapsed 0:19:56, exp. remaining 0:02:51, complete 87.46%
att-weights epoch 481, step 620, max_size:classes 13, max_size:data 468, mem_usage:GPU:0 1.0GB, num_seqs 8, 8.689 sec/step, elapsed 0:20:05, exp. remaining 0:02:47, complete 87.83%
att-weights epoch 481, step 621, max_size:classes 12, max_size:data 386, mem_usage:GPU:0 1.0GB, num_seqs 10, 1.914 sec/step, elapsed 0:20:07, exp. remaining 0:02:41, complete 88.24%
att-weights epoch 481, step 622, max_size:classes 10, max_size:data 401, mem_usage:GPU:0 1.0GB, num_seqs 9, 1.830 sec/step, elapsed 0:20:09, exp. remaining 0:02:35, complete 88.61%
att-weights epoch 481, step 623, max_size:classes 11, max_size:data 498, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.488 sec/step, elapsed 0:20:10, exp. remaining 0:02:30, complete 88.94%
att-weights epoch 481, step 624, max_size:classes 12, max_size:data 442, mem_usage:GPU:0 1.0GB, num_seqs 9, 1.513 sec/step, elapsed 0:20:12, exp. remaining 0:02:24, complete 89.35%
att-weights epoch 481, step 625, max_size:classes 12, max_size:data 400, mem_usage:GPU:0 1.0GB, num_seqs 10, 1.883 sec/step, elapsed 0:20:14, exp. remaining 0:02:19, complete 89.68%
att-weights epoch 481, step 626, max_size:classes 12, max_size:data 346, mem_usage:GPU:0 1.0GB, num_seqs 9, 3.679 sec/step, elapsed 0:20:17, exp. remaining 0:02:14, complete 90.05%
att-weights epoch 481, step 627, max_size:classes 13, max_size:data 482, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.602 sec/step, elapsed 0:20:19, exp. remaining 0:02:08, complete 90.46%
att-weights epoch 481, step 628, max_size:classes 11, max_size:data 458, mem_usage:GPU:0 1.0GB, num_seqs 8, 1.613 sec/step, elapsed 0:20:21, exp. remaining 0:02:03, complete 90.83%
att-weights epoch 481, step 629, max_size:classes 11, max_size:data 398, mem_usage:GPU:0 1.0GB, num_seqs 10, 19.739 sec/step, elapsed 0:20:40, exp. remaining 0:01:59, complete 91.19%
att-weights epoch 481, step 630, max_size:classes 10, max_size:data 338, mem_usage:GPU:0 1.0GB, num_seqs 11, 1.463 sec/step, elapsed 0:20:42, exp. remaining 0:01:53, complete 91.60%
att-weights epoch 481, step 631, max_size:classes 12, max_size:data 371, mem_usage:GPU:0 1.0GB, num_seqs 10, 1.706 sec/step, elapsed 0:20:44, exp. remaining 0:01:48, complete 91.97%
att-weights epoch 481, step 632, max_size:classes 11, max_size:data 403, mem_usage:GPU:0 1.0GB, num_seqs 9, 1.624 sec/step, elapsed 0:20:45, exp. remaining 0:01:43, complete 92.34%
att-weights epoch 481, step 633, max_size:classes 13, max_size:data 351, mem_usage:GPU:0 1.0GB, num_seqs 11, 1.744 sec/step, elapsed 0:20:47, exp. remaining 0:01:37, complete 92.75%
att-weights epoch 481, step 634, max_size:classes 9, max_size:data 434, mem_usage:GPU:0 1.0GB, num_seqs 9, 1.384 sec/step, elapsed 0:20:48, exp. remaining 0:01:31, complete 93.16%
att-weights epoch 481, step 635, max_size:classes 10, max_size:data 371, mem_usage:GPU:0 1.0GB, num_seqs 10, 12.037 sec/step, elapsed 0:21:00, exp. remaining 0:01:26, complete 93.60%
att-weights epoch 481, step 636, max_size:classes 12, max_size:data 345, mem_usage:GPU:0 1.0GB, num_seqs 11, 8.404 sec/step, elapsed 0:21:09, exp. remaining 0:01:20, complete 94.01%
att-weights epoch 481, step 637, max_size:classes 9, max_size:data 378, mem_usage:GPU:0 1.0GB, num_seqs 10, 1.460 sec/step, elapsed 0:21:10, exp. remaining 0:01:14, complete 94.45%
att-weights epoch 481, step 638, max_size:classes 11, max_size:data 370, mem_usage:GPU:0 1.0GB, num_seqs 10, 1.617 sec/step, elapsed 0:21:12, exp. remaining 0:01:08, complete 94.86%
att-weights epoch 481, step 639, max_size:classes 11, max_size:data 359, mem_usage:GPU:0 1.0GB, num_seqs 11, 1.666 sec/step, elapsed 0:21:13, exp. remaining 0:01:03, complete 95.26%
att-weights epoch 481, step 640, max_size:classes 10, max_size:data 372, mem_usage:GPU:0 1.0GB, num_seqs 10, 1.670 sec/step, elapsed 0:21:15, exp. remaining 0:00:57, complete 95.67%
att-weights epoch 481, step 641, max_size:classes 9, max_size:data 367, mem_usage:GPU:0 1.0GB, num_seqs 10, 1.701 sec/step, elapsed 0:21:17, exp. remaining 0:00:51, complete 96.12%
att-weights epoch 481, step 642, max_size:classes 10, max_size:data 360, mem_usage:GPU:0 1.0GB, num_seqs 11, 1.743 sec/step, elapsed 0:21:19, exp. remaining 0:00:46, complete 96.49%
att-weights epoch 481, step 643, max_size:classes 9, max_size:data 336, mem_usage:GPU:0 1.0GB, num_seqs 11, 1.708 sec/step, elapsed 0:21:20, exp. remaining 0:00:40, complete 96.97%
att-weights epoch 481, step 644, max_size:classes 9, max_size:data 309, mem_usage:GPU:0 1.0GB, num_seqs 12, 3.250 sec/step, elapsed 0:21:24, exp. remaining 0:00:34, complete 97.41%
att-weights epoch 481, step 645, max_size:classes 10, max_size:data 335, mem_usage:GPU:0 1.0GB, num_seqs 11, 2.325 sec/step, elapsed 0:21:26, exp. remaining 0:00:28, complete 97.85%
att-weights epoch 481, step 646, max_size:classes 10, max_size:data 311, mem_usage:GPU:0 1.0GB, num_seqs 12, 20.305 sec/step, elapsed 0:21:46, exp. remaining 0:00:22, complete 98.34%
att-weights epoch 481, step 647, max_size:classes 8, max_size:data 344, mem_usage:GPU:0 1.0GB, num_seqs 11, 17.069 sec/step, elapsed 0:22:03, exp. remaining 0:00:15, complete 98.85%
att-weights epoch 481, step 648, max_size:classes 10, max_size:data 277, mem_usage:GPU:0 1.0GB, num_seqs 11, 1.005 sec/step, elapsed 0:22:04, exp. remaining 0:00:08, complete 99.33%
att-weights epoch 481, step 649, max_size:classes 10, max_size:data 362, mem_usage:GPU:0 1.0GB, num_seqs 11, 0.953 sec/step, elapsed 0:22:05, exp. remaining 0:00:02, complete 99.82%
att-weights epoch 481, step 650, max_size:classes 8, max_size:data 331, mem_usage:GPU:0 1.0GB, num_seqs 12, 0.998 sec/step, elapsed 0:22:06, exp. remaining 0:00:02, complete 99.82%
att-weights epoch 481, step 651, max_size:classes 7, max_size:data 366, mem_usage:GPU:0 1.0GB, num_seqs 10, 0.768 sec/step, elapsed 0:22:07, exp. remaining 0:00:02, complete 99.82%
att-weights epoch 481, step 652, max_size:classes 11, max_size:data 289, mem_usage:GPU:0 1.0GB, num_seqs 13, 1.073 sec/step, elapsed 0:22:08, exp. remaining 0:00:02, complete 99.82%
att-weights epoch 481, step 653, max_size:classes 11, max_size:data 292, mem_usage:GPU:0 1.0GB, num_seqs 12, 1.042 sec/step, elapsed 0:22:09, exp. remaining 0:00:02, complete 99.82%
att-weights epoch 481, step 654, max_size:classes 11, max_size:data 329, mem_usage:GPU:0 1.0GB, num_seqs 12, 1.223 sec/step, elapsed 0:22:10, exp. remaining 0:00:02, complete 99.82%
att-weights epoch 481, step 655, max_size:classes 8, max_size:data 288, mem_usage:GPU:0 1.0GB, num_seqs 13, 0.979 sec/step, elapsed 0:22:11, exp. remaining 0:00:02, complete 99.82%
att-weights epoch 481, step 656, max_size:classes 8, max_size:data 281, mem_usage:GPU:0 1.0GB, num_seqs 14, 1.237 sec/step, elapsed 0:22:13, exp. remaining 0:00:02, complete 99.82%
att-weights epoch 481, step 657, max_size:classes 6, max_size:data 304, mem_usage:GPU:0 1.0GB, num_seqs 13, 1.076 sec/step, elapsed 0:22:14, exp. remaining 0:00:02, complete 99.82%
att-weights epoch 481, step 658, max_size:classes 4, max_size:data 288, mem_usage:GPU:0 1.0GB, num_seqs 13, 0.856 sec/step, elapsed 0:22:14, exp. remaining 0:00:02, complete 99.82%
att-weights epoch 481, step 659, max_size:classes 4, max_size:data 178, mem_usage:GPU:0 1.0GB, num_seqs 6, 0.381 sec/step, elapsed 0:22:15, exp. remaining 0:00:02, complete 99.82%
Stats:
  mem_usage:GPU:0: Stats(mean=1.0GB, std_dev=0.0B, min=1.0GB, max=1.0GB, num_seqs=660, avg_data_len=1)
att-weights epoch 481, finished after 660 steps, 0:22:15 elapsed (25.8% computing time)
Layer 'dec_02_att_weights' Stats:
  2703 seqs, 102444704 total frames, 37900.371439 average frames
  Mean: 0.00553859768854618
  Std dev: 0.040394714599241426
  Min/max: 0.0 / 1.0
Layer 'dec_01_att_weights' Stats:
  2703 seqs, 102444704 total frames, 37900.371439 average frames
  Mean: 0.0055385976792928756
  Std dev: 0.07206521331300893
  Min/max: 0.0 / 1.0
Layer 'dec_03_att_weights' Stats:
  2703 seqs, 102444704 total frames, 37900.371439 average frames
  Mean: 0.005538597689868077
  Std dev: 0.04681535810279359
  Min/max: 0.0 / 0.9999988
Layer 'dec_04_att_weights' Stats:
  2703 seqs, 102444704 total frames, 37900.371439 average frames
  Mean: 0.005538597684803908
  Std dev: 0.04357248117472374
  Min/max: 0.0 / 0.99999595
Layer 'dec_05_att_weights' Stats:
  2703 seqs, 102444704 total frames, 37900.371439 average frames
  Mean: 0.0055385976811174725
  Std dev: 0.07047108981855361
  Min/max: 0.0 / 1.0
Layer 'dec_06_att_weights' Stats:
  2703 seqs, 102444704 total frames, 37900.371439 average frames
  Mean: 0.005538597675457502
  Std dev: 0.06587331733406884
  Min/max: 0.0 / 1.0
Layer 'dec_07_att_weights' Stats:
  2703 seqs, 102444704 total frames, 37900.371439 average frames
  Mean: 0.005538597675215455
  Std dev: 0.0697083881583307
  Min/max: 0.0 / 1.0
Layer 'dec_08_att_weights' Stats:
  2703 seqs, 102444704 total frames, 37900.371439 average frames
  Mean: 0.005538597670169906
  Std dev: 0.06865581450004055
  Min/max: 0.0 / 1.0
Layer 'dec_09_att_weights' Stats:
  2703 seqs, 102444704 total frames, 37900.371439 average frames
  Mean: 0.005538597684748039
  Std dev: 0.07099630803953694
  Min/max: 0.0 / 1.0
Layer 'dec_10_att_weights' Stats:
  2703 seqs, 102444704 total frames, 37900.371439 average frames
  Mean: 0.005538597670840136
  Std dev: 0.06943524531046363
  Min/max: 0.0 / 1.0
Layer 'dec_11_att_weights' Stats:
  2703 seqs, 102444704 total frames, 37900.371439 average frames
  Mean: 0.005538597662424684
  Std dev: 0.06797576060331503
  Min/max: 0.0 / 1.0
Layer 'dec_12_att_weights' Stats:
  2703 seqs, 102444704 total frames, 37900.371439 average frames
  Mean: 0.005538597685027315
  Std dev: 0.0670860914104262
  Min/max: 0.0 / 1.0
Quitting
+------- EPILOGUE SCRIPT -----------------------------------------------
|
| Job ID ..............: 9505992
| Stopped at ..........: Tue Jul  2 12:34:58 CEST 2019
| Resources requested .: gpu=1,h_rt=7200,h_rss=4G,h_vmem=1536G,s_core=0,pxe=ubuntu_16.04,num_proc=5,h_fsize=20G,scratch_free=5G
| Resources used ......: cpu=00:36:05, mem=8170.56604 GB s, io=11.44690 GB, vmem=4.049G, maxvmem=4.073G, last_file_cache=339M, last_rss=3M, max-cache=3.669G
| Memory used .........: 4.000G / 4.000G (100.0%)
| Total time used .....: 0:24:19
|
+------- EPILOGUE SCRIPT -----------------------------------------------
