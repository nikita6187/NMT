+------- PROLOGUE SCRIPT -----------------------------------------------
|
| Job ID ...........: 9519751
| Started at .......: Fri Jul  5 14:02:45 CEST 2019
| Execution host ...: cluster-cn-259
| Cluster queue ....: 4-GPU-1080
| Script ...........: /var/spool/sge/cluster-cn-259/job_scripts/9519751
| > #!/bin/bash
| > 
| > export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64:/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda/extras/CUPTI/lib64/:/usr/local/cuda-9.1/extras/CUPTI/lib64/:/usr/local/cuda-9.0/extras/CUPTI/lib64:/usr/lib/nvidia-418
| > #export LD_LIBRARY_PATH="/usr/local/lib::/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda-9.0/lib64:/usr/lib/nvidia-390:/opt/rbi/sge/lib/glinux:/opt/rbi/sge/lib/glinux"
| > 
| > source /u/bahar/settings/python3-returnn-tf1.9/bin/activate
| > 
| > python3 ~/returnn-parnia-2/tools/get-attention-weights.py /u/bahar/workspace/asr/librispeech/test-20190121/config-train/base2.smlp2.specaug.datarndperm_noscale.bs18k.curric3.retrain1.config --epoch 183 --data 'config:get_dataset("'$1'")' --dump_dir ~/makarov/NMT/hmm-factorization/experiments/asr2019/lstm-prior/forward-$1 --layers "att_weights" --rec_layer "output" --batch_size 4000 --tf_log_dir /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/lstm-prior/forward-$1/tf_log_dir --reset_seq_ordering sorted_reverse
|
+------- PROLOGUE SCRIPT -----------------------------------------------
RETURNN get-attention-weights starting up.
RETURNN starting up, version unknown(git exception: CalledProcessError(128, ('git', 'show', '-s', '--format=%ci', 'HEAD'))), date/time 2019-07-05-14-02-47 (UTC+0200), pid 1898, cwd /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/exp3.ctc.ogg-priorAtt-lr0005-k10-specAug, Python /u/bahar/settings/python3-returnn-tf1.9/bin/python3
RETURNN config: /u/bahar/workspace/asr/librispeech/test-20190121/config-train/base2.smlp2.specaug.datarndperm_noscale.bs18k.curric3.retrain1.config
RETURNN command line options: ()
Hostname: cluster-cn-259
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
       incarnation: 9237638296877052713
  2/2: name: "/device:GPU:0"
       device_type: "GPU"
       memory_limit: 10911236096
       locality {
         bus_id: 1
         links {
         }
       }
       incarnation: 13806442587739918680
       physical_device_desc: "device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1"
Using gpu device 0: GeForce GTX 1080 Ti
Device not set explicitly, and we found a GPU, which we will use.
Model file prefix: net-model/network
NOTE: We will use 'sorted_reverse' seq ordering.
Setup tf.Session with options {'device_count': {'GPU': 1}, 'log_device_placement': False} ...
layer root/'data' output: Data(name='data', shape=(None, 40))
layer root/'source' output: Data(name='source_output', shape=(None, 40))
Exception creating layer root/'source' of class EvalLayer with opts:
{'eval': "self.network.get_config().typed_value('transform')(source(0), "
         'network=self.network)',
 'name': 'source',
 'network': <TFNetwork 'root' train=<tf.Tensor 'globals/train_flag:0' shape=() dtype=bool>>,
 'output': Data(name='source_output', shape=(None, 40)),
 'sources': [<SourceLayer 'data' out_type=Data(shape=(None, 40))>]}
Unhandled exception <class 'ImportError'> in thread <_MainThread(MainThread, started 47072171211776)>, proc 1898.

Thread current, main, <_MainThread(MainThread, started 47072171211776)>:
(Excluded thread.)

That were all threads.
+------- EPILOGUE SCRIPT -----------------------------------------------
|
| Job ID ..............: 9519751
| Stopped at ..........: Fri Jul  5 14:03:06 CEST 2019
| Resources requested .: scratch_free=5G,h_fsize=20G,num_proc=5,s_core=0,h_vmem=1536G,pxe=ubuntu_16.04,h_rss=8G,h_rt=7200,gpu=1
| Resources used ......: cpu=00:00:12, mem=7.27158 GB s, io=0.02933 GB, vmem=777.848M, maxvmem=777.848M, last_file_cache=636K, last_rss=2M, max-cache=752M
| Memory used .........: 752M / 8.000G (9.2%)
| Total time used .....: 0:00:22
|
+------- EPILOGUE SCRIPT -----------------------------------------------
+------- PROLOGUE SCRIPT -----------------------------------------------
|
| Job ID ...........: 9519817
| Started at .......: Fri Jul  5 14:20:57 CEST 2019
| Execution host ...: cluster-cn-235
| Cluster queue ....: 4-GPU-1080
| Script ...........: /var/spool/sge/cluster-cn-235/job_scripts/9519817
| > #!/bin/bash
| > 
| > export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64:/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda/extras/CUPTI/lib64/:/usr/local/cuda-9.1/extras/CUPTI/lib64/:/usr/local/cuda-9.0/extras/CUPTI/lib64:/usr/lib/nvidia-418
| > #export LD_LIBRARY_PATH="/usr/local/lib::/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda-9.0/lib64:/usr/lib/nvidia-390:/opt/rbi/sge/lib/glinux:/opt/rbi/sge/lib/glinux"
| > 
| > source /u/bahar/settings/python3-returnn-tf1.9/bin/activate
| > 
| > python3 ~/returnn-parnia-2/tools/get-attention-weights.py /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/lstm-prior/exp3.ctc.ogg-priorAtt-lr0005-k10-specAug.config --epoch 183 --data 'config:get_dataset("'$1'")' --dump_dir ~/makarov/NMT/hmm-factorization/experiments/asr2019/lstm-prior/forward-$1 --layers "att_weights" --rec_layer "output" --batch_size 4000 --tf_log_dir /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/lstm-prior/forward-$1/tf_log_dir --reset_seq_ordering sorted_reverse
|
+------- PROLOGUE SCRIPT -----------------------------------------------
RETURNN get-attention-weights starting up.
RETURNN starting up, version unknown(git exception: CalledProcessError(128, ('git', 'show', '-s', '--format=%ci', 'HEAD'))), date/time 2019-07-05-14-21-00 (UTC+0200), pid 2113, cwd /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/exp3.ctc.ogg-priorAtt-lr0005-k10-specAug, Python /u/bahar/settings/python3-returnn-tf1.9/bin/python3
RETURNN config: /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/lstm-prior/exp3.ctc.ogg-priorAtt-lr0005-k10-specAug.config
RETURNN command line options: ()
Hostname: cluster-cn-235
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
       incarnation: 17389062351836499180
  2/2: name: "/device:GPU:0"
       device_type: "GPU"
       memory_limit: 10911236096
       locality {
         bus_id: 2
         numa_node: 1
         links {
         }
       }
       incarnation: 3024653634620227825
       physical_device_desc: "device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:81:00.0, compute capability: 6.1"
Using gpu device 2: GeForce GTX 1080 Ti
Device not set explicitly, and we found a GPU, which we will use.
Model file prefix: net-model/network
NOTE: We will use 'sorted_reverse' seq ordering.
Setup tf.Session with options {'log_device_placement': False, 'device_count': {'GPU': 1}} ...
layer root/'data' output: Data(name='data', shape=(None, 40))
layer root/'source' output: Data(name='source_output', shape=(None, 40))
Exception creating layer root/'source' of class EvalLayer with opts:
{'eval': "self.network.get_config().typed_value('transform')(source(0), "
         'network=self.network)',
 'name': 'source',
 'network': <TFNetwork 'root' train=<tf.Tensor 'globals/train_flag:0' shape=() dtype=bool>>,
 'output': Data(name='source_output', shape=(None, 40)),
 'sources': [<SourceLayer 'data' out_type=Data(shape=(None, 40))>]}
Unhandled exception <class 'ImportError'> in thread <_MainThread(MainThread, started 47968628804608)>, proc 2113.

Thread current, main, <_MainThread(MainThread, started 47968628804608)>:
(Excluded thread.)

That were all threads.
+------- EPILOGUE SCRIPT -----------------------------------------------
|
| Job ID ..............: 9519817
| Stopped at ..........: Fri Jul  5 14:21:06 CEST 2019
| Resources requested .: h_rss=8G,h_rt=7200,gpu=1,s_core=0,pxe=ubuntu_16.04,h_vmem=1536G,num_proc=5,scratch_free=5G,h_fsize=20G
| Resources used ......: cpu=00:00:00, mem=0.00000 GB s, io=0.00000 GB, vmem=N/A, maxvmem=N/A, last_file_cache=5M, last_rss=2M, max-cache=744M
| Memory used .........: 749M / 8.000G (9.1%)
| Total time used .....: 0:00:10
|
+------- EPILOGUE SCRIPT -----------------------------------------------
+------- PROLOGUE SCRIPT -----------------------------------------------
|
| Job ID ...........: 9519819
| Started at .......: Fri Jul  5 14:23:27 CEST 2019
| Execution host ...: cluster-cn-280
| Cluster queue ....: 4-GPU-1080-128G
| Script ...........: /var/spool/sge/cluster-cn-280/job_scripts/9519819
| > #!/bin/bash
| > 
| > export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64:/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda/extras/CUPTI/lib64/:/usr/local/cuda-9.1/extras/CUPTI/lib64/:/usr/local/cuda-9.0/extras/CUPTI/lib64:/usr/lib/nvidia-418
| > #export LD_LIBRARY_PATH="/usr/local/lib::/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda-9.0/lib64:/usr/lib/nvidia-390:/opt/rbi/sge/lib/glinux:/opt/rbi/sge/lib/glinux"
| > 
| > source /u/bahar/settings/python3-returnn-tf1.9/bin/activate
| > 
| > python3 ~/returnn-parnia-2/tools/get-attention-weights.py /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/lstm-prior/exp3.ctc.ogg-priorAtt-lr0005-k10-specAug.config --epoch 183 --data 'config:get_dataset("'$1'")' --dump_dir ~/makarov/NMT/hmm-factorization/experiments/asr2019/lstm-prior/forward-$1 --layers "att_weights" --rec_layer "output" --batch_size 4000 --tf_log_dir /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/lstm-prior/forward-$1/tf_log_dir --reset_seq_ordering sorted_reverse
|
+------- PROLOGUE SCRIPT -----------------------------------------------
RETURNN get-attention-weights starting up.
RETURNN starting up, version unknown(git exception: CalledProcessError(128, ('git', 'show', '-s', '--format=%ci', 'HEAD'))), date/time 2019-07-05-14-23-29 (UTC+0200), pid 3795, cwd /work/smt3/bahar/expriments/asr/librispeech/test/20190121-end2end-hmm/data-train/exp3.ctc.ogg-priorAtt-lr0005-k10-specAug, Python /u/bahar/settings/python3-returnn-tf1.9/bin/python3
RETURNN config: /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/lstm-prior/exp3.ctc.ogg-priorAtt-lr0005-k10-specAug.config
RETURNN command line options: ()
Hostname: cluster-cn-280
TensorFlow: 1.10.0 (v1.10.0-0-g656e7a2b34) (<site-package> in /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow)
Error while getting SGE num_proc: FileNotFoundError(2, "No such file or directory: 'qstat'")
Setup TF inter and intra global thread pools, num_threads None, session opts {'log_device_placement': False, 'device_count': {'GPU': 0}}.
CUDA_VISIBLE_DEVICES is set to '3'.
Collecting TensorFlow device list...
Local devices available to TensorFlow:
  1/2: name: "/device:CPU:0"
       device_type: "CPU"
       memory_limit: 268435456
       locality {
       }
       incarnation: 11424345951664434414
  2/2: name: "/device:GPU:0"
       device_type: "GPU"
       memory_limit: 10911236096
       locality {
         bus_id: 2
         numa_node: 1
         links {
         }
       }
       incarnation: 8264220052033890168
       physical_device_desc: "device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:82:00.0, compute capability: 6.1"
Using gpu device 3: GeForce GTX 1080 Ti
Device not set explicitly, and we found a GPU, which we will use.
Model file prefix: net-model/network
NOTE: We will use 'sorted_reverse' seq ordering.
Setup tf.Session with options {'log_device_placement': False, 'device_count': {'GPU': 1}} ...
layer root/'data' output: Data(name='data', shape=(None, 40))
layer root/'source' output: Data(name='source_output', shape=(None, 40))
layer root/'lstm0_fw' output: Data(name='lstm0_fw_output', shape=(None, 1024), batch_dim_axis=1)
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9519819.1.4-GPU-1080-128G/makarov/returnn_tf_cache/ops/NativeLstm2/807b76c5bb/NativeLstm2.cc -o /var/tmp/9519819.1.4-GPU-1080-128G/makarov/returnn_tf_cache/ops/NativeLstm2/807b76c5bb/NativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
OpCodeCompiler call: /usr/local/cuda-9.0/bin/nvcc -shared -O2 -std=c++11 -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include -I /u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-9.0/include -L /usr/local/cuda-9.0/lib64 -x cu -v -DGOOGLE_CUDA=1 -Xcompiler -fPIC -Xcompiler -v -arch compute_61 -D_GLIBCXX_USE_CXX11_ABI=0 -g /var/tmp/9519819.1.4-GPU-1080-128G/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef93791ddb/GradOfNativeLstm2.cc -o /var/tmp/9519819.1.4-GPU-1080-128G/makarov/returnn_tf_cache/ops/GradOfNativeLstm2/ef93791ddb/GradOfNativeLstm2.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/numpy/.libs -l:libopenblasp-r0-39a31c03.2.18.so -L/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
layer root/'lstm0_bw' output: Data(name='lstm0_bw_output', shape=(None, 1024), batch_dim_axis=1)
layer root/'lstm0_pool' output: Data(name='lstm0_pool_output', shape=(None, 2048))
layer root/'lstm1_fw' output: Data(name='lstm1_fw_output', shape=(None, 1024), batch_dim_axis=1)
layer root/'lstm1_bw' output: Data(name='lstm1_bw_output', shape=(None, 1024), batch_dim_axis=1)
layer root/'lstm1_pool' output: Data(name='lstm1_pool_output', shape=(None, 2048))
layer root/'lstm2_fw' output: Data(name='lstm2_fw_output', shape=(None, 1024), batch_dim_axis=1)
layer root/'lstm2_bw' output: Data(name='lstm2_bw_output', shape=(None, 1024), batch_dim_axis=1)
layer root/'lstm2_pool' output: Data(name='lstm2_pool_output', shape=(None, 2048), batch_dim_axis=1)
layer root/'lstm3_fw' output: Data(name='lstm3_fw_output', shape=(None, 1024), batch_dim_axis=1)
layer root/'lstm3_bw' output: Data(name='lstm3_bw_output', shape=(None, 1024), batch_dim_axis=1)
layer root/'lstm3_pool' output: Data(name='lstm3_pool_output', shape=(None, 2048), batch_dim_axis=1)
layer root/'lstm4_fw' output: Data(name='lstm4_fw_output', shape=(None, 1024), batch_dim_axis=1)
layer root/'lstm4_bw' output: Data(name='lstm4_bw_output', shape=(None, 1024), batch_dim_axis=1)
layer root/'lstm4_pool' output: Data(name='lstm4_pool_output', shape=(None, 2048), batch_dim_axis=1)
layer root/'lstm5_fw' output: Data(name='lstm5_fw_output', shape=(None, 1024), batch_dim_axis=1)
layer root/'lstm5_bw' output: Data(name='lstm5_bw_output', shape=(None, 1024), batch_dim_axis=1)
layer root/'encoder' output: Data(name='encoder_output', shape=(None, 2048), batch_dim_axis=1)
layer root/'ctc' output: Data(name='ctc_output', shape=(None, 10026), batch_dim_axis=1)
layer root/'enc_ctx' output: Data(name='enc_ctx_output', shape=(None, 1024), batch_dim_axis=1)
layer root/'enc_value' output: Data(name='enc_value_output', shape=(None, 1, 2048), batch_dim_axis=1)
layer root/'inv_fertility' output: Data(name='inv_fertility_output', shape=(None, 1), batch_dim_axis=1)
layer root/'output' output: Data(name='output_output', shape=(None,), dtype='int32', sparse=True, dim=10025, batch_dim_axis=1)
Rec layer sub net:
  Input layers moved out of loop: (#: 2)
    output
    target_embed
  Output layers moved out of loop: (#: 3)
    output_prob
    base_encoder
    prev_outputs_transformed
  Layers in loop: (#: 10)
    att_weights
    energy
    energy_tanh
    energy_in
    s_transformed
    s
    att
    att0
    weight_feedback
    accum_att_weights
  Unused layers: (#: 2)
    end
    readout_in
layer root/output:rec-subnet-input/'output' output: Data(name='classes', shape=(None,), dtype='int32', sparse=True, dim=10025)
layer root/output:rec-subnet-input/'target_embed' output: Data(name='target_embed_output', shape=(None, 621))
layer root/output:rec-subnet/'weight_feedback' output: Data(name='weight_feedback_output', shape=(None, 1024), time_dim_axis=None)
layer root/output:rec-subnet/'prev:target_embed' output: Data(name='target_embed_output', shape=(621,), time_dim_axis=None)
layer root/output:rec-subnet/'s' output: Data(name='s_output', shape=(1000,), time_dim_axis=None)
layer root/output:rec-subnet/'s_transformed' output: Data(name='s_transformed_output', shape=(1024,), time_dim_axis=None)
layer root/output:rec-subnet/'energy_in' output: Data(name='energy_in_output', shape=(None, 1024), batch_dim_axis=1)
layer root/output:rec-subnet/'energy_tanh' output: Data(name='energy_tanh_output', shape=(None, 1024), batch_dim_axis=1)
layer root/output:rec-subnet/'energy' output: Data(name='energy_output', shape=(None, 1), batch_dim_axis=1)
layer root/output:rec-subnet/'att_weights' output: Data(name='att_weights_output', shape=(None, 1), batch_dim_axis=1)
layer root/output:rec-subnet/'accum_att_weights' output: Data(name='accum_att_weights_output', shape=(None, 1), time_dim_axis=None)
layer root/output:rec-subnet/'att0' output: Data(name='att0_output', shape=(1, 2048), time_dim_axis=None)
layer root/output:rec-subnet/'att' output: Data(name='att_output', shape=(2048,), time_dim_axis=None)
layer root/output:rec-subnet-output/'att_weights' output: Data(name='att_weights_output', shape=(None, None, 1), batch_dim_axis=2)
layer root/output:rec-subnet-output/'base_encoder' output: Data(name='base_encoder_output', shape=(None, 1024), batch_dim_axis=1)
layer root/output:rec-subnet-output/'s_transformed' output: Data(name='s_transformed_output', shape=(None, 1024), batch_dim_axis=1)
layer root/output:rec-subnet-output/'prev:target_embed' output: Data(name='target_embed_output', shape=(None, 621), batch_dim_axis=1)
layer root/output:rec-subnet-output/'prev_outputs_transformed' output: Data(name='prev_outputs_transformed_output', shape=(None, 1024), batch_dim_axis=1)
layer root/output:rec-subnet-output/'output_prob' output: Data(name='att_weights_output', shape=(None, 10025), batch_dim_axis=1)
layer root/'decision' output: Data(name='output_output', shape=(None,), dtype='int32', sparse=True, dim=10025, batch_dim_axis=1)
Warning: using numerical unstable sparse Cross-Entropy loss calculation
Network layer topology:
  extern data: classes: Data(shape=(None,), dtype='int32', sparse=True, dim=10025, available_for_inference=False), data: Data(shape=(None, 40))
  used data keys: ['classes', 'data']
  layer softmax 'ctc' #: 10026
  layer source 'data' #: 40
  layer decide 'decision' #: 10025
  layer linear 'enc_ctx' #: 1024
  layer split_dims 'enc_value' #: 2048
  layer copy 'encoder' #: 2048
  layer linear 'inv_fertility' #: 1
  layer rec 'lstm0_bw' #: 1024
  layer rec 'lstm0_fw' #: 1024
  layer pool 'lstm0_pool' #: 2048
  layer rec 'lstm1_bw' #: 1024
  layer rec 'lstm1_fw' #: 1024
  layer pool 'lstm1_pool' #: 2048
  layer rec 'lstm2_bw' #: 1024
  layer rec 'lstm2_fw' #: 1024
  layer pool 'lstm2_pool' #: 2048
  layer rec 'lstm3_bw' #: 1024
  layer rec 'lstm3_fw' #: 1024
  layer pool 'lstm3_pool' #: 2048
  layer rec 'lstm4_bw' #: 1024
  layer rec 'lstm4_fw' #: 1024
  layer pool 'lstm4_pool' #: 2048
  layer rec 'lstm5_bw' #: 1024
  layer rec 'lstm5_fw' #: 1024
  layer rec 'output' #: 10025
  layer eval 'source' #: 40
net params #: 191119711
net trainable params: [<tf.Variable 'ctc/W:0' shape=(2048, 10026) dtype=float32_ref>, <tf.Variable 'ctc/b:0' shape=(10026,) dtype=float32_ref>, <tf.Variable 'enc_ctx/W:0' shape=(2048, 1024) dtype=float32_ref>, <tf.Variable 'enc_ctx/b:0' shape=(1024,) dtype=float32_ref>, <tf.Variable 'inv_fertility/W:0' shape=(2048, 1) dtype=float32_ref>, <tf.Variable 'lstm0_bw/rec/W:0' shape=(40, 4096) dtype=float32_ref>, <tf.Variable 'lstm0_bw/rec/W_re:0' shape=(1024, 4096) dtype=float32_ref>, <tf.Variable 'lstm0_bw/rec/b:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'lstm0_fw/rec/W:0' shape=(40, 4096) dtype=float32_ref>, <tf.Variable 'lstm0_fw/rec/W_re:0' shape=(1024, 4096) dtype=float32_ref>, <tf.Variable 'lstm0_fw/rec/b:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'lstm1_bw/rec/W:0' shape=(2048, 4096) dtype=float32_ref>, <tf.Variable 'lstm1_bw/rec/W_re:0' shape=(1024, 4096) dtype=float32_ref>, <tf.Variable 'lstm1_bw/rec/b:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'lstm1_fw/rec/W:0' shape=(2048, 4096) dtype=float32_ref>, <tf.Variable 'lstm1_fw/rec/W_re:0' shape=(1024, 4096) dtype=float32_ref>, <tf.Variable 'lstm1_fw/rec/b:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'lstm2_bw/rec/W:0' shape=(2048, 4096) dtype=float32_ref>, <tf.Variable 'lstm2_bw/rec/W_re:0' shape=(1024, 4096) dtype=float32_ref>, <tf.Variable 'lstm2_bw/rec/b:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'lstm2_fw/rec/W:0' shape=(2048, 4096) dtype=float32_ref>, <tf.Variable 'lstm2_fw/rec/W_re:0' shape=(1024, 4096) dtype=float32_ref>, <tf.Variable 'lstm2_fw/rec/b:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'lstm3_bw/rec/W:0' shape=(2048, 4096) dtype=float32_ref>, <tf.Variable 'lstm3_bw/rec/W_re:0' shape=(1024, 4096) dtype=float32_ref>, <tf.Variable 'lstm3_bw/rec/b:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'lstm3_fw/rec/W:0' shape=(2048, 4096) dtype=float32_ref>, <tf.Variable 'lstm3_fw/rec/W_re:0' shape=(1024, 4096) dtype=float32_ref>, <tf.Variable 'lstm3_fw/rec/b:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'lstm4_bw/rec/W:0' shape=(2048, 4096) dtype=float32_ref>, <tf.Variable 'lstm4_bw/rec/W_re:0' shape=(1024, 4096) dtype=float32_ref>, <tf.Variable 'lstm4_bw/rec/b:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'lstm4_fw/rec/W:0' shape=(2048, 4096) dtype=float32_ref>, <tf.Variable 'lstm4_fw/rec/W_re:0' shape=(1024, 4096) dtype=float32_ref>, <tf.Variable 'lstm4_fw/rec/b:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'lstm5_bw/rec/W:0' shape=(2048, 4096) dtype=float32_ref>, <tf.Variable 'lstm5_bw/rec/W_re:0' shape=(1024, 4096) dtype=float32_ref>, <tf.Variable 'lstm5_bw/rec/b:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'lstm5_fw/rec/W:0' shape=(2048, 4096) dtype=float32_ref>, <tf.Variable 'lstm5_fw/rec/W_re:0' shape=(1024, 4096) dtype=float32_ref>, <tf.Variable 'lstm5_fw/rec/b:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'output/rec/base_encoder/W:0' shape=(1024, 1024) dtype=float32_ref>, <tf.Variable 'output/rec/energy/W:0' shape=(1024, 1) dtype=float32_ref>, <tf.Variable 'output/rec/output_prob/dense/kernel:0' shape=(1024, 10025) dtype=float32_ref>, <tf.Variable 'output/rec/prev_outputs_transformed/W:0' shape=(621, 1024) dtype=float32_ref>, <tf.Variable 'output/rec/s/rec/lstm_cell/bias:0' shape=(4000,) dtype=float32_ref>, <tf.Variable 'output/rec/s/rec/lstm_cell/kernel:0' shape=(3669, 4000) dtype=float32_ref>, <tf.Variable 'output/rec/s_transformed/W:0' shape=(1000, 1024) dtype=float32_ref>, <tf.Variable 'output/rec/target_embed/W:0' shape=(10025, 621) dtype=float32_ref>, <tf.Variable 'output/rec/weight_feedback/W:0' shape=(1, 1024) dtype=float32_ref>]
loading weights from net-model/network.183
TF: log_dir: /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/lstm-prior/forward-dev-other/tf_log_dir/prefix:dev-other-183-2019-07-05-12-23-28
Note: There are still these uninitialized variables: ['learning_rate:0']
att-weights epoch 183, step 0, max_size:classes 94, max_size:data 3206, mem_usage:GPU:0 1.4GB, num_seqs 1, 2.520 sec/step, elapsed 0:00:03, exp. remaining 0:17:07, complete 0.38%
att-weights epoch 183, step 1, max_size:classes 93, max_size:data 3516, mem_usage:GPU:0 1.4GB, num_seqs 1, 1.036 sec/step, elapsed 0:00:05, exp. remaining 0:20:05, complete 0.42%
att-weights epoch 183, step 2, max_size:classes 83, max_size:data 3322, mem_usage:GPU:0 1.4GB, num_seqs 1, 1.005 sec/step, elapsed 0:00:06, exp. remaining 0:22:24, complete 0.45%
att-weights epoch 183, step 3, max_size:classes 90, max_size:data 2413, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.881 sec/step, elapsed 0:00:07, exp. remaining 0:23:56, complete 0.49%
att-weights epoch 183, step 4, max_size:classes 79, max_size:data 2611, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.778 sec/step, elapsed 0:00:07, exp. remaining 0:25:05, complete 0.52%
att-weights epoch 183, step 5, max_size:classes 84, max_size:data 3350, mem_usage:GPU:0 1.4GB, num_seqs 1, 1.055 sec/step, elapsed 0:00:09, exp. remaining 0:26:49, complete 0.56%
att-weights epoch 183, step 6, max_size:classes 78, max_size:data 2592, mem_usage:GPU:0 1.4GB, num_seqs 1, 1.347 sec/step, elapsed 0:00:10, exp. remaining 0:29:05, complete 0.59%
att-weights epoch 183, step 7, max_size:classes 80, max_size:data 2232, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.670 sec/step, elapsed 0:00:11, exp. remaining 0:29:19, complete 0.63%
att-weights epoch 183, step 8, max_size:classes 83, max_size:data 2463, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.832 sec/step, elapsed 0:00:12, exp. remaining 0:29:57, complete 0.66%
att-weights epoch 183, step 9, max_size:classes 89, max_size:data 2577, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.789 sec/step, elapsed 0:00:12, exp. remaining 0:30:24, complete 0.70%
att-weights epoch 183, step 10, max_size:classes 92, max_size:data 2618, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.843 sec/step, elapsed 0:00:13, exp. remaining 0:30:57, complete 0.73%
att-weights epoch 183, step 11, max_size:classes 86, max_size:data 3177, mem_usage:GPU:0 1.4GB, num_seqs 1, 1.052 sec/step, elapsed 0:00:14, exp. remaining 0:31:53, complete 0.77%
att-weights epoch 183, step 12, max_size:classes 82, max_size:data 2822, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.842 sec/step, elapsed 0:00:15, exp. remaining 0:32:13, complete 0.80%
att-weights epoch 183, step 13, max_size:classes 73, max_size:data 2951, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.924 sec/step, elapsed 0:00:16, exp. remaining 0:32:41, complete 0.84%
att-weights epoch 183, step 14, max_size:classes 72, max_size:data 2463, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.753 sec/step, elapsed 0:00:17, exp. remaining 0:32:48, complete 0.87%
att-weights epoch 183, step 15, max_size:classes 76, max_size:data 2211, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.719 sec/step, elapsed 0:00:18, exp. remaining 0:32:50, complete 0.91%
att-weights epoch 183, step 16, max_size:classes 78, max_size:data 2964, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.902 sec/step, elapsed 0:00:18, exp. remaining 0:33:11, complete 0.94%
att-weights epoch 183, step 17, max_size:classes 76, max_size:data 2016, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.651 sec/step, elapsed 0:00:19, exp. remaining 0:33:05, complete 0.98%
att-weights epoch 183, step 18, max_size:classes 87, max_size:data 2598, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.837 sec/step, elapsed 0:00:20, exp. remaining 0:33:18, complete 1.01%
att-weights epoch 183, step 19, max_size:classes 74, max_size:data 2578, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.782 sec/step, elapsed 0:00:21, exp. remaining 0:33:24, complete 1.05%
att-weights epoch 183, step 20, max_size:classes 110, max_size:data 2909, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.894 sec/step, elapsed 0:00:22, exp. remaining 0:33:41, complete 1.08%
att-weights epoch 183, step 21, max_size:classes 70, max_size:data 1753, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.615 sec/step, elapsed 0:00:22, exp. remaining 0:33:31, complete 1.12%
att-weights epoch 183, step 22, max_size:classes 77, max_size:data 2778, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.826 sec/step, elapsed 0:00:23, exp. remaining 0:33:41, complete 1.15%
att-weights epoch 183, step 23, max_size:classes 83, max_size:data 2699, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.865 sec/step, elapsed 0:00:24, exp. remaining 0:33:53, complete 1.19%
att-weights epoch 183, step 24, max_size:classes 63, max_size:data 3014, mem_usage:GPU:0 1.4GB, num_seqs 1, 1.079 sec/step, elapsed 0:00:25, exp. remaining 0:34:21, complete 1.22%
att-weights epoch 183, step 25, max_size:classes 63, max_size:data 3232, mem_usage:GPU:0 1.4GB, num_seqs 1, 1.061 sec/step, elapsed 0:00:26, exp. remaining 0:33:49, complete 1.29%
att-weights epoch 183, step 26, max_size:classes 62, max_size:data 1935, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.713 sec/step, elapsed 0:00:27, exp. remaining 0:32:56, complete 1.36%
att-weights epoch 183, step 27, max_size:classes 69, max_size:data 2024, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.646 sec/step, elapsed 0:00:27, exp. remaining 0:32:02, complete 1.43%
att-weights epoch 183, step 28, max_size:classes 66, max_size:data 1976, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.671 sec/step, elapsed 0:00:28, exp. remaining 0:32:01, complete 1.47%
att-weights epoch 183, step 29, max_size:classes 69, max_size:data 2385, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.781 sec/step, elapsed 0:00:29, exp. remaining 0:32:07, complete 1.50%
att-weights epoch 183, step 30, max_size:classes 58, max_size:data 2818, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.939 sec/step, elapsed 0:00:30, exp. remaining 0:32:23, complete 1.54%
att-weights epoch 183, step 31, max_size:classes 60, max_size:data 1849, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.633 sec/step, elapsed 0:00:30, exp. remaining 0:32:19, complete 1.57%
att-weights epoch 183, step 32, max_size:classes 67, max_size:data 2141, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.823 sec/step, elapsed 0:00:31, exp. remaining 0:32:26, complete 1.61%
att-weights epoch 183, step 33, max_size:classes 61, max_size:data 2119, mem_usage:GPU:0 1.4GB, num_seqs 1, 1.067 sec/step, elapsed 0:00:32, exp. remaining 0:32:48, complete 1.64%
att-weights epoch 183, step 34, max_size:classes 64, max_size:data 1820, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.811 sec/step, elapsed 0:00:33, exp. remaining 0:32:13, complete 1.71%
att-weights epoch 183, step 35, max_size:classes 64, max_size:data 1807, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.752 sec/step, elapsed 0:00:34, exp. remaining 0:31:37, complete 1.78%
att-weights epoch 183, step 36, max_size:classes 66, max_size:data 1775, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.888 sec/step, elapsed 0:00:35, exp. remaining 0:31:12, complete 1.85%
att-weights epoch 183, step 37, max_size:classes 61, max_size:data 1991, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.772 sec/step, elapsed 0:00:36, exp. remaining 0:30:42, complete 1.92%
att-weights epoch 183, step 38, max_size:classes 63, max_size:data 2176, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.811 sec/step, elapsed 0:00:36, exp. remaining 0:30:16, complete 1.99%
att-weights epoch 183, step 39, max_size:classes 65, max_size:data 1634, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.675 sec/step, elapsed 0:00:37, exp. remaining 0:29:45, complete 2.06%
att-weights epoch 183, step 40, max_size:classes 58, max_size:data 2835, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.892 sec/step, elapsed 0:00:38, exp. remaining 0:29:26, complete 2.13%
att-weights epoch 183, step 41, max_size:classes 60, max_size:data 1838, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.598 sec/step, elapsed 0:00:39, exp. remaining 0:28:56, complete 2.20%
att-weights epoch 183, step 42, max_size:classes 68, max_size:data 2190, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.684 sec/step, elapsed 0:00:39, exp. remaining 0:28:30, complete 2.27%
att-weights epoch 183, step 43, max_size:classes 63, max_size:data 1632, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.803 sec/step, elapsed 0:00:40, exp. remaining 0:28:38, complete 2.30%
att-weights epoch 183, step 44, max_size:classes 59, max_size:data 1934, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.832 sec/step, elapsed 0:00:41, exp. remaining 0:28:46, complete 2.34%
att-weights epoch 183, step 45, max_size:classes 64, max_size:data 1987, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.817 sec/step, elapsed 0:00:42, exp. remaining 0:28:28, complete 2.41%
att-weights epoch 183, step 46, max_size:classes 62, max_size:data 1990, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.842 sec/step, elapsed 0:00:43, exp. remaining 0:28:12, complete 2.48%
att-weights epoch 183, step 47, max_size:classes 59, max_size:data 1627, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.690 sec/step, elapsed 0:00:43, exp. remaining 0:27:51, complete 2.55%
att-weights epoch 183, step 48, max_size:classes 58, max_size:data 1779, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.666 sec/step, elapsed 0:00:44, exp. remaining 0:27:30, complete 2.62%
att-weights epoch 183, step 49, max_size:classes 55, max_size:data 1813, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.695 sec/step, elapsed 0:00:45, exp. remaining 0:27:33, complete 2.65%
att-weights epoch 183, step 50, max_size:classes 58, max_size:data 1785, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.771 sec/step, elapsed 0:00:45, exp. remaining 0:27:39, complete 2.69%
att-weights epoch 183, step 51, max_size:classes 60, max_size:data 1824, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.845 sec/step, elapsed 0:00:46, exp. remaining 0:27:47, complete 2.72%
att-weights epoch 183, step 52, max_size:classes 57, max_size:data 1478, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.780 sec/step, elapsed 0:00:47, exp. remaining 0:27:32, complete 2.79%
att-weights epoch 183, step 53, max_size:classes 64, max_size:data 2023, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.659 sec/step, elapsed 0:00:48, exp. remaining 0:27:33, complete 2.83%
att-weights epoch 183, step 54, max_size:classes 62, max_size:data 1990, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.939 sec/step, elapsed 0:00:49, exp. remaining 0:27:24, complete 2.90%
att-weights epoch 183, step 55, max_size:classes 56, max_size:data 1850, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.396 sec/step, elapsed 0:00:50, exp. remaining 0:27:50, complete 2.93%
att-weights epoch 183, step 56, max_size:classes 54, max_size:data 1942, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.016 sec/step, elapsed 0:00:51, exp. remaining 0:28:03, complete 2.97%
att-weights epoch 183, step 57, max_size:classes 61, max_size:data 1607, mem_usage:GPU:0 1.4GB, num_seqs 2, 5.365 sec/step, elapsed 0:00:56, exp. remaining 0:30:14, complete 3.04%
att-weights epoch 183, step 58, max_size:classes 60, max_size:data 1529, mem_usage:GPU:0 1.4GB, num_seqs 1, 1.465 sec/step, elapsed 0:00:58, exp. remaining 0:30:18, complete 3.11%
att-weights epoch 183, step 59, max_size:classes 62, max_size:data 2044, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.944 sec/step, elapsed 0:00:59, exp. remaining 0:30:05, complete 3.18%
att-weights epoch 183, step 60, max_size:classes 55, max_size:data 2007, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.731 sec/step, elapsed 0:00:59, exp. remaining 0:29:47, complete 3.25%
att-weights epoch 183, step 61, max_size:classes 51, max_size:data 1820, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.970 sec/step, elapsed 0:01:00, exp. remaining 0:29:37, complete 3.32%
att-weights epoch 183, step 62, max_size:classes 62, max_size:data 2501, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.840 sec/step, elapsed 0:01:01, exp. remaining 0:29:23, complete 3.39%
att-weights epoch 183, step 63, max_size:classes 59, max_size:data 1603, mem_usage:GPU:0 1.4GB, num_seqs 2, 2.734 sec/step, elapsed 0:01:04, exp. remaining 0:30:02, complete 3.46%
att-weights epoch 183, step 64, max_size:classes 55, max_size:data 1503, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.731 sec/step, elapsed 0:01:05, exp. remaining 0:29:45, complete 3.53%
att-weights epoch 183, step 65, max_size:classes 58, max_size:data 2102, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.718 sec/step, elapsed 0:01:05, exp. remaining 0:29:28, complete 3.60%
att-weights epoch 183, step 66, max_size:classes 56, max_size:data 1594, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.302 sec/step, elapsed 0:01:07, exp. remaining 0:29:28, complete 3.67%
att-weights epoch 183, step 67, max_size:classes 57, max_size:data 1603, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.818 sec/step, elapsed 0:01:08, exp. remaining 0:29:15, complete 3.74%
att-weights epoch 183, step 68, max_size:classes 57, max_size:data 1912, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.724 sec/step, elapsed 0:01:08, exp. remaining 0:28:59, complete 3.81%
att-weights epoch 183, step 69, max_size:classes 50, max_size:data 1795, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.702 sec/step, elapsed 0:01:09, exp. remaining 0:28:44, complete 3.88%
att-weights epoch 183, step 70, max_size:classes 55, max_size:data 1574, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.741 sec/step, elapsed 0:01:10, exp. remaining 0:28:46, complete 3.91%
att-weights epoch 183, step 71, max_size:classes 53, max_size:data 1496, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.674 sec/step, elapsed 0:01:10, exp. remaining 0:28:47, complete 3.95%
att-weights epoch 183, step 72, max_size:classes 58, max_size:data 1601, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.726 sec/step, elapsed 0:01:11, exp. remaining 0:28:33, complete 4.02%
att-weights epoch 183, step 73, max_size:classes 51, max_size:data 1366, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.734 sec/step, elapsed 0:01:12, exp. remaining 0:28:20, complete 4.09%
att-weights epoch 183, step 74, max_size:classes 55, max_size:data 1479, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.698 sec/step, elapsed 0:01:13, exp. remaining 0:28:06, complete 4.16%
att-weights epoch 183, step 75, max_size:classes 48, max_size:data 1657, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.836 sec/step, elapsed 0:01:13, exp. remaining 0:27:56, complete 4.22%
att-weights epoch 183, step 76, max_size:classes 53, max_size:data 1560, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.678 sec/step, elapsed 0:01:14, exp. remaining 0:27:43, complete 4.29%
att-weights epoch 183, step 77, max_size:classes 58, max_size:data 1576, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.076 sec/step, elapsed 0:01:15, exp. remaining 0:27:38, complete 4.36%
att-weights epoch 183, step 78, max_size:classes 54, max_size:data 1657, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.846 sec/step, elapsed 0:01:16, exp. remaining 0:27:29, complete 4.43%
att-weights epoch 183, step 79, max_size:classes 50, max_size:data 1349, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.515 sec/step, elapsed 0:01:17, exp. remaining 0:27:13, complete 4.50%
att-weights epoch 183, step 80, max_size:classes 53, max_size:data 2082, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.771 sec/step, elapsed 0:01:17, exp. remaining 0:27:03, complete 4.57%
att-weights epoch 183, step 81, max_size:classes 54, max_size:data 1891, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.147 sec/step, elapsed 0:01:18, exp. remaining 0:27:14, complete 4.61%
att-weights epoch 183, step 82, max_size:classes 54, max_size:data 1364, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.793 sec/step, elapsed 0:01:19, exp. remaining 0:27:05, complete 4.68%
att-weights epoch 183, step 83, max_size:classes 45, max_size:data 1750, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.862 sec/step, elapsed 0:01:20, exp. remaining 0:26:57, complete 4.75%
att-weights epoch 183, step 84, max_size:classes 50, max_size:data 1418, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.075 sec/step, elapsed 0:01:21, exp. remaining 0:26:54, complete 4.82%
att-weights epoch 183, step 85, max_size:classes 61, max_size:data 1682, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.042 sec/step, elapsed 0:01:22, exp. remaining 0:26:50, complete 4.89%
att-weights epoch 183, step 86, max_size:classes 54, max_size:data 1357, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.793 sec/step, elapsed 0:01:23, exp. remaining 0:26:41, complete 4.96%
att-weights epoch 183, step 87, max_size:classes 55, max_size:data 1520, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.725 sec/step, elapsed 0:01:24, exp. remaining 0:26:31, complete 5.03%
att-weights epoch 183, step 88, max_size:classes 50, max_size:data 1434, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.758 sec/step, elapsed 0:01:25, exp. remaining 0:26:23, complete 5.10%
att-weights epoch 183, step 89, max_size:classes 52, max_size:data 1789, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.973 sec/step, elapsed 0:01:26, exp. remaining 0:26:18, complete 5.17%
att-weights epoch 183, step 90, max_size:classes 59, max_size:data 2713, mem_usage:GPU:0 1.4GB, num_seqs 1, 1.034 sec/step, elapsed 0:01:27, exp. remaining 0:26:14, complete 5.24%
att-weights epoch 183, step 91, max_size:classes 52, max_size:data 1987, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.741 sec/step, elapsed 0:01:27, exp. remaining 0:26:06, complete 5.31%
att-weights epoch 183, step 92, max_size:classes 51, max_size:data 1756, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.817 sec/step, elapsed 0:01:28, exp. remaining 0:25:59, complete 5.38%
att-weights epoch 183, step 93, max_size:classes 49, max_size:data 1426, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.727 sec/step, elapsed 0:01:29, exp. remaining 0:26:01, complete 5.41%
att-weights epoch 183, step 94, max_size:classes 48, max_size:data 1450, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.647 sec/step, elapsed 0:01:29, exp. remaining 0:25:51, complete 5.48%
att-weights epoch 183, step 95, max_size:classes 49, max_size:data 1606, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.704 sec/step, elapsed 0:01:30, exp. remaining 0:25:42, complete 5.55%
att-weights epoch 183, step 96, max_size:classes 49, max_size:data 1593, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.755 sec/step, elapsed 0:01:31, exp. remaining 0:25:35, complete 5.62%
att-weights epoch 183, step 97, max_size:classes 54, max_size:data 1701, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.723 sec/step, elapsed 0:01:32, exp. remaining 0:25:27, complete 5.69%
att-weights epoch 183, step 98, max_size:classes 47, max_size:data 1596, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.733 sec/step, elapsed 0:01:32, exp. remaining 0:25:19, complete 5.76%
att-weights epoch 183, step 99, max_size:classes 49, max_size:data 1878, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.810 sec/step, elapsed 0:01:33, exp. remaining 0:25:13, complete 5.83%
att-weights epoch 183, step 100, max_size:classes 55, max_size:data 1407, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.672 sec/step, elapsed 0:01:34, exp. remaining 0:25:05, complete 5.90%
att-weights epoch 183, step 101, max_size:classes 48, max_size:data 1524, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.707 sec/step, elapsed 0:01:35, exp. remaining 0:24:57, complete 5.97%
att-weights epoch 183, step 102, max_size:classes 52, max_size:data 2377, mem_usage:GPU:0 1.4GB, num_seqs 1, 0.860 sec/step, elapsed 0:01:35, exp. remaining 0:24:52, complete 6.04%
att-weights epoch 183, step 103, max_size:classes 52, max_size:data 1682, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.705 sec/step, elapsed 0:01:36, exp. remaining 0:24:45, complete 6.11%
att-weights epoch 183, step 104, max_size:classes 50, max_size:data 1310, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.780 sec/step, elapsed 0:01:37, exp. remaining 0:24:39, complete 6.18%
att-weights epoch 183, step 105, max_size:classes 46, max_size:data 1659, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.876 sec/step, elapsed 0:01:38, exp. remaining 0:24:34, complete 6.25%
att-weights epoch 183, step 106, max_size:classes 56, max_size:data 1578, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.160 sec/step, elapsed 0:01:39, exp. remaining 0:24:34, complete 6.32%
att-weights epoch 183, step 107, max_size:classes 49, max_size:data 1712, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.983 sec/step, elapsed 0:01:40, exp. remaining 0:24:31, complete 6.39%
att-weights epoch 183, step 108, max_size:classes 49, max_size:data 1813, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.789 sec/step, elapsed 0:01:41, exp. remaining 0:24:26, complete 6.46%
att-weights epoch 183, step 109, max_size:classes 48, max_size:data 1637, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.961 sec/step, elapsed 0:01:42, exp. remaining 0:24:23, complete 6.53%
att-weights epoch 183, step 110, max_size:classes 52, max_size:data 1820, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.893 sec/step, elapsed 0:01:43, exp. remaining 0:24:19, complete 6.60%
att-weights epoch 183, step 111, max_size:classes 43, max_size:data 1717, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.276 sec/step, elapsed 0:01:44, exp. remaining 0:24:20, complete 6.67%
att-weights epoch 183, step 112, max_size:classes 44, max_size:data 1363, mem_usage:GPU:0 1.4GB, num_seqs 2, 2.377 sec/step, elapsed 0:01:46, exp. remaining 0:24:37, complete 6.74%
att-weights epoch 183, step 113, max_size:classes 51, max_size:data 1431, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.355 sec/step, elapsed 0:01:48, exp. remaining 0:24:39, complete 6.81%
att-weights epoch 183, step 114, max_size:classes 48, max_size:data 1879, mem_usage:GPU:0 1.4GB, num_seqs 2, 2.155 sec/step, elapsed 0:01:50, exp. remaining 0:24:52, complete 6.88%
att-weights epoch 183, step 115, max_size:classes 50, max_size:data 1384, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.428 sec/step, elapsed 0:01:51, exp. remaining 0:24:55, complete 6.95%
att-weights epoch 183, step 116, max_size:classes 44, max_size:data 1359, mem_usage:GPU:0 1.4GB, num_seqs 2, 9.368 sec/step, elapsed 0:02:01, exp. remaining 0:26:43, complete 7.02%
att-weights epoch 183, step 117, max_size:classes 47, max_size:data 1672, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.990 sec/step, elapsed 0:02:02, exp. remaining 0:26:39, complete 7.09%
att-weights epoch 183, step 118, max_size:classes 45, max_size:data 1593, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.335 sec/step, elapsed 0:02:03, exp. remaining 0:26:40, complete 7.16%
att-weights epoch 183, step 119, max_size:classes 45, max_size:data 1746, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.193 sec/step, elapsed 0:02:04, exp. remaining 0:26:39, complete 7.23%
att-weights epoch 183, step 120, max_size:classes 47, max_size:data 1635, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.277 sec/step, elapsed 0:02:05, exp. remaining 0:26:38, complete 7.30%
att-weights epoch 183, step 121, max_size:classes 48, max_size:data 1607, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.795 sec/step, elapsed 0:02:06, exp. remaining 0:26:32, complete 7.37%
att-weights epoch 183, step 122, max_size:classes 59, max_size:data 1396, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.275 sec/step, elapsed 0:02:07, exp. remaining 0:26:32, complete 7.44%
att-weights epoch 183, step 123, max_size:classes 46, max_size:data 1434, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.920 sec/step, elapsed 0:02:09, exp. remaining 0:26:39, complete 7.51%
att-weights epoch 183, step 124, max_size:classes 39, max_size:data 1269, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.555 sec/step, elapsed 0:02:11, exp. remaining 0:26:34, complete 7.61%
att-weights epoch 183, step 125, max_size:classes 46, max_size:data 1530, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.428 sec/step, elapsed 0:02:12, exp. remaining 0:26:36, complete 7.68%
att-weights epoch 183, step 126, max_size:classes 39, max_size:data 1782, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.430 sec/step, elapsed 0:02:14, exp. remaining 0:26:37, complete 7.75%
att-weights epoch 183, step 127, max_size:classes 43, max_size:data 1598, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.786 sec/step, elapsed 0:02:15, exp. remaining 0:26:31, complete 7.82%
att-weights epoch 183, step 128, max_size:classes 47, max_size:data 1236, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.080 sec/step, elapsed 0:02:16, exp. remaining 0:26:28, complete 7.89%
att-weights epoch 183, step 129, max_size:classes 43, max_size:data 1534, mem_usage:GPU:0 1.4GB, num_seqs 2, 2.149 sec/step, elapsed 0:02:18, exp. remaining 0:26:38, complete 7.96%
att-weights epoch 183, step 130, max_size:classes 47, max_size:data 1391, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.289 sec/step, elapsed 0:02:19, exp. remaining 0:26:30, complete 8.07%
att-weights epoch 183, step 131, max_size:classes 43, max_size:data 1494, mem_usage:GPU:0 1.4GB, num_seqs 2, 3.592 sec/step, elapsed 0:02:23, exp. remaining 0:26:56, complete 8.14%
att-weights epoch 183, step 132, max_size:classes 45, max_size:data 1531, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.889 sec/step, elapsed 0:02:24, exp. remaining 0:26:51, complete 8.21%
att-weights epoch 183, step 133, max_size:classes 41, max_size:data 1304, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.005 sec/step, elapsed 0:02:25, exp. remaining 0:26:47, complete 8.28%
att-weights epoch 183, step 134, max_size:classes 45, max_size:data 1371, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.863 sec/step, elapsed 0:02:25, exp. remaining 0:26:42, complete 8.34%
att-weights epoch 183, step 135, max_size:classes 48, max_size:data 1380, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.041 sec/step, elapsed 0:02:26, exp. remaining 0:26:39, complete 8.41%
att-weights epoch 183, step 136, max_size:classes 45, max_size:data 1664, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.076 sec/step, elapsed 0:02:28, exp. remaining 0:26:29, complete 8.52%
att-weights epoch 183, step 137, max_size:classes 41, max_size:data 1115, mem_usage:GPU:0 1.4GB, num_seqs 2, 3.290 sec/step, elapsed 0:02:31, exp. remaining 0:26:43, complete 8.62%
att-weights epoch 183, step 138, max_size:classes 41, max_size:data 1760, mem_usage:GPU:0 1.4GB, num_seqs 2, 7.507 sec/step, elapsed 0:02:38, exp. remaining 0:27:40, complete 8.73%
att-weights epoch 183, step 139, max_size:classes 44, max_size:data 1268, mem_usage:GPU:0 1.4GB, num_seqs 3, 0.816 sec/step, elapsed 0:02:39, exp. remaining 0:27:34, complete 8.80%
att-weights epoch 183, step 140, max_size:classes 41, max_size:data 1466, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.027 sec/step, elapsed 0:02:40, exp. remaining 0:27:31, complete 8.87%
att-weights epoch 183, step 141, max_size:classes 41, max_size:data 1377, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.005 sec/step, elapsed 0:02:41, exp. remaining 0:27:27, complete 8.94%
att-weights epoch 183, step 142, max_size:classes 47, max_size:data 1233, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.811 sec/step, elapsed 0:02:42, exp. remaining 0:27:21, complete 9.01%
att-weights epoch 183, step 143, max_size:classes 40, max_size:data 1558, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.787 sec/step, elapsed 0:02:43, exp. remaining 0:27:08, complete 9.11%
att-weights epoch 183, step 144, max_size:classes 38, max_size:data 1389, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.742 sec/step, elapsed 0:02:44, exp. remaining 0:26:55, complete 9.22%
att-weights epoch 183, step 145, max_size:classes 42, max_size:data 1079, mem_usage:GPU:0 1.4GB, num_seqs 3, 0.908 sec/step, elapsed 0:02:44, exp. remaining 0:26:50, complete 9.29%
att-weights epoch 183, step 146, max_size:classes 41, max_size:data 1246, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.162 sec/step, elapsed 0:02:46, exp. remaining 0:26:42, complete 9.39%
att-weights epoch 183, step 147, max_size:classes 42, max_size:data 1182, mem_usage:GPU:0 1.4GB, num_seqs 3, 0.993 sec/step, elapsed 0:02:47, exp. remaining 0:26:38, complete 9.46%
att-weights epoch 183, step 148, max_size:classes 40, max_size:data 1375, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.761 sec/step, elapsed 0:02:47, exp. remaining 0:26:26, complete 9.57%
att-weights epoch 183, step 149, max_size:classes 41, max_size:data 974, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.950 sec/step, elapsed 0:02:48, exp. remaining 0:26:22, complete 9.64%
att-weights epoch 183, step 150, max_size:classes 38, max_size:data 1528, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.912 sec/step, elapsed 0:02:49, exp. remaining 0:26:18, complete 9.71%
att-weights epoch 183, step 151, max_size:classes 43, max_size:data 1376, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.926 sec/step, elapsed 0:02:50, exp. remaining 0:26:14, complete 9.78%
att-weights epoch 183, step 152, max_size:classes 40, max_size:data 1006, mem_usage:GPU:0 1.4GB, num_seqs 3, 0.934 sec/step, elapsed 0:02:51, exp. remaining 0:26:04, complete 9.88%
att-weights epoch 183, step 153, max_size:classes 43, max_size:data 1126, mem_usage:GPU:0 1.4GB, num_seqs 3, 0.904 sec/step, elapsed 0:02:52, exp. remaining 0:25:54, complete 9.99%
att-weights epoch 183, step 154, max_size:classes 45, max_size:data 1416, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.779 sec/step, elapsed 0:02:53, exp. remaining 0:25:43, complete 10.09%
att-weights epoch 183, step 155, max_size:classes 39, max_size:data 1275, mem_usage:GPU:0 1.4GB, num_seqs 3, 0.674 sec/step, elapsed 0:02:53, exp. remaining 0:25:32, complete 10.20%
att-weights epoch 183, step 156, max_size:classes 39, max_size:data 1359, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.758 sec/step, elapsed 0:02:54, exp. remaining 0:25:27, complete 10.27%
att-weights epoch 183, step 157, max_size:classes 44, max_size:data 1266, mem_usage:GPU:0 1.4GB, num_seqs 3, 0.725 sec/step, elapsed 0:02:55, exp. remaining 0:25:21, complete 10.34%
att-weights epoch 183, step 158, max_size:classes 40, max_size:data 1008, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.612 sec/step, elapsed 0:02:56, exp. remaining 0:25:15, complete 10.41%
att-weights epoch 183, step 159, max_size:classes 43, max_size:data 1530, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.860 sec/step, elapsed 0:02:56, exp. remaining 0:25:11, complete 10.47%
att-weights epoch 183, step 160, max_size:classes 33, max_size:data 1605, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.942 sec/step, elapsed 0:02:57, exp. remaining 0:25:08, complete 10.54%
att-weights epoch 183, step 161, max_size:classes 46, max_size:data 1117, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.901 sec/step, elapsed 0:02:59, exp. remaining 0:25:13, complete 10.61%
att-weights epoch 183, step 162, max_size:classes 41, max_size:data 1300, mem_usage:GPU:0 1.4GB, num_seqs 3, 6.721 sec/step, elapsed 0:03:06, exp. remaining 0:25:52, complete 10.72%
att-weights epoch 183, step 163, max_size:classes 41, max_size:data 1281, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.970 sec/step, elapsed 0:03:08, exp. remaining 0:25:52, complete 10.82%
att-weights epoch 183, step 164, max_size:classes 41, max_size:data 1307, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.636 sec/step, elapsed 0:03:10, exp. remaining 0:25:54, complete 10.89%
att-weights epoch 183, step 165, max_size:classes 41, max_size:data 1429, mem_usage:GPU:0 1.4GB, num_seqs 2, 2.117 sec/step, elapsed 0:03:12, exp. remaining 0:25:55, complete 11.00%
att-weights epoch 183, step 166, max_size:classes 38, max_size:data 1771, mem_usage:GPU:0 1.4GB, num_seqs 2, 3.060 sec/step, elapsed 0:03:15, exp. remaining 0:26:03, complete 11.10%
att-weights epoch 183, step 167, max_size:classes 42, max_size:data 1241, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.808 sec/step, elapsed 0:03:16, exp. remaining 0:25:58, complete 11.17%
att-weights epoch 183, step 168, max_size:classes 43, max_size:data 1530, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.767 sec/step, elapsed 0:03:16, exp. remaining 0:25:53, complete 11.24%
att-weights epoch 183, step 169, max_size:classes 42, max_size:data 1191, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.674 sec/step, elapsed 0:03:17, exp. remaining 0:25:42, complete 11.35%
att-weights epoch 183, step 170, max_size:classes 43, max_size:data 1350, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.707 sec/step, elapsed 0:03:18, exp. remaining 0:25:37, complete 11.42%
att-weights epoch 183, step 171, max_size:classes 38, max_size:data 1203, mem_usage:GPU:0 1.4GB, num_seqs 3, 0.881 sec/step, elapsed 0:03:19, exp. remaining 0:25:33, complete 11.49%
att-weights epoch 183, step 172, max_size:classes 41, max_size:data 1228, mem_usage:GPU:0 1.4GB, num_seqs 3, 0.868 sec/step, elapsed 0:03:19, exp. remaining 0:25:30, complete 11.56%
att-weights epoch 183, step 173, max_size:classes 42, max_size:data 1591, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.864 sec/step, elapsed 0:03:20, exp. remaining 0:25:21, complete 11.66%
att-weights epoch 183, step 174, max_size:classes 43, max_size:data 1327, mem_usage:GPU:0 1.4GB, num_seqs 3, 0.912 sec/step, elapsed 0:03:21, exp. remaining 0:25:17, complete 11.73%
att-weights epoch 183, step 175, max_size:classes 41, max_size:data 1307, mem_usage:GPU:0 1.4GB, num_seqs 3, 0.906 sec/step, elapsed 0:03:22, exp. remaining 0:25:14, complete 11.80%
att-weights epoch 183, step 176, max_size:classes 37, max_size:data 1059, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.561 sec/step, elapsed 0:03:23, exp. remaining 0:25:03, complete 11.91%
att-weights epoch 183, step 177, max_size:classes 41, max_size:data 1392, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.888 sec/step, elapsed 0:03:24, exp. remaining 0:24:54, complete 12.01%
att-weights epoch 183, step 178, max_size:classes 37, max_size:data 1081, mem_usage:GPU:0 1.4GB, num_seqs 3, 0.850 sec/step, elapsed 0:03:24, exp. remaining 0:24:51, complete 12.08%
att-weights epoch 183, step 179, max_size:classes 38, max_size:data 1461, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.931 sec/step, elapsed 0:03:25, exp. remaining 0:24:43, complete 12.19%
att-weights epoch 183, step 180, max_size:classes 44, max_size:data 1108, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.900 sec/step, elapsed 0:03:26, exp. remaining 0:24:35, complete 12.29%
att-weights epoch 183, step 181, max_size:classes 41, max_size:data 1351, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.820 sec/step, elapsed 0:03:27, exp. remaining 0:24:27, complete 12.40%
att-weights epoch 183, step 182, max_size:classes 38, max_size:data 1161, mem_usage:GPU:0 1.4GB, num_seqs 3, 0.923 sec/step, elapsed 0:03:28, exp. remaining 0:24:19, complete 12.50%
att-weights epoch 183, step 183, max_size:classes 41, max_size:data 1526, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.418 sec/step, elapsed 0:03:29, exp. remaining 0:24:20, complete 12.57%
att-weights epoch 183, step 184, max_size:classes 40, max_size:data 1952, mem_usage:GPU:0 1.4GB, num_seqs 2, 3.275 sec/step, elapsed 0:03:33, exp. remaining 0:24:28, complete 12.67%
att-weights epoch 183, step 185, max_size:classes 42, max_size:data 1272, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.584 sec/step, elapsed 0:03:34, exp. remaining 0:24:25, complete 12.78%
att-weights epoch 183, step 186, max_size:classes 34, max_size:data 1238, mem_usage:GPU:0 1.4GB, num_seqs 3, 2.390 sec/step, elapsed 0:03:37, exp. remaining 0:24:33, complete 12.85%
att-weights epoch 183, step 187, max_size:classes 47, max_size:data 1351, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.592 sec/step, elapsed 0:03:38, exp. remaining 0:24:30, complete 12.95%
att-weights epoch 183, step 188, max_size:classes 38, max_size:data 1263, mem_usage:GPU:0 1.4GB, num_seqs 3, 2.052 sec/step, elapsed 0:03:40, exp. remaining 0:24:25, complete 13.09%
att-weights epoch 183, step 189, max_size:classes 38, max_size:data 1061, mem_usage:GPU:0 1.4GB, num_seqs 3, 2.774 sec/step, elapsed 0:03:43, exp. remaining 0:24:35, complete 13.16%
att-weights epoch 183, step 190, max_size:classes 37, max_size:data 1254, mem_usage:GPU:0 1.4GB, num_seqs 3, 2.278 sec/step, elapsed 0:03:45, exp. remaining 0:24:36, complete 13.27%
att-weights epoch 183, step 191, max_size:classes 41, max_size:data 1299, mem_usage:GPU:0 1.4GB, num_seqs 3, 2.147 sec/step, elapsed 0:03:48, exp. remaining 0:24:37, complete 13.37%
att-weights epoch 183, step 192, max_size:classes 38, max_size:data 1447, mem_usage:GPU:0 1.4GB, num_seqs 2, 2.558 sec/step, elapsed 0:03:50, exp. remaining 0:24:40, complete 13.48%
att-weights epoch 183, step 193, max_size:classes 37, max_size:data 1163, mem_usage:GPU:0 1.4GB, num_seqs 3, 2.271 sec/step, elapsed 0:03:52, exp. remaining 0:24:41, complete 13.58%
att-weights epoch 183, step 194, max_size:classes 43, max_size:data 1024, mem_usage:GPU:0 1.4GB, num_seqs 3, 2.498 sec/step, elapsed 0:03:55, exp. remaining 0:24:44, complete 13.69%
att-weights epoch 183, step 195, max_size:classes 33, max_size:data 1395, mem_usage:GPU:0 1.4GB, num_seqs 2, 2.283 sec/step, elapsed 0:03:57, exp. remaining 0:24:45, complete 13.79%
att-weights epoch 183, step 196, max_size:classes 37, max_size:data 1269, mem_usage:GPU:0 1.4GB, num_seqs 3, 2.837 sec/step, elapsed 0:04:00, exp. remaining 0:24:45, complete 13.93%
att-weights epoch 183, step 197, max_size:classes 35, max_size:data 912, mem_usage:GPU:0 1.4GB, num_seqs 4, 3.026 sec/step, elapsed 0:04:03, exp. remaining 0:24:55, complete 14.00%
att-weights epoch 183, step 198, max_size:classes 35, max_size:data 1369, mem_usage:GPU:0 1.4GB, num_seqs 2, 0.862 sec/step, elapsed 0:04:04, exp. remaining 0:24:47, complete 14.11%
att-weights epoch 183, step 199, max_size:classes 33, max_size:data 972, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.007 sec/step, elapsed 0:04:05, exp. remaining 0:24:37, complete 14.25%
att-weights epoch 183, step 200, max_size:classes 36, max_size:data 1295, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.510 sec/step, elapsed 0:04:06, exp. remaining 0:24:33, complete 14.35%
att-weights epoch 183, step 201, max_size:classes 37, max_size:data 1160, mem_usage:GPU:0 1.4GB, num_seqs 3, 2.892 sec/step, elapsed 0:04:09, exp. remaining 0:24:38, complete 14.46%
att-weights epoch 183, step 202, max_size:classes 35, max_size:data 1027, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.857 sec/step, elapsed 0:04:11, exp. remaining 0:24:40, complete 14.53%
att-weights epoch 183, step 203, max_size:classes 36, max_size:data 1230, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.520 sec/step, elapsed 0:04:13, exp. remaining 0:24:37, complete 14.63%
att-weights epoch 183, step 204, max_size:classes 36, max_size:data 1042, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.281 sec/step, elapsed 0:04:14, exp. remaining 0:24:36, complete 14.70%
att-weights epoch 183, step 205, max_size:classes 37, max_size:data 968, mem_usage:GPU:0 1.4GB, num_seqs 4, 4.863 sec/step, elapsed 0:04:19, exp. remaining 0:24:52, complete 14.80%
att-weights epoch 183, step 206, max_size:classes 33, max_size:data 1366, mem_usage:GPU:0 1.4GB, num_seqs 2, 2.225 sec/step, elapsed 0:04:21, exp. remaining 0:24:52, complete 14.91%
att-weights epoch 183, step 207, max_size:classes 34, max_size:data 1093, mem_usage:GPU:0 1.4GB, num_seqs 3, 2.640 sec/step, elapsed 0:04:24, exp. remaining 0:24:51, complete 15.05%
att-weights epoch 183, step 208, max_size:classes 36, max_size:data 994, mem_usage:GPU:0 1.4GB, num_seqs 4, 2.260 sec/step, elapsed 0:04:26, exp. remaining 0:24:51, complete 15.15%
att-weights epoch 183, step 209, max_size:classes 40, max_size:data 1010, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.966 sec/step, elapsed 0:04:28, exp. remaining 0:24:50, complete 15.26%
att-weights epoch 183, step 210, max_size:classes 38, max_size:data 1276, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.774 sec/step, elapsed 0:04:30, exp. remaining 0:24:48, complete 15.36%
att-weights epoch 183, step 211, max_size:classes 35, max_size:data 1355, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.501 sec/step, elapsed 0:04:31, exp. remaining 0:24:44, complete 15.47%
att-weights epoch 183, step 212, max_size:classes 33, max_size:data 1069, mem_usage:GPU:0 1.4GB, num_seqs 3, 3.797 sec/step, elapsed 0:04:35, exp. remaining 0:24:53, complete 15.57%
att-weights epoch 183, step 213, max_size:classes 40, max_size:data 1448, mem_usage:GPU:0 1.4GB, num_seqs 2, 4.761 sec/step, elapsed 0:04:40, exp. remaining 0:25:07, complete 15.68%
att-weights epoch 183, step 214, max_size:classes 38, max_size:data 1323, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.706 sec/step, elapsed 0:04:41, exp. remaining 0:25:04, complete 15.78%
att-weights epoch 183, step 215, max_size:classes 34, max_size:data 1177, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.156 sec/step, elapsed 0:04:43, exp. remaining 0:24:58, complete 15.89%
att-weights epoch 183, step 216, max_size:classes 37, max_size:data 982, mem_usage:GPU:0 1.4GB, num_seqs 4, 2.979 sec/step, elapsed 0:04:46, exp. remaining 0:25:02, complete 15.99%
att-weights epoch 183, step 217, max_size:classes 34, max_size:data 1000, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.865 sec/step, elapsed 0:04:47, exp. remaining 0:25:00, complete 16.10%
att-weights epoch 183, step 218, max_size:classes 34, max_size:data 1072, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.814 sec/step, elapsed 0:04:49, exp. remaining 0:24:58, complete 16.20%
att-weights epoch 183, step 219, max_size:classes 36, max_size:data 1197, mem_usage:GPU:0 1.4GB, num_seqs 3, 2.011 sec/step, elapsed 0:04:51, exp. remaining 0:24:57, complete 16.31%
att-weights epoch 183, step 220, max_size:classes 33, max_size:data 1013, mem_usage:GPU:0 1.4GB, num_seqs 3, 2.833 sec/step, elapsed 0:04:54, exp. remaining 0:24:56, complete 16.45%
att-weights epoch 183, step 221, max_size:classes 36, max_size:data 930, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.922 sec/step, elapsed 0:04:56, exp. remaining 0:24:55, complete 16.55%
att-weights epoch 183, step 222, max_size:classes 35, max_size:data 1123, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.915 sec/step, elapsed 0:04:58, exp. remaining 0:24:49, complete 16.69%
att-weights epoch 183, step 223, max_size:classes 33, max_size:data 1042, mem_usage:GPU:0 1.4GB, num_seqs 3, 2.076 sec/step, elapsed 0:05:00, exp. remaining 0:24:48, complete 16.79%
att-weights epoch 183, step 224, max_size:classes 33, max_size:data 1186, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.905 sec/step, elapsed 0:05:02, exp. remaining 0:24:47, complete 16.90%
att-weights epoch 183, step 225, max_size:classes 35, max_size:data 1170, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.615 sec/step, elapsed 0:05:04, exp. remaining 0:24:43, complete 17.00%
att-weights epoch 183, step 226, max_size:classes 37, max_size:data 1025, mem_usage:GPU:0 1.4GB, num_seqs 3, 2.964 sec/step, elapsed 0:05:06, exp. remaining 0:24:47, complete 17.11%
att-weights epoch 183, step 227, max_size:classes 32, max_size:data 1013, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.595 sec/step, elapsed 0:05:08, exp. remaining 0:24:44, complete 17.21%
att-weights epoch 183, step 228, max_size:classes 35, max_size:data 1177, mem_usage:GPU:0 1.4GB, num_seqs 3, 2.520 sec/step, elapsed 0:05:11, exp. remaining 0:24:41, complete 17.35%
att-weights epoch 183, step 229, max_size:classes 35, max_size:data 955, mem_usage:GPU:0 1.4GB, num_seqs 4, 3.828 sec/step, elapsed 0:05:14, exp. remaining 0:24:49, complete 17.46%
att-weights epoch 183, step 230, max_size:classes 31, max_size:data 1265, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.462 sec/step, elapsed 0:05:16, exp. remaining 0:24:45, complete 17.56%
att-weights epoch 183, step 231, max_size:classes 37, max_size:data 981, mem_usage:GPU:0 1.4GB, num_seqs 4, 2.953 sec/step, elapsed 0:05:19, exp. remaining 0:24:44, complete 17.70%
att-weights epoch 183, step 232, max_size:classes 35, max_size:data 1093, mem_usage:GPU:0 1.4GB, num_seqs 3, 3.200 sec/step, elapsed 0:05:22, exp. remaining 0:24:48, complete 17.81%
att-weights epoch 183, step 233, max_size:classes 33, max_size:data 1208, mem_usage:GPU:0 1.4GB, num_seqs 3, 3.568 sec/step, elapsed 0:05:26, exp. remaining 0:24:58, complete 17.88%
att-weights epoch 183, step 234, max_size:classes 37, max_size:data 1028, mem_usage:GPU:0 1.4GB, num_seqs 3, 3.288 sec/step, elapsed 0:05:29, exp. remaining 0:24:58, complete 18.02%
att-weights epoch 183, step 235, max_size:classes 35, max_size:data 1012, mem_usage:GPU:0 1.4GB, num_seqs 3, 2.625 sec/step, elapsed 0:05:32, exp. remaining 0:25:00, complete 18.12%
att-weights epoch 183, step 236, max_size:classes 31, max_size:data 1283, mem_usage:GPU:0 1.4GB, num_seqs 3, 2.867 sec/step, elapsed 0:05:34, exp. remaining 0:24:59, complete 18.26%
att-weights epoch 183, step 237, max_size:classes 34, max_size:data 963, mem_usage:GPU:0 1.4GB, num_seqs 4, 3.077 sec/step, elapsed 0:05:37, exp. remaining 0:25:02, complete 18.37%
att-weights epoch 183, step 238, max_size:classes 33, max_size:data 1087, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.566 sec/step, elapsed 0:05:39, exp. remaining 0:24:58, complete 18.47%
att-weights epoch 183, step 239, max_size:classes 38, max_size:data 1071, mem_usage:GPU:0 1.4GB, num_seqs 3, 3.525 sec/step, elapsed 0:05:43, exp. remaining 0:25:03, complete 18.58%
att-weights epoch 183, step 240, max_size:classes 34, max_size:data 931, mem_usage:GPU:0 1.4GB, num_seqs 4, 2.415 sec/step, elapsed 0:05:45, exp. remaining 0:25:03, complete 18.68%
att-weights epoch 183, step 241, max_size:classes 29, max_size:data 1052, mem_usage:GPU:0 1.4GB, num_seqs 3, 2.260 sec/step, elapsed 0:05:47, exp. remaining 0:25:00, complete 18.82%
att-weights epoch 183, step 242, max_size:classes 34, max_size:data 1371, mem_usage:GPU:0 1.4GB, num_seqs 2, 1.835 sec/step, elapsed 0:05:49, exp. remaining 0:24:57, complete 18.92%
att-weights epoch 183, step 243, max_size:classes 36, max_size:data 982, mem_usage:GPU:0 1.4GB, num_seqs 4, 3.164 sec/step, elapsed 0:05:52, exp. remaining 0:25:00, complete 19.03%
att-weights epoch 183, step 244, max_size:classes 34, max_size:data 1332, mem_usage:GPU:0 1.4GB, num_seqs 3, 3.211 sec/step, elapsed 0:05:55, exp. remaining 0:25:04, complete 19.13%
att-weights epoch 183, step 245, max_size:classes 31, max_size:data 969, mem_usage:GPU:0 1.4GB, num_seqs 4, 4.564 sec/step, elapsed 0:06:00, exp. remaining 0:25:13, complete 19.24%
att-weights epoch 183, step 246, max_size:classes 33, max_size:data 1179, mem_usage:GPU:0 1.4GB, num_seqs 3, 6.669 sec/step, elapsed 0:06:07, exp. remaining 0:25:31, complete 19.34%
att-weights epoch 183, step 247, max_size:classes 32, max_size:data 1085, mem_usage:GPU:0 1.4GB, num_seqs 3, 4.312 sec/step, elapsed 0:06:11, exp. remaining 0:25:38, complete 19.45%
att-weights epoch 183, step 248, max_size:classes 28, max_size:data 1044, mem_usage:GPU:0 1.4GB, num_seqs 3, 7.359 sec/step, elapsed 0:06:18, exp. remaining 0:25:55, complete 19.59%
att-weights epoch 183, step 249, max_size:classes 34, max_size:data 1326, mem_usage:GPU:0 1.4GB, num_seqs 3, 3.065 sec/step, elapsed 0:06:21, exp. remaining 0:25:54, complete 19.73%
att-weights epoch 183, step 250, max_size:classes 34, max_size:data 879, mem_usage:GPU:0 1.4GB, num_seqs 4, 3.994 sec/step, elapsed 0:06:25, exp. remaining 0:26:00, complete 19.83%
att-weights epoch 183, step 251, max_size:classes 32, max_size:data 1032, mem_usage:GPU:0 1.4GB, num_seqs 3, 3.042 sec/step, elapsed 0:06:28, exp. remaining 0:26:02, complete 19.94%
att-weights epoch 183, step 252, max_size:classes 33, max_size:data 1070, mem_usage:GPU:0 1.4GB, num_seqs 3, 3.593 sec/step, elapsed 0:06:32, exp. remaining 0:26:06, complete 20.04%
att-weights epoch 183, step 253, max_size:classes 30, max_size:data 958, mem_usage:GPU:0 1.4GB, num_seqs 3, 3.808 sec/step, elapsed 0:06:36, exp. remaining 0:26:11, complete 20.15%
att-weights epoch 183, step 254, max_size:classes 29, max_size:data 1214, mem_usage:GPU:0 1.4GB, num_seqs 3, 2.617 sec/step, elapsed 0:06:38, exp. remaining 0:26:11, complete 20.25%
att-weights epoch 183, step 255, max_size:classes 33, max_size:data 1108, mem_usage:GPU:0 1.4GB, num_seqs 3, 3.550 sec/step, elapsed 0:06:42, exp. remaining 0:26:14, complete 20.36%
att-weights epoch 183, step 256, max_size:classes 30, max_size:data 1229, mem_usage:GPU:0 1.4GB, num_seqs 3, 3.005 sec/step, elapsed 0:06:45, exp. remaining 0:26:16, complete 20.46%
att-weights epoch 183, step 257, max_size:classes 34, max_size:data 884, mem_usage:GPU:0 1.4GB, num_seqs 4, 3.770 sec/step, elapsed 0:06:49, exp. remaining 0:26:17, complete 20.60%
att-weights epoch 183, step 258, max_size:classes 34, max_size:data 907, mem_usage:GPU:0 1.4GB, num_seqs 4, 4.711 sec/step, elapsed 0:06:54, exp. remaining 0:26:22, complete 20.74%
att-weights epoch 183, step 259, max_size:classes 32, max_size:data 1249, mem_usage:GPU:0 1.4GB, num_seqs 3, 4.564 sec/step, elapsed 0:06:58, exp. remaining 0:26:29, complete 20.84%
att-weights epoch 183, step 260, max_size:classes 31, max_size:data 1104, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.319 sec/step, elapsed 0:06:59, exp. remaining 0:26:24, complete 20.95%
att-weights epoch 183, step 261, max_size:classes 33, max_size:data 1026, mem_usage:GPU:0 1.4GB, num_seqs 3, 3.693 sec/step, elapsed 0:07:03, exp. remaining 0:26:28, complete 21.05%
att-weights epoch 183, step 262, max_size:classes 39, max_size:data 1026, mem_usage:GPU:0 1.4GB, num_seqs 3, 2.258 sec/step, elapsed 0:07:05, exp. remaining 0:26:23, complete 21.19%
att-weights epoch 183, step 263, max_size:classes 34, max_size:data 1010, mem_usage:GPU:0 1.4GB, num_seqs 3, 2.509 sec/step, elapsed 0:07:08, exp. remaining 0:26:22, complete 21.30%
att-weights epoch 183, step 264, max_size:classes 30, max_size:data 1215, mem_usage:GPU:0 1.4GB, num_seqs 3, 2.503 sec/step, elapsed 0:07:10, exp. remaining 0:26:22, complete 21.40%
att-weights epoch 183, step 265, max_size:classes 30, max_size:data 1177, mem_usage:GPU:0 1.4GB, num_seqs 3, 2.862 sec/step, elapsed 0:07:13, exp. remaining 0:26:19, complete 21.54%
att-weights epoch 183, step 266, max_size:classes 36, max_size:data 848, mem_usage:GPU:0 1.4GB, num_seqs 4, 5.517 sec/step, elapsed 0:07:19, exp. remaining 0:26:26, complete 21.68%
att-weights epoch 183, step 267, max_size:classes 30, max_size:data 906, mem_usage:GPU:0 1.4GB, num_seqs 4, 2.166 sec/step, elapsed 0:07:21, exp. remaining 0:26:24, complete 21.79%
att-weights epoch 183, step 268, max_size:classes 31, max_size:data 1044, mem_usage:GPU:0 1.4GB, num_seqs 3, 2.077 sec/step, elapsed 0:07:23, exp. remaining 0:26:22, complete 21.89%
att-weights epoch 183, step 269, max_size:classes 27, max_size:data 1224, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.677 sec/step, elapsed 0:07:25, exp. remaining 0:26:12, complete 22.07%
att-weights epoch 183, step 270, max_size:classes 29, max_size:data 1077, mem_usage:GPU:0 1.4GB, num_seqs 3, 3.366 sec/step, elapsed 0:07:28, exp. remaining 0:26:11, complete 22.21%
att-weights epoch 183, step 271, max_size:classes 31, max_size:data 868, mem_usage:GPU:0 1.4GB, num_seqs 4, 8.504 sec/step, elapsed 0:07:37, exp. remaining 0:26:31, complete 22.31%
att-weights epoch 183, step 272, max_size:classes 32, max_size:data 1176, mem_usage:GPU:0 1.4GB, num_seqs 3, 7.263 sec/step, elapsed 0:07:44, exp. remaining 0:26:43, complete 22.45%
att-weights epoch 183, step 273, max_size:classes 34, max_size:data 1218, mem_usage:GPU:0 1.4GB, num_seqs 3, 2.769 sec/step, elapsed 0:07:47, exp. remaining 0:26:43, complete 22.56%
att-weights epoch 183, step 274, max_size:classes 32, max_size:data 845, mem_usage:GPU:0 1.4GB, num_seqs 4, 2.890 sec/step, elapsed 0:07:49, exp. remaining 0:26:40, complete 22.70%
att-weights epoch 183, step 275, max_size:classes 32, max_size:data 865, mem_usage:GPU:0 1.4GB, num_seqs 4, 2.080 sec/step, elapsed 0:07:52, exp. remaining 0:26:38, complete 22.80%
att-weights epoch 183, step 276, max_size:classes 31, max_size:data 1180, mem_usage:GPU:0 1.4GB, num_seqs 3, 2.169 sec/step, elapsed 0:07:54, exp. remaining 0:26:36, complete 22.91%
att-weights epoch 183, step 277, max_size:classes 31, max_size:data 1005, mem_usage:GPU:0 1.4GB, num_seqs 3, 3.424 sec/step, elapsed 0:07:57, exp. remaining 0:26:35, complete 23.04%
att-weights epoch 183, step 278, max_size:classes 29, max_size:data 793, mem_usage:GPU:0 1.4GB, num_seqs 5, 7.046 sec/step, elapsed 0:08:04, exp. remaining 0:26:45, complete 23.18%
att-weights epoch 183, step 279, max_size:classes 29, max_size:data 994, mem_usage:GPU:0 1.4GB, num_seqs 4, 4.029 sec/step, elapsed 0:08:08, exp. remaining 0:26:49, complete 23.29%
att-weights epoch 183, step 280, max_size:classes 29, max_size:data 1115, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.231 sec/step, elapsed 0:08:09, exp. remaining 0:26:44, complete 23.39%
att-weights epoch 183, step 281, max_size:classes 29, max_size:data 949, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.030 sec/step, elapsed 0:08:10, exp. remaining 0:26:38, complete 23.50%
att-weights epoch 183, step 282, max_size:classes 32, max_size:data 1248, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.071 sec/step, elapsed 0:08:12, exp. remaining 0:26:29, complete 23.64%
att-weights epoch 183, step 283, max_size:classes 28, max_size:data 858, mem_usage:GPU:0 1.4GB, num_seqs 4, 0.853 sec/step, elapsed 0:08:12, exp. remaining 0:26:20, complete 23.78%
att-weights epoch 183, step 284, max_size:classes 31, max_size:data 1176, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.034 sec/step, elapsed 0:08:13, exp. remaining 0:26:11, complete 23.92%
att-weights epoch 183, step 285, max_size:classes 30, max_size:data 1074, mem_usage:GPU:0 1.4GB, num_seqs 3, 0.823 sec/step, elapsed 0:08:14, exp. remaining 0:26:04, complete 24.02%
att-weights epoch 183, step 286, max_size:classes 28, max_size:data 829, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.038 sec/step, elapsed 0:08:15, exp. remaining 0:25:59, complete 24.13%
att-weights epoch 183, step 287, max_size:classes 29, max_size:data 788, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.413 sec/step, elapsed 0:08:17, exp. remaining 0:25:54, complete 24.23%
att-weights epoch 183, step 288, max_size:classes 34, max_size:data 993, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.096 sec/step, elapsed 0:08:18, exp. remaining 0:25:46, complete 24.37%
att-weights epoch 183, step 289, max_size:classes 30, max_size:data 1088, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.351 sec/step, elapsed 0:08:19, exp. remaining 0:25:38, complete 24.51%
att-weights epoch 183, step 290, max_size:classes 28, max_size:data 1039, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.057 sec/step, elapsed 0:08:20, exp. remaining 0:25:30, complete 24.65%
att-weights epoch 183, step 291, max_size:classes 29, max_size:data 923, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.123 sec/step, elapsed 0:08:21, exp. remaining 0:25:22, complete 24.79%
att-weights epoch 183, step 292, max_size:classes 30, max_size:data 881, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.104 sec/step, elapsed 0:08:22, exp. remaining 0:25:14, complete 24.93%
att-weights epoch 183, step 293, max_size:classes 32, max_size:data 824, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.095 sec/step, elapsed 0:08:24, exp. remaining 0:25:06, complete 25.07%
att-weights epoch 183, step 294, max_size:classes 30, max_size:data 1012, mem_usage:GPU:0 1.4GB, num_seqs 3, 0.926 sec/step, elapsed 0:08:24, exp. remaining 0:24:58, complete 25.21%
att-weights epoch 183, step 295, max_size:classes 29, max_size:data 956, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.006 sec/step, elapsed 0:08:25, exp. remaining 0:24:52, complete 25.31%
att-weights epoch 183, step 296, max_size:classes 29, max_size:data 1138, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.089 sec/step, elapsed 0:08:27, exp. remaining 0:24:47, complete 25.42%
att-weights epoch 183, step 297, max_size:classes 28, max_size:data 908, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.654 sec/step, elapsed 0:08:28, exp. remaining 0:24:44, complete 25.52%
att-weights epoch 183, step 298, max_size:classes 29, max_size:data 836, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.279 sec/step, elapsed 0:08:30, exp. remaining 0:24:34, complete 25.70%
att-weights epoch 183, step 299, max_size:classes 29, max_size:data 886, mem_usage:GPU:0 1.4GB, num_seqs 4, 3.061 sec/step, elapsed 0:08:33, exp. remaining 0:24:35, complete 25.80%
att-weights epoch 183, step 300, max_size:classes 32, max_size:data 991, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.616 sec/step, elapsed 0:08:34, exp. remaining 0:24:29, complete 25.94%
att-weights epoch 183, step 301, max_size:classes 28, max_size:data 916, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.252 sec/step, elapsed 0:08:35, exp. remaining 0:24:22, complete 26.08%
att-weights epoch 183, step 302, max_size:classes 32, max_size:data 835, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.744 sec/step, elapsed 0:08:37, exp. remaining 0:24:16, complete 26.22%
att-weights epoch 183, step 303, max_size:classes 26, max_size:data 885, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.347 sec/step, elapsed 0:08:39, exp. remaining 0:24:09, complete 26.36%
att-weights epoch 183, step 304, max_size:classes 27, max_size:data 918, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.052 sec/step, elapsed 0:08:40, exp. remaining 0:24:02, complete 26.50%
att-weights epoch 183, step 305, max_size:classes 30, max_size:data 1127, mem_usage:GPU:0 1.4GB, num_seqs 3, 0.913 sec/step, elapsed 0:08:41, exp. remaining 0:23:52, complete 26.68%
att-weights epoch 183, step 306, max_size:classes 33, max_size:data 1053, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.260 sec/step, elapsed 0:08:42, exp. remaining 0:23:45, complete 26.82%
att-weights epoch 183, step 307, max_size:classes 28, max_size:data 783, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.080 sec/step, elapsed 0:08:43, exp. remaining 0:23:38, complete 26.96%
att-weights epoch 183, step 308, max_size:classes 30, max_size:data 1033, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.604 sec/step, elapsed 0:08:44, exp. remaining 0:23:32, complete 27.09%
att-weights epoch 183, step 309, max_size:classes 32, max_size:data 909, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.089 sec/step, elapsed 0:08:46, exp. remaining 0:23:25, complete 27.23%
att-weights epoch 183, step 310, max_size:classes 26, max_size:data 812, mem_usage:GPU:0 1.4GB, num_seqs 4, 2.099 sec/step, elapsed 0:08:48, exp. remaining 0:23:21, complete 27.37%
att-weights epoch 183, step 311, max_size:classes 30, max_size:data 960, mem_usage:GPU:0 1.4GB, num_seqs 4, 3.773 sec/step, elapsed 0:08:51, exp. remaining 0:23:21, complete 27.51%
att-weights epoch 183, step 312, max_size:classes 31, max_size:data 895, mem_usage:GPU:0 1.4GB, num_seqs 4, 2.110 sec/step, elapsed 0:08:54, exp. remaining 0:23:17, complete 27.65%
att-weights epoch 183, step 313, max_size:classes 29, max_size:data 973, mem_usage:GPU:0 1.4GB, num_seqs 4, 2.165 sec/step, elapsed 0:08:56, exp. remaining 0:23:10, complete 27.83%
att-weights epoch 183, step 314, max_size:classes 29, max_size:data 709, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.270 sec/step, elapsed 0:08:57, exp. remaining 0:23:04, complete 27.97%
att-weights epoch 183, step 315, max_size:classes 27, max_size:data 898, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.046 sec/step, elapsed 0:08:58, exp. remaining 0:22:57, complete 28.11%
att-weights epoch 183, step 316, max_size:classes 25, max_size:data 801, mem_usage:GPU:0 1.4GB, num_seqs 4, 0.908 sec/step, elapsed 0:08:59, exp. remaining 0:22:50, complete 28.25%
att-weights epoch 183, step 317, max_size:classes 26, max_size:data 957, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.204 sec/step, elapsed 0:09:00, exp. remaining 0:22:46, complete 28.35%
att-weights epoch 183, step 318, max_size:classes 28, max_size:data 834, mem_usage:GPU:0 1.4GB, num_seqs 4, 3.992 sec/step, elapsed 0:09:04, exp. remaining 0:22:49, complete 28.46%
att-weights epoch 183, step 319, max_size:classes 26, max_size:data 909, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.189 sec/step, elapsed 0:09:05, exp. remaining 0:22:42, complete 28.60%
att-weights epoch 183, step 320, max_size:classes 27, max_size:data 913, mem_usage:GPU:0 1.4GB, num_seqs 4, 0.953 sec/step, elapsed 0:09:06, exp. remaining 0:22:35, complete 28.74%
att-weights epoch 183, step 321, max_size:classes 29, max_size:data 942, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.314 sec/step, elapsed 0:09:08, exp. remaining 0:22:29, complete 28.88%
att-weights epoch 183, step 322, max_size:classes 28, max_size:data 764, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.068 sec/step, elapsed 0:09:09, exp. remaining 0:22:25, complete 28.98%
att-weights epoch 183, step 323, max_size:classes 28, max_size:data 952, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.114 sec/step, elapsed 0:09:10, exp. remaining 0:22:19, complete 29.12%
att-weights epoch 183, step 324, max_size:classes 26, max_size:data 942, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.039 sec/step, elapsed 0:09:11, exp. remaining 0:22:15, complete 29.22%
att-weights epoch 183, step 325, max_size:classes 27, max_size:data 845, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.327 sec/step, elapsed 0:09:12, exp. remaining 0:22:09, complete 29.36%
att-weights epoch 183, step 326, max_size:classes 26, max_size:data 636, mem_usage:GPU:0 1.4GB, num_seqs 3, 0.835 sec/step, elapsed 0:09:13, exp. remaining 0:22:02, complete 29.50%
att-weights epoch 183, step 327, max_size:classes 26, max_size:data 1029, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.099 sec/step, elapsed 0:09:14, exp. remaining 0:21:53, complete 29.68%
att-weights epoch 183, step 328, max_size:classes 26, max_size:data 917, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.396 sec/step, elapsed 0:09:15, exp. remaining 0:21:46, complete 29.85%
att-weights epoch 183, step 329, max_size:classes 30, max_size:data 780, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.077 sec/step, elapsed 0:09:17, exp. remaining 0:21:37, complete 30.03%
att-weights epoch 183, step 330, max_size:classes 27, max_size:data 864, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.126 sec/step, elapsed 0:09:18, exp. remaining 0:21:32, complete 30.17%
att-weights epoch 183, step 331, max_size:classes 27, max_size:data 1036, mem_usage:GPU:0 1.4GB, num_seqs 3, 0.967 sec/step, elapsed 0:09:19, exp. remaining 0:21:23, complete 30.34%
att-weights epoch 183, step 332, max_size:classes 26, max_size:data 716, mem_usage:GPU:0 1.4GB, num_seqs 4, 0.994 sec/step, elapsed 0:09:20, exp. remaining 0:21:15, complete 30.52%
att-weights epoch 183, step 333, max_size:classes 24, max_size:data 1075, mem_usage:GPU:0 1.4GB, num_seqs 3, 0.926 sec/step, elapsed 0:09:21, exp. remaining 0:21:09, complete 30.66%
att-weights epoch 183, step 334, max_size:classes 30, max_size:data 876, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.184 sec/step, elapsed 0:09:22, exp. remaining 0:21:03, complete 30.80%
att-weights epoch 183, step 335, max_size:classes 25, max_size:data 887, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.158 sec/step, elapsed 0:09:23, exp. remaining 0:20:57, complete 30.94%
att-weights epoch 183, step 336, max_size:classes 25, max_size:data 741, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.067 sec/step, elapsed 0:09:24, exp. remaining 0:20:51, complete 31.08%
att-weights epoch 183, step 337, max_size:classes 29, max_size:data 725, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.310 sec/step, elapsed 0:09:25, exp. remaining 0:20:46, complete 31.22%
att-weights epoch 183, step 338, max_size:classes 29, max_size:data 785, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.258 sec/step, elapsed 0:09:27, exp. remaining 0:20:41, complete 31.35%
att-weights epoch 183, step 339, max_size:classes 28, max_size:data 956, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.221 sec/step, elapsed 0:09:28, exp. remaining 0:20:34, complete 31.53%
att-weights epoch 183, step 340, max_size:classes 26, max_size:data 758, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.110 sec/step, elapsed 0:09:29, exp. remaining 0:20:28, complete 31.67%
att-weights epoch 183, step 341, max_size:classes 27, max_size:data 785, mem_usage:GPU:0 1.4GB, num_seqs 5, 0.971 sec/step, elapsed 0:09:30, exp. remaining 0:20:22, complete 31.81%
att-weights epoch 183, step 342, max_size:classes 28, max_size:data 896, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.489 sec/step, elapsed 0:09:31, exp. remaining 0:20:17, complete 31.95%
att-weights epoch 183, step 343, max_size:classes 25, max_size:data 995, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.441 sec/step, elapsed 0:09:33, exp. remaining 0:20:13, complete 32.09%
att-weights epoch 183, step 344, max_size:classes 26, max_size:data 922, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.572 sec/step, elapsed 0:09:34, exp. remaining 0:20:06, complete 32.26%
att-weights epoch 183, step 345, max_size:classes 24, max_size:data 885, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.022 sec/step, elapsed 0:09:35, exp. remaining 0:19:59, complete 32.44%
att-weights epoch 183, step 346, max_size:classes 29, max_size:data 850, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.148 sec/step, elapsed 0:09:36, exp. remaining 0:19:54, complete 32.58%
att-weights epoch 183, step 347, max_size:classes 26, max_size:data 896, mem_usage:GPU:0 1.4GB, num_seqs 4, 0.928 sec/step, elapsed 0:09:37, exp. remaining 0:19:48, complete 32.72%
att-weights epoch 183, step 348, max_size:classes 28, max_size:data 724, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.113 sec/step, elapsed 0:09:39, exp. remaining 0:19:43, complete 32.86%
att-weights epoch 183, step 349, max_size:classes 27, max_size:data 858, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.053 sec/step, elapsed 0:09:40, exp. remaining 0:19:37, complete 33.00%
att-weights epoch 183, step 350, max_size:classes 25, max_size:data 922, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.282 sec/step, elapsed 0:09:41, exp. remaining 0:19:33, complete 33.14%
att-weights epoch 183, step 351, max_size:classes 26, max_size:data 687, mem_usage:GPU:0 1.4GB, num_seqs 4, 0.976 sec/step, elapsed 0:09:42, exp. remaining 0:19:25, complete 33.31%
att-weights epoch 183, step 352, max_size:classes 31, max_size:data 863, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.356 sec/step, elapsed 0:09:43, exp. remaining 0:19:19, complete 33.48%
att-weights epoch 183, step 353, max_size:classes 26, max_size:data 669, mem_usage:GPU:0 1.4GB, num_seqs 5, 2.501 sec/step, elapsed 0:09:46, exp. remaining 0:19:17, complete 33.62%
att-weights epoch 183, step 354, max_size:classes 26, max_size:data 688, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.043 sec/step, elapsed 0:09:47, exp. remaining 0:19:10, complete 33.80%
att-weights epoch 183, step 355, max_size:classes 32, max_size:data 887, mem_usage:GPU:0 1.4GB, num_seqs 4, 2.020 sec/step, elapsed 0:09:49, exp. remaining 0:19:05, complete 33.97%
att-weights epoch 183, step 356, max_size:classes 28, max_size:data 941, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.232 sec/step, elapsed 0:09:50, exp. remaining 0:19:00, complete 34.11%
att-weights epoch 183, step 357, max_size:classes 25, max_size:data 811, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.136 sec/step, elapsed 0:09:51, exp. remaining 0:18:53, complete 34.29%
att-weights epoch 183, step 358, max_size:classes 27, max_size:data 928, mem_usage:GPU:0 1.4GB, num_seqs 4, 4.577 sec/step, elapsed 0:09:56, exp. remaining 0:18:55, complete 34.43%
att-weights epoch 183, step 359, max_size:classes 25, max_size:data 987, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.401 sec/step, elapsed 0:09:57, exp. remaining 0:18:49, complete 34.60%
att-weights epoch 183, step 360, max_size:classes 24, max_size:data 697, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.155 sec/step, elapsed 0:09:58, exp. remaining 0:18:41, complete 34.81%
att-weights epoch 183, step 361, max_size:classes 24, max_size:data 795, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.249 sec/step, elapsed 0:10:00, exp. remaining 0:18:35, complete 34.99%
att-weights epoch 183, step 362, max_size:classes 23, max_size:data 912, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.210 sec/step, elapsed 0:10:01, exp. remaining 0:18:30, complete 35.13%
att-weights epoch 183, step 363, max_size:classes 24, max_size:data 721, mem_usage:GPU:0 1.4GB, num_seqs 5, 7.038 sec/step, elapsed 0:10:08, exp. remaining 0:18:34, complete 35.30%
att-weights epoch 183, step 364, max_size:classes 26, max_size:data 769, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.191 sec/step, elapsed 0:10:09, exp. remaining 0:18:30, complete 35.44%
att-weights epoch 183, step 365, max_size:classes 26, max_size:data 824, mem_usage:GPU:0 1.4GB, num_seqs 4, 0.996 sec/step, elapsed 0:10:10, exp. remaining 0:18:25, complete 35.58%
att-weights epoch 183, step 366, max_size:classes 24, max_size:data 738, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.291 sec/step, elapsed 0:10:11, exp. remaining 0:18:19, complete 35.75%
att-weights epoch 183, step 367, max_size:classes 24, max_size:data 827, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.096 sec/step, elapsed 0:10:12, exp. remaining 0:18:14, complete 35.89%
att-weights epoch 183, step 368, max_size:classes 26, max_size:data 777, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.145 sec/step, elapsed 0:10:13, exp. remaining 0:18:08, complete 36.07%
att-weights epoch 183, step 369, max_size:classes 29, max_size:data 665, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.287 sec/step, elapsed 0:10:15, exp. remaining 0:18:02, complete 36.24%
att-weights epoch 183, step 370, max_size:classes 26, max_size:data 708, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.095 sec/step, elapsed 0:10:16, exp. remaining 0:17:57, complete 36.38%
att-weights epoch 183, step 371, max_size:classes 27, max_size:data 878, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.138 sec/step, elapsed 0:10:17, exp. remaining 0:17:51, complete 36.56%
att-weights epoch 183, step 372, max_size:classes 25, max_size:data 660, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.338 sec/step, elapsed 0:10:18, exp. remaining 0:17:45, complete 36.73%
att-weights epoch 183, step 373, max_size:classes 26, max_size:data 771, mem_usage:GPU:0 1.4GB, num_seqs 4, 0.944 sec/step, elapsed 0:10:19, exp. remaining 0:17:39, complete 36.91%
att-weights epoch 183, step 374, max_size:classes 24, max_size:data 915, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.061 sec/step, elapsed 0:10:20, exp. remaining 0:17:33, complete 37.08%
att-weights epoch 183, step 375, max_size:classes 24, max_size:data 601, mem_usage:GPU:0 1.4GB, num_seqs 5, 0.988 sec/step, elapsed 0:10:21, exp. remaining 0:17:28, complete 37.22%
att-weights epoch 183, step 376, max_size:classes 23, max_size:data 842, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.148 sec/step, elapsed 0:10:22, exp. remaining 0:17:21, complete 37.43%
att-weights epoch 183, step 377, max_size:classes 24, max_size:data 745, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.515 sec/step, elapsed 0:10:24, exp. remaining 0:17:17, complete 37.57%
att-weights epoch 183, step 378, max_size:classes 24, max_size:data 798, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.666 sec/step, elapsed 0:10:26, exp. remaining 0:17:12, complete 37.74%
att-weights epoch 183, step 379, max_size:classes 25, max_size:data 911, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.251 sec/step, elapsed 0:10:27, exp. remaining 0:17:07, complete 37.92%
att-weights epoch 183, step 380, max_size:classes 23, max_size:data 760, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.332 sec/step, elapsed 0:10:28, exp. remaining 0:17:00, complete 38.13%
att-weights epoch 183, step 381, max_size:classes 24, max_size:data 762, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.239 sec/step, elapsed 0:10:30, exp. remaining 0:16:54, complete 38.30%
att-weights epoch 183, step 382, max_size:classes 24, max_size:data 711, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.976 sec/step, elapsed 0:10:31, exp. remaining 0:16:50, complete 38.48%
att-weights epoch 183, step 383, max_size:classes 28, max_size:data 635, mem_usage:GPU:0 1.4GB, num_seqs 5, 0.976 sec/step, elapsed 0:10:32, exp. remaining 0:16:44, complete 38.65%
att-weights epoch 183, step 384, max_size:classes 26, max_size:data 849, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.699 sec/step, elapsed 0:10:34, exp. remaining 0:16:39, complete 38.83%
att-weights epoch 183, step 385, max_size:classes 24, max_size:data 628, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.878 sec/step, elapsed 0:10:36, exp. remaining 0:16:37, complete 38.97%
att-weights epoch 183, step 386, max_size:classes 23, max_size:data 819, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.194 sec/step, elapsed 0:10:37, exp. remaining 0:16:31, complete 39.14%
att-weights epoch 183, step 387, max_size:classes 24, max_size:data 701, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.263 sec/step, elapsed 0:10:38, exp. remaining 0:16:24, complete 39.35%
att-weights epoch 183, step 388, max_size:classes 23, max_size:data 681, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.007 sec/step, elapsed 0:10:40, exp. remaining 0:16:20, complete 39.49%
att-weights epoch 183, step 389, max_size:classes 24, max_size:data 577, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.216 sec/step, elapsed 0:10:41, exp. remaining 0:16:13, complete 39.70%
att-weights epoch 183, step 390, max_size:classes 24, max_size:data 765, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.288 sec/step, elapsed 0:10:42, exp. remaining 0:16:08, complete 39.87%
att-weights epoch 183, step 391, max_size:classes 24, max_size:data 746, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.155 sec/step, elapsed 0:10:43, exp. remaining 0:16:03, complete 40.05%
att-weights epoch 183, step 392, max_size:classes 25, max_size:data 697, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.485 sec/step, elapsed 0:10:45, exp. remaining 0:15:58, complete 40.22%
att-weights epoch 183, step 393, max_size:classes 26, max_size:data 676, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.259 sec/step, elapsed 0:10:46, exp. remaining 0:15:55, complete 40.36%
att-weights epoch 183, step 394, max_size:classes 29, max_size:data 844, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.040 sec/step, elapsed 0:10:47, exp. remaining 0:15:48, complete 40.57%
att-weights epoch 183, step 395, max_size:classes 21, max_size:data 754, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.448 sec/step, elapsed 0:10:48, exp. remaining 0:15:43, complete 40.75%
att-weights epoch 183, step 396, max_size:classes 24, max_size:data 626, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.403 sec/step, elapsed 0:10:50, exp. remaining 0:15:37, complete 40.96%
att-weights epoch 183, step 397, max_size:classes 22, max_size:data 827, mem_usage:GPU:0 1.4GB, num_seqs 4, 0.904 sec/step, elapsed 0:10:51, exp. remaining 0:15:32, complete 41.13%
att-weights epoch 183, step 398, max_size:classes 24, max_size:data 627, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.300 sec/step, elapsed 0:10:52, exp. remaining 0:15:27, complete 41.31%
att-weights epoch 183, step 399, max_size:classes 27, max_size:data 786, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.278 sec/step, elapsed 0:10:53, exp. remaining 0:15:23, complete 41.45%
att-weights epoch 183, step 400, max_size:classes 23, max_size:data 796, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.175 sec/step, elapsed 0:10:54, exp. remaining 0:15:18, complete 41.62%
att-weights epoch 183, step 401, max_size:classes 22, max_size:data 763, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.388 sec/step, elapsed 0:10:56, exp. remaining 0:15:12, complete 41.83%
att-weights epoch 183, step 402, max_size:classes 25, max_size:data 861, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.040 sec/step, elapsed 0:10:57, exp. remaining 0:15:07, complete 42.00%
att-weights epoch 183, step 403, max_size:classes 22, max_size:data 664, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.423 sec/step, elapsed 0:10:58, exp. remaining 0:15:03, complete 42.18%
att-weights epoch 183, step 404, max_size:classes 24, max_size:data 754, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.574 sec/step, elapsed 0:11:00, exp. remaining 0:15:00, complete 42.32%
att-weights epoch 183, step 405, max_size:classes 26, max_size:data 627, mem_usage:GPU:0 1.4GB, num_seqs 6, 2.914 sec/step, elapsed 0:11:03, exp. remaining 0:14:56, complete 42.53%
att-weights epoch 183, step 406, max_size:classes 23, max_size:data 706, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.189 sec/step, elapsed 0:11:04, exp. remaining 0:14:51, complete 42.70%
att-weights epoch 183, step 407, max_size:classes 22, max_size:data 750, mem_usage:GPU:0 1.4GB, num_seqs 5, 3.009 sec/step, elapsed 0:11:07, exp. remaining 0:14:49, complete 42.88%
att-weights epoch 183, step 408, max_size:classes 25, max_size:data 809, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.440 sec/step, elapsed 0:11:08, exp. remaining 0:14:44, complete 43.05%
att-weights epoch 183, step 409, max_size:classes 20, max_size:data 705, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.655 sec/step, elapsed 0:11:10, exp. remaining 0:14:39, complete 43.26%
att-weights epoch 183, step 410, max_size:classes 21, max_size:data 657, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.533 sec/step, elapsed 0:11:12, exp. remaining 0:14:36, complete 43.40%
att-weights epoch 183, step 411, max_size:classes 24, max_size:data 759, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.382 sec/step, elapsed 0:11:13, exp. remaining 0:14:32, complete 43.58%
att-weights epoch 183, step 412, max_size:classes 22, max_size:data 767, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.326 sec/step, elapsed 0:11:14, exp. remaining 0:14:28, complete 43.72%
att-weights epoch 183, step 413, max_size:classes 22, max_size:data 864, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.090 sec/step, elapsed 0:11:15, exp. remaining 0:14:22, complete 43.92%
att-weights epoch 183, step 414, max_size:classes 23, max_size:data 655, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.108 sec/step, elapsed 0:11:17, exp. remaining 0:14:17, complete 44.13%
att-weights epoch 183, step 415, max_size:classes 21, max_size:data 776, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.431 sec/step, elapsed 0:11:18, exp. remaining 0:14:12, complete 44.31%
att-weights epoch 183, step 416, max_size:classes 23, max_size:data 773, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.948 sec/step, elapsed 0:11:20, exp. remaining 0:14:09, complete 44.48%
att-weights epoch 183, step 417, max_size:classes 22, max_size:data 723, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.251 sec/step, elapsed 0:11:21, exp. remaining 0:14:03, complete 44.69%
att-weights epoch 183, step 418, max_size:classes 25, max_size:data 639, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.703 sec/step, elapsed 0:11:23, exp. remaining 0:13:59, complete 44.87%
att-weights epoch 183, step 419, max_size:classes 23, max_size:data 946, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.197 sec/step, elapsed 0:11:24, exp. remaining 0:13:55, complete 45.04%
att-weights epoch 183, step 420, max_size:classes 26, max_size:data 768, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.397 sec/step, elapsed 0:11:25, exp. remaining 0:13:49, complete 45.25%
att-weights epoch 183, step 421, max_size:classes 21, max_size:data 815, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.265 sec/step, elapsed 0:11:27, exp. remaining 0:13:45, complete 45.43%
att-weights epoch 183, step 422, max_size:classes 23, max_size:data 623, mem_usage:GPU:0 1.4GB, num_seqs 6, 2.306 sec/step, elapsed 0:11:29, exp. remaining 0:13:42, complete 45.60%
att-weights epoch 183, step 423, max_size:classes 22, max_size:data 636, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.410 sec/step, elapsed 0:11:30, exp. remaining 0:13:38, complete 45.78%
att-weights epoch 183, step 424, max_size:classes 20, max_size:data 630, mem_usage:GPU:0 1.4GB, num_seqs 5, 6.568 sec/step, elapsed 0:11:37, exp. remaining 0:13:40, complete 45.95%
att-weights epoch 183, step 425, max_size:classes 21, max_size:data 722, mem_usage:GPU:0 1.4GB, num_seqs 5, 5.220 sec/step, elapsed 0:11:42, exp. remaining 0:13:40, complete 46.12%
att-weights epoch 183, step 426, max_size:classes 21, max_size:data 647, mem_usage:GPU:0 1.4GB, num_seqs 6, 2.230 sec/step, elapsed 0:11:44, exp. remaining 0:13:36, complete 46.33%
att-weights epoch 183, step 427, max_size:classes 22, max_size:data 707, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.317 sec/step, elapsed 0:11:46, exp. remaining 0:13:32, complete 46.51%
att-weights epoch 183, step 428, max_size:classes 21, max_size:data 735, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.834 sec/step, elapsed 0:11:48, exp. remaining 0:13:26, complete 46.75%
att-weights epoch 183, step 429, max_size:classes 24, max_size:data 598, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.488 sec/step, elapsed 0:11:49, exp. remaining 0:13:22, complete 46.93%
att-weights epoch 183, step 430, max_size:classes 20, max_size:data 770, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.440 sec/step, elapsed 0:11:51, exp. remaining 0:13:18, complete 47.10%
att-weights epoch 183, step 431, max_size:classes 23, max_size:data 696, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.133 sec/step, elapsed 0:11:52, exp. remaining 0:13:16, complete 47.21%
att-weights epoch 183, step 432, max_size:classes 19, max_size:data 708, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.184 sec/step, elapsed 0:11:53, exp. remaining 0:13:11, complete 47.42%
att-weights epoch 183, step 433, max_size:classes 20, max_size:data 652, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.451 sec/step, elapsed 0:11:54, exp. remaining 0:13:07, complete 47.59%
att-weights epoch 183, step 434, max_size:classes 21, max_size:data 700, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.431 sec/step, elapsed 0:11:56, exp. remaining 0:13:03, complete 47.77%
att-weights epoch 183, step 435, max_size:classes 22, max_size:data 621, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.572 sec/step, elapsed 0:11:57, exp. remaining 0:12:57, complete 48.01%
att-weights epoch 183, step 436, max_size:classes 21, max_size:data 776, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.031 sec/step, elapsed 0:11:58, exp. remaining 0:12:50, complete 48.25%
att-weights epoch 183, step 437, max_size:classes 20, max_size:data 547, mem_usage:GPU:0 1.4GB, num_seqs 7, 1.466 sec/step, elapsed 0:12:00, exp. remaining 0:12:45, complete 48.46%
att-weights epoch 183, step 438, max_size:classes 21, max_size:data 777, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.530 sec/step, elapsed 0:12:01, exp. remaining 0:12:43, complete 48.60%
att-weights epoch 183, step 439, max_size:classes 21, max_size:data 653, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.170 sec/step, elapsed 0:12:03, exp. remaining 0:12:38, complete 48.81%
att-weights epoch 183, step 440, max_size:classes 22, max_size:data 1093, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.238 sec/step, elapsed 0:12:04, exp. remaining 0:12:33, complete 49.02%
att-weights epoch 183, step 441, max_size:classes 20, max_size:data 522, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.288 sec/step, elapsed 0:12:05, exp. remaining 0:12:29, complete 49.20%
att-weights epoch 183, step 442, max_size:classes 20, max_size:data 606, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.132 sec/step, elapsed 0:12:06, exp. remaining 0:12:24, complete 49.41%
att-weights epoch 183, step 443, max_size:classes 22, max_size:data 751, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.568 sec/step, elapsed 0:12:08, exp. remaining 0:12:19, complete 49.62%
att-weights epoch 183, step 444, max_size:classes 21, max_size:data 566, mem_usage:GPU:0 1.4GB, num_seqs 7, 2.078 sec/step, elapsed 0:12:10, exp. remaining 0:12:15, complete 49.83%
att-weights epoch 183, step 445, max_size:classes 18, max_size:data 570, mem_usage:GPU:0 1.4GB, num_seqs 7, 1.385 sec/step, elapsed 0:12:11, exp. remaining 0:12:10, complete 50.03%
att-weights epoch 183, step 446, max_size:classes 21, max_size:data 604, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.768 sec/step, elapsed 0:12:13, exp. remaining 0:12:07, complete 50.21%
att-weights epoch 183, step 447, max_size:classes 20, max_size:data 814, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.466 sec/step, elapsed 0:12:14, exp. remaining 0:12:01, complete 50.45%
att-weights epoch 183, step 448, max_size:classes 19, max_size:data 592, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.374 sec/step, elapsed 0:12:16, exp. remaining 0:11:57, complete 50.66%
att-weights epoch 183, step 449, max_size:classes 20, max_size:data 620, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.462 sec/step, elapsed 0:12:17, exp. remaining 0:11:52, complete 50.87%
att-weights epoch 183, step 450, max_size:classes 20, max_size:data 697, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.438 sec/step, elapsed 0:12:19, exp. remaining 0:11:47, complete 51.08%
att-weights epoch 183, step 451, max_size:classes 21, max_size:data 641, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.521 sec/step, elapsed 0:12:20, exp. remaining 0:11:43, complete 51.29%
att-weights epoch 183, step 452, max_size:classes 19, max_size:data 591, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.035 sec/step, elapsed 0:12:21, exp. remaining 0:11:39, complete 51.47%
att-weights epoch 183, step 453, max_size:classes 22, max_size:data 654, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.416 sec/step, elapsed 0:12:23, exp. remaining 0:11:35, complete 51.64%
att-weights epoch 183, step 454, max_size:classes 20, max_size:data 599, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.362 sec/step, elapsed 0:12:24, exp. remaining 0:11:32, complete 51.82%
att-weights epoch 183, step 455, max_size:classes 23, max_size:data 677, mem_usage:GPU:0 1.4GB, num_seqs 5, 0.906 sec/step, elapsed 0:12:25, exp. remaining 0:11:27, complete 52.03%
att-weights epoch 183, step 456, max_size:classes 20, max_size:data 558, mem_usage:GPU:0 1.4GB, num_seqs 7, 1.366 sec/step, elapsed 0:12:26, exp. remaining 0:11:22, complete 52.23%
att-weights epoch 183, step 457, max_size:classes 17, max_size:data 612, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.349 sec/step, elapsed 0:12:28, exp. remaining 0:11:19, complete 52.41%
att-weights epoch 183, step 458, max_size:classes 20, max_size:data 643, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.320 sec/step, elapsed 0:12:29, exp. remaining 0:11:14, complete 52.62%
att-weights epoch 183, step 459, max_size:classes 21, max_size:data 585, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.209 sec/step, elapsed 0:12:30, exp. remaining 0:11:10, complete 52.83%
att-weights epoch 183, step 460, max_size:classes 20, max_size:data 606, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.315 sec/step, elapsed 0:12:32, exp. remaining 0:11:05, complete 53.04%
att-weights epoch 183, step 461, max_size:classes 20, max_size:data 681, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.326 sec/step, elapsed 0:12:33, exp. remaining 0:11:00, complete 53.28%
att-weights epoch 183, step 462, max_size:classes 24, max_size:data 620, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.102 sec/step, elapsed 0:12:34, exp. remaining 0:10:55, complete 53.49%
att-weights epoch 183, step 463, max_size:classes 19, max_size:data 683, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.343 sec/step, elapsed 0:12:35, exp. remaining 0:10:51, complete 53.70%
att-weights epoch 183, step 464, max_size:classes 20, max_size:data 530, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.542 sec/step, elapsed 0:12:37, exp. remaining 0:10:50, complete 53.81%
att-weights epoch 183, step 465, max_size:classes 20, max_size:data 596, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.409 sec/step, elapsed 0:12:38, exp. remaining 0:10:45, complete 54.05%
att-weights epoch 183, step 466, max_size:classes 19, max_size:data 670, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.465 sec/step, elapsed 0:12:40, exp. remaining 0:10:39, complete 54.29%
att-weights epoch 183, step 467, max_size:classes 20, max_size:data 592, mem_usage:GPU:0 1.4GB, num_seqs 6, 0.989 sec/step, elapsed 0:12:41, exp. remaining 0:10:35, complete 54.50%
att-weights epoch 183, step 468, max_size:classes 19, max_size:data 645, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.567 sec/step, elapsed 0:12:42, exp. remaining 0:10:31, complete 54.71%
att-weights epoch 183, step 469, max_size:classes 22, max_size:data 573, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.584 sec/step, elapsed 0:12:44, exp. remaining 0:10:28, complete 54.89%
att-weights epoch 183, step 470, max_size:classes 18, max_size:data 497, mem_usage:GPU:0 1.4GB, num_seqs 7, 1.631 sec/step, elapsed 0:12:45, exp. remaining 0:10:24, complete 55.10%
att-weights epoch 183, step 471, max_size:classes 21, max_size:data 622, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.194 sec/step, elapsed 0:12:47, exp. remaining 0:10:19, complete 55.34%
att-weights epoch 183, step 472, max_size:classes 19, max_size:data 625, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.422 sec/step, elapsed 0:12:48, exp. remaining 0:10:14, complete 55.55%
att-weights epoch 183, step 473, max_size:classes 20, max_size:data 1012, mem_usage:GPU:0 1.4GB, num_seqs 3, 1.144 sec/step, elapsed 0:12:49, exp. remaining 0:10:11, complete 55.73%
att-weights epoch 183, step 474, max_size:classes 18, max_size:data 554, mem_usage:GPU:0 1.4GB, num_seqs 7, 1.210 sec/step, elapsed 0:12:50, exp. remaining 0:10:06, complete 55.97%
att-weights epoch 183, step 475, max_size:classes 20, max_size:data 548, mem_usage:GPU:0 1.4GB, num_seqs 7, 1.299 sec/step, elapsed 0:12:52, exp. remaining 0:10:02, complete 56.18%
att-weights epoch 183, step 476, max_size:classes 23, max_size:data 616, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.390 sec/step, elapsed 0:12:53, exp. remaining 0:09:59, complete 56.35%
att-weights epoch 183, step 477, max_size:classes 18, max_size:data 606, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.459 sec/step, elapsed 0:12:55, exp. remaining 0:09:56, complete 56.53%
att-weights epoch 183, step 478, max_size:classes 20, max_size:data 686, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.841 sec/step, elapsed 0:12:56, exp. remaining 0:09:52, complete 56.74%
att-weights epoch 183, step 479, max_size:classes 18, max_size:data 619, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.498 sec/step, elapsed 0:12:58, exp. remaining 0:09:48, complete 56.95%
att-weights epoch 183, step 480, max_size:classes 16, max_size:data 548, mem_usage:GPU:0 1.4GB, num_seqs 7, 1.396 sec/step, elapsed 0:12:59, exp. remaining 0:09:43, complete 57.19%
att-weights epoch 183, step 481, max_size:classes 18, max_size:data 578, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.198 sec/step, elapsed 0:13:01, exp. remaining 0:09:40, complete 57.37%
att-weights epoch 183, step 482, max_size:classes 18, max_size:data 693, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.409 sec/step, elapsed 0:13:02, exp. remaining 0:09:37, complete 57.54%
att-weights epoch 183, step 483, max_size:classes 18, max_size:data 523, mem_usage:GPU:0 1.4GB, num_seqs 7, 1.595 sec/step, elapsed 0:13:04, exp. remaining 0:09:32, complete 57.79%
att-weights epoch 183, step 484, max_size:classes 21, max_size:data 616, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.399 sec/step, elapsed 0:13:05, exp. remaining 0:09:28, complete 58.03%
att-weights epoch 183, step 485, max_size:classes 19, max_size:data 654, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.600 sec/step, elapsed 0:13:07, exp. remaining 0:09:24, complete 58.24%
att-weights epoch 183, step 486, max_size:classes 18, max_size:data 713, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.653 sec/step, elapsed 0:13:08, exp. remaining 0:09:19, complete 58.52%
att-weights epoch 183, step 487, max_size:classes 17, max_size:data 513, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.385 sec/step, elapsed 0:13:10, exp. remaining 0:09:16, complete 58.66%
att-weights epoch 183, step 488, max_size:classes 19, max_size:data 604, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.397 sec/step, elapsed 0:13:11, exp. remaining 0:09:13, complete 58.87%
att-weights epoch 183, step 489, max_size:classes 19, max_size:data 567, mem_usage:GPU:0 1.4GB, num_seqs 7, 1.561 sec/step, elapsed 0:13:13, exp. remaining 0:09:09, complete 59.08%
att-weights epoch 183, step 490, max_size:classes 17, max_size:data 711, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.302 sec/step, elapsed 0:13:14, exp. remaining 0:09:03, complete 59.36%
att-weights epoch 183, step 491, max_size:classes 17, max_size:data 696, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.415 sec/step, elapsed 0:13:15, exp. remaining 0:09:00, complete 59.57%
att-weights epoch 183, step 492, max_size:classes 18, max_size:data 525, mem_usage:GPU:0 1.4GB, num_seqs 7, 2.057 sec/step, elapsed 0:13:17, exp. remaining 0:08:56, complete 59.78%
att-weights epoch 183, step 493, max_size:classes 16, max_size:data 517, mem_usage:GPU:0 1.4GB, num_seqs 7, 1.476 sec/step, elapsed 0:13:19, exp. remaining 0:08:53, complete 59.99%
att-weights epoch 183, step 494, max_size:classes 17, max_size:data 656, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.691 sec/step, elapsed 0:13:20, exp. remaining 0:08:48, complete 60.23%
att-weights epoch 183, step 495, max_size:classes 17, max_size:data 500, mem_usage:GPU:0 1.4GB, num_seqs 8, 1.758 sec/step, elapsed 0:13:22, exp. remaining 0:08:46, complete 60.41%
att-weights epoch 183, step 496, max_size:classes 17, max_size:data 805, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.010 sec/step, elapsed 0:13:23, exp. remaining 0:08:43, complete 60.58%
att-weights epoch 183, step 497, max_size:classes 15, max_size:data 607, mem_usage:GPU:0 1.4GB, num_seqs 6, 0.994 sec/step, elapsed 0:13:24, exp. remaining 0:08:39, complete 60.79%
att-weights epoch 183, step 498, max_size:classes 21, max_size:data 612, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.215 sec/step, elapsed 0:13:25, exp. remaining 0:08:36, complete 60.93%
att-weights epoch 183, step 499, max_size:classes 18, max_size:data 498, mem_usage:GPU:0 1.4GB, num_seqs 8, 1.420 sec/step, elapsed 0:13:27, exp. remaining 0:08:34, complete 61.07%
att-weights epoch 183, step 500, max_size:classes 16, max_size:data 524, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.459 sec/step, elapsed 0:13:28, exp. remaining 0:08:31, complete 61.28%
att-weights epoch 183, step 501, max_size:classes 16, max_size:data 662, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.565 sec/step, elapsed 0:13:30, exp. remaining 0:08:27, complete 61.49%
att-weights epoch 183, step 502, max_size:classes 17, max_size:data 609, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.534 sec/step, elapsed 0:13:31, exp. remaining 0:08:22, complete 61.77%
att-weights epoch 183, step 503, max_size:classes 18, max_size:data 546, mem_usage:GPU:0 1.4GB, num_seqs 7, 1.548 sec/step, elapsed 0:13:33, exp. remaining 0:08:19, complete 61.98%
att-weights epoch 183, step 504, max_size:classes 17, max_size:data 612, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.451 sec/step, elapsed 0:13:34, exp. remaining 0:08:15, complete 62.19%
att-weights epoch 183, step 505, max_size:classes 16, max_size:data 710, mem_usage:GPU:0 1.4GB, num_seqs 5, 2.376 sec/step, elapsed 0:13:37, exp. remaining 0:08:11, complete 62.43%
att-weights epoch 183, step 506, max_size:classes 15, max_size:data 595, mem_usage:GPU:0 1.4GB, num_seqs 6, 2.104 sec/step, elapsed 0:13:39, exp. remaining 0:08:07, complete 62.71%
att-weights epoch 183, step 507, max_size:classes 17, max_size:data 533, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.287 sec/step, elapsed 0:13:40, exp. remaining 0:08:03, complete 62.92%
att-weights epoch 183, step 508, max_size:classes 17, max_size:data 821, mem_usage:GPU:0 1.4GB, num_seqs 4, 1.435 sec/step, elapsed 0:13:42, exp. remaining 0:08:00, complete 63.13%
att-weights epoch 183, step 509, max_size:classes 19, max_size:data 601, mem_usage:GPU:0 1.4GB, num_seqs 6, 2.273 sec/step, elapsed 0:13:44, exp. remaining 0:07:55, complete 63.44%
att-weights epoch 183, step 510, max_size:classes 18, max_size:data 584, mem_usage:GPU:0 1.4GB, num_seqs 6, 2.156 sec/step, elapsed 0:13:46, exp. remaining 0:07:51, complete 63.69%
att-weights epoch 183, step 511, max_size:classes 18, max_size:data 478, mem_usage:GPU:0 1.4GB, num_seqs 8, 4.050 sec/step, elapsed 0:13:50, exp. remaining 0:07:48, complete 63.93%
att-weights epoch 183, step 512, max_size:classes 16, max_size:data 556, mem_usage:GPU:0 1.4GB, num_seqs 6, 5.017 sec/step, elapsed 0:13:55, exp. remaining 0:07:45, complete 64.21%
att-weights epoch 183, step 513, max_size:classes 17, max_size:data 616, mem_usage:GPU:0 1.4GB, num_seqs 6, 3.931 sec/step, elapsed 0:13:59, exp. remaining 0:07:43, complete 64.42%
att-weights epoch 183, step 514, max_size:classes 16, max_size:data 561, mem_usage:GPU:0 1.4GB, num_seqs 7, 31.581 sec/step, elapsed 0:14:31, exp. remaining 0:07:56, complete 64.66%
att-weights epoch 183, step 515, max_size:classes 19, max_size:data 487, mem_usage:GPU:0 1.4GB, num_seqs 8, 3.450 sec/step, elapsed 0:14:34, exp. remaining 0:07:52, complete 64.94%
att-weights epoch 183, step 516, max_size:classes 16, max_size:data 636, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.563 sec/step, elapsed 0:14:36, exp. remaining 0:07:47, complete 65.19%
att-weights epoch 183, step 517, max_size:classes 17, max_size:data 659, mem_usage:GPU:0 1.4GB, num_seqs 6, 2.524 sec/step, elapsed 0:14:38, exp. remaining 0:07:44, complete 65.40%
att-weights epoch 183, step 518, max_size:classes 16, max_size:data 418, mem_usage:GPU:0 1.4GB, num_seqs 9, 2.423 sec/step, elapsed 0:14:41, exp. remaining 0:07:40, complete 65.68%
att-weights epoch 183, step 519, max_size:classes 17, max_size:data 546, mem_usage:GPU:0 1.4GB, num_seqs 7, 1.606 sec/step, elapsed 0:14:42, exp. remaining 0:07:37, complete 65.89%
att-weights epoch 183, step 520, max_size:classes 16, max_size:data 564, mem_usage:GPU:0 1.4GB, num_seqs 7, 1.984 sec/step, elapsed 0:14:44, exp. remaining 0:07:33, complete 66.13%
att-weights epoch 183, step 521, max_size:classes 16, max_size:data 492, mem_usage:GPU:0 1.4GB, num_seqs 8, 2.632 sec/step, elapsed 0:14:47, exp. remaining 0:07:30, complete 66.31%
att-weights epoch 183, step 522, max_size:classes 15, max_size:data 606, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.704 sec/step, elapsed 0:14:49, exp. remaining 0:07:27, complete 66.52%
att-weights epoch 183, step 523, max_size:classes 14, max_size:data 555, mem_usage:GPU:0 1.4GB, num_seqs 7, 2.152 sec/step, elapsed 0:14:51, exp. remaining 0:07:23, complete 66.76%
att-weights epoch 183, step 524, max_size:classes 17, max_size:data 455, mem_usage:GPU:0 1.4GB, num_seqs 8, 1.931 sec/step, elapsed 0:14:53, exp. remaining 0:07:19, complete 67.04%
att-weights epoch 183, step 525, max_size:classes 22, max_size:data 520, mem_usage:GPU:0 1.4GB, num_seqs 7, 1.761 sec/step, elapsed 0:14:54, exp. remaining 0:07:15, complete 67.28%
att-weights epoch 183, step 526, max_size:classes 20, max_size:data 587, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.528 sec/step, elapsed 0:14:56, exp. remaining 0:07:11, complete 67.53%
att-weights epoch 183, step 527, max_size:classes 16, max_size:data 464, mem_usage:GPU:0 1.4GB, num_seqs 8, 2.375 sec/step, elapsed 0:14:58, exp. remaining 0:07:07, complete 67.77%
att-weights epoch 183, step 528, max_size:classes 16, max_size:data 572, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.580 sec/step, elapsed 0:15:00, exp. remaining 0:07:04, complete 67.98%
att-weights epoch 183, step 529, max_size:classes 15, max_size:data 502, mem_usage:GPU:0 1.4GB, num_seqs 7, 2.540 sec/step, elapsed 0:15:02, exp. remaining 0:06:59, complete 68.30%
att-weights epoch 183, step 530, max_size:classes 21, max_size:data 679, mem_usage:GPU:0 1.4GB, num_seqs 5, 1.521 sec/step, elapsed 0:15:04, exp. remaining 0:06:54, complete 68.58%
att-weights epoch 183, step 531, max_size:classes 17, max_size:data 648, mem_usage:GPU:0 1.4GB, num_seqs 6, 7.421 sec/step, elapsed 0:15:11, exp. remaining 0:06:53, complete 68.82%
att-weights epoch 183, step 532, max_size:classes 15, max_size:data 505, mem_usage:GPU:0 1.4GB, num_seqs 7, 1.378 sec/step, elapsed 0:15:13, exp. remaining 0:06:49, complete 69.06%
att-weights epoch 183, step 533, max_size:classes 14, max_size:data 462, mem_usage:GPU:0 1.4GB, num_seqs 8, 1.905 sec/step, elapsed 0:15:15, exp. remaining 0:06:45, complete 69.31%
att-weights epoch 183, step 534, max_size:classes 22, max_size:data 539, mem_usage:GPU:0 1.4GB, num_seqs 7, 1.902 sec/step, elapsed 0:15:17, exp. remaining 0:06:40, complete 69.62%
att-weights epoch 183, step 535, max_size:classes 16, max_size:data 515, mem_usage:GPU:0 1.4GB, num_seqs 7, 1.447 sec/step, elapsed 0:15:18, exp. remaining 0:06:35, complete 69.90%
att-weights epoch 183, step 536, max_size:classes 16, max_size:data 557, mem_usage:GPU:0 1.4GB, num_seqs 7, 1.509 sec/step, elapsed 0:15:20, exp. remaining 0:06:30, complete 70.18%
att-weights epoch 183, step 537, max_size:classes 15, max_size:data 611, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.575 sec/step, elapsed 0:15:21, exp. remaining 0:06:26, complete 70.46%
att-weights epoch 183, step 538, max_size:classes 16, max_size:data 423, mem_usage:GPU:0 1.4GB, num_seqs 9, 1.541 sec/step, elapsed 0:15:23, exp. remaining 0:06:21, complete 70.74%
att-weights epoch 183, step 539, max_size:classes 14, max_size:data 471, mem_usage:GPU:0 1.4GB, num_seqs 8, 1.866 sec/step, elapsed 0:15:24, exp. remaining 0:06:17, complete 71.02%
att-weights epoch 183, step 540, max_size:classes 14, max_size:data 520, mem_usage:GPU:0 1.4GB, num_seqs 7, 1.693 sec/step, elapsed 0:15:26, exp. remaining 0:06:13, complete 71.30%
att-weights epoch 183, step 541, max_size:classes 15, max_size:data 507, mem_usage:GPU:0 1.4GB, num_seqs 7, 1.388 sec/step, elapsed 0:15:28, exp. remaining 0:06:08, complete 71.58%
att-weights epoch 183, step 542, max_size:classes 15, max_size:data 528, mem_usage:GPU:0 1.4GB, num_seqs 7, 1.568 sec/step, elapsed 0:15:29, exp. remaining 0:06:03, complete 71.89%
att-weights epoch 183, step 543, max_size:classes 14, max_size:data 402, mem_usage:GPU:0 1.4GB, num_seqs 9, 1.913 sec/step, elapsed 0:15:31, exp. remaining 0:05:57, complete 72.24%
att-weights epoch 183, step 544, max_size:classes 15, max_size:data 487, mem_usage:GPU:0 1.4GB, num_seqs 8, 1.451 sec/step, elapsed 0:15:33, exp. remaining 0:05:53, complete 72.52%
att-weights epoch 183, step 545, max_size:classes 16, max_size:data 466, mem_usage:GPU:0 1.4GB, num_seqs 8, 1.598 sec/step, elapsed 0:15:34, exp. remaining 0:05:49, complete 72.77%
att-weights epoch 183, step 546, max_size:classes 15, max_size:data 485, mem_usage:GPU:0 1.4GB, num_seqs 8, 1.571 sec/step, elapsed 0:15:36, exp. remaining 0:05:45, complete 73.04%
att-weights epoch 183, step 547, max_size:classes 14, max_size:data 490, mem_usage:GPU:0 1.4GB, num_seqs 8, 1.403 sec/step, elapsed 0:15:37, exp. remaining 0:05:40, complete 73.36%
att-weights epoch 183, step 548, max_size:classes 14, max_size:data 488, mem_usage:GPU:0 1.4GB, num_seqs 8, 1.535 sec/step, elapsed 0:15:39, exp. remaining 0:05:35, complete 73.67%
att-weights epoch 183, step 549, max_size:classes 14, max_size:data 500, mem_usage:GPU:0 1.4GB, num_seqs 8, 1.804 sec/step, elapsed 0:15:40, exp. remaining 0:05:32, complete 73.92%
att-weights epoch 183, step 550, max_size:classes 13, max_size:data 493, mem_usage:GPU:0 1.4GB, num_seqs 8, 6.680 sec/step, elapsed 0:15:47, exp. remaining 0:05:30, complete 74.16%
att-weights epoch 183, step 551, max_size:classes 16, max_size:data 415, mem_usage:GPU:0 1.4GB, num_seqs 9, 2.937 sec/step, elapsed 0:15:50, exp. remaining 0:05:24, complete 74.55%
att-weights epoch 183, step 552, max_size:classes 13, max_size:data 398, mem_usage:GPU:0 1.4GB, num_seqs 10, 7.490 sec/step, elapsed 0:15:58, exp. remaining 0:05:22, complete 74.83%
att-weights epoch 183, step 553, max_size:classes 14, max_size:data 475, mem_usage:GPU:0 1.4GB, num_seqs 8, 1.842 sec/step, elapsed 0:15:59, exp. remaining 0:05:17, complete 75.14%
att-weights epoch 183, step 554, max_size:classes 15, max_size:data 562, mem_usage:GPU:0 1.4GB, num_seqs 7, 1.468 sec/step, elapsed 0:16:01, exp. remaining 0:05:12, complete 75.45%
att-weights epoch 183, step 555, max_size:classes 14, max_size:data 463, mem_usage:GPU:0 1.4GB, num_seqs 8, 2.071 sec/step, elapsed 0:16:03, exp. remaining 0:05:09, complete 75.70%
att-weights epoch 183, step 556, max_size:classes 14, max_size:data 396, mem_usage:GPU:0 1.4GB, num_seqs 9, 1.538 sec/step, elapsed 0:16:04, exp. remaining 0:05:05, complete 75.98%
att-weights epoch 183, step 557, max_size:classes 13, max_size:data 440, mem_usage:GPU:0 1.4GB, num_seqs 9, 2.089 sec/step, elapsed 0:16:07, exp. remaining 0:05:00, complete 76.29%
att-weights epoch 183, step 558, max_size:classes 15, max_size:data 460, mem_usage:GPU:0 1.4GB, num_seqs 7, 1.499 sec/step, elapsed 0:16:08, exp. remaining 0:04:55, complete 76.61%
att-weights epoch 183, step 559, max_size:classes 16, max_size:data 534, mem_usage:GPU:0 1.4GB, num_seqs 7, 1.799 sec/step, elapsed 0:16:10, exp. remaining 0:04:51, complete 76.92%
att-weights epoch 183, step 560, max_size:classes 14, max_size:data 362, mem_usage:GPU:0 1.4GB, num_seqs 11, 2.009 sec/step, elapsed 0:16:12, exp. remaining 0:04:46, complete 77.27%
att-weights epoch 183, step 561, max_size:classes 13, max_size:data 445, mem_usage:GPU:0 1.4GB, num_seqs 8, 1.659 sec/step, elapsed 0:16:14, exp. remaining 0:04:41, complete 77.58%
att-weights epoch 183, step 562, max_size:classes 13, max_size:data 433, mem_usage:GPU:0 1.4GB, num_seqs 9, 1.589 sec/step, elapsed 0:16:15, exp. remaining 0:04:37, complete 77.83%
att-weights epoch 183, step 563, max_size:classes 13, max_size:data 444, mem_usage:GPU:0 1.4GB, num_seqs 9, 1.708 sec/step, elapsed 0:16:17, exp. remaining 0:04:34, complete 78.07%
att-weights epoch 183, step 564, max_size:classes 13, max_size:data 539, mem_usage:GPU:0 1.4GB, num_seqs 7, 1.632 sec/step, elapsed 0:16:18, exp. remaining 0:04:31, complete 78.28%
att-weights epoch 183, step 565, max_size:classes 17, max_size:data 496, mem_usage:GPU:0 1.4GB, num_seqs 8, 1.487 sec/step, elapsed 0:16:20, exp. remaining 0:04:26, complete 78.60%
att-weights epoch 183, step 566, max_size:classes 13, max_size:data 415, mem_usage:GPU:0 1.4GB, num_seqs 9, 1.663 sec/step, elapsed 0:16:22, exp. remaining 0:04:23, complete 78.88%
att-weights epoch 183, step 567, max_size:classes 13, max_size:data 426, mem_usage:GPU:0 1.4GB, num_seqs 9, 1.649 sec/step, elapsed 0:16:23, exp. remaining 0:04:19, complete 79.16%
att-weights epoch 183, step 568, max_size:classes 13, max_size:data 407, mem_usage:GPU:0 1.4GB, num_seqs 9, 1.899 sec/step, elapsed 0:16:25, exp. remaining 0:04:14, complete 79.47%
att-weights epoch 183, step 569, max_size:classes 14, max_size:data 368, mem_usage:GPU:0 1.4GB, num_seqs 10, 2.244 sec/step, elapsed 0:16:27, exp. remaining 0:04:09, complete 79.82%
att-weights epoch 183, step 570, max_size:classes 14, max_size:data 391, mem_usage:GPU:0 1.4GB, num_seqs 9, 1.515 sec/step, elapsed 0:16:29, exp. remaining 0:04:05, complete 80.13%
att-weights epoch 183, step 571, max_size:classes 13, max_size:data 539, mem_usage:GPU:0 1.4GB, num_seqs 7, 1.229 sec/step, elapsed 0:16:30, exp. remaining 0:04:00, complete 80.48%
att-weights epoch 183, step 572, max_size:classes 12, max_size:data 472, mem_usage:GPU:0 1.4GB, num_seqs 7, 1.708 sec/step, elapsed 0:16:32, exp. remaining 0:03:56, complete 80.76%
att-weights epoch 183, step 573, max_size:classes 13, max_size:data 600, mem_usage:GPU:0 1.4GB, num_seqs 6, 1.355 sec/step, elapsed 0:16:33, exp. remaining 0:03:53, complete 81.01%
att-weights epoch 183, step 574, max_size:classes 13, max_size:data 426, mem_usage:GPU:0 1.4GB, num_seqs 9, 1.725 sec/step, elapsed 0:16:35, exp. remaining 0:03:49, complete 81.28%
att-weights epoch 183, step 575, max_size:classes 11, max_size:data 469, mem_usage:GPU:0 1.4GB, num_seqs 8, 2.121 sec/step, elapsed 0:16:37, exp. remaining 0:03:45, complete 81.56%
att-weights epoch 183, step 576, max_size:classes 13, max_size:data 491, mem_usage:GPU:0 1.4GB, num_seqs 8, 2.578 sec/step, elapsed 0:16:40, exp. remaining 0:03:40, complete 81.91%
att-weights epoch 183, step 577, max_size:classes 13, max_size:data 424, mem_usage:GPU:0 1.4GB, num_seqs 9, 1.724 sec/step, elapsed 0:16:41, exp. remaining 0:03:34, complete 82.37%
att-weights epoch 183, step 578, max_size:classes 17, max_size:data 398, mem_usage:GPU:0 1.4GB, num_seqs 10, 6.427 sec/step, elapsed 0:16:48, exp. remaining 0:03:30, complete 82.72%
att-weights epoch 183, step 579, max_size:classes 14, max_size:data 415, mem_usage:GPU:0 1.4GB, num_seqs 9, 2.307 sec/step, elapsed 0:16:50, exp. remaining 0:03:26, complete 83.03%
att-weights epoch 183, step 580, max_size:classes 14, max_size:data 375, mem_usage:GPU:0 1.4GB, num_seqs 10, 1.960 sec/step, elapsed 0:16:52, exp. remaining 0:03:22, complete 83.34%
att-weights epoch 183, step 581, max_size:classes 11, max_size:data 403, mem_usage:GPU:0 1.4GB, num_seqs 8, 5.282 sec/step, elapsed 0:16:57, exp. remaining 0:03:18, complete 83.66%
att-weights epoch 183, step 582, max_size:classes 14, max_size:data 505, mem_usage:GPU:0 1.4GB, num_seqs 7, 1.719 sec/step, elapsed 0:16:59, exp. remaining 0:03:13, complete 84.04%
att-weights epoch 183, step 583, max_size:classes 12, max_size:data 462, mem_usage:GPU:0 1.4GB, num_seqs 8, 1.631 sec/step, elapsed 0:17:01, exp. remaining 0:03:08, complete 84.39%
att-weights epoch 183, step 584, max_size:classes 13, max_size:data 496, mem_usage:GPU:0 1.4GB, num_seqs 8, 1.693 sec/step, elapsed 0:17:02, exp. remaining 0:03:03, complete 84.78%
att-weights epoch 183, step 585, max_size:classes 13, max_size:data 384, mem_usage:GPU:0 1.4GB, num_seqs 10, 1.777 sec/step, elapsed 0:17:04, exp. remaining 0:02:59, complete 85.09%
att-weights epoch 183, step 586, max_size:classes 11, max_size:data 291, mem_usage:GPU:0 1.4GB, num_seqs 13, 1.841 sec/step, elapsed 0:17:06, exp. remaining 0:02:54, complete 85.44%
att-weights epoch 183, step 587, max_size:classes 12, max_size:data 392, mem_usage:GPU:0 1.4GB, num_seqs 10, 1.851 sec/step, elapsed 0:17:08, exp. remaining 0:02:49, complete 85.82%
att-weights epoch 183, step 588, max_size:classes 12, max_size:data 435, mem_usage:GPU:0 1.4GB, num_seqs 9, 1.802 sec/step, elapsed 0:17:10, exp. remaining 0:02:45, complete 86.17%
att-weights epoch 183, step 589, max_size:classes 12, max_size:data 423, mem_usage:GPU:0 1.4GB, num_seqs 9, 1.849 sec/step, elapsed 0:17:11, exp. remaining 0:02:41, complete 86.49%
att-weights epoch 183, step 590, max_size:classes 13, max_size:data 405, mem_usage:GPU:0 1.4GB, num_seqs 9, 1.351 sec/step, elapsed 0:17:13, exp. remaining 0:02:37, complete 86.77%
att-weights epoch 183, step 591, max_size:classes 12, max_size:data 348, mem_usage:GPU:0 1.4GB, num_seqs 11, 1.741 sec/step, elapsed 0:17:15, exp. remaining 0:02:32, complete 87.15%
att-weights epoch 183, step 592, max_size:classes 12, max_size:data 372, mem_usage:GPU:0 1.4GB, num_seqs 10, 1.539 sec/step, elapsed 0:17:16, exp. remaining 0:02:29, complete 87.40%
att-weights epoch 183, step 593, max_size:classes 12, max_size:data 356, mem_usage:GPU:0 1.4GB, num_seqs 11, 2.007 sec/step, elapsed 0:17:18, exp. remaining 0:02:25, complete 87.71%
att-weights epoch 183, step 594, max_size:classes 12, max_size:data 422, mem_usage:GPU:0 1.4GB, num_seqs 9, 1.811 sec/step, elapsed 0:17:20, exp. remaining 0:02:20, complete 88.09%
att-weights epoch 183, step 595, max_size:classes 12, max_size:data 399, mem_usage:GPU:0 1.4GB, num_seqs 10, 1.671 sec/step, elapsed 0:17:22, exp. remaining 0:02:16, complete 88.41%
att-weights epoch 183, step 596, max_size:classes 13, max_size:data 358, mem_usage:GPU:0 1.4GB, num_seqs 11, 1.744 sec/step, elapsed 0:17:23, exp. remaining 0:02:11, complete 88.83%
att-weights epoch 183, step 597, max_size:classes 10, max_size:data 377, mem_usage:GPU:0 1.4GB, num_seqs 10, 2.273 sec/step, elapsed 0:17:26, exp. remaining 0:02:07, complete 89.14%
att-weights epoch 183, step 598, max_size:classes 11, max_size:data 388, mem_usage:GPU:0 1.4GB, num_seqs 9, 1.811 sec/step, elapsed 0:17:27, exp. remaining 0:02:03, complete 89.42%
att-weights epoch 183, step 599, max_size:classes 9, max_size:data 478, mem_usage:GPU:0 1.4GB, num_seqs 8, 1.845 sec/step, elapsed 0:17:29, exp. remaining 0:01:58, complete 89.87%
att-weights epoch 183, step 600, max_size:classes 13, max_size:data 331, mem_usage:GPU:0 1.4GB, num_seqs 11, 11.333 sec/step, elapsed 0:17:41, exp. remaining 0:01:53, complete 90.33%
att-weights epoch 183, step 601, max_size:classes 10, max_size:data 513, mem_usage:GPU:0 1.4GB, num_seqs 7, 2.196 sec/step, elapsed 0:17:43, exp. remaining 0:01:48, complete 90.71%
att-weights epoch 183, step 602, max_size:classes 14, max_size:data 420, mem_usage:GPU:0 1.4GB, num_seqs 9, 12.706 sec/step, elapsed 0:17:56, exp. remaining 0:01:45, complete 91.06%
att-weights epoch 183, step 603, max_size:classes 11, max_size:data 347, mem_usage:GPU:0 1.4GB, num_seqs 11, 1.864 sec/step, elapsed 0:17:57, exp. remaining 0:01:40, complete 91.48%
att-weights epoch 183, step 604, max_size:classes 11, max_size:data 434, mem_usage:GPU:0 1.4GB, num_seqs 9, 1.900 sec/step, elapsed 0:17:59, exp. remaining 0:01:35, complete 91.86%
att-weights epoch 183, step 605, max_size:classes 13, max_size:data 332, mem_usage:GPU:0 1.4GB, num_seqs 12, 2.307 sec/step, elapsed 0:18:02, exp. remaining 0:01:30, complete 92.28%
att-weights epoch 183, step 606, max_size:classes 11, max_size:data 359, mem_usage:GPU:0 1.4GB, num_seqs 9, 2.234 sec/step, elapsed 0:18:04, exp. remaining 0:01:25, complete 92.67%
att-weights epoch 183, step 607, max_size:classes 10, max_size:data 448, mem_usage:GPU:0 1.4GB, num_seqs 8, 2.675 sec/step, elapsed 0:18:07, exp. remaining 0:01:21, complete 93.05%
att-weights epoch 183, step 608, max_size:classes 11, max_size:data 297, mem_usage:GPU:0 1.4GB, num_seqs 13, 2.368 sec/step, elapsed 0:18:09, exp. remaining 0:01:16, complete 93.47%
att-weights epoch 183, step 609, max_size:classes 16, max_size:data 296, mem_usage:GPU:0 1.4GB, num_seqs 13, 1.795 sec/step, elapsed 0:18:11, exp. remaining 0:01:11, complete 93.85%
att-weights epoch 183, step 610, max_size:classes 9, max_size:data 339, mem_usage:GPU:0 1.4GB, num_seqs 11, 2.213 sec/step, elapsed 0:18:13, exp. remaining 0:01:06, complete 94.24%
att-weights epoch 183, step 611, max_size:classes 9, max_size:data 385, mem_usage:GPU:0 1.4GB, num_seqs 10, 1.837 sec/step, elapsed 0:18:15, exp. remaining 0:01:02, complete 94.59%
att-weights epoch 183, step 612, max_size:classes 9, max_size:data 319, mem_usage:GPU:0 1.4GB, num_seqs 12, 2.192 sec/step, elapsed 0:18:17, exp. remaining 0:00:57, complete 95.01%
att-weights epoch 183, step 613, max_size:classes 11, max_size:data 349, mem_usage:GPU:0 1.4GB, num_seqs 11, 2.290 sec/step, elapsed 0:18:19, exp. remaining 0:00:53, complete 95.39%
att-weights epoch 183, step 614, max_size:classes 10, max_size:data 328, mem_usage:GPU:0 1.4GB, num_seqs 12, 2.381 sec/step, elapsed 0:18:22, exp. remaining 0:00:47, complete 95.84%
att-weights epoch 183, step 615, max_size:classes 10, max_size:data 319, mem_usage:GPU:0 1.4GB, num_seqs 11, 2.200 sec/step, elapsed 0:18:24, exp. remaining 0:00:42, complete 96.26%
att-weights epoch 183, step 616, max_size:classes 9, max_size:data 337, mem_usage:GPU:0 1.4GB, num_seqs 11, 1.937 sec/step, elapsed 0:18:26, exp. remaining 0:00:36, complete 96.79%
att-weights epoch 183, step 617, max_size:classes 9, max_size:data 323, mem_usage:GPU:0 1.4GB, num_seqs 12, 1.703 sec/step, elapsed 0:18:27, exp. remaining 0:00:32, complete 97.17%
att-weights epoch 183, step 618, max_size:classes 11, max_size:data 351, mem_usage:GPU:0 1.4GB, num_seqs 11, 1.858 sec/step, elapsed 0:18:29, exp. remaining 0:00:28, complete 97.52%
att-weights epoch 183, step 619, max_size:classes 9, max_size:data 348, mem_usage:GPU:0 1.4GB, num_seqs 11, 1.978 sec/step, elapsed 0:18:31, exp. remaining 0:00:23, complete 97.94%
att-weights epoch 183, step 620, max_size:classes 8, max_size:data 386, mem_usage:GPU:0 1.4GB, num_seqs 10, 1.731 sec/step, elapsed 0:18:33, exp. remaining 0:00:17, complete 98.43%
att-weights epoch 183, step 621, max_size:classes 8, max_size:data 327, mem_usage:GPU:0 1.4GB, num_seqs 12, 3.119 sec/step, elapsed 0:18:36, exp. remaining 0:00:12, complete 98.92%
att-weights epoch 183, step 622, max_size:classes 9, max_size:data 340, mem_usage:GPU:0 1.4GB, num_seqs 11, 1.409 sec/step, elapsed 0:18:38, exp. remaining 0:00:07, complete 99.37%
att-weights epoch 183, step 623, max_size:classes 7, max_size:data 303, mem_usage:GPU:0 1.4GB, num_seqs 13, 0.942 sec/step, elapsed 0:18:39, exp. remaining 0:00:01, complete 99.90%
att-weights epoch 183, step 624, max_size:classes 9, max_size:data 331, mem_usage:GPU:0 1.4GB, num_seqs 12, 0.998 sec/step, elapsed 0:18:40, exp. remaining 0:00:01, complete 99.90%
att-weights epoch 183, step 625, max_size:classes 7, max_size:data 248, mem_usage:GPU:0 1.4GB, num_seqs 15, 1.152 sec/step, elapsed 0:18:41, exp. remaining 0:00:01, complete 99.90%
att-weights epoch 183, step 626, max_size:classes 7, max_size:data 342, mem_usage:GPU:0 1.4GB, num_seqs 11, 0.791 sec/step, elapsed 0:18:41, exp. remaining 0:00:01, complete 99.90%
att-weights epoch 183, step 627, max_size:classes 7, max_size:data 369, mem_usage:GPU:0 1.4GB, num_seqs 10, 0.691 sec/step, elapsed 0:18:42, exp. remaining 0:00:01, complete 99.90%
att-weights epoch 183, step 628, max_size:classes 6, max_size:data 317, mem_usage:GPU:0 1.4GB, num_seqs 12, 0.801 sec/step, elapsed 0:18:43, exp. remaining 0:00:01, complete 99.90%
att-weights epoch 183, step 629, max_size:classes 6, max_size:data 278, mem_usage:GPU:0 1.4GB, num_seqs 14, 1.028 sec/step, elapsed 0:18:44, exp. remaining 0:00:01, complete 99.90%
att-weights epoch 183, step 630, max_size:classes 6, max_size:data 282, mem_usage:GPU:0 1.4GB, num_seqs 14, 0.915 sec/step, elapsed 0:18:45, exp. remaining 0:00:01, complete 99.90%
att-weights epoch 183, step 631, max_size:classes 5, max_size:data 287, mem_usage:GPU:0 1.4GB, num_seqs 13, 0.838 sec/step, elapsed 0:18:46, exp. remaining 0:00:01, complete 99.90%
att-weights epoch 183, step 632, max_size:classes 7, max_size:data 253, mem_usage:GPU:0 1.4GB, num_seqs 15, 1.062 sec/step, elapsed 0:18:47, exp. remaining 0:00:01, complete 99.90%
att-weights epoch 183, step 633, max_size:classes 3, max_size:data 135, mem_usage:GPU:0 1.4GB, num_seqs 4, 0.338 sec/step, elapsed 0:18:47, exp. remaining 0:00:01, complete 99.90%
Stats:
  mem_usage:GPU:0: Stats(mean=1.4GB, std_dev=0.0B, min=1.4GB, max=1.4GB, num_seqs=634, avg_data_len=1)
att-weights epoch 183, finished after 634 steps, 0:18:47 elapsed (37.8% computing time)
Layer 'att_weights' Stats:
  2864 seqs, 11035331 total frames, 3853.118366 average frames
  Mean: 0.006072314454487043
  Std dev: 0.07148662415567758
  Min/max: 0.0 / 0.9999999
Quitting
+------- EPILOGUE SCRIPT -----------------------------------------------
|
| Job ID ..............: 9519819
| Stopped at ..........: Fri Jul  5 14:43:18 CEST 2019
| Resources requested .: h_rss=8G,h_rt=7200,gpu=1,h_vmem=1536G,pxe=ubuntu_16.04,s_core=0,num_proc=5,scratch_free=5G,h_fsize=20G
| Resources used ......: cpu=00:50:59, mem=11960.81507 GB s, io=3.11434 GB, vmem=4.137G, maxvmem=4.158G, last_file_cache=1.249G, last_rss=3M, max-cache=3.751G
| Memory used .........: 4.999G / 8.000G (62.5%)
| Total time used .....: 0:19:52
|
+------- EPILOGUE SCRIPT -----------------------------------------------
