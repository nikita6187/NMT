2019-07-02 11:39:25.907883: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-07-02 11:39:27.675216: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:03:00.0
totalMemory: 10.92GiB freeMemory: 10.76GiB
2019-07-02 11:39:27.675261: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-02 11:39:27.675274: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-02 11:39:27.675281: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-02 11:39:27.675287: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-02 11:39:27.677312: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-02 11:39:28.681728: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-02 11:39:28.681772: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-02 11:39:28.681779: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-02 11:39:28.682098: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/device:GPU:0 with 10405 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1)
[31;1mEXCEPTION[0m
[34mTraceback (most recent call last):[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/tools/[0m[36;1mget-attention-weights.py[0m[36m"[0m, [34mline [0m[35m414[0m, [34min [0m<module>
[34m    line: [0mmain[34m([0msys[34m.[0margv[34m)[0m
[34m    locals:[0m
      main [34;1m= [0m[34m<local> [0m[34m<[0mfunction main at 0x2b6d7124c7b8[34m>[0m
      sys [34;1m= [0m[34m<local> [0m[34m<[0mmodule [36m'sys'[0m [34m([0mbuilt[34m-[0m[34min[0m[34m)[0m[34m>[0m
      sys[34;1m.[0margv [34;1m= [0m[34m<local> [0m[34m[[0m[36m'/u/makarov/returnn-parnia/tools/get-attention-weights.py'[0m[34m,[0m [36m'/u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc.config'[0m[34m,[0m [36m'--epoch'[0m[34m,[0m [36m'481'[0m[34m,[0m [36m'--data'[0m[34m,[0m [36m'config:get_dataset("${DATA}")'[0m[34m,[0m [36m'--dump_dir'[0m[34m,[0m [36m'/u/makarov/makarov/NMT/hmm-fa..., len = 40, _[0]: {len = 56}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/tools/[0m[36;1mget-attention-weights.py[0m[36m"[0m, [34mline [0m[35m221[0m, [34min [0mmain
[34m    line: [0mdataset [34m=[0m init_dataset[34m([0mdataset_str[34m,[0m extra_kwargs[34m=[0mextra_dataset_kwargs[34m)[0m
[34m    locals:[0m
      dataset [34;1m= [0m[34m<not found>[0m
      init_dataset [34;1m= [0m[34m<global> [0m[34m<[0mfunction init_dataset at 0x2b6d19812268[34m>[0m
      dataset_str [34;1m= [0m[34m<local> [0m[36m'config:get_dataset("${DATA}")'[0m[34m,[0m len [34m=[0m 29
      extra_kwargs [34;1m= [0m[34m<not found>[0m
      extra_dataset_kwargs [34;1m= [0m[34m<local> [0m[34m{[0m[36m'seq_ordering'[0m[34m:[0m [36m'sorted_reverse'[0m[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/[0m[36;1mDataset.py[0m[36m"[0m, [34mline [0m[35m884[0m, [34min [0minit_dataset
[34m    line: [0m[34mreturn [0minit_dataset_via_str[34m([0mconfig_str[34m=[0mconfig_str[34m,[0m [34m*[0m[34m*[0mkwargs[34m)[0m
[34m    locals:[0m
      init_dataset_via_str [34;1m= [0m[34m<global> [0m[34m<[0mfunction init_dataset_via_str at 0x2b6d198122f0[34m>[0m
      config_str [34;1m= [0m[34m<local> [0m[36m'config:get_dataset("${DATA}")'[0m[34m,[0m len [34m=[0m 29
      kwargs [34;1m= [0m[34m<local> [0m[34m{[0m[36m'seq_ordering'[0m[34m:[0m [36m'sorted_reverse'[0m[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/[0m[36;1mDataset.py[0m[36m"[0m, [34mline [0m[35m927[0m, [34min [0minit_dataset_via_str
[34m    line: [0m[34mreturn [0minit_dataset[34m([0mdata[34m)[0m
[34m    locals:[0m
      init_dataset [34;1m= [0m[34m<global> [0m[34m<[0mfunction init_dataset at 0x2b6d19812268[34m>[0m
      data [34;1m= [0m[34m<local> [0m[34m{[0m[36m'path'[0m[34m:[0m [36m'/u/bahar/workspace/asr/librispeech/test-20190121/dataset/ogg-zips'[0m[34m,[0m [36m'use_zip'[0m[34m:[0m [34mTrue[0m[34m,[0m [36m'fixed_random_seed'[0m[34m:[0m 1[34m,[0m [36m'seq_ordering'[0m[34m:[0m [36m'sorted_reverse'[0m[34m,[0m [36m'use_cache_manager'[0m[34m:[0m [34mTrue[0m[34m,[0m [36m'prefix'[0m[34m:[0m [36m'${DATA}'[0m[34m,[0m [36m'audio'[0m[34m:[0m [34m{[0m[36m'norm_mean'[0m[34m:[0m [36m'/u/bahar/workspace/asr/librispeech/test-20190121/dataset/stats.mean.txt'[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 10
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/[0m[36;1mDataset.py[0m[36m"[0m, [34mline [0m[35m897[0m, [34min [0minit_dataset
[34m    line: [0mobj [34m=[0m clazz[34m([0m[34m*[0m[34m*[0mkwargs[34m)[0m
[34m    locals:[0m
      obj [34;1m= [0m[34m<not found>[0m
      clazz [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'GeneratingDataset.LibriSpeechCorpus'[0m[34m>[0m
      kwargs [34;1m= [0m[34m<local> [0m[34m{[0m[36m'use_cache_manager'[0m[34m:[0m [34mTrue[0m[34m,[0m [36m'prefix'[0m[34m:[0m [36m'${DATA}'[0m[34m,[0m [36m'audio'[0m[34m:[0m [34m{[0m[36m'norm_mean'[0m[34m:[0m [36m'/u/bahar/workspace/asr/librispeech/test-20190121/dataset/stats.mean.txt'[0m[34m,[0m [36m'norm_std_dev'[0m[34m:[0m [36m'/u/bahar/workspace/asr/librispeech/test-20190121/dataset/stats.std_dev.txt'[0m[34m}[0m[34m,[0m [36m'fixed_random_seed'[0m[34m:[0m 1[34m,[0m [36m'bpe'[0m[34m:[0m [34m{[0m[36m'bpe_file'[0m[34m:[0m [36m'/u/bahar/w..., len = 9[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/[0m[36;1mGeneratingDataset.py[0m[36m"[0m, [34mline [0m[35m1845[0m, [34min [0m__init__
[34m    line: [0m[34massert [0mzip_fns[34m,[0m [36m"no files found: %r"[0m [34m%[0m zip_fn_pattern
[34m    locals:[0m
      zip_fns [34;1m= [0m[34m<local> [0m[34m[[0m[34m][0m
      zip_fn_pattern [34;1m= [0m[34m<local> [0m[36m'/u/bahar/workspace/asr/librispeech/test-20190121/dataset/ogg-zips/${DATA}*.zip'[0m[34m,[0m len [34m=[0m 78
[31mAssertionError[0m: no files found: '/u/bahar/workspace/asr/librispeech/test-20190121/dataset/ogg-zips/${DATA}*.zip'
2019-07-02 12:06:50.588452: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:03:00.0
totalMemory: 10.92GiB freeMemory: 10.76GiB
2019-07-02 12:06:50.588906: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-02 12:06:50.588955: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-02 12:06:50.588968: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-02 12:06:50.588980: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-02 12:06:50.592868: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-02 12:06:51.210352: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-02 12:06:51.210426: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-02 12:06:51.210439: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-02 12:06:51.210808: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/device:GPU:0 with 10405 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1)
ERROR: cannot read default config file. using build-in defaults [main]
LOG: connected to ('10.6.100.1', 10321)
LOG: destination: /var/tmp/makarov/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: request: /work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: using existing copy /var/tmp/irie/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
2019-07-02 12:06:55.991386: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-02 12:06:55.991524: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-02 12:06:55.991545: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-02 12:06:55.991565: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-02 12:06:55.991956: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10405 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1)
2019-07-02 12:13:21.253015: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-07-02 12:13:22.404599: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:03:00.0
totalMemory: 10.92GiB freeMemory: 10.76GiB
2019-07-02 12:13:22.405930: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-02 12:13:22.405951: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-02 12:13:22.405960: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-02 12:13:22.405968: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-02 12:13:22.408998: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-02 12:13:22.846545: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-02 12:13:22.846602: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-02 12:13:22.846610: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-02 12:13:22.847079: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/device:GPU:0 with 10405 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1)
ERROR: cannot read default config file. using build-in defaults [main]
LOG: connected to ('10.6.100.1', 10321)
LOG: destination: /var/tmp/makarov/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: request: /work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: checking status of cluster-cn-223:/var/tmp/zeyer/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: cannot get stat of cluster-cn-223:/var/tmp/zeyer/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: checking status of cluster-cn-255:/var/tmp/irie/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: cannot get stat of cluster-cn-255:/var/tmp/irie/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: checking status of cluster-cn-216:/var/tmp/irie/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: cannot get stat of cluster-cn-216:/var/tmp/irie/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: checking status of cluster-cn-222:/var/tmp/zeineldeen/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: cannot get stat of cluster-cn-222:/var/tmp/zeineldeen/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: checking status of cluster-cn-222:/var/tmp/zeyer/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: cannot get stat of cluster-cn-222:/var/tmp/zeyer/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: checking status of cluster-cn-243:/var/tmp/zeineldeen/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: cannot get stat of cluster-cn-243:/var/tmp/zeineldeen/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: checking status of cluster-cn-231:/var/tmp/zeineldeen/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: cannot get stat of cluster-cn-231:/var/tmp/zeineldeen/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: checking status of cluster-cn-234:/var/tmp/zeineldeen/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: cannot get stat of cluster-cn-234:/var/tmp/zeineldeen/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: checking status of cluster-cn-273:/var/tmp/zeineldeen/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: cannot get stat of cluster-cn-273:/var/tmp/zeineldeen/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: checking status of cluster-cn-224:/var/tmp/zeyer/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: cannot get stat of cluster-cn-224:/var/tmp/zeyer/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: checking status of cluster-cn-250:/var/tmp/zeineldeen/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: cannot get stat of cluster-cn-250:/var/tmp/zeineldeen/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: checking status of cluster-cn-254:/var/tmp/zeineldeen/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: cannot get stat of cluster-cn-254:/var/tmp/zeineldeen/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: checking status of cluster-cn-260:/var/tmp/zeyer/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: cannot get stat of cluster-cn-260:/var/tmp/zeyer/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: checking status of cluster-cn-241:/var/tmp/zeyer/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: cannot get stat of cluster-cn-241:/var/tmp/zeyer/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: checking status of cluster-cn-260:/var/tmp/zeineldeen/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: cannot get stat of cluster-cn-260:/var/tmp/zeineldeen/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: start copying /work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: copied /work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
2019-07-02 12:13:28.300181: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-02 12:13:28.300264: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-02 12:13:28.300278: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-02 12:13:28.300287: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-02 12:13:28.302560: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10405 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1)
/var/spool/sge/cluster-cn-244/job_scripts/9505996: line 8: 21792 Killed                  python3 ~/returnn-parnia/tools/get-attention-weights.py /u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc.config --epoch 481 --data 'config:get_dataset("'$1'")' --dump_dir ~/makarov/NMT/hmm-factorization/experiments/asr2019/transformer/forward-$1 --layers "dec_02_att_weights" --layers "dec_01_att_weights" --layers "dec_03_att_weights" --layers "dec_04_att_weights" --layers "dec_05_att_weights" --layers "dec_06_att_weights" --layers "dec_07_att_weights" --layers "dec_08_att_weights" --layers "dec_09_att_weights" --layers "dec_10_att_weights" --layers "dec_11_att_weights" --layers "dec_12_att_weights" --rec_layer "output" --batch_size 4000 --tf_log_dir /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/transformer/forward/tf_log_dir --reset_seq_ordering sorted_reverse
2019-07-02 12:35:14.119006: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:03:00.0
totalMemory: 10.92GiB freeMemory: 10.76GiB
2019-07-02 12:35:14.119436: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-02 12:35:14.119517: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-02 12:35:14.119544: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-02 12:35:14.119570: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-02 12:35:14.122823: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-02 12:35:14.762620: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-02 12:35:14.762697: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-02 12:35:14.762711: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-02 12:35:14.763144: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/device:GPU:0 with 10405 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1)
ERROR: cannot read default config file. using build-in defaults [main]
LOG: connected to ('10.6.100.1', 10321)
LOG: destination: /var/tmp/makarov/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: request: /work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: checking status of cluster-cn-244:/var/tmp/makarov/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: cannot get stat of cluster-cn-244:/var/tmp/makarov/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: start copying /work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
LOG: copied /work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/test-clean.zip
2019-07-02 12:35:22.134607: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-02 12:35:22.134812: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-02 12:35:22.134826: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-02 12:35:22.134838: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-02 12:35:22.135112: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10405 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1)
