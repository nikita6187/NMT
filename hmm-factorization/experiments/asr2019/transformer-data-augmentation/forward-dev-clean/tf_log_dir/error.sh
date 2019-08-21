2019-07-02 11:38:25.327930: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-07-02 11:38:26.428616: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:81:00.0
totalMemory: 10.92GiB freeMemory: 10.76GiB
2019-07-02 11:38:26.429093: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-02 11:38:26.429141: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-02 11:38:26.429152: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-02 11:38:26.429161: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-02 11:38:26.432401: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-02 11:38:26.847386: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-02 11:38:26.847450: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-02 11:38:26.847458: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-02 11:38:26.847920: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/device:GPU:0 with 10405 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:81:00.0, compute capability: 6.1)
[31;1mEXCEPTION[0m
[34mTraceback (most recent call last):[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/tools/[0m[36;1mget-attention-weights.py[0m[36m"[0m, [34mline [0m[35m414[0m, [34min [0m<module>
[34m    line: [0mmain[34m([0msys[34m.[0margv[34m)[0m
[34m    locals:[0m
      main [34;1m= [0m[34m<local> [0m[34m<[0mfunction main at 0x2b0cb630c7b8[34m>[0m
      sys [34;1m= [0m[34m<local> [0m[34m<[0mmodule [36m'sys'[0m [34m([0mbuilt[34m-[0m[34min[0m[34m)[0m[34m>[0m
      sys[34;1m.[0margv [34;1m= [0m[34m<local> [0m[34m[[0m[36m'/u/makarov/returnn-parnia/tools/get-attention-weights.py'[0m[34m,[0m [36m'/u/bahar/workspace/asr/librispeech/test-20190121/config-train/trafo.specaug.datarndperm_noscale.12l.ffdim4.pretrain3.hpc.config'[0m[34m,[0m [36m'--epoch'[0m[34m,[0m [36m'481'[0m[34m,[0m [36m'--data'[0m[34m,[0m [36m'config:get_dataset("${DATA}")'[0m[34m,[0m [36m'--dump_dir'[0m[34m,[0m [36m'/u/makarov/makarov/NMT/hmm-fa..., len = 40, _[0]: {len = 56}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/tools/[0m[36;1mget-attention-weights.py[0m[36m"[0m, [34mline [0m[35m221[0m, [34min [0mmain
[34m    line: [0mdataset [34m=[0m init_dataset[34m([0mdataset_str[34m,[0m extra_kwargs[34m=[0mextra_dataset_kwargs[34m)[0m
[34m    locals:[0m
      dataset [34;1m= [0m[34m<not found>[0m
      init_dataset [34;1m= [0m[34m<global> [0m[34m<[0mfunction init_dataset at 0x2b0c5e8e3268[34m>[0m
      dataset_str [34;1m= [0m[34m<local> [0m[36m'config:get_dataset("${DATA}")'[0m[34m,[0m len [34m=[0m 29
      extra_kwargs [34;1m= [0m[34m<not found>[0m
      extra_dataset_kwargs [34;1m= [0m[34m<local> [0m[34m{[0m[36m'seq_ordering'[0m[34m:[0m [36m'sorted_reverse'[0m[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/[0m[36;1mDataset.py[0m[36m"[0m, [34mline [0m[35m884[0m, [34min [0minit_dataset
[34m    line: [0m[34mreturn [0minit_dataset_via_str[34m([0mconfig_str[34m=[0mconfig_str[34m,[0m [34m*[0m[34m*[0mkwargs[34m)[0m
[34m    locals:[0m
      init_dataset_via_str [34;1m= [0m[34m<global> [0m[34m<[0mfunction init_dataset_via_str at 0x2b0c5e8e32f0[34m>[0m
      config_str [34;1m= [0m[34m<local> [0m[36m'config:get_dataset("${DATA}")'[0m[34m,[0m len [34m=[0m 29
      kwargs [34;1m= [0m[34m<local> [0m[34m{[0m[36m'seq_ordering'[0m[34m:[0m [36m'sorted_reverse'[0m[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/[0m[36;1mDataset.py[0m[36m"[0m, [34mline [0m[35m927[0m, [34min [0minit_dataset_via_str
[34m    line: [0m[34mreturn [0minit_dataset[34m([0mdata[34m)[0m
[34m    locals:[0m
      init_dataset [34;1m= [0m[34m<global> [0m[34m<[0mfunction init_dataset at 0x2b0c5e8e3268[34m>[0m
      data [34;1m= [0m[34m<local> [0m[34m{[0m[36m'use_ogg'[0m[34m:[0m [34mTrue[0m[34m,[0m [36m'use_cache_manager'[0m[34m:[0m [34mTrue[0m[34m,[0m [36m'path'[0m[34m:[0m [36m'/u/bahar/workspace/asr/librispeech/test-20190121/dataset/ogg-zips'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'LibriSpeechCorpus'[0m[34m,[0m [36m'seq_ordering'[0m[34m:[0m [36m'sorted_reverse'[0m[34m,[0m [36m'bpe'[0m[34m:[0m [34m{[0m[36m'bpe_file'[0m[34m:[0m [36m'/u/bahar/workspace/asr/librispeech/test-20190121/dataset/trans.bpe.codes'[0m[34m,[0m [36m'vocab_file'[0m[34m:[0m [36m'..., len = 10[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/[0m[36;1mDataset.py[0m[36m"[0m, [34mline [0m[35m897[0m, [34min [0minit_dataset
[34m    line: [0mobj [34m=[0m clazz[34m([0m[34m*[0m[34m*[0mkwargs[34m)[0m
[34m    locals:[0m
      obj [34;1m= [0m[34m<not found>[0m
      clazz [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'GeneratingDataset.LibriSpeechCorpus'[0m[34m>[0m
      kwargs [34;1m= [0m[34m<local> [0m[34m{[0m[36m'bpe'[0m[34m:[0m [34m{[0m[36m'bpe_file'[0m[34m:[0m [36m'/u/bahar/workspace/asr/librispeech/test-20190121/dataset/trans.bpe.codes'[0m[34m,[0m [36m'vocab_file'[0m[34m:[0m [36m'/u/bahar/workspace/asr/librispeech/test-20190121/dataset/trans.bpe.vocab'[0m[34m,[0m [36m'seq_postfix'[0m[34m:[0m [34m[[0m0[34m][0m[34m,[0m [36m'unknown_label'[0m[34m:[0m [36m'<unk>'[0m[34m}[0m[34m,[0m [36m'path'[0m[34m:[0m [36m'/u/bahar/workspace/asr/librispeech/test-20190121/datas..., len = 9[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/[0m[36;1mGeneratingDataset.py[0m[36m"[0m, [34mline [0m[35m1845[0m, [34min [0m__init__
[34m    line: [0m[34massert [0mzip_fns[34m,[0m [36m"no files found: %r"[0m [34m%[0m zip_fn_pattern
[34m    locals:[0m
      zip_fns [34;1m= [0m[34m<local> [0m[34m[[0m[34m][0m
      zip_fn_pattern [34;1m= [0m[34m<local> [0m[36m'/u/bahar/workspace/asr/librispeech/test-20190121/dataset/ogg-zips/${DATA}*.zip'[0m[34m,[0m len [34m=[0m 78
[31mAssertionError[0m: no files found: '/u/bahar/workspace/asr/librispeech/test-20190121/dataset/ogg-zips/${DATA}*.zip'
2019-07-02 12:01:46.143840: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-07-02 12:01:47.409600: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:81:00.0
totalMemory: 10.92GiB freeMemory: 10.76GiB
2019-07-02 12:01:47.409663: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-02 12:01:47.409680: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-02 12:01:47.409690: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-02 12:01:47.409699: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-02 12:01:47.412139: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-02 12:01:47.902552: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-02 12:01:47.902611: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-02 12:01:47.902620: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-02 12:01:47.902986: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/device:GPU:0 with 10405 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:81:00.0, compute capability: 6.1)
ERROR: cannot read default config file. using build-in defaults [main]
LOG: connected to ('10.6.100.1', 10321)
LOG: destination: /var/tmp/makarov/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-clean.zip
LOG: request: /work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-clean.zip
LOG: checking status of cluster-cn-234:/var/tmp/zeineldeen/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-clean.zip
LOG: cannot get stat of cluster-cn-234:/var/tmp/zeineldeen/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-clean.zip
LOG: checking status of cluster-cn-276:/var/tmp/irie/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-clean.zip
LOG: cannot get stat of cluster-cn-276:/var/tmp/irie/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-clean.zip
LOG: checking status of cluster-cn-01:/var/tmp/zeineldeen/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-clean.zip
LOG: start copying cluster-cn-01:/var/tmp/zeineldeen/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-clean.zip
LOG: copied cluster-cn-01:/var/tmp/zeineldeen/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-clean.zip
2019-07-02 12:02:00.379613: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-02 12:02:00.379704: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-02 12:02:00.379717: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-02 12:02:00.379726: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-02 12:02:00.384695: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10405 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:81:00.0, compute capability: 6.1)
2019-07-02 12:06:21.474864: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-07-02 12:06:22.927868: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:02:00.0
totalMemory: 10.92GiB freeMemory: 10.76GiB
2019-07-02 12:06:22.928152: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-02 12:06:22.928175: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-02 12:06:22.928185: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-02 12:06:22.928194: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-02 12:06:22.934649: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-02 12:06:23.422405: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-02 12:06:23.422476: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-02 12:06:23.422494: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-02 12:06:23.423021: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/device:GPU:0 with 10405 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
ERROR: cannot read default config file. using build-in defaults [main]
LOG: connected to ('10.6.100.1', 10321)
LOG: destination: /var/tmp/makarov/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-clean.zip
LOG: request: /work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-clean.zip
LOG: using existing copy /var/tmp/irie/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-clean.zip
2019-07-02 12:06:51.312689: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-02 12:06:51.312783: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-02 12:06:51.312807: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-02 12:06:51.312819: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-02 12:06:51.313117: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10405 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
2019-07-02 12:10:45.537564: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:03:00.0
totalMemory: 10.92GiB freeMemory: 10.76GiB
2019-07-02 12:10:45.537782: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-02 12:10:45.537808: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-02 12:10:45.537820: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-02 12:10:45.537832: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-02 12:10:45.542816: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-02 12:10:46.174795: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-02 12:10:46.174874: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-02 12:10:46.174888: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-02 12:10:46.175295: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/device:GPU:0 with 10405 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1)
ERROR: cannot read default config file. using build-in defaults [main]
LOG: connected to ('10.6.100.1', 10321)
LOG: destination: /var/tmp/makarov/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-clean.zip
LOG: request: /work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-clean.zip
LOG: using existing copy /var/tmp/irie/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-clean.zip
2019-07-02 12:10:48.011201: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-02 12:10:48.011322: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-02 12:10:48.011344: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-02 12:10:48.011365: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-02 12:10:48.014750: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10405 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1)
