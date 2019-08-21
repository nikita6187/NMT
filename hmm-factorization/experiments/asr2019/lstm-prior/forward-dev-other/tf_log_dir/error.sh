fatal: Not a git repository (or any parent up to mount point /u/makarov)
Stopping at filesystem boundary (GIT_DISCOVERY_ACROSS_FILESYSTEM not set).
2019-07-05 14:02:47.919722: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-07-05 14:02:48.958559: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:02:00.0
totalMemory: 10.92GiB freeMemory: 10.76GiB
2019-07-05 14:02:48.958612: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-05 14:02:48.958628: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-05 14:02:48.958638: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-05 14:02:48.958646: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-05 14:02:48.963177: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-05 14:02:49.414238: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-05 14:02:49.414293: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-05 14:02:49.414301: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-05 14:02:49.414659: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/device:GPU:0 with 10405 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
ERROR: cannot read default config file. using build-in defaults [main]
LOG: connected to ('10.6.100.1', 10321)
LOG: destination: /var/tmp/makarov/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-other.zip
LOG: request: /work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-other.zip
LOG: using existing copy /var/tmp/irie/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-other.zip
2019-07-05 14:03:04.978341: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-05 14:03:04.978427: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-05 14:03:04.978441: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-05 14:03:04.978451: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-05 14:03:04.978716: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10405 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
[31;1mEXCEPTION[0m
[34mTraceback (most recent call last):[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/tools/[0m[36;1mget-attention-weights.py[0m[36m"[0m, [34mline [0m[35m414[0m, [34min [0m<module>
[34m    line: [0mmain[34m([0msys[34m.[0margv[34m)[0m
[34m    locals:[0m
      main [34;1m= [0m[34m<local> [0m[34m<[0mfunction main at 0x2ad08725eae8[34m>[0m
      sys [34;1m= [0m[34m<local> [0m[34m<[0mmodule [36m'sys'[0m [34m([0mbuilt[34m-[0m[34min[0m[34m)[0m[34m>[0m
      sys[34;1m.[0margv [34;1m= [0m[34m<local> [0m[34m[[0m[36m'/u/makarov/returnn-parnia-2/tools/get-attention-weights.py'[0m[34m,[0m [36m'/u/bahar/workspace/asr/librispeech/test-20190121/config-train/base2.smlp2.specaug.datarndperm_noscale.bs18k.curric3.retrain1.config'[0m[34m,[0m [36m'--epoch'[0m[34m,[0m [36m'183'[0m[34m,[0m [36m'--data'[0m[34m,[0m [36m'config:get_dataset("dev-other")'[0m[34m,[0m [36m'--dump_dir'[0m[34m,[0m [36m'/u/makarov/makarov/NM..., len = 18, _[0]: {len = 58}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/tools/[0m[36;1mget-attention-weights.py[0m[36m"[0m, [34mline [0m[35m231[0m, [34min [0mmain
[34m    line: [0minit_net[34m([0margs[34m,[0m layers[34m)[0m
[34m    locals:[0m
      init_net [34;1m= [0m[34m<global> [0m[34m<[0mfunction init_net at 0x2ad08725ea60[34m>[0m
      args [34;1m= [0m[34m<local> [0mNamespace[34m([0masr[34m=[0m[34mFalse[0m[34m,[0m batch_size[34m=[0m4000[34m,[0m beam_size[34m=[0m12[34m,[0m config_file[34m=[0m[36m'/u/bahar/workspace/asr/librispeech/test-20190121/config-train/base2.smlp2.specaug.datarndperm_noscale.bs18k.curric3.retrain1.config'[0m[34m,[0m data[34m=[0m[36m'config:get_dataset("dev-other")'[0m[34m,[0m device[34m=[0m[34mNone[0m[34m,[0m do_search[34m=[0m[34mFalse[0m[34m,[0m dropout[34m=[0m[34mNone[0m[34m,[0m dump_dir[34m=[0m[36m'/u/m...[0m
      layers [34;1m= [0m[34m<local> [0m[34m[[0m[36m'att_weights'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 11[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/tools/[0m[36;1mget-attention-weights.py[0m[36m"[0m, [34mline [0m[35m139[0m, [34min [0minit_net
[34m    line: [0mrnn[34m.[0mengine[34m.[0minit_network_from_config[34m([0mconfig[34m=[0mconfig[34m,[0m net_dict_post_proc[34m=[0mnet_dict_post_proc[34m)[0m
[34m    locals:[0m
      rnn [34;1m= [0m[34m<global> [0m[34m<[0mmodule [36m'rnn'[0m [34mfrom [0m[36m'/u/makarov/returnn-parnia-2/rnn.py'[0m[34m>[0m
      rnn[34;1m.[0mengine [34;1m= [0m[34m<global> [0m[34m<[0mTFEngine[34m.[0mEngine object at 0x2acfd81e5ac8[34m>[0m
      rnn[34;1m.[0mengine[34;1m.[0minit_network_from_config [34;1m= [0m[34m<global> [0m[34m<[0mbound method Engine[34m.[0minit_network_from_config of [34m<[0mTFEngine[34m.[0mEngine object at 0x2acfd81e5ac8[34m>[0m[34m>[0m
      config [34;1m= [0m[34m<global> [0m[34m<[0mConfig[34m.[0mConfig object at 0x2ad087266748[34m>[0m
      net_dict_post_proc [34;1m= [0m[34m<local> [0m[34m<[0mfunction init_net[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mnet_dict_post_proc at 0x2ad0de0940d0[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFEngine.py[0m[36m"[0m, [34mline [0m[35m936[0m, [34min [0minit_network_from_config
[34m    line: [0mself[34m.[0m_init_network[34m([0mnet_desc[34m=[0mnet_dict[34m,[0m epoch[34m=[0mself[34m.[0mepoch[34m)[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFEngine[34m.[0mEngine object at 0x2acfd81e5ac8[34m>[0m
      self[34;1m.[0m_init_network [34;1m= [0m[34m<local> [0m[34m<[0mbound method Engine[34m.[0m_init_network of [34m<[0mTFEngine[34m.[0mEngine object at 0x2acfd81e5ac8[34m>[0m[34m>[0m
      net_desc [34;1m= [0m[34m<not found>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'source'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_pool'[0m[34m:[0m [34m{[0m[36m'trainable'[0m[34m:[0m [34mFalse[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'pool_size'[0m[34m:[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      epoch [34;1m= [0m[34m<local> [0m183
      self[34;1m.[0mepoch [34;1m= [0m[34m<local> [0m183
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFEngine.py[0m[36m"[0m, [34mline [0m[35m1060[0m, [34min [0m_init_network
[34m    line: [0mself[34m.[0mnetwork[34m,[0m self[34m.[0mupdater [34m=[0m self[34m.[0mcreate_network[34m([0m
            config[34m=[0mself[34m.[0mconfig[34m,[0m
            rnd_seed[34m=[0mnet_random_seed[34m,[0m
            train_flag[34m=[0mtrain_flag[34m,[0m eval_flag[34m=[0mself[34m.[0muse_eval_flag[34m,[0m search_flag[34m=[0mself[34m.[0muse_search_flag[34m,[0m
            initial_learning_rate[34m=[0mgetattr[34m([0mself[34m,[0m [36m"initial_learning_rate"[0m[34m,[0m [34mNone[0m[34m)[0m[34m,[0m
            net_dict[34m=[0mnet_desc[34m)[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFEngine[34m.[0mEngine object at 0x2acfd81e5ac8[34m>[0m
      self[34;1m.[0mnetwork [34;1m= [0m[34m<local> [0m[34mNone[0m
      self[34;1m.[0mupdater [34;1m= [0m[34m<local> [0m[34mNone[0m
      self[34;1m.[0mcreate_network [34;1m= [0m[34m<local> [0m[34m<[0mbound method Engine[34m.[0mcreate_network of [34m<[0m[34mclass [0m[36m'TFEngine.Engine'[0m[34m>[0m[34m>[0m
      config [34;1m= [0m[34m<not found>[0m
      self[34;1m.[0mconfig [34;1m= [0m[34m<local> [0m[34m<[0mConfig[34m.[0mConfig object at 0x2ad087266748[34m>[0m
      rnd_seed [34;1m= [0m[34m<not found>[0m
      net_random_seed [34;1m= [0m[34m<local> [0m183
      train_flag [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m
      eval_flag [34;1m= [0m[34m<not found>[0m
      self[34;1m.[0muse_eval_flag [34;1m= [0m[34m<local> [0m[34mTrue[0m
      search_flag [34;1m= [0m[34m<not found>[0m
      self[34;1m.[0muse_search_flag [34;1m= [0m[34m<local> [0m[34mFalse[0m
      initial_learning_rate [34;1m= [0m[34m<not found>[0m
      getattr [34;1m= [0m[34m<builtin> [0m[34m<[0mbuilt[34m-[0m[34min [0mfunction getattr[34m>[0m
      net_dict [34;1m= [0m[34m<not found>[0m
      net_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'source'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_pool'[0m[34m:[0m [34m{[0m[36m'trainable'[0m[34m:[0m [34mFalse[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'pool_size'[0m[34m:[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFEngine.py[0m[36m"[0m, [34mline [0m[35m1091[0m, [34min [0mcreate_network
[34m    line: [0mnetwork[34m.[0mconstruct_from_dict[34m([0mnet_dict[34m)[0m
[34m    locals:[0m
      network [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      network[34;1m.[0mconstruct_from_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_from_dict of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'source'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_pool'[0m[34m:[0m [34m{[0m[36m'trainable'[0m[34m:[0m [34mFalse[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'pool_size'[0m[34m:[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m338[0m, [34min [0mconstruct_from_dict
[34m    line: [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m,[0m name[34m)[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'source'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_pool'[0m[34m:[0m [34m{[0m[36m'trainable'[0m[34m:[0m [34mFalse[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'pool_size'[0m[34m:[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<local> [0m[36m'ctc'[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.SoftmaxLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method LayerBase[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkLayer.SoftmaxLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'loss'[0m[34m:[0m [36m'ctc'[0m[34m,[0m [36m'target'[0m[34m:[0m [36m'classes'[0m[34m,[0m [36m'loss_opts'[0m[34m:[0m [34m{[0m[36m'ctc_opts'[0m[34m:[0m [34m{[0m[36m'ignore_longer_outputs_than_inputs'[0m[34m:[0m [34mTrue[0m[34m}[0m[34m,[0m [36m'beam_width'[0m[34m:[0m 1[34m}[0m[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de0af6a8[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m362[0m, [34min [0mtransform_config_dict
[34m    line: [0m[34mfor [0msrc_name [34min [0msrc_names
[34m    locals:[0m
      src_name [34;1m= [0m[34m<not found>[0m
      src_names [34;1m= [0m[34m<local> [0m[34m[[0m[36m'encoder'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 7[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m363[0m, [34min [0m<listcomp>
[34m    line: [0md[34m[[0m[36m"sources"[0m[34m][0m [34m=[0m [34m[[0m
            get_layer[34m([0msrc_name[34m)[0m
            [34mfor [0msrc_name [34min [0msrc_names
            [34mif [0m[34mnot [0msrc_name [34m=[0m[34m=[0m [36m"none"[0m[34m][0m
[34m    locals:[0m
      d [34;1m= [0m[34m<not found>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de0af6a8[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'encoder'[0m[34m,[0m len [34m=[0m 7
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'source'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_pool'[0m[34m:[0m [34m{[0m[36m'trainable'[0m[34m:[0m [34mFalse[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'pool_size'[0m[34m:[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'encoder'[0m[34m,[0m len [34m=[0m 7
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.CopyLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method LayerBase[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkLayer.CopyLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de15f0d0[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m362[0m, [34min [0mtransform_config_dict
[34m    line: [0m[34mfor [0msrc_name [34min [0msrc_names
[34m    locals:[0m
      src_name [34;1m= [0m[34m<not found>[0m
      src_names [34;1m= [0m[34m<local> [0m[34m[[0m[36m'lstm5_fw'[0m[34m,[0m [36m'lstm5_bw'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 8[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m363[0m, [34min [0m<listcomp>
[34m    line: [0md[34m[[0m[36m"sources"[0m[34m][0m [34m=[0m [34m[[0m
            get_layer[34m([0msrc_name[34m)[0m
            [34mfor [0msrc_name [34min [0msrc_names
            [34mif [0m[34mnot [0msrc_name [34m=[0m[34m=[0m [36m"none"[0m[34m][0m
[34m    locals:[0m
      d [34;1m= [0m[34m<not found>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de15f0d0[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm5_fw'[0m[34m,[0m len [34m=[0m 8
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'source'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_pool'[0m[34m:[0m [34m{[0m[36m'trainable'[0m[34m:[0m [34mFalse[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'pool_size'[0m[34m:[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm5_fw'[0m[34m,[0m len [34m=[0m 8
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method RecLayer[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m 1[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de16b400[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkRecLayer.py[0m[36m"[0m, [34mline [0m[35m230[0m, [34min [0mtransform_config_dict
[34m    line: [0msuper[34m([0mRecLayer[34m,[0m cls[34m)[0m[34m.[0mtransform_config_dict[34m([0md[34m,[0m network[34m=[0mnetwork[34m,[0m get_layer[34m=[0mget_layer[34m)[0m  [37m# everything except "unit"[0m
[34m    locals:[0m
      super [34;1m= [0m[34m<builtin> [0m[34m<[0m[34mclass [0m[36m'super'[0m[34m>[0m
      RecLayer [34;1m= [0m[34m<global> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      cls [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      transform_config_dict [34;1m= [0m[34m<not found>[0m
      d [34;1m= [0m[34m<local> [0m[34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m 1[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m}[0m
      network [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de16b400[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m362[0m, [34min [0mtransform_config_dict
[34m    line: [0m[34mfor [0msrc_name [34min [0msrc_names
[34m    locals:[0m
      src_name [34;1m= [0m[34m<not found>[0m
      src_names [34;1m= [0m[34m<local> [0m[34m[[0m[36m'lstm4_pool'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 10[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m363[0m, [34min [0m<listcomp>
[34m    line: [0md[34m[[0m[36m"sources"[0m[34m][0m [34m=[0m [34m[[0m
            get_layer[34m([0msrc_name[34m)[0m
            [34mfor [0msrc_name [34min [0msrc_names
            [34mif [0m[34mnot [0msrc_name [34m=[0m[34m=[0m [36m"none"[0m[34m][0m
[34m    locals:[0m
      d [34;1m= [0m[34m<not found>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de16b400[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm4_pool'[0m[34m,[0m len [34m=[0m 10
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'source'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_pool'[0m[34m:[0m [34m{[0m[36m'trainable'[0m[34m:[0m [34mFalse[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'pool_size'[0m[34m:[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm4_pool'[0m[34m,[0m len [34m=[0m 10
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method LayerBase[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'trainable'[0m[34m:[0m [34mFalse[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'pool_size'[0m[34m:[0m [34m([0m1[34m,[0m[34m)[0m[34m,[0m [36m'mode'[0m[34m:[0m [36m'max'[0m[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de16b510[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m362[0m, [34min [0mtransform_config_dict
[34m    line: [0m[34mfor [0msrc_name [34min [0msrc_names
[34m    locals:[0m
      src_name [34;1m= [0m[34m<not found>[0m
      src_names [34;1m= [0m[34m<local> [0m[34m[[0m[36m'lstm4_fw'[0m[34m,[0m [36m'lstm4_bw'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 8[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m363[0m, [34min [0m<listcomp>
[34m    line: [0md[34m[[0m[36m"sources"[0m[34m][0m [34m=[0m [34m[[0m
            get_layer[34m([0msrc_name[34m)[0m
            [34mfor [0msrc_name [34min [0msrc_names
            [34mif [0m[34mnot [0msrc_name [34m=[0m[34m=[0m [36m"none"[0m[34m][0m
[34m    locals:[0m
      d [34;1m= [0m[34m<not found>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de16b510[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm4_fw'[0m[34m,[0m len [34m=[0m 8
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'source'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_pool'[0m[34m:[0m [34m{[0m[36m'trainable'[0m[34m:[0m [34mFalse[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'pool_size'[0m[34m:[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm4_fw'[0m[34m,[0m len [34m=[0m 8
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method RecLayer[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m 1[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de16b620[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkRecLayer.py[0m[36m"[0m, [34mline [0m[35m230[0m, [34min [0mtransform_config_dict
[34m    line: [0msuper[34m([0mRecLayer[34m,[0m cls[34m)[0m[34m.[0mtransform_config_dict[34m([0md[34m,[0m network[34m=[0mnetwork[34m,[0m get_layer[34m=[0mget_layer[34m)[0m  [37m# everything except "unit"[0m
[34m    locals:[0m
      super [34;1m= [0m[34m<builtin> [0m[34m<[0m[34mclass [0m[36m'super'[0m[34m>[0m
      RecLayer [34;1m= [0m[34m<global> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      cls [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      transform_config_dict [34;1m= [0m[34m<not found>[0m
      d [34;1m= [0m[34m<local> [0m[34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m 1[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m}[0m
      network [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de16b620[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m362[0m, [34min [0mtransform_config_dict
[34m    line: [0m[34mfor [0msrc_name [34min [0msrc_names
[34m    locals:[0m
      src_name [34;1m= [0m[34m<not found>[0m
      src_names [34;1m= [0m[34m<local> [0m[34m[[0m[36m'lstm3_pool'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 10[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m363[0m, [34min [0m<listcomp>
[34m    line: [0md[34m[[0m[36m"sources"[0m[34m][0m [34m=[0m [34m[[0m
            get_layer[34m([0msrc_name[34m)[0m
            [34mfor [0msrc_name [34min [0msrc_names
            [34mif [0m[34mnot [0msrc_name [34m=[0m[34m=[0m [36m"none"[0m[34m][0m
[34m    locals:[0m
      d [34;1m= [0m[34m<not found>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de16b620[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm3_pool'[0m[34m,[0m len [34m=[0m 10
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'source'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_pool'[0m[34m:[0m [34m{[0m[36m'trainable'[0m[34m:[0m [34mFalse[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'pool_size'[0m[34m:[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm3_pool'[0m[34m,[0m len [34m=[0m 10
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method LayerBase[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'trainable'[0m[34m:[0m [34mFalse[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'pool_size'[0m[34m:[0m [34m([0m1[34m,[0m[34m)[0m[34m,[0m [36m'mode'[0m[34m:[0m [36m'max'[0m[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de16b730[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m362[0m, [34min [0mtransform_config_dict
[34m    line: [0m[34mfor [0msrc_name [34min [0msrc_names
[34m    locals:[0m
      src_name [34;1m= [0m[34m<not found>[0m
      src_names [34;1m= [0m[34m<local> [0m[34m[[0m[36m'lstm3_fw'[0m[34m,[0m [36m'lstm3_bw'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 8[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m363[0m, [34min [0m<listcomp>
[34m    line: [0md[34m[[0m[36m"sources"[0m[34m][0m [34m=[0m [34m[[0m
            get_layer[34m([0msrc_name[34m)[0m
            [34mfor [0msrc_name [34min [0msrc_names
            [34mif [0m[34mnot [0msrc_name [34m=[0m[34m=[0m [36m"none"[0m[34m][0m
[34m    locals:[0m
      d [34;1m= [0m[34m<not found>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de16b730[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm3_fw'[0m[34m,[0m len [34m=[0m 8
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'source'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_pool'[0m[34m:[0m [34m{[0m[36m'trainable'[0m[34m:[0m [34mFalse[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'pool_size'[0m[34m:[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm3_fw'[0m[34m,[0m len [34m=[0m 8
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method RecLayer[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m 1[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de16b840[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkRecLayer.py[0m[36m"[0m, [34mline [0m[35m230[0m, [34min [0mtransform_config_dict
[34m    line: [0msuper[34m([0mRecLayer[34m,[0m cls[34m)[0m[34m.[0mtransform_config_dict[34m([0md[34m,[0m network[34m=[0mnetwork[34m,[0m get_layer[34m=[0mget_layer[34m)[0m  [37m# everything except "unit"[0m
[34m    locals:[0m
      super [34;1m= [0m[34m<builtin> [0m[34m<[0m[34mclass [0m[36m'super'[0m[34m>[0m
      RecLayer [34;1m= [0m[34m<global> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      cls [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      transform_config_dict [34;1m= [0m[34m<not found>[0m
      d [34;1m= [0m[34m<local> [0m[34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m 1[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m}[0m
      network [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de16b840[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m362[0m, [34min [0mtransform_config_dict
[34m    line: [0m[34mfor [0msrc_name [34min [0msrc_names
[34m    locals:[0m
      src_name [34;1m= [0m[34m<not found>[0m
      src_names [34;1m= [0m[34m<local> [0m[34m[[0m[36m'lstm2_pool'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 10[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m363[0m, [34min [0m<listcomp>
[34m    line: [0md[34m[[0m[36m"sources"[0m[34m][0m [34m=[0m [34m[[0m
            get_layer[34m([0msrc_name[34m)[0m
            [34mfor [0msrc_name [34min [0msrc_names
            [34mif [0m[34mnot [0msrc_name [34m=[0m[34m=[0m [36m"none"[0m[34m][0m
[34m    locals:[0m
      d [34;1m= [0m[34m<not found>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de16b840[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm2_pool'[0m[34m,[0m len [34m=[0m 10
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'source'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_pool'[0m[34m:[0m [34m{[0m[36m'trainable'[0m[34m:[0m [34mFalse[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'pool_size'[0m[34m:[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm2_pool'[0m[34m,[0m len [34m=[0m 10
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method LayerBase[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'trainable'[0m[34m:[0m [34mFalse[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'pool_size'[0m[34m:[0m [34m([0m1[34m,[0m[34m)[0m[34m,[0m [36m'mode'[0m[34m:[0m [36m'max'[0m[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de16b950[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m362[0m, [34min [0mtransform_config_dict
[34m    line: [0m[34mfor [0msrc_name [34min [0msrc_names
[34m    locals:[0m
      src_name [34;1m= [0m[34m<not found>[0m
      src_names [34;1m= [0m[34m<local> [0m[34m[[0m[36m'lstm2_fw'[0m[34m,[0m [36m'lstm2_bw'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 8[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m363[0m, [34min [0m<listcomp>
[34m    line: [0md[34m[[0m[36m"sources"[0m[34m][0m [34m=[0m [34m[[0m
            get_layer[34m([0msrc_name[34m)[0m
            [34mfor [0msrc_name [34min [0msrc_names
            [34mif [0m[34mnot [0msrc_name [34m=[0m[34m=[0m [36m"none"[0m[34m][0m
[34m    locals:[0m
      d [34;1m= [0m[34m<not found>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de16b950[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm2_fw'[0m[34m,[0m len [34m=[0m 8
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'source'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_pool'[0m[34m:[0m [34m{[0m[36m'trainable'[0m[34m:[0m [34mFalse[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'pool_size'[0m[34m:[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm2_fw'[0m[34m,[0m len [34m=[0m 8
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method RecLayer[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m 1[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de16ba60[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkRecLayer.py[0m[36m"[0m, [34mline [0m[35m230[0m, [34min [0mtransform_config_dict
[34m    line: [0msuper[34m([0mRecLayer[34m,[0m cls[34m)[0m[34m.[0mtransform_config_dict[34m([0md[34m,[0m network[34m=[0mnetwork[34m,[0m get_layer[34m=[0mget_layer[34m)[0m  [37m# everything except "unit"[0m
[34m    locals:[0m
      super [34;1m= [0m[34m<builtin> [0m[34m<[0m[34mclass [0m[36m'super'[0m[34m>[0m
      RecLayer [34;1m= [0m[34m<global> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      cls [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      transform_config_dict [34;1m= [0m[34m<not found>[0m
      d [34;1m= [0m[34m<local> [0m[34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m 1[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m}[0m
      network [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de16ba60[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m362[0m, [34min [0mtransform_config_dict
[34m    line: [0m[34mfor [0msrc_name [34min [0msrc_names
[34m    locals:[0m
      src_name [34;1m= [0m[34m<not found>[0m
      src_names [34;1m= [0m[34m<local> [0m[34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 10[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m363[0m, [34min [0m<listcomp>
[34m    line: [0md[34m[[0m[36m"sources"[0m[34m][0m [34m=[0m [34m[[0m
            get_layer[34m([0msrc_name[34m)[0m
            [34mfor [0msrc_name [34min [0msrc_names
            [34mif [0m[34mnot [0msrc_name [34m=[0m[34m=[0m [36m"none"[0m[34m][0m
[34m    locals:[0m
      d [34;1m= [0m[34m<not found>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de16ba60[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm1_pool'[0m[34m,[0m len [34m=[0m 10
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'source'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_pool'[0m[34m:[0m [34m{[0m[36m'trainable'[0m[34m:[0m [34mFalse[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'pool_size'[0m[34m:[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm1_pool'[0m[34m,[0m len [34m=[0m 10
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method LayerBase[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'trainable'[0m[34m:[0m [34mFalse[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'pool_size'[0m[34m:[0m [34m([0m2[34m,[0m[34m)[0m[34m,[0m [36m'mode'[0m[34m:[0m [36m'max'[0m[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de16bb70[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m362[0m, [34min [0mtransform_config_dict
[34m    line: [0m[34mfor [0msrc_name [34min [0msrc_names
[34m    locals:[0m
      src_name [34;1m= [0m[34m<not found>[0m
      src_names [34;1m= [0m[34m<local> [0m[34m[[0m[36m'lstm1_fw'[0m[34m,[0m [36m'lstm1_bw'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 8[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m363[0m, [34min [0m<listcomp>
[34m    line: [0md[34m[[0m[36m"sources"[0m[34m][0m [34m=[0m [34m[[0m
            get_layer[34m([0msrc_name[34m)[0m
            [34mfor [0msrc_name [34min [0msrc_names
            [34mif [0m[34mnot [0msrc_name [34m=[0m[34m=[0m [36m"none"[0m[34m][0m
[34m    locals:[0m
      d [34;1m= [0m[34m<not found>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de16bb70[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm1_fw'[0m[34m,[0m len [34m=[0m 8
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'source'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_pool'[0m[34m:[0m [34m{[0m[36m'trainable'[0m[34m:[0m [34mFalse[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'pool_size'[0m[34m:[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm1_fw'[0m[34m,[0m len [34m=[0m 8
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method RecLayer[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m 1[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de16bc80[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkRecLayer.py[0m[36m"[0m, [34mline [0m[35m230[0m, [34min [0mtransform_config_dict
[34m    line: [0msuper[34m([0mRecLayer[34m,[0m cls[34m)[0m[34m.[0mtransform_config_dict[34m([0md[34m,[0m network[34m=[0mnetwork[34m,[0m get_layer[34m=[0mget_layer[34m)[0m  [37m# everything except "unit"[0m
[34m    locals:[0m
      super [34;1m= [0m[34m<builtin> [0m[34m<[0m[34mclass [0m[36m'super'[0m[34m>[0m
      RecLayer [34;1m= [0m[34m<global> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      cls [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      transform_config_dict [34;1m= [0m[34m<not found>[0m
      d [34;1m= [0m[34m<local> [0m[34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m 1[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m}[0m
      network [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de16bc80[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m362[0m, [34min [0mtransform_config_dict
[34m    line: [0m[34mfor [0msrc_name [34min [0msrc_names
[34m    locals:[0m
      src_name [34;1m= [0m[34m<not found>[0m
      src_names [34;1m= [0m[34m<local> [0m[34m[[0m[36m'lstm0_pool'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 10[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m363[0m, [34min [0m<listcomp>
[34m    line: [0md[34m[[0m[36m"sources"[0m[34m][0m [34m=[0m [34m[[0m
            get_layer[34m([0msrc_name[34m)[0m
            [34mfor [0msrc_name [34min [0msrc_names
            [34mif [0m[34mnot [0msrc_name [34m=[0m[34m=[0m [36m"none"[0m[34m][0m
[34m    locals:[0m
      d [34;1m= [0m[34m<not found>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de16bc80[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm0_pool'[0m[34m,[0m len [34m=[0m 10
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'source'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_pool'[0m[34m:[0m [34m{[0m[36m'trainable'[0m[34m:[0m [34mFalse[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'pool_size'[0m[34m:[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm0_pool'[0m[34m,[0m len [34m=[0m 10
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method LayerBase[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'trainable'[0m[34m:[0m [34mFalse[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'pool_size'[0m[34m:[0m [34m([0m3[34m,[0m[34m)[0m[34m,[0m [36m'mode'[0m[34m:[0m [36m'max'[0m[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de16bd90[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m362[0m, [34min [0mtransform_config_dict
[34m    line: [0m[34mfor [0msrc_name [34min [0msrc_names
[34m    locals:[0m
      src_name [34;1m= [0m[34m<not found>[0m
      src_names [34;1m= [0m[34m<local> [0m[34m[[0m[36m'lstm0_fw'[0m[34m,[0m [36m'lstm0_bw'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 8[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m363[0m, [34min [0m<listcomp>
[34m    line: [0md[34m[[0m[36m"sources"[0m[34m][0m [34m=[0m [34m[[0m
            get_layer[34m([0msrc_name[34m)[0m
            [34mfor [0msrc_name [34min [0msrc_names
            [34mif [0m[34mnot [0msrc_name [34m=[0m[34m=[0m [36m"none"[0m[34m][0m
[34m    locals:[0m
      d [34;1m= [0m[34m<not found>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de16bd90[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm0_fw'[0m[34m,[0m len [34m=[0m 8
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'source'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_pool'[0m[34m:[0m [34m{[0m[36m'trainable'[0m[34m:[0m [34mFalse[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'pool_size'[0m[34m:[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm0_fw'[0m[34m,[0m len [34m=[0m 8
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method RecLayer[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m 1[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de16bea0[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkRecLayer.py[0m[36m"[0m, [34mline [0m[35m230[0m, [34min [0mtransform_config_dict
[34m    line: [0msuper[34m([0mRecLayer[34m,[0m cls[34m)[0m[34m.[0mtransform_config_dict[34m([0md[34m,[0m network[34m=[0mnetwork[34m,[0m get_layer[34m=[0mget_layer[34m)[0m  [37m# everything except "unit"[0m
[34m    locals:[0m
      super [34;1m= [0m[34m<builtin> [0m[34m<[0m[34mclass [0m[36m'super'[0m[34m>[0m
      RecLayer [34;1m= [0m[34m<global> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      cls [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      transform_config_dict [34;1m= [0m[34m<not found>[0m
      d [34;1m= [0m[34m<local> [0m[34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m 1[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m}[0m
      network [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de16bea0[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m362[0m, [34min [0mtransform_config_dict
[34m    line: [0m[34mfor [0msrc_name [34min [0msrc_names
[34m    locals:[0m
      src_name [34;1m= [0m[34m<not found>[0m
      src_names [34;1m= [0m[34m<local> [0m[34m[[0m[36m'source'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 6[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m363[0m, [34min [0m<listcomp>
[34m    line: [0md[34m[[0m[36m"sources"[0m[34m][0m [34m=[0m [34m[[0m
            get_layer[34m([0msrc_name[34m)[0m
            [34mfor [0msrc_name [34min [0msrc_names
            [34mif [0m[34mnot [0msrc_name [34m=[0m[34m=[0m [36m"none"[0m[34m][0m
[34m    locals:[0m
      d [34;1m= [0m[34m<not found>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ad0de16bea0[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'source'[0m[34m,[0m len [34m=[0m 6
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'source'[0m[34m][0m[34m}[0m[34m,[0m [36m'lstm0_pool'[0m[34m:[0m [34m{[0m[36m'trainable'[0m[34m:[0m [34mFalse[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'pool_size'[0m[34m:[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'source'[0m[34m,[0m len [34m=[0m 6
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m418[0m, [34min [0mconstruct_layer
[34m    line: [0m[34mreturn [0madd_layer[34m([0mname[34m=[0mname[34m,[0m layer_class[34m=[0mlayer_class[34m,[0m [34m*[0m[34m*[0mlayer_desc[34m)[0m
[34m    locals:[0m
      add_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0madd_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      name [34;1m= [0m[34m<local> [0m[36m'source'[0m[34m,[0m len [34m=[0m 6
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.EvalLayer'[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'eval'[0m[34m:[0m [36m"self.network.get_config().typed_value('transform')(source(0), network=self.network)"[0m[34m,[0m [36m'sources'[0m[34m:[0m [34m[[0m[34m<[0mSourceLayer [36m'data'[0m out_type[34m=[0mData[34m([0mshape[34m=[0m[34m([0m[34mNone[0m[34m,[0m 40[34m)[0m[34m)[0m[34m>[0m[34m][0m[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m507[0m, [34min [0madd_layer
[34m    line: [0mlayer [34m=[0m self[34m.[0m_create_layer[34m([0mname[34m=[0mname[34m,[0m layer_class[34m=[0mlayer_class[34m,[0m [34m*[0m[34m*[0mlayer_desc[34m)[0m
[34m    locals:[0m
      layer [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0m_create_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0m_create_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      name [34;1m= [0m[34m<local> [0m[36m'source'[0m[34m,[0m len [34m=[0m 6
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.EvalLayer'[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'eval'[0m[34m:[0m [36m"self.network.get_config().typed_value('transform')(source(0), network=self.network)"[0m[34m,[0m [36m'sources'[0m[34m:[0m [34m[[0m[34m<[0mSourceLayer [36m'data'[0m out_type[34m=[0mData[34m([0mshape[34m=[0m[34m([0m[34mNone[0m[34m,[0m 40[34m)[0m[34m)[0m[34m>[0m[34m][0m[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m464[0m, [34min [0m_create_layer
[34m    line: [0mlayer [34m=[0m layer_class[34m([0m[34m*[0m[34m*[0mlayer_desc[34m)[0m
[34m    locals:[0m
      layer [34;1m= [0m[34m<not found>[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.EvalLayer'[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'network'[0m[34m:[0m [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m,[0m [36m'eval'[0m[34m:[0m [36m"self.network.get_config().typed_value('transform')(source(0), network=self.network)"[0m[34m,[0m [36m'name'[0m[34m:[0m [36m'source'[0m[34m,[0m [36m'sources'[0m[34m:[0m [34m[[0m[34m<[0mSourceLayer [36m'data'[0m out_type[34m=[0mData[34m([0mshape[34m=[0m[34m([0m[34mNone[0m[34m,[0m 40[34m)[0m[34m)[0m[34m>[0m[34m][0m[34m,[0m [36m'output'[0m[34m:[0m Data[34m([0mname[34m=[0m[36m'sou...[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m4426[0m, [34min [0m__init__
[34m    line: [0msuper[34m([0mEvalLayer[34m,[0m self[34m)[0m[34m.[0m__init__[34m([0mkind[34m=[0m[36m"eval"[0m[34m,[0m eval[34m=[0meval[34m,[0m [34m*[0m[34m*[0mkwargs[34m)[0m
[34m    locals:[0m
      super [34;1m= [0m[34m<builtin> [0m[34m<[0m[34mclass [0m[36m'super'[0m[34m>[0m
      EvalLayer [34;1m= [0m[34m<global> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.EvalLayer'[0m[34m>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mEvalLayer [36m'source'[0m out_type[34m=[0mData[34m([0mshape[34m=[0m[34m([0m[34mNone[0m[34m,[0m 40[34m)[0m[34m)[0m[34m>[0m
      __init__ [34;1m= [0m[34m<not found>[0m
      kind [34;1m= [0m[34m<not found>[0m
      eval [34;1m= [0m[34m<local> [0m[36m"self.network.get_config().typed_value('transform')(source(0), network=self.network)"[0m[34m,[0m len [34m=[0m 83
      kwargs [34;1m= [0m[34m<local> [0m[34m{[0m[36m'name'[0m[34m:[0m [36m'source'[0m[34m,[0m [36m'network'[0m[34m:[0m [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m,[0m [36m'sources'[0m[34m:[0m [34m[[0m[34m<[0mSourceLayer [36m'data'[0m out_type[34m=[0mData[34m([0mshape[34m=[0m[34m([0m[34mNone[0m[34m,[0m 40[34m)[0m[34m)[0m[34m>[0m[34m][0m[34m,[0m [36m'output'[0m[34m:[0m Data[34m([0mname[34m=[0m[36m'source_output'[0m[34m,[0m shape[34m=[0m[34m([0m[34mNone[0m[34m,[0m 40[34m)[0m[34m)[0m[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m4285[0m, [34min [0m__init__
[34m    line: [0mx [34m=[0m op[34m([0msources[34m)[0m
[34m    locals:[0m
      x [34;1m= [0m[34m<not found>[0m
      op [34;1m= [0m[34m<local> [0m[34m<[0mfunction CombineLayer[34m.[0m_get_op[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mwrap_eval_op at 0x2ad0de16f0d0[34m>[0m
      sources [34;1m= [0m[34m<local> [0m[34m[[0m[34m<[0mSourceLayer [36m'data'[0m out_type[34m=[0mData[34m([0mshape[34m=[0m[34m([0m[34mNone[0m[34m,[0m 40[34m)[0m[34m)[0m[34m>[0m[34m][0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m4409[0m, [34min [0mwrap_eval_op
[34m    line: [0m[34mreturn [0mself[34m.[0m_op_kind_eval[34m([0msources[34m,[0m eval_str[34m=[0meval_str[34m,[0m eval_locals[34m=[0meval_locals[34m)[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mEvalLayer [36m'source'[0m out_type[34m=[0mData[34m([0mshape[34m=[0m[34m([0m[34mNone[0m[34m,[0m 40[34m)[0m[34m)[0m[34m>[0m
      self[34;1m.[0m_op_kind_eval [34;1m= [0m[34m<local> [0m[34m<[0mbound method CombineLayer[34m.[0m_op_kind_eval of [34m<[0mEvalLayer [36m'source'[0m out_type[34m=[0mData[34m([0mshape[34m=[0m[34m([0m[34mNone[0m[34m,[0m 40[34m)[0m[34m)[0m[34m>[0m[34m>[0m
      sources [34;1m= [0m[34m<local> [0m[34m[[0m[34m<[0mSourceLayer [36m'data'[0m out_type[34m=[0mData[34m([0mshape[34m=[0m[34m([0m[34mNone[0m[34m,[0m 40[34m)[0m[34m)[0m[34m>[0m[34m][0m
      eval_str [34;1m= [0m[34m<local> [0m[36m"self.network.get_config().typed_value('transform')(source(0), network=self.network)"[0m[34m,[0m len [34m=[0m 83
      eval_locals [34;1m= [0m[34m<local> [0m[34mNone[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m4399[0m, [34min [0m_op_kind_eval
[34m    line: [0mx [34m=[0m eval[34m([0meval_str[34m,[0m vs[34m)[0m
[34m    locals:[0m
      x [34;1m= [0m[34m<not found>[0m
      eval [34;1m= [0m[34m<builtin> [0m[34m<[0mbuilt[34m-[0m[34min [0mfunction eval[34m>[0m
      eval_str [34;1m= [0m[34m<local> [0m[36m"self.network.get_config().typed_value('transform')(source(0), network=self.network)"[0m[34m,[0m len [34m=[0m 83
      vs [34;1m= [0m[34m<local> [0m[34m{[0m[36m'supported_devices_for_op'[0m[34m:[0m [34m<[0mfunction supported_devices_for_op at 0x2ad08723c158[34m>[0m[34m,[0m [36m'tile_transposed'[0m[34m:[0m [34m<[0mfunction tile_transposed at 0x2ad08722e620[34m>[0m[34m,[0m [36m'identity_with_check_numerics'[0m[34m:[0m [34m<[0mfunction identity_with_check_numerics at 0x2ad08722ba60[34m>[0m[34m,[0m [36m'to_float32'[0m[34m:[0m [34m<[0mfunction to_float32 at 0x2ad08723bf28[34m>[0m[34m,[0m [36m's..., len = 217[0m
  [34;1mFile [0m[36m"[0m[36;1m<string>[0m[36m"[0m, [34mline [0m[35m1[0m, [34min [0m<module>
[34m    -- code not available --[0m
  [34;1mFile [0m[36m"/u/bahar/workspace/asr/librispeech/test-20190121/config-train/[0m[36;1mbase2.smlp2.specaug.datarndperm_noscale.bs18k.curric3.retrain1.config[0m[36m"[0m, [34mline [0m[35m183[0m, [34min [0mtransform
[34m    line: [0mx [34m=[0m network[34m.[0mcond_on_train[34m([0mget_masked[34m,[0m [34mlambda[0m[34m:[0m x[34m)[0m
[34m    locals:[0m
      x [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'source/clip_by_value:0'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m,[0m 40[34m)[0m dtype[34m=[0mfloat32[34m>[0m
      network [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      network[34;1m.[0mcond_on_train [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mcond_on_train of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      get_masked [34;1m= [0m[34m<local> [0m[34m<[0mfunction transform[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_masked at 0x2ad01ec7e488[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m1027[0m, [34min [0mcond_on_train
[34m    line: [0m[34mreturn [0mcond[34m([0mself[34m.[0mtrain_flag[34m,[0m fn_train[34m,[0m fn_eval[34m)[0m
[34m    locals:[0m
      cond [34;1m= [0m[34m<local> [0m[34m<[0mfunction cond at 0x2ad0872332f0[34m>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mtrain_flag [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m
      fn_train [34;1m= [0m[34m<local> [0m[34m<[0mfunction transform[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_masked at 0x2ad01ec7e488[34m>[0m
      fn_eval [34;1m= [0m[34m<local> [0m[34m<[0mfunction transform[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0m[34m<[0m[34mlambda[0m[34m>[0m at 0x2ad0de16f1e0[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFUtil.py[0m[36m"[0m, [34mline [0m[35m3768[0m, [34min [0mcond
[34m    line: [0m[34mreturn [0mcontrol_flow_ops[34m.[0mcond[34m([0mpred[34m,[0m fn1[34m,[0m fn2[34m,[0m name[34m=[0mname[34m)[0m
[34m    locals:[0m
      control_flow_ops [34;1m= [0m[34m<local> [0m[34m<[0mmodule [36m'tensorflow.python.ops.control_flow_ops'[0m [34mfrom [0m[36m'/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/python/ops/control_flow_ops.py'[0m[34m>[0m
      control_flow_ops[34;1m.[0mcond [34;1m= [0m[34m<local> [0m[34m<[0mfunction cond at 0x2ad086288d08[34m>[0m
      pred [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m
      fn1 [34;1m= [0m[34m<local> [0m[34m<[0mfunction transform[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_masked at 0x2ad01ec7e488[34m>[0m
      fn2 [34;1m= [0m[34m<local> [0m[34m<[0mfunction transform[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0m[34m<[0m[34mlambda[0m[34m>[0m at 0x2ad0de16f1e0[34m>[0m
      name [34;1m= [0m[34m<local> [0m[34mNone[0m
  [34;1mFile [0m[36m"/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/python/util/[0m[36;1mdeprecation.py[0m[36m"[0m, [34mline [0m[35m454[0m, [34min [0mnew_func
[34m    line: [0m[34mreturn [0mfunc[34m([0m[34m*[0margs[34m,[0m [34m*[0m[34m*[0mkwargs[34m)[0m
[34m    locals:[0m
      func [34;1m= [0m[34m<local> [0m[34m<[0mfunction cond at 0x2ad086288bf8[34m>[0m
      args [34;1m= [0m[34m<local> [0m[34m([0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m,[0m [34m<[0mfunction transform[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_masked at 0x2ad01ec7e488[34m>[0m[34m,[0m [34m<[0mfunction transform[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0m[34m<[0m[34mlambda[0m[34m>[0m at 0x2ad0de16f1e0[34m>[0m[34m)[0m
      kwargs [34;1m= [0m[34m<local> [0m[34m{[0m[36m'name'[0m[34m:[0m [34mNone[0m[34m}[0m
  [34;1mFile [0m[36m"/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/python/ops/[0m[36;1mcontrol_flow_ops.py[0m[36m"[0m, [34mline [0m[35m2048[0m, [34min [0mcond
[34m    line: [0morig_res_t[34m,[0m res_t [34m=[0m context_t[34m.[0mBuildCondBranch[34m([0mtrue_fn[34m)[0m
[34m    locals:[0m
      orig_res_t [34;1m= [0m[34m<not found>[0m
      res_t [34;1m= [0m[34m<not found>[0m
      context_t [34;1m= [0m[34m<local> [0m[34m<[0mtensorflow[34m.[0mpython[34m.[0mops[34m.[0mcontrol_flow_ops[34m.[0mCondContext object at 0x2ad0de172470[34m>[0m
      context_t[34;1m.[0mBuildCondBranch [34;1m= [0m[34m<local> [0m[34m<[0mbound method CondContext[34m.[0mBuildCondBranch of [34m<[0mtensorflow[34m.[0mpython[34m.[0mops[34m.[0mcontrol_flow_ops[34m.[0mCondContext object at 0x2ad0de172470[34m>[0m[34m>[0m
      true_fn [34;1m= [0m[34m<local> [0m[34m<[0mfunction transform[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_masked at 0x2ad01ec7e488[34m>[0m
  [34;1mFile [0m[36m"/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/python/ops/[0m[36;1mcontrol_flow_ops.py[0m[36m"[0m, [34mline [0m[35m1895[0m, [34min [0mBuildCondBranch
[34m    line: [0moriginal_result [34m=[0m fn[34m([0m[34m)[0m
[34m    locals:[0m
      original_result [34;1m= [0m[34m<not found>[0m
      fn [34;1m= [0m[34m<local> [0m[34m<[0mfunction transform[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_masked at 0x2ad01ec7e488[34m>[0m
  [34;1mFile [0m[36m"/u/bahar/workspace/asr/librispeech/test-20190121/config-train/[0m[36;1mbase2.smlp2.specaug.datarndperm_noscale.bs18k.curric3.retrain1.config[0m[36m"[0m, [34mline [0m[35m179[0m, [34min [0mget_masked
[34m    line: [0mx_masked [34m=[0m random_mask[34m([0mx_masked[34m,[0m axis[34m=[0m1[34m,[0m min_num[34m=[0m1[34m,[0m max_num[34m=[0mtf[34m.[0mmaximum[34m([0mtf[34m.[0mshape[34m([0mx[34m)[0m[34m[[0m1[34m][0m [34m/[0m[34m/[0m 100[34m,[0m 1[34m)[0m[34m,[0m max_dims[34m=[0m20[34m)[0m
[34m    locals:[0m
      x_masked [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'source/clip_by_value:0'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m,[0m 40[34m)[0m dtype[34m=[0mfloat32[34m>[0m
      random_mask [34;1m= [0m[34m<global> [0m[34m<[0mfunction random_mask at 0x2ad087273840[34m>[0m
      axis [34;1m= [0m[34m<not found>[0m
      min_num [34;1m= [0m[34m<not found>[0m
      max_num [34;1m= [0m[34m<not found>[0m
      tf [34;1m= [0m[34m<local> [0m[34m<[0mmodule [36m'tensorflow'[0m [34mfrom [0m[36m'/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/__init__.py'[0m[34m>[0m
      tf[34;1m.[0mmaximum [34;1m= [0m[34m<local> [0m[34m<[0mfunction maximum at 0x2ad085e6f510[34m>[0m
      tf[34;1m.[0mshape [34;1m= [0m[34m<local> [0m[34m<[0mfunction shape at 0x2ad085ea4400[34m>[0m
      x [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'source/clip_by_value:0'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m,[0m 40[34m)[0m dtype[34m=[0mfloat32[34m>[0m
      max_dims [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/bahar/workspace/asr/librispeech/test-20190121/config-train/[0m[36;1mbase2.smlp2.specaug.datarndperm_noscale.bs18k.curric3.retrain1.config[0m[36m"[0m, [34mline [0m[35m168[0m, [34min [0mrandom_mask
[34m    line: [0m_[34m,[0m x [34m=[0m tf[34m.[0mwhile_loop[34m([0m
              cond[34m=[0m[34mlambda [0mi[34m,[0m _[34m:[0m tf[34m.[0mless[34m([0mi[34m,[0m tf[34m.[0mreduce_max[34m([0mnum[34m)[0m[34m)[0m[34m,[0m
              body[34m=[0m[34mlambda [0mi[34m,[0m x[34m:[0m [34m([0m
                  i [34m+[0m 1[34m,[0m 
                  tf[34m.[0mwhere[34m([0m
                      tf[34m.[0mless[34m([0mi[34m,[0m num[34m)[0m[34m,[0m
                      _mask[34m([0mx[34m,[0m axis[34m=[0maxis[34m,[0m pos[34m=[0mindices[34m[[0m[34m:[0m[34m,[0m i[34m][0m[34m,[0m max_amount[34m=[0mmax_dims[34m)[0m[34m,[0m
                      x[34m)[0m[34m)[0m[34m,[0m
              loop_vars[34m=[0m[34m([0m0[34m,[0m x[34m)[0m[34m)[0m
[34m    locals:[0m
      _ [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'source/cond/TopKV2:0'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m)[0m dtype[34m=[0mfloat32[34m>[0m
      x [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'source/clip_by_value:0'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m,[0m 40[34m)[0m dtype[34m=[0mfloat32[34m>[0m
      tf [34;1m= [0m[34m<local> [0m[34m<[0mmodule [36m'tensorflow'[0m [34mfrom [0m[36m'/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/__init__.py'[0m[34m>[0m
      tf[34;1m.[0mwhile_loop [34;1m= [0m[34m<local> [0m[34m<[0mfunction while_loop at 0x2ad086288a60[34m>[0m
      cond [34;1m= [0m[34m<not found>[0m
      i [34;1m= [0m[34m<not found>[0m
      tf[34;1m.[0mless [34;1m= [0m[34m<local> [0m[34m<[0mfunction less at 0x2ad085e6d8c8[34m>[0m
      tf[34;1m.[0mreduce_max [34;1m= [0m[34m<local> [0m[34m<[0mfunction reduce_max at 0x2ad0861ad598[34m>[0m
      num [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'source/cond/random_uniform:0'[0m shape[34m=[0m[34m([0m?[34m,[0m[34m)[0m dtype[34m=[0mint32[34m>[0m
      body [34;1m= [0m[34m<not found>[0m
      tf[34;1m.[0mwhere [34;1m= [0m[34m<local> [0m[34m<[0mfunction where at 0x2ad085fcf510[34m>[0m
      _mask [34;1m= [0m[34m<global> [0m[34m<[0mfunction _mask at 0x2ad0872737b8[34m>[0m
      axis [34;1m= [0m[34m<local> [0m1
      pos [34;1m= [0m[34m<not found>[0m
      indices [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'source/cond/TopKV2:1'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m)[0m dtype[34m=[0mint32[34m>[0m
      max_amount [34;1m= [0m[34m<not found>[0m
      max_dims [34;1m= [0m[34m<local> [0m20
      loop_vars [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/python/ops/[0m[36;1mcontrol_flow_ops.py[0m[36m"[0m, [34mline [0m[35m3232[0m, [34min [0mwhile_loop
[34m    line: [0mresult [34m=[0m loop_context[34m.[0mBuildLoop[34m([0mcond[34m,[0m body[34m,[0m loop_vars[34m,[0m shape_invariants[34m,[0m
                                          return_same_structure[34m)[0m
[34m    locals:[0m
      result [34;1m= [0m[34m<not found>[0m
      loop_context [34;1m= [0m[34m<local> [0m[34m<[0mtensorflow[34m.[0mpython[34m.[0mops[34m.[0mcontrol_flow_ops[34m.[0mWhileContext object at 0x2ad0dfe15dd8[34m>[0m
      loop_context[34;1m.[0mBuildLoop [34;1m= [0m[34m<local> [0m[34m<[0mbound method WhileContext[34m.[0mBuildLoop of [34m<[0mtensorflow[34m.[0mpython[34m.[0mops[34m.[0mcontrol_flow_ops[34m.[0mWhileContext object at 0x2ad0dfe15dd8[34m>[0m[34m>[0m
      cond [34;1m= [0m[34m<local> [0m[34m<[0mfunction random_mask[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0m[34m<[0m[34mlambda[0m[34m>[0m at 0x2ad01ec909d8[34m>[0m
      body [34;1m= [0m[34m<local> [0m[34m<[0mfunction random_mask[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0m[34m<[0m[34mlambda[0m[34m>[0m at 0x2ad0de16f488[34m>[0m
      loop_vars [34;1m= [0m[34m<local> [0m[34m([0m0[34m,[0m [34m<[0mtf[34m.[0mTensor [36m'source/clip_by_value:0'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m,[0m 40[34m)[0m dtype[34m=[0mfloat32[34m>[0m[34m)[0m
      shape_invariants [34;1m= [0m[34m<local> [0m[34mNone[0m
      return_same_structure [34;1m= [0m[34m<local> [0m[34mFalse[0m
  [34;1mFile [0m[36m"/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/python/ops/[0m[36;1mcontrol_flow_ops.py[0m[36m"[0m, [34mline [0m[35m2952[0m, [34min [0mBuildLoop
[34m    line: [0moriginal_body_result[34m,[0m exit_vars [34m=[0m self[34m.[0m_BuildLoop[34m([0m
              pred[34m,[0m body[34m,[0m original_loop_vars[34m,[0m loop_vars[34m,[0m shape_invariants[34m)[0m
[34m    locals:[0m
      original_body_result [34;1m= [0m[34m<not found>[0m
      exit_vars [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mtensorflow[34m.[0mpython[34m.[0mops[34m.[0mcontrol_flow_ops[34m.[0mWhileContext object at 0x2ad0dfe15dd8[34m>[0m
      self[34;1m.[0m_BuildLoop [34;1m= [0m[34m<local> [0m[34m<[0mbound method WhileContext[34m.[0m_BuildLoop of [34m<[0mtensorflow[34m.[0mpython[34m.[0mops[34m.[0mcontrol_flow_ops[34m.[0mWhileContext object at 0x2ad0dfe15dd8[34m>[0m[34m>[0m
      pred [34;1m= [0m[34m<local> [0m[34m<[0mfunction random_mask[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0m[34m<[0m[34mlambda[0m[34m>[0m at 0x2ad01ec909d8[34m>[0m
      body [34;1m= [0m[34m<local> [0m[34m<[0mfunction random_mask[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0m[34m<[0m[34mlambda[0m[34m>[0m at 0x2ad0de16f488[34m>[0m
      original_loop_vars [34;1m= [0m[34m<local> [0m[34m([0m0[34m,[0m [34m<[0mtf[34m.[0mTensor [36m'source/clip_by_value:0'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m,[0m 40[34m)[0m dtype[34m=[0mfloat32[34m>[0m[34m)[0m
      loop_vars [34;1m= [0m[34m<local> [0m[34m[[0m[34m<[0mtf[34m.[0mTensor [36m'source/cond/while/Const:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mint32[34m>[0m[34m,[0m [34m<[0mtf[34m.[0mTensor [36m'source/clip_by_value:0'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m,[0m 40[34m)[0m dtype[34m=[0mfloat32[34m>[0m[34m][0m
      shape_invariants [34;1m= [0m[34m<local> [0m[34mNone[0m
  [34;1mFile [0m[36m"/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/python/ops/[0m[36;1mcontrol_flow_ops.py[0m[36m"[0m, [34mline [0m[35m2887[0m, [34min [0m_BuildLoop
[34m    line: [0mbody_result [34m=[0m body[34m([0m[34m*[0mpacked_vars_for_body[34m)[0m
[34m    locals:[0m
      body_result [34;1m= [0m[34m<not found>[0m
      body [34;1m= [0m[34m<local> [0m[34m<[0mfunction random_mask[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0m[34m<[0m[34mlambda[0m[34m>[0m at 0x2ad0de16f488[34m>[0m
      packed_vars_for_body [34;1m= [0m[34m<local> [0m[34m([0m[34m<[0mtf[34m.[0mTensor [36m'source/cond/while/Identity:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mint32[34m>[0m[34m,[0m [34m<[0mtf[34m.[0mTensor [36m'source/cond/while/Identity_1:0'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m,[0m 40[34m)[0m dtype[34m=[0mfloat32[34m>[0m[34m)[0m
  [34;1mFile [0m[36m"/u/bahar/workspace/asr/librispeech/test-20190121/config-train/[0m[36;1mbase2.smlp2.specaug.datarndperm_noscale.bs18k.curric3.retrain1.config[0m[36m"[0m, [34mline [0m[35m166[0m, [34min [0m<lambda>
[34m    line: [0m_mask[34m([0mx[34m,[0m axis[34m=[0maxis[34m,[0m pos[34m=[0mindices[34m[[0m[34m:[0m[34m,[0m i[34m][0m[34m,[0m max_amount[34m=[0mmax_dims[34m)[0m[34m,[0m
[34m    locals:[0m
      _mask [34;1m= [0m[34m<global> [0m[34m<[0mfunction _mask at 0x2ad0872737b8[34m>[0m
      x [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'source/cond/while/Identity_1:0'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m,[0m 40[34m)[0m dtype[34m=[0mfloat32[34m>[0m
      axis [34;1m= [0m[34m<local> [0m1
      pos [34;1m= [0m[34m<not found>[0m
      indices [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'source/cond/TopKV2:1'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m)[0m dtype[34m=[0mint32[34m>[0m
      i [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'source/cond/while/Identity:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mint32[34m>[0m
      max_amount [34;1m= [0m[34m<not found>[0m
      max_dims [34;1m= [0m[34m<local> [0m20
  [34;1mFile [0m[36m"/u/bahar/workspace/asr/librispeech/test-20190121/config-train/[0m[36;1mbase2.smlp2.specaug.datarndperm_noscale.bs18k.curric3.retrain1.config[0m[36m"[0m, [34mline [0m[35m138[0m, [34min [0m_mask
[34m    line: [0m[34mfrom [0mTFUtil [34mimport [0mwhere_bc
[34m    locals:[0m
      TFUtil [34;1m= [0m[34m<not found>[0m
      where_bc [34;1m= [0m[34m<not found>[0m
[31mImportError[0m: cannot import name 'where_bc'
fatal: Not a git repository (or any parent up to mount point /u/makarov)
Stopping at filesystem boundary (GIT_DISCOVERY_ACROSS_FILESYSTEM not set).
2019-07-05 14:21:00.823654: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-07-05 14:21:02.285277: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:81:00.0
totalMemory: 10.92GiB freeMemory: 10.76GiB
2019-07-05 14:21:02.285320: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-05 14:21:02.285331: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-05 14:21:02.285337: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-05 14:21:02.285343: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-05 14:21:02.289586: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-05 14:21:02.984473: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-05 14:21:02.984523: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-05 14:21:02.984530: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-05 14:21:02.984897: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/device:GPU:0 with 10405 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:81:00.0, compute capability: 6.1)
ERROR: cannot read default config file. using build-in defaults [main]
LOG: connected to ('10.6.100.1', 10321)
LOG: destination: /var/tmp/makarov/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-other.zip
LOG: request: /work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-other.zip
LOG: using existing copy /var/tmp/irie/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-other.zip
2019-07-05 14:21:05.272654: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-05 14:21:05.272707: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-05 14:21:05.272714: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-05 14:21:05.272719: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-05 14:21:05.272907: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10405 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:81:00.0, compute capability: 6.1)
[31;1mEXCEPTION[0m
[34mTraceback (most recent call last):[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/tools/[0m[36;1mget-attention-weights.py[0m[36m"[0m, [34mline [0m[35m414[0m, [34min [0m<module>
[34m    line: [0mmain[34m([0msys[34m.[0margv[34m)[0m
[34m    locals:[0m
      main [34;1m= [0m[34m<local> [0m[34m<[0mfunction main at 0x2ba1402deae8[34m>[0m
      sys [34;1m= [0m[34m<local> [0m[34m<[0mmodule [36m'sys'[0m [34m([0mbuilt[34m-[0m[34min[0m[34m)[0m[34m>[0m
      sys[34;1m.[0margv [34;1m= [0m[34m<local> [0m[34m[[0m[36m'/u/makarov/returnn-parnia-2/tools/get-attention-weights.py'[0m[34m,[0m [36m'/work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/lstm-prior/exp3.ctc.ogg-priorAtt-lr0005-k10-specAug.config'[0m[34m,[0m [36m'--epoch'[0m[34m,[0m [36m'183'[0m[34m,[0m [36m'--data'[0m[34m,[0m [36m'config:get_dataset("dev-other")'[0m[34m,[0m [36m'--dump_dir'[0m[34m,[0m [36m'/u/makarov/makarov/NMT/hmm-factor..., len = 18, _[0]: {len = 58}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/tools/[0m[36;1mget-attention-weights.py[0m[36m"[0m, [34mline [0m[35m231[0m, [34min [0mmain
[34m    line: [0minit_net[34m([0margs[34m,[0m layers[34m)[0m
[34m    locals:[0m
      init_net [34;1m= [0m[34m<global> [0m[34m<[0mfunction init_net at 0x2ba1402dea60[34m>[0m
      args [34;1m= [0m[34m<local> [0mNamespace[34m([0masr[34m=[0m[34mFalse[0m[34m,[0m batch_size[34m=[0m4000[34m,[0m beam_size[34m=[0m12[34m,[0m config_file[34m=[0m[36m'/work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/lstm-prior/exp3.ctc.ogg-priorAtt-lr0005-k10-specAug.config'[0m[34m,[0m data[34m=[0m[36m'config:get_dataset("dev-other")'[0m[34m,[0m device[34m=[0m[34mNone[0m[34m,[0m do_search[34m=[0m[34mFalse[0m[34m,[0m dropout[34m=[0m[34mNone[0m[34m,[0m dump_dir[34m=[0m[36m'/u/makarov/makar...[0m
      layers [34;1m= [0m[34m<local> [0m[34m[[0m[36m'att_weights'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 11[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/tools/[0m[36;1mget-attention-weights.py[0m[36m"[0m, [34mline [0m[35m139[0m, [34min [0minit_net
[34m    line: [0mrnn[34m.[0mengine[34m.[0minit_network_from_config[34m([0mconfig[34m=[0mconfig[34m,[0m net_dict_post_proc[34m=[0mnet_dict_post_proc[34m)[0m
[34m    locals:[0m
      rnn [34;1m= [0m[34m<global> [0m[34m<[0mmodule [36m'rnn'[0m [34mfrom [0m[36m'/u/makarov/returnn-parnia-2/rnn.py'[0m[34m>[0m
      rnn[34;1m.[0mengine [34;1m= [0m[34m<global> [0m[34m<[0mTFEngine[34m.[0mEngine object at 0x2ba09127aac8[34m>[0m
      rnn[34;1m.[0mengine[34;1m.[0minit_network_from_config [34;1m= [0m[34m<global> [0m[34m<[0mbound method Engine[34m.[0minit_network_from_config of [34m<[0mTFEngine[34m.[0mEngine object at 0x2ba09127aac8[34m>[0m[34m>[0m
      config [34;1m= [0m[34m<global> [0m[34m<[0mConfig[34m.[0mConfig object at 0x2ba1402e6748[34m>[0m
      net_dict_post_proc [34;1m= [0m[34m<local> [0m[34m<[0mfunction init_net[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mnet_dict_post_proc at 0x2ba1430d5158[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFEngine.py[0m[36m"[0m, [34mline [0m[35m936[0m, [34min [0minit_network_from_config
[34m    line: [0mself[34m.[0m_init_network[34m([0mnet_desc[34m=[0mnet_dict[34m,[0m epoch[34m=[0mself[34m.[0mepoch[34m)[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFEngine[34m.[0mEngine object at 0x2ba09127aac8[34m>[0m
      self[34;1m.[0m_init_network [34;1m= [0m[34m<local> [0m[34m<[0mbound method Engine[34m.[0m_init_network of [34m<[0mTFEngine[34m.[0mEngine object at 0x2ba09127aac8[34m>[0m[34m>[0m
      net_desc [34;1m= [0m[34m<not found>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm3_fw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm2_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m[34m,[0m [36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m}[0m[34m,[0m [36m'output'[0m[34m:[0m [34m{[0m[36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'target'[0m[34m:[0m [36m'classes'[0m[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      epoch [34;1m= [0m[34m<local> [0m183
      self[34;1m.[0mepoch [34;1m= [0m[34m<local> [0m183
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFEngine.py[0m[36m"[0m, [34mline [0m[35m1060[0m, [34min [0m_init_network
[34m    line: [0mself[34m.[0mnetwork[34m,[0m self[34m.[0mupdater [34m=[0m self[34m.[0mcreate_network[34m([0m
            config[34m=[0mself[34m.[0mconfig[34m,[0m
            rnd_seed[34m=[0mnet_random_seed[34m,[0m
            train_flag[34m=[0mtrain_flag[34m,[0m eval_flag[34m=[0mself[34m.[0muse_eval_flag[34m,[0m search_flag[34m=[0mself[34m.[0muse_search_flag[34m,[0m
            initial_learning_rate[34m=[0mgetattr[34m([0mself[34m,[0m [36m"initial_learning_rate"[0m[34m,[0m [34mNone[0m[34m)[0m[34m,[0m
            net_dict[34m=[0mnet_desc[34m)[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFEngine[34m.[0mEngine object at 0x2ba09127aac8[34m>[0m
      self[34;1m.[0mnetwork [34;1m= [0m[34m<local> [0m[34mNone[0m
      self[34;1m.[0mupdater [34;1m= [0m[34m<local> [0m[34mNone[0m
      self[34;1m.[0mcreate_network [34;1m= [0m[34m<local> [0m[34m<[0mbound method Engine[34m.[0mcreate_network of [34m<[0m[34mclass [0m[36m'TFEngine.Engine'[0m[34m>[0m[34m>[0m
      config [34;1m= [0m[34m<not found>[0m
      self[34;1m.[0mconfig [34;1m= [0m[34m<local> [0m[34m<[0mConfig[34m.[0mConfig object at 0x2ba1402e6748[34m>[0m
      rnd_seed [34;1m= [0m[34m<not found>[0m
      net_random_seed [34;1m= [0m[34m<local> [0m183
      train_flag [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m
      eval_flag [34;1m= [0m[34m<not found>[0m
      self[34;1m.[0muse_eval_flag [34;1m= [0m[34m<local> [0m[34mTrue[0m
      search_flag [34;1m= [0m[34m<not found>[0m
      self[34;1m.[0muse_search_flag [34;1m= [0m[34m<local> [0m[34mFalse[0m
      initial_learning_rate [34;1m= [0m[34m<not found>[0m
      getattr [34;1m= [0m[34m<builtin> [0m[34m<[0mbuilt[34m-[0m[34min [0mfunction getattr[34m>[0m
      net_dict [34;1m= [0m[34m<not found>[0m
      net_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm3_fw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm2_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m[34m,[0m [36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m}[0m[34m,[0m [36m'output'[0m[34m:[0m [34m{[0m[36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'target'[0m[34m:[0m [36m'classes'[0m[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFEngine.py[0m[36m"[0m, [34mline [0m[35m1091[0m, [34min [0mcreate_network
[34m    line: [0mnetwork[34m.[0mconstruct_from_dict[34m([0mnet_dict[34m)[0m
[34m    locals:[0m
      network [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      network[34;1m.[0mconstruct_from_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_from_dict of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm3_fw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm2_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m[34m,[0m [36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m}[0m[34m,[0m [36m'output'[0m[34m:[0m [34m{[0m[36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'target'[0m[34m:[0m [36m'classes'[0m[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m338[0m, [34min [0mconstruct_from_dict
[34m    line: [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m,[0m name[34m)[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm3_fw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm2_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m[34m,[0m [36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m}[0m[34m,[0m [36m'output'[0m[34m:[0m [34m{[0m[36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'target'[0m[34m:[0m [36m'classes'[0m[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<local> [0m[36m'ctc'[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.SoftmaxLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method LayerBase[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkLayer.SoftmaxLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'loss'[0m[34m:[0m [36m'ctc'[0m[34m,[0m [36m'target'[0m[34m:[0m [36m'classes'[0m[34m,[0m [36m'loss_opts'[0m[34m:[0m [34m{[0m[36m'ctc_opts'[0m[34m:[0m [34m{[0m[36m'ignore_longer_outputs_than_inputs'[0m[34m:[0m [34mTrue[0m[34m}[0m[34m,[0m [36m'beam_width'[0m[34m:[0m 1[34m}[0m[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1430f3b70[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m362[0m, [34min [0mtransform_config_dict
[34m    line: [0m[34mfor [0msrc_name [34min [0msrc_names
[34m    locals:[0m
      src_name [34;1m= [0m[34m<not found>[0m
      src_names [34;1m= [0m[34m<local> [0m[34m[[0m[36m'encoder'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 7[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m363[0m, [34min [0m<listcomp>
[34m    line: [0md[34m[[0m[36m"sources"[0m[34m][0m [34m=[0m [34m[[0m
            get_layer[34m([0msrc_name[34m)[0m
            [34mfor [0msrc_name [34min [0msrc_names
            [34mif [0m[34mnot [0msrc_name [34m=[0m[34m=[0m [36m"none"[0m[34m][0m
[34m    locals:[0m
      d [34;1m= [0m[34m<not found>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1430f3b70[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'encoder'[0m[34m,[0m len [34m=[0m 7
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm3_fw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm2_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m[34m,[0m [36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m}[0m[34m,[0m [36m'output'[0m[34m:[0m [34m{[0m[36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'target'[0m[34m:[0m [36m'classes'[0m[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'encoder'[0m[34m,[0m len [34m=[0m 7
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.CopyLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method LayerBase[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkLayer.CopyLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431a4158[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m362[0m, [34min [0mtransform_config_dict
[34m    line: [0m[34mfor [0msrc_name [34min [0msrc_names
[34m    locals:[0m
      src_name [34;1m= [0m[34m<not found>[0m
      src_names [34;1m= [0m[34m<local> [0m[34m[[0m[36m'lstm5_fw'[0m[34m,[0m [36m'lstm5_bw'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 8[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m363[0m, [34min [0m<listcomp>
[34m    line: [0md[34m[[0m[36m"sources"[0m[34m][0m [34m=[0m [34m[[0m
            get_layer[34m([0msrc_name[34m)[0m
            [34mfor [0msrc_name [34min [0msrc_names
            [34mif [0m[34mnot [0msrc_name [34m=[0m[34m=[0m [36m"none"[0m[34m][0m
[34m    locals:[0m
      d [34;1m= [0m[34m<not found>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431a4158[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm5_fw'[0m[34m,[0m len [34m=[0m 8
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm3_fw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm2_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m[34m,[0m [36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m}[0m[34m,[0m [36m'output'[0m[34m:[0m [34m{[0m[36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'target'[0m[34m:[0m [36m'classes'[0m[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm5_fw'[0m[34m,[0m len [34m=[0m 8
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method RecLayer[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431ad488[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkRecLayer.py[0m[36m"[0m, [34mline [0m[35m230[0m, [34min [0mtransform_config_dict
[34m    line: [0msuper[34m([0mRecLayer[34m,[0m cls[34m)[0m[34m.[0mtransform_config_dict[34m([0md[34m,[0m network[34m=[0mnetwork[34m,[0m get_layer[34m=[0mget_layer[34m)[0m  [37m# everything except "unit"[0m
[34m    locals:[0m
      super [34;1m= [0m[34m<builtin> [0m[34m<[0m[34mclass [0m[36m'super'[0m[34m>[0m
      RecLayer [34;1m= [0m[34m<global> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      cls [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      transform_config_dict [34;1m= [0m[34m<not found>[0m
      d [34;1m= [0m[34m<local> [0m[34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m
      network [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431ad488[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m362[0m, [34min [0mtransform_config_dict
[34m    line: [0m[34mfor [0msrc_name [34min [0msrc_names
[34m    locals:[0m
      src_name [34;1m= [0m[34m<not found>[0m
      src_names [34;1m= [0m[34m<local> [0m[34m[[0m[36m'lstm4_pool'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 10[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m363[0m, [34min [0m<listcomp>
[34m    line: [0md[34m[[0m[36m"sources"[0m[34m][0m [34m=[0m [34m[[0m
            get_layer[34m([0msrc_name[34m)[0m
            [34mfor [0msrc_name [34min [0msrc_names
            [34mif [0m[34mnot [0msrc_name [34m=[0m[34m=[0m [36m"none"[0m[34m][0m
[34m    locals:[0m
      d [34;1m= [0m[34m<not found>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431ad488[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm4_pool'[0m[34m,[0m len [34m=[0m 10
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm3_fw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm2_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m[34m,[0m [36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m}[0m[34m,[0m [36m'output'[0m[34m:[0m [34m{[0m[36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'target'[0m[34m:[0m [36m'classes'[0m[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm4_pool'[0m[34m,[0m len [34m=[0m 10
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method LayerBase[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'mode'[0m[34m:[0m [36m'max'[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'pool_size'[0m[34m:[0m [34m([0m1[34m,[0m[34m)[0m[34m,[0m [36m'trainable'[0m[34m:[0m [34mFalse[0m[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431ad598[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m362[0m, [34min [0mtransform_config_dict
[34m    line: [0m[34mfor [0msrc_name [34min [0msrc_names
[34m    locals:[0m
      src_name [34;1m= [0m[34m<not found>[0m
      src_names [34;1m= [0m[34m<local> [0m[34m[[0m[36m'lstm4_fw'[0m[34m,[0m [36m'lstm4_bw'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 8[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m363[0m, [34min [0m<listcomp>
[34m    line: [0md[34m[[0m[36m"sources"[0m[34m][0m [34m=[0m [34m[[0m
            get_layer[34m([0msrc_name[34m)[0m
            [34mfor [0msrc_name [34min [0msrc_names
            [34mif [0m[34mnot [0msrc_name [34m=[0m[34m=[0m [36m"none"[0m[34m][0m
[34m    locals:[0m
      d [34;1m= [0m[34m<not found>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431ad598[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm4_fw'[0m[34m,[0m len [34m=[0m 8
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm3_fw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm2_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m[34m,[0m [36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m}[0m[34m,[0m [36m'output'[0m[34m:[0m [34m{[0m[36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'target'[0m[34m:[0m [36m'classes'[0m[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm4_fw'[0m[34m,[0m len [34m=[0m 8
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method RecLayer[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431ad6a8[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkRecLayer.py[0m[36m"[0m, [34mline [0m[35m230[0m, [34min [0mtransform_config_dict
[34m    line: [0msuper[34m([0mRecLayer[34m,[0m cls[34m)[0m[34m.[0mtransform_config_dict[34m([0md[34m,[0m network[34m=[0mnetwork[34m,[0m get_layer[34m=[0mget_layer[34m)[0m  [37m# everything except "unit"[0m
[34m    locals:[0m
      super [34;1m= [0m[34m<builtin> [0m[34m<[0m[34mclass [0m[36m'super'[0m[34m>[0m
      RecLayer [34;1m= [0m[34m<global> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      cls [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      transform_config_dict [34;1m= [0m[34m<not found>[0m
      d [34;1m= [0m[34m<local> [0m[34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m
      network [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431ad6a8[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m362[0m, [34min [0mtransform_config_dict
[34m    line: [0m[34mfor [0msrc_name [34min [0msrc_names
[34m    locals:[0m
      src_name [34;1m= [0m[34m<not found>[0m
      src_names [34;1m= [0m[34m<local> [0m[34m[[0m[36m'lstm3_pool'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 10[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m363[0m, [34min [0m<listcomp>
[34m    line: [0md[34m[[0m[36m"sources"[0m[34m][0m [34m=[0m [34m[[0m
            get_layer[34m([0msrc_name[34m)[0m
            [34mfor [0msrc_name [34min [0msrc_names
            [34mif [0m[34mnot [0msrc_name [34m=[0m[34m=[0m [36m"none"[0m[34m][0m
[34m    locals:[0m
      d [34;1m= [0m[34m<not found>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431ad6a8[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm3_pool'[0m[34m,[0m len [34m=[0m 10
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm3_fw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm2_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m[34m,[0m [36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m}[0m[34m,[0m [36m'output'[0m[34m:[0m [34m{[0m[36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'target'[0m[34m:[0m [36m'classes'[0m[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm3_pool'[0m[34m,[0m len [34m=[0m 10
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method LayerBase[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'mode'[0m[34m:[0m [36m'max'[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'pool_size'[0m[34m:[0m [34m([0m1[34m,[0m[34m)[0m[34m,[0m [36m'trainable'[0m[34m:[0m [34mFalse[0m[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431ad7b8[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m362[0m, [34min [0mtransform_config_dict
[34m    line: [0m[34mfor [0msrc_name [34min [0msrc_names
[34m    locals:[0m
      src_name [34;1m= [0m[34m<not found>[0m
      src_names [34;1m= [0m[34m<local> [0m[34m[[0m[36m'lstm3_fw'[0m[34m,[0m [36m'lstm3_bw'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 8[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m363[0m, [34min [0m<listcomp>
[34m    line: [0md[34m[[0m[36m"sources"[0m[34m][0m [34m=[0m [34m[[0m
            get_layer[34m([0msrc_name[34m)[0m
            [34mfor [0msrc_name [34min [0msrc_names
            [34mif [0m[34mnot [0msrc_name [34m=[0m[34m=[0m [36m"none"[0m[34m][0m
[34m    locals:[0m
      d [34;1m= [0m[34m<not found>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431ad7b8[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm3_fw'[0m[34m,[0m len [34m=[0m 8
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm3_fw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm2_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m[34m,[0m [36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m}[0m[34m,[0m [36m'output'[0m[34m:[0m [34m{[0m[36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'target'[0m[34m:[0m [36m'classes'[0m[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm3_fw'[0m[34m,[0m len [34m=[0m 8
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method RecLayer[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431ad8c8[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkRecLayer.py[0m[36m"[0m, [34mline [0m[35m230[0m, [34min [0mtransform_config_dict
[34m    line: [0msuper[34m([0mRecLayer[34m,[0m cls[34m)[0m[34m.[0mtransform_config_dict[34m([0md[34m,[0m network[34m=[0mnetwork[34m,[0m get_layer[34m=[0mget_layer[34m)[0m  [37m# everything except "unit"[0m
[34m    locals:[0m
      super [34;1m= [0m[34m<builtin> [0m[34m<[0m[34mclass [0m[36m'super'[0m[34m>[0m
      RecLayer [34;1m= [0m[34m<global> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      cls [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      transform_config_dict [34;1m= [0m[34m<not found>[0m
      d [34;1m= [0m[34m<local> [0m[34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m
      network [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431ad8c8[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m362[0m, [34min [0mtransform_config_dict
[34m    line: [0m[34mfor [0msrc_name [34min [0msrc_names
[34m    locals:[0m
      src_name [34;1m= [0m[34m<not found>[0m
      src_names [34;1m= [0m[34m<local> [0m[34m[[0m[36m'lstm2_pool'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 10[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m363[0m, [34min [0m<listcomp>
[34m    line: [0md[34m[[0m[36m"sources"[0m[34m][0m [34m=[0m [34m[[0m
            get_layer[34m([0msrc_name[34m)[0m
            [34mfor [0msrc_name [34min [0msrc_names
            [34mif [0m[34mnot [0msrc_name [34m=[0m[34m=[0m [36m"none"[0m[34m][0m
[34m    locals:[0m
      d [34;1m= [0m[34m<not found>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431ad8c8[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm2_pool'[0m[34m,[0m len [34m=[0m 10
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm3_fw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm2_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m[34m,[0m [36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m}[0m[34m,[0m [36m'output'[0m[34m:[0m [34m{[0m[36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'target'[0m[34m:[0m [36m'classes'[0m[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm2_pool'[0m[34m,[0m len [34m=[0m 10
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method LayerBase[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'mode'[0m[34m:[0m [36m'max'[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'pool_size'[0m[34m:[0m [34m([0m1[34m,[0m[34m)[0m[34m,[0m [36m'trainable'[0m[34m:[0m [34mFalse[0m[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431ad9d8[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m362[0m, [34min [0mtransform_config_dict
[34m    line: [0m[34mfor [0msrc_name [34min [0msrc_names
[34m    locals:[0m
      src_name [34;1m= [0m[34m<not found>[0m
      src_names [34;1m= [0m[34m<local> [0m[34m[[0m[36m'lstm2_fw'[0m[34m,[0m [36m'lstm2_bw'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 8[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m363[0m, [34min [0m<listcomp>
[34m    line: [0md[34m[[0m[36m"sources"[0m[34m][0m [34m=[0m [34m[[0m
            get_layer[34m([0msrc_name[34m)[0m
            [34mfor [0msrc_name [34min [0msrc_names
            [34mif [0m[34mnot [0msrc_name [34m=[0m[34m=[0m [36m"none"[0m[34m][0m
[34m    locals:[0m
      d [34;1m= [0m[34m<not found>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431ad9d8[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm2_fw'[0m[34m,[0m len [34m=[0m 8
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm3_fw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm2_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m[34m,[0m [36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m}[0m[34m,[0m [36m'output'[0m[34m:[0m [34m{[0m[36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'target'[0m[34m:[0m [36m'classes'[0m[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm2_fw'[0m[34m,[0m len [34m=[0m 8
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method RecLayer[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431adae8[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkRecLayer.py[0m[36m"[0m, [34mline [0m[35m230[0m, [34min [0mtransform_config_dict
[34m    line: [0msuper[34m([0mRecLayer[34m,[0m cls[34m)[0m[34m.[0mtransform_config_dict[34m([0md[34m,[0m network[34m=[0mnetwork[34m,[0m get_layer[34m=[0mget_layer[34m)[0m  [37m# everything except "unit"[0m
[34m    locals:[0m
      super [34;1m= [0m[34m<builtin> [0m[34m<[0m[34mclass [0m[36m'super'[0m[34m>[0m
      RecLayer [34;1m= [0m[34m<global> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      cls [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      transform_config_dict [34;1m= [0m[34m<not found>[0m
      d [34;1m= [0m[34m<local> [0m[34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m
      network [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431adae8[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m362[0m, [34min [0mtransform_config_dict
[34m    line: [0m[34mfor [0msrc_name [34min [0msrc_names
[34m    locals:[0m
      src_name [34;1m= [0m[34m<not found>[0m
      src_names [34;1m= [0m[34m<local> [0m[34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 10[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m363[0m, [34min [0m<listcomp>
[34m    line: [0md[34m[[0m[36m"sources"[0m[34m][0m [34m=[0m [34m[[0m
            get_layer[34m([0msrc_name[34m)[0m
            [34mfor [0msrc_name [34min [0msrc_names
            [34mif [0m[34mnot [0msrc_name [34m=[0m[34m=[0m [36m"none"[0m[34m][0m
[34m    locals:[0m
      d [34;1m= [0m[34m<not found>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431adae8[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm1_pool'[0m[34m,[0m len [34m=[0m 10
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm3_fw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm2_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m[34m,[0m [36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m}[0m[34m,[0m [36m'output'[0m[34m:[0m [34m{[0m[36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'target'[0m[34m:[0m [36m'classes'[0m[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm1_pool'[0m[34m,[0m len [34m=[0m 10
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method LayerBase[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'mode'[0m[34m:[0m [36m'max'[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'pool_size'[0m[34m:[0m [34m([0m2[34m,[0m[34m)[0m[34m,[0m [36m'trainable'[0m[34m:[0m [34mFalse[0m[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431adbf8[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m362[0m, [34min [0mtransform_config_dict
[34m    line: [0m[34mfor [0msrc_name [34min [0msrc_names
[34m    locals:[0m
      src_name [34;1m= [0m[34m<not found>[0m
      src_names [34;1m= [0m[34m<local> [0m[34m[[0m[36m'lstm1_fw'[0m[34m,[0m [36m'lstm1_bw'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 8[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m363[0m, [34min [0m<listcomp>
[34m    line: [0md[34m[[0m[36m"sources"[0m[34m][0m [34m=[0m [34m[[0m
            get_layer[34m([0msrc_name[34m)[0m
            [34mfor [0msrc_name [34min [0msrc_names
            [34mif [0m[34mnot [0msrc_name [34m=[0m[34m=[0m [36m"none"[0m[34m][0m
[34m    locals:[0m
      d [34;1m= [0m[34m<not found>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431adbf8[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm1_fw'[0m[34m,[0m len [34m=[0m 8
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm3_fw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm2_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m[34m,[0m [36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m}[0m[34m,[0m [36m'output'[0m[34m:[0m [34m{[0m[36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'target'[0m[34m:[0m [36m'classes'[0m[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm1_fw'[0m[34m,[0m len [34m=[0m 8
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method RecLayer[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431add08[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkRecLayer.py[0m[36m"[0m, [34mline [0m[35m230[0m, [34min [0mtransform_config_dict
[34m    line: [0msuper[34m([0mRecLayer[34m,[0m cls[34m)[0m[34m.[0mtransform_config_dict[34m([0md[34m,[0m network[34m=[0mnetwork[34m,[0m get_layer[34m=[0mget_layer[34m)[0m  [37m# everything except "unit"[0m
[34m    locals:[0m
      super [34;1m= [0m[34m<builtin> [0m[34m<[0m[34mclass [0m[36m'super'[0m[34m>[0m
      RecLayer [34;1m= [0m[34m<global> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      cls [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      transform_config_dict [34;1m= [0m[34m<not found>[0m
      d [34;1m= [0m[34m<local> [0m[34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m
      network [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431add08[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m362[0m, [34min [0mtransform_config_dict
[34m    line: [0m[34mfor [0msrc_name [34min [0msrc_names
[34m    locals:[0m
      src_name [34;1m= [0m[34m<not found>[0m
      src_names [34;1m= [0m[34m<local> [0m[34m[[0m[36m'lstm0_pool'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 10[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m363[0m, [34min [0m<listcomp>
[34m    line: [0md[34m[[0m[36m"sources"[0m[34m][0m [34m=[0m [34m[[0m
            get_layer[34m([0msrc_name[34m)[0m
            [34mfor [0msrc_name [34min [0msrc_names
            [34mif [0m[34mnot [0msrc_name [34m=[0m[34m=[0m [36m"none"[0m[34m][0m
[34m    locals:[0m
      d [34;1m= [0m[34m<not found>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431add08[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm0_pool'[0m[34m,[0m len [34m=[0m 10
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm3_fw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm2_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m[34m,[0m [36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m}[0m[34m,[0m [36m'output'[0m[34m:[0m [34m{[0m[36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'target'[0m[34m:[0m [36m'classes'[0m[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm0_pool'[0m[34m,[0m len [34m=[0m 10
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method LayerBase[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'mode'[0m[34m:[0m [36m'max'[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'pool_size'[0m[34m:[0m [34m([0m3[34m,[0m[34m)[0m[34m,[0m [36m'trainable'[0m[34m:[0m [34mFalse[0m[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431ade18[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m362[0m, [34min [0mtransform_config_dict
[34m    line: [0m[34mfor [0msrc_name [34min [0msrc_names
[34m    locals:[0m
      src_name [34;1m= [0m[34m<not found>[0m
      src_names [34;1m= [0m[34m<local> [0m[34m[[0m[36m'lstm0_fw'[0m[34m,[0m [36m'lstm0_bw'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 8[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m363[0m, [34min [0m<listcomp>
[34m    line: [0md[34m[[0m[36m"sources"[0m[34m][0m [34m=[0m [34m[[0m
            get_layer[34m([0msrc_name[34m)[0m
            [34mfor [0msrc_name [34min [0msrc_names
            [34mif [0m[34mnot [0msrc_name [34m=[0m[34m=[0m [36m"none"[0m[34m][0m
[34m    locals:[0m
      d [34;1m= [0m[34m<not found>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431ade18[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm0_fw'[0m[34m,[0m len [34m=[0m 8
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm3_fw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm2_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m[34m,[0m [36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m}[0m[34m,[0m [36m'output'[0m[34m:[0m [34m{[0m[36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'target'[0m[34m:[0m [36m'classes'[0m[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm0_fw'[0m[34m,[0m len [34m=[0m 8
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method RecLayer[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431adf28[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkRecLayer.py[0m[36m"[0m, [34mline [0m[35m230[0m, [34min [0mtransform_config_dict
[34m    line: [0msuper[34m([0mRecLayer[34m,[0m cls[34m)[0m[34m.[0mtransform_config_dict[34m([0md[34m,[0m network[34m=[0mnetwork[34m,[0m get_layer[34m=[0mget_layer[34m)[0m  [37m# everything except "unit"[0m
[34m    locals:[0m
      super [34;1m= [0m[34m<builtin> [0m[34m<[0m[34mclass [0m[36m'super'[0m[34m>[0m
      RecLayer [34;1m= [0m[34m<global> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      cls [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      transform_config_dict [34;1m= [0m[34m<not found>[0m
      d [34;1m= [0m[34m<local> [0m[34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m
      network [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431adf28[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m362[0m, [34min [0mtransform_config_dict
[34m    line: [0m[34mfor [0msrc_name [34min [0msrc_names
[34m    locals:[0m
      src_name [34;1m= [0m[34m<not found>[0m
      src_names [34;1m= [0m[34m<local> [0m[34m[[0m[36m'source'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 6[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m363[0m, [34min [0m<listcomp>
[34m    line: [0md[34m[[0m[36m"sources"[0m[34m][0m [34m=[0m [34m[[0m
            get_layer[34m([0msrc_name[34m)[0m
            [34mfor [0msrc_name [34min [0msrc_names
            [34mif [0m[34mnot [0msrc_name [34m=[0m[34m=[0m [36m"none"[0m[34m][0m
[34m    locals:[0m
      d [34;1m= [0m[34m<not found>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2ba1431adf28[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'source'[0m[34m,[0m len [34m=[0m 6
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'lstm3_fw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm2_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m[34m,[0m [36m'lstm2_bw'[0m[34m:[0m [34m{[0m[36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm1_pool'[0m[34m][0m[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m}[0m[34m,[0m [36m'output'[0m[34m:[0m [34m{[0m[36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'target'[0m[34m:[0m [36m'classes'[0m[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'source'[0m[34m,[0m len [34m=[0m 6
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m418[0m, [34min [0mconstruct_layer
[34m    line: [0m[34mreturn [0madd_layer[34m([0mname[34m=[0mname[34m,[0m layer_class[34m=[0mlayer_class[34m,[0m [34m*[0m[34m*[0mlayer_desc[34m)[0m
[34m    locals:[0m
      add_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0madd_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      name [34;1m= [0m[34m<local> [0m[36m'source'[0m[34m,[0m len [34m=[0m 6
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.EvalLayer'[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'sources'[0m[34m:[0m [34m[[0m[34m<[0mSourceLayer [36m'data'[0m out_type[34m=[0mData[34m([0mshape[34m=[0m[34m([0m[34mNone[0m[34m,[0m 40[34m)[0m[34m)[0m[34m>[0m[34m][0m[34m,[0m [36m'eval'[0m[34m:[0m [36m"self.network.get_config().typed_value('transform')(source(0), network=self.network)"[0m[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m507[0m, [34min [0madd_layer
[34m    line: [0mlayer [34m=[0m self[34m.[0m_create_layer[34m([0mname[34m=[0mname[34m,[0m layer_class[34m=[0mlayer_class[34m,[0m [34m*[0m[34m*[0mlayer_desc[34m)[0m
[34m    locals:[0m
      layer [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0m_create_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0m_create_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      name [34;1m= [0m[34m<local> [0m[36m'source'[0m[34m,[0m len [34m=[0m 6
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.EvalLayer'[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'sources'[0m[34m:[0m [34m[[0m[34m<[0mSourceLayer [36m'data'[0m out_type[34m=[0mData[34m([0mshape[34m=[0m[34m([0m[34mNone[0m[34m,[0m 40[34m)[0m[34m)[0m[34m>[0m[34m][0m[34m,[0m [36m'eval'[0m[34m:[0m [36m"self.network.get_config().typed_value('transform')(source(0), network=self.network)"[0m[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m464[0m, [34min [0m_create_layer
[34m    line: [0mlayer [34m=[0m layer_class[34m([0m[34m*[0m[34m*[0mlayer_desc[34m)[0m
[34m    locals:[0m
      layer [34;1m= [0m[34m<not found>[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.EvalLayer'[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'sources'[0m[34m:[0m [34m[[0m[34m<[0mSourceLayer [36m'data'[0m out_type[34m=[0mData[34m([0mshape[34m=[0m[34m([0m[34mNone[0m[34m,[0m 40[34m)[0m[34m)[0m[34m>[0m[34m][0m[34m,[0m [36m'name'[0m[34m:[0m [36m'source'[0m[34m,[0m [36m'eval'[0m[34m:[0m [36m"self.network.get_config().typed_value('transform')(source(0), network=self.network)"[0m[34m,[0m [36m'output'[0m[34m:[0m Data[34m([0mname[34m=[0m[36m'source_output'[0m[34m,[0m shape[34m=[0m[34m([0m[34mNone[0m[34m,[0m 40[34m)[0m[34m)[0m[34m,[0m [36m'network'[0m[34m:[0m [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train...[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m4426[0m, [34min [0m__init__
[34m    line: [0msuper[34m([0mEvalLayer[34m,[0m self[34m)[0m[34m.[0m__init__[34m([0mkind[34m=[0m[36m"eval"[0m[34m,[0m eval[34m=[0meval[34m,[0m [34m*[0m[34m*[0mkwargs[34m)[0m
[34m    locals:[0m
      super [34;1m= [0m[34m<builtin> [0m[34m<[0m[34mclass [0m[36m'super'[0m[34m>[0m
      EvalLayer [34;1m= [0m[34m<global> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.EvalLayer'[0m[34m>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mEvalLayer [36m'source'[0m out_type[34m=[0mData[34m([0mshape[34m=[0m[34m([0m[34mNone[0m[34m,[0m 40[34m)[0m[34m)[0m[34m>[0m
      __init__ [34;1m= [0m[34m<not found>[0m
      kind [34;1m= [0m[34m<not found>[0m
      eval [34;1m= [0m[34m<local> [0m[36m"self.network.get_config().typed_value('transform')(source(0), network=self.network)"[0m[34m,[0m len [34m=[0m 83
      kwargs [34;1m= [0m[34m<local> [0m[34m{[0m[36m'sources'[0m[34m:[0m [34m[[0m[34m<[0mSourceLayer [36m'data'[0m out_type[34m=[0mData[34m([0mshape[34m=[0m[34m([0m[34mNone[0m[34m,[0m 40[34m)[0m[34m)[0m[34m>[0m[34m][0m[34m,[0m [36m'name'[0m[34m:[0m [36m'source'[0m[34m,[0m [36m'network'[0m[34m:[0m [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m,[0m [36m'output'[0m[34m:[0m Data[34m([0mname[34m=[0m[36m'source_output'[0m[34m,[0m shape[34m=[0m[34m([0m[34mNone[0m[34m,[0m 40[34m)[0m[34m)[0m[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m4285[0m, [34min [0m__init__
[34m    line: [0mx [34m=[0m op[34m([0msources[34m)[0m
[34m    locals:[0m
      x [34;1m= [0m[34m<not found>[0m
      op [34;1m= [0m[34m<local> [0m[34m<[0mfunction CombineLayer[34m.[0m_get_op[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mwrap_eval_op at 0x2ba1431b1158[34m>[0m
      sources [34;1m= [0m[34m<local> [0m[34m[[0m[34m<[0mSourceLayer [36m'data'[0m out_type[34m=[0mData[34m([0mshape[34m=[0m[34m([0m[34mNone[0m[34m,[0m 40[34m)[0m[34m)[0m[34m>[0m[34m][0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m4409[0m, [34min [0mwrap_eval_op
[34m    line: [0m[34mreturn [0mself[34m.[0m_op_kind_eval[34m([0msources[34m,[0m eval_str[34m=[0meval_str[34m,[0m eval_locals[34m=[0meval_locals[34m)[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mEvalLayer [36m'source'[0m out_type[34m=[0mData[34m([0mshape[34m=[0m[34m([0m[34mNone[0m[34m,[0m 40[34m)[0m[34m)[0m[34m>[0m
      self[34;1m.[0m_op_kind_eval [34;1m= [0m[34m<local> [0m[34m<[0mbound method CombineLayer[34m.[0m_op_kind_eval of [34m<[0mEvalLayer [36m'source'[0m out_type[34m=[0mData[34m([0mshape[34m=[0m[34m([0m[34mNone[0m[34m,[0m 40[34m)[0m[34m)[0m[34m>[0m[34m>[0m
      sources [34;1m= [0m[34m<local> [0m[34m[[0m[34m<[0mSourceLayer [36m'data'[0m out_type[34m=[0mData[34m([0mshape[34m=[0m[34m([0m[34mNone[0m[34m,[0m 40[34m)[0m[34m)[0m[34m>[0m[34m][0m
      eval_str [34;1m= [0m[34m<local> [0m[36m"self.network.get_config().typed_value('transform')(source(0), network=self.network)"[0m[34m,[0m len [34m=[0m 83
      eval_locals [34;1m= [0m[34m<local> [0m[34mNone[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m4399[0m, [34min [0m_op_kind_eval
[34m    line: [0mx [34m=[0m eval[34m([0meval_str[34m,[0m vs[34m)[0m
[34m    locals:[0m
      x [34;1m= [0m[34m<not found>[0m
      eval [34;1m= [0m[34m<builtin> [0m[34m<[0mbuilt[34m-[0m[34min [0mfunction eval[34m>[0m
      eval_str [34;1m= [0m[34m<local> [0m[36m"self.network.get_config().typed_value('transform')(source(0), network=self.network)"[0m[34m,[0m len [34m=[0m 83
      vs [34;1m= [0m[34m<local> [0m[34m{[0m[36m'self'[0m[34m:[0m [34m<[0mEvalLayer [36m'source'[0m out_type[34m=[0mData[34m([0mshape[34m=[0m[34m([0m[34mNone[0m[34m,[0m 40[34m)[0m[34m)[0m[34m>[0m[34m,[0m [36m'dot'[0m[34m:[0m [34m<[0mfunction dot at 0x2ba1402ad730[34m>[0m[34m,[0m [36m'mem_usage_for_dev'[0m[34m:[0m [34m<[0mfunction mem_usage_for_dev at 0x2ba1402b5ea0[34m>[0m[34m,[0m [36m'expand_multiple_dims'[0m[34m:[0m [34m<[0mfunction expand_multiple_dims at 0x2ba1402ae598[34m>[0m[34m,[0m [36m'nested_get_shapes'[0m[34m:[0m [34m<[0mfunction nested_get_shapes [34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 217
  [34;1mFile [0m[36m"[0m[36;1m<string>[0m[36m"[0m, [34mline [0m[35m1[0m, [34min [0m<module>
[34m    -- code not available --[0m
  [34;1mFile [0m[36m"/work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/lstm-prior/[0m[36;1mexp3.ctc.ogg-priorAtt-lr0005-k10-specAug.config[0m[36m"[0m, [34mline [0m[35m178[0m, [34min [0mtransform
[34m    line: [0mx [34m=[0m network[34m.[0mcond_on_train[34m([0mget_masked[34m,[0m [34mlambda[0m[34m:[0m x[34m)[0m
[34m    locals:[0m
      x [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'source/clip_by_value:0'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m,[0m 40[34m)[0m dtype[34m=[0mfloat32[34m>[0m
      network [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      network[34;1m.[0mcond_on_train [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mcond_on_train of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      get_masked [34;1m= [0m[34m<local> [0m[34m<[0mfunction transform[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_masked at 0x2ba0d7d13488[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m1027[0m, [34min [0mcond_on_train
[34m    line: [0m[34mreturn [0mcond[34m([0mself[34m.[0mtrain_flag[34m,[0m fn_train[34m,[0m fn_eval[34m)[0m
[34m    locals:[0m
      cond [34;1m= [0m[34m<local> [0m[34m<[0mfunction cond at 0x2ba1402b32f0[34m>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mtrain_flag [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m
      fn_train [34;1m= [0m[34m<local> [0m[34m<[0mfunction transform[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_masked at 0x2ba0d7d13488[34m>[0m
      fn_eval [34;1m= [0m[34m<local> [0m[34m<[0mfunction transform[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0m[34m<[0m[34mlambda[0m[34m>[0m at 0x2ba1431b1268[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFUtil.py[0m[36m"[0m, [34mline [0m[35m3768[0m, [34min [0mcond
[34m    line: [0m[34mreturn [0mcontrol_flow_ops[34m.[0mcond[34m([0mpred[34m,[0m fn1[34m,[0m fn2[34m,[0m name[34m=[0mname[34m)[0m
[34m    locals:[0m
      control_flow_ops [34;1m= [0m[34m<local> [0m[34m<[0mmodule [36m'tensorflow.python.ops.control_flow_ops'[0m [34mfrom [0m[36m'/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/python/ops/control_flow_ops.py'[0m[34m>[0m
      control_flow_ops[34;1m.[0mcond [34;1m= [0m[34m<local> [0m[34m<[0mfunction cond at 0x2ba13f308d08[34m>[0m
      pred [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m
      fn1 [34;1m= [0m[34m<local> [0m[34m<[0mfunction transform[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_masked at 0x2ba0d7d13488[34m>[0m
      fn2 [34;1m= [0m[34m<local> [0m[34m<[0mfunction transform[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0m[34m<[0m[34mlambda[0m[34m>[0m at 0x2ba1431b1268[34m>[0m
      name [34;1m= [0m[34m<local> [0m[34mNone[0m
  [34;1mFile [0m[36m"/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/python/util/[0m[36;1mdeprecation.py[0m[36m"[0m, [34mline [0m[35m454[0m, [34min [0mnew_func
[34m    line: [0m[34mreturn [0mfunc[34m([0m[34m*[0margs[34m,[0m [34m*[0m[34m*[0mkwargs[34m)[0m
[34m    locals:[0m
      func [34;1m= [0m[34m<local> [0m[34m<[0mfunction cond at 0x2ba13f308bf8[34m>[0m
      args [34;1m= [0m[34m<local> [0m[34m([0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m,[0m [34m<[0mfunction transform[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_masked at 0x2ba0d7d13488[34m>[0m[34m,[0m [34m<[0mfunction transform[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0m[34m<[0m[34mlambda[0m[34m>[0m at 0x2ba1431b1268[34m>[0m[34m)[0m
      kwargs [34;1m= [0m[34m<local> [0m[34m{[0m[36m'name'[0m[34m:[0m [34mNone[0m[34m}[0m
  [34;1mFile [0m[36m"/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/python/ops/[0m[36;1mcontrol_flow_ops.py[0m[36m"[0m, [34mline [0m[35m2048[0m, [34min [0mcond
[34m    line: [0morig_res_t[34m,[0m res_t [34m=[0m context_t[34m.[0mBuildCondBranch[34m([0mtrue_fn[34m)[0m
[34m    locals:[0m
      orig_res_t [34;1m= [0m[34m<not found>[0m
      res_t [34;1m= [0m[34m<not found>[0m
      context_t [34;1m= [0m[34m<local> [0m[34m<[0mtensorflow[34m.[0mpython[34m.[0mops[34m.[0mcontrol_flow_ops[34m.[0mCondContext object at 0x2ba1431b55f8[34m>[0m
      context_t[34;1m.[0mBuildCondBranch [34;1m= [0m[34m<local> [0m[34m<[0mbound method CondContext[34m.[0mBuildCondBranch of [34m<[0mtensorflow[34m.[0mpython[34m.[0mops[34m.[0mcontrol_flow_ops[34m.[0mCondContext object at 0x2ba1431b55f8[34m>[0m[34m>[0m
      true_fn [34;1m= [0m[34m<local> [0m[34m<[0mfunction transform[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_masked at 0x2ba0d7d13488[34m>[0m
  [34;1mFile [0m[36m"/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/python/ops/[0m[36;1mcontrol_flow_ops.py[0m[36m"[0m, [34mline [0m[35m1895[0m, [34min [0mBuildCondBranch
[34m    line: [0moriginal_result [34m=[0m fn[34m([0m[34m)[0m
[34m    locals:[0m
      original_result [34;1m= [0m[34m<not found>[0m
      fn [34;1m= [0m[34m<local> [0m[34m<[0mfunction transform[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_masked at 0x2ba0d7d13488[34m>[0m
  [34;1mFile [0m[36m"/work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/lstm-prior/[0m[36;1mexp3.ctc.ogg-priorAtt-lr0005-k10-specAug.config[0m[36m"[0m, [34mline [0m[35m174[0m, [34min [0mget_masked
[34m    line: [0mx_masked [34m=[0m random_mask[34m([0mx_masked[34m,[0m axis[34m=[0m1[34m,[0m min_num[34m=[0m1[34m,[0m max_num[34m=[0mtf[34m.[0mmaximum[34m([0mtf[34m.[0mshape[34m([0mx[34m)[0m[34m[[0m1[34m][0m [34m/[0m[34m/[0m 100[34m,[0m 1[34m)[0m[34m,[0m max_dims[34m=[0m20[34m)[0m
[34m    locals:[0m
      x_masked [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'source/clip_by_value:0'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m,[0m 40[34m)[0m dtype[34m=[0mfloat32[34m>[0m
      random_mask [34;1m= [0m[34m<global> [0m[34m<[0mfunction random_mask at 0x2ba1402f3840[34m>[0m
      axis [34;1m= [0m[34m<not found>[0m
      min_num [34;1m= [0m[34m<not found>[0m
      max_num [34;1m= [0m[34m<not found>[0m
      tf [34;1m= [0m[34m<local> [0m[34m<[0mmodule [36m'tensorflow'[0m [34mfrom [0m[36m'/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/__init__.py'[0m[34m>[0m
      tf[34;1m.[0mmaximum [34;1m= [0m[34m<local> [0m[34m<[0mfunction maximum at 0x2ba13eef0510[34m>[0m
      tf[34;1m.[0mshape [34;1m= [0m[34m<local> [0m[34m<[0mfunction shape at 0x2ba13ef25400[34m>[0m
      x [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'source/clip_by_value:0'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m,[0m 40[34m)[0m dtype[34m=[0mfloat32[34m>[0m
      max_dims [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/lstm-prior/[0m[36;1mexp3.ctc.ogg-priorAtt-lr0005-k10-specAug.config[0m[36m"[0m, [34mline [0m[35m163[0m, [34min [0mrandom_mask
[34m    line: [0m_[34m,[0m x [34m=[0m tf[34m.[0mwhile_loop[34m([0m
              cond[34m=[0m[34mlambda [0mi[34m,[0m _[34m:[0m tf[34m.[0mless[34m([0mi[34m,[0m tf[34m.[0mreduce_max[34m([0mnum[34m)[0m[34m)[0m[34m,[0m
              body[34m=[0m[34mlambda [0mi[34m,[0m x[34m:[0m [34m([0m
                  i [34m+[0m 1[34m,[0m 
                  tf[34m.[0mwhere[34m([0m
                      tf[34m.[0mless[34m([0mi[34m,[0m num[34m)[0m[34m,[0m
                      _mask[34m([0mx[34m,[0m axis[34m=[0maxis[34m,[0m pos[34m=[0mindices[34m[[0m[34m:[0m[34m,[0m i[34m][0m[34m,[0m max_amount[34m=[0mmax_dims[34m)[0m[34m,[0m
                      x[34m)[0m[34m)[0m[34m,[0m
              loop_vars[34m=[0m[34m([0m0[34m,[0m x[34m)[0m[34m)[0m
[34m    locals:[0m
      _ [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'source/cond/TopKV2:0'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m)[0m dtype[34m=[0mfloat32[34m>[0m
      x [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'source/clip_by_value:0'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m,[0m 40[34m)[0m dtype[34m=[0mfloat32[34m>[0m
      tf [34;1m= [0m[34m<local> [0m[34m<[0mmodule [36m'tensorflow'[0m [34mfrom [0m[36m'/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/__init__.py'[0m[34m>[0m
      tf[34;1m.[0mwhile_loop [34;1m= [0m[34m<local> [0m[34m<[0mfunction while_loop at 0x2ba13f308a60[34m>[0m
      cond [34;1m= [0m[34m<not found>[0m
      i [34;1m= [0m[34m<not found>[0m
      tf[34;1m.[0mless [34;1m= [0m[34m<local> [0m[34m<[0mfunction less at 0x2ba13eeee8c8[34m>[0m
      tf[34;1m.[0mreduce_max [34;1m= [0m[34m<local> [0m[34m<[0mfunction reduce_max at 0x2ba13f22d598[34m>[0m
      num [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'source/cond/random_uniform:0'[0m shape[34m=[0m[34m([0m?[34m,[0m[34m)[0m dtype[34m=[0mint32[34m>[0m
      body [34;1m= [0m[34m<not found>[0m
      tf[34;1m.[0mwhere [34;1m= [0m[34m<local> [0m[34m<[0mfunction where at 0x2ba13f04f510[34m>[0m
      _mask [34;1m= [0m[34m<global> [0m[34m<[0mfunction _mask at 0x2ba1402f37b8[34m>[0m
      axis [34;1m= [0m[34m<local> [0m1
      pos [34;1m= [0m[34m<not found>[0m
      indices [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'source/cond/TopKV2:1'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m)[0m dtype[34m=[0mint32[34m>[0m
      max_amount [34;1m= [0m[34m<not found>[0m
      max_dims [34;1m= [0m[34m<local> [0m20
      loop_vars [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/python/ops/[0m[36;1mcontrol_flow_ops.py[0m[36m"[0m, [34mline [0m[35m3232[0m, [34min [0mwhile_loop
[34m    line: [0mresult [34m=[0m loop_context[34m.[0mBuildLoop[34m([0mcond[34m,[0m body[34m,[0m loop_vars[34m,[0m shape_invariants[34m,[0m
                                          return_same_structure[34m)[0m
[34m    locals:[0m
      result [34;1m= [0m[34m<not found>[0m
      loop_context [34;1m= [0m[34m<local> [0m[34m<[0mtensorflow[34m.[0mpython[34m.[0mops[34m.[0mcontrol_flow_ops[34m.[0mWhileContext object at 0x2ba143e52e10[34m>[0m
      loop_context[34;1m.[0mBuildLoop [34;1m= [0m[34m<local> [0m[34m<[0mbound method WhileContext[34m.[0mBuildLoop of [34m<[0mtensorflow[34m.[0mpython[34m.[0mops[34m.[0mcontrol_flow_ops[34m.[0mWhileContext object at 0x2ba143e52e10[34m>[0m[34m>[0m
      cond [34;1m= [0m[34m<local> [0m[34m<[0mfunction random_mask[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0m[34m<[0m[34mlambda[0m[34m>[0m at 0x2ba0d7d269d8[34m>[0m
      body [34;1m= [0m[34m<local> [0m[34m<[0mfunction random_mask[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0m[34m<[0m[34mlambda[0m[34m>[0m at 0x2ba1431b1510[34m>[0m
      loop_vars [34;1m= [0m[34m<local> [0m[34m([0m0[34m,[0m [34m<[0mtf[34m.[0mTensor [36m'source/clip_by_value:0'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m,[0m 40[34m)[0m dtype[34m=[0mfloat32[34m>[0m[34m)[0m
      shape_invariants [34;1m= [0m[34m<local> [0m[34mNone[0m
      return_same_structure [34;1m= [0m[34m<local> [0m[34mFalse[0m
  [34;1mFile [0m[36m"/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/python/ops/[0m[36;1mcontrol_flow_ops.py[0m[36m"[0m, [34mline [0m[35m2952[0m, [34min [0mBuildLoop
[34m    line: [0moriginal_body_result[34m,[0m exit_vars [34m=[0m self[34m.[0m_BuildLoop[34m([0m
              pred[34m,[0m body[34m,[0m original_loop_vars[34m,[0m loop_vars[34m,[0m shape_invariants[34m)[0m
[34m    locals:[0m
      original_body_result [34;1m= [0m[34m<not found>[0m
      exit_vars [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mtensorflow[34m.[0mpython[34m.[0mops[34m.[0mcontrol_flow_ops[34m.[0mWhileContext object at 0x2ba143e52e10[34m>[0m
      self[34;1m.[0m_BuildLoop [34;1m= [0m[34m<local> [0m[34m<[0mbound method WhileContext[34m.[0m_BuildLoop of [34m<[0mtensorflow[34m.[0mpython[34m.[0mops[34m.[0mcontrol_flow_ops[34m.[0mWhileContext object at 0x2ba143e52e10[34m>[0m[34m>[0m
      pred [34;1m= [0m[34m<local> [0m[34m<[0mfunction random_mask[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0m[34m<[0m[34mlambda[0m[34m>[0m at 0x2ba0d7d269d8[34m>[0m
      body [34;1m= [0m[34m<local> [0m[34m<[0mfunction random_mask[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0m[34m<[0m[34mlambda[0m[34m>[0m at 0x2ba1431b1510[34m>[0m
      original_loop_vars [34;1m= [0m[34m<local> [0m[34m([0m0[34m,[0m [34m<[0mtf[34m.[0mTensor [36m'source/clip_by_value:0'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m,[0m 40[34m)[0m dtype[34m=[0mfloat32[34m>[0m[34m)[0m
      loop_vars [34;1m= [0m[34m<local> [0m[34m[[0m[34m<[0mtf[34m.[0mTensor [36m'source/cond/while/Const:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mint32[34m>[0m[34m,[0m [34m<[0mtf[34m.[0mTensor [36m'source/clip_by_value:0'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m,[0m 40[34m)[0m dtype[34m=[0mfloat32[34m>[0m[34m][0m
      shape_invariants [34;1m= [0m[34m<local> [0m[34mNone[0m
  [34;1mFile [0m[36m"/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/python/ops/[0m[36;1mcontrol_flow_ops.py[0m[36m"[0m, [34mline [0m[35m2887[0m, [34min [0m_BuildLoop
[34m    line: [0mbody_result [34m=[0m body[34m([0m[34m*[0mpacked_vars_for_body[34m)[0m
[34m    locals:[0m
      body_result [34;1m= [0m[34m<not found>[0m
      body [34;1m= [0m[34m<local> [0m[34m<[0mfunction random_mask[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0m[34m<[0m[34mlambda[0m[34m>[0m at 0x2ba1431b1510[34m>[0m
      packed_vars_for_body [34;1m= [0m[34m<local> [0m[34m([0m[34m<[0mtf[34m.[0mTensor [36m'source/cond/while/Identity:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mint32[34m>[0m[34m,[0m [34m<[0mtf[34m.[0mTensor [36m'source/cond/while/Identity_1:0'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m,[0m 40[34m)[0m dtype[34m=[0mfloat32[34m>[0m[34m)[0m
  [34;1mFile [0m[36m"/work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/lstm-prior/[0m[36;1mexp3.ctc.ogg-priorAtt-lr0005-k10-specAug.config[0m[36m"[0m, [34mline [0m[35m161[0m, [34min [0m<lambda>
[34m    line: [0m_mask[34m([0mx[34m,[0m axis[34m=[0maxis[34m,[0m pos[34m=[0mindices[34m[[0m[34m:[0m[34m,[0m i[34m][0m[34m,[0m max_amount[34m=[0mmax_dims[34m)[0m[34m,[0m
[34m    locals:[0m
      _mask [34;1m= [0m[34m<global> [0m[34m<[0mfunction _mask at 0x2ba1402f37b8[34m>[0m
      x [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'source/cond/while/Identity_1:0'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m,[0m 40[34m)[0m dtype[34m=[0mfloat32[34m>[0m
      axis [34;1m= [0m[34m<local> [0m1
      pos [34;1m= [0m[34m<not found>[0m
      indices [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'source/cond/TopKV2:1'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m)[0m dtype[34m=[0mint32[34m>[0m
      i [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'source/cond/while/Identity:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mint32[34m>[0m
      max_amount [34;1m= [0m[34m<not found>[0m
      max_dims [34;1m= [0m[34m<local> [0m20
  [34;1mFile [0m[36m"/work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/lstm-prior/[0m[36;1mexp3.ctc.ogg-priorAtt-lr0005-k10-specAug.config[0m[36m"[0m, [34mline [0m[35m133[0m, [34min [0m_mask
[34m    line: [0m[34mfrom [0mTFUtil [34mimport [0mwhere_bc
[34m    locals:[0m
      TFUtil [34;1m= [0m[34m<not found>[0m
      where_bc [34;1m= [0m[34m<not found>[0m
[31mImportError[0m: cannot import name 'where_bc'
fatal: Not a git repository (or any parent up to mount point /u/makarov)
Stopping at filesystem boundary (GIT_DISCOVERY_ACROSS_FILESYSTEM not set).
2019-07-05 14:23:29.711743: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-07-05 14:23:30.784558: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:82:00.0
totalMemory: 10.92GiB freeMemory: 10.76GiB
2019-07-05 14:23:30.784625: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-05 14:23:30.784639: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-05 14:23:30.784647: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-05 14:23:30.784653: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-05 14:23:30.789734: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-05 14:23:31.207408: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-05 14:23:31.207465: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-05 14:23:31.207474: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-05 14:23:31.207855: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/device:GPU:0 with 10405 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:82:00.0, compute capability: 6.1)
ERROR: cannot read default config file. using build-in defaults [main]
LOG: connected to ('10.6.100.1', 10321)
LOG: destination: /var/tmp/makarov/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-other.zip
LOG: request: /work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-other.zip
LOG: using existing copy /var/tmp/irie/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-other.zip
2019-07-05 14:23:32.315827: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-05 14:23:32.315918: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-05 14:23:32.315933: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-05 14:23:32.315944: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-05 14:23:32.316208: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10405 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:82:00.0, compute capability: 6.1)
