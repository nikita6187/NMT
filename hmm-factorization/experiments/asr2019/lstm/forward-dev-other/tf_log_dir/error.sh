2019-07-05 11:50:25.176594: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-07-05 11:50:26.575777: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
name: GeForce GTX 980 major: 5 minor: 2 memoryClockRate(GHz): 1.266
pciBusID: 0000:83:00.0
totalMemory: 3.95GiB freeMemory: 3.87GiB
2019-07-05 11:50:26.575847: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-05 11:50:26.575867: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-05 11:50:26.575877: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-05 11:50:26.575886: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-05 11:50:26.580061: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-05 11:50:26.960282: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-05 11:50:26.960341: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-05 11:50:26.960348: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-05 11:50:26.960547: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/device:GPU:0 with 3593 MB memory) -> physical GPU (device: 0, name: GeForce GTX 980, pci bus id: 0000:83:00.0, compute capability: 5.2)
ERROR: cannot read default config file. using build-in defaults [main]
LOG: connected to ('10.6.100.1', 10321)
LOG: destination: /var/tmp/makarov/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-other.zip
LOG: request: /work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-other.zip
LOG: using existing copy /var/tmp/zeyer/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-other.zip
2019-07-05 11:50:28.559975: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-05 11:50:28.560079: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-05 11:50:28.560102: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-05 11:50:28.560120: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-05 11:50:28.560412: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3593 MB memory) -> physical GPU (device: 0, name: GeForce GTX 980, pci bus id: 0000:83:00.0, compute capability: 5.2)
[31;1mEXCEPTION[0m
[34mTraceback (most recent call last):[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/tools/[0m[36;1mget-attention-weights.py[0m[36m"[0m, [34mline [0m[35m414[0m, [34min [0m<module>
[34m    line: [0mmain[34m([0msys[34m.[0margv[34m)[0m
[34m    locals:[0m
      main [34;1m= [0m[34m<local> [0m[34m<[0mfunction main at 0x2b636988b8c8[34m>[0m
      sys [34;1m= [0m[34m<local> [0m[34m<[0mmodule [36m'sys'[0m [34m([0mbuilt[34m-[0m[34min[0m[34m)[0m[34m>[0m
      sys[34;1m.[0margv [34;1m= [0m[34m<local> [0m[34m[[0m[36m'/u/makarov/returnn-parnia/tools/get-attention-weights.py'[0m[34m,[0m [36m'/u/bahar/workspace/asr/librispeech/test-20190121/config-train/base2.smlp2.specaug.datarndperm_noscale.bs18k.curric3.retrain1.config'[0m[34m,[0m [36m'--epoch'[0m[34m,[0m [36m'250'[0m[34m,[0m [36m'--data'[0m[34m,[0m [36m'config:get_dataset("dev-other")'[0m[34m,[0m [36m'--dump_dir'[0m[34m,[0m [36m'/u/makarov/makarov/NMT/..., len = 18, _[0]: {len = 56}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/tools/[0m[36;1mget-attention-weights.py[0m[36m"[0m, [34mline [0m[35m231[0m, [34min [0mmain
[34m    line: [0minit_net[34m([0margs[34m,[0m layers[34m)[0m
[34m    locals:[0m
      init_net [34;1m= [0m[34m<global> [0m[34m<[0mfunction init_net at 0x2b636988b840[34m>[0m
      args [34;1m= [0m[34m<local> [0mNamespace[34m([0masr[34m=[0m[34mFalse[0m[34m,[0m batch_size[34m=[0m4000[34m,[0m beam_size[34m=[0m12[34m,[0m config_file[34m=[0m[36m'/u/bahar/workspace/asr/librispeech/test-20190121/config-train/base2.smlp2.specaug.datarndperm_noscale.bs18k.curric3.retrain1.config'[0m[34m,[0m data[34m=[0m[36m'config:get_dataset("dev-other")'[0m[34m,[0m device[34m=[0m[34mNone[0m[34m,[0m do_search[34m=[0m[34mFalse[0m[34m,[0m dropout[34m=[0m[34mNone[0m[34m,[0m dump_dir[34m=[0m[36m'/u/m...[0m
      layers [34;1m= [0m[34m<local> [0m[34m[[0m[36m'att_weights'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 11[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/tools/[0m[36;1mget-attention-weights.py[0m[36m"[0m, [34mline [0m[35m139[0m, [34min [0minit_net
[34m    line: [0mrnn[34m.[0mengine[34m.[0minit_network_from_config[34m([0mconfig[34m=[0mconfig[34m,[0m net_dict_post_proc[34m=[0mnet_dict_post_proc[34m)[0m
[34m    locals:[0m
      rnn [34;1m= [0m[34m<global> [0m[34m<[0mmodule [36m'rnn'[0m [34mfrom [0m[36m'/u/makarov/returnn-parnia/rnn.py'[0m[34m>[0m
      rnn[34;1m.[0mengine [34;1m= [0m[34m<global> [0m[34m<[0mTFEngine[34m.[0mEngine object at 0x2b62ca854ac8[34m>[0m
      rnn[34;1m.[0mengine[34;1m.[0minit_network_from_config [34;1m= [0m[34m<global> [0m[34m<[0mbound method Engine[34m.[0minit_network_from_config of [34m<[0mTFEngine[34m.[0mEngine object at 0x2b62ca854ac8[34m>[0m[34m>[0m
      config [34;1m= [0m[34m<global> [0m[34m<[0mConfig[34m.[0mConfig object at 0x2b636988e668[34m>[0m
      net_dict_post_proc [34;1m= [0m[34m<local> [0m[34m<[0mfunction init_net[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mnet_dict_post_proc at 0x2b6440c1ff28[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/[0m[36;1mTFEngine.py[0m[36m"[0m, [34mline [0m[35m936[0m, [34min [0minit_network_from_config
[34m    line: [0mself[34m.[0m_init_network[34m([0mnet_desc[34m=[0mnet_dict[34m,[0m epoch[34m=[0mself[34m.[0mepoch[34m)[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFEngine[34m.[0mEngine object at 0x2b62ca854ac8[34m>[0m
      self[34;1m.[0m_init_network [34;1m= [0m[34m<local> [0m[34m<[0mbound method Engine[34m.[0m_init_network of [34m<[0mTFEngine[34m.[0mEngine object at 0x2b62ca854ac8[34m>[0m[34m>[0m
      net_desc [34;1m= [0m[34m<not found>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'encoder'[0m[34m:[0m [34m{[0m[36m'from'[0m[34m:[0m [34m[[0m[36m'lstm5_fw'[0m[34m,[0m [36m'lstm5_bw'[0m[34m][0m[34m,[0m [36m'class'[0m[34m:[0m [36m'copy'[0m[34m}[0m[34m,[0m [36m'source'[0m[34m:[0m [34m{[0m[36m'eval'[0m[34m:[0m [36m"self.network.get_config().typed_value('transform')(source(0), network=self.network)"[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'eval'[0m[34m}[0m[34m,[0m [36m'lstm5_fw'[0m[34m:[0m [34m{[0m[36m'direction'[0m[34m:[0m 1[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm4_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'..., len = 25[0m
      epoch [34;1m= [0m[34m<local> [0m250
      self[34;1m.[0mepoch [34;1m= [0m[34m<local> [0m250
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/[0m[36;1mTFEngine.py[0m[36m"[0m, [34mline [0m[35m1059[0m, [34min [0m_init_network
[34m    line: [0mself[34m.[0mnetwork[34m,[0m self[34m.[0mupdater [34m=[0m self[34m.[0mcreate_network[34m([0m
            config[34m=[0mself[34m.[0mconfig[34m,[0m
            rnd_seed[34m=[0mnet_random_seed[34m,[0m
            train_flag[34m=[0mtrain_flag[34m,[0m eval_flag[34m=[0mself[34m.[0muse_eval_flag[34m,[0m search_flag[34m=[0mself[34m.[0muse_search_flag[34m,[0m
            initial_learning_rate[34m=[0mgetattr[34m([0mself[34m,[0m [36m"initial_learning_rate"[0m[34m,[0m [34mNone[0m[34m)[0m[34m,[0m
            net_dict[34m=[0mnet_desc[34m)[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFEngine[34m.[0mEngine object at 0x2b62ca854ac8[34m>[0m
      self[34;1m.[0mnetwork [34;1m= [0m[34m<local> [0m[34mNone[0m
      self[34;1m.[0mupdater [34;1m= [0m[34m<local> [0m[34mNone[0m
      self[34;1m.[0mcreate_network [34;1m= [0m[34m<local> [0m[34m<[0mbound method Engine[34m.[0mcreate_network of [34m<[0m[34mclass [0m[36m'TFEngine.Engine'[0m[34m>[0m[34m>[0m
      config [34;1m= [0m[34m<not found>[0m
      self[34;1m.[0mconfig [34;1m= [0m[34m<local> [0m[34m<[0mConfig[34m.[0mConfig object at 0x2b636988e668[34m>[0m
      rnd_seed [34;1m= [0m[34m<not found>[0m
      net_random_seed [34;1m= [0m[34m<local> [0m250
      train_flag [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m
      eval_flag [34;1m= [0m[34m<not found>[0m
      self[34;1m.[0muse_eval_flag [34;1m= [0m[34m<local> [0m[34mTrue[0m
      search_flag [34;1m= [0m[34m<not found>[0m
      self[34;1m.[0muse_search_flag [34;1m= [0m[34m<local> [0m[34mFalse[0m
      initial_learning_rate [34;1m= [0m[34m<not found>[0m
      getattr [34;1m= [0m[34m<builtin> [0m[34m<[0mbuilt[34m-[0m[34min [0mfunction getattr[34m>[0m
      net_dict [34;1m= [0m[34m<not found>[0m
      net_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'encoder'[0m[34m:[0m [34m{[0m[36m'from'[0m[34m:[0m [34m[[0m[36m'lstm5_fw'[0m[34m,[0m [36m'lstm5_bw'[0m[34m][0m[34m,[0m [36m'class'[0m[34m:[0m [36m'copy'[0m[34m}[0m[34m,[0m [36m'source'[0m[34m:[0m [34m{[0m[36m'eval'[0m[34m:[0m [36m"self.network.get_config().typed_value('transform')(source(0), network=self.network)"[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'eval'[0m[34m}[0m[34m,[0m [36m'lstm5_fw'[0m[34m:[0m [34m{[0m[36m'direction'[0m[34m:[0m 1[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm4_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'..., len = 25[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/[0m[36;1mTFEngine.py[0m[36m"[0m, [34mline [0m[35m1090[0m, [34min [0mcreate_network
[34m    line: [0mnetwork[34m.[0mconstruct_from_dict[34m([0mnet_dict[34m)[0m
[34m    locals:[0m
      network [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      network[34;1m.[0mconstruct_from_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_from_dict of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'encoder'[0m[34m:[0m [34m{[0m[36m'from'[0m[34m:[0m [34m[[0m[36m'lstm5_fw'[0m[34m,[0m [36m'lstm5_bw'[0m[34m][0m[34m,[0m [36m'class'[0m[34m:[0m [36m'copy'[0m[34m}[0m[34m,[0m [36m'source'[0m[34m:[0m [34m{[0m[36m'eval'[0m[34m:[0m [36m"self.network.get_config().typed_value('transform')(source(0), network=self.network)"[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'eval'[0m[34m}[0m[34m,[0m [36m'lstm5_fw'[0m[34m:[0m [34m{[0m[36m'direction'[0m[34m:[0m 1[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm4_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'..., len = 25[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m338[0m, [34min [0mconstruct_from_dict
[34m    line: [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m,[0m name[34m)[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'encoder'[0m[34m:[0m [34m{[0m[36m'from'[0m[34m:[0m [34m[[0m[36m'lstm5_fw'[0m[34m,[0m [36m'lstm5_bw'[0m[34m][0m[34m,[0m [36m'class'[0m[34m:[0m [36m'copy'[0m[34m}[0m[34m,[0m [36m'source'[0m[34m:[0m [34m{[0m[36m'eval'[0m[34m:[0m [36m"self.network.get_config().typed_value('transform')(source(0), network=self.network)"[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'eval'[0m[34m}[0m[34m,[0m [36m'lstm5_fw'[0m[34m:[0m [34m{[0m[36m'direction'[0m[34m:[0m 1[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm4_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'..., len = 25[0m
      name [34;1m= [0m[34m<local> [0m[36m'decision'[0m[34m,[0m len [34m=[0m 8
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m412[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.DecideLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method LayerBase[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.DecideLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'loss_opts'[0m[34m:[0m [34m{[0m[34m}[0m[34m,[0m [36m'loss'[0m[34m:[0m [36m'edit_distance'[0m[34m,[0m [36m'target'[0m[34m:[0m [36m'classes'[0m[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b6440c6b598[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m360[0m, [34min [0mtransform_config_dict
[34m    line: [0m[34mfor [0msrc_name [34min [0msrc_names
[34m    locals:[0m
      src_name [34;1m= [0m[34m<not found>[0m
      src_names [34;1m= [0m[34m<local> [0m[34m[[0m[36m'output'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 6[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m361[0m, [34min [0m<listcomp>
[34m    line: [0md[34m[[0m[36m"sources"[0m[34m][0m [34m=[0m [34m[[0m
            get_layer[34m([0msrc_name[34m)[0m
            [34mfor [0msrc_name [34min [0msrc_names
            [34mif [0m[34mnot [0msrc_name [34m=[0m[34m=[0m [36m"none"[0m[34m][0m
[34m    locals:[0m
      d [34;1m= [0m[34m<not found>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b6440c6b598[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'output'[0m[34m,[0m len [34m=[0m 6
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m402[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'encoder'[0m[34m:[0m [34m{[0m[36m'from'[0m[34m:[0m [34m[[0m[36m'lstm5_fw'[0m[34m,[0m [36m'lstm5_bw'[0m[34m][0m[34m,[0m [36m'class'[0m[34m:[0m [36m'copy'[0m[34m}[0m[34m,[0m [36m'source'[0m[34m:[0m [34m{[0m[36m'eval'[0m[34m:[0m [36m"self.network.get_config().typed_value('transform')(source(0), network=self.network)"[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'eval'[0m[34m}[0m[34m,[0m [36m'lstm5_fw'[0m[34m:[0m [34m{[0m[36m'direction'[0m[34m:[0m 1[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm4_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m,[0m [36m'..., len = 25[0m
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'output'[0m[34m,[0m len [34m=[0m 6
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0m[34mreturn [0madd_layer[34m([0mname[34m=[0mname[34m,[0m layer_class[34m=[0mlayer_class[34m,[0m [34m*[0m[34m*[0mlayer_desc[34m)[0m
[34m    locals:[0m
      add_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0madd_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      name [34;1m= [0m[34m<local> [0m[36m'output'[0m[34m,[0m len [34m=[0m 6
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'loss'[0m[34m:[0m [34mNone[0m[34m,[0m [36m'sources'[0m[34m:[0m [34m[[0m[34m][0m[34m,[0m [36m'max_seq_len'[0m[34m:[0m [34m<[0mtf[34m.[0mTensor [36m'max_seq_len_encoder:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mint32[34m>[0m[34m,[0m [36m'target'[0m[34m:[0m [36m'classes'[0m[34m,[0m [36m'n_out'[0m[34m:[0m [34mNone[0m[34m,[0m [36m'unit'[0m[34m:[0m [34m{[0m[36m'readout_in'[0m[34m:[0m [34m{[0m[36m'from'[0m[34m:[0m [34m[[0m[36m's'[0m[34m,[0m [36m'prev:target_embed'[0m[34m,[0m [36m'att'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1000[34m,[0m [36m'class'[0m[34m:[0m [36m'linear'[0m[34m,[0m [36m'activation'[0m[34m:[0m [34mNone[0m[34m}[0m[34m,[0m [36m'energy'[0m[34m:[0m [34m{[0m[36m'mode'[0m[34m:[0m [36m'sum'[0m[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'..., len = 7[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m500[0m, [34min [0madd_layer
[34m    line: [0mlayer [34m=[0m self[34m.[0m_create_layer[34m([0mname[34m=[0mname[34m,[0m layer_class[34m=[0mlayer_class[34m,[0m [34m*[0m[34m*[0mlayer_desc[34m)[0m
[34m    locals:[0m
      layer [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0m_create_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0m_create_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      name [34;1m= [0m[34m<local> [0m[36m'output'[0m[34m,[0m len [34m=[0m 6
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'loss'[0m[34m:[0m [34mNone[0m[34m,[0m [36m'sources'[0m[34m:[0m [34m[[0m[34m][0m[34m,[0m [36m'max_seq_len'[0m[34m:[0m [34m<[0mtf[34m.[0mTensor [36m'max_seq_len_encoder:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mint32[34m>[0m[34m,[0m [36m'target'[0m[34m:[0m [36m'classes'[0m[34m,[0m [36m'n_out'[0m[34m:[0m [34mNone[0m[34m,[0m [36m'unit'[0m[34m:[0m [34m{[0m[36m'readout_in'[0m[34m:[0m [34m{[0m[36m'from'[0m[34m:[0m [34m[[0m[36m's'[0m[34m,[0m [36m'prev:target_embed'[0m[34m,[0m [36m'att'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1000[34m,[0m [36m'class'[0m[34m:[0m [36m'linear'[0m[34m,[0m [36m'activation'[0m[34m:[0m [34mNone[0m[34m}[0m[34m,[0m [36m'energy'[0m[34m:[0m [34m{[0m[36m'mode'[0m[34m:[0m [36m'sum'[0m[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'..., len = 7[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m460[0m, [34min [0m_create_layer
[34m    line: [0mlayer [34m=[0m layer_class[34m([0m[34m*[0m[34m*[0mlayer_desc[34m)[0m
[34m    locals:[0m
      layer [34;1m= [0m[34m<not found>[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'sources'[0m[34m:[0m [34m[[0m[34m][0m[34m,[0m [36m'name'[0m[34m:[0m [36m'output'[0m[34m,[0m [36m'network'[0m[34m:[0m [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m,[0m [36m'target'[0m[34m:[0m [36m'classes'[0m[34m,[0m [36m'loss'[0m[34m:[0m [34mNone[0m[34m,[0m [36m'n_out'[0m[34m:[0m [34mNone[0m[34m,[0m [36m'max_seq_len'[0m[34m:[0m [34m<[0mtf[34m.[0mTensor [36m'max_seq_len_encoder:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mint32[34m>[0m[34m,[0m [36m'unit'[0m[34m:[0m [34m{[0m[36m'readout_in'[0m[34m:[0m [34m{[0m[36m'from'[0m[34m:[0m [34m[[0m[36m's'[0m[34m,[0m [36m'prev:targ..., len = 10[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/[0m[36;1mTFNetworkRecLayer.py[0m[36m"[0m, [34mline [0m[35m178[0m, [34min [0m__init__
[34m    line: [0my [34m=[0m self[34m.[0m_get_output_subnet_unit[34m([0mself[34m.[0mcell[34m)[0m
[34m    locals:[0m
      y [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mRecLayer [36m'output'[0m out_type[34m=[0mData[34m([0mshape[34m=[0m[34m([0m[34mNone[0m[34m,[0m[34m)[0m[34m,[0m dtype[34m=[0m[36m'int32'[0m[34m,[0m sparse[34m=[0m[34mTrue[0m[34m,[0m dim[34m=[0m10025[34m,[0m batch_dim_axis[34m=[0m1[34m)[0m[34m>[0m
      self[34;1m.[0m_get_output_subnet_unit [34;1m= [0m[34m<local> [0m[34m<[0mbound method RecLayer[34m.[0m_get_output_subnet_unit of [34m<[0mRecLayer [36m'output'[0m out_type[34m=[0mData[34m([0mshape[34m=[0m[34m([0m[34mNone[0m[34m,[0m[34m)[0m[34m,[0m dtype[34m=[0m[36m'int32'[0m[34m,[0m sparse[34m=[0m[34mTrue[0m[34m,[0m dim[34m=[0m10025[34m,[0m batch_dim_axis[34m=[0m1[34m)[0m[34m>[0m[34m>[0m
      self[34;1m.[0mcell [34;1m= [0m[34m<local> [0m[34m<[0mTFNetworkRecLayer[34m.[0m_SubnetworkRecCell object at 0x2b6458097f98[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/[0m[36;1mTFNetworkRecLayer.py[0m[36m"[0m, [34mline [0m[35m709[0m, [34min [0m_get_output_subnet_unit
[34m    line: [0moutput[34m,[0m search_choices [34m=[0m cell[34m.[0mget_output[34m([0mrec_layer[34m=[0mself[34m)[0m
[34m    locals:[0m
      output [34;1m= [0m[34m<not found>[0m
      search_choices [34;1m= [0m[34m<not found>[0m
      cell [34;1m= [0m[34m<local> [0m[34m<[0mTFNetworkRecLayer[34m.[0m_SubnetworkRecCell object at 0x2b6458097f98[34m>[0m
      cell[34;1m.[0mget_output [34;1m= [0m[34m<local> [0m[34m<[0mbound method _SubnetworkRecCell[34m.[0mget_output of [34m<[0mTFNetworkRecLayer[34m.[0m_SubnetworkRecCell object at 0x2b6458097f98[34m>[0m[34m>[0m
      rec_layer [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mRecLayer [36m'output'[0m out_type[34m=[0mData[34m([0mshape[34m=[0m[34m([0m[34mNone[0m[34m,[0m[34m)[0m[34m,[0m dtype[34m=[0m[36m'int32'[0m[34m,[0m sparse[34m=[0m[34mTrue[0m[34m,[0m dim[34m=[0m10025[34m,[0m batch_dim_axis[34m=[0m1[34m)[0m[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia/[0m[36;1mTFNetworkRecLayer.py[0m[36m"[0m, [34mline [0m[35m1522[0m, [34min [0mget_output
[34m    line: [0m[34mif [0m[34mnot [0mrec_layer[34m.[0moutput[34m.[0mis_same_time_dim[34m([0mlayer[34m.[0moutput[34m)[0m[34m:[0m
[34m    locals:[0m
      rec_layer [34;1m= [0m[34m<local> [0m[34m<[0mRecLayer [36m'output'[0m out_type[34m=[0mData[34m([0mshape[34m=[0m[34m([0m[34mNone[0m[34m,[0m[34m)[0m[34m,[0m dtype[34m=[0m[36m'int32'[0m[34m,[0m sparse[34m=[0m[34mTrue[0m[34m,[0m dim[34m=[0m10025[34m,[0m batch_dim_axis[34m=[0m1[34m)[0m[34m>[0m
      rec_layer[34;1m.[0moutput [34;1m= [0m[34m<local> [0mData[34m([0mname[34m=[0m[36m'output_output'[0m[34m,[0m shape[34m=[0m[34m([0m[34mNone[0m[34m,[0m[34m)[0m[34m,[0m dtype[34m=[0m[36m'int32'[0m[34m,[0m sparse[34m=[0m[34mTrue[0m[34m,[0m dim[34m=[0m10025[34m,[0m batch_dim_axis[34m=[0m1[34m)[0m
      rec_layer[34;1m.[0moutput[34;1m.[0mis_same_time_dim [34;1m= [0m[34m<local> [0m!AttributeError: 'Data' object has no attribute 'is_same_time_dim'
      layer [34;1m= [0m[34m<local> [0m[34m<[0mChoiceLayer [36m'output'[0m out_type[34m=[0mData[34m([0mshape[34m=[0m[34m([0m[34mNone[0m[34m,[0m[34m)[0m[34m,[0m dtype[34m=[0m[36m'int32'[0m[34m,[0m sparse[34m=[0m[34mTrue[0m[34m,[0m dim[34m=[0m10025[34m)[0m[34m>[0m
      layer[34;1m.[0moutput [34;1m= [0m[34m<local> [0mData[34m([0mname[34m=[0m[36m'classes'[0m[34m,[0m shape[34m=[0m[34m([0m[34mNone[0m[34m,[0m[34m)[0m[34m,[0m dtype[34m=[0m[36m'int32'[0m[34m,[0m sparse[34m=[0m[34mTrue[0m[34m,[0m dim[34m=[0m10025[34m)[0m
[31mAttributeError[0m: 'Data' object has no attribute 'is_same_time_dim'
fatal: Not a git repository (or any parent up to mount point /u/makarov)
Stopping at filesystem boundary (GIT_DISCOVERY_ACROSS_FILESYSTEM not set).
2019-07-05 12:10:34.974069: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-07-05 12:10:35.999072: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:02:00.0
totalMemory: 10.92GiB freeMemory: 10.76GiB
2019-07-05 12:10:35.999127: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-05 12:10:35.999142: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-05 12:10:35.999159: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-05 12:10:35.999167: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-05 12:10:36.001922: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-05 12:10:36.386068: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-05 12:10:36.386120: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-05 12:10:36.386128: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-05 12:10:36.386467: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/device:GPU:0 with 10405 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
ERROR: cannot read default config file. using build-in defaults [main]
LOG: connected to ('10.6.100.1', 10321)
LOG: destination: /var/tmp/makarov/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-other.zip
LOG: request: /work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-other.zip
LOG: using existing copy /var/tmp/irie/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-other.zip
2019-07-05 12:10:37.452091: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-05 12:10:37.452147: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-05 12:10:37.452156: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-05 12:10:37.452172: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-05 12:10:37.452378: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10405 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
[31;1mEXCEPTION[0m
[34mTraceback (most recent call last):[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/tools/[0m[36;1mget-attention-weights.py[0m[36m"[0m, [34mline [0m[35m414[0m, [34min [0m<module>
[34m    line: [0mmain[34m([0msys[34m.[0margv[34m)[0m
[34m    locals:[0m
      main [34;1m= [0m[34m<local> [0m[34m<[0mfunction main at 0x2b8735143840[34m>[0m
      sys [34;1m= [0m[34m<local> [0m[34m<[0mmodule [36m'sys'[0m [34m([0mbuilt[34m-[0m[34min[0m[34m)[0m[34m>[0m
      sys[34;1m.[0margv [34;1m= [0m[34m<local> [0m[34m[[0m[36m'/u/makarov/returnn-parnia-2/tools/get-attention-weights.py'[0m[34m,[0m [36m'/u/bahar/workspace/asr/librispeech/test-20190121/config-train/base2.smlp2.specaug.datarndperm_noscale.bs18k.curric3.retrain1.config'[0m[34m,[0m [36m'--epoch'[0m[34m,[0m [36m'250'[0m[34m,[0m [36m'--data'[0m[34m,[0m [36m'config:get_dataset("dev-other")'[0m[34m,[0m [36m'--dump_dir'[0m[34m,[0m [36m'/u/makarov/makarov/NM..., len = 18, _[0]: {len = 58}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/tools/[0m[36;1mget-attention-weights.py[0m[36m"[0m, [34mline [0m[35m231[0m, [34min [0mmain
[34m    line: [0minit_net[34m([0margs[34m,[0m layers[34m)[0m
[34m    locals:[0m
      init_net [34;1m= [0m[34m<global> [0m[34m<[0mfunction init_net at 0x2b87351437b8[34m>[0m
      args [34;1m= [0m[34m<local> [0mNamespace[34m([0masr[34m=[0m[34mFalse[0m[34m,[0m batch_size[34m=[0m4000[34m,[0m beam_size[34m=[0m12[34m,[0m config_file[34m=[0m[36m'/u/bahar/workspace/asr/librispeech/test-20190121/config-train/base2.smlp2.specaug.datarndperm_noscale.bs18k.curric3.retrain1.config'[0m[34m,[0m data[34m=[0m[36m'config:get_dataset("dev-other")'[0m[34m,[0m device[34m=[0m[34mNone[0m[34m,[0m do_search[34m=[0m[34mFalse[0m[34m,[0m dropout[34m=[0m[34mNone[0m[34m,[0m dump_dir[34m=[0m[36m'/u/m...[0m
      layers [34;1m= [0m[34m<local> [0m[34m[[0m[36m'att_weights'[0m[34m][0m[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 11[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/tools/[0m[36;1mget-attention-weights.py[0m[36m"[0m, [34mline [0m[35m139[0m, [34min [0minit_net
[34m    line: [0mrnn[34m.[0mengine[34m.[0minit_network_from_config[34m([0mconfig[34m=[0mconfig[34m,[0m net_dict_post_proc[34m=[0mnet_dict_post_proc[34m)[0m
[34m    locals:[0m
      rnn [34;1m= [0m[34m<global> [0m[34m<[0mmodule [36m'rnn'[0m [34mfrom [0m[36m'/u/makarov/returnn-parnia-2/rnn.py'[0m[34m>[0m
      rnn[34;1m.[0mengine [34;1m= [0m[34m<global> [0m[34m<[0mTFEngine[34m.[0mEngine object at 0x2b86855f7ac8[34m>[0m
      rnn[34;1m.[0mengine[34;1m.[0minit_network_from_config [34;1m= [0m[34m<global> [0m[34m<[0mbound method Engine[34m.[0minit_network_from_config of [34m<[0mTFEngine[34m.[0mEngine object at 0x2b86855f7ac8[34m>[0m[34m>[0m
      config [34;1m= [0m[34m<global> [0m[34m<[0mConfig[34m.[0mConfig object at 0x2b8735145198[34m>[0m
      net_dict_post_proc [34;1m= [0m[34m<local> [0m[34m<[0mfunction init_net[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mnet_dict_post_proc at 0x2b8734582d90[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFEngine.py[0m[36m"[0m, [34mline [0m[35m936[0m, [34min [0minit_network_from_config
[34m    line: [0mself[34m.[0m_init_network[34m([0mnet_desc[34m=[0mnet_dict[34m,[0m epoch[34m=[0mself[34m.[0mepoch[34m)[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFEngine[34m.[0mEngine object at 0x2b86855f7ac8[34m>[0m
      self[34;1m.[0m_init_network [34;1m= [0m[34m<local> [0m[34m<[0mbound method Engine[34m.[0m_init_network of [34m<[0mTFEngine[34m.[0mEngine object at 0x2b86855f7ac8[34m>[0m[34m>[0m
      net_desc [34;1m= [0m[34m<not found>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'encoder'[0m[34m:[0m [34m{[0m[36m'from'[0m[34m:[0m [34m[[0m[36m'lstm5_fw'[0m[34m,[0m [36m'lstm5_bw'[0m[34m][0m[34m,[0m [36m'class'[0m[34m:[0m [36m'copy'[0m[34m}[0m[34m,[0m [36m'lstm1_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm0_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m}[0m[34m,[0m [36m'lstm4_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm3_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      epoch [34;1m= [0m[34m<local> [0m250
      self[34;1m.[0mepoch [34;1m= [0m[34m<local> [0m250
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFEngine.py[0m[36m"[0m, [34mline [0m[35m1060[0m, [34min [0m_init_network
[34m    line: [0mself[34m.[0mnetwork[34m,[0m self[34m.[0mupdater [34m=[0m self[34m.[0mcreate_network[34m([0m
            config[34m=[0mself[34m.[0mconfig[34m,[0m
            rnd_seed[34m=[0mnet_random_seed[34m,[0m
            train_flag[34m=[0mtrain_flag[34m,[0m eval_flag[34m=[0mself[34m.[0muse_eval_flag[34m,[0m search_flag[34m=[0mself[34m.[0muse_search_flag[34m,[0m
            initial_learning_rate[34m=[0mgetattr[34m([0mself[34m,[0m [36m"initial_learning_rate"[0m[34m,[0m [34mNone[0m[34m)[0m[34m,[0m
            net_dict[34m=[0mnet_desc[34m)[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFEngine[34m.[0mEngine object at 0x2b86855f7ac8[34m>[0m
      self[34;1m.[0mnetwork [34;1m= [0m[34m<local> [0m[34mNone[0m
      self[34;1m.[0mupdater [34;1m= [0m[34m<local> [0m[34mNone[0m
      self[34;1m.[0mcreate_network [34;1m= [0m[34m<local> [0m[34m<[0mbound method Engine[34m.[0mcreate_network of [34m<[0m[34mclass [0m[36m'TFEngine.Engine'[0m[34m>[0m[34m>[0m
      config [34;1m= [0m[34m<not found>[0m
      self[34;1m.[0mconfig [34;1m= [0m[34m<local> [0m[34m<[0mConfig[34m.[0mConfig object at 0x2b8735145198[34m>[0m
      rnd_seed [34;1m= [0m[34m<not found>[0m
      net_random_seed [34;1m= [0m[34m<local> [0m250
      train_flag [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m
      eval_flag [34;1m= [0m[34m<not found>[0m
      self[34;1m.[0muse_eval_flag [34;1m= [0m[34m<local> [0m[34mTrue[0m
      search_flag [34;1m= [0m[34m<not found>[0m
      self[34;1m.[0muse_search_flag [34;1m= [0m[34m<local> [0m[34mFalse[0m
      initial_learning_rate [34;1m= [0m[34m<not found>[0m
      getattr [34;1m= [0m[34m<builtin> [0m[34m<[0mbuilt[34m-[0m[34min [0mfunction getattr[34m>[0m
      net_dict [34;1m= [0m[34m<not found>[0m
      net_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'encoder'[0m[34m:[0m [34m{[0m[36m'from'[0m[34m:[0m [34m[[0m[36m'lstm5_fw'[0m[34m,[0m [36m'lstm5_bw'[0m[34m][0m[34m,[0m [36m'class'[0m[34m:[0m [36m'copy'[0m[34m}[0m[34m,[0m [36m'lstm1_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm0_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m}[0m[34m,[0m [36m'lstm4_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm3_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFEngine.py[0m[36m"[0m, [34mline [0m[35m1091[0m, [34min [0mcreate_network
[34m    line: [0mnetwork[34m.[0mconstruct_from_dict[34m([0mnet_dict[34m)[0m
[34m    locals:[0m
      network [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      network[34;1m.[0mconstruct_from_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_from_dict of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'encoder'[0m[34m:[0m [34m{[0m[36m'from'[0m[34m:[0m [34m[[0m[36m'lstm5_fw'[0m[34m,[0m [36m'lstm5_bw'[0m[34m][0m[34m,[0m [36m'class'[0m[34m:[0m [36m'copy'[0m[34m}[0m[34m,[0m [36m'lstm1_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm0_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m}[0m[34m,[0m [36m'lstm4_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm3_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m338[0m, [34min [0mconstruct_from_dict
[34m    line: [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m,[0m name[34m)[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'encoder'[0m[34m:[0m [34m{[0m[36m'from'[0m[34m:[0m [34m[[0m[36m'lstm5_fw'[0m[34m,[0m [36m'lstm5_bw'[0m[34m][0m[34m,[0m [36m'class'[0m[34m:[0m [36m'copy'[0m[34m}[0m[34m,[0m [36m'lstm1_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm0_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m}[0m[34m,[0m [36m'lstm4_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm3_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<local> [0m[36m'ctc'[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.SoftmaxLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method LayerBase[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkLayer.SoftmaxLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'loss'[0m[34m:[0m [36m'ctc'[0m[34m,[0m [36m'target'[0m[34m:[0m [36m'classes'[0m[34m,[0m [36m'loss_opts'[0m[34m:[0m [34m{[0m[36m'ctc_opts'[0m[34m:[0m [34m{[0m[36m'ignore_longer_outputs_than_inputs'[0m[34m:[0m [34mTrue[0m[34m}[0m[34m,[0m [36m'beam_width'[0m[34m:[0m 1[34m}[0m[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b8734685400[34m>[0m
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
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b8734685400[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'encoder'[0m[34m,[0m len [34m=[0m 7
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'encoder'[0m[34m:[0m [34m{[0m[36m'from'[0m[34m:[0m [34m[[0m[36m'lstm5_fw'[0m[34m,[0m [36m'lstm5_bw'[0m[34m][0m[34m,[0m [36m'class'[0m[34m:[0m [36m'copy'[0m[34m}[0m[34m,[0m [36m'lstm1_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm0_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m}[0m[34m,[0m [36m'lstm4_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm3_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
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
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b8734777e18[34m>[0m
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
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b8734777e18[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm5_fw'[0m[34m,[0m len [34m=[0m 8
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'encoder'[0m[34m:[0m [34m{[0m[36m'from'[0m[34m:[0m [34m[[0m[36m'lstm5_fw'[0m[34m,[0m [36m'lstm5_bw'[0m[34m][0m[34m,[0m [36m'class'[0m[34m:[0m [36m'copy'[0m[34m}[0m[34m,[0m [36m'lstm1_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm0_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m}[0m[34m,[0m [36m'lstm4_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm3_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
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
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b873477e158[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkRecLayer.py[0m[36m"[0m, [34mline [0m[35m230[0m, [34min [0mtransform_config_dict
[34m    line: [0msuper[34m([0mRecLayer[34m,[0m cls[34m)[0m[34m.[0mtransform_config_dict[34m([0md[34m,[0m network[34m=[0mnetwork[34m,[0m get_layer[34m=[0mget_layer[34m)[0m  [37m# everything except "unit"[0m
[34m    locals:[0m
      super [34;1m= [0m[34m<builtin> [0m[34m<[0m[34mclass [0m[36m'super'[0m[34m>[0m
      RecLayer [34;1m= [0m[34m<global> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      cls [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      transform_config_dict [34;1m= [0m[34m<not found>[0m
      d [34;1m= [0m[34m<local> [0m[34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m 1[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m}[0m
      network [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b873477e158[34m>[0m
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
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b873477e158[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm4_pool'[0m[34m,[0m len [34m=[0m 10
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'encoder'[0m[34m:[0m [34m{[0m[36m'from'[0m[34m:[0m [34m[[0m[36m'lstm5_fw'[0m[34m,[0m [36m'lstm5_bw'[0m[34m][0m[34m,[0m [36m'class'[0m[34m:[0m [36m'copy'[0m[34m}[0m[34m,[0m [36m'lstm1_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm0_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m}[0m[34m,[0m [36m'lstm4_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm3_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm4_pool'[0m[34m,[0m len [34m=[0m 10
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method LayerBase[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'mode'[0m[34m:[0m [36m'max'[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'trainable'[0m[34m:[0m [34mFalse[0m[34m,[0m [36m'pool_size'[0m[34m:[0m [34m([0m1[34m,[0m[34m)[0m[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b873477e268[34m>[0m
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
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b873477e268[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm4_fw'[0m[34m,[0m len [34m=[0m 8
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'encoder'[0m[34m:[0m [34m{[0m[36m'from'[0m[34m:[0m [34m[[0m[36m'lstm5_fw'[0m[34m,[0m [36m'lstm5_bw'[0m[34m][0m[34m,[0m [36m'class'[0m[34m:[0m [36m'copy'[0m[34m}[0m[34m,[0m [36m'lstm1_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm0_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m}[0m[34m,[0m [36m'lstm4_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm3_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
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
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b873477e378[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkRecLayer.py[0m[36m"[0m, [34mline [0m[35m230[0m, [34min [0mtransform_config_dict
[34m    line: [0msuper[34m([0mRecLayer[34m,[0m cls[34m)[0m[34m.[0mtransform_config_dict[34m([0md[34m,[0m network[34m=[0mnetwork[34m,[0m get_layer[34m=[0mget_layer[34m)[0m  [37m# everything except "unit"[0m
[34m    locals:[0m
      super [34;1m= [0m[34m<builtin> [0m[34m<[0m[34mclass [0m[36m'super'[0m[34m>[0m
      RecLayer [34;1m= [0m[34m<global> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      cls [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      transform_config_dict [34;1m= [0m[34m<not found>[0m
      d [34;1m= [0m[34m<local> [0m[34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m 1[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m}[0m
      network [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b873477e378[34m>[0m
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
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b873477e378[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm3_pool'[0m[34m,[0m len [34m=[0m 10
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'encoder'[0m[34m:[0m [34m{[0m[36m'from'[0m[34m:[0m [34m[[0m[36m'lstm5_fw'[0m[34m,[0m [36m'lstm5_bw'[0m[34m][0m[34m,[0m [36m'class'[0m[34m:[0m [36m'copy'[0m[34m}[0m[34m,[0m [36m'lstm1_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm0_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m}[0m[34m,[0m [36m'lstm4_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm3_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm3_pool'[0m[34m,[0m len [34m=[0m 10
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method LayerBase[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'mode'[0m[34m:[0m [36m'max'[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'trainable'[0m[34m:[0m [34mFalse[0m[34m,[0m [36m'pool_size'[0m[34m:[0m [34m([0m1[34m,[0m[34m)[0m[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b873477e488[34m>[0m
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
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b873477e488[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm3_fw'[0m[34m,[0m len [34m=[0m 8
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'encoder'[0m[34m:[0m [34m{[0m[36m'from'[0m[34m:[0m [34m[[0m[36m'lstm5_fw'[0m[34m,[0m [36m'lstm5_bw'[0m[34m][0m[34m,[0m [36m'class'[0m[34m:[0m [36m'copy'[0m[34m}[0m[34m,[0m [36m'lstm1_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm0_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m}[0m[34m,[0m [36m'lstm4_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm3_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
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
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b873477e598[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkRecLayer.py[0m[36m"[0m, [34mline [0m[35m230[0m, [34min [0mtransform_config_dict
[34m    line: [0msuper[34m([0mRecLayer[34m,[0m cls[34m)[0m[34m.[0mtransform_config_dict[34m([0md[34m,[0m network[34m=[0mnetwork[34m,[0m get_layer[34m=[0mget_layer[34m)[0m  [37m# everything except "unit"[0m
[34m    locals:[0m
      super [34;1m= [0m[34m<builtin> [0m[34m<[0m[34mclass [0m[36m'super'[0m[34m>[0m
      RecLayer [34;1m= [0m[34m<global> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      cls [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      transform_config_dict [34;1m= [0m[34m<not found>[0m
      d [34;1m= [0m[34m<local> [0m[34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m 1[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m}[0m
      network [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b873477e598[34m>[0m
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
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b873477e598[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm2_pool'[0m[34m,[0m len [34m=[0m 10
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'encoder'[0m[34m:[0m [34m{[0m[36m'from'[0m[34m:[0m [34m[[0m[36m'lstm5_fw'[0m[34m,[0m [36m'lstm5_bw'[0m[34m][0m[34m,[0m [36m'class'[0m[34m:[0m [36m'copy'[0m[34m}[0m[34m,[0m [36m'lstm1_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm0_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m}[0m[34m,[0m [36m'lstm4_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm3_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm2_pool'[0m[34m,[0m len [34m=[0m 10
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method LayerBase[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'mode'[0m[34m:[0m [36m'max'[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'trainable'[0m[34m:[0m [34mFalse[0m[34m,[0m [36m'pool_size'[0m[34m:[0m [34m([0m1[34m,[0m[34m)[0m[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b873477e6a8[34m>[0m
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
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b873477e6a8[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm2_fw'[0m[34m,[0m len [34m=[0m 8
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'encoder'[0m[34m:[0m [34m{[0m[36m'from'[0m[34m:[0m [34m[[0m[36m'lstm5_fw'[0m[34m,[0m [36m'lstm5_bw'[0m[34m][0m[34m,[0m [36m'class'[0m[34m:[0m [36m'copy'[0m[34m}[0m[34m,[0m [36m'lstm1_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm0_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m}[0m[34m,[0m [36m'lstm4_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm3_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
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
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b873477e7b8[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkRecLayer.py[0m[36m"[0m, [34mline [0m[35m230[0m, [34min [0mtransform_config_dict
[34m    line: [0msuper[34m([0mRecLayer[34m,[0m cls[34m)[0m[34m.[0mtransform_config_dict[34m([0md[34m,[0m network[34m=[0mnetwork[34m,[0m get_layer[34m=[0mget_layer[34m)[0m  [37m# everything except "unit"[0m
[34m    locals:[0m
      super [34;1m= [0m[34m<builtin> [0m[34m<[0m[34mclass [0m[36m'super'[0m[34m>[0m
      RecLayer [34;1m= [0m[34m<global> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      cls [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      transform_config_dict [34;1m= [0m[34m<not found>[0m
      d [34;1m= [0m[34m<local> [0m[34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m 1[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m}[0m
      network [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b873477e7b8[34m>[0m
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
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b873477e7b8[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm1_pool'[0m[34m,[0m len [34m=[0m 10
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'encoder'[0m[34m:[0m [34m{[0m[36m'from'[0m[34m:[0m [34m[[0m[36m'lstm5_fw'[0m[34m,[0m [36m'lstm5_bw'[0m[34m][0m[34m,[0m [36m'class'[0m[34m:[0m [36m'copy'[0m[34m}[0m[34m,[0m [36m'lstm1_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm0_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m}[0m[34m,[0m [36m'lstm4_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm3_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm1_pool'[0m[34m,[0m len [34m=[0m 10
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method LayerBase[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'mode'[0m[34m:[0m [36m'max'[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'trainable'[0m[34m:[0m [34mFalse[0m[34m,[0m [36m'pool_size'[0m[34m:[0m [34m([0m2[34m,[0m[34m)[0m[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b873477e8c8[34m>[0m
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
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b873477e8c8[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm1_fw'[0m[34m,[0m len [34m=[0m 8
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'encoder'[0m[34m:[0m [34m{[0m[36m'from'[0m[34m:[0m [34m[[0m[36m'lstm5_fw'[0m[34m,[0m [36m'lstm5_bw'[0m[34m][0m[34m,[0m [36m'class'[0m[34m:[0m [36m'copy'[0m[34m}[0m[34m,[0m [36m'lstm1_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm0_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m}[0m[34m,[0m [36m'lstm4_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm3_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
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
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b873477e9d8[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkRecLayer.py[0m[36m"[0m, [34mline [0m[35m230[0m, [34min [0mtransform_config_dict
[34m    line: [0msuper[34m([0mRecLayer[34m,[0m cls[34m)[0m[34m.[0mtransform_config_dict[34m([0md[34m,[0m network[34m=[0mnetwork[34m,[0m get_layer[34m=[0mget_layer[34m)[0m  [37m# everything except "unit"[0m
[34m    locals:[0m
      super [34;1m= [0m[34m<builtin> [0m[34m<[0m[34mclass [0m[36m'super'[0m[34m>[0m
      RecLayer [34;1m= [0m[34m<global> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      cls [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      transform_config_dict [34;1m= [0m[34m<not found>[0m
      d [34;1m= [0m[34m<local> [0m[34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m 1[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m}[0m
      network [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b873477e9d8[34m>[0m
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
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b873477e9d8[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm0_pool'[0m[34m,[0m len [34m=[0m 10
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'encoder'[0m[34m:[0m [34m{[0m[36m'from'[0m[34m:[0m [34m[[0m[36m'lstm5_fw'[0m[34m,[0m [36m'lstm5_bw'[0m[34m][0m[34m,[0m [36m'class'[0m[34m:[0m [36m'copy'[0m[34m}[0m[34m,[0m [36m'lstm1_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm0_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m}[0m[34m,[0m [36m'lstm4_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm3_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm0_pool'[0m[34m,[0m len [34m=[0m 10
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method LayerBase[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkLayer.PoolLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'mode'[0m[34m:[0m [36m'max'[0m[34m,[0m [36m'padding'[0m[34m:[0m [36m'same'[0m[34m,[0m [36m'trainable'[0m[34m:[0m [34mFalse[0m[34m,[0m [36m'pool_size'[0m[34m:[0m [34m([0m3[34m,[0m[34m)[0m[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b873477eae8[34m>[0m
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
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b873477eae8[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm0_fw'[0m[34m,[0m len [34m=[0m 8
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'encoder'[0m[34m:[0m [34m{[0m[36m'from'[0m[34m:[0m [34m[[0m[36m'lstm5_fw'[0m[34m,[0m [36m'lstm5_bw'[0m[34m][0m[34m,[0m [36m'class'[0m[34m:[0m [36m'copy'[0m[34m}[0m[34m,[0m [36m'lstm1_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm0_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m}[0m[34m,[0m [36m'lstm4_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm3_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
      name [34;1m= [0m[34m<not found>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'lstm0_fw'[0m[34m,[0m len [34m=[0m 8
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m415[0m, [34min [0mconstruct_layer
[34m    line: [0mlayer_class[34m.[0mtransform_config_dict[34m([0mlayer_desc[34m,[0m network[34m=[0mself[34m,[0m get_layer[34m=[0mget_layer[34m)[0m
[34m    locals:[0m
      layer_class [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      layer_class[34;1m.[0mtransform_config_dict [34;1m= [0m[34m<local> [0m[34m<[0mbound method RecLayer[34m.[0mtransform_config_dict of [34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m[34m>[0m
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m
      network [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b873477ebf8[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkRecLayer.py[0m[36m"[0m, [34mline [0m[35m230[0m, [34min [0mtransform_config_dict
[34m    line: [0msuper[34m([0mRecLayer[34m,[0m cls[34m)[0m[34m.[0mtransform_config_dict[34m([0md[34m,[0m network[34m=[0mnetwork[34m,[0m get_layer[34m=[0mget_layer[34m)[0m  [37m# everything except "unit"[0m
[34m    locals:[0m
      super [34;1m= [0m[34m<builtin> [0m[34m<[0m[34mclass [0m[36m'super'[0m[34m>[0m
      RecLayer [34;1m= [0m[34m<global> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      cls [34;1m= [0m[34m<local> [0m[34m<[0m[34mclass [0m[36m'TFNetworkRecLayer.RecLayer'[0m[34m>[0m
      transform_config_dict [34;1m= [0m[34m<not found>[0m
      d [34;1m= [0m[34m<local> [0m[34m{[0m[36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m 1[34m}[0m
      network [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b873477ebf8[34m>[0m
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
      get_layer [34;1m= [0m[34m<local> [0m[34m<[0mfunction TFNetwork[34m.[0mconstruct_layer[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_layer at 0x2b873477ebf8[34m>[0m
      src_name [34;1m= [0m[34m<local> [0m[36m'source'[0m[34m,[0m len [34m=[0m 6
      src_names [34;1m= [0m[34m<not found>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m388[0m, [34min [0mget_layer
[34m    line: [0m[34mreturn [0mself[34m.[0mconstruct_layer[34m([0mnet_dict[34m=[0mnet_dict[34m,[0m name[34m=[0msrc_name[34m)[0m  [37m# set get_layer to wrap construct_layer[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mconstruct_layer [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mconstruct_layer of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      net_dict [34;1m= [0m[34m<local> [0m[34m{[0m[36m'encoder'[0m[34m:[0m [34m{[0m[36m'from'[0m[34m:[0m [34m[[0m[36m'lstm5_fw'[0m[34m,[0m [36m'lstm5_bw'[0m[34m][0m[34m,[0m [36m'class'[0m[34m:[0m [36m'copy'[0m[34m}[0m[34m,[0m [36m'lstm1_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm0_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m [36m'class'[0m[34m:[0m [36m'rec'[0m[34m}[0m[34m,[0m [36m'lstm4_bw'[0m[34m:[0m [34m{[0m[36m'unit'[0m[34m:[0m [36m'nativelstm2'[0m[34m,[0m [36m'direction'[0m[34m:[0m [34m-[0m1[34m,[0m [36m'dropout'[0m[34m:[0m 0[34m.[0m3[34m,[0m [36m'from'[0m[34m:[0m [34m[[0m[36m'lstm3_pool'[0m[34m][0m[34m,[0m [36m'n_out'[0m[34m:[0m 1024[34m,[0m[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 25
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
      layer_desc [34;1m= [0m[34m<local> [0m[34m{[0m[36m'name'[0m[34m:[0m [36m'source'[0m[34m,[0m [36m'output'[0m[34m:[0m Data[34m([0mname[34m=[0m[36m'source_output'[0m[34m,[0m shape[34m=[0m[34m([0m[34mNone[0m[34m,[0m 40[34m)[0m[34m)[0m[34m,[0m [36m'eval'[0m[34m:[0m [36m"self.network.get_config().typed_value('transform')(source(0), network=self.network)"[0m[34m,[0m [36m'sources'[0m[34m:[0m [34m[[0m[34m<[0mSourceLayer [36m'data'[0m out_type[34m=[0mData[34m([0mshape[34m=[0m[34m([0m[34mNone[0m[34m,[0m 40[34m)[0m[34m)[0m[34m>[0m[34m][0m[34m,[0m [36m'network'[0m[34m:[0m [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train...[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m4426[0m, [34min [0m__init__
[34m    line: [0msuper[34m([0mEvalLayer[34m,[0m self[34m)[0m[34m.[0m__init__[34m([0mkind[34m=[0m[36m"eval"[0m[34m,[0m eval[34m=[0meval[34m,[0m [34m*[0m[34m*[0mkwargs[34m)[0m
[34m    locals:[0m
      super [34;1m= [0m[34m<builtin> [0m[34m<[0m[34mclass [0m[36m'super'[0m[34m>[0m
      EvalLayer [34;1m= [0m[34m<global> [0m[34m<[0m[34mclass [0m[36m'TFNetworkLayer.EvalLayer'[0m[34m>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mEvalLayer [36m'source'[0m out_type[34m=[0mData[34m([0mshape[34m=[0m[34m([0m[34mNone[0m[34m,[0m 40[34m)[0m[34m)[0m[34m>[0m
      __init__ [34;1m= [0m[34m<not found>[0m
      kind [34;1m= [0m[34m<not found>[0m
      eval [34;1m= [0m[34m<local> [0m[36m"self.network.get_config().typed_value('transform')(source(0), network=self.network)"[0m[34m,[0m len [34m=[0m 83
      kwargs [34;1m= [0m[34m<local> [0m[34m{[0m[36m'name'[0m[34m:[0m [36m'source'[0m[34m,[0m [36m'sources'[0m[34m:[0m [34m[[0m[34m<[0mSourceLayer [36m'data'[0m out_type[34m=[0mData[34m([0mshape[34m=[0m[34m([0m[34mNone[0m[34m,[0m 40[34m)[0m[34m)[0m[34m>[0m[34m][0m[34m,[0m [36m'output'[0m[34m:[0m Data[34m([0mname[34m=[0m[36m'source_output'[0m[34m,[0m shape[34m=[0m[34m([0m[34mNone[0m[34m,[0m 40[34m)[0m[34m)[0m[34m,[0m [36m'network'[0m[34m:[0m [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m}[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetworkLayer.py[0m[36m"[0m, [34mline [0m[35m4285[0m, [34min [0m__init__
[34m    line: [0mx [34m=[0m op[34m([0msources[34m)[0m
[34m    locals:[0m
      x [34;1m= [0m[34m<not found>[0m
      op [34;1m= [0m[34m<local> [0m[34m<[0mfunction CombineLayer[34m.[0m_get_op[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mwrap_eval_op at 0x2b873477ed90[34m>[0m
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
      vs [34;1m= [0m[34m<local> [0m[34m{[0m[36m'get_sparse_tensor_length'[0m[34m:[0m [34m<[0mfunction get_sparse_tensor_length at 0x2b873484a730[34m>[0m[34m,[0m [36m'bpe_idx_to_bpe_string'[0m[34m:[0m [34m<[0mfunction bpe_idx_to_bpe_string at 0x2b873484a488[34m>[0m[34m,[0m [36m'opt_reuse_name_scope'[0m[34m:[0m [34m<[0mfunction opt_reuse_name_scope at 0x2b87350ff1e0[34m>[0m[34m,[0m [36m'slice_pad_zeros'[0m[34m:[0m [34m<[0mfunction slice_pad_zeros at 0x2b8734830b7[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 217
  [34;1mFile [0m[36m"[0m[36;1m<string>[0m[36m"[0m, [34mline [0m[35m1[0m, [34min [0m<module>
[34m    -- code not available --[0m
  [34;1mFile [0m[36m"/u/bahar/workspace/asr/librispeech/test-20190121/config-train/[0m[36;1mbase2.smlp2.specaug.datarndperm_noscale.bs18k.curric3.retrain1.config[0m[36m"[0m, [34mline [0m[35m183[0m, [34min [0mtransform
[34m    line: [0mx [34m=[0m network[34m.[0mcond_on_train[34m([0mget_masked[34m,[0m [34mlambda[0m[34m:[0m x[34m)[0m
[34m    locals:[0m
      x [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'source/clip_by_value:0'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m,[0m 40[34m)[0m dtype[34m=[0mfloat32[34m>[0m
      network [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      network[34;1m.[0mcond_on_train [34;1m= [0m[34m<local> [0m[34m<[0mbound method TFNetwork[34m.[0mcond_on_train of [34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m[34m>[0m
      get_masked [34;1m= [0m[34m<local> [0m[34m<[0mfunction transform[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_masked at 0x2b873477eea0[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFNetwork.py[0m[36m"[0m, [34mline [0m[35m1027[0m, [34min [0mcond_on_train
[34m    line: [0m[34mreturn [0mcond[34m([0mself[34m.[0mtrain_flag[34m,[0m fn_train[34m,[0m fn_eval[34m)[0m
[34m    locals:[0m
      cond [34;1m= [0m[34m<local> [0m[34m<[0mfunction cond at 0x2b8734830048[34m>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFNetwork [36m'root'[0m train[34m=[0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m>[0m
      self[34;1m.[0mtrain_flag [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m
      fn_train [34;1m= [0m[34m<local> [0m[34m<[0mfunction transform[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_masked at 0x2b873477eea0[34m>[0m
      fn_eval [34;1m= [0m[34m<local> [0m[34m<[0mfunction transform[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0m[34m<[0m[34mlambda[0m[34m>[0m at 0x2b86cc08f620[34m>[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-parnia-2/[0m[36;1mTFUtil.py[0m[36m"[0m, [34mline [0m[35m3768[0m, [34min [0mcond
[34m    line: [0m[34mreturn [0mcontrol_flow_ops[34m.[0mcond[34m([0mpred[34m,[0m fn1[34m,[0m fn2[34m,[0m name[34m=[0mname[34m)[0m
[34m    locals:[0m
      control_flow_ops [34;1m= [0m[34m<local> [0m[34m<[0mmodule [36m'tensorflow.python.ops.control_flow_ops'[0m [34mfrom [0m[36m'/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/python/ops/control_flow_ops.py'[0m[34m>[0m
      control_flow_ops[34;1m.[0mcond [34;1m= [0m[34m<local> [0m[34m<[0mfunction cond at 0x2b873368aa60[34m>[0m
      pred [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m
      fn1 [34;1m= [0m[34m<local> [0m[34m<[0mfunction transform[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_masked at 0x2b873477eea0[34m>[0m
      fn2 [34;1m= [0m[34m<local> [0m[34m<[0mfunction transform[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0m[34m<[0m[34mlambda[0m[34m>[0m at 0x2b86cc08f620[34m>[0m
      name [34;1m= [0m[34m<local> [0m[34mNone[0m
  [34;1mFile [0m[36m"/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/python/util/[0m[36;1mdeprecation.py[0m[36m"[0m, [34mline [0m[35m454[0m, [34min [0mnew_func
[34m    line: [0m[34mreturn [0mfunc[34m([0m[34m*[0margs[34m,[0m [34m*[0m[34m*[0mkwargs[34m)[0m
[34m    locals:[0m
      func [34;1m= [0m[34m<local> [0m[34m<[0mfunction cond at 0x2b873368a950[34m>[0m
      args [34;1m= [0m[34m<local> [0m[34m([0m[34m<[0mtf[34m.[0mTensor [36m'globals/train_flag:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mbool[34m>[0m[34m,[0m [34m<[0mfunction transform[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_masked at 0x2b873477eea0[34m>[0m[34m,[0m [34m<[0mfunction transform[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0m[34m<[0m[34mlambda[0m[34m>[0m at 0x2b86cc08f620[34m>[0m[34m)[0m
      kwargs [34;1m= [0m[34m<local> [0m[34m{[0m[36m'name'[0m[34m:[0m [34mNone[0m[34m}[0m
  [34;1mFile [0m[36m"/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/python/ops/[0m[36;1mcontrol_flow_ops.py[0m[36m"[0m, [34mline [0m[35m2048[0m, [34min [0mcond
[34m    line: [0morig_res_t[34m,[0m res_t [34m=[0m context_t[34m.[0mBuildCondBranch[34m([0mtrue_fn[34m)[0m
[34m    locals:[0m
      orig_res_t [34;1m= [0m[34m<not found>[0m
      res_t [34;1m= [0m[34m<not found>[0m
      context_t [34;1m= [0m[34m<local> [0m[34m<[0mtensorflow[34m.[0mpython[34m.[0mops[34m.[0mcontrol_flow_ops[34m.[0mCondContext object at 0x2b873477ff98[34m>[0m
      context_t[34;1m.[0mBuildCondBranch [34;1m= [0m[34m<local> [0m[34m<[0mbound method CondContext[34m.[0mBuildCondBranch of [34m<[0mtensorflow[34m.[0mpython[34m.[0mops[34m.[0mcontrol_flow_ops[34m.[0mCondContext object at 0x2b873477ff98[34m>[0m[34m>[0m
      true_fn [34;1m= [0m[34m<local> [0m[34m<[0mfunction transform[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_masked at 0x2b873477eea0[34m>[0m
  [34;1mFile [0m[36m"/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/python/ops/[0m[36;1mcontrol_flow_ops.py[0m[36m"[0m, [34mline [0m[35m1895[0m, [34min [0mBuildCondBranch
[34m    line: [0moriginal_result [34m=[0m fn[34m([0m[34m)[0m
[34m    locals:[0m
      original_result [34;1m= [0m[34m<not found>[0m
      fn [34;1m= [0m[34m<local> [0m[34m<[0mfunction transform[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mget_masked at 0x2b873477eea0[34m>[0m
  [34;1mFile [0m[36m"/u/bahar/workspace/asr/librispeech/test-20190121/config-train/[0m[36;1mbase2.smlp2.specaug.datarndperm_noscale.bs18k.curric3.retrain1.config[0m[36m"[0m, [34mline [0m[35m179[0m, [34min [0mget_masked
[34m    line: [0mx_masked [34m=[0m random_mask[34m([0mx_masked[34m,[0m axis[34m=[0m1[34m,[0m min_num[34m=[0m1[34m,[0m max_num[34m=[0mtf[34m.[0mmaximum[34m([0mtf[34m.[0mshape[34m([0mx[34m)[0m[34m[[0m1[34m][0m [34m/[0m[34m/[0m 100[34m,[0m 1[34m)[0m[34m,[0m max_dims[34m=[0m20[34m)[0m
[34m    locals:[0m
      x_masked [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'source/clip_by_value:0'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m,[0m 40[34m)[0m dtype[34m=[0mfloat32[34m>[0m
      random_mask [34;1m= [0m[34m<global> [0m[34m<[0mfunction random_mask at 0x2b87345ea598[34m>[0m
      axis [34;1m= [0m[34m<not found>[0m
      min_num [34;1m= [0m[34m<not found>[0m
      max_num [34;1m= [0m[34m<not found>[0m
      tf [34;1m= [0m[34m<local> [0m[34m<[0mmodule [36m'tensorflow'[0m [34mfrom [0m[36m'/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/__init__.py'[0m[34m>[0m
      tf[34;1m.[0mmaximum [34;1m= [0m[34m<local> [0m[34m<[0mfunction maximum at 0x2b86dcd9a268[34m>[0m
      tf[34;1m.[0mshape [34;1m= [0m[34m<local> [0m[34m<[0mfunction shape at 0x2b86dce52158[34m>[0m
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
      tf[34;1m.[0mwhile_loop [34;1m= [0m[34m<local> [0m[34m<[0mfunction while_loop at 0x2b873368a7b8[34m>[0m
      cond [34;1m= [0m[34m<not found>[0m
      i [34;1m= [0m[34m<not found>[0m
      tf[34;1m.[0mless [34;1m= [0m[34m<local> [0m[34m<[0mfunction less at 0x2b86dcd98620[34m>[0m
      tf[34;1m.[0mreduce_max [34;1m= [0m[34m<local> [0m[34m<[0mfunction reduce_max at 0x2b87327252f0[34m>[0m
      num [34;1m= [0m[34m<local> [0m[34m<[0mtf[34m.[0mTensor [36m'source/cond/random_uniform:0'[0m shape[34m=[0m[34m([0m?[34m,[0m[34m)[0m dtype[34m=[0mint32[34m>[0m
      body [34;1m= [0m[34m<not found>[0m
      tf[34;1m.[0mwhere [34;1m= [0m[34m<local> [0m[34m<[0mfunction where at 0x2b86dd07a268[34m>[0m
      _mask [34;1m= [0m[34m<global> [0m[34m<[0mfunction _mask at 0x2b87345ea510[34m>[0m
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
      loop_context [34;1m= [0m[34m<local> [0m[34m<[0mtensorflow[34m.[0mpython[34m.[0mops[34m.[0mcontrol_flow_ops[34m.[0mWhileContext object at 0x2b873494e4e0[34m>[0m
      loop_context[34;1m.[0mBuildLoop [34;1m= [0m[34m<local> [0m[34m<[0mbound method WhileContext[34m.[0mBuildLoop of [34m<[0mtensorflow[34m.[0mpython[34m.[0mops[34m.[0mcontrol_flow_ops[34m.[0mWhileContext object at 0x2b873494e4e0[34m>[0m[34m>[0m
      cond [34;1m= [0m[34m<local> [0m[34m<[0mfunction random_mask[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0m[34m<[0m[34mlambda[0m[34m>[0m at 0x2b86cc0a1b70[34m>[0m
      body [34;1m= [0m[34m<local> [0m[34m<[0mfunction random_mask[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0m[34m<[0m[34mlambda[0m[34m>[0m at 0x2b8734a2c1e0[34m>[0m
      loop_vars [34;1m= [0m[34m<local> [0m[34m([0m0[34m,[0m [34m<[0mtf[34m.[0mTensor [36m'source/clip_by_value:0'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m,[0m 40[34m)[0m dtype[34m=[0mfloat32[34m>[0m[34m)[0m
      shape_invariants [34;1m= [0m[34m<local> [0m[34mNone[0m
      return_same_structure [34;1m= [0m[34m<local> [0m[34mFalse[0m
  [34;1mFile [0m[36m"/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/python/ops/[0m[36;1mcontrol_flow_ops.py[0m[36m"[0m, [34mline [0m[35m2952[0m, [34min [0mBuildLoop
[34m    line: [0moriginal_body_result[34m,[0m exit_vars [34m=[0m self[34m.[0m_BuildLoop[34m([0m
              pred[34m,[0m body[34m,[0m original_loop_vars[34m,[0m loop_vars[34m,[0m shape_invariants[34m)[0m
[34m    locals:[0m
      original_body_result [34;1m= [0m[34m<not found>[0m
      exit_vars [34;1m= [0m[34m<not found>[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mtensorflow[34m.[0mpython[34m.[0mops[34m.[0mcontrol_flow_ops[34m.[0mWhileContext object at 0x2b873494e4e0[34m>[0m
      self[34;1m.[0m_BuildLoop [34;1m= [0m[34m<local> [0m[34m<[0mbound method WhileContext[34m.[0m_BuildLoop of [34m<[0mtensorflow[34m.[0mpython[34m.[0mops[34m.[0mcontrol_flow_ops[34m.[0mWhileContext object at 0x2b873494e4e0[34m>[0m[34m>[0m
      pred [34;1m= [0m[34m<local> [0m[34m<[0mfunction random_mask[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0m[34m<[0m[34mlambda[0m[34m>[0m at 0x2b86cc0a1b70[34m>[0m
      body [34;1m= [0m[34m<local> [0m[34m<[0mfunction random_mask[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0m[34m<[0m[34mlambda[0m[34m>[0m at 0x2b8734a2c1e0[34m>[0m
      original_loop_vars [34;1m= [0m[34m<local> [0m[34m([0m0[34m,[0m [34m<[0mtf[34m.[0mTensor [36m'source/clip_by_value:0'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m,[0m 40[34m)[0m dtype[34m=[0mfloat32[34m>[0m[34m)[0m
      loop_vars [34;1m= [0m[34m<local> [0m[34m[[0m[34m<[0mtf[34m.[0mTensor [36m'source/cond/while/Const:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mint32[34m>[0m[34m,[0m [34m<[0mtf[34m.[0mTensor [36m'source/clip_by_value:0'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m,[0m 40[34m)[0m dtype[34m=[0mfloat32[34m>[0m[34m][0m
      shape_invariants [34;1m= [0m[34m<local> [0m[34mNone[0m
  [34;1mFile [0m[36m"/u/bahar/settings/python3-returnn-tf1.9/lib/python3.5/site-packages/tensorflow/python/ops/[0m[36;1mcontrol_flow_ops.py[0m[36m"[0m, [34mline [0m[35m2887[0m, [34min [0m_BuildLoop
[34m    line: [0mbody_result [34m=[0m body[34m([0m[34m*[0mpacked_vars_for_body[34m)[0m
[34m    locals:[0m
      body_result [34;1m= [0m[34m<not found>[0m
      body [34;1m= [0m[34m<local> [0m[34m<[0mfunction random_mask[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0m[34m<[0m[34mlambda[0m[34m>[0m at 0x2b8734a2c1e0[34m>[0m
      packed_vars_for_body [34;1m= [0m[34m<local> [0m[34m([0m[34m<[0mtf[34m.[0mTensor [36m'source/cond/while/Identity:0'[0m shape[34m=[0m[34m([0m[34m)[0m dtype[34m=[0mint32[34m>[0m[34m,[0m [34m<[0mtf[34m.[0mTensor [36m'source/cond/while/Identity_1:0'[0m shape[34m=[0m[34m([0m?[34m,[0m ?[34m,[0m 40[34m)[0m dtype[34m=[0mfloat32[34m>[0m[34m)[0m
  [34;1mFile [0m[36m"/u/bahar/workspace/asr/librispeech/test-20190121/config-train/[0m[36;1mbase2.smlp2.specaug.datarndperm_noscale.bs18k.curric3.retrain1.config[0m[36m"[0m, [34mline [0m[35m166[0m, [34min [0m<lambda>
[34m    line: [0m_mask[34m([0mx[34m,[0m axis[34m=[0maxis[34m,[0m pos[34m=[0mindices[34m[[0m[34m:[0m[34m,[0m i[34m][0m[34m,[0m max_amount[34m=[0mmax_dims[34m)[0m[34m,[0m
[34m    locals:[0m
      _mask [34;1m= [0m[34m<global> [0m[34m<[0mfunction _mask at 0x2b87345ea510[34m>[0m
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
usage: get-attention-weights.py [-h] [--epoch EPOCH] [--data DATA]
                                [--do_search] [--beam_size BEAM_SIZE]
                                [--dump_dir DUMP_DIR]
                                [--output_file OUTPUT_FILE] [--device DEVICE]
                                [--layers LAYERS] [--rec_layer REC_LAYER]
                                [--enc_layer ENC_LAYER]
                                [--batch_size BATCH_SIZE]
                                [--seq_list SEQ_LIST]
                                [--min_seq_len MIN_SEQ_LEN]
                                [--num_seqs NUM_SEQS]
                                [--output_format OUTPUT_FORMAT]
                                [--dropout DROPOUT] [--train_flag]
                                [--reset_partition_epoch RESET_PARTITION_EPOCH]
                                [--reset_seq_ordering RESET_SEQ_ORDERING]
                                [--reset_epoch_wise_filter RESET_EPOCH_WISE_FILTER]
                                config_file
get-attention-weights.py: error: unrecognized arguments: --tf_log_dir /work/smt2/makarov/NMT/hmm-factorization/experiments/asr2019/lstm/forward-dev-other/tf_log_dir
2019-07-05 12:14:15.066278: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-07-05 12:14:16.300019: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:03:00.0
totalMemory: 10.92GiB freeMemory: 10.76GiB
2019-07-05 12:14:16.300082: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-05 12:14:16.300103: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-05 12:14:16.300113: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-05 12:14:16.300124: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-05 12:14:16.303288: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-05 12:14:16.722034: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-05 12:14:16.722098: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-05 12:14:16.722107: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-05 12:14:16.722519: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/device:GPU:0 with 10405 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1)
ERROR: cannot read default config file. using build-in defaults [main]
LOG: connected to ('10.6.100.1', 10321)
LOG: destination: /var/tmp/makarov/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-other.zip
LOG: request: /work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-other.zip
LOG: using existing copy /var/tmp/irie/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-other.zip
2019-07-05 12:14:18.076363: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-05 12:14:18.076438: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-05 12:14:18.076458: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-05 12:14:18.076468: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-05 12:14:18.078095: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10405 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1)
[31;1mEXCEPTION[0m
[34mTraceback (most recent call last):[0m
  [34;1mFile [0m[36m"/u/makarov/returnn-vanilla/[0m[36;1mTFEngine.py[0m[36m"[0m, [34mline [0m[35m579[0m, [34min [0mrun
[34m    line: [0mself[34m.[0m_maybe_handle_extra_fetches[34m([0mfetches_results[34m)[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFEngine[34m.[0mRunner object at 0x2b4ce50420f0[34m>[0m
      self[34;1m.[0m_maybe_handle_extra_fetches [34;1m= [0m[34m<local> [0m[34m<[0mbound method Runner[34m.[0m_maybe_handle_extra_fetches of [34m<[0mTFEngine[34m.[0mRunner object at 0x2b4ce50420f0[34m>[0m[34m>[0m
      fetches_results [34;1m= [0m[34m<local> [0m[34m{[0m[36m'error:ctc'[0m[34m:[0m 6[34m.[0m0[34m,[0m [36m'extra:output_len'[0m[34m:[0m array[34m([0m[34m[[0m94[34m][0m[34m,[0m dtype[34m=[0mint32[34m)[0m[34m,[0m [36m'error:output/output_prob'[0m[34m:[0m 4[34m.[0m0[34m,[0m [36m'cost:output/output_prob'[0m[34m:[0m 18[34m.[0m55054[34m,[0m [36m'error:decision'[0m[34m:[0m 0[34m.[0m0[34m,[0m [36m'loss_norm_factor:decision'[0m[34m:[0m 0[34m.[0m010638298[34m,[0m [36m'extra:seq_tag'[0m[34m:[0m array[34m([0m[34m[[0m[36m'dev-other-7601-291468-0000'[0m[34m][0m[34m,[0m dtype[34m=[0mobject[34m)[0m[34m,[0m [36m'loss_norm_factor:ctc'[0m[34m:[0m 0[34m.[0m0[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 23
  [34;1mFile [0m[36m"/u/makarov/returnn-vanilla/[0m[36;1mTFEngine.py[0m[36m"[0m, [34mline [0m[35m358[0m, [34min [0m_maybe_handle_extra_fetches
[34m    line: [0mself[34m.[0mextra_fetches_callback[34m([0m[34m*[0m[34m*[0md[34m)[0m
[34m    locals:[0m
      self [34;1m= [0m[34m<local> [0m[34m<[0mTFEngine[34m.[0mRunner object at 0x2b4ce50420f0[34m>[0m
      self[34;1m.[0mextra_fetches_callback [34;1m= [0m[34m<local> [0m[34m<[0mfunction main[34m.[0m[34m<[0mlocals[34m>[0m[34m.[0mfetch_callback at 0x2b496995d488[34m>[0m
      d [34;1m= [0m[34m<local> [0m[34m{[0m[36m'output'[0m[34m:[0m array[34m([0m[34m[[0m[34m[[0m  46[34m,[0m    2[34m,[0m 4836[34m,[0m 3458[34m,[0m 2064[34m,[0m  194[34m,[0m  825[34m,[0m   19[34m,[0m   70[34m,[0m    5[34m,[0m 1516[34m,[0m
                            74[34m,[0m   13[34m,[0m 3610[34m,[0m    4[34m,[0m    2[34m,[0m 6777[34m,[0m   10[34m,[0m  152[34m,[0m 3172[34m,[0m  267[34m,[0m 3732[34m,[0m
                          1281[34m,[0m   39[34m,[0m    4[34m,[0m   13[34m,[0m 2404[34m,[0m 1997[34m,[0m  839[34m,[0m 4925[34m,[0m  616[34m,[0m 9256[34m,[0m    2[34m,[0m
                          3358[34m,[0m   10[34m,[0m    2[34m,[0m 1263[34m,[0m   19[34m,[0m   65[34m,[0m  212[34m,[0m    5[34m,[0m   26[34m,[0m 48[34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 8
  [34;1mFile [0m[36m"/u/makarov/returnn-vanilla/tools/[0m[36;1mget-attention-weights.py[0m[36m"[0m, [34mline [0m[35m266[0m, [34min [0mfetch_callback
[34m    line: [0m[34massert [0mout[34m.[0mshape[34m[[0m0[34m][0m [34m>[0m[34m=[0m output_len[34m[[0mi[34m][0m [34mand [0mout[34m.[0mshape[34m[[0m1[34m][0m [34m>[0m[34m=[0m encoder_len[34m[[0mi[34m][0m
[34m    locals:[0m
      out [34;1m= [0m[34m<local> [0marray[34m([0m[34m[[0m[34m[[0m[34m[[0m1[34m.[0m2674613e[34m-[0m02[34m,[0m 3[34m.[0m4933843e[34m-[0m02[34m,[0m 1[34m.[0m9389097e[34m-[0m02[34m,[0m [34m.[0m[34m.[0m[34m.[0m[34m,[0m
                             3[34m.[0m1357166e[34m-[0m17[34m,[0m 1[34m.[0m1040296e[34m-[0m15[34m,[0m 3[34m.[0m5440155e[34m-[0m16[34m][0m[34m][0m[34m,[0m
                    
                           [34m[[0m[34m[[0m7[34m.[0m8406818e[34m-[0m03[34m,[0m 1[34m.[0m9062437e[34m-[0m01[34m,[0m 1[34m.[0m6740145e[34m-[0m02[34m,[0m [34m.[0m[34m.[0m[34m.[0m[34m,[0m
                             7[34m.[0m9451022e[34m-[0m17[34m,[0m 1[34m.[0m3712369e[34m-[0m15[34m,[0m 1[34m.[0m9998926e[34m-[0m15[34m][0m[34m][0m[34m,[0m
                    
                           [34m[[0m[34m[[0m4[34m.[0m6953177e[34m-[0m03[34m,[0m 7[34m.[0m1483470e[34m-[0m02[34m,[0m 2[34m.[0m0558701e[34m-[0m03[34m,[0m [34m.[0m[34m.[0m[34m.[0m[34m,[0m
                          [34m.[0m[34m.[0m[34m.[0m[34m,[0m len [34m=[0m 94[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 1[34m,[0m _[34m[[0m0[34m][0m[34m:[0m [34m{[0mlen [34m=[0m 535[34m}[0m[34m}[0m
      out[34;1m.[0mshape [34;1m= [0m[34m<local> [0m[34m([0m94[34m,[0m 1[34m,[0m 535[34m)[0m
      output_len [34;1m= [0m[34m<local> [0marray[34m([0m[34m[[0m94[34m][0m[34m,[0m dtype[34m=[0mint32[34m)[0m[34m,[0m len [34m=[0m 1
      i [34;1m= [0m[34m<local> [0m0
      encoder_len [34;1m= [0m[34m<local> [0marray[34m([0m[34m[[0m535[34m][0m[34m,[0m dtype[34m=[0mint32[34m)[0m[34m,[0m len [34m=[0m 1
[31mAssertionError[0m
2019-07-05 12:17:30.871471: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-07-05 12:17:31.954381: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:82:00.0
totalMemory: 10.92GiB freeMemory: 10.76GiB
2019-07-05 12:17:31.954601: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-05 12:17:31.954625: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-05 12:17:31.954632: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-05 12:17:31.954638: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-05 12:17:31.957203: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-05 12:17:32.340273: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-05 12:17:32.340317: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-05 12:17:32.340324: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-05 12:17:32.340613: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/device:GPU:0 with 10405 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:82:00.0, compute capability: 6.1)
ERROR: cannot read default config file. using build-in defaults [main]
LOG: connected to ('10.6.100.1', 10321)
LOG: destination: /var/tmp/makarov/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-other.zip
LOG: request: /work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-other.zip
LOG: using existing copy /var/tmp/irie/work/asr3/zeyer/setups-data/librispeech/dataset/ogg-zips/dev-other.zip
2019-07-05 12:17:33.444964: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-07-05 12:17:33.445030: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-05 12:17:33.445044: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-07-05 12:17:33.445053: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-07-05 12:17:33.445271: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10405 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:82:00.0, compute capability: 6.1)
