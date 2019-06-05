
## 1st run the get_attention_weights.py tool.

Here's an example, note that it needs to be launched from the location of the models folder.

```
python3 /work/smt2/makarov/returnn-hmm/tools/get-attention-weights.py ../../transformer-newBaseline.config --epoch 221 --data config:dev --dump_dir ./log-dump/ --layers "dec_06_att_weights" --rec_layer "output" --batch_size 600
```

Extra in case of all layers and options:

```
python3 /work/smt2/makarov/returnn-hmm/tools/get-attention-weights.py /u/bahar/workspace/wmt/2019/en-zh--2019-04-16/config-train/transformer-newBaseline-hmm-nolinear-k5.config --epoch 160 --data config:newstest2017 --dump_dir /work/smt2/makarov/NMT/hmm-factorization/experiments/parnia-en-zh-transformer-latent-2017/ --layers "dec_02_att_weights" --layers "dec_01_att_weights" --layers "dec_03_att_weights" --layers "dec_04_att_weights" --layers "dec_05_att_weights" --layers "dec_06_att_weights" --rec_layer "output" --batch_size 600 --tf_log_dir /work/smt2/makarov/NMT/hmm-factorization/experiments/parnia-en-zh-transformer-latent-2017/tf_log_dir
```


## 2nd run the visualization tool

Here's an example.
TODO: make better indexing.

```
python3 /work/smt2/makarov/NMT/visualizations/attention_weights/visualization_attention_returnn.py transformer-newBaseline_ep221_data_103_112.npy 4 --show_labels

```
