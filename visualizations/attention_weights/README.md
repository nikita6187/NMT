
## 1st run the get_attention_weights.py tool.

Here's an example, note that it needs to be launched from the location of the models folder.

```
python3 /work/smt2/makarov/returnn-hmm/tools/get-attention-weights.py ../../transformer-newBaseline.config --epoch 221 --data config:dev --dump_dir ./log-dump/ --layers "dec_06_att_weights" --rec_layer "output" --batch_size 600
```

## 2nd run the visualization tool

Here's an example.
TODO: make better indexing.

```
python3 /work/smt2/makarov/NMT/visualizations/attention_weights/visualization_attention_returnn.py transformer-newBaseline_ep221_data_103_112.npy 4 --show_labels

```
