
## 1st run the get_attention_weights.py tool.

Here's an example, note that it needs to be launched from the location of the models folder.

```
python3 /work/smt2/makarov/returnn-hmm/tools/get-attention-weights.py ../../transformer-newBaseline.config --epoch 221 --data config:dev --dump_dir ./log-dump/ --layers "dec_06_att_weights" --rec_layer "output" --batch_size 600
```
