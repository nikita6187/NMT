# NMT Experiments at i6

## Structure
The structure contains the configs and some results on the work on the Attention Mixture Model (also called HMM Factorization), as well as the topic aware decomposition for domain adaptation. 
The repo is located at ```/work/smt2/makarov/NMT/```, with all results being collected their under the respective ```/logs/``` folders, seen under each location where configs are found.

The hmm factorization folder contains three subfolders: ``de-en``,`zh-en` and `experiments`. ``de-en``,`zh-en` contain results on the language pairs, with ``de-en/hard-baseline`` having the most performant configs. `experiments` contains various results from extra experiments, such as for ASR and alignments.

The topic factorization (https://github.com/nikita68/NMT/blob/master/topic-factorization/Topic_Aware_NMT_Summary.pdf) folder first contains the formal derivations and initial experiments, as well as more in-depth configs on ``de-en``.  Here the subfolders `no-optimization` show configs without the mapping optimization and `mapping-generation` contains the scripts to generate the mappings from the data.

`visualizations` contains the subfolders for `attention_weights` and `distributions`. `attention_weights` looks into the visualization of attention weights of NMT models, with a detailed README inside. `distributions` has a few basic experiments into how the posterior distribution looks like in the Transformer models.

The `experiments` subfolder contains multiple small random experiments.

## Configs
Many subfolders contain configs, which should be run using my RETURNN fork: https://github.com/nikita68/returnn. The log folder for each model should have a `log` subfolder. The script `launch_config.py` shows how to launch it on the cluster with correct logging settings. For example, to launch `hmm-factorization/de-en/hard-baseline/transformer-newBaseline.config`, use the following line from `hmm-factorization/de-en/hard-baseline/`: `python3 /work/smt2/makarov/NMT/launch_config.py transformer-newBaseline.config`.

## Scripts
The repo contains many helper scripts, to help have an overview of the experiments. The main scripts are located in the root. The most important one is `status.py`, which shows the status of all models. An example (AMM, de-en) can be run with `python3 /work/smt2/makarov/NMT/status.py /work/smt2/makarov/NMT/hmm-factorization/de-en/hard-baseline/logs`, from any location on the cluster. For AMM zh-en, it is `python3 /work/smt2/makarov/NMT/status.py /work/smt2/makarov/NMT/hmm-factorization/zh-en/logs/`, and for topic factorization is is `python3 /work/smt2/makarov/NMT/status.py /work/smt2/makarov/NMT/topic-factorization/de-en/logs/`.

## Attention Mixture Model in RETURNN
- Located in https://github.com/nikita68/returnn
- Please sync with the main branch using this guide https://help.github.com/en/articles/syncing-a-fork
- The layer is found in https://github.com/nikita68/returnn/blob/master/TFNetworkHMMFactorization.py
- All of the documentation is directly in the layer
- Usage demos found in the `returnn/demos` subfolder, the Transformer demo with many features being here: https://github.com/nikita68/returnn/blob/master/demos/demo-tf-hmm-factorization-transformer.config Note that there are many different ones, including dynamic K, random head, RNN and independent decoder.

## Topic Factorization in RETURNN
- Located in https://github.com/nikita68/returnn
- Exact formulation found here https://github.com/nikita68/NMT/blob/master/topic-factorization/Topic_Aware_NMT_Summary.pdf
- Please sync with the main branch using this guide https://help.github.com/en/articles/syncing-a-fork
- The layer is found in https://github.com/nikita68/returnn/blob/master/TFNetworkTopicFactorization.py
- All of the documentation is directly in the layer
- Usage demos found in the `returnn/demos` subfolder, the Transformer demo with many features being here: https://github.com/nikita68/returnn/blob/master/demos/demo-tf-transformer-topic-factorization.config Note, again, that there are many other demos, showing full softmax and pre-loading weights.

## Geometric Loss
As part of my sampled softmax implementations, I briefly looked into using the l2 (Euclidean) distance as the loss, between the target embeddings and the current decoder state, as this can add more fine grained sample methods. Due to limited time, I only looked at some basic experiments, which did not turn out to be fruitful. I also tried combining the loss with cross entropy. All of the layer logic can be found at the bottom of https://github.com/nikita68/returnn/blob/master/TFNetworkHMMFactorization.py, with some demos being located here: https://github.com/nikita68/returnn/blob/master/demos/demo-tf-transformer-geo-softmax-ce.config, https://github.com/nikita68/returnn/blob/master/demos/demo-tf-transformer-geo-softmax.config.




