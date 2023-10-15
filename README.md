# Attention_guided_weight_mixup_BLO
​
This folder contains the implementation of our proposed method using BERT-LARGE model.
​
### Create and activate conda env
```console
conda env create -f environment.yml
```
### Start the experiments
```console
python multi_lanuch_script.py
bash run.sh
```
(or)
```console
bash run.sh
```
### Code file description

bert_modeling.py - BERT modeling with alpha parameters in the resultant weight node estimation

blo.py - implementation of the weights mixup using alpha parameters

run_glue_mlo.py - the main code to run

multi_lanuch_script.py - the script to run multiple experiments using multiprocessing.
