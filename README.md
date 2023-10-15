# Attention_guided_weight_mixup_BLO
​
This folder contains the implementation of our proposed method in BERT.
​
### Create and activate conda env
```console
conda env create -f environment.yml
```
### Start the experiments
```console
python multi_lanuch_script.py (or)
bash run.sh
```

Here are the description of each code file:

bert_modeling.py contains the code for the BERT model backbone with alpha parameters

blo.py is the implementation of the weights mixup using alpha parameters

run_glue_mlo.py is the main code to run

multi_lanuch_script.py is the script to run multiple experiments using multiprocessing.
