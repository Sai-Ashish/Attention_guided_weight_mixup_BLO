# Attention guided weight mixup using bi-level optimization

Generalizable and Stable Finetuning of Pretrained Language Models on Low-Resource Texts - NAACL'24
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
```
(or)
```console
bash run.sh
```
### 📁 Code File Descriptions

- 📄 **bert_modeling.py**: Contains BERT modeling enhancements with alpha parameters used for the resultant weight node estimation.

- 📄 **blo.py**: Implements the mixup of weights using alpha parameters.

- 📄 **run_glue_mlo.py**: This is the primary script to execute.

- 📄 **multi_launch_script.py**: Utilizes multiprocessing to execute multiple experiments concurrently.
