import json
import re
from itertools import cycle
import random
from torch.multiprocessing import Pool
import torch.multiprocessing as multiprocessing
import os
from itertools import product
from run_glue_mlo import main
import numpy as np

# arguments
num_gpus = 1
TASK_NAME_LIST=["qnli", "mnli", "stsb"]
GPU=0
TRAIN_SPLIT=300
MODEL="bert-large-cased"
BSZ=16
EPOCH=3.0
MLO_EPOCHS=3
MLO_WARMUP=0
MODEL_LR=2e-5
UNROLL_STEPS_LIST=[1]
SEED_list= #GIVE SOME RANDOM SEED
LR_LIST=[2e-5]
ALPHA_LR_LIST=[2e-3]
ALPHA_WARMUP_RATIO_LIST=[0.1]
ALPHA_WEIGHT_DECAY_LIST=[0]
ALPHA_L1_FACTOR=0
USE_L1="false"
AVG="true"

### POOLER LAYER ALPHA DISABLED ###

dict_tasktometric = {
                        "cola": "eval_matthews_correlation",
                        "mrpc": "eval_f1",
                        "rte":  "eval_accuracy",
                        "stsb": "eval_spearmanr",
                        "sst2": "eval_accuracy",
                        "qnli": "eval_accuracy",
                        "qqp" : "eval_accuracy",
                        "mnli": "eval_accuracy",
                        "snli": "eval_accuracy"
                    }


os.environ['CUDA_VISIBLE_DEVICES'] = f'{GPU}'

# function to parse arguments
def prepare_launch_string(string, string_list):
    launch_string = [s for s in re.split('\s+', string) if s]

    string_list.append(launch_string)

# functions for arguments
launch_strings = []

for TASK_NAME in TASK_NAME_LIST:
    for UNROLL_STEPS in UNROLL_STEPS_LIST:
        for LR in LR_LIST:
            for ALPHA_LR in ALPHA_LR_LIST:
                for ALPHA_WARMUP_RATIO in ALPHA_WARMUP_RATIO_LIST:
                    for ALPHA_WEIGHT_DECAY in ALPHA_WEIGHT_DECAY_LIST:
                        ### do train is missing ###
                        for SEED in SEED_list:
                            # arguments
                            launch_string = f"""
                                --model_name_or_path {MODEL} \
                                --do_train \
                                --do_eval \
                                --task_name {TASK_NAME} \
                                --max_seq_length 128 \
                                --seed {SEED} \
                                --per_device_train_batch_size {BSZ} \
                                --learning_rate {LR} \
                                --num_train_epochs {EPOCH} \
                                --warmup_ratio 0.1 \
                                --weight_decay 0.01 \
                                --output_dir ./Output/{TASK_NAME}/run_{SEED}_{TASK_NAME} \
                                --train_split {TRAIN_SPLIT} \
                                --use_mlo true \
                                --mlo_sample_dataset true \
                                --save_total_limit 1 \
                                --save_steps 30000 \
                                --overwrite_output_dir \
                                --MLO_warm_up {MLO_WARMUP} \
                                --MLO_epochs {MLO_EPOCHS} \
                                --unroll_steps {UNROLL_STEPS} \
                                --model_learning_rate {MODEL_LR} \
                                --alpha_learning_rate {ALPHA_LR} \
                                --alpha_warmup_ratio {ALPHA_WARMUP_RATIO} \
                                --alpha_weight_decay {ALPHA_WEIGHT_DECAY} \
                                --use_l1 {USE_L1} \
                                --L1factor {ALPHA_L1_FACTOR} \
                                --exp_name {SEED}_{TASK_NAME} \
                                --report_freq 100 \
                                --alpha_lr_scheduler_type cosine \
                                --cross_valid 5.0 \
                                --total_avg {AVG}
                            """
                            prepare_launch_string(launch_string, launch_strings)

                        # write the final results
                        with open('Hyperparameters.json', 'a') as wfh:
                            with Pool(num_gpus, maxtasksperchild=1) as p:
                                avg_results = {}
                                list_result = []
                                cross_valid_result = []
                                results = p.imap(main, launch_strings, chunksize=1)
                                
                                for index, result in enumerate(results):
                                    list_result.append((result[0])[dict_tasktometric[TASK_NAME]])
                                    cross_valid_result.append((result[1]))
                                
                                avg_results['mean'] = np.mean(np.array(list_result))
                                avg_results['std'] = np.std(np.array(list_result))

                                avg_results['cross_mean'] = np.mean(np.array(cross_valid_result))
                                avg_results['cross_std'] = np.std(np.array(cross_valid_result))

                                np.savetxt('Results/results_'+str(TASK_NAME)+"_"+str(LR)+"_"+str(EPOCH)+'.out', np.array(list_result), delimiter=',')
                                launch_strings[index].remove('--seed')
                                launch_strings[index].remove(str(SEED))
                                avg_results['setting'] = launch_strings[index]   # 1 is the queue
                                print(json.dumps(avg_results), file=wfh)
                                wfh.flush()

                        launch_strings = []