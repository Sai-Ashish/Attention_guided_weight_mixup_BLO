# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import os
import re
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    get_scheduler,
    get_constant_schedule,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from ChildTuningD import ChildTuningDtrainer
from ChildTuningF import ChildTuningFtrainer
from MLOTuning import MLOTuningTrainer
from bert_modeling import *
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from load_weights import *
import logging

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    # myparams
    reserve_p: float = field(
        default=0.0
    )
    mode: str = field(
        default=None
    )
    train_split: int = field(
        default=-1,
        metadata={"help": "Number of train data examples."},
    )
    use_mlo: bool = field(
        default=False, metadata={"help": "Use MLO or not."}
    )
    mlo_sample_dataset: bool = field(
        default=True, metadata={"help": "Sample MLO train valid dataset or not?"}
    )
    unroll_steps: int = field(
        default=1,
        metadata={"help": "Number of unroll steps for MLO."},
    )
    report_freq: int = field(
        default=50,
        metadata={"help": "Report after report_freq number of steps"},
    )
    device: str = field(
        default="cuda:0",
        metadata={
            "help": "Set the device to run the code."},
    )
    MLO_epochs: int = field(
        default=3,
        metadata={"help": "number of MLO epochs"},
    )
    MLO_warm_up: int = field(
        default=3,
        metadata={"help": "number of warm up MLO steps"},
    )
    L1factor: float = field(
        default=0,
        metadata={
            "help": "the learning rate of alpha"},
    )
    cross_valid: float = field(
        default=5.0,
        metadata={
            "help": "the cross validation coefficient"},
    )
    model_learning_rate: float = field(
        default=2e-5,
        metadata={
            "help": "the learning rate of alpha"},
    )
    alpha_learning_rate: float = field(
        default=2e-5,
        metadata={
            "help": "the learning rate of alpha"},
    )
    alpha_weight_decay: float = field(
        default=0.01,
        metadata={
            "help": "the learning rate of alpha"},
    )
    alpha_warmup_ratio: float = field(
        default=0,
        metadata={
            "help": "the learning rate of alpha"},
    )
    exp_name: str = field(
        default='exp',
        metadata={
            "help": "experiment name"},
    )
    alpha_lr_scheduler_type: str = field(
        default='linear',
        metadata={
            "help": "alpha lr scheduler type"},
    )
    use_l1: bool = field(
        default=False, metadata={"help": "Use L1 loss for alpha"}
    )
    total_avg: bool = field(
        default=False, metadata={"help": "Total average of the model"}
    )

def main(args=''):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    gpu_number = None

    if len(args) == 2:
        arg_strings_list, gpu_queue = args
    else:
        arg_strings_list = args

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif arg_strings_list:
        # if gpu_queue is not None:
        #     gpu_number = gpu_queue.get()
        #     for index, string in enumerate(arg_strings_list):
        #         if 'GPU_NUMBER' in string:
        #             new_string = string.replace("GPU_NUMBER", str(gpu_number))
        #             arg_strings_list[index] = new_string

        model_args, data_args, training_args = parser.parse_args_into_dataclasses(arg_strings_list)

    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue.py", data_args.task_name)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if model_args.use_mlo:

        pretrained_model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = BertForSequenceClassification(config)

        # load the pretrained weights
        load_bert_weights(model, pretrained_model)

        no_decay = ["bias", "LayerNorm.weight"]

        architect = ["alpha", "pretrained_layer"]

        # model weights without the alpha parameter for inner in MLO and finetuning later
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay) and not any(atn in n for atn in architect)],
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           any(nd in n for nd in no_decay) and not any(atn in n for atn in architect)],
                "weight_decay": 0.0,
            },
        ]

        # alpha parameters for outer
        alpha_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if any(atn in n for atn in ["alpha"])],
                "weight_decay": model_args.alpha_weight_decay,
            }
        ]

    else:

        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=data_args.max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[l] for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    train_dataset = datasets["train"]

    ### check the dataset re-distribution ###
    dataset_split = {} # dictionary to resplit the dataset for MLO
    # train set
    if len(train_dataset) > model_args.train_split and model_args.train_split != -1:
        training_subsample_indices = random.sample(range(len(train_dataset)), model_args.train_split)
        dataset_split["train"] = train_dataset.select(training_subsample_indices)
        # dev set
        remaining_indices = list(set(np.arange(len(train_dataset))) - set(training_subsample_indices))
        dev_subsample_indices = random.sample(remaining_indices, int(model_args.train_split/6))
        dataset_split["dev"] = train_dataset.select(dev_subsample_indices)
    else:
        model_args.train_split = len(train_dataset)
        training_subsample_indices = random.sample(range(len(train_dataset)), model_args.train_split)
        dataset_split["train"] = train_dataset
        dataset_split["dev"] = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
    
    if model_args.mlo_sample_dataset:

        dataset_split["MLO-train"] = {} # MLO training set

        dataset_split["MLO-valid"] = {} # MLO validation set

        for i in range(int(model_args.cross_valid)):
            ### MLO train set ###
            MLO_train_indices = random.sample(training_subsample_indices, int(0.8*model_args.train_split))
            dataset_split["MLO-train"][i] = train_dataset.select(MLO_train_indices)
            ### MLO valid set ###
            MLO_valid_indices = list(set(training_subsample_indices) - set(MLO_train_indices))
            dataset_split["MLO-valid"][i]   = train_dataset.select(MLO_valid_indices)

    else:
        ### MLO train set ###
        dataset_split["MLO-train"] = dataset_split["train"]
        ### MLO valid set ###
        dataset_split["MLO-valid"] = dataset_split["train"]
    
    # load the changed/resplit dataset for training and MLO
    train_dataset = dataset_split["train"]
    eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
    dev_dataset = dataset_split["dev"]
    MLO_train_dataset = dataset_split["MLO-train"]
    MLO_valid_dataset = dataset_split["MLO-valid"]
    
    if data_args.task_name is not None or data_args.test_file is not None:
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]

    if model_args.mlo_sample_dataset:

        print("-------------------------------")
        print(len(train_dataset), len(dev_dataset), len(MLO_train_dataset), len(MLO_valid_dataset), len(eval_dataset), len(test_dataset))
        print("-------------------------------")

    else:

        print("-------------------------------")
        print(len(train_dataset), len(dev_dataset), len(MLO_train_dataset[0]), len(MLO_valid_dataset[0]), len(eval_dataset), len(test_dataset))
        print("-------------------------------")
        
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("metric.py", data_args.task_name)
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
        
    # Initialize our Trainer
    assert model_args.mode in ['ChildTuning-F', 'ChildTuning-D', None]
    if model_args.mode is None:

        if model_args.use_mlo:

            trainer = MLOTuningTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,
                gradient_mask=architect
            )

        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
    else:
        if model_args.mode == 'ChildTuning-F':
            trainer_cls = ChildTuningFtrainer
        elif model_args.mode == 'ChildTuning-D':
            trainer_cls = ChildTuningDtrainer
        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
            reserve_p=model_args.reserve_p,
            mode=model_args.mode
        )

    ####################################################
    if model_args.use_mlo:
        from betty.engine import Engine
        from betty.configs import Config, EngineConfig
        from betty.problems import ImplicitProblem

        # define a few parameters for MLO
        # the epochs, training iterations etc.
        if model_args.mlo_sample_dataset:
            num_update_steps_per_epoch = int(( len(MLO_train_dataset[0]) // training_args.per_device_train_batch_size + 1) * model_args.unroll_steps)
        else:
            num_update_steps_per_epoch = int(( len(MLO_train_dataset) // training_args.per_device_train_batch_size + 1) * model_args.unroll_steps)
        
        train_iters = math.ceil(model_args.MLO_epochs * num_update_steps_per_epoch)

        print("---------------------------")
        print("MLO_epochs: ", model_args.MLO_epochs)
        print("train_iters: ", train_iters)
        print("unroll_steps: ", model_args.unroll_steps)
        print("---------------------------")

        # outer model

        # function to normalize the alphas
        def normalise(alpha_list):

            for p in alpha_list:
                with torch.no_grad():
                    p.clamp_(0,1)

        # start of the MLO outer and inner modules
        val_avg_loss = torch.zeros(model_args.report_freq,1)

        class Outer(ImplicitProblem):
            def forward(self, inputs):
                return self.module(**inputs)
            
            def training_step(self, inputs):

                # check alpha is updating
                # print("**************alpha not changing: ",[p for n, p in self.module.named_parameters() if any(atn in n for atn in ["alpha"])][0])

                # check pretrained layer is not updating
                # print([p for n, p in self.module.named_parameters() if any(atn in n for atn in ["pretrained_layer"])][0])
                
                try:
                    inputs.pop('idx')
                except:
                    pass
                
                # normalize alphas
                normalise(self.trainable_parameters())

                outputs = self.inner(inputs)

                loss = outputs.loss

                # L-1 loss
                if model_args.use_l1:

                    reg_loss = 0

                    for param in self.trainable_parameters():
                        reg_loss = reg_loss + torch.norm(param, 1)/(param.view(-1).shape[0])

                    loss = loss + (model_args.L1factor)*reg_loss

                val_avg_loss[(self.count-1)%model_args.report_freq] = loss.item()

                if self.count % model_args.report_freq == 0 and self.count!=0:
                    print(f"step {self.count} || Outer loss: {val_avg_loss.mean()}")

                # print('Outer lr: '+str(self.scheduler.get_lr()[0]))

                return loss

            def trainable_parameters(self):
                # alpha parameters for outer
                return [p for n, p in self.module.named_parameters() if any(atn in n for atn in ["alpha"])]
                
            def param_groups(self):
                return alpha_grouped_parameters

        # inner model
        train_avg_loss = torch.zeros(model_args.report_freq,1)

        class Inner(ImplicitProblem):
            def forward(self, inputs):
                return self.module(**inputs)

            def training_step(self, inputs):

                #check alpha is not updating
                # print("--------------alpha changing: ", [p for n, p in self.module.named_parameters() if any(atn in n for atn in ["alpha"])][0])

                #check pretrained layer is not updating
                # print([p for n, p in self.module.named_parameters() if any(atn in n for atn in ["pretrained_layer"])][0])
                try:
                    inputs.pop('idx')
                except:
                    pass
                
                # normalize alphas
                normalise(self.outer.trainable_parameters())

                outputs = self.outer(inputs)

                loss = outputs.loss

                train_avg_loss[(self.count-1) % model_args.report_freq] = loss.item()

                if self.count % model_args.report_freq == 0 and self.count!=0:
                    print(f"step {self.count} || Inner loss: {train_avg_loss.mean()}")

                # print('Inner: '+str(self.scheduler.get_lr()[0]))

                return loss

            def trainable_parameters(self):
                # W parameters for inner
                return [p for n, p in self.module.named_parameters() if not any(atn in n for atn in architect)]

            def param_groups(self):
                return optimizer_grouped_parameters
        
        eval_loss = [] # to append the evaluation loss of the dev set

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
                
        class NASEngine(Engine):
            @torch.no_grad()
            def validation(self):

                tasks = [data_args.task_name]
                eval_datasets = [dev_dataset]

                with torch.no_grad():

                    model.eval()

                    for _dataset, task in zip(eval_datasets, tasks):

                        eval_result = trainer.evaluate(eval_dataset=_dataset)

                        if trainer.is_world_process_zero():
                            for key, value in sorted(eval_result.items()):
                                print(f"{key} = {value}\n")
                                if key=="eval_loss":
                                    eval_loss.append(value)

                return {"Metric": eval_result[dict_tasktometric[data_args.task_name]]} # change for other datasets for only RTE
        
        if model_args.mlo_sample_dataset:

            train_queue = {}
            
            valid_queue = {}

            for i in range(int(model_args.cross_valid)):
                # dataloader
                valid_queue[i] = torch.utils.data.DataLoader(
                    MLO_valid_dataset[i],
                    sampler=torch.utils.data.sampler.RandomSampler(MLO_valid_dataset[i]),
                    batch_size=training_args.per_device_train_batch_size,
                    collate_fn=data_collator,
                    drop_last=training_args.dataloader_drop_last,
                    num_workers=training_args.dataloader_num_workers,
                    pin_memory=training_args.dataloader_pin_memory
                )

                # inner model dataloader
                train_queue[i] = torch.utils.data.DataLoader(
                    MLO_train_dataset[i],
                    sampler=torch.utils.data.sampler.RandomSampler(MLO_train_dataset[i]),
                    batch_size=training_args.per_device_train_batch_size,
                    collate_fn=data_collator,
                    drop_last=training_args.dataloader_drop_last,
                    num_workers=training_args.dataloader_num_workers,
                    pin_memory=training_args.dataloader_pin_memory
                )
        else:
            valid_queue = torch.utils.data.DataLoader(
                MLO_valid_dataset,
                sampler=torch.utils.data.sampler.RandomSampler(MLO_valid_dataset),
                batch_size=training_args.per_device_train_batch_size,
                collate_fn=data_collator,
                drop_last=training_args.dataloader_drop_last,
                num_workers=training_args.dataloader_num_workers,
                pin_memory=training_args.dataloader_pin_memory
            )

            # inner model dataloader
            train_queue = torch.utils.data.DataLoader(
                MLO_train_dataset,
                sampler=torch.utils.data.sampler.RandomSampler(MLO_train_dataset),
                batch_size=training_args.per_device_train_batch_size,
                collate_fn=data_collator,
                drop_last=training_args.dataloader_drop_last,
                num_workers=training_args.dataloader_num_workers,
                pin_memory=training_args.dataloader_pin_memory
            )

        # Putting together the pieces and running using Betty
        outer_config = Config(type="darts", retain_graph=True, first_order=True, log_step=model_args.report_freq * model_args.unroll_steps)
        inner_config = Config(type="darts", unroll_steps=model_args.unroll_steps, darts_preconditioned=True, log_step=model_args.report_freq * model_args.unroll_steps, gradient_clipping=5.0)
        engine_config = EngineConfig(
            valid_step=model_args.report_freq * model_args.unroll_steps,
            train_iters=train_iters,
            roll_back=True,
            logger_type='tensorboard',
            # name=model_args.exp_name,
        )

        if model_args.mlo_sample_dataset:

            alpha = {}

            # initialize alphas
            for n, p in model.named_parameters():
                if model_args.total_avg:
                    alpha[n] = 0
                else:
                    if "alpha" in n:
                        alpha[n] = 0
            
            cross_valid_result = []

            for i in range(int(model_args.cross_valid)):

                # inner model optimizer and scheduler
                optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)

                optimizer_kwargs['lr'] = model_args.model_learning_rate

                optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                
                scheduler = get_scheduler(
                    name=training_args.lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=(train_iters*training_args.warmup_ratio)//(model_args.unroll_steps),
                    num_training_steps=(train_iters)//(model_args.unroll_steps),
                )

                # outer model optimizer and scheduler
                alpha_optimizer = torch.optim.AdamW(alpha_grouped_parameters, lr=model_args.alpha_learning_rate)

                alpha_scheduler = get_scheduler(
                    name=model_args.alpha_lr_scheduler_type,
                    optimizer=alpha_optimizer,
                    num_warmup_steps=(train_iters*model_args.alpha_warmup_ratio)/(model_args.unroll_steps),
                    num_training_steps=(train_iters)/(model_args.unroll_steps),
                )   

                outer = Outer(name="outer", module=model, optimizer=alpha_optimizer, scheduler=alpha_scheduler, train_data_loader=valid_queue[i], config=outer_config, device=training_args.device)
                inner = Inner(name="inner", module=model, optimizer=optimizer, scheduler=scheduler, train_data_loader=train_queue[i], config=inner_config, device=training_args.device)

                problems = [outer, inner]
                l2u = {inner: [outer]}
                u2l = {outer: [inner]}
                dependencies = {"l2u": l2u, "u2l": u2l}

                engine = NASEngine(config=engine_config, problems=problems, dependencies=dependencies)
                        
                # normalize alphas
                normalise(engine.outer.trainable_parameters())
                
                engine.run()

                # normalize alphas
                normalise(engine.outer.trainable_parameters())

                # get results
                validation_step = engine.validation()
                cross_valid_result.append( validation_step["Metric"] )

                # store alphas
                for n, p in model.named_parameters():
                    if model_args.total_avg:
                        alpha[n] += p.data.detach().clone()
                    else:
                        if "alpha" in n:
                            alpha[n] += p.data.detach().clone()

                ## load a fresh model
                load_bert_weights(model.to(training_args.device), pretrained_model.to(training_args.device))

            # load the alpha weights
            for n, p in model.named_parameters():
                if model_args.total_avg:
                    p.data = ((alpha[n]/model_args.cross_valid).detach().clone())
                else:
                    if "alpha" in n:
                        p.data = ((alpha[n]/model_args.cross_valid).detach().clone())

            # trainer.save_model()

        else:

            # inner model optimizer and scheduler
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)

            optimizer_kwargs['lr'] = model_args.model_learning_rate

            optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            
            scheduler = get_scheduler(
                name=training_args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=(train_iters*training_args.warmup_ratio)//(model_args.unroll_steps),
                num_training_steps=(train_iters)//(model_args.unroll_steps),
            )

            # outer model optimizer and scheduler
            alpha_optimizer = torch.optim.AdamW(alpha_grouped_parameters, lr=model_args.alpha_learning_rate)
            
            alpha_scheduler = get_scheduler(
                name=model_args.alpha_lr_scheduler_type,
                optimizer=alpha_optimizer,
                num_warmup_steps=(train_iters*model_args.alpha_warmup_ratio)/(model_args.unroll_steps),
                num_training_steps=(train_iters)/(model_args.unroll_steps),
            ) 

            outer = Outer(name="outer", module=model, optimizer=alpha_optimizer, scheduler=alpha_scheduler, train_data_loader=valid_queue, config=outer_config, device=training_args.device)
            inner = Inner(name="inner", module=model, optimizer=optimizer, scheduler=scheduler, train_data_loader=train_queue, config=inner_config, device=training_args.device)

            problems = [outer, inner]
            l2u = {inner: [outer]}
            u2l = {outer: [inner]}
            dependencies = {"l2u": l2u, "u2l": u2l}

            engine = NASEngine(config=engine_config, problems=problems, dependencies=dependencies)
                    
            # normalize alphas
            normalise(engine.outer.trainable_parameters())
            
            engine.run()

            # normalize alphas
            normalise(engine.outer.trainable_parameters())
    ###############################################

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics

        # trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info(f"***** Eval results {task} *****")
                    for key, value in sorted(eval_result.items()):
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")

            eval_results.update(eval_result)

    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            test_dataset.remove_columns_("label")
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")
    
    if model_args.mlo_sample_dataset:
        return eval_results, max(cross_valid_result)
    else:
        return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
