import os
import random
import numpy as np
import torch
import argparse
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from datasets import load_dataset
import evaluate
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.nn as nn
import pickle
from utils.bert_utils import *
from pdb import set_trace as st
from utils.bert_methods import *


parser = argparse.ArgumentParser(description="loading")

parser.add_argument("--dataset", default="sst2", type=str, help="dataset")
parser.add_argument("--model", default="bert-base-cased", type=str, help="Model you want to fine-tune")
parser.add_argument("--max_length", default=512, type=int, help="Max sequence length with padding")
parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate for training")
parser.add_argument("--bs", default=32, type=int, help="batch size")
parser.add_argument("--epochs", default=3, type=int, help="epochs for finetune")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay for training")
parser.add_argument("--gpus", type=str, default="2,3", help="gpu ids")
parser.add_argument("--recover_lr", default=1e-5, type=float, help="Learning rate for recovering")
parser.add_argument("--recover_epochs", default=3, type=int, help="epochs for recovering")
parser.add_argument("--obfus", default="translinkguard", type=str, help="obfuscation method")
parser.add_argument("--blackbox_dir", default="results/blackbox_results", type=str, help="blackbox model dir")
parser.add_argument("--recover_data_dir", default="data/recover_data", type=str, help="data for recovering finetune")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if args.model == "bert-base-cased":
    model_name = "bert"
else:
    raise ValueError("Invalid model name")

args.blackbox_dir = f"{args.blackbox_dir}/{model_name}/{args.dataset}"
args.recover_data_dir = f"{args.recover_data_dir}/{model_name}/{args.dataset}"
os.makedirs(args.blackbox_dir, exist_ok=True)
os.makedirs(args.recover_data_dir, exist_ok=True)
set_seed()

# number of classes in the dataset
actual_task = "mnli" if args.dataset == "mnli-mm" else args.dataset
num_labels = 3 if actual_task.startswith("mnli") else (1 if actual_task == "stsb" else 2)
validation_key = "validation_mismatched" if args.dataset == "mnli-mm" else "validation_matched" if args.dataset == "mnli" else "validation"

# Prepare data
print("Preparing data..")
task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mnli-mm": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
sentence1_key, sentence2_key = task_to_keys[args.dataset]
trainset, evalset, tokenizer = prepare_data(actual_task, args.model, validation_key, sentence1_key, sentence2_key, args.max_length)

# metric
print("Loading metric..")
metric = evaluate.load('glue', actual_task)

# 保存recover_data
# 如果存在recover_data，则不再生成
path = f"{args.recover_data_dir}/recover_data.json"
if not os.path.exists(f"{args.recover_data_dir}/recover_data.json"):
    prepare_recover_data(model, trainset, args.bs, path, ratio = 0.01)
recover_dataset = load_dataset("json", data_files=path)["train"]
print("recover_data prepared!")
#st()


# Trainer
blackmodel_args = TrainingArguments(
    output_dir=f"{args.blackbox_dir}",
    eval_strategy='epoch',  # 每个epoch进行评估
    logging_strategy='epoch',
    save_strategy="epoch",  # 每个epoch保存
    learning_rate=args.recover_lr,
    per_device_train_batch_size=args.bs,
    per_device_eval_batch_size=args.bs,
    num_train_epochs=args.recover_epochs,
    weight_decay=args.weight_decay,
    dataloader_num_workers=4,  # 使用数据加载器的并行线程
    seed=42,
)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if actual_task != "stsb":
        predictions = np.argmax(predictions, axis=-1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

if not os.path.exists(f"{args.blackbox_dir}/final_checkpoint"):    
    set_seed()
    init_model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels, use_safetensors=True)
    trainer = Trainer(
        model=init_model,
        args=blackmodel_args,
        train_dataset=recover_dataset,
        eval_dataset=evalset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    blackbox_results = trainer.evaluate(eval_dataset=evalset)
    print(f"黑盒情况下的结果:{blackbox_results}")

else:
    set_seed()
    blackbox_model = AutoModelForSequenceClassification.from_pretrained(f"{args.blackbox_dir}/final_checkpoint", num_labels=num_labels, use_safetensors=True) 
    trainer = Trainer(
        model=blackbox_model,
        args=blackmodel_args,
        train_dataset=recover_dataset,
        eval_dataset=evalset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    blackbox_results = trainer.evaluate(eval_dataset=evalset)
    print(f"黑盒情况下的结果:{blackbox_results}")