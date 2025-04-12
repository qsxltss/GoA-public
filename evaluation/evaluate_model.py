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
#from datasets import load_dataset
import evaluate
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.nn as nn
import pickle
from pdb import set_trace as st
from utils.utils import *


parser = argparse.ArgumentParser(description="loading")

parser.add_argument("--dataset", default="cola", type=str, help="dataset")
parser.add_argument("--model", default="bert-base-cased", type=str, help="Model you want to fine-tune")
parser.add_argument("--max_length", default=512, type=int, help="Max sequence length with padding")
parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate for training")
parser.add_argument("--bs", default=32, type=int, help="batch size")
parser.add_argument("--epochs", default=3, type=int, help="epochs for finetune")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay for training")
parser.add_argument("--gpus", type=str, default="0,1", help="gpu ids") # GPUs to use

parser.add_argument("--obfus", default="none", type=str, help="obfuscation method")
parser.add_argument("--output_dir", default="evaluate_results", type=str, help="output directory")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.makedirs(f"{args.output_dir}/{args.obfus}/{args.dataset}", exist_ok=True)

if args.model == "bert-base-cased":
    model_name = "bert"

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

# trainer & metrics
print("Loading metric..")
metric = evaluate.load('glue', actual_task)

# weight_dir
if args.obfus == "none":
    weight_dir = f"results/train_results/{model_name}/{args.dataset}/final_checkpoint"
elif args.obfus == "translinkguard" or args.obfus == "soter" or args.obfus == "tsqp" or args.obfus == "tempo" or args.obfus == "shadownet":
    weight_dir = f"results/arrowmatch_results/{model_name}/{args.obfus}/{args.dataset}/final_checkpoint"
else:
    raise ValueError(f"Obfuscation method {args.obfus} not supported")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if actual_task != "stsb":
        predictions = np.argmax(predictions, axis=-1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)


# Model
print("Building model..")
if model_name == "bert":
    set_seed()
    model = AutoModelForSequenceClassification.from_pretrained(
        weight_dir,  # 指定权重文件所在的目录
        num_labels=num_labels,
        use_safetensors=True  # 添加这一行，以明确使用 safetensors 格式
    )
else:
    # vit and gpt2 will be added soon
    raise ValueError(f"Model {args.model} not supported")


training_args = TrainingArguments(
    output_dir=f"{args.output_dir}/{args.obfus}/{args.dataset}",
    eval_strategy='no',  # 模型的评估频率
    save_strategy="epoch",  # 模型的保存频率
    learning_rate=args.lr,
    per_device_train_batch_size=args.bs,
    per_device_eval_batch_size=args.bs,
    num_train_epochs=args.epochs,
    weight_decay=args.weight_decay,
    dataloader_num_workers=4,  
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=trainset,
    eval_dataset=evalset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
results = trainer.evaluate(eval_dataset=evalset)
print("Evaluation results: ", results)