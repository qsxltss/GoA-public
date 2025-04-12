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
from pdb import set_trace as st
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import *
import torch.nn as nn


os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Fine-tune a BERT-like model on a GLUE task.')
    parser.add_argument("--dataset", default="sst2", type=str, help="Any dataset in GLUE")
    parser.add_argument("--model", default="bert-base-cased", type=str, help="Model you want to fine-tune")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training")
    parser.add_argument("--max_length", default=512, type=int, help="Max sequence length with padding")
    parser.add_argument("--lr", default=2e-5, type=float, help="Learning rate for training")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay for training")
    parser.add_argument("--epochs", default=3, type=int, help="Number of epochs")
    parser.add_argument("--output_dir", default="results/train_results", type=str, help="Directory to save fine-tuned model")
    parser.add_argument("--seed", default=42, type=int, help="Seed for reproducibility")
    parser.add_argument("--gpus", default="2,3", type=str, help="GPU to use")
    return parser.parse_args()


#计算model权重的平方和
def loss1(model):
    loss = 0
    for name, param in model.named_parameters():
        if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name or "output.dense.bias" in name or "intermediate.dense.bias" in name:
            loss += torch.sum(param ** 2)
    return loss

#计算pre_model和model的权重的L2距离
def loss2(model, pre_model):
    loss = 0
    #print()
    for name, param in model.named_parameters():
        if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name or "output.dense.bias" in name or "intermediate.dense.bias" in name:
            name = name.replace("module.", "")
            pre_data = pre_model.state_dict()[name]
            #将pre_data传到param.data的设备上
            pre_data = pre_data.to(param.device)
            #print(torch.sum((param - pre_data) ** 2))
            loss += torch.sum((param - pre_data) ** 2)
    return torch.sqrt(loss+1e-8)

class CustomTrainer(Trainer):
    def __init__(self, pre_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_model = pre_model
        
    def compute_loss(self, model, inputs, return_outputs=False):
        # 获取模型的输出
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        # 自定义损失逻辑
        loss_fn = nn.CrossEntropyLoss()
        #print(self.pre_model.state_dict().keys())
        #st()
        loss_1 = 1e-4*loss1(model)
        loss_2 = 1e-4*loss2(model, self.pre_model)
        loss = loss_fn(logits, labels)+loss_1-loss_2
        return (loss, outputs) if return_outputs else loss


def fine_tune(args):
    """Fine-tune the model with specified arguments."""
    set_seed(args.seed)
    
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

    actual_task = "mnli" if args.dataset == "mnli-mm" else args.dataset
    num_labels = 3 if actual_task.startswith("mnli") else (1 if actual_task == "stsb" else 2)
    validation_key = "validation_mismatched" if args.dataset == "mnli-mm" else "validation_matched" if args.dataset == "mnli" else "validation"

    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_dataset("glue", actual_task)
    
    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], padding='max_length', truncation=True, max_length=args.max_length)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], padding='max_length', truncation=True, max_length=args.max_length)
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Load model
    print("Loading model...")
    set_seed()
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
    set_seed()
    pre_model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
    
    # Load metric
    print("Loading metric...")
    metric = evaluate.load('glue', actual_task)

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/{args.dataset}",
        evaluation_strategy='epoch',  # 模型的评估频率
        save_strategy="epoch",  # 模型的保存频率
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_dir="./logs", 
        logging_steps=100,  
        dataloader_num_workers=4,  
        local_rank=-1,  
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if actual_task != "stsb":
            predictions = np.argmax(predictions, axis=-1)
        else:
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)

    # Initialize Trainer
    print("Initializing Trainer...")
    trainer = CustomTrainer(
        model=model,
        pre_model = pre_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets[validation_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Evaluate the pre-trained model
    print("Evaluate the pre-trained model...")
    pretrained_model_perf =  trainer.evaluate(eval_dataset=tokenized_datasets[validation_key])
    print(pretrained_model_perf)
    
    # Train the model
    print("Start fine-tuning...")
    trainer.train()

    # Evaluate the fine-tuned model
    print("Evaluate the fine-tuned model...")
    fine_tuned_model_perf = trainer.evaluate(eval_dataset=tokenized_datasets[validation_key])
    print(fine_tuned_model_perf)
    
    print(f"Pre-trained model performance on {args.dataset}: {pretrained_model_perf}")
    print(f"Fine-tuned model performance on {args.dataset}: {fine_tuned_model_perf}")

if __name__ == "__main__":
    args = parse_args()
    if args.dataset == "mnli" or args.dataset == "sst2":
        args.lr = 2e-5
    elif args.dataset == "qnli" or args.dataset == "qqp":
        args.lr = 3e-5
    else:
        raise ValueError("Invalid dataset")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    fine_tune(args)