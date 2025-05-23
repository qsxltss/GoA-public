import sys
import time
import torch
import torch.nn as nn
import os
import datasets
import torchvision
import torchvision.transforms as transforms
import random
import pickle
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
#from vtab import get_data, get_classes_num
from pdb import set_trace as st
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    GPT2ForSequenceClassification, 
    TrainerCallback
)
from datasets import load_dataset
import json
import matplotlib.pyplot as plt

def set_seed(seed=42):
    """Set seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def prepare_data(task,model,validation_key,sentence1_key,sentence2_key, max_length=512):
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model)
    dataset = load_dataset("glue", task)
    
    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], padding='max_length', truncation=True, max_length=max_length)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], padding='max_length', truncation=True, max_length=max_length)
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets[validation_key]
    return train_dataset, eval_dataset, tokenizer

def prepare_recover_data(model, trainset, batch_size, path, ratio=1):
    set_seed()
    all_indices = list(range(len(trainset)))
    num_samples = int(len(trainset) * ratio)
    indices = random.sample(all_indices, num_samples)

    recover_data = {key: [] for key in trainset[0].keys() if key != 'label'}
    recover_data['label'] = []  
    true_labels = []  
    
    for idx in indices:
        item = trainset[idx]
        for key in recover_data.keys():
            if key != 'label':
                recover_data[key].append(item[key])
        true_labels.append(item['label']) 

    subset = torch.utils.data.Subset(trainset, indices)
    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        token_type_ids = [item['token_type_ids'] for item in batch]
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }
    dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    correct, tot = 0, 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing"):
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                'token_type_ids': batch['token_type_ids'].to(device)
            }
            outputs = model(**inputs)
            outputs = outputs.logits if hasattr(outputs, 'logits') else outputs
            predictions = torch.argmax(outputs, dim=1)
            for i in range(inputs["input_ids"].size(0)):
                predicted_label = predictions[i].cpu().tolist()
                recover_data["label"].append(predicted_label)  
                tot += 1
                if predicted_label == true_labels[len(recover_data["label"]) - 1]:
                    correct += 1

    accuracy = correct / tot
    print(f"预测的label和真实label的正确比例: {accuracy:.4f}")
    
    with open(path, "w") as f:
        json.dump(recover_data, f, indent=4)

def loss1(model):
    loss = 0
    for name, param in model.named_parameters():
        if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name or "output.dense.bias" in name or "intermediate.dense.bias" in name:
            loss += torch.sum(param ** 2)
    return loss

def loss2(model, pre_model):
    loss = 0
    #print()
    for name, param in model.named_parameters():
        if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name or "output.dense.bias" in name or "intermediate.dense.bias" in name:
            name = name.replace("module.", "")
            pre_data = pre_model.state_dict()[name]
            pre_data = pre_data.to(param.device)
            loss += torch.sum((param - pre_data) ** 2)
    return torch.sqrt(loss+1e-8)

class CustomTrainer(Trainer):
    def __init__(self, pre_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_model = pre_model
        
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        loss_fn = nn.CrossEntropyLoss()
        loss_1 = 1e-4*loss1(model)
        loss_2 = 1e-4*loss2(model, self.pre_model)
        loss = loss_fn(logits, labels)+loss_1-loss_2
        return (loss, outputs) if return_outputs else loss


def row_restore_perm(pre_model_mat, model_mat, threshold=0.0):
    model_mat_cpu = model_mat.cpu().numpy()
    pre_model_mat_cpu = pre_model_mat.cpu().numpy()
    similarty_matrix = cosine_similarity(model_mat_cpu, pre_model_mat_cpu)
    perm = np.argmax(similarty_matrix, axis=1)
    restored_matrix = np.empty_like(model_mat_cpu)
    success = []
    for i, row in enumerate(model_mat_cpu):
        max_similarity = similarty_matrix[i, perm[i]]
        if max_similarity >= threshold:
            restored_matrix[perm[i]] = model_mat_cpu[i]
            success.append(perm[i])
    for i in range(len(restored_matrix)):
        if i not in success:
            restored_matrix[i] = pre_model_mat_cpu[i]
    restored_matrix = torch.from_numpy(restored_matrix).to(model_mat.device)
    return perm, success, restored_matrix

def col_restore_perm(pre_model_mat, model_mat, threshold=0.0, perm = None):
    model_mat_cpu = model_mat.cpu().numpy()
    pre_model_mat_cpu = pre_model_mat.cpu().numpy()
    similarty_matrix = cosine_similarity(model_mat_cpu.T, pre_model_mat_cpu.T)
    if perm is None:
        perm = np.argmax(similarty_matrix, axis=1)
    restored_matrix = np.empty_like(model_mat_cpu)
    success = []
    for i, col in enumerate(model_mat_cpu.T):
        max_similarity = similarty_matrix[i, perm[i]]
        if max_similarity >= threshold:
            restored_matrix[:,perm[i]] = model_mat_cpu[:,i]
            success.append(perm[i])
    for i in range(len(restored_matrix[0])):
        if i not in success:
            restored_matrix[:,i] = pre_model_mat_cpu[:,i]
    restored_matrix = torch.from_numpy(restored_matrix).to(model_mat.device)
    return perm, success, restored_matrix
    

def fix_factor(num, mini=1.0, max=6.0):
    if(num < mini):
        return mini
    elif(num > max):
        return max
    else:
        return num

def check_diff_perm(perm, true_perm):
    tot = 0
    for i in range(len(perm)):
        if perm[i] == true_perm[i]:
            tot += 1
    same_ratio = tot / len(perm)
    return same_ratio

def check_similarity(pre_model_mat, model_mat,threshold=0.8):
    similarty_matrix = cosine_similarity(model_mat, pre_model_mat)
    for i in range(len(similarty_matrix)):
        if similarty_matrix[i][i] < threshold:
            model_mat[i] = pre_model_mat[i]
            print(f"index: {i}, similarity: {similarty_matrix[i][i]}")
            break
    return model_mat
    
       


def check_similar(model_mat,pre_model_mat, name=""):
    model_mat_cpu = model_mat.cpu().numpy()
    pre_model_mat_cpu = pre_model_mat.cpu().numpy()
    similarity_matrix = cosine_similarity(model_mat_cpu, pre_model_mat_cpu)
    # 初始化结果列表
    max_values = []
    second_max_values = []
    mean_values = []
    max_positions = []
    second_max_positions = []
    for i, row in enumerate(similarity_matrix):
        mean_value = np.mean(row)
        max_value = np.max(row)
        max_pos = np.argmax(row)
        row[max_pos] = -np.inf
        second_max_value = np.max(row)
        row[max_pos] = max_value
        max_values.append(max_value)
        second_max_values.append(second_max_value)
        mean_values.append(mean_value)
    # 输出结果
    print("Row-wise Mean values:", np.mean(mean_values))
    print("Row-wise Max values:", np.mean(max_values))
    print("Row-wise Second Max values:", np.mean(second_max_values))   
    return np.mean(mean_values), np.mean(max_values), np.mean(second_max_values)

def check_distance(model_mat, pre_model_mat, name=""):
    model_mat_cpu = model_mat.cpu().numpy()
    pre_model_mat_cpu = pre_model_mat.cpu().numpy()

    max_values_l2 = []
    second_max_values_l2 = []
    all_values_l2 = [] 
    max_values_linf = []
    second_max_values_linf = []
    all_values_linf = [] 
    for i in range(model_mat_cpu.shape[0]):
        model_mat_cpu[i] = model_mat_cpu[i] / np.linalg.norm(model_mat_cpu[i], ord=2)
        pre_model_mat_cpu[i] = pre_model_mat_cpu[i] / np.linalg.norm(pre_model_mat_cpu[i], ord=2)
        
    for i in range(model_mat_cpu.shape[0]):
        row1 = model_mat_cpu[i]
        row2 = pre_model_mat_cpu[i]
        
        max_value_l2 = np.linalg.norm(row1 - row2, ord=2)
        max_value_linf = np.max(np.abs(row1 - row2))
        max_values_l2.append(max_value_l2)
        max_values_linf.append(max_value_linf)

        diffs_l2 = np.linalg.norm(pre_model_mat_cpu - row1, axis=1, ord=2)
        diffs_linf = np.max(np.abs(pre_model_mat_cpu - row1), axis=1)
        
        mean_value_l2 = np.mean(diffs_l2) 
        mean_value_linf = np.mean(diffs_linf) 
        all_values_l2.append(mean_value_l2)
        all_values_linf.append(mean_value_linf)
        
        diffs_l2[i] = np.inf
        diffs_linf[i] = np.inf
        
        second_min_value_l2 = np.min(diffs_l2)  
        second_min_value_linf = np.min(diffs_linf) 

        second_max_values_l2.append(second_min_value_l2)
        second_max_values_linf.append(second_min_value_linf)

    # 计算所有行的均值
    mean_value_l2 = np.mean(all_values_l2)  # 所有行的 L2 均值
    mean_value_linf = np.mean(all_values_linf)  # 所有行的 L∞ 均值

    # 输出结果
    print(f"{name} L2 mean distance (row-wise avg): {mean_value_l2}")
    print(f"{name} L2 min distance: {np.mean(max_values_l2)}")
    print(f"{name} L2 second min distance: {np.mean(second_max_values_l2)}")
    print(f"{name} Linf mean distance (row-wise avg): {mean_value_linf}")
    print(f"{name} Linf min distance: {np.mean(max_values_linf)}")
    print(f"{name} Linf second min distance: {np.mean(second_max_values_linf)}")

    return mean_value_l2, np.mean(max_values_l2), np.mean(second_max_values_l2), mean_value_linf, np.mean(max_values_linf), np.mean(second_max_values_linf)


#loss曲线画图
class SaveLossPlotCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.train_losses = []
        self.eval_losses = []
        self.output_dir = output_dir

    def on_log(self, args, state, control, **kwargs):
        if state.log_history:
            latest_log = state.log_history[-1]
            if 'loss' in latest_log:
                self.train_losses.append(latest_log['loss'])
            if 'eval_loss' in latest_log:
                self.eval_losses.append(latest_log['eval_loss'])

    def on_train_end(self, args, state, control, **kwargs):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.eval_losses, label='Evaluation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Evaluation Loss Over Epochs')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'loss_plot.png'))
        plt.close()

