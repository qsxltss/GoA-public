import sys
import time
import torch
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

    # 初始化recover_data包含与trainset相同的字段（除了label）
    recover_data = {key: [] for key in trainset[0].keys() if key != 'label'}
    recover_data['label'] = []  # 单独添加label字段，用于存放模型预测结果
    true_labels = []  # 用于存放真实标签
    
    # 逐个提取数据项并添加到recover_data中
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
                recover_data["label"].append(predicted_label)  # 记录预测的label
                tot += 1
                # 比较预测结果和真实标签
                if predicted_label == true_labels[len(recover_data["label"]) - 1]:
                    correct += 1
    # 计算并输出正确比例
    accuracy = correct / tot
    print(f"预测的label和真实label的正确比例: {accuracy:.4f}")
    
    with open(path, "w") as f:
        json.dump(recover_data, f, indent=4)


def row_restore_perm(pre_model_mat, model_mat, threshold=0.0):
    # 将矩阵从GPU中转移到CPU并转为numpy数组
    model_mat_cpu = model_mat.cpu().numpy()
    pre_model_mat_cpu = pre_model_mat.cpu().numpy()
    # 计算cosine similarity矩阵
    similarty_matrix = cosine_similarity(model_mat_cpu, pre_model_mat_cpu)
    # 根据similarity_matrix找出每一行的最佳匹配
    perm = np.argmax(similarty_matrix, axis=1)
    # 构建恢复的矩阵
    restored_matrix = np.empty_like(model_mat_cpu)
    success = []
    for i, row in enumerate(model_mat_cpu):
        # 取这一行的最大similarity值
        max_similarity = similarty_matrix[i, perm[i]]
        if max_similarity >= threshold:
            # 如果similarity大于等于阈值, 使用perm对应的行
            restored_matrix[perm[i]] = model_mat_cpu[i]
            success.append(perm[i])
    for i in range(len(restored_matrix)):
        if i not in success:
            restored_matrix[i] = pre_model_mat_cpu[i]
    # 转换为PyTorch张量并移回GPU
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
    #print(f"total same_ratio: {same_ratio}")
    return same_ratio

# 检查similarity是否有异常
# 有异常则替换为pre_model_mat
def check_similarity(pre_model_mat, model_mat,threshold=0.8):
    similarty_matrix = cosine_similarity(model_mat, pre_model_mat)
    for i in range(len(similarty_matrix)):
        if similarty_matrix[i][i] < threshold:
            model_mat[i] = pre_model_mat[i]
            print(f"index: {i}, similarity: {similarty_matrix[i][i]}")
            break
    return model_mat
    
       


def check_similar(model_mat,pre_model_mat, name=""):
    # 将矩阵从GPU中转移到CPU并转为numpy数组
    model_mat_cpu = model_mat.cpu().numpy()
    pre_model_mat_cpu = pre_model_mat.cpu().numpy()
    # 计算cosine similarity矩阵
    similarity_matrix = cosine_similarity(model_mat_cpu, pre_model_mat_cpu)
    # 初始化结果列表
    max_values = []
    second_max_values = []
    mean_values = []
    max_positions = []
    second_max_positions = []
    # 遍历每一行，计算最大值、第二大值和平均值
    for i, row in enumerate(similarity_matrix):
        # 计算每行的平均值
        mean_value = np.mean(row)
        # 找到每行的最大值和对应的位置
        max_value = np.max(row)
        max_pos = np.argmax(row)
        # 将最大值位置的元素设为负无穷，以找到第二大值
        row[max_pos] = -np.inf
        second_max_value = np.max(row)
        #second_max_pos = np.argmax(row)
        # 恢复原始最大值
        row[max_pos] = max_value
        # 将结果存入列表
        max_values.append(max_value)
        second_max_values.append(second_max_value)
        mean_values.append(mean_value)
        #max_positions.append(max_pos)
        #second_max_positions.append(second_max_pos)
    # 输出结果
    print("Row-wise Mean values:", np.mean(mean_values))
    print("Row-wise Max values:", np.mean(max_values))
    #print("Row-wise Max positions:", np.mean(max_positions))
    print("Row-wise Second Max values:", np.mean(second_max_values))
    #print("Row-wise Second Max positions:", second_max_positions)   
    return np.mean(mean_values), np.mean(max_values), np.mean(second_max_values)

def check_distance(model_mat, pre_model_mat, name=""):
    # 将数据一次性转换为 NumPy 数组
    model_mat_cpu = model_mat.cpu().numpy()
    pre_model_mat_cpu = pre_model_mat.cpu().numpy()

    # 初始化结果列表
    max_values_l2 = []
    second_max_values_l2 = []
    all_values_l2 = []  # 用于记录每一行与其他行的 L2 距离均值
    max_values_linf = []
    second_max_values_linf = []
    all_values_linf = []  # 用于记录每一行与其他行的 Linf 距离均值
    for i in range(model_mat_cpu.shape[0]):
        #归一化
        model_mat_cpu[i] = model_mat_cpu[i] / np.linalg.norm(model_mat_cpu[i], ord=2)
        pre_model_mat_cpu[i] = pre_model_mat_cpu[i] / np.linalg.norm(pre_model_mat_cpu[i], ord=2)
    # 计算每一行之间的距离
    for i in range(model_mat_cpu.shape[0]):
        row1 = model_mat_cpu[i]
        row2 = pre_model_mat_cpu[i]
        
        max_value_l2 = np.linalg.norm(row1 - row2, ord=2)
        max_value_linf = np.max(np.abs(row1 - row2))
        max_values_l2.append(max_value_l2)
        max_values_linf.append(max_value_linf)

        # 向量化计算所有行之间的 L2 和 Linf 距离
        diffs_l2 = np.linalg.norm(pre_model_mat_cpu - row1, axis=1, ord=2)
        diffs_linf = np.max(np.abs(pre_model_mat_cpu - row1), axis=1)
        
        mean_value_l2 = np.mean(diffs_l2) 
        mean_value_linf = np.mean(diffs_linf) 
        all_values_l2.append(mean_value_l2)
        all_values_linf.append(mean_value_linf)
        
        # 排除自己与自己计算的距离
        diffs_l2[i] = np.inf
        diffs_linf[i] = np.inf
        
        second_min_value_l2 = np.min(diffs_l2)  # 第二小 L2 距离
        second_min_value_linf = np.min(diffs_linf)  # 第二小 L∞ 距离

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
        # 在日志记录时捕获训练损失
        if state.log_history:
            latest_log = state.log_history[-1]
            if 'loss' in latest_log:
                self.train_losses.append(latest_log['loss'])
            if 'eval_loss' in latest_log:
                self.eval_losses.append(latest_log['eval_loss'])

    def on_train_end(self, args, state, control, **kwargs):
        # 训练结束后绘制损失曲线图
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.eval_losses, label='Evaluation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Evaluation Loss Over Epochs')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'loss_plot.png'))
        plt.close()