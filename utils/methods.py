from pdb import set_trace as st
import torch
import numpy as np
import random
from utils.utils import *
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from math import sqrt

def ob_translinkguard(model):
    set_seed()
    layer_permutations = {}
    rows = 0
    for name, module in model.named_parameters():
        #print(name)
        if "query.weight" in name:
            w_q = module.data
            layer_name = name.rsplit(".")[3]
            # 打乱行顺序
            # 注意data的行列是反过来的：fc2.data = 768*3072
            num_rows = w_q.shape[1]
            rows = num_rows
            permutation = torch.randperm(num_rows)
            layer_permutations[layer_name] = permutation
            ob_w_q = w_q[:,permutation]
            module.data = ob_w_q
        elif "key.weight" in name:
            w_k = module.data
            layer_name = name.rsplit(".")[3]
            permutation = layer_permutations[layer_name]
            ob_w_k = w_k[:,permutation]
            module.data = ob_w_k
        elif "value.weight" in name:
            w_v = module.data
            layer_name = name.rsplit(".")[3]
            permutation = layer_permutations[layer_name]
            ob_w_v = w_v[:,permutation]
            module.data = ob_w_v
        elif "attention.output.dense.weight" in name:
            w_o = module.data
            layer_name = name.rsplit(".")[3]
            permutation = layer_permutations[layer_name]
            inv_perm = torch.argsort(permutation)
            ob_o = w_o[inv_perm, :]
            module.data = ob_o
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            w_proj = module.data
            if(w_proj.shape[1] == rows):
                layer_name = name.rsplit(".")[3]
                permutation = layer_permutations[layer_name]
                ob_proj = w_proj[:, permutation]
                module.data = ob_proj
    return model, layer_permutations, rows

""""""
# 针对translinkguard的攻击
def attack_translinkguard(model, pre_model, rows, permutations=None):
    set_seed()
    restore_perm = {}        
    for name, module in model.named_parameters():
        if "query.weight" in name:
            ob_wq = module.data
            layer_name = name.rsplit(".")[3]
            pre_wq = pre_model.state_dict()[name]
            perm, _, restore_wq = col_restore_perm(pre_wq, ob_wq)
            restore_perm[layer_name] = torch.tensor(perm)
            inv_perm = torch.argsort(torch.tensor(perm))
            restore_wq = ob_wq[:, inv_perm]
            module.data = restore_wq
        elif "key.weight" in name:
            ob_wk = module.data
            layer_name = name.rsplit(".")[3]
            perm = restore_perm[layer_name]
            inv_perm = torch.argsort(perm)
            restore_wk = ob_wk[:, inv_perm]
            module.data = restore_wk
        elif "value.weight" in name:
            ob_wv = module.data
            layer_name = name.rsplit(".")[3]
            perm = restore_perm[layer_name]
            inv_perm = torch.argsort(perm)
            restore_wv = ob_wv[:, inv_perm]
            module.data = restore_wv
        elif "attention.output.dense.weight" in name:
            ob_o = module.data
            layer_name = name.rsplit(".")[3]
            perm = restore_perm[layer_name]
            #inv_perm = torch.argsort(perm)
            restore_o = ob_o[perm, :]
            module.data = restore_o
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            ob_proj = module.data
            if(ob_proj.shape[1] == rows):
                layer_name = name.rsplit(".")[3]
                perm = restore_perm[layer_name]
                inv_perm = torch.argsort(perm)
                restore_proj = ob_proj[:, inv_perm]
                module.data = restore_proj
        elif "classifier" in name or "pooler" in name:
            module.data = pre_model.state_dict()[name].data
    return model        
        

def ob_tsqp(model):
    set_seed()  
    scaling_factors = {}  
    # 对模型参数进行遍历
    for name, module in model.named_parameters():
        if "query.weight" in name:
            w_q = module.data
            scale_q = 1 + 5 * torch.rand(1).item()
            w_q *= scale_q
            module.data = w_q
            scaling_factors[name] = scale_q
        elif "key.weight" in name:
            w_k = module.data
            scale_k = 1 + 5 * torch.rand(1).item()
            w_k *= scale_k
            module.data = w_k
            scaling_factors[name] = scale_k
        elif "value.weight" in name:
            w_v = module.data
            scale_v = 1 + 5 * torch.rand(1).item()
            w_v *= scale_v
            module.data = w_v
            scaling_factors[name] = scale_v
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            w_proj = module.data
            scale_proj = 1 + 5 * torch.rand(1).item()
            w_proj *= scale_proj
            module.data = w_proj
            scaling_factors[name] = scale_proj
    # 返回混淆后的模型和系数
    return model, scaling_factors

def attack_tsqp(model, pre_model, scaling_factors=None):
    set_seed()
    restore_scaling_factors = {}
    for name, module in model.named_parameters():
        if "query.weight" in name:
            w_q = module.data
            pre_q = pre_model.state_dict()[name].data
            k = fix_factor(sqrt(torch.var(w_q).item()/torch.var(pre_q).item()))
            new_w_q = w_q/k
            restore_scaling_factors[name] = k
            module.data = new_w_q
        elif "key.weight" in name:
            w_k = module.data
            pre_k = pre_model.state_dict()[name].data
            k = fix_factor(sqrt(torch.var(w_k).item()/torch.var(pre_k).item()))
            new_w_k = w_k/k
            restore_scaling_factors[name] = k
            module.data = new_w_k
        elif "value.weight" in name:
            w_v = module.data
            pre_v = pre_model.state_dict()[name].data
            k = fix_factor(sqrt(torch.var(w_v).item()/torch.var(pre_v).item()))
            new_w_v = w_v/k
            restore_scaling_factors[name] = k
            module.data = new_w_v
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:      
            w_proj = module.data
            pre_proj = pre_model.state_dict()[name].data
            k = fix_factor(sqrt(torch.var(w_proj).item()/torch.var(pre_proj).item()))
            new_w_proj = w_proj/k
            restore_scaling_factors[name] = k
            module.data = new_w_proj
        elif "classifier" in name or "pooler" in name or "LayerNorm" or "bias" in name:
            module.data = pre_model.state_dict()[name].data
    return model, restore_scaling_factors

def ob_soter(model, pre_model):
    set_seed()
    scaling_factors = {}  
    init_layers = {}
    # 随机选取一些权重块加载pre_model的权重,比例为20%
    for name, module in model.named_parameters():
        if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name:
            init_layers[name] = module.data
    # 计算需要替换的层数（20%）
    num_layers_to_replace = int(len(init_layers) * 0.2)
    layers_to_replace = random.sample(list(init_layers.keys()), num_layers_to_replace)
    
    # 替换选中层的权重
    for name, module in model.named_parameters():
        if name in layers_to_replace:
            module.data = pre_model.state_dict()[name].data
            
    for name, module in model.named_parameters():
        if name not in layers_to_replace:
            if "query.weight" in name:
                w_q = module.data
                scale_q = 1 + 5 * torch.rand(1).item()
                w_q *= scale_q
                module.data = w_q
                scaling_factors[name] = scale_q
            elif "key.weight" in name:
                w_k = module.data
                scale_k = 1 + 5 * torch.rand(1).item()
                w_k *= scale_k
                module.data = w_k
                scaling_factors[name] = scale_k
            elif "value.weight" in name:
                w_v = module.data
                scale_v = 1 + 5 * torch.rand(1).item()
                w_v *= scale_v
                module.data = w_v
                scaling_factors[name] = scale_v
            elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
                w_proj = module.data
                scale_proj = 1 + 5 * torch.rand(1).item()
                w_proj *= scale_proj
                module.data = w_proj
                scaling_factors[name] = scale_proj
    return model, scaling_factors, layers_to_replace

def attack_soter(model, pre_model, scaling_factors=None):
    set_seed()
    restore_scaling_factors = {}
    # 对每个权重快恢复对应的系数
    # 恢复系数定义为model和pre_model相同的权重块权重总和的比值
    for name, module in model.named_parameters():
        if "query.weight" in name:
            w_q = module.data
            pre_q = pre_model.state_dict()[name].data
            k = fix_factor(sqrt(torch.var(w_q).item()/torch.var(pre_q).item()))
            new_w_q = w_q/k
            restore_scaling_factors[name] = k
            module.data = new_w_q
        elif "key.weight" in name:
            w_k = module.data
            pre_k = pre_model.state_dict()[name].data
            k = fix_factor(sqrt(torch.var(w_k).item()/torch.var(pre_k).item()))
            new_w_k = w_k/k
            restore_scaling_factors[name] = k
            module.data = new_w_k
        elif "value.weight" in name:
            w_v = module.data
            pre_v = pre_model.state_dict()[name].data
            k = fix_factor(sqrt(torch.var(w_v).item()/torch.var(pre_v).item()))
            new_w_v = w_v/k
            restore_scaling_factors[name] = k
            module.data = new_w_v
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            w_proj = module.data
            pre_proj = pre_model.state_dict()[name].data
            k = fix_factor(sqrt(torch.var(w_proj).item()/torch.var(pre_proj).item()))
            new_w_proj = w_proj/k
            restore_scaling_factors[name] = k
            module.data = new_w_proj
        elif "classifier" in name or "pooler" in name or "LayerNorm" or "bias" in name:
            module.data = pre_model.state_dict()[name].data
    return model, restore_scaling_factors

def ob_shadownet(model):
    set_seed()
    layer_permutations = {}
    scaling_factors = {}
    for name, module in model.named_parameters():
        if "query.weight" in name:
            w_q = module.data
            num_rows = w_q.shape[1]
            #对ob_w_q的每一列乘以一个随机系数
            ratios_q = []
            for i in range(num_rows):
                ratio_q = 1 + 5 * torch.rand(1).item()
                w_q[:,i] *= ratio_q
                ratios_q.append(ratio_q)
            scaling_factors[name] = ratios_q
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            ob_w_q = w_q[:, permutation]
            module.data = ob_w_q
        elif "key.weight" in name:
            w_k = module.data
            num_rows = w_k.shape[1]
            ratios_k = []
            for i in range(num_rows):
                ratio_k = 1 + 5 * torch.rand(1).item()
                w_k[:,i] *= ratio_k
                ratios_k.append(ratio_k)
            scaling_factors[name] = ratios_k
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            ob_w_k = w_k[:, permutation]
            module.data = ob_w_k
        elif "value.weight" in name:
            w_v = module.data
            num_rows = w_v.shape[1]
            ratios_v = []
            for i in range(num_rows):
                ratio_v = 1 + 5 * torch.rand(1).item()
                w_v[:,i] *= ratio_v
                ratios_v.append(ratio_v)
            scaling_factors[name] = ratios_v
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            ob_w_v = w_v[:, permutation]
            module.data = ob_w_v
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            w_proj = module.data
            num_rows = w_proj.shape[1]
            ratios = []
            for i in range(num_rows):
                ratio = 1 + 5 * torch.rand(1).item()
                w_proj[:,i] *= ratio
                ratios.append(ratio)
            scaling_factors[name] = ratios
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            ob_proj = w_proj[:, permutation]
            module.data = ob_proj
    return model, layer_permutations, scaling_factors

def attack_shadownet(model, pre_model, layer_permutations=None, scaling_factors=None):
    set_seed()
    restore_perm = {}
    for name, module in model.named_parameters():
        if "query.weight" in name:
            ob_wq = module.data
            pre_wq = pre_model.state_dict()[name].data
            perm, similarity, restore_wq = col_restore_perm(pre_wq, ob_wq)
            for i in range(ob_wq.shape[1]):
                ratio_q = fix_factor(sqrt(torch.var(restore_wq[:,i]).item()/torch.var(pre_wq[:,i]).item()))
                restore_wq[:,i] /= ratio_q
            module.data = restore_wq
        elif "key.weight" in name:
            ob_wk = module.data
            pre_wk = pre_model.state_dict()[name].data
            perm, similarity, restore_wk = col_restore_perm(pre_wk, ob_wk)
            for i in range(ob_wk.shape[1]):
                ratio_k = fix_factor(sqrt(torch.var(restore_wk[:,i]).item()/torch.var(pre_wk[:,i]).item()))
                restore_wk[:,i] /= ratio_k
            module.data = restore_wk
        elif "value.weight" in name:
            ob_wv = module.data
            pre_wv = pre_model.state_dict()[name].data
            perm, similarity, restore_wv = col_restore_perm(pre_wv, ob_wv)
            for i in range(ob_wv.shape[1]):
                ratio_v = fix_factor(sqrt(torch.var(restore_wv[:,i]).item()/torch.var(pre_wv[:,i]).item()))
                restore_wv[:,i] /= ratio_v
            module.data = restore_wv
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            ob_proj = module.data
            pre_proj = pre_model.state_dict()[name].data
            perm, similarity, restore_proj = col_restore_perm(pre_proj, ob_proj)
            for i in range(ob_proj.shape[1]):
                ratio = fix_factor(sqrt(torch.var(restore_proj[:,i]).item()/torch.var(pre_proj[:,i]).item()))
                restore_proj[:,i] /= ratio
            module.data = restore_proj
        elif "classifier" in name or "pooler" in name or "LayerNorm" or "bias" in name:
            module.data = pre_model.state_dict()[name].data
    return model, restore_perm

def ob_tempo(model):
    set_seed()
    layer_permutations = {}
    scaling_factors = {}
    for name, module in model.named_parameters():
        if "query.weight" in name:
            w_q = module.data
            num_cols = w_q.shape[0]
            ratios_q = []
            for i in range(num_cols):
                ratio_q = 1 + 5 * torch.rand(1).item()
                w_q[i] *= ratio_q
                ratios_q.append(ratio_q)
            scaling_factors[name] = ratios_q
            permutation = torch.randperm(num_cols)
            layer_permutations[name] = permutation
            ob_w_q = w_q[permutation,:]
            module.data = ob_w_q
        elif "key.weight" in name:
            w_k = module.data
            num_cols = w_k.shape[0]
            ratios_k = []
            for i in range(num_cols):
                ratio_k = 1 + 5 * torch.rand(1).item()
                w_k[i] *= ratio_k
                ratios_k.append(ratio_k)
            scaling_factors[name] = ratios_k
            permutation = torch.randperm(num_cols)
            layer_permutations[name] = permutation
            ob_w_k = w_k[permutation,:]
            module.data = ob_w_k
        elif "value.weight" in name:
            w_v = module.data
            num_cols = w_v.shape[0]
            ratios_v = []
            for i in range(num_cols):
                ratio_v = 1 + 5 * torch.rand(1).item()
                w_v[i] *= ratio_v
                ratios_v.append(ratio_v)
            scaling_factors[name] = ratios_v
            permutation = torch.randperm(num_cols)
            layer_permutations[name] = permutation
            ob_w_v = w_v[permutation,:]
            module.data = ob_w_v
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            w_proj = module.data
            num_cols = w_proj.shape[0]
            ratios = []
            for i in range(num_cols):
                ratio = 1 + 5 * torch.rand(1).item()
                w_proj[i] *= ratio
                ratios.append(ratio)
            scaling_factors[name] = ratios
            permutation = torch.randperm(num_cols)
            layer_permutations[name] = permutation
            ob_proj = w_proj[permutation,:]
            module.data = ob_proj
    return model, layer_permutations, scaling_factors

def attack_tempo(model, pre_model):
    set_seed()
    restore_perm = {}
    for name, module in model.named_parameters():
        if "query.weight" in name:
            ob_wq = module.data
            pre_wq = pre_model.state_dict()[name].data
            perm, similarity, restore_wq = row_restore_perm(pre_wq, ob_wq)
            for i in range(ob_wq.shape[0]):
                ratio_q = fix_factor(sqrt(torch.var(restore_wq[i]).item()/torch.var(pre_wq[i]).item()))
                restore_wq[i] /= ratio_q
            module.data = restore_wq
        elif "key.weight" in name:
            ob_wk = module.data
            pre_wk = pre_model.state_dict()[name].data
            perm, similarity, restore_wk = row_restore_perm(pre_wk, ob_wk)
            for i in range(ob_wk.shape[0]):
                ratio_k = fix_factor(sqrt(torch.var(restore_wk[i]).item()/torch.var(pre_wk[i]).item()))
                restore_wk[i] /= ratio_k
            module.data = restore_wk
        elif "value.weight" in name:
            ob_wv = module.data
            pre_wv = pre_model.state_dict()[name].data
            perm, similarity, restore_wv = row_restore_perm(pre_wv, ob_wv)
            for i in range(ob_wv.shape[0]):
                ratio_v = fix_factor(sqrt(torch.var(restore_wv[i]).item()/torch.var(pre_wv[i]).item()))
                restore_wv[i] /= ratio_v
            module.data = restore_wv
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            ob_proj = module.data
            pre_proj = pre_model.state_dict()[name].data
            perm, similarity, restore_proj = row_restore_perm(pre_proj, ob_proj)
            for i in range(ob_proj.shape[0]):
                ratio = fix_factor(sqrt(torch.var(restore_proj[i]).item()/torch.var(pre_proj[i]).item()))
                restore_proj[i] /= ratio
            module.data = restore_proj
        elif "classifier" in name or "pooler" in name or "LayerNorm" or "bias" in name:
            module.data = pre_model.state_dict()[name].data
    return model, restore_perm

def ob_arrowcloak(model):
    set_seed()
    layer_permutations = {}
    layer_masks = {}
    layer_factors = {}
    weight_factors = {}
    for name, module in model.named_parameters():
        if "query.weight" in name:
            w_q = module.data
            num_rows = w_q.shape[0]
            device = w_q.device
            coeff = torch.randint(0,5,(w_q.shape[0],), device=device)
            mask = torch.matmul(w_q.T, coeff.float())
            layer_masks[name] = mask
            ratios_q = []
            ratios_q2 = []
            for i in range(num_rows):
                ratio = (torch.randint(0, 11, (1,), device=device)-5).float()
                ratio2 = (torch.randint(1, 3, (1,), device=device)).float()
                w_q[i] *= ratio2
                mask_qi = mask * ratio
                w_q[i] += mask_qi
                ratios_q.append(ratio)
                ratios_q2.append(ratio2)
            layer_factors[name] = ratios_q
            weight_factors[name] = ratios_q2
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            w_q = w_q[permutation]
            module.data = w_q
        elif "key.weight" in name:
            w_k = module.data
            num_rows = w_k.shape[0]
            device = w_k.device
            coeff = torch.randint(0,5,(w_k.shape[0],), device=device)
            mask = torch.matmul(w_k.T, coeff.float())
            layer_masks[name] = mask
            ratios_k = []
            ratios_k2 = []
            for i in range(num_rows):
                ratio = (torch.randint(0, 11, (1,), device=device)-5).float()
                ratio2 = (torch.randint(1, 3, (1,), device=device)).float()
                w_k[i] *= ratio2
                mask_ki = mask * ratio
                w_k[i] += mask_ki
                ratios_k.append(ratio)
                ratios_k2.append(ratio2)
            layer_factors[name] = ratios_k
            weight_factors[name] = ratios_k2
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            w_k = w_k[permutation]
            module.data = w_k
        elif "value.weight" in name:
            w_v = module.data
            num_rows = w_v.shape[0]
            device = w_v.device
            coeff = torch.randint(0,5,(w_v.shape[0],), device=device)
            mask = torch.matmul(w_v.T, coeff.float())
            layer_masks[name] = mask
            ratios_v = []
            ratios_v2 = []
            for i in range(num_rows):
                ratio = (torch.randint(0, 11, (1,), device=device)-5).float()
                ratio2 = (torch.randint(1, 3, (1,), device=device)).float()
                w_v[i] *= ratio2
                mask_vi = mask * ratio
                w_v[i] += mask_vi
                ratios_v.append(ratio)
                ratios_v2.append(ratio2)
            layer_factors[name] = ratios_v
            weight_factors[name] = ratios_v2
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            w_v = w_v[permutation]
            module.data = w_v
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            w_proj = module.data
            num_rows = w_proj.shape[0]
            device = w_proj.device
            coeff = torch.randint(0,5,(w_proj.shape[0],), device=device)
            mask = torch.matmul(w_proj.T, coeff.float())
            layer_masks[name] = mask
            ratios = []
            ratios2 = []
            for i in range(num_rows):
                ratio = (torch.randint(0, 11, (1,), device=device)-5).float()
                ratio2 = (torch.randint(1, 3, (1,), device=device)).float()
                w_proj[i] *= ratio2
                mask_i = mask * ratio
                w_proj[i] += mask_i
                ratios.append(ratio)
                ratios2.append(ratio2)
            layer_factors[name] = ratios
            weight_factors[name] = ratios2
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            w_proj = w_proj[permutation]
            module.data = w_proj
    return model, layer_permutations, layer_masks, layer_factors, weight_factors