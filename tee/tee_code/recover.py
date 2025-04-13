from pdb import set_trace as st
import torch
import numpy as np
import torch.nn as nn
import time
import os


cpu_num = 1 
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
os.environ['KMP_BLOCKTIME'] = '1'
os.environ['KMP_SETTINGS'] = '1'
os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'
torch.set_num_threads(cpu_num)

repeated_times = 3
layers = 12
tot_time = 0

def recover_otp(obfus_result, otp):
    """"""
    obfus_result -= otp
    return obfus_result

def recover_tsqp(obfus_result, otp, scale1):
    """"""
    obfus_result -= otp
    obfus_result /= scale1  
    return obfus_result

def recover_shadownet(obfus_result, deshuffle, mask_vector, otp, scale1):
    obfus_result -= otp
    obfus_result = obfus_result[:,:, deshuffle]
    obfus_result -= mask_vector
    obfus_result/= scale1
    return obfus_result

def recover(obfus_result, X, deshuffle, mask_vector, otp, scale1, scale2, W=None):
    """"""
    obfus_result -= otp
    obfus_result = obfus_result[:,:, deshuffle]
    # 计算v
    v = X @ mask_vector   
    # 将 scale1 和 v 相乘的部分进行计算，减少多次操作
    obfus_result -= scale1 * v.unsqueeze(1)
    obfus_result /= scale2
    
    return obfus_result

