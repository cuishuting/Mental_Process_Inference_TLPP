import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))


def log_likelihood_mental(mental_time, mental_type, haz_out):
    haz_out = haz_out.squeeze(-1)
    
    log_likelihood_mental = haz_out * (mental_time > 0) + (1 - haz_out) * (mental_time == 0)
    
    
    log_likelihood_mental = torch.log(log_likelihood_mental)
    log_likelihood_mental = torch.sum(log_likelihood_mental, dim=-1)
    log_likelihood_mental = torch.sum(log_likelihood_mental, dim=-1)
    ##### 最终计算的是一个batch上的mental loglikelihood之和
    return log_likelihood_mental

