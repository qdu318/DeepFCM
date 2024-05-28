import torch
from torch.autograd import Variable
import json
from math import radians, cos, sin, asin, sqrt
import numpy as np

config = json.load(open('./config.json', 'r'))

def geo_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, map(float, [lon1, lat1, lon2, lat2]))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

def Z_Score(x):
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / std

def Max_Min(x):
    max = np.max(x)
    min = np.min(x)
    return (x - min) / (max - min)

def Decimal_Scaling(x):
    j = (-1)
    return x / 10 ** j

def unnormalize(x, key):
    mean = config[key + '_mean']
    std = config[key + '_std']
    return x * std + mean

def pad_sequence(sequences, lengths):
    padded = torch.zeros(len(sequences), lengths[0]).float()
    for i, seq in enumerate(sequences):
        seq = torch.Tensor(seq)
        padded[i, :lengths[i]] = seq[:]
    return padded

def to_var(var):
    if torch.is_tensor(var):
        var = Variable(var)
        if torch.cuda.is_available():
            var = var.cuda()
        return var
    if isinstance(var, int) or isinstance(var, float):
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key])
        return var
    if isinstance(var, list):
        var = map(lambda x: to_var(x), var)
        return var

def get_local_seq(full_seq, kernel_size, mean, std):
    seq_len = full_seq.size()[1]

    if torch.cuda.is_available():
        indices = torch.cuda.LongTensor(seq_len)
    else:
        indices = torch.LongTensor(seq_len)

    torch.arange(0, seq_len, out = indices)

    indices = Variable(indices, requires_grad = False)

    first_seq = torch.index_select(full_seq, dim = 1, index = indices[kernel_size - 1:])
    second_seq = torch.index_select(full_seq, dim = 1, index = indices[:-kernel_size + 1])

    local_seq = first_seq - second_seq
    local_seq = (local_seq - mean) / std

    return local_seq

def MAE(pred_dict):
    return torch.mean(torch.abs(torch.squeeze(pred_dict['pred'], 1) - torch.squeeze(pred_dict['label'], 1)))

def RMSE(pred_dict):
    return torch.sqrt(torch.mean(torch.pow(torch.squeeze(pred_dict['pred'], 1) - torch.squeeze(pred_dict['label'], 1), 2)))

def MAPE(pred_dict):
    return torch.mean(torch.abs(torch.squeeze(pred_dict['pred'], 1) - torch.squeeze(pred_dict['label'], 1))
                      / (torch.squeeze(pred_dict['label'], 1) + 1e-5))

def CalConfusionMatrix(confusion_matrix):
    TP, FP, FN, TN, precise, recall, f1_score = 0, 0, 0, 0, 0, 0, 0
    n = confusion_matrix.shape[0]
    fpr, tpr = [], []
    for i in range(n):
        TP = confusion_matrix[i][i]
        FP = (confusion_matrix[i].sum() - TP)
        FN = (confusion_matrix[:, i].sum() - TP)
        TN = (confusion_matrix.sum() - TP - FP - FN)
        if TP != 0:
            precise_temp = Precise(TP, FP)
            precise += precise_temp
            recall_temp = ReCall(TP, FN)
            recall += recall_temp
            f1_score += F1_Score(precise_temp, recall_temp)
            #fpr += FPR(FP, TN)
            #tpr += ReCall(TP, FN)
            #fpr.append(FPR(FP, TN))
            #tpr.append(ReCall(TP, FN))
        else:
            precise += 0.
            recall += 0.
            f1_score += 0.
            #fpr.append(0.)
            #tpr.append(0.)
            #fpr += 0.
            #tpr += 0.
    return precise / n, recall / n, f1_score / n

def Precise(TP, FP):
    return TP / (TP + FP)

def ReCall(TP, FN):
    return TP / (TP + FN)

def FPR(FP, TN):
    return FP / (FP + TN)

def TPR(TP,FN):
    return TP / (TP + FN)

def F1_Score(P, R):
    return 2 * P * R / (P + R)