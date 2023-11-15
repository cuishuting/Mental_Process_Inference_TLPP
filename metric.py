import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import zlib

class Metric(nn.Module):
    def __init__(self):
        super(Metric, self).__init__()
        
    """ Euclidean Distance """
    def Euclidean_Distance(self, seq1, seq2):
        # 计算序列长度
        len1 = len(seq1)
        len2 = len(seq2)
    
        # 找到较短和较长序列的长度
        min_len = min(len1, len2)
        max_len = max(len1, len2)
    
        # 计算差值的平方和
        squared_diff_sum = 0
        for i in range(min_len):
            squared_diff_sum += (seq1[i] - seq2[i]) ** 2
    
        # 如果序列长度不同，将剩余部分的差值平方和加到总和中
        if len1 != len2:
            if len1 > len2:
                for i in range(min_len, max_len):
                    squared_diff_sum += seq1[i] ** 2
            else:
                for i in range(min_len, max_len):
                    squared_diff_sum += seq2[i] ** 2
    
        # 计算欧几里德距离并返回结果
        euclidean_dist = np.sqrt(squared_diff_sum)
        return round(euclidean_dist, 4)

    """ Dynamic Time Warping (DTW) """
    def DTW_Distance(self, seq1, seq2):
        seq1 = np.array(seq1)
        seq2 = np.array(seq2)
        distance, path = fastdtw(seq1, seq2, dist=euclidean)
        
        return round(distance, 4)

    """ Cosine Similarity """ 
    def Cosine_Similarity(self, seq1, seq2):
        # 计算序列长度
        len1 = len(seq1)
        len2 = len(seq2)
    
        # 找到较短和较长序列的长度
        min_len = min(len1, len2)
        max_len = max(len1, len2)
    
        # 计算点积
        dot_product = np.dot(seq1[:min_len], seq2[:min_len])
    
        # 如果序列长度不同，将剩余部分的平方和加到点积中
        if len1 != len2:
            if len1 > len2:
                dot_product += np.dot(seq1[min_len:], seq1[min_len:])
            else:
                dot_product += np.dot(seq2[min_len:], seq2[min_len:])
    
        # 计算向量的范数
        norm_seq1 = np.linalg.norm(seq1)
        norm_seq2 = np.linalg.norm(seq2)
    
        # 计算余弦相似度并返回结果
        cosine_sim = dot_product / (norm_seq1 * norm_seq2)
        return round(cosine_sim, 4)

    """ LB Keogh """
    def LB_Keogh(self, seq1, seq2):
        distance, _ = fastdtw(seq1, seq2, dist=lambda x, y: np.abs(x - y), radius=1)
        return round(distance, 4)
    
    """ DISSIM """
    def DISSIM(self, ori_seq1, ori_seq2):
        seq1 = []
        seq2 = []
        for item in ori_seq1:
            seq1.append([item])
        for item in ori_seq2:
            seq2.append([item])
            
        len1 = len(seq1)
        len2 = len(seq2)
        
        # 计算欧几里德距离
        def euclidean_distance(x, y):
            return np.sqrt(np.sum(np.square(np.array(x) - np.array(y))))
    
        # 计算DISSIM
        dissim = 0.0
        for i in range(len1):
            min_dist = float('inf')
            for j in range(len2):
                dist = euclidean_distance(seq1[i], seq2[j])
                if dist < min_dist:
                    min_dist = dist
            dissim += min_dist
        
        return round(dissim / len1, 4)    

    """ MAE count """ 
    def Event_MAE_Count(self, seq1, seq2):
        # 计算序列长度
        len1 = len(seq1)
        len2 = len(seq2)
    
        event_mae_count = abs(len1 - len2)
    
        return round(event_mae_count, 4)
    
    """ Weighted Average Recall (WARP) """
    def Compute_WARP(self, seq1, seq2):
        # Determine the lengths of the input sequences
        len1 = len(seq1)
        len2 = len(seq2)

        # Create a cost matrix with dimensions (len1 x len2)
        cost_matrix = np.zeros((len1, len2))

        # Initialize the first row and column of the cost matrix
        cost_matrix[0, 0] = abs(seq1[0] - seq2[0])
        for i in range(1, len1):
            cost_matrix[i, 0] = cost_matrix[i-1, 0] + abs(seq1[i] - seq2[0])
        for j in range(1, len2):
            cost_matrix[0, j] = cost_matrix[0, j-1] + abs(seq1[0] - seq2[j])

        # Compute the rest of the cost matrix
        for i in range(1, len1):
            for j in range(1, len2):
                cost = abs(seq1[i] - seq2[j])
                cost_matrix[i, j] = cost + min(cost_matrix[i-1, j], cost_matrix[i, j-1], cost_matrix[i-1, j-1])

        # Compute the WARP as the average cost per aligned element
        warp = cost_matrix[-1, -1] / (len1 + len2 - 1)

        return round(warp, 4)

    def Compute_TQuest(self, seq1, seq2, threshold):
        # Determine the lengths of the input sequences
        len1 = len(seq1)
        len2 = len(seq2)

        # Initialize the count of threshold queries
        tquest_count = 0

        # Iterate over the shorter sequence using a sliding window
        for i in range(len1):
            # Determine the start and end indices of the window in sequence 2
            start_index = max(0, i - len2 + 1)
            end_index = min(i + 1, len2)

            # Check if any element in the window satisfies the threshold condition
            if any(abs(seq1[i] - seq2[j]) <= threshold for j in range(start_index, end_index)):
                tquest_count += 1

        return round(tquest_count, 3)
    
    """ Compression Dissimilarity (CDM) """
    def Compute_CDM(self, seq1, seq2):
        # Convert sequences to strings
        str1 = ''.join(map(str, seq1))
        str2 = ''.join(map(str, seq2))

        # Compute compressed sizes of the sequences
        compressed_size1 = len(zlib.compress(str1.encode('utf-8')))
        compressed_size2 = len(zlib.compress(str2.encode('utf-8')))

        # Compute the CDM as the absolute difference of compressed sizes
        cdm = abs(compressed_size1 - compressed_size2)

        return cdm
    
if __name__ == '__main__':
    metric = Metric()

    # 示例用法
    seq1 = [1.2, 3.6, 7.8]
    seq2 = [1.5, 4.0, 7.2, 8.9, 10.0]

    # distance = metric.euclidean_distance(seq1, seq2)
    # print('Euclidean Distance:', distance)
    
    # distance = metric.DTW_distance(seq1, seq2)
    # print('DTM Distance:', distance)

    # similarity = metric.cosine_similarity(seq1, seq2)
    # print('Cosine Similarity:', similarity)

    # distance = metric.LB_Keogh(seq1, seq2)
    # print('LB-Keogh distance:', distance)

    # result = metric.DISSIM(seq1, seq2)
    # print('DISSIM:', result)

    # event_mae_count = metric.event_mae_count(seq1, seq2)
    # print('Event MAE Count:', event_mae_count)
    
    warp = metric.Compute_WARP(seq1, seq2)
    print('WARP:', warp)
    
    TQuest = metric.Compute_TQuest(seq1, seq2, threshold=2)
    print('TQuest:', TQuest)
    
    cdm = metric.Compute_CDM(seq1, seq2)
    print('CDM:', cdm)
