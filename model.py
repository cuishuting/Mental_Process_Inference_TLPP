import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
from tqdm import *
import itertools
import datetime
import time
import random
from utils import redirect_log_file, Timer
import math
import constant
from SubLayers import EncoderLayer, ScaledDotProductAttention, MultiHeadAttention

def get_non_pad_mask(seq):
    """ Get the non-padding positions. """
    assert seq.dim() == 3 # B * D * L
    return seq.ne(constant.PAD).type(torch.float)

class SamplingMental(nn.Module):
    def __init__(self, opt):
        super(SamplingMental, self).__init__()
        self.num_predicate = len(opt.head_predicate_set)
        self.num_mental_predicate = len(opt.mental_predicate_set)
        self.num_action_predicate = len(opt.action_predicate_set)
        self.mental_predicate_set = opt.mental_predicate_set
        self.action_predicate_set = opt.action_predicate_set
        self.head_predicate_set = opt.head_predicate_set
        self.time_horizon = opt.time_horizon
        self.d_model = opt.d_model
        self.device = opt.device

        self.grid_length = 0.1
        self.sigma = 1

        self.hidden_size = 128
        self.lstm = nn.LSTM(input_size=self.d_model, hidden_size=self.hidden_size).to(self.device)
        self.linear_layer_1 = nn.Linear(self.hidden_size, 5, bias=True)
        self.linear_layer_2 = nn.Linear(5, 1, bias=True)

        self.elu = nn.ELU()

        self.initial_U = 0.01
        self.U_matrix = self.initial_U * torch.ones((self.num_predicate, self.num_predicate), requires_grad=True)

        self.d_k = self.d_model

        # self.attention = ScaledDotProductAttention(np.power(self.d_k, 0.5))
        self.slf_attn = MultiHeadAttention(
            opt.n_head, opt.d_model, opt.d_k, opt.d_v, dropout=opt.dropout, normalize_before=False).to(self.device)

        self.action_emb = nn.Embedding(self.num_action_predicate + 1, self.d_model, padding_idx=0)
        self.k_action_emb = nn.Embedding(self.num_action_predicate + 1, self.d_model, padding_idx=0)
        self.v_action_emb = nn.Embedding(self.num_action_predicate + 1, self.d_model, padding_idx=0)

        self.mental_emb = nn.Embedding(self.num_mental_predicate + 1, self.d_model, padding_idx=0)

        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / self.d_model) for i in range(self.d_model)],
            device=torch.device('cpu')
        )

    def current_time_embedding(self, mental_emb, cur_time):
        ##### 'L'是mental process的维度, 'D'是d_model的值
        L, D = mental_emb.shape

        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)).to(
            self.device)
        my_tensor = (torch.tensor(cur_time)).unsqueeze(-1).to(self.device)
        result = my_tensor / self.position_vec.to(self.device)
        my_tensor = my_tensor.expand(-1, self.d_model // 2)
        result[:, 0::2] = torch.sin(my_tensor * div_term)
        result[:, 1::2] = torch.cos(my_tensor * div_term)
        result = result.unsqueeze(0).expand(L, -1, -1)

        _, L, D = result.shape
        mental_emb = mental_emb.unsqueeze(0).expand(-1, L, -1).to(self.device)
        ##### 这里的query是对离散化后的所有时间点做完embedding之后再加上对mental event type的embedding
        q = result + mental_emb

        return q

    def kv_embedding(self, sample):

        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)).to(self.device)

        my_tensor = sample.unsqueeze(-1).to(self.device)
        result = my_tensor / self.position_vec.to(self.device)
        my_tensor = my_tensor.expand(-1, -1, -1, self.d_model // 2)
        result[:, :, :, 0::2] = torch.sin(my_tensor * div_term)
        result[:, :, :, 1::2] = torch.cos(my_tensor * div_term)

        return result

    def LSTM_sample_mental_event(self, sample):

        my_tensor = (torch.tensor(self.mental_predicate_set)).to(self.device)
        
        
        # print('----- my_tensor -----')
        # print(my_tensor)
        
        
        ##### TODO: 这里如果mental event type是0的话, 那么做完embedding之后仍然全是0
        ##### TODO: 因此我们要么让所有的event type都从1开始而不是从0开始, 要么把现在的对event type的embedding改成one-hot embedding
        mental_emb = self.mental_emb(my_tensor)
            
        # print('----- mental_emb -----')
        # print(mental_emb)
        
        ##### 将时间轴离散化
        num_grids = int(self.time_horizon / self.grid_length)
        all_cur_time = [(grid_idx + 1) * self.grid_length for grid_idx in range(num_grids)]
        
        ##### 在每一个被离散化的时间点上都有一个query, 并且做了embedding
        q_all_mental = self.current_time_embedding(mental_emb, all_cur_time)

        ##### 'non_pad_mask'记录了一个batch的action_event中哪些位置是经过padding之后得到的
        non_pad_mask = get_non_pad_mask(sample)
        ##### 'non_pad_mask'按照d_model的值做reshape
        non_pad_mask = non_pad_mask.unsqueeze(-1).expand(-1, -1, -1, self.d_model).to(self.device)
        
        # print('----- non_pad_mask -----')
        # print(non_pad_mask)       
        
        ##### 对action_data的时间做embedding
        kv_embedding = self.kv_embedding(sample).to(self.device)
        
        ##### 'B'表示一个batch中有多少seq
        ##### 'L'表示一个seq中有多少个action event type
        ##### 'T'表示一个batch中的所有seq的所有action维度中, action event最多的个数
        ##### 'D'表示d_model的值
        B, L, T, D = kv_embedding.shape

        my_tensor = (torch.tensor(self.action_predicate_set)).to(self.device)
        
        k_action_emb = self.k_action_emb(my_tensor)    
        
        v_action_emb = self.v_action_emb(my_tensor)
        k_action_emb = k_action_emb.unsqueeze(0).unsqueeze(2).expand(B, L, T, D).to(self.device)
        v_action_emb = v_action_emb.unsqueeze(0).unsqueeze(2).expand(B, L, T, D).to(self.device)
        k = kv_embedding + k_action_emb
        v = kv_embedding + v_action_emb
        ##### 做完embedding之后的action event乘以mask, 把padding的位置对应的值mask掉
        k = k * non_pad_mask
        v = v * non_pad_mask
        ##### 对所有action event type维度进行加和
        k = torch.sum(k, dim=1).to(self.device)
        v = torch.sum(v, dim=1).to(self.device)

        q = q_all_mental.unsqueeze(0).expand(B, -1, -1, -1).to(self.device)

        ##### output表示sample每个离散时间点上的mental event时要用到的所有action event的weighted combination
        ##### attention表示sample每个离散时间点上的mental event时要用到的所有action event的attention score
        output, attention = self.slf_attn(q, k, v)
        
        ##### 'B'表示一个batch中的seq个数
        ##### 'L'表示将时间轴离散化之后所有的离散时间点的个数
        B, _, L, _ = q.shape

        current_hazard_mental = torch.zeros((B, L)).to(self.device)

        # print('----- current_hazard_mental -----')
        # print(current_hazard_mental)
        
        hazard_rate_dic = {}
        for mental_id in self.mental_predicate_set:
            hazard_rate_dic[mental_id] = []

        for i in self.mental_predicate_set:
            hx, cx = self.lstm(output[:, i, :, :].squeeze())
            ##### 两个linear层加上一个sigmoid non-linear激活函数
            hazard_rate = torch.sigmoid(self.linear_layer_2(self.linear_layer_1(hx)))
            hazard_rate = hazard_rate.squeeze()
            hazard_rate_dic[i] = hazard_rate
            current_hazard_mental = torch.stack((current_hazard_mental, hazard_rate), 1)

        # print('----- current_hazard_mental -----')
        # print(current_hazard_mental)
        
        prob_mental = torch.sum(current_hazard_mental, dim=1)
        
        # print('----- prob_mental -----')
        # print(prob_mental)
        
        time_indexs = torch.bernoulli(prob_mental)

        mental_event = []
        for i in range(B):
            mental_event_dic = {}
            for mental_id in self.mental_predicate_set:
                mental_event_dic[mental_id] = {}
                mental_event_dic[mental_id]['time'] = []
            for j in range(len(time_indexs[i])):
                if time_indexs[i, j] == 1:
                    probs = current_hazard_mental[i, 1:, j]
                    pd = Categorical(probs=probs)
                    index = pd.sample()
                    cur_time = (j + 1) * self.grid_length
                    mental_event_dic[self.mental_predicate_set[index]]['time'].append(cur_time)
                else:
                    pass
            mental_event.append(mental_event_dic)
        
        # print('----- mental_event -----')
        # print(mental_event)
        # print('----- hazard_rate_dic -----')
        # print(hazard_rate_dic)
        # exit()
        
        return mental_event, hazard_rate_dic

 
class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.BEFORE = 'BEFORE'
        self.EQUAL = 'EQUAL'
        self.AFTER = 'AFTER'

        self.num_predicate = len(opt.head_predicate_set)
        self.num_mental_predicate = len(opt.mental_predicate_set)
        self.num_action_predicate = len(opt.action_predicate_set)
        self.mental_predicate_set = opt.mental_predicate_set
        self.action_predicate_set = opt.action_predicate_set
        self.head_predicate_set = opt.head_predicate_set
        self.time_horizon = opt.time_horizon
        self.device = opt.device

        self.T_max = self.time_horizon
        self.S_for_sampling = 10

        self.SamplingMental = SamplingMental(opt)

        # Here the logic rule set is given as prior knowledge
        self.initial_mu = 0.1
        self.initial_a_0 = 0.1
        self.initial_a_1 = 0.05

        self.logic_template = self.logic_rule()

        self.model_parameter = {}
        self.initialize_model_parameter(self.initial_mu, self.initial_a_0,
                                        self.initial_a_1)  # initialize self.model_parameter

        self.time_tolerance = 0.1
        self.integral_resolution = 0.03
        self.decay_rate = 1

    def logic_rule(self):
        '''
        This function encodes the content of logic rules
        logic_template = {0:{},1:{},...,6:{}}
        '''

        '''
        Only head predicates may have negative values, because we only record each body predicate's boosted time 
        (states of body predicates are always be 1) 
        body predicates must happen before head predicate in the same logic rule
        '''

        logic_template = {}

        '''
        Mental predicate: [0]
        '''

        '''
        Action predicates: [1]
        '''

        head_predicate_idx = 1
        logic_template[head_predicate_idx] = {}

        # NOTE: rule content: 0 and before(0, 1) to 1
        formula_idx = 0
        logic_template[head_predicate_idx][formula_idx] = {}
        logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0]
        logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]
        logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0, 1]]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [self.BEFORE]

        return logic_template

    def initialize_model_parameter(self, initial_mu, initial_a_0, initial_a_1):

        ##### head predicate index: 1
        head_predicate_idx = 1
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] = torch.tensor([initial_mu], dtype=torch.float64,
                                                                        requires_grad=True)
        formula_idx = 0
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight_para'] = torch.tensor([initial_a_0, initial_a_1],
                                                                                            dtype=torch.float64,
                                                                                            requires_grad=True)
        # self.model_parameter[head_predicate_idx][formula_idx]['weight_para'] = torch.tensor([initial_a_0], dtype=torch.float64, requires_grad=True)

    def get_formula_effect(self, template):
        if template['head_predicate_sign'][0] == 1:
            formula_effect = 1
        else:
            formula_effect = -1
        return formula_effect

    ##### Here we need preprocess the history data, since the mental event is sampled
    def get_feature(self, cur_time, head_predicate_idx, history, template):
        occur_time_dic = {}
        feature = 0
        for idx, body_predicate_idx in enumerate(template['body_predicate_idx']):
            occur_time = np.array(history[body_predicate_idx]['time'])
            mask = (occur_time <= cur_time)  # find corresponding history
            # mask: filter all the history time points that satisfies the conditions which will boost the head predicate
            occur_time_dic[body_predicate_idx] = occur_time[mask]
        occur_time_dic[head_predicate_idx] = [cur_time]
        ### get weight
        # compute features whenever any item of the transition_item_dic is nonempty
        history_transition_len = [len(i) for i in occur_time_dic.values()]
        if min(history_transition_len) > 0:
            # need to compute feature using logic rules
            time_combination = np.array(list(itertools.product(*occur_time_dic.values())))
            # get all possible time combinations
            time_combination_dic = {}
            for i, idx in enumerate(list(occur_time_dic.keys())):
                time_combination_dic[idx] = time_combination[:, i]
            temporal_kernel = np.ones(len(time_combination))
            for idx, temporal_relation_idx in enumerate(template['temporal_relation_idx']):
                time_difference = time_combination_dic[temporal_relation_idx[0]] - time_combination_dic[
                    temporal_relation_idx[1]]
                if template['temporal_relation_type'][idx] == 'BEFORE':
                    temporal_kernel *= (time_difference < -self.time_tolerance) * np.exp(
                        -self.decay_rate * (cur_time - time_combination_dic[temporal_relation_idx[0]]))
                if template['temporal_relation_type'][idx] == 'EQUAL':
                    temporal_kernel *= (abs(time_difference) <= self.time_tolerance) * np.exp(
                        -self.decay_rate * (cur_time - time_combination_dic[temporal_relation_idx[0]]))
                if template['temporal_relation_type'][idx] == 'AFTER':
                    temporal_kernel *= (time_difference > self.time_tolerance) * np.exp(
                        -self.decay_rate * (cur_time - time_combination_dic[temporal_relation_idx[1]]))
            feature = np.sum(temporal_kernel)
        return feature

    def intensity(self, cur_time, head_predicate_idx, history):
        feature_formula = []
        weight_formula = []
        effect_formula = []
        # print("cur head: ", head_predicate_idx)
        for formula_idx in list(self.logic_template[head_predicate_idx].keys()):
            cur_weight = torch.matmul(self.model_parameter[head_predicate_idx][formula_idx]['weight_para'],
                                      torch.tensor([1, cur_time], dtype=torch.float64))
            # cur_weight = torch.matmul(self.model_parameter[head_predicate_idx][formula_idx]['weight_para'], torch.tensor([1], dtype=torch.float64))

            weight_formula.append(cur_weight.reshape(-1))
            cur_feature = self.get_feature(cur_time, head_predicate_idx, history,
                                           self.logic_template[head_predicate_idx][formula_idx])
            # print("feature: ", cur_feature)
            feature_formula.append(torch.tensor([cur_feature], dtype=torch.float64))
            cur_effect = self.get_formula_effect(self.logic_template[head_predicate_idx][formula_idx])
            effect_formula.append(torch.tensor([cur_effect], dtype=torch.float64))

        intensity = torch.cat(weight_formula, dim=0) * torch.cat(feature_formula, dim=0) * torch.cat(effect_formula,
                                                                                                     dim=0)
        intensity = self.model_parameter[head_predicate_idx]['base'] + torch.sum(intensity)
        # todo: below method getting intensity works but why?
        if intensity.item() >= 0:
            if intensity.item() >= self.model_parameter[head_predicate_idx]['base'].item():
                final_intensity = intensity
            else:
                final_intensity = self.model_parameter[head_predicate_idx]['base']
            # final_intensity = torch.max(torch.tensor([intensity.item(), self.model_parameter[head_predicate_idx]['base']]))
            return final_intensity
        else:
            return torch.exp(intensity)

    #################### First term of ELBO -- related to the log-likelihood
    ##### Here we need preprocess the data set, since the mental event is sampled
    def intensity_log_sum(self, head_predicate_idx, data_sample):
        intensity_transition = []
        for t in data_sample[head_predicate_idx]['time'][1:]:
            # NOTE: compute the intensity at transition times
            cur_intensity = self.intensity(t, head_predicate_idx, data_sample)
            intensity_transition.append(cur_intensity)

        if len(intensity_transition) == 0:  # only survival term, no event happens
            log_sum = torch.tensor([0], dtype=torch.float64)
        else:
            log_sum = torch.sum(torch.log(torch.cat(intensity_transition, dim=0)))
        return log_sum

    def intensity_integral(self, head_predicate_idx, data_sample, T_max):
        start_time = 0
        end_time = T_max
        intensity_grid = []
        for t in np.arange(start_time, end_time, self.integral_resolution):
            # NOTE: evaluate the intensity values at the chosen time points
            cur_intensity = self.intensity(t, head_predicate_idx, data_sample)
            intensity_grid.append(cur_intensity)
        # NOTE: approximately calculate the integral
        integral = torch.sum(torch.cat(intensity_grid, dim=0) * self.integral_resolution)
        return integral

    def log_likelihood(self, new_data_sample, T_max):
        '''
        This function calculates the log-likehood given the new data sample
        log-likelihood = \sum log(intensity(transition_time)) + int_0^T intensity dt
        '''
        # multiple samples then take average
        log_likelihood = torch.tensor([0], dtype=torch.float64)
        # iterate over head predicates; each predicate corresponds to one intensity
        for head_predicate_idx in self.head_predicate_set:
            # NOTE: compute the summation of log intensities at the transition times
            intensity_log_sum = self.intensity_log_sum(head_predicate_idx, new_data_sample)
            # NOTE: compute the integration of intensity function over the time horizon
            intensity_integral = self.intensity_integral(head_predicate_idx, new_data_sample, T_max)
            log_likelihood += (intensity_log_sum - intensity_integral)

        return log_likelihood

    #################### Second term of ELBO -- related to the approximated posterior, which can be converted to the entropy
    def compute_entropy(self, hazard_dic_sample):

        B, L = hazard_dic_sample[self.mental_predicate_set[0]].shape

        entropy_all_mental = torch.zeros((B, L)).to(self.device)
        hazard_all_mental = torch.zeros((B, L)).to(self.device)
        for mental_id in self.mental_predicate_set:
            hazard_all_mental += hazard_dic_sample[mental_id]
            entropy = (-hazard_dic_sample[mental_id]) * torch.log(hazard_dic_sample[mental_id])
            entropy_all_mental += entropy
        entropy_no_mental = (-(1 - hazard_all_mental)) * torch.log(1 - hazard_all_mental)
        entire_entropy = entropy_all_mental + entropy_no_mental

        entire_entropy = torch.sum(entire_entropy)
        return entire_entropy

    #################### ELBO
    def compute_ELBO(self, batch_data):

        multisample_log_likelihood = []
        multisample_entropy = []
        ##### NOTE: 'batch_data'是一个list, 
        ##### NOTE: list中第一个元素是batch中每个seq的所有action组成的tensor, 并且按照batch中action event个数最多的那个维度补齐(补-1)
        ##### NOTE: list中第二个元素是batch中每个seq的所有mental组成的tensor, 并且按照batch中mental event个数最多的那个维度补齐(补-1)
        ##### NOTE: list中第三个元素是batch中每个seq的所有原始数据格式的action event组成的tuple
        action_data, mental_data, ori_action = batch_data
        
        ##### 'B' 是指batch size
        B, _, _ = action_data.shape
    
        ##### 这里我们根据action_data利用LSTM去sample mental_event
        mental_event, hazard_rate = self.SamplingMental.LSTM_sample_mental_event(action_data)

        # print('----- mental_event -----')
        # print(mental_event)
        # print('----- hazard_rate -----')
        # print(hazard_rate)
        
        entire_entropy = self.compute_entropy(hazard_rate)
        multisample_entropy.append(entire_entropy)

        for i in range(B):
            new_data_sample = {**mental_event[i], **ori_action[i]}
            log_likelihood = self.log_likelihood(new_data_sample, self.T_max)
            multisample_log_likelihood.append(log_likelihood)

        first_term = sum(multisample_log_likelihood)
        second_term = sum(multisample_entropy)
        ELBO = first_term.to(self.device) + second_term.to(self.device)

        return ELBO
     
    #################### Compare the original mental events and the learned mental events

    
##### TODO: 似乎没有在每一个mental event发生时, starting point从0开始重新计算
##### TODO: 似乎不需要starting point从0开始重新计算, 因为如果loss function设计的准确, LSTM应该能够准确的拟合hazard rate的变化趋势
##### TODO: 包括在每一个mental event发生时, hazard rate变成0的趋势

