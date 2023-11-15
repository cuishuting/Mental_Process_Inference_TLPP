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
##################################################################
random_seed = 1024
if random_seed:
    print('Random Seed: {}'.format(random_seed))
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)


class ScaledDotProductAttention(nn.Module):
    '''
    Scaled Dot-Product Attention
    '''
    def __init__(self, scale):
        super(ScaledDotProductAttention, self).__init__()
        
        self.scale = scale
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, q, k, v, mask=None):
        # 1. Matmul
        u = torch.bmm(q, k.transpose(1, 2))

        # 2. Scale
        u = u / self.scale

        if mask is not None:
            # 3. Mask
            u = u.masked_fill(mask, -np.inf) 
            
        # 4. Softmax
        attn = self.softmax(u)
        
        # 5. Output
        output = torch.bmm(attn, v)
        
        return attn, output


class SamplingMental(nn.Module):
    def __init__(self):
        super(SamplingMental, self).__init__()
        self.num_predicate = 3 
        self.num_mental_predicate = 1
        self.num_action_predicate = 2
        self.mental_predicate_set = [0]
        self.action_predicate_set = [1, 2]
        self.head_predicate_set = [0, 1, 2]

        self.time_horizon = 10
        
        ##### action embedding
        self.d_model = 5  # d_model is the number of model
        
        ##### discretize the time horizon
        self.grid_length = 0.1
        self.sigma = 1
        
        ##### LSTM for sampling mental event. Note we should use LSTMCell
        # since we consider attention action embedding, current mental embedding, and current time embedding as input
        self.input_size = (self.d_model + self.num_predicate) * 2 + self.d_model 
        self.hidden_size = 128
        self.lstm_cell = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size)
        
        ##### a fully-connect layer that outputs a distribution over the hazard rate, 
        ##### given the weighted combination of action process and LSTM output
        self.linear_layer_1 = nn.Linear(self.hidden_size, 5, bias=True)
        self.linear_layer_2 = nn.Linear(5, 1, bias=True)
        ##### a nonlinear activation function
        self.elu = nn.ELU()
        
        ##### learnable matrix for event type embedding
        self.initial_U = 0.01
        # todo: change from cst: since Variable is deprecated in PyTorch
        self.U_matrix = self.initial_U * torch.ones((self.num_predicate, self.num_predicate), requires_grad=True)
        # todo original:
        # self.U_matrix = Variable(self.initial_U * torch.ones((self.num_predicate, self.num_predicate)), requires_grad=True)
        # self.U_matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        ##### attention
        self.d_k = self.d_model + self.num_predicate
        self.attention = ScaledDotProductAttention(scale=np.power(self.d_k, 0.5))
        
    def event_type_embedding(self):
        # todo: confuse: what if head_predicate_set doesn't include all predicates?
        type_embedding = {}
        # todo: check whether the autograd will affect the one-hot encoded K??
        K = F.one_hot(torch.arange(0, self.num_predicate)) # one-hot encoding for event type
        # todo: change from cst
        for idx, head_predicate_id in enumerate(self.head_predicate_set):
            type_embedding[head_predicate_id] = K[idx]
        # for predicate_id in self.head_predicate_set:
        #     corresponding_index = self.head_predicate_set.index(predicate_id)
        #     ##### (1) one-hot type embedding
        #     type_embedding[predicate_id] = K[corresponding_index]
        #     ##### (2) more flexible type embedding
        #     # type_embedding[predicate_id] = (self.U_matrix * K[predicate_id])[:, predicate_id]
        return type_embedding

    ##### embedding of discrete time for specific mental dimension
    def current_time_embedding(self, type_embedding, cur_time):
        q_all_mental = {}       
        for mental_id in self.mental_predicate_set:
            q = torch.zeros(1, 1, self.d_model)
            # concat mental dimension specific type embedding
            print(type_embedding[mental_id].shape)
            print(type_embedding[mental_id])
            print(type_embedding[mental_id].repeat(1, 1).shape)
            print(type_embedding[mental_id].repeat(1, 1))
            q = torch.cat((q, type_embedding[mental_id].repeat(1, 1).unsqueeze(0)), 2)
            for i in range(self.d_model):
                if i % 2 == 0:
                    q[0, 0, i] = torch.sin(torch.tensor(cur_time / 10000 ** (2 * i / self.d_model))) 
                else:
                    # todo: check for odd positions: i, cos(time/1e4**(2*(i-1)/d_model))?
                    q[0, 0, i] = torch.cos(torch.tensor(cur_time / 10000 ** (2 * i / self.d_model)))
            print(q.shape)
            q_all_mental[mental_id] = q
        print('q_all_mental', q_all_mental)
        return q_all_mental
    
    ##### embedding for action event 
    def action_event_time_embedding(self, sample):  
        action_time_embedding = {} # NOTE: the first event of the action process occurs at 0, this might cause potential numerical bug
        for action_id in self.action_predicate_set:
            seq = sample[action_id]['time']
            embedding = np.zeros((len(seq), self.d_model))
            for pos_id in range(len(seq)):
                pos = seq[pos_id]
                for i in range(self.d_model):
                    if i % 2 == 0:
                        embedding[pos_id, i] = np.sin(pos / 10000 ** (2 * i / self.d_model))
                    else:
                        # todo: check for odd positions: i, cos(time/1e4**(2*(i-1)/d_model))?
                        embedding[pos_id, i] = np.cos(pos / 10000 ** (2 * i / self.d_model))
            action_time_embedding[action_id] = torch.tensor(embedding, dtype=torch.float32)
        return action_time_embedding

    def action_event_embedding(self, sample, type_embedding):
        # todo: simplify
        action_time_embedding = self.action_event_time_embedding(sample)
        action_embedding = {}
        for action_id in self.action_predicate_set:
            ##### concatenate: event embedding = concatenate(time embedding, type embedding)
            action_embedding[action_id] = torch.cat((action_time_embedding[action_id], 
                                                     type_embedding[action_id].repeat(action_time_embedding[action_id].shape[0], 1)), 1)
        
        if len(self.action_predicate_set) == 1: # only one action dimension
            k = action_embedding[self.action_predicate_set[0]]
            v = action_embedding[self.action_predicate_set[0]]
        else: # more than one action dimension
            k = action_embedding[self.action_predicate_set[0]]
            v = action_embedding[self.action_predicate_set[0]]
            for action_id_idx in range(self.num_action_predicate):
                if action_id_idx == 0:
                    continue
                else:
                    k = torch.cat((k, action_embedding[self.action_predicate_set[action_id_idx]]), 0)
                    v = torch.cat((v, action_embedding[self.action_predicate_set[action_id_idx]]), 0)

        # reshape tensor dimension
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        
        return k, v
    
    def fast_time_embedding(self, input, time):
        output = input
        for i in range(self.d_model):
            if i % 2 == 0:
                output[0, 0, i] = torch.sin(torch.tensor(time / 10000 ** (2 * i / self.d_model))) 
            else:
                # todo: check for odd positions: i, cos(time/1e4**(2*(i-1)/d_model))?
                output[0, 0, i] = torch.cos(torch.tensor(time / 10000 ** (2 * i / self.d_model)))
        return output
            
    def LSTM_sample_mental_event(self, sample):
        print("LSTM_sample_mental_event")
        print('sample', sample)
        type_embedding = self.event_type_embedding()
        
        ##### initialize the hidden state and cell state
        ##### each mental dimension has a LSTM cell but shares parameters
        # todo: generalize batch_size to a param: batch_size not 1
        hx = torch.zeros(1, self.hidden_size).float()  # batch_size = 1
        cx = torch.zeros(1, self.hidden_size).float()
        
        mental_event_dic = {}
        hazard_rate_dic = {}
        dynamic_time_dic = {}
        for mental_id in self.mental_predicate_set:
            hazard_rate_dic[mental_id] = []
            mental_event_dic[mental_id] = {}
            mental_event_dic[mental_id]['time'] = [0]
            dynamic_time_dic[mental_id] = 0
        
        num_grids = int(self.time_horizon/self.grid_length)
        
        last_mental_event = None
        for grid_idx in range(num_grids):
            cur_time = (grid_idx + 1) * self.grid_length
            print('cur_time', cur_time)
            q_all_mental = self.current_time_embedding(type_embedding=type_embedding, cur_time=cur_time)
            
            # NOTE: compute current mental embedding 
            if last_mental_event == None: # no mental events have been sampled so far
                mental_embedding = torch.zeros(1, 1, self.d_model+self.num_predicate)
            else: # use the lastest mental event embedding
                mental_embedding = torch.zeros(1, 1, self.d_model)
                # concat mental dimension specific type embedding
                mental_embedding = torch.cat((mental_embedding, 
                                              type_embedding[last_mental_event['mental_id']].repeat(1, 1).unsqueeze(0)), 2)
                mental_embedding = self.fast_time_embedding(input=mental_embedding, time=last_mental_event['time'])
                
            current_hazard_mental = torch.tensor([])
            for mental_id in self.mental_predicate_set:
                # update dynamic time dictionary
                dynamic_time_dic[mental_id] = dynamic_time_dic[mental_id] + self.grid_length
                
                q = q_all_mental[mental_id]
                print(q.shape)
                k, v = self.action_event_embedding(sample=sample, type_embedding=type_embedding)
                print(k.shape)

                _, attention_acton_embedding = self.attention(q, k, v, mask=None)
                print(attention_acton_embedding.shape)
                # NOTE: compute dynamic current time embedding
                cur_time_embedding = torch.zeros(1, 1, self.d_model)
                cur_time_embedding = self.fast_time_embedding(input=cur_time_embedding, time=dynamic_time_dic[mental_id])
                print(cur_time_embedding.shape)
                print(mental_embedding.shape)
                # NOTE: concatenate for LSTM input: input = concat(action_embedding, cur_mental_embedding, dynamic_time_embedding)
                input = torch.cat((attention_acton_embedding, mental_embedding, cur_time_embedding), 2).squeeze(0)
                print(input.shape)
                hx, cx = self.lstm_cell(input, (hx, cx))
                ##### map hidden state to hazard rate
                hazard_rate = torch.sigmoid(self.linear_layer_2(self.linear_layer_1(hx)))
                print(hazard_rate.shape)
                hazard_rate_dic[mental_id].append(hazard_rate.squeeze())
                print(hazard_rate.squeeze(0).shape)
                current_hazard_mental = torch.cat((current_hazard_mental, hazard_rate.squeeze(0)), 0)
            print("********************************")
            print(current_hazard_mental.shape)
            ##### sample according to hazard rate
            prob_mental = sum(current_hazard_mental)
            s = torch.bernoulli(prob_mental)
            if s.item() == 0: # no mental event occurs
                continue
            else: # a mental event occurs, then we assign it to one mental dimension
                # torch.Categorical would normalize the given probabilities
                print(current_hazard_mental.shape)

                pd = Categorical(probs=current_hazard_mental)
                index = pd.sample()
                print(index)

                mental_event_dic[self.mental_predicate_set[index]]['time'].append(cur_time)
                
                last_mental_event = {}
                last_mental_event['mental_id'] = self.mental_predicate_set[index]
                last_mental_event['time'] = cur_time
                
                # the survival process repeat, the time would start from 0
                dynamic_time_dic[self.mental_predicate_set[index]] = 0
        print("last")
        print(mental_event_dic)
        print(hazard_rate_dic)
        return mental_event_dic, hazard_rate_dic    


class Learning(nn.Module):
    def __init__(self):
        super(Learning, self).__init__()
        self.BEFORE = 'BEFORE'
        self.EQUAL = 'EQUAL'
        self.AFTER = 'AFTER' 
        self.num_predicate = 3 
        self.num_mental_predicate = 1
        self.num_action_predicate = 2
        self.mental_predicate_set = [0]
        self.action_predicate_set = [1, 2]
        self.head_predicate_set = [0, 1, 2]
        self.time_horizon = 10    
        
        self.SamplingMental = SamplingMental()
        
        # Here the logic rule set is given as prior knowledge
        self.initial_mu = 0.1
        self.initial_a_0 = 0.1
        self.initial_a_1 = 0.05
        self.logic_template = self.logic_rule()
        self.model_parameter = {}
        self.initialize_model_parameter(self.initial_mu, self.initial_a_0, self.initial_a_1) # initialize self.model_parameter
    
        self.time_tolerance = 0.1
        self.integral_resolution = 0.03
        self.decay_rate = 1
        
        self.T_max = self.time_horizon
        
        self.S_for_sampling = 10
        
        self.batch_size = 10
        
        self.num_epoch = 1
        
        self.start_time = time.time()
        
        self.lr_base = 1e-3
        self.lr_A = 1e-3
        self.lr_LSTM = 1e-3
        self.lr = 1e-3
         
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

        head_predicate_idx = 0
        logic_template[head_predicate_idx] = {}

        # NOTE: rule content: 1 and before(1, 0) to 0
        formula_idx = 0
        logic_template[head_predicate_idx][formula_idx] = {}
        logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [1]
        logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]  # use 1 to indicate True; use 0 to indicate False
        logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[1, 0]]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [self.BEFORE]

        '''
        Action predicates: [1, 2]
        '''

        head_predicate_idx = 1
        logic_template[head_predicate_idx] = {}

        # TODO "2 to 1" & "1 and 0 and before(1,0) to \neg 2" contradict ?
        # NOTE: rule content: 2 and before(2, 1) to 1
        formula_idx = 0
        logic_template[head_predicate_idx][formula_idx] = {}
        logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [2]
        logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]
        logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[2, 1]]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [self.BEFORE]

        head_predicate_idx = 2
        logic_template[head_predicate_idx] = {}

        # NOTE: rule content: 0 and before(0,2) to 2
        formula_idx = 0
        logic_template[head_predicate_idx][formula_idx] = {}
        logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0]
        logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]
        logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0, 2]]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [self.BEFORE]

        return logic_template

    def initialize_model_parameter(self, initial_mu, initial_a_0, initial_a_1):
        ##### head predicate index: 0
        head_predicate_idx = 0
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] = torch.tensor([initial_mu], dtype=torch.float64, requires_grad=True) 
        formula_idx = 0
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight_para'] = torch.tensor([initial_a_0, initial_a_1], dtype=torch.float64, requires_grad=True)
        # self.model_parameter[head_predicate_idx][formula_idx]['weight_para'] = torch.tensor([initial_a_0], dtype=torch.float64, requires_grad=True)
        
        ##### head predicate index: 1
        head_predicate_idx = 1
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] = torch.tensor([initial_mu], dtype=torch.float64, requires_grad=True)
        formula_idx = 0
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight_para'] = torch.tensor([initial_a_0, initial_a_1], dtype=torch.float64, requires_grad=True)
        # self.model_parameter[head_predicate_idx][formula_idx]['weight_para'] = torch.tensor([initial_a_0], dtype=torch.float64, requires_grad=True)
        
        ##### head predicate index: 2
        head_predicate_idx = 2
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] = torch.tensor([initial_mu], dtype=torch.float64, requires_grad=True)
        formula_idx = 0
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight_para'] = torch.tensor([initial_a_0, initial_a_1], dtype=torch.float64, requires_grad=True)
        # self.model_parameter[head_predicate_idx][formula_idx]['weight_para'] = torch.tensor([initial_a_0], dtype=torch.float64, requires_grad=True)
    
    def sampling_using_LSTM(self, sample):
        new_sample = {}
        
        mental_event_dic, hazard_rate_dic = self.SamplingMental.LSTM_sample_mental_event(sample)
        
        ##### the mental events are sampled
        for mental_id in self.mental_predicate_set:
            new_sample[mental_id] = mental_event_dic[mental_id]
        ##### the action events are given
        for action_id in self.action_predicate_set:
            new_sample[action_id] = sample[action_id]
         
        return new_sample, hazard_rate_dic
     
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
            occur_time = np.array(history[body_predicate_idx]['time'][1:])
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
                time_difference = time_combination_dic[temporal_relation_idx[0]] - time_combination_dic[temporal_relation_idx[1]]
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
            
            cur_weight = torch.matmul(self.model_parameter[head_predicate_idx][formula_idx]['weight_para'], torch.tensor([1, cur_time], dtype=torch.float64))
            # cur_weight = torch.matmul(self.model_parameter[head_predicate_idx][formula_idx]['weight_para'], torch.tensor([1], dtype=torch.float64))
            
            weight_formula.append(cur_weight.reshape(-1))
            cur_feature = self.get_feature(cur_time, head_predicate_idx, history, self.logic_template[head_predicate_idx][formula_idx])
            # print("feature: ", cur_feature)
            feature_formula.append(torch.tensor([cur_feature], dtype=torch.float64))
            cur_effect = self.get_formula_effect(self.logic_template[head_predicate_idx][formula_idx])
            effect_formula.append(torch.tensor([cur_effect], dtype=torch.float64))
     
        intensity = torch.cat(weight_formula, dim=0) * torch.cat(feature_formula, dim=0) * torch.cat(effect_formula, dim=0)
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
        # todo: change to be more flexible
        num_grid = len(hazard_dic_sample[self.mental_predicate_set[0]])
        # todo: original
        # num_grid = len(hazard_dic_sample[0])
        
        entire_entropy = torch.tensor([0], dtype=torch.float64)
        # iterate over time grids
        for grid_id in range(num_grid):
            # iterate over mental dimensions

            entropy_all_mental = torch.tensor([0], dtype=torch.float64)
            hazard_all_mental = torch.tensor([0], dtype=torch.float64)
            for mental_id in self.mental_predicate_set:
                hazard_all_mental += hazard_dic_sample[mental_id][grid_id]
                entropy = (-hazard_dic_sample[mental_id][grid_id]) * torch.log(hazard_dic_sample[mental_id][grid_id])
                entropy_all_mental += entropy
            # count for no mental event occurring
            entropy_no_mental = (-(1 - hazard_all_mental)) * torch.log(1 - hazard_all_mental) 
            
            entire_entropy = entire_entropy + entropy_all_mental + entropy_no_mental
        
        return entire_entropy
            
    #################### ELBO
    def compute_ELBO(self, dataset, sample_ID_batch):
        
        first_term = torch.tensor([0], dtype=torch.float64)
        second_term = torch.tensor([0], dtype=torch.float64)
        
        # iterate over samples
        for sample_ID in sample_ID_batch:
            data_sample = dataset[sample_ID]
            multisample_log_likelihood = torch.tensor([0], dtype=torch.float64)
            multisample_entropy = torch.tensor([0], dtype=torch.float64)
            ## multiple sampling then compute the mean
            for s in range(self.S_for_sampling):  # self.S_for_sampling = 10
                ##### compute the first term of ELBO
                # sample the mental events
                new_data_sample, hazard_dic_sample = self.sampling_using_LSTM(sample=data_sample)

                log_likelihood = self.log_likelihood(new_data_sample, self.T_max)

                multisample_log_likelihood = multisample_log_likelihood + log_likelihood
                
                ##### compute the second term of ELBO
                entire_entropy = self.compute_entropy(hazard_dic_sample=hazard_dic_sample)
                multisample_entropy = multisample_entropy + entire_entropy
                
            first_term += 1/self.S_for_sampling * multisample_log_likelihood
            second_term += 1/self.S_for_sampling * multisample_entropy

        # todo: check whether an outside average of ELBO on num of sample_ID_batch is needed
        ELBO = first_term + second_term
        return ELBO
            
    #################### Training
    def optimize_ELBO(self, dataset):
        print('--------------- start training ---------------')
        start_time = time.time()
        
        # declaim the parameters
        base_params = list()
        A_params = list()
        for head_predicate_idx in self.head_predicate_set:
            base_params.append(self.model_parameter[head_predicate_idx]['base'])
            sub_rule_list = list(self.model_parameter[head_predicate_idx].keys())
            for sub_rule_idx in sub_rule_list:
                if sub_rule_idx != 'base':
                    A_params.append(self.model_parameter[head_predicate_idx][sub_rule_idx]['weight_para']) 



        base_params_dic = {'params': base_params, 'lr': self.lr_base}
        A_params_dic = {'params': A_params, 'lr': self.lr_A}
        # todo: check whether add a list() outside of self.SamplingMental.parameters() is correct
        LSTM_params_dic = {'params': self.SamplingMental.parameters(), 'lr': self.lr_LSTM}


        params_dics = [base_params_dic, A_params_dic, LSTM_params_dic]
        
        optimizer = optim.SGD(params=params_dics, lr=self.lr)
        
        # divide the data into several mini-batches
        #todo change:
        batch_num = len(dataset) // self.batch_size
        # todo: original:
        # batch_num = len(dataset.keys()) // self.batch_size

        for i in range(self.num_epoch):
            print('---------- start the {}-th epoch ----------'.format(i))
            sample_ID_list = list(dataset.keys())

            random.shuffle(sample_ID_list) # get the random data order
            for batch_idx in range(batch_num):
                sample_ID_batch = sample_ID_list[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]

                loss = -self.compute_ELBO(dataset, sample_ID_batch=sample_ID_batch)

                optimizer.zero_grad()
                
                # print('========== parameters of LSTM: before updating ==========')
                # for name, parms in self.SamplingMental.named_parameters():
                #     print('--> name:', name)
                #     print('--> para:', parms)
                #     print('--> grad_requirs:', parms.requires_grad)
                #     print('--> grad_value:', parms.grad)
                #     print("===")

                loss.backward()
                optimizer.step()
                
                # print('========== parameters of LSTM: after updating ==========')
                # for name, parms in self.SamplingMental.named_parameters():
                #     print('--> name:', name)
                #     print('--> para:', parms)
                #     print('--> grad_requirs:', parms.requires_grad)
                #     print('--> grad_value:', parms.grad)
                #     print("===")
                    
                t = time.time() - self.start_time
                print('loss = {}, t = {}s'.format(loss, t))

            end_time = time.time()
            time_cost = end_time - start_time
            # print('----- model_parameter -----')
            # print(self.model_parameter)
            print('---------- the {}-th epoch, using t = {}s ----------'.format(i, time_cost))
        
        total_running_time = time.time() - self.start_time
        print('--------------- finish training, using t = {}s ---------------'.format(total_running_time))
        
        # after training, we can sample the mental events, they should be accurate
        new_dataset = {}
        for sample_id in list(data.keys()):
            new_sample, _ = self.sampling_using_LSTM(sample=data[sample_id])
            new_dataset[sample_id] = new_sample
        
        return new_dataset, self.model_parameter
            
## load synthetic dataset
data = np.load("./Synthetic_Data/dataset_100seqs_changing_weights.npy", allow_pickle=True).item()

print("Start time is", datetime.datetime.now(), flush=1)
with Timer("Total running time") as t:
    # redirect_log_file('0731_10epochs_100seqs.txt')
    sampling_for_mental = SamplingMental()
    learning = Learning()
    new_dataset, model_parameter = learning.optimize_ELBO(dataset=data)
    # print('----- model_parameter -----')
    # print(model_parameter)
    # for sample_id in list(new_dataset.keys()):
    #     print('----- The {}-th sample -----'.format(sample_id))
    #     print(new_dataset[sample_id][0])
print("Exit time is", datetime.datetime.now(), flush=1)
