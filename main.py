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
from dataset import get_dataloader
import argparse
from model import Model
import json
import constant
from metric import Metric


##################################################################
random_seed = 1024
if random_seed:
    print('Random Seed: {}'.format(random_seed))
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

def prepare_dataloader(opt, batch_size):
    """ Load data and prepare dataloader. """

    def load_data(name):
        with open(name, 'rb') as f:
            data = np.load(f, allow_pickle=True).item()
            return data

    train_data = load_data('./Synthetic_Data/' + opt.data + '.npy')
    trainloader = get_dataloader(train_data, batch_size, shuffle=False)
    return trainloader, train_data


original_data = np.load("./Synthetic_Data/dataset_3seqs_20T.npy", allow_pickle=True).item()
print('Original data...............')
print(original_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', type=str, default='dataset_3seqs_20T')
    parser.add_argument('-epoch', type=int, default=3)
    parser.add_argument('-batch_size', type=int, default=3)
    parser.add_argument('-patience', type=int, default=3)

    parser.add_argument('-d_model', type=int, default=8)
    parser.add_argument('-d_rnn', type=int, default=128)
    parser.add_argument('-d_inner_hid', type=int, default=512)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=1)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-4)

    parser.add_argument('-log', type=str, default='log.txt')
    parser.add_argument('-model', type=str, default='Transformer')
    # parser.add_argument('-model', type=str, default='LSTM')
    opt = parser.parse_args()
    opt.log = opt.data + '-' + opt.log
    # default device is CUDA
    opt.device = torch.device('cuda')
    # opt.device = torch.device('cpu')

    opt.mental_predicate_set = constant.mental_predicate_set
    opt.action_predicate_set = constant.action_predicate_set
    opt.head_predicate_set = constant.head_predicate_set
    opt.time_horizon = 10

    opt.lr_base = 1e-3
    opt.lr_A = 1e-3
    opt.lr_LSTM = 1e-3
    opt.lr = 1e-3

    """ prepare dataloader """
    print('Loading data...............')
    trainloader, train_data = prepare_dataloader(opt, opt.batch_size)
    model = Model(opt).to(opt.device)
    metric = Metric()

    model_parameter = model.model_parameter
    
    print("Start time is", datetime.datetime.now(), flush=1)
    with Timer("Total running time") as t:
        print('--------------- start training ---------------')
        start_time = time.time()

        #################### claim learnable parameters ####################
        base_params = list()
        A_params = list()
        for head_predicate_idx in opt.head_predicate_set:
            base_params.append(model_parameter[head_predicate_idx]['base'])
            sub_rule_list = list(model_parameter[head_predicate_idx].keys())
            for sub_rule_idx in sub_rule_list:
                if sub_rule_idx != 'base':
                    A_params.append(model_parameter[head_predicate_idx][sub_rule_idx]['weight_para'])
        base_params_dic = {'params': base_params, 'lr': opt.lr_base}
        A_params_dic = {'params': A_params, 'lr': opt.lr_A}
        # TODO: check whether add a list() outside of self.SamplingMental.parameters() is correct
        LSTM_params_dic = {'params': model.SamplingMental.parameters(), 'lr': opt.lr_LSTM}
        params_dics = [base_params_dic, A_params_dic, LSTM_params_dic]
        ################################################################################
        
        #################### claim the optimizer ####################
        optimizer = optim.SGD(params=params_dics, lr=opt.lr)
        ################################################################################
        
        for i in range(opt.epoch):
            train_loss = []
            print('---------- start the {}-th epoch ----------'.format(i))
            for j, batch_data in enumerate(trainloader):
                
                # print('----- batch_data -----')
                # print(batch_data)
                ##### NOTE: 处理完的'batch_data'是一个list, 
                ##### NOTE: list中第一个元素是batch中每个seq的所有action组成的tensor, 并且按照batch中action event个数最多的那个维度补齐(补-1)
                ##### NOTE: list中第二个元素是batch中每个seq的所有mental组成的tensor, 并且按照batch中mental event个数最多的那个维度补齐(补-1)
                ##### NOTE: list中第三个元素是batch中每个seq的所有原始数据格式的action event组成的tuple
                
                optimizer.zero_grad()
                loss = -model.compute_ELBO(batch_data)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                print('loss = {}'.format(loss))
            train_loss = np.average(train_loss)
            end_time = time.time()
            time_cost = end_time - start_time
            print('---------- the {}-th epoch, using t = {}s , loss = {}----------'.format(i, time_cost, train_loss))
        total_running_time = time.time() - start_time
        print('--------------- finish training, using t = {}s ---------------'.format(total_running_time))
        
        
        """ process the original data so that we can sample mental events via the well trained model. """
        original_data = np.load('./Synthetic_Data/' + opt.data + '.npy', allow_pickle=True).item()
        original_mental_data = []
        for seq_id in list(original_data.keys()):
            original_mental_data.append(original_data[seq_id])
        
        print('----- original_mental_data -----')
        print(original_mental_data)
        
        # after training, we can sample the mental events, they should be accurate
        # 以完整的sequence个数作为batch_size
        alldataloader, all_data = prepare_dataloader(opt, batch_size=len(list(original_data.keys())))
        for j, batch_data in enumerate(alldataloader):
            action_data, mental_data, ori_action = batch_data
            learned_mental_event, hazard_rate_dic = model.SamplingMental.LSTM_sample_mental_event(action_data)
        
        print('----- learned_mental_event -----')
        print(learned_mental_event)
        
        all_err = []
        for seq_id in list(original_data.keys()):
            for mental_predicate_idx in opt.mental_predicate_set:
                err = metric.event_mae_count(original_mental_data[seq_id][mental_predicate_idx]['time'], learned_mental_event[seq_id][mental_predicate_idx]['time'])
                all_err.append(err)
            
        avg_err = sum(all_err) / len(all_err)
        
    print("Exit time is", datetime.datetime.now(), flush=1)
