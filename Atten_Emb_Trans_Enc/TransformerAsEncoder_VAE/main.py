from Utils import MultiHeadedAttention, PositionwiseFeedForward
from copy import deepcopy as dc
from Data_Gen import Logic_Model_Generator
from Get_DataLoader import get_dataloader
from Embedding import Embeddings
import numpy as np
import argparse
import torch
import os


parser = argparse.ArgumentParser()
parser.add_argument('-action_type_list', type=list, nargs='+')
parser.add_argument('-mental_type_list', type=list, nargs='+')
parser.add_argument('-time_tolerance', type=float)
parser.add_argument('-decay_rate', type=float)
parser.add_argument('-time_horizon', type=int)
parser.add_argument('-num_sample', type=int)
parser.add_argument('-sep_for_grids', type=float)
parser.add_argument('-sep_for_data_syn', type=float)
parser.add_argument('-d_emb', type=int)
parser.add_argument('-d_h', type=int)  # num of heads in MultiheadAttn
parser.add_argument('-d_k', type=int)
parser.add_argument('-d_v', type=int)
parser.add_argument('-d_hid', type=int)
parser.add_argument('-dropout', type=float)
parser.add_argument('-batch_size', type=int)
parser.add_argument('-lr', type=float)
parser.add_argument('-num_iter', type=int)
param = parser.parse_args()

param.device = torch.device('cuda')
action_type_list = list(map(int, param.action_type_list[0]))  # cur: [1, 2]
mental_type_list = list(map(int, param.mental_type_list[0]))  # cur: [0]
num_grids = int(param.time_horizon / param.sep_for_grids)


attn = MultiHeadedAttention(h=param.d_h, d_model=param.d_emb)
ff = PositionwiseFeedForward(d_model=param.d_emb, d_ff=param.d_hid)

"""Generate train data"""
org_train_data_file_path = "./Synthetic_Data/test_for_train_data_gen.npy"
if os.path.exists(org_train_data_file_path):
    org_train_data = np.load(org_train_data_file_path, allow_pickle=True).item()
else:
    data_generator = Logic_Model_Generator(param.time_tolerance, param.decay_rate, param.sep_for_data_syn, param.sep_for_grids)
    org_train_data = data_generator.generate_data(param.num_sample, param.time_horizon)
    np.save(org_train_data_file_path, org_train_data)

train_dataloader = get_dataloader(org_train_data, action_type_list, mental_type_list, param.batch_size)
"""test for input embedding first"""
input_embedding_layer = Embeddings(param.d_emb, len(action_type_list))
for idx, (pad_a_time_batch, pad_a_type_batch, pad_m_time_batch, pad_m_type_batch) in enumerate(train_dataloader):
    print("batch id: ", idx)
    cur_input_emb = input_embedding_layer(pad_a_type_batch, pad_a_time_batch)
    print("pad_a_type_batch shape: ", pad_a_type_batch.shape)  # [batch_size, num_a_types*max_pad_len_in_cur_batch]
    print("pad_a_time_batch shape: ", pad_a_time_batch.shape)  # [batch_size, num_a_types*max_pad_len_in_cur_batch]


    print("cur_input_emb shape: ", cur_input_emb.shape)
    # todo: [batch_size, num_a_types*max_pad_len_in_cur_batch, param.d_emb], will be regarded as k,v in self-attn of transformer's encoder





