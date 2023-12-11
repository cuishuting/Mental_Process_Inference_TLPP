from Utils import MultiHeadedAttention, PositionwiseFeedForward
from copy import deepcopy as dc
from Data_Gen import Logic_Model_Generator
import argparse
import torch


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





