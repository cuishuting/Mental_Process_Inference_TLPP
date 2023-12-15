from Utils import MultiHeadedAttention, PositionwiseFeedForward, Get_q_all_grids, subsequent_mask, NegLogLikelihood, get_m_occur_grids_list
from Transformer import EncoderDecoder, Generator
from Encoder import Encoder, EncoderLayer
from Decoder import Decoder, DecoderLayer
from Embedding import Embeddings
from copy import deepcopy as dc
from Data_Gen import Logic_Model_Generator
from Get_DataLoader import get_dataloader
import numpy as np
import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim


def make_model(param, action_type_list, mental_type_list):
    attn = MultiHeadedAttention(h=param.d_h, d_model=param.d_emb)
    ff = PositionwiseFeedForward(d_model=param.d_emb, d_ff=param.d_hid, dropout=param.dropout)
    model = EncoderDecoder(Encoder(EncoderLayer(param.d_emb, dc(attn), dc(ff), dropout=param.dropout), N=param.num_sublayer),
                           Decoder(DecoderLayer(param.d_emb, dc(attn), dc(attn), dc(ff), dropout=param.dropout), N=param.num_sublayer),
                           Embeddings(param.d_emb, len(action_type_list)),
                           Embeddings(param.d_emb, len(mental_type_list)+1),
                           Generator(param.d_emb, len(mental_type_list)+1))
    """Initialize parameters with Glorot / fan_avg."""
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


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
parser.add_argument('-d_hid', type=int)
parser.add_argument('-dropout', type=float)
parser.add_argument('-num_sublayer', type=int)
parser.add_argument('-batch_size', type=int)
parser.add_argument('-lr', type=float)
parser.add_argument('-num_iter', type=int)
param = parser.parse_args()

param.device = torch.device('cuda')
action_type_list = list(map(int, param.action_type_list[0]))  # cur: [1, 2]
mental_type_list = list(map(int, param.mental_type_list[0]))  # cur: [0]
num_grids = int(param.time_horizon / param.sep_for_grids)




"""Generate train data"""
org_train_data_file_path = "./Synthetic_Data/test_for_train_data_gen.npy"
if os.path.exists(org_train_data_file_path):
    org_train_data = np.load(org_train_data_file_path, allow_pickle=True).item()
else:
    data_generator_train = Logic_Model_Generator(param.time_tolerance, param.decay_rate, param.sep_for_data_syn, param.sep_for_grids)
    org_train_data = data_generator_train.generate_data(param.num_sample, param.time_horizon)
    np.save(org_train_data_file_path, org_train_data)

train_dataloader = get_dataloader(org_train_data, action_type_list, mental_type_list, param.batch_size)

"""Generate test data"""
org_test_data_file_path = "./Synthetic_Data/test_for_test_data_gen.npy"
if os.path.exists(org_test_data_file_path):
    org_test_data = np.load(org_test_data_file_path, allow_pickle=True).item()
else:
    data_generator_test = Logic_Model_Generator(param.time_tolerance, param.decay_rate, param.sep_for_data_syn, param.sep_for_grids)
    org_test_data = data_generator_test.generate_data(param.batch_size, param.time_horizon)
    np.save(org_test_data_file_path, org_test_data)

test_dataloader = get_dataloader(org_test_data, action_type_list, mental_type_list, param.batch_size)


model = make_model(param, action_type_list, mental_type_list)
criterion = NegLogLikelihood(param.batch_size)

def train_epoch(train_dataloader, model, param, num_grids, mental_type_list, optimizer):
    for idx, (pad_a_time_batch, pad_a_type_batch, pad_m_time_batch, pad_m_type_batch) in enumerate(train_dataloader):
        m_occur_grids_list_curb = get_m_occur_grids_list(pad_m_time_batch, param.time_horizon, param.sep_for_grids, param.batch_size)
        src_mask = pad_a_time_batch.eq(0)  # [batch_size, num_a_types*max_pad_len_in_cur_batch]
        q_input = Get_q_all_grids(param.time_horizon, param.sep_for_grids, param.d_emb, param.batch_size)
        src = (pad_a_time_batch, pad_a_type_batch, q_input)
        memory = model.encode(src, src_mask)  # [batch_size, num_grids, param.d_emb]
        history_grids_time = torch.zeros((param.batch_size, 1))  # initialize begin token's time
        begin_token = torch.ones((param.batch_size, 1)).type_as(src[1]) # 1 represent none mental occurs currently
        history_grids_type = begin_token
        pred_hz = torch.zeros((param.batch_size, num_grids, len(mental_type_list)+1)).requires_grad_(True)
        # todo: currently we only have one mental type, we use 0 to represent the only real mental type
        #  and 1 to represent none mental occurs
        for i in range(num_grids):
            out = model.decode(memory, src_mask, (history_grids_time, history_grids_type), subsequent_mask(history_grids_time.size(1)))
            prob_all_types = model.generator(out)  # [batch_size, i+1, num_mental_types]
            _, history_grids_type = torch.max(prob_all_types, dim=-1) # [batch_size, i+1]
            cur_grid_time = torch.tensor([(i+1/2) * param.sep_for_grids]*param.batch_size).view(param.batch_size, 1)
            history_grids_time = torch.cat([history_grids_time, cur_grid_time], dim=-1)
            history_grids_type = torch.cat((begin_token, history_grids_type), dim=-1)
            if i == num_grids-1:
                pred_hz = prob_all_types

        """loss computation & model optim"""
        neg_ll = criterion(pred_hz, m_occur_grids_list_curb)
        neg_ll.backward()
        optimizer.step()
        optimizer.zero_grad()












print("begin of test")
train_epoch(train_dataloader, model, param, num_grids, mental_type_list)





