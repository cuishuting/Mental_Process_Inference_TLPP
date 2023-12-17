from Utils import MultiHeadedAttention, PositionwiseFeedForward, Get_q_all_grids, subsequent_mask, neg_log_likelihood, get_m_occur_grids_list
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
import matplotlib.pyplot as plt


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


def model_pred_hz(model, pad_a_time_batch, pad_a_type_batch, param, num_grids, mental_type_list):
    src_mask = pad_a_time_batch.eq(0)  # [batch_size, num_a_types*max_pad_len_in_cur_batch]
    q_input = Get_q_all_grids(param.time_horizon, param.sep_for_grids, param.d_emb, param.batch_size)
    src = (pad_a_time_batch, pad_a_type_batch, q_input)
    memory = model.encode(src, src_mask)  # [batch_size, num_grids, param.d_emb]
    history_grids_time = torch.zeros((param.batch_size, 1))  # initialize begin token's time
    begin_token = torch.ones((param.batch_size, 1)).type_as(src[1])  # 1 represent none mental occurs currently
    history_grids_type = begin_token
    pred_hz = torch.zeros((param.batch_size, num_grids, len(mental_type_list) + 1)).requires_grad_(True)
    # todo: currently we only have one mental type, we use 0 to represent the only real mental type
    #  and 1 to represent none mental occurs
    for i in range(num_grids):
        out = model.decode(memory, src_mask, (history_grids_time, history_grids_type),
                           subsequent_mask(history_grids_time.size(1)))
        prob_all_types = model.generator(out)  # [batch_size, i+1, num_mental_types]
        _, history_grids_type = torch.max(prob_all_types, dim=-1)  # [batch_size, i+1]
        cur_grid_time = torch.tensor([(i + 1 / 2) * param.sep_for_grids] * param.batch_size).view(param.batch_size, 1)
        history_grids_time = torch.cat((history_grids_time, cur_grid_time), dim=-1)
        history_grids_type = torch.cat((begin_token, history_grids_type), dim=-1)

        if i == num_grids - 1:
            pred_hz = prob_all_types
    return pred_hz


def train_epoch(train_dataloader, model, param, num_grids, mental_type_list, optimizer):
    model.train()
    for idx, (pad_a_time_batch, pad_a_type_batch, pad_m_time_batch, pad_m_type_batch) in enumerate(train_dataloader):
        """get pred hz and ground truth mental occurrence in curb"""
        pred_hz = model_pred_hz(model, pad_a_time_batch, pad_a_type_batch, param, num_grids, mental_type_list)
        m_occur_grids_list_curb, _ = get_m_occur_grids_list(pad_m_time_batch, param.time_horizon, param.sep_for_grids,
                                                         param.batch_size)
        """loss computation & model optim"""
        neg_ll = neg_log_likelihood(param.batch_size, pred_hz, m_occur_grids_list_curb)
        if idx % param.batch_size == 1:
            print(("Epoch Step in batch: %6d | Loss: %6.2f | Learning Rate: %6.1e") % (idx, neg_ll, optimizer.state_dict()['param_groups'][0]['lr']))
        neg_ll.backward()
        optimizer.step()
        optimizer.zero_grad()


def eval_epoch(model, test_dataloader, param, num_grids, mental_type_list):
    model.eval()
    grids = np.arange(0, param.time_horizon, param.sep_for_grids)
    with torch.no_grad():
        for idx, (pad_a_time_batch, pad_a_type_batch, pad_m_time_batch, pad_m_type_batch) in enumerate(test_dataloader):
            m_occur_grids_list_curb, org_time_dict = get_m_occur_grids_list(pad_m_time_batch, param.time_horizon, param.sep_for_grids,
                                                             param.batch_size)
            # shape: [batch_size, num_of_grids]
            # org_time_dict[0][b_id] to get real mental occur time list
            pred_hz = model_pred_hz(model, pad_a_time_batch, pad_a_type_batch, param, num_grids, mental_type_list)
            # shape: [batch_size, num_grids, num_mental_types]

            for t_id in range(param.batch_size):
                cur_pred_hz = pred_hz[t_id, :, 0].cpu()
                plt.figure(figsize=(8, 5))
                """plot predicted hazard function on each grid"""
                plt.plot(grids, cur_pred_hz, color='blue', label='predicted hazard function')
                """plot ground truth mental occurrence on pred hazard func"""
                cur_mental_oc_gr_list = m_occur_grids_list_curb[t_id]
                scatter_pred_hz_list = []
                for g_id in range(num_grids):
                    if cur_mental_oc_gr_list[g_id]:
                        scatter_pred_hz_list.append(cur_pred_hz[g_id])
                plt.scatter(org_time_dict[0][t_id].cpu(), scatter_pred_hz_list, marker='o', color='red')
                plt.title('pred hazard func of mental event with sep_for_grids: ' + str(param.sep_for_grids) + " train samples: " + str(param.num_sample))
                plt.xlabel("grids")
                plt.ylabel("hazard function")
                plt.savefig("./result_visual/sample_" + str(param.num_sample) + "_grid_size_" + str(param.sep_for_grids) + "_batch_" + str(param.batch_size) + "/" + "fig_" + str(t_id) + ".png")
                plt.close()


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
parser.add_argument('-lr_scheduler_step', type=int)
parser.add_argument('-num_iter', type=int)
param = parser.parse_args()

param.device = torch.device('cuda')
action_type_list = list(map(int, param.action_type_list[0]))  # cur: [1, 2]
mental_type_list = list(map(int, param.mental_type_list[0]))  # cur: [0]
num_grids = int(param.time_horizon / param.sep_for_grids)


"""Generate train data"""
org_train_data_file_path = "./Synthetic_Data/data_" + str(param.num_sample) + "_sample_" + str(param.batch_size) + "_batch_" + str(param.sep_for_grids) + "_sep_grids/train_data_gen_sep_" + str(param.sep_for_data_syn) + ".npy"
if os.path.exists(org_train_data_file_path):
    org_train_data = np.load(org_train_data_file_path, allow_pickle=True).item()
else:
    print("begin generating training data: sample size: ", param.num_sample)
    data_generator_train = Logic_Model_Generator(param.time_tolerance, param.decay_rate, param.sep_for_data_syn, param.sep_for_grids)
    org_train_data = data_generator_train.generate_data(param.num_sample, param.time_horizon)
    np.save(org_train_data_file_path, org_train_data)
print("finish generating training data!")
train_dataloader = get_dataloader(org_train_data, action_type_list, mental_type_list, param.batch_size)


"""Generate test data"""
org_test_data_file_path = "./Synthetic_Data/data_" + str(param.num_sample) + "_sample_" + str(param.batch_size) + "_batch_" + str(param.sep_for_grids) + "_sep_grids/test_data_gen_sep_" + str(param.sep_for_data_syn) + ".npy"
if os.path.exists(org_test_data_file_path):
    org_test_data = np.load(org_test_data_file_path, allow_pickle=True).item()
else:
    print("begin generating testing data: sample size: ", param.batch_size)
    data_generator_test = Logic_Model_Generator(param.time_tolerance, param.decay_rate, param.sep_for_data_syn, param.sep_for_grids)
    org_test_data = data_generator_test.generate_data(param.batch_size, param.time_horizon)
    np.save(org_test_data_file_path, org_test_data)
print("finish generating testing data!")
test_dataloader = get_dataloader(org_test_data, action_type_list, mental_type_list, param.batch_size)


model = make_model(param, action_type_list, mental_type_list)
optimizer = optim.SGD(model.parameters(), lr=param.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=param.lr_scheduler_step, gamma=0.5)
print("begin of training")
for epoch in range(param.num_iter):
    print("epoch: ", epoch+1)
    train_epoch(train_dataloader, model, param, num_grids, mental_type_list, optimizer)
    scheduler.step()

print("begin of testing")
eval_epoch(model, test_dataloader, param, num_grids, mental_type_list)

print("The End!")






