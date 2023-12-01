import numpy as np
from Data_simulation import Logic_Model_Generator
import argparse
import torch
import os
from GetDataloader import get_dataloader
from InputEmbLayer import InputEmbLayer

# todo: def an extra func to "get mental_occur_grids_list" to cal obj_func for encoder

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
parser.add_argument('-batch_size', type=int)
parser.add_argument('-lr', type=float)
parser.add_argument('-num_iter', type=int)
parser.add_argument('-log', type=str)
param = parser.parse_args()

param.device = torch.device('cuda')
action_type_list = list(map(int, param.action_type_list[0]))  # cur: [1, 2]
mental_type_list = list(map(int, param.mental_type_list[0]))  # cur: [0]


"""
Generate org_train_dict data, data has form: org_train_data_dict[sample_ID][predicate_idx]['time'] = [...]
"""
train_data_file_str = "./Synthetic_Data/org_train_data_dict" + "_" + str(param.time_horizon) + "_" + str(param.num_sample) + "_" + str(param.sep_for_data_syn) + ".npy"
if os.path.exists(train_data_file_str):
    org_train_data_dict = np.load(train_data_file_str, allow_pickle=True).item()
else:
    data_generator = Logic_Model_Generator(param.time_tolerance, param.decay_rate, param.sep_for_data_syn)
    org_train_data_dict = data_generator.generate_data(param.num_sample, param.time_horizon)
    np.save(train_data_file_str, org_train_data_dict)

"""
Generate org_test_dict data
"""
num_sample_test = param.batch_size
test_data_file_str = "./Synthetic_Data/test_data_dict" + "_" + str(param.time_horizon) + "_" + str(num_sample_test) + "_" + str(param.sep_for_data_syn) + ".npy"
if os.path.exists(test_data_file_str):
    test_data_dict = np.load(test_data_file_str, allow_pickle=True).item()

else:
    test_data_generator = Logic_Model_Generator(param.time_tolerance, param.decay_rate, param.sep_for_data_syn)
    test_data_dict = test_data_generator.generate_data(num_sample_test, param.time_horizon)
    np.save(test_data_file_str, test_data_dict)

train_dataloader = get_dataloader(org_train_data_dict, action_type_list, mental_type_list, param.batch_size)

for id, (a_pad_batch, m_pad_batch, real_a_time_num, real_m_time_num) in enumerate(train_dataloader):
    if id == 0:
        model = InputEmbLayer(a_type_list=action_type_list,
                              m_type_list=mental_type_list,
                              d_emb=param.d_emb,
                              batch_size=param.batch_size,
                              time_horizon=param.time_horizon,
                              sep_for_grids=param.sep_for_grids,
                              real_a_seq_len=real_a_time_num,
                              real_m_seq_len=real_m_time_num)
        input_all_grids = model((a_pad_batch, m_pad_batch))
        print("********")
        print(m_pad_batch)  # dict with padded time seq, like: {0: torch.tensor([[1,3,0], [4,0,0], [3,4,5]])}
        print("********")

