import numpy as np
from Data_simulation import Logic_Model_Generator
import argparse
import torch
import os
from GetDataloader import get_dataloader
from Encoder import Encoder
from Utils import get_m_occur_grids_list
import torch.optim as optim
import matplotlib.pyplot as plt


def train_epoch(model, train_dataloader, param, action_type_list, mental_type_list, optimizer):
    model.train()
    for id, (a_pad_batch, m_pad_batch, real_a_time_num, real_m_time_num) in enumerate(train_dataloader):
        for a in action_type_list:
            a_pad_batch[a] = a_pad_batch[a].to(param.device)
            real_a_time_num[a] = real_a_time_num[a].to(param.device)
        for m in mental_type_list:
            m_pad_batch[m] = m_pad_batch[m].to(param.device)
            real_m_time_num[m] = real_m_time_num[m].to(param.device)

        pred_hz = model((a_pad_batch, m_pad_batch), real_a_time_num)
        mental_occur_grids_list, _ = get_m_occur_grids_list(m_pad_batch, param.time_horizon, param.sep_for_grids,
                                                         mental_type_list, real_m_time_num, param.batch_size)
        # mental_occur_grids_list: True-False list with shape: [batch_size, len(m_type_list), num_of_grids] with True
        # signifying mental occurrence in cur grid
        # todo: only consider one mental predicate here
        neg_ll = model.obj_function(pred_hz, mental_occur_grids_list[:, 0, :])
        if (id+1) % 10 == 0:
            print("current neg ll during training is: ", neg_ll)

        """backpropagation"""
        neg_ll.backward()
        optimizer.step()
        optimizer.zero_grad()


def eval_epoch(model, test_dataloader, param, mental_type_list, action_type_list):
    model.eval()

    grids = np.arange(0, param.time_horizon, param.sep_for_grids)
    with torch.no_grad():
        for id, (a_pad_batch, m_pad_batch, real_a_time_num, real_m_time_num) in enumerate(test_dataloader):
            for a in action_type_list:
                a_pad_batch[a] = a_pad_batch[a].to(param.device)
                real_a_time_num[a] = real_a_time_num[a].to(param.device)
            for m in mental_type_list:
                m_pad_batch[m] = m_pad_batch[m].to(param.device)
                real_m_time_num[m] = real_m_time_num[m].to(param.device)
            pred_hz = model((a_pad_batch, m_pad_batch), real_a_time_num)  # [batch_size, num_grids, 1]
            mental_occur_grids_list_test, org_time_dict = get_m_occur_grids_list(m_pad_batch, param.time_horizon, param.sep_for_grids,
                                                             mental_type_list, real_m_time_num, param.batch_size)
            # mental_occur_grids_list_test: shape: [batch_size, len(m_type_list), num_of_grids]
            # org_time_dict: org_time_dict[m][b_id] to get mental m's real time seq in batch b_id
            """
            result visualization
            """
            for t_id in range(param.batch_size):
                cur_pred_hz = pred_hz[t_id].reshape(-1).cpu()

                fig, (f1_hz, f2_srv) = plt.subplots(2, 1)
                """plot predicted hazard function on each grid"""
                f1_hz.plot(grids, cur_pred_hz, color='blue', label='predicted hazard function')

                # todo: for current simple case, only one mental predicate with index 0
                cur_mental_oc_gr_list = mental_occur_grids_list_test[t_id][mental_type_list[0]]
                scatter_pred_hz_list = []
                for g_id in range(int(param.time_horizon / param.sep_for_grids)):
                    if cur_mental_oc_gr_list[g_id]:
                        scatter_pred_hz_list.append(cur_pred_hz[g_id])
                # todo: for current simple case, only one mental predicate with index 0
                """plot ground truth mental occurrence on pred hazard func"""
                f1_hz.scatter(org_time_dict[0][t_id].cpu(), scatter_pred_hz_list, marker='o', color='red')
                f1_hz.set_title('pred hazard func of mental event with sep_for_grids: ' + str(param.sep_for_grids) + " train samples: " + str(param.num_sample))
                f1_hz.set_xlabel("grids")
                f1_hz.set_ylabel("hazard function")

                survival_rate = []
                last_grid_occur = False
                for g_id in range(int(param.time_horizon / param.sep_for_grids)):
                    if not mental_occur_grids_list_test[t_id][0][g_id]:
                        if g_id == 0 or last_grid_occur:
                            survival_rate.append(1 - cur_pred_hz[g_id])
                            last_grid_occur = False
                        else:
                            survival_rate.append(survival_rate[g_id - 1] * (1 - cur_pred_hz[g_id]))
                    else:
                        survival_rate.append(survival_rate[g_id - 1] * (1 - cur_pred_hz[g_id]))
                        last_grid_occur = True
                """plot survival function(rate) based on pred hazard func"""
                f2_srv.plot(grids, survival_rate, color='green', label='survival rate')
                f2_srv.set_title("survival rate")
                f2_srv.set_xlabel('grids')
                f2_srv.set_ylabel('survival rate')
                plt.savefig("./result_visual/result_add_srv_" + str(t_id) + "_" + str(param.num_iter) + "_trans_encoder.png")
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
test_dataloader = get_dataloader(test_data_dict, action_type_list, mental_type_list, param.batch_size)

encoder_for_hz = Encoder(d_emb=param.d_emb,
                         d_k=param.d_k,
                         d_v=param.d_v,
                         d_hid=param.d_hid,
                         dropout=param.dropout,
                         a_type_list=action_type_list,
                         batch_size=param.batch_size,
                         time_horizon=param.time_horizon,
                         sep_for_grids=param.sep_for_grids,
                         device=param.device)
encoder_for_hz.to(param.device)

optimizer = optim.SGD(encoder_for_hz.parameters(), lr=param.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

"""Train encoder"""
for epoch in range(param.num_iter):
    train_epoch(encoder_for_hz, train_dataloader, param, action_type_list, mental_type_list, optimizer)
    scheduler.step()
"""Val encoder"""
eval_epoch(encoder_for_hz, test_dataloader, param, mental_type_list, action_type_list)











