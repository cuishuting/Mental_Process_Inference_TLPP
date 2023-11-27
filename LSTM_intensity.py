import numpy as np
import torch.nn as nn
import torch
from Data_simulation import Logic_Model_Generator
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os


class LSTM_intensity(nn.Module):
    def __init__(self, input_size, hidden_size, num_of_grids, batch_size, time_horizon, sep):
        super(LSTM_intensity, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_of_grids = num_of_grids
        self.batch_size = batch_size
        self.sep = torch.tensor(sep)
        self.time_horizon = time_horizon
        self.bi_lstm_input = nn.LSTM(self.input_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.lstm_hz = nn.LSTM(self.hidden_size*2, self.hidden_size*4, batch_first=True)
        self.dense1 = nn.Linear(self.hidden_size*4, self.hidden_size*3)
        self.dense2 = nn.Linear(self.hidden_size*3, self.hidden_size*2)
        self.dense3 = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.dense4 = nn.Linear(self.hidden_size, 10)
        self.dense5 = nn.Linear(10, 5)
        self.dense6 = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        # input_data shape: [num_sample(batch_size), num_of_grids, ini_input_size(num_action_types+1)]
        """
        bi-lstm: get input embedding of each grid for encoder(lstm+mlp currently)
        """
        h_0_bi_lstm = torch.randn(2, self.batch_size, self.hidden_size).cuda()
        c_0_bi_lstm = torch.randn(2, self.batch_size, self.hidden_size).cuda()
        output_bi_lstm, (_, _) = self.bi_lstm_input(input_data, (h_0_bi_lstm, c_0_bi_lstm))
        # output_bi_lstm shape: [batch_size, num_grids, 2*hidden_size]
        """
        lstm+mlp: encoder 
        """
        h_0_encoder = torch.randn(1, self.batch_size, self.hidden_size*4).cuda()
        c_0_encoder = torch.randn(1, self.batch_size, self.hidden_size*4).cuda()
        output_encoder, (_, _) = self.lstm_hz(output_bi_lstm, (h_0_encoder, c_0_encoder))
        # output_encoder shape: [batch_size, num_grids, 4*hidden_size]

        tmp_hz = self.dense1(output_encoder)
        tmp_hz = self.dense2(tmp_hz)
        tmp_hz = self.dense3(tmp_hz)
        tmp_hz = self.dense4(tmp_hz)
        tmp_hz = self.dense5(tmp_hz)
        tmp_hz = self.dense6(tmp_hz)

        final_hz = self.sigmoid(tmp_hz) # shape : [batch_size, num_grids， 1]
        return final_hz


    def obj_function(self, pred_hz_list, mental_occur_grids_list):
        """
        discrete time repeated survival process model's negative log likelihood
        param:
        a. pred_hz_list: shape [batch_size, num_grids， 1], from encoder
        b. mental_occur_grid_list: True-False array with True signifying ground truth mental occurrence in cur grid, shape: [batch_size, num_of_grids]
        """
        log_likelihood = torch.tensor(0.0).cuda()
        for b_id in range(self.batch_size):
            mental_oc_gr_cur_batch = mental_occur_grids_list[b_id]
            pred_hz_cur_batch = pred_hz_list[b_id].reshape(-1)
            for g_id in range(self.num_of_grids):
                if mental_oc_gr_cur_batch[g_id]:
                    log_likelihood += torch.log(pred_hz_cur_batch[g_id])
                else:
                    log_likelihood += torch.log(1 - pred_hz_cur_batch[g_id])
        log_likelihood = log_likelihood / self.batch_size
        return (-1) * log_likelihood


class Get_Data:
    """
    data preprocessing:
    a. extend org_data_dict with the same length as num_of_grids, each grid: 0/t_j
    b. final input for self.bi_lstm_input: 0/t_j concat one-hot type vector where "none action occurs" is one type
    """
    def __init__(self, num_sample, time_horizon, sep, mental_predicates_set, action_predicates_set, org_data_dic):
        self.num_sample = num_sample
        self.time_horizon = time_horizon
        self.sep = sep
        self.mental_predicates_set = mental_predicates_set
        self.action_predicates_set = action_predicates_set
        self.ini_input_size = len(self.action_predicates_set) + 1  # +1 represents circumstance: none event occur
        self.org_data_dic = org_data_dic
        self.num_of_grids = int(self.time_horizon / self.sep)

    def extend_org_data_dict(self):
        processed_data = np.zeros([self.num_sample, len(self.mental_predicates_set)+len(self.action_predicates_set), self.num_of_grids])

        for idx, sample in enumerate(self.org_data_dic):
            for m_id in self.mental_predicates_set:
                cur_mental_occur_time_list = self.org_data_dic[idx][m_id]['time']
                cur_check_occur_time = 0
                for g_id in np.arange(self.num_of_grids):
                    cur_grid_right_time = (g_id + 1) * self.sep
                    if (cur_check_occur_time < len(cur_mental_occur_time_list)) and \
                            (cur_mental_occur_time_list[cur_check_occur_time] <= cur_grid_right_time):
                        processed_data[idx][m_id][g_id] = cur_mental_occur_time_list[cur_check_occur_time]
                        cur_check_occur_time += 1
                    else:
                        continue

            for a_id in self.action_predicates_set:
                cur_action_occur_time_list = self.org_data_dic[idx][a_id]['time']
                cur_check_occur_time = 0
                for g_id in np.arange(self.num_of_grids):
                    cur_grid_right_time = (g_id + 1) * self.sep
                    if (cur_check_occur_time < len(cur_action_occur_time_list)) and \
                            (cur_action_occur_time_list[cur_check_occur_time] <= cur_grid_right_time):
                        processed_data[idx][a_id][g_id] = cur_action_occur_time_list[cur_check_occur_time]
                        cur_check_occur_time += 1
                    else:
                        continue

        return processed_data

    # def time_embedding(self, cur_time):
    #     time_emb = torch.zeros(self.time_emb_size)
    #     for i in range(self.time_emb_size):
    #         if i % 2 == 0:
    #             time_emb[i] = torch.sin(torch.tensor(cur_time / 10000 ** (2 * i / self.time_emb_size)))
    #         else:
    #             time_emb[i] = torch.cos(torch.tensor(cur_time / 10000 ** (2 * i / self.time_emb_size)))
    #     return time_emb

    def get_bi_LSTM_input(self):
        """
        get initial input encoding for bi-lstm to get input embedding of each small grid for lstm(encoder) of hazard function
        """
        input_lstm = torch.zeros([self.num_sample, self.num_of_grids, self.ini_input_size])
        processed_data = self.extend_org_data_dict()
        # processed_data size: [num_sample, len(mental_predicate_set)+len(action_predicate_set), num_of_grids]
        input_lstm[:, :, :len(self.action_predicates_set)] = torch.transpose(torch.tensor(processed_data[:, 1:, :]), 1, 2)
        return input_lstm

    def get_mental_occur_grid_list(self):
        processed_data = self.extend_org_data_dict()
        # todo: for simplicity, only considering one mental predicate
        extend_mental = processed_data[:, self.mental_predicates_set[0], :]  # [num_sample, num_of_grids]
        mental_occur_grids_list = (extend_mental > 0) # True-False array with shape [num_sample, num_of_grids]
        return mental_occur_grids_list

    def get_LSTM_hazard_func_input(self):
        # todo: based on mental observation time window (repeated survival process), get accumulated action history emb as shown in overleaf
        mental_survival_window = {} # for later use to compute obj_func(log-likelihood) over predicted hazard func
        processed_data = self.extend_org_data_dict()
        extend_mental = processed_data[:, self.mental_predicates_set[0], :]
        extend_action = processed_data[:, self.action_predicates_set, :]
        input_lstm_for_hazard = torch.zeros([self.num_sample, len(self.action_predicates_set), self.num_of_grids, self.time_emb_size])
        # input_lstm_for_hazard contains accumulated action time embedding in each mental survival window
        emb_for_time_0 = self.time_embedding(cur_time = 0) # for later accumulated action emb computation
        for sample_id in range(self.num_sample):
            cur_sample_mental_occur_grid_list = np.nonzero(extend_mental[sample_id])[0] # only consider one mental predicate's case here
            for win_id in range(len(cur_sample_mental_occur_grid_list)):
                if win_id == 0:
                    win_start_grid_idx = 0
                    win_end_grid_idx = cur_sample_mental_occur_grid_list[win_id]
                else:
                    win_start_grid_idx = cur_sample_mental_occur_grid_list[win_id-1] + 1
                    win_end_grid_idx = cur_sample_mental_occur_grid_list[win_id]
                for grid_idx in np.arange(win_start_grid_idx, win_end_grid_idx+1, 1):
                    for a_id in range(len(self.action_predicates_set)):
                        cur_time_emb = self.time_embedding(extend_action[sample_id][a_id][grid_idx])
                        if grid_idx == win_start_grid_idx:
                            input_lstm_for_hazard[sample_id][a_id][grid_idx] = cur_time_emb
                            continue
                        if extend_action[sample_id][a_id][grid_idx] != 0:
                            if input_lstm_for_hazard[sample_id][a_id][grid_idx - 1].equal(emb_for_time_0):
                                input_lstm_for_hazard[sample_id][a_id][grid_idx] = cur_time_emb
                            else:
                                input_lstm_for_hazard[sample_id][a_id][grid_idx] = input_lstm_for_hazard[sample_id][a_id][grid_idx - 1] + cur_time_emb

                        else:
                            input_lstm_for_hazard[sample_id][a_id][grid_idx] = input_lstm_for_hazard[sample_id][a_id][grid_idx - 1]

        # todo: for current simple case, only one action predicate
        final_input_lstm_for_h = input_lstm_for_hazard[:, 0, :, :]  # shape: [num_sample, num_grids, time_emb_size]
        mask_mental_occur = torch.tensor((processed_data[:, self.mental_predicates_set[0], :] > 0))
        # shape: [num_sample, num_of_grids], true if mental occurs in certain grid for each sample
        return mask_mental_occur, final_input_lstm_for_h


"""
Generate training data and train model LSTM_intensity
"""
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print(f"Using {device} device")
time_tolerance = 0.1
decay_rate = 0.8
time_horizon = 50
num_sample = 2000
sep = 0.1  # discrete small grids length
mental_predicate_set = [0]
action_predicate_set = [1, 2]
train_data_file_str = "./Synthetic_Data/org_train_data_dict" + "_" + str(time_horizon) + "_" + str(num_sample) + "_" + str(sep) +"debug.npy"
if os.path.exists(train_data_file_str):
    org_train_data_dict = np.load(train_data_file_str, allow_pickle=True).item()
    # print(org_train_data_dict)
else:
    data_generator = Logic_Model_Generator(time_tolerance, decay_rate, sep)
    org_train_data_dict = data_generator.generate_data(num_sample, time_horizon)
    np.save(train_data_file_str, org_train_data_dict)


get_train_data = Get_Data(num_sample, time_horizon, sep, mental_predicate_set, action_predicate_set, org_train_data_dict)
# mask_mental_occur_train, transformed_input_train = get_train_data.get_LSTM_intensity_input()
ini_input_train = get_train_data.get_bi_LSTM_input()
mental_occur_grids_list = get_train_data.get_mental_occur_grid_list()
ini_input_train = ini_input_train.to(device)

hidden_size = 20
batch_size = 20
model = LSTM_intensity(input_size=len(action_predicate_set) + 1, hidden_size=hidden_size, num_of_grids=int(time_horizon/sep),
                       batch_size=batch_size, time_horizon=time_horizon, sep=sep)
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
num_iter = 1000
model.train()
for iter in range(num_iter):
    shuffled_input, shuffled_mental_oc_gr_list = shuffle(ini_input_train, mental_occur_grids_list, random_state=iter)
    for b_id in range(int(num_sample/batch_size)):
        input_data = shuffled_input[b_id*batch_size:(b_id+1)*batch_size, :, :] # shape: [batch_size, num_of_grids, time_emb*2]
        pred_mental_intensity_list = model(input_data)
        mental_oc_gr_list_cur_batch = shuffled_mental_oc_gr_list[b_id * batch_size:(b_id + 1) * batch_size, :]
        avg_neg_log_likelihood = model.obj_function(pred_mental_intensity_list, mental_oc_gr_list_cur_batch)
        # backpropogation
        avg_neg_log_likelihood.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (b_id == 99) and (iter % 20 == 0):
            print("current iter: ", iter+1, ", average negative log likelihood:", avg_neg_log_likelihood.item())

"""
Generate testing data and  get model predicted mental intensity function
"""
num_sample_test = 5
test_data_file_str = "./Synthetic_Data/test_data_dict" + "_" + str(time_horizon) + "_" + str(num_sample_test) + "_" + str(sep) +"debug.npy"
if os.path.exists(test_data_file_str):
    test_data_dict = np.load(test_data_file_str, allow_pickle=True).item()

else:
    test_data_generator = Logic_Model_Generator(time_tolerance, decay_rate, sep)
    test_data_dict = test_data_generator.generate_data(num_sample_test, time_horizon)
    np.save(test_data_file_str, test_data_dict)

get_test_data = Get_Data(num_sample_test, time_horizon, sep, mental_predicate_set, action_predicate_set, test_data_dict)
ini_input_test = get_test_data.get_bi_LSTM_input()
mental_occur_grids_list_test = get_test_data.get_mental_occur_grid_list()  # shape: [num_sample_test, num_of_grids]
ini_input_test = ini_input_test.to(device)

model.eval()
with torch.no_grad():
    pred_mental_hazard_list_test = model(ini_input_test)  # shape: [batch_size, num_grids， 1]

"""
result visualization
"""
grids = np.arange(0, time_horizon, sep)
# todo: modify result visualization: only contains pred hazard function (x axis are ground-truth mental occur time)
#  + scatter for ground-truth mental occurrence (y axis are corresponding pred hazard function)
for t_id in range(num_sample_test):
    cur_pred_hz = pred_mental_hazard_list_test[t_id].reshape(-1).cpu()
    plt.plot(grids, cur_pred_hz, color='blue', label='predicted hazard function')
    # todo: for current simple case, only one mental predicate with index 0
    cur_mental_oc_gr_list = mental_occur_grids_list_test[t_id]
    scatter_pred_hz_list = []
    for g_id in range(int(time_horizon/sep)):
        if cur_mental_oc_gr_list[g_id]:
            scatter_pred_hz_list.append(cur_pred_hz[g_id])
    plt.scatter(test_data_dict[t_id][0]['time'], scatter_pred_hz_list, marker='o', color='red')
    plt.xlabel('time')
    plt.ylabel('hazard function')
    plt.title('predicted hazard function of mental event')
    plt.savefig("./result_visual/result_" + str(t_id) + "_" + str(num_sample) + "_cst.png")
    plt.close()


print("The End!!")