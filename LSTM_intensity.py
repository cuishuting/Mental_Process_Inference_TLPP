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
        self.lstm_cell = nn.LSTMCell(self.input_size, self.hidden_size)
        self.dense1 = nn.Linear(self.hidden_size, 10)
        self.dense2 = nn.Linear(10, 5)
        self.dense3 = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, input_data): # input_data shape: [num_of_grids, batch_size, input_size(time_emb_size*2)]
        h_t = torch.randn(self.batch_size, self.hidden_size).cuda()
        c_t = torch.randn(self.batch_size, self.hidden_size).cuda()
        # h_t = torch.randn(self.batch_size, self.hidden_size)
        # c_t = torch.randn(self.batch_size, self.hidden_size)
        output_intensity_all_grids = torch.zeros((self.batch_size, self.num_of_grids)).cuda()
        # output_intensity_all_grids = torch.zeros((self.batch_size, self.num_of_grids))

        for g_id in range(self.num_of_grids):
            h_t, c_t = self.lstm_cell(input_data[g_id], (h_t, c_t))
            tmp_pred_t = self.dense1(h_t)
            tmp_pred_t = self.dense2(tmp_pred_t)
            tmp_pred_t = self.dense3(tmp_pred_t) # shape: [batch_size, 1]
            final_pred_t = self.sigmoid(tmp_pred_t)
            # print(final_pred_t.reshape(-1))
            output_intensity_all_grids[:, g_id] = final_pred_t.reshape(-1)
            # lstmcell_output_list[g_id] = h_t

        return output_intensity_all_grids

    def obj_function(self, mental_intensity_list, mask_mental_occur):
        """
        obj func is log-likelihood of mental predicates based on intensity function learned from lstm_cell (regard action
        predicates' log-likelihood as constant)
        """
        occured_mental_intensity_list = mental_intensity_list * mask_mental_occur  # shape: [batch_size, seq_len]
        occured_mental_intensity_list = torch.transpose(occured_mental_intensity_list, 0, 1)  # new shape: [seq_len, batch_size]
        occur_grid_idx = torch.nonzero(occured_mental_intensity_list)
        log_likelihood = torch.tensor(0.0, requires_grad=True).cuda()
        # first part: sum of occurred mental intensity
        for b_id in range(self.batch_size):
            cur_batch_occur_grids_list = occur_grid_idx[occur_grid_idx[:, 1] == b_id][:, 0]
            for g_id in cur_batch_occur_grids_list:
                log_likelihood = log_likelihood + torch.log(occured_mental_intensity_list[g_id][b_id])
        # second part: integral of intensity function for all batches
        squeezed_mental_intensity_list = mental_intensity_list.reshape(-1)
        for intensity in squeezed_mental_intensity_list:
            log_likelihood = log_likelihood - intensity * self.sep
        log_likelihood = log_likelihood / self.batch_size
        return (-1) * log_likelihood


class Get_Data:
    """
    data preprocessing
    """
    def __init__(self, num_sample, time_horizon, sep, mental_predicates_set, action_predicates_set, time_emb_size, org_data_dic):
        self.num_sample = num_sample
        self.time_horizon = time_horizon
        self.sep = sep
        self.mental_predicates_set = mental_predicates_set
        self.action_predicates_set = action_predicates_set
        self.time_emb_size = time_emb_size
        self.org_data_dic = org_data_dic
        self.num_of_grids = int(self.time_horizon / self.sep)

    def extend_org_data_dict(self):
        processed_data = np.zeros([self.num_sample, len(self.mental_predicates_set)+len(self.action_predicates_set), self.num_of_grids])
        # print("####")
        # print(self.org_data_dic)
        # # print(len(self.org_data_dic))
        # print("####")

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

    def time_embedding(self, cur_time):
        time_emb = torch.zeros(self.time_emb_size)
        for i in range(self.time_emb_size):
            if i % 2 == 0:
                time_emb[i] = torch.sin(torch.tensor(cur_time / 10000 ** (2 * i / self.time_emb_size)))
            else:
                time_emb[i] = torch.cos(torch.tensor(cur_time / 10000 ** (2 * i / self.time_emb_size)))
        return time_emb

    def get_LSTM_intensity_input(self):
        input_lstm = torch.zeros([self.num_sample, len(self.mental_predicates_set) + len(self.action_predicates_set), self.num_of_grids, self.time_emb_size])
        processed_data = self.extend_org_data_dict()
        # processed_data size: [num_sample, len(mental_predicate_set)+len(action_predicate_set), num_of_grids]
        for sample_id in range(self.num_sample):
            for m_id in self.mental_predicates_set:
                input_lstm[sample_id][m_id] = torch.stack([self.time_embedding(processed_data[sample_id][m_id][g_id]) for g_id in range(self.num_of_grids)])
            for a_id in self.action_predicates_set:
                input_lstm[sample_id][a_id] = torch.stack([self.time_embedding(processed_data[sample_id][a_id][g_id]) for g_id in range(self.num_of_grids)])
        # todo: for simplicity, only one mental predicate is considered
        mask_mental_occur = torch.tensor((processed_data[:, self.mental_predicates_set[0], :] > 0))
        # shape: [num_sample, num_of_grids], true if mental occurs in certain grid for each sample

        transformed_input = torch.transpose(input_lstm, 1, 2)
        transformed_input = transformed_input.reshape(self.num_sample, self.num_of_grids, self.time_emb_size * 2)
        return mask_mental_occur, transformed_input

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
time_horizon = 50 # todo: increase
num_sample = 100
sep = 0.1  # discrete small grids length
mental_predicate_set = [0]
action_predicate_set = [1, 2]
time_emb_size = 5
train_data_file_str = "./Synthetic_Data/org_train_data_dict"+ "_" + str(time_horizon) + "_" + str(num_sample)+ "_" + str(sep) +".npy"
if os.path.exists(train_data_file_str):
    org_train_data_dict = np.load(train_data_file_str, allow_pickle=True).item()
    # print(org_train_data_dict)
else:
    data_generator = Logic_Model_Generator(time_tolerance, decay_rate, sep)
    org_train_data_dict = data_generator.generate_data(num_sample, time_horizon)
    np.save(train_data_file_str, org_train_data_dict)
print("#######")
print(org_train_data_dict)
print("########")

get_train_data = Get_Data(num_sample, time_horizon, sep, mental_predicate_set, action_predicate_set, time_emb_size, org_train_data_dict)
# mask_mental_occur_train, transformed_input_train = get_train_data.get_LSTM_intensity_input()
mask_mental_occur_train, transformed_input_train = get_train_data.get_LSTM_hazard_func_input()
mask_mental_occur_train = mask_mental_occur_train.to(device)
transformed_input_train = transformed_input_train.to(device)


# mask_mental_occur_train shape: [num_sample, num_of_grids]
# transformed_input_train shape: [num_sample,num_of_grids,time_emb_size*2 or time_emb_size(pred intensity/hazard_func)]


hidden_size = 20
batch_size = 5
model = LSTM_intensity(input_size=time_emb_size, hidden_size=hidden_size, num_of_grids=int(time_horizon / sep), batch_size=batch_size,
                       time_horizon=time_horizon, sep=sep)
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
num_iter = 500
model.train()
for iter in range(num_iter):
    shuffled_input, shuffled_mask = shuffle(transformed_input_train, mask_mental_occur_train, random_state=iter)
    for b_id in range(int(num_sample/batch_size)):
        input_data = shuffled_input[b_id*batch_size:(b_id+1)*batch_size, :, :] # shape: [batch_size, num_of_grids, time_emb*2]
        input_data = torch.transpose(input_data, 0, 1)
        pred_mental_intensity_list = model(input_data)
        mask_mental_occur_cur_batch = shuffled_mask[b_id * batch_size:(b_id + 1) * batch_size, :]
        mask_mental_occur = torch.transpose(mask_mental_occur_cur_batch, 0, 1).reshape(pred_mental_intensity_list.shape)
        avg_neg_log_likelihood = model.obj_function(pred_mental_intensity_list, mask_mental_occur)
        # backpropogation
        avg_neg_log_likelihood.backward()
        optimizer.step()
        optimizer.zero_grad()
        if b_id == 3:
            print("current average negative log likelihood:", avg_neg_log_likelihood.item())

"""
Generate testing data and  get model predicted mental intensity function
"""
num_sample_test = batch_size
test_data_file_str = "./Synthetic_Data/test_data_dict" + "_" + str(time_horizon) + "_" + str(num_sample_test) + "_" + str(sep) +".npy"
if os.path.exists(test_data_file_str):
    test_data_dict = np.load(test_data_file_str, allow_pickle=True).item()

else:
    test_data_generator = Logic_Model_Generator(time_tolerance, decay_rate, sep)
    test_data_dict = test_data_generator.generate_data(num_sample_test, time_horizon)
    np.save(test_data_file_str, test_data_dict)

get_test_data = Get_Data(num_sample_test, time_horizon, sep, mental_predicate_set, action_predicate_set, time_emb_size, test_data_dict)
mask_mental_occur_test, transformed_input_test = get_test_data.get_LSTM_hazard_func_input()
mask_mental_occur_test = mask_mental_occur_train.to(device)
transformed_input_test = transformed_input_test.to(device)
# transformed_input_test shape: [num_sample, num_of_grids, time_emb_size*2 or time_emb_size(pred intensity/hazard_func)]
test_input = torch.transpose(transformed_input_test, 0, 1)
model.eval()
with torch.no_grad():
    pred_mental_hazard_list_test = model(test_input) # shape: [batch_size, num_of_grids]
# print("########")
# print(pred_mental_intensity_list_test)
# print("########")
"""
result visualization
"""
grids = np.arange(0, time_horizon, sep)[:int(time_horizon / sep)]


# todo: modify result visualization: only contains pred hazard function (x axis are ground-truth mental occur time)
#  + scatter for ground-truth mental occurrence (y axis are corresponding pred hazard function)
for t_id in range(num_sample_test):

    # 1. plot ground truth hazard function (currently ratio of keeping time_to_event)
    plt.plot(test_t_dict[t_id], test_gs_hazard_func_dict[t_id], color='red', label='ground truth ratio')
    # 2. plot ground truth conditional intensity function
    plt.plot(test_t_dict[t_id], test_gs_intensity_dict[t_id], color='green', label='ground truth intensity')


    # plot pred hazard func
    plot_pred_mental_hazard_idx_list = [int(time/sep) for time in test_t_dict[t_id]]
    plot_pred_mental_hazard_list = [pred_mental_hazard_list_test[t_id][g_id].item() for g_id in plot_pred_mental_hazard_idx_list]
    plt.plot(test_t_dict[t_id], plot_pred_mental_hazard_list, color='blue', label='predicted intensity')

    # plot observed mental occurrence point (ratio as ground truth)
    plt.scatter(test_occur_t_dict[t_id], test_occurred_mental_ratio_dict[t_id], marker='o', color='red')
    # plot observed mental occurrence point (conditional intensity as ground truth)
    plt.scatter(test_occur_t_dict[t_id], test_occurred_mental_intensity_dict[t_id], marker='o', color='green')
    plt.xlabel('time')
    plt.ylabel('hazard function')
    plt.title('predicted hazard function of mental event')
    plt.show()















