import numpy as np
import torch.nn as nn
import torch
from generate_changing_weight_simple_find_intensity import Logic_Model_Generator
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


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
        self.dense1 = nn.Linear(self.hidden_size, 5)
        self.dense2 = nn.Linear(5, 1)
        self.ELU = nn.ELU()


    def forward(self, input_data): # input_data shape: [num_of_grids, batch_size, input_size(time_emb_size*2)]
        h_t = torch.randn(self.batch_size, self.hidden_size)
        c_t = torch.randn(self.batch_size, self.hidden_size)
        mental_intensity_list = [] # output of lstm
        for g_id in range(self.num_of_grids):
            h_t, c_t = self.lstm_cell(input_data[g_id], (h_t, c_t))
            tmp_intensity = self.dense1(h_t)
            tmp_intensity = self.dense2(tmp_intensity)
            output_intensity = self.ELU(tmp_intensity)
            mental_intensity_list.append(output_intensity)
        return torch.stack(mental_intensity_list)  # shape: [num_of_grids, batch_size, 1]

    def obj_function(self, mental_intensity_list, b_id, mask_mental_occur):
        """
        obj func is log-likelihood of mental predicates based on intensity function learned from lstm_cell (regard action
        predicates' log-likelihood as constant)
        """
        mask_mental_occur_cur_batch = mask_mental_occur[b_id*self.batch_size:(b_id+1)*self.batch_size, :]
        mask_mental_occur = torch.transpose(mask_mental_occur_cur_batch, 0, 1).reshape(mental_intensity_list.shape)
        occured_mental_intensity_list = mental_intensity_list * mask_mental_occur  # new shape: [num_of_grids, batch_size, 1]
        occured_mental_intensity_list = torch.transpose(occured_mental_intensity_list, 0, 1) # [batch_size, num_of_grids, 1]
        occur_grid_idx = torch.nonzero(occured_mental_intensity_list)
        log_likelihood = torch.tensor(0.0, requires_grad=True)
        # first part: sum of occurred mental intensity
        for b_id in range(self.batch_size):
            cur_batch_occur_grids_list = occur_grid_idx[occur_grid_idx[:,0] == b_id][:, 1]
            for g_id in cur_batch_occur_grids_list:
                log_likelihood = log_likelihood + occured_mental_intensity_list[b_id][g_id][0]
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


"""
Generate training data and train model LSTM_intensity
"""
time_tolerance = 0
decay_rate = 1
time_horizon = 50
num_sample = 100
sep = 0.5  # discrete small grids length
mental_predicate_set = [0]
action_predicate_set = [1]
time_emb_size = 5
data_generator = Logic_Model_Generator(time_tolerance, decay_rate, time_horizon, sep)
org_train_data_dict, _, _ = data_generator.generate_data(num_sample, time_horizon)

get_train_data = Get_Data(num_sample, time_horizon, sep, mental_predicate_set, action_predicate_set, time_emb_size, org_train_data_dict)
mask_mental_occur_train, transformed_input_train = get_train_data.get_LSTM_intensity_input()
# mask_mental_occur_train shape: [num_sample, num_of_grids]
# transformed_input_train shape: [num_sample, num_of_grids, time_emb_size*2]

hidden_size = 20
batch_size = 5
model = LSTM_intensity(input_size=time_emb_size*2, hidden_size=hidden_size, num_of_grids=int(time_horizon / sep), batch_size=batch_size,
                       time_horizon=time_horizon, sep=sep)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
num_iter = 50
for iter in range(num_iter):
    shuffled_input, shuffled_mask = shuffle(transformed_input_train, mask_mental_occur_train, random_state=iter)
    for b_id in range(int(num_sample/batch_size)):
        input_data = shuffled_input[b_id*batch_size:(b_id+1)*batch_size, :, :] # shape: [batch_size, num_of_grids, time_emb*2]
        input_data = torch.transpose(input_data, 0, 1)
        # print(input_data.shape)
        pred_mental_intensity_list = model(input_data)
        avg_neg_log_likelihood = model.obj_function(pred_mental_intensity_list, b_id, shuffled_mask)
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
test_data_generator = Logic_Model_Generator(time_tolerance, decay_rate, time_horizon, sep)
org_test_data_dict, test_t_list_dict, test_ground_truth_intensity_list_dict = test_data_generator.generate_data(num_sample_test, time_horizon)

get_test_data = Get_Data(num_sample_test, time_horizon, sep, mental_predicate_set, action_predicate_set, time_emb_size, org_test_data_dict)
mask_mental_occur_test, transformed_input_test = get_test_data.get_LSTM_intensity_input()
# transformed_input_test shape: [num_sample, num_of_grids, time_emb_size*2]
test_input = torch.transpose(transformed_input_test, 0, 1)
pred_mental_intensity_list_test = model(test_input) # shape: [num_of_grids, batch_size, 1]

"""
result visualization
"""
grids = np.arange(0, time_horizon, sep)[:int(time_horizon / sep)]


for t_id in range(num_sample_test):

    plt.plot(test_t_list_dict[t_id], test_ground_truth_intensity_list_dict[t_id], color='red', label='ground truth intensity')
    plt.show()
    # print(ground_truth_mental_intensity)
    plt.plot(grids, pred_mental_intensity_list_test[:, t_id, :].reshape(-1).detach().numpy(), color='blue', label='predicted intensity')
    plt.xlabel('time')
    plt.ylabel('intensity')
    plt.title('predicted intensity of mental event')
    plt.show()















