import torch.nn as nn
import torch
import math


def get_time_emb(data, d_emb, device):
    """
    return time encoding for each event's time stamp, which will not be included in autograd;
    data shape: [batch_size, max_seq_len_cur_batch]
    d_emb is the dimension of both time and type encoding
    """
    pad_mask = ~data.eq(0).to(device)  # True-False tensor with shape [batch_size, max_len_cur_batch],
    # where False means respective value in data are the padded 0
    pos_vec = torch.tensor([math.pow(10000.0, 2.0 * (i // 2) / d_emb) for i in range(d_emb)]).to(device)  # d_emb length
    time_emb = data.unsqueeze(-1) / pos_vec
    time_emb[:, :, 0::2] = torch.sin(time_emb[:, :, 0::2])
    time_emb[:, :, 1::2] = torch.cos(time_emb[:, :, 1::2])
    pad_mask_expand = pad_mask.unsqueeze(-1).expand(pad_mask.shape[0], pad_mask.shape[1], d_emb)
    final_time_emb = time_emb * pad_mask_expand  # padded value will be encoded into d_emb-length zero vector
    return final_time_emb, pad_mask_expand   # [batch_size, max_seq_len, time_emb_size]


def get_q_all_grids(time_horizon, sep_for_grids, d_emb):
    # todo: return query tensor on all grids with shape [n, d_emb], n is num of grids
    num_grids = int(time_horizon / sep_for_grids)
    mid_time_list = torch.tensor([(i+1/2) * sep_for_grids for i in range(num_grids)])
    pos_vec = torch.tensor([math.pow(10000.0, 2.0 * (i // 2) / d_emb) for i in range(d_emb)])
    q_all_grids = mid_time_list.unsqueeze(-1).expand(num_grids, d_emb) / pos_vec
    q_all_grids[:, 0::2] = torch.sin(q_all_grids[:, 0::2])
    q_all_grids[:, 1::2] = torch.cos(q_all_grids[:, 1::2])
    return q_all_grids  # shape: [num_grids, d_emb]


class InputEmbLayer(nn.Module):
    def __init__(self, a_type_list, d_emb, batch_size, time_horizon, sep_for_grids, device):
        super().__init__()
        self.a_type_list = a_type_list
        self.d_emb = d_emb
        self.batch_size = batch_size
        self.time_horizon = time_horizon
        self.sep_for_grids = sep_for_grids
        self.device = device
        self.a_type_emb = nn.Embedding(len(self.a_type_list), self.d_emb)
        self.softmax_attn_weights = nn.Softmax(dim=0)  # softmax apply on all action emb in cur batch

    def forward(self, data, real_a_seq_len):
        """
        data, tuple, with data[0]: padded action seq batch, data[1]: padded mental seq batch
        real_a_seq_len: dict with structure:{1: tensor([11, 12, 12, 12, 13]), 2: tensor([11, 13, 21, 16,  8])}
        """
        a_data = data[0]  # dict with padded time seq, like: {1: torch.tensor([[1,3,0], [4,0,0], [3,4,5]]), 2: torch.tensor([[1,0,0], [2,3,5], [4,5,0]])}
        L = a_data[self.a_type_list[0]].shape[1]  # L:max_a_seq_len after padding
        all_a_emb = torch.zeros((self.batch_size, len(self.a_type_list)*L, self.d_emb))
        for id, a_type in enumerate(self.a_type_list):
            time_emb, pad_mask_exp = get_time_emb(a_data[a_type], self.d_emb, self.device)  # [batch_size, L, d_emb]
            type_emb = self.a_type_emb(torch.LongTensor([id]).to(self.device))
            type_emb = type_emb * pad_mask_exp
            cur_emb = time_emb + type_emb  # [batch_size, L, d_emb], with padded time stamps valued as zeros
            all_a_emb[:, id*L:(id+1)*L, :] = cur_emb
        q_all_grids = get_q_all_grids(self.time_horizon, self.sep_for_grids, self.d_emb)  # shape: [num_grids, d_emb]
        num_grids = int(self.time_horizon/self.sep_for_grids)
        attn_weights = torch.zeros((self.batch_size, len(self.a_type_list)*L, num_grids))
        for g_id in range(num_grids):
            for a_emb_id in range(len(self.a_type_list)*L):
                attn_weights[:, a_emb_id, g_id] = torch.tensor([torch.dot(all_a_emb[b_id, a_emb_id, :], q_all_grids[g_id, :]) /
                                                                pow(self.d_emb, 0.5) for b_id in range(self.batch_size)])
        # todo: begin of not using matrix calculation, but calculate on each batch separately, below is example when batch_id == 0
        final_input_all_grids = torch.zeros((self.batch_size, num_grids, self.d_emb))
        for b_id in range(self.batch_size):
            cur_batch_w = attn_weights[b_id, :, :]  # [num_a_types*L, num_grids]
            real_w = [cur_batch_w[id*L:id*L+real_a_seq_len[a_type][b_id], :] for (id, a_type) in enumerate(self.a_type_list)]
            real_w = torch.cat(real_w, dim=0)  # [num_real_a_times, num_grids]
            final_real_w = self.softmax_attn_weights(real_w).transpose(0, 1)  # [num_grids, num_real_a_times]
            real_a_emb = [all_a_emb[b_id, id*L:id*L+real_a_seq_len[a_type][b_id], :] for (id, a_type) in enumerate(self.a_type_list)]
            final_real_a_emb = torch.cat(real_a_emb, dim=0)
            for g_id in range(num_grids):
                tmp_input_emb = torch.zeros(self.d_emb)
                for a_id in range(final_real_w.shape[1]):
                    tmp_input_emb += final_real_w[g_id][a_id] * final_real_a_emb[a_id]
                final_input_all_grids[b_id][g_id] = tmp_input_emb

        return final_input_all_grids



