import torch.nn as nn
import torch
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout):
        super().__init__()
        self.temperature = temperature  # i.e., sqrt(d_emb)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        """
        q: [batch_size, num_grids, d_q_emb == d_k]
        k: [batch_size, num_grids, d_k_emb == d_k]
        v: [batch_size, num_grids, d_v]
        """
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))
        attn = self.dropout(self.softmax(attn))
        output = torch.matmul(attn, v)  # [num_grids, d_v]

        return output


class Encoder(nn.Module):
    def __init__(self, d_emb, d_k, d_v, d_hid, dropout):
        super().__init__()
        self.d_emb = d_emb  # d_emb is encoding dim for each action event (for both temporal and type encoding)
        self.d_k = d_k  # d_q_emb == d_k_emb == d_k
        self.d_v = d_v
        self.d_hid = d_hid
        # self.normalize_before = normalize_before  ???
        self.w_q = nn.Linear(self.d_emb, self.d_k, bias=False)
        self.w_k = nn.Linear(self.d_emb, self.d_k, bias=False)
        self.w_v = nn.Linear(self.d_emb, self.d_v, bias=False)
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5, attn_dropout=dropout)
        # self.layer_norm = nn.LayerNorm(self.d_v, eps=1e-6)  ==> leave or not??
        self.dense1 = nn.Linear(self.d_v, self.d_hid*2)
        self.dense2 = nn.Linear(self.d_hid*2, self.d_hid)
        self.dense3 = nn.Linear(self.d_hid, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_all_grids):
        """
        input_all_grids: shape [batch_size, num_grids, d_emb]
        """
        q = self.w_q(input_all_grids)  # shape: [batch_size, num_grids, d_k]
        k = self.w_k(input_all_grids)  # shape: [batch_size, num_grids, d_k]
        v = self.w_v(input_all_grids)  # shape: [batch_size, num_grids, d_v]
        attn_v = self.attention(q, k, v)  # shape: [batch_size, num_grids, d_v]
        tmp_out = self.dense1(attn_v)
        tmp_out = self.dropout(tmp_out)
        tmp_out = self.dense2(tmp_out)
        tmp_out = self.dropout(tmp_out)
        tmp_out = self.dense3(tmp_out)
        pred_hz = self.sigmoid(tmp_out)  # shape: [batch_size, num_grids, 1]

        return pred_hz

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
