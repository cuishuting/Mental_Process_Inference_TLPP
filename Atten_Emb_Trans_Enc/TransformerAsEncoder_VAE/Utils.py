import torch.nn as nn
import copy
import torch
import math
import numpy as np


def clones(module, N):
    """Produce N identical layers."""""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 1


def get_m_occur_grids_list(pad_m_time_batch, time_horizon, sep_for_grids, batch_size):
    """
    pad_m_time_batch: shape: [batch_size, max_len_after_padded_m0]
    output: True-False array with True signifying mental occurrence in cur grid, shape: [batch_size, num_of_grids](only 1 mental currently)
    """
    org_time_dict = dict()
    org_time_dict[0] = dict()
    num_of_grids = int(time_horizon / sep_for_grids)
    processed_data = np.zeros([batch_size, num_of_grids])
    for b_id in range(batch_size):
        if len(np.where(pad_m_time_batch[b_id] == 0)[0]) == 0:
            org_time_dict[0][b_id] = pad_m_time_batch[b_id]
        else:
            begin_pad_idx = np.where(pad_m_time_batch[b_id] == 0)[0][0]
            org_time_dict[0][b_id] = pad_m_time_batch[b_id][:begin_pad_idx]
        cur_check_time_pos = 0
        cur_real_time_seq = org_time_dict[0][b_id]
        for g_id in range(num_of_grids):
            cur_grid_right_time = (g_id + 1) * sep_for_grids
            if (cur_check_time_pos < len(cur_real_time_seq)) and (cur_real_time_seq[cur_check_time_pos] <= cur_grid_right_time):
                processed_data[b_id][g_id] = cur_real_time_seq[cur_check_time_pos]
                cur_check_time_pos += 1
            else:
                continue
    return processed_data != 0


def Get_q_all_grids(time_horizon, sep_for_grids, d_emb, batch_size):
    # todo: return query tensor on all grids with shape [n, d_emb], n is num of grids, expand first dim as batch_size
    num_grids = int(time_horizon / sep_for_grids)
    mid_time_list = torch.tensor([(i+1/2) * sep_for_grids for i in range(num_grids)])
    pos_vec = torch.tensor([math.pow(10000.0, 2.0 * (i // 2) / d_emb) for i in range(d_emb)])
    q_all_grids = mid_time_list.unsqueeze(-1).expand(num_grids, d_emb) / pos_vec
    q_all_grids[:, 0::2] = torch.sin(q_all_grids[:, 0::2])
    q_all_grids[:, 1::2] = torch.cos(q_all_grids[:, 1::2])
    q_all_grids = q_all_grids.unsqueeze(0).expand(batch_size, num_grids, d_emb)
    return q_all_grids.requires_grad_(False)  # shape: [batch_size, num_grids, d_emb]


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill_(mask, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size: d_model and number of heads: h."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        num_grids = query.shape[1]
        nbatches = query.size(0)
        if mask is not None:
            if not (query.shape[1] == key.shape[1]):  # the case of self-attn in encoder
                # Same mask applied to all h heads.
                mask = mask.unsqueeze(1).unsqueeze(2).repeat(1, self.h, num_grids, 1)
            else:  # the case of self-attn in decoder
                mask = mask.unsqueeze(1).repeat(nbatches, self.h, 1, 1)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        # shape of q, k, v from: [nbatches, num_grids/num_of_action_types*max_pad_len, d_model]
        # => to: [nbatches, h, num_grids/num_of_action_types*max_pad_len, d_k]  (d_model=d_k*h)



        # 2) Apply attention on all the projected vectors in batch.


        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)   # before transpose: [nbatches, rows_of_query, d_model]
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)  # [nbatches, rows_of_query, d_model]


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class NegLogLikelihood(nn.Module):
    """discrete time repeated survival process model's negative log likelihood"""
    def __init__(self, batch_size):
        super(NegLogLikelihood, self).__init__()
        self.log_likelihood = torch.zeros(1).requires_grad_(True)
        self.batch_size = batch_size

    def forward(self, pred_hz, target_m):
        """
        1. pred_hz: output from transformer's decoder's generator with shape: [batch_size, num_grids, num_mental_types]
        currently, we use pred_hz[:,:,0] to represent mental 0
        2. target_m: True-False array with True signifying ground truth mental occurrence in cur grid, shape:
        [batch_size, num_of_grids]
        """
        pred_hz_m0 = pred_hz[:, :, 0]
        num_grids = target_m.shape[1]
        for b_id in range(self.batch_size):
            target_m_curb = target_m[b_id]
            pred_hz_m0_curb = pred_hz_m0[b_id]
            for g_id in range(num_grids):
                if target_m_curb[g_id]:
                    self.log_likelihood += torch.log(pred_hz_m0_curb[g_id])
                else:
                    self.log_likelihood += torch.log(1 - pred_hz_m0_curb[g_id])
        return -1 * self.log_likelihood / self.batch_size