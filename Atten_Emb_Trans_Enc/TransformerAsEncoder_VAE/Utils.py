import torch.nn as nn
import copy
import torch
import math
import numpy as np


def logic_rule():
    logic_template = {}
    '''
    Mental predicate: [1]
    '''
    head_predicate_idx = 1
    logic_template[head_predicate_idx] = {}

    # NOTE: rule content: 2 and before(2, 1) to 1
    formula_idx = 0
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [2]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[2, 1]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = ['BEFORE']

    '''
    Action predicates: [2, 3]
    '''
    head_predicate_idx = 2
    logic_template[head_predicate_idx] = {}

    # NOTE: rule content: 3 and before(3,2) to 2
    formula_idx = 0
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [3]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[3, 2]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = ['BEFORE']

    head_predicate_idx = 3
    logic_template[head_predicate_idx] = {}

    # NOTE: rule content: 1 and before(1,3) to 3
    formula_idx = 0
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [1]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[1, 3]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = ['BEFORE']

    return logic_template


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
    return processed_data != 0, org_time_dict


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
    """
    in the case of self-attn in encoder:
        shape of q,k,v: [batche_size, h, num_grids(q)/num_of_action_types*max_pad_len(k&v), d_k]  (d_model=d_k*h)
        shape of mask: [batch_size, h, num_grids, num_a_types*max_pad_len_in_cur_batch]
    """
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
        """
        query: [batch_size, num_grids, param.d_emb]
        key&value: [batch_size, num_a_types * max_pad_len_in_cur_batch, param.d_emb]
        mask: [batch_size, num_a_types*max_pad_len_in_cur_batch], False: real time stamp, True: padded 0
        """
        num_grids = query.shape[1]
        nbatches = query.shape[0]
        if mask is not None:
            if not (query.shape[1] == key.shape[1]):  # the case of self-attn in encoder
                # Same mask applied to all h heads.
                mask = mask.unsqueeze(1).unsqueeze(2).repeat(1, self.h, num_grids, 1).to('cuda')
            else:  # the case of self-attn in decoder
                mask = mask.unsqueeze(1).repeat(nbatches, self.h, 1, 1).to('cuda')
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask)
        # todo: x's shape: [batch_size, h, num_grids, d_k]

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)  # todo: shape: [batch_size, num_grids, d_model]


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


def neg_log_likelihood(batch_size, pred_hz, target_m):
    log_likelihood = torch.zeros(1).to('cuda')
    pred_hz_m0 = pred_hz[:, :, 0]
    num_grids = target_m.shape[1]
    for b_id in range(batch_size):
        target_m_curb = target_m[b_id]
        pred_hz_m0_curb = pred_hz_m0[b_id]
        for g_id in range(num_grids):
            if target_m_curb[g_id]:
                log_likelihood = log_likelihood + torch.log(pred_hz_m0_curb[g_id])
            else:
                log_likelihood = log_likelihood + torch.log(1 - pred_hz_m0_curb[g_id])
    final_neg_log_likelihood = -1 * log_likelihood / batch_size
    return final_neg_log_likelihood


def get_sampled_m_time_type_batch(sampled_m, sep_for_grids):
    """
    sampled_m: [nbatch, num_grids, num_m_types], last dim is one-hot vector, from Gumbel-softmax
    """
    cur_grids_num = sampled_m.shape[1]
    batch_size = sampled_m.shape[0]
    m_time_batch = torch.tensor([(g_id + 0.5) * sep_for_grids for g_id in range(cur_grids_num)]).unsqueeze(0).repeat(batch_size, 1).to('cuda')  # shape: [nbatch, cur_grids_num]
    m_type_batch = torch.argmax(sampled_m, dim=-1).to('cuda')
    # m_type_batch shape: [nbatch, cur_grids_num], 0 is for mental occurrence, 1 is for none mental
    masked_m_time = m_time_batch * m_type_batch.eq(0)  # shape: [nbatch, cur_grids_num]
    # masked_m_time: only keep time stamps where certain m sampled in crsp grid, with all other grids valued 0
    occur_m_time = dict()

    for i in range(batch_size):
        cur_b_mask = masked_m_time[i].eq(0)
        occur_m_time[i] = torch.masked_select(masked_m_time[i], ~cur_b_mask).to('cuda')
    cur_b_max_m_len = max([len(occur_m_time[i]) for i in range(batch_size)])
    history_m_time_batch = torch.zeros((batch_size, cur_b_max_m_len)).to('cuda')
    history_m_type_batch = torch.zeros((batch_size, cur_b_max_m_len)).to('cuda')
    for i in range(batch_size):
        history_m_time_batch[i] = torch.cat([occur_m_time[i], torch.tensor([0] * (cur_b_max_m_len-len(occur_m_time[i]))).to('cuda')], dim=-1)
        history_m_type_batch[i][len(occur_m_time[i]):] = 1  # todo: 0 is the only real mental type, 1 is none mental type

    return history_m_time_batch, history_m_type_batch


def rec_a_log_likelihood(pred_next_a_types_probs, pred_lambda_time2next_a, real_next_a_type, real_time2next_a, delta_time2next, loss_pred_a_type):
    """
    Get log_likelihood on one accepted pred_next_a
    pred_next_a_types_probs: [batch_size, num_action_types]
    pred_lambda_time2next_a: [batch_size, 1] -> conditional intensity func, lambda
    real_next_a_type: [batch_size]
    real_time2next_a: [batch_size]
    """
    loss_a_type_mean = loss_pred_a_type(input=pred_next_a_types_probs, target=real_next_a_type-1)
    loss_a_time2next_mean = torch.sum(torch.log(1 - torch.exp(-1*pred_lambda_time2next_a.reshape(-1)*delta_time2next))
                                      - pred_lambda_time2next_a.reshape(-1)*real_time2next_a) / len(real_time2next_a)
    return loss_a_type_mean + loss_a_time2next_mean
