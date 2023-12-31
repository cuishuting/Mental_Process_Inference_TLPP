import torch.nn as nn
import torch
import math
"""class Embedding is only used to get input and output emb, for getting input query,we'll write a separate func outside 
without considering type emb because we only use time emb in each grid as query"""


class Embeddings(nn.Module):
    """To get action or mental predicates' summation of type and time embedding"""
    def __init__(self, d_model, num_types):
        super(Embeddings, self).__init__()
        self.type_emb = nn.Embedding(num_types, d_model, padding_idx=0)
        self.d_model = d_model

    def get_time_emb(self, time_tensor):
        pad_mask = ~time_tensor.eq(0)
        pos_vec = torch.tensor([math.pow(10000.0, 2.0 * (i // 2) / self.d_model) for i in range(self.d_model)]).to('cuda')
        time_emb = time_tensor.unsqueeze(-1) / pos_vec
        time_emb[:, :, 0::2] = torch.sin(time_emb[:, :, 0::2])
        time_emb[:, :, 1::2] = torch.cos(time_emb[:, :, 1::2])
        pad_mask_expand = pad_mask.unsqueeze(-1).expand(pad_mask.shape[0], pad_mask.shape[1], self.d_model)
        final_time_emb = time_emb * pad_mask_expand
        return final_time_emb.requires_grad_(False)

    def forward(self, src):
        """return summation of type and time embedding,where type emb needs autograd but time emb does not"""
        time_tensor = src[0]
        type_tensor = src[1]
        type_emb = self.type_emb(torch.LongTensor(type_tensor.cpu().numpy()).to('cuda')) * math.sqrt(self.d_model)
        time_emb = self.get_time_emb(time_tensor)
        if len(src) == 3:  # for encoder's self-attn block
            # todo: type_emb + time_emb shape: [batch_size, num_a_types*max_pad_len_in_cur_batch, d_model]
            # todo: src[2] is query in encoder with shape: [batch_size, num_grids, d_model]
            return type_emb + time_emb, src[2]
        else:  # for decoder's src-attn and cross-attn block
            return type_emb + time_emb  # todo: shape: [batch_size, cur_output_emb_grids_num, d_model]

