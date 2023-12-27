import torch.nn as nn
from Utils import clones
from Add_Norm import SublayerConnection


class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, tgt_mask):
        # todo: x is from Embeddings.forward(src)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, tgt_mask):
        """
        a. x: from Embeddings.forward(src=(history_grids_time, history_grids_type)) with
        shape: [batch_size, cur_output_emb_grids_num, d_model]
        b. memory: from model.encode(), with shape [batch_size, num_grids, param.d_emb]
        c. tgt_mask: subsequent mask with shape: [1, cur_output_emb_grids_num, cur_output_emb_grids_num]
        """
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m))
        return self.sublayer[2](x, self.feed_forward)