import torch.nn as nn
from Utils import clones
from Add_Norm import SublayerConnection


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)  # layer.size: param.d_emb in run.sh

    def forward(self, x, mask):
        for idx, layer in enumerate(self.layers):
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x[1], x[0], x[0], mask))
        # in this case, x is a tuple belike: (type_emb + time_emb, src[2])
        # where src[2] is query comprising of grids' mid times
        return self.sublayer[1](x, self.feed_forward)
        # todo: finally returned shape: [batch_size, num_grids, d_model]
