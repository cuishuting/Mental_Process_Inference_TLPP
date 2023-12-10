import torch.nn as nn
from Utils import clones
from Add_Norm import SublayerConnection


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)  #todo: call for nn.LayerNorm may have bug: check what is param.layer

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
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

    def forward(self, q, k, v, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0]((q, k, v), lambda q,k,v: self.self_attn(q, k, v, mask))
        return self.sublayer[1](x, self.feed_forward)