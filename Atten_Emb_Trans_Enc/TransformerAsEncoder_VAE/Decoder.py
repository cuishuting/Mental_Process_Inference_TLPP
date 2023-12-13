import torch.nn as nn
from Utils import clones
from Add_Norm import SublayerConnection


class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
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

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m))
        # todo: check whether it's right to not to feed src_mask into cross-attn in decoder
        #  because we assume that k&v from encoder in cross-attn has already consider the effect of masks
        #  and because the row number of q and k&v in encoder's self-attn are various so the src_mask, which is
        #  obtained from padded org_data and has correlation with input k&v's num of rows, can not explain the padding
        #  situation of output memory (as k&v in decoder's cross attn) from encoder whose num of rows equals to num of grids

        return self.sublayer[2](x, self.feed_forward)