import torch.nn as nn
from torch.nn.functional import softmax, gumbel_softmax


class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences"""
        return self.decode(self.encode(src, src_mask), tgt, tgt_mask)


class Generator(nn.Module):
    """Define standard linear + logsoftmax generation step for each grid's hazard func."""

    def __init__(self, d_model, num_mental_types, temperature):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, num_mental_types)
        self.tau = temperature

    def forward(self, x):
        # todo: an important advantage about using log_softmax in addition to numerical stability: this activation
        #  function heavily penalizes wrong class prediction as compared to its Softmax counterpart
        # return log_softmax(self.proj(x), dim=-1)
        logits = softmax(self.proj(x), dim=-1)  # consider output[0] as hz of mental 0
        mental_samples = gumbel_softmax(logits, tau=self.tau, hard=True)
        return mental_samples






