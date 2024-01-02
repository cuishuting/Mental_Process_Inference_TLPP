import torch.nn as nn
from torch.nn.functional import softmax, gumbel_softmax
import torch


class EncoderDecoder(nn.Module):

    def __init__(self, transformer_encoder, transformer_decoder, action_embed, mental_embed, generator, mental_sampler, vae_decoder):
        super(EncoderDecoder, self).__init__()
        self.transformer_encoder = transformer_encoder
        self.transformer_decoder = transformer_decoder
        self.action_embed = action_embed
        self.mental_embed = mental_embed
        self.generator = generator
        self.mental_sampler = mental_sampler
        self.vae_decoder = vae_decoder

    def encode(self, src, src_mask):
        return self.transformer_encoder(self.action_embed(src), src_mask)

    def decode(self, memory, tgt, tgt_mask):
        return self.transformer_decoder(self.mental_embed(tgt), memory, tgt_mask)

    def get_next_a_type_lamb(self, history_a, history_m):
        pred_next_a_types_prob, pred_lambda_time2next_a = self.vae_decoder(self.action_embed, self.mental_embed, history_a, history_m)
        return pred_next_a_types_prob, pred_lambda_time2next_a

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences"""
        return self.decode(self.encode(src, src_mask), tgt, tgt_mask)


class Generator(nn.Module):
    """Define standard linear + logsoftmax generation step for each grid's hazard func."""

    def __init__(self, d_model, num_mental_types):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, num_mental_types)

    def forward(self, x):
        # todo: an important advantage about using log_softmax in addition to numerical stability: this activation
        #  function heavily penalizes wrong class prediction as compared to its Softmax counterpart
        # return log_softmax(self.proj(x), dim=-1)
        logits = softmax(self.proj(x), dim=-1)  # consider output[0] as hz of mental 0
        return logits


class MentalSampler(nn.Module):
    def __init__(self, temperature):
        super(MentalSampler, self).__init__()
        self.tau = temperature

    def forward(self, logits): # logits shape: [batch_size, num_grids, num_mental_types]
        mental_samples = gumbel_softmax(logits, tau=self.tau, hard=True)
        return mental_samples


class DecoderForVAE(nn.Module):
    def __init__(self, d_model, d_hid, dropout, action_type_list):
        # todo: action_type_list doesn't include none type
        super(DecoderForVAE, self).__init__()
        self.d_model = d_model
        self.d_ff = d_hid
        self.dropout = dropout
        self.decoder_type_a = nn.Sequential(nn.Linear(self.d_model, self.d_ff),
                                            nn.ReLU(),
                                            nn.Dropout(self.dropout),
                                            nn.Linear(self.d_ff, self.d_model),
                                            nn.ReLU(),
                                            nn.Dropout(self.dropout),
                                            nn.Linear(self.d_model, len(action_type_list)),
                                            nn.Softmax(dim=-1))
        # constructing decoder for next time event, the last activation function is
        # softplus function to predict intensity function for next a
        self.decoder_time2next_a = nn.Sequential(nn.Linear(self.d_model, self.d_ff),
                                                 nn.ReLU(),
                                                 nn.Dropout(self.dropout),
                                                 nn.Linear(self.d_ff, self.d_model),
                                                 nn.ReLU(),
                                                 nn.Dropout(self.dropout),
                                                 nn.Linear(self.d_model, 1),
                                                 nn.Softplus())

    def forward(self, decoder_input_emb_a, decoder_input_emb_m, history_a, history_m):
        # history_a: (history_a_time, history_a_type), tuple
        # history_m: (history_m_time, history_m_type), tuple
        history_a_emb = decoder_input_emb_a(history_a)  # todo: check whether need to include history_a as VAE_decoder's input
        history_m_emb = decoder_input_emb_m(history_m)
        # todo: check whether it's reasonable to get combination of his_a & his_m via concat
        decoder_input = torch.cat([history_a_emb, history_m_emb], dim=1)
        pred_next_a_type = self.decoder_type_a(decoder_input)
        pred_lambda_time2next_a = self.decoder_time2next_a(decoder_input)
        return pred_next_a_type, pred_lambda_time2next_a


















