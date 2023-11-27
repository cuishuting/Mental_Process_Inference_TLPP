import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.Constants as Constants
from transformer.Layers import EncoderLayer



class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout):
        super().__init__()

        self.d_model = d_model

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cuda'))

        # event type embedding
        ##### 设置padding_idx=Constants.PAD使得补0位置的type embed全为0
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)
        
        ##### nn.ModuleList是一个储存不同module, 并自动将每个module的parameters添加到网络之中的容器
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

    def temporal_enc(self, time):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result

    def forward(self, event_time, event_type):
        """ Encode event sequences via masked self-attention. """

        ##### there is no need to use mask

        '''
        原始代码中'slf_attn_mask'的作用是mask掉future信息, 文章中的原话是: When computing the attention output S(j,:) (the j-th row of S),
        we mask all the future positions, i.e., we set Q(j, j+1), Q(j, j+1), ..., Q(j, L) to inf.
        This will avoid the softmax function from assigning dependency to events in the future
        
        在我们的setting下需要考虑所有的历史信息, 因此不需要使用mask
        '''
        
        tem_enc = self.temporal_enc(event_time)
        enc_output = self.event_emb(event_type)

        # print('----- event_time -----')
        # print(event_time)
        # print('----- event_type -----')
        # print(event_type)

        # print('----- tem_enc -----')
        # print(tem_enc)
        # print('----- enc_output -----')
        # print(enc_output)
        
        # print(enc_output.size())

        for enc_layer in self.layer_stack:
            enc_output += tem_enc ##### 直接把temporal encoding和type encoding加起来作为input

            enc_output, _ = enc_layer(enc_output)
            
            # print('----- enc_output -----')
            # print(enc_output)
            
            # print(enc_output.size())
            # exit()
            
            
        return enc_output


class RNN_layers(nn.Module):
    """
    Recurrent layers. We use the output of Transformer as the input of LSTM layers to fit the hazard rate of mental event
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, 1) ##### 因为这里只考虑一个mental维度

    def forward(self, data):

        out, (hn, cn) = self.rnn(data)        
        out = self.projection(out)
        
        return out


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            num_types, d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1):
        super().__init__()

        self.encoder = Encoder(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

        self.num_types = num_types

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_types)

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))
        
        # Recurrent layer, used to predict hazard rate
        self.rnn = RNN_layers(d_model, d_rnn)


    def forward(self, event_time, event_type):
        """
        Return the hidden representations.
        Input: event_type: batch_size*seq_len;
               event_time: batch_size*seq_len.
        Output: enc_output: batch_size*seq_len*model_dim.
        """
        enc_output = self.encoder(event_time, event_type)
        
        haz_output = self.rnn(enc_output)
        
        haz_output = torch.sigmoid(haz_output) ##### 用sigmoid函数把hazard rate限制在0-1之间

        return enc_output, haz_output
