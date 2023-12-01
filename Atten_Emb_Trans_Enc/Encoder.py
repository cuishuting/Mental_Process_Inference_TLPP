import torch.nn as nn
import torch
import math


class Encoder(nn.Module):
    def __init__(self, d_emb, num_action_types):
        super().__init__()
        self.d_emb = d_emb  # d_emb is encoding dim for each action event (for both temporal and type encoding)
        self.num_action_types = num_action_types
        self.position_vec = torch.tensor([])

    def forward(self, data):
        pass
        # data: org_action_events_list