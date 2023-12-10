import torch.nn as nn


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        """x can be one of two types: (1)tuple, when sublayer is encoder/decoder layer (2)tensor,when sublayer is ffw"""
        """when it comes to (1), we add k/v as the add term in residual operation"""
        if type(x).__name__ == 'tuple':
            return x[1] + self.dropout(self.norm(sublayer(x)))  # todo: check layernorm before or after residual?
        else:
            return x + self.dropout(self.norm(sublayer(x)))
