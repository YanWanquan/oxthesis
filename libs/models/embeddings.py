# --- --- ---
# embeddings.py
# Sven Giegerich / 13.05.2021
# --- --- ---

import torch
from torch import nn
import math

class SimplePositionalEncoding(nn.Module):
    """
    This positional encoding is direclty based on the paper “Attention Is All You Need” using sin & cos addition.
    For more see either the paper,
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017
    or this blog article: https://kazemnejad.com/blog/transformer_architecture_positional_encoding/.
    """

    def __init__(self, d_model, max_len=5000):
        super(SimplePositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe) # exclude from model's paramters

    def forward(self, x):
        return x + self.pe[:x.size(0), :]