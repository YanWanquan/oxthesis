# --- --- ---
# transformer.py
# Sven Giegerich / 13.05.2021
# --- --- ---

import torch
from torch import nn
import math

from libs.models.embeddings import *
from libs.losses import LossHelper

# T: window size, B: batch size, C: dim of covariates


class TransformerEncoder(nn.Module):
    """
    The architecture is based on the paper “Attention Is All You Need”. 
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017.
    """
    name = 'transformer'
    batch_first = True

    def __init__(self, d_model, d_input, d_output, n_head, n_layer, d_hidden, dropout, device,
                 len_input_window, len_output_window, loss_type):
        super(TransformerEncoder, self).__init__()
        self.len_input_window = len_input_window
        self.len_output_window = len_output_window

        self.src_mask = self.generate_causal_mask(
            self.len_input_window).to(device)

        # for simplicity give encoder and decoder the same size
        d_hidden_encoder = d_hidden
        n_layer_encoder = n_layer

        # preprocess
        # changed embedding to linear layer
        self.embedding = nn.Linear(d_input, d_model)
        self.pos_encoder = SimplePositionalEncoding(d_model)
        # encoder
        encoder_norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head,
                                                   dim_feedforward=d_hidden_encoder, dropout=dropout)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, n_layer_encoder, encoder_norm)
        # "decoder"
        self.decoder = nn.Linear(d_model, d_output)
        self.output_fn = LossHelper.get_output_activation(loss_type)

        self.init_weights()

    def forward(self, src, src_mask=None):
        src = src.permute(1, 0, 2)  # TransformerEncoder expects dim: T x B x C

        if src_mask is None:
            src_mask = self.src_mask

        memory = self.encode(src, src_mask)
        return self.decode(memory).permute(1, 0, 2)

    def encode(self, src, src_mask):
        return self.encoder(self.pos_encoder(self.embedding(src)), mask=src_mask)

    def decode(self, memory):
        return self.output_fn(self.decoder(memory))

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def generate_causal_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def get_attention(self, src, src_mask=None):
        # currently self_attn just returns the output of multi-head attn..
        # ... and not the heads separately
        # TBD

        src = src.permute(1, 0, 2)

        if src_mask is None:
            src_mask = self.src_mask

        emb = self.pos_encoder(self.embedding(src))

        attention_layers = []
        for layer in self.encoder.layers:
            attn_l = layer.self_attn(
                emb, emb, emb, attn_mask=src_mask)
            attention_layers.append(attn_l[1])

        attn = torch.stack(attention_layers)
        if len(attention_layers) == 1:
            attn = attn.unsqueeze(0)
        attn = attn.permute(1, 0, 2, 3)
        return attn  # B x n_layer x L x L
