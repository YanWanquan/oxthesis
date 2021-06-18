# --- --- ---
# transformer.py
# Sven Giegerich / 13.05.2021
# --- --- ---

import torch
from torch import nn
import math

from libs.models.embeddings import *
from libs.losses import LossHelper
from libs.models.embeddings import TimeFeatureEmbedding

# T: window size, B: batch size, C: dim of covariates

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerEncoder(nn.Module):
    """
    The architecture is based on the paper “Attention Is All You Need”. 
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017.
    """
    name = 'transformer'
    batch_first = True

    def __init__(self, d_model, d_input, d_output, n_head, n_layer, d_hidden,
                 dropout, win_len, loss_type,
                 embedding_add='projection', embedding_pos='simple',
                 embedding_tmp=None, embedding_entity=None,  n_categories=None):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.d_input = d_input
        self.d_output = d_output
        self.n_head = n_head
        self.n_layer = n_layer
        self.d_hidden = d_hidden
        self.dropout = dropout
        self.win_len = win_len
        self.loss_type = loss_type

        self.embedding_pos = embedding_pos
        self.embedding_add = embedding_add
        self.embedding_entity = embedding_entity
        self.n_categories = n_categories
        self.embedding_tmp = embedding_tmp
        self.d_embd = 10

        self.src_mask = self.generate_causal_mask(
            self.win_len).to(device)

        # embedding ----
        if self.embedding_add == 'projection':
            self.d_embd = self.d_model
            self.projection = nn.Linear(d_input, d_model)
        elif self.embedding_add == 'separate':
            self.d_model = self.d_input + self.d_embd
        else:
            raise ValueError("Wrong parameter")

        if self.embedding_entity:
            print("> use entity embedding")
            self.entity_embedding = nn.Embedding(
                self.n_categories, self.d_embd)

        if self.embedding_tmp:
            print("> use temporal embedding")
            # tmp: feature or embedding?
            freq = 'd'
            self.temporal_embedding = TimeFeatureEmbedding(
                self.d_embd, freq=freq)

        if self.embedding_pos == 'simple':
            self.pos_embedding = SimplePositionalEncoding(
                self.d_embd, add_x=False)
        elif self.embedding_pos == 'learn':
            self.pos_embedding = nn.Embedding(
                self.win_len, self.d_embd)
        else:
            raise ValueError("Wrong parameter")

        # encoder ----
        encoder_norm = nn.LayerNorm(self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_head,
                                                   dim_feedforward=self.d_hidden, dropout=self.dropout)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, self.n_layer, encoder_norm)

        # "decoder" ----
        self.decoder = nn.Linear(self.d_model, self.d_output)
        self.output_fn = LossHelper.get_output_activation(self.loss_type)

        self.init_weights()

    def forward(self, x, x_time=None, x_entity=None, src_mask=None):
        B, L, D = x.shape
        x = x.permute(1, 0, 2)  # TransformerEncoder expects dim: L x B x C

        # embedding ----
        # ... positional
        if self.embedding_pos == 'simple':
            embedding = self.pos_embedding(torch.zeros(L)).expand(-1, B, -1)
        elif self.embedding_pos == 'learn':
            pos = torch.arange(L).to(device)
            embedding = self.pos_embedding(
                pos).unsqueeze(-2).expand(-1, B, -1)

        # ... temporal
        if self.embedding_tmp:
            temporal_encoding = self.temporal_embedding(
                x_time).permute(1, 0, 2)
            embedding = embedding + temporal_encoding

        # ... entity
        if self.embedding_entity:
            entity_encoding = self.entity_embedding(x_entity).unsqueeze(0)
            embedding = embedding + entity_encoding

        # ... embedding add type
        if self.embedding_add == 'projection':
            proj = self.projection(x)
            embedding = proj + embedding
        elif self.embedding_add == 'separate':
            embedding = torch.cat((x, embedding), dim=-1)

        if src_mask is None:
            src_mask = self.src_mask

        # encoding ----
        memory = self.encode(embedding, src_mask)

        return self.decode(memory).permute(1, 0, 2)

    def encode(self, embedding, src_mask):
        return self.encoder(embedding, mask=src_mask)

    def decode(self, memory):
        return self.output_fn(self.decoder(memory))

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" in name:
                nn.init.zeros_(p)
            elif p.dim() > 1:
                #nn.init.normal_(p, 0, 0.01)
                nn.init.xavier_uniform_(p)

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
            # [0]: attn_output, [1]: attn_output_weights
            attention_layers.append(attn_l[1])

        attn = torch.stack(attention_layers)
        attn = attn.permute(1, 0, 2, 3)
        return attn  # B x n_layer x L x L

# --- ---


class Transformer(nn.Module):
    name = "transformer"
    batch_first = True

    def __init__(self, d_model, d_input, d_output, n_head, n_layer,
                 d_hidden, dropout, device,
                 len_input_window, len_output_window, loss_type):
        super(Transformer, self).__init__()
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

    def forward(self):
        pass
