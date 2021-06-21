# --- --- ---
# conv_transformer.py
# Sven Giegerich / 18.05.2021
# --- --- ---

# Code comes directly from the orignal authors via mail.
# Li, S., Jin, X., Xuan, Y., Zhou, X., Chen, W., Wang, Y.X. and Yan, X., 2019.
# Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting.
# Advances in Neural Information Processing Systems, 32, pp.5243-5253.

import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from torch.distributions.normal import Normal
import copy
from torch.nn.parameter import Parameter

from libs.models.embeddings import *
from libs.losses import LossHelper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvTransformerEncoder(nn.Module):
    """
    The architecture is based on the paper Li et al. (2019).
    """

    name = 'conv_transformer'
    batch_first = True

    def __init__(self, args, d_input, n_head, n_layer,
                 d_model, d_hidden, dropout, win_len, d_output, loss_type,
                 embedding_add='projection', embedding_pos='simple',
                 embedding_tmp=None, embedding_entity=None,  n_categories=None):
        super(ConvTransformerEncoder, self).__init__()

        if n_layer > 1 and isinstance(args['q_len'], int):
            args['q_len'] = [args['q_len'] for _ in range(n_layer)]

        self.d_input = d_input
        self.d_output = d_output
        self.n_head = n_head
        self.d_hidden = d_hidden
        self.d_model = d_model
        self.n_layer = n_layer
        self.win_len = win_len
        self.loss_type = loss_type
        self.dropout = dropout

        self.embedding_pos = embedding_pos
        self.embedding_add = embedding_add
        self.embedding_entity = embedding_entity
        self.n_categories = n_categories
        self.embedding_tmp = embedding_tmp
        self.d_embd = 20

        # embedding ----
        if self.embedding_add == 'projection':
            self.d_embd = self.d_model
            self.projection = nn.Linear(d_input, d_model)
        elif self.embedding_add == 'separate':
            self.d_model = self.d_input + self.d_embd
        else:
            raise ValueError(
                f"Wrong embedding_add parameter: {self.embedding_add}")

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

        self.drop_embedding = nn.Dropout(self.dropout)

        if self.embedding_pos == 'simple':
            self.pos_embedding = SimplePositionalEncoding(
                self.d_embd, add_x=False)
        elif self.embedding_pos == 'learn':
            self.pos_embedding = nn.Embedding(
                self.win_len, self.d_embd)
        else:
            raise ValueError(
                f"Wrong embedding_pos parameter: {self.embedding_pos}")

        # block ----
        self.blocks = nn.ModuleList([
            Block(args, self.n_head, self.win_len, self.d_model,
                  self.d_hidden, self.dropout,
                  scale=args['scale_att'], q_len=args['q_len'][l])
            for l in range(self.n_layer)])

        # output ---
        self.projection_out = nn.Linear(
            self.d_model, self.d_output)
        self.output_fn = LossHelper.get_output_activation(loss_type)

        self.init_weights()

    def forward(self, enc, enc_time=None, enc_entity=None):
        # embedding ----
        x = self.embedding(enc, enc_time, enc_entity)

        # attention block ----
        attn_weights = []
        for block in self.blocks:
            x, attn_weight = block(x)
            attn_weights.append(attn_weight)

        # .. format attention weights for easier handling
        attn_weights = torch.stack(attn_weights, dim=1)

        # out ----
        out = self.projection_out(x)
        return self.output_fn(out), attn_weights

    def embedding(self, enc, enc_time, enc_entity):
        B, L, D = enc.shape

        # ... positional
        if self.embedding_pos == 'simple':
            embedding = self.pos_embedding(torch.zeros(L)).expand(-1, B, -1)
            embedding = embedding.transpose(1, 0)  # batch first
        elif self.embedding_pos == 'learn':
            pos = torch.arange(L).to(device)
            embedding = self.pos_embedding(
                pos).unsqueeze(-2).expand(-1, B, -1)
            embedding = embedding.transpose(1, 0)  # batch first

        # ... temporal
        if self.embedding_tmp:
            temporal_encoding = self.temporal_embedding(
                enc_time)
            embedding = embedding + temporal_encoding

        # ... entity
        if self.embedding_entity:
            entity_encoding = self.entity_embedding(enc_entity).unsqueeze(1)
            embedding = embedding + entity_encoding

        embedding = self.drop_embedding(embedding)

        # ... embedding add type
        if self.embedding_add == 'projection':
            proj = self.projection(enc)
            embedding = proj + embedding
        elif self.embedding_add == 'separate':
            embedding = torch.cat((enc, embedding), dim=-1)

        return embedding

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" in name:
                nn.init.zeros_(p)
            elif p.dim() > 1:
                # nn.init.normal_(p, 0, 0.01)
                nn.init.xavier_uniform_(p)


class Attention(nn.Module):

    def __init__(self, args, n_head, n_embd, win_len, scale, q_len):
        super(Attention, self).__init__()

        print(f"> use convolutional kernel {q_len}")
        if(args['sparse']):
            print(f'> activate log sparse for layer')
            mask = self.log_mask(win_len, args['sub_len'])
        else:
            mask = torch.tril(torch.ones(win_len, win_len)
                              ).view(1, 1, win_len, win_len)

        self.register_buffer('mask_tri', mask)
        self.n_head = n_head
        self.split_size = n_embd*self.n_head
        self.scale = scale
        self.q_len = q_len
        self.query_key = nn.Conv1d(n_embd, n_embd*n_head*2, self.q_len)
        self.value = Conv1D(n_embd*n_head, 1, n_embd)
        self.c_proj = Conv1D(n_embd, 1, n_embd*self.n_head)
        self.attn_dropout = nn.Dropout(args['attn_pdrop'])
        self.resid_dropout = nn.Dropout(args['resid_pdrop'])

    def log_mask(self, win_len, sub_len):
        mask = torch.zeros((win_len, win_len), dtype=torch.float)
        for i in range(win_len):
            mask[i] = self.row_mask(i, sub_len, win_len)
        return mask.view(1, 1, mask.size(0), mask.size(1))

    def row_mask(self, index, sub_len, win_len):
        """
        Remark:
        1 . Currently, dense matrices with sparse multiplication are not supported by Pytorch. Efficient implementation
            should deal with CUDA kernel, which we haven't implemented yet.

        2 . Our default setting here use Local attention and Restart attention.

        3 . For index-th row, if its past is smaller than the number of cells the last
            cell can attend, we can allow current cell to attend all past cells to fully
            utilize parallel computing in dense matrices with sparse multiplication."""
        log_l = math.ceil(np.log2(sub_len))
        mask = torch.zeros((win_len), dtype=torch.float)
        if((win_len//sub_len)*2*(log_l) > index):
            mask[:(index+1)] = 1
        else:
            while(index >= 0):
                if((index - log_l+1) < 0):
                    mask[:index] = 1
                    break
                mask[index-log_l+1:(index+1)] = 1  # Local attention
                for i in range(0, log_l):
                    new_index = index - log_l + 1 - 2**i
                    if((index-new_index) <= sub_len and new_index >= 0):
                        mask[new_index] = 1
                index -= sub_len
        return mask

    def attn(self, query, key, value):

        pre_att = torch.matmul(query, key)
        if self.scale:
            pre_att = pre_att / math.sqrt(value.size(-1))
        mask = self.mask_tri[:, :, :pre_att.size(-2), :pre_att.size(-1)]
        pre_att = pre_att * mask + -1e9 * (1 - mask)
        pre_att = nn.Softmax(dim=-1)(pre_att)
        att_weights = pre_att
        pre_att = self.attn_dropout(pre_att)
        attn = torch.matmul(pre_att, value)

        return (attn, att_weights)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):
        value = self.value(x)
        qk_x = nn.functional.pad(x.permute(0, 2, 1), pad=(self.q_len-1, 0))
        query_key = self.query_key(qk_x).permute(0, 2, 1)
        query, key = query_key.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        attn, attn_weights = self.attn(query, key, value)
        attn = self.merge_heads(attn)
        attn = self.c_proj(attn)
        attn = self.resid_dropout(attn)
        return attn, attn_weights


class Conv1D(nn.Module):
    def __init__(self, out_dim, rf, in_dim):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.out_dim = out_dim
        if rf == 1:
            w = torch.empty(in_dim, out_dim)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(out_dim))
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.out_dim,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class Block(nn.Module):
    def __init__(self, args, n_head, win_len, d_model, d_hidden, dropout, scale, q_len):
        super(Block, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.attn = Attention(args, n_head, d_model, win_len, scale, q_len)

        self.linear1 = nn.Linear(d_model, d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hidden, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn, attn_weight = self.attn(x)
        x = x + self.dropout1(attn)
        x = self.norm1(x)

        x2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x, attn_weight
