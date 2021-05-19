# --- --- ---
# conv_transformer.py
# Sven Giegerich / 18.05.2021
# --- --- ---

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

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def swish(x):
    return x * torch.sigmoid(x)

ACT_FNS = {
    'relu': nn.ReLU(),
    'swish': swish,
    'gelu': gelu
}

class Attention(nn.Module):
    def __init__(self,args,n_head,n_embd, win_len, scale,q_len):
        super(Attention, self).__init__()

        if(args['sparse']):
            print('Activate log sparse!')
            mask = self.log_mask(win_len,args['sub_len'])
        else:
            mask = torch.tril(torch.ones(win_len, win_len)).view(1, 1, win_len, win_len)

        self.register_buffer('mask_tri', mask)
        self.n_head = n_head
        self.split_size = n_embd*self.n_head
        self.scale = scale
        self.q_len = q_len
        self.query_key = nn.Conv1d(n_embd, n_embd*n_head*2, self.q_len)
        self.value = Conv1D(n_embd*n_head,1,n_embd)
        self.c_proj = Conv1D(n_embd, 1, n_embd*self.n_head)
        self.attn_dropout = nn.Dropout(args['attn_pdrop'])
        self.resid_dropout = nn.Dropout(args['resid_pdrop'])

    def log_mask(self,win_len,sub_len):
        mask = torch.zeros((win_len,win_len),dtype=torch.float)
        for i in range(win_len):
            mask[i] = self.row_mask(i,sub_len,win_len)
        return mask.view(1,1,mask.size(0),mask.size(1))

    def row_mask(self,index,sub_len,win_len):
        """
        Remark:
        1 . Currently, dense matrices with sparse multiplication are not supported by Pytorch. Efficient implementation
            should deal with CUDA kernel, which we haven't implemented yet.

        2 . Our default setting here use Local attention and Restart attention.

        3 . For index-th row, if its past is smaller than the number of cells the last
            cell can attend, we can allow current cell to attend all past cells to fully
            utilize parallel computing in dense matrices with sparse multiplication."""
        log_l = math.ceil(np.log2(sub_len))
        mask = torch.zeros((win_len),dtype=torch.float)
        if((win_len//sub_len)*2*(log_l)>index):
            mask[:(index+1)]=1
        else:
            while(index>=0):
                if((index - log_l+1)<0):
                    mask[:index] = 1
                    break
                mask[index-log_l+1:(index+1)]=1 # Local attention
                for i in range(0,log_l):
                    new_index = index - log_l + 1 -2**i
                    if((index-new_index)<=sub_len and new_index >=0):
                        mask[new_index]=1
                index -= sub_len
        return mask

    def attn(self, query, key, value):

        pre_att = torch.matmul(query, key)
        if self.scale:
            pre_att = pre_att / math.sqrt(value.size(-1))
        mask = self.mask_tri[:, :, :pre_att.size(-2), :pre_att.size(-1)]
        pre_att = pre_att * mask + -1e9 * (1 - mask)
        pre_att = nn.Softmax(dim=-1)(pre_att)
        pre_att = self.attn_dropout(pre_att)
        attn = torch.matmul(pre_att, value)

        return attn

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
        qk_x = nn.functional.pad(x.permute(0,2,1), pad=(self.q_len-1,0))
        query_key = self.query_key(qk_x).permute(0,2,1)
        query, key = query_key.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        attn = self.attn(query, key, value)
        attn = self.merge_heads(attn)
        attn = self.c_proj(attn)
        attn = self.resid_dropout(attn)
        return attn

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

class LayerNorm(nn.Module):
    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

    def __init__(self, n_embd, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_embd))
        self.b = nn.Parameter(torch.zeros(n_embd))
        self.e = e

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = (x - mu).pow(2).mean(-1, keepdim=True)
        x = (x - mu) / torch.sqrt(sigma + self.e)
        return self.g * x + self.b

class MLP(nn.Module):
    def __init__(self,n_state,n_embd,acf='relu'):
        super(MLP, self).__init__()
        n_embd = n_embd
        self.c_fc = Conv1D(n_state, 1, n_embd)
        self.c_proj = Conv1D(n_embd, 1, n_state)
        self.act =ACT_FNS[acf]
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        hidden1 = self.act(self.c_fc(x))
        hidden2 = self.c_proj(hidden1)
        return self.dropout(hidden2)


class Block(nn.Module):
    def __init__(self,args,n_head, win_len,n_embd, scale,q_len):
        super(Block, self).__init__()
        n_embd = n_embd
        self.attn = Attention(args,n_head, n_embd, win_len, scale,q_len)
        self.ln_1 = LayerNorm(n_embd)
        self.mlp = MLP(4 * n_embd, n_embd)
        self.ln_2 = LayerNorm(n_embd)

    def forward(self, x):
        attn = self.attn(x)
        ln1 = self.ln_1(x + attn)
        mlp = self.mlp(ln1)
        hidden = self.ln_2(ln1 + mlp)
        return hidden

class ConvTransformerEncoder(nn.Module):
    """
    The architecture is based on the paper "...".
    """

    name = 'conv_transformer'
    batch_first = True

    """ Transformer model """
    def __init__(self,args,d_input, n_head, n_layer, d_model, win_len, d_output, loss_type, seq_num=None):
        super(ConvTransformerEncoder, self).__init__()
        # SVEN
        # d_model is n_embd here
        # TMP: deactivated sequence embedding
        self.d_input = d_input
        self.n_head = n_head
        #self.seq_num = seq_num
        self.n_embd = d_model
        self.n_layer = n_layer
        self.win_len = win_len
        self.d_output = d_output
        #self.id_embed = nn.Embedding(seq_num,n_embd)
        self.po_embed = nn.Embedding(self.win_len,self.n_embd)
        self.drop_em = nn.Dropout(args['embd_pdrop'])
        block = Block(args,n_head,win_len,self.n_embd+self.d_input, scale=args['scale_att'],q_len=args['q_len'])
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(self.n_layer)])

        # SVEN
        self.decoder = nn.Linear(self.d_input + self.n_embd, self.d_output)
        self.output_fn = LossHelper.get_output_activation(loss_type)

        self.init_weights()

    def forward(self, x, series_id=None):
        #id_embedding = self.id_embed(series_id)
        length = x.size(1) # (Batch_size,length,input_dim)
        position = torch.arange(length).type(torch.long).to(device)
        po_embedding = self.po_embed(position)
        batch_size = x.size(0)
        embedding_sum = torch.zeros(batch_size,length,self.n_embd).to(device)
        embedding_sum[:] = po_embedding
        embedding_sum = embedding_sum # + id_embedding.unsqueeze(1)
        x = torch.cat((x,embedding_sum),dim=2)
        for block in self.blocks:
            x = block(x)

        # SVEN
        pred = self.decoder(x)
        return self.output_fn(pred)

    # SVEN
    def init_weights(self):
        #nn.init.normal_(self.id_embed.weight, std=0.02)
        nn.init.normal_(self.po_embed.weight, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, 0,0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    