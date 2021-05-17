# --- --- ---
# lstm.py
# Sven Giegerich / 15.05.2021
# --- --- ---

import torch
from torch import nn
from libs.losses import LossHelper
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
from typing import *


class LSTM(nn.Module):
    name = 'lstm'
    batch_first = True

    def __init__(self, d_input, d_output, d_hidden, dropout, loss_type):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.n_hidden = d_hidden
        self.dropout = dropout
        n_layers = 1 # no stacked LSTM

        dropouti = dropout
        dropoutw = dropout
        dropouto = dropout

        self.output_fn = LossHelper.get_output_activation(loss_type)

        self.lstm = LSTMwDropout(input_size=d_input, hidden_size = d_hidden, num_layers=n_layers, batch_first=True, dropoutw=dropoutw, dropouti=dropouti, dropouto=dropouto)
        self.decoder = nn.Linear(d_hidden, d_output)

    def forward(self, src):
        lstm_out, _ = self.lstm(src)
        decoder_out = self.decoder(lstm_out)
        out = self.output_fn(decoder_out)
        return out


# --- --- ---

# LSTM with weight dropout, and variational dropout in input & output layers
# Based on https://github.com/keitakurita/Better_LSTM_PyTorch

class VariationalDropout(nn.Module):
    """
    Applies the same dropout mask across the temporal dimension
    See https://arxiv.org/abs/1512.05287 for more details.
    Note that this is not applied to the recurrent activations in the LSTM like the above paper.
    Instead, it is applied to the inputs and outputs of the recurrent layer.
    """
    def __init__(self, dropout: float, batch_first: Optional[bool]=False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, seq_lens = pad_packed_sequence(x, batch_first=self.batch_first)
        max_batch_size = x.size(0 if self.batch_first else 1)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return pack_padded_sequence(x, seq_lens, batch_first=self.batch_first)
        else:
            return x 

class LSTMwDropout(nn.LSTM):
    def __init__(self, *args, dropouti: float=0.,
                 dropoutw: float=0., dropouto: float=0.,
                 batch_first=True, unit_forget_bias=True, **kwargs):
        super().__init__(*args, **kwargs, batch_first=batch_first)
        self.unit_forget_bias = unit_forget_bias
        self.dropoutw = dropoutw
        self.input_drop = VariationalDropout(dropouti,
                                             batch_first=batch_first)
        self.output_drop = VariationalDropout(dropouto,
                                              batch_first=batch_first)
        self._init_weights()

    def _init_weights(self):
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0 except for forget gate
        """
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name and self.unit_forget_bias:
                nn.init.zeros_(param.data)
                param.data[self.hidden_size:2 * self.hidden_size] = 1

    def _drop_weights(self):
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                getattr(self, name).data = \
                    torch.nn.functional.dropout(param.data, p=self.dropoutw,
                                                training=self.training).contiguous()

    def forward(self, input, hx=None):
        self._drop_weights()
        self.flatten_parameters()
        input = self.input_drop(input)
        seq, state = super().forward(input, hx=hx)
        return self.output_drop(seq), state