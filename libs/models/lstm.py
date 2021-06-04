import torch
from torch import nn
import torch.nn as nn
from libs.losses import LossHelper


class LSTM(nn.Module):
    name = 'lstm'
    batch_first = True

    def __init__(self, d_input, d_output, n_layer, d_hidden, loss_type, dropout=0., dropouti=0., dropoutw=0., dropouto=0.):
        """
        Based on AWD-LSTM
        https://github.com/salesforce/awd-lstm-lm/blob/master/model.py
        Args
            d_hidden (list): (dim: n_layer)
        """
        super().__init__()

        self.d_input = d_input
        self.d_output = d_output
        self.d_hidden = d_hidden
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropoutw = dropoutw
        self.dropouto = dropouto
        self.n_layer = n_layer

        self.lstm = nn.LSTM(input_size=d_input,
                            hidden_size=d_hidden, num_layers=n_layer, batch_first=True)

        self.decoder = nn.Linear(d_hidden, d_output)
        self.output_fn = LossHelper.get_output_activation(loss_type)

    def forward(self, src):
        x, _ = self.lstm(src)
        self.lstm.flatten_parameters()
        decoder_out = self.decoder(x)
        out = self.output_fn(decoder_out)
        return out
