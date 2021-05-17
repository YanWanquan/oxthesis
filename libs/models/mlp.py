# --- --- ---
# mlp.py
# Sven Giegerich / 15.05.2021
# --- --- ---

import torch
from torch import nn
from libs.losses import LossHelper

class MLP(nn.Module):

    name = 'MLP'

    def __init__(self, d_input, d_output, d_hidden, n_layer, dropout, loss_type):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.d_hidden = d_hidden
        self.n_layer = n_layer
        self.dropout = dropout
        self.loss_type= loss_type

        self.activation_fn = nn.Tanh
        self.output_fn = LossHelper.get_output_activation(loss_type)

        modules = []
        modules.append(nn.Linear(d_input, d_hidden)) # input layer
        for i in range(n_layer): # hidden layers
            modules.append(nn.Linear(d_hidden, d_hidden))
            modules.append(self.activation_fn())
            modules.append(nn.Dropout(dropout))
        modules.append(nn.Linear(d_hidden, d_output)) # output layer

        self.feedforward = torch.nn.Sequential(*modules)

    def forward(self, src):
        return self.output_fn(self.feedforward(src))







