# --- --- ---
# lstm.py
# Sven Giegerich / 15.05.2021
# --- --- ---

import torch
from torch import nn
from torch.nn import Parameter
from functools import wraps
from libs.losses import LossHelper
from torch.autograd import Variable


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
        if not isinstance(d_hidden, list):
            d_hidden = [d_hidden]
        self.d_input = d_input
        self.d_output = d_output
        self.d_hidden = d_hidden
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropoutw = dropoutw
        self.dropouto = dropouto
        self.n_layer = n_layer
        if dropout == .0 and dropouto != .0:
            # rnn output dropout (of the last RNN layer)
            self.dropout = self.dropouto

        self.lockdrop = LockedDropout()

        self.lstms = [nn.LSTM(input_size=d_input if l == 0 else d_hidden[l - 1], hidden_size=d_hidden[l], num_layers=n_layer, dropout=0,
                              batch_first=True) for l in range(n_layer)]

        if dropoutw > 0:
            self.lstms = [WeightDrop(
                l, ['weight_hh_l0', 'weight_ih_l0'], dropout=dropoutw) for l in self.lstms]

        self.lstms = nn.ModuleList(self.lstms)
        self.decoder = nn.Linear(d_hidden[-1], d_output)
        self.output_fn = LossHelper.get_output_activation(loss_type)

    def forward(self, src, hidden=None):
        batch_size, seq_length, feat_size = src.size()
        emb = self.lockdrop(src, self.dropouti)
        raw_output = emb

        if hidden is None:
            hidden = self.init_hidden(batch_size)

        new_hidden = []
        raw_outputs = []
        outputs = []

        for l, lstm in enumerate(self.lstms):
            # calculate hidden states and output from the l RNN layer
            raw_output, new_h = lstm(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.n_layer - 1:
                # apply dropout to the output of the l-th RNN layer (dropouto)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                # save 'dropped-out outputs' in a list
                outputs.append(raw_output)

        hidden = new_hidden
        # Dropout to the output of the last RNN layer (dropout)
        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)
        result = output

        decoder_out = self.decoder(result)
        out = self.output_fn(decoder_out)
        return out

    def init_hidden(self, bsz):
        """
        Based on https://github.com/mourga/variational-lstm/blob/ce7f9affc40061405f044f2443914ffcbdd8eb3b/rnn_module.py#L9.
        Initialise the hidden and cell state (h0, c0) for the first timestep (t=0).
        Both h0, c0 are of shape (num_layers * num_directions, batch_size, hidden_size)
        :param bsz: batch size
        :return:
        """
        weight = next(self.parameters()).data
        weight.new(1, bsz, self.d_hidden[0]).zero_()

        return [(weight.new(1, bsz, self.d_hidden[l]).zero_(),
                 weight.new(1, bsz, self.d_hidden[l]).zero_())
                for l in range(self.n_layer)]


# --- --- ---

# --- ---
# weight dropout
# based on https://github.com/salesforce/awd-lstm-lm/blob/master/weight_drop.py
# --- ---

class WeightDrop(nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda:
                    mask = mask.cuda()
                mask = torch.nn.functional.dropout(
                    mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = torch.nn.functional.dropout(
                    raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


# --- ---
# locked dropout
# based on https://github.com/salesforce/awd-lstm-lm/blob/master/locked_dropout.py
# --- ---

class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x
