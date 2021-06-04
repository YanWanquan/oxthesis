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

    def __init__(self, d_input, d_output, n_layer, d_hidden, loss_type, dropout=0.,
                 dropouti=0., dropoutw=0., dropouto=0., variational_dropout=True):
        """
        Based on AWD-LSTM
        https://github.com/salesforce/awd-lstm-lm/blob/master/model.py
        """
        super().__init__()

        # tmp
        self.pack = True
        self.last = False

        if not isinstance(d_hidden, list):
            d_hidden = [d_hidden]

        self.d_input = d_input
        self.d_output = d_output
        self.d_hidden = d_hidden
        self.dropout = dropout
        self.dropouti = dropouti  # input dropout
        self.dropoutw = dropoutw  # recurrent dropout
        self.dropouto = dropouto  # output dropout
        self.n_layer = n_layer
        self.variational_dropout = variational_dropout
        if dropout == .0 and dropouto != .0:
            # rnn output dropout (of the last RNN layer)
            self.dropout = self.dropouto

        self.lockdrop = LockedDropout()

        self.lstms = [nn.LSTM(input_size=d_input if l == 0 else d_hidden[l - 1],
                              hidden_size=d_hidden[l], num_layers=1, dropout=0,
                              batch_first=True) for l in range(n_layer)]

        if dropoutw > 0:
            self.lstms = [WeightDrop(
                lstm, ['weight_hh_l0', 'weight_ih_l0'], dropout=dropoutw, variational=variational_dropout) for lstm in self.lstms]

        print(self.lstms)

        self.lstms = nn.ModuleList(self.lstms)
        self.decoder = nn.Linear(d_hidden[-1], d_output)
        self.output_fn = LossHelper.get_output_activation(loss_type)

    def forward(self, src, hidden=None):
        batch_size, seq_length, feat_size = src.size()
        emb = self.lockdrop(src, self.dropouti)

        if hidden is None:
            hidden = self.init_hidden(batch_size)

        raw_output = emb  # input first layer
        new_hidden = []
        raw_outputs = []
        outputs = []

        """
        `batch_first = True` use of PyTorch RNN module
        shapes of input and output tensors
        -----------------------------------------------
        output, (hn, cn) = rnn(input, (h0, c0))
        -----------------------------------------------
        input: (batch_size, seq_len, input_size)
        h0: (num_layers * num_directions, batch_size, feature_size)
        c0: (num_layers * num_directions, batch_size, feature_size)
        -----------------------------------------------
        output: (batch_size, seq_len, num_directions * hidden_size])
        contains the output features `(h_t)` from the last layer of the LSTM, for each `t`
        hn: (num_layers * num_directions, batch_size, feature_size)
        contains the hidden state for `t = seq_len`
        cn: (num_layers * num_directions, batch_size, feature_size)
        contains the cell state for `t = seq_len`
        """

        for l, lstm in enumerate(self.lstms):
            # calculate hidden states and output from the l RNN layer
            raw_output, new_h = lstm(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.n_layer - 1:
                # apply dropout to the output of the l-th RNN layer (dropouto)
                raw_output = self.lockdrop(raw_output, self.dropouto)
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
        return [(weight.new(1, bsz, self.d_hidden[l]).zero_(),
                 weight.new(1, bsz, self.d_hidden[l]).zero_())
                for l in range(self.n_layer)]


# --- --- ---

# --- ---
# weight dropout
# based on https://github.com/salesforce/awd-lstm-lm/blob/master/weight_drop.py
# and https://github.com/mourga/variational-lstm/blob/master/weight_drop.py
# --- ---

class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=True):
        """
        Dropout class that is paired with a torch module to make sure that the SAME mask
        will be sampled and applied to ALL timesteps.
        :param module: nn. module (e.g. nn.Linear, nn.LSTM)
        :param weights: which weights to apply dropout (names of weights of module)
        :param dropout: dropout to be applied
        :param variational: if True applies Variational Dropout, if False applies DropConnect (different masks!!!)
        """
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        """
        Smerity code I don't understand.
        """
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        """
        This function renames each 'weight name' to 'weight name' + '_raw'
        (e.g. weight_hh_l0 -> weight_hh_l0_raw)
        :return:
        """
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('> Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        """
        This function samples & applies a dropout mask to the weights of the recurrent layers.
        Specifically, for an LSTM, each gate has
        - a W matrix ('weight_ih') that is multiplied with the input (x_t)
        - a U matrix ('weight_hh') that is multiplied with the previous hidden state (h_t-1)
        We sample a mask (either with Variational Dropout or with DropConnect) and apply it to
        the matrices U and/or W.
        The matrices to be dropped-out are in self.weights.
        A 'weight_hh' matrix is of shape (4*nhidden, nhidden)
        while a 'weight_ih' matrix is of shape (4*nhidden, ninput).
        **** Variational Dropout ****
        With this method, we sample a mask from the tensor (4*nhidden, 1) PER ROW
        and expand it to the full matrix.
        **** DropConnect ****
        With this method, we sample a mask from the tensor (4*nhidden, nhidden) directly
        which means that we apply dropout PER ELEMENT/NEURON.
        :return:
        """
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None

            if self.variational:
                #######################################################
                # Variational dropout (as proposed by Gal & Ghahramani)
                #######################################################
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda:
                    mask = mask.cuda()
                mask = torch.nn.functional.dropout(
                    mask, p=self.dropout, training=self.training)  # True or tmp?
                w = mask.expand_as(raw_w) * raw_w
            else:
                #######################################################
                # DropConnect (as presented in the AWD paper)
                #######################################################
                w = torch.nn.functional.dropout(
                    raw_w, p=self.dropout, training=self.training)

            # Fix
            # see: https://github.com/salesforce/awd-lstm-lm/issues/79
            # in evaluation mode dropout returns a parameter instead of a tensor
            if not self.training:
                w = w.data

            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        self.module.flatten_parameters()
        return self.module.forward(*args)


# --- ---
# locked dropout
# based on https://github.com/salesforce/awd-lstm-lm/blob/master/locked_dropout.py
# --- ---

class LockedDropout(nn.Module):
    """
    This function applies dropout to the input tensor x.
    The shape of the tensor x in our implementation is (batch_size, seq_len, feature_size)
    (contrary to Merity's AWD that uses (seq_len, batch_size, feature_size)).
    So, we sample a mask from the 'feature_size' dimension,
    but a different mask for each 'batch_size' dimension,
    and expand it in the 'sequence_length' dimension so that
    we apply the SAME mask FOR EACH TIMESTEP of the RNN (= 'seq_len' dim.).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, dropout):
        if not self.training or dropout == 0.0:
            return x
        batch_size, seq_length, feat_size = x.size()
        m = x.data.new(batch_size, 1, feat_size).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x
