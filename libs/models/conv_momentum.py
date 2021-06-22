# --- --- ---
# conv_momentum.py
# Sven Giegerich / 16.06.2021
# --- --- ---

import torch
from torch import nn
from torch.nn import Parameter
from functools import wraps
from libs.losses import LossHelper
from torch.autograd import Variable
import torch.nn.functional as F

from libs.models.lstm_dropout import LockedDropout, WeightDrop
from libs.models.embeddings import DataEmbedding, MomentumEmbedding, SimplePositionalEncoding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvMomentum(nn.Module):
    name = 'conv_momentum'
    batch_first = True

    def __init__(
            self, d_input, d_output, n_layer_lstm, d_hidden, loss_type,
            len_input_window, n_head, n_categories,
            embedding_add, embedding_entity,
            n_layer_attn=1, dropout=0., dropouti=0., dropoutw=0.,
            dropouto=0., variational_dropout=True):
        super().__init__()
        if not isinstance(d_hidden, list):
            d_hidden = [d_hidden]

        # Params ----
        # ... LSTM ----
        self.d_input = d_input
        self.d_output = d_output
        self.d_hidden = d_hidden
        self.dropout = dropout
        self.dropouti = dropouti  # input dropout
        self.dropoutw = dropoutw  # recurrent dropout
        self.dropouto = dropouto  # output dropout
        self.n_layer_lstm = n_layer_lstm
        self.variational_dropout = variational_dropout
        if dropout == .0 and dropouto != .0:
            # rnn output dropout (of the last RNN layer)
            self.dropout = self.dropouto

        # ... embedding ----
        self.embedding_add = embedding_add
        self.embedding_entity = embedding_entity
        self.n_categories = n_categories
        self.embedding_tmp = False
        self.d_embd = 10

        # ... attention ----
        self.len_input_window = len_input_window
        self.n_head = n_head
        self.n_layer_attn = n_layer_attn

        # --- --- ---

        # Modules ----
        self.d_hidden_lstm_in = self.d_hidden[0]
        self.d_hidden_lstm_out = self.d_hidden[-1]

        # ... embedding ----
        if self.embedding_add == 'projection':
            self.d_embd = self.d_hidden_lstm_in
            self.projection = nn.Linear(d_input, self.d_hidden_lstm_in)
            self.d_input_lstm = self.d_hidden_lstm_in
        elif self.embedding_add == 'separate':
            self.d_input_lstm = self.d_input

        else:
            raise ValueError(
                f"Wrong embedding_add parameter: {self.embedding_add}")

        if self.embedding_entity:
            print("> use entity embedding")
            self.entity_embedding = nn.Embedding(
                self.n_categories, self.d_embd)

        self.drop_embedding = nn.Dropout(self.dropout)

        # ... gated residual networks (GRNs) ----
        self.grn_init_hidden_lstm = GatedResidualNetwork(
            self.d_embd, self.d_hidden_lstm_in, self.d_hidden_lstm_in, dropout=self.dropout)
        self.grn_init_cell_lstm = GatedResidualNetwork(
            self.d_embd, self.d_hidden_lstm_in, self.d_hidden_lstm_in, dropout=self.dropout)

        self.gate_post_lstm = GateAddNorm(
            self.d_hidden_lstm_out, self.d_input_lstm)
        self.grn_post_lstm = GatedResidualNetwork(
            self.d_hidden_lstm_out, self.d_hidden_lstm_out, self.d_hidden_lstm_out, dropout=0)

        self.gate_post_attn = GateAddNorm(
            self.d_hidden_lstm_out, self.d_hidden_lstm_out)
        self.grn_post_attn = GatedResidualNetwork(
            self.d_hidden_lstm_out, self.d_hidden_lstm_out, self.d_hidden_lstm_out, dropout=0)

        # ... LSTM ----
        self.lockdrop = LockedDropout()
        self.lstms = [nn.LSTM(input_size=self.d_input_lstm if l == 0 else d_hidden[l - 1],
                              hidden_size=d_hidden[l], num_layers=1, dropout=0,
                              batch_first=True) for l in range(n_layer_lstm)]
        if dropoutw > 0:
            self.lstms = [WeightDrop(
                lstm, ['weight_hh_l0', 'weight_ih_l0'], dropout=dropoutw, variational=variational_dropout) for lstm in self.lstms]
        self.lstms = nn.ModuleList(self.lstms)

        # ... Attention ----
        self.src_mask = self.generate_causal_mask(
            self.len_input_window).to(device)

        self.attn = nn.MultiheadAttention(
            self.d_hidden_lstm_out, self.n_head, self.dropout)
        self.attn_norm = nn.LayerNorm(self.d_hidden_lstm_out)

        # ... Decoder ----
        self.output = nn.Linear(self.d_hidden_lstm_out, d_output)
        self.output_fn = LossHelper.get_output_activation(loss_type)

    # ----

    def forward(self, x_var, x_static, src_mask=None):
        # input x_time is not used

        # Embedding ----
        embd_var, embd_static = self.embedding(x_var.permute(1, 0, 2),
                                               x_static)
        if embd_var is None:
            embd_var = x_var
        else:
            embd_var = embd_var.permute(1, 0, 2)
            embd_static = embd_static.permute(1, 0, 2)

        # LSTM ----
        init_hidden_lstm = self.grn_init_hidden_lstm(
            embd_static).expand(self.n_layer_lstm, -1, -1)  # n_layer x B x d_embed
        init_cell_lstm = self.grn_init_cell_lstm(
            embd_static).expand(self.n_layer_lstm, -1, -1)  # same

        out_lstm = self.encoder_lstm(
            embd_var, hidden=(init_hidden_lstm, init_cell_lstm))

        # GRN (I) ----
        pre_attn = self.gate_post_lstm(out_lstm, embd_var)
        pre_attn = self.grn_post_lstm(pre_attn)

        # Attn ----
        # tmp: why no pos embedding?

        if src_mask is None:
            src_mask = self.src_mask
        pre_attn = pre_attn.permute(1, 0, 2)
        out_attn, attn_weights = self.attn(
            pre_attn, pre_attn, pre_attn, attn_mask=src_mask)
        out_attn = out_attn.permute(1, 0, 2)
        pre_attn = pre_attn.permute(1, 0, 2)
        out_attn = pre_attn + self.attn_norm(out_attn)

        # GRN (II) ----
        post_attn = self.gate_post_attn(out_attn, pre_attn)
        post_attn = self.grn_post_attn(post_attn)

        # Decoder
        out = self.output(post_attn)
        out = self.output_fn(out)
        return out

    def embedding(self, enc, enc_entity=None):
        L, B, D = enc.shape

        # ... entity
        if self.embedding_entity:
            entity_encoding = self.entity_embedding(enc_entity).unsqueeze(0)
            embedding = entity_encoding

        embedding = self.drop_embedding(embedding)

        # ... embedding add type
        if self.embedding_add == 'projection':
            proj = self.projection(enc)
            return ((proj + embedding), embedding)
        elif self.embedding_add == 'separate':
            return None, embedding
        else:
            raise ValueError("Embedding add not supported!")

    def generate_causal_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encoder_lstm(self, emb, hidden=None):
        batch_size, seq_length, dim_emb = emb.size()

        if hidden is None:
            hidden = self.init_lstm_hidden(batch_size)
        else:
            hidden = [(hidden[0], hidden[1]) for _ in range(self.n_layer_lstm)]

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
            if l != self.n_layer_lstm - 1:
                # apply dropout to the output of the l-th RNN layer (dropouto)
                raw_output = self.lockdrop(raw_output, self.dropouto)
                # save 'dropped-out outputs' in a list
                outputs.append(raw_output)

        hidden = new_hidden

        # Dropout to the output of the last RNN layer (dropout)
        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        return output

    def init_lstm_hidden(self, bsz):
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
                for l in range(self.n_layer_lstm)]

# --- ---


class GatedLinearUnit(nn.Module):
    def __init__(self, d_inp, d_hidden=None, dropout=0.):
        super().__init__()

        self.d_inp = d_inp
        self.d_hidden = d_hidden or d_inp
        self.dropout = dropout

        self.dropout = nn.Dropout(self.dropout)
        self.dense = nn.Linear(self.d_inp, self.d_hidden * 2)  # A & B for GLU

        self.init_weights()

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = nn.functional.glu(x, dim=-1)
        return x

    def init_weights(self):
        for n, p in self.named_parameters():
            if "bias" in n:
                torch.nn.init.zeros_(p)
            elif "dense" in n:
                torch.nn.init.xavier_uniform_(p)


class GateAddNorm(nn.Module):
    def __init__(self, d_inp, d_skip, dropout=0.):
        super().__init__()
        self.d_inp = d_inp
        self.dropout = dropout
        self.d_skip = d_skip

        if self.d_skip != self.d_inp:
            self.dense_skip = nn.Linear(self.d_skip, self.d_inp)

        self.glu = GatedLinearUnit(self.d_inp, dropout=self.dropout)
        self.norm = nn.LayerNorm(self.d_inp)

    def forward(self, x, skip):
        if self.d_skip != self.d_inp:
            skip = self.dense_skip(skip)

        x = self.glu(x)
        out = self.norm(x + skip)
        return out


class GatedResidualNetwork(nn.Module):
    def __init__(self, d_inp, d_hidden, d_output=None, dropout=0.):
        super().__init__()
        # tmp: no context vector
        self.d_inp = d_inp
        self.d_hidden = d_hidden or d_inp
        self.d_output = d_output or d_inp
        self.dropout = dropout

        self.dense1 = nn.Linear(self.d_inp, self.d_hidden)
        self.elu = nn.ELU()
        self.dense2 = nn.Linear(self.d_hidden, self.d_hidden)
        self.glu = GatedLinearUnit(self.d_hidden, self.d_output, self.dropout)

        self.norm = nn.LayerNorm(self.d_output)
        self.dropout = nn.Dropout(self.dropout)

        self.init_weights()

    def forward(self, x):
        x = self.elu(self.dense1(x))
        x = self.dense2(x)
        x = self.glu(x)
        x = self.norm(self.dropout(x))
        return x

    def init_weights(self):
        for n, p in self.named_parameters():
            if "bias" in n:
                torch.nn.init.zeros_(p)
            elif "dense1" in n:
                # see https://stats.stackexchange.com/questions/229885/whats-the-recommended-weight-initialization-strategy-when-using-the-elu-activat
                torch.nn.init.kaiming_normal_(
                    p, a=0, mode="fan_in", nonlinearity="relu")
            elif "dense2" in n:
                torch.nn.init.xavier_uniform_(p)
