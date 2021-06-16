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
            len_input_window, n_head, d_model, d_attn_hidden, n_categories,
            embed_type='fixed', freq='d', use_embed=True,
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

        # TMP
        self.use_embed = use_embed
        self.embed_type = embed_type
        self.n_categories = n_categories

        # ... ATTN ----
        self.len_input_window = len_input_window
        self.n_head = n_head
        self.d_model = d_model
        self.d_attn_hidden = d_attn_hidden
        self.n_layer_attn = n_layer_attn

        # --- ---
        # --- ---

        # Modules ----
        self.d_hidden_lstm_in = self.d_hidden[0]
        self.d_hidden_lstm_out = self.d_hidden[-1]

        if self.use_embed:
            if self.embed_type == "simple":
                print("> Use simple positional encoding")
                self.enc_simple_embedding = nn.Linear(d_input, d_model)
                self.pos_encoder = SimplePositionalEncoding(d_model=d_model)
            elif self.embed_type == 'momentum':
                print("> Use momentum embedding")
                self.enc_momentum_embedding = MomentumEmbedding(
                    c_in=d_input, d_model=d_model, n_categories=n_categories, embed_type='timeF', freq=freq, dropout=dropout, only_encoder=True)
            else:
                print("> Use data embedding")
                self.enc_data_embedding = DataEmbedding(
                    c_in=d_input, d_model=d_model, embed_type=embed_type, freq=freq, dropout=dropout, only_encoder=True)

            d_input_lstm = d_hidden[0]
        else:
            d_input_lstm = d_input

        # ... LSTM ----
        self.lockdrop = LockedDropout()
        self.lstms = [nn.LSTM(input_size=d_input_lstm if l == 0 else d_hidden[l - 1],
                              hidden_size=d_hidden[l], num_layers=1, dropout=0,
                              batch_first=True) for l in range(n_layer_lstm)]
        if dropoutw > 0:
            self.lstms = [WeightDrop(
                lstm, ['weight_hh_l0', 'weight_ih_l0'], dropout=dropoutw, variational=variational_dropout) for lstm in self.lstms]
        self.lstms = nn.ModuleList(self.lstms)

        # ... Attention ----
        self.src_mask = self.generate_causal_mask(
            self.len_input_window).to(device)
        attn_norm = nn.LayerNorm(self.d_model)
        encoder_attn_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_head,
                                                        dim_feedforward=self.d_attn_hidden, dropout=dropout)
        self.attention_module = nn.TransformerEncoder(
            encoder_attn_layer, self.n_layer_attn, attn_norm)

        # ... Decoder ----
        self.decoder = nn.Linear(self.d_model, d_output)
        self.output_fn = LossHelper.get_output_activation(loss_type)

    # ----

    def forward(self, x_var, x_time, x_static, src_mask=None):
        if self.use_embed:
            if self.embed_type == "simple":
                # opt1: simple pos encoding (original attention is all you need)
                emb = self.pos_encoder(self.enc_simple_embedding(x_var))
            elif self.embed_type == 'momentum':
                emb = self.enc_momentum_embedding(x_var, x_time, x_static)
            else:
                emb = self.enc_data_embedding(x_var, x_time)
        else:
            emb = x_var

        # LSTM
        ouput_lstm = self.encoder_lstm(emb)

        # Attn
        if src_mask is None:
            src_mask = self.src_mask
        output_attention = self.attention_module(
            ouput_lstm.permute(1, 0, 2), src_mask).permute(1, 0, 2)

        x = output_attention

        # Decoder
        decoder_out = self.decoder(x)
        out = self.output_fn(decoder_out)
        return out

    def generate_causal_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encoder_lstm(self, emb, hidden=None):
        batch_size, seq_length, dim_emb = emb.size()

        if hidden is None:
            hidden = self.init_lstm_hidden(batch_size)

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
