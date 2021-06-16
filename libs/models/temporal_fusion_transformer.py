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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TemporalFusionTransformer(nn.Module):
    name = 'conv_momentum'
    batch_first = True

    def __init__(
            self, d_input, d_output, n_layer_lstm, d_hidden, loss_type,
            len_input_window, n_head, d_model, d_attn_hidden, n_categories,
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

        self.embedding = nn.Linear(self.d_input, self.d_hidden_lstm_in)
        self.embedding_static = nn.Embedding(
            self.n_categories, self.d_hidden_lstm_in)

        # ... static variable ----
        self.static_context_variable_selection = GatedResidualNetwork(
            input_size=self.d_hidden_lstm_in,
            hidden_size=self.d_hidden_lstm_in,
            output_size=self.d_hidden_lstm_in,
            dropout=self.dropout,
        )

        # ... Variable selection ----
        self.encoder_variable_selection = VariableSelectionNetwork(
            input_sizes={str(i): self.len_input_window for i in range(
                self.d_hidden_lstm_in)},
            hidden_size=self.d_hidden_lstm_in,
            input_embedding_flags={
                name: False for name in range(self.d_input)},
            dropout=self.dropout,
            context_size=self.d_hidden_lstm_in,
            single_variable_grns={}
        )

        # ... LSTM ----
        self.lockdrop = LockedDropout()
        self.lstms = [nn.LSTM(input_size=d_hidden[0] if l == 0 else d_hidden[l - 1],
                              hidden_size=d_hidden[l], num_layers=1, dropout=0,
                              batch_first=True) for l in range(n_layer_lstm)]
        if dropoutw > 0:
            self.lstms = [WeightDrop(
                lstm, ['weight_hh_l0', 'weight_ih_l0'], dropout=dropoutw, variational=variational_dropout) for lstm in self.lstms]
        self.lstms = nn.ModuleList(self.lstms)

        # ... GRNs (I) ----
        # self.post_lstm_gate_norm = GateAddNorm(
        #    self.d_hidden_lstm_out, dropout=None)

        # ... Attention ----
        self.src_mask = self.generate_causal_mask(
            self.len_input_window).to(device)
        attn_norm = nn.LayerNorm(self.d_model)
        encoder_attn_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_head,
                                                        dim_feedforward=self.d_attn_hidden, dropout=dropout)
        self.attention_module = nn.TransformerEncoder(
            encoder_attn_layer, self.n_layer_attn, attn_norm)

        # ... GRNs (II) ----
        self.post_attn_gate_norm = GateAddNorm(
            self.d_hidden_lstm_out, dropout=self.dropout)

        self.pos_wise_ff = GatedResidualNetwork(
            self.d_hidden_lstm_out, self.d_hidden_lstm_out, self.d_hidden_lstm_out, dropout=self.dropout)

        self.pre_output_gate_norm = GateAddNorm(
            self.d_hidden_lstm_out, dropout=0)

        # ... Decoder ----
        self.decoder = nn.Linear(self.d_model, d_output)
        self.output_fn = LossHelper.get_output_activation(loss_type)

    # ----

    def forward(self, x_var, x_static=None, hidden=None, src_mask=None):
        B, L, D = x_var.shape

        # Embedding
        embedding = self.embedding(x_var)
        embedding = self.lockdrop(embedding, self.dropouti)

        # ... static variable(s)
        if x_static is not None:
            static_embedding = self.embedding_static(x_static)
            # tmp: currently excluded as just one meta-data currently
            # static_embedding, static_variable_selection = self.static_variable_selection(
            #    static_embedding)
        else:
            # just input zeros
            static_embedding = torch.zeros(
                (x_var.size(0), self.d_hidden_lstm_in), dtype=torch.double, device=device
            )
            static_variable_selection = torch.zeros(
                (x_var.size(0), 0), dtype=torch.double, device=device)

        static_context_variable_selection = self.expand_static_context(
            self.static_context_variable_selection(static_embedding), L
        )

        # variable selection
        embeddings_varying_encoder = {
            str(i): embedding[:, :, i] for i in range(embedding.shape[-1])
        }
        embeddings_varying_encoder, encoder_sparse_weights = self.encoder_variable_selection(
            embeddings_varying_encoder,
            static_context_variable_selection
        )

        # LSTM

        # tmp: input initial state as context initial state with static embedding
        # embeddings_varying_encoder leaks information somehow...
        # ... currently changed back to embed (without static information)
        ouput_lstm = self.encoder_lstm(embedding)

        # GRNs (I)
        # ...

        # Attn
        if src_mask is None:
            src_mask = self.src_mask
        output_attention = self.attention_module(
            ouput_lstm.permute(1, 0, 2), src_mask).permute(1, 0, 2)

        # GRNs (II)
        x = self.post_attn_gate_norm(output_attention, ouput_lstm)
        x = self.pos_wise_ff(x)
        x = self.pre_output_gate_norm(x, ouput_lstm)

        # Decoder
        decoder_out = self.decoder(x)
        out = self.output_fn(decoder_out)
        return out

    def generate_causal_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def expand_static_context(self, context, timesteps):
        """
        add time dimension to static context
        """
        return context[:, None].expand(-1, timesteps, -1)

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


class TimeDistributedEmbeddingBag(nn.EmbeddingBag):
    def __init__(self, *args, batch_first: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.forward(x)

        # Squash samples and timesteps into a single axis
        # (samples * timesteps, input_size)
        x_reshape = x.contiguous().view(-1, x.size(-1))

        y = super().forward(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            # (samples, timesteps, output_size)
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            # (timesteps, samples, output_size)
            y = y.view(-1, x.size(1), y.size(-1))
        return y


class TimeDistributedInterpolation(nn.Module):
    def __init__(self, output_size: int, batch_first: bool = False, trainable: bool = False):
        super().__init__()
        self.output_size = output_size
        self.batch_first = batch_first
        self.trainable = trainable
        if self.trainable:
            self.mask = nn.Parameter(torch.zeros(
                self.output_size, dtype=torch.double))
            self.gate = nn.Sigmoid()

    def interpolate(self, x):
        upsampled = F.interpolate(x.unsqueeze(
            1), self.output_size, mode="linear", align_corners=True).squeeze(1)
        if self.trainable:
            upsampled = upsampled * self.gate(self.mask.unsqueeze(0)) * 2.0
        return upsampled

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.interpolate(x)

        # Squash samples and timesteps into a single axis
        # (samples * timesteps, input_size)
        x_reshape = x.contiguous().view(-1, x.size(-1))

        y = self.interpolate(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            # (samples, timesteps, output_size)
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            # (timesteps, samples, output_size)
            y = y.view(-1, x.size(1), y.size(-1))

        return y


class ResampleNorm(nn.Module):
    def __init__(self, input_size: int, output_size: int = None, trainable_add: bool = True):
        super().__init__()

        self.input_size = input_size
        self.trainable_add = trainable_add
        self.output_size = output_size or input_size

        if self.input_size != self.output_size:
            self.resample = TimeDistributedInterpolation(
                self.output_size, batch_first=True, trainable=False)

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(
                self.output_size, dtype=torch.double))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_size != self.output_size:
            x = self.resample(x)

        if self.trainable_add:
            x = x * self.gate(self.mask) * 2.0

        output = self.norm(x)
        return output


class GatedLinearUnit(nn.Module):
    def __init__(self, input_size, hidden_size=None, dropout=0):
        super().__init__()

        # why linear necessary?

        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size or input_size
        self.dense = nn.Linear(input_size, self.hidden_size * 2)

        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if "bias" in n:
                torch.nn.init.zeros_(p)
            elif "dense" in n:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = F.glu(x, dim=-1)
        return x


class AddNorm(nn.Module):
    def __init__(self, input_size, skip_size=None, trainable_add=False):
        super().__init__()

        self.input_size = input_size
        self.trainable_add = trainable_add
        self.skip_size = skip_size or input_size

        if self.input_size != self.skip_size:
            raise NotImplementedError()

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(
                self.input_size, dtype=torch.double))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.input_size)

    def forward(self, x, skip):
        if self.trainable_add:
            skip = skip * self.gate(self.mask) * 2.0

        output = self.norm(x + skip)
        return output


class GateAddNorm(nn.Module):
    def __init__(self, input_size, hidden_size=None, skip_size=None, dropout=0):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size or input_size
        self.skip_size = skip_size or self.hidden_size
        self.dropout = dropout

        self.glu = GatedLinearUnit(
            self.input_size, hidden_size=self.hidden_size, dropout=self.dropout)
        self.add_norm = AddNorm(
            self.hidden_size, skip_size=self.skip_size, trainable_add=False)
        pass

    def forward(self, x, skip):
        output = self.glu(x)
        output = self.add_norm(output, skip)
        return output


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size,
                 hidden_size,
                 output_size,
                 dropout=0.1,
                 context_size=None,
                 residual=False):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.residual = residual

        if self.input_size != self.output_size and not self.residual:
            residual_size = self.input_size
        else:
            residual_size = self.output_size

        if self.output_size != residual_size:
            self.resample_norm = ResampleNorm(residual_size, self.output_size)

        self.dense_1 = nn.Linear(self.input_size, self.hidden_size)
        self.dense_2 = nn.Linear(self.hidden_size, self.output_size)

        self.elu = nn.ELU()

        if self.context_size is not None:
            self.context = nn.Linear(
                self.context_size, self.hidden_size, bias=False)

        self.gate_norm = GateAddNorm(
            input_size=self.hidden_size,
            skip_size=self.output_size,
            hidden_size=self.output_size,
            dropout=self.dropout
        )

        self.init_weights()

    def forward(self, x, context=None, residual=None):
        if residual is None:
            residual = x

        if self.input_size != self.output_size and not self.residual:
            residual = self.resample_norm(residual)

        x = self.dense_1(x)
        if context is not None:
            context = self.context(context)
            x = x + context
        x = self.elu(x)
        x = self.dense_2(x)
        x = self.gate_norm(x, residual)
        return x

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" in name:
                torch.nn.init.zeros_(p)
            elif "dense_1" in name or "dense_2" in name:
                torch.nn.init.kaiming_normal_(
                    p, a=0, mode="fan_in", nonlinearity="leaky_relu")
            elif "context" in name:
                torch.nn.init.xavier_uniform_(p)


class VariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        input_sizes,
        hidden_size,
        input_embedding_flags={},
        dropout=0,
        context_size=None,
        single_variable_grns={},
    ):
        """
        Calcualte weights for ``num_inputs`` variables  which are each of size ``input_size``
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.input_sizes = input_sizes
        self.input_embedding_flags = input_embedding_flags
        self.dropout = dropout
        self.context_size = context_size

        if self.num_inputs > 1:
            if self.context_size is not None:
                self.flattened_grn = GatedResidualNetwork(
                    self.input_size_total,
                    min(self.hidden_size, self.num_inputs),
                    self.num_inputs,
                    self.dropout,
                    self.context_size,
                    residual=False,
                )
            else:
                self.flattened_grn = GatedResidualNetwork(
                    self.input_size_total,
                    min(self.hidden_size, self.num_inputs),
                    self.num_inputs,
                    self.dropout,
                    residual=False,
                )

        self.single_variable_grns = nn.ModuleDict()
        for name, input_size in self.input_sizes.items():
            if name in single_variable_grns:
                self.single_variable_grns[name] = single_variable_grns[name]
            elif self.input_embedding_flags.get(name, False):
                self.single_variable_grns[name] = ResampleNorm(
                    input_size, self.hidden_size)
            else:
                self.single_variable_grns[name] = GatedResidualNetwork(
                    input_size,
                    min(input_size, self.hidden_size),
                    output_size=self.hidden_size,
                    dropout=self.dropout,
                )

        self.softmax = nn.Softmax(dim=-1)

    @property
    def input_size_total(self):
        return sum(size if name in self.input_embedding_flags else size for name, size in self.input_sizes.items())

    @property
    def num_inputs(self):
        return len(self.input_sizes)

    def forward(self, x, context=None):
        if self.num_inputs > 1:
            # transform single variables
            var_outputs = []
            weight_inputs = []
            for name in self.input_sizes.keys():
                # select embedding belonging to a single input
                variable_embedding = x[name]
                weight_inputs.append(variable_embedding)
                # var_outputs.append(
                #    self.single_variable_grns[name](variable_embedding))
                # tmp: dimension issue?!
                var_outputs.append(variable_embedding)
            var_outputs = torch.stack(var_outputs, dim=-1)

            # calculate variable weights
            flat_embedding = torch.cat(weight_inputs, dim=-1)
            sparse_weights = self.flattened_grn(
                flat_embedding, context=None)  # tmp: set context to none
            sparse_weights = self.softmax(sparse_weights).unsqueeze(-2)

            outputs = var_outputs * sparse_weights
            # outputs = outputs.sum(dim=-1) tmp: dim issue?!
        else:  # for one input, do not perform variable selection but just encoding
            name = next(iter(self.single_variable_grns.keys()))
            variable_embedding = x[name]
            outputs = self.single_variable_grns[name](
                variable_embedding)  # fast forward if only one variable
            if outputs.ndim == 3:  # -> batch size, time, hidden size, n_variables
                sparse_weights = torch.ones(outputs.size(
                    0), outputs.size(1), 1, 1, device=outputs.device)  #
            else:  # ndim == 2 -> batch size, hidden size, n_variables
                sparse_weights = torch.ones(
                    outputs.size(0), 1, 1, device=outputs.device)
        return outputs, sparse_weights
