from __future__ import print_function

import copy

from torch import nn

from modelHelpers.misc import Intermediate, Output
from modelHelpers.attention import Attention


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        attention_probs_dropout_prob,
        hidden_dropout_prob,
    ):
        super(Encoder, self).__init__()
        self.attention = Attention(
            hidden_size,
            num_attention_heads,
            attention_probs_dropout_prob,
            hidden_dropout_prob,
        )
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class Encoder_MultipleLayers(nn.Module):
    def __init__(
        self,
        n_layer,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        attention_probs_dropout_prob,
        hidden_dropout_prob,
    ):
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder(
            hidden_size,
            intermediate_size,
            num_attention_heads,
            attention_probs_dropout_prob,
            hidden_dropout_prob,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states
