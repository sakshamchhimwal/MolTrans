from __future__ import print_function
import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import math
import copy

torch.manual_seed(1)
np.random.seed(1)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        # Define two learnable parameters: gamma and beta
        # gamma: a scaling factor
        # beta: an offset
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        # Small constant to avoid division by zero
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        # Calculate the mean of the input along the last dimension (features)
        u = x.mean(-1, keepdim=True)
        # Calculate the variance of the input along the last dimension (features)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        # Normalize the input by subtracting the mean and dividing by the standard deviation
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        # Scale the normalized input by the learned gamma factor and shift it by the learned beta factor
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        # Create word embeddings for the input IDs
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        # Create position embeddings for the input sequence positions
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        # Apply layer normalization to the combined embeddings
        self.LayerNorm = LayerNorm(hidden_size)
        # Apply dropout to the normalized embeddings
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        # Get the sequence length from the input IDs
        seq_length = input_ids.size(1)

        # Create position IDs based on the sequence length
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Get the word embeddings for the input IDs
        words_embeddings = self.word_embeddings(input_ids)
        # Get the position embeddings for the position IDs
        position_embeddings = self.position_embeddings(position_ids)

        # Add the word and position embeddings
        embeddings = words_embeddings + position_embeddings
        # Apply layer normalization to the combined embeddings
        embeddings = self.LayerNorm(embeddings)
        # Apply dropout to the normalized embeddings
        embeddings = self.dropout(embeddings)

        # Return the final embeddings
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention heads ({num_attention_heads})"
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def _split_into_heads(self, tensor):
        batch_size, seq_length, _ = tensor.size()
        tensor = tensor.view(
            batch_size, seq_length, self.num_attention_heads, self.attention_head_size
        )
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        query_layer = self._split_into_heads(self.query(hidden_states))
        key_layer = self._split_into_heads(self.key(hidden_states))
        value_layer = self._split_into_heads(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        output_size = context_layer.size()[:-2] + (self.all_head_size,)
        return context_layer.view(*output_size)


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        # Define a linear layer to transform the input features
        self.dense = nn.Linear(hidden_size, hidden_size)
        # Define a layer normalization layer to normalize the output
        self.LayerNorm = LayerNorm(hidden_size)
        # Define a dropout layer to prevent overfitting
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # Pass the input through the linear layer
        hidden_states = self.dense(hidden_states)
        # Apply dropout to the transformed hidden states
        hidden_states = self.dropout(hidden_states)
        # Apply layer normalization to the sum of the transformed hidden states and the input tensor
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # Return the normalized output
        return hidden_states


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_probs_dropout_prob,
        hidden_dropout_prob,
    ):
        super(Attention, self).__init__()
        # Create a Self-Attention layer
        self.self = SelfAttention(
            hidden_size, num_attention_heads, attention_probs_dropout_prob
        )
        # Create a Self-Output layer
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        # Pass the input through the Self-Attention layer
        self_output = self.self(input_tensor, attention_mask)
        # Pass the Self-Attention output through the Self-Output layer
        attention_output = self.output(self_output, input_tensor)
        # Return the final attention output
        return attention_output


class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        # Define a linear layer to transform the hidden features to the intermediate size
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        # Pass the input hidden states through the linear layer
        hidden_states = self.dense(hidden_states)
        # Apply a ReLU activation function to the transformed hidden states
        hidden_states = F.relu(hidden_states)
        # Return the activated intermediate features
        return hidden_states


class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        # Define a linear layer to transform the intermediate features to the hidden size
        self.dense = nn.Linear(intermediate_size, hidden_size)
        # Define a layer normalization layer to normalize the output
        self.LayerNorm = LayerNorm(hidden_size)
        # Define a dropout layer to prevent overfitting
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # Pass the hidden states through the linear layer
        hidden_states = self.dense(hidden_states)
        # Apply dropout to the transformed hidden states
        hidden_states = self.dropout(hidden_states)
        # Apply layer normalization to the sum of the transformed hidden states and the input tensor
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # Return the normalized output
        return hidden_states


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
        # Create an Attention layer
        self.attention = Attention(
            hidden_size,
            num_attention_heads,
            attention_probs_dropout_prob,
            hidden_dropout_prob,
        )
        # Create an Intermediate layer
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        # Create an Output layer
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        # Pass the input through the Attention layer
        attention_output = self.attention(hidden_states, attention_mask)
        # Pass the Attention layer's output through the Intermediate layer
        intermediate_output = self.intermediate(attention_output)
        # Pass the Intermediate layer's output through the Output layer
        layer_output = self.output(intermediate_output, attention_output)
        # Return the final output of the Encoder layer
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
        # Create a single encoder layer
        layer = Encoder(
            hidden_size,
            intermediate_size,
            num_attention_heads,
            attention_probs_dropout_prob,
            hidden_dropout_prob,
        )
        # Create a list of n_layer encoder layers by making copies of the single layer
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        # Pass the input through each of the encoder layers
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)

        return hidden_states
