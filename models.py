from __future__ import print_function
import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import collections
import math
import copy

from modelHelpers.embeddings import Embeddings
from modelHelpers.encoder import Encoder_MultipleLayers

torch.manual_seed(1)
np.random.seed(1)

class BIN_Interaction_Flat(nn.Sequential):
    """
    Interaction Network with 2D interaction map
    """

    def __init__(self, **config):
        super(BIN_Interaction_Flat, self).__init__()
        self.max_drug_seq_len = config["max_drug_seq"]
        self.max_protein_seq_len = config["max_protein_seq"]
        self.embedding_dim = config["emb_size"]
        self.dropout_rate = config["dropout_rate"]

        # DenseNet parameters
        self.scale_down_ratio = config["scale_down_ratio"]
        self.growth_rate = config["growth_rate"]
        self.transition_rate = config["transition_rate"]
        self.num_dense_blocks = config["num_dense_blocks"]
        self.kernel_dense_size = config["kernal_dense_size"]
        self.batch_size = config["batch_size"]
        self.input_dim_drug = config["input_dim_drug"]
        self.input_dim_protein = config["input_dim_target"]
        self.num_gpus = 1
        self.num_layers = 2

        # Encoder parameters
        self.hidden_dim = config["emb_size"]
        self.intermediate_dim = config["intermediate_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_dropout_prob = config["attention_probs_dropout_prob"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]

        self.flatten_dim = config["flat_dim"]

        # Specialized embedding with positional encoding
        self.drug_embedding = Embeddings(
            self.input_dim_drug, self.embedding_dim, self.max_drug_seq_len, self.dropout_rate
        )
        self.protein_embedding = Embeddings(
            self.input_dim_protein, self.embedding_dim, self.max_protein_seq_len, self.dropout_rate
        )

        self.drug_encoder = Encoder_MultipleLayers(
            self.num_layers,
            self.hidden_dim,
            self.intermediate_dim,
            self.num_attention_heads,
            self.attention_dropout_prob,
            self.hidden_dropout_prob,
        )
        self.protein_encoder = Encoder_MultipleLayers(
            self.num_layers,
            self.hidden_dim,
            self.intermediate_dim,
            self.num_attention_heads,
            self.attention_dropout_prob,
            self.hidden_dropout_prob,
        )

        self.interaction_cnn = nn.Conv2d(1, 3, kernel_size=3, padding=0)

        self.decoder = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),
            # Output layer
            nn.Linear(32, 1),
        )

    def forward(self, drug_seq, protein_seq, drug_mask, protein_mask):

        extended_drug_mask = drug_mask.unsqueeze(1).unsqueeze(2)
        extended_protein_mask = protein_mask.unsqueeze(1).unsqueeze(2)

        extended_drug_mask = (1.0 - extended_drug_mask) * -10000.0
        extended_protein_mask = (1.0 - extended_protein_mask) * -10000.0

        drug_embeddings = self.drug_embedding(drug_seq)  # batch_size x seq_length x embed_size
        protein_embeddings = self.protein_embedding(protein_seq)

        encoded_drug = self.drug_encoder(drug_embeddings.float(), extended_drug_mask.float())
        encoded_protein = self.protein_encoder(protein_embeddings.float(), extended_protein_mask.float())

        drug_augmented = torch.unsqueeze(encoded_drug, 2).repeat(1, 1, self.max_protein_seq_len, 1)
        protein_augmented = torch.unsqueeze(encoded_protein, 1).repeat(1, self.max_drug_seq_len, 1, 1)

        interaction_matrix = drug_augmented * protein_augmented
        interaction_view = interaction_matrix.view(
            int(self.batch_size / self.num_gpus), -1, self.max_drug_seq_len, self.max_protein_seq_len
        )
        interaction_view = torch.sum(interaction_view, dim=1)
        interaction_view = torch.unsqueeze(interaction_view, 1)

        interaction_view = F.dropout(interaction_view, p=self.dropout_rate)

        interaction_features = self.interaction_cnn(interaction_view)

        interaction_features = interaction_features.view(int(self.batch_size / self.num_gpus), -1)

        score = self.decoder(interaction_features)
        return score
