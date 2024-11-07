from __future__ import print_function
from helper import Embeddings, Encoder_MultipleLayers
import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import collections
import math
import copy

torch.manual_seed(1)
np.random.seed(1)


class BIN_Interaction_Flat(nn.Sequential):
    """
    Interaction Network with 2D interaction map
    """

    def __init__(self, **config):
        super(BIN_Interaction_Flat, self).__init__()
        self.max_d = config["max_drug_seq"]
        self.max_p = config["max_protein_seq"]
        self.emb_size = config["emb_size"]
        self.dropout_rate = config["dropout_rate"]

        # densenet
        self.scale_down_ratio = config["scale_down_ratio"]
        self.growth_rate = config["growth_rate"]
        self.transition_rate = config["transition_rate"]
        self.num_dense_blocks = config["num_dense_blocks"]
        self.kernal_dense_size = config["kernal_dense_size"]
        self.batch_size = config["batch_size"]
        self.input_dim_drug = config["input_dim_drug"]
        self.input_dim_target = config["input_dim_target"]
        self.gpus = torch.cuda.device_count()
        self.n_layer = 2
        # encoder
        self.hidden_size = config["emb_size"]
        self.intermediate_size = config["intermediate_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_probs_dropout_prob = config["attention_probs_dropout_prob"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]

        self.flatten_dim = config["flat_dim"]

        # specialized embedding with positional one
        self.drug_embed = Embeddings(
            self.input_dim_drug, self.emb_size, self.max_d, self.dropout_rate
        )
        self.protein_embed = Embeddings(
            self.input_dim_target, self.emb_size, self.max_p, self.dropout_rate
        )

        self.drug_encoder = Encoder_MultipleLayers(
            self.n_layer,
            self.hidden_size,
            self.intermediate_size,
            self.num_attention_heads,
            self.attention_probs_dropout_prob,
            self.hidden_dropout_prob,
        )
        self.protein_encoder = Encoder_MultipleLayers(
            self.n_layer,
            self.hidden_size,
            self.intermediate_size,
            self.num_attention_heads,
            self.attention_probs_dropout_prob,
            self.hidden_dropout_prob,
        )

        self.icnn = nn.Conv2d(1, 3, 3, padding=0)

        self.decoder = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),
            # output layer
            nn.Linear(32, 1),
        )

    def forward(self, drug, protein, drug_mask, protein_mask):

        ex_drug_mask = drug_mask.unsqueeze(1).unsqueeze(2)
        ex_protein_mask = protein_mask.unsqueeze(1).unsqueeze(2)

        ex_drug_mask = (1.0 - ex_drug_mask) * -10000.0
        ex_protein_mask = (1.0 - ex_protein_mask) * -10000.0

        drug_embed = self.drug_embed(drug)  # batch_size x seq_length x embed_size
        protein_embed = self.protein_embed(protein)

        # set output_all_encoded_layers be false, to obtain the last layer hidden states only...

        drug_encoded_layers = self.drug_encoder(
            drug_embed.float(), ex_drug_mask.float()
        )
        # print(d_encoded_layers.shape)
        protein_encoded_layers = self.protein_encoder(
            protein_embed.float(), ex_protein_mask.float()
        )
        # print(p_encoded_layers.shape)

        # repeat to have the same tensor size for aggregation
        d_aug = torch.unsqueeze(drug_encoded_layers, 2).repeat(
            1, 1, self.max_p, 1
        )  # repeat along protein size
        p_aug = torch.unsqueeze(protein_encoded_layers, 1).repeat(
            1, self.max_d, 1, 1
        )  # repeat along drug size

        i = d_aug * p_aug  # interaction
        i_v = i.view(int(self.batch_size / self.gpus), -1, self.max_d, self.max_p)
        # batch_size x embed size x max_drug_seq_len x max_protein_seq_len
        i_v = torch.sum(i_v, dim=1)
        # print(i_v.shape)
        i_v = torch.unsqueeze(i_v, 1)
        # print(i_v.shape)

        i_v = F.dropout(i_v, p=self.dropout_rate)

        # f = self.icnn2(self.icnn1(i_v))
        f = self.icnn(i_v)

        # print(f.shape)

        # f = self.dense_net(f)
        # print(f.shape)

        f = f.view(int(self.batch_size / self.gpus), -1)
        # print(f.shape)

        # f_encode = torch.cat((d_encoded_layers[:,-1], p_encoded_layers[:,-1]), dim = 1)

        # score = self.decoder(torch.cat((f, f_encode), dim = 1))
        score = self.decoder(f)
        return score
