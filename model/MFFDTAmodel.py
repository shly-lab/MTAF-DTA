from collections import OrderedDict
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from torch_geometric.nn import global_mean_pool
from torch_geometric import data as DATA
import numpy as np
from featuresExtraction import DrugEncoder
from featuresExtraction import ProteinEncoder


class qkvAttentionClass(nn.Module):
    def __init__(self, hid_dim, multiORsingle_head_, dropout):
        super().__init__()
        if multiORsingle_head_ == 'single':
            self.multiORsingle_head = 1
        else:
            print("please enter the num of head!")
        self.hid_dim = hid_dim

        assert hid_dim % self.multiORsingle_head == 0

        self.q_weight = nn.Linear(hid_dim, hid_dim)
        self.k_weight = nn.Linear(hid_dim, hid_dim)
        self.v_weight = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)
        device = torch.device('cuda:0')
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // self.multiORsingle_head])).to(device)

    def forward(self, query, key, value, mask=None):

        batch_size = query.shape[0]

        Q = self.q_weight(query)
        K = self.k_weight(key)
        V = self.v_weight(value)

        Q = Q.view(batch_size, self.multiORsingle_head, self.hid_dim // self.multiORsingle_head).unsqueeze(3)
        K_T = K.view(batch_size, self.multiORsingle_head, self.hid_dim // self.multiORsingle_head).unsqueeze(3).transpose(2,3)
        V = V.view(batch_size, self.multiORsingle_head, self.hid_dim // self.multiORsingle_head).unsqueeze(3)

        energy = torch.matmul(Q, K_T) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        weighter_matrix = torch.matmul(attention, V)

        weighter_matrix = weighter_matrix.permute(0, 2, 1, 3).contiguous()

        weighter_matrix = weighter_matrix.view(batch_size, self.multiORsingle_head * (self.hid_dim // self.multiORsingle_head))

        weighter_matrix = self.do(self.fc(weighter_matrix))

        return weighter_matrix

class SpiralAttention(nn.Module):

    def __init__(self, channels=128, r=4):

        super(SpiralAttention, self).__init__()
        inter_channels = int(channels // r)
        self.att = qkvAttentionClass(hid_dim = 128, multiORsingle_head_ = 'single', dropout = 0.2)
        self.att1 = nn.Sequential(
            nn.Linear(channels, inter_channels),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Linear(inter_channels, channels),
            nn.BatchNorm1d(channels)
        )

        self.att2 = nn.Sequential(
            nn.Linear(channels, inter_channels),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Linear(inter_channels, channels),
            nn.BatchNorm1d(channels)
        )

        self.sigmoid = nn.Sigmoid()

    
    def forward(self, drug_feature, protein_feature):

        w1 = self.sigmoid(self.att1(drug_feature + protein_feature))
        fout1 = drug_feature * w1 + protein_feature * (1 - w1)

        w2 = self.sigmoid(self.att2(fout1))
        fdfp_attention = drug_feature * w2 + protein_feature * (1 - w2)
        drug_feature = drug_feature + self.att(protein_feature,drug_feature,drug_feature)
        drug_feature = self.att(drug_feature,drug_feature,drug_feature)
        output = self.att(drug_feature, fdfp_attention, fdfp_attention)

        return output
class mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.protein_encoder = ProteinEncoder()
        self.ligand_encoder = DrugEncoder(num_input_features=87, out_dim=128, block_config=[8, 8, 8], bn_sizes=[2, 2, 2])

        self.sat = SpiralAttention()
        self.classifier = nn.Sequential(
            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, data):
        ProcessedDrug = self.ligand_encoder(data)
        ProcessedProtein = self.protein_encoder(data.target)

        x = self.sat(ProcessedDrug, ProcessedProtein)
        y = self.classifier(x)

        return y

if sys.platform == 'linux':
    with open(__file__,"r",encoding="utf-8") as f:
        for line in f.readlines():
            print(line)

if __name__ == '__main__':
    def get_model_size(model):
        return sum(p.numel() for p in model.parameters())
    
    model = mymodel()

    total_params = get_model_size(model)
    print(f"Total params: {(get_model_size(model) / 1e6):.2f}M")
