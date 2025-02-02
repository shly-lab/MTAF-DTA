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


class Conv1dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )
    
    def forward(self, x):

        return self.inc(x)
    

class StackCNN(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.inc = nn.Sequential(OrderedDict([('conv_layer0', Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))]))
        for layer_idx in range(layer_num - 1):
            self.inc.add_module('conv_layer%d' % (layer_idx + 1), Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))

        self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))

    def forward(self, x):

        return self.inc(x).squeeze(-1)


class ProteinEncoder(nn.Module):
    def __init__(self, block_num=3, vocab_size=25+1, embedding_num=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        self.block_list = nn.ModuleList()
        for block_idx in range(1, block_num + 1):
            self.block_list.append(
                StackCNN(block_idx, embedding_num, 128, 3)
            )
        self.linear = nn.Linear(block_num * 128, 128)
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=9)
        self.bn3 = nn.BatchNorm1d(128)
        self.advpool = nn.AdaptiveMaxPool1d(1)
        
    def forward(self, v):
        v = self.embed(v)
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = self.advpool(v).squeeze(-1)
        # print("===v",v.shape)
        return v

class NodeLevelBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(NodeLevelBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        return torch.functional.F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={eps}, ' \
               'affine={affine}'.format(**self.__dict__)

class GraphConvBn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = gnn.GraphConv(in_channels, out_channels)
        self.norm = NodeLevelBatchNorm(out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        data.x = x

        return data

class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        self.conv1 = GraphConvBn(num_input_features, int(growth_rate * bn_size))
        self.conv2 = GraphConvBn(int(growth_rate * bn_size), growth_rate)

    def forward(self, data):
        if isinstance(data.x, Tensor):
            data.x = [data.x]
        data.x = torch.cat(data.x, 1)

        data = self.conv1(data)
        data = self.conv2(data)

        return data

class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, growth_rate=32, bn_size=2):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size)
            self.add_module('layer%d' % (i + 1), layer)

    def forward(self, data):
        features = [data.x]
        for name, layer in self.items():
            data = layer(data)
            features.append(data.x)
            data.x = features

        data.x = torch.cat(data.x, 1)

        return data

class DrugEncoder(nn.Module):
    def __init__(self, num_input_features, out_dim, growth_rate=32, block_config=[8, 8, 8], bn_sizes=[2, 2, 2], channels=128, r=4):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', GraphConvBn(num_input_features, 32))]))
        num_input_features = 32

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers, num_input_features, growth_rate, bn_size=bn_sizes[i]
            )
            self.features.add_module('block%d' % (i+1), block)
            num_input_features += int(num_layers * growth_rate)

            trans = GraphConvBn(num_input_features, num_input_features // 2)

            self.features.add_module("transition%d" % (i+1), trans)
            num_input_features = num_input_features // 2

        self.classifer = nn.Linear(num_input_features, out_dim)

        inter_channels = int(channels // r)

        self.att1 = nn.Sequential(
            nn.Linear(228, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 228),
            nn.BatchNorm1d(228)
        )
        self.att2 = nn.Sequential(
            nn.Linear(228, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 228),
            nn.BatchNorm1d(228)
        )


        self.sigmoid = nn.Sigmoid()
        self.fc_mg = nn.Linear(2048, 228)
        self.fc_ava = nn.Linear(512, 228)

    def forward(self, data):
        data = self.features(data)
        graph_feature = gnn.global_mean_pool(data.x, data.batch)

        morgan = torch.tensor(data.morgan).float().to('cuda:0')
        morgan = self.fc_mg(morgan)
        Avalon = torch.tensor(data.Avalon).float().to('cuda:0')
        Avalon = self.fc_ava(Avalon)

        w1 = self.sigmoid(self.att1(graph_feature + morgan))
        gm_f = graph_feature * w1 + morgan * (1 - w1)

        w2 = self.sigmoid(self.att2(gm_f + Avalon))
        
        Agm_f = gm_f * w2 + Avalon * (1 - w2)
        fout = self.classifer(Agm_f)

        return fout
