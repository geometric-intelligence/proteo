import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import LayerNorm, Linear, Parameter, ReLU, Sequential
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch

from proteo.datasets.mlagnn import *


class GATv5(torch.nn.Module):
    def __init__(self, opt, in_channels, out_channels, hidden_channels=4, heads=2, dropout=0.6):
        super(GATv5, self).__init__()
        self.conv1 = GATv2Conv(
            in_channels=in_channels, out_channels=hidden_channels, heads=heads, dropout=dropout
        )
        self.linear1 = Linear(hidden_channels * heads, 1)
        self.conv2 = GATv2Conv(
            hidden_channels, out_channels=hidden_channels, heads=heads, dropout=dropout
        )
        self.linear2 = Linear(hidden_channels * heads, 1)

        self.encoder = Sequential(
            Linear(opt.num_nodes, 64),  # TODO: Fix hardcoding where 7289 is the number of nodes
            ReLU(),
            Linear(64, 32),
            ReLU(),
            Linear(32, out_channels),
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        for layer in self.encoder:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, edge_index):
        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1)
        x1 = self.linear1(x1)

        x2 = self.conv2(x1, edge_index)
        x2 = F.relu(x2)
        x2 = self.linear2(x2)

        x_concat = torch.concat((x, x1, x2), dim=1)

        pred = self.encoder(x_concat)

        if self.act is not None:
            pred = self.act(pred)

            if isinstance(self.act, nn.Sigmoid):
                pred = pred * self.output_range + self.output_shift

        return pred
