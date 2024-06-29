import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_batch


class GATv4(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        heads,
        num_layers,
        fc_dim,
        num_fc_layers,
        which_layer,
        num_nodes,
        use_layer_norm,
        fc_dropout,
    ):
        super(GATv4, self).__init__()
        self.hidden_channels = [
            hidden_channels,
        ] * num_layers
        self.heads = [
            heads,
        ] * num_layers
        self.fc_dim = [
            fc_dim,
        ] * num_fc_layers
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.fc_dropout = fc_dropout
        self.which_layer = which_layer
        self.use_layer_norm = use_layer_norm
        self.lin_input_dim = num_nodes * len(which_layer)

        # GAT layers
        self.convs = nn.ModuleList()
        self.build_gat_layers()

        # Pooling layers
        self.pools = nn.ModuleList()
        self.build_pooling_layers()

        # Layer normalization
        self.layer_norm = LayerNorm(num_nodes)

        # Fully connected layers
        self.encoder = self.build_fc_layers()

    def build_gat_layers(self):
        input_dim = self.in_channels
        for hidden_dim, num_heads in zip(self.hidden_channels, self.heads):
            self.convs.append(
                GATConv(input_dim, hidden_dim, heads=num_heads, dropout=self.fc_dropout)
            )
            input_dim = hidden_dim * num_heads

    def build_pooling_layers(self):
        for hidden_dim, num_heads in zip(self.hidden_channels, self.heads):
            self.pools.append(nn.Linear(hidden_dim * num_heads, 1))

    def build_fc_layers(self):
        layers = []
        fc_input_dim = self.lin_input_dim
        for fc_dim in self.fc_dim:
            layers.append(
                nn.Sequential(
                    nn.Linear(fc_input_dim, fc_dim),
                    # nn.ELU(),
                    nn.Tanh(),
                    nn.AlphaDropout(p=self.fc_dropout, inplace=True),
                )
            )
            fc_input_dim = fc_dim
        layers.append(nn.Linear(fc_dim, self.out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, edge_index=None, data=None):
        if not isinstance(data, Batch):
            data = Batch().from_data_list([data])

        batch = data.batch
        edge_index = data.edge_index

        # Initial operations before GAT layers
        x = x.requires_grad_()  # [bs*nodes, in_channels]
        # This does mean on features, per subgraph and per node.
        # [bs, nodes], converts it to be broken up by batches
        x0, _ = to_dense_batch(torch.mean(x, dim=-1), batch=batch)

        # Apply first GAT layer and pooling
        x = F.dropout(x, p=0.1, training=self.training)
        # apply dropout if we are training, reduced this to 0.1 from 0.2
        x = self.convs[0](x, edge_index)
        # [bs*nodes, hidden_channels[0]*heads[0]], Apply the gatconv layer
        x = F.elu(x)  # [bs*nodes, hidden_channels[0]*heads[0]], Apply elu activation function
        x1 = self.pools[0](x)  # [bs*nodes, 1]
        x1 = x1.squeeze(-1)  # [bs*nodes]
        x1, _ = to_dense_batch(x1, batch=batch)  # [bs, nodes]

        # Apply second GAT layer and pooling
        x = F.dropout(x, p=0.1, training=self.training)  # apply dropout if we are training
        x = self.convs[1](x, edge_index)
        x = F.elu(x)  # [bs*nodes, hidden_channels[1]*heads[1]]
        x2 = self.pools[1](x)  # [bs*nodes, 1]
        x2 = x2.squeeze(-1)  # [bs*nodes]
        x2, _ = to_dense_batch(x2, batch=batch)  # [bs, nodes]

        # Apply layer normalization to each individual graph to have mean 0, std 1
        if self.use_layer_norm:
            x0 = self.layer_norm(x0)
            x1 = self.layer_norm(x1)
            x2 = self.layer_norm(x2)

        # Concatenate multiscale features
        multiscale_features = {'layer1': x0, 'layer2': x1, 'layer3': x2}
        multiscale_features = torch.cat(
            [multiscale_features[layer] for layer in self.which_layer], dim=1
        )

        # Pass through fully connected layers
        pred = self.encoder(multiscale_features)

        return pred
