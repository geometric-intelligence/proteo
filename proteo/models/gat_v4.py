from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, Parameter
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import to_dense_batch


# Overriding parameter intitialization
class CustomGATConv(GATConv):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        weight_initializer=None,  # The only thing changed from orig. implementation
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.weight_init = weight_initializer  # The only thing changed from orig. implementation

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=add_self_loops,
            edge_dim=edge_dim,
            fill_value=fill_value,
            bias=bias,
            **kwargs,
        )

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        self.lin = self.lin_src = self.lin_dst = None
        if isinstance(in_channels, int):
            self.lin = Linear(
                in_channels, heads * out_channels, bias=False, weight_initializer='glorot'
            )
        else:
            self.lin_src = Linear(
                in_channels[0], heads * out_channels, False, weight_initializer='glorot'
            )
            self.lin_dst = Linear(
                in_channels[1], heads * out_channels, False, weight_initializer='glorot'
            )

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = Parameter(torch.empty(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(
                edge_dim, heads * out_channels, bias=False, weight_initializer='glorot'
            )
            self.att_edge = Parameter(torch.empty(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self, **kwargs):
        super().reset_parameters()
        if self.lin is not None:
            self.weight_init(self.lin.weight)
        if self.lin_src is not None:
            self.weight_init(self.lin_src.weight)
        if self.lin_dst is not None:
            self.weight_init(self.lin_dst.weight)
        if self.lin_edge is not None:
            self.weight_init(self.lin_edge)
        if self.att_edge is not None:
            self.weight_init(self.att_edge)
        self.weight_init(self.att_src)
        self.weight_init(self.att_dst)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class GATv4(nn.Module):
    ACT_MAP = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
    }
    INIT_MAP = {
        "uniform": nn.init.uniform_,
        "xavier": nn.init.xavier_uniform_,
        "kaiming": nn.init.kaiming_uniform_,
        "orthogonal": nn.init.orthogonal_,
        "truncated_normal": torch.nn.init.trunc_normal_,
    }

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        heads,
        dropout,
        act,
        which_layer,
        use_layer_norm,
        fc_dim,
        fc_dropout,
        fc_act,
        num_nodes,
        weight_initializer,
    ):
        super(GATv4, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.act = act
        self.which_layer = which_layer
        self.use_layer_norm = use_layer_norm
        self.fc_dim = fc_dim
        self.fc_dropout = fc_dropout
        self.fc_act = fc_act
        self.fc_input_dim = num_nodes * len(which_layer)
        self.weight_initializer = self.INIT_MAP[weight_initializer]
        self.num_nodes = num_nodes

        # GAT layers
        self.convs = nn.ModuleList()
        self.build_gat_layers()

        # Pooling layers
        self.pools = nn.ModuleList()
        self.build_pooling_layers()

        # Layer normalization
        self.layer_norm = LayerNorm(self.num_nodes)

        # Fully connected layers
        self.encoder = self.build_fc_layers()
        self.sex_encoder = self.build_sex_encoder()
        self.mutation_encoder = self.build_mutation_encoder()
        self.age_encoder = self.build_age_encoder()

        # Initialize weights
        self.reset_parameters()

    def build_gat_layers(self):
        input_dim = self.in_channels
        for hidden_dim, num_heads in zip(self.hidden_channels, self.heads):
            self.convs.append(
                CustomGATConv(
                    in_channels=input_dim,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    dropout=self.dropout,
                    weight_initializer=self.weight_initializer,
                )
            )
            input_dim = hidden_dim * num_heads

    def build_pooling_layers(self):
        for hidden_dim, num_heads in zip(self.hidden_channels, self.heads):
            self.pools.append(nn.Linear(hidden_dim * num_heads, 1))

    def build_fc_layers(self):
        layers = []
        fc_layer_input_dim = self.fc_input_dim
        for fc_dim in self.fc_dim:
            layers.append(
                nn.Sequential(
                    nn.Linear(fc_layer_input_dim, fc_dim),
                    self.ACT_MAP[self.fc_act],
                    nn.AlphaDropout(p=self.fc_dropout, inplace=True),
                )
            )
            fc_layer_input_dim = fc_dim
        layers.append(nn.Linear(fc_dim, self.out_channels))
        return nn.Sequential(*layers)

    def build_sex_encoder(self):
        return nn.Embedding(num_embeddings=2, embedding_dim=self.num_nodes) #2 sexes

    def build_mutation_encoder(self):
        return nn.Embedding(num_embeddings=4, embedding_dim=self.num_nodes) #4 mutations
    
    def build_age_encoder(self):
        return nn.Linear(1, self.num_nodes) #1 age #TODO: Add nonlinearity?

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for pool in self.pools:
            self.weight_initializer(pool.weight)
            if pool.bias is not None:
                pool.bias.data.fill_(0)

        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                self.weight_initializer(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0)

    def forward(self, x, edge_index=None, data=None):
        if not isinstance(data, Batch):
            data = Batch().from_data_list([data])

        batch = data.batch
        edge_index = data.edge_index 
        sex = data.sex # [bs] - 0 or 1
        mutation = data.mutation # [bs] - 0, 1, 2, 3
        age = data.age  # [bs] - age

        # Initial operations before GAT layers
        x = x.requires_grad_()  # [bs*nodes, in_channels]
        # This does mean on features, per subgraph and per node.
        # [bs, nodes], converts it to be broken up by batches
        x0, _ = to_dense_batch(torch.mean(x, dim=-1), batch=batch)

        # Apply first GAT layer and pooling
        x = F.dropout(x, p=self.dropout, training=self.training)
        # apply dropout if we are training, reduced this to 0.1 from 0.2
        x = self.convs[0](x, edge_index)
        # [bs*nodes, hidden_channels[0]*heads[0]], Apply the gatconv layer
        x = self.ACT_MAP[self.act](
            x
        )  # [bs*nodes, hidden_channels[0]*heads[0]], Apply elu activation function
        x1 = self.pools[0](x)  # [bs*nodes, 1]
        x1 = x1.squeeze(-1)  # [bs*nodes]
        x1, _ = to_dense_batch(x1, batch=batch)  # [bs, nodes]

        # Apply second GAT layer and pooling
        x = F.dropout(x, p=self.dropout, training=self.training)  # apply dropout if we are training
        x = self.convs[1](x, edge_index)
        x = self.ACT_MAP[self.act](x)  # [bs*nodes, hidden_channels[1]*heads[1]]
        x2 = self.pools[1](x)  # [bs*nodes, 1]
        x2 = x2.squeeze(-1)  # [bs*nodes]
        x2, _ = to_dense_batch(x2, batch=batch)  # [bs, nodes]

        # Apply layer normalization to each individual graph to have mean 0, std 1
        if self.use_layer_norm:
            x0 = self.layer_norm(x0)  # [bs, nodes]
            x1 = self.layer_norm(x1)  # [bs, nodes]
            x2 = self.layer_norm(x2)  # [bs, nodes]

        # Concatenate multiscale features - results in [bs, 3*nodes]
        multiscale_features = {'layer1': x0, 'layer2': x1, 'layer3': x2}
        multiscale_features = torch.cat(
            [multiscale_features[layer] for layer in self.which_layer[0:3]], dim=1 #just take first 3 gat layers
        )

        # Pass through fully connected layers and encode graph level data
        if all(feature in self.which_layer for feature in ['sex', 'mutation', 'age']):
            sex_features = self.sex_encoder(data.sex)
            mutation_features = self.mutation_encoder(mutation)
            age_features = self.age_encoder(age.view(-1,1)) #reshape to [bs, 1] for linear layer
            demographic_features = torch.cat(
                [sex_features, mutation_features, age_features], dim=1
            )
            

            total_features = torch.cat([demographic_features, multiscale_features], dim=1)
                
            pred = self.encoder(total_features)
        else:
            pred = self.encoder(multiscale_features)
        '''
        pred_nfl = self.encoder_nfl(pred)
        pred_cdr_multiclass = self.encoder_cdr_multiclass(pred)
        pred_cognitive_decline = self.encoder_cognitive_decline(pred)
        pred = self.encoder(multiscale_features)
        '''
        aux = [x0, x1, x2, multiscale_features]

        return pred, aux
