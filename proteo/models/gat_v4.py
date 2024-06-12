import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Batch, Data
from torch.nn import LayerNorm, Parameter
from torch_geometric.utils import to_dense_batch

class GATv4(nn.Module):
    def __init__(self, opt, in_channels, out_channels):
        super(GATv4, self).__init__()
        self.opt = opt
        #self.act = define_act_layer(act_type=opt.act_type)

        self.hidden_channels = opt.hidden_channels
        self.heads = opt.heads
        self.fc_dim = opt.fc_dim
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.lin_input_dim = opt.num_nodes * len(opt.which_layer)

        # GAT layers
        self.convs = nn.ModuleList()
        self.build_gat_layers()

        # Pooling layers
        self.pools = nn.ModuleList()
        self.build_pooling_layers()

        # Layer normalization
        self.layer_norm = LayerNorm(opt.num_nodes)

        # Fully connected layers
        self.encoder = self.build_fc_layers()
        self.last_layer = nn.Linear(opt.omic_dim, self.out_channels)

        # Output range and shift
        #self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        #self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def build_gat_layers(self):
        input_dim = self.in_channels
        for hidden_dim, num_heads in zip(self.hidden_channels, self.heads):
            self.convs.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=self.opt.fc_dropout))
            input_dim = hidden_dim * num_heads

    def build_pooling_layers(self):
        for hidden_dim, num_heads in zip(self.hidden_channels, self.heads):
            self.pools.append(nn.Linear(hidden_dim * num_heads, 1))

    def build_fc_layers(self):
        layers = []
        fc_input_dim = self.lin_input_dim
        for fc_dim in self.fc_dim:
            layers.append(nn.Sequential(
                nn.Linear(fc_input_dim, fc_dim),
                nn.ELU(),
                nn.AlphaDropout(p=self.opt.fc_dropout, inplace=True),
            ))
            fc_input_dim = fc_dim
        layers.append(nn.Sequential(
            nn.Linear(fc_input_dim, self.opt.omic_dim),
            nn.ELU(),
            nn.AlphaDropout(p=self.opt.fc_dropout, inplace=True),
        ))
        return nn.Sequential(*layers)

    def forward(self, x, edge_index=None, data=None):
        if not isinstance(data, Batch):
            data = Batch().from_data_list([data])

        batch = data.batch
        edge_index = data.edge_index

        # Initial operations before GAT layers
        x = x.requires_grad_()    # [bs*nodes, in_channels]
        x0 = to_dense_batch(torch.mean(x, dim=-1), batch=batch)[0]  # [bs, nodes], converts it to be broken up by batches

        # Apply first GAT layer and pooling
        x = F.dropout(x, p=0.1, training=self.training) #apply dropout if we are training, reduced this to 0.1 from 0.2
        x = self.convs[0](x, edge_index) # [bs*nodes, hidden_channels[0]*heads[0]], Apply the gatconv layer
        x = F.elu(x)  # [bs*nodes, hidden_channels[0]*heads[0]], Apply elu activation function
        x1 = self.pools[0](x) # [bs*nodes, 1]
        x1 = x1.squeeze(-1)  # [bs*nodes]
        x1 = to_dense_batch(x1, batch=batch)[0] # [bs, nodes]

        # Apply second GAT layer and pooling
        x = F.dropout(x, p=0.1, training=self.training)  #apply dropout if we are training
        x = self.convs[1](x, edge_index)
        x = F.elu(x)  # [bs*nodes, hidden_channels[1]*heads[1]]
        x2 = self.pools[1](x) # [bs*nodes, 1]
        x2 = x2.squeeze(-1) # [bs*nodes]
        x2 = to_dense_batch(x2, batch=batch)[0] # [bs, nodes]

        # Apply layer normalization if specified
        if self.opt.layer_norm:
            x0 = self.layer_norm(x0)
            x1 = self.layer_norm(x1)
            x2 = self.layer_norm(x2)

        # Concatenate multiscale features
        multiscale_features = {'layer1': x0, 'layer2': x1, 'layer3': x2}
        multiscale_features = torch.cat(
            [multiscale_features[layer] for layer in self.opt.which_layer], dim=1
        )

        # Pass through fully connected layers
        encoded_features = self.encoder(multiscale_features)
        pred = self.last_layer(encoded_features)

        # Apply activation function if specified
        '''
        if self.act:
            "Im in here!"
            pred = self.act(pred)
            if isinstance(self.act, nn.Sigmoid):
                pred = pred * self.output_range + self.output_shift'''

        return pred


# can delete this all 
def define_scheduler(opt, optimizer):
    if opt.lr_policy == 'linear':

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1) / float(opt.num_epochs + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5
        )
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler
