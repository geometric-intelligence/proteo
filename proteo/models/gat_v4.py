import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import LayerNorm, Parameter
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_batch
from proteo.datasets.mlagnn import *


class GATv4(torch.nn.Module):
    def __init__(self, opt, out_channels):
        super(GATv4, self).__init__()
        self.fc_dropout = opt.fc_dropout
        self.GAT_dropout = opt.fc_dropout  # opt.GAT_dropout: TODO Check where GAT_dropout is
        self.act = define_act_layer(act_type=opt.act_type)

        self.hidden_channels = opt.hidden_channels #[8, 16, 12] #number of hidden units for each layer
        self.heads = opt.heads #number of attention heads for each GAT layer.
        self.fc_dim = opt.fc_dim #defines the dimensions (or the number of neurons/units) for a series of fully connected (FC) layers
        self.out_channels = out_channels #dimensions of fully connected (FC) layers that are part of the model's encoder

        self.conv1 = GATConv(
            opt.input_dim, self.hidden_channels[0], heads=self.heads[0], dropout=self.GAT_dropout
        )
        print(f"conv1 is:{self.conv1}")
        self.conv2 = GATConv(
            self.hidden_channels[0] * self.heads[0],
            self.hidden_channels[1],
            heads=self.heads[1],
            dropout=self.GAT_dropout,
        )
        print(f"conv2 is:{self.conv2}")
        self.conv3 = GATConv(
            self.hidden_channels[1] * self.heads[1],
            self.hidden_channels[2],
            heads=self.heads[2],
            dropout=self.GAT_dropout,
        )

        self.pool1 = torch.nn.Linear(self.hidden_channels[0] * self.heads[0], 1)
        self.pool2 = torch.nn.Linear(self.hidden_channels[1] * self.heads[1], 1)
        self.pool3 = torch.nn.Linear(self.hidden_channels[2] * self.heads[2], 1)

        self.layer_norm0 = LayerNorm(opt.num_nodes) #num_nodes = batch_size * nodes_per_graph
        self.layer_norm1 = LayerNorm(opt.num_nodes) 
        self.layer_norm2 = LayerNorm(opt.num_nodes) 
        self.layer_norm3 = LayerNorm(opt.num_nodes)

        fc1 = nn.Sequential(
            nn.Linear(opt.lin_input_dim, self.fc_dim[0]),
            nn.ELU(),
            nn.AlphaDropout(p=self.fc_dropout, inplace=True),
        )

        fc2 = nn.Sequential(
            nn.Linear(self.fc_dim[0], self.fc_dim[1]),
            nn.ELU(),
            nn.AlphaDropout(p=self.fc_dropout, inplace=False),
        )

        fc3 = nn.Sequential(
            nn.Linear(self.fc_dim[1], self.fc_dim[2]),
            nn.ELU(),
            nn.AlphaDropout(p=self.fc_dropout, inplace=False),
        )

        fc4 = nn.Sequential(
            nn.Linear(self.fc_dim[2], opt.omic_dim),
            nn.ELU(),
            nn.AlphaDropout(p=self.fc_dropout, inplace=False),
        )

        self.encoder = nn.Sequential(fc1, fc2, fc3, fc4)
        self.last_layer = nn.Sequential(nn.Linear(opt.omic_dim, self.out_channels))

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, data, opt):
        batch = data.batch.cuda()  # [batch_size * nodes]
        ### layer1
        edge_index = torch.tensor(data.edge_index).cuda()
        print(f"edge_index is:{edge_index.shape}")
        edge_index = torch.transpose(edge_index, 2, 0)
        print(f"edge_index is:{edge_index.shape}")
        edge_index = torch.reshape(edge_index, (2, -1))
        print(f"edge_index is:{edge_index.shape}")
        data.x = torch.tensor(data.x).requires_grad_()
        print(f"data.x is:{data.x.shape}") 
        x0 = data.x.cuda()  # [batch_size, nodes]

        ### layer2
        x = F.dropout(x0, p=0.2, training=self.training).to(edge_index.device)    # [batch_size, nodes]
        print("Before conv1")
        print(f"x is:{x.shape}")
        x = F.elu(self.conv1(x, edge_index))  # [bs*nodes, hidden_channels[0]*heads[0]]
        print("After conv1")
        print(f"x is:{x.shape}")
        x1 = to_dense_batch(self.pool1(x).squeeze(-1), batch=batch)[0].to(
            edge_index.device
        )  # [bs, nodes]
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv2(x, edge_index))  # [bs*nodes, hidden_channels[0]*heads[0]]
        print("After conv2")
        print(f"x is:{x.shape}")
        x2 = to_dense_batch(self.pool2(x).squeeze(-1), batch=batch)[0].to(
            edge_index.device
        )  # [bs, nodes]
        print(f"x2 is:{x2.shape}")
        #TO DO: Erroring here now 
        if opt.layer_norm:
            x0 = self.layer_norm0(x0)
            x1 = self.layer_norm1(x1)
            x2 = self.layer_norm0(x2)

        if opt.which_layer == 'all':
            multiscale_features = torch.cat([x0, x1, x2], dim=1)

        elif opt.which_layer == 'layer1':
            multiscale_features = x0
        elif opt.which_layer == 'layer2':
            multiscale_features = x1
        elif opt.which_layer == 'layer3':
            multiscale_features = x2

        encoded_features = self.encoder(multiscale_features)

        pred = self.last_layer(encoded_features)

        if self.act is not None:
            pred = self.act(pred)

            if isinstance(self.act, nn.Sigmoid):
                pred = pred * self.output_range + self.output_shift

        return multiscale_features, encoded_features, pred


def define_optimizer(opt, model):
    optimizer = None
    if opt.optimizer_type == 'adabound':
        optimizer = adabound.AdaBound(model.parameters(), lr=opt.lr, final_lr=opt.final_lr)
    elif opt.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay
        )
    elif opt.optimizer_type == 'adagrad':
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=opt.lr,
            weight_decay=opt.weight_decay,
            initial_accumulator_value=0.1,
        )
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % opt.optimizer)
    return optimizer


def define_reg(model):
    for W in model.parameters():
        loss_reg = torch.abs(W).sum()

    return loss_reg


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
