import torch
import torch.nn.functional as F
import torch_geometric.nn.conv as conv
from torch.autograd import Variable
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool


class MyGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super(MyGAT, self).__init__()
        torch.manual_seed(12345)
        self.in_channels = in_channels
        self.conv1 = conv.GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
        )
        self.conv2 = conv.GATConv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
        )
        self.conv3 = conv.GATConv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
        )
        self.conv_classification = conv.GATConv(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=heads,
        )
        self.max_dim = 2
        self.lin = Linear(hidden_channels * heads, out_channels)

    def forward(self, data, dim):
        """Forward pass of the model.

        Parameters
        ----------
        data : Complex or ComplexBatch(Complex)
            The input data.
        dim : int
            The dimension of the cochain to process.
        self.in_channels : int, optional
            The number of classes to use for one-hot encoding.
        """
        batch = data.cochains[dim].batch

        # TODO: Check the function get_all_cochain_params here, see if we get order 2 features
        # Note: here, data is a Complex object
        params = data.get_all_cochain_params(max_dim=self.max_dim, include_down_features=False)
        # print("x of rank 2 is:")
        # print(params[2].x)
        # print("up_index of rank 2 is:")
        # print(params[2].up_index)
        # print("down_index of rank 2 is:")
        # print(params[2].down_index)
        x = params[dim].x

        if self.in_channels > 0:
            x = F.one_hot(x, num_classes=self.in_channels)
        x = x.squeeze()
        up_index = params[dim].up_index
        down_index = params[dim].down_index
        updown_index = up_index
        if down_index is not None:
            updown_index += down_index

        x = F.dropout(x.to(torch.float32), p=0.6, training=self.training)
        x = self.conv1(x, up_index)

        x = x.relu()
        if down_index is not None:
            x = self.conv2(x, down_index)
            x = x.relu()

        x = self.conv3(x, updown_index)

        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.05, training=self.training)
        x = self.lin(x)
        return x
