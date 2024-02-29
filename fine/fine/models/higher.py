import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool

def square_boundary_index(boundary_index):
    """
    Recover lifted graph where edge-to-node boundary relationships of a graph with n_nodes and n_edges
    can be represented as up-adjacency node relations. There are n_nodes+n_edges
    nodes in this lifted graph.

    Parameters
    ----------
    boundary_index : list of lists
        boundary_index[0]: list of node ids in the boundary of edge stored in boundary_index[1]
        boundary_index[1]: list of new_ids of edges

    Returns
    -------
    edge_index : list of lists
        edge_index[0][i] and edge_index[1][i] are the two nodes of edge i
    """
    n_nodes = torch.unique(boundary_index[0]).size(0) + 1
    max_node_id = int(max(boundary_index[0])) + 1

    edge_index = [[], []]
    for n, e in zip(boundary_index[0], boundary_index[1]):
        n = int(n)
        e = int(e)
        edge_index[0].append(n)
        edge_index[1].append(e + max_node_id)
        edge_index[0].append(e + max_node_id)
        edge_index[1].append(n)

    # index_mapping = None
    # if max(boundary_index[0]) >= n_nodes or 0 not in boundary_index[0]:
    edge_index, index_mapping = recount_edge_index(edge_index)
    return torch.tensor(edge_index), index_mapping


def recount_edge_index(edge_index):
    """
    Recount edge index so that it is a valid edge index for a graph with n_nodes nodes.

    Parameters
    ----------
    edge_index : list of lists
        edge_index[0][i] and edge_index[1][i] are the two node ids of edge i

    Returns
    -------
    edge_index : list of lists
        edge_index[0][i] and edge_index[1][i] are the two node indexes of edge i
    index_mapping : dict
        Mapping from modified edge index to original edge index
    """
    node_id_to_index = {}  # Dictionary to map node IDs to indexes
    new_edge_index = [[0] * len(edge_index[0]), [0] * len(edge_index[0])]
    index_mapping = {}  # Mapping from modified edge index to original edge index

    for i in range(len(edge_index[0])):
        for j in range(2):  # Iterate over both nodes in the edge
            node_id = edge_index[j][i]
            if node_id not in node_id_to_index:
                new_index = len(node_id_to_index)
                node_id_to_index[node_id] = new_index
                # index_mapping.append([node_id, node_id_to_index[node_id]])
                # store in index_mapping the node_id_to_index[node_id] for the node_ide value
                index_mapping[node_id] = new_index
    # Update new_edge_index with node indexes
    for i in range(len(edge_index[0])):
        new_edge_index[0][i] = node_id_to_index[edge_index[0][i]]
        new_edge_index[1][i] = node_id_to_index[edge_index[1][i]]

    return new_edge_index, index_mapping


class Higher(torch.nn.Module):
    """High-order GNN model.

    This class takes a GNN and its kwargs as inputs, and turns it into a HighGNN.

    Parameters
    ----------
    max_dim : int
        The maximum dimension of the cochains to process.
    GNN : torch.nn.Module, a class not an object
        The GNN class to use.
    **gnn_kwargs : dict
        The kwargs to pass to the GNN class to build an object from that class.
    """

    def __init__(self, max_dim, GNN, **gnn_kwargs):
        # TODO: How to pass the learned weights for the module of dim=0, i.e. the pretrained one, instead of creating a new one.
        super(Higher, self).__init__()
        self.max_dim = max_dim
        # self.init_layer = torch.nn.Linear()
        self.layers = torch.nn.ModuleList()
        # for dim in range(max_dim + 1):
        #     if dim == 0:
        #         self.layers.append(
        #             GNN(**gnn_kwargs)
        #         )  # self.layers[dim] = torch.nn.ModuleList([GNN(**gnn_kwargs)])

        #     elif dim == 1: #elif dim != self.max_dim:
        #         self.layers.append(GNN(**gnn_kwargs))
        #         self.layers.append(GNN(**gnn_kwargs))
        #         self.layers.append(GNN(**gnn_kwargs))

        #     else:
        #         self.layers.append(GNN(**gnn_kwargs))
        #         self.layers.append(GNN(**gnn_kwargs))
        
        if self.max_dim == 2:
            for _ in range(6):
                self.layers.append(GNN(**gnn_kwargs))
        
        elif self.max_dim == 1:
            for _ in range(3):
                self.layers.append(GNN(**gnn_kwargs))

        self.in_channels = list(gnn_kwargs.values())[0]

    def forward(self, x, **forward_kwargs):
        """Forward pass of the model.

        Parameters
        ----------
        x : Complex or ComplexBatch(Complex)
            The input data.
        forward_kwargs : dict
            The kwargs to pass to the GNN forward pass.
        """
        
        params = x.get_all_cochain_params(max_dim=self.max_dim, include_down_features=True)

        xout_per_dim = {}
        n_edges_total = torch.unique(params[1].boundary_index[1]).size(0)
        if len(params) < (self.max_dim + 1):
            print("skipping this example")
            return None

        for dim in range(self.max_dim + 1):
            params_dim = params[dim]
            x_per_dim = params_dim.x
            if (x_per_dim < 0).any() or (x_per_dim >= self.in_channels).any():
                print("Invalid indices for one-hot encoding.")
                return None

            xin_of_dim = F.one_hot(x_per_dim, num_classes=self.in_channels).float()
            xin_of_dim = torch.sum(xin_of_dim, dim=1)

            if len(params) <= dim:
                print(f"dim={dim} is out of range")
                continue

            if dim == 0:
                xout_per_dim[dim] = self.layers[dim](
                    xin_of_dim, edge_index=params_dim.up_index, **forward_kwargs
                )

            elif dim == 1:
                square_bdry_index, _ = square_boundary_index(params_dim.boundary_index)
                n_nodes = torch.unique(params_dim.boundary_index[0]).size(0)

                zeros_on_nodes = torch.zeros(n_nodes, self.in_channels).to(xin_of_dim.device)
                x_in_of_dim_square = torch.vstack([zeros_on_nodes, xin_of_dim]).to(
                    xin_of_dim.device
                )
                out_bdry_square = self.layers[1](
                    x_in_of_dim_square,
                    edge_index=square_bdry_index.to(xin_of_dim.device),
                    **forward_kwargs,
                )
                out_bdry = out_bdry_square[:n_nodes]
                xout_per_dim[dim - 1] += out_bdry

                out_adj_down = self.layers[2](
                    xin_of_dim, edge_index=params_dim.down_index, **forward_kwargs
                )
                if self.max_dim == 2:
                    out_adj_up = self.layers[3](
                        xin_of_dim, edge_index=params_dim.up_index, **forward_kwargs
                    )

                xout_per_dim[dim] = out_adj_up + out_adj_down if self.max_dim==2 else out_adj_down
            else:
                if params_dim.down_index is not None:
                    out_adj_down = self.layers[4](
                        xin_of_dim, edge_index=params_dim.down_index, **forward_kwargs
                    )
                else:
                    out_adj_down = torch.zeros(xin_of_dim.shape[0], xout_per_dim[1].shape[1])
                xout_per_dim[dim] = out_adj_down.to(xin_of_dim.device)

                square_bdry_index, index_mapping = square_boundary_index(params_dim.boundary_index)
                n_edges_involved_in_bounding = torch.unique(params_dim.boundary_index[0]).size(0)
                zeros_on_edges = torch.zeros(n_edges_involved_in_bounding, self.in_channels).to(
                    xin_of_dim.device
                )
                x_in_of_dim_square = torch.vstack([zeros_on_edges, xin_of_dim])

                square_out_bdry = self.layers[5](
                    x_in_of_dim_square,
                    edge_index=square_bdry_index.to(xin_of_dim.device),
                    **forward_kwargs,
                )

                full_out_bdry = torch.zeros(xout_per_dim[1].shape).to(xin_of_dim.device)
                if index_mapping is not None:
                    n_faces_total = torch.unique(params_dim.boundary_index[1]).size(0)
                    faces_to_skip = sorted(index_mapping.keys(), reverse=True)[:n_faces_total]
                    for og_id, new_id in index_mapping.items():
                        og_id -= 1
                        if og_id in faces_to_skip:
                            continue
                        full_out_bdry[og_id] = square_out_bdry[new_id]
                else:
                    for edge_id in params_dim.boundary_index[0]:
                        edge_id = int(edge_id - 1)
                        full_out_bdry[edge_id] = square_out_bdry[edge_id]

                xout_per_dim[dim - 1] += full_out_bdry

        node_membership = x.cochains[0].batch.unsqueeze(1).squeeze()
        edge_membership = x.cochains[1].batch.unsqueeze(1).squeeze()
        face_membership = x.cochains[2].batch.unsqueeze(1).squeeze()

        xout_0 = global_mean_pool(xout_per_dim[0], node_membership)
        xout_1 = global_mean_pool(xout_per_dim[1], edge_membership)
        xout_2 = global_mean_pool(xout_per_dim[2], face_membership)
        print(xout_0.shape, xout_1.shape, xout_2.shape)
        x = torch.cat([xout_0, xout_1, xout_2], dim=1)
        # x = torch.cat(list(xout_per_dim.values()), dim=0)
        x = torch.sum(x, dim=1)
        x = F.sigmoid(x)
        return x
