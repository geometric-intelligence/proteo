from checkpoint_analysis import load_checkpoint, load_config
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from proteo.train import construct_datasets

def load_attention_coefficient():
    # 1) Load your Lightning module from checkpoint
    module = load_checkpoint("/scratch/lcornelis/outputs/ray_results/TorchTrainer_2025-02-03_16-34-34/model=gat-v4,seed=42423_3995_act=sigmoid,adj_thresh=0.7000,batch_size=32,dropout=0.2000,l1_lambda=0.0000,lr=0.0087,lr_scheduler=La_2025-02-05_12-48-31/checkpoint_000631")
    config = load_config(module)
    module.model.eval()  # 'module.model' is the GATv4 instance typically

    train_dataset, test_dataset = construct_datasets(config)

    # 2) Suppose you have a single participant's Data
    participant_data = test_dataset[0]  # torch_geometric.data.Data or a Batch

    # 3) Forward pass to get attention from the first layer
    with torch.no_grad():
        pred, aux, (edge_index, alpha_1) = module.model(
            participant_data.x,
            edge_index=participant_data.edge_index,
            data=participant_data,
            return_attention_weights=True
        )

    # alpha_1 might be shape [E, num_heads] -> average if needed
    if alpha_1.dim() == 2:
        print("more than one head")
        alpha_1 = alpha_1.mean(dim=1)  # shape [E]
    alpha_1 = alpha_1.cpu().numpy()

    print("edge_index shape:", edge_index.shape)
    print("Alpha shape:", alpha_1.shape) 

    edge_index = participant_data.edge_index  # shape [2, E]
    E = edge_index.size(1)

# Suppose you have a Series or array of importances for this participant, 
# e.g. from a CSV:
# row = importance_scores.loc["ParticipantXYZ"]
# row is length = total number of nodes

# # 1) Identify top-30 node IDs
# top_30_ids = row.nlargest(30).index.to_list()  # if row's index are integers matching node IDs
# top_30_set = set(top_30_ids)

# # 2) Build adjacency (30x30) from alpha_1
# adj_30 = np.zeros((30, 30), dtype=np.float32)

# # map node ID -> subgraph index [0..29]
# top_30_list = sorted(list(top_30_set))
# node2idx = {nid: i for i, nid in enumerate(top_30_list)}

# for e in range(E):
#     src = edge_index[0, e].item()
#     dst = edge_index[1, e].item()
#     if (src in top_30_set) and (dst in top_30_set):
#         i_src = node2idx[src]
#         i_dst = node2idx[dst]
#         adj_30[i_src, i_dst] = alpha_1[e]

# # 3) Visualize with NetworkX
# G_sub = nx.from_numpy_array(adj_30, create_using=nx.DiGraph)
# # rename nodes to their "real" IDs
# mapping = {i: top_30_list[i] for i in range(30)}
# G_sub = nx.relabel_nodes(G_sub, mapping)

# # node sizes ~ importance
# node_sizes = [3000 * row[node] for node in G_sub.nodes()]

# # edge widths ~ attention
# edge_widths = []
# for (u, v) in G_sub.edges():
#     w = G_sub[u][v]['weight']
#     edge_widths.append(3.0 * w)

# pos = nx.spring_layout(G_sub, k=0.8, seed=42)

# plt.figure(figsize=(10, 10), dpi=150)
# nx.draw_networkx_nodes(G_sub, pos,
#                        node_size=node_sizes,
#                        node_color="skyblue",
#                        alpha=0.8)
# nx.draw_networkx_edges(G_sub, pos,
#                        width=edge_widths,
#                        edge_color="gray",
#                        alpha=0.7)
# nx.draw_networkx_labels(G_sub, pos, font_size=8)
# plt.title("Top-30 Subgraph: First-Layer Attention")
# plt.axis("off")
# plt.tight_layout()
# plt.show()

def __main__():
    load_attention_coefficient()

if __name__ == "__main__":
    __main__()