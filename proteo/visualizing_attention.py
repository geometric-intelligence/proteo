import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from checkpoint_analysis import load_checkpoint, load_config
from proteo.train import construct_datasets

def load_attention_coefficient(
    checkpoint_path: str,
    csv_path: str,
    participant_id: str,
    participant_id_num: int,
    mutation: str,
    top_n: int = 30,
    edge_threshold: float = 0.01,
    layout: str = "spring"
):
    """
    Loads a GAT model from checkpoint, retrieves the first-layer attention 
    for the given participant (by index in the test dataset), then reads 
    the participant's importance scores from a CSV, and visualizes 
    the subgraph of the top-N most important nodes (proteins).

    Args:
        checkpoint_path (str): Path to your trained checkpoint file.
        csv_path (str): Path to the CSV containing importance scores.
            - Rows = participants; columns = node IDs (or feature IDs).
        participant_id (str): The participant to visualize (must match CSV).
        top_n (int): Number of top nodes (by importance) to include in subgraph.
        edge_threshold (float): Remove edges whose attention < this value.
        layout (str): Which layout to use for visualization. 
                      Options: "spring", "kamada", etc.
    """

    # 1) Load your Lightning module from checkpoint
    module = load_checkpoint(checkpoint_path)
    config = load_config(module)
    model = module.model  # GAT instance
    model.eval()

    # 2) Construct or load your dataset
    train_dataset, test_dataset = construct_datasets(config)

    # Make sure test_dataset[0] truly corresponds to participant_id 
    # (if not, map participant_id -> correct index).
    participant_data = test_dataset[participant_id_num]
    print("participant_data.edge_index", participant_data.edge_index)

    # 3) Forward pass to get attention from the first layer
    with torch.no_grad():
        pred, aux, (edge_index, alpha_1) = model(
            participant_data.x,
            edge_index=participant_data.edge_index,
            data=participant_data,
            return_attention_weights=True
        )

    # If multiple heads, average across them
    if alpha_1.dim() == 2:
        alpha_1 = alpha_1.mean(dim=1)  # shape [E]
    alpha_1 = alpha_1.cpu().numpy()  # numpy array of attention

    print("edge_index shape:", edge_index.shape)  # [2, E]
    print("Alpha shape:", alpha_1.shape)
    E = edge_index.size(1)

    # 4) Load the CSV containing the importance scores
    importance_df = pd.read_csv(csv_path, index_col=0)

    # Exclude first four columns if needed (as in your example)
    protein_columns = importance_df.columns[4:]
    print("Number of protein_columns:", len(protein_columns))

    if participant_id not in importance_df.index:
        raise ValueError(f"Participant {participant_id} not found in the CSV index.")

    # Select the row for the specified participant
    row = importance_df.loc[participant_id, protein_columns]
    row = pd.to_numeric(row, errors='coerce')  # convert to numeric if needed

    # Create mapping from protein to node index
    protein_to_node = {protein: idx for idx, protein in enumerate(protein_columns)}

    # 5) Identify top-N node indices by importance
    top_n_proteins = row.nlargest(top_n).index.to_list()
    top_n_indices = [protein_to_node[protein] for protein in top_n_proteins]
    top_n_set = set(top_n_indices)

    #print("Top N indices:", top_n_indices)

    # --------------------------
    # OPTIONAL: Filter edges before building the subgraph
    #  (a) remove self-loops
    #  (b) remove low-attention edges
    # --------------------------
    src, dst = edge_index[0], edge_index[1]

    #(a) remove self-loops
    no_self_loop_mask = (src != dst)

    final_mask = no_self_loop_mask

    # Filter edge_index and alpha_1
    edge_index = edge_index[:, final_mask]
    alpha_1 = alpha_1[final_mask]

    # 6) Re-build adjacency (top_n x top_n) from the filtered edges
    E_filtered = edge_index.size(1)
    node2idx = {nid: i for i, nid in enumerate(sorted(top_n_set))}
    adj_top_n = np.zeros((top_n, top_n), dtype=np.float32)
    edges_list = []

    for e in range(E_filtered):
        s = edge_index[0, e].item()
        t = edge_index[1, e].item()
        if (s in top_n_set) and (t in top_n_set):
            i_src = node2idx[s]
            i_dst = node2idx[t]
            adj_top_n[i_src, i_dst] = alpha_1[e]
            edges_list.append((i_src, i_dst))
            print("i_src", i_src)
            print("i_dst", i_dst)
            print("alpha_1[e]", alpha_1[e])

    # 7) Visualize with NetworkX
    G_sub = nx.from_numpy_array(adj_top_n, create_using=nx.DiGraph)
    # rename nodes to their real protein IDs
    mapping = {i: top_n_proteins[i] for i in range(top_n)}
    G_sub = nx.relabel_nodes(G_sub, mapping)

    # Node size ~ importance
    node_sizes = [3000 * row[protein] for protein in top_n_proteins]

    # Edge widths ~ attention
    edge_widths = []
    for (u, v) in G_sub.edges():
        w = G_sub[u][v]['weight']
        # scale for visualization
        edge_widths.append(3.0 * w)

    # Choose a layout
    if layout == "spring":
        pos = nx.spring_layout(G_sub, k=0.3, iterations=50, seed=42)
    elif layout == "kamada":
        pos = nx.kamada_kawai_layout(G_sub, weight='weight')
    else:
        # fallback
        pos = nx.spring_layout(G_sub, seed=42)

    # Optionally shorten labels by splitting on '|'
    labels = {node: node.split('|')[0] for node in G_sub.nodes()}

    # Plot
    plt.figure(figsize=(20, 20), dpi=150)
    nx.draw_networkx_nodes(G_sub, pos,
                           node_size=node_sizes,
                           node_color="skyblue",
                           alpha=0.8)
    nx.draw_networkx_edges(G_sub, pos,
                           width=edge_widths,
                           edge_color="gray",
                           alpha=0.7)
    nx.draw_networkx_labels(G_sub, pos, labels=labels, font_size=8)
    plt.title(f"Participant {participant_id} - Top-{top_n} Subgraph (Mutation={mutation})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close()
    return participant_data.edge_index, edge_index, adj_top_n, edges_list

if __name__ == "__main__":
    checkpoint_path_global_age = (
        "/scratch/lcornelis/outputs/ray_results/"
        "TorchTrainer_2025-02-03_16-34-34/model=gat-v4,seed=42423_3995_act=sigmoid,"
        "adj_thresh=0.7000,batch_size=32,dropout=0.2000,l1_lambda=0.0000,lr=0.0087,"
        "lr_scheduler=La_2025-02-05_12-48-31/checkpoint_000631"
    )
    csv_path_global_age = (
        "percent_importances_model=gat-v4,seed=42423_3995_act=sigmoid,adj_thresh=0.7000,"
        "batch_size=32,dropout=0.2000,l1_lambda=0.0000,lr=0.0087,lr_scheduler=La_2025-02-05_12-48-31.csv"
    )

    checkpoint_path_nfl = "/scratch/lcornelis/outputs/ray_results/TorchTrainer_2024-10-29_13-49-44/model=gat-v4,seed=47436_511_act=tanh,adj_thresh=0.9000,batch_size=8,dropout=0.1000,l1_lambda=0.0000,lr=0.0001,lr_scheduler=CosineA_2024-10-29_18-37-51/checkpoint_000342"
    csv_path_nfl = "percent_importances_model=gat-v4,seed=47436_511_act=tanh,adj_thresh=0.9000,batch_size=8,dropout=0.1000,l1_lambda=0.0000,lr=0.0001,lr_scheduler=CosineA_2024-10-29_18-37-51.csv"

    participant_ids = [
        203658808, 203015047, 203424662, 203675150, 203444149, 203698781,
        203767335, 203920017, 203024967, 203107651, 203208248, 203846123,
        203874005, 203633810, 203237749, 203579685, 203754322, 203037518,
        203394884, 203075010, 203307931, 203056536, 203445036, 203484970,
        203117034, 203955060, 203711414, 203854714, 203025100, 203649860,
        203545890, 203363904, 203485014, 203292476, 203836355, 203083561,
        203412030, 203052412, 203787564, 203697600, 203232724, 203510962,
        203609568, 203557584, 203162182, 203686947, 203041416, 203228793,
        203554899, 203681871, 203132940
    ]

    # List of mutations corresponding to participant_ids
    mutations = [
        "MAPT", "C9orf72", "CTL", "MAPT", "MAPT", "CTL", "C9orf72", "C9orf72", "C9orf72", "CTL",
        "CTL", "CTL", "MAPT", "MAPT", "C9orf72", "MAPT", "C9orf72", "CTL", "C9orf72", "MAPT",
        "GRN", "GRN", "CTL", "GRN", "MAPT", "CTL", "C9orf72", "MAPT", "GRN", "MAPT", "GRN",
        "MAPT", "MAPT", "MAPT", "C9orf72", "C9orf72", "CTL", "C9orf72", "MAPT", "C9orf72",
        "C9orf72", "GRN", "GRN", "CTL", "CTL", "C9orf72", "GRN", "CTL", "CTL", "C9orf72", "GRN"
    ]

    # Iterate over participant_ids and mutations
    for idx, (participant_id, mutation) in enumerate(zip(participant_ids, mutations)):
        # Call the function with the current participant_id
        edge_index, attention_edge_index, adj_top_n, edges_list = load_attention_coefficient(
            checkpoint_path=checkpoint_path_global_age,
            csv_path=csv_path_global_age,
            participant_id=participant_id,
            participant_id_num=idx,
            mutation=mutation,
            top_n=200,  # Adjust as needed
            edge_threshold=0.01,  # Adjust if needed
            layout="spring"       # or "kamada"
        )

    # # Run for participant 203025100 with all nodes
    # edge_index, attention_edge_index, adj_top_n, edges_list_0 = load_attention_coefficient(
    #     checkpoint_path=checkpoint_path_global_age,
    #     csv_path=csv_path_global_age,
    #     participant_id=203025100,
    #     participant_id_num=participant_ids.index(203025100),
    #     top_n=7257,
    #     edge_threshold=0.01,  # Adjust if needed
    #     layout="spring"       # or "kamada"
    # )

    # # Run for participant 203658808 with all nodes
    # edge_index, attention_edge_index, adj_top_n, edges_list_1 = load_attention_coefficient(
    #     checkpoint_path=checkpoint_path_global_age,
    #     csv_path=csv_path_global_age,
    #     participant_id=203658808,
    #     participant_id_num=0,
    #     top_n=7257,
    #     edge_threshold=0.01,  # Adjust if needed
    #     layout="spring"       # or "kamada"
    # )

    # if edges_list_0 == edges_list_1:
    #     print("edges_list_0 and edges_list_1 are the same.")
    # else:
    #     print("edges_list_0 and edges_list_1 are different.")