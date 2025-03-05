import torch
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from proteo.datasets.ftd_folds import FTDDataset
from checkpoint_analysis import load_checkpoint, load_config
from proteo.train import construct_datasets
from sklearn.model_selection import train_test_split

def load_patient_demographics(config):
    # Make an instance of the FTDDataset class to get the top proteins
    root = config.data_dir
    random_state = config.random_state
    train_dataset = FTDDataset(root, "train", config)
    _, _, _, filtered_sex_col, filtered_mutation_col, filtered_age_col, filtered_did_col, filtered_gene_col= train_dataset.load_csv_data_pre_pt_files(config)
    # Splitting indices only
    train_sex_labels, test_sex_labels, train_mutation_labels, test_mutation_labels, train_age_labels, test_age_labels, train_did_labels, test_did_labels, train_gene_col, test_gene_col = train_test_split(filtered_sex_col, filtered_mutation_col, filtered_age_col, filtered_did_col, filtered_gene_col, test_size=0.20, random_state=random_state)
    return train_did_labels, test_did_labels, train_sex_labels, test_sex_labels, train_mutation_labels, test_mutation_labels, train_age_labels, test_age_labels, train_gene_col, test_gene_col

def load_attention_coefficient(
    checkpoint_path: str,
    csv_path: str,
    participant_id_num: int,
    top_n: int = 30,
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
        participant_id_num (int): The index of the participant to visualize.
        top_n (int): Number of top nodes (by importance) to include in subgraph.
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

    # Make sure test_dataset[0] truly corresponds to participant_id_num 
    # (if not, map participant_id_num -> correct index).
    participant_data = test_dataset[participant_id_num]
    print("participant_data.edge_index", participant_data.edge_index)

    # 3) Forward pass to get attention from the first layer
    with torch.no_grad():
        pred, aux, (edge_index1, alpha_1), (edge_index2, alpha_2) = model(
            participant_data.x,
            edge_index=participant_data.edge_index,
            data=participant_data,
            return_attention_weights=True
        )

    # If multiple heads, average across them
    if alpha_1.dim() == 2:
        alpha_1 = alpha_1.mean(dim=1)  # shape [E]
        print("alpha_1", alpha_1)
    alpha_1 = alpha_1.cpu().numpy()  # numpy array of attention

    if alpha_2.dim() == 2:
        alpha_2 = alpha_2.mean(dim=1)  # shape [E]
        print("alpha_2", alpha_2)
    alpha_2 = alpha_2.cpu().numpy()  # numpy array of attention

    print("edge_index shape:", edge_index1.shape)  # [2, E]
    print("Alpha shape:", alpha_1.shape)
    E = edge_index1.size(1)

    # 4) Load the CSV containing the importance scores
    importance_df = pd.read_csv(csv_path, index_col=0)

    # Exclude first four columns if needed (as in your example)
    protein_columns = importance_df.columns[4:]
    print("Number of protein_columns:", len(protein_columns))


    # Load patient demographics
    _, test_did_labels, _, _, _, test_mutation_labels, _, test_age_labels, _, _ = load_patient_demographics(config)
    # Get participant_id, mutation, and age for the given participant_id_num
    participant_id = test_did_labels.iloc[participant_id_num]  # Use iloc for positional indexing
    mutation = test_mutation_labels.iloc[participant_id_num]
    age = test_age_labels.iloc[participant_id_num]

    # Select the row for the specified participant
    row = importance_df.loc[participant_id, protein_columns]
    row = pd.to_numeric(row, errors='coerce')  # convert to numeric if needed

    # Identify top-N highest and lowest proteins by mean importance
    top_n_highest_proteins = row.nlargest(top_n).index.to_list()
    top_n_lowest_proteins = row.nsmallest(top_n).index.to_list()
    print("len top_n_lowest_proteins", len(top_n_lowest_proteins))
    top_n_proteins = top_n_highest_proteins + top_n_lowest_proteins

    # Create mapping from protein to node index
    protein_to_node = {protein: idx for idx, protein in enumerate(protein_columns)}
    top_n_indices = [protein_to_node[protein] for protein in top_n_proteins]
    top_n_set = set(top_n_indices)


    #print("Top N indices:", top_n_indices)

    # --------------------------
    # OPTIONAL: Filter edges before building the subgraph
    #  (a) remove self-loops

    src, dst = edge_index1[0], edge_index1[1]

    #(a) remove self-loops
    no_self_loop_mask = (src != dst)

    # Filter edge_index and alpha_1
    edge_index = edge_index1[:, no_self_loop_mask]
    alpha_1 = alpha_1[no_self_loop_mask]

    # 6) Re-build adjacency (top_n x top_n) from the filtered edges
    E_filtered = edge_index.size(1)
    node2idx = {nid: i for i, nid in enumerate(sorted(top_n_set))}
    adj_top_n = np.zeros((2 * top_n, 2 * top_n), dtype=np.float32)
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
    # Create a mapping from the node index in the subgraph to the protein name
    mapping = {i: protein_columns[top_n_indices[i]] for i in range(len(top_n_indices))}
    G_sub = nx.relabel_nodes(G_sub, mapping)

    # Node size ~ importance
    node_sizes = [3000 * row[protein] for protein in top_n_proteins]

    # Node color based on importance
    node_colors = ['red' if protein in top_n_highest_proteins else 'blue' for protein in top_n_proteins]

    # Edge widths ~ attention
    edge_widths = []
    for (u, v) in G_sub.edges():
        w = G_sub[u][v]['weight']
        # scale for visualization
        edge_widths.append(10 * w)

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
                           node_color=node_colors,
                           alpha=0.8)
    nx.draw_networkx_edges(G_sub, pos,
                           width=edge_widths,
                           edge_color="gray",
                           alpha=0.7)
    nx.draw_networkx_labels(G_sub, pos, labels=labels, font_size=12)
    plt.title(f"Participant {participant_id} - Age: {age} - Mutation: {mutation} - Top-{top_n} Subgraph")
    plt.axis("off")
    plt.tight_layout()
    output_dir = "gnn_vis"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"gnn_vis/node_visual_{participant_id}.png")
    plt.close()
    return participant_data.edge_index, edge_index1, adj_top_n, edges_list, alpha_1

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

    # Iterate over participant indices
    for participant_id_num in range(51):
        print("participant_id_num", participant_id_num)
        edge_index, attention_edge_index, adj_top_n, edges_list, alpha_1 = load_attention_coefficient(
            checkpoint_path=checkpoint_path_nfl,
            csv_path=csv_path_nfl,
            participant_id_num=participant_id_num,
            top_n=200,  # Adjust as needed
            layout="spring"       # or "kamada"
        )
    '''
    # Run for participant 203025100 with all nodes
    edge_index, attention_edge_index1, attention_edge_index2, adj_top_n, edges_list_0, alpha_1_0, alpha_2_0 = load_attention_coefficient(
        checkpoint_path=checkpoint_path_global_age,
        csv_path=csv_path_global_age,
        participant_id_num=10,
        top_n=1000,
        layout="spring"       # or "kamada"
    )

    # Run for participant 203658808 with all nodes
    edge_index, attention_edge_index1, attention_edge_index2, adj_top_n, edges_list_1, alpha_1_1, alpha_2_1 = load_attention_coefficient(
        checkpoint_path=checkpoint_path_global_age,
        csv_path=csv_path_global_age,
        participant_id_num=15,
        top_n=1000,
        layout="spring"       # or "kamada"
    )

    if np.array_equal(alpha_2_0, alpha_2_1):
        print("edges_list_0 and edges_list_1 are the same.")
    else:
        for idx, (val_0, val_1) in enumerate(zip(alpha_1_0, alpha_1_1)):
            if abs(val_0 - val_1) > 0.000001:
                print(f"different at index {idx} with values: {val_0} {val_1}")
        print("edges_list_0 and edges_list_1 are different.")
    '''

   