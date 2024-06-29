import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyWGCNA
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, InMemoryDataset


class FTDDataset(InMemoryDataset):
    """This is dataset used in FTD.
    This is a graph regression task.

    **Rows:**
    - 0: Column Headers
    - 1 - 531 : Patient ID Number *(int)*

    **Columns:**
    - 0: DID *(int):* Patient ID
    - 1: Mutation *(string)*: CTL (Control), MAPT, C9orf72, GRN
    - 2: AGE_AT_VISIT *(int)*
    - 3: SEX_AT_BIRTH *(string)*: M, F
    - 4: Carrier.Status *(string)*: Carrier, CTL
    - 5: Gene.Dx *(string)*:  mutation status + clinical status
    (“PreSx” suffix = presymptomatic and “Sx” suffix = symptomatic)
    - 6: GLOBALCOG.ZCORE *(float)*: global cognition composite score
    - 7: FTLDCDR_SBL *(int)*: CDR sum of boxes - Clinical Dementia Rating Scale (CDR)
    is a global assessment instrument that yields global and Sum of Boxes (SOB) scores,
    with the global score regularly used in clinical and research settings
    to stage dementia severity. Higher is worse.
    - 8: NFL3_MEAN *(float):* plasma NfL concentrations

    - 9: HasPlasma? *(int)*: 1, 0 (519 Yes)
    - 10 - 7298: Proteins *(float)*:

    Protein variables are annotated as
      Protein Symbol | UniProt ID^Sequence ID| Matrix (CSF or PLASMA).
      The sequence ID is present only if there is more than one target
      for a given protein: e.g.,
      ABL2|P42684^SL010488@seq.3342.76|PLASMA ,
      ABL2|P42684^SL010488@seq.5261.13|PLASMA

    - 7299: HasCSF? *(int)*: 1, 0 (254 Yes)
    - 7300 - 14588: Proteins *(float)*:
    - 14589 - 15212: Clinical Data - maybe not necessary for right now.

    """

    def __init__(self, root, split, config):
        self.name = 'FTD'
        self.root = root
        self.split = split
        assert split in ["train", "test"]
        self.config = config
        self.has_plasma_col_id = 9
        self.plasma_protein_col_range = (10, 7299)  # 7299
        self.nfl_col_id = 8
        self.adj_str = f'adj_thresh_{config.adj_thresh}'

        super(FTDDataset, self).__init__(root)
        self.feature_dim = 1  # protein concentration is a scalar, ie, dim 1
        self.label_dim = 1  # NfL is a scalar, ie, dim 1

        path = os.path.join(self.processed_dir, f'{self.name}_{self.adj_str}_{split}.pt')
        self.load(path)

    @property
    def raw_file_names(self):
        """Files that must be present in order to skip downloading them from somewhere.

        Then, the grandparent Dataset class automatically defines raw_paths as:
        raw_path = self.raw_dir + raw_filename
        where: self.processed_dir = self.root + "raw"

        See Also
        --------
        https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/dataset.py
        """
        return [self.config.raw_file_name]

    @property
    def processed_file_names(self):
        """Files that must be present in order to skip processing.

        The, the grandparent Dataset class automatically defines processed_paths as:
        processed_path = self.processed_dir + processed_filename
        where: self.processed_dir = self.root + "processed"

        See Also
        --------
        https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/dataset.py
        """
        return [f"{self.name}_{self.adj_str}_train.pt", f"{self.name}_{self.adj_str}_test.pt"]

    def create_graph_data(self, feature, label, adj_matrix):
        """Create Data object for each graph.

        Compute attributes x, edge_index, and y for each graph.
        """
        x = feature  # protein concentrations: what is on the nodes
        adj_tensor = torch.tensor(adj_matrix)
        # Find the indices where the matrix has non-zero elements
        pairs_indices = torch.nonzero(adj_tensor, as_tuple=False)
        # Extract the pairs of connected nodes
        edge_index = torch.tensor(pairs_indices.tolist())
        edge_index = torch.transpose(edge_index, 0, 1)  # reshape(edge_index, (2, -1))
        return Data(x=x, edge_index=edge_index, y=label)

    def process(self):
        """Read data into huge `Data` list, i.e., a list of graphs"""
        train_features, train_labels, test_features, test_labels, adj_matrix = self.load_csv_data(
            self.config
        )

        train_data_list = []
        for feature, label in zip(train_features, train_labels):
            data = self.create_graph_data(feature, label, adj_matrix)
            train_data_list.append(data)

        test_data_list = []
        for feature, label in zip(test_features, test_labels):
            data = self.create_graph_data(feature, label, adj_matrix)
            test_data_list.append(data)
        self.save(train_data_list, self.processed_paths[0])
        self.save(test_data_list, self.processed_paths[1])

    def load_csv_data(self, config):
        csv_path = self.raw_paths[0]
        print("Loading data from:", csv_path)
        csv_data = np.array(pd.read_csv(csv_path))
        has_plasma = csv_data[:, self.has_plasma_col_id].astype(int)
        has_plasma = has_plasma == 1  # Converting from indices to boolean
        nfl = csv_data[has_plasma, self.nfl_col_id].astype(float)
        nfl_mask = ~np.isnan(nfl)
        # Extract and convert the plasma_protein values for rows
        # where has_plasma is True and nfl is not NaN.
        plasma_protein = csv_data[
            has_plasma, self.plasma_protein_col_range[0] : self.plasma_protein_col_range[1]
        ][nfl_mask].astype(float)
        # Remove NaN values from nfl
        nfl = nfl[nfl_mask]
        nfl = log_transform(nfl)
        hist_path = os.path.join(self.processed_dir, 'histogram.jpg')
        plot_histogram(pd.DataFrame(nfl), save_to=hist_path)

        features = plasma_protein
        labels = nfl

        train_features, test_features, train_labels, test_labels = train_test_split(
            features, labels, test_size=0.20, random_state=42
        )
        train_features = torch.FloatTensor(train_features.reshape(-1, train_features.shape[1], 1))
        test_features = torch.FloatTensor(test_features.reshape(-1, test_features.shape[1], 1))
        train_labels = torch.FloatTensor(train_labels)
        test_labels = torch.FloatTensor(test_labels)
        print("Training features and labels:", train_features.shape, train_labels.shape)
        print("Testing features and labels:", test_features.shape, test_labels.shape)
        print("--> Total features and labels:", features.shape, labels.shape)

        adj_path = os.path.join(self.processed_dir, f'adjacency_{config.adj_thresh}.csv')
        # Calculate and save adjacency matrix
        if not os.path.exists(adj_path):
            calculate_adjacency_matrix(plasma_protein, save_to=adj_path)
        print(f"Loading adjacency matrix from: {adj_path}...")
        adj_matrix = np.array(pd.read_csv(adj_path, header=None)).astype(float)
        # Threshold adjacency matrix
        adj_matrix = torch.FloatTensor(np.where(adj_matrix > config.adj_thresh, 1, 0))
        print("Adjacency matrix:", adj_matrix.shape)
        print("Number of edges:", adj_matrix.sum())

        # Plot and save adjacency matrix as jpg
        cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "black"])
        plt.figure()
        plt.imshow(adj_matrix, cmap=cmap)
        plt.colorbar(ticks=[0, 1], label='Adjacency Value')
        plt.title("Visualization of Adjacency Matrix")
        plt.savefig(os.path.join(self.processed_dir, f'adjacency_{config.adj_thresh}.jpg'))
        plt.close()

        return train_features, train_labels, test_features, test_labels, adj_matrix


def calculate_adjacency_matrix(plasma_protein, save_to):
    """Calculate and save adjacency matrix."""
    plasma_protein_df = pd.DataFrame(plasma_protein)
    softThreshold = PyWGCNA.WGCNA.pickSoftThreshold(plasma_protein_df)
    print("Soft threshold:", softThreshold[0])
    adjacency = PyWGCNA.WGCNA.adjacency(
        plasma_protein, power=softThreshold[0], adjacencyType="signed hybrid"
    )
    # Using adjacency matrix calculate the topological overlap matrix (TOM).
    # TOM = PyWGCNA.WGCNA.TOMsimilarity(adjacency)
    adjacency_df = pd.DataFrame(adjacency)
    print(f"Saving adjacency matrix to: {save_to}...")
    adjacency_df.to_csv(save_to, header=None, index=False)


def plot_histogram(data, save_to):
    plt.hist(data, bins=30, alpha=0.5)
    plt.xlabel('NFL3_MEAN')
    plt.ylabel('Frequency')
    plt.title('Histogram of NFL3_MEAN')
    plt.savefig(save_to, format='jpg')


def log_transform(data):
    # Log transformation
    log_data = np.log(data)

    mean = np.mean(log_data)
    std = np.std(log_data)
    standardized_log_data = (log_data - mean) / std
    return standardized_log_data


def reverse_log_transform(standardized_log_data):
    # De-standardize the data
    mean = np.mean(standardized_log_data)
    std = np.std(standardized_log_data)
    log_data = (standardized_log_data * std) + mean

    # Reverse the log transformation by applying the exponential function
    original_data = np.exp(log_data)

    return original_data
