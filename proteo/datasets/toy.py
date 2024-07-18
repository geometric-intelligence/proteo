import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyWGCNA
import torch
from scipy.stats import chi2_contingency, ks_2samp, ttest_ind
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, InMemoryDataset


class ToyDataset(InMemoryDataset):
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
        self.name = 'toy'
        self.root = root
        self.split = split
        assert split in ["train", "test"]
        self.config = config
        self.n_nodes = config.n_nodes
        self.n_graphs = config.n_graphs
        self.mean_control = config.mean_control  # mean = 10
        self.mean_carrier = config.mean_carrier  # mean = 20
        self.std_control = config.std_control  # std = 1
        self.std_carrier = config.std_carrier
        # right now, this uses the same normal distribution for each portein,
        #    the mean and std only depend on control versus carrier
        self.feature_dim = 1  # protein concentration is a scalar, ie, dim 1
        self.label_dim = 1  # NfL is a scalar, ie, dim 1

        path = os.path.join(self.processed_dir, f'{self.name}_{split}.pt')
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
        return [
            f"{self.name}_train.pt",
            f"{self.name}_test.pt",
        ]

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
        (
            train_features,
            train_labels,
            test_features,
            test_labels,
            adj_matrix,
        ) = self.generate_toy_data(self.config)

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

    def find_top_ks_values(self, csv_data):
        # Finding 30 most different  Kolmogorov–Smirnov values
        ks_stats = []
        for i in range(self.plasma_protein_col_range[0], self.plasma_protein_col_range[1]):
            protein_column = csv_data.columns[i]

            # Separate data by "Carrier" and "CTL"
            carrier_data = csv_data[csv_data['Carrier.Status'] == 'Carrier'][protein_column]
            ctl_data = csv_data[csv_data['Carrier.Status'] == 'CTL'][protein_column]

            carrier_data = carrier_data.dropna()
            ctl_data = ctl_data.dropna()
            carrier_data = carrier_data[np.isfinite(carrier_data)]
            ctl_data = ctl_data[np.isfinite(ctl_data)]
            # Perform Kolmogorov-Smirnov Test

            ks_statistic, ks_p_value = ks_2samp(carrier_data, ctl_data)
            ks_stats.append((protein_column, i, ks_statistic, ks_p_value))

        # Convert to DataFrame for easy sorting
        ks_stats_df = pd.DataFrame(
            ks_stats, columns=['Protein', 'Column', 'KS_Statistic', 'P Value']
        )

        # Sort by KS statistic in descending order and get top 30
        top_10_columns = ks_stats_df.sort_values(by='P Value', ascending=True).head(10)
        # Return "Column" values
        return top_10_columns['Column'].values

    def generate_toy_data(self, config):
        """Generate toy data for the toy dataset."""

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

    def load_nfl_values(self, csv_data, x_values):
        nfl = csv_data[x_values, self.nfl_col_id].astype(float)
        test_nfl_mean(nfl)
        nfl_mask = ~np.isnan(nfl)
        # Remove NaN values from nfl
        nfl = nfl[nfl_mask]
        test_nfl_mean_no_nan(nfl)
        nfl = log_transform(nfl)
        hist_path = os.path.join(self.processed_dir, 'histogram.jpg')
        plot_histogram(pd.DataFrame(nfl), save_to=hist_path)
        return nfl, nfl_mask

    def load_carrier(self, csv_data, x_values):
        carrier = csv_data[x_values, self.carrier_col_id].astype(str)
        carrier_mask = ~pd.isna(carrier)
        carrier = carrier[carrier_mask]
        carrier = np.where(carrier == 'Carrier', 1, np.where(carrier == 'CTL', 0, None)).astype(
            float
        )
        # Check for any None values, which indicate an unrecognized status
        if None in carrier:
            raise ValueError(
                "Encountered an unrecognized carrier status. Only 'Carrier' and 'CTL' are allowed."
            )
        return carrier, carrier_mask


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
