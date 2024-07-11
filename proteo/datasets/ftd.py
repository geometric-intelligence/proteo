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
        self.name = 'ftd'
        self.root = root
        self.split = split
        assert split in ["train", "test"]
        self.config = config
        self.has_plasma_col_id = 9
        self.plasma_protein_col_range = (10, 7299)
        self.has_csf_col_id = 7299
        self.csf_protein_col_range = (7300, 14588)
        self.nfl_col_id = 8
        self.carrier_status_col_id = 4
        self.mutation_status_col_id = 1
        self.adj_str = f'adj_thresh_{config.adj_thresh}'
        self.y_val_str = f'y_val_{config.y_val}'
        self.num_nodes_str = f'num_nodes_{config.num_nodes}'
        self.mutation_status_str = f'mutation_status_{config.mutation_status}'
        self.plasma_or_csf_str = f'{config.plasma_or_csf}'

        super(FTDDataset, self).__init__(root)
        self.feature_dim = 1  # protein concentration is a scalar, ie, dim 1
        self.label_dim = 1  # NfL is a scalar, ie, dim 1

        path = os.path.join(
            self.processed_dir,
            f'{self.name}_{self.y_val_str}_{self.adj_str}_{self.num_nodes_str}_{self.mutation_status_str}_{self.plasma_or_csf_str}_{split}.pt',
        )
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
            f"{self.name}_{self.y_val_str}_{self.adj_str}_{self.num_nodes_str}_{self.mutation_status_str}_{self.plasma_or_csf_str}_train.pt",
            f"{self.name}_{self.y_val_str}_{self.adj_str}_{self.num_nodes_str}_{self.mutation_status_str}_{self.plasma_or_csf_str}_test.pt",
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
        ) = self.load_csv_data(self.config)

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

    def find_top_ks_values(self, csv_data, config, measurement_col_range):
        ks_stats = []

        # Direct comparison for mutation status
        mutation_status = config.mutation_status
        
        if mutation_status in ["GRN", "MAPT", "C9orf72", "CTL"]:
            condition1 = csv_data['Mutation'] == mutation_status
            condition2 = csv_data['Mutation'] == "CTL"
        else:
            raise ValueError("Invalid mutation status specified.")

        for i in range(measurement_col_range[0], measurement_col_range[1]):
            protein_column = csv_data.columns[i]

            mutation_data = csv_data[condition1][protein_column]
            other_data = csv_data[condition2][protein_column]

            mutation_data = mutation_data.dropna()
            other_data = other_data.dropna()
            mutation_data = mutation_data[np.isfinite(mutation_data)]
            other_data = other_data[np.isfinite(other_data)]

            ks_statistic, ks_p_value = ks_2samp(mutation_data, other_data)
            ks_stats.append((protein_column, i, ks_statistic, ks_p_value))

        ks_stats_df = pd.DataFrame(
            ks_stats, columns=['Protein', 'Column', 'KS_Statistic', 'P Value']
        )
        top_columns = ks_stats_df.sort_values(by='P Value', ascending=True).head(config.num_nodes)

        # Save the plasma_protein_names to a file
        plasma_protein_names = top_columns['Protein'].values
        file_path = os.path.join(
            self.processed_dir,
            f'top_proteins_num_nodes_{config.num_nodes}_mutation_status_{config.mutation_status}_{config.plasma_or_csf}.npy',
        )
        np.save(file_path, plasma_protein_names)

        return top_columns['Column'].values

    def load_csv_data(self, config):
        csv_path = self.raw_paths[0]
        print("Loading data from:", csv_path)
        pre_array_csv_data = pd.read_csv(csv_path)  # for KS scores
        csv_data = np.array(pre_array_csv_data)
        if config.plasma_or_csf == 'plasma':
            print("Using plasma data.")
            has_measurement_col_id = self.has_plasma_col_id
            measurement_col_range = self.plasma_protein_col_range
        elif config.plasma_or_csf == 'csf':
            print("Using CSF data.")
            has_measurement_col_id = self.has_csf_col_id
            measurement_col_range = self.csf_protein_col_range
        else:
            raise ValueError("Invalid plasma_or_csf. Must be 'plasma' or 'csf'.")
        # Get the indices of the rows where has_measurement is True
        has_measurement = csv_data[:, has_measurement_col_id].astype(int)
        #test_has_plasma_col_id(has_plasma)
        has_measurement = has_measurement == 1  # Converting from indices to boolean
        print("Number of patients with measurements:", np.sum(has_measurement))
        #test_boolean_plasma(has_plasma)

        # Additional filtering based on mutation_status, always take mutation status and control
        if config.mutation_status in ['GRN', 'MAPT', 'C9orf72']:
            mutation_filter = np.isin(csv_data[:, self.mutation_status_col_id], (config.mutation_status, 'CTL'))
        elif config.mutation_status == 'CTL':
            mutation_filter = np.ones_like(has_measurement, dtype=bool)
        else: 
            raise ValueError("Invalid mutation status specified.")
        print("Number of patients with mutation status:", np.sum(mutation_filter))
        combined_filter = has_measurement & mutation_filter
        print("Number of patients with measurements and mutation status:", np.sum(combined_filter))

        if config.y_val == 'nfl':
            y_val, y_val_mask = self.load_nfl_values(csv_data, combined_filter)
        elif config.y_val == 'carrier_status':
            y_val, y_val_mask = self.load_carrier_status(csv_data, combined_filter)
        else:
            "Invalid y_val. Must be 'nfl' or 'carrier_status'."
        # Extract and convert the plasma_protein values for rows
        # where has_plasma is True and nfl is not NaN.
        top_protein_col_indices = self.find_top_ks_values(pre_array_csv_data, config, measurement_col_range)
        top_proteins = csv_data[combined_filter, :][:, top_protein_col_indices][y_val_mask].astype(
            float
        )
        # test_plasma_protein(plasma_protein) TODO: remove?
        features = top_proteins
        labels = y_val

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

        adj_path = os.path.join(
            self.processed_dir,
            f'adjacency_{config.adj_thresh}_num_nodes_{config.num_nodes}_mutation_status_{config.mutation_status}_{config.plasma_or_csf}.csv',
        )
        # Calculate and save adjacency matrix
        if not os.path.exists(adj_path):
            calculate_adjacency_matrix(top_proteins, save_to=adj_path)
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
        plt.savefig(
            os.path.join(
                self.processed_dir,
                f'adjacency_{config.adj_thresh}_num_nodes_{config.num_nodes}_mutation_status_{config.mutation_status}_{config.plasma_or_csf}.jpg',
            )
        )
        plt.close()

        return (
            train_features,
            train_labels,
            test_features,
            test_labels,
            adj_matrix,
        )

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

    def load_carrier_status(self, csv_data, x_values):
        carrier_status = csv_data[x_values, self.carrier_status_col_id].astype(str)
        carrier_mask = ~pd.isna(carrier_status)
        carrier_status = carrier_status[carrier_mask]
        carrier_status = np.where(
            carrier_status == 'Carrier', 1, np.where(carrier_status == 'CTL', 0, None)
        ).astype(float)
        # Check for any None values, which indicate an unrecognized status
        if None in carrier_status:
            raise ValueError(
                "Encountered an unrecognized carrier status. Only 'Carrier' and 'CTL' are allowed."
            )
        return carrier_status, carrier_mask


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


# Unit Tests:


def test_has_plasma_col_id(has_plasma):
    expected_start = [1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    expected_end = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    assert np.array_equal(
        has_plasma[0:10], expected_start
    ), f"Expected start does not match: {has_plasma[0:10]}"
    assert np.array_equal(
        has_plasma[-10:], expected_end
    ), f"Expected end does not match: {has_plasma[-10:]}"


def test_boolean_plasma(has_plasma):
    expected_start = [True, True, False, True, True, True, True, True, True, True]
    expected_end = [True, True, True, True, True, True, True, True, True, True]

    assert np.array_equal(
        has_plasma[0:10], expected_start
    ), f"Expected start does not match: {has_plasma[0:10]}"
    assert np.array_equal(
        has_plasma[-10:], expected_end
    ), f"Expected end does not match: {has_plasma[-10:]}"


def test_nfl_mean(nfl):
    expected_start = [
        5.373789749,
        4.96802777,
        7.849056381,
        9.952806685,
        19.09750386,
        3.020361499,
        3.955332738,
        5.609756833,
        2.543991562,
        20.35120703,
    ]
    expected_end = [
        4.704597833,
        4.451686665,
        10.11666739,
        np.nan,
        4.906927378,
        np.nan,
        32.57894769,
        10.47815389,
        13.82665788,
        5.090618157,
    ]

    assert np.allclose(
        nfl[0:10], expected_start, equal_nan=True
    ), f"Expected start does not match: {nfl[0:10]}"
    assert np.allclose(
        nfl[-10:], expected_end, equal_nan=True
    ), f"Expected end does not match: {nfl[-10:]}"


def test_nfl_mean_no_nan(nfl):
    expected_start = [
        5.373789749,
        4.96802777,
        7.849056381,
        9.952806685,
        19.09750386,
        3.020361499,
        3.955332738,
        5.609756833,
        2.543991562,
        20.35120703,
    ]
    expected_end = [
        3.193790446,
        5.375672435,
        4.704597833,
        4.451686665,
        10.11666739,
        4.906927378,
        32.57894769,
        10.47815389,
        13.82665788,
        5.090618157,
    ]
    assert np.allclose(nfl[0:10], expected_start), f"Expected start does not match: {nfl[0:10]}"
    assert np.allclose(nfl[-10:], expected_end), f"Expected end does not match: {nfl[-10:]}"


def test_plasma_protein(plasma_protein):
    first_col = [
        10.42909285,
        10.40492866,
        10.39360497,
        10.7173337,
        10.59516422,
        10.91101741,
        11.27035373,
        10.27972664,
        10.79498438,
        10.67551604,
    ]
    last_col = [
        14.29598412,
        14.13749552,
        13.9971883,
        14.03512556,
        14.00098588,
        14.59941311,
        13.65365122,
        13.65650307,
        13.76113565,
        13.8820411,
    ]

    assert np.allclose(
        plasma_protein[:10, 0], first_col
    ), f"First column does not match: {plasma_protein[:10, 0]}"
    # Only works if you take all the columns
    # assert np.allclose(plasma_protein[:10, -1], last_col), f"Last column does not match: {plasma_protein[:10, -1]}"
