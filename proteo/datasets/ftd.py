import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyWGCNA
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, InMemoryDataset

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

ADJACENCY_FOLDER = os.path.join(ROOT_DIR, "data", "ftd", "processed")
ADJACENCY_PATH = os.path.join(ADJACENCY_FOLDER, "adjacency_matrix.csv")
CSV_PATH = os.path.join(ROOT_DIR, "data", "ALLFTD_dataset_for_nina_louisa.csv")


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
        super(FTDDataset, self).__init__(root)
        self.feature_dim = 1  # protein concentration is a scalar, ie, dim 1
        self.label_dim = 1  # NfL is a scalar, ie, dim 1
        path = os.path.join(self.processed_dir, f'{self.name}_{split}.pt')

        # Note: It seems that this is needed to load the data
        # However, it is taking forever and is the reason why multi GPUs is failing.
        self.load(path)

    @property  # TO DO: Is this needed?
    def raw_file_names(self):
        return ['test.csv']

    @property
    def processed_file_names(self):
        name = self.name
        return [f'{name}_train.pt', f'{name}_test.pt']

    def create_graph_data(self, feature, label, adj_matrix):
        x = feature  # protein concentrations: what is on the nodes
        adj_tensor = torch.tensor(adj_matrix)
        # Find the indices where the matrix has non-zero elements
        pairs_indices = torch.nonzero(adj_tensor, as_tuple=False)
        # Extract the pairs of connected nodes
        edge_index = torch.tensor(pairs_indices.tolist())
        edge_index = torch.transpose(edge_index, 0, 1)  # reshape(edge_index, (2, -1))
        return Data(x=x, edge_index=edge_index, y=label)

    def process(self):
        # Read data into huge `Data` list which will be a list of graphs
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
        self.save(train_data_list, os.path.join(self.processed_dir, f'{self.name}_train.pt'))
        self.save(test_data_list, os.path.join(self.processed_dir, f'{self.name}_test.pt'))

    def load_csv_data(self, config):
        print("Loading data from:", CSV_PATH)
        csv_data = np.array(pd.read_csv(CSV_PATH))
        has_plasma = csv_data[:, self.has_plasma_col_id].astype(int)
        has_plasma = has_plasma == 1  # Converting from indices to boolean
        nfl = csv_data[has_plasma, self.nfl_col_id].astype(float)
        nfl_mask = ~np.isnan(nfl)
        plasma_protein = csv_data[
            has_plasma, self.plasma_protein_col_range[0] : self.plasma_protein_col_range[1]
        ][nfl_mask].astype(float) #Extract and convert the plasma_protein values for rows where has_plasma is True and nfl is not NaN.
        nfl = nfl[nfl_mask] # Remove NaN values from nfl 
        nfl = log_transform(nfl)
        plot_histogram(pd.DataFrame(nfl))

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

        if not os.path.exists(ADJACENCY_PATH):
            print(plasma_protein[0:5])
            calculate_adjacency_matrix(config, plasma_protein)
        adj_matrix = np.array(pd.read_csv(ADJACENCY_PATH, header=None)).astype(float)
        adj_matrix = torch.FloatTensor(
            np.where(adj_matrix > config.adj_thresh, 1, 0)
        )  # thresholding!

        # Plotting adjacency matrix
        cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "black"])
        plt.figure()
        plt.imshow(adj_matrix, cmap=cmap)
        plt.colorbar(ticks=[0, 1], label='Adjacency Value')
        plt.title("Visualization of Adjacency Matrix")
        plt.savefig(os.path.join(ADJACENCY_FOLDER, 'adjacency.jpg'))
        plt.close()

        print("Adjacency matrix:", adj_matrix.shape)
        print("Number of edges:", adj_matrix.sum())

        return train_features, train_labels, test_features, test_labels, adj_matrix


def calculate_adjacency_matrix(config, plasma_protein):
    # WGCNA parameters
    print(plasma_protein.shape)  # rows = samples ; cols = proteins,

    # Calculate adjacency matrix.
    plasma_protein_df = pd.DataFrame(plasma_protein)
    softThreshold = PyWGCNA.WGCNA.pickSoftThreshold(plasma_protein_df)
    print("Soft threshold:", softThreshold[0])
    adjacency = PyWGCNA.WGCNA.adjacency(
        plasma_protein, power=softThreshold[0], adjacencyType="signed hybrid"
    )
    # Using adjacency matrix calculate the topological overlap matrix (TOM).
    # TOM = PyWGCNA.WGCNA.TOMsimilarity(adjacency)
    # Convert to dataframe.
    adjacency_df = pd.DataFrame(adjacency)
    print(adjacency_df.shape)

    # if ADJACENCY_FOLDER doesn't exist, create it:
    if not os.path.exists(ADJACENCY_FOLDER):
        os.makedirs(ADJACENCY_FOLDER)
    adjacency_df.to_csv(ADJACENCY_PATH, header=None, index=False)
    #    similarity_matrix = np.array(
    #     pd.read_csv(),
    # ).astype(float)
    # adj_matrix = torch.LongTensor(np.where(similarity_matrix > config.adj_thresh, 1, 0))
    # adj_matrix = adj_matrix[80:, 80:]


def plot_histogram(data):
    plt.hist(data, bins=30, alpha=0.5)
    plt.xlabel('NFL3_MEAN')
    plt.ylabel('Frequency')
    plt.title('Histogram of NFL3_MEAN')
    histogram_path = os.path.join(ADJACENCY_FOLDER, 'histogram.jpg')
    plt.savefig(histogram_path, format='jpg')


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
