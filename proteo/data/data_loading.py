import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    auc,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import LabelBinarizer
from torch_geometric.data import Data, InMemoryDataset
import PyWGCNA

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR =  os.path.abspath(os.path.join(ROOT_DIR, '..', '..', '..','..','..'))
FEATURES_LABELS_FOLDER = os.path.join(
    TARGET_DIR, "data"
) #To Do: Need to find exact path 
ADJACENCY_FOLDER = os.path.join(PARENT_DIR, "MLA-GNN/example_data/input_adjacency_matrix/split") #To Do need to find exact path


################
# Data Utils
################


class AllFTDDataset(InMemoryDataset):
    """This is dataset used in AllFTD.
    This is a graph regression task.
    """

    def __init__(self, root, split, config):
        self.name = 'AllFTD'
        self.root = root
        self.split = split
        assert split in ["train", "test"]
        self.config = config
        super(AllFTDDataset, self).__init__(root)
        self.feature_dim = 1  # protein concentration is a scalar, ie, dim 1
        self.label_dim = 1  # survival is a scalar, ie, dim 1, CHANGE THIS FOR CLASSIFICATION, # of classes you have in grading

        path = os.path.join(self.processed_dir, f'{self.name}_{split}.pt')
        self.load(path)

    @property #TO DO: Is this needed?
    def raw_file_names(self):
        return ['test.csv']

    @property
    def processed_file_names(self):
        name = self.name
        return [f'{name}_train.pt', f'{name}_test.pt']


    def createst_graph_data(self, feature, label, adj_matrix):
        x = feature  # protein concentrations: what is on the nodes
        adj_tensor = torch.tensor(adj_matrix)
        # Find the indices where the matrix has non-zero elements
        pairs_indices = torch.nonzero(adj_tensor, as_tuple=False)
        # Extract the pairs of connected nodes
        edge_index = pairs_indices.tolist()
        return Data(x=x, edge_index=edge_index, y=label)

    def process(self):
        # Read data into huge `Data` list which will be a list of graphs
        train_features, train_labels, test_features, test_labels, adj_matrix = load_csv_data(
            1, self.config
        )

        train_data_list = []
        for feature, label in zip(train_features, train_labels):
            data = self.createst_graph_data(feature, label, adj_matrix)
            train_data_list.append(data)

        test_data_list = []
        for feature, label in zip(test_features, test_labels):
            data = self.createst_graph_data(feature, label, adj_matrix)
            test_data_list.append(data)
        self.save(train_data_list, os.path.join(self.processed_dir, f'{self.name}_train.pt'))
        self.save(test_data_list, os.path.join(self.processed_dir, f'{self.name}_test.pt'))


def load_csv_data(k, config):
    print("Loading data from:", FEATURES_LABELS_FOLDER + str(k))
    train_data_path = (
        FEATURES_LABELS_FOLDER + str(k) + '_train_320d_features_labels.csv'
    )  # TODO: Put into os.path.join format
    train_data = np.array(pd.read_csv(train_data_path, header=None))[1:, 2:].astype(float)
    print(train_data[:, 80:320].shape)  # 81-320 becomes 80-319ie, 80:320 because last one not taken
    train_features = torch.FloatTensor(
        train_data[:, 80:320].reshape(-1, 240, 1)
    )  # reshape goes from (840, 320) to [840, 320, 1]
    print(train_features.shape)
    train_labels = torch.LongTensor(train_data[:, 320:])
    print("Training features and labels:", train_features.shape, train_labels.shape)

    test_data_path = FEATURES_LABELS_FOLDER + str(k) + '_test_320d_features_labels.csv'
    test_data = np.array(pd.read_csv(test_data_path, header=None))[1:, 2:].astype(float)
    print(test_data[:, 81:320].shape)
    test_features = torch.FloatTensor(test_data[:, 80:320].reshape(-1, 240, 1))
    test_labels = torch.LongTensor(test_data[:, 320:])
    print("Testing features and labels:", test_features.shape, test_labels.shape)

    similarity_matrix = np.array(
        pd.read_csv(ADJACENCY_FOLDER + str(k) + '_adjacency_matrix.csv', header=None),
    ).astype(float)
    adj_matrix = torch.LongTensor(np.where(similarity_matrix > config.adj_thresh, 1, 0))
    adj_matrix = adj_matrix[80:, 80:]
    print("Adjacency matrix:", adj_matrix.shape)
    print("Number of edges:", adj_matrix.sum())

    if config.task == "grad":
        train_ids = train_labels[:, 2] >= 0
        train_labels = train_labels[train_ids]  # Tensor of format [   1, 1448,    2]
        train_features = train_features[train_ids, :]
        print(
            "Training features and grade labels after deleting NA labels:",
            train_features.shape,
            train_labels.shape,
        )

        test_ids = test_labels[:, 2] >= 0
        test_labels = test_labels[test_ids]
        test_features = test_features[test_ids, :]
        print(
            "Testing features and grade labels after deleting NA labels:",
            test_features.shape,
            test_labels.shape,
        )
    train_labels = train_labels[:, 1]  # Taking survival time
    print(train_labels.shape)
    test_labels = test_labels[:, 1]  # Taking survival time
    print(test_labels.shape)

    return train_features, train_labels, test_features, test_labels, adj_matrix


def load_dataset(k, config): #To Do : Why do we need this?
    train_features, train_labels, test_features, test_labels, _ = load_csv_data(1, config)
    # transform these vsriables to create an object of dataset class
    dataset = MyDataset(train_features, train_labels, test_features, test_labels, _)
    return dataset


def calculate_adjacency_matrix(config):
    # WGCNA parameters
    config.wgcna_power = 9
    config.wgcna_minModuleSize = 10
    config.wgcna_mergeCutHeight = 0.25


    # Read data
    geneExp = pd.read_csv(data_file) # col = genes, rows = samples 
    #Get rid of gene and sample labels
    print(geneExp.shape)
    geneExp = geneExp.iloc[:, 2:322].values #can change this to 81:322 to match what they trained on
    print(geneExp.shape)
    print(geneExp[1].shape)
    # Convert elements to float.
    geneExp = geneExp.astype(float)

    # Calculate adjacency matrix.
    adjacency = PyWGCNA.WGCNA.adjacency(geneExp, power = config.wgcna_power, adjacencyType="signed hybrid")
    # Using adjacency matrix calculate the topological overlap matrix (TOM).
    # TOM = PyWGCNA.WGCNA.TOMsimilarity(adjacency)

    #Convert to dataframe.
    adjacency_df = pd.DataFrame(adjacency)
    print(adjacency_df.shape)

    adjacency_df.to_csv(os.path.join(OUTPUT_FOLDER, "split1_adjacency_matrix.csv"), index=None, header=None)``