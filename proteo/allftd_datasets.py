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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from torch_geometric.data import Data, InMemoryDataset
import PyWGCNA

# TODO: decide on folder
ADJACENCY_FOLDER = os.path.join(
    PARENT_DIR, "MLA-GNN/example_data/input_adjacency_matrix/split") #To Do need to find exact path

CSV_PATH = "/home/data/ALLFTD_dataset_for_nina_louisa.csv"

class AllFTDDataset(InMemoryDataset):
    """This is dataset used in AllFTD.
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
        self.name = 'AllFTD'
        self.root = root
        self.split = split
        assert split in ["train", "test"]
        self.config = config
        super(AllFTDDataset, self).__init__(root)
        self.feature_dim = 1  # protein concentration is a scalar, ie, dim 1
        self.label_dim = 1  # NfL is a scalar, ie, dim 1

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
    print("Loading data from:", CSV_PATH)
    csv_data = np.array(pd.read_csv(CSV_PATH))
    has_plasma = csv_data[:, 9].astype(int)
    # Note: Only select 50 proteins to debug
    plasma_protein = csv_data[has_plasma, 10:50].astype(float)
    nfl = csv_data[:, 8].astype(float)

    features = plasma_protein
    labels = nfl

    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.20, random_state=42)

    adj_matrix_path = os.path.join(ADJACENCY_FOLDER, 'allftd_adj_matrix.csv')

    if os.path.exists(adj_matrix_path):
        adj_matrix = np.array(pd.read_csv(adj_matrix_path)).astype(float)
    else:
        adj_matrix = calculate_adjacency_matrix(config)

    print("Adjacency matrix:", adj_matrix.shape)
    print("Number of edges:", adj_matrix.sum())

    return train_features, train_labels, test_features, test_labels, adj_matrix



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

    adjacency_df.to_csv(os.path.join(OUTPUT_FOLDER, "split1_adjacency_matrix.csv"), index=None, header=None)

    #    similarity_matrix = np.array(
    #     pd.read_csv(),
    # ).astype(float)
    # adj_matrix = torch.LongTensor(np.where(similarity_matrix > config.adj_thresh, 1, 0))
    # adj_matrix = adj_matrix[80:, 80:]