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

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(ROOT_DIR)
FEATURES_LABELS_FOLDER = os.path.join(
    PARENT_DIR, "MLA-GNN/example_data/input_features_labels/split"
)
ADJACENCY_FOLDER = os.path.join(PARENT_DIR, "MLA-GNN/example_data/input_adjacency_matrix/split")


# import lifelines
# from lifelines.utils import concordance_index
# from lifelines.statistics import logrank_test


################
# Data Utils
################


class MLAGNNDataset(InMemoryDataset):
    """This is dataset used in MLAGNN.
    This is a graph regression task.

    root will be something like os.path.join(ROOT_DIR, "data", "FAD")
    """

    def __init__(self, root, split, config):
        self.name = 'MLAGNN'
        self.root = root
        self.split = split
        assert split in ["train", "test"]
        self.config = config
        super(MLAGNNDataset, self).__init__(root)
        # self.load(self.processed_paths[0])
        self.feature_dim = 1  # protein concentration is a scalar, ie, dim 1
        self.label_dim = 1  # survival is a scalar, ie, dim 1, CHANGE THIS FOR CLASSIFICATION, # of classes you have in grading
        self.test_features = None
        self.test_labels = None
        self.adj_matrix = None
        path = os.path.join(self.processed_dir, f'{self.name}_{split}.pt')
        self.load(path)

    @property
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
        self.test_features = test_features
        self.test_labels = test_labels
        self.adj_matrix = adj_matrix

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


def load_dataset(k, config):
    train_features, train_labels, test_features, test_labels, _ = load_csv_data(1, config)
    # transform these vsriables to create an object of dataset class
    dataset = MyDataset(train_features, train_labels, test_features, test_labels, _)
    return dataset


################
# Grading Utils
################
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def print_model(model, optimizer):
    print(model)
    print("Model's statest_dict:")
    # Print model's statest_dict
    for param_tensor in model.statest_dict():
        print(param_tensor, "\t", model.statest_dict()[param_tensor].size())
    print("optimizer's statest_dict:")
    # Print optimizer's statest_dict
    for var_name in optimizer.statest_dict():
        print(var_name, "\t", optimizer.statest_dict()[var_name])


def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1.0 / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


def computest_ROC_AUC(test_pred, gt_labels):
    enc = LabelBinarizer()
    enc.fit(gt_labels)
    labels_oh = enc.transform(gt_labels)  ## convert to one_hot grade labels.
    # print(gt_labels, labels_oh, test_pred.shape)
    fpr, tpr, thresh = roc_curve(labels_oh.ravel(), test_pred.ravel())
    aucroc = auc(fpr, tpr)

    return aucroc


def computest_metrics(test_pred, gt_labels):
    enc = LabelBinarizer()
    enc.fit(gt_labels)
    labels_oh = enc.transform(gt_labels)  ## convert to one_hot grade labels.

    # print(gt_labels, labels_oh, test_pred.shape)
    # print(labels_oh, test_pred)
    idx = np.argmax(test_pred, axis=1)
    # print(gt_labels, idx)
    labels_and_pred = np.concatenate((gt_labels, idx))
    test_pred = enc.fit(labels_and_pred).transform(labels_and_pred)[gt_labels.shape[0] :, :]
    # print(test_pred)
    macro_f1_score = f1_score(labels_oh, test_pred, average='macro')
    # micro_f1_score = f1_score(labels_oh, test_pred, average='micro') #equal to accuracy.
    precision = precision_score(labels_oh, test_pred, average='macro')
    recall = recall_score(labels_oh, test_pred, average='macro')
    # kappa = cohen_kappa_score(labels_oh, test_pred)

    return macro_f1_score, precision, recall


################
# Survival Utils
################
def CoxLoss(survtime, censor, hazard_pred):
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    # print("R mat shape:", R_mat.shape)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i, j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).cuda()
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    # print("censor and theta shape:", censor.shape, theta.shape)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor)
    return loss_cox


def accuracy_cox(hazardsdata, labels):
    # This accuracy is based on estimated survival events against true survival events
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)


def cox_log_rank(hazardsdata, labels, survtime_all):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return pvalue_pred


def CIndex(hazards, labels, survtime_all):
    concord = 0.0
    total = 0.0
    N_test = labels.shape[0]
    for i in range(N_test):
        if labels[i] == 1:
            for j in range(N_test):
                if survtime_all[j] > survtime_all[i]:
                    total += 1
                    if hazards[j] < hazards[i]:
                        concord += 1
                    elif hazards[j] < hazards[i]:
                        concord += 0.5

    return concord / total


def CIndex_lifeline(hazards, labels, survtime_all):
    return concordance_index(survtime_all, -hazards, labels)


################
# Layer Utils
################
def define_act_layer(act_type='Tanh'):
    if act_type == 'Tanh':
        act_layer = nn.Tanh()
    elif act_type == 'ReLU':
        act_layer = nn.ReLU()
    elif act_type == 'Sigmoid':
        act_layer = nn.Sigmoid()
    elif act_type == 'LSM':
        act_layer = nn.LogSoftmax(dim=1)
    elif act_type == "none":
        act_layer = None
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return act_layer
