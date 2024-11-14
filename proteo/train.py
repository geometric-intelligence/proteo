"""Train one single neural network.

Lightning manages the training,
and thus we use:
- Lightning's WandbLogger logger, in our CustomWandbCallback,
- Lightning's ModelCheckpoint callback.

Here, pl_module.logger is WandbLogger's logger.


Notes
-----
When we do hyperparameter search in main.py,
Ray[Tune] takes over the training process, 
and thus we use instead:
- wandb.log, in our CustomWandbCallback,
- Ray's CheckpointConfig.

Here, pl_module.logger is Ray's dedicated logger.
"""

import gc
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
from config_utils import CONFIG_FILE, Config, read_config_from_file
from focal_loss.focal_loss import FocalLoss
from models.gat_v4 import GATv4
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import strategies as pl_strategies
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    LambdaLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAT, GCN, global_mean_pool
from torch_geometric.nn.models import MLP

import proteo.callbacks as proteo_callbacks
from proteo.datasets.ftd import (
    BINARY_Y_VALS_MAP,
    MULTICLASS_Y_VALS_MAP,
    Y_VALS_TO_NORMALIZE,
    FTDDataset,
)


class Proteo(pl.LightningModule):
    """Proteo Lightning Module that handles batching the graphs in training and validation.

    Parameters
    ----------
    config : Config
        The config object.
    in_channels : int, the number of features per node of the input graph. Currently, this is 1 because it is one protein measurement per node.
    out_channels : int, the number of features per output node.
    avg_node_degree : float
    """

    LOSS_MAP = {
        "binary_classification": torch.nn.BCEWithLogitsLoss,  # "binary cross-entropy loss with logits"
        "multiclass_classification": FocalLoss,
        "l1_regression": torch.nn.L1Loss,
        "mse_regression": torch.nn.MSELoss,
    }

    def __init__(
        self,
        config: Config,
        in_channels,
        out_channels,
        avg_node_degree,
        pos_weight,
        focal_loss_weight,
    ):
        """Initializes the proteo module by defining self.model according to the model specified in config.yml."""
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        # Save model parameters in a separate config object
        self.config_model = Config.parse_obj(getattr(config, config.model))
        self.avg_node_degree = avg_node_degree
        self.train_preds = []
        self.val_preds = []
        self.train_targets = []
        self.val_targets = []
        self.val_losses = []
        self.x0 = []
        self.x1 = []
        self.x2 = []
        self.multiscale = []
        self.pos_weight = pos_weight
        self.focal_loss_weight = focal_loss_weight
        self.min_val_loss = 1000
        self.min_train_loss = 1000

        if config.model == 'gat-v4':
            self.model = GATv4(
                in_channels=in_channels,
                hidden_channels=self.config_model.hidden_channels,
                out_channels=out_channels,
                heads=self.config_model.heads,
                dropout=self.config.dropout,
                act=self.config.act,
                which_layer=self.config_model.which_layer,
                use_layer_norm=self.config_model.use_layer_norm,
                fc_dim=self.config_model.fc_dim,
                fc_dropout=self.config_model.fc_dropout,
                fc_act=self.config_model.fc_act,
                num_nodes=self.config.num_nodes,
                weight_initializer=self.config_model.weight_initializer,
                use_master_nodes=self.config.use_master_nodes,
                master_nodes=self.config.master_nodes,
            )
        elif config.model == 'gat':
            self.model = GAT(
                in_channels=in_channels,
                hidden_channels=self.config_model.hidden_channels,
                out_channels=out_channels,
                num_layers=self.config_model.num_layers,
                heads=self.config_model.heads,
                v2=self.config_model.v2,
                dropout=self.config.dropout,
                act=self.config.act,
            )
        elif config.model == 'gcn':
            self.model = GCN(
                in_channels=in_channels,
                hidden_channels=self.config_model.hidden_channels,
                out_channels=out_channels,
                num_layers=self.config_model.num_layers,
                dropout=self.config.dropout,
                act=self.config.act,
            )
        elif config.model == "mlp":
            dropout = self.config.dropout
            dropout = [dropout] * (len(self.config_model.channel_list) - 1) #repeat same dropout for all layers
            self.model = MLP(
                channel_list = self.config_model.channel_list,
                dropout = dropout,
                act=self.config.act,
                norm = self.config_model.norm,
                plain_last = self.config_model.plain_last
            )
        else:
            raise NotImplementedError('Model not implemented yet')

    def forward(self, batch):
        """Forward pass of the model on the input batch.

        Parameters
        ----------
        batch: DataBatch object with attributes x, edge_index, y, and batch.
        batch.x: torch.Tensor of shape [num_nodes * batch_size, num_features]
        batch.edge_index: torch.Tensor of shape [2, num_edges * batch_size]
        batch.y: torch.Tensor of shape [batch_size]
        batch.batch: torch.Tensor of shape [num_nodes * batch_size]
        """
        if self.config.model == "gat-v4":
            pred, aux = self.model(batch.x, batch.edge_index, batch)
            self.x0.append(aux[0])  # check which format is best here.
            self.x1.append(aux[1])
            self.x2.append(aux[2])
            self.multiscale.append(aux[3])
            return pred
        if self.config.model == 'gat':
            # This returns a pred value for each node in the big graph
            pred_nodes = self.model(batch.x, batch.edge_index, batch=batch.batch)
            # Aggregate node features into graph-level features
            return global_mean_pool(pred_nodes, batch.batch)
        if self.config.model == 'gcn':
            pred_nodes = self.model(batch.x, batch.edge_index, batch=batch.batch)
            return global_mean_pool(pred_nodes, batch.batch)
        if self.config.model == 'mlp':
            if self.config.use_master_nodes:
                input_dim = self.config.num_nodes + len(self.config.master_nodes)
            else:
                input_dim = self.config.num_nodes
            batch_size = batch.x.shape[0]//input_dim
            batch.x = batch.x.view(batch_size, input_dim)
            return self.model(batch.x)
        raise NotImplementedError('Model not implemented yet')

    def training_step(self, batch):
        """Run single step in the training loop.

        It specifies how a batch of data is processed during training, including making predictions, calculating the loss, and logging metrics.

        Parameters
        ----------
        batch: DataBatch object with attributes x, edge_index, y, and batch.
        batch.x: torch.Tensor of shape [num_nodes * batch_size, num_features]
        batch.edge_index: torch.Tensor of shape [2, num_edges * batch_size]
        batch.y: torch.Tensor of shape [batch_size]
        batch.batch: torch.Tensor of shape [num_nodes * batch_size]
        """
        # Pred is shape [batch_size,1] and targets is shape [batch_size]
        pred = self.forward(batch)
        if torch.isnan(pred).any():
            # Check if there is nans in the input data
            print("train: pred has nan")
            print(f"?batch.x has nan: {torch.isnan(batch.x).any()}")
            raise ValueError
        if self.config.y_val != 'clinical_dementia_rating_global':
            target = batch.y.view(pred.shape)
        else:
            target = batch.y
        # Store predictions
        self.train_preds.append(pred.clone().detach().cpu())
        self.train_targets.append(target.clone().detach().cpu())

        # Reduction = "mean" averages over all samples in the batch,
        # providing a single average per batch.
        if self.config.y_val in BINARY_Y_VALS_MAP:
            # pos_weight is used to weight the positive class in the loss function
            device = pred.device
            self.pos_weight = self.pos_weight.to(device)
            loss_fn = self.LOSS_MAP["binary_classification"](
                pos_weight=self.pos_weight, reduction="mean"
            )
        elif self.config.y_val in MULTICLASS_Y_VALS_MAP:
            device = pred.device
            self.focal_loss_weight = self.focal_loss_weight.to(device)
            loss_fn = self.LOSS_MAP["multiclass_classification"](
                weights=self.focal_loss_weight, gamma=2, reduction="mean"
            )
            # Convert targets to ints for the loss function
            target = target.long()
            # Convert to probabilites before taking loss
            pred = torch.nn.Softmax(dim=-1)(pred)
        else:
            loss_fn = self.LOSS_MAP["mse_regression"](reduction="mean")
        loss = loss_fn(pred, target)

        # Calculate L1 regularization term to encourage sparsity
        # FIXME: With L1 regularization, the train_RMSE is not the RMSE
        # FIXME: L2 regularization not applied to val_loss, -> train and val losses cannot be compared
        if self.config.l1_lambda > 0:
            l1_lambda = self.config.l1_lambda  # Regularization strength
            l1_norm = sum(p.abs().sum() for p in self.parameters())
            loss = loss + l1_lambda * l1_norm

        return loss

    def validation_step(self, batch):
        """Run single step in the validation loop. It specifies how a batch of data is processed during validation, including making predictions, calculating the loss, and logging metrics.

        Parameters
        ----------
        batch: DataBatch object with attributes x, edge_index, y, and batch.
        batch.x: torch.Tensor of shape [num_nodes * batch_size, num_features]
        batch.edge_index: torch.Tensor of shape [2, num_edges * batch_size]
        batch.y: torch.Tensor of shape [batch_size]
        batch.batch: torch.Tensor of shape [num_nodes * batch_size]
        """
        pred = self.forward(batch)
        # Pred is shape [batch_size,1] and targets is shape [batch_size], only reshape if not doing multiclass classification
        if self.config.y_val != 'clinical_dementia_rating_global':
            target = batch.y.view(pred.shape)
        else:
            target = batch.y
        # Store predictions
        self.val_preds.append(pred.clone())
        self.val_targets.append(target.clone())

        # Reduction = "mean" averages over all samples in the batch,
        # providing a single average per batch.
        if self.config.y_val in BINARY_Y_VALS_MAP:
            # pos_weight is used to weight the positive class in the loss function
            device = pred.device
            self.pos_weight = self.pos_weight.to(device)
            loss_fn = self.LOSS_MAP["binary_classification"](
                pos_weight=self.pos_weight, reduction='none'
            )
        elif self.config.y_val in MULTICLASS_Y_VALS_MAP:
            device = pred.device
            self.focal_loss_weight = self.focal_loss_weight.to(device)
            loss_fn = self.LOSS_MAP["multiclass_classification"](
                weights=self.focal_loss_weight, gamma=2, reduction='none'
            )
            # Convert targets to ints for the loss function
            target = target.long()
            # Convert to probabilites before taking loss
            pred = torch.nn.Softmax(dim=-1)(pred)
        else:
            loss_fn = self.LOSS_MAP["mse_regression"](reduction='none')
        loss = loss_fn(pred, target)
        self.val_losses.append(loss)
        #self.log("val_loss", loss, on_epoch=True)
        #return loss
    
    # def on_validation_epoch_end(self):
    #     # outs is a list of whatever you returned in `validation_step`
    #     loss = torch.stack(self.val_losses).mean()
    #     self.log("val_loss", loss)
    #     return loss

    def configure_optimizers(self):
        assert self.config.optimizer == 'Adam'

        optimizer = Adam(
            self.parameters(),
            lr=self.config.lr,
            betas=(0.9, 0.999),
            weight_decay=self.config.weight_decay,
        )

        if self.config.lr_scheduler == 'LambdaLR':

            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + 1) / float(self.config.epochs + 1)
                return lr_l

            scheduler = LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif self.config.lr_scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
            )
        elif self.config.lr_scheduler == 'ExponentialLR':
            scheduler = ExponentialLR(optimizer, 0.1, last_epoch=-1)
        elif self.config.lr_scheduler == 'StepLR':
            scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        elif self.config.lr_scheduler == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs, eta_min=0)
        else:
            return NotImplementedError('scheduler not implemented:', self.config.lr_scheduler)

        # HACKALERT: Validation loss is logged twice, as val_loss and val_loss
        # So that Proteo's module can be used in train.py and main.py
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


def compute_avg_node_degree(dataset):
    # Calculate the average node degree for logging purposes
    num_nodes, _ = dataset.x.shape
    _, num_edges = dataset.edge_index.shape
    num_edges = num_edges / 2
    avg_node_degree = num_edges / num_nodes
    return avg_node_degree


def compute_pos_weight(test_dataset, train_dataset):
    """Function that computes the positive weight (ratio of ctl to carriers) for the binary cross-entropy loss function."""
    test_y_values = test_dataset.y
    train_y_values = train_dataset.y
    test_carrier_count = torch.sum(test_y_values == 1).item()
    train_carrier_count = torch.sum(train_y_values == 1).item()
    test_ctl_count = torch.sum(test_y_values == 0).item()
    train_ctl_count = torch.sum(train_y_values == 0).item()
    return torch.FloatTensor(
        [(train_ctl_count + test_ctl_count) / (train_carrier_count + test_carrier_count)]
    )


def compute_focal_loss_weight(config, test_dataset, train_dataset):
    '''Function that computes the weights (prevalence of) classes in the shape [1, num_classes] to be used in the focal loss function.'''
    test_y_values = test_dataset.y
    train_y_values = train_dataset.y
    complete_y_values = torch.cat((test_y_values, train_y_values))
    frequencies = []
    for key, value in MULTICLASS_Y_VALS_MAP[config.y_val].items():
        count = torch.sum(complete_y_values == value).item()
        frequencies.append(count)
    # Calculate weights inversely proportional to the frequencies
    frequencies = torch.tensor(frequencies, dtype=torch.float32)
    weights = 1.0 / frequencies
    return weights


def construct_datasets(config):
    root = config.data_dir
    print(f"Loading datasets from {root}")
    print(f"Absolute path: {os.path.abspath(root)}")
    print(f"Directory contents: {os.listdir(root)}")
    
    try:
        train_dataset = FTDDataset(root, "train", config)
        print("Train dataset loaded successfully")
    except Exception as e:
        print(f"Error loading train dataset: {str(e)}")
        raise
    
    try:
        test_dataset = FTDDataset(root, "test", config)
        print("Test dataset loaded successfully")
    except Exception as e:
        print(f"Error loading test dataset: {str(e)}")
        raise
    
    return train_dataset, test_dataset


def print_datasets(train_dataset, test_dataset):
    print(f"Train: {len(train_dataset)} samples")
    print(f"- train_dataset.x.shape: {train_dataset.x.shape}")
    print(f"- train_dataset.y.shape: {train_dataset.y.shape}")
    print(f"- train_dataset.sex.shape: {train_dataset.sex.shape}")
    print(f"- train_dataset.mutation.shape: {train_dataset.mutation.shape}")
    print(f"- train_dataset.age.shape: {train_dataset.age.shape}")
    print(f"- train_dataset.edge_index.shape: {train_dataset.edge_index.shape}")
    print(f"---- train_dataset[0].x.shape: {train_dataset[0].x.shape}")
    print(f"---- train_dataset[0].edge_index.shape: {train_dataset[0].edge_index.shape}")
    print(f"---- train_dataset[0].sex.shape: {train_dataset[0].sex.shape}")
    print(f"---- train_dataset[0].mutation.shape: {train_dataset[0].mutation.shape}")
    print(f"---- train_dataset[0].age.shape: {train_dataset[0].age.shape}")
    print(f"---- train_dataset[0].y.shape: {train_dataset[0].y.shape}")
    print(f"Test: {len(test_dataset)} samples")
    print(f"- test_dataset.x.shape: {test_dataset.x.shape}")
    print(f"- test_dataset.y.shape: {test_dataset.y.shape}")
    print(f"- test_dataset.sex.shape: {test_dataset.sex.shape}")
    print(f"- test_dataset.mutation.shape: {test_dataset.mutation.shape}")
    print(f"- test_dataset.age.shape: {test_dataset.age.shape}")
    print(f"- test_dataset.edge_index.shape: {test_dataset.edge_index.shape}")
    print(f"---- test_dataset[0].x.shape: {test_dataset[0].x.shape}")
    print(f"---- test_dataset[0].edge_index.shape: {test_dataset[0].edge_index.shape}")


def construct_loaders(config, train_dataset, test_dataset):
    # Make DataLoader objects to handle batching
    train_loader = DataLoader(  # makes into one big graph
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    return train_loader, test_loader


def print_loaders(train_loader, test_loader):
    first_train_batch = next(iter(train_loader))
    first_test_batch = next(iter(test_loader))
    print(f"First train batch: {len(first_train_batch)} samples")
    print(f"- first_train_batch.x.shape: {first_train_batch.x.shape}")
    print(f"- first_train_batch.y.shape: {first_train_batch.y.shape}")
    print(f"- first_train_batch.edge_index.shape: {first_train_batch.edge_index.shape}")
    print(f"- first_train_batch.batch.shape: {first_train_batch.batch.shape}")
    print(f"First test batch: {len(first_test_batch)} samples")
    print(f"- first_test_batch.x.shape: {first_test_batch.x.shape}")
    print(f"- first_test_batch.y.shape: {first_test_batch.y.shape}")
    print(f"- first_test_batch.edge_index.shape: {first_test_batch.edge_index.shape}")
    print(f"- first_test_batch.batch.shape: {first_test_batch.batch.shape}")
    print(f"- first_test_batch.sex.shape: {first_test_batch.sex.shape}")
    print(f"- first_test_batch.mutation.shape: {first_test_batch.mutation.shape}")
    print(f"- first_test_batch.age.shape: {first_test_batch.age.shape}")


def get_wandb_logger(config):
    """Get Weights and Biases logger."""
    wandb_api_key_path = os.path.join(config.root_dir, config.wandb_api_key_path)
    with open(wandb_api_key_path, 'r') as f:
        wandb_api_key = f.read().strip()
    os.environ['WANDB_API_KEY'] = wandb_api_key
    return pl_loggers.WandbLogger(
        config=config,
        project=config.project,
        save_dir=config.wandb_tmp_dir,  # dir needs to exist, otherwise wandb saves in /tmp
        offline=config.wandb_offline,
    )


def read_protein_file(processed_dir, config):
    file_path = os.path.join(
        processed_dir,
        f'top_proteins_num_nodes_{config.num_nodes}_mutation_{config.mutation}_{config.modality}_{config.sex}.npy',
    )
    print(f"Reading protein file: {file_path}")
    if os.path.exists(file_path):
        return np.load(file_path, allow_pickle=True)
    else:
        raise FileNotFoundError(f"The file {file_path} does not exist.")


def main():
    """Training and evaluation script for experiments."""
    torch.set_float32_matmul_precision('medium')  # for performance
    config = read_config_from_file(CONFIG_FILE)
    '''
    yaml_string = """
    wgcna_minModuleSize: 10
    gpu_per_worker: 1
    checkpoint_dir: '/scratch/lcornelis/outputs/checkpoints'
    wandb_api_key_path: 'wandb_api_key.txt'
    l1_lambda_max: 0.1
    gat_v4_fc_dropout: [0.1, 0.2, 0.5]
    devices: [0]
    gat_v4_weight_initializer: ['xavier', 'kaiming', 'orthogonal', 'truncated_normal', 'uniform']
    gcn_hidden_channels: [8, 32, 128]
    wandb_offline: false
    l1_lambda: 0.00013885569041041751
    num_samples: 1000
    mlp_channel_lists: [[7261, 1], [7261, 1028, 1], [7261, 128, 64, 1], [7261, 1028, 128, 1], [7261, 1028, 256, 64, 1], [7261, 1028, 512, 128, 1], [7261, 1028, 256, 128, 64, 1]]
    pin_memory: true
    model: 'gat-v4'
    seed: 59012
    use_master_nodes: false
    lr_scheduler_choices: ['LambdaLR', 'ReduceLROnPlateau', 'ExponentialLR', 'StepLR', 'CosineAnnealingLR']
    gat_v4_fc_act: ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu']
    epochs: 1000
    lr: 0.006193218418598052
    master_nodes: ['sex', 'mutation', 'age']
    project: 'proteo'
    batch_size: 8
    sync_batchnorm: false
    mutation_choices: [['GRN', 'MAPT', 'C9orf72', 'CTL']]
    l1_lambda_min: 1e-05
    precision: '32-true'
    weight_decay: 0
    sex: ['M', 'F']
    adj_thresh: 0.35
    y_val: 'nfl'
    y_val_choices: ['nfl']
    output_dir: '/scratch/lcornelis/outputs'
    use_progress_bar: true
    gcn_num_layers: [2, 3, 4]
    ray_results_dir: '/scratch/lcornelis/outputs/ray_results'
    reduction_factor: 6
    trainer_accelerator: 'gpu'
    gat_v4_fc_dim: [[64, 128, 128, 32], [128, 256, 256, 64], [256, 512, 512, 128]]
    act_choices: ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu']
    model_grid_search: ['gat-v4']
    optimizer: 'Adam'
    mlp:
        channel_list: [7261, 32, 64, 128, 1]
        norm: 'batch_norm'
        plain_last: true
    data_dir: '/scratch/lcornelis/data/data_louisa'
    gat_v4_hidden_channels: [[8, 16], [32, 64], [64, 128]]
    gat_heads: [1, 2, 4, 8]
    dropout_choices: [0, 0.05, 0.1, 0.2, 0.3, 0.5]
    num_to_keep: 3
    checkpoint_every_n_epochs_train: 1
    num_nodes_choices: [7258]
    mutation: ['GRN', 'MAPT', 'C9orf72', 'CTL']
    error_protein_file_name: 'bimodal_aptamers_for_removal.xlsx'
    wandb_tmp_dir: '/tmp'
    log_every_n_steps: 10
    wgcna_mergeCutHeight: 0.25
    raw_file_name: 'ALLFTD_dataset_for_nina_louisa_071124_age_adjusted.csv'
    sex_choices: [['M', 'F']]
    modality_choices: ['csf']
    accumulate_grad_batches: 1
    gat_hidden_channels: [8, 32, 128, 256]
    q_gpu: true
    gcn:
        num_layers: 3
        hidden_channels: 32
    gat_v4_heads: [[2, 3], [2, 2], [4, 4]]
    act: 'tanh'
    gat_num_layers: [2, 4, 6, 12]
    grace_period: 30
    lr_max: 0.1
    dataset_name: 'ftd'
    mlp_norms: ['batch_norm', 'layer_norm']
    lr_min: 1e-06
    num_nodes: 7258
    mlp_plain_last: [true, false]
    lr_scheduler: 'CosineAnnealingLR'
    batch_size_choices: [8, 16, 32, 50]
    dropout: 0.1
    ray_tmp_dir: '/scratch/lcornelis/tmp'
    modality: 'csf'
    use_gpu: true
    sex_specific_adj: true
    gat-v4:
        hidden_channels: [32, 64]
        heads: [2, 2]
        use_layer_norm: true
        which_layer: ['layer1', 'layer2', 'layer3', 'sex', 'mutation', 'age']
        fc_dim: [256, 512, 512, 128]
        fc_dropout: 0.1
        fc_act: 'elu'
        weight_initializer: 'xavier'
        num_layers: null
    nodes_count: 1
    gat:
        num_layers: 2
        hidden_channels: 256
        heads: 4
        v2: true
    cpu_per_worker: 1
    adj_thresh_choices: [0.1, 0.2, 0.35, 0.5, 0.7, 0.9]
    root_dir: '/home/lcornelis/code/proteo'
    num_workers: 16
    channel_list: null
    norm: null
    plain_last: null
    """
    config_dict = yaml.safe_load(yaml_string)
    config = config_instance = Config(**config_dict)
    print(config.model)
    '''
    pl.seed_everything(config.seed)
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config.devices))

    train_dataset, test_dataset = construct_datasets(config)
    train_loader, test_loader = construct_loaders(config, train_dataset, test_dataset)
    avg_node_degree = compute_avg_node_degree(test_dataset)
    pos_weight = 1.0  # default value
    focal_loss_weight = [1.0]  # default value
    if config.y_val in BINARY_Y_VALS_MAP:
        pos_weight = compute_pos_weight(test_dataset, train_dataset)
        print(f"pos_weight used for loss function: {pos_weight}")
    elif config.y_val in MULTICLASS_Y_VALS_MAP:
        focal_loss_weight = compute_focal_loss_weight(config, test_dataset, train_dataset)
        print(f"focal_loss_weight used for loss function: {focal_loss_weight}")

    module = Proteo(
        config,
        in_channels=train_dataset.feature_dim,  # 1 dim of input
        out_channels=train_dataset.label_dim,  # 1 dim of result,
        avg_node_degree=avg_node_degree,
        pos_weight=pos_weight,
        focal_loss_weight=focal_loss_weight,
    )

    logger = get_wandb_logger(config)
    images_to_log = [
        os.path.join(
            train_dataset.processed_dir,
            f'{config.y_val}_{config.sex}_{config.mutation}_{config.modality}_histogram.jpg',
        )
    ]
    # Check if sex-specific adjacency is enabled
    if config.sex_specific_adj:
        # Load both male and female adjacency images
        images_to_log.extend(
            [
                os.path.join(
                    train_dataset.processed_dir,
                    f'adjacency_num_nodes_{config.num_nodes}_adjthresh_{config.adj_thresh}_mutation_{config.mutation}_{config.modality}_sex_{config.sex}_masternodes_{config.use_master_nodes}_sex_specific_{config.sex_specific_adj}_M.jpg',
                ),
                os.path.join(
                    train_dataset.processed_dir,
                    f'adjacency_num_nodes_{config.num_nodes}_adjthresh_{config.adj_thresh}_mutation_{config.mutation}_{config.modality}_sex_{config.sex}_masternodes_{config.use_master_nodes}_sex_specific_{config.sex_specific_adj}_F.jpg',
                ),
            ]
        )
    else:
        # Load the single adjacency image if sex-specific adjacency is not enabled
        images_to_log.append(
            os.path.join(
                train_dataset.processed_dir,
                f'adjacency_{config.adj_thresh}_num_nodes_{config.num_nodes}_adjthresh_{config.adj_thresh}_mutation_{config.mutation}_{config.modality}_sex_{config.sex}_masternodes_{config.use_master_nodes}_sex_specific_{config.sex_specific_adj}.jpg',
            )
        )

    if (
        config.y_val in Y_VALS_TO_NORMALIZE
    ):  # add histogram of non normalized data if y is normalized
        images_to_log.append(
            os.path.join(
                train_dataset.processed_dir,
                f'{config.y_val}_{config.sex}_{config.mutation}_{config.modality}_orig_histogram.jpg',
            )
        )

    logger.log_image(
        key="dataset_statistics",
        images=images_to_log,
    )
    # Log top proteins, note this is in order from most to least different
    protein_file_data = read_protein_file(train_dataset.processed_dir, config)
    protein_names = protein_file_data['Protein']
    metrics = protein_file_data['Metric']
    # Create a list of lists for logging
    top_proteins_data = [[protein, metric] for protein, metric in zip(protein_names, metrics)]
    logger.log_text(key="top_proteins", columns=["Protein", "Metric"], data=top_proteins_data)
    logger.log_text(
        key="Parameters",
        columns=["Medium", "Mutation", "Target", "Sex", "Avg Node Degree"],
        data=[
            [
                config.modality,
                config.mutation,
                config.y_val,
                config.sex,
                avg_node_degree,
            ]
        ],
    )

    ckpt_callback = pl_callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=config.checkpoint_dir,
        filename=config.model + '-{epoch}' + '-{val_loss:.4f}',
        mode='min',
        every_n_epochs=config.checkpoint_every_n_epochs_train,
    )
    lr_callback = pl_callbacks.LearningRateMonitor(logging_interval='epoch')
    trainer_callbacks = [
        ckpt_callback,
        lr_callback,
        proteo_callbacks.CustomWandbCallback(),
        proteo_callbacks.progress_bar(),
    ]

    trainer = pl.Trainer(
        enable_progress_bar=config.use_progress_bar,
        accumulate_grad_batches=config.accumulate_grad_batches,
        max_epochs=config.epochs,
        callbacks=trainer_callbacks,
        logger=logger,
        accelerator=config.trainer_accelerator,
        devices=config.devices,
        num_nodes=config.nodes_count,
        strategy=pl_strategies.DDPStrategy(find_unused_parameters=False),
        sync_batchnorm=config.sync_batchnorm,
        log_every_n_steps=config.log_every_n_steps,
        precision=config.precision,
        num_sanity_val_steps=1,
        deterministic=True,
    )

    # Code that only runs on the rank 0 GPU, in multi-GPUs setup
    if trainer.strategy.cluster_environment.global_rank() == 0:
        # Put wandb's experiment id into checkpoints' filenames
        ckpt_callback.filename = (
            f"run-{datetime.now().strftime('%Y%m%d_%H%Mxx')}"
            + f"-{str(logger.experiment.id)}"
            + f"-{ckpt_callback.filename}"
        )
        # Print these only for rank 0 GPU, to avoid cluttering the console
        print(f"Using device(s) {config.devices}")
        print_datasets(train_dataset, test_dataset)
        print_loaders(train_loader, test_loader)
        print(f"Outputs will be saved into:\n {logger.save_dir}")
        print(f"Checkpoints will be saved into:\n {ckpt_callback.dirpath}/{ckpt_callback.filename}")

    trainer.fit(module, train_loader, test_loader)


if __name__ == "__main__":
    main()
