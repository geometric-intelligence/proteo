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
from models.readout import Readout
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
from torch_geometric.utils import to_dense_batch

import proteo.callbacks_no_val as proteo_callbacks
from proteo.datasets.ftd_folds import (
    BINARY_Y_VALS_MAP,
    MULTICLASS_Y_VALS_MAP,
    Y_VALS_TO_NORMALIZE,
    FTDDataset,
)


def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.unsqueeze(1).expand_as(loss)
    loss = torch.mean(loss)
    return loss

class ProteoNoVal(pl.LightningModule):
    """Proteo Lightning Module that handles batching the graphs in training.

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
        "mse_regression_weighted": weighted_mse_loss,
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
        self.train_targets = []
        self.train_losses = []
        self.x0 = []
        self.x1 = []
        self.x2 = []
        self.multiscale = []
        self.pos_weight = pos_weight
        self.focal_loss_weight = focal_loss_weight
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
                num_nodes=self.config.num_nodes,
                weight_initializer=self.config_model.weight_initializer,
            )
            if self.config.use_feature_encoder:
                fc_input_dim = self.config.num_nodes * len(self.config.which_layer)
            else:
                fc_input_dim = (self.config.num_nodes * 3) + 3
            self.readout = Readout(feature_output_dim=self.config.num_nodes, which_layer=self.config.which_layer, fc_dim=self.config.fc_dim, fc_dropout=self.config.fc_dropout, fc_act=self.config.fc_act, out_channels=out_channels, fc_input_dim=fc_input_dim)
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
            fc_input_dim = (self.config.num_nodes * 2) -1 #-1 bc modulo division
            self.readout = Readout(feature_output_dim=self.config.num_nodes//3, which_layer=self.config.which_layer, fc_dim=self.config.fc_dim, fc_dropout=self.config.fc_dropout, fc_act=self.config.fc_act, out_channels=out_channels, fc_input_dim=fc_input_dim)
        elif config.model == 'gcn':
            self.model = GCN(
                in_channels=in_channels,
                hidden_channels=self.config_model.hidden_channels,
                out_channels=out_channels,
                num_layers=self.config_model.num_layers,
                dropout=self.config.dropout,
                act=self.config.act,
            )
            fc_input_dim = (self.config.num_nodes * 2) -1 #-1 bc modulo division
            self.readout = Readout(feature_output_dim=self.config.num_nodes//3, which_layer=self.config.which_layer, fc_dim=self.config.fc_dim, fc_dropout=self.config.fc_dropout, fc_act=self.config.fc_act, out_channels=out_channels, fc_input_dim=fc_input_dim)
        elif config.model == "mlp":
            if self.config.use_feature_encoder:
                fc_input_dim = (self.config.num_nodes * 2) -1
            else:
                fc_input_dim = config.num_nodes + 3
            self.readout = Readout(feature_output_dim=self.config.num_nodes//3, which_layer=self.config.which_layer, fc_dim=self.config_model.channel_list, fc_dropout=self.config.dropout, fc_act=self.config.act, out_channels=out_channels, fc_input_dim=fc_input_dim, use_feature_encoder=self.config.use_feature_encoder)
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
            graph_features, aux = self.model(batch.x, batch.edge_index, batch) #[bs, 3*nodes]
            pred = self.readout_class.forward(graph_features, batch) #[bs, 1]
            self.x0.append(aux[0]) 
            self.x1.append(aux[1])
            self.x2.append(aux[2])
            self.multiscale.append(aux[3])
            return pred
        if self.config.model == 'gat':
            graph_features = self.model(batch.x, batch.edge_index, batch=batch.batch) #[bs*num_nodes, hidden_channels] -> [bs*num_nodes, 1]
            graph_features = graph_features.squeeze(-1) #[bs*num_nodes]
            graph_features, _ = to_dense_batch(graph_features, batch=batch.batch) #[bs, num_nodes]
            pred = self.readout.forward(graph_features, batch) # [bs, 1]
            return pred
        if self.config.model == 'gcn':
            graph_features = self.model(batch.x, batch.edge_index, batch=batch.batch) #[bs*num_nodes, 1]
            graph_features = graph_features.squeeze(-1) #[bs*num_nodes]
            graph_features, _ = to_dense_batch(graph_features, batch=batch.batch) #[bs, num_nodes]
            pred = self.readout.forward(graph_features, batch) # [bs, 1]
            return pred
        if self.config.model == 'mlp':
            input = batch.x.squeeze(-1) # [bs*num_nodes, 1]
            input, _ = to_dense_batch(input, batch=batch.batch) # [bs, num_nodes]
            pred = self.readout.forward(input, batch) # [bs, 1]
            return pred
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

        # Reduction = "none" provides a loss per sample in the batch, we avg all in callbacks
        if self.config.y_val in BINARY_Y_VALS_MAP:
            # pos_weight is used to weight the positive class in the loss function
            device = pred.device
            self.pos_weight = self.pos_weight.to(device)
            loss_fn = self.LOSS_MAP["binary_classification"](
                pos_weight=self.pos_weight, reduction="none"
            )
        elif self.config.y_val in MULTICLASS_Y_VALS_MAP:
            device = pred.device
            self.focal_loss_weight = self.focal_loss_weight.to(device)
            loss_fn = self.LOSS_MAP["multiclass_classification"](
                weights=self.focal_loss_weight, gamma=2, reduction="none"
            )
            # Convert targets to ints for the loss function
            target = target.long()
            # Convert to probabilites before taking loss
            pred = torch.nn.Softmax(dim=-1)(pred)
        else:
            loss_fn = self.LOSS_MAP["mse_regression"](reduction="none")

        loss = loss_fn(pred, target)

        # Calculate L1 regularization term to encourage sparsity
        # FIXME: With L1 regularization, the train_RMSE is not the RMSE
        # FIXME: L2 regularization not applied to val_loss, -> train and val losses cannot be compared
        if self.config.l1_lambda > 0:
            l1_lambda = self.config.l1_lambda  # Regularization strength
            l1_norm = sum(p.abs().sum() for p in self.parameters())
            loss = loss + l1_lambda * l1_norm

        self.train_losses.append(loss) 


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

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}


def compute_avg_node_degree(dataset):
    # Calculate the average node degree for logging purposes
    num_nodes, _ = dataset.x.shape
    _, num_edges = dataset.edge_index.shape
    num_edges = num_edges / 2
    avg_node_degree = num_edges / num_nodes
    return avg_node_degree


def compute_pos_weight(train_dataset):
    """Function that computes the positive weight (ratio of ctl to carriers) for the binary cross-entropy loss function."""
    train_y_values = train_dataset.y
    train_carrier_count = torch.sum(train_y_values == 1).item()
    train_ctl_count = torch.sum(train_y_values == 0).item()
    return torch.FloatTensor(
        [(train_ctl_count) / (train_carrier_count)]
    )


def compute_focal_loss_weight(config, train_dataset):
    '''Function that computes the weights (prevalence of) classes in the shape [1, num_classes] to be used in the focal loss function.'''
    train_y_values = train_dataset.y
    frequencies = []
    for key, value in MULTICLASS_Y_VALS_MAP[config.y_val].items():
        count = torch.sum(train_y_values == value).item()
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
    
    return train_dataset


def print_datasets(train_dataset):
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



def construct_loaders(config, train_dataset):
    # Make DataLoader objects to handle batching
    train_loader = DataLoader(  # makes into one big graph
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    return train_loader


def print_loaders(train_loader):
    first_train_batch = next(iter(train_loader))
    print(f"First train batch: {len(first_train_batch)} samples")
    print(f"- first_train_batch.x.shape: {first_train_batch.x.shape}")
    print(f"- first_train_batch.y.shape: {first_train_batch.y.shape}")
    print(f"- first_train_batch.edge_index.shape: {first_train_batch.edge_index.shape}")
    print(f"- first_train_batch.batch.shape: {first_train_batch.batch.shape}")


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


def main():
    """Training and evaluation script for experiments."""
    torch.set_float32_matmul_precision('medium')  # for performance
    config = read_config_from_file(CONFIG_FILE)
    pl.seed_everything(config.seed)
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config.devices))

    train_dataset = construct_datasets(config)
    train_loader = construct_loaders(config, train_dataset)
    avg_node_degree = compute_avg_node_degree(train_dataset)
    pos_weight = 1.0  # default value
    focal_loss_weight = [1.0]  # default value
    if config.y_val in BINARY_Y_VALS_MAP:
        pos_weight = compute_pos_weight(train_dataset)
        print(f"pos_weight used for loss function: {pos_weight}")
    elif config.y_val in MULTICLASS_Y_VALS_MAP:
        focal_loss_weight = compute_focal_loss_weight(config, train_dataset)
        print(f"focal_loss_weight used for loss function: {focal_loss_weight}")

    module = ProteoNoVal(
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
            f'{config.y_val}_{config.sex}_{config.mutation}_{config.modality}_{config.num_folds}fold_{config.fold}_histogram.jpg',
        )
    ]
    # Load the single adjacency image if sex-specific adjacency is not enabled
    images_to_log.append(
        os.path.join(
            train_dataset.processed_dir,
            f'adjacency_{config.adj_thresh}_num_nodes_{config.num_nodes}_adjthresh_{config.adj_thresh}_mutation_{config.mutation}_{config.modality}_sex_{config.sex}_{config.num_folds}fold_{config.fold}.jpg',
        )
    )

    if (
        config.y_val in Y_VALS_TO_NORMALIZE
    ):  # add histogram of non normalized data if y is normalized
        images_to_log.append(
            os.path.join(
                train_dataset.processed_dir,
                f'{config.y_val}_{config.sex}_{config.mutation}_{config.modality}_{config.num_folds}fold_{config.fold}_orig_histogram.jpg',
            )
        )

    logger.log_image(
        key="dataset_statistics",
        images=images_to_log,
    )
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
        monitor='train_loss',
        dirpath=config.checkpoint_dir,
        filename=config.model + '-{epoch}' + '-{train_loss:.4f}',
        mode='min',
        every_n_epochs=config.checkpoint_every_n_epochs_train,
    )
    # Add EarlyStopping callback on train_loss
    early_stop_callback = pl_callbacks.EarlyStopping(
        monitor="train_loss",
        mode="min",
        patience=10,
        verbose=True
    )
    lr_callback = pl_callbacks.LearningRateMonitor(logging_interval='epoch')
    trainer_callbacks = [
        ckpt_callback,
        early_stop_callback,
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
        print_datasets(train_dataset)
        print_loaders(train_loader)
        print(f"Outputs will be saved into:\n {logger.save_dir}")
        print(f"Checkpoints will be saved into:\n {ckpt_callback.dirpath}/{ckpt_callback.filename}")

    trainer.fit(module, train_loader)


if __name__ == "__main__":
    main()
