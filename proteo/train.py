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
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
from config_utils import CONFIG_FILE, Config, read_config_from_file
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
from focal_loss.focal_loss import FocalLoss
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAT, GCN, global_mean_pool

import proteo.callbacks as proteo_callbacks
from proteo.datasets.ftd import FTDDataset


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

    def __init__(self, config: Config, in_channels, out_channels, avg_node_degree, pos_weight, focal_loss_weight):
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
        self.pos_weight = pos_weight
        self.focal_loss_weight = focal_loss_weight

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
            return self.model(batch.x, batch.edge_index, batch)
        if self.config.model == 'gat':
            # This returns a pred value for each node in the big graph
            pred_nodes = self.model(batch.x, batch.edge_index, batch=batch.batch)
            # Aggregate node features into graph-level features
            return global_mean_pool(pred_nodes, batch.batch)
        if self.config.model == 'gcn':
            pred_nodes = self.model(batch.x, batch.edge_index, batch=batch.batch)
            return global_mean_pool(pred_nodes, batch.batch)
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
        target = batch.y.view(pred.shape)
        # Store predictions
        self.train_preds.append(pred.clone().detach().cpu())
        self.train_targets.append(target.clone().detach().cpu())

        # Reduction = "mean" averages over all samples in the batch,
        # providing a single average per batch.
        if self.config.y_val == "carrier_status":
            # pos_weight is used to weight the positive class in the loss function
            device = pred.device
            self.pos_weight = self.pos_weight.to(device)
            loss_fn = self.LOSS_MAP["binary_classification"](
                pos_weight=self.pos_weight, reduction="mean"
            )
        elif self.config.y_val == 'clinical_dementia_rating_global':
            device = pred.device
            self.focal_loss_weight = self.focal_loss_weight.to(device)
            loss_fn = self.LOSS_MAP["multiclass_classification"](
                weights=self.focal_loss_weight, gamma=2, reduction="mean"
            )
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
        # Pred is shape [batch_size,1] and targets is shape [batch_size]
        target = batch.y.view(pred.shape)
        # Store predictions
        self.val_preds.append(pred.clone())
        self.val_targets.append(target.clone())

        # Reduction = "mean" averages over all samples in the batch,
        # providing a single average per batch.
        if self.config.y_val == "carrier_status":
            # pos_weight is used to weight the positive class in the loss function
            device = pred.device
            self.pos_weight = self.pos_weight.to(device)
            loss_fn = self.LOSS_MAP["binary_classification"](
                pos_weight=self.pos_weight, reduction="mean"
            )
        elif self.config.y_val == 'clinical_dementia_rating_global':
            device = pred.device
            self.focal_loss_weight = self.focal_loss_weight.to(device)
            loss_fn = self.LOSS_MAP["multiclass_classification"](
                weights=self.focal_loss_weight, gamma=2, reduction="mean"
            )
            #TODO: is this correct here
            pred = torch.softmax(pred, dim=1)
        else:
            loss_fn = self.LOSS_MAP["mse_regression"](reduction="mean")
        loss = loss_fn(pred, target)
        return loss

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


def compute_pos_weight(config):  # TODO: this doesn't take into account filtering out NA values
    csv_path = os.path.join(config.root_dir, config.data_dir, "raw", config.raw_file_name)
    df = pd.read_csv(csv_path)

    # Count the occurrences of 'CTL' in the 'Mutation' column
    control_count = df['Mutation'].value_counts().get('CTL', 0)

    # Count the occurrences of config.mutation_status in the 'Mutation' column
    mutation_status_count = df['Mutation'].value_counts().get(config.mutation_status, 0)
    return torch.FloatTensor([control_count / mutation_status_count])

def compute_focal_loss_weight(config):
    #TODO - not the best because this code is copied from ftd.py, but weight needs to be recomputed for each set
    # Get cdr global values for specific subset
    csv_path = os.path.join(config.root_dir, config.data_dir, "raw", config.raw_file_name)
    df = pd.read_csv(csv_path)
    if config.plasma_or_csf == 'plasma':
        has_measurement_col = 'HasPlasma?'
    elif config.plasma_or_csf == 'csf':
        has_measurement_col = 'HasCSF?'
    # Get the indices of the rows where has_measurement is True
    has_measurement = df[has_measurement_col].astype(int) == 1

    # Additional filtering based on mutation_status, always take mutation status and control
    if config.mutation_status in ['GRN', 'MAPT', 'C9orf72']:
        mutation_filter = df['Mutation'].isin(
            [config.mutation_status, 'CTL']
        )
    elif config.mutation_status == 'CTL':
        mutation_filter = pd.Series([True] * len(df))
    # Additional filtering based on sex
    if config.sex in ['M', 'F']:
        sex_filter = df['SEX_AT_BIRTH'] == config.sex
    elif config.sex == 'All':
        sex_filter = pd.Series([True] * len(df))
    combined_filter = has_measurement & mutation_filter & sex_filter
    cdrglob_values = df.loc[combined_filter, 'CDRGLOB'].astype(float)
    # Compute the weight for the focal loss
    specific_values = [0, 0.5, 1, 2, 3]
    frequencies = []
    for value in specific_values:
        count = (cdrglob_values == value).sum()
        frequencies.append(count)
    # Calculate weights inversely proportional to the frequencies
    frequencies = torch.tensor(frequencies, dtype=torch.float32)
    weights = 1.0 / frequencies 
    return weights



def construct_datasets(config):
    # Load the datasets, which are InMemoryDataset objects
    root = os.path.join(config.root_dir, config.data_dir)
    train_dataset = FTDDataset(root, "train", config)
    test_dataset = FTDDataset(root, "test", config)
    return train_dataset, test_dataset


def print_datasets(train_dataset, test_dataset):
    print(f"Train: {len(train_dataset)} samples")
    print(f"- train_dataset.x.shape: {train_dataset.x.shape}")
    print(f"- train_dataset.y.shape: {train_dataset.y.shape}")
    print(f"- train_dataset.edge_index.shape: {train_dataset.edge_index.shape}")
    print(f"---- train_dataset[0].x.shape: {train_dataset[0].x.shape}")
    print(f"---- train_dataset[0].edge_index.shape: {train_dataset[0].edge_index.shape}")
    print(f"Test: {len(test_dataset)} samples")
    print(f"- test_dataset.x.shape: {test_dataset.x.shape}")
    print(f"- test_dataset.y.shape: {test_dataset.y.shape}")
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


def get_wandb_logger(config):
    """Get Weights and Biases logger."""
    wandb_api_key_path = os.path.join(config.root_dir, config.wandb_api_key_path)
    with open(wandb_api_key_path, 'r') as f:
        wandb_api_key = f.read().strip()
    os.environ['WANDB_API_KEY'] = wandb_api_key
    output_dir = os.path.join(config.root_dir, config.output_dir)
    return pl_loggers.WandbLogger(
        config=config,
        project=config.project,
        save_dir=output_dir,  # dir needs to exist, otherwise wandb saves in /tmp
        offline=config.wandb_offline,
    )


def read_protein_file(processed_dir, config):
    file_path = os.path.join(
        processed_dir,
        f'top_proteins_num_nodes_{config.num_nodes}_mutation_status_{config.mutation_status}_{config.plasma_or_csf}.npy',
    )
    if os.path.exists(file_path):
        return np.load(file_path, allow_pickle=True)
    else:
        raise FileNotFoundError(f"The file {file_path} does not exist.")


def main():
    """Training and evaluation script for experiments."""
    torch.set_float32_matmul_precision('medium')  # for performance
    config = read_config_from_file(CONFIG_FILE)
    pl.seed_everything(config.seed)

    output_dir = os.path.join(config.root_dir, config.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config.device))

    train_dataset, test_dataset = construct_datasets(config)
    train_loader, test_loader = construct_loaders(config, train_dataset, test_dataset)
    avg_node_degree = compute_avg_node_degree(test_dataset)
    pos_weight = compute_pos_weight(config)
    focal_loss_weight = compute_focal_loss_weight(config)
    if config.y_val == "carrier_status":
        print(f"pos_weight used for loss function: {pos_weight}")
    elif config.y_val == 'clinical_dementia_rating_global':
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

    logger.log_image(
        key="dataset_statistics",
        images=[
            os.path.join(
                train_dataset.processed_dir,
                f'{config.y_val}_{config.sex}_{config.mutation_status}_histogram.jpg',
            ),
            os.path.join(
                train_dataset.processed_dir,
                f"adjacency_{config.adj_thresh}_num_nodes_{config.num_nodes}_mutation_status_{config.mutation_status}_{config.plasma_or_csf}_sex_{config.sex}.jpg",
            ),
        ],
    )
    # Log top proteins, note this is in order from most to least different
    plasma_protein_names = read_protein_file(train_dataset.processed_dir, config)
    # Create a list of lists for logging
    top_proteins_data = [[protein] for protein in plasma_protein_names]
    logger.log_text(key="top_proteins", columns=["Protein"], data=top_proteins_data)
    logger.log_text(
        key="Parameters",
        columns=["Medium", "Mutation", "Target", "Sex", "Avg Node Degree"],
        data=[
            [
                config.plasma_or_csf,
                config.mutation_status,
                config.y_val,
                config.sex,
                avg_node_degree,
            ]
        ],
    )

    ckpt_callback = pl_callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(config.root_dir, config.checkpoint_dir),
        filename=config.model + '-{epoch}' + '-{val_loss:.4f}',
        mode='min',
        every_n_epochs=config.checkpoint_every_n_epochs,
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
        devices=len(config.device),
        num_nodes=config.nodes_count,
        strategy=pl_strategies.DDPStrategy(find_unused_parameters=False),
        sync_batchnorm=config.sync_batchnorm,
        log_every_n_steps=config.log_every_n_steps,
        precision=config.precision,
        num_sanity_val_steps=1,
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
        print(f"Using device(s) {config.device}")
        print_datasets(train_dataset, test_dataset)
        print_loaders(train_loader, test_loader)
        print(f"Outputs will be saved into:\n {logger.save_dir}")
        print(f"Checkpoints will be saved into:\n {ckpt_callback.dirpath}/{ckpt_callback.filename}")

    trainer.fit(module, train_loader, test_loader)


if __name__ == "__main__":
    main()
