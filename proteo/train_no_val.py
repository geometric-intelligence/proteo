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

class Proteo(pl.LightningModule):
    """
    Proteo Lightning Module using only the training set.
    """

    LOSS_MAP = {
        "binary_classification": torch.nn.BCEWithLogitsLoss,  # binary cross-entropy with logits
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
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.config_model = Config.parse_obj(getattr(config, config.model))
        self.avg_node_degree = avg_node_degree
        self.train_preds = []
        self.train_targets = []
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
        elif config.model == "mlp":
            dropout = [self.config.dropout] * (len(self.config_model.channel_list) - 1)
            self.model = MLP(
                channel_list=self.config_model.channel_list,
                dropout=dropout,
                act=self.config.act,
                norm=self.config_model.norm,
                plain_last=self.config_model.plain_last
            )
        else:
            raise NotImplementedError('Model not implemented yet')

    def forward(self, batch):
        if self.config.model == "gat-v4":
            pred, aux = self.model(batch.x, batch.edge_index, batch)
            return pred
        if self.config.model in ['gat', 'gcn']:
            pred_nodes = self.model(batch.x, batch.edge_index, batch=batch.batch)
            return global_mean_pool(pred_nodes, batch.batch)
        if self.config.model == 'mlp':
            input_dim = self.config.num_nodes + len(self.config.master_nodes) if self.config.use_master_nodes else self.config.num_nodes
            batch_size = batch.x.shape[0] // input_dim
            batch.x = batch.x.view(batch_size, input_dim)
            return self.model(batch.x)
        raise NotImplementedError('Model not implemented yet')

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch)
        if torch.isnan(pred).any():
            print("train: pred has nan")
            print(f"?batch.x has nan: {torch.isnan(batch.x).any()}")
            raise ValueError
        target = batch.y.view(pred.shape) if self.config.y_val != 'clinical_dementia_rating_global' else batch.y
        self.train_preds.append(pred.clone().detach().cpu())
        self.train_targets.append(target.clone().detach().cpu())

        if self.config.y_val in BINARY_Y_VALS_MAP:
            device = pred.device
            self.pos_weight = self.pos_weight.to(device)
            loss_fn = self.LOSS_MAP["binary_classification"](pos_weight=self.pos_weight, reduction="mean")
        elif self.config.y_val in MULTICLASS_Y_VALS_MAP:
            device = pred.device
            self.focal_loss_weight = self.focal_loss_weight.to(device)
            loss_fn = self.LOSS_MAP["multiclass_classification"](weights=self.focal_loss_weight, gamma=2, reduction="mean")
            target = target.long()
            pred = torch.nn.Softmax(dim=-1)(pred)
        else:
            loss_fn = self.LOSS_MAP["mse_regression_weighted"]

        loss = loss_fn(pred, target)

        if self.config.l1_lambda > 0:
            l1_lambda = self.config.l1_lambda
            l1_norm = sum(p.abs().sum() for p in self.parameters())
            loss = loss + l1_lambda * l1_norm

        # Log training loss so that callbacks can monitor it
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    # Removed the validation_step since we are not using a validation set

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.config.lr,
            betas=(0.9, 0.999),
            weight_decay=self.config.weight_decay,
        )

        if self.config.lr_scheduler == 'LambdaLR':
            def lambda_rule(epoch):
                return 1.0 - max(0, epoch + 1) / float(self.config.epochs + 1)
            scheduler = LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif self.config.lr_scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2, threshold=0.01, patience=5)
        elif self.config.lr_scheduler == 'ExponentialLR':
            scheduler = ExponentialLR(optimizer, 0.1, last_epoch=-1)
        elif self.config.lr_scheduler == 'StepLR':
            scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        elif self.config.lr_scheduler == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs, eta_min=0)
        else:
            raise NotImplementedError('scheduler not implemented:', self.config.lr_scheduler)

        # Now monitor "train_loss" instead of "val_loss"
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

def compute_avg_node_degree(dataset):
    num_nodes, _ = dataset.x.shape
    _, num_edges = dataset.edge_index.shape
    num_edges = num_edges / 2
    return num_edges / num_nodes

def construct_dataset(config):
    root = config.data_dir
    print(f"Loading train dataset from {root}")
    try:
        train_dataset = FTDDataset(root, "train", config)
        print("Train dataset loaded successfully")
    except Exception as e:
        print(f"Error loading train dataset: {str(e)}")
        raise
    return train_dataset

def print_dataset(train_dataset):
    print(f"Train: {len(train_dataset)} samples")
    print(f"- train_dataset.x.shape: {train_dataset.x.shape}")
    print(f"- train_dataset.y.shape: {train_dataset.y.shape}")
    print(f"- train_dataset.sex.shape: {train_dataset.sex.shape}")
    print(f"- train_dataset.mutation.shape: {train_dataset.mutation.shape}")
    print(f"- train_dataset.age.shape: {train_dataset.age.shape}")
    print(f"- train_dataset.edge_index.shape: {train_dataset.edge_index.shape}")
    print(f"---- train_dataset[0].x.shape: {train_dataset[0].x.shape}")
    print(f"---- train_dataset[0].edge_index.shape: {train_dataset[0].edge_index.shape}")

def construct_loader(config, train_dataset):
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    return train_loader

def get_wandb_logger(config):
    wandb_api_key_path = os.path.join(config.root_dir, config.wandb_api_key_path)
    with open(wandb_api_key_path, 'r') as f:
        wandb_api_key = f.read().strip()
    os.environ['WANDB_API_KEY'] = wandb_api_key
    return pl_loggers.WandbLogger(
        config=config,
        project=config.project,
        save_dir=config.wandb_tmp_dir,
        offline=config.wandb_offline,
    )

def main():
    torch.set_float32_matmul_precision('medium')
    config = read_config_from_file(CONFIG_FILE)
    pl.seed_everything(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    train_dataset = construct_dataset(config)
    train_loader = construct_loader(config, train_dataset)
    avg_node_degree = compute_avg_node_degree(train_dataset)
    pos_weight = torch.FloatTensor([1.0])  # Modify as needed (e.g., computed from the train set)
    focal_loss_weight = torch.tensor([1.0])  # Modify as needed

    if config.y_val in BINARY_Y_VALS_MAP:
        print(f"pos_weight used for loss function: {pos_weight}")
    elif config.y_val in MULTICLASS_Y_VALS_MAP:
        print(f"focal_loss_weight used for loss function: {focal_loss_weight}")

    module = Proteo(
        config,
        in_channels=train_dataset.feature_dim,
        out_channels=train_dataset.label_dim,
        avg_node_degree=avg_node_degree,
        pos_weight=pos_weight,
        focal_loss_weight=focal_loss_weight,
    )

    logger = get_wandb_logger(config)
    images_to_log = [
        os.path.join(
            train_dataset.processed_dir,
            f'{config.y_val}_{config.sex}_{config.mutation}_{config.modality}_{config.num_folds}fold_{config.fold}_histogram.jpg',
        ),
        os.path.join(
            train_dataset.processed_dir,
            f'adjacency_{config.adj_thresh}_num_nodes_{config.num_nodes}_adjthresh_{config.adj_thresh}_mutation_{config.mutation}_{config.modality}_sex_{config.sex}_{config.num_folds}fold_{config.fold}.jpg',
        )
    ]
    if config.y_val in Y_VALS_TO_NORMALIZE:
        images_to_log.append(
            os.path.join(
                train_dataset.processed_dir,
                f'{config.y_val}_{config.sex}_{config.mutation}_{config.modality}_{config.num_folds}fold_{config.fold}_orig_histogram.jpg',
            )
        )
    logger.log_image(key="dataset_statistics", images=images_to_log)
    logger.log_text(
        key="Parameters",
        columns=["Medium", "Mutation", "Target", "Sex", "Avg Node Degree"],
        data=[[config.modality, config.mutation, config.y_val, config.sex, avg_node_degree]],
    )

    # Update the checkpoint callback to monitor training loss
    ckpt_callback = pl_callbacks.ModelCheckpoint(
        monitor='train_loss',
        dirpath=config.checkpoint_dir,
        filename=config.model + '-{epoch}' + '-{train_loss:.4f}',
        mode='min',
        every_n_epochs=config.checkpoint_every_n_epochs_train,
    )
    lr_callback = pl_callbacks.LearningRateMonitor(logging_interval='epoch')
    early_stop_callback = pl_callbacks.EarlyStopping(
        monitor='train_loss',
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode='min'
    )
    trainer_callbacks = [
        ckpt_callback,
        lr_callback,
        early_stop_callback,
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
        num_sanity_val_steps=0,  # No validation, so set to 0
        deterministic=True,
    )

    if trainer.strategy.cluster_environment.global_rank() == 0:
        ckpt_callback.filename = (
            f"run-{datetime.now().strftime('%Y%m%d_%H%Mxx')}"
            + f"-{str(logger.experiment.id)}"
            + f"-{ckpt_callback.filename}"
        )
        print(f"Using device(s) {config.devices}")
        print_dataset(train_dataset)
        print(f"Checkpoints will be saved into:\n {ckpt_callback.dirpath}/{ckpt_callback.filename}")

    # Train using only the training loader
    trainer.fit(module, train_loader)

if __name__ == "__main__":
    main()