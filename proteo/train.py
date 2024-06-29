import gc
import math
import os
from datetime import datetime

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
import wandb
from config_utils import CONFIG_FILE, Config, read_config_from_file
from models.gat_v4 import GATv4
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import strategies as pl_strategies
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAT, GCN, global_mean_pool

import proteo.rich_gi as rich_gi
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
        "l1_regression": torch.nn.L1Loss,
        "mse_regression": torch.nn.MSELoss,
    }

    def __init__(self, config: Config, in_channels, out_channels, avg_node_degree):
        """Initializes the proteo module by defining self.model according to the model specified in config.yml."""
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        # Save model parameters in a separate config object
        self.model_parameters = Config.parse_obj(getattr(config, config.model))
        self.avg_node_degree = avg_node_degree
        self.train_preds = []
        self.val_preds = []
        self.train_targets = []
        self.val_targets = []

        if config.model == 'gat-v4':
            self.model = GATv4(
                in_channels=in_channels,
                hidden_channels=self.model_parameters.hidden_channels,
                out_channels=out_channels,
                heads=self.model_parameters.heads,
                num_nodes=self.model_parameters.num_nodes,
                which_layer=self.model_parameters.which_layer,
                use_layer_norm=self.model_parameters.use_layer_norm,
                fc_dim=self.model_parameters.fc_dim,
                fc_dropout=self.model_parameters.fc_dropout,
                fc_act=self.model_parameters.fc_act,
            )
        elif config.model == 'gat':
            self.model = GAT(
                in_channels=in_channels,
                hidden_channels=self.model_parameters.hidden_channels,
                out_channels=out_channels,
                num_layers=self.model_parameters.num_layers,
                heads=self.model_parameters.heads,
                v2=self.model_parameters.v2,
                dropout=self.model_parameters.dropout,
                act=self.model_parameters.act,
            )
        elif config.model == 'gcn':
            self.model = GCN(
                in_channels=in_channels,
                hidden_channels=self.model_parameters.hidden_channels,
                out_channels=out_channels,
                num_layers=self.model_parameters.num_layers,
                dropout=self.model_parameters.dropout,
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
            pred = self.model(batch.x, batch.edge_index, batch=batch.batch)
            # Aggregate node features into graph-level features
            return global_mean_pool(pred, batch.batch)
        if self.config.model == 'gcn':
            pred = self.model(batch.x, batch.edge_index, batch=batch.batch)
            return global_mean_pool(pred, batch.batch)
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
        # self.train_preds.append(pred)
        # self.train_targets.append(target)

        # Reduction = "mean" averages over all samples in the batch,
        # providing a single average per batch.
        loss_fn = self.LOSS_MAP[self.config.task_type](reduction="mean")
        loss = loss_fn(pred, target)

        # Calculate L1 regularization term to encourage sparsity
        if self.model_parameters.l1_lambda > 0:
            l1_lambda = self.model_parameters.l1_lambda  # Regularization strength
            l1_norm = sum(p.abs().sum() for p in self.parameters())
            loss = loss + l1_lambda * l1_norm

        self.log(
            'train_loss',
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.config.batch_size,
        )
        self.log('train_RMSE', math.sqrt(loss), on_step=False, on_epoch=True, sync_dist=True)
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
        # self.val_preds.append(pred)
        # self.val_targets.append(target)

        # Reduction = "mean" averages over all samples in the batch,
        # providing a single average per batch.
        loss_fn = self.LOSS_MAP[self.config.task_type](reduction="mean")
        loss = loss_fn(pred, target)

        self.log(
            'val_loss',
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.config.batch_size,
        )
        self.log('val_RMSE', math.sqrt(loss), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    # def on_train_epoch_end(self):
    #     train_preds = torch.vstack(self.train_preds).detach().cpu()
    #     train_targets = torch.vstack(self.train_targets).detach().cpu()
    #     params = torch.concat([p.flatten() for p in self.parameters()]).detach().cpu()
    #     print(f"self.logger.experiment.__dict__: {self.logger.experiment.__dict__}")
    #     self.logger.experiment.log(
    #         {
    #             "train_preds": wandb.Histogram(train_preds),
    #             "train_targets": wandb.Histogram(train_targets),
    #             "parameters": wandb.Histogram(params),
    #         }
    #     )
    #     self.train_preds.clear()  # free memory
    #     self.train_targets.clear()

    # def on_validation_epoch_end(self):
    #     if not self.trainer.sanity_checking:
    #         val_preds = torch.vstack(self.val_preds).detach().cpu()
    #         val_targets = torch.vstack(self.val_targets).detach().cpu()
    #         print(f"self.logger.experiment.__dict__: {self.logger.experiment.__dict__}")
    #         self.logger.experiment.log(
    #             {
    #                 "val_preds": wandb.Histogram(val_preds),
    #                 "val_targets": wandb.Histogram(val_targets),
    #             }
    #         )
    #         self.val_preds.clear()  # free memory
    #         self.val_targets.clear()

    def configure_optimizers(self):
        assert self.model_parameters.optimizer == 'Adam'
        assert self.model_parameters.lr_scheduler in ['LambdaLR', 'ReduceLROnPlateau']

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.model_parameters.lr,
            betas=(0.9, 0.999),
            weight_decay=self.model_parameters.weight_decay,
        )

        if self.model_parameters.lr_scheduler == 'LambdaLR':

            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + 1) / float(self.config.epochs + 1)
                return lr_l

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


def avg_node_degree(dataset):
    # Calculate the average node degree for logging purposes
    num_nodes, _ = dataset.x.shape
    _, num_edges = dataset.edge_index.shape
    num_edges = num_edges / 2
    avg_node_degree = num_edges / num_nodes
    return avg_node_degree


def construct_datasets(config):
    # Load the datasets, which are InMemoryDataset objects
    if config.dataset_name == "ftd":
        root = os.path.join(config.root_dir, config.data_dir, "ftd")
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


def main():
    """Training and evaluation script for experiments."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.set_float32_matmul_precision('high')
    config = read_config_from_file(CONFIG_FILE)
    pl.seed_everything(config.seed)

    output_dir = os.path.join(config.root_dir, config.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # this is where we pick the CUDA device(s) to use
    if isinstance(config.device, list):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config.device))
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.device)

    train_dataset, test_dataset = construct_datasets(config)
    train_loader, test_loader = construct_loaders(config, train_dataset, test_dataset)

    in_channels = train_dataset.feature_dim  # 1 dim of input
    out_channels = train_dataset.label_dim  # 1 dim of result
    avg_node_deg = avg_node_degree(test_dataset)

    wandb_api_key_path = os.path.join(config.root_dir, config.wandb_api_key_path)
    with open(wandb_api_key_path, 'r') as f:
        wandb_api_key = f.read().strip()
    os.environ['WANDB_API_KEY'] = wandb_api_key
    logger = pl_loggers.WandbLogger(
        config=config,
        project=config.project,
        save_dir=output_dir,  # dir needs to exist, otherwise wandb saves in /tmp
        offline=config.wandb_offline,
    )

    logger.log_image(
        key="dataset_statistics",
        images=[
            os.path.join(config.root_dir, config.data_dir, "ftd/processed/histogram.jpg"),
            os.path.join(config.root_dir, config.data_dir, "ftd/processed/adjacency.jpg"),
        ],
    )
    logger.log_text(key="avg_node_deg", columns=["avg_node_deg"], data=[[avg_node_deg]])

    ckpt_callback = pl_callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(config.root_dir, config.checkpoint_dir),
        filename=config.model + '-{epoch}' + '-{val_loss:.2f}',
        mode='min',
    )
    lr_callback = pl_callbacks.LearningRateMonitor(logging_interval='epoch')
    trainer_callbacks = [ckpt_callback, lr_callback, rich_gi.progress_bar()]

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
        print(f"Using device(s) {config.device}")
        print_datasets(train_dataset, test_dataset)
        print_loaders(train_loader, test_loader)
        # Put wandb's experiment id into checkpoints' filenames
        ckpt_callback.filename = (
            "run-"
            + datetime.now().strftime('%Y%m%d_%H%Mxx')
            + "-"
            + str(logger.experiment.id)
            + "-"
            + ckpt_callback.filename
        )
        print(f"Outputs saved into:\n {logger.save_dir}")
        print(f"Checkpoints saved into:\n {ckpt_callback.dirpath}/{ckpt_callback.filename}")

    module = Proteo(config, in_channels, out_channels, avg_node_deg)
    trainer.fit(module, train_loader, test_loader)


if __name__ == "__main__":
    main()
