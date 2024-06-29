import math
import os
import gc
from datetime import date

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import rich_gi
import torch
import wandb
from config_utils import CONFIG_FILE, Config, read_config_from_file
from models.gat_v4 import GATv4
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import strategies as pl_strategies
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAT, global_mean_pool
import matplotlib.pyplot as plt

from proteo.datasets.ftd import FTDDataset


class AttrDict(dict):
    """Convert a dict into an object where attributes are accessed with "."

    This is needed for the utils.load() function.
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


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
        "classification": torch.nn.CrossEntropyLoss(),
        "bin_classification": torch.nn.BCEWithLogitsLoss(),
        "regression": torch.nn.L1Loss(),
        "mse_regression": torch.nn.MSELoss(),
    }

    def __init__(self, config: Config, in_channels, out_channels, avg_node_degree):
        """Initializes the proteo module by defining self.model according to the model specified in config.yml."""
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        # Save the model specific parameters in a separate object
        self.model_parameters = getattr(config, config.model)
        self.model_parameters = AttrDict(self.model_parameters)
        self.avg_node_degree = avg_node_degree
        self.train_predictions = []
        self.val_predictions = []

        if config.model == 'gat':
            self.model = GAT(
                in_channels=in_channels,
                hidden_channels=self.model_parameters.hidden_channels,
                num_layers=self.model_parameters.num_layers,
                out_channels=out_channels,
                heads=self.model_parameters.heads,
                v2=self.model_parameters.v2,
            )
        elif config.model == 'gat-v4':
            self.model = GATv4(
                opt=self.model_parameters, in_channels=in_channels, out_channels=out_channels
            )
        else:
            raise NotImplementedError('Model not implemented yet')
        

    def forward(self, batch):
        """Performs one forward pass of the model on the input batch.
        Parameters
        ----------
        batch: DataBatch object with attributes x, edge_index, y, and batch.
        batch.x: torch.Tensor of shape [num_nodes * batch_size, num_features]
        batch.edge_index: torch.Tensor of shape [2, num_edges * batch_size]
        batch.y: torch.Tensor of shape [batch_size]
        batch.batch: torch.Tensor of shape [num_nodes * batch_size]
        """
        if self.config.model == 'gat':
            # This returns a pred value for each node in the big graph
            pred = self.model(batch.x, batch.edge_index, batch=batch.batch)
            # Aggregate node features into graph-level features
            return global_mean_pool(pred, batch.batch)
        if self.config.model == "gat-v4":
            return self.model(batch.x, batch.edge_index, batch)
        raise NotImplementedError('Model not implemented yet')

    def training_step(self, batch):
        """Defines a single step in the training loop. It specifies how a batch of data is processed during training, including making predictions, calculating the loss, and logging metrics.
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
        targets = batch.y.view(pred.shape)

        # Reduction = "mean" averages over all samples in the batch, providing a single average per batch.
        loss_fn = torch.nn.MSELoss(reduction="mean")
        loss = loss_fn(pred, targets)

        # Calculate L1 regularization term to encourage sparsity
        if self.model_parameters.l1_lambda > 0:
            l1_lambda = self.model_parameters.l1_lambda  # Regularization strength
            l1_norm = sum(p.abs().sum() for p in self.parameters())
            loss = loss + l1_lambda * l1_norm

        self.log(
            'train_loss',
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.config.batch_size,
        )
        self.log('train_RMSE', math.sqrt(loss), on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch):
        """Defines a single step in the validation loop. It specifies how a batch of data is processed during validation, including making predictions, calculating the loss, and logging metrics.
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
        targets = batch.y.view(pred.shape)
        # Store predictions
        self.val_predictions.append(targets.detach().cpu().numpy())

        # Reduction = "mean" averages over all samples in the batch, providing a single average per batch.
        loss_fn = torch.nn.MSELoss(reduction="mean")
        loss = loss_fn(pred, targets)

        self.log(
            'val_loss',
            loss,
            batch_size=self.config.batch_size,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        self.log('val_RMSE', math.sqrt(loss), on_step=True, on_epoch=True, sync_dist=True)

        # Log the current learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = None
        if self.model_parameters.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.model_parameters.lr,
                betas=(0.9, 0.999),
                weight_decay=self.model_parameters.weight_decay,
            )
        else:
            raise NotImplementedError('Optimizer not implemented yet')
        mode = self.config.mode

        scheduler_type = self.model_parameters.lr_scheduler
        scheduler_params = self.model_parameters.lr_scheduler_params

        if scheduler_type == 'LambdaLR':

            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + 1) / float(self.config.epochs + 1)
                return lr_l

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=mode, **scheduler_params
            )

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
        root = os.path.join(config.root_dir, "data", "ftd")
        test_dataset = FTDDataset(root, "test", config)
        train_dataset = FTDDataset(root, "train", config)
    return train_dataset, test_dataset


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


def main():
    """Training and evaluation script for experiments."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.set_float32_matmul_precision('high')

    config = read_config_from_file(CONFIG_FILE)

    # this is where we pick the CUDA device(s) to use
    if isinstance(config.device, list):
        print(f"Using devices {config.device}")
        device_count = len(config.device)
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config.device))
    else:
        print(f"Using device {config.device}")
        device_count = 1
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.device)

    pl.seed_everything(config.seed)

    train_dataset, test_dataset = construct_datasets(config)
    train_loader, test_loader = construct_loaders(config, train_dataset, test_dataset)

    in_channels = train_dataset.feature_dim  # 1 dim of input
    out_channels = train_dataset.label_dim  # 1 dim of result

    avg_node_deg = avg_node_degree(test_dataset)

    print("Loaders created")

    # Configure WandB Logger
    # logger = None
    # wandb_api_key_path = os.path.join(config.root_dir, config.wandb_api_key_path)
    # if wandb_api_key_path and not config.wandb_offline:
    #     with open(config.wandb_api_key_path, 'r') as f:
    #         wandb_api_key = f.read().strip()
    #     os.environ['WANDB_API_KEY'] = wandb_api_key
    #wandb_config = {'project': config.project_name}
    logger = pl_loggers.WandbLogger(config=config, project=config.project)

    logger.log_image(
        key="output_hist",
        images=[
            os.path.join(config.root_dir, "data/ftd/processed/histogram.jpg"),
            os.path.join(config.root_dir, "data/ftd/processed/adjacency.jpg"),
        ],
    )

    trainer_callbacks = [
        pl_callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath=config.checkpoint_dir,
            filename=config.checkpoint_name_pattern
            + "-"
            + config.model
            + "-"
            + date.today().strftime('%d-%m-%Y-%h-%M')
            + '{epoch}',
            mode='min',
        ),
        rich_gi.progress_bar(),
    ]

    trainer = pl.Trainer(
        enable_progress_bar=config.use_progress_bar,
        accumulate_grad_batches=config.accumulate_grad_batches,
        max_epochs=config.epochs,
        callbacks=trainer_callbacks,
        logger=logger,
        accelerator=config.trainer_accelerator,
        devices=device_count,
        num_nodes=config.nodes_count,
        strategy=pl_strategies.DDPStrategy(find_unused_parameters=False),
        sync_batchnorm=config.sync_batchnorm,
        log_every_n_steps=config.log_every_n_steps,  # Controls the frequency of logging within training, by specifying how many training steps should occur between each logging event.
        precision=config.precision,
        num_sanity_val_steps=1,
    )

    if config.wandb_api_key_path:
        env = trainer.strategy.cluster_environment
        if env.global_rank() != 0 and env.local_rank() == 0:
            wandb.init(config=config, project=config.project)
            wandb.log(
                {
                    "nfl_hist": wandb.Image(
                        os.path.join(config.root_dir, "datasets/data/ftd/processed/histogram.svg")
                    )
                }
            )

    module = Proteo(config, in_channels, out_channels, avg_node_deg)
    trainer.fit(module, train_loader, test_loader)
    # Plot the histogram for validation predictions
    plt.hist(module.val_predictions, bins=30, edgecolor='k')
    plt.title('Histogram of Validation Predicted Values')
    plt.xlabel('Predicted Value')
    plt.ylabel('Frequency')
    plt.show()


if __name__ == "__main__":
    main()
