import os

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
import wandb
from config_utils import CONFIG_FILE, Config, read_config_from_file
from data import complex
from data.data_loading import DataLoader, load_dataset
from models.gat_v4 import GAT
from models.gats import MyGAT
from models.higher import Higher
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import strategies as pl_strategies
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from torch_geometric.nn import GAT
from utils import load_csv_data


class Proteo(pl.LightningModule):
    """Proteo Lightning Module.

    Parameters
    ----------
    config : Config
        The config object.
    in_channels : int
    out_channels : int
    """

    LOSS_MAP = {
        "classification": torch.nn.CrossEntropyLoss(),
        "bin_classification": torch.nn.BCEWithLogitsLoss(),
        "regression": torch.nn.L1Loss(),
        "mse_regression": torch.nn.MSELoss(),
    }

    def __init__(self, config: Config, in_channels, out_channels):
        super().__init__()
        self.config = config
        self.model_name = config.model
        model_parameters = getattr(config, config.model)

        if config.model == 'gat':
            self.model = MyGAT(
                in_channels=in_channels,
                hidden_channels=config.hidden_channels,  # TODO This will break as is.
                out_channels=out_channels,
                heads=config.heads,
            )
        elif config.model == 'higher-gat':
            self.model = Higher(
                max_dim=config.max_dim,
                GNN=GAT,
                in_channels=in_channels,
                hidden_channels=config.hidden_channels,
                num_layers=config[config.model]['num_layers'],
                out_channels=out_channels,
            )
        elif config.model == 'gat-v4':
            self.model = GAT(opt=config[config.model])  # TODO Check if this is correct
        else:
            raise NotImplementedError('Model not implemented yet')

    def forward(self, x):
        if self.config.model == 'gat':
            return self.model(x, dim=self.config.dim)
        elif self.config.model == 'higher-gat':
            return self.model(x)
        elif self.config.model == "gat-v4":
            _, _, _, _, adj = load_csv_data(1, self.config)
            return self.model(x, adj, x.batch, self.config[self.config.model])
        else:
            raise NotImplementedError('Model not implemented yet')

    def training_step(self, batch, batch_idx):
        if self.config.model == 'gat':
            pred = self.model(batch, dim=self.config.dim)
        elif self.config.model == 'higher-gat':
            pred = self.model(batch)
        elif self.config.model == 'gat-v4':
            _, _, _, _, adj = load_csv_data(1, self.config)
            _, _, pred = self.model(
                x, adj, x.batch, self.config[self.config.model]
            )  # TODO define other inputs
        targets = batch.y.view(pred.shape)

        loss_fn = self.LOSS_MAP[self.config.task_type]
        loss = loss_fn(pred, targets)
        self.log(
            'train_loss',
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.config.batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        if self.config.model == 'gat':
            pred = self.model(batch, dim=self.config.dim)
        elif self.config.model == 'higher-gat':
            pred = self.model(batch)
        elif self.config.model == 'gat-v4':
            pred = self.model(batch)  # TODO define other inputs

        targets = batch.y

        loss_fn = self.LOSS_MAP[self.config.task_type]
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

        return loss

    def configure_optimizers(self, config: Config):
        # Do not change this
        self.config = config

        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        mode = 'min' if self.config.minimize else 'max'

        scheduler_type = config[config.model]['lr_scheduler']
        scheduler_params = config[config.model]['lr_scheduler_params']

        if scheduler_type == 'LambdaLR':
            # TO DO: Define num epochs?
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + 1) / float(self.config.epochs + 1)
                return lr_l

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=mode, **scheduler_params
            )

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


def main():
    """Training and evaluation script for experiments."""
    torch.set_float32_matmul_precision('high')

    config = read_config_from_file(CONFIG_FILE)

    device_id = config.device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)

    pl.seed_everything(config.seed)

    dataset = load_dataset(
        config.dataset,
        max_dim=config.max_dim,
        fold=config.fold,
        init_method=config.init_method,
        emb_dim=config.emb_dim,
        flow_points=config.flow_points,
        flow_classes=config.flow_classes,
        max_ring_size=config.max_ring_size,
        use_edge_features=config.use_edge_features,
        include_down_adj=config.include_down_adj,
        simple_features=config.simple_features,
        n_jobs=config.preproc_jobs,
        train_orient=config.train_orient,
        test_orient=config.test_orient,
    )

    in_channels = dataset.num_node_type
    out_channels = dataset.num_classes

    train_features, train_labels, test_features, test_labels, _ = load_csv_data(1, config)

    train_loader = DataLoader(
        # DEBUG: Set to "valid" to go faster through epoch 1
        [train_features, train_labels],  # TODO: transform this into a InMemoryDataset object
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        max_dim=dataset.max_dim,
    )
    test_loader = DataLoader(
        dataset.get_split("test"),  # TODO do the same for the test
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        max_dim=dataset.max_dim,
    )

    # Almost don't touch what's below here
    logger = None
    if config.wandb_api_key_path and config.wandb_offline is False:
        with open(config.wandb_api_key_path, 'r') as f:
            wandb_api_key = f.read().strip()
        os.environ['WANDB_API_KEY'] = wandb_api_key
        wandb_config = {'project': config.project_name}
        logger = pl_loggers.WandbLogger(config=config, **wandb_config)

    custom_theme = RichProgressBarTheme(
        **{
            "description": "deep_sky_blue1",
            "progress_bar": "orange1",
            "progress_bar_finished": "orange3",
            "progress_bar_pulse": "deep_sky_blue3",
            "batch_progress": "deep_sky_blue1",
            "time": "grey82",
            "processing_speed": "grey82",
            "metrics": "grey82",
            "metrics_text_delimiter": "\n",
            "metrics_format": ".3e",
        }
    )

    trainer_callbacks = [
        pl_callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath=config.checkpoint_dir,
            filename=config.checkpoint_name_pattern,
            mode='min',
        ),
        pl_callbacks.RichProgressBar(theme=custom_theme),
    ]

    trainer = pl.Trainer(
        enable_progress_bar=config.use_progress_bar,
        accumulate_grad_batches=config.accumulate_grad_batches,
        max_epochs=config.epochs,
        callbacks=trainer_callbacks,
        logger=logger,
        accelerator=config.trainer_accelerator,
        devices=config.devices_count,
        num_nodes=config.nodes_count,
        strategy=pl_strategies.DDPStrategy(find_unused_parameters=False),
        sync_batchnorm=config.sync_batchnorm,
        log_every_n_steps=config.log_every_n_steps,
        precision=config.precision,
        num_sanity_val_steps=1,
    )

    if config.wandb_api_key_path:
        env = trainer.strategy.cluster_environment
        if env.global_rank() != 0 and env.local_rank() == 0:
            wandb.init(config=config, **wandb_config)

    model = Proteo(config, in_channels, out_channels)
    print(model)
    trainer.fit(model, train_loader, test_loader)


if __name__ == "__main__":
    main()
