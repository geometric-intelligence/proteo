import os

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
import wandb
from config_utils import CONFIG_FILE, Config, read_config_from_file
from models.gat_v4 import GATv4
from models.gats import MyGAT
from models.higher import Higher
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import strategies as pl_strategies
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAT

from proteo.datasets.ftd import ROOT_DIR, FTDDataset
from proteo.datasets.mlagnn import ROOT_DIR, MLAGNNDataset


class AttrDict(dict):
    """Convert a dict into an object where attributes are accessed with "."

    This is needed for the utils.load() function.
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


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

    def __init__(self, config: Config, in_channels, out_channels, test_loader):
        super().__init__()
        self.config = config
        self.model_parameters = getattr(config, config.model)
        self.model_parameters = AttrDict(self.model_parameters)
        # These parameters are to get the test() function to run
        self.test_loader = test_loader

        if config.model == 'gat':
            self.model = MyGAT(
                in_channels=in_channels,
                hidden_channels=self.model_parameters.hidden_channels,
                out_channels=out_channels,
                heads=self.model_parameters.heads,
            )
        elif config.model == 'higher-gat':
            self.model = Higher(
                max_dim=config.max_dim,
                GNN=GAT,
                in_channels=in_channels,
                hidden_channels=self.model_parameters.hidden_channels,
                num_layers=self.model_parameters.num_layers,
                out_channels=out_channels,
            )
        elif config.model == 'gat-v4':
            model = GATv4(
                opt=self.model_parameters, in_channels=in_channels, out_channels=out_channels
            )
            self.model = model
        else:
            raise NotImplementedError('Model not implemented yet')

    def forward(self, x):
        if self.config.model == 'gat':
            return self.model(x, dim=self.config.dim)
        elif self.config.model == 'higher-gat':
            return self.model(x)
        elif self.config.model == "gat-v4":
            return self.model(
                x, x.batch, self.model_parameters
            )  # TO DO: understand why 3 inputs here but 2 in gatv4
        else:
            raise NotImplementedError('Model not implemented yet')

    def training_step(self, batch):
        if self.config.model == 'gat':
            pred = self.model(batch, dim=self.config.dim)
        elif self.config.model == 'higher-gat':
            pred = self.model(batch)
        elif self.config.model == 'gat-v4':
            _, _, pred = self.model(batch, self.model_parameters)

        targets = batch.y.view(pred.shape)

        loss_fn = torch.nn.MSELoss()   #self.LOSS_MAP[self.config.task_type]
        # HACK ALERT: only training on survival even though we predict censor and survival
        print(pred[0:10])
        print(targets[0:10])
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

    def validation_step(self, batch):
        if self.config.model == 'gat':
            pred = self.model(batch, dim=self.config.dim)
        elif self.config.model == 'higher-gat':
            pred = self.model(batch)
        elif self.config.model == 'gat-v4':
            _, _, pred = self.model(batch, self.model_parameters)
        targets = batch.y.view(pred.shape)

        loss_fn = torch.nn.MSELoss()  # self.LOSS_MAP[self.config.task_type]
        # HACK ALERT: only training on survival even though we predict censor and survival
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

    def configure_optimizers(self):
        # Do not change this

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.model_parameters.lr,
            betas=(0.9, 0.999),
            weight_decay=self.model_parameters.weight_decay,
        )
        mode = 'min' if self.config.minimize else 'max'  # Could set this in the config file

        scheduler_type = self.model_parameters.lr_scheduler
        scheduler_params = self.model_parameters.lr_scheduler_params

        if scheduler_type == 'LambdaLR':
            # TODO: Define num epochs?
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

    # this is where we pick the CUDA device(s) to use
    if isinstance(config.device, list):
        device_count = len(config.device)
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config.device))
    else:
        print(f"Using device {config.device}")
        device_count = 1
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.device)

    pl.seed_everything(config.seed)

    # The first time you call this it creates train.pt and test.pt files, and afterwards it loads them

    if config.dataset_name == "mlagnn":
        root = os.path.join(ROOT_DIR, "data", "FAD")
        test_dataset = MLAGNNDataset(root, "test", config)
        train_dataset = MLAGNNDataset(root, "train", config)

    elif config.dataset_name == "ftd":
        root = os.path.join(ROOT_DIR, "data", "ftd")
        test_dataset = FTDDataset(root, "test", config)
        train_dataset = FTDDataset(root, "train", config)

    in_channels = train_dataset.feature_dim  # 1 dim of input

    out_channels = train_dataset.label_dim  # 1 dim of result

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
        devices=device_count,
        num_nodes=config.nodes_count,
        strategy=pl_strategies.DDPStrategy(find_unused_parameters=True),  # Debug later
        sync_batchnorm=config.sync_batchnorm,
        log_every_n_steps=config.log_every_n_steps,
        precision=config.precision,
        num_sanity_val_steps=1,
    )

    if config.wandb_api_key_path:
        env = trainer.strategy.cluster_environment
        if env.global_rank() != 0 and env.local_rank() == 0:
            wandb.init(config=config, **wandb_config)

    module = Proteo(config, in_channels, out_channels, test_loader)
    print(module)
    trainer.fit(module, train_loader, test_loader)


if __name__ == "__main__":
    main()
