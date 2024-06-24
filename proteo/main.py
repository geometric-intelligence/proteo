import math
import os
from datetime import date

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
import wandb
from config_utils import CONFIG_FILE, read_config_from_file, Config
from models.gat_v4 import GATv4
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import strategies as pl_strategies
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAT, global_mean_pool
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train import lightning as ray_lightning
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler


from proteo.datasets.ftd import ROOT_DIR, FTDDataset

MAX_SEED = 65535

class AttrDict(dict):
    """Convert a dict into an object where attributes are accessed with "."

    This is needed for the utils.load() function.
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Proteo(pl.LightningModule):
    """Proteo Lightning Module that handles batching the graphs in traning and validation.

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

        if config.model == 'gat':
            self.model = GAT(
                in_channels=in_channels,
                hidden_channels=self.model_parameters.hidden_channels,
                num_layers=self.model_parameters.num_layers,
                out_channels=out_channels,
                heads=self.model_parameters.heads,
            )
        elif config.model == 'gat-v4':
            model = GATv4(
                opt=self.model_parameters, in_channels=in_channels, out_channels=out_channels
            )
            self.model = model
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
            pred = self.model(
                batch.x, batch.edge_index, batch=batch.batch
            )  # This returns a pred value for each node in the big graph
            return global_mean_pool(
                pred, batch.batch
            )  # Aggregate node features into graph-level features
        elif self.config.model == "gat-v4":
            return self.model(batch.x, batch.edge_index, batch)
        else:
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
        targets = batch.y.view(pred.shape)

        loss_fn = torch.nn.MSELoss(
            reduction="mean"
        )  # reduction = "mean" averages over all samples in the batch, providing a single average per batch.
        loss = loss_fn(pred, targets)

        # Calculate L1 regularization term to encourage sparsity
        if self.model_parameters.l1_lambda > 0:
            l1_lambda = self.model_parameters.l1_lambda  # Regularization strength
            l1_norm = sum(p.abs().sum() for p in self.parameters())
            loss = loss + l1_lambda * l1_norm

        # --- LOGGING ---
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
        loss_fn = torch.nn.MSELoss(
            reduction="mean"
        )  # self.LOSS_MAP[self.config.task_type], reduction = "mean" averages over all samples in the batch, providing a single average per batch.
        loss = loss_fn(pred, targets)
        # --- LOGGING ---
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



def main(search_config):
    """Training and evaluation script for one set of hyperparameters.
    
    Parameters
    ----------
    search_config : dict
        Dictionary containing the hyperparameters currently tested.
        For us: keys are model, hidden_channels, heads, seed.
    """
    torch.set_float32_matmul_precision('high')
    config = read_config_from_file(CONFIG_FILE)

    config.update(search_config)

    # # this is where we pick the CUDA device(s) to use
    # if isinstance(config.device, list):
    #     print(f"Using devices {config.device}")
    #     device_count = len(config.device)
    #     os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config.device))
    # else:
    #     print(f"Using device {config.device}")
    #     device_count = 1
    #     os.environ['CUDA_VISIBLE_DEVICES'] = str(config.device)

    pl.seed_everything(config.seed)

    # Load the datasets, which are InMemoryDataset objects
    if config.dataset_name == "ftd":
        root = os.path.join(ROOT_DIR, "data", "ftd")
        test_dataset = FTDDataset(root, "test", config)
        train_dataset = FTDDataset(root, "train", config)

    in_channels = train_dataset.feature_dim  # 1 dim of input
    out_channels = train_dataset.label_dim  # 1 dim of result

    # Calculate the average node degree for logging purposes
    num_nodes, _ = test_dataset.x.shape
    _, num_edges = test_dataset.edge_index.shape
    num_edges = num_edges / 2
    avg_node_degree = num_edges / num_nodes

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
    print("Loaders created")

    #Configure WandB Logger
    logger = None
    if config.wandb_api_key_path and config.wandb_offline is False:
        with open(config.wandb_api_key_path, 'r') as f:
            wandb_api_key = f.read().strip()
        os.environ['WANDB_API_KEY'] = wandb_api_key
        wandb_config = {'project': config.project_name}
        logger = pl_loggers.WandbLogger(config=config, **wandb_config)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    logger.log_image(
        key="output_hist",
        images=[
            os.path.join(dir_path, "datasets/data/ftd/processed/histogram.jpg"),
            os.path.join(dir_path, "datasets/data/ftd/processed/adjacency.jpg"),
        ],
    )

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

    # Things that happen regularly during training
    trainer_callbacks = [
        EarlyStopping(monitor='val_loss', mode = "min", patience=3),
        pl_callbacks.ModelCheckpoint( #TODO: Where is this going?
            monitor='val_loss',
            dirpath=config.checkpoint_dir,
            filename=config.checkpoint_name_pattern 
            + "-" + config.model
            + "-" + date.today().strftime('%d-%m-%Y-%h-%M')
            + '{epoch}',
            mode='min',
        ),
        pl_callbacks.RichProgressBar(theme=custom_theme),
        ray_lightning.RayTrainReportCallback()
    ]

    trainer = pl.Trainer(
        enable_progress_bar=config.use_progress_bar,
        accumulate_grad_batches=config.accumulate_grad_batches,
        max_epochs=config.epochs,
        callbacks=trainer_callbacks,
        logger=logger,
        accelerator=config.trainer_accelerator,
        devices='auto',
        #num_nodes=config.nodes_count,
        strategy=ray_lightning.RayDDPStrategy(), 
        sync_batchnorm=config.sync_batchnorm,
        log_every_n_steps=config.log_every_n_steps,  # Controls the frequency of logging within training, by specifying how many training steps should occur between each logging event.
        precision=config.precision,
        num_sanity_val_steps=1,
        plugins=[ray_lightning.RayLightningEnvironment()]
    )
    
    '''
    if config.wandb_api_key_path:
        env = trainer.strategy.cluster_environment
        if env.global_rank() != 0 and env.local_rank() == 0:
            wandb.init(config=config, **wandb_config)
            wandb.log(
                {
                    "nfl_hist": wandb.Image(
                        "/home/lcornelis/code/proteo/proteo/datasets/data/ftd/processed/histogram.svg"
                    )
                }
            )'''

    module = Proteo(config, in_channels, out_channels, avg_node_degree)
    # print(module)
    trainer = ray_lightning.prepare_trainer(trainer)
    trainer.fit(module, train_loader, test_loader)


def search_hyperparameters():
    scaling_config = ScalingConfig(
        num_workers=1, use_gpu=True, resources_per_worker={'CPU': 8, 'GPU': 1}
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute='val_loss',
            checkpoint_score_order='min',
        ),
    )

    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        main,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    def hidden_channels(spec):
        hidden_channels_map = {'gat': [3, 10, 20, 25] , 'gat-v4': [[8, 8, 12], [10, 7, 12], [8, 16, 12]]}
        config = spec['train_loop_config']
        return hidden_channels_map[config['model']]
    
    def heads(spec):
        heads_map = {'gat': [1, 3, 5] , 'gat-v4': [[2, 2, 3], [3, 4, 5], [4, 4, 6]]}
        config = spec['train_loop_config'] 
        return heads_map[config['model']]


    search_space = {
        'model': tune.grid_search(['gat']),
        'heads': tune.sample_from([heads]),
        'seed': tune.randint(0, MAX_SEED),
        'hidden_channels': tune.sample_from(hidden_channels),
    }

    def trial_str_creator(trial):
        config = trial.config['train_loop_config']
        seed = config['seed']
        return f"model={config['model']},heads={config['heads']},seed={seed}"

    tuner = tune.Tuner(
        ray_trainer,
        param_space={'train_loop_config': search_space},
        tune_config=tune.TuneConfig(
            metric='val_loss',
            mode='min',
            # TODO: Infer num samples from search space.
            num_samples=2,
            trial_name_creator=trial_str_creator,
        ),
    )
    return tuner.fit()



if __name__ == '__main__':
    results = search_hyperparameters()
    results.get_dataframe().to_csv('ray_results_search_hyperparameters.csv')