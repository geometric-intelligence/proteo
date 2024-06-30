"""Run several trainings with hyper-parameter search.

Ray[Tune] manages the training of many neural networks, 
and thus we use:
- wandb.log in our custom CustomLogsCallback,
- Ray's CheckpointConfig.

Here, pl_module.logger is Ray's logger.

Notes
-----
This differs from training one single neural network in train.py, 
which only requires Lightning,
and thus we use:
- Lightning's WandbLogger logger, in our custom CustomLogsCallback,
- Lightning's ModelCheckpoint callback.

Here, pl_module.logger is Wandb's logger.

Remark on the Missing logger folder warning:
https://github.com/Lightning-AI/pytorch-lightning/discussions/12276
Seems OK to disregard.
"""

import os

import numpy as np
import pytorch_lightning as pl
import ray
import torch
import train as proteo_train
from config_utils import CONFIG_FILE, read_config_from_file
from ray import tune
from ray.air.integrations.wandb import setup_wandb
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train import lightning as ray_lightning
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler

import proteo.callbacks_ray as proteo_callbacks_ray

MAX_SEED = 65535


def train_func(search_config):
    """Train one neural network with Lightning.

    Configure Lightning with Ray.

    Parameters
    ----------
    search_config: Configuration parameters for training.
    """
    torch.set_float32_matmul_precision('medium')  # for performance
    config = read_config_from_file(CONFIG_FILE)
    pl.seed_everything(config.seed)

    # TODO: Hacky - is there a better way?
    # Update model specific parameters
    model_name = search_config['model']
    updated_parameters = {
        'hidden_channels': search_config['hidden_channels'],
        'heads': search_config['heads'],
        'num_layers': search_config['num_layers'],
        'lr': search_config['lr'],
    }
    config[model_name].update(updated_parameters)
    # Remove keys that were already updated in nested configuration
    for key in updated_parameters:
        search_config.pop(key)
    config.update(search_config)

    setup_wandb(
        search_config,
        project=config.project,
        api_key_file=os.path.join(config.root_dir, config.wandb_api_key_path),
        dir=os.path.join(
            config.root_dir, config.output_dir
        ),  # dir needs to exist, otherwise wandb saves in /tmp
        mode="offline" if config.wandb_offline else "online",
    )

    train_dataset, test_dataset = proteo_train.construct_datasets(config)
    train_loader, test_loader = proteo_train.construct_loaders(config, train_dataset, test_dataset)

    module = proteo_train.Proteo(
        config,
        in_channels=train_dataset.feature_dim,  # 1 dim of input
        out_channels=train_dataset.label_dim,  # 1 dim of result
        avg_node_degree=proteo_train.avg_node_degree(test_dataset),
    )

    # Define Lightning's Trainer that will be wrapped by Ray's TorchTrainer
    trainer = pl.Trainer(
        devices='auto',
        accelerator='auto',
        strategy=ray_lightning.RayDDPStrategy(),
        callbacks=[
            proteo_callbacks_ray.CustomRayLogsCallback(),
            proteo_callbacks_ray.CustomRayTrainReportCallback(
                checkpoint_interval=config.checkpoint_interval),
        ],
        plugins=[
            ray_lightning.RayLightningEnvironment()
        ],  # How ray interacts with pytorch lightning
        enable_progress_bar=False,
        max_epochs=config.epochs,
    )
    trainer = ray_lightning.prepare_trainer(trainer)
    # FIXME: When a trial errors, Wandb still shows it as "running".
    trainer.fit(module, train_loader, test_loader)


def main():
    config = read_config_from_file(CONFIG_FILE)
    output_dir = os.path.join(config.root_dir, config.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(config.ray_tmp_dir, exist_ok=True)

    ray.init(_temp_dir=config.ray_tmp_dir)

    # Wrap Lightning's Trainer with Ray's TorchTrainer for Tuner
    scaling_config = ScalingConfig(
        num_workers=1,
        use_gpu=True,
        resources_per_worker={'CPU': config.cpu_per_worker, 'GPU': config.gpu_per_worker},
    )

    run_config = RunConfig(
        storage_path=os.path.join(config.root_dir, config.ray_results_dir),
        checkpoint_config=CheckpointConfig(
            num_to_keep=config.num_to_keep,
            checkpoint_score_attribute='val_loss',
            checkpoint_score_order='min',
        ),
    )

    ray_trainer = TorchTrainer(
        train_func,  # Contains Lightning's Trainer
        scaling_config=scaling_config,
        run_config=run_config,
    )

    # Configure hyperparameter search for Tuner
    def hidden_channels(spec):
        # Note that hidden channels must be divisible by heads for gat
        hidden_channels_map = {
            'gat': config.gat_hidden_channels,
            'gat-v4': config.gat_v4_hidden_channels,
            'gcn': config.gcn_hidden_channels,
        }
        model = spec['train_loop_config']['model']
        model_params = hidden_channels_map[model]
        random_index = np.random.choice(len(model_params))
        return model_params[random_index]

    def heads(spec):
        heads_map = {
            'gat': config.gat_heads,
            'gat-v4': config.gat_v4_heads,
            'gcn': [None],  # Unused. Here for compatibility.
        }
        model = spec['train_loop_config']['model']
        model_params = heads_map[model]
        random_index = np.random.choice(len(model_params))
        return model_params[random_index]

    search_space = {
        'model': tune.grid_search(config.model_grid_search),
        'seed': tune.randint(0, MAX_SEED),
        'hidden_channels': tune.sample_from(hidden_channels),
        'num_layers': tune.choice(config.num_layers_choice),
        'heads': tune.sample_from(heads),
        'lr': tune.loguniform(config.lr_min, config.lr_max),
        'batch_size': tune.choice(config.batch_size_choice),
        'scheduler': tune.choice(config.scheduler_choice),
        'dropout': tune.choice(config.dropout_choice),
    }

    def trial_str_creator(trial):
        config = trial.config['train_loop_config']
        seed = config['seed']
        return f"model={config['model']},num_layers={config['num_layers']},seed={seed}"

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=config['epochs'],
        grace_period=config.grace_period,
        reduction_factor=config.reduction_factor,
    )

    tuner = tune.Tuner(
        ray_trainer,
        param_space={'train_loop_config': search_space},
        tune_config=tune.TuneConfig(
            metric='val_loss',
            mode='min',
            num_samples=config.num_samples,  # Repeats grid search options n times through
            trial_name_creator=trial_str_creator,
            scheduler=scheduler,
        ),
    )
    results = tuner.fit()
    results.get_dataframe().to_csv('ray_results_search_hyperparameters.csv')


if __name__ == '__main__':
    main()
