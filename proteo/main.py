"""Run several trainings with hyper-parameter search.

Ray[Tune] manages the training of many neural networks, 
and thus we use:
- wandb.log in our custom CustomWandbCallback,
- Ray's CheckpointConfig.

Here, pl_module.logger is Ray's logger.

Notes
-----
This differs from training one single neural network in train.py, 
which only requires Lightning,
and thus we use:
- Lightning's WandbLogger logger, in our custom CustomWandbCallback,
- Lightning's ModelCheckpoint callback.

Here, pl_module.logger is Wandb's logger.

Remark on the Missing logger folder warning:
https://github.com/Lightning-AI/pytorch-lightning/discussions/12276
Seems OK to disregard.
"""

import os
import random
import time

import numpy as np
import pytorch_lightning as pl
import ray
import torch
import train as proteo_train
import wandb
from config_utils import CONFIG_FILE, read_config_from_file
from ray import tune
from ray.air.integrations.wandb import setup_wandb
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train import lightning as ray_lightning
from ray.train.lightning import RayTrainReportCallback
from ray.train.torch import TorchTrainer
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler

import proteo.callbacks_ray as proteo_callbacks_ray
from proteo.datasets.ftd import BINARY_Y_VALS_MAP, MULTICLASS_Y_VALS_MAP, Y_VALS_TO_NORMALIZE

MAX_SEED = 65535


def train_func(train_loop_config):
    """Train one neural network with Lightning.

    Configure Lightning with Ray.

    Parameters
    ----------
    search_config: Configuration parameters for training.
    """
    torch.set_float32_matmul_precision('medium')  # for performance
    config = read_config_from_file(CONFIG_FILE)
    pl.seed_everything(config.seed)

    # Update default config with sweep-specific train_loop_config
    # Update model specific parameters: hacky - is there a better way?
    model = train_loop_config['model']
    train_loop_config_model = {
        'hidden_channels': train_loop_config['hidden_channels'],
        'heads': train_loop_config['heads'],
        'num_layers': train_loop_config['num_layers'],
        'fc_dim': train_loop_config['fc_dim'],
        'fc_dropout': train_loop_config['fc_dropout'],
        'fc_act': train_loop_config['fc_act'],
        'weight_initializer': train_loop_config['weight_initializer'],
    }
    config[model].update(train_loop_config_model)
    # Remove keys that were already updated in nested configuration
    for key in train_loop_config_model:
        train_loop_config.pop(key)
    config.update(train_loop_config)

    setup_wandb(  # wandb.init, but for ray
        config.dict(),  # Transform Config object into dict for wandb
        project=config.project,
        api_key_file=os.path.join(config.root_dir, config.wandb_api_key_path),
        # Directory in dir needs to exist, otherwise wandb saves in /tmp
        dir=config.wandb_tmp_dir,
        mode="offline" if config.wandb_offline else "online",
    )

    train_dataset, test_dataset = proteo_train.construct_datasets(config)
    train_loader, test_loader = proteo_train.construct_loaders(config, train_dataset, test_dataset)

    avg_node_degree = proteo_train.compute_avg_node_degree(test_dataset)
    pos_weight = 1.0  # default value
    focal_loss_weight = [1.0]  # default value
    if config.y_val in BINARY_Y_VALS_MAP:
        pos_weight = proteo_train.compute_pos_weight(test_dataset, train_dataset)
    elif config.y_val in MULTICLASS_Y_VALS_MAP:
        focal_loss_weight = proteo_train.compute_focal_loss_weight(
            config, test_dataset, train_dataset
        )
    # For wandb logging top proteins
    protein_file_data = proteo_train.read_protein_file(train_dataset.processed_dir, config)
    protein_names = protein_file_data['Protein']
    metrics = protein_file_data['Metric']
    top_proteins_data = [[protein, metric] for protein, metric in zip(protein_names, metrics)]

    module = proteo_train.Proteo(
        config,
        in_channels=train_dataset.feature_dim,  # 1 dim of input
        out_channels=train_dataset.label_dim,  # 1 dim of result
        avg_node_degree=avg_node_degree,
        pos_weight=pos_weight,
        focal_loss_weight=focal_loss_weight,
    )
    if config.y_val in Y_VALS_TO_NORMALIZE:
        wandb.log(
            {
                "histogram original": wandb.Image(
                    os.path.join(
                        train_dataset.processed_dir,
                        f'{config.y_val}_{config.sex}_{config.mutation}_{config.modality}_orig_histogram.jpg',
                    )
                )
            }
        )
    wandb.log(
        {
            "histogram": wandb.Image(
                os.path.join(
                    train_dataset.processed_dir,
                    f'{config.y_val}_{config.sex}_{config.mutation}_{config.modality}_histogram.jpg',
                )
            ),
            "adjacency": wandb.Image(
                os.path.join(
                    train_dataset.processed_dir,
                    f"adjacency_{config.adj_thresh}_num_nodes_{config.num_nodes}_mutation_{config.mutation}_{config.modality}_sex_{config.sex}_masternodes_{config.use_master_nodes}.jpg",
                )
            ),
            "top_proteins": wandb.Table(
                columns=["Protein", "Metric"], data=top_proteins_data
            ),  # note this is in order from most to least different
            "parameters": wandb.Table(
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
            ),
        }
    )

    # Define Lightning's Trainer that will be wrapped by Ray's TorchTrainer
    trainer = pl.Trainer(
        devices='auto',
        accelerator='auto',
        strategy=ray_lightning.RayDDPStrategy(),
        callbacks=[
            proteo_callbacks_ray.CustomRayWandbCallback(),
            proteo_callbacks_ray.CustomRayReportLossCallback(),
            TuneReportCheckpointCallback(
                metrics={"val_loss": "val_loss", "train_loss": "train_loss"},
                filename=f"checkpoint.ckpt",
                on="validation_end",
            ),
        ],
        # How ray interacts with pytorch lightning
        plugins=[ray_lightning.RayLightningEnvironment()],
        enable_progress_bar=False,
        max_epochs=config.epochs,
        log_every_n_steps=config.log_every_n_steps,
        deterministic=True,
    )
    trainer = ray_lightning.prepare_trainer(trainer)
    # FIXME: When a trial errors, Wandb still shows it as "running".
    trainer.fit(module, train_loader, test_loader)
    torch.cuda.empty_cache()


def main():
    """Run hyper-parameter search with Ray[Tune].

    We pass a param_space dict to the tuner (a tune.Tuner).
    The tuner passes **param_space to ray_trainer (a TorchTrainer's that expects "train_loop_config").

    One set of hyperparams in param_space becomes:
    - trial.config["train_loop_config"] for trial_str_creator(trial)
    - trial_config["train_loop_config"] for functions(trial_config) that go into sample_from
    - train_loop_config for train_func(train_loop_config)

    See Also
    --------
    Inputs expected by a TorchTrainer for its init:
    https://docs.ray.io/en/latest/_modules/ray/train/torch/torch_trainer.html#TorchTrainer
    """
    config = read_config_from_file(CONFIG_FILE)
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    ray_tmp_dir = config.ray_tmp_dir
    os.makedirs(ray_tmp_dir, exist_ok=True)
    os.makedirs(config.ray_tmp_dir, exist_ok=True)
    ray.init(_temp_dir=config.ray_tmp_dir)

    # Wrap Lightning's Trainer with Ray's TorchTrainer for Tuner
    scaling_config = ScalingConfig(
        num_workers=1,
        use_gpu=True,
        resources_per_worker={'CPU': config.cpu_per_worker, 'GPU': config.gpu_per_worker},
    )

    run_config = RunConfig(
        storage_path=config.ray_results_dir,
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
    def num_layers(trial_config):
        num_layers_map = {
            'gat-v4': [None],  # Unused. Here for compatibility.
            'gat': config.gat_num_layers,
            'gcn': config.gcn_num_layers,
        }
        model = trial_config['train_loop_config']['model']
        return random.choice(num_layers_map[model])

    def hidden_channels(trial_config):
        hidden_channels_map = {
            'gat-v4': config.gat_v4_hidden_channels,
            'gat': config.gat_hidden_channels,
            'gcn': config.gcn_hidden_channels,
        }
        model = trial_config['train_loop_config']['model']
        return random.choice(hidden_channels_map[model])

    def heads(trial_config):
        heads_map = {
            'gat-v4': config.gat_v4_heads,
            'gat': config.gat_heads,
            'gcn': [None],  # Unused. Here for compatibility.
        }
        model = trial_config['train_loop_config']['model']
        return random.choice(heads_map[model])

    def fc_dim(trial_config):
        fc_dim_map = {
            'gat-v4': config.gat_v4_fc_dim,
            'gat': [None],  # Unused. Here for compatibility.
            'gcn': [None],
        }
        model = trial_config['train_loop_config']['model']
        return random.choice(fc_dim_map[model])

    def fc_dropout(trial_config):
        fc_dropout_map = {
            'gat-v4': config.gat_v4_fc_dropout,
            'gat': [None],  # Unused. Here for compatibility.
            'gcn': [None],  # Unused. Here for compatibility.
        }
        model = trial_config['train_loop_config']['model']
        return random.choice(fc_dropout_map[model])

    def fc_act(trial_config):
        fc_act_map = {
            'gat-v4': config.gat_v4_fc_act,
            'gat': [None],  # Unused. Here for compatibility.
            'gcn': [None],  # Unused. Here for compatibility.
        }
        model = trial_config['train_loop_config']['model']
        return random.choice(fc_act_map[model])

    def weight_initializer(trial_config):
        weight_initializer_map = {
            'gat-v4': config.gat_v4_weight_initializer,
            'gat': [None],  # Unused. Here for compatibility.
            'gcn': [None],  # Unused. Here for compatibility.
        }
        model = trial_config['train_loop_config']['model']
        return random.choice(weight_initializer_map[model])

    def trial_str_creator(trial):
        train_loop_config = trial.config['train_loop_config']
        model = train_loop_config['model']
        seed = train_loop_config['seed']
        return f"model={model},seed={seed}"

    search_space = {
        'model': tune.grid_search(config.model_grid_search),
        # Model-specific parameters
        'num_layers': tune.sample_from(num_layers),
        'hidden_channels': tune.sample_from(hidden_channels),
        'heads': tune.sample_from(heads),
        'fc_dim': tune.sample_from(fc_dim),
        'fc_dropout': tune.sample_from(fc_dropout),
        'fc_act': tune.sample_from(fc_act),
        'weight_initializer': tune.sample_from(weight_initializer),
        # Shared parameters
        'seed': tune.randint(0, MAX_SEED),
        'lr': tune.loguniform(config.lr_min, config.lr_max),
        'batch_size': tune.choice(config.batch_size_choices),
        'lr_scheduler': tune.choice(config.lr_scheduler_choices),
        'dropout': tune.choice(config.dropout_choices),
        'l1_lambda': tune.loguniform(config.l1_lambda_min, config.l1_lambda_max),
        'act': tune.choice(config.act_choices),
        'num_nodes': tune.choice(config.num_nodes_choices),
        'adj_thresh': tune.choice(config.adj_thresh_choices),
        'mutation': tune.grid_search(config.mutation_choices),
        'sex': tune.grid_search(config.sex_choices),
        'modality': tune.grid_search(config.modality_choices),
        'y_val': tune.grid_search(config.y_val_choices),
    }

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=config['epochs'],
        grace_period=config.grace_period,
        reduction_factor=config.reduction_factor,
    )
    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric='val_loss',
            mode='min',
            num_samples=config.num_samples,  # Repeats grid search options n times through
            trial_name_creator=trial_str_creator,
            scheduler=scheduler,
        ),
    )
    results = tuner.fit()
    results.get_dataframe(filter_metric="val_loss", filter_mode="min").to_csv(
        'ray_results_search_hyperparameters.csv'
    )


if __name__ == '__main__':
    main()
