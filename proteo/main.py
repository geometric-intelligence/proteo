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
        'channel_list': train_loop_config['channel_list'],
        'norm': train_loop_config['norm'],
        'plain_last': train_loop_config['plain_last'],
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

    # Use the fold from train_loop_config
    fold = train_loop_config['fold']
    train_dataset, test_dataset = proteo_train.construct_datasets(config, fold)
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

    module = proteo_train.Proteo(
        config,
        in_channels=train_dataset.feature_dim,  # 1 dim of input
        out_channels=train_dataset.label_dim,  # 1 dim of result
        avg_node_degree=avg_node_degree,
        pos_weight=pos_weight,
        focal_loss_weight=focal_loss_weight,
        use_weights=config.use_weights
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
    # Initialize the dictionary to log
    log_data = {
        "histogram": wandb.Image(
            os.path.join(
                train_dataset.processed_dir,
                f'{config.y_val}_{config.sex}_{config.mutation}_{config.modality}_histogram.jpg',
            )
        ),
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
    if config.sex_specific_adj:
        # Add both male and female adjacency images to the log
        log_data["adjacency_M"] = wandb.Image(
            os.path.join(
                train_dataset.processed_dir,
                f"adjacency_num_nodes_{config.num_nodes}_adjthresh_{config.adj_thresh}_mutation_{config.mutation}_{config.modality}_sex_{config.sex}_masternodes_{config.use_master_nodes}_sex_specific_{config.sex_specific_adj}_M.jpg",
            )
        )
        log_data["adjacency_F"] = wandb.Image(
            os.path.join(
                train_dataset.processed_dir,
                f'adjacency_num_nodes_{config.num_nodes}_adjthresh_{config.adj_thresh}_mutation_{config.mutation}_{config.modality}_sex_{config.sex}_masternodes_{config.use_master_nodes}_sex_specific_{config.sex_specific_adj}_F.jpg',
            )
        )
    else:
        # Add the single adjacency image to the log
        log_data["adjacency"] = wandb.Image(
            os.path.join(
                train_dataset.processed_dir,
                f'adjacency_{config.adj_thresh}_num_nodes_{config.num_nodes}_adjthresh_{config.adj_thresh}_mutation_{config.mutation}_{config.modality}_sex_{config.sex}_masternodes_{config.use_master_nodes}_sex_specific_{config.sex_specific_adj}.jpg',
            )
        )
    wandb.log(log_data)

    # Define Lightning's Trainer that will be wrapped by Ray's TorchTrainer
    trainer = pl.Trainer(
        devices=[0],  # because we constrain one GPU per worker, this will always find one
        accelerator=config.trainer_accelerator,
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
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config.devices))

    # Wrap Lightning's Trainer with Ray's TorchTrainer for Tuner
    if config.use_gpu:
        resources_per_worker = {'GPU': config.gpu_per_worker}
    else:
        resources_per_worker = {'CPU': config.cpu_per_worker}
    scaling_config = ScalingConfig(
        num_workers=1,
        use_gpu=config.use_gpu,
        resources_per_worker=resources_per_worker,
        placement_strategy="STRICT_SPREAD",
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
    model_param_grid = {
        'gat-v4': {
            'num_layers': [None],
            'hidden_channels': config.gat_v4_hidden_channels,
            'heads': config.gat_v4_heads,
            'fc_dim': config.gat_v4_fc_dim,
            'fc_dropout': config.gat_v4_fc_dropout,
            'fc_act': config.gat_v4_fc_act,
            'weight_initializer': config.gat_v4_weight_initializer,
            'channel_list': [None],
            'norm': [None],
            'plain_last': [None]
        },
        'gat': {
            'num_layers': config.gat_num_layers,
            'hidden_channels': config.gat_hidden_channels,
            'heads': config.gat_heads,
            'fc_dim': [None],
            'fc_dropout': [None],
            'fc_act': [None],
            'weight_initializer': [None],
            'channel_list': [None],
            'norm': [None],
            'plain_last': [None]
        },
        'gcn': {
            'num_layers': config.gcn_num_layers,
            'hidden_channels': config.gcn_hidden_channels,
            'heads': [None],
            'fc_dim': [None],
            'fc_dropout': [None],
            'fc_act': [None],
            'weight_initializer': [None],
            'channel_list': [None],
            'norm': [None],
            'plain_last': [None]
        },
        'mlp': {
            'num_layers': [None],
            'hidden_channels': [None],
            'heads': [None],
            'fc_dim': [None],
            'fc_dropout': [None],
            'fc_act': [None],
            'weight_initializer': [None],
            'channel_list': config.mlp_channel_lists,
            'norm': config.mlp_norms,
            'plain_last': config.mlp_plain_last
        }
    }

    # Build search space with model-specific parameter grids
    search_spaces = []
    
    # Create a separate search space for each model type
    for model_name in config.model_grid_search:
        model_space = {
            'model': model_name,
            # Shared parameters
            'seed': config.seed,  # Use fixed seed instead of randint
            'fold': tune.grid_search(list(range(config.num_folds))),  # Add fold as grid search parameter
            'lr': tune.grid_search(np.logspace(np.log10(config.lr_min), np.log10(config.lr_max), 5)),
            'batch_size': tune.grid_search(config.batch_size_choices),
            'lr_scheduler': tune.grid_search(config.lr_scheduler_choices),
            'dropout': tune.grid_search(config.dropout_choices),
            'l1_lambda': tune.grid_search(np.logspace(np.log10(config.l1_lambda_min), np.log10(config.l1_lambda_max), 5)),
            'act': tune.grid_search(config.act_choices),
            'num_nodes': tune.grid_search(config.num_nodes_choices),
            'adj_thresh': tune.grid_search(config.adj_thresh_choices),
            'mutation': tune.grid_search(config.mutation_choices),
            'sex': tune.grid_search(config.sex_choices),
            'modality': tune.grid_search(config.modality_choices),
            'y_val': tune.grid_search(config.y_val_choices),
            'sex_specific_adj': tune.grid_search(config.sex_specific_adj_choices),
            'use_weights': tune.grid_search(config.use_weights_choices)
        }
        
        # Add model-specific parameters
        for param, values in model_param_grid[model_name].items():
            if values != [None]:
                model_space[param] = tune.grid_search(values)
            else:
                model_space[param] = None
                
        search_spaces.append(model_space)

    # Combine all search spaces
    search_space = tune.grid_search(search_spaces)

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric='val_loss',
            mode='min',
            trial_name_creator=trial_str_creator,
        ),
    )
    results = tuner.fit()
    results.get_dataframe(filter_metric="val_loss", filter_mode="min").to_csv(
        'ray_results_search_hyperparameters.csv'
    )

if __name__ == '__main__':
    main()
