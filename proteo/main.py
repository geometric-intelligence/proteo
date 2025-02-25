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
import itertools

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
from proteo.datasets.ftd_folds import BINARY_Y_VALS_MAP, MULTICLASS_Y_VALS_MAP, Y_VALS_TO_NORMALIZE, FTDDataset
from proteo.train import construct_datasets

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
    train_dataset, val_dataset = construct_datasets(config)
    train_loader, val_loader = proteo_train.construct_loaders(config, train_dataset, val_dataset)

    avg_node_degree = proteo_train.compute_avg_node_degree(val_dataset)
    pos_weight = 1.0  # default value
    focal_loss_weight = [1.0]  # default value
    if config.y_val in BINARY_Y_VALS_MAP:
        pos_weight = proteo_train.compute_pos_weight(val_dataset, train_dataset)
    elif config.y_val in MULTICLASS_Y_VALS_MAP:
        focal_loss_weight = proteo_train.compute_focal_loss_weight(
            config, val_dataset, train_dataset
        )

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
                        f'{config.y_val}_{config.sex}_{config.mutation}_{config.modality}_{config.num_folds}fold_{config.fold}_orig_histogram.jpg',
                    )
                )
            }
        )
    # Update the file paths for logging and saving
    log_data = {
        "histogram": wandb.Image(
            os.path.join(
                train_dataset.processed_dir,
                f'{config.y_val}_{config.sex}_{config.mutation}_{config.modality}_{config.num_folds}fold_{config.fold}_histogram.jpg',
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
    # Add the single adjacency image to the log
    log_data["adjacency"] = wandb.Image(
        os.path.join(
            train_dataset.processed_dir,
            f'adjacency_{config.adj_thresh}_num_nodes_{config.num_nodes}_adjthresh_{config.adj_thresh}_mutation_{config.mutation}_{config.modality}_sex_{config.sex}_{config.num_folds}fold_{config.fold}.jpg',
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
    trainer.fit(module, train_loader, val_loader)
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

    # Global parameters (apply to all models)
    global_space = {
        'seed': [config.seed],
        'fold': list(range(config.num_folds)),
        #'lr': list(np.logspace(np.log10(config.lr_min), np.log10(config.lr_max), 5)),
        'batch_size': config.batch_size_choices,
        'lr_scheduler': config.lr_scheduler_choices,
        'dropout': config.dropout_choices,
        #'l1_lambda': list(np.logspace(np.log10(config.l1_lambda_min), np.log10(config.l1_lambda_max), 5)),
        'act': config.act_choices,
        'num_nodes': config.num_nodes_choices,
        'adj_thresh': config.adj_thresh_choices,
        'mutation': config.mutation_choices,
        'sex': config.sex_choices,
        'modality': config.modality_choices,
        'y_val': config.y_val_choices,
    }


    ## Model-specific parameters for each model type:
    model_space = {
        'gat-v4': {
            'num_layers': [None],
            'hidden_channels': config.gat_v4_hidden_channels,  # list of candidate values
            'heads': config.gat_v4_heads,
            'fc_dim': config.gat_v4_fc_dim,
            'fc_dropout': config.gat_v4_fc_dropout,
            'fc_act': config.gat_v4_fc_act,
            'weight_initializer': config.gat_v4_weight_initializer,
            'channel_list': [None],  # not used
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

    def trial_str_creator(trial):
        train_loop_config = trial.config['train_loop_config']
        model = train_loop_config['model']
        return f"model={model},seed={config.seed},num_folds={config.num_folds}"
    
    def flatten_configs(global_dict, model_dict, model_name):
        """Generates a list of full config dictionaries for a given model."""
        # Copy global parameters and fix 'model' to the given model_name.
        global_config = global_dict.copy()
        global_config['model'] = [model_name]
        
        # Ensure each value is a list
        global_keys = list(global_config.keys())
        global_values = [global_config[k] if isinstance(global_config[k], list) else [global_config[k]] for k in global_keys]
        
        model_keys = list(model_dict.keys())
        model_values = [model_dict[k] if isinstance(model_dict[k], list) else [model_dict[k]] for k in model_keys]
        
        full_configs = []
        for g_vals in itertools.product(*global_values):
            gc = dict(zip(global_keys, g_vals))
            for m_vals in itertools.product(*model_values):
                mc = dict(zip(model_keys, m_vals))
                config_combo = {}
                config_combo.update(gc)
                config_combo.update(mc)
                full_configs.append(config_combo)
        return full_configs
    
    all_configs = []
    for model in config.model_grid_search:
        # Ensure model-specific parameters are lists:
        for key, value in model_space[model].items():
            if not isinstance(value, list):
                model_space[model][key] = list(value)
        all_configs.extend(flatten_configs(global_space, model_space[model], model))

    search_space = tune.grid_search(all_configs)
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
