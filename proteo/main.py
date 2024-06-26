import os
import numpy as np
import pytorch_lightning as pl
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train import lightning as ray_lightning
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler
from proteo.datasets.ftd import ROOT_DIR, FTDDataset
from torch_geometric.loader import DataLoader
from config_utils import CONFIG_FILE, Config, read_config_from_file

import train as proteo_train

MAX_SEED = 65535


def train_func(search_config):
    """
    Trains a PyTorch model using the provided configuration.

    Args:
        config (dict): Configuration parameters for training.

    Returns:
        None
    """
    config = read_config_from_file(CONFIG_FILE)
    #TODO: Hacky - is there a better way?
    # Update model specific parameters
    model_name = search_config['model']
    updated_parameters = {'hidden_channels': search_config['hidden_channels'], 
                         'heads':search_config['heads'], 'lr': search_config['lr']}
    config[model_name].update(updated_parameters)
    # Remove keys that were already updated in nested configuration
    for key in updated_parameters:
        search_config.pop(key)
    config.update(search_config)
    setup_wandb(search_config, api_key_file = config.wandb_api_key_path)
    

    train_dataset, test_dataset = proteo_train.construct_datasets(config)
    train_loader, test_loader = proteo_train.construct_loaders(config, train_dataset, test_dataset)

    in_channels = train_dataset.feature_dim  # 1 dim of input
    out_channels = train_dataset.label_dim  # 1 dim of resul÷≥…t
    avg_node_degree = proteo_train.avg_node_degree(test_dataset)

    module = proteo_train.Proteo(config, in_channels, out_channels, avg_node_degree)

    pl.seed_everything(config.seed)
    trainer = pl.Trainer(
        devices='auto',
        accelerator='auto',
        strategy=ray_lightning.RayDDPStrategy(),
        callbacks=[ray_lightning.RayTrainReportCallback()],
        plugins=[ray_lightning.RayLightningEnvironment()], # How ray interacts with pytorch lightning
        enable_progress_bar=False,
        max_epochs=config.epochs,
    )
    trainer = ray_lightning.prepare_trainer(trainer)
    trainer.fit(module, train_loader, test_loader)




def search_hyperparameters():
    # training should use one worker, one GPU, and 8 CPUs per worker.
    scaling_config = ScalingConfig(
        num_workers=1, use_gpu=True, resources_per_worker={'CPU': 8, 'GPU': 1}
    )

    # keeping only the best checkpoint based on minimum validation loss
    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute='val_loss',
            checkpoint_score_order='min',
        ),
    )

    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    def hidden_channels(spec):
        # Note that hidden channels must be divisible by heads for gat
        hidden_channels_map = {'gat': [4, 8, 12, 20] , 'gat-v4': [[8, 8, 12], [10, 7, 12], [8, 16, 12]]}
        config = spec['train_loop_config']
        model_params = hidden_channels_map[config['model']]
        random_index = np.random.choice(len(model_params))
        return model_params[random_index]
    
    def heads(spec):
        heads_map = {'gat': [1, 2, 4], 'gat-v4': [[2, 2, 3], [3, 4, 5], [4, 4, 6]]}
        config = spec['train_loop_config'] 
        model_params = heads_map[config['model']]
        random_index = np.random.choice(len(model_params))
        return model_params[random_index]

    search_space = {
        'model': tune.grid_search(['gat', 'gat-v4']),
        'seed': tune.randint(0, MAX_SEED),
        'hidden_channels': tune.sample_from(hidden_channels),
        'heads': tune.sample_from(heads),
        'lr': tune.loguniform(1e-4, 1e-1),
        'batch_size': tune.choice([6, 12, 32, 64, 80]),
}

    def trial_str_creator(trial):
        config = trial.config['train_loop_config']
        seed = config['seed']
        return f"model={config['model']},heads={config['heads']},seed={seed}"

    config = read_config_from_file(CONFIG_FILE)
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t= config['epochs'],
        grace_period=2,
        reduction_factor=4,
    )
    
    tuner = tune.Tuner(
        ray_trainer,
        param_space={'train_loop_config': search_space},
        tune_config=tune.TuneConfig(
            metric='val_loss',
            mode='min',
            # TODO: Infer num samples from search space.
            num_samples=30, # Repeats grid search options 2 times through with random seed
            trial_name_creator=trial_str_creator,
            scheduler=scheduler,
        ),
    )
    return tuner.fit()



if __name__ == '__main__':
    results = search_hyperparameters()
    results.get_dataframe().to_csv('ray_results_search_hyperparameters.csv')