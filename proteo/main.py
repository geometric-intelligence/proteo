import os
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

    config.update(search_config)
    setup_wandb(search_config, api_key_file = config.wandb_api_key_path)
    

    train_dataset, test_dataset = proteo_train.construct_datasets(config)
    train_loader, test_loader = proteo_train.construct_loaders(config, train_dataset, test_dataset)

    in_channels = train_dataset.feature_dim  # 1 dim of input
    out_channels = train_dataset.label_dim  # 1 dim of result

    avg_node_degree = proteo_train.avg_node_degree(test_dataset)

    module = proteo_train.Proteo(config, in_channels, out_channels, avg_node_degree)

    pl.seed_everything(config.seed)
    trainer = pl.Trainer(
        devices='auto',
        accelerator='auto',
        strategy=ray_lightning.RayDDPStrategy(),
        callbacks=[ray_lightning.RayTrainReportCallback()],
        plugins=[ray_lightning.RayLightningEnvironment()],
        enable_progress_bar=False,
        max_epochs=config.epochs,
    )
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
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    def hidden_channels(spec):
        hidden_channels_map = {'gat': [3, 10, 20, 25] , 'gat-v4': [[8, 8, 12], [10, 7, 12], [8, 16, 12]]}
        config = spec['train_loop_config']
        return hidden_channels_map[config['model']]
    
    def heads(spec):
        heads_map = {'gat': [1, 3, 5], 'gat-v4': [[2, 2, 3], [3, 4, 5], [4, 4, 6]]}
        config = spec['train_loop_config'] 
        return heads_map[config['model']]


    search_space = {
        'model': tune.grid_search(['gat']),
        'heads': tune.sample_from(heads),
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