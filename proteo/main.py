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
    #TODO: Hacky - is there a better way?
    model_name = search_config['train_loop_config']['model']
    search_config['model'] = model_name
    updated_parameters = {'hidden_channels': search_config['train_loop_config']['hidden_channels'], 
                         'heads':search_config['train_loop_config']['heads']}
    config[model_name].update(updated_parameters)
    search_config.pop('train_loop_config')
    print("=====================================")
    print(search_config)
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

    hidden_channels_map = {'gat': [3, 10, 20, 25], 'gat-v4': [[8, 8, 12], [10, 7, 12], [8, 16, 12]]}
    heads_map = {'gat': [1, 3, 5], 'gat-v4': [[2, 2, 3], [3, 4, 5], [4, 4, 6]]}

    def generate_configurations():
        configurations = []
        for model in ['gat', 'gat-v4']:
            for hidden_channels in hidden_channels_map[model]:
                for heads in heads_map[model]:
                    configurations.append({
                        'model': model,
                        'hidden_channels': hidden_channels,
                        'heads': heads
                    })
        return configurations

    search_space = {
        'train_loop_config': tune.grid_search(generate_configurations()),
        'seed': tune.randint(0, MAX_SEED),
        'lr': tune.loguniform(1e-4, 1e-1),
}

    def trial_str_creator(trial):
        config = trial.config['train_loop_config']
        seed = config['seed']
        nested_config = config['train_loop_config']
        return f"model={nested_config['model']},heads={nested_config['heads']},seed={seed}"

    tuner = tune.Tuner(
        ray_trainer,
        param_space={'train_loop_config': search_space},
        tune_config=tune.TuneConfig(
            metric='val_loss',
            mode='min',
            # TODO: Infer num samples from search space.
            num_samples=2, # Repeats grid search options 2 times through with random seed
            trial_name_creator=trial_str_creator,
        ),
    )
    return tuner.fit()



if __name__ == '__main__':
    results = search_hyperparameters()
    results.get_dataframe().to_csv('ray_results_search_hyperparameters.csv')