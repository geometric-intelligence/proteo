import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import train as proteo_train
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from proteo.datasets.ftd import FTDDataset, reverse_log_transform

device = 'cuda' if torch.cuda.is_available() else 'cpu'
c9_mean_dict = {
    "['M']_csf": 2.20139473218633,
    "['F']_plasma": 2.5069125020915246,
    "['F']_csf": 2.3905483112831987,
    "['M', 'F']_plasma": 2.4382370774886417,
    "['M', 'F']_csf": 2.323617044833538,
}
c9_std_dict = {
    "['M']_csf": 0.9414006476156331,
    "['F']_plasma": 0.9801098341235991,
    "['F']_csf": 0.95108017948172,
    "['M', 'F']_plasma": 0.9639665529956777,
    "['M', 'F']_csf": 0.951972757962228,
}
MAPT_mean_dict = {
    "['M']_csf": 2.080694213697065,
    "['M']_plasma": 2.1657279439973016,
    "['F']_csf": 2.0152637385189265,
    "['M', 'F']_csf": 2.0454624193703754,
}
MAPT_std_dict = {
    "['M']_csf": 0.6213240141321779,
    "['M']_plasma": 0.6840496344783593,
    "['F']_csf": 0.7999340389937927,
    "['M', 'F']_csf": 0.7237378322971036,
}
GRN_mean_dict = {
    "['M']_csf": 2.178815827183045,
    "['F']_plasma": 3.120974866855634,
    "['F']_csf": 3.2586357196385385,
    "['M', 'F']_csf": 2.752470145050026,
}
GRN_std_dict = {
    "['M']_csf": 0.7776541040264751,
    "['F']_plasma": 1.2401561087499366,
    "['F']_csf": 1.1764975422138229,
    "['M', 'F']_csf": 1.1441881493582908,
}
all_nodes_csf_mean = 2.124088581365514
all_nodes_csf_std = 0.8733420033790319
all_nodes_plasma_mean = 2.1761493077110043
all_nodes_plasma_std = 0.9054411007915015


def load_checkpoint(relative_checkpoint_path):
    '''Load the checkpoint as a module. Note levels_up depends on the directory structure of the ray_results folder'''
    relative_checkpoint_path = os.path.join(relative_checkpoint_path, 'checkpoint.ckpt')
    # Check if the file exists to avoid errors
    if not os.path.isfile(relative_checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {relative_checkpoint_path}")

    # Load the checkpoint dictionary using PyTorch directly to modify the config
    checkpoint = torch.load(relative_checkpoint_path, map_location=torch.device('cpu'))

    # Ensure that the 'use_master_nodes' attribute is in the checkpoint's config
    if not hasattr(checkpoint['hyper_parameters']['config'], 'use_master_nodes') or not hasattr(
        checkpoint['hyper_parameters']['config'], 'master_nodes'
    ):
        checkpoint['hyper_parameters']['config'].use_master_nodes = False  # Add default value
        checkpoint['hyper_parameters']['config'].master_nodes = []

    torch.save(checkpoint, relative_checkpoint_path)
    module = proteo_train.Proteo.load_from_checkpoint(relative_checkpoint_path)
    return module


def load_config(module):
    '''Load the config from the module  and return it'''
    config = module.config
    return config

def construct_loaders_eval(config, train_dataset, test_dataset):
    # Make DataLoader objects to handle batching
    train_loader = DataLoader(  # makes into one big graph
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
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
    return train_loader, test_loader

def load_model_and_predict(module, config, device='cuda'):
    '''Run the module with the correct train and test datasets and return the predictions and targets'''
    module.to(device)
    module.eval()
    # pl.seed_everything(config.seed)
    train_dataset, test_dataset = proteo_train.construct_datasets(config)
    train_loader, test_loader = construct_loaders_eval(config, train_dataset, test_dataset)
    # Get predictions and targets for the training set
    train_preds, train_targets = [], []
    for batch in train_loader:
        batch.to(device)
        with torch.no_grad():
            # Forward pass
            pred = module(batch)
            target = batch.y.view(pred.shape)
        # Store predictions and targets
        train_preds.append(pred.cpu())
        train_targets.append(target.cpu())
    train_preds = torch.cat(train_preds)
    train_targets = torch.cat(train_targets)

    # Calculate MSE for training set
    train_mse = F.mse_loss(train_preds, train_targets).item()

    # Get predictions and targets for the validation set
    val_preds, val_targets = [], []
    for batch in test_loader:
        batch.to(device)
        # Forward pass
        pred = module(batch)
        target = batch.y.view(pred.shape)

        # Store predictions and targets
        val_preds.append(pred.cpu())
        val_targets.append(target.cpu())
    val_preds = torch.cat(val_preds)
    val_targets = torch.cat(val_targets)

    # Calculate MSE for validation set
    val_mse = F.mse_loss(val_preds, val_targets).item()
    print("Normalized Val MSE:", val_mse)
    print("Normalized train MSE:", train_mse)
    return train_preds, train_targets, train_mse, val_preds, val_targets, val_mse


def full_load_and_run_and_convert(relative_checkpoint_path, device, mean, std):
    '''Call all the functions to load the checkpoint, run the model and convert the predictions back to the original units'''
    module = load_checkpoint(relative_checkpoint_path)
    config = load_config(module)
    train_preds, train_targets, train_mse, val_preds, val_targets, val_mse = load_model_and_predict(
        module, config, device
    )
    train_preds = reverse_log_transform(train_preds, mean, std)
    train_targets = reverse_log_transform(train_targets, mean, std)
    train_mse = F.mse_loss(train_preds, train_targets)
    train_rmse = torch.sqrt(train_mse)
    val_preds = reverse_log_transform(val_preds, mean, std)
    val_targets = reverse_log_transform(val_targets, mean, std)
    val_mse = F.mse_loss(val_preds, val_targets)
    val_rmse = torch.sqrt(val_mse)
    print(val_preds.view(-1).detach().cpu().numpy())
    val_z_scores = zscore(
        val_preds.view(-1).detach().cpu().numpy() - val_targets.view(-1).detach().cpu().numpy(),
        ddof=1,
    )
    print("Original Units Val MSE:", val_mse)
    print("Original Units Val RMSE:", val_rmse)
    print("Val Z scores:", val_z_scores)
    return [
        train_preds,
        train_targets,
        train_mse,
        train_rmse,
        val_preds,
        val_targets,
        val_mse,
        val_rmse,
        val_z_scores,
    ]

def process_checkpoints(checkpoint_paths, mean_dict, std_dict, device):
    results = []
    i = 1
    for checkpoint_path in checkpoint_paths:
        print(f"Loading checkpoint from: {checkpoint_path}")
        module = load_checkpoint(checkpoint_path)
        config = load_config(module)
        print(config)
        # print("Config being used:", config)
        print(f"{i} best checkpoint for {config.sex} and {config.modality}")
        key = f"{config.sex}_{config.modality}"
        mean = mean_dict[key]
        std = std_dict[key]

        result = full_load_and_run_and_convert(checkpoint_path, device, mean, std)
        results.append(result)
        i += 1
    return results

def get_sex_mutation_age_distribution(config):
    # Make an instance of the FTDDataset class to use the load_csv_data_pre_pt_files method
    root = config.data_dir
    train_dataset = FTDDataset(root, "train", config)
    (
        _,
        _,
        _,
        _,
        filtered_sex_col,
        filtered_mutation_col,
        filtered_age_col,
    ) = train_dataset.load_csv_data_pre_pt_files(config)
    # Splitting indices only
    (
        train_sex_labels,
        test_sex_labels,
        train_mutation_labels,
        test_mutation_labels,
        train_age_labels,
        test_age_labels,
    ) = train_test_split(
        filtered_sex_col, filtered_mutation_col, filtered_age_col, test_size=0.20, random_state=42
    )
    return (
        train_sex_labels,
        test_sex_labels,
        train_mutation_labels,
        test_mutation_labels,
        train_age_labels,
        test_age_labels,
    )

def predict_for_subgroups_with_labels(relative_checkpoint_path, device, mean, std):
    module = load_checkpoint(relative_checkpoint_path)
    config = load_config(module)
    # Load the model and get the predictions and targets
    train_preds, train_targets, _, val_preds, val_targets, _ = load_model_and_predict(
        module, config, device
    )

    # Convert the predictions back to the original units
    train_preds = reverse_log_transform(train_preds, mean, std)
    train_targets = reverse_log_transform(train_targets, mean, std)
    val_preds = reverse_log_transform(val_preds, mean, std)
    val_targets = reverse_log_transform(val_targets, mean, std)
    # Get the group labels for the training and validation sets
    (
        train_sex_labels,
        test_sex_labels,
        train_mutation_labels,
        test_mutation_labels,
        train_age_labels,
        test_age_labels,
    ) = get_sex_mutation_age_distribution(config)

    # Define subgroups
    sex_labels = ['M', 'F']
    mutation_labels = ["C9orf72", "MAPT", "GRN", "CTL"]
    age_bins = [(10, 30), (30, 50), (50, 70), (70, 90)]

    # Calculate MSE for each subgroup in training data
    train_rmse_results = {}
    val_rmse_results = {}
    for sex in sex_labels:
        for mutation in mutation_labels:
            for age_range in age_bins:
                subgroup_name = f"{sex}_{mutation}_{age_range[0]}-{age_range[1]}"

                # Create mask for current subgroup
                train_mask = (
                    (train_sex_labels == sex)
                    & (train_mutation_labels == mutation)
                    & (train_age_labels >= age_range[0])
                    & (train_age_labels < age_range[1])
                )
                train_mask = torch.tensor(train_mask.values, dtype=torch.bool)

                # Filter predictions and targets based on mask
                train_subgroup_preds = train_preds[train_mask]
                train_subgroup_targets = train_targets[train_mask]

                # Compute MSE if there are any samples in the subgroup
                if len(train_subgroup_preds) > 0:
                    mse = F.mse_loss(train_subgroup_preds, train_subgroup_targets)
                    rmse = torch.sqrt(mse)
                    train_rmse_results[subgroup_name] = rmse, len(train_subgroup_preds)
                else:
                    train_rmse_results[subgroup_name] = "No samples in subgroup"

                val_mask = (
                    (test_sex_labels == sex)
                    & (test_mutation_labels == mutation)
                    & (test_age_labels >= age_range[0])
                    & (test_age_labels < age_range[1])
                )
                val_mask = torch.tensor(val_mask.values, dtype=torch.bool)
                # Filter predictions and targets based on mask
                val_subgroup_preds = val_preds[val_mask]
                val_subgroup_targets = val_targets[val_mask]

                # Compute MSE if there are any samples in the subgroup
                if len(val_subgroup_preds) > 0:
                    mse = F.mse_loss(val_subgroup_preds, val_subgroup_targets)
                    rmse = torch.sqrt(mse)
                    val_rmse_results[subgroup_name] = rmse, len(val_subgroup_preds)
                else:
                    val_rmse_results[subgroup_name] = "No samples in subgroup"

    return train_rmse_results, val_rmse_results

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #Best run passing in sex, mutation and age before encoder
    train_rmse_results_personalized, val_rmse_results_personalized = predict_for_subgroups_with_labels('/scratch/lcornelis/outputs/ray_results/TorchTrainer_2024-08-13_15-49-20/model=gat-v4,seed=31061_269_act=sigmoid,adj_thresh=0.1000,batch_size=8,dropout=0.1000,l1_lambda=0.0008,lr=0.0000,lr_scheduler=Lamb_2024-08-13_16-58-56/checkpoint_000005', device, 2.124088581365514, 0.8733420033790319)
    print("Train RMSE Results Personalized:")
    print(train_rmse_results_personalized)
    print("Val RMSE Results Personalized:")
    print(val_rmse_results_personalized)
    train_rmse_results, val_rmse_results = predict_for_subgroups_with_labels('/scratch/lcornelis/outputs/ray_results/TorchTrainer_2024-08-15_10-15-54/model=gat-v4,seed=4565_459_act=elu,adj_thresh=0.7000,batch_size=32,dropout=0,l1_lambda=0.0644,lr=0.0000,lr_scheduler=ReduceLROnPla_2024-08-15_12-16-06/checkpoint_000000', device, 2.124088581365514, 0.8733420033790319)
    print("Train RMSE Results Not personalized:")
    print(train_rmse_results)
    print("Val RMSE Results Not personalized:")
    print(val_rmse_results)
    loss_difference_val = {
    key: (
        val_rmse_results[key][0].item() - val_rmse_results_personalized[key][0].item()
        if not isinstance(val_rmse_results[key][0], str) and not isinstance(val_rmse_results_personalized[key][0], str)
        else "NA"
    )
    for key in train_rmse_results
}
    print("Loss Difference Val:")
    print(loss_difference_val)
    loss_difference_train = {
    key: (
        train_rmse_results[key][0].item() - train_rmse_results_personalized[key][0].item()
        if not isinstance(train_rmse_results[key][0], str) and not isinstance(train_rmse_results_personalized[key][0], str)
        else "NA"
    )
    for key in train_rmse_results
}
    print("Loss Difference Train:")
    print(loss_difference_train)

if __name__ == "__main__":
    main()