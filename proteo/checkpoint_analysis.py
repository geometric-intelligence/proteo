import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import train as proteo_train
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import json

from proteo.datasets.ftd import FTDDataset, reverse_log_transform

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_checkpoint(relative_checkpoint_path):
    '''Load the checkpoint as a module. Note levels_up depends on the directory structure of the ray_results folder'''
    relative_checkpoint_path = os.path.join(relative_checkpoint_path, 'checkpoint.ckpt')
    # Check if the file exists to avoid errors
    if not os.path.isfile(relative_checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {relative_checkpoint_path}")

    # Load the checkpoint dictionary using PyTorch directly to modify the config
    checkpoint = torch.load(relative_checkpoint_path, map_location=torch.device('cpu'))

    # Ensure that the 'use_master_nodes' attribute is in the checkpoint's config
    if not hasattr(checkpoint['hyper_parameters']['config'], 'kfold') or not hasattr(checkpoint['hyper_parameters']['config'], 'num_folds') or not hasattr(checkpoint['hyper_parameters']['config'], 'fold'):
        checkpoint['hyper_parameters']['config'].kfold = False
        checkpoint['hyper_parameters']['config'].num_folds = 1
        checkpoint['hyper_parameters']['config'].fold = 0
    
    if not hasattr(checkpoint['hyper_parameters']['config'], 'random_state'):
        checkpoint['hyper_parameters']['config'].random_state = 42

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
    print(config)
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
    val_losses = []
    loss_fn = torch.nn.MSELoss(reduction='none')
    for batch in test_loader:
        batch.to(device)
        # Forward pass
        pred = module(batch)
        print(pred.shape)
        target = batch.y.view(pred.shape)
        val_losses.append(loss_fn(pred, target))

        # Store predictions and targets
        val_preds.append(pred.cpu())
        val_targets.append(target.cpu())
    val_preds = torch.cat(val_preds)
    val_targets = torch.cat(val_targets)

    # Calculate MSE for validation set
    val_mse = F.mse_loss(val_preds, val_targets).mean()
    val_mse2 = torch.vstack(val_losses).detach().cpu().mean()
    print("Normalized Val MSE:", val_mse)
    print("Normalized Val MSE2:", val_mse2)
    print("Normalized train MSE:", train_mse)
    return train_preds, train_targets, train_mse, val_preds, val_targets, val_mse

def load_stats_from_json(config):
    #Just make an instance to get mean and std path 
    root = config.data_dir
    dataset = FTDDataset(root, "train", config)
    if config.kfold:
        mean_std_file_name = f"{dataset.experiment_id}_train_random_state_{config.random_state}_{config.num_folds}fold_{config.fold}.json"
    else:
        mean_std_file_name = f"{dataset.experiment_id}_train_random_state_{config.random_state}.json"
    mean_std_file_path = os.path.join(dataset.processed_dir, mean_std_file_name)
    with open(mean_std_file_path, 'r') as f:
        stats = json.load(f)
    
    mean = stats['mean']
    std = stats['std']
    
    return mean, std

def full_load_and_run_and_convert(relative_checkpoint_path, device):
    '''Call all the functions to load the checkpoint, run the model and convert the predictions back to the original units'''
    module = load_checkpoint(relative_checkpoint_path)
    config = load_config(module)

    mean, std = load_stats_from_json(config)


    train_preds, train_targets, train_mse, val_preds, val_targets, val_mse = load_model_and_predict(
        module, config, device
    )
    #return module, config, train_preds, train_targets, train_mse, val_preds, val_targets, val_mse
    #print("Train Preds:")
    #print(train_preds)
    train_preds = reverse_log_transform(train_preds, mean, std)
    train_targets = reverse_log_transform(train_targets, mean, std)
    train_mse = F.mse_loss(train_preds, train_targets)
    train_rmse = torch.sqrt(train_mse)
    val_preds = reverse_log_transform(val_preds, mean, std)
    val_targets = reverse_log_transform(val_targets, mean, std)
    val_mse = F.mse_loss(val_preds, val_targets)
    val_rmse = torch.sqrt(val_mse)
    #print(val_preds.view(-1).detach().cpu().numpy())
    val_z_scores = zscore(
        val_preds.view(-1).detach().cpu().numpy() - val_targets.view(-1).detach().cpu().numpy(),
        ddof=1,
    )
    print("Original Units Train MSE:", train_mse)
    print("Original Units Train RMSE:", train_rmse)
    print("Original Units Val MSE:", val_mse)
    print("Original Units Val RMSE:", val_rmse)
    print("Val Z scores:", val_z_scores)

    # Plot Train and Validation Predictions vs. Targets side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot Train Predictions vs. Targets
    axes[0].scatter(train_targets.cpu().numpy(), train_preds.cpu().numpy(), color='blue', alpha=0.6)
    axes[0].plot([min(train_targets), max(train_targets)], [min(train_targets), max(train_targets)], 'r--', lw=2)
    axes[0].set_title("Train Predictions vs Targets")
    axes[0].set_xlabel("True Train Targets")
    axes[0].set_ylabel("Predicted Train Targets")
    
    # Plot Validation Predictions vs. Targets
    axes[1].scatter(val_targets.cpu().detach().numpy(), val_preds.cpu().detach().numpy(), color='green', alpha=0.6)
    axes[1].plot([min(val_targets), max(val_targets)], [min(val_targets), max(val_targets)], 'r--', lw=2)
    axes[1].set_title("Validation Predictions vs Targets")
    axes[1].set_xlabel("True Validation Targets")
    axes[1].set_ylabel("Predicted Validation Targets")
    
    plt.tight_layout()
    plt.show()

    return [
        train_preds,
        train_targets,
        train_mse,
        train_rmse,
        val_preds,
        val_targets,
        val_mse,
        val_rmse,
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
    random_state = config.random_state
    train_dataset = FTDDataset(root, "train", config)
    (
        _,
        _,
        _,
        _,
        filtered_sex_col,
        filtered_mutation_col,
        filtered_age_col,
        filtered_did_col,
        filtered_gene_col
    ) = train_dataset.load_csv_data_pre_pt_files(config)
    # Splitting indices only
    (
        train_sex_labels,
        test_sex_labels,
        train_mutation_labels,
        test_mutation_labels,
        train_age_labels,
        test_age_labels,
        train_did_labels, 
        test_did_labels,
        train_gene_col,
        test_gene_col
    ) = train_test_split(
        filtered_sex_col, filtered_mutation_col, filtered_age_col, filtered_did_col, filtered_gene_col, test_size=0.20, random_state=random_state
    )
    print("train did labels", train_did_labels)
    print("test did labels", test_did_labels )
    return (
        train_sex_labels,
        test_sex_labels,
        train_mutation_labels,
        test_mutation_labels,
        train_age_labels,
        test_age_labels,
        train_did_labels,
        test_did_labels,
        train_gene_col, 
        test_gene_col
    )


def predict_for_subgroups_with_labels(relative_checkpoint_path, device):
    module = load_checkpoint(relative_checkpoint_path)
    config = load_config(module)
    mean, std = load_stats_from_json(config)
    # Load the model and get the predictions and targets
    train_preds, train_targets, _, val_preds, val_targets, _ = load_model_and_predict(
        module, config, device
    )

    '''# Convert the predictions back to the original units - not for now
    train_preds = reverse_log_transform(train_preds, mean, std)
    train_targets = reverse_log_transform(train_targets, mean, std)
    val_preds = reverse_log_transform(val_preds, mean, std)
    val_targets = reverse_log_transform(val_targets, mean, std)'''

    # Get the group labels for the training and validation sets
    (
        train_sex_labels,
        test_sex_labels,
        train_mutation_labels,
        test_mutation_labels,
        _,  # Removing age labels
        _,
        train_did_labels,
        train_did_labels,
    ) = get_sex_mutation_age_distribution(config)

    # Define subgroups
    sex_labels = ['M', 'F']
    mutation_labels = ["C9orf72", "MAPT", "GRN", "CTL"]

    # Calculate RMSE for each subgroup in training data
    train_rmse_results = {}
    val_rmse_results = {}
    for sex in sex_labels:
        for mutation in mutation_labels:
            subgroup_name = f"{sex}_{mutation}"

            # Create mask for current subgroup
            train_mask = (train_sex_labels == sex) & (train_mutation_labels == mutation)
            train_mask = torch.tensor(train_mask.values, dtype=torch.bool)

            # Filter predictions and targets based on mask
            train_subgroup_preds = train_preds[train_mask]
            train_subgroup_targets = train_targets[train_mask]

            # Compute RMSE if there are any samples in the subgroup
            if len(train_subgroup_preds) > 0:
                mse = F.mse_loss(train_subgroup_preds, train_subgroup_targets)
                rmse = torch.sqrt(mse)
                train_rmse_results[subgroup_name] = rmse, len(train_subgroup_preds)
            else:
                train_rmse_results[subgroup_name] = "No samples in subgroup"

            val_mask = (test_sex_labels == sex) & (test_mutation_labels == mutation)
            val_mask = torch.tensor(val_mask.values, dtype=torch.bool)

            # Filter predictions and targets based on mask
            val_subgroup_preds = val_preds[val_mask]
            val_subgroup_targets = val_targets[val_mask]

            # Compute RMSE if there are any samples in the subgroup
            if len(val_subgroup_preds) > 0:
                mse = F.mse_loss(val_subgroup_preds, val_subgroup_targets)
                rmse = torch.sqrt(mse)
                val_rmse_results[subgroup_name] = rmse, len(val_subgroup_preds)
            else:
                val_rmse_results[subgroup_name] = "No samples in subgroup"

    return train_rmse_results, val_rmse_results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Best run passing in sex, mutation and age before encoder
    (
        train_rmse_results_personalized_m,
        val_rmse_results_personalized_m,
    ) = predict_for_subgroups_with_labels(
        '/scratch/lcornelis/outputs/ray_results/TorchTrainer_2024-08-23_13-20-08/model=gat-v4,seed=44609_31_act=relu,adj_thresh=0.9000,batch_size=50,dropout=0,l1_lambda=0.0104,lr=0.0034,lr_scheduler=ReduceLROnPl_2024-08-23_13-20-08/checkpoint_000006',
        device,
        10,
        10,
    )

    (
        train_rmse_results_personalized_f,
        val_rmse_results_personalized_f,
    ) = predict_for_subgroups_with_labels(
        '/scratch/lcornelis/outputs/ray_results/TorchTrainer_2024-08-13_15-49-20/model=gat-v4,seed=31061_269_act=sigmoid,adj_thresh=0.1000,batch_size=8,dropout=0.1000,l1_lambda=0.0008,lr=0.0000,lr_scheduler=Lamb_2024-08-13_16-58-56/checkpoint_000005',
        device,
        2.124088581365514,
        0.8733420033790319,
    )
    print("Train RMSE Results Personalized:")
    print(train_rmse_results_personalized_f)
    # print("Val RMSE Results Personalized:")
    # print(val_rmse_results_personalized_f)

    (
        train_rmse_results_personalized_m,
        val_rmse_results_personalized_m,
    ) = predict_for_subgroups_with_labels(
        '/scratch/lcornelis/outputs/ray_results/TorchTrainer_2024-08-23_13-20-08/model=gat-v4,seed=44609_31_act=relu,adj_thresh=0.9000,batch_size=50,dropout=0,l1_lambda=0.0104,lr=0.0034,lr_scheduler=ReduceLROnPl_2024-08-23_13-20-08/checkpoint_000006',
        device,
        2.124088581365514,
        0.8733420033790319,
    )
    print("Train RMSE Results Personalized_m:")
    print(train_rmse_results_personalized_m)

    train_rmse_results, val_rmse_results = predict_for_subgroups_with_labels(
        '/scratch/lcornelis/outputs/ray_results/TorchTrainer_2024-08-15_10-15-54/model=gat-v4,seed=4565_459_act=elu,adj_thresh=0.7000,batch_size=32,dropout=0,l1_lambda=0.0644,lr=0.0000,lr_scheduler=ReduceLROnPla_2024-08-15_12-16-06/checkpoint_000000',
        device,
        2.124088581365514,
        0.8733420033790319,
    )
    print("Train RMSE Results Not personalized:")
    print(train_rmse_results)
    # print("Val RMSE Results Not personalized:")
    # print(val_rmse_results)
    loss_difference_val_f = {
        key: (
            val_rmse_results[key][0].item() - val_rmse_results_personalized_f[key][0].item()
            if not isinstance(val_rmse_results[key][0], str)
            and not isinstance(val_rmse_results_personalized_f[key][0], str)
            else "NA"
        )
        for key in train_rmse_results
    }
    # print("Loss Difference Val:")
    # print(loss_difference_val_f)
    loss_difference_train_f = {
        key: (
            train_rmse_results[key][0].item() - train_rmse_results_personalized_f[key][0].item()
            if not isinstance(train_rmse_results[key][0], str)
            and not isinstance(train_rmse_results_personalized_f[key][0], str)
            else "NA"
        )
        for key in train_rmse_results
    }
    print("Loss Difference Train for fully connected:")
    print(loss_difference_train_f)

    loss_difference_train_m = {
        key: (
            train_rmse_results[key][0].item() - train_rmse_results_personalized_m[key][0].item()
            if not isinstance(train_rmse_results[key][0], str)
            and not isinstance(train_rmse_results_personalized_m[key][0], str)
            else "NA"
        )
        for key in train_rmse_results
    }
    print("Loss Difference Train for master nodes:")
    print(loss_difference_train_m)


if __name__ == "__main__":
    main()
