import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.svm import SVR
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from proteo.datasets.ftd_folds import FTDDataset
from config_utils import CONFIG_FILE, read_config_from_file
from proteo.train import construct_datasets
import itertools
import torch.nn as nn
import os

def load_mean_std_for_fold(config):
    """
    Given the configuration for a fold (which includes fold, seed, etc.),
    construct the JSON filename, load it, and return (mu, sigma).
    
    Expected JSON structure:
      {
        "mean": <float>,
        "std": <float>
      }
    """
    # Extract fields from config
    # (Make sure that your config object has these attributes.)
    y_val = config.y_val
    adj   = config.adj_thresh
    num_nodes = config.num_nodes
    
    # Assume mutation and sex are stored as lists in the config
    mutation = config.mutation
    mutation_str = f"mutation_{','.join(mutation)}"
    
    sex = config.sex
    sex_str = f"sex_{','.join(sex)}"
    
    modality = config.modality
    
    y_val_str = f"y_val_{y_val}"
    adj_str   = f"adj_thresh_{adj}"
    num_nodes_str = f"num_nodes_{num_nodes}"
    
    experiment_id = (
        f"ftd_{y_val_str}_{adj_str}_{num_nodes_str}_{mutation_str}_{modality}_{sex_str}"
    )
    
    split = getattr(config, "split", "train")
    random_state = config.seed
    num_folds = config.num_folds
    fold = config.fold
    
    # Construct the filename (adjust the pattern if needed)
    filename = (
        f"{experiment_id}_{split}_random_state_{random_state}_"
        f"{num_folds}fold_{fold}.json"
    )
    
    # Adjust the directory path as needed
    full_path = os.path.join("/scratch/lcornelis/data/data_louisa/processed", filename)
    
    with open(full_path, "r") as f:
        lines = f.read().splitlines()
    
    # Parse the JSON content manually (assuming the same structure)
    mean_str = lines[0].split(":")[1].strip()
    std_str  = lines[1].split(":")[1].strip()
    
    mean_val = float(mean_str)
    std_val  = float(std_str)
    
    return mean_val, std_val


def load_and_prepare_data(config):
    # Load data using your existing function
    train_dataset, test_dataset = construct_datasets(config)
    train_dataset_features = train_dataset.x.view(train_dataset.y.shape[0], -1)
    train_dataset_labels = train_dataset.y
    train_dataset_sex = train_dataset.sex
    train_dataset_mutation = train_dataset.mutation
    train_dataset_age = train_dataset.age
    val_dataset_features = test_dataset.x.view(test_dataset.y.shape[0], -1)
    val_dataset_labels = test_dataset.y
    val_dataset_sex = test_dataset.sex
    val_dataset_mutation = test_dataset.mutation
    val_dataset_age = test_dataset.age
    
    return train_dataset_features, train_dataset_labels, train_dataset_sex, train_dataset_mutation, train_dataset_age, val_dataset_features, val_dataset_labels, val_dataset_sex, val_dataset_mutation, val_dataset_age

def stack_features(train_features, train_labels, train_sex, train_mutation, train_age, val_features, val_labels, val_sex, val_mutation, val_age):
    target_shape = train_features.shape[1] // 3
    if train_sex.shape[1] < target_shape:
        # Calculate how many times to repeat train_sex along the second axis
        repeat_times = target_shape // train_sex.shape[1]
    
        # Expand train_sex
        train_sex = np.tile(train_sex, (1, repeat_times))
        val_sex = np.tile(val_sex, (1, repeat_times))
        train_mutation = np.tile(train_mutation, (1, repeat_times))
        val_mutation = np.tile(val_mutation, (1, repeat_times))
        train_age = np.tile(train_age, (1, repeat_times))
        val_age = np.tile(val_age, (1, repeat_times))

    train_combined = np.hstack((train_features, train_sex, train_mutation, train_age))
    print(train_combined.shape)
    val_combined = np.hstack((val_features, val_sex, val_mutation, val_age))
    print(val_combined.shape)
    return train_combined, val_combined, train_labels, val_labels


def run_lasso_folds(config):
    alphas = np.logspace(-4, 2, 50)  # Range of alpha values to try
    # Dictionaries to store MSE scores for each alpha
    alpha_results_norm = {alpha: [] for alpha in alphas}
    alpha_results_orig = {alpha: [] for alpha in alphas}
    
    for fold in range(config.num_folds):
        # Update config with the current fold number
        config.fold = fold
        
        # Load data for this fold
        (train_features, train_labels, train_sex, train_mutation, train_age, 
         val_features, val_labels, val_sex, val_mutation, val_age) = load_and_prepare_data(config)
        
        # Stack features
        train_combined, val_combined, train_labels, val_labels = stack_features(
            train_features, train_labels, train_sex, train_mutation, train_age,
            val_features, val_labels, val_sex, val_mutation, val_age
        )
        
        # Load original sigma (and mean, if needed) for this fold
        # We only need sigma for converting MSE (MSE_orig = sigma^2 * MSE_norm)
        _, sigma = load_mean_std_for_fold(config)
        
        # Try different alpha values
        for alpha in alphas:
            lasso_model = Lasso(alpha=alpha, random_state=42)
            lasso_model.fit(train_combined, train_labels)
            
            # Predict and compute MSE in normalized space
            predictions = lasso_model.predict(val_combined)
            mse_norm = mean_squared_error(val_labels, predictions)
            
            # Convert to original scale
            mse_orig = mse_norm * (sigma ** 2)
            
            alpha_results_norm[alpha].append(mse_norm)
            alpha_results_orig[alpha].append(mse_orig)
        
        print(f"Completed fold {fold}")
    
    # Print statistics for each alpha
    print("\nResults for each alpha value across folds (normalized and original):")
    print("Alpha\tMean Norm MSE ± Std Norm MSE\tMean Orig MSE ± Std Orig MSE")
    print("-" * 80)
    
    best_mean_norm_mse = float('inf')
    best_alpha = None
    
    for alpha in alphas:
        mean_norm_mse = np.mean(alpha_results_norm[alpha])
        std_norm_mse = np.std(alpha_results_norm[alpha])
        mean_orig_mse = np.mean(alpha_results_orig[alpha])
        std_orig_mse = np.std(alpha_results_orig[alpha])
        print(f"{alpha:.3f}\t{mean_norm_mse:.4f} ± {std_norm_mse:.4f}\t\t{mean_orig_mse:.4f} ± {std_orig_mse:.4f}")
        
        if mean_norm_mse < best_mean_norm_mse:
            best_mean_norm_mse = mean_norm_mse
            best_alpha = alpha
    
    print(f"\nBest alpha across all folds: {best_alpha} (Normalized MSE: {best_mean_norm_mse:.4f})")
    
    return alpha_results_norm, alpha_results_orig

# --------------------------------------------------
# Similarly, modified run_svr_folds function:
def run_svr_folds(config):
    # Define the parameter grid
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'epsilon': [0.1, 0.2, 0.5, 1],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }
    
    # Create all parameter combinations
    param_combinations = [
        dict(zip(param_grid.keys(), v)) 
        for v in itertools.product(*param_grid.values())
    ]
    
    # Dictionaries to store MSE scores for each parameter combination
    results_norm = {str(params): [] for params in param_combinations}
    results_orig = {str(params): [] for params in param_combinations}
    
    for fold in range(config.num_folds):
        config.fold = fold
        
        # Load data for this fold
        (train_features, train_labels, train_sex, train_mutation, train_age, 
         val_features, val_labels, val_sex, val_mutation, val_age) = load_and_prepare_data(config)
        
        train_combined, val_combined, train_labels, val_labels = stack_features(
            train_features, train_labels, train_sex, train_mutation, train_age,
            val_features, val_labels, val_sex, val_mutation, val_age
        )
        
        # Load sigma for conversion
        _, sigma = load_mean_std_for_fold(config)
        
        for params in param_combinations:
            svr_model = SVR(**params)
            svr_model.fit(train_combined, train_labels)
            
            predictions = svr_model.predict(val_combined)
            mse_norm = mean_squared_error(val_labels, predictions)
            mse_orig = mse_norm * (sigma ** 2)
            
            results_norm[str(params)].append(mse_norm)
            results_orig[str(params)].append(mse_orig)
        
        print(f"Completed fold {fold}")
    
    print("\nResults for each parameter combination across folds (normalized and original):")
    print("Parameters\tMean Norm MSE ± Std Norm MSE\tMean Orig MSE ± Std Orig MSE")
    print("-" * 100)
    
    best_mean_norm_mse = float('inf')
    best_params = None
    
    for params_str, mse_scores in results_norm.items():
        mean_norm_mse = np.mean(mse_scores)
        std_norm_mse = np.std(mse_scores)
        mean_orig_mse = np.mean(results_orig[params_str])
        std_orig_mse = np.std(results_orig[params_str])
        print(f"{params_str}\t{mean_norm_mse:.4f} ± {std_norm_mse:.4f}\t\t{mean_orig_mse:.4f} ± {std_orig_mse:.4f}")
        
        if mean_norm_mse < best_mean_norm_mse:
            best_mean_norm_mse = mean_norm_mse
            best_params = eval(params_str)
    
    print(f"\nBest parameters across all folds: {best_params}")
    print(f"Best normalized MSE: {best_mean_norm_mse:.4f}")
    
    return results_norm, results_orig

def main():
    # Read configuration
    config = read_config_from_file(CONFIG_FILE)
    
    # Run Lasso with k-fold cross validation
    print("Running Lasso...")
    lasso_results = run_lasso_folds(config)
    
    #Run SVR with k-fold cross validation
    print("\nRunning SVR...")
    svr_results = run_svr_folds(config)

if __name__ == "__main__":
    main()