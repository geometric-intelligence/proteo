import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.svm import SVR
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from proteo.proteo.datasets.ftd_folds import FTDDataset
from config_utils import CONFIG_FILE, read_config_from_file
from proteo.train import construct_datasets
import itertools

def load_and_prepare_data(config):
    # Load data using your existing function
    train_dataset, test_dataset = construct_datasets(config)
    train_dataset_features = train_dataset.x
    train_dataset_labels = train_dataset.y
    train_dataset_sex = train_dataset.sex
    train_dataset_mutation = train_dataset.mutation
    train_dataset_age = train_dataset.age
    val_dataset_features = test_dataset.x
    val_dataset_labels = test_dataset.y
    val_dataset_sex = test_dataset.sex
    val_dataset_mutation = test_dataset.mutation
    val_dataset_age = test_dataset.age
    
    return train_dataset_features, train_dataset_labels, train_dataset_sex, train_dataset_mutation, train_dataset_age, val_dataset_features, val_dataset_labels, val_dataset_sex, val_dataset_mutation, val_dataset_age

def stack_features(train_features, train_labels, train_sex, train_mutation, train_age, val_features, val_labels, val_sex, val_mutation, val_age):
    train_combined = np.hstack((train_features, train_sex, train_mutation, train_age))
    val_combined = np.hstack((val_features, val_sex, val_mutation, val_age))
    return train_combined, val_combined, train_labels, val_labels

def find_best_alpha(features, labels):
    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Use LassoCV to find the best alpha
    lasso_cv = LassoCV(cv=5, random_state=42)
    lasso_cv.fit(features_scaled, labels)

    print(f"Best alpha found: {lasso_cv.alpha_}")
    return lasso_cv.alpha_

def run_lasso_folds(config):
    alphas = np.logspace(-4, 2, 50) # Range of alpha values to try
    # Dictionary to store MSE scores for each alpha
    alpha_results = {alpha: [] for alpha in alphas}
    
    for fold in range(config.num_folds):
        # Update config fold number
        config.fold = fold
        
        # Load data for this fold
        train_features, train_labels, train_sex, train_mutation, train_age, \
        val_features, val_labels, val_sex, val_mutation, val_age = load_and_prepare_data(config)
        
        # Stack features
        train_combined, val_combined, train_labels, val_labels = stack_features(
            train_features, train_labels, train_sex, train_mutation, train_age,
            val_features, val_labels, val_sex, val_mutation, val_age
        )
        
        # Try different alpha values
        for alpha in alphas:
            lasso_model = Lasso(alpha=alpha, random_state=42)
            lasso_model.fit(train_combined, train_labels)
            
            # Predict and evaluate
            predictions = lasso_model.predict(val_combined)
            mse = mean_squared_error(val_labels, predictions)
            alpha_results[alpha].append(mse)
        
        print(f"Completed fold {fold}")
    
    # Calculate and print statistics for each alpha
    print("\nResults for each alpha value across folds:")
    print("Alpha\tMean MSE ± Std MSE")
    print("-" * 30)
    
    best_mean_mse = float('inf')
    best_alpha = None
    
    for alpha in alphas:
        mean_mse = np.mean(alpha_results[alpha])
        std_mse = np.std(alpha_results[alpha])
        print(f"{alpha:.3f}\t{mean_mse:.4f} ± {std_mse:.4f}")
        
        if mean_mse < best_mean_mse:
            best_mean_mse = mean_mse
            best_alpha = alpha
    
    print(f"\nBest alpha across all folds: {best_alpha} (MSE: {best_mean_mse:.4f})")
    
    return alpha_results


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
    
    # Dictionary to store MSE scores for each parameter combination
    results = {str(params): [] for params in param_combinations}
    
    for fold in range(config.num_folds):
        # Update config fold number
        config.fold = fold
        
        # Load data for this fold
        train_features, train_labels, train_sex, train_mutation, train_age, \
        val_features, val_labels, val_sex, val_mutation, val_age = load_and_prepare_data(config)
        
        # Stack features
        train_combined, val_combined, train_labels, val_labels = stack_features(
            train_features, train_labels, train_sex, train_mutation, train_age,
            val_features, val_labels, val_sex, val_mutation, val_age
        )
        
        # Try different parameter combinations
        for params in param_combinations:
            svr_model = SVR(**params)
            svr_model.fit(train_combined, train_labels)
            
            # Predict and evaluate
            predictions = svr_model.predict(val_combined)
            mse = mean_squared_error(val_labels, predictions)
            results[str(params)].append(mse)
        
        print(f"Completed fold {fold}")
    
    # Calculate and print statistics for each parameter combination
    print("\nResults for each parameter combination across folds:")
    print("Parameters\tMean MSE ± Std MSE")
    print("-" * 50)
    
    best_mean_mse = float('inf')
    best_params = None
    
    for params_str, mse_scores in results.items():
        mean_mse = np.mean(mse_scores)
        std_mse = np.std(mse_scores)
        print(f"{params_str}\t{mean_mse:.4f} ± {std_mse:.4f}")
        
        if mean_mse < best_mean_mse:
            best_mean_mse = mean_mse
            best_params = eval(params_str)
    
    print(f"\nBest parameters across all folds: {best_params}")
    print(f"Best MSE: {best_mean_mse:.4f}")
    
    return results

def main():
    # Read configuration
    config = read_config_from_file(CONFIG_FILE)
    
    # Run Lasso with k-fold cross validation
    print("Running Lasso...")
    lasso_results = run_lasso_folds(config)
    
    # Run SVR with k-fold cross validation
    print("\nRunning SVR...")
    svr_results = run_svr_folds(config)

if __name__ == "__main__":
    main()