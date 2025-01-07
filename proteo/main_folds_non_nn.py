import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.svm import SVR
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from proteo.proteo.datasets.ftd_folds import FTDDataset
from config_utils import CONFIG_FILE, read_config_from_file

def load_and_prepare_data(config):
    # Load data using your existing function
    dataset = FTDDataset(config.data_dir, "train", 0, config)  # Fold is set to 0 for initial loading
    (
        features,
        labels,
        filtered_data_for_adj,
        top_protein_columns,
        filtered_sex_col,
        filtered_mutation_col,
        filtered_age_col,
        filtered_did_col, 
        filtered_gene_col
    ) = dataset.load_csv_data_pre_pt_files(config)

    # Convert sex and mutation to categorical labels
    sex_labels = np.array(filtered_sex_col.astype('category').cat.codes)
    mutation_labels = np.array(filtered_mutation_col.astype('category').cat.codes)

    # Combine features if using master nodes
    if config.use_master_nodes:
        master_node_dict = {
            'sex': sex_labels.reshape(-1, 1),
            'mutation': mutation_labels.reshape(-1, 1),
            'age': filtered_age_col.values.reshape(-1, 1),
        }
        master_node_features = [
            master_node_dict[feature]
            for feature in config.master_nodes
            if feature in master_node_dict
        ]
        if master_node_features:
            master_node_features = np.concatenate(master_node_features, axis=1)
            features = np.concatenate((features, master_node_features), axis=1)
        print("Using master nodes")
        print("Features shape after master nodes:", features.shape)

    return features, labels, sex_labels, mutation_labels, filtered_age_col.values

def k_fold_split_and_standardize(features, labels, sex_labels, mutation_labels, age_values, k=3):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(features):
        # Split data into train and test sets
        train_features = features[train_index]
        test_features = features[test_index]
        train_labels = labels[train_index]
        test_labels = labels[test_index]
        train_sex = sex_labels[train_index]
        test_sex = sex_labels[test_index]
        train_mutation = mutation_labels[train_index]
        test_mutation = mutation_labels[test_index]
        train_age = age_values[train_index]
        test_age = age_values[test_index]

        # Combine additional features
        train_combined = np.hstack((train_features, train_sex.reshape(-1, 1), train_mutation.reshape(-1, 1), train_age.reshape(-1, 1)))
        test_combined = np.hstack((test_features, test_sex.reshape(-1, 1), test_mutation.reshape(-1, 1), test_age.reshape(-1, 1)))

        # Standardize the features
        scaler = StandardScaler()
        train_combined = scaler.fit_transform(train_combined)
        test_combined = scaler.transform(test_combined)

        yield train_combined, test_combined, train_labels, test_labels

def find_best_alpha(features, labels):
    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Use LassoCV to find the best alpha
    lasso_cv = LassoCV(cv=5, random_state=42)
    lasso_cv.fit(features_scaled, labels)

    print(f"Best alpha found: {lasso_cv.alpha_}")
    return lasso_cv.alpha_

def run_lasso_k_fold(features, labels, sex_labels, mutation_labels, age_values, best_alpha, k=3):
    mse_scores = []

    for train_combined, test_combined, train_labels, test_labels in k_fold_split_and_standardize(features, labels, sex_labels, mutation_labels, age_values, k):
        # Train Lasso regression with the best alpha
        lasso_model = Lasso(alpha=best_alpha)
        lasso_model.fit(train_combined, train_labels)

        # Predict and evaluate
        predictions = lasso_model.predict(test_combined)
        mse = mean_squared_error(test_labels, predictions)
        mse_scores.append(mse)

    return np.mean(mse_scores), np.std(mse_scores)

def find_best_svr_params(features, labels):
    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'epsilon': [0.1, 0.2, 0.5, 1],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }

    # Initialize the SVR model
    svr = SVR()

    # Use GridSearchCV to find the best parameters
    grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(features_scaled, labels)

    print(f"Best SVR parameters: {grid_search.best_params_}")
    return grid_search.best_params_

def run_svr_k_fold(features, labels, sex_labels, mutation_labels, age_values, best_params, k=3):
    mse_scores = []

    for train_combined, test_combined, train_labels, test_labels in k_fold_split_and_standardize(features, labels, sex_labels, mutation_labels, age_values, k):
        # Train SVR with the best parameters
        svr_model = SVR(**best_params)
        svr_model.fit(train_combined, train_labels)

        # Predict and evaluate
        predictions = svr_model.predict(test_combined)
        mse = mean_squared_error(test_labels, predictions)
        mse_scores.append(mse)

    return np.mean(mse_scores), np.std(mse_scores)

def main():
    # Read configuration
    config = read_config_from_file(CONFIG_FILE)

    # Load and prepare data
    features, labels, sex_labels, mutation_labels, age_values = load_and_prepare_data(config)

    # Find the best alpha for Lasso
    best_alpha = find_best_alpha(features, labels)

    # Run Lasso regression with k-fold cross-validation
    mean_mse, std_mse = run_lasso_k_fold(features, labels, sex_labels, mutation_labels, age_values, best_alpha)
    print(f"Lasso Mean MSE: {mean_mse:.2f} ± {std_mse:.2f}")

    # Find the best SVR parameters
    best_svr_params = find_best_svr_params(features, labels)

    # Run SVR with k-fold cross-validation
    mean_mse_svr, std_mse_svr = run_svr_k_fold(features, labels, sex_labels, mutation_labels, age_values, best_svr_params)
    print(f"SVR Mean MSE: {mean_mse_svr:.2f} ± {std_mse_svr:.2f}")

if __name__ == "__main__":
    main()