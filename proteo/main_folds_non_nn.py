import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.svm import SVR
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
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
        protein_columns,
        filtered_sex_col,
        filtered_mutation_col,
        filtered_age_col,
        filtered_did_col, 
        filtered_gene_col
    ) = dataset.load_csv_data_pre_pt_files(config)

    # Convert sex and mutation to categorical labels
    sex_labels = np.array(filtered_sex_col.astype('category').cat.codes)
    mutation_labels = np.array(filtered_mutation_col.astype('category').cat.codes)

    # Split data into train_val and test sets
    train_val_features, test_features, train_val_labels, test_labels, train_val_sex, test_sex, train_val_mutation, test_mutation, train_val_age, test_age = train_test_split(
        features, labels, sex_labels, mutation_labels, filtered_age_col.values, val_size=0.2, random_state=config.random_state
    )

    return train_val_features, train_val_labels, train_val_sex, train_val_mutation, train_val_age.values

def k_fold_split_and_standardize(features, labels, sex_labels, mutation_labels, age_values, config):
    kf = KFold(n_splits=config.num_folds, shuffle=True, random_state=config.random_state)
    for train_index, val_index in kf.split(features):
        # Split data into train and val sets
        train_features = features[train_index]
        val_features = features[val_index]
        train_labels = labels[train_index]
        val_labels = labels[val_index]
        train_sex = sex_labels[train_index]
        val_sex = sex_labels[val_index]
        train_mutation = mutation_labels[train_index]
        val_mutation = mutation_labels[val_index]
        train_age = age_values[train_index]
        val_age = age_values[val_index]

        # Standardize the features
        scaler_features = StandardScaler()
        train_features = scaler_features.fit_transform(train_features)
        val_features = scaler_features.transform(val_features)

        # Standardize the sex labels
        scaler_sex = StandardScaler()
        train_sex = scaler_sex.fit_transform(train_sex.reshape(-1, 1))
        val_sex = scaler_sex.transform(val_sex.reshape(-1, 1))

        # Standardize the mutation labels
        scaler_mutation = StandardScaler()
        train_mutation = scaler_mutation.fit_transform(train_mutation.reshape(-1, 1))
        val_mutation = scaler_mutation.transform(val_mutation.reshape(-1, 1))

        # Standardize the age values
        scaler_age = StandardScaler()
        train_age = scaler_age.fit_transform(train_age.reshape(-1, 1))
        val_age = scaler_age.transform(val_age.reshape(-1, 1))

        # Combine additional features
        train_combined = np.hstack((train_features, train_sex, train_mutation, train_age))
        val_combined = np.hstack((val_features, val_sex, val_mutation, val_age))

        yield train_combined, val_combined, train_labels, val_labels

def find_best_alpha(features, labels):
    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Use LassoCV to find the best alpha
    lasso_cv = LassoCV(cv=5, random_state=42)
    lasso_cv.fit(features_scaled, labels)

    print(f"Best alpha found: {lasso_cv.alpha_}")
    return lasso_cv.alpha_

def run_lasso_k_fold(features, labels, sex_labels, mutation_labels, age_values, best_alpha, config):
    mse_scores = []

    for train_combined, val_combined, train_labels, val_labels in k_fold_split_and_standardize(features, labels, sex_labels, mutation_labels, age_values, config):
        # Train Lasso regression with the best alpha
        lasso_model = Lasso(alpha=best_alpha)
        lasso_model.fit(train_combined, train_labels)

        # Predict and evaluate
        predictions = lasso_model.predict(val_combined)
        mse = mean_squared_error(val_labels, predictions)
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

def run_svr_k_fold(features, labels, sex_labels, mutation_labels, age_values, best_params, config):
    mse_scores = []

    for train_combined, val_combined, train_labels, val_labels in k_fold_split_and_standardize(features, labels, sex_labels, mutation_labels, age_values, config):
        # Train SVR with the best parameters
        svr_model = SVR(**best_params)
        svr_model.fit(train_combined, train_labels)

        # Predict and evaluate
        predictions = svr_model.predict(val_combined)
        mse = mean_squared_error(val_labels, predictions)
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
    mean_mse, std_mse = run_lasso_k_fold(features, labels, sex_labels, mutation_labels, age_values, best_alpha, config)
    print(f"Lasso Mean MSE: {mean_mse:.2f} ± {std_mse:.2f}")

    # Find the best SVR parameters
    best_svr_params = find_best_svr_params(features, labels)

    # Run SVR with k-fold cross-validation
    mean_mse_svr, std_mse_svr = run_svr_k_fold(features, labels, sex_labels, mutation_labels, age_values, best_svr_params, config)
    print(f"SVR Mean MSE: {mean_mse_svr:.2f} ± {std_mse_svr:.2f}")

if __name__ == "__main__":
    main()