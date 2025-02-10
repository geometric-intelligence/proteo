import os
import torch
import torch_geometric
from torch_geometric.explain import Explainer, CaptumExplainer
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict, Counter
import re
# Imports for dimensionality reduction and clustering
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
# Custom imports from proteo
from proteo.datasets.ftd import FTDDataset, remove_erroneous_columns
import train as proteo_train
import plotly.graph_objs as go
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler

#############FUNCTIONS TO GET EXPLANATIONS AND COMPILE RESULTS####################
def load_config(module):
    '''Load the config from the module  and return it'''
    config = module.config
    # Check if 'use_master_nodes' attribute exists, added for runs with master nodes
    if not hasattr(config, 'use_master_nodes'):
        # Add the attribute with a default value (e.g., False)
        setattr(config, 'use_master_nodes', False)
    return config

#Load model checkpoint - Note when the wrapper class is not necessary you can use this function from checkpoint_analysis.py
def load_checkpoint(relative_checkpoint_path):
    '''Load the checkpoint as a module. Note levels_up depends on the directory structure of the ray_results folder'''
    # Construct the full path to the checkpoint
    checkpoint_path = os.path.join(relative_checkpoint_path, 'checkpoint.ckpt')
    print("checkpoint_path", checkpoint_path)

    # Check if the file exists to avoid errors
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load the checkpoint dictionary using PyTorch directly to modify the config
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Ensure that the 'use_master_nodes' attribute is in the checkpoint's config
    if not hasattr(checkpoint['hyper_parameters']['config'], 'use_master_nodes') or not hasattr(checkpoint['hyper_parameters']['config'], 'master_nodes'):
        checkpoint['hyper_parameters']['config'].use_master_nodes = False  # Add default value
        checkpoint['hyper_parameters']['config'].master_nodes = []
    
    if not hasattr(checkpoint['hyper_parameters']['config'], 'random_state'):
        checkpoint['hyper_parameters']['config'].random_state = 42
        
    torch.save(checkpoint, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    module = proteo_train.Proteo.load_from_checkpoint(checkpoint_path)


    original_forward = module.model.forward
    # Redefine the forward method of module.model to return only pred
    def new_forward(self, x, edge_index=None, data=None):
        pred, _ = original_forward(x, edge_index, data)
        return pred
    
    module.model.forward = new_forward.__get__(module.model)
    return module


def get_explainer_baseline(config):
    '''Function to get the baseline normal expression data to use for explainer results'''
    # Get the Scaler we use to transform all the data
    root = config.data_dir
    random_state = config.random_state
    train_dataset = FTDDataset(root, "train", config)
    features, _, _, top_protein_columns, _, _, _, _, _ = train_dataset.load_csv_data_pre_pt_files(config)
    train_features, test_features = train_test_split(features, test_size=0.20, random_state=random_state)
    combined_features = np.concatenate((train_features, test_features), axis=0)
    scaler = StandardScaler()
    scaler.fit(train_features)

    #Get the CTL Data to transform and get mean
    csv_data = pd.read_csv(train_dataset.raw_paths[0])
    csv_data = remove_erroneous_columns(config, csv_data, train_dataset.raw_dir)
    condition_ctl = csv_data["Mutation"].isin(["CTL"])
    ctl_data = csv_data[condition_ctl]

    # Extract relevant protein columns and scale
    proteins_ctl = ctl_data[top_protein_columns].dropna()
    proteins_ctl_scaled = scaler.transform(proteins_ctl)
    # Calculate the mean for each protein
    baseline_mean = proteins_ctl_scaled.mean(axis=0)
    return baseline_mean, combined_features



#Load protein ids
def get_protein_ids(config):
    # Make an instance of the FTDDataset class to get the top proteins
    root = config.data_dir
    train_dataset = FTDDataset(root, "train", config)
    _, _, _, top_protein_columns, _, _, _, _, _ = train_dataset.load_csv_data_pre_pt_files(config)
    top_protein_columns.extend(['sex', 'mutation', 'age'])
    return np.array(top_protein_columns)

def get_sex_mutation_age_distribution(config):
    # Make an instance of the FTDDataset class to get the top proteins
    root = config.data_dir
    random_state = config.random_state
    train_dataset = FTDDataset(root, "train", config)
    _, _, _, _, filtered_sex_col, filtered_mutation_col, filtered_age_col, filtered_did_col, filtered_gene_col= train_dataset.load_csv_data_pre_pt_files(config)
    # Splitting indices only
    train_sex_labels, test_sex_labels, train_mutation_labels, test_mutation_labels, train_age_labels, test_age_labels, train_did_labels, test_did_labels, train_gene_col, test_gene_col = train_test_split(filtered_sex_col, filtered_mutation_col, filtered_age_col, filtered_did_col, filtered_gene_col, test_size=0.20, random_state=random_state)
    total_sex_labels = np.concatenate((train_sex_labels, test_sex_labels))
    total_mutation_labels = np.concatenate((train_mutation_labels, test_mutation_labels))
    total_age_labels = np.concatenate((train_age_labels, test_age_labels))
    total_did_labels = np.concatenate((train_did_labels, test_did_labels))
    total_gene_labels = np.concatenate((train_gene_col, test_gene_col))
    return total_sex_labels, total_mutation_labels, total_age_labels, total_did_labels, train_did_labels, test_did_labels, total_gene_labels

# Helper function to plot importance values
def plot_importance_scores(explanations, labels, filename, title, ylabel):
    plt.figure()  # Create a new figure for each plot
    for i, importance in enumerate(explanations):
        # Sort the importance scores in descending order (by magnitude) for plotting
        plt.plot(sorted(importance, reverse=True), label=f'Person {i}')
    
    # Add legend and labels
    plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=1)
    plt.xlabel('Protein')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Save the plot if needed
    plot_filename = os.path.join('explainer_plots', filename)
    # plt.savefig(plot_filename)  # Uncomment this to save the plot
    plt.show()

# Main explainer function
def run_explainer_single_dataset(dataset, explainer, protein_ids, dids, filename):
    n_people = len(dataset)
    n_nodes = len(dataset[0].x)
    print(f"n_people = {n_people}, n_nodes = {n_nodes}")
    
    sum_node_importance_raw = {protein_id: 0 for protein_id in protein_ids}
    sum_node_importance_percent = {protein_id: 0 for protein_id in protein_ids}
    positive_percent_by_protein = {protein_id: 0 for protein_id in protein_ids}
    negative_percent_by_protein = {protein_id: 0 for protein_id in protein_ids}
    all_raw_importances = []
    all_percent_importances = []
    all_top_proteins = []
    all_labels = []

    
    for i, data in enumerate(dataset):
        # Run the explainer
        explanation = explainer(data.x, data.edge_index, data=data, target=None, index=None)

        # Flatten and store node importance
        node_importance = np.array(explanation.node_mask.cpu().detach().numpy()).flatten().tolist()
        # Check if node_importance is all zeros
        if all(importance == 0 for importance in node_importance):
            print(f"Warning: Person {i} has a node importance list with all zeros.")

        all_raw_importances.append(node_importance)

        # Convert raw scores to percentage importance (preserving the sign)
        total_importance = np.sum(np.abs(node_importance))  # Sum of original importance (with signs)
        importance_percentages = (node_importance / total_importance) * 100 
        all_percent_importances.append(importance_percentages)

        # Finding percentage of negative and positive
        node_importance = np.array(node_importance)
        total_positive_importance = np.sum(node_importance[node_importance > 0])
        total_negative_importance = np.sum(np.abs(node_importance[node_importance < 0]))

        # Sort indices by raw importance
        sorted_indices = np.argsort(node_importance)[::-1]  # Sort by original value

        # Get the top 5 proteins by importance
        top_5_proteins = [protein_ids[idx] for idx in sorted_indices[:5]]
        all_top_proteins.append([protein_ids[idx] for idx in sorted_indices])

        # Add label for legend
        dids = dids.reset_index(drop=True)
        patient_id = dids[i] if i < len(dids) else f"unknown_{i}"
        all_labels.append(f'Top 5 for person {patient_id}: {", ".join(top_5_proteins)}')

        # Update cumulative importance for each protein
        for idx, importance in enumerate(node_importance):
            sum_node_importance_raw[protein_ids[idx]] += importance
            sum_node_importance_percent[protein_ids[idx]] += importance_percentages[idx]
            if importance > 0:
                positive_percent_by_protein[protein_ids[idx]] = (importance / total_positive_importance) * 100 if total_positive_importance != 0 else 0
            elif importance < 0:    
                negative_percent_by_protein[protein_ids[idx]] = (np.abs(importance) / total_negative_importance) * 100 if total_negative_importance != 0 else 0

    # Plot raw importance scores
    plot_importance_scores(all_raw_importances, all_labels, f'{filename}_raw.png',
                           'Sorted Raw Node Importance with Top 5 Protein IDs', 'Importance')

    # Plot percentage importance scores with signs preserved
    plot_importance_scores(all_percent_importances, all_labels, f'{filename}_percent.png',
                           'Sorted Node Importance as Percentage of Total with Top 5 Protein IDs', 'Importance (%)')

    # Count important proteins across all individuals
    all_important_proteins = [protein for top_proteins in all_top_proteins for protein in top_proteins]
    protein_count = Counter(all_important_proteins)

    # Return both raw and percent importance scores, along with other info
    return sum_node_importance_raw, sum_node_importance_percent, positive_percent_by_protein, negative_percent_by_protein, protein_count, all_raw_importances, all_percent_importances, all_top_proteins

def run_explainer_train_and_test(checkpoint_path):
    module = load_checkpoint(checkpoint_path)
    config = load_config(module)
    print(config)
    # Load datasets and labels
    total_sex_labels, total_mutation_labels, total_age_labels, total_did_labels, train_did, test_did, total_gene_col= get_sex_mutation_age_distribution(config)
    train_dataset, test_dataset = proteo_train.construct_datasets(config)
    print("dim train_dataset", len(train_dataset))
    print("dim test_dataset", len(test_dataset))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset.to(device)
    test_dataset.to(device)

    baseline_mean, features = get_explainer_baseline(config)
    baseline_mean_tensor = torch.tensor(baseline_mean, dtype=torch.float32).to(device)
    baseline_mean_tensor = baseline_mean_tensor.unsqueeze(0).unsqueeze(2)
    # Construct Explainer and set parameters
    explainer = Explainer(
        model=module.model.to(device),
        algorithm=CaptumExplainer('IntegratedGradients'), #, baselines=baseline_mean_tensor),
        explanation_type='model',
        model_config=dict(
            mode='regression',
            task_level='graph',  # Explain why the model predicts a certain property or label for the entire graph (nodes + edges)
            return_type='raw'
        ),
        node_mask_type= 'attributes', #'object', # Generate masks that indicate the importance of individual node features
        edge_mask_type=None,
        threshold_config=dict( #keep only the top 7258 most important proteins and set the rest to 0
            threshold_type='topk',
            value=7258,
        ),
    )
    protein_ids = get_protein_ids(config)

    pattern = r'/(model=.*?)(/checkpoint_\d+)?$'
    model_name = re.search(pattern, checkpoint_path).group(1)
    
    # Run explainer for both training and testing datasets
    train_sum_node_importance_raw, train_sum_node_importance_percent, train_positive_percent, train_negative_percent, train_protein_count, all_raw_importances_train, all_percent_importances_train, all_top_proteins_train = run_explainer_single_dataset(train_dataset, explainer, protein_ids, train_did,  filename=model_name + "_train.png")
    test_sum_node_importance_raw, test_sum_node_importance_percent,  test_positive_percent, test_negative_percent, test_protein_count, all_raw_importances_test, all_percent_importances_test, all_top_proteins_test = run_explainer_single_dataset(test_dataset, explainer, protein_ids, test_did, filename=model_name + "_test.png")

    # Combine sum_node_importance for train and test datasets
    combined_sum_node_importance_raw = {key: train_sum_node_importance_raw.get(key, 0) + test_sum_node_importance_raw.get(key, 0)
                                    for key in set(train_sum_node_importance_raw) | set(test_sum_node_importance_raw)}
    
    combined_sum_node_importance_percent = {key: train_sum_node_importance_percent.get(key, 0) + test_sum_node_importance_percent.get(key, 0)
                                    for key in set(train_sum_node_importance_percent) | set(test_sum_node_importance_percent)}
    
    combined_sum_node_importance_positive = {key: train_positive_percent.get(key, 0) + test_positive_percent.get(key, 0)
                                    for key in set(train_positive_percent) | set(test_positive_percent)}
    
    combined_sum_node_importance_negative = {key: train_negative_percent.get(key, 0) + test_negative_percent.get(key, 0)
                                    for key in set(train_negative_percent) | set(test_negative_percent)}

    
    # Combine protein_count for train and test datasets
    combined_protein_count = train_protein_count + test_protein_count

    # Combine raw and percent importance scores
    all_raw_importances = all_raw_importances_train + all_raw_importances_test
    all_percent_importances = all_percent_importances_train + all_percent_importances_test
    print("all_explanations shape (raw importance):", np.array(all_raw_importances).shape)
    print("all_explanations shape (percent importance):", np.array(all_percent_importances).shape)

    # Save to a csv with model name appended to the filename
    model_name_suffix = f"_{model_name}.csv"

    df_percent_importances = pd.DataFrame(all_percent_importances, index=total_did_labels, columns=protein_ids[0:np.array(all_raw_importances).shape[1]])
    df_percent_importances.insert(0, "SEX", total_sex_labels)
    df_percent_importances.insert(1, "AGE", total_age_labels)
    df_percent_importances.insert(2, "Mutation", total_mutation_labels)
    df_percent_importances.insert(3, "Gene.Dx", total_gene_col)
    df_percent_importances.to_csv(f"percent_importances{model_name_suffix}")

    df_raw_importances = pd.DataFrame(all_raw_importances, index=total_did_labels, columns=protein_ids[0:np.array(all_raw_importances).shape[1]])
    df_raw_importances.insert(0, "SEX", total_sex_labels)
    df_raw_importances.insert(1, "AGE", total_age_labels)
    df_raw_importances.insert(2, "Mutation", total_mutation_labels)
    df_raw_importances.insert(3, "Gene.Dx", total_gene_col)
    df_raw_importances.to_csv(f"raw_importances{model_name_suffix}")

    df_raw_expression = pd.DataFrame(features, index=total_did_labels, columns=protein_ids[0:np.array(all_raw_importances).shape[1]])
    df_raw_expression.insert(0, "SEX", total_sex_labels)
    df_raw_expression.insert(1, "AGE", total_age_labels)
    df_raw_expression.insert(2, "Mutation", total_mutation_labels)
    df_raw_expression.insert(3, "Gene.Dx", total_gene_col)
    df_raw_expression.to_csv("raw_expression.csv")


    # Combine top proteins for both train and test datasets
    all_top_proteins = all_top_proteins_train + all_top_proteins_test
    
    return combined_sum_node_importance_raw, combined_sum_node_importance_percent, combined_sum_node_importance_positive, combined_sum_node_importance_negative, combined_protein_count, config, all_raw_importances, all_percent_importances, all_top_proteins, protein_ids

##############FUNCTIONS TO PLOT AND VISUALIZE RESULTS############################
### CLUSTERING FUNCTIONS ###
def get_sex_per_cluster(clusters, config):
    participant_sex, _, _, _, _, _, _= get_sex_mutation_age_distribution(config)
    # Group sex by clusters
    cluster_sex = defaultdict(list)
    for i, cluster in enumerate(clusters):
        cluster_sex[cluster].append(participant_sex[i])
    
    return cluster_sex

def create_protein_matrix(participant_top_proteins, all_proteins):
    num_participants = len(participant_top_proteins)
    num_proteins = len(all_proteins)
    
    # Create an empty matrix of zeros
    protein_matrix = np.zeros((num_participants, num_proteins), dtype=int)
    
    # Fill the matrix
    for i, top_proteins in enumerate(participant_top_proteins):
        for protein in top_proteins:
            protein_indices = np.where(all_proteins == protein)[0]
            if protein_indices.size > 0:
                protein_index = protein_indices[0]
                protein_matrix[i, protein_index] = 1

    return protein_matrix

def visualize_sex_distribution(cluster_sex_dict):
    # Convert the dictionary into a DataFrame
    data = []
    for cluster, sex_list in cluster_sex_dict.items():
        for sex in sex_list:
            data.append({'Cluster': f"Cluster {cluster}", 'Sex': sex})
    
    df = pd.DataFrame(data)
    
    # Create a bar plot
    sns.countplot(x='Cluster', hue='Sex', data=df)
    plt.title("Sex Distribution in Clusters")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.show()

def cluster_protein_matrix(all_top_proteins, protein_ids, config, num_clusters=2):
    # Apply KMeans clustering
    protein_matrix = create_protein_matrix(all_top_proteins, protein_ids)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(protein_matrix)
    

    sex_clusters = get_sex_per_cluster(clusters, config)
    visualize_sex_distribution(sex_clusters)
    print("clusters", clusters)
    print("sex clusters", sex_clusters)
    return clusters, sex_clusters


### PCA PLOTTING FUNCTIONS ###
def perform_pca(all_explanations, n_components=10):
    """Performs PCA and returns the reduced vectors, explained variance, and loadings for the specified number of components."""
    pca = PCA(n_components=n_components)
    reduced_vectors = pca.fit_transform(all_explanations)
    explained_variance = pca.explained_variance_ratio_
    loadings = pca.components_
    return reduced_vectors, explained_variance, loadings

def plot_pca_3d(reduced_vectors, labels, explained_variance, title):
    labels = np.where(labels == 'F', 'pink', 'blue')
    """Plots a 3D scatter plot for the first 3 principal components."""
    fig = go.Figure(data=[go.Scatter3d(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        z=reduced_vectors[:, 2],
        mode='markers',
        marker=dict(size=5, color=labels, opacity=0.6)
    )])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f'PC1 ({explained_variance[0]*100:.2f}%)',
            yaxis_title=f'PC2 ({explained_variance[1]*100:.2f}%)',
            zaxis_title=f'PC3 ({explained_variance[2]*100:.2f}%)'
        )
    )
    fig.show()

def plot_pca_all_components_2d(reduced_vectors, explained_variance, labels_sex, labels_mutation, labels_age):
    """Plots PCA components 1 vs 2 and 2 vs 3 colored by sex, mutation, and age, scaled by variance."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    labels_sex = np.where(labels_sex == 'F', 'pink', 'blue')

    mutation_color_map = {
        'C9orf72': 'red',
        'MAPT': 'green',
        'GRN': 'blue',
        'CTL': 'purple'
    }

    # Map mutation labels to colors
    labels_mutation = np.array([mutation_color_map[label] for label in labels_mutation])

    # Determine scale factors for the aspect ratio based on explained variance
    scale_pc1_pc2 = np.sqrt(explained_variance[1]) / np.sqrt(explained_variance[0])
    scale_pc2_pc3 = np.sqrt(explained_variance[2]) / np.sqrt(explained_variance[1])

    # PC 1 vs PC 2, colored by Sex
    axes[0, 0].scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=labels_sex)
    axes[0, 0].set_title("PC1 vs PC2 Colored by Sex")
    axes[0, 0].set_xlabel(f"PC1 ({explained_variance[0]:.2%} variance)")
    axes[0, 0].set_ylabel(f"PC2 ({explained_variance[1]:.2%} variance)")
    axes[0, 0].set_aspect(scale_pc1_pc2)  # Set aspect ratio scaled by variance

    # PC 2 vs PC 3, colored by Sex
    axes[0, 1].scatter(reduced_vectors[:, 1], reduced_vectors[:, 2], c=labels_sex)
    axes[0, 1].set_title("PC2 vs PC3 Colored by Sex")
    axes[0, 1].set_xlabel(f"PC2 ({explained_variance[1]:.2%} variance)")
    axes[0, 1].set_ylabel(f"PC3 ({explained_variance[2]:.2%} variance)")
    axes[0, 1].set_aspect(scale_pc2_pc3)  # Set aspect ratio scaled by variance

    # PC 1 vs PC 2, colored by Mutation
    scatter_mutation_1 = axes[1, 0].scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=labels_mutation)
    axes[1, 0].set_title("PC1 vs PC2 Colored by Mutation")
    axes[1, 0].set_xlabel(f"PC1 ({explained_variance[0]:.2%} variance)")
    axes[1, 0].set_ylabel(f"PC2 ({explained_variance[1]:.2%} variance)")
    axes[1, 0].set_aspect(scale_pc1_pc2)  # Set aspect ratio scaled by variance

    # PC 2 vs PC 3, colored by Mutation
    scatter_mutation_2 = axes[1, 1].scatter(reduced_vectors[:, 1], reduced_vectors[:, 2], c=labels_mutation)
    axes[1, 1].set_title("PC2 vs PC3 Colored by Mutation")
    axes[1, 1].set_xlabel(f"PC2 ({explained_variance[1]:.2%} variance)")
    axes[1, 1].set_ylabel(f"PC3 ({explained_variance[2]:.2%} variance)")
    axes[1, 1].set_aspect(scale_pc2_pc3)  # Set aspect ratio scaled by variance

    # Create the mutation legend
    mutation_legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) for label, color in mutation_color_map.items()]
    axes[1, 0].legend(handles=mutation_legend_elements, title="Mutation", loc='upper right')

    # PC 1 vs PC 2, colored by Age
    scatter_age_1 = axes[2, 0].scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=labels_age, cmap='viridis')
    axes[2, 0].set_title("PC1 vs PC2 Colored by Age")
    axes[2, 0].set_xlabel(f"PC1 ({explained_variance[0]:.2%} variance)")
    axes[2, 0].set_ylabel(f"PC2 ({explained_variance[1]:.2%} variance)")
    axes[2, 0].set_aspect(scale_pc1_pc2)  # Set aspect ratio scaled by variance

    # PC 2 vs PC 3, colored by Age
    scatter_age_2 = axes[2, 1].scatter(reduced_vectors[:, 1], reduced_vectors[:, 2], c=labels_age, cmap='viridis')
    axes[2, 1].set_title("PC2 vs PC3 Colored by Age")
    axes[2, 1].set_xlabel(f"PC2 ({explained_variance[1]:.2%} variance)")
    axes[2, 1].set_ylabel(f"PC3 ({explained_variance[2]:.2%} variance)")
    axes[2, 1].set_aspect(scale_pc2_pc3)  # Set aspect ratio scaled by variance

    # Add colorbar for Age on the last plot
    plt.colorbar(scatter_age_2, ax=axes[2, 1], label='Age')

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

def plot_variance_percentage(explained_variance):
    """Plots the explained variance percentage for each principal component."""
    plt.figure(figsize=(10, 6))
    components = np.arange(1, len(explained_variance) + 1)
    plt.bar(components, explained_variance * 100)
    plt.title('Explained Variance by Principal Component')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance (%)')
    plt.xticks(components)
    plt.grid(True)
    plt.show()

def plot_pca_loadings_line(loadings, protein_ids, threshold=0.04):
    """Plots the PCA loadings, filtered by a threshold, for the first two components."""
    plt.figure(figsize=(20, 6))

    # Plot loadings for PC1 and PC2
    for i, loading in enumerate(loadings[:2]):
        filtered_indices = [index for index, value in enumerate(loading) if abs(value) > threshold]
        filtered_values = [value for value in loading if abs(value) > threshold]
        filtered_names = [protein_ids[index] for index in filtered_indices]

        plt.bar(filtered_names, filtered_values, width=0.5, label=f'PC{i+1}', alpha=0.7)
    
    plt.title('PCA Loadings for PC1 and PC2')
    plt.xlabel('Protein ID')
    plt.ylabel('Loading')
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

### General Plotting Functions ###


def plot_bar_chart(protein_dict, title, x_label, y_label, filename=None, top_n=100):
    """Creates two bar charts: one for the top N highest values and one for the top N lowest values from a dictionary."""
    
    # Sort the dictionary by values in descending order for highest and ascending order for lowest
    sorted_items_desc = dict(sorted(protein_dict.items(), key=lambda item: item[1], reverse=True))
    sorted_items_asc = dict(sorted(protein_dict.items(), key=lambda item: item[1]))

    # Get the top N highest and lowest items
    top_highest = dict(list(sorted_items_desc.items())[:top_n])
    top_lowest = dict(list(sorted_items_asc.items())[:top_n])
    
    # Plot top N highest values
    x_highest = list(top_highest.keys())
    y_highest = list(top_highest.values())

    plt.figure(figsize=(28, 10), dpi=300)
    bar_width = 0.6
    plt.bar(x_highest, y_highest, color='lightcoral', width=bar_width)
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.title(f"Top {top_n} Highest - {title}", fontsize=24)
    plt.xticks(rotation=90, ha='right', fontsize=12)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    if filename:
        plt.savefig(f"{filename}_highest.png")
    plt.show()
    
    # Plot top N lowest values
    x_lowest = list(top_lowest.keys())
    y_lowest = list(top_lowest.values())

    plt.figure(figsize=(28, 10), dpi=300)
    plt.bar(x_lowest, y_lowest, color='skyblue', width=bar_width)
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.title(f"Top {top_n} Lowest - {title}", fontsize=24)
    plt.xticks(rotation=90, ha='right', fontsize=12)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    if filename:
        plt.savefig(f"{filename}_lowest.png")
    plt.show()

def plot_top_bar_chart(protein_dict, title, x_label, y_label, filename=None, top_n=100):
    """Creates a bar chart for the top N highest values from a dictionary."""

    # Sort the dictionary by values in descending order for the highest values
    sorted_items_desc = dict(sorted(protein_dict.items(), key=lambda item: item[1], reverse=True))

    # Get the top N highest items
    top_highest = dict(list(sorted_items_desc.items())[:top_n])

    # Extract keys and values for plotting
    x_highest = list(top_highest.keys())
    y_highest = list(top_highest.values())

    # Plot top N highest values
    plt.figure(figsize=(28, 10))
    bar_width = 0.6
    plt.bar(x_highest, y_highest, color='skyblue', width=bar_width)
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.title(f"Top {top_n} - {title}", fontsize=24)
    plt.xticks(rotation=90, ha='right', fontsize=12)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    if filename:
        plt.savefig(f"{filename}_top_{top_n}.png")
    plt.show()

def divide_dict_values(dict1, dict2):
    """Divides values of dict2 by dict1 for matching keys, ensuring both values are numerical."""
    result = {}
    for key in dict1:
        if key in dict2:
            try:
                # Ensure both values are numbers before dividing
                if isinstance(dict1[key], (int, float)) and isinstance(dict2[key], (int, float)):
                    if dict1[key] != 0:  # Avoid division by zero
                        result[key] = dict2[key] / dict1[key]
                    else:
                        print("I am in dict1 0 loop", dict1[key])
                        result[key] = None  # Handle division by zero
                else:
                    print("I am in non-numeric 0 loop", dict1[key])
                    print(type(dict1[key]))
                    print(type(dict2[key]))
                    result[key] = None  # Handle non-numeric values

            except TypeError:
                print("I am in type error loop", key)
                result[key] = None  # Handle potential type errors gracefully

    return result
### Comparison Function ###

def plot_importance_comparison_men_vs_women(all_explanations_percent, protein_ids, config, exclude_ctl=False):
    """Compares feature importance between men and women."""
    total_sex_labels, total_mutation_labels, _, _, _, _, _ = get_sex_mutation_age_distribution(config)
    all_explanations_percent = np.array(all_explanations_percent)


    # Optionally exclude individuals with mutation = 'CTL'
    if exclude_ctl:
        ctl_indices = np.where(total_mutation_labels == 'CTL')[0]
        keep_indices = np.setdiff1d(np.arange(len(total_mutation_labels)), ctl_indices)
        total_sex_labels = total_sex_labels[keep_indices]
        all_explanations_percent = all_explanations_percent[keep_indices]
    
    # Get indices for men and women
    men_indices = np.where(total_sex_labels == 'M')[0]
    women_indices = np.where(total_sex_labels == 'F')[0]

    # Calculate mean importance scores (percentages) for men and women
    mean_importance_men = np.mean(all_explanations_percent[men_indices], axis=0)
    mean_importance_women = np.mean(all_explanations_percent[women_indices], axis=0)

    # Scatter plot to compare men vs women
    plt.figure(figsize=(8, 8))
    plt.scatter(mean_importance_men, mean_importance_women, alpha=0.6, color='gray')
    min_len = min(len(mean_importance_men), len(mean_importance_women), len(protein_ids))
    # Define threshold for labeling
    threshold = 0.05 * max(max(mean_importance_men), max(mean_importance_women))
    # Annotate proteins
    for i in range(min_len):
        # Extract the part before the first '|' in protein_ids
        protein_id_x = protein_ids[i].split('|')[0]
        x, y = mean_importance_men[i], mean_importance_women[i]
        
        # Check if point is within threshold of the line y = x
        if abs(x - y) > threshold:
            plt.annotate(protein_id_x, (x, y), textcoords="offset points", xytext=(0, 5), 
                         ha='center', fontsize=6)

    # Determine the common axis limits based on the data in both axes
    min_value = min(min(mean_importance_men), min(mean_importance_women))
    max_value = max(max(mean_importance_men), max(mean_importance_women))

    # Set the same range for both x and y axes
    plt.xlim(min_value - 0.001, max_value + 0.001)
    plt.ylim(min_value - 0.001, max_value + 0.001)

    # Add labels and reference line
    plt.xlabel('Mean Importance Score (%) - Men')
    plt.ylabel('Mean Importance Score (%) - Women')
    plt.title('Feature Importance Scores Comparison: Men vs Women (Percentages)')
    plt.plot([min_value, max_value], [min_value, max_value], 
             color='red', linestyle='--', label='Equal Importance')

    # Show plot
    plt.legend()
    plt.grid(True)
    plt.show()

### Analysis Functions ###

def plot_explainer_results(config, all_explanations_percent, protein_ids, filename):
    """Performs PCA analysis and creates relevant plots for the first 3 components."""
    
    reduced_vectors, explained_variance, loadings = perform_pca(np.array(all_explanations_percent))
    
    # Get labels for sex, mutation, and age from config
    total_sex_labels, total_mutation_labels, total_age_labels, _, _, _, _ = get_sex_mutation_age_distribution(config)

    # Plot the first 3 PCA components in 2D
    plot_pca_all_components_2d(reduced_vectors, explained_variance, total_sex_labels, total_mutation_labels, total_age_labels)
    
    # Plot the first 3 PCA components in 3D
    plot_pca_3d(reduced_vectors, total_sex_labels, explained_variance, "3D PCA Plot Colored by Sex")

    # Plot variance percentage for the first 10 components
    plot_variance_percentage(explained_variance[:10])

    # Plot loadings for the first 10 components
    plot_pca_loadings_line(loadings[:10], protein_ids)

### Protein Importance Plotting ###

def create_protein_plots(combined_sum_node_importance_raw, combined_sum_node_importance_percent, combined_positive_percent, combined_negative_percent, combined_protein_count, all_percent_importances, protein_ids, config, checkpoint_path):
    """Creates a set of plots for protein importance and PCA analysis."""
    pattern = r'/(model=.*?)(/checkpoint_\d+)?$'
    model_name = re.search(pattern, checkpoint_path).group(1)
    sum_node_importance_avg_percent = divide_dict_values(combined_protein_count, combined_sum_node_importance_percent)

    plot_bar_chart(
        sum_node_importance_avg_percent,
        f'Top Proteins Average Percentage Importance for {config.y_val} {config.sex} {config.mutation} {config.modality}',
        'Protein ID', 'Importance Value (%)',
    )
    # Most positive proteins
    combined_positive_avg_percent = divide_dict_values(combined_protein_count, combined_positive_percent)
    plot_top_bar_chart(
        combined_positive_avg_percent,
        f'Top Positive Proteins Average Percentage Importance for {config.y_val} {config.sex} {config.mutation} {config.modality}',
        'Protein ID', 'Importance Value (%)',
    )
    # Most Negative Proteins
    combined_negative_avg_percent = divide_dict_values(combined_protein_count, combined_negative_percent)
    plot_top_bar_chart(
        combined_negative_avg_percent,
        f'Top Negative Proteins Average Percentage Importance for {config.y_val} {config.sex} {config.mutation} {config.modality}',
        'Protein ID', 'Importance Value (%)',
    )
    # Plot the sum of node importance for each protein (top 300)
    plot_bar_chart(
        combined_sum_node_importance_raw,
        f'Sum of node importance for each protein for {config.y_val} {config.sex} {config.mutation} {config.modality}',
        'Protein', 'Sum of node importance',
    )
    # Calculate and plot the average node importance
    sum_node_importance_avg = divide_dict_values(combined_protein_count, combined_sum_node_importance_raw)
    plot_bar_chart(
        sum_node_importance_avg,
        f'Top Proteins Average Importance Value for {config.y_val} {config.sex} {config.mutation} {config.modality}',
        'Protein ID', 'Importance Value',
    )

    # Plot the number of people with each protein (top 300)
    plot_bar_chart(
        combined_protein_count,
        f'Top Proteins by Count for {config.y_val} {config.sex} {config.mutation} {config.modality}',
        'Protein ID', 'Number of People with this protein',
    )

    # PCA analysis and plotting
    plot_explainer_results(config, all_percent_importances, protein_ids, f'explainer_plots/{model_name}_pca.png')

    # Comparison plot for men vs women
    plot_importance_comparison_men_vs_women(all_percent_importances, protein_ids, config, exclude_ctl=True)


############FUNCTIONS TO COMPARE RESULTS############################################
def compare_top_n(dict1, dict2, dict3=None, n=100):
    """
    Compare the top 'n' keys in two or three dictionaries based on their values and return
    how many keys they have in common. Also prints the common keys, each on its own line.

    Args:
    dict1 (dict): The first dictionary.
    dict2 (dict): The second dictionary.
    dict3 (dict, optional): The third dictionary (optional).
    n (int): The number of top values to compare (default is 100).

    Returns:
    int: The number of common keys in the top 'n' values across the dictionaries.
    """
    # Sort the dictionaries by their values in descending order and extract the top 'n' keys
    top_n_keys_dict1 = set([k for k, v in sorted(dict1.items(), key=lambda item: item[1], reverse=True)[:n]])
    top_n_keys_dict2 = set([k for k, v in sorted(dict2.items(), key=lambda item: item[1], reverse=True)[:n]])

    # If dict3 is provided, compare with it as well
    if dict3 is not None:
        top_n_keys_dict3 = set([k for k, v in sorted(dict3.items(), key=lambda item: item[1], reverse=True)[:n]])
        common_keys = top_n_keys_dict1.intersection(top_n_keys_dict2).intersection(top_n_keys_dict3)
    else:
        common_keys = top_n_keys_dict1.intersection(top_n_keys_dict2)
    
    # Print the common keys, each on its own line
    if common_keys:
        print(f"Common keys in the top {n} values across the dictionaries:")
        for key in common_keys:
            print(f"- {key}")
    else:
        print(f"No common keys in the top {n} values across the dictionaries.")
    
    # Return the number of common keys
    return len(common_keys)