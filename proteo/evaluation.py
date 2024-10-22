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
from proteo.datasets.ftd import FTDDataset
import train as proteo_train
import plotly.graph_objs as go


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

#Load protein ids
def get_protein_ids(config):
    # Make an instance of the FTDDataset class to get the top proteins
    root = config.data_dir
    train_dataset = FTDDataset(root, "train", config)
    _, _, _, top_protein_columns, _, _, _ = train_dataset.load_csv_data_pre_pt_files(config)
    top_protein_columns.extend(['sex', 'mutation', 'age'])
    return np.array(top_protein_columns)

def run_explainer_single_dataset(dataset, explainer, protein_ids, filename):
    # Run explainer on each person in dataset
    all_proteins = []
    n_nodes = len(dataset[0].x)
    n_people = len(dataset)
    print(f"n_people = {n_people}")
    print(f"n_nodes = {n_nodes}")
    # Initialize sum_node_importance as a dictionary with protein_ids as keys
    sum_node_importance = {protein_id: 0 for protein_id in protein_ids}

    all_labels = []
    all_explanations = []
    all_top_proteins = []
    
    for i, data in enumerate(dataset):
        # Ensure data.x and data.edge_index are tensors
        if not isinstance(data.x, torch.Tensor) or not isinstance(data.edge_index, torch.Tensor):
            raise TypeError("data.x and data.edge_index must be torch.Tensor")

        # Get the explanation
        explanation = explainer(
            data.x,
            data.edge_index,
            data=data,
            target=None,
            index=None
        )

        # Node importance is of format [[0], [0],[0],...,[.5]] with length equal to the number of nodes
        node_importance = np.array(explanation.node_mask.cpu().detach().numpy().tolist())
        flat_node_importance = np.array(node_importance.flatten().tolist())

        # Calculate the total importance per person and the importance as percentage
        total_importance = np.sum(np.abs(flat_node_importance))
        importance_percentages = (np.abs(flat_node_importance) / total_importance) * 100

        all_explanations.append(flat_node_importance)

        # Sort indices by raw importance 
        sorted_indices = np.argsort(flat_node_importance)[::-1] #np.argsort(node_importance[:, 0])[::-1]

        # Get the top 5 indices and corresponding protein IDs
        top_5_indices = sorted_indices[:5]
        top_5_proteins = [protein_ids[index] for index in top_5_indices]

        # Store the top proteins for each person
        top_proteins = [protein_ids[index] for index in sorted_indices]
        all_top_proteins.append(top_proteins)

        # Plot the raw importance scores for each person
        plt.plot(sorted(flat_node_importance, reverse=True), label=f'Raw Importance for Person {i}')

        # Add top 5 protein information to the legend
        all_labels.append(f'Top 5 Proteins for person {i}: {", ".join(top_5_proteins)}')

        # Find row indices of non-zero elements in node_importance
        indv_important_proteins_indices = np.nonzero(node_importance)[0]

        # Get the protein names of the important proteins
        indv_important_proteins = protein_ids[indv_important_proteins_indices]
        
        # Append the important proteins to the list of all important proteins
        all_proteins.append(indv_important_proteins)

        # Update sum_node_importance dictionary using protein_ids as keys
        for idx, importance in enumerate(node_importance):
            if importance[0] == 0:
                continue
            protein_id = protein_ids[idx]
            sum_node_importance[protein_id] += importance[0]

    # Plot all the raw importance scores on the same graph
    plt.legend(all_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=1)
    plt.xlabel('Protein')
    plt.ylabel('Importance')
    plt.title('Sorted Raw Node Importance with Top 5 Protein IDs')
    raw_filename = os.path.join('explainer_plots', f'{filename}_raw.png')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    #plt.savefig(raw_filename)
    plt.show()

    # --- New Plot for Percentages ---
    plt.figure()  # Start a new figure for percentages
    for i, data in enumerate(dataset):
        # Plot the importance percentages for each person
        importance_percentages = (np.abs(all_explanations[i]) / np.sum(np.abs(all_explanations[i]))) * 100
        plt.plot(sorted(importance_percentages, reverse=True), label=f'Importance % for Person {i}')

    # Create a legend with the same top 5 protein information
    plt.legend(all_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=1)
    plt.xlabel('Protein')
    plt.ylabel('Importance (%)')
    plt.title('Sorted Node Importance as Percentage of Total with Top 5 Protein IDs')
    percent_filename = os.path.join('explainer_plots', f'{filename}_percent.png')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    #plt.savefig(percent_filename)
    plt.show()

    # Flatten the list of lists into a single list
    full_count = [item for sublist in all_proteins for item in sublist]

    # Use Counter to count the occurrences of each element
    protein_count = Counter(full_count)
    
    return sum_node_importance, protein_count, all_explanations, all_top_proteins

def run_explainer_train_and_test(checkpoint_path):

    module = load_checkpoint(checkpoint_path)
    config = load_config(module)
    print(config)
    # Load datasets
    train_dataset, test_dataset = proteo_train.construct_datasets(config)
    print("dim train_dataset", len(train_dataset))
    print("dim test_dataset", len(test_dataset))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset.to(device)
    test_dataset.to(device)

    # Construct Explainer and set parameters
    explainer = Explainer(
        model=module.model.to(device),
        algorithm=CaptumExplainer('IntegratedGradients'),
        explanation_type='model',
        model_config=dict(
            mode='regression',
            task_level='graph',  # Explain why the model predicts a certain property or label for the entire graph (nodes + edges)
            return_type='raw'
        ),
        node_mask_type= 'attributes', #'object', # Generate masks that indicate the importance of individual node features
        edge_mask_type=None,
        threshold_config=dict( #keep only the top 300 most important proteins and set the rest to 0
            threshold_type='topk',
            value=7000,
        ),
    )
    protein_ids = get_protein_ids(config)

    pattern = r'/(model=.*?)(/checkpoint_\d+)?$'
    model_name = re.search(pattern, checkpoint_path).group(1)
    
    train_sum_node_importance, train_protein_count, all_explanations_train, all_top_proteins_train = run_explainer_single_dataset(train_dataset, explainer, protein_ids, filename=model_name + "_train.png")
    test_sum_node_importance, test_protein_count, all_explanations_test, all_top_proteins_test = run_explainer_single_dataset(test_dataset, explainer, protein_ids, filename=model_name + "_test.png")

    # Combine sum_node_importance
    combined_sum_node_importance = {key: train_sum_node_importance.get(key, 0) + test_sum_node_importance.get(key, 0)
                                for key in set(train_sum_node_importance) | set(test_sum_node_importance)}
    
    # Combine protein_count
    combined_protein_count = train_protein_count + test_protein_count

    # Combine all explanations and top 5 proteins
    all_explanations = all_explanations_train + all_explanations_test
    print("all_explanations shape", np.array(all_explanations).shape)
    all_top_proteins = all_top_proteins_train + all_top_proteins_test
    return combined_sum_node_importance, combined_protein_count, config, all_explanations, all_top_proteins, protein_ids

def plot_a_dictionary(protein_dict, title, x_label, y_label, filename):
    sorted_sum_node_importance = dict(sorted(protein_dict.items(), key=lambda item: item[1], reverse=True))
    top_300_items = dict(list(sorted_sum_node_importance.items())[:100])
    # Separate the items into x and y components for plotting
    keys = list(top_300_items.keys())
    values = list(top_300_items.values()) # Value is a list of one object

    # Plot the data
    plt.figure(figsize=(28, 10))
    bar_width = 0.6
    plt.bar(keys, values, color='skyblue', width=bar_width)
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.title(title, fontsize=24)
    plt.xticks(rotation=90, ha='right', fontsize=12)  # Rotate x-ticks for better readability
    plt.yticks(fontsize=14)  # Increase y-tick size for better readability
    plt.tight_layout()  # Adjust layout to ensure everything fits without overlapping
    #plt.gca().invert_xaxis()  # Invert y-axis to have the highest values on top
    #plt.savefig(filename)
    plt.show()

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

def divide_dict_values(dict1, dict2):
    result = {}
    for key in dict1:
        if key in dict2:
            if dict1[key] != 0:  # Avoid division by zero
                result[key] = dict2[key] / dict1[key]
            else:
                result[key] = None  # or handle division by zero as needed
    return result

def get_sex_mutation_age_distribution(config):
    # Make an instance of the FTDDataset class to get the top proteins
    root = config.data_dir
    train_dataset = FTDDataset(root, "train", config)
    _, _, _, _, filtered_sex_col, filtered_mutation_col, filtered_age_col= train_dataset.load_csv_data_pre_pt_files(config)
    # Splitting indices only
    train_sex_labels, test_sex_labels, train_mutation_labels, test_mutation_labels, train_age_labels, test_age_labels = train_test_split(filtered_sex_col, filtered_mutation_col, filtered_age_col, test_size=0.20, random_state=42)
    total_sex_labels = np.concatenate((train_sex_labels, test_sex_labels))
    total_mutation_labels = np.concatenate((train_mutation_labels, test_mutation_labels))
    total_age_labels = np.concatenate((train_age_labels, test_age_labels))
    return total_sex_labels, total_mutation_labels, total_age_labels

def plot_pca_subplot(ax, reduced_vectors, labels, title, explained_variance, cmap=None, colorbar_label=None):
    scatter = ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=labels, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(f"PC 1 ({explained_variance[0]:.2%} variance)")
    ax.set_ylabel(f"PC 2 ({explained_variance[1]:.2%} variance)")
    if colorbar_label:
        plt.colorbar(scatter, ax=ax, label=colorbar_label)
    return scatter

def plot_pca_loadings_line(loadings, protein_ids, threshold=0.04):
    plt.figure(figsize=(20, 6))

    # Only plot for the first two principal components
    for i, loading in enumerate(loadings[:2]):  # Limit to the first two loadings (PC1 and PC2)
        # Apply the threshold to filter the loading values
        filtered_indices = [index for index, value in enumerate(loading) if abs(value) > threshold]
        filtered_values = [value for value in loading if abs(value) > threshold]
        # Convert indices to corresponding item names
        filtered_names = [protein_ids[index] for index in filtered_indices]

        # Plot only the filtered loading values
        plt.bar(filtered_names, filtered_values, width=0.5, label=f'PC{i+1}', alpha=0.7)
    
    plt.title('PCA Loadings for PC1 and PC2')
    plt.xlabel('Protein ID')
    plt.ylabel('Loading')
    plt.legend()
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to fit rotated labels
    plt.show()

def plot_importance_comparison_men_vs_women(all_explanations, protein_ids, config):
    # Extract the sex list from the config
    total_sex_labels, total_mutation_labels, total_age_labels = get_sex_mutation_age_distribution(config)
    print("Shape and type of sex_labels", total_sex_labels.shape, type(total_sex_labels))
    all_explanations = np.array(all_explanations)
    print("Shape and type of all explanations", all_explanations.shape, type(all_explanations))
    
    # Get indices for men and women
    men_indices = np.where(total_sex_labels == 'M')[0]
    women_indices = np.where(total_sex_labels == 'F')[0]

    # Calculate mean importance scores for men and women across all features
    mean_importance_men = np.mean(all_explanations[men_indices], axis=0)
    mean_importance_women = np.mean(all_explanations[women_indices], axis=0)

    # Create a scatter plot to compare men and women for each feature
    plt.figure(figsize=(8, 8))
    plt.scatter(mean_importance_men, mean_importance_women, alpha=0.6, color='gray')
    min_len = min(len(mean_importance_men), len(mean_importance_women), len(protein_ids))
    for i in range(min_len):
        plt.annotate(protein_ids[i], (mean_importance_men[i], mean_importance_women[i]), 
                    textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)

    # Add labels and title
    plt.xlabel('Mean Importance Score (Men)')
    plt.ylabel('Mean Importance Score (Women)')
    plt.title('Feature Importance Scores: Men vs. Women')

    # Add a reference line where x = y (if the importance is the same for men and women)
    plt.plot([min(mean_importance_men), max(mean_importance_men)], 
             [min(mean_importance_men), max(mean_importance_men)], 
             color='red', linestyle='--', label='Equal Importance')

    # Show the plot
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_explainer_results(config, all_explanations, protein_ids, filename):
    total_sex_labels, total_mutation_labels, total_age_labels = get_sex_mutation_age_distribution(config)

    all_explanations = np.array(all_explanations)
    print("all_explanations shape", all_explanations.shape)
    
    # Set PCA to have 10 components for variance plotting
    pca = PCA(n_components=10)
    reduced_vectors = pca.fit_transform(all_explanations)

    # Calculate explained variance ratio
    explained_variance = pca.explained_variance_ratio_

    # Get loadings (principal component vectors)
    loadings = pca.components_

    # Define mutation color map using the actual mutation names as keys
    mutation_color_map = {
        'C9orf72': 'red',
        'MAPT': 'green',
        'GRN': 'blue',
        'CTL': 'purple'
    }

    # Map mutation labels to colors
    mutation_colors = np.array([mutation_color_map[label] for label in total_mutation_labels])

    # Create subplots for the PCA scatter plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 1, 1]})

    # Determine scale factor for the aspect ratio
    scale_x = np.sqrt(explained_variance[0])
    scale_y = np.sqrt(explained_variance[1])

    # Plot for Sex
    plot_pca_subplot(axes[0], reduced_vectors[:, :2], np.where(total_sex_labels == 'F', 'pink', 'blue'), "Colored by Sex", explained_variance[:2])
    axes[0].set_aspect(scale_y/scale_x)
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

    # Plot for Mutation
    plot_pca_subplot(axes[1], reduced_vectors[:, :2], mutation_colors, "Colored by Mutation", explained_variance[:2])
    axes[1].set_aspect(scale_y/scale_x)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
               for label, color in mutation_color_map.items()]
    axes[1].legend(handles=handles, title="Mutation", loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)

    # Plot for Age (continuous values)
    scatter = axes[2].scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=total_age_labels, cmap='viridis')
    axes[2].set_xlabel(f'PC 1 ({explained_variance[0]*100:.2f}% variance)')
    axes[2].set_ylabel(f'PC 2 ({explained_variance[1]*100:.2f}% variance)')
    axes[2].set_title("Colored by Age")
    axes[2].set_aspect(scale_y/scale_x)

    # Create a new axis below the last plot for the colorbar
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # Adjust position and size as needed
    plt.colorbar(scatter, cax=cbar_ax, orientation='horizontal', label='Age')

    # Update the title with the new information
    plt.suptitle(f"PCA Dimensionality Reduction on Explainer Results for {config.y_val} {config.sex} {config.mutation} {config.modality}\nOne data point = importance scores of proteins for one participant", 
                 fontsize=16, fontweight='bold')

    # Use tight_layout to adjust the layout and reduce white space
    plt.tight_layout(pad=2.0, rect=[0, 0.1, 1, 0.95])  # Adjust rect to provide room for the suptitle and colorbar
    plt.subplots_adjust(hspace=0.5, bottom=0.15)  # Increase hspace to separate plots

    #plt.savefig(filename, bbox_inches='tight')  # Save with tight bounding box to include legend
    plt.show()

    # Plot the loadings as a separate, wider plot
    plot_pca_loadings_line(loadings, protein_ids)

    # Plot the variance percentages of the first 10 components
    plot_variance_percentage(explained_variance)
    plot_3d_pca_scatter_sex(reduced_vectors, total_sex_labels, explained_variance)

def plot_explainer_results_pca23(config, all_explanations, protein_ids, filename):
    total_sex_labels, total_mutation_labels, total_age_labels = get_sex_mutation_age_distribution(config)

    all_explanations = np.array(all_explanations)
    print("all_explanations shape", all_explanations.shape)
    
    # Set PCA to have 10 components for variance plotting
    pca = PCA(n_components=10)
    reduced_vectors = pca.fit_transform(all_explanations)

    # Calculate explained variance ratio
    explained_variance = pca.explained_variance_ratio_

    # Get loadings (principal component vectors)
    loadings = pca.components_

    # Define mutation color map using the actual mutation names as keys
    mutation_color_map = {
        'C9orf72': 'red',
        'MAPT': 'green',
        'GRN': 'blue',
        'CTL': 'purple'
    }

    # Map mutation labels to colors
    mutation_colors = np.array([mutation_color_map[label] for label in total_mutation_labels])

    # Create subplots for the PCA scatter plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 1, 1]})

    # Determine scale factor for the aspect ratio using components 2 and 3
    scale_x = np.sqrt(explained_variance[1])
    scale_y = np.sqrt(explained_variance[2])

    # Plot for Sex using components 2 and 3
    plot_pca_subplot(axes[0], reduced_vectors[:, 1:3], np.where(total_sex_labels == 'F', 'pink', 'blue'), "Colored by Sex (PC 2 vs PC 3)", explained_variance[1:3])
    axes[0].set_aspect(scale_y/scale_x)
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

    # Plot for Mutation using components 2 and 3
    plot_pca_subplot(axes[1], reduced_vectors[:, 1:3], mutation_colors, "Colored by Mutation (PC 2 vs PC 3)", explained_variance[1:3])
    axes[1].set_aspect(scale_y/scale_x)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
               for label, color in mutation_color_map.items()]
    axes[1].legend(handles=handles, title="Mutation", loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)

    # Plot for Age (continuous values) using components 2 and 3
    scatter = axes[2].scatter(reduced_vectors[:, 1], reduced_vectors[:, 2], c=total_age_labels, cmap='viridis')
    axes[2].set_xlabel(f'PC 2 ({explained_variance[1]*100:.2f}% variance)')
    axes[2].set_ylabel(f'PC 3 ({explained_variance[2]*100:.2f}% variance)')
    axes[2].set_title("Colored by Age (PC 2 vs PC 3)")
    axes[2].set_aspect(scale_y/scale_x)

    # Create a new axis below the last plot for the colorbar
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # Adjust position and size as needed
    plt.colorbar(scatter, cax=cbar_ax, orientation='horizontal', label='Age')

    # Update the title with the new information
    plt.suptitle(f"PCA Dimensionality Reduction on Explainer Results for {config.y_val} {config.sex} {config.mutation} {config.modality}\nOne data point = importance scores of proteins for one participant (PC 2 vs PC 3)", 
                 fontsize=16, fontweight='bold')

    # Use tight_layout to adjust the layout and reduce white space
    plt.tight_layout(pad=2.0, rect=[0, 0.1, 1, 0.95])  # Adjust rect to provide room for the suptitle and colorbar
    plt.subplots_adjust(hspace=0.5, bottom=0.15)  # Increase hspace to separate plots

    #plt.savefig(filename, bbox_inches='tight')  # Save with tight bounding box to include legend
    plt.show()

    # Plot the loadings as a separate, wider plot
    plot_pca_loadings_line(loadings, protein_ids)

    # Plot the variance percentages of the first 10 components
    plot_variance_percentage(explained_variance)
    plot_3d_pca_scatter_sex(reduced_vectors, total_sex_labels, explained_variance)    

def plot_3d_pca_scatter_sex(reduced_vectors, sex_labels, explained_variance):
    # Map sex labels to colors (F = pink, M = blue)
    sex_colors = np.where(sex_labels == 'F', 'pink', 'blue')

    # Create a 3D scatter plot
    fig = go.Figure(
        data=[go.Scatter3d(
            x=reduced_vectors[:, 0],
            y=reduced_vectors[:, 1],
            z=reduced_vectors[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=sex_colors,  # Use mapped colors
                opacity=0.6
            ),
            text=sex_labels  # Display sex labels on hover
        )]
    )

    # Set plot layout and axis labels
    fig.update_layout(
        title='3D Visualization of the First 3 PCA Components (Colored by Sex)',
        scene=dict(
            xaxis_title=f'PC1 ({explained_variance[0] * 100:.2f}%)',
            yaxis_title=f'PC2 ({explained_variance[1] * 100:.2f}%)',
            zaxis_title=f'PC3 ({explained_variance[2] * 100:.2f}%)'
        ),
        legend=dict(
            title='Sex',
            itemsizing='constant'
        )
    )

    # Display the plot
    fig.show()

def plot_variance_percentage(explained_variance):
    plt.figure(figsize=(10, 6))
    components = np.arange(1, len(explained_variance) + 1)
    plt.bar(components, explained_variance * 100)
    plt.title('Explained Variance by Principal Component')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance (%)')
    plt.xticks(components)
    plt.grid(True)
    plt.show()

def get_sex_per_cluster(clusters, config):
    participant_sex, _, _ = get_sex_mutation_age_distribution(config)
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

def create_protein_plots(sum_node_importance, protein_count, all_explanations, protein_ids, config, checkpoint_path):
    # Plot the sum of node importance for each protein
    pattern = r'/(model=.*?)(/checkpoint_\d+)?$'
    model_name = re.search(pattern, checkpoint_path).group(1)
    plot_a_dictionary(sum_node_importance, f'Sum of node importance for each protein for {config.y_val} {config.sex} {config.mutation} {config.modality}', 'Protein', 'Sum of node importance', f'explainer_plots/{model_name}_sum_node_importance.png')
    sum_node_importance_avg = divide_dict_values(protein_count, sum_node_importance)
    # Plot the average node importance for each protein
    plot_a_dictionary(sum_node_importance_avg, f'Top Proteins Average Importance Value for {config.y_val} {config.sex} {config.mutation} {config.modality}', 'Protein ID', 'Importance Value', f'explainer_plots/{model_name}_average_node_importance.png')
    # Plot the number of people with each protein
    plot_a_dictionary(protein_count, f'Top Proteins by Count for {config.y_val} {config.sex} {config.mutation} {config.modality}', 'Protein ID', 'Number of People with this protein', f'explainer_plots/{model_name}_protein_count.png')
    plot_importance_comparison_men_vs_women(all_explanations, protein_ids, config)
    # Plot the PCA visualization of the node importance vectors
    plot_explainer_results(config, all_explanations, protein_ids, f'explainer_plots/{model_name}_pca_top_2.png')
    plot_explainer_results_pca23(config, all_explanations, protein_ids, f'explainer_plots/{model_name}_pca_top_2.png')