import os
import re
from collections import Counter, defaultdict

import numpy as np
import torch
import train as proteo_train
from checkpoint_analysis import get_sex_mutation_age_distribution
from torch_geometric.explain import CaptumExplainer, Explainer, GNNExplainer
from torch_geometric.explain.config import ExplanationType, ModelMode
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt


############### METRICS ################
def fidelity(explainer, explanation):
    """Evaluates the fidelity of an
    :class:`~torch_geometric.explain.Explainer` given an
    :class:`~torch_geometric.explain.Explanation`, as described in the
    `"GraphFramEx: Towards Systematic Evaluation of Explainability Methods for
    Graph Neural Networks" <https://arxiv.org/abs/2206.09677>`_ paper.

    Fidelity evaluates the contribution of the produced explanatory subgraph
    to the initial prediction, either by giving only the subgraph to the model
    (fidelity-) or by removing it from the entire graph (fidelity+).
    The fidelity scores capture how good an explainable model reproduces the
    natural phenomenon or the GNN model logic.


    For **model** explanations, the fidelity scores are given by:

    .. math::
        \textrm{fid}_{+} &= 1 - \frac{1}{N} \sum_{i = 1}^N
        \mathbb{1}( \hat{y}_i^{G_{C \setminus S}} = \hat{y}_i)

        \textrm{fid}_{-} &= 1 - \frac{1}{N} \sum_{i = 1}^N
        \mathbb{1}( \hat{y}_i^{G_S} = \hat{y}_i)

    Args:
        explainer (Explainer): The explainer to evaluate.
        explanation (Explanation): The explanation to evaluate.
    """

    node_mask = explanation.get('node_mask')
    edge_mask = explanation.get('edge_mask')
    kwargs = {key: explanation[key] for key in explanation._model_args}

    y = explanation.target #these are the preds from the model

    explain_y_hat = explainer.get_masked_prediction(
        explanation.x,
        explanation.edge_index,
        node_mask,
        edge_mask,
        **kwargs,
    )
    explain_y_hat = explainer.get_target(explain_y_hat)

    complement_y_hat = explainer.get_masked_prediction(
        explanation.x,
        explanation.edge_index,
        1.0 - node_mask if node_mask is not None else None,
        1.0 - edge_mask if edge_mask is not None else None,
        **kwargs,
    )
    complement_y_hat = explainer.get_target(complement_y_hat)

    # TODO: Check if I need this part
    if explanation.get('index') is not None:
        y = y[explanation.index]
        if explainer.explanation_type == ExplanationType.phenomenon:
            y_hat = y_hat[explanation.index]
        explain_y_hat = explain_y_hat[explanation.index]
        complement_y_hat = complement_y_hat[explanation.index]

    # Changed this for regression definition
    if explainer.explanation_type == ExplanationType.model:
        # You want pos_fidelity to be as high as possible, big difference between complement_y_hat and y when you remove the subgraph
        pos_fidelity = torch.nn.functional.mse_loss(complement_y_hat, y)
        # You want neg_fidelity to be as low as possible
        neg_fidelity = 1/torch.nn.functional.mse_loss(explain_y_hat, y)

    return float(pos_fidelity), float(neg_fidelity), y


def characterization_score(pos_fidelity, neg_fidelity, pos_weight=0.5, neg_weight=0.5):
    # TODO: Redefine this for regression
    """Returns the componentwise characterization score as described in the
    `"GraphFramEx: Towards Systematic Evaluation of Explainability Methods for
    Graph Neural Networks" <https://arxiv.org/abs/2206.09677>`_ paper.

    ..  math::
       \textrm{charact} = \frac{w_{+} + w_{-}}{\frac{w_{+}}{\textrm{fid}_{+}} +
        \frac{w_{-}}{1 - \textrm{fid}_{-}}}

    Args:
        pos_fidelity (torch.Tensor): The positive fidelity
            :math:`\textrm{fid}_{+}`.
        neg_fidelity (torch.Tensor): The negative fidelity
            :math:`\textrm{fid}_{-}`.
        pos_weight (float, optional): The weight :math:`w_{+}` for
            :math:`\textrm{fid}_{+}`. (default: :obj:`0.5`)
        neg_weight (float, optional): The weight :math:`w_{-}` for
            :math:`\textrm{fid}_{-}`. (default: :obj:`0.5`)
    """
    if (pos_weight + neg_weight) != 1.0:
        raise ValueError(f"The weights need to sum up to 1 " f"(got {pos_weight} and {neg_weight})")

    numer = neg_weight + pos_weight
    denom = (neg_weight / neg_fidelity) + (pos_weight / (1.0 / pos_fidelity))
    return numer / denom


############### FUNCTIONS ################
def load_config(module):
    '''Load the config from the module  and return it'''
    config = module.config
    # Check if 'use_master_nodes' attribute exists, added for runs with master nodes
    if not hasattr(config, 'use_master_nodes'):
        # Add the attribute with a default value (e.g., False)
        setattr(config, 'use_master_nodes', False)
    setattr(config, 'data_dir', '/scratch/lcornelis/data/data_louisa')
    return config


# Load model checkpoint - Note when the wrapper class is not necessary you can use this function from checkpoint_analysis.py
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
    if not hasattr(checkpoint['hyper_parameters']['config'], 'use_master_nodes') or not hasattr(
        checkpoint['hyper_parameters']['config'], 'master_nodes'
    ):
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


def run_explainer_fidelity_single_dataset(dataset, explainer):
    # Run explainer on each person in dataset and find fidelity and characterization score
    n_nodes = len(dataset[0].x)
    n_people = len(dataset)
    print(f"n_people = {n_people}")
    print(f"n_nodes = {n_nodes}")

    all_fidelity_plus = []
    all_fidelity_minus = []
    all_y_preds = []
    all_characterization_scores = []
    for data in dataset:
        # Ensure data.x and data.edge_index are tensors
        if not isinstance(data.x, torch.Tensor) or not isinstance(data.edge_index, torch.Tensor):
            raise TypeError("data.x and data.edge_index must be torch.Tensor")
        explanation = explainer(data.x, data.edge_index, data=data, target=None, index=None)
        fidelity_plus, fidelity_minus, y_preds = fidelity(explainer, explanation)
        all_fidelity_plus.append(fidelity_plus)
        all_fidelity_minus.append(fidelity_minus)
        all_y_preds.extend(y_preds.flatten().cpu())
        characterization_score_val = characterization_score(fidelity_plus, fidelity_minus)
        all_characterization_scores.append(characterization_score_val)

    return (
        np.array(all_fidelity_plus),
        np.array(all_fidelity_minus),
        np.array(all_characterization_scores),
        np.array(all_y_preds)
    )


def compute_subgroup_metrics(mask, fidelity_plus, fidelity_minus, y_true, y_preds):
    """Helper function to compute metrics for a given subgroup."""
    fid_plus = fidelity_plus[mask]
    fid_minus = fidelity_minus[mask]
    y = y_true[mask]
    y_preds = y_preds[mask]
    squared_error = (y - y_preds) ** 2
    
    mse = np.mean(squared_error) if len(y) > 0 else float('nan')
    return fid_plus, fid_minus, squared_error, mse

def fidelity_per_subgroup_train_and_test(checkpoint_path):
    module = load_checkpoint(checkpoint_path)
    config = load_config(module)
    
    # Load datasets
    train_dataset, test_dataset = proteo_train.construct_datasets(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset.to(device)
    test_dataset.to(device)
    train_y = np.array((train_dataset.y).cpu())
    test_y = np.array((test_dataset.y).cpu())

    # Construct Explainer and set parameters
    explainer = Explainer(
        model=module.model.to(device),
        algorithm=CaptumExplainer('IntegratedGradients'),
        explanation_type='model',
        model_config=dict(
            mode='regression',
            task_level='graph',  
            return_type='raw',
        ),
        node_mask_type='attributes',
        edge_mask_type=None,
        threshold_config=dict(
            threshold_type='hard', 
            value=0.1,
        ),
    )

    # Run explainer on each person in dataset and find fidelity and characterization score for train and test
    (
        train_fidelity_plus,
        train_fidelity_minus,
        train_characterization_scores,
        train_y_preds
    ) = run_explainer_fidelity_single_dataset(train_dataset, explainer)
    (
        test_fidelity_plus,
        test_fidelity_minus,
        test_characterization_scores,
        test_y_preds
    ) = run_explainer_fidelity_single_dataset(test_dataset, explainer)

    # Define subgroups
    sex_labels = ['M', 'F']
    mutation_labels = ["Carrier", "CTL"]

    # Get characteristics of each person in the dataset
    (
        train_sex_labels,
        test_sex_labels,
        train_mutation_labels,
        test_mutation_labels,
        _,  # Removing age labels
        _,
    ) = get_sex_mutation_age_distribution(config)
    
    # Simplify mutation labels
    train_mutation_labels = np.where(np.isin(train_mutation_labels, ["C9orf72", "MAPT", "GRN"]), "Carrier", train_mutation_labels)
    test_mutation_labels = np.where(np.isin(test_mutation_labels, ["C9orf72", "MAPT", "GRN"]), "Carrier", test_mutation_labels)

    train_fidelity_results = {}
    test_fidelity_results = {}

    for sex in sex_labels:
        for mutation in mutation_labels:
            subgroup_name = f"{sex}_{mutation}"

            # Create mask for current subgroup
            train_mask = (train_sex_labels == sex) & (train_mutation_labels == mutation)
            test_mask = (test_sex_labels == sex) & (test_mutation_labels == mutation)

            # Compute metrics for training and testing subgroups
            train_fid_plus, train_fid_minus, train_squared_error, train_mse = compute_subgroup_metrics(
                train_mask, train_fidelity_plus, train_fidelity_minus, train_y, train_y_preds
            )
            test_fid_plus, test_fid_minus, test_squared_error, test_mse = compute_subgroup_metrics(
                test_mask, test_fidelity_plus, test_fidelity_minus, test_y, test_y_preds
            )

            # Store results
            train_fidelity_results[subgroup_name] = (
                np.sum(train_mask),
                np.mean(train_fid_plus),
                train_fid_plus,
                np.mean(train_fid_minus),
                train_fid_minus,
                train_mse,
                train_squared_error
            ) if len(train_fid_plus) > 0 else "No samples in subgroup"

            test_fidelity_results[subgroup_name] = (
                np.sum(test_mask),
                np.mean(test_fid_plus),
                test_fid_plus,
                np.mean(test_fid_minus),
                test_fid_minus,
                test_mse,
                test_squared_error
            ) if len(test_fid_plus) > 0 else "No samples in subgroup"

    return train_fidelity_results, test_fidelity_results

def plot_histogram(ax, data, title, xlabel, color, std_value):
    """Helper function to plot histograms and annotate standard deviations."""
    ax.hist(data, bins=30, alpha=0.5, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.text(0.8, 0.9, f'Std: {std_value:.2f}', transform=ax.transAxes)

def plot_difference_histograms(train_fidelity_results_personalized, train_fidelity_results_non_personalized):
    # Create a new figure
    fig, axes = plt.subplots(len(train_fidelity_results_personalized), 3, figsize=(18, 8 * len(train_fidelity_results_personalized)))
    
    # Iterate through the subgroups to plot histograms of the differences
    for idx, key in enumerate(train_fidelity_results_personalized):
        if train_fidelity_results_personalized[key] != "No samples in subgroup" and train_fidelity_results_non_personalized[key] != "No samples in subgroup":
            # Extract the arrays from both results
            personalized_fid_plus = train_fidelity_results_personalized[key][2]
            non_personalized_fid_plus = train_fidelity_results_non_personalized[key][2]
            
            personalized_fid_minus = train_fidelity_results_personalized[key][4]
            non_personalized_fid_minus = train_fidelity_results_non_personalized[key][4]
            
            personalized_squared_error = train_fidelity_results_personalized[key][6]
            non_personalized_squared_error = train_fidelity_results_non_personalized[key][6]
            
            # Compute the element-wise differences
            fid_plus_diff = non_personalized_fid_plus - personalized_fid_plus
            fid_minus_diff = non_personalized_fid_minus - personalized_fid_minus
            squared_error_diff = non_personalized_squared_error - personalized_squared_error

            # Compute the mean and std of the differences
            mean_fid_plus_diff = np.mean(fid_plus_diff)
            mean_fid_minus_diff = np.mean(fid_minus_diff)
            mean_squared_error_diff = np.mean(squared_error_diff)

            std_fid_plus_diff = np.std(fid_plus_diff)
            std_fid_minus_diff = np.std(fid_minus_diff)
            std_squared_error_diff = np.std(squared_error_diff)

            # Print the mean differences for the current subgroup
            print(f"\nSubgroup: {key}")
            print(f"Mean Difference (Incomprehensiveness): {mean_fid_plus_diff:.4f}")
            print(f"Mean Difference (Sufficiency): {mean_fid_minus_diff:.4f}")
            print(f"Mean Difference (Squared Error): {mean_squared_error_diff:.4f}")

            # Plot the histograms
            row = idx

            # Incomprehensiveness Difference Histogram
            plot_histogram(axes[row, 0], fid_plus_diff, f'{key} Incomprehensiveness Difference', 'Difference', f'C{row}', std_fid_plus_diff)

            # Sufficiency Difference Histogram
            plot_histogram(axes[row, 1], fid_minus_diff, f'{key} Sufficiency Difference', 'Difference', f'C{row}', std_fid_minus_diff)

            # Squared Error Difference Histogram
            plot_histogram(axes[row, 2], squared_error_diff, f'{key} Squared Error Difference', 'Difference', f'C{row}', std_squared_error_diff)

    plt.tight_layout()
    plt.savefig("differences_histograms_with_std.png")
    plt.show()

    
def main():
    # Load and compute fidelity for personalized model
    checkpoint_path_personalized = '/scratch/lcornelis/outputs/ray_results/TorchTrainer_2024-08-23_13-20-08/model=gat-v4,seed=44609_31_act=relu,adj_thresh=0.9000,batch_size=50,dropout=0,l1_lambda=0.0104,lr=0.0034,lr_scheduler=ReduceLROnPl_2024-08-23_13-20-08/checkpoint_000006'
    train_fidelity_results_personalized, test_fidelity_results_personalized = fidelity_per_subgroup_train_and_test(
        checkpoint_path_personalized
    )

    # Load and compute fidelity for non-personalized model
    checkpoint_path_non_personalized = '/scratch/lcornelis/outputs/ray_results/TorchTrainer_2024-09-26_14-59-55/model=gat-v4,seed=32248_160_act=elu,adj_thresh=0.5000,batch_size=50,dropout=0.0500,l1_lambda=0.0001,lr=0.0017,lr_scheduler=CosineA_2024-09-26_14-59-56/checkpoint_000047'
    train_fidelity_results, test_fidelity_results = fidelity_per_subgroup_train_and_test(
        checkpoint_path_non_personalized
    )

    # Compute the differences between the personalized and non-personalized test results
    test_fidelity_differences = {}
    for key in test_fidelity_results_personalized:
        if key in test_fidelity_results and test_fidelity_results_personalized[key] != "No samples in subgroup" and test_fidelity_results[key] != "No samples in subgroup":
            # Subtract element-wise for fidelity_plus, fidelity_minus, and squared_error
            test_fidelity_differences[key] = {
                'fidelity_plus_diff': test_fidelity_results[key][1] - test_fidelity_results_personalized[key][1],
                'fidelity_minus_diff': test_fidelity_results[key][3] - test_fidelity_results_personalized[key][3],
                'squared_error_diff': test_fidelity_results[key][5] - test_fidelity_results_personalized[key][5]
            }

    print("Differences between personalized and non-personalized test sets:")
    print(test_fidelity_differences)

    # Plot the differences
    plot_difference_histograms(train_fidelity_results_personalized, train_fidelity_results)

    
if __name__ == "__main__":
    main()
