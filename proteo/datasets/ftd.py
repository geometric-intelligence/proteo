import os

os.environ["RPY2_RINTERFACE_SIGINT"] = "FALSE"
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyWGCNA
import torch
from scipy.stats import chi2_contingency, kendalltau, ks_2samp, ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data, InMemoryDataset

LABEL_DIM_MAP = {
    "clinical_dementia_rating_global": 5,
    "clinical_dementia_rating_binary": 1,  # binary classification CDR=0 versus CDR>0
    "clinical_dementia_rating": 1,
    "carrier": 1,
    "disease_age": 1,
    "executive_function": 1,
    "memory": 1,
    "nfl": 1,
    "cog_z_score": 1,
}
SEXES = [["M"], ["F"], ["M", "F"], ["F", "M"]]
MODALITIES = ["plasma", "csf"]

Y_VALS_TO_NORMALIZE = ["nfl", "cog_z_score", "clinical_dementia_rating"]
CONTINOUS_Y_VALS = [
    "nfl",
    "disease_age",
    "executive_function",
    "memory",
    "clinical_dementia_rating",
    "cog_z_score"
]
BINARY_Y_VALS_MAP = {
    "clinical_dementia_rating_binary": {0: 0, 0.5: 1, 1: 1, 2: 1, 3: 1},
    "carrier": {"CTL": 0, "Carrier": 1},
}
MULTICLASS_Y_VALS_MAP = {"clinical_dementia_rating_global": {0: 0, 0.5: 1, 1: 2, 2: 3, 3: 4}}

HAS_MODALITY_COL = {
    "plasma": "HasPlasma?",
    "csf": "HasCSF?",
}
MODALITY_COL_END = {
    "plasma": "|PLASMA",
    "csf": "|CSF",
}
Y_VAL_COL_MAP = {
    'nfl': "NFL3_MEAN",
    'disease_age': "disease.age",
    'executive_function': "ef.unadj.slope",
    'memory': "mem.unadj.slope",
    'clinical_dementia_rating': "FTLDCDR_SB",
    'clinical_dementia_rating_global': "CDRGLOB",
    'clinical_dementia_rating_binary': "CDRGLOB",
    'carrier': "Carrier.Status",
    'cog_z_score': "GLOBALCOG.ZSCORE"
}

mutation_col = "Mutation"
sex_col = "SEX_AT_BIRTH"
age_col = "AGE_AT_VISIT"
did_col = "DID"


class FTDDataset(InMemoryDataset):
    """This is dataset used in FTD.
    This is a graph regression task.

    **Rows:**
    - 0: Column Headers
    - 1 - 531 : Patient ID Number *(int)*

    **Columns:**
    - 0: DID *(int):* Patient ID
    - 1: Mutation *(string)*: CTL (Control), MAPT, C9orf72, GRN
    - 2: AGE_AT_VISIT *(int)*
    - 3: SEX_AT_BIRTH *(string)*: M, F
    - 4: Carrier.Status *(string)*: Carrier, CTL
    - 5: Gene.Dx *(string)*:  mutation status + clinical status
    (“PreSx” suffix = presymptomatic and “Sx” suffix = symptomatic)
    - 6: GLOBALCOG.ZCORE *(float)*: global cognition composite score
    - 7: FTLDCDR_SBL *(int)*: CDR sum of boxes - Clinical Dementia Rating Scale (CDR)
    is a global assessment instrument that yields global and Sum of Boxes (SOB) scores,
    with the global score regularly used in clinical and research settings
    to stage dementia severity. Higher is worse.
    - 8: NFL3_MEAN *(float):* plasma NfL concentrations
    - 9 : ef.unadj.intercept: Executive function unadjusted intercept
    - 10 : ef.unadj.slope: Executive function unadjusted slope
    - 11: ef.adj.intercept: Executive function adjusted intercept
    - 12: ef.adj.slope: Executive function adjusted slope
    - 13: mem.unadj.intercept: Memory unadjusted intercept
    - 14: mem.unadj.slope: Memory unadjusted slope
    - 15: mem.adj.intercept: Memory adjusted intercept
    - 16: mem.adj.slope: Memory adjusted slope
    - 17: disease.age: Disease age

    - 9: HasPlasma? *(int)*: 1, 0 (519 Yes)
    - 19 - 7307: Proteins *(float)*:

    Protein variables are annotated as
      Protein Symbol | UniProt ID^Sequence ID| Matrix (CSF or PLASMA).
      The sequence ID is present only if there is more than one target
      for a given protein: e.g.,
      ABL2|P42684^SL010488@seq.3342.76|PLASMA ,
      ABL2|P42684^SL010488@seq.5261.13|PLASMA

    - 7308: HasCSF? *(int)*: 1, 0 (254 Yes)
    - 7309 - 14597: Proteins *(float)*:
    - 14598 - 15221: Clinical Data - maybe not necessary for right now.

    """

    def __init__(self, root, split, config):
        self.name = 'ftd'
        self.root = root
        self.split = split
        assert split in ["train", "test"]

        assert config.sex in SEXES
        assert config.modality in MODALITIES
        assert config.y_val in LABEL_DIM_MAP
        if config.y_val == 'carrier':
            assert len(config.mutation) > 1 and "CTL" in config.mutation

        self.config = config
        self.adj_str = f'adj_thresh_{config.adj_thresh}'
        self.y_val_str = f'y_val_{config.y_val}'
        self.num_nodes_str = f'num_nodes_{config.num_nodes}'
        self.mutation_str = f'mutation_{",".join(config.mutation)}'
        self.modality_str = f'{config.modality}'
        self.sex_str = f'sex_{",".join(config.sex)}'
        self.hist_path_str = f'{self.config.y_val}_{self.config.sex}_{self.config.mutation}_{self.config.modality}_histogram.jpg'
        self.orig_hist_path_str = f'{self.config.y_val}_{self.config.sex}_{self.config.mutation}_{self.config.modality}_orig_histogram.jpg'

        super(FTDDataset, self).__init__(root, transform=None, pre_transform=None)
        self.feature_dim = 1  # protein concentration is a scalar, ie, dim 1
        self.label_dim = LABEL_DIM_MAP[self.config.y_val]

        path = os.path.join(
            self.processed_dir,
            f'{self.name}_{self.y_val_str}_{self.adj_str}_{self.num_nodes_str}_{self.mutation_str}_{self.modality_str}_{self.sex_str}_masternodes_{self.config.use_master_nodes}_sex_specific_{self.config.sex_specific_adj}_{split}_nolog.pt',
        )
        print("Loading data from:", path)
        self.load(path)

    @property
    def raw_file_names(self):
        """Files that must be present in order to skip downloading them from somewhere.

        Then, the grandparent Dataset class automatically defines raw_paths as:
        raw_path = self.raw_dir + raw_filename
        where: self.processed_dir = self.root + "raw"

        See Also
        --------
        https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/dataset.py
        """
        return [self.config.raw_file_name]

    @property
    def processed_file_names(self):
        """Files that must be present in order to skip processing.

        The, the grandparent Dataset class automatically defines processed_paths as:
        processed_path = self.processed_dir + processed_filename
        where: self.processed_dir = self.root + "processed"

        See Also
        --------
        https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/dataset.py
        """
        files= [
            f"{self.name}_{self.y_val_str}_{self.adj_str}_{self.num_nodes_str}_{self.mutation_str}_{self.modality_str}_{self.sex_str}_masternodes_{self.config.use_master_nodes}_sex_specific_{self.config.sex_specific_adj}_train_nolog.pt",
            f"{self.name}_{self.y_val_str}_{self.adj_str}_{self.num_nodes_str}_{self.mutation_str}_{self.modality_str}_{self.sex_str}_masternodes_{self.config.use_master_nodes}_sex_specific_{self.config.sex_specific_adj}_test_nolog.pt",
        ]
        print("Processed file names:", files)
        return files

    def create_graph_data(
        self,
        feature,
        label,
        adj_matrix,
        sex,
        mutation,
        age,
    ):
        """Create Data object for each graph.

        Compute attributes x, edge_index, and y for each graph.
        """
        x = feature  # protein concentrations: what is on the nodes
        adj_tensor = torch.tensor(adj_matrix)
        # Find the indices where the matrix has non-zero elements
        pairs_indices = torch.nonzero(adj_tensor, as_tuple=False)
        # Extract the pairs of connected nodes
        edge_index = torch.tensor(pairs_indices.tolist())
        edge_index = torch.transpose(edge_index, 0, 1)  # reshape(edge_index, (2, -1))
        return Data(x=x, edge_index=edge_index, y=label, sex=sex, mutation=mutation, age=age)

    def process(self):
        """
        Read data into huge `Data` list, i.e., a list of graphs.
        """
        # Unpack the return values from load_csv_data
        (
            train_features,
            train_labels,
            test_features,
            test_labels,
            train_sex,
            test_sex,
            train_mutation,
            test_mutation,
            train_age,
            test_age,
            adj_matrices,  # This will be a list
        ) = self.load_csv_data(self.config)

        train_data_list = []
        test_data_list = []

        # Check if we are using sex-specific adjacency matrices
        if self.config.sex_specific_adj:
            # Unpack male and female adjacency matrices
            adj_matrix_M, adj_matrix_F = adj_matrices

            # Iterate through train data and assign the correct adjacency matrix based on sex
            for feature, label, sex, mutation, age in zip(
                train_features, train_labels, train_sex, train_mutation, train_age
            ):
                # Choose male or female adjacency matrix based on sex label (assuming 1 = Male, 0 = Female)
                adj_matrix = adj_matrix_M if sex.item() == 1 else adj_matrix_F
                data = self.create_graph_data(feature, label, adj_matrix, sex, mutation, age)
                train_data_list.append(data)

            # Iterate through test data and assign the correct adjacency matrix based on sex
            for feature, label, sex, mutation, age in zip(
                test_features, test_labels, test_sex, test_mutation, test_age
            ):
                adj_matrix = adj_matrix_M if sex.item() == 1 else adj_matrix_F
                data = self.create_graph_data(feature, label, adj_matrix, sex, mutation, age)
                test_data_list.append(data)

        else:
            # Single adjacency matrix is used for both male and female data
            adj_matrix = adj_matrices[0]

            # Iterate through train data and use the single adjacency matrix
            for feature, label, sex, mutation, age in zip(
                train_features, train_labels, train_sex, train_mutation, train_age
            ):
                data = self.create_graph_data(feature, label, adj_matrix, sex, mutation, age)
                train_data_list.append(data)

            # Iterate through test data and use the single adjacency matrix
            for feature, label, sex, mutation, age in zip(
                test_features, test_labels, test_sex, test_mutation, test_age
            ):
                data = self.create_graph_data(feature, label, adj_matrix, sex, mutation, age)
                test_data_list.append(data)

        # Save the train and test data lists
        self.save(train_data_list, self.processed_paths[0])
        self.save(test_data_list, self.processed_paths[1])

    # -----------------------------FUNCTIONS TO GET FEATURES---------------------------------#
    def find_top_proteins(self, filtered_data, y_val):
        # Get the column names of the correct proteins (plasma or csf)
        modality_cols = [
            col
            for col in filtered_data.columns
            if col.endswith(MODALITY_COL_END[self.config.modality])
        ]

        # Case where y_val = "carrier", "clinical_dementia_rating_binary", ks test - binary
        if self.config.y_val in BINARY_Y_VALS_MAP:
            top_columns, metric = self.get_top_columns_binary_classification(
                filtered_data, modality_cols, y_val
            )
        # Case where y_val = "nfl", "disease_age", "executive_function", "memory", "clinical_dementia_rating", regression
        elif self.config.y_val in CONTINOUS_Y_VALS:
            top_columns, metric = self.get_top_columns_regression(
                filtered_data, modality_cols, y_val
            )
        # Case where y_val = "clinical_dementia_rating_global", multiclass
        elif self.config.y_val in MULTICLASS_Y_VALS_MAP:
            top_columns, metric = self.get_top_columns_multiclass(
                filtered_data, modality_cols, y_val
            )

        # Save the plasma_protein_names to a file for wandb
        protein_names = top_columns['Protein'].values
        file_path = os.path.join(
            self.processed_dir,
            f'top_proteins_num_nodes_{self.config.num_nodes}_mutation_{self.config.mutation}_{self.config.modality}_{self.config.sex}.npy',
        )
        # Combine into a structured array
        structured_array = np.rec.array(
            (protein_names, metric), dtype=[('Protein', 'U50'), ('Metric', 'f8')]
        )
        np.save(file_path, structured_array)

        return top_columns['Protein'].tolist()

    def get_top_columns_binary_classification(self, filtered_data, modality_cols, y_val):
        '''For binary classification, use the KS test to find the most different proteins between the two groups.'''
        # Compare the group with y=1 to the group with y=0
        # Convert to pandas Series to align with filtered_data
        y_val = pd.Series(y_val, index=filtered_data.index)
        group_1 = filtered_data[y_val == 1]
        group_0 = filtered_data[y_val == 0]
        ks_stats = []
        for protein_column in modality_cols:
            data1 = group_1[protein_column]
            data0 = group_0[protein_column]
            ks_statistic, ks_p_value = ks_2samp(data1, data0)
            ks_stats.append((protein_column, ks_statistic, ks_p_value))
        ks_stats_df = pd.DataFrame(ks_stats, columns=['Protein', 'KS_Statistic', 'P Value'])
        top_columns = ks_stats_df.sort_values(by='P Value', ascending=True).head(
            self.config.num_nodes
        )
        return top_columns, top_columns['P Value'].values

    def get_top_columns_regression(self, filtered_data, modality_cols, y_val):
        '''Find the top n proteins with the highest correlation to y_val using Kendall's tau.'''
        correlations = []
        # Compute Kendall's tau correlation for each protein column with y_val
        for protein_column in modality_cols:
            tau, p_value = kendalltau(filtered_data[protein_column], y_val)
            correlations.append((protein_column, tau, p_value))
        correlations_df = pd.DataFrame(correlations, columns=['Protein', 'Kendall_Tau', 'P Value'])
        correlations_df['Abs_Tau'] = correlations_df['Kendall_Tau'].abs()  # take absolute value
        top_columns = correlations_df.sort_values(by='Abs_Tau', ascending=False).head(
            self.config.num_nodes
        )
        return top_columns, top_columns['Kendall_Tau'].values

    def get_top_columns_multiclass(self, filtered_data, modality_cols, y_val):
        '''Find the top n proteins with the highest p score between cdr = 0 and cdr>0. NOTE: this is hardcoded for cdr global right now.'''
        y_val = pd.Series(y_val, index=filtered_data.index)
        group_1 = filtered_data[y_val.isin([1, 2, 3, 4])]
        group_0 = filtered_data[y_val == 0]
        ks_stats = []
        for protein_column in modality_cols:
            data1 = group_1[protein_column]
            data0 = group_0[protein_column]
            ks_statistic, ks_p_value = ks_2samp(data1, data0)
            ks_stats.append((protein_column, ks_statistic, ks_p_value))
        ks_stats_df = pd.DataFrame(ks_stats, columns=['Protein', 'KS_Statistic', 'P Value'])
        top_columns = ks_stats_df.sort_values(by='P Value', ascending=True).head(
            self.config.num_nodes
        )
        return top_columns, top_columns['P Value'].values

    # -----------------------------FUNCTIONS TO GET LABELS---------------------------------#
    def load_y_vals(self, filtered_data):
        '''Find the y_val values based on the config.'''
        y_vals = filtered_data[Y_VAL_COL_MAP[self.config.y_val]]
        if self.config.y_val in Y_VALS_TO_NORMALIZE:
            hist_path = os.path.join(self.processed_dir, self.orig_hist_path_str)
            plot_histogram(pd.DataFrame(y_vals), f'original {self.config.y_val}', save_to=hist_path)
            y_train, y_test = train_test_split(y_vals, test_size=0.20, random_state=42)
            y_vals, mean, std = log_transform(y_train, y_vals)
        y_vals_mask = ~y_vals.isna()
        y_vals = y_vals[y_vals_mask]

        if self.config.y_val in BINARY_Y_VALS_MAP:
            y_vals = self.load_binary_y_values(y_vals)
        elif self.config.y_val in MULTICLASS_Y_VALS_MAP:
            y_vals = self.load_multiclass_y_values(y_vals)
        # Remove NaN values from y_vals and return filter to remove rows where y_val is NaN

        # Plot histogram of y_vals
        hist_path = os.path.join(
            self.processed_dir,
            self.hist_path_str,
        )
        plot_histogram(pd.DataFrame(y_vals), self.config.y_val, save_to=hist_path)
        return y_vals, y_vals_mask

    def load_binary_y_values(self, y_vals):
        '''Load the binary y_val values using dictionary that maps values to keys.'''
        mapping_dict = BINARY_Y_VALS_MAP[self.config.y_val]
        mapped_values = [mapping_dict[value] for value in y_vals]
        return mapped_values

    def load_multiclass_y_values(self, y_vals):
        '''Load multiclass y_values and encode as index targets for focal loss'''
        mapping_dict = MULTICLASS_Y_VALS_MAP[self.config.y_val]
        mapped_values = [mapping_dict[value] for value in y_vals]
        return mapped_values

    def load_csv_data_pre_pt_files(self, config):
        '''Load the csv data features and labels'''
        csv_path = self.raw_paths[0]
        print("Loading data from:", csv_path)
        csv_data = pd.read_csv(csv_path)
        # Remove bimodal columns
        csv_data = remove_erroneous_columns(config, csv_data, self.raw_dir)
        # Get the correct subset of proteins based on the mutation, if they have the correct modality measurements, and sex and then use those to find the top proteins and labels
        condition_sex = csv_data[sex_col].isin(self.config.sex)
        condition_modality = csv_data[HAS_MODALITY_COL[self.config.modality]]
        condition_mutation = csv_data[mutation_col].isin(self.config.mutation)
        sex_mutation_modality_filter = condition_sex & condition_mutation & condition_modality
        print("Number of patients with measurements:", condition_modality.sum())
        print(
            f"Number of patients with mutation status in {self.config.mutation}:",
            condition_mutation.sum(),
        )
        print(f"Number of patients with sex in {self.config.sex}:", condition_sex.sum())
        print("Total number of patients with all conditions", sex_mutation_modality_filter.sum())
        filtered_data = csv_data[
            sex_mutation_modality_filter
        ]  # Select rows that meet all conditions

        if config.sex_specific_adj:
            # Ensure that both "M" and "F" are present in config.sex if sex-specific adjacency is enabled
            if "M" not in config.sex or "F" not in config.sex:
                raise ValueError(
                    "When `sex_specific_adj` is True, `config.sex` must contain both 'M' and 'F'."
                )
        filtered_data_for_adj = (
            filtered_data  # don't remove values with nfl = NaN to find corr for adj matrix.
        )
        # Extract the y_val values
        y_vals, y_val_mask = self.load_y_vals(filtered_data)
        filtered_data = filtered_data[y_val_mask]  # Remove rows where y_val is NaN
        # Extract the top proteins (features) for building datasets
        top_protein_columns = self.find_top_proteins(filtered_data, y_vals)
        #top_protein_columns = sorted(
        #    top_protein_columns
        #)  # to ensure same ordering when using all proteins
        top_proteins = filtered_data[top_protein_columns]
        filtered_data_for_adj = filtered_data_for_adj[
            top_protein_columns + [sex_col]
        ]  # keep sex for filtering later

        # Extract column labels for sex to understand explainer results
        filtered_sex_col = filtered_data[sex_col]
        print("Dimensions of filtered sex col", filtered_sex_col.shape)
        filtered_mutation_col = filtered_data[mutation_col]
        filtered_age_col = filtered_data[age_col]
        filtered_did_col = filtered_data[did_col]

        features = np.array(top_proteins)
        print("Features shape before master nodes:", features.shape)

        labels = np.array(y_vals)

        return (
            features,
            labels,
            filtered_data_for_adj,
            top_protein_columns,
            filtered_sex_col,
            filtered_mutation_col,
            filtered_age_col,
            filtered_did_col
        )  # NOTE: Just returning top_protein_cols to use it in finding top proteins in evaluation.ipynb

    def load_csv_data(self, config):
        (
            features,
            labels,
            filtered_data_for_adj,
            top_protein_columns,
            filtered_sex_col,
            filtered_mutation_col,
            filtered_age_col,
            filtered_did_col
        ) = self.load_csv_data_pre_pt_files(config)

        # Convert sex and mutation to categorical labels
        sex_labels = np.array(filtered_sex_col.astype('category').cat.codes)
        mutation_labels = np.array(filtered_mutation_col.astype('category').cat.codes)

        if self.config.use_master_nodes:
            master_node_dict = {
                'sex': sex_labels.reshape(-1, 1),
                'mutation': mutation_labels.reshape(-1, 1),
                'age': filtered_age_col.values.reshape(-1, 1),
            }
            master_node_features = [
                master_node_dict[feature]
                for feature in self.config.master_nodes
                if feature in master_node_dict
            ]
            if master_node_features:
                master_node_features = np.concatenate(master_node_features, axis=1)
                features = np.concatenate((features, master_node_features), axis=1)
            print("using master nodes")
            print("features shape after master nodes", features.shape)

        (
            train_features,
            test_features,
            train_labels,
            test_labels,
            train_sex,
            test_sex,
            train_mutation,
            test_mutation,
            train_age,
            test_age,
        ) = train_test_split(
            features,
            labels,
            sex_labels,
            mutation_labels,
            filtered_age_col,
            test_size=0.20,
            random_state=42,
        )
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features) 
        train_age = scaler.fit_transform(train_age.values.reshape(-1, 1)).reshape(-1)
        test_age = scaler.transform(test_age.values.reshape(-1, 1)).reshape(-1)

        train_features = torch.FloatTensor(train_features.reshape(-1, train_features.shape[1], 1))
        test_features = torch.FloatTensor(test_features.reshape(-1, test_features.shape[1], 1))
        train_labels = torch.FloatTensor(train_labels)
        test_labels = torch.FloatTensor(test_labels)
        train_sex = torch.IntTensor(train_sex)
        test_sex = torch.IntTensor(test_sex)
        train_mutation = torch.IntTensor(train_mutation)
        test_mutation = torch.IntTensor(test_mutation)
        train_age = torch.FloatTensor(train_age)
        test_age = torch.FloatTensor(test_age)
        print("Training features and labels:", train_features.shape, train_labels.shape)
        print("Testing features and labels:", test_features.shape, test_labels.shape)
        print("--> Total features and labels:", features.shape, labels.shape)
        print(
            "Training sex, mutation and age labels shape:",
            train_sex.shape,
            train_mutation.shape,
            train_age.shape,
        )
        print(
            "Testing sex, mutation and age labels shape:",
            test_sex.shape,
            test_mutation.shape,
            test_age.shape,
        )
        adj_matrices = []
        if config.sex_specific_adj:
            adj_path_M = os.path.join(
                self.processed_dir,
                f'adjacency_num_nodes_{config.num_nodes}_mutation_{config.mutation}_{config.modality}_sex_{config.sex}_masternodes_{config.use_master_nodes}_sex_specific_{config.sex_specific_adj}_M.csv',
            )
            adj_path_F = os.path.join(
                self.processed_dir,
                f'adjacency_num_nodes_{config.num_nodes}_mutation_{config.mutation}_{config.modality}_sex_{config.sex}_masternodes_{config.use_master_nodes}_sex_specific_{config.sex_specific_adj}_F.csv',
            )
            if not os.path.exists(adj_path_M) or not os.path.exists(adj_path_F):
                print("Sex-specific adjacency matrices do not exist. Calculating now...")
                filtered_data_M = filtered_data_for_adj[filtered_data_for_adj[sex_col] == "M"].drop(
                    columns=[sex_col]
                )
                print("Length of filtered M:", len(filtered_data_M))
                filtered_data_F = filtered_data_for_adj[filtered_data_for_adj[sex_col] == "F"].drop(
                    columns=[sex_col]
                )
                print("Length of filtered F:", len(filtered_data_F))
                calculate_adjacency_matrix(config, filtered_data_M, save_to=adj_path_M)
                calculate_adjacency_matrix(config, filtered_data_F, save_to=adj_path_F)

            adj_matrix_M = self.load_adjacency_matrix(adj_path_M, config.adj_thresh, config)
            adj_matrix_F = self.load_adjacency_matrix(adj_path_F, config.adj_thresh, config)
            self.plot_adj_matrix(
                adj_matrix_M,
                os.path.join(
                    self.processed_dir,
                    f'adjacency_num_nodes_{config.num_nodes}_adjthresh_{config.adj_thresh}_mutation_{config.mutation}_{config.modality}_sex_{config.sex}_masternodes_{config.use_master_nodes}_sex_specific_{config.sex_specific_adj}_M.jpg',
                ),
            )
            self.plot_adj_matrix(
                adj_matrix_F,
                os.path.join(
                    self.processed_dir,
                    f'adjacency_num_nodes_{config.num_nodes}_adjthresh_{config.adj_thresh}_mutation_{config.mutation}_{config.modality}_sex_{config.sex}_masternodes_{config.use_master_nodes}_sex_specific_{config.sex_specific_adj}_F.jpg',
                ),
            )
            adj_matrices.extend([adj_matrix_M, adj_matrix_F])
        else:
            adj_path = os.path.join(
                self.processed_dir,
                f'adjacency_num_nodes_{config.num_nodes}_mutation_{config.mutation}_{config.modality}_sex_{config.sex}_masternodes_{config.use_master_nodes}_sex_specific_{config.sex_specific_adj}.csv',
            )
            # Calculate and save adjacency matrix
            if not os.path.exists(adj_path):
                filtered_data_no_sex = filtered_data_for_adj.drop(columns=[sex_col])
                calculate_adjacency_matrix(config, filtered_data_no_sex, save_to=adj_path)

            adj_matrix = self.load_adjacency_matrix(adj_path, config.adj_thresh, config)
            # Plot and save adjacency matrix as jpg
            self.plot_adj_matrix(
                adj_matrix,
                os.path.join(
                    self.processed_dir,
                    f'adjacency_{config.adj_thresh}_num_nodes_{config.num_nodes}_adjthresh_{config.adj_thresh}_mutation_{config.mutation}_{config.modality}_sex_{config.sex}_masternodes_{config.use_master_nodes}_sex_specific_{config.sex_specific_adj}.jpg',
                ),
            )
            adj_matrices.append(adj_matrix)
        return (
            train_features,
            train_labels,
            test_features,
            test_labels,
            train_sex,
            test_sex,
            train_mutation,
            test_mutation,
            train_age,
            test_age,
            adj_matrices,
        )

    def load_adjacency_matrix(self, path, adj_thresh, config):
        """
        Load and threshold an adjacency matrix.

        Parameters:
        - path: Path to the CSV file containing the adjacency matrix.
        - adj_thresh: Threshold value to convert the matrix to a binary form.

        Returns:
        - adj_matrix: Thresholded adjacency matrix as a torch FloatTensor.
        """
        print(f"Loading adjacency matrix from: {path}...")
        adj_matrix = np.array(pd.read_csv(path, header=None)).astype(float)
        adj_matrix = torch.FloatTensor(np.where(adj_matrix > adj_thresh, 1, 0))  # Thresholding
        print("Adjacency matrix shape:", adj_matrix.shape)
        expected_shape = (
            config.num_nodes + len(config.master_nodes) if config.use_master_nodes else config.num_nodes,
            config.num_nodes + len(config.master_nodes) if config.use_master_nodes else config.num_nodes,
        )
        # Assert the shape matches the expected shape
        assert adj_matrix.shape == expected_shape, f"Unexpected shape: {adj_matrix.shape}. Expected shape: {expected_shape}"
        print("Number of edges:", adj_matrix.sum())
        return adj_matrix

    def plot_adj_matrix(self, adj_matrix, path):
        cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "black"])
        plt.figure()
        plt.imshow(adj_matrix.cpu().numpy(), cmap=cmap)
        plt.colorbar(ticks=[0, 1], label='Adjacency Value')
        plt.title("Visualization of Adjacency Matrix")
        plt.savefig(path)
        plt.close()


def remove_erroneous_columns(config, csv_data, raw_dir):
    """Remove columns that have bimodal distributions."""
    csv_path = os.path.join(raw_dir, config.error_protein_file_name)
    error_proteins_df = pd.read_excel(csv_path)
    # Extract column names under "Plasma" and "CSF"
    modality_columns = error_proteins_df['Plasma'].dropna().tolist()
    csf_columns = error_proteins_df['CSF'].dropna().tolist()
    columns_to_remove = list(set(modality_columns + csf_columns))
    if config.y_val == 'nfl':
        columns_to_remove.extend(
            ['NEFL|P07196|CSF', 'NEFH|P12036|CSF', 'NEFL|P07196|PLASMA', 'NEFH|P12036|PLASMA']
        )
    # Remove the columns
    csv_data = csv_data.drop(columns=columns_to_remove)
    return csv_data


def calculate_adjacency_matrix(config, plasma_protein, save_to):
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects import r
    from rpy2.robjects import r as R
    from rpy2.robjects.packages import importr

    pandas2ri.activate()
    wgcna = importr('WGCNA')
    """Calculate and save adjacency matrix using R's WGCNA."""
    # Convert the input `plasma_protein` (NumPy array) to a pandas DataFrame if necessary
    if not isinstance(plasma_protein, pd.DataFrame):
        plasma_protein_df = pd.DataFrame(plasma_protein)
    else:
        plasma_protein_df = plasma_protein

    # Convert pandas DataFrame to R DataFrame
    r_plasma_protein = pandas2ri.py2rpy(plasma_protein_df)

    # Call R's `pickSoftThreshold` function to determine the soft thresholding power
    soft_threshold_result = wgcna.pickSoftThreshold(
        r_plasma_protein, corFnc="bicor", networkType="signed"
    )
    soft_threshold_power = soft_threshold_result.rx2('powerEstimate')[
        0
    ]  # Extract the estimated power
    if config.modality == 'csf' and all(
        mut in config.mutation for mut in ["C9orf72", "MAPT", "GRN", "CTL"]
    ):
        soft_threshold_power = 9  # went over values w Rowan and selected this.
    print(f"Soft threshold power: {soft_threshold_power}")

    # Call R's `adjacency` function using the chosen power and the desired correlation function (bicor)
    adjacency_matrix_r = wgcna.adjacency(
        r_plasma_protein,
        power=soft_threshold_power,
        type="signed",  # Specify network type
        corFnc="bicor",  # Use biweight midcorrelation
    )

    # Convert the resulting adjacency matrix from R back to a NumPy array
    adjacency_matrix = adjacency_matrix_r

    print("Adjacency matrix shape:", adjacency_matrix.shape)

    # Pad the adjacency matrix if master nodes are used
    if config.use_master_nodes:
        padding_size = len(config.master_nodes)
        adjacency_matrix = np.pad(
            adjacency_matrix,
            pad_width=((0, padding_size), (0, padding_size)),
            mode='constant',
            constant_values=1,
        )
        print("Adjacency matrix shape after padding:", adjacency_matrix.shape)

    # Save the adjacency matrix to the specified file path
    adjacency_df = pd.DataFrame(adjacency_matrix)
    with open(save_to, 'w') as f:
        adjacency_df.to_csv(f, header=None, index=False)
    print(f"Adjacency matrix saved to: {save_to}")


def plot_histogram(data, x_label, save_to):
    plt.hist(data, bins=30, alpha=0.5)
    plt.xlabel(x_label)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {x_label}')
    plt.savefig(save_to, format='jpg')
    plt.close()


def log_transform(train_data, data, log=False):
    if log:
    # Log transformation
        data = np.log(data)
    mean = np.mean(train_data)
    std = np.std(train_data)
    print("mean log train", mean)
    print("std log train", std)
    standardized_log_data = (data - mean) / std
    return standardized_log_data, mean, std


def reverse_log_transform(standardized_log_data, mean, std, log=False):
    # De-standardize the data
    data = standardized_log_data * std + mean
    if log:
        data = torch.exp(data)
    return data
