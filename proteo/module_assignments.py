# General-purpose imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# R and Python interoperability
from rpy2.robjects import r, pandas2ri, numpy2ri
from rpy2.robjects.packages import importr

# Scikit-learn preprocessing and clustering
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA, NMF
from sklearn.mixture import GaussianMixture

# Statistical analysis
from scipy.stats import pearsonr

# UMAP dimensionality reduction
import umap.umap_ as umap



def find_modules_wgcna():
    pandas2ri.activate()
    numpy2ri.activate()
    # Import WGCNA package
    wgcna = importr('WGCNA')
    base = importr('base')

    # Load your data (assuming it is a pandas DataFrame)
    df = pd.read_csv("percent_importances.csv")  # Replace with your data file
    # Step 1: Select only protein columns (assuming they start from the 4th column onward)
    pids = df.iloc[:, 0]
    protein_data = df.iloc[:, 4:]
    top_300_proteins = protein_data.sum(axis=0).nlargest(7258).index
    print("Top 300 proteins by summed value across samples:", top_300_proteins.tolist())
    df_top_proteins = protein_data[top_300_proteins].transpose()
    df_top_proteins.columns = pids
    print(df_top_proteins.shape)

    # Convert data to R format
    r_data = pandas2ri.py2rpy(df_top_proteins)

    soft_threshold_result = wgcna.pickSoftThreshold(df_top_proteins, powerVector = np.arange(1, 20.5, 0.5), corFnc="bicor", networkType="signed")
    # Set parameters, TODO: iterate on these 
    # PICK SOFT THRESHOLD TO GET POWEr, mean k below 100, r sq above .8
    power = 12.5
    deepSplit = 2
    minModuleSize = 10
    mergeCutHeight = 0.15

    # Call the blockwiseModules function
    net = wgcna.blockwiseModules(
        r_data,
        power=power,
        deepSplit=deepSplit,
        minModuleSize=minModuleSize,
        mergeCutHeight=mergeCutHeight,
        networkType="signed",
        TOMType="signed",
        pamStage=True,
        pamRespectsDendro=True,
        TOMDenom="mean",
        corType="bicor",
        numericLabels=True,
        saveTOMs=False,
        maxBlockSize=r.nrow(r_data)[0] + 1
    )

    # Extract the module labels and module eigengenes
    print("length of assignments",  net.rx2("colors").shape)
    module_colors = net.rx2("colors")  # Get the numeric labels
    #module_colors = wgcna.labels2colors(module_numeric_labels)
    print("Module colors (assignments):", module_colors)
    MEs = net.rx2("MEs")  # Module eigengenes
    # Compute kME table (correlation between each person and each module eigengene)
    kME_table = wgcna.signedKME(r_data, MEs, corFnc="bicor")
    print(kME_table)
        # Post-hoc Cleanup (iteratively reassign proteins as needed)
    max_iterations = 100
    for iteration in range(max_iterations):
        changed = False  # Track changes for convergence
        
        # Step 2a: Remove proteins with low intramodular kME (< 0.30)
        for i in range(len(module_colors)):
            module_color = int(module_colors[i])  # Convert to integer to use as index
            if module_color != 0 and kME_table.rx(i + 1, module_color + 1)[0] < 0.30:  # Check intramodular kME
                module_colors[i] = 0  # Assign to '0' as a grey equivalent
                changed = True

        # Step 2b: Reassign grey proteins to modules if max kME > 0.30
        gray_proteins = np.where(module_colors == 0)[0]
        for i in gray_proteins:
            max_kME = max(kME_table.rx(i + 1, True))  # Get maximum kME for this protein
            best_module = np.argmax(kME_table.rx(i + 1, True))  # Best module index (already 0-based)
            if max_kME > 0.30:
                module_colors[i] = best_module
                changed = True

        # Step 2c: Reassign proteins with intramodular kME > 0.10 below max kME
        for i in range(len(module_colors)):
            module_color = int(module_colors[i])
            if module_color != 0:
                max_kME = max(kME_table.rx(i + 1, True))
                if kME_table.rx(i + 1, module_color + 1)[0] < max_kME - 0.10:
                    best_module = np.argmax(kME_table.rx(i + 1, True))  # Reassign to highest kME module
                    module_colors[i] = best_module
                    changed = True

        # Recalculate MEs and kME table after reassignment
        MEs = wgcna.moduleEigengenes(r_data, colors=module_colors).rx2("eigengenes")
        new_kME_table = wgcna.signedKME(r_data, MEs, corFnc="bicor")

        # Check if kME table is stable (no changes), then break
        if not changed:
            print(f"Convergence reached after {iteration + 1} iterations.")
            break
        kME_table = new_kME_table
        
    print(module_colors)
    print(kME_table)
    # Count the occurrences of each module
    unique, counts = np.unique(module_colors, return_counts=True)

    # Calculate the percentages and print them
    percentages = {module: (count / len(module_colors)) * 100 for module, count in zip(unique, counts)}

    for module, percentage in percentages.items():
        print(f"Module {int(module)}: {percentage:.2f}%")
    print(type(MEs))
    return MEs, kME_table, module_colors

def visualize_wgcna_results(MEs, module_colors, protein_data):
    # Convert R module_colors to NumPy array for easier handling
    module_colors = np.array(module_colors)

    # Perform PCA on the protein data
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(protein_data)

    # Create a scatter plot of the PCA results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        c=module_colors,
        cmap='rainbow',  # Use a colormap to visualize modules
        s=50,
        alpha=0.8
    )

    plt.title("WGCNA Clustering Visualization (PCA - 2D)", fontsize=16)
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)
    plt.colorbar(scatter, label="Module Color")
    plt.savefig("wgcna_pca_visualization.png")
    plt.show()

    # Visualize the module eigengenes (optional)
    plt.figure(figsize=(10, 6))
    plt.plot(MEs)
    plt.title("Module Eigengenes")
    plt.xlabel("Samples")
    plt.ylabel("Eigengene Value")
    plt.legend([f"Module {i}" for i in range(MEs.columns.size)])
    plt.savefig("module_eigengenes.png")
    plt.show()

def find_modules_kmeans_umap():
    # Load your data
    df = pd.read_csv("percent_importances.csv")  # Replace with your data file
    
    # Step 1: Select only protein columns (assuming they start from the 4th column onward)
    pids = df.iloc[:, 0]  # First column assumed to be person IDs
    protein_data = df.iloc[:, 4:]  # Proteins as columns, people as rows
    
    # Select top 150 proteins by summed values
    top_300_proteins = protein_data.sum(axis=0).nlargest(1000).index
    print("Top 300 proteins by summed value across samples:", top_300_proteins.tolist())

    # Subset the data to include only the top proteins
    df_top_proteins = protein_data[top_300_proteins]
    df_top_proteins.index = pids  # Set person IDs as index
    print("Shape of data for clustering (people x proteins):", df_top_proteins.shape)
    
    # Scale the data using MinMaxScaler
    #scaler = MinMaxScaler()
    #df_top_proteins = scaler.fit_transform(df_top_proteins)
    #print("Data successfully scaled using MinMaxScaler.")

    # Plot the elbow graph to determine the optimal number of clusters
    distortions = []
    K = range(1, 30)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=100)
        kmeans.fit(df_top_proteins)
        distortions.append(kmeans.inertia_)

    # Elbow method visualization
    plt.figure(figsize=(8, 5))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal k')
    plt.savefig('elbow_plot.png')
    plt.show()

    # Define the optimal number of clusters (adjust based on the elbow plot)
    optimal_k = 5  # Replace with the actual elbow point

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(df_top_proteins)

    # Add cluster assignments to the original DataFrame
    df['Cluster'] = clusters

    # UMAP for 2D visualization
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced_data_2d = reducer.fit_transform(df_top_proteins)

    # Plot UMAP visualization
    plt.figure(figsize=(8, 6))
    for cluster in range(optimal_k):
        cluster_points = reduced_data_2d[clusters == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')
    
    plt.title('KMeans Clustering Visualization (UMAP - 2D)')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend()
    plt.savefig('umap_2d_visualization.png')
    plt.show()

    # Calculate the breakdown metrics
    cluster_stats = {}

    for cluster in range(optimal_k):
        cluster_data = df[df['Cluster'] == cluster]
        
        # Count of each sex
        sex_counts = cluster_data['SEX'].value_counts().to_dict()
        
        # Mean age
        mean_age = cluster_data['AGE'].mean()
        
        # Count of each mutation
        mutation_counts = cluster_data['Mutation'].value_counts().to_dict()
        
        # Store the metrics
        cluster_stats[cluster] = {
            'Sex Counts': sex_counts,
            'Mean Age': mean_age,
            'Mutation Counts': mutation_counts,
            'Total Members': len(cluster_data)
        }

    # Print the cluster statistics
    print("\nCluster Statistics:")
    for cluster, stats in cluster_stats.items():
        print(f"\nCluster {cluster}:")
        print(f"  Total Members: {stats['Total Members']}")
        print(f"  Sex Counts: {stats['Sex Counts']}")
        print(f"  Mean Age: {stats['Mean Age']:.2f}")
        print(f"  Mutation Counts: {stats['Mutation Counts']}")

    # Save cluster assignments to a CSV file
    df.to_csv("people_cluster_assignments.csv", index=False)


def find_modules_kmeans_with_multiple_increments_variance():
    # Load your data
    df = pd.read_csv("percent_importances.csv")  # Replace with your data file
    
    # Step 1: Select only protein columns (assuming they start from the 4th column onward)
    pids = df.iloc[:, 0]  # First column assumed to be person IDs
    protein_data = df.iloc[:, 4:]  # Proteins as columns, people as rows

    # Compute variance for each protein
    protein_variances = protein_data.var(axis=0)
    max_proteins = protein_variances.shape[0]
    increments = range(100, max_proteins + 1, 100)  # Steps of 100 proteins

    # Determine grid layout for subplots
    num_increments = len(increments)
    rows = int(np.ceil(num_increments / 4))  # 4 columns per row
    cols = min(4, num_increments)

    # Prepare subplots for elbow plots
    fig_elbow, axes_elbow = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    fig_umap, axes_umap = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

    # Flatten axes for easier indexing
    axes_elbow = axes_elbow.flatten()
    axes_umap = axes_umap.flatten()

    # Loop through each increment of top proteins
    for idx, num_proteins in enumerate(increments):
        print(f"Clustering with top {num_proteins} proteins...")

        # Select the top `num_proteins` by variance
        top_proteins = protein_variances.nlargest(num_proteins).index
        df_top_proteins = protein_data[top_proteins]
        df_top_proteins.index = pids  # Set person IDs as index

        # Scale the data using MinMaxScaler
        #scaler = MinMaxScaler()
        #df_top_proteins_scaled = scaler.fit_transform(df_top_proteins)

        # Calculate distortions for the Elbow Method
        distortions = []
        K = range(1, 30)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(df_top_proteins)
            distortions.append(kmeans.inertia_)

        # Plot the elbow graph for the current increment
        ax_elbow = axes_elbow[idx]
        ax_elbow.plot(K, distortions, 'bx-')
        ax_elbow.set_title(f'Top {num_proteins} Proteins')
        ax_elbow.set_xlabel('Number of Clusters (k)')
        if idx % cols == 0:
            ax_elbow.set_ylabel('Distortion')

        # Perform KMeans clustering with an optimal k (hardcoded or inferred)
        optimal_k = 6  # Replace with the actual elbow point or a dynamic method
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(df_top_proteins)

        # Add cluster assignments to the original DataFrame
        df[f'Cluster_{num_proteins}'] = clusters

        # UMAP for 2D visualization
        reducer = umap.UMAP(n_components=2, random_state=42)
        print(df_top_proteins[:0])
        reduced_data_2d = reducer.fit_transform(df_top_proteins)

        # Plot UMAP with cluster colors
        ax_umap = axes_umap[idx]
        for cluster in range(optimal_k):
            cluster_points = reduced_data_2d[clusters == cluster]
            ax_umap.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}', alpha=0.7)
        ax_umap.set_title(f'Top {num_proteins} Proteins')
        ax_umap.set_xlabel('UMAP Dimension 1')
        if idx % cols == 0:
            ax_umap.set_ylabel('UMAP Dimension 2')

    # Hide unused subplots
    for ax in axes_elbow[num_increments:]:
        ax.axis('off')
    for ax in axes_umap[num_increments:]:
        ax.axis('off')

    # Finalize and display all elbow plots
    fig_elbow.tight_layout()
    fig_elbow.suptitle('Elbow Plots for Different Protein Increments', fontsize=16)
    fig_elbow.savefig('elbow_plots_comparison.png')
    plt.show()

    # Finalize and display all UMAP plots
    fig_umap.tight_layout()
    fig_umap.suptitle('UMAP Visualizations for Different Protein Increments', fontsize=16)
    fig_umap.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title="Clusters")
    fig_umap.savefig('umap_plots_comparison.png')
    plt.show()

def find_modules_dbscan():
    # Load your data
    df = pd.read_csv("percent_importances.csv")  # Replace with your data file

    # Step 1: Select only protein columns (assuming they start from the 4th column onward)
    pids = df.iloc[:, 0]  # First column assumed to be person IDs
    protein_data = df.iloc[:, 4:]  # Proteins as columns, people as rows

    # Select top 300 proteins by summed values
    top_300_proteins = protein_data.sum(axis=0).nlargest(300).index
    print("Top 300 proteins by summed value across samples:", top_300_proteins.tolist())

    # Subset the data to include only the top proteins
    df_top_proteins = protein_data[top_300_proteins]
    df_top_proteins.index = pids  # Set person IDs as index
    print("Shape of data for clustering (people x proteins):", df_top_proteins.shape)

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_top_proteins)
    print("Data successfully scaled using MinMaxScaler.")

    # Plot the k-distance graph to determine a good eps
    def plot_k_distance(data, k=4, save_path="k_distance_plot.png"):
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(data)
        distances, indices = neighbors_fit.kneighbors(data)
        distances = np.sort(distances[:, k-1], axis=0)  # k-th nearest neighbor distances
        plt.figure(figsize=(8, 5))
        plt.plot(distances)
        plt.xlabel('Points sorted by distance')
        plt.ylabel(f'{k}-th Nearest Neighbor Distance')
        plt.title('k-Distance Graph')
        plt.savefig(save_path)  # Save the plot
        print(f"k-Distance plot saved as: {save_path}")
        plt.show()

    # Generate and save the k-distance plot
    plot_k_distance(scaled_data, k=4, save_path="k_distance_plot.png")

    # Set DBSCAN parameters (adjust eps based on k-distance plot)
    eps = 2.75  # Replace this with the elbow point from the k-distance graph
    min_samples = 10

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    cluster_labels = dbscan.fit_predict(scaled_data)

    # Add cluster assignments to the original DataFrame
    df['Cluster'] = cluster_labels

    # Identify the number of clusters (excluding noise)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")

    # Visualize the clusters in 2D using PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_data)

    plt.figure(figsize=(10, 7))
    for cluster in set(cluster_labels):
        cluster_points = reduced_data[cluster_labels == cluster]
        if cluster == -1:  # Noise points
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color='gray', label='Noise', alpha=0.6)
        else:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('DBSCAN Clustering Visualization (PCA - 2D)')
    plt.legend()
    plt.savefig('dbscan_clustering_pca.png')
    plt.show()

def find_modules_kmeans_with_pca_and_umap():
    # Load your data
    df = pd.read_csv("percent_importances.csv")  # Replace with your data file
    
    # Step 1: Select only protein columns (assuming they start from the 4th column onward)
    pids = df.iloc[:, 0]  # First column assumed to be person IDs
    protein_data = df.iloc[:, 4:]  # Proteins as columns, people as rows
    print("Shape of original protein data (people x proteins):", protein_data.shape)
    
    # Step 2: Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    scaled_protein_data = scaler.fit_transform(protein_data)
    print("Minimum value after scaling:", scaled_protein_data.min())
    print("Maximum value after scaling:", scaled_protein_data.max())
    print("Data successfully scaled using MinMaxScaler.")

    # Step 3: Perform PCA to preserve 95% variance
    pca = PCA(n_components=0.95, random_state=42)  # Automatically choose components to preserve 95% variance
    pca_components = pca.fit_transform(scaled_protein_data)
    print(f"PCA reduced data to {pca_components.shape[1]} dimensions, preserving 95% variance.")

    # Step 4: Plot the elbow graph to determine the optimal number of clusters
    distortions = []
    K = range(1, 30)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(pca_components)
        distortions.append(kmeans.inertia_)

    # Elbow method visualization
    plt.figure(figsize=(8, 5))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal k')
    plt.savefig('pca_elbow_plot.png')
    plt.show()

    # Step 5: Define the optimal number of clusters (adjust based on the elbow plot)
    optimal_k = 7  # Replace with the actual elbow point

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(pca_components)

    # Add cluster assignments to the original DataFrame
    df['Cluster'] = clusters

    # Step 6: UMAP for 2D visualization
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced_data_2d = reducer.fit_transform(pca_components)

    # Plot UMAP with cluster colors
    plt.figure(figsize=(8, 6))
    for cluster in range(optimal_k):
        cluster_points = reduced_data_2d[clusters == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')
    
    plt.title('KMeans Clustering Visualization (UMAP - 2D)')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend()
    plt.savefig('umap_2d_visualization.png')
    plt.show()

    # Step 7: Calculate the breakdown metrics
    cluster_stats = {}

    for cluster in range(optimal_k):
        cluster_data = df[df['Cluster'] == cluster]
        
        # Count of each sex
        sex_counts = cluster_data['SEX'].value_counts().to_dict()
        
        # Mean age
        mean_age = cluster_data['AGE'].mean()
        
        # Count of each mutation
        mutation_counts = cluster_data['Mutation'].value_counts().to_dict()
        
        # Store the metrics
        cluster_stats[cluster] = {
            'Sex Counts': sex_counts,
            'Mean Age': mean_age,
            'Mutation Counts': mutation_counts,
            'Total Members': len(cluster_data)
        }

    # Print the cluster statistics
    print("\nCluster Statistics:")
    for cluster, stats in cluster_stats.items():
        print(f"\nCluster {cluster}:")
        print(f"  Total Members: {stats['Total Members']}")
        print(f"  Sex Counts: {stats['Sex Counts']}")
        print(f"  Mean Age: {stats['Mean Age']:.2f}")
        print(f"  Mutation Counts: {stats['Mutation Counts']}")

    # Save cluster assignments to a CSV file
    df.to_csv("people_cluster_assignments_with_pca.csv", index=False)

def find_modules_nmf_with_multiple_increments_and_visualizations():
    # Load your data
    df = pd.read_csv("percent_importances.csv")  # Replace with your data file
    
    # Step 1: Select only protein columns (assuming they start from the 4th column onward)
    pids = df.iloc[:, 0]  # First column assumed to be person IDs
    protein_data = df.iloc[:, 4:]  # Proteins as columns, people as rows

    # Compute variance for each protein
    protein_variances = protein_data.var(axis=0)
    max_proteins = protein_variances.shape[0]
    increments = range(100, max_proteins + 1, 100)  # Steps of 100 proteins

    # Determine grid layout for subplots
    num_increments = len(increments)
    rows = int(np.ceil(num_increments / 4))  # 4 columns per row
    cols = min(4, num_increments)

    # Prepare subplots for reconstruction error, UMAP, and silhouette plots
    fig_error, axes_error = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    fig_umap, axes_umap = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    fig_silhouette, axes_silhouette = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

    # Flatten axes for easier indexing
    axes_error = axes_error.flatten()
    axes_umap = axes_umap.flatten()
    axes_silhouette = axes_silhouette.flatten()

    # Loop through each increment of top proteins
    for idx, num_proteins in enumerate(increments):
        print(f"Clustering with top {num_proteins} proteins...")

        # Select the top `num_proteins` by variance
        top_proteins = protein_variances.nlargest(num_proteins).index
        df_top_proteins = protein_data[top_proteins]
        df_top_proteins.index = pids  # Set person IDs as index

        # Scale the data using MinMaxScaler
        scaler = MinMaxScaler()
        df_top_proteins_scaled = scaler.fit_transform(df_top_proteins)

        # Perform NMF clustering
        n_components = 4  # Adjust the number of components as needed
        nmf_model = NMF(n_components=n_components, random_state=42)
        W = nmf_model.fit_transform(df_top_proteins_scaled)  # Latent representation
        H = nmf_model.components_  # Basis vectors
        reconstruction_error = nmf_model.reconstruction_err_

        # Assign clusters based on the maximum latent feature
        clusters = np.argmax(W, axis=1)
        df[f'Cluster_{num_proteins}'] = clusters

        # Compute silhouette score
        silhouette_avg = silhouette_score(df_top_proteins_scaled, clusters)

        # Plot the reconstruction error for the current increment
        ax_error = axes_error[idx]
        ax_error.bar([1], [reconstruction_error], color='blue')
        ax_error.set_title(f'Top {num_proteins} Proteins')
        ax_error.set_xlabel('Reconstruction Error')
        if idx % cols == 0:
            ax_error.set_ylabel('Error Magnitude')

        # Plot the silhouette score for the current increment
        ax_silhouette = axes_silhouette[idx]
        ax_silhouette.bar([1], [silhouette_avg], color='green')
        ax_silhouette.set_title(f'Top {num_proteins} Proteins')
        ax_silhouette.set_xlabel('Silhouette Score')
        if idx % cols == 0:
            ax_silhouette.set_ylabel('Score Magnitude')

        # UMAP for 2D visualization
        reducer = umap.UMAP(n_components=2, random_state=42)
        reduced_data_2d = reducer.fit_transform(W)

        # Plot UMAP with cluster colors
        ax_umap = axes_umap[idx]
        for cluster in np.unique(clusters):
            cluster_points = reduced_data_2d[clusters == cluster]
            ax_umap.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}', alpha=0.7)
        ax_umap.set_title(f'Top {num_proteins} Proteins')
        ax_umap.set_xlabel('UMAP Dimension 1')
        if idx % cols == 0:
            ax_umap.set_ylabel('UMAP Dimension 2')
        ax_umap.legend(loc='upper right', fontsize='small')

    # Hide unused subplots
    for ax in axes_error[num_increments:]:
        ax.axis('off')
    for ax in axes_umap[num_increments:]:
        ax.axis('off')
    for ax in axes_silhouette[num_increments:]:
        ax.axis('off')

    # Finalize and display all reconstruction error plots
    fig_error.tight_layout()
    fig_error.suptitle('Reconstruction Errors for Different Protein Increments', fontsize=16)
    fig_error.savefig('reconstruction_errors_comparison.png')
    plt.show()

    # Finalize and display all silhouette score plots
    fig_silhouette.tight_layout()
    fig_silhouette.suptitle('Silhouette Scores for Different Protein Increments', fontsize=16)
    fig_silhouette.savefig('silhouette_scores_comparison.png')
    plt.show()

    # Finalize and display all UMAP plots
    fig_umap.tight_layout()
    fig_umap.suptitle('UMAP Visualizations for Different Protein Increments', fontsize=16)
    fig_umap.savefig('umap_plots_nmf_comparison.png')
    plt.show()

def find_modules_kmeans_with_pca_and_multiple_increments():
    # Load your data
    df = pd.read_csv("percent_importances.csv")  # Replace with your data file

    # Step 1: Select only protein columns (assuming they start from the 4th column onward)
    pids = df.iloc[:, 0]  # First column assumed to be person IDs
    protein_data = df.iloc[:, 4:]  # Proteins as columns, people as rows

    # Compute variance for each protein
    protein_variances = protein_data.var(axis=0)
    max_proteins = protein_variances.shape[0]
    increments = range(100, max_proteins + 1, 100)  # Steps of 100 proteins

    # Determine grid layout for subplots
    num_increments = len(increments)
    rows = int(np.ceil(num_increments / 4))  # 4 columns per row
    cols = min(4, num_increments)

    # Prepare subplots for elbow plots and PCA visualizations
    fig_elbow, axes_elbow = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    fig_pca, axes_pca = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

    # Flatten axes for easier indexing
    axes_elbow = axes_elbow.flatten()
    axes_pca = axes_pca.flatten()

    # Loop through each increment of top proteins
    for idx, num_proteins in enumerate(increments):
        print(f"Clustering with top {num_proteins} proteins...")

        # Select the top `num_proteins` by variance
        top_proteins = protein_variances.nlargest(num_proteins).index
        df_top_proteins = protein_data[top_proteins]
        df_top_proteins.index = pids  # Set person IDs as index

        # Scale the data using MinMaxScaler
        scaler = MinMaxScaler()
        df_top_proteins_scaled = scaler.fit_transform(df_top_proteins)

        # Calculate distortions for the Elbow Method
        distortions = []
        K = range(1, 30)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(df_top_proteins_scaled)
            distortions.append(kmeans.inertia_)

        # Plot the elbow graph for the current increment
        ax_elbow = axes_elbow[idx]
        ax_elbow.plot(K, distortions, 'bx-')
        ax_elbow.set_title(f'Top {num_proteins} Proteins')
        ax_elbow.set_xlabel('Number of Clusters (k)')
        if idx % cols == 0:
            ax_elbow.set_ylabel('Distortion')

        # Perform KMeans clustering with an optimal k (hardcoded or inferred)
        optimal_k = 4  # Replace with the actual elbow point or a dynamic method
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(df_top_proteins_scaled)

        # Add cluster assignments to the original DataFrame
        df[f'Cluster_{num_proteins}'] = clusters

        # PCA for 2D visualization
        pca = PCA(n_components=2)
        pca_data_2d = pca.fit_transform(df_top_proteins_scaled)
        explained_variance = pca.explained_variance_ratio_ * 100  # Convert to percentages

        # Plot PCA with cluster colors
        ax_pca = axes_pca[idx]
        for cluster in range(optimal_k):
            cluster_points = pca_data_2d[clusters == cluster]
            ax_pca.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}', alpha=0.7)
        ax_pca.set_title(f'Top {num_proteins} Proteins')
        ax_pca.set_xlabel(f'PC1 ({explained_variance[0]:.2f}% Variance)')
        if idx % cols == 0:
            ax_pca.set_ylabel(f'PC2 ({explained_variance[1]:.2f}% Variance)')
        ax_pca.legend(loc='upper right', fontsize='small')

    # Hide unused subplots
    for ax in axes_elbow[num_increments:]:
        ax.axis('off')
    for ax in axes_pca[num_increments:]:
        ax.axis('off')

    # Finalize and display all elbow plots
    fig_elbow.tight_layout()
    fig_elbow.suptitle('Elbow Plots for Different Protein Increments', fontsize=16)
    fig_elbow.savefig('elbow_plots_comparison.png')
    plt.show()

    # Finalize and display all PCA plots
    fig_pca.tight_layout()
    fig_pca.suptitle('PCA Visualizations for Different Protein Increments', fontsize=16)
    fig_pca.savefig('pca_plots_comparison.png')
    plt.show()

def nmf_clustering_stability_analysis():
    # Load your data
    df = pd.read_csv("percent_importances.csv")  # Replace with your data file
    
    # Step 1: Select only protein columns (assuming they start from the 4th column onward)
    protein_data = df.iloc[:, 4:]  # Proteins as columns, people as rows
    print("Shape of original protein data (people x proteins):", protein_data.shape)
    
    # Step 2: Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    scaled_protein_data = scaler.fit_transform(protein_data)
    print("Data successfully scaled using MinMaxScaler.")
    
    # Parameters
    max_clusters = 10  # Maximum number of clusters to test
    runs = 50  # Number of NMF runs for stability analysis
    random_fit = []  # Store random fit for comparison
    metrics = []  # Metrics table

    # Loop over the number of clusters
    for n_clusters in range(2, max_clusters + 1):
        print(f"Evaluating {n_clusters} clusters...")

        # Run NMF multiple times to calculate co-phonetic coefficient
        all_clusters = []
        reconstruction_errors = []
        W_all = []  # Store W matrices for co-phonetic coefficient calculation

        for i in range(runs):
            nmf = NMF(n_components=n_clusters, init='random', random_state=i, max_iter=1000)
            W = nmf.fit_transform(scaled_protein_data)
            H = nmf.components_
            reconstruction_errors.append(nmf.reconstruction_err_)
            W_all.append(W)
            clusters = np.argmax(W, axis=1)
            all_clusters.append(clusters)

        # Calculate the co-phonetic coefficient
        co_phonetic_matrix = np.zeros((len(all_clusters), len(all_clusters)))
        for i in range(len(all_clusters)):
            for j in range(i, len(all_clusters)):
                co_phonetic_matrix[i, j] = pearsonr(all_clusters[i], all_clusters[j])[0]
        co_phonetic_coefficient = np.mean(co_phonetic_matrix)

        # Calculate silhouette width
        final_clusters = np.argmax(W_all[-1], axis=1)  # Final clustering solution
        silhouette_avg = silhouette_score(scaled_protein_data, final_clusters)

        # Compare fit with random solutions
        random_nmf = NMF(n_components=n_clusters, init='random', random_state=None, max_iter=500)
        random_fit_value = random_nmf.fit(scaled_protein_data).reconstruction_err_
        random_fit.append(random_fit_value)
        fit_ratio = np.mean(reconstruction_errors) / random_fit_value

        # Store metrics for this number of clusters
        metrics.append({
            "Number of Clusters": n_clusters,
            "Co-phonetic Correlation": co_phonetic_coefficient,
            "Random Fit": random_fit_value,
            "Observed Fit": np.mean(reconstruction_errors),
            "Fold Improved Fit": fit_ratio,
            "Silhouette Width": silhouette_avg
        })

    # Create a DataFrame to store the results
    metrics_df = pd.DataFrame(metrics)
    metrics_df["% Variance Explained Over Lower Cluster Solution"] = (
        metrics_df["Observed Fit"].pct_change(periods=1) * -1 * 100
    )
    print(metrics_df)

    # Determine optimal number of clusters
    optimal_clusters = metrics_df[
        (metrics_df["Co-phonetic Correlation"] > 0.7) &
        (metrics_df["Fold Improved Fit"] > 2) &
        (metrics_df["Silhouette Width"] > 0.5)
    ]["Number of Clusters"]
    print(f"Optimal number of clusters: {optimal_clusters.tolist()}")

    # Visualize metrics
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df["Number of Clusters"], metrics_df["Co-phonetic Correlation"], marker='o', label='Co-phonetic Correlation')
    plt.plot(metrics_df["Number of Clusters"], metrics_df["Fold Improved Fit"], marker='o', label='Fold Improved Fit')
    plt.plot(metrics_df["Number of Clusters"], metrics_df["Silhouette Width"], marker='o', label='Silhouette Width')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Metric Value')
    plt.title('Metrics for Cluster Stability and Fit')
    plt.legend()
    plt.grid(True)
    plt.savefig('nmf_metrics_plot.png')
    plt.show()

    return metrics_df


def find_modules_kmeans_with_multiple_increments_sum():
    # Load your data
    df = pd.read_csv("percent_importances.csv")  # Replace with your data file
    
    # Step 1: Select only protein columns (assuming they start from the 4th column onward)
    pids = df.iloc[:, 0]  # First column assumed to be person IDs
    protein_data = df.iloc[:, 4:]  # Proteins as columns, people as rows

    # Compute sums for each protein
    protein_sums = protein_data.sum(axis=0)
    max_proteins = protein_sums.shape[0]
    increments = list(range(100, max_proteins + 1, 100))  # Steps of 100 proteins
    increments.append(7258)

    # Determine grid layout for subplots
    num_increments = len(increments)
    rows = int(np.ceil(num_increments / 4))  # 4 columns per row
    cols = min(4, num_increments)

    # Prepare subplots for elbow plots
    fig_elbow, axes_elbow = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    fig_umap, axes_umap = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

    # Flatten axes for easier indexing
    axes_elbow = axes_elbow.flatten()
    axes_umap = axes_umap.flatten()

    # Loop through each increment of top proteins
    for idx, num_proteins in enumerate(increments):
        print(f"Clustering with top {num_proteins} proteins...")

        # Select the top `num_proteins` by sum
        top_proteins = protein_sums.nlargest(num_proteins).index
        df_top_proteins = protein_data[top_proteins]
        df_top_proteins.index = pids  # Set person IDs as index

        # Scale the data using MinMaxScaler
        #scaler = MinMaxScaler()
        #df_top_proteins_scaled = scaler.fit_transform(df_top_proteins)

        # Calculate distortions for the Elbow Method
        distortions = []
        K = range(1, 30)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(df_top_proteins)
            distortions.append(kmeans.inertia_)

        # Plot the elbow graph for the current increment
        ax_elbow = axes_elbow[idx]
        ax_elbow.plot(K, distortions, 'bx-')
        ax_elbow.set_title(f'Top {num_proteins} Proteins')
        ax_elbow.set_xlabel('Number of Clusters (k)')
        if idx % cols == 0:
            ax_elbow.set_ylabel('Distortion')

        # Perform KMeans clustering with an optimal k (hardcoded or inferred)
        optimal_k = 6  # Replace with the actual elbow point or a dynamic method
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(df_top_proteins)

        # Add cluster assignments to the original DataFrame
        df[f'Cluster_{num_proteins}'] = clusters

        # UMAP for 2D visualization
        reducer = umap.UMAP(n_components=2, random_state=42)
        reduced_data_2d = reducer.fit_transform(df_top_proteins)

        # Plot UMAP with cluster colors
        ax_umap = axes_umap[idx]
        for cluster in range(optimal_k):
            cluster_points = reduced_data_2d[clusters == cluster]
            ax_umap.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}', alpha=0.7)
        ax_umap.set_title(f'Top {num_proteins} Proteins')
        ax_umap.set_xlabel('UMAP Dimension 1')
        if idx % cols == 0:
            ax_umap.set_ylabel('UMAP Dimension 2')

    # Hide unused subplots
    for ax in axes_elbow[num_increments:]:
        ax.axis('off')
    for ax in axes_umap[num_increments:]:
        ax.axis('off')

    # Finalize and display all elbow plots
    fig_elbow.tight_layout()
    fig_elbow.suptitle('Elbow Plots for Different Protein Increments', fontsize=16)
    fig_elbow.savefig('elbow_plots_comparison.png')
    plt.show()

    # Finalize and display all UMAP plots
    fig_umap.tight_layout()
    fig_umap.suptitle('UMAP Visualizations for Different Protein Increments', fontsize=16)
    fig_umap.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title="Clusters")
    fig_umap.savefig('umap_plots_comparison.png')
    plt.show()


if __name__ == '__main__':
    find_modules_kmeans_with_multiple_increments_sum()