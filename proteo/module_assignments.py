import pandas as pd
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
import numpy as np

def find_modules():
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
    top_150_proteins = protein_data.sum(axis=0).nlargest(150).index
    print("Top 150 proteins by summed value across samples:", top_150_proteins.tolist())
    df_top_proteins = protein_data[top_150_proteins].transpose()
    df_top_proteins.columns = pids
    print(df_top_proteins.shape)

    # Convert data to R format
    r_data = pandas2ri.py2rpy(df_top_proteins)

    # Set parameters, TODO: iterate on these 
    power = 10
    deepSplit = 2
    minModuleSize = 10
    mergeCutHeight = 0.07

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
        print(iteration)
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
    return MEs, kME_table, module_colors


if __name__ == '__main__':
    find_modules()
