import os
import pandas as pd
import numpy as np
import networkx as nx
import PyWGCNA

data_folder = os.path.abspath("..")
output_folder = "./example_data/input_adjacency_matrix"

data_file = data_folder + "/5xFAD_paper/expressionList.csv"
os.makedirs(output_folder, exist_ok=True)

# WGCNA parameters
# Probably not necessary since 6 is the default
wgcna_power = 9
wgcna_minModuleSize = 10
wgcna_mergeCutHeight = 0.25


# Read data
geneExp = pd.read_csv(data_file) # col = genes, rows = samples 
#Get rid of gene and sample labels
geneExp = geneExp.iloc[:20, 1:20]
# Convert elements to float.
geneExp = geneExp.astype(float)
print(geneExp.iloc[0:,])

# Calculate adjacency matrix.
adjacency = PyWGCNA.WGCNA.adjacency(geneExp, power = wgcna_power, adjacencyType="signed hybrid")
print(adjacency)
# Using adjacency matrix calculate the topological overlap matrix (TOM).
# TOM = PyWGCNA.WGCNA.TOMsimilarity(adjacency)

#Convert to dataframe.
adjacency_df = pd.DataFrame(adjacency)

adjacency_df.to_csv(os.path.join(str(output_folder), "5xFAD_paper_split15_adjacency_matrix.csv"), index=False, header=False)


# check if preprocessing removes empty, if not remove myself
