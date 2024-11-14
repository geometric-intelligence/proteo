import pandas as pd
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

def find_modules():
    pandas2ri.activate()
    numpy2ri.activate()
    # Import WGCNA package
    wgcna = importr('WGCNA')
    base = importr('base')

    # Load your data (assuming it is a pandas DataFrame)
    data = pd.read_csv("percent_importances.csv")  # Replace with your data file
    print(data.shape)

    # Convert data to R format
    r_data = pandas2ri.py2rpy(data)

    # Set parameters, TODO: iterate on these 
    power = 8
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
        maxBlockSize=r.nrow(r_data) + 1
    )

    # Extract the module labels and module eigengenes
    module_colors = base.unlist(net.rx2("colors"))
    MEs = net.rx2("MEs")  # Module eigengenes

if __name__ == '__main__':
    find_modules()
