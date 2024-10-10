# AI Proteomics

# Set up

```shell
$ conda create -n proteo --file conda-linux-64.lock
$ conda activate proteo
$ R
$ if (!requireNamespace("BiocManager", quietly = TRUE))
$    install.packages("BiocManager")

$ BiocManager::install("WGCNA")
$ q()
$ poetry install --with=dev,gpu
```

# Dev

Only run if changes are made to the environment files.

To recreate the conda lock, after modifying conda.yaml:
```shell
pip install conda-lock
make conda-linux-64.lock
```

To recreate the poetry lock, after modifying pyproject.toml:
```shell
make poetry.lock
```
Note: you need to have poetry installed for this, which means that you need to be in proteo.

And then go back to set up step above.