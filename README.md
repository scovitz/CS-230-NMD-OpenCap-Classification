This repository contains the code required to replicate the results from the following publication (in submission).

> **Video-based biomechanical analysis captures disease-specific movement signatures of different neuromuscular diseases**
>
> Parker S. Ruth\*, Scott D. Uhlrich\*, Constance de Monts, Antoine Falisse, Julie Muccini, Paxton Ataide, John Day, Tina Duong, Scott Delp
>
> \*Contributed equally

This code has been tested on MacOS.

## Code Installation

1. Install  [(mini)conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [mamba](https://mamba.readthedocs.io/en/stable/installation/mamba-installation.html) Python environment manager

2. Initialize the conda environment for the repository

```
conda env create -f environment.yml -n opensim-nmd
```

## Data Access

The dataset can be accessed from https://doi.org/10.5281/zenodo.13788592.

In the top level of the code repository, create a soft link to the dataset. This can be done by running the line below with `[path_to_dataset]` replaced by an absolute or relative path to the downloaded data.

```
ln -s [path_to_dataset] datadir
```

## Feature Extraction

The feature extraction pipeline uses the [snakemake](https://snakemake.github.io/).

```
cd feature_extraction
conda activate opensim-nmd
snakemake -c1
```

To parallelize over n cores use `-c[n]` (e.g. `-c4` parallelizes over 4 cores).

## Figure Generation

The code for figure generation code is in Jupyter notebooks. Use these commands to launch Jupyter Lab.

```
cd feature_generation
conda activate opensim-nmd
jupyter lab
```

Intraclass correlation is calculated in `fig3_icc.ipynb` by calling an R script from Python via a shell. On MacOS, the `Rscript` command can be acquired by installing [RStudio](https://posit.co/downloads/). If R is not available on the computer, an alternative is to run the `fig3_icc_no_Rscript.ipynb` file, which reads all of the ICC from a pre-computed CSV file.
