This repository contains the code required to replicate the results from the following publication (in submission).

> **Video-based biomechanical analysis captures disease-specific movement signatures of different neuromuscular diseases**
>
> Parker S. Ruth\*, Scott D. Uhlrich\*, Constance de Monts, Antoine Falisse, Julie Muccini, Sydney Covitz, John Day, Tina Duong†, Scott Delp†
>
> \*Contributed equally
> †Contributed equally

This code has been tested on MacOS.

## Code Installation

1. Install the [(mini)conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [mamba](https://mamba.readthedocs.io/en/stable/installation/mamba-installation.html) Python environment manager.

2. Open a terminal in the directory where you wish to download the code. Clone the GitHub repository and enter it:

3. Initialize the conda environment for the repository:

```
git clone https://github.com/stanfordnmbl/opencap-fshd-dm-analysis.git
cd opencap-fshd-dm-analysis
conda env create -f environment.yml -n opensim-nmd
cd ..
```


## Data Access

To access the dataset, go to https://doi.org/10.5281/zenodo.13788592 and press "Download All"

Unzip the downloaded file and rename the unzipped folder to `datadir`.

Move the `datadir` directory to be contained inside the same parent directory as the code repository `opencap-fshd-dm-analysis`.

At this point, your working directory should be organized as follows:

```
├── datadir
│   ├── README.md
│   ├── feature_key.csv
│   ├── opencap_data.zip
│   ├── opencap_markers.png
│   ├── participant_info.csv
│   └── video_features.csv
└── opencap-fshd-dm-analysis
    ├── LICENSE
    ├── README.md
    ├── environment.yml
    ├── feature_extraction
    └── figure_generation
```

Unzip the `opencap_data .zip` directory.

```
unzip datadir/opencap_data.zip -d datadir
```

## Feature Extraction

The feature extraction pipeline uses the [snakemake](https://snakemake.github.io/). This is a Makefile-like tool that automates feature extraction across all of the activities. This may take 10 minutes to run or more depending on your machine. Use the lines below to run the pipeline. At any time, the pipeline can be halted by pressing `CTL-C`. When restarted, it snakemake will resume from where it left off.

```
cd feature_extraction
conda activate opensim-nmd
snakemake -c1
```

To parallelize over n cores use `-c[n]` (e.g. `-c4` parallelizes over 4 cores).

To reduce the verbosity of the terminal output,  add the `--quiet` flag.

## Figure Generation

The code for figure generation code is in Jupyter notebooks. Use these commands to launch Jupyter Lab.

```
cd feature_generation
conda activate opensim-nmd
jupyter lab
```

Intraclass correlation is calculated in `fig3_icc.ipynb` by calling an R script from Python via a shell. On MacOS, the `Rscript` command can be acquired by installing [RStudio](https://posit.co/downloads/). If R is not available on the computer, an alternative is to run the `fig3_icc_no_Rscript.ipynb` file, which reads all of the ICC from a pre-computed CSV file.

