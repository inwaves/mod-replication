#!/bin/bash

# Stop on error
set -e

# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# Install Miniconda in batch mode to the default location
bash miniconda.sh -b -p $HOME/miniconda3

# Remove the installer to save space
rm miniconda.sh

# Initialize Conda for shell interaction
$HOME/miniconda3/bin/conda init

# Close and reopen your shell or source the profile to make 'conda' available
source ~/.bashrc

# Create a new conda environment with Python 3.10
$HOME/miniconda3/bin/conda create -n mod python=3.10 -y

# Activate the newly created environment
# NOTE: This might not work until you open a new shell or source your profile again due to the way conda init modifies the shell initialization scripts.
# $HOME/miniconda3/bin/conda activate .venv