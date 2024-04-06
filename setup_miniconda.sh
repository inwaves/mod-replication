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

# Use the environment.yml to create the Conda environment
$HOME/miniconda3/bin/conda env create -f env.yml

# Activate the newly created environment
# Ensure this command is run in a new shell or after re-sourcing your .bashrc
# conda activate mod
