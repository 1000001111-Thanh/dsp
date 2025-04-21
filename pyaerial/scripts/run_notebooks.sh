#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

function gen_notebook() {
   echo Running notebook $1...
   jupyter nbconvert --execute $1 --output-dir docs/source/notebooks --to notebook
}

# Exit on first error
set -e

# Switch to PROJECT_ROOT directory
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)
PROJECT_ROOT=$(dirname $SCRIPT_DIR)
echo $SCRIPT starting...
cd $PROJECT_ROOT

echo Run notebook examples...
rm -rf docs/source/notebooks
mkdir -p docs/source/notebooks

# pyAerial (cuPHY) examples.
gen_notebook notebooks/example_pusch_simulation.ipynb
gen_notebook notebooks/example_ldpc_coding.ipynb

# ML and LLRNet examples.
gen_notebook notebooks/example_simulated_dataset.ipynb
gen_notebook notebooks/llrnet_dataset_generation.ipynb
gen_notebook notebooks/llrnet_model_training.ipynb
gen_notebook notebooks/example_neural_receiver.ipynb
gen_notebook notebooks/channel_estimation/channel_estimation.ipynb

# Data Lake examples.
gen_notebook notebooks/datalake_channel_estimation.ipynb
gen_notebook notebooks/datalake_pusch_decoding.ipynb

# Finished
echo $SCRIPT finished
