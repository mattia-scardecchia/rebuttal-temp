#!/bin/bash

# paths
export SCRATCH=/scratch/$USER
export PERMANENT=/projects/work/yang-lab/users/$USER

# conda
export CONDA_ENV=$PERMANENT/conda_envs/rebuttal

# full tracebacks with hydra
export HYDRA_FULL_ERROR=1

# avoid memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True