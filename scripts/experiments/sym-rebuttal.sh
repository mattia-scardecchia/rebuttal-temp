#!/bin/bash -e
#SBATCH --job-name=sym
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 23:59:00
#SBATCH --output=logs/sym_%j.out
#SBATCH --error=logs/sym_%j.err
#SBATCH --account=torch_pr_147_courant


PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
mkdir -p "${PROJECT_ROOT}/logs"
source "${PROJECT_ROOT}/scripts/experiments/constants.sh"

module load anaconda3/2025.06
source "$(conda info --base)/etc/profile.d/conda.sh"

# perceptron rule
conda run -p "$CONDA_ENV" python scripts/grid_search.py -cn sym name=symmetric_rules/perceptron \
    '+J_D_values=[0.0,0.5]' \
    '+lambda_input_skip_values=[5.0]' \
    '+double_dynamics_values=[true,false]' \
    '+symmetric_J_init_values=[true,false]' \
    '+seed_values=[0,1,2]' \
    num_epochs=30

# symmetric rules
conda run -p "$CONDA_ENV" python scripts/grid_search.py -cn sym name=symmetric_rules/sym1 \
    '+J_D_values=[0.0,0.5]' \
    '+lambda_input_skip_values=[5.0]' \
    '+double_dynamics_values=[true,false]' \
    symmetric_J_init=true \
    symmetric_threshold_internal_couplings=true \
    '+seed_values=[0,1,2]' \
    num_epochs=30
conda run -p "$CONDA_ENV" python scripts/grid_search.py -cn sym name=symmetric_rules/sym2 \
    '+J_D_values=[0.0,0.5]' \
    '+lambda_input_skip_values=[5.0]' \
    '+double_dynamics_values=[true,false]' \
    symmetric_J_init=true \
    symmetric_update_internal_couplings=true \
    '+seed_values=[0,1,2]' \
    num_epochs=30
