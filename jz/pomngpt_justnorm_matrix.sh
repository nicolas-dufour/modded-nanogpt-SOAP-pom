#!/bin/bash
#SBATCH --job-name=justnorm_matrix
#SBATCH --output=logs/justnorm_matrix/%j.out
#SBATCH --error=logs/justnorm_matrix/%j.err
#SBATCH --account=wre@h100
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread

module purge

module load arch/h100
module load miniforge/24.9.0
conda activate pom

# Run the script
cd $WORK/modded-nanogpt-SOAP-pom

export WANDB_MODE=offline

set -x
srun torchrun --standalone --nproc_per_node=4 train.py \
     experiment=pomngpt_justnorm_matrix
